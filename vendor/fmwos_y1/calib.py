# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/calib.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""P1 calibration: per-campus priority mapping (v2) and per-(campus,trade) crew size.

Both artifacts are deterministic functions of the cleaned FMUCD dataframe
(``fmwos.io.clean`` output) and are written to ``results/p1_calib/``.

Priority mapping v2 (evidence-based; interfaces.md + decisions.md 2026-07-04)
----------------------------------------------------------------------------
The raw ``WOPriority`` field mixes planned-work categories with urgency, and
several campuses' numeric scales run in *opposite* directions (campus 12's
"30" completes in a median 75 d vs "50" in 25 d). The mapping is therefore a
function of (campus, raw value, is_pm):

  R5a  is_pm (PPM) work orders -> class 4 regardless of raw value (planned =
       calendared; labeled campuses put PM at the bottom of their scales).
  R5b  CM with a keyword label -> keyword class (case-insensitive substring):
       EMERG->1, URG->2, ROUTINE/EXPEDITED/CALENDARED->3, DEFER/PLANNED->4.
  R5c  CM numeric scales: common values = those with >= 0.5% of the campus's
       CM rows; keep the scale's own numeric order but flip direction if
       Spearman(value rank, median realized completion duration of the CM
       subset per value) < 0 (urgent work completes sooner); then
       class = ceil(rank / N * 4) over the N common values (rank 1 = most
       urgent).
  R5d  rare / unmappable / missing -> class 3.

Realized completion duration = (WOEndDate - WOStartDate) in total days,
computed on the cleaned data (rows without WOEndDate are excluded from the
median). Every (campus, raw value, pm/cm-split) row is emitted to
priority_mapping.csv with counts, the evidence columns, and the rule that
fired.

Crew size, per (campus, trade)
------------------------------
crew = max(1, ceil(p95(weekly summed LaborHours) / 40)), computed on train
years (WOStartDate <= 2017-12-31). Weekly hours are summed over ISO weeks that
have >= 1 order (observed weekly trade-hours). Trades with < 1000 rows in a
campus across *all* years are merged into "MISC" for that campus *before* the
capacity computation (interfaces.md).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

CAMPUSES = [1, 2, 5, 9, 10, 12]
TRAIN_END = pd.Timestamp("2017-12-31 23:59:59")
RARE_FRAC = 0.005        # < 0.5% of a campus's CM rows -> "rare"
MISC_MIN_ROWS = 1000     # trades below this (all years) merge into MISC
CREW_HOURS = 40.0        # one crew-week of capacity, in bh
MISSING_TOKEN = "<MISSING>"


# --------------------------------------------------------------------------- #
# Trade merge (MISC)                                                          #
# --------------------------------------------------------------------------- #
def trade_merge_map(clean: pd.DataFrame) -> dict[int, dict[str, str]]:
    """Per campus, {raw_trade -> raw_trade or 'MISC'} using all-year counts."""
    out: dict[int, dict[str, str]] = {}
    counts = clean.groupby(["UniversityID", "trade"], observed=True).size()
    for campus, sub in counts.groupby(level=0):
        m = {}
        for (_, trade), n in sub.items():
            m[trade] = trade if n >= MISC_MIN_ROWS else "MISC"
        out[int(campus)] = m
    return out


def apply_trade_merge(clean: pd.DataFrame, tmap: dict[int, dict[str, str]]) -> pd.Series:
    """Return a Series of merged trade (``trade_m``) aligned to ``clean``.

    Vectorised via a merge on (campus, trade).
    """
    lut = pd.DataFrame(
        [(c, t, tm) for c, m in tmap.items() for t, tm in m.items()],
        columns=["campus", "trade", "trade_m"],
    )
    left = pd.DataFrame({
        "campus": clean["UniversityID"].astype("int64").to_numpy(),
        "trade": clean["trade"].astype("object").to_numpy(),
    })
    merged = left.merge(lut, on=["campus", "trade"], how="left")
    # Any (campus, trade) not in the lut keeps its own trade (defensive).
    tm = merged["trade_m"].where(merged["trade_m"].notna(), merged["trade"])
    return pd.Series(tm.to_numpy(), index=clean.index, name="trade_m")


# --------------------------------------------------------------------------- #
# Priority mapping (v2)                                                       #
# --------------------------------------------------------------------------- #
def _keyword_class(raw_upper: str) -> int | None:
    """R5b keyword anchors (CM rows only). Note: no 'PM' keyword in v2 --
    planned-maintenance semantics are carried by is_pm (R5a)."""
    if "EMERG" in raw_upper:
        return 1
    if "URG" in raw_upper:
        return 2
    if "ROUTINE" in raw_upper or "EXPEDITED" in raw_upper or "CALENDARED" in raw_upper:
        return 3
    if "DEFER" in raw_upper or "PLANNED" in raw_upper:
        return 4
    return None


def _as_float(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties); nan if < 2 points."""
    if len(x) < 2:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy(dtype="float64")
    ry = pd.Series(y).rank().to_numpy(dtype="float64")
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def build_priority_mapping(clean: pd.DataFrame) -> pd.DataFrame:
    """Return the v2 per-campus priority mapping table.

    One row per (campus, raw value, pm/cm split) with evidence columns:
    rows, median_cm_duration_days (null for pm rows), pm_share_of_value,
    mapped_class, rule (r5a/r5b/r5c/r5d), spearman_rho + direction (r5c only).
    """
    is_pm = clean["is_pm"].fillna(False).astype(bool).to_numpy()
    dur_days = (
        (clean["WOEndDate"] - clean["WOStartDate"]).dt.total_seconds() / 86400.0
    ).to_numpy(dtype="float64")
    base = pd.DataFrame({
        "campus": clean["UniversityID"].astype("int64").to_numpy(),
        "raw_value": clean["WOPriority"].astype("string").fillna(MISSING_TOKEN)
        .astype(object).to_numpy(),
        "is_pm_split": np.where(is_pm, "pm", "cm"),
        "dur_days": dur_days,
    })

    rows: list[dict] = []
    for campus, sub in base.groupby("campus"):
        cnt = (
            sub.groupby(["raw_value", "is_pm_split"]).size().unstack(fill_value=0)
            .reindex(columns=["cm", "pm"], fill_value=0)
        )
        cm = sub[sub["is_pm_split"] == "cm"]
        cm_total = int(cnt["cm"].sum())
        rare_cut = RARE_FRAC * cm_total
        # median realized completion duration per raw value, CM subset only
        med_dur = cm.groupby("raw_value")["dur_days"].median()

        # ---- R5c: the campus's common CM numeric scale ---------------------
        commons: list[tuple[float, str]] = []  # (numeric value, raw string)
        for v in cnt.index:
            if v == MISSING_TOKEN or int(cnt.loc[v, "cm"]) < rare_cut:
                continue
            f = _as_float(str(v))
            if f is not None:
                commons.append((f, str(v)))
        commons.sort(key=lambda t: t[0])
        n_common = len(commons)
        rho = float("nan")
        direction = ""
        scale: dict[str, int] = {}
        if n_common >= 1:
            meds = np.array([med_dur.get(raw, np.nan) for _, raw in commons])
            ok = ~np.isnan(meds)
            rho = _spearman(np.arange(n_common)[ok].astype("float64"), meds[ok])
            flip = (not math.isnan(rho)) and rho < 0
            direction = "descending" if flip else "ascending"
            for i, (_, raw) in enumerate(commons):
                value_rank = i + 1                       # ascending numeric order
                urg_rank = (n_common + 1 - value_rank) if flip else value_rank
                scale[raw] = max(1, min(4, int(math.ceil(urg_rank / n_common * 4))))

        # ---- emit one row per (raw value, pm/cm split) ----------------------
        for v in cnt.index:
            n_cm = int(cnt.loc[v, "cm"])
            n_pm = int(cnt.loc[v, "pm"])
            n_tot = n_cm + n_pm
            pm_share = round(n_pm / n_tot, 4) if n_tot else 0.0
            common = dict(campus=int(campus), raw_value=str(v),
                          pm_share_of_value=pm_share)
            if n_pm > 0:  # R5a: PM -> 4 regardless of raw value
                rows.append(dict(common, is_pm_split="pm", rows=n_pm,
                                 median_cm_duration_days=None,
                                 mapped_class=4, rule="r5a",
                                 spearman_rho=None, direction=None))
            if n_cm > 0:
                md = med_dur.get(v, np.nan)
                md = None if (md is None or (isinstance(md, float) and math.isnan(md))) \
                    else round(float(md), 2)
                if v == MISSING_TOKEN:
                    cls, rule, r_rho, r_dir = 3, "r5d", None, None
                else:
                    kw = _keyword_class(str(v).upper())
                    if kw is not None:
                        cls, rule, r_rho, r_dir = kw, "r5b", None, None
                    elif str(v) in scale:
                        cls, rule = scale[str(v)], "r5c"
                        r_rho = round(rho, 3) if not math.isnan(rho) else None
                        r_dir = direction
                    else:
                        cls, rule, r_rho, r_dir = 3, "r5d", None, None
                rows.append(dict(common, is_pm_split="cm", rows=n_cm,
                                 median_cm_duration_days=md,
                                 mapped_class=int(cls), rule=rule,
                                 spearman_rho=r_rho, direction=r_dir))

    cols = ["campus", "raw_value", "is_pm_split", "rows",
            "median_cm_duration_days", "pm_share_of_value",
            "mapped_class", "rule", "spearman_rho", "direction"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["campus", "is_pm_split", "mapped_class", "rows"],
                          ascending=[True, True, True, False]).reset_index(drop=True)


def priority_class_series(clean: pd.DataFrame, mapping: pd.DataFrame) -> pd.Series:
    """Merge the v2 mapping onto the cleaned frame -> integer priority Series.

    Merge is on (campus, raw WOPriority, pm/cm split). Rows not covered by the
    mapping (e.g. campuses outside CAMPUSES) default to R5a/R5d: 4 if PM else 3.
    """
    is_pm = clean["is_pm"].fillna(False).astype(bool).to_numpy()
    left = pd.DataFrame({
        "campus": clean["UniversityID"].astype("int64").to_numpy(),
        "raw_value": clean["WOPriority"].astype("string").fillna(MISSING_TOKEN)
        .astype(object).to_numpy(),
        "is_pm_split": np.where(is_pm, "pm", "cm"),
    })
    lut = mapping[["campus", "raw_value", "is_pm_split", "mapped_class"]]
    merged = left.merge(lut, on=["campus", "raw_value", "is_pm_split"], how="left")
    cls = merged["mapped_class"].to_numpy(dtype="float64")
    default = np.where(is_pm, 4.0, 3.0)
    out = np.where(np.isnan(cls), default, cls).astype("int64")
    return pd.Series(out, index=clean.index, name="priority")


# --------------------------------------------------------------------------- #
# Crew capacity                                                               #
# --------------------------------------------------------------------------- #
def build_capacity(clean: pd.DataFrame, trade_m: pd.Series) -> pd.DataFrame:
    """Return the per-(campus,trade) capacity table.

    Covers every (campus, trade_m) present in *any* year so instances always
    have >= 1 technician for a work order's trade; p95 uses train years only.
    """
    df = clean[["UniversityID", "WOStartDate", "LaborHours"]].copy()
    df["trade_m"] = trade_m.to_numpy()
    df["campus"] = df["UniversityID"].astype("int64")
    df["week"] = df["WOStartDate"].dt.to_period("W")

    all_pairs = (
        df.groupby(["campus", "trade_m"], observed=True).size().index.tolist()
    )
    train = df[df["WOStartDate"] <= TRAIN_END]

    # weekly summed LaborHours per (campus, trade, week) on train years
    weekly = (
        train.groupby(["campus", "trade_m", "week"], observed=True)["LaborHours"]
        .sum()
    )

    rows: list[dict] = []
    train_rows = train.groupby(["campus", "trade_m"], observed=True).size()
    for campus, trade in all_pairs:
        # weekly summed LaborHours for this (campus, trade) on train years
        try:
            wser = weekly.loc[(campus, trade)]
        except KeyError:
            wser = pd.Series(dtype="float64")
        if len(wser) > 0:
            p95 = float(np.quantile(wser.to_numpy(), 0.95))
        else:
            p95 = 0.0
        crew = max(1, int(math.ceil(p95 / CREW_HOURS)))
        n_rows = int(train_rows.get((campus, trade), 0))
        rows.append(dict(campus=int(campus), trade=trade, crew=crew,
                         p95_weekly_hours=round(p95, 2), rows=n_rows))

    cap = pd.DataFrame(rows, columns=["campus", "trade", "crew",
                                      "p95_weekly_hours", "rows"])
    return cap.sort_values(["campus", "trade"]).reset_index(drop=True)


def write_calibration(clean: pd.DataFrame, out_dir: str | Path):
    """Compute + write both calibration tables. Returns (mapping, capacity, tmap)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mapping = build_priority_mapping(clean)
    mapping = mapping[mapping["campus"].isin(CAMPUSES)].reset_index(drop=True)
    mapping.to_csv(out / "priority_mapping.csv", index=False)

    tmap = trade_merge_map(clean)
    trade_m = apply_trade_merge(clean, tmap)
    capacity = build_capacity(clean, trade_m)
    capacity = capacity[capacity["campus"].isin(CAMPUSES)].reset_index(drop=True)
    capacity.to_csv(out / "capacity.csv", index=False)

    return mapping, capacity, tmap, trade_m
