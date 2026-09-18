#!/usr/bin/env python
"""Roster-calibration variants (E-CAL) and subset rosters (E-SUB).

The released roster sizes a trade's crew from the weekly hours that trade
actually booked, ``crew = max(1, ceil(p95(weekly summed LaborHours) / 40))``,
on the Y1 train years (WOStartDate <= 2017-12-31), per (campus, trade), with
trades below 1000 rows across all years merged into MISC first
(``fmwos_y1.calib.build_capacity``). The quantile and the window are
estimator choices, so this module rebuilds the same table with q and the
window as parameters, and with an optional row filter for the corrective-only
and short-job streams.

Cleaning and trade merging come from the vendored Y1 modules unchanged
(``fmwos_y1.io.clean``, ``fmwos_y1.calib.trade_merge_map`` /
``apply_trade_merge``). The crew rule itself is re-expressed here only to
admit q, the window, and the filter; ``--verify`` asserts that the
parameterized builder at (q = 0.95, train window, no filter) returns exactly
what ``fmwos_y1.calib.build_capacity`` returns on the same cleaned frame, and
that both equal the released capacity table.

The trade universe is held at the one the unfiltered corpus produces, so a
filtered stream never removes a trade and every work order keeps at least one
eligible technician; a trade with no hours in the filtered stream gets the
floor of one technician.

Cache
-----
The raw corpus is 1.4 GB. One pass writes a compact weekly aggregate
(campus, trade, ISO week, train flag, preventive flag, short flag -> summed
hours and row count) to ``results/ecal/cache/weekly.parquet``; every
calibration is then a group-by over that cache. ``--rebuild`` forces the raw
pass.

Outputs
-------
  results/ecal/capacity_<label>.csv     q and window variants
  results/esub/capacity_<label>.csv     corrective-only and short-job rosters
  results/ecal/cache/weekly.parquet     cleaning cache (input artifact)
  results/ecal/cache/clean_audit.json   the cleaning audit for that cache

Usage:
  PYTHONPATH=.:vendor python experiments/recalibrate.py --verify
  PYTHONPATH=.:vendor python experiments/recalibrate.py --all
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from fmwos_y1 import calib, io                              # noqa: E402

FROZEN = ROOT / "data" / "y1_frozen_v10"


def resolve_y1_root(override=None):
    """Locate the Y1 corpus root that every runner in this family reads.

    Delegates to ``experiments/y1_root.py``, the shared resolver every runner
    uses, and falls back to its order (explicit override, the FMWOS_Y1_ROOT
    environment variable, this project's frozen copy of the corpus the
    released results were computed on, the sibling repository) when that
    module is absent. Instances and the released calibration table are read
    from the same root, so the two corpus versions can never be mixed.
    """
    try:
        from experiments.y1_root import y1_root          # noqa: WPS433
    except Exception:                                    # noqa: BLE001
        y1_root = None
    if y1_root is not None:
        return Path(y1_root(override))
    if override:
        return Path(override)
    env = os.environ.get("FMWOS_Y1_ROOT")
    if env:
        return Path(env)
    if (FROZEN / "results/p1_calib/capacity.csv").exists():
        return FROZEN
    return ROOT.parent / "PaperY-FMScheduling"


def raw_csv_path(y1_root=None):
    """The raw FMUCD corpus. A frozen root need not carry the 1.4 GB file;
    the corpus itself never changed (its checksum is pinned in
    ``fmwos_y1.io.RAW_SHA256``), so the sibling repository supplies it when
    the resolved root does not."""
    root = Path(y1_root) if y1_root else resolve_y1_root()
    p = root / "data/raw/FMUCD.csv"
    if p.exists():
        return p
    return ROOT.parent / "PaperY-FMScheduling" / "data/raw/FMUCD.csv"


Y1 = resolve_y1_root()
RAW_CSV = raw_csv_path(Y1)
RELEASED_CAP = Y1 / "results/p1_calib/capacity.csv"
ECAL_DIR = ROOT / "results" / "ecal"
ESUB_DIR = ROOT / "results" / "esub"
CACHE_DIR = ECAL_DIR / "cache"
CACHE = CACHE_DIR / "weekly.parquet"
PAIRS_CACHE = CACHE_DIR / "trade_pairs.parquet"
AUDIT = CACHE_DIR / "clean_audit.json"

SHORT_BH = 8.0            # one technician-day; the short-job class boundary
CAP_COLUMNS = ["campus", "trade", "crew", "p95_weekly_hours", "rows"]

# label -> (quantile, window, row filter)
ECAL_VARIANTS = {
    "q090_train": (0.90, "train", None),
    "q095_train": (0.95, "train", None),
    "q0975_train": (0.975, "train", None),
    "q099_train": (0.99, "train", None),
    "q095_all": (0.95, "all", None),
}
ESUB_VARIANTS = {
    "cm_q095_train": (0.95, "train", "cm"),
    "le8_q095_train": (0.95, "train", "short"),
}


# --------------------------------------------------------------------------- #
# Deterministic dominant-line tie-break                                        #
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def dominant_line(mode="stable"):
    """R7 dominant-line tie-break: ``stable`` or the vendored ``legacy``."""
    if mode == "legacy":
        yield
        return
    if mode != "stable":
        raise ValueError("dominant_sort must be 'stable' or 'legacy'")
    with stable_dominant_line():
        yield


@contextlib.contextmanager
def stable_dominant_line():
    """Force the R7 dominant-line tie-break to follow raw-file order.

    R7 aggregates the labor lines of one work order and takes every other
    field from the line with the most hours. Roughly half of the multi-line
    work orders report equal hours on every line, so the "dominant" line is
    undefined on hours alone, and pandas' default quicksort then picks an
    arbitrary one. The released calibration corresponds to the first line in
    raw-file order, so this context manager makes the sort inside
    ``fmwos_y1.io.clean`` stable for the duration of the call. Nothing in
    ``vendor/`` is modified, and the regression check against the released
    capacity table is what proves the choice.
    """
    original = pd.DataFrame.sort_values

    def sort_values(self, by, *args, **kwargs):
        kwargs.setdefault("kind", "stable")
        return original(self, by, *args, **kwargs)

    pd.DataFrame.sort_values = sort_values
    try:
        yield
    finally:
        pd.DataFrame.sort_values = original


# --------------------------------------------------------------------------- #
# One raw pass -> compact weekly aggregate                                     #
# --------------------------------------------------------------------------- #
def build_cache(raw_csv=None, verbose=True, dominant_sort="stable"):
    """Clean the raw corpus once and write the weekly aggregate cache.

    Returns (weekly, pairs, audit). ``weekly`` has one row per
    (campus, trade, week, in_train, is_pm, is_short) with the summed
    LaborHours and the row count; ``pairs`` is the unfiltered
    (campus, trade) universe.
    """
    raw_csv = Path(raw_csv) if raw_csv else raw_csv_path()
    t0 = time.perf_counter()
    if verbose:
        print("reading %s (dominant line: %s) ..."
              % (raw_csv.name, dominant_sort), flush=True)
    with dominant_line(dominant_sort):
        clean, audit = io.clean(io.load_raw(raw_csv))
    if verbose:
        print("  cleaned %d work orders in %.0f s (PM share %.4f)"
              % (len(clean), time.perf_counter() - t0, audit["pm_share"]),
              flush=True)

    tmap = calib.trade_merge_map(clean)
    trade_m = calib.apply_trade_merge(clean, tmap)

    df = pd.DataFrame({
        "campus": clean["UniversityID"].astype("int64").to_numpy(),
        "trade": trade_m.to_numpy(),
        "week": clean["WOStartDate"].dt.to_period("W").astype(str).to_numpy(),
        "in_train": (clean["WOStartDate"] <= calib.TRAIN_END).to_numpy(),
        "is_pm": clean["is_pm"].fillna(False).astype(bool).to_numpy(),
        "hours": clean["LaborHours"].astype("float64").to_numpy(),
    })
    # p_bh in an instance is round(LaborHours, 4); the short class uses the
    # same rounded value, so the raw stream and the instances select the same
    # orders.
    df["is_short"] = np.round(df["hours"].to_numpy(), 4) <= SHORT_BH

    pairs = (df[["campus", "trade"]].drop_duplicates()
             .sort_values(["campus", "trade"]).reset_index(drop=True))
    weekly = (df.groupby(["campus", "trade", "week", "in_train", "is_pm",
                          "is_short"], observed=True)
              .agg(hours=("hours", "sum"), rows=("hours", "size"))
              .reset_index())

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(CACHE, index=False)
    pairs.to_parquet(PAIRS_CACHE, index=False)
    with open(AUDIT, "w") as f:
        json.dump({"generated": _dt.datetime.now().isoformat(
            timespec="seconds"),
            "raw_csv": raw_csv.name,
            "dominant_sort": dominant_sort,
            "crew_hours_per_week": calib.CREW_HOURS,
            "train_end": str(calib.TRAIN_END),
            "short_bh": SHORT_BH,
            "audit": {k: (float(v) if isinstance(v, float) else int(v))
                      for k, v in audit.items()}}, f, indent=2)
    if verbose:
        print("  cache: %d weekly rows, %d (campus, trade) pairs -> %s"
              % (len(weekly), len(pairs), CACHE), flush=True)
    return weekly, pairs, audit, clean, trade_m


def load_cache(rebuild=False, verbose=True, dominant_sort="stable"):
    """Weekly aggregate + trade universe, from cache when available."""
    if CACHE.exists() and PAIRS_CACHE.exists() and not rebuild:
        return (pd.read_parquet(CACHE), pd.read_parquet(PAIRS_CACHE),
                json.load(open(AUDIT))["audit"] if AUDIT.exists() else {},
                None, None)
    return build_cache(verbose=verbose, dominant_sort=dominant_sort)


# --------------------------------------------------------------------------- #
# Parameterized crew table                                                     #
# --------------------------------------------------------------------------- #
def build_capacity_variant(weekly, pairs, q=0.95, window="train",
                           row_filter=None, campuses=None):
    """Per-(campus, trade) crew table at quantile ``q`` on ``window``.

    ``window`` is "train" (WOStartDate <= 2017-12-31, the released rule) or
    "all" (every recorded year). ``row_filter`` is None, "cm" (drop preventive
    orders) or "short" (keep orders of at most 8 bh). The quantile runs over
    the weeks that have at least one order in the selected stream, matching
    the released rule; a (campus, trade) pair with no such week keeps the
    floor of one technician.

    The column name ``p95_weekly_hours`` is kept at every q so the table is a
    drop-in for ``overlays.build.load_crews``; the header of each written file
    records the quantile that produced it.
    """
    sub = weekly
    if window == "train":
        sub = sub[sub["in_train"]]
    elif window != "all":
        raise ValueError("window must be 'train' or 'all'")
    if row_filter == "cm":
        sub = sub[~sub["is_pm"]]
    elif row_filter == "short":
        sub = sub[sub["is_short"]]
    elif row_filter is not None:
        raise ValueError("row_filter must be None, 'cm' or 'short'")

    wk = (sub.groupby(["campus", "trade", "week"], observed=True)["hours"]
          .sum())
    n_rows = sub.groupby(["campus", "trade"], observed=True)["rows"].sum()

    rows = []
    for campus, trade in pairs.itertuples(index=False):
        try:
            wser = wk.loc[(campus, trade)]
        except KeyError:
            wser = pd.Series(dtype="float64")
        vol = (float(np.quantile(wser.to_numpy(), q)) if len(wser) > 0
               else 0.0)
        crew = max(1, int(math.ceil(vol / calib.CREW_HOURS)))
        rows.append({"campus": int(campus), "trade": str(trade),
                     "crew": crew, "p95_weekly_hours": round(vol, 2),
                     "rows": int(n_rows.get((campus, trade), 0))})
    cap = pd.DataFrame(rows, columns=CAP_COLUMNS)
    keep = calib.CAMPUSES if campuses is None else list(campuses)
    cap = cap[cap["campus"].isin(keep)]
    return cap.sort_values(["campus", "trade"]).reset_index(drop=True)


def write_capacity(cap, path, label, q, window, row_filter):
    """Write a capacity table plus a sidecar provenance file.

    The table itself is byte-compatible with the released capacity.csv (same
    header, same column order, no comment lines), so
    ``overlays.build.load_crews`` reads it unchanged; the quantile, window and
    filter that produced it live in ``<name>.meta.json`` beside it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cap.to_csv(path, index=False)
    meta = {"generator": "experiments/recalibrate.py", "label": label,
            "quantile": float(q), "window": window,
            "row_filter": row_filter or "none",
            "note": ("column p95_weekly_hours holds the quantile of weekly "
                     "summed labor hours at this quantile"),
            "trades": int(len(cap)), "headcount": int(cap["crew"].sum())}
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return path


def read_capacity(path):
    """Read a table written by ``write_capacity`` (or the released one)."""
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Regression checks                                                            #
# --------------------------------------------------------------------------- #
def released_capacity(y1_root=None):
    """The released calibration table of the resolved Y1 root."""
    root = Path(y1_root) if y1_root else resolve_y1_root()
    path = root / "results/p1_calib/capacity.csv"
    return pd.read_csv(path)[CAP_COLUMNS].sort_values(
        ["campus", "trade"]).reset_index(drop=True)


def compare(cap, ref, name_a="rebuilt", name_b="released"):
    """Exact comparison of two capacity tables. Returns a list of diffs."""
    diffs = []
    a = cap.set_index(["campus", "trade"])
    b = ref.set_index(["campus", "trade"])
    only_a = sorted(set(a.index) - set(b.index))
    only_b = sorted(set(b.index) - set(a.index))
    for k in only_a:
        diffs.append("(campus %s, trade %s) in %s only" % (k[0], k[1], name_a))
    for k in only_b:
        diffs.append("(campus %s, trade %s) in %s only" % (k[0], k[1], name_b))
    for k in sorted(set(a.index) & set(b.index)):
        for col in ("crew", "p95_weekly_hours", "rows"):
            va, vb = a.loc[k, col], b.loc[k, col]
            if float(va) != float(vb):
                diffs.append("(campus %s, trade %s) %s: %s=%r %s=%r"
                             % (k[0], k[1], col, name_a, va, name_b, vb))
    return diffs


def chain_orders(cap):
    """{campus -> chain order} the overlay builder derives from a table."""
    from overlays.build import chain_order
    out = {}
    for campus in sorted(cap["campus"].unique()):
        sub = cap[cap["campus"] == campus]
        crews = [{"trade": r.trade, "crew": int(r.crew),
                  "volume": float(r.p95_weekly_hours)}
                 for r in sub.itertuples()]
        out[int(campus)] = chain_order(crews)
    return out


def verify(rebuild=False, dominant_sort="stable", y1_root=None):
    """Regression check 1, in three graded parts.

    (1a) the crew column of the rebuilt q = 0.95 / train table equals the
         released one, campus by campus and trade by trade. The crew count is
         the roster, so a difference here would make every alternative
         calibration incomparable; it is a hard gate.
    (1b) the volume column and the chain order it induces. A volume that
         differs moves a trade in the workload-ordered chain, so any
         difference is reported with the campus it affects.
    (1c) the parameterized builder at (q = 0.95, train, no filter) equals
         ``fmwos_y1.calib.build_capacity`` on the same cleaned frame, which
         pins the parameterization to the vendored rule.
    """
    weekly, pairs, _audit, clean, trade_m = load_cache(
        rebuild=rebuild, dominant_sort=dominant_sort)
    cap = build_capacity_variant(weekly, pairs, 0.95, "train", None)
    ref = released_capacity(y1_root)

    crew_diffs = [d for d in compare(cap, ref) if " crew:" in d]
    print("[check 1a] rebuilt (q=0.95, train) crews vs the released table: "
          "%d row(s), %d difference(s)" % (len(cap), len(crew_diffs)))
    for d in crew_diffs[:20]:
        print("   ", d)

    vol_diffs = [d for d in compare(cap, ref)
                 if " p95_weekly_hours:" in d or " rows:" in d]
    o_new, o_ref = chain_orders(cap), chain_orders(ref)
    moved = [c for c in o_ref if o_new.get(c) != o_ref[c]]
    print("[check 1b] volume and row-count differences: %d; campuses whose "
          "chain order moves: %s" % (len(vol_diffs),
                                     moved if moved else "none"))
    for d in vol_diffs[:20]:
        print("   ", d)

    diffs_v = []
    if clean is not None:
        vend = calib.build_capacity(clean, trade_m)
        vend = vend[vend["campus"].isin(calib.CAMPUSES)].sort_values(
            ["campus", "trade"]).reset_index(drop=True)
        diffs_v = compare(cap, vend, "parameterized", "vendored")
        print("[check 1c] parameterized vs fmwos_y1.calib.build_capacity on "
              "the same cleaned frame: %d difference(s)" % len(diffs_v))
        for d in diffs_v[:20]:
            print("   ", d)
    else:
        print("[check 1c] skipped (cache hit; rerun with --rebuild to "
              "re-derive the vendored table from the raw corpus)")
    return {"crew_diffs": crew_diffs, "volume_diffs": vol_diffs,
            "chain_order_moved": moved, "vendored_diffs": diffs_v,
            "pass": len(crew_diffs) == 0 and len(diffs_v) == 0,
            "exact": (len(crew_diffs) == 0 and len(vol_diffs) == 0
                      and len(diffs_v) == 0)}


# --------------------------------------------------------------------------- #
def write_all(rebuild=False, dominant_sort="stable", y1_root=None):
    weekly, pairs, _a, _c, _t = load_cache(rebuild=rebuild,
                                           dominant_sort=dominant_sort)
    out = {}
    # The released roster is copied through verbatim, never rebuilt, so the
    # label that anchors the sweep to the manuscript is the table the released
    # results were produced with.
    rel = released_capacity(y1_root)
    p = write_capacity(rel, ECAL_DIR / "capacity_released.csv", "released",
                       0.95, "train", None)
    out["released"] = p
    print("%-14s trades=%3d headcount=%5d  -> %s (copied from the resolved "
          "Y1 root)" % ("released", len(rel), int(rel["crew"].sum()), p.name))
    for label, (q, window, filt) in ECAL_VARIANTS.items():
        cap = build_capacity_variant(weekly, pairs, q, window, filt)
        p = write_capacity(cap, ECAL_DIR / ("capacity_%s.csv" % label),
                           label, q, window, filt)
        out[label] = p
        print("%-14s trades=%3d headcount=%5d  -> %s"
              % (label, len(cap), int(cap["crew"].sum()), p.name))
    for label, (q, window, filt) in ESUB_VARIANTS.items():
        cap = build_capacity_variant(weekly, pairs, q, window, filt)
        p = write_capacity(cap, ESUB_DIR / ("capacity_%s.csv" % label),
                           label, q, window, filt)
        out[label] = p
        print("%-14s trades=%3d headcount=%5d  -> %s"
              % (label, len(cap), int(cap["crew"].sum()), p.name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="re-read the raw corpus instead of the cache")
    ap.add_argument("--verify", action="store_true",
                    help="run the released-table regression check only")
    ap.add_argument("--all", action="store_true",
                    help="write every calibration table")
    ap.add_argument("--y1-root", default=None,
                    help="corpus root holding the instances and the released "
                         "calibration table (default: the resolver)")
    ap.add_argument("--dominant-sort", default="stable",
                    choices=["stable", "legacy"],
                    help="R7 tie-break between labor lines of equal hours")
    args = ap.parse_args()

    root = resolve_y1_root(args.y1_root)
    print("Y1 root: %s" % root.name)
    res = verify(rebuild=args.rebuild, dominant_sort=args.dominant_sort,
                 y1_root=root)
    if not res["pass"]:
        sys.exit("REGRESSION CHECK FAILED: the rebuilt q = 0.95 / train "
                 "roster does not match the released crews")
    print("[check 1] %s" % ("PASS (exact)" if res["exact"]
                            else "PASS on crews; volume differences listed "
                                 "above"))
    with open(ECAL_DIR / "calibration_check.json", "w") as f:
        ECAL_DIR.mkdir(parents=True, exist_ok=True)
        json.dump({"y1_root": root.name,
                   "dominant_sort": args.dominant_sort,
                   "crew_differences": res["crew_diffs"],
                   "volume_differences": res["volume_diffs"],
                   "chain_order_moved": res["chain_order_moved"],
                   "vendored_differences": res["vendored_diffs"],
                   "exact": res["exact"]}, f, indent=2)
    if not args.all:
        return
    write_all(rebuild=False, dominant_sort=args.dominant_sort, y1_root=root)


if __name__ == "__main__":
    main()
