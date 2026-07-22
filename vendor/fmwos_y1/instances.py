# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/instances.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""Replay instance builder (P1; sampling v2 = first-N-releases).

Produces instance JSONs matching the schema in notes/interfaces.md exactly.

Sampling v2 (interfaces.md "Replay sampling protocol (v2)"; decisions.md
2026-07-04, supersedes the fixed-window +-20% rule which oversampled lull
days on high-variance campuses):

- an instance = the FIRST ``size`` work orders of the campus released at/after
  a weekday-08:00 anchor ``t0`` -- always exactly ``size`` WOs;
- ``window_bh`` = release_bh of the last included WO (min 1.0 bh), i.e. the
  window is adaptive per anchor instead of fixed per campus;
- anchor acceptance (in scripts/p1_instances.py): every weekday 08:00 in the
  campus span, shuffled (seed 301), greedy, with NON-OVERLAPPING
  ``[t0, t0+window]`` ranges within a (campus, size, split) cell; windows must
  not cross the train/test boundary; skip anchors with fewer than ``size`` WOs
  remaining in the stream. Cap 100 accepted per cell.

Per-WO fields: ``release_bh = to_bh(WOStartDate, t0)``,
``due_bh = release_bh + SLA_BH[priority]``, ``p_bh = LaborHours`` (already
capped), ``trade`` after the MISC merge, ``priority`` after the calibrated v2
mapping, ``building`` = BuildingID or null, ``is_pm`` from ``io.clean``.

Technicians are replicated from the crew counts in the campus's capacity table,
covering ALL of the campus's trades (not just those present in a given
instance), so every WO trade always has >= 1 eligible technician.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import timeaxis as ta

CAMPUS_SET = [1, 2, 5, 9, 10, 12]
SIZES = [50, 150, 400]
WINDOW_MIN_BH = 1.0      # recorded window never below 1 bh


# --------------------------------------------------------------------------- #
# Technician pool                                                             #
# --------------------------------------------------------------------------- #
def technicians_for_campus(capacity: pd.DataFrame, campus: int):
    """Return (trades_list, technicians_list) for a campus from its crew counts."""
    sub = capacity[capacity["campus"] == campus].sort_values("trade")
    trades = sub["trade"].tolist()
    techs: list[dict] = []
    tid = 0
    for _, r in sub.iterrows():
        for _ in range(int(r["crew"])):
            techs.append({"id": f"T{tid}", "trade": r["trade"]})
            tid += 1
    return trades, techs


# --------------------------------------------------------------------------- #
# Per-campus preparation                                                       #
# --------------------------------------------------------------------------- #
def prepare_campus(clean: pd.DataFrame, campus: int,
                   trade_m: pd.Series, priority: pd.Series) -> dict:
    """Pre-sort a campus's WOs by absolute bh for fast first-N queries."""
    mask = clean["UniversityID"].astype("int64") == campus
    sub = clean[mask]
    abs_bh = ta.abs_bh_series(sub["WOStartDate"])
    order = np.argsort(abs_bh, kind="stable")

    b_isna = sub["BuildingID"].isna().to_numpy()
    b_vals = sub["BuildingID"].astype("object").to_numpy()
    building = np.where(b_isna, None, b_vals).astype(object)
    return {
        "abs_bh": abs_bh[order],
        "wo_id": sub["WOID"].astype("object").to_numpy()[order],
        "trade": trade_m[mask].to_numpy()[order],
        "p_bh": sub["LaborHours"].to_numpy(dtype="float64")[order],
        "priority": priority[mask].to_numpy().astype("int64")[order],
        "building": building[order],
        "is_pm": sub["is_pm"].fillna(False).astype(bool).to_numpy()[order],
    }


# --------------------------------------------------------------------------- #
# Sampling v2: probe + build                                                   #
# --------------------------------------------------------------------------- #
def probe_window(prep: dict, t0_abs: float, size: int):
    """Locate the first ``size`` releases at/after bh-coordinate ``t0_abs``.

    Returns (lo, hi, window_bh) or None if fewer than ``size`` WOs remain in
    the campus stream. ``window_bh`` = release_bh of the size-th WO (rounded
    to 4 decimals, exactly like the stored releases), floored at 1.0 bh.
    """
    abs_bh = prep["abs_bh"]
    lo = int(np.searchsorted(abs_bh, t0_abs, side="left"))
    hi = lo + size
    if hi > len(abs_bh):
        return None
    last_release = float(np.round(abs_bh[hi - 1] - t0_abs, 4))
    window_bh = max(last_release, WINDOW_MIN_BH)
    return lo, hi, window_bh


def build_instance(prep: dict, t0: pd.Timestamp, t0_abs: float,
                   lo: int, hi: int, window_bh: float, size: int,
                   inst_id: str, campus: int, trades: list, technicians: list) -> dict:
    """Assemble the instance dict for a probed [lo, hi) slice."""
    release = np.round(prep["abs_bh"][lo:hi] - t0_abs, 4)
    prio = prep["priority"][lo:hi]
    p_bh = np.round(prep["p_bh"][lo:hi], 4)
    wo_id = prep["wo_id"][lo:hi]
    trade = prep["trade"][lo:hi]
    building = prep["building"][lo:hi]
    is_pm = prep["is_pm"][lo:hi]

    work_orders = []
    for k in range(hi - lo):
        pr = int(prio[k])
        rb = float(release[k])
        work_orders.append({
            "id": str(wo_id[k]),
            "trade": str(trade[k]),
            "p_bh": float(p_bh[k]),
            "release_bh": rb,
            "due_bh": round(rb + ta.SLA_BH[pr], 4),
            "priority": pr,
            "weight": ta.WEIGHT[pr],
            "building": (None if building[k] is None else str(building[k])),
            "is_pm": bool(is_pm[k]),
        })

    return {
        "meta": {
            "id": inst_id,
            "campus": int(campus),
            "track": "replay",
            "size_class": int(size),
            "window_start": t0.isoformat(),
            "window_bh": round(float(window_bh), 4),
            "provenance": "R",
            "seed": None,
        },
        "trades": list(trades),
        "technicians": list(technicians),
        "work_orders": work_orders,
    }
