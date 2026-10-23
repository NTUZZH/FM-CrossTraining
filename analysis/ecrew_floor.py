#!/usr/bin/env python
"""Length floor of weighted tardiness per verdict instance (E-CREW companion).

An order cannot finish before its release plus its shortest possible elapsed
time, so sum_j w_j * max(0, r_j + e_j - d_j) is a lower bound on the TWT of
every schedule of the instance, whatever the crew design or the dispatcher.
Here e_j is the elapsed time of order j on a full-speed crew of its scenario
size: p_j for c_j = 1 and ceil_0.01(p_j / c_j) for c_j >= 2, with c_j from the
E-CREW crew-size rule (env/crew_engine.crew_sizes) on the dedicated roster at
crew multiplier m. Tardiness above the floor arises only from waiting for a
technician (start later than release), so L0 TWT minus the floor is the part
of tardiness that a workforce design can remove.

Writes results/ecrew/floor.csv (instance_id, campus, m, c_max, floor) for
analysis/gen_ecrew.py.

Usage (CPU only):
  PYTHONPATH=.:vendor taskset -c 0-3 python analysis/ecrew_floor.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

import experiments.run_ecrew as rc          # noqa: E402
from env.crew_engine import crew_sizes      # noqa: E402


def floor_of(work_orders, sizes) -> float:
    total = 0.0
    for w in work_orders:
        c = sizes[w["id"]]
        p = float(w["p_bh"])
        e = p if c == 1 else math.ceil(p / c * 100.0 - 1e-6) / 100.0
        total += w["weight"] * max(0.0, w["release_bh"] + e - w["due_bh"])
    return total


def main():
    out = ROOT / "results/ecrew/floor.csv"
    rows = []
    for cfg in rc.configs_for(None, [1], "/nonexistent"):
        with open(cfg["path"]) as f:
            inst = json.load(f)
        for m in rc.MS:
            techs = rc.overlay_for(cfg["campus"], "dedicated", None, 1.0,
                                   m)["technicians"]
            for c_max in (1, 2, 3, 4):
                sizes = crew_sizes(inst["work_orders"], techs, c_max)
                rows.append({"instance_id": cfg["instance_id"],
                             "campus": cfg["campus"], "m": m,
                             "c_max": c_max,
                             "floor": floor_of(inst["work_orders"], sizes)})
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote %d rows to %s" % (len(rows), out))


if __name__ == "__main__":
    main()
