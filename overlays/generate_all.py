#!/usr/bin/env python
"""Materialise every overlay used anywhere in Y2 as released JSON files
(the Benchmark v2 asset).

Covers: all 6 campuses x the full ladder x eta {1.0, 0.8, 0.9, 0.75} x
m {1.0, 0.8, 0.6, 0.45, 0.75} (the E5' crew-scale composites) plus the
three chain-permutation variants on the designated family.

Usage: PYTHONPATH=.:vendor python overlays/generate_all.py
Writes overlays/generated/<overlay_id>.json + an index.csv.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from overlays.build import build_overlay, load_crews, save_overlay  # noqa

Y1_ROOT = Path(os.environ.get("FMWOS_Y1_ROOT", ROOT.parent / "FM-Scheduling"))
CAP = str(Y1_ROOT / "results/p1_calib/capacity.csv")
OUT = ROOT / "overlays" / "generated"

CAMPUSES = [1, 2, 5, 9, 10, 12]
MAIN_ETAS = [1.0, 0.8]
SENS_ETAS = [0.9, 0.75]
MAIN_MS = [1.0, 0.8, 0.6]
E5_MS = [0.45, 0.75]                     # 0.6 x {0.75, 1.25}
PERM_SEEDS = [20260708, 20260709, 20260710]

STRUCTS = [("dedicated", None), ("chain", 0.25), ("chain", 0.5),
           ("chain", 1.0), ("generalist", None), ("full", None)]


def main():
    rows = []
    for c in CAMPUSES:
        crews = load_crews(CAP, c)
        for (st, phi) in STRUCTS:
            etas = [1.0] if st == "dedicated" else MAIN_ETAS + SENS_ETAS
            for eta in etas:
                for m in MAIN_MS + E5_MS:
                    ov = build_overlay(c, crews, st, phi, eta, m)
                    save_overlay(ov, OUT)
                    rows.append(ov)
        # chain permutations (designated family: CHAIN(1.0), eta .8, m .6)
        for seed in PERM_SEEDS:
            ov = build_overlay(c, crews, "chain", 1.0, 0.8, 0.6,
                               perm_seed=seed)
            save_overlay(ov, OUT)
            rows.append(ov)
    with open(OUT / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["overlay_id", "campus", "structure", "phi", "eta",
                    "crew_multiplier", "headcount", "budget_B"])
        for ov in rows:
            w.writerow([ov["overlay_id"], ov["campus"], ov["structure"],
                        ov["phi"], ov["eta"], ov["crew_multiplier"],
                        ov["headcount"], ov["budget_B"]])
    print("wrote %d overlays -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
