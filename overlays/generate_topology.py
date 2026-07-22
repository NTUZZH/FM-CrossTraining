#!/usr/bin/env python
"""Materialise the E-B topology overlays as JSON assets.

Writes overlays/generated_overlays/<overlay_id>.json + index.csv for:
  * chain_adj   : verdict campuses x eta {1.0, 0.8} x m 0.6
  * perm<seed>  : verdict campuses x eta 1.0 x m 0.6, seeds 20260801..810
  * tsel<seed>  : verdict campuses x eta 1.0 x m 0.6, CHAIN(0.5),
                  seeds 20260821..823

Separate from the released overlays/generated/ tree so nothing here can be
swept into a released-benchmark analysis. Construction reuses
overlays.topology_overlays (byte-for-byte build.py conventions).

Usage: PYTHONPATH=.:vendor python overlays/generate_topology.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from overlays.build import load_crews, save_overlay, build_overlay  # noqa
from overlays import topology_overlays as rt                        # noqa

Y1 = Path(os.environ.get("FMWOS_Y1_ROOT", ROOT.parent / "FM-Scheduling"))
CAP = str(Y1 / "results/p1_calib/capacity.csv")
OUT = ROOT / "overlays" / "generated_overlays"

CAMPUSES = [5, 9, 10, 12]
M = 0.6
CHAIN_ADJ_ETAS = [1.0, 0.8]
PERM_SEEDS = list(range(20260801, 20260811))
TSEL_SEEDS = [20260821, 20260822, 20260823]


def _check(ov, ref, kind):
    assert ov["headcount"] == ref["headcount"], (kind, "headcount")
    assert ov["budget_B"] == ref["budget_B"], (kind, "budget_B", ov["overlay_id"])
    p1 = {t["id"]: t["primary"] for t in ov["technicians"]}
    p2 = {t["id"]: t["primary"] for t in ref["technicians"]}
    assert p1 == p2, (kind, "primary")


def main():
    rows = []
    for c in CAMPUSES:
        crews = load_crews(CAP, c)
        adj = rt.adjacency_order(crews)
        for eta in CHAIN_ADJ_ETAS:
            ref = build_overlay(c, crews, "chain", 1.0, eta, M)
            ov = rt.build_chain_variant(c, crews, 1.0, eta, M, order=adj,
                                        struct_label="chain_adj",
                                        variant="chain_adj")
            _check(ov, ref, "chain_adj")
            save_overlay(ov, OUT)
            rows.append((ov, "chain_adj"))
        for s in PERM_SEEDS:
            ref = build_overlay(c, crews, "chain", 1.0, 1.0, M)
            ov = rt.build_chain_variant(c, crews, 1.0, 1.0, M, perm_seed=s,
                                        struct_label="chain", variant="perm%d" % s)
            _check(ov, ref, "perm")
            save_overlay(ov, OUT)
            rows.append((ov, "perm%d" % s))
        for s in TSEL_SEEDS:
            ref = build_overlay(c, crews, "chain", 0.5, 1.0, M)
            ov = rt.build_chain_variant(c, crews, 0.5, 1.0, M, tech_seed=s,
                                        struct_label="chain", variant="tsel%d" % s)
            _check(ov, ref, "tsel")
            save_overlay(ov, OUT)
            rows.append((ov, "tsel%d" % s))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["overlay_id", "variant", "campus", "structure", "phi",
                    "eta", "crew_multiplier", "headcount", "budget_B",
                    "selection", "chain_order"])
        for ov, var in rows:
            w.writerow([ov["overlay_id"], var, ov["campus"], ov["structure"],
                        ov["phi"], ov["eta"], ov["crew_multiplier"],
                        ov["headcount"], ov["budget_B"], ov["selection"],
                        " ".join(ov["chain_order"])])
    print("wrote %d topology overlays -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
