#!/usr/bin/env python
"""Record the verdict class (protocol log 4) BEFORE any test evaluation.

Pre-registered rule: the class (pair-MLP vs pair-attention) with the LOWER
MEAN PER-SEED DEVELOPMENT MINIMUM of the primary dev signal becomes the
Gate P pool. Reads results/train/{mlp,attn}_seed*/curves.csv, writes
results/gates/verdict_class.json, and appends a dated amendment to
protocol/Y2_protocol.md. Refuses to run if any main-pool seed is missing
or unfinished (all 20 done.json required).

Usage: PYTHONPATH=. python experiments/verdict_class.py [--force]
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "results/train"
OUT = ROOT / "results/gates"

MLP_SEEDS = list(range(301, 311))
ATTN_SEEDS = list(range(401, 411))


def per_seed_minima(arch, seeds):
    out = {}
    for s in seeds:
        d = TRAIN / ("%s_seed%d" % (arch, s))
        if not (d / "done.json").exists():
            return None, "missing done.json for %s seed %d" % (arch, s)
        c = pd.read_csv(d / "curves.csv")
        out[s] = float(c["dev_primary"].min())
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="proceed even if some seeds are unfinished")
    args = ap.parse_args()

    mlp, err1 = per_seed_minima("mlp", MLP_SEEDS)
    attn, err2 = per_seed_minima("attn", ATTN_SEEDS)
    if (err1 or err2) and not args.force:
        sys.exit("NOT READY: %s" % (err1 or err2))

    mean_mlp = sum(mlp.values()) / len(mlp) if mlp else float("inf")
    mean_attn = sum(attn.values()) / len(attn) if attn else float("inf")
    verdict = "mlp" if mean_mlp <= mean_attn else "attn"

    OUT.mkdir(parents=True, exist_ok=True)
    rec = {
        "date": datetime.date.today().isoformat(),
        "rule": "lower mean per-seed dev minimum on the primary signal "
                "(protocol log 4, pre-registered)",
        "mean_dev_min_mlp": mean_mlp,
        "mean_dev_min_attn": mean_attn,
        "per_seed_mlp": mlp,
        "per_seed_attn": attn,
        "verdict_class": verdict,
    }
    with open(OUT / "verdict_class.json", "w") as f:
        json.dump(rec, f, indent=2)

    proto = ROOT / "protocol/Y2_protocol.md"
    txt = open(proto).read()
    if "Amendment A1" not in txt:
        txt = txt.replace(
            "## Amendments\n\n(none yet)",
            "## Amendments\n\n"
            "### Amendment A1 (%s): verdict class recorded\n"
            "Applying the pre-registered rule of Section 4 to the completed "
            "training pools, BEFORE any test-set evaluation of either "
            "class: mean per-seed development minimum (primary signal) "
            "pair-MLP = %.4f, pair-attention = %.4f. The verdict class is "
            "**pair-%s**; the other class is reported as an ablation. "
            "Per-seed minima in results/gates/verdict_class.json.\n"
            % (rec["date"], mean_mlp, mean_attn, verdict))
        open(proto, "w").write(txt)
    print("verdict class: pair-%s (mlp %.4f vs attn %.4f)"
          % (verdict, mean_mlp, mean_attn))
    print("amendment appended to protocol log; COMMIT before test eval.")


if __name__ == "__main__":
    main()
