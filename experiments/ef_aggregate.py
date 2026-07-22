"""E-F aggregation: after the 3-seed x3600 probe finishes, evaluate each probe
seed's dev_primary checkpoint (best.pt) AND its dev_full checkpoint
(best_full.pt) on the penalized-FULL cells and the Gate P scope, compare against
the released pool, summarize dev curves, and write results/train_probe/
summary.json. Read-only apart from summary.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from analysis import gates  # noqa: E402
from experiments.eval_ckpt_full import evaluate  # noqa: E402

PROBE = ROOT / "results/train_probe"
TIER1 = pd.read_parquet(ROOT / "results/tier1/results.parquet")
MLP = ["v2mlp%d" % s for s in range(301, 311)]
SEEDS = [901, 902, 903]
RULES = gates.RANKED + ["random"]


def _cfgpool(df, keys=("instance_id", "m")):
    d = df.copy()
    d["cfg"] = d[list(keys)].astype(str).agg("|".join, axis=1)
    return d.groupby("cfg").twt.mean()


def released_full():
    out = {}
    for m in (0.6, 0.8):
        cell = TIER1[(TIER1.structure == "full") & (TIER1.eta == 0.8)
                     & (TIER1.m == m)]
        pol = float(_cfgpool(cell[cell.method.isin(MLP)]).mean())
        seedmeans = [float(cell[cell.method == mm].twt.mean()) for mm in MLP]
        out["m%.1f" % m] = {
            "policy_pool": round(pol, 2),
            "per_seed_mean": round(float(np.mean(seedmeans)), 2),
            "per_seed_std": round(float(np.std(seedmeans)), 2),
            "random": round(float(cell[cell.method == "random"].twt.mean()), 2),
            "edd": round(float(cell[cell.method == "edd"].twt.mean()), 2)}
    return out


def curve_summary(seed):
    c = pd.read_csv(PROBE / ("v2mlp%d" % seed) / "curves.csv")
    ev = c[(c["update"] % 20 == 0) | (c["update"] == c["update"].max())]
    dp = ev["dev_primary"].values
    ups = ev["update"].values
    bi = int(np.argmin(dp))
    df = ev["dev_full"].values
    return {
        "updates": int(c["update"].max() + 1),
        "best_primary": round(float(dp.min()), 2),
        "best_primary_update": int(ups[bi]),
        "final_primary": round(float(dp[-1]), 2),
        "final_minus_best": round(float(dp[-1] - dp.min()), 2),
        "improving_at_end": bool(ups[bi] >= 0.8 * ups[-1]),
        "best_full": round(float(df.min()), 2),
        "best_full_update": int(ups[int(np.argmin(df))]),
        "final_full": round(float(df[-1]), 2),
        "median_s_per_update": round(
            float(c[c["update"] > 0].seconds.median()), 3),
    }


def eval_pool(which, scope, workers=10):
    """which in {'best','best_full'}; returns concatenated probe rows."""
    fname = "best.pt" if which == "best" else "best_full.pt"
    frames = []
    for s in SEEDS:
        ck = PROBE / ("v2mlp%d" % s) / fname
        df = evaluate(str(ck), "probe%d_%s" % (s, which), scope, workers)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def full_numbers(rows):
    out = {}
    for m in (0.6, 0.8):
        sub = rows[(rows.structure == "full") & (rows.eta == 0.8)
                   & (rows.m == m)]
        pool = float(_cfgpool(sub).mean())
        seedmeans = [float(sub[sub.method == mm].twt.mean())
                     for mm in sub.method.unique()]
        out["m%.1f" % m] = {
            "policy_pool": round(pool, 2),
            "per_seed_mean": round(float(np.mean(seedmeans)), 2),
            "per_seed_std": round(float(np.std(seedmeans)), 2),
            "per_seed": {mm: round(float(sub[sub.method == mm].twt.mean()), 2)
                         for mm in sorted(sub.method.unique())},
            "infeasible": int((sub.validator_ok == 0).sum())}
    return out


def gatep_numbers(gatep_rows):
    """Combine tier1 rules with probe policy rows, run gate_p."""
    flex = TIER1[TIER1.method.isin(RULES)
                 & TIER1.structure.isin(("chain", "full"))
                 & TIER1.m.isin((0.6, 0.8)) & TIER1.eta.isin((1.0, 0.8))]
    cols = ["instance_id", "structure", "eta", "m", "method", "twt"]
    probe = gatep_rows.copy()
    probe_methods = sorted(probe.method.unique())
    df = pd.concat([flex[cols], probe[cols]], ignore_index=True)
    res = gates.gate_p(df, probe_methods, probe_methods)
    out = {}
    for sc in ("pooled", "m08", "m06"):
        r = res[sc]
        out[sc] = {"policy_pool": round(r["policy_pooled_mean"], 2),
                   "best_rule": r["best_rule"],
                   "best_rule_mean": round(r["best_rule_mean"], 2),
                   "seeds_beating_best_rule": r["seeds_beating_best_rule"],
                   "n_seeds": r["n_seeds"], "pass": r["pass"]}
    return out


def main():
    for s in SEEDS:
        if not (PROBE / ("v2mlp%d" % s) / "done.json").exists():
            raise SystemExit("seed %d not finished (no done.json); abort" % s)
    summary = {"released_pool": {
        "full_penalized": released_full(),
        "gatep": {"pooled_policy": 326.67, "pooled_best_rule_edd": 319.15,
                  "seeds_beating": 1, "verdict": "FAIL (as registered)"}},
        "probe": {"updates": 3600, "seeds": SEEDS, "curves": {}}}
    for s in SEEDS:
        summary["probe"]["curves"]["seed%d" % s] = curve_summary(s)

    for which in ("best", "best_full"):
        full_rows = eval_pool(which, "full08")
        gp_rows = eval_pool(which, "gatep")
        summary["probe"][which] = {
            "selection": ("dev_primary (as released)" if which == "best"
                          else "dev_full (selection-mismatch test)"),
            "full_penalized": full_numbers(full_rows),
            "gatep": gatep_numbers(gp_rows)}
        full_rows.to_parquet(PROBE / ("probe_full08_%s.parquet" % which))
        gp_rows.to_parquet(PROBE / ("probe_gatep_%s.parquet" % which))

    with open(PROBE / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
