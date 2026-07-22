#!/usr/bin/env python
"""Supplementary statistics computed from the released result files.

Computes, from the released result files only:
  - patient-rule (E6) dividends against the plain-EDD dedicated baseline,
    per structure and crew multiplier, plus the patient capture ratios;
  - the ablation (pair-attention) class's pooled seed-mean per Gate P
    scope against the best ranked rule;
  - Frame U policy-versus-best-rule at U >= 1.1, both eta values;
  - the m = 1.0 full-envelope dividend (units and % of L0);
  - trades per verdict campus (K);
  - an instance-cluster bootstrap CI for the fixed-EDD capture ratio
    (resampling base instances within campus, B = 10,000, seed 20260718,
    the same convention as cluster_stats_out.json).

Usage: PYTHONPATH=.:vendor python notes/supplementary/dividend_stats.py
Writes notes/supplementary/dividend_stats_out.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

import analysis.gates as G                                   # noqa: E402

RANKED = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
BOOT_B, BOOT_SEED = 10_000, 20260718


def read(family):
    return pd.read_csv(RES / family / "results.csv")


def policy_pools(df):
    mlp = sorted({m for m in df.method.unique() if isinstance(m, str)
                  and m.startswith("v2mlp") and 301 <= int(m[5:]) <= 310})
    attn = sorted({m for m in df.method.unique() if isinstance(m, str)
                   and m.startswith("v2attn") and 401 <= int(m[6:]) <= 410})
    return mlp, attn


def pool_seed_mean(d, methods):
    c = d[d.method.isin(methods)].copy()
    c["cfg"] = G.config_key(c)
    return float(c.groupby("method").twt.mean().mean())


def best_rule(d):
    return min(float(d[d.method == r].twt.mean()) for r in RANKED)


def main():
    out = {}
    t1 = G.expand_l0(read("tier1"))
    mlp, attn = policy_pools(t1)

    # ---- patient dividends (EDD-fixed baseline) ------------------------
    e6 = read("e6_patient")
    pat = {}
    for m in (0.6, 0.8):
        l0 = float(t1[(t1.m == m) & (t1.eta == 0.8)
                      & (t1.structure == "dedicated")
                      & (t1.method == "edd")].twt.mean())
        row = {"l0_edd": l0}
        for st in ("chain", "full"):
            pe = float(e6[(e6.m == m) & (e6.structure == st)
                          & (e6.method == "edd_patient")].twt.mean())
            row["div_patient_%s" % st] = l0 - pe
        row["capture_patient"] = (row["div_patient_chain"]
                                  / row["div_patient_full"])
        pat["m%.1f" % m] = row
    out["patient_dividends"] = pat

    # ---- ablation class per Gate P scope -------------------------------
    flex = t1[(t1.structure.isin(["chain", "full"]))
              & (t1.m.isin([0.6, 0.8]))]
    flex = flex[(flex.structure != "chain") | (flex.phi == 1.0)]
    scopes = {"pooled": flex, "m0.8": flex[flex.m == 0.8],
              "m0.6": flex[flex.m == 0.6]}
    out["attn_scopes"] = {
        k: {"attn": pool_seed_mean(v, attn), "best_rule": best_rule(v)}
        for k, v in scopes.items()}

    # ---- Frame U policy vs best rule at U >= 1.1 ------------------------
    e3 = G.expand_l0(read("e3"))
    m3, _ = policy_pools(e3)
    fu = {}
    for eta in (1.0, 0.8):
        d = e3[(e3.u_target >= 1.1) & (e3.eta == eta)]
        br, pol = best_rule(d), pool_seed_mean(d, m3)
        fu["eta%.1f" % eta] = {"best_rule": br, "policy": pol,
                               "gap_pct": 100 * (pol - br) / br}
    out["frameu_policy"] = fu

    # ---- m = 1.0 envelope dividend --------------------------------------
    pools = {r: [r] for r in RANKED + ["random"]}
    if mlp:
        pools["pm"] = mlp
    if attn:
        pools["pa"] = attn
    sub = t1[(t1.m == 1.0) & (t1.eta == 1.0)]
    v_l0, _ = G.twt_best(sub[sub.structure == "dedicated"], pools)
    v_f, _ = G.twt_best(sub[sub.structure == "full"], pools)
    out["m10"] = {"l0": v_l0, "full": v_f, "delta": v_l0 - v_f,
                  "delta_pct": 100 * (v_l0 - v_f) / v_l0}

    # ---- trades per verdict campus --------------------------------------
    from overlays.build import load_crews
    cap = os.environ.get(
        "FMWOS_Y1_CAP",
        str(ROOT.parent / "FM-Scheduling/results/p1_calib/capacity.csv"))
    out["trades"] = {str(c): len(load_crews(cap, c)) for c in (5, 9, 10, 12)}

    # ---- fixed-EDD capture ratio, instance-cluster bootstrap ------------
    fam = t1[(t1.m == 0.6) & (t1.eta == 1.0) & (t1.method == "edd")]
    fam = fam[(fam.structure != "chain") | (fam.phi == 1.0)]
    piv = {}
    for st in ("dedicated", "chain", "full"):
        s = fam[fam.structure == st].copy()
        # Base-instance key: the same replayed order stream under the
        # three overlays (EDD is deterministic, one row per instance).
        s["base"] = (s.campus.astype(str) + "|" + s["size"].astype(str)
                     + "|" + s.track.astype(str) + "|"
                     + s.instance_id.astype(str))
        piv[st] = s.set_index("base")[["campus", "twt"]]
    common = piv["dedicated"].index
    for st in ("chain", "full"):
        common = common.intersection(piv[st].index)
    camp = piv["dedicated"].loc[common, "campus"].to_numpy()
    v = {st: piv[st].loc[common, "twt"].to_numpy()
         for st in ("dedicated", "chain", "full")}
    point = ((v["dedicated"] - v["chain"]).mean()
             / (v["dedicated"] - v["full"]).mean())
    rng = np.random.default_rng(BOOT_SEED)
    stats = []
    idx_by_c = {c: np.flatnonzero(camp == c) for c in np.unique(camp)}
    for _ in range(BOOT_B):
        take = np.concatenate([
            rs[rng.integers(0, len(rs), len(rs))]
            for rs in idx_by_c.values()])
        num = (v["dedicated"][take] - v["chain"][take]).mean()
        den = (v["dedicated"][take] - v["full"][take]).mean()
        stats.append(num / den)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    out["edd_rho"] = {"point": float(point), "ci": [float(lo), float(hi)],
                      "n_instances": int(len(common)), "B": BOOT_B,
                      "seed": BOOT_SEED}

    dst = Path(__file__).with_name("dividend_stats_out.json")
    json.dump(out, open(dst, "w"), indent=1, sort_keys=True)
    print("wrote", dst)
    for k, val in out.items():
        print(k, json.dumps(val)[:150])


if __name__ == "__main__":
    main()
