#!/usr/bin/env python
"""Same-budget sparse-topology panel: capture ratios and structure metrics.

Computes, from released result files only (tier1, tier2, e7_topology):
  - pooled rules-envelope TWT (8 methods incl. Random) and fixed-EDD TWT for
    every structure at m = 0.6, eta in {1.0, 0.8}: L0, CHAIN(1.0), chain_adj,
    GEN, FULL, and the sparse controls pairs / star / feas / rand1 x 3;
  - envelope and fixed-EDD capture ratios per structure;
  - paired chain-vs-pairs EDD difference with an instance-cluster bootstrap
    CI (resampling base instances within campus, B = 10,000, seed 20260718)
    and a paired two-sided Wilcoxon p-value, at both eta values;
  - trade-level topology descriptors per control (coverage share, weak
    components, max in-degree, workload-weighted coverage), pooled over the
    verdict campuses from sparse_config_check.json + the capacity table;
  - the policy-undercut check: pooled policy seed-means never beat the rules
    envelope in any contended flexible cell, so the rules envelope is the
    operative dividend estimator for structures without policy runs.

Usage: PYTHONPATH=.:vendor python notes/supplementary/sparse_topology_stats.py
Writes notes/supplementary/sparse_topology_out.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

import analysis.gates as G                                   # noqa: E402
from overlays.build import load_crews                        # noqa: E402

RANKED = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
ENVELOPE = RANKED + ["random"]
BOOT_B, BOOT_SEED = 10_000, 20260718
CAMPUSES = (5, 9, 10, 12)
RAND1_SEEDS = (20260901, 20260902, 20260903)


def read(family):
    return pd.read_csv(RES / family / "results.csv")


def base_key(s):
    return (s.campus.astype(str) + "|" + s["size"].astype(str) + "|"
            + s.track.astype(str) + "|" + s.instance_id.astype(str))


def envelope_twt(d, ids):
    d = d[d.method.isin(ENVELOPE) & d.instance_id.isin(ids)]
    g = d.groupby("method").twt.mean()
    return float(g.min()), str(g.idxmin())


def edd_twt(d, ids):
    d = d[(d.method == "edd") & d.instance_id.isin(ids)]
    return float(d.twt.mean())


def paired_boot_ci(a, b, camp):
    """CI for mean(a - b), resampling base instances within campus."""
    rng = np.random.default_rng(BOOT_SEED)
    idx_by_c = {c: np.flatnonzero(camp == c) for c in np.unique(camp)}
    stats = []
    for _ in range(BOOT_B):
        take = np.concatenate([rs[rng.integers(0, len(rs), len(rs))]
                               for rs in idx_by_c.values()])
        stats.append((a[take] - b[take]).mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    out = {}
    t1 = G.expand_l0(read("tier1"))
    t2 = G.expand_l0(read("tier2"))
    # The e7 results.csv also carries held-out campuses 1/2 and m = 0.8 rows
    # (transfer closure); this is a verdict-scope analysis, so restrict to
    # the verdict campuses at m = 0.6 before anything else.
    e7 = read("e7_topology")
    e7 = e7[e7.campus.isin(CAMPUSES) & (e7.m == 0.6)]

    ids = sorted(set(e7[(e7.variant == "pairs")
                        & (e7.eta == 1.0)].instance_id))
    assert len(ids) == 763, "expected 763 verdict replay instances"
    out["n_instances"] = len(ids)

    sane = e7[e7.variant.isin(["pairs", "star", "feas"]
                              + ["rand1_%d" % s for s in RAND1_SEEDS])]
    assert int((sane.validator_ok == 0).sum()) == 0, "infeasible sparse rows"

    # ---- per-structure pooled TWT and capture, both estimators ----------
    for eta in (1.0, 0.8):
        block = {}
        srcs = {
            "l0": t1[(t1.m == 0.6) & (t1.eta == eta)
                     & (t1.structure == "dedicated")],
            "chain": t1[(t1.m == 0.6) & (t1.eta == eta)
                        & (t1.structure == "chain") & (t1.phi == 1.0)],
            "full": t1[(t1.m == 0.6) & (t1.eta == eta)
                       & (t1.structure == "full")],
            "gen": t2[(t2.m == 0.6) & (t2.eta == eta)
                      & (t2.structure == "generalist")],
            "chain_adj": e7[(e7.variant == "chain_adj") & (e7.eta == eta)],
            "pairs": e7[(e7.variant == "pairs") & (e7.eta == eta)],
            "star": e7[(e7.variant == "star") & (e7.eta == eta)],
            "feas": e7[(e7.variant == "feas") & (e7.eta == eta)],
        }
        if eta == 1.0:
            for s in RAND1_SEEDS:
                srcs["rand1_%d" % s] = e7[e7.variant == "rand1_%d" % s]
        twt_env, twt_edd = {}, {}
        for name, d in srcs.items():
            if len(d) == 0:
                continue
            twt_env[name], best = envelope_twt(d, ids)
            twt_edd[name] = edd_twt(d, ids)
            block[name] = {"twt_envelope": twt_env[name],
                           "best_method": best, "twt_edd": twt_edd[name]}
        d_env = twt_env["l0"] - twt_env["full"]
        d_edd = twt_edd["l0"] - twt_edd["full"]
        for name in block:
            if name == "l0":
                continue
            block[name]["capture_envelope"] = (
                (twt_env["l0"] - twt_env[name]) / d_env)
            block[name]["capture_edd"] = (
                (twt_edd["l0"] - twt_edd[name]) / d_edd)
        block["delta_full_envelope"] = d_env
        block["delta_full_edd"] = d_edd
        out["eta%.1f" % eta] = block

    # ---- paired chain vs pairs (fixed EDD), CI + Wilcoxon ---------------
    cmp_out = {}
    for eta in (1.0, 0.8):
        ch = t1[(t1.m == 0.6) & (t1.eta == eta) & (t1.structure == "chain")
                & (t1.phi == 1.0) & (t1.method == "edd")].copy()
        pr = e7[(e7.variant == "pairs") & (e7.eta == eta)
                & (e7.method == "edd")].copy()
        ch["base"], pr["base"] = base_key(ch), base_key(pr)
        m = ch.set_index("base")[["campus", "twt"]].join(
            pr.set_index("base")[["twt"]], rsuffix="_pairs", how="inner")
        a = m.twt_pairs.to_numpy()          # pairs
        b = m.twt.to_numpy()                # chain
        lo, hi = paired_boot_ci(a, b, m.campus.to_numpy())
        try:
            p = float(wilcoxon(a, b).pvalue)
        except ValueError:                  # all-zero differences
            p = 1.0
        cmp_out["eta%.1f" % eta] = {
            "mean_diff_pairs_minus_chain": float((a - b).mean()),
            "ci": [lo, hi], "wilcoxon_p": p, "n": int(len(m))}
    out["pairs_vs_chain_edd"] = cmp_out

    # ---- topology descriptors pooled over campuses ----------------------
    chk = json.load(open(RES / "e7_topology" / "sparse_config_check.json"))
    cap = os.environ.get(
        "FMWOS_Y1_CAP",
        str(ROOT.parent / "FM-Scheduling/results/p1_calib/capacity.csv"))
    vols = {c: {r["trade"]: r["volume"] for r in load_crews(cap, c)}
            for c in CAMPUSES}
    desc = {}
    for variant in ["pairs", "star", "feas"] + [
            "rand1_%d" % s for s in RAND1_SEEDS]:
        rows = []
        for c in CAMPUSES:
            d = chk["c%02d" % c][variant]
            v = vols[c]
            covered = [t for t in d["in_degree"] if d["in_degree"][t] > 0]
            wcov = sum(v[t] for t in covered) / sum(v.values())
            rows.append({"coverage": d["coverage_share"], "wcov": wcov,
                         "components": d["weak_components"],
                         "indeg_max": d["in_degree_max"]})
        desc[variant] = {
            "coverage_mean": float(np.mean([r["coverage"] for r in rows])),
            "coverage_range": [float(min(r["coverage"] for r in rows)),
                               float(max(r["coverage"] for r in rows))],
            "wcov_mean": float(np.mean([r["wcov"] for r in rows])),
            "components_range": [int(min(r["components"] for r in rows)),
                                 int(max(r["components"] for r in rows))],
            "indeg_max_range": [int(min(r["indeg_max"] for r in rows)),
                                int(max(r["indeg_max"] for r in rows))]}
    out["descriptors"] = desc

    # ---- policy-undercut check ------------------------------------------
    mlp = sorted({m for m in t1.method.unique() if isinstance(m, str)
                  and m.startswith("v2mlp")})
    attn = sorted({m for m in t1.method.unique() if isinstance(m, str)
                   and m.startswith("v2attn")})
    und = {}
    for eta in (1.0, 0.8):
        for st in ("chain", "full"):
            d = t1[(t1.m == 0.6) & (t1.eta == eta) & (t1.structure == st)]
            d = d[(d.structure != "chain") | (d.phi == 1.0)]
            env, _ = envelope_twt(d, ids)
            pol = min(
                float(d[d.method.isin(mlp)].groupby("method").twt
                      .mean().mean()),
                float(d[d.method.isin(attn)].groupby("method").twt
                      .mean().mean()))
            und["%s_eta%.1f" % (st, eta)] = {
                "envelope": env, "best_policy_class_mean": pol,
                "policy_undercuts": bool(pol < env)}
    out["policy_undercut_check"] = und

    dst = Path(__file__).with_name("sparse_topology_out.json")
    json.dump(out, open(dst, "w"), indent=1, sort_keys=True)
    print("wrote", dst)
    for eta in ("eta1.0", "eta0.8"):
        b = out[eta]
        for k in sorted(b):
            if isinstance(b[k], dict) and "capture_envelope" in b[k]:
                print("%s %-16s env %.3f edd %.3f" % (
                    eta, k, b[k]["capture_envelope"], b[k]["capture_edd"]))
    print("pairs-vs-chain:", json.dumps(out["pairs_vs_chain_edd"]))
    print("undercut:", json.dumps(out["policy_undercut_check"], indent=0))


if __name__ == "__main__":
    main()
