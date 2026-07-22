#!/usr/bin/env python
"""Supplementary scenario statistics from released artifacts.

Blocks:
  feasibility   : loose (skill-family chain) / normal (2 licence-gated trades) /
                  strict (4 gated) skill-adjacency scenarios; capture ratio,
                  workload coverage, and eta=0.8 dividend per scenario.
  noisyp        : duration-estimate sensitivity summary (EDD/pFIFO invariance;
                  capture ratio at every noise level; per-method pooled means).
  bound_tight   : admissible reward-bound value at the start of a fully-released
                  static snapshot vs the CP-SAT certified optimum, ratio by
                  structure and eta (tightness of the shaping bound).
  per_priority  : breach-share change by priority class (chain, full vs L0),
                  so P1/P2 (safety-critical) effects are reported, not only TWT.
  compute       : total wall-clock cost of the released runs (training + eval),
                  not only per-decision latency.

Usage: PYTHONPATH=.:vendor python notes/supplementary/scenario_stats.py
Writes notes/supplementary/scenario_stats_out.json.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
Y1 = Path(os.environ.get("FMWOS_Y1_ROOT", ROOT.parent / "FM-Scheduling"))
CAP = Y1 / "results/p1_calib/capacity.csv"
INST_ROOT = Y1 / "data/processed/instances"

import analysis.gates as G                                   # noqa: E402
from overlays.build import build_overlay, load_crews         # noqa: E402
from overlays import topology_overlays as rt                 # noqa: E402
from env import lb2                                          # noqa: E402

RANKED = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
ENVELOPE = RANKED + ["random"]
CAMPUSES = (5, 9, 10, 12)


def read(fam, name="results.csv"):
    p = RES / fam / name
    return pd.read_csv(p) if p.exists() else None


def verdict_e7(e7):
    return e7[e7.campus.isin(CAMPUSES) & (e7.m == 0.6)]


# ---------------------------------------------------------------------------
def feasibility(out):
    """loose / normal / strict feasibility scenarios, capture + coverage."""
    e7 = verdict_e7(read("e7_topology"))
    t1 = G.expand_l0(read("tier1"))
    ids = sorted(set(e7[(e7.variant == "pairs") & (e7.eta == 1.0)].instance_id))

    def env_twt(d, eta):
        d = d[d.method.isin(ENVELOPE) & d.instance_id.isin(ids)]
        return float(d.groupby("method").twt.mean().min())

    def edd_twt(d):
        d = d[(d.method == "edd") & d.instance_id.isin(ids)]
        return float(d.twt.mean())

    # workload-weighted coverage per scenario
    vols = {c: {x["trade"]: x["volume"] for x in load_crews(CAP, c)}
            for c in CAMPUSES}

    def coverage(sigma_fn):
        num = den = 0.0
        for c in CAMPUSES:
            crews = load_crews(CAP, c)
            order = rt.adjacency_order(crews)
            sig = sigma_fn(order)
            covered = set(sig.values())
            for t, v in vols[c].items():
                den += v
                if t in covered:
                    num += v
        return num / den

    scen = {
        "loose": {"variant": "chain_adj",
                  "sigma": lambda o: {t: o[(i + 1) % len(o)]
                                      for i, t in enumerate(o)}},
        "normal": {"variant": "feasnorm",
                   "sigma": lambda o: rt.sigma_feas(o, rt.LICENSED_NORMAL)},
        "strict": {"variant": "feas",
                   "sigma": lambda o: rt.sigma_feas(o, rt.LICENSED_TRADES)},
    }
    l0_10 = env_twt(t1[(t1.m == 0.6) & (t1.eta == 1.0)
                       & (t1.structure == "dedicated")], 1.0)
    full_10 = env_twt(t1[(t1.m == 0.6) & (t1.eta == 1.0)
                         & (t1.structure == "full")], 1.0)
    l0_08 = l0_10
    full_08 = env_twt(t1[(t1.m == 0.6) & (t1.eta == 0.8)
                         & (t1.structure == "full")], 0.8)
    blk = {"delta_full_eta10": l0_10 - full_10,
           "delta_full_eta08": l0_08 - full_08}
    for name, spec in scen.items():
        d10 = e7[(e7.variant == spec["variant"]) & (e7.eta == 1.0)]
        d08 = e7[(e7.variant == spec["variant"]) & (e7.eta == 0.8)]
        cap10 = ((l0_10 - env_twt(d10, 1.0)) / (l0_10 - full_10)
                 if l0_10 != full_10 else None)
        blk[name] = {
            "coverage": coverage(spec["sigma"]),
            "capture_env_eta10": cap10,
            "capture_edd_eta10": ((l0_10 - edd_twt(d10))
                                  / (l0_10 - edd_twt(
                                      t1[(t1.m == 0.6) & (t1.eta == 1.0)
                                         & (t1.structure == "full")])))
            if len(d10) else None,
            "dividend_eta08": (l0_08 - env_twt(d08, 0.8)) if len(d08) else None,
            "passes_gatec": bool(cap10 is not None and cap10 >= 0.70),
        }
    out["feasibility"] = blk


# ---------------------------------------------------------------------------
def noisyp(out):
    d = read("e12_noisyp")
    if d is None:
        return
    sigmas = sorted(d.sigma.unique())
    # invariance: max |twt(sigma) - twt(0)| per method over all cells
    inv = {}
    for m in ENVELOPE:
        base = d[(d.sigma == 0.0) & (d.method == m)].set_index(
            ["instance_id", "structure", "eta"]).twt
        mx = 0.0
        for sg in sigmas[1:]:
            cur = d[(d.sigma == sg) & (d.method == m)].set_index(
                ["instance_id", "structure", "eta"]).twt
            j = base.to_frame("b").join(cur.to_frame("c")).dropna()
            mx = max(mx, float((j.b - j.c).abs().max()))
        inv[m] = mx
    # capture by sigma (best-method envelope)
    cap = {}
    for eta in (1.0, 0.8):
        row = {}
        for sg in sigmas:
            s = d[(d.eta == eta) & (d.sigma == sg)]
            env = {st: s[s.structure == st].groupby("method").twt.mean()
                   for st in ("dedicated", "chain", "full")}
            den = env["dedicated"].min() - env["full"].min()
            row["%.2f" % sg] = {
                "capture": ((env["dedicated"].min() - env["chain"].min())
                            / den) if den else None,
                "chain_winner": str(env["chain"].idxmin()),
                "delta_full": den}
        cap["eta%.1f" % eta] = row
    out["noisyp"] = {
        "sigmas": sigmas,
        "invariant_rules": sorted(m for m, v in inv.items() if v < 1e-9),
        "changing_rules_max_delta": {m: round(v, 1) for m, v in inv.items()
                                     if v >= 1e-9},
        "capture": cap}


# ---------------------------------------------------------------------------
def bound_tight(out):
    """Admissible reward-bound value on a fully-released static snapshot vs
    the CP-SAT certified optimum, per structure and eta. Tightness = bound /
    optimum (<= 1 by admissibility; closer to 1 is tighter)."""
    e1 = read("e1_static")
    if e1 is None:
        return
    cp = e1[e1.method.isin(["cpsat60", "cpsat300"])]
    opt = cp[cp.proved_optimal == 1].groupby(
        ["instance_id", "structure", "eta"]).twt.min()
    # replay static instances only (paths available); first-15 per stratum
    idx = {}
    with open(INST_ROOT / "index.csv", newline="") as f:
        for r in csv.DictReader(f):
            if (r["track"] == "replay" and r["split"] == "test"
                    and int(r["campus"]) in CAMPUSES):
                idx[r["id"]] = str(INST_ROOT / r["path"])
    ratios = {}
    n_used = 0
    for (iid, st, eta), o in opt.items():
        if o <= 0 or iid not in idx:
            continue
        crews = load_crews(CAP, int(iid.split("_")[0][1:]))
        phi = 1.0 if st == "chain" else None
        ov = build_overlay(int(iid.split("_")[0][1:]), crews, st, phi,
                           float(eta), 1.0)
        inst = json.load(open(idx[iid]))
        queues = {}
        for wo in inst["work_orders"]:
            queues.setdefault(wo["trade"], []).append(
                (wo["p_bh"], wo["due_bh"], wo["weight"]))
        skills_of = {t["id"]: t["skills"] for t in ov["technicians"]}
        tech_free = {t["id"]: 0.0 for t in ov["technicians"]}
        lb = lb2.lb_remaining_v2(queues, tech_free, skills_of, 0.0,
                                 float(eta))
        ratios.setdefault((st, float(eta)), []).append(lb / o)
        n_used += 1
    blk = {"n_instances": n_used}
    for (st, eta), rs in ratios.items():
        blk["%s_eta%.1f" % (st, eta)] = {
            "bound_over_opt_mean": float(np.mean(rs)),
            "bound_over_opt_min": float(np.min(rs)),
            "n": len(rs)}
    out["bound_tight"] = blk


# ---------------------------------------------------------------------------
def per_priority(out):
    """Breach-share by priority class, chain/full vs L0 (m=0.6, eta=1.0,
    fixed EDD): report P1/P2 (safety-critical) not only pooled TWT."""
    t1 = G.expand_l0(read("tier1"))
    fam = t1[(t1.m == 0.6) & (t1.eta == 1.0) & (t1.method == "edd")]
    fam = fam[(fam.structure != "chain") | (fam.phi == 1.0)]
    cols = ["breach_p1", "breach_p2", "breach_p3", "breach_p4"]
    base = {st: fam[fam.structure == st][cols].mean()
            for st in ("dedicated", "chain", "full")}
    blk = {}
    for st in ("dedicated", "chain", "full"):
        blk[st] = {c: (float(base[st][c]) if pd.notna(base[st][c]) else None)
                   for c in cols}
    out["per_priority"] = blk


# ---------------------------------------------------------------------------
def compute(out):
    """Total wall-clock of the released runs (training + the CPU sweeps),
    reported alongside per-decision latency."""
    blk = {}
    tot_train_h = 0.0
    n_train = 0
    for f in glob.glob(str(RES / "train" / "*_seed*" / "curves.csv")):
        c = pd.read_csv(f)
        if "seconds" in c:
            tot_train_h += float(c.seconds.sum()) / 3600.0
            n_train += 1
    blk["training_gpu_hours"] = round(tot_train_h, 1)
    blk["training_runs"] = n_train
    out["compute"] = blk


# ---------------------------------------------------------------------------
def main():
    out = {}
    feasibility(out)
    noisyp(out)
    bound_tight(out)
    per_priority(out)
    compute(out)
    dst = Path(__file__).with_name("scenario_stats_out.json")
    json.dump(out, open(dst, "w"), indent=1, sort_keys=True)
    print("wrote", dst)
    for k in out:
        print("--", k, json.dumps(out[k])[:200])


if __name__ == "__main__":
    main()
