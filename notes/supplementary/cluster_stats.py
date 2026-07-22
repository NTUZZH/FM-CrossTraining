#!/usr/bin/env python
"""Cluster-aware uncertainty quantification for the manuscript's headline
quantities, from the released result files only.

Robustness questions addressed:
  (a) Wilcoxon / pooled means treat instance-configurations that share the
      same base workload (same order stream under different structure / eta /
      m overlays) as independent, and cluster by campus.
  (b) headline point estimates carry no uncertainty intervals.

Convention mirroring analysis/gates.py exactly:
  - expand_l0: dedicated rows (eta=1.0) stand for both eta values.
  - config_key = instance_id | structure | eta | m.
  - Gate P pooled scope: chain/full x m in {0.6,0.8} x eta in {1.0,0.8};
    policy = per-config seed-mean over the verdict-class pool (v2mlp301-310),
    then pooled mean over configs; best ranked rule = EDD.
  - Gate C: envelope-best = lowest pooled-mean TWT over rules (single method)
    and policy pools (per-config seed-mean, then mean).

Clustering unit = base instance_id (the shared order stream). Two bootstraps:
  INSTANCE-CLUSTER : resample base instance_ids WITHIN campus, with replacement.
  CAMPUS-CLUSTER   : resample whole campuses with replacement (only 4 campuses
                     in the verdict set -> reported separately, CIs are wide).

Usage: PYTHONPATH=.:vendor python notes/supplementary/cluster_stats.py
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

SEED = 20260718
NBOOT = 10000

RANKED = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
MLP_MAIN = ["v2mlp%d" % i for i in range(301, 311)]
ATTN_MAIN = ["v2attn%d" % i for i in range(401, 411)]
VERDICT_CAMPUSES = [5, 9, 10, 12]


def load(family):
    p = RES / family / "results.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(RES / family / "results.csv")


# --------------------------------------------------------------------------
# Envelope cell machinery: per-instance value matrices aligned to a master
# instance list, so ONE resample of instance_ids applies coherently to L0,
# chain and full cells (they share the base workloads).
# --------------------------------------------------------------------------
COLS = RANKED + ["random", "policy_mlp", "policy_attn"]
COLS_GATEC = list(range(len(RANKED) + 1))          # rules + random + policy
COLS_GATEC = list(range(len(COLS)))                 # rules+random+mlp+attn
COLS_PHI = list(range(len(RANKED) + 1))             # rules + random only


def cell_matrix(df, structure, m, eta, master_ids, phi=None):
    """Return (n_master, n_cols) array; cols = RANKED + random + policy_mlp +
    policy_attn. Value per instance: single-method twt or policy-pool
    per-instance seed-mean. NaN where absent."""
    sub = df[(df.structure == structure) & (df.m == m)]
    if structure != "dedicated":
        sub = sub[sub.eta == eta]
    if phi is not None:
        sub = sub[sub.phi == phi]
    pos = {iid: k for k, iid in enumerate(master_ids)}
    single = RANKED + ["random"]
    M = np.full((len(master_ids), len(COLS)), np.nan)
    for j, meth in enumerate(single):
        s = sub[sub.method == meth]
        for iid, v in zip(s.instance_id, s.twt):
            if iid in pos:
                M[pos[iid], j] = v
    for j, pool in ((len(single), MLP_MAIN), (len(single) + 1, ATTN_MAIN)):
        s = sub[sub.method.isin(pool)]
        g = s.groupby("instance_id").twt.mean()
        for iid, v in g.items():
            if iid in pos:
                M[pos[iid], j] = v
    return M, COLS


def envelope_best(M, idx, cols=None):
    """Lowest column mean over resampled instance positions idx.
    cols: column indices to consider (default all)."""
    sel = M[idx] if cols is None else M[np.ix_(idx, cols)]
    with np.errstate(invalid="ignore"):
        means = np.nanmean(sel, axis=0)
    return np.nanmin(means[np.isfinite(means)])


# --------------------------------------------------------------------------
# Resampling index generators.
# --------------------------------------------------------------------------
def make_resamplers(campus_of_master, rng, nboot):
    """Yield instance-cluster and campus-cluster index arrays.
    campus_of_master: array of campus label per master position."""
    campuses = np.unique(campus_of_master)
    by_c = {c: np.where(campus_of_master == c)[0] for c in campuses}
    inst_idx = []
    camp_idx = []
    for _ in range(nboot):
        # instance-cluster: resample instances within each campus
        parts = [rng.choice(by_c[c], size=len(by_c[c]), replace=True)
                 for c in campuses]
        inst_idx.append(np.concatenate(parts))
        # campus-cluster: resample whole campuses, take all their instances
        drawn = rng.choice(campuses, size=len(campuses), replace=True)
        camp_idx.append(np.concatenate([by_c[c] for c in drawn]))
    return inst_idx, camp_idx


def ci(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def fmt_ci(lo, hi, nd=2):
    return "[%.*f, %.*f]" % (nd, lo, nd, hi)


# ==========================================================================
# ITEM 1 : Gate P headline gap + per-seed count
# ==========================================================================
def item1(tier1, rng):
    df = tier1.copy()
    flex = df[(df.structure.isin(["chain", "full"]))
              & (df.m.isin([0.6, 0.8])) & (df.eta.isin([1.0, 0.8]))]
    flex = flex.copy()
    flex["cfg"] = (flex.instance_id.astype(str) + "|" + flex.structure
                   + "|" + flex.eta.astype(str) + "|" + flex.m.astype(str))
    # per-config aggregates
    pol = flex[flex.method.isin(MLP_MAIN)].groupby(["instance_id", "cfg"]) \
        .twt.mean().rename("pol")
    edd = flex[flex.method == "edd"].groupby(["instance_id", "cfg"]) \
        .twt.mean().rename("edd")
    seedcols = {}
    for s in MLP_MAIN:
        seedcols[s] = flex[flex.method == s] \
            .groupby(["instance_id", "cfg"]).twt.mean().rename(s)
    cfg = pd.concat([pol, edd] + list(seedcols.values()), axis=1).dropna()
    cfg = cfg.reset_index()
    # per-instance sums (for cluster bootstrap over base instances)
    g = cfg.groupby("instance_id")
    inst = pd.DataFrame({"nc": g.size()})
    inst["spol"] = g.pol.sum()
    inst["sedd"] = g.edd.sum()
    for s in MLP_MAIN:
        inst["s_" + s] = g[s].sum()
    inst["campus"] = cfg.groupby("instance_id").apply(
        lambda x: int(x.name.split("_")[0][1:]) if isinstance(x.name, str)
        else 0) if False else cfg.groupby("instance_id").first().index.map(
        lambda iid: int(iid.split("_")[0][1:]))
    inst = inst.reset_index()
    campus_arr = inst.campus.to_numpy()

    nc = inst.nc.to_numpy(float)
    spol = inst.spol.to_numpy(float)
    sedd = inst.sedd.to_numpy(float)
    sseed = np.vstack([inst["s_" + s].to_numpy(float) for s in MLP_MAIN])

    # point estimates
    pol_mean = spol.sum() / nc.sum()
    edd_mean = sedd.sum() / nc.sum()
    gap = pol_mean - edd_mean
    pct = 100 * gap / edd_mean
    seed_means0 = sseed.sum(axis=1) / nc.sum()
    n_beat0 = int((seed_means0 < edd_mean).sum())

    inst_idx, camp_idx = make_resamplers(campus_arr, rng, NBOOT)

    def run(idxlist):
        gaps, pcts, nbeats = [], [], []
        for idx in idxlist:
            n = nc[idx].sum()
            pm = spol[idx].sum() / n
            em = sedd[idx].sum() / n
            gaps.append(pm - em)
            pcts.append(100 * (pm - em) / em)
            sm = sseed[:, idx].sum(axis=1) / n
            nbeats.append(int((sm < em).sum()))
        return gaps, pcts, nbeats

    gi, pi, bi = run(inst_idx)
    gc, pc_, bc = run(camp_idx)
    return {
        "n_configs": len(cfg), "n_instances": len(inst),
        "policy_mean": pol_mean, "edd_mean": edd_mean,
        "gap": gap, "pct": pct, "n_beat": n_beat0,
        "gap_ci_inst": ci(gi), "pct_ci_inst": ci(pi), "beat_ci_inst": ci(bi),
        "gap_ci_camp": ci(gc), "pct_ci_camp": ci(pc_), "beat_ci_camp": ci(bc),
    }


# ==========================================================================
# ITEM 2 & 3 : Gate C capture ratio rho and dividends; partial adoption
# ==========================================================================
def item23(tier1, tier2, rng):
    both = pd.concat([tier1, tier2], ignore_index=True)
    # master instance list from dedicated cell of campus set
    master = tier1[(tier1.structure == "dedicated")
                   & (tier1.campus.isin(VERDICT_CAMPUSES))]
    master_ids = sorted(master.instance_id.unique())
    campus_of = np.array([int(i.split("_")[0][1:]) for i in master_ids])

    def build(m):
        L0 = cell_matrix(both, "dedicated", m, 1.0, master_ids)[0]
        cells = {"L0": L0}
        cells["chain10"] = cell_matrix(both, "chain", m, 1.0, master_ids,
                                       phi=1.0)[0]
        cells["full"] = cell_matrix(both, "full", m, 1.0, master_ids)[0]
        cells["chain05"] = cell_matrix(both, "chain", m, 1.0, master_ids,
                                       phi=0.5)[0]
        cells["chain025"] = cell_matrix(both, "chain", m, 1.0, master_ids,
                                        phi=0.25)[0]
        return cells

    C6 = build(0.6)
    C8 = build(0.8)
    inst_idx, camp_idx = make_resamplers(campus_of, rng, NBOOT)

    full_idx = np.arange(len(master_ids))

    # Gate C (item 2): policy-inclusive envelope, mirroring gate_c pools.
    # Phi-share (item 3): rules+random envelope, mirroring build_all's tier2
    # phi block (which does NOT include the policy pools).
    def point(C, m06=False):
        l0 = envelope_best(C["L0"], full_idx, COLS_GATEC)
        ch = envelope_best(C["chain10"], full_idx, COLS_GATEC)
        fu = envelope_best(C["full"], full_idx, COLS_GATEC)
        dch, dfu = l0 - ch, l0 - fu
        d = dict(l0=l0, dch=dch, dfu=dfu, rho=dch / dfu,
                 dfu_pct=100 * dfu / l0)
        if m06:
            # phi-share uses the rules+random L0 and chain(1.0) (build_all)
            l0p = envelope_best(C["L0"], full_idx, COLS_PHI)
            d100p = l0p - envelope_best(C["chain10"], full_idx, COLS_PHI)
            d["r05"] = (l0p - envelope_best(C["chain05"], full_idx,
                                            COLS_PHI)) / d100p
            d["r025"] = (l0p - envelope_best(C["chain025"], full_idx,
                                             COLS_PHI)) / d100p
        return d

    p6 = point(C6, m06=True)
    p8 = point(C8)

    def boot(C, idxlist, m06=False):
        out = {k: [] for k in ["rho", "dch", "dfu", "dfu_pct",
                               "r05", "r025"]}
        for idx in idxlist:
            l0 = envelope_best(C["L0"], idx, COLS_GATEC)
            ch = envelope_best(C["chain10"], idx, COLS_GATEC)
            fu = envelope_best(C["full"], idx, COLS_GATEC)
            dch, dfu = l0 - ch, l0 - fu
            out["rho"].append(dch / dfu if dfu != 0 else np.nan)
            out["dch"].append(dch)
            out["dfu"].append(dfu)
            out["dfu_pct"].append(100 * dfu / l0)
            if m06:
                l0p = envelope_best(C["L0"], idx, COLS_PHI)
                d100p = l0p - envelope_best(C["chain10"], idx, COLS_PHI)
                d05 = l0p - envelope_best(C["chain05"], idx, COLS_PHI)
                d025 = l0p - envelope_best(C["chain025"], idx, COLS_PHI)
                out["r05"].append(d05 / d100p if d100p != 0 else np.nan)
                out["r025"].append(d025 / d100p if d100p != 0 else np.nan)
        return out

    b6i = boot(C6, inst_idx, m06=True)
    b6c = boot(C6, camp_idx, m06=True)
    b8i = boot(C8, inst_idx)
    b8c = boot(C8, camp_idx)
    return {"p6": p6, "p8": p8,
            "b6i": {k: ci(v) for k, v in b6i.items() if v},
            "b6c": {k: ci(v) for k, v in b6c.items() if v},
            "b8i": {k: ci(v) for k, v in b8i.items() if v},
            "b8c": {k: ci(v) for k, v in b8c.items() if v},
            "n_instances": len(master_ids)}


# ==========================================================================
# ITEM 4 : Campus-2 transfer stress test (e4)
# ==========================================================================
def item4(e4, rng):
    c2 = e4[e4.campus == 2]
    master_ids = sorted(c2[c2.structure == "dedicated"].instance_id.unique())
    full_idx = np.arange(len(master_ids))
    # single campus -> instance-level resampling only
    idxlist = [rng.choice(full_idx, size=len(full_idx), replace=True)
               for _ in range(NBOOT)]
    res = {}
    for m in (0.8, 0.6):
        L0 = cell_matrix(c2, "dedicated", m, 0.8, master_ids)[0]
        CH = cell_matrix(c2, "chain", m, 0.8, master_ids, phi=1.0)[0]
        FU = cell_matrix(c2, "full", m, 0.8, master_ids)[0]

        def red(struct, idx):
            l0 = envelope_best(L0, idx)
            v = envelope_best(struct, idx)
            return 100 * (l0 - v) / l0
        pch = red(CH, full_idx)
        pfu = red(FU, full_idx)
        bch = [red(CH, idx) for idx in idxlist]
        bfu = [red(FU, idx) for idx in idxlist]
        res["m%s" % m] = {"chain": pch, "full": pfu,
                          "chain_ci": ci(bch), "full_ci": ci(bfu)}
    res["n_instances"] = len(master_ids)
    return res


# ==========================================================================
# ITEM 5 : effect sizes for the 7 Gate P Wilcoxon comparisons (pooled)
# ==========================================================================
def item5(tier1, rng):
    df = tier1.copy()
    flex = df[(df.structure.isin(["chain", "full"]))
              & (df.m.isin([0.6, 0.8])) & (df.eta.isin([1.0, 0.8]))].copy()
    flex["cfg"] = (flex.instance_id.astype(str) + "|" + flex.structure
                   + "|" + flex.eta.astype(str) + "|" + flex.m.astype(str))
    flex["iid"] = flex.instance_id
    pol = flex[flex.method.isin(MLP_MAIN)].groupby(["iid", "cfg"]).twt.mean()
    out = {}
    for rule in RANKED:
        rs = flex[flex.method == rule].groupby(["iid", "cfg"]).twt.mean()
        j = pd.concat([pol.rename("pol"), rs.rename("rule")], axis=1).dropna()
        diff = (j.pol - j.rule).to_numpy()      # >0 => policy WORSE
        nz = diff[diff != 0]
        med = float(np.median(diff))
        med_nz = float(np.median(nz)) if len(nz) else float("nan")
        # rank-biserial for signed-rank (sign: positive rank sum share)
        r = np.abs(nz)
        ranks = pd.Series(r).rank().to_numpy()
        Rpos = ranks[nz > 0].sum()
        Rneg = ranks[nz < 0].sum()
        tot = Rpos + Rneg
        rb = (Rpos - Rneg) / tot           # >0 => policy worse (higher TWT)
        try:
            stat, p = wilcoxon(nz, zero_method="wilcox",
                               alternative="two-sided")
        except ValueError:
            p = 1.0
        # instance-cluster bootstrap CI on median diff and rank-biserial
        iids = j.index.get_level_values("iid").to_numpy()
        campus = np.array([int(i.split("_")[0][1:]) for i in iids])
        uid = np.unique(iids)
        cof = {i: int(i.split("_")[0][1:]) for i in uid}
        by_c = {}
        for i in uid:
            by_c.setdefault(cof[i], []).append(i)
        by_c = {c: np.array(v) for c, v in by_c.items()}
        # map iid -> row positions
        pos = {}
        for k, i in enumerate(iids):
            pos.setdefault(i, []).append(k)
        pos = {i: np.array(p_) for i, p_ in pos.items()}
        meds, rbs = [], []
        for _ in range(2000):
            drawn = np.concatenate([np.random.choice(by_c[c], len(by_c[c]),
                                                     replace=True)
                                    for c in by_c])
            rows = np.concatenate([pos[i] for i in drawn])
            d = diff[rows]
            z = d[d != 0]
            if len(z) < 2:
                continue
            meds.append(np.median(d))
            rr = pd.Series(np.abs(z)).rank().to_numpy()
            Rp = rr[z > 0].sum()
            Rn = rr[z < 0].sum()
            rbs.append((Rp - Rn) / (Rp + Rn))
        out[rule] = {"median_diff": med, "median_diff_nz": med_nz,
                     "rank_biserial": rb, "p_raw": p,
                     "n_common": len(diff), "n_nonzero": len(nz),
                     "median_ci": ci(meds), "rb_ci": ci(rbs)}
    return out


# ==========================================================================
# ITEM 6 : E5 sensitivity -- best-method envelope TWT movement per variant
# ==========================================================================
def item6(e5, rng):
    variants = sorted(e5.variant.unique())
    campus_of = {}
    out = {}
    # reference cells: chain family -> base ; full family -> full_base
    def env_cell(sub, master_ids, idx):
        pos = {iid: k for k, iid in enumerate(master_ids)}
        cols = RANKED + ["policy_mlp", "policy_attn"]
        M = np.full((len(master_ids), len(cols)), np.nan)
        for j, meth in enumerate(RANKED):
            s = sub[sub.method == meth]
            for iid, v in zip(s.instance_id, s.twt):
                if iid in pos:
                    M[pos[iid], j] = v
        for j, pool in ((7, MLP_MAIN), (8, ATTN_MAIN)):
            s = sub[sub.method.isin(pool)]
            g = s.groupby("instance_id").twt.mean()
            for iid, v in g.items():
                if iid in pos:
                    M[pos[iid], j] = v
        return envelope_best(M, idx), M

    # common instance master + resamplers (shared across variants)
    master_ids = sorted(e5[e5.variant == "base"].instance_id.unique())
    campus_of = np.array([int(i.split("_")[0][1:]) for i in master_ids])
    full_idx = np.arange(len(master_ids))
    inst_idx, camp_idx = make_resamplers(campus_of, rng, NBOOT)

    Ms = {}
    bests = {}
    for v in variants:
        sub = e5[e5.variant == v]
        b, M = env_cell(sub, master_ids, full_idx)
        Ms[v] = M
        bests[v] = b

    def ref_of(v):
        return "full_base" if v.endswith("_full") or v == "full_base" \
            else "base"

    for v in variants:
        ref = ref_of(v)
        # movement recomputed per resample (paired instance set)
        mov_i = []
        for idx in inst_idx:
            bv = envelope_best(Ms[v], idx)
            br = envelope_best(Ms[ref], idx)
            mov_i.append(bv - br)
        mov_c = []
        for idx in camp_idx:
            bv = envelope_best(Ms[v], idx)
            br = envelope_best(Ms[ref], idx)
            mov_c.append(bv - br)
        base_best = bests[ref]
        out[v] = {"best_twt": bests[v], "ref": ref, "ref_best": base_best,
                  "move_abs": bests[v] - base_best,
                  "move_pct": 100 * (bests[v] - base_best) / base_best,
                  "move_abs_ci_inst": ci(mov_i),
                  "move_abs_ci_camp": ci(mov_c)}
    out["_n_instances"] = len(master_ids)
    return out


# ==========================================================================
def main():
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    tier1 = load("tier1")
    tier2 = load("tier2")
    e4 = load("e4")
    e5 = load("e5")

    print("=" * 74)
    print("CLUSTER-AWARE UNCERTAINTY  (seed=%d, B=%d)" % (SEED, NBOOT))
    print("clustering unit = base instance_id; verdict campuses 5/9/10/12")
    print("=" * 74)

    r1 = item1(tier1, rng)
    print("\n### ITEM 1  Gate P headline gap (pooled contended flexible)")
    print("n_configs=%d  n_base_instances=%d" % (r1["n_configs"],
                                                 r1["n_instances"]))
    print("policy pooled mean=%.2f  EDD pooled mean=%.2f" %
          (r1["policy_mean"], r1["edd_mean"]))
    print("abs gap = %.2f   %%gap = %.2f%%   seeds beating EDD = %d/10" %
          (r1["gap"], r1["pct"], r1["n_beat"]))
    print("  abs gap  95%% CI inst-cluster %s ; campus-cluster %s" %
          (fmt_ci(*r1["gap_ci_inst"]), fmt_ci(*r1["gap_ci_camp"])))
    print("  %% gap    95%% CI inst-cluster %s ; campus-cluster %s" %
          (fmt_ci(*r1["pct_ci_inst"]), fmt_ci(*r1["pct_ci_camp"])))
    print("  #seeds   95%% CI inst-cluster %s ; campus-cluster %s" %
          (fmt_ci(*r1["beat_ci_inst"], nd=1),
           fmt_ci(*r1["beat_ci_camp"], nd=1)))

    r23 = item23(tier1, tier2, rng)
    p6, p8 = r23["p6"], r23["p8"]
    print("\n### ITEM 2  Gate C capture rho & FULL dividend  (n_inst=%d)" %
          r23["n_instances"])
    print("m=0.6 eta=1.0:  rho=%.3f  d_chain=%.2f  d_full=%.2f (%.1f%% of L0)"
          % (p6["rho"], p6["dch"], p6["dfu"], p6["dfu_pct"]))
    print("  rho     95%% CI inst %s ; campus %s" %
          (fmt_ci(*r23["b6i"]["rho"], nd=3),
           fmt_ci(*r23["b6c"]["rho"], nd=3)))
    print("  d_chain 95%% CI inst %s ; campus %s" %
          (fmt_ci(*r23["b6i"]["dch"]), fmt_ci(*r23["b6c"]["dch"])))
    print("  d_full  95%% CI inst %s ; campus %s" %
          (fmt_ci(*r23["b6i"]["dfu"]), fmt_ci(*r23["b6c"]["dfu"])))
    print("  d_full%% 95%% CI inst %s ; campus %s" %
          (fmt_ci(*r23["b6i"]["dfu_pct"]), fmt_ci(*r23["b6c"]["dfu_pct"])))
    print("m=0.8 eta=1.0:  rho=%.3f  d_full=%.2f (%.1f%% of L0)" %
          (p8["rho"], p8["dfu"], p8["dfu_pct"]))
    print("  d_full  95%% CI inst %s ; campus %s" %
          (fmt_ci(*r23["b8i"]["dfu"]), fmt_ci(*r23["b8c"]["dfu"])))
    print("  d_full%% 95%% CI inst %s ; campus %s" %
          (fmt_ci(*r23["b8i"]["dfu_pct"]), fmt_ci(*r23["b8c"]["dfu_pct"])))

    print("\n### ITEM 3  Partial adoption capture (m=0.6 eta=1.0)")
    print("CHAIN(0.5)/CHAIN(1.0) dividend ratio = %.1f%%" % (100 * p6["r05"]))
    print("  95%% CI inst %s ; campus %s" %
          (fmt_ci(100 * r23["b6i"]["r05"][0], 100 * r23["b6i"]["r05"][1], 1),
           fmt_ci(100 * r23["b6c"]["r05"][0], 100 * r23["b6c"]["r05"][1], 1)))
    print("CHAIN(0.25)/CHAIN(1.0) dividend ratio = %.1f%%" %
          (100 * p6["r025"]))
    print("  95%% CI inst %s ; campus %s" %
          (fmt_ci(100 * r23["b6i"]["r025"][0], 100 * r23["b6i"]["r025"][1], 1),
           fmt_ci(100 * r23["b6c"]["r025"][0],
                  100 * r23["b6c"]["r025"][1], 1)))

    r4 = item4(e4, rng)
    print("\n### ITEM 4  Campus-2 transfer rescue (e4, single campus, n=%d)" %
          r4["n_instances"])
    print("  single campus -> INSTANCE-level resampling only (no campus CI)")
    for m in ("0.8", "0.6"):
        d = r4["m%s" % m]
        tag = "  (paper's 27/32 headline)" if m == "0.6" else ""
        print("m=%s eta=0.8: chain reduction=%.1f%% CI %s ; "
              "full reduction=%.1f%% CI %s%s" %
              (m, d["chain"], fmt_ci(*d["chain_ci"], nd=1),
               d["full"], fmt_ci(*d["full_ci"], nd=1), tag))

    r5 = item5(tier1, rng)
    print("\n### ITEM 5  Gate P effect sizes (pooled; sign: +=policy WORSE)")
    print("(median_diff over ALL configs is ~0 because most configs are "
          "exact ties;\n med_nz = median over the %d-odd nonzero pairs.)" %
          r5["edd"]["n_nonzero"])
    print("%-9s %8s %8s %8s %9s %6s %6s" %
          ("rule", "med_all", "med_nz", "rankbis", "p_raw", "ncom", "nnz"))
    for rule in RANKED:
        e = r5[rule]
        print("%-9s %8.2f %8.2f %8.3f %9.1e %6d %6d" %
              (rule, e["median_diff"], e["median_diff_nz"],
               e["rank_biserial"], e["p_raw"], e["n_common"], e["n_nonzero"]))
        print("           rank-biserial 95%% CI %s" % fmt_ci(*e["rb_ci"], 3))

    r6 = item6(e5, rng)
    print("\n### ITEM 6  E5 sensitivity: best-method envelope TWT movement")
    print("  CAVEAT: E5 has NO dedicated (L0) rows and each variant carries")
    print("  a single structure, so a Gate-C FULL dividend (L0-full) and rho")
    print("  (dChain/dFull) are NOT computable from E5. Reported instead:")
    print("  envelope best-method TWT and its shift vs the matching")
    print("  reference cell (chain->base, full->full_base). n_inst=%d" %
          r6["_n_instances"])
    print("%-13s %6s %9s %9s %9s   %s" %
          ("variant", "ref", "best_twt", "move_abs", "move_%", "move CI inst"))
    for v in sorted(k for k in r6 if not k.startswith("_")):
        e = r6[v]
        print("%-13s %6s %9.2f %9.2f %8.2f%%   %s" %
              (v, e["ref"], e["best_twt"], e["move_abs"], e["move_pct"],
               fmt_ci(*e["move_abs_ci_inst"])))

    # dump machine-readable
    out = {"item1": r1, "item2_3": {"p6": p6, "p8": p8,
                                    "ci": {"b6i": r23["b6i"], "b6c": r23["b6c"],
                                           "b8i": r23["b8i"],
                                           "b8c": r23["b8c"]}},
           "item4": r4, "item5": r5, "item6": r6}
    dump = Path(__file__).resolve().parent / "cluster_stats_out.json"
    with open(dump, "w") as f:
        json.dump(out, f, indent=1, default=lambda x: (
            list(x) if isinstance(x, (tuple, np.ndarray)) else float(x)))
    print("\nwrote %s" % dump)


if __name__ == "__main__":
    main()
