#!/usr/bin/env python
"""Supplementary robustness statistics computable from released
artifacts (plus the two new CPU runs e7-sparse and e6-variants).

Blocks (each independent; a missing input skips its block):
  static_bounds    : all-instance gap vs the strongest CP-SAT lower bound per
                     cell (rules and GA), vacuous-bound counts, and 60->300 s
                     bound-progress statistics.
  normalised       : per-order and per-100-order effects, breach deltas, and
                     instance-cluster bootstrap CIs for the absolute fixed-EDD
                     dividends in the evaluable family (m=0.6, eta=1.0).
  loco             : leave-one-campus-out capture ratios and Gate P margins;
                     per-campus and per-size fixed-EDD dividends.
  holm6            : Holm correction over the six distinct rule hypotheses
                     (EDD and pFIFO produce identical replay schedules).
  weights          : priority-class weight vector as replayed.
  counts           : instance counts by size and campus in the pooled scopes.
  phi_realised     : realised adoption fraction of CHAIN(0.5) per campus.
  rolling          : rolling CP-SAT subsample dispersion per Tier-1 cell.
  cost_scenarios   : skill-membership budgets, enrolled-technician counts,
                     same-cluster shares, and three cost scenarios per
                     structure, with pooled envelope TWT at both eta.
  patient_variants : patient WSPT/ATC/LFJ-ATC/ATC-eta vs their plain rules on
                     the four penalised cells + waiting statistics.
  utilisation      : per-campus offered-load utilisation and per-trade
                     overload shares at each crew multiplier, streamed from
                     the replay test instances.

Usage: PYTHONPATH=.:vendor python notes/supplementary/robustness_stats.py
Writes notes/supplementary/robustness_stats_out.json.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
Y1 = Path(os.environ.get("FMWOS_Y1_ROOT",
                         ROOT.parent / "FM-Scheduling"))
CAP = Y1 / "results/p1_calib/capacity.csv"
INST_ROOT = Y1 / "data/processed/instances"

import analysis.gates as G                                   # noqa: E402
from overlays.build import (build_overlay, load_crews,       # noqa: E402
                            scaled_crews)
from overlays import topology_overlays as rt                 # noqa: E402

RANKED = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
ENVELOPE = RANKED + ["random"]
BOOT_B, BOOT_SEED = 10_000, 20260718
CAMPUSES = (5, 9, 10, 12)
WEEK_BH = 40.0            # business hours per week (8 h x 5 d), Y1 axis
WINDOW_BH = 80.0          # dynamic replay window length


def read(family, name="results.csv"):
    p = RES / family / name
    return pd.read_csv(p) if p.exists() else None


def base_key(s):
    return (s.campus.astype(str) + "|" + s["size"].astype(str) + "|"
            + s.track.astype(str) + "|" + s.instance_id.astype(str))


def boot_ci(values_fn, camp, n_items):
    rng = np.random.default_rng(BOOT_SEED)
    idx_by_c = {c: np.flatnonzero(camp == c) for c in np.unique(camp)}
    stats = []
    for _ in range(BOOT_B):
        take = np.concatenate([rs[rng.integers(0, len(rs), len(rs))]
                               for rs in idx_by_c.values()])
        stats.append(values_fn(take))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
def static_bounds(out):
    e1 = read("e1_static")          # released CSV carries the bound columns
    cp = e1[e1.method.isin(["cpsat60", "cpsat300"])]
    key = ["instance_id", "structure", "eta"]
    bound = cp.groupby(key).best_bound_bh.max()
    blk = {}
    for tag, st, eta in [("l0", "dedicated", 1.0), ("c10", "chain", 1.0),
                         ("c08", "chain", 0.8), ("f10", "full", 1.0),
                         ("f08", "full", 0.8)]:
        row = {}
        for name, mset in (("rule", RANKED), ("ga", ["ga"])):
            br = e1[e1.method.isin(mset)].groupby(key).twt.min()
            j = pd.concat([bound, br], axis=1).dropna()
            j.columns = ["b", "m"]
            mask = ((j.index.get_level_values("structure") == st)
                    & (j.index.get_level_values("eta") == eta))
            jj = j[mask]
            vac = (jj.b <= 0) & (jj.m > 0)
            inf = jj[~vac]
            g = np.where(inf.m > 0,
                         100 * (inf.m - inf.b).clip(lower=0) / inf.m, 0.0)
            row[name] = {"n": int(len(jj)), "n_vacuous": int(vac.sum()),
                         "mean_gap_vs_bound_pct": float(g.mean()),
                         "max_gap_vs_bound_pct": float(g.max())}
        blk[tag] = row
    b60 = cp[cp.method == "cpsat60"].set_index(key)
    b300 = cp[cp.method == "cpsat300"].set_index(key)
    j = b60[["best_bound_bh", "proved_optimal"]].join(
        b300[["best_bound_bh"]], rsuffix="_300", how="inner")
    un = j[j.proved_optimal != 1]
    imp = un.best_bound_bh_300 - un.best_bound_bh
    blk["tail_progress"] = {
        "n_rerun_unproved_at_60": int(len(un)),
        "n_bound_improves": int((imp > 1e-9).sum()),
        "share_bound_improves": float((imp > 1e-9).mean())}
    out["static_bounds"] = blk


# ---------------------------------------------------------------------------
def normalised(out):
    t1 = G.expand_l0(read("tier1"))
    fam = t1[(t1.m == 0.6) & (t1.eta == 1.0) & (t1.method == "edd")]
    fam = fam[(fam.structure != "chain") | (fam.phi == 1.0)]
    piv = {}
    for st in ("dedicated", "chain", "full"):
        s = fam[fam.structure == st].copy()
        s["base"] = base_key(s)
        piv[st] = s.set_index("base")[["campus", "size", "twt",
                                       "breach_share"]]
    common = piv["dedicated"].index
    for st in ("chain", "full"):
        common = common.intersection(piv[st].index)
    camp = piv["dedicated"].loc[common, "campus"].to_numpy()
    size = piv["dedicated"].loc[common, "size"].to_numpy()
    v = {st: piv[st].loc[common, "twt"].to_numpy()
         for st in piv}
    bs = {st: piv[st].loc[common, "breach_share"].to_numpy()
          for st in piv}
    blk = {"n": int(len(common))}
    tot_orders = size.sum()
    for st in piv:
        blk[st] = {
            "twt_per_100_orders": float(100 * v[st].sum() / tot_orders),
            "breaches_per_100_orders": float(
                100 * (bs[st] * size).sum() / tot_orders)}
    for st in ("chain", "full"):
        d = v["dedicated"] - v[st]
        lo, hi = boot_ci(lambda take, d=d: d[take].mean(), camp, len(common))
        blk["dividend_%s" % st] = {
            "units": float(d.mean()), "ci_units": [lo, hi],
            "pct_of_l0": float(100 * d.mean() / v["dedicated"].mean()),
            "per_100_orders": float(100 * d.sum() / tot_orders),
            "breach_delta_per_100_orders": float(
                100 * ((bs["dedicated"] - bs[st]) * size).sum()
                / tot_orders)}
    out["normalised"] = blk


# ---------------------------------------------------------------------------
def loco(out):
    t1 = G.expand_l0(read("tier1"))
    fam = t1[(t1.m == 0.6) & (t1.eta == 1.0)]
    fam = fam[(fam.structure != "chain") | (fam.phi == 1.0)]

    def rho_of(sub, methods):
        tw = {}
        for st in ("dedicated", "chain", "full"):
            d = sub[(sub.structure == st) & sub.method.isin(methods)]
            tw[st] = d.groupby("method").twt.mean().min()
        den = tw["dedicated"] - tw["full"]
        return ((tw["dedicated"] - tw["chain"]) / den if den else None,
                den)

    blk = {"leave_out": {}, "per_campus": {}, "per_size": {}}
    for c in CAMPUSES:
        keep = fam[fam.campus != c]
        r_env, d_env = rho_of(keep, ENVELOPE)
        r_edd, d_edd = rho_of(keep, ["edd"])
        blk["leave_out"][str(c)] = {
            "rho_envelope": float(r_env), "rho_edd": float(r_edd),
            "delta_full_envelope": float(d_env),
            "delta_full_edd": float(d_edd)}
        only = fam[fam.campus == c]
        r_env, d_env = rho_of(only, ENVELOPE)
        r_edd, d_edd = rho_of(only, ["edd"])
        blk["per_campus"][str(c)] = {
            "rho_envelope": float(r_env), "rho_edd": float(r_edd),
            "delta_full_envelope": float(d_env),
            "delta_full_edd": float(d_edd)}
    for sz in (150, 400):
        onl = fam[fam["size"] == sz]
        r_env, d_env = rho_of(onl, ENVELOPE)
        r_edd, d_edd = rho_of(onl, ["edd"])
        blk["per_size"][str(sz)] = {
            "rho_envelope": float(r_env), "rho_edd": float(r_edd),
            "delta_full_envelope": float(d_env)}

    # Gate P pooled margin leaving one campus out (policy minus best rule).
    mlp = sorted({m for m in t1.method.unique() if isinstance(m, str)
                  and m.startswith("v2mlp")})
    flex = t1[(t1.m.isin([0.6, 0.8])) & (t1.structure.isin(
        ["chain", "full"]))]
    flex = flex[(flex.structure != "chain") | (flex.phi == 1.0)]
    gp = {}
    for c in ("none",) + CAMPUSES:
        keep = flex if c == "none" else flex[flex.campus != c]
        pol = keep[keep.method.isin(mlp)].groupby("method").twt.mean().mean()
        br = min(keep[keep.method == r].twt.mean() for r in RANKED)
        gp[str(c)] = {"policy_pooled": float(pol), "best_rule": float(br),
                      "margin": float(pol - br),
                      "policy_beats": bool(pol < br)}
    blk["gatep_leave_out"] = gp

    # Campus-equal-weight (macro-average) point estimates: each of the four
    # verdict campuses weighted 1/4, so a large campus cannot dominate the
    # instance-weighted pool. Capture ratio and the Gate P policy-minus-best
    # rule gap, both computed per single campus then averaged.
    caps_env = [blk["per_campus"][str(c)]["rho_envelope"] for c in CAMPUSES]
    caps_edd = [blk["per_campus"][str(c)]["rho_edd"] for c in CAMPUSES]
    gp_gaps = []
    for c in CAMPUSES:
        onl = flex[flex.campus == c]
        pol = onl[onl.method.isin(mlp)].groupby("method").twt.mean().mean()
        br = min(onl[onl.method == r].twt.mean() for r in RANKED)
        gp_gaps.append(float(pol - br))
    blk["campus_equal_weight"] = {
        "rho_envelope": float(np.mean(caps_env)),
        "rho_edd": float(np.mean(caps_edd)),
        "gatep_gap": float(np.mean(gp_gaps)),
        "per_campus_gatep_gap": gp_gaps}
    out["loco"] = blk


# ---------------------------------------------------------------------------
def holm6(out):
    """Gate P passes a scope only if the policy is LOWER than every ranked
    rule and every comparison is Holm-significant. Dropping pFIFO (EDD's
    replay duplicate) relaxes the correction to six hypotheses; the gate
    still fails wherever the policy fails to undercut at least one rule."""
    g = json.load(open(RES / "gates" / "gates.json"))["gate_p"]
    blk = {}
    for scope in ("pooled", "m08", "m06"):
        rules = g[scope]["rules"]
        distinct = {r: v for r, v in rules.items() if r != "pfifo"}
        ps = sorted((v["wilcoxon_p_raw"], r) for r, v in distinct.items())
        holm_sig = {}
        for i, (p, r) in enumerate(ps):
            holm_sig[r] = min(1.0, (len(ps) - i) * p) < 0.05
        passes = all(v["policy_lower"] and holm_sig[r]
                     for r, v in distinct.items())
        blk[scope] = {
            "n_hypotheses": len(ps),
            "rules_policy_fails_to_beat": sorted(
                r for r, v in distinct.items() if not v["policy_lower"]),
            "gate_passes_holm6": passes,
            "verdict_unchanged": not passes}
    out["holm6"] = blk


# ---------------------------------------------------------------------------
def weights_counts_phi(out):
    with open(INST_ROOT / "index.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["track"] == "replay" and r["split"] == "test"
                and int(r["campus"]) in CAMPUSES
                and int(r["size_class"]) in (150, 400)]
    wmap = {}
    for r in rows:
        for o in json.load(open(INST_ROOT / r["path"]))["work_orders"]:
            wmap.setdefault(int(o["priority"]), float(o["weight"]))
        if len(wmap) >= 4:
            break
    out["weights"] = {"by_priority": {str(k): wmap[k]
                                      for k in sorted(wmap)}}
    cnt = {}
    for r in rows:
        key = "c%02d" % int(r["campus"])
        cnt.setdefault(key, {"150": 0, "400": 0})
        cnt[key][r["size_class"]] += 1
    tot = {"150": sum(v["150"] for v in cnt.values()),
           "400": sum(v["400"] for v in cnt.values())}
    out["counts"] = {"per_campus": cnt, "total": tot,
                     "n_instances": len(rows)}

    phi = {}
    for c in CAMPUSES:
        crews = load_crews(CAP, c)
        for m in (1.0, 0.6):
            ov = build_overlay(c, crews, "chain", 0.5, 1.0, m)
            phi.setdefault("c%02d" % c, {})["m%.1f" % m] = (
                ov["budget_B"] / ov["headcount"])
    out["phi_realised"] = phi


# ---------------------------------------------------------------------------
def rolling(out):
    t1 = G.expand_l0(read("tier1"))
    rc = t1[t1.method == "rollcp2"]
    blk = {}
    for (st, eta, m), sub in rc.groupby(["structure", "eta", "m"]):
        if m not in (0.6, 0.8) and st != "dedicated":
            continue
        ids = set(sub.instance_id)
        env = t1[(t1.structure == st) & (t1.eta == eta) & (t1.m == m)
                 & t1.method.isin(RANKED) & t1.instance_id.isin(ids)]
        br = env.groupby("method").twt.mean().min()
        blk["%s_eta%.1f_m%.1f" % (st, eta, m)] = {
            "n": int(sub.instance_id.nunique()),
            "roll_mean": float(sub.twt.mean()),
            "roll_min": float(sub.twt.min()),
            "roll_max": float(sub.twt.max()),
            "best_rule_same_subsample": float(br)}
    out["rolling"] = blk


# ---------------------------------------------------------------------------
def verdict_e7(e7):
    """The e7 results.csv now also carries held-out campuses 1/2 (transfer
    closure) and m = 0.8 rows for them. Every VERDICT-scope reader must
    restrict to the verdict campuses at m = 0.6, or campus 2's overloaded
    TWT leaks into pooled means. Returns the verdict-scope slice."""
    return e7[e7.campus.isin(CAMPUSES) & (e7.m == 0.6)]


def cost_scenarios(out):
    e7 = verdict_e7(read("e7_topology"))
    t1 = G.expand_l0(read("tier1"))
    t2 = G.expand_l0(read("tier2"))
    ids = sorted(set(e7[(e7.variant == "pairs")
                        & (e7.eta == 1.0)].instance_id))
    assert len(ids) == 763, "verdict pairs instance set drifted: %d" % len(ids)

    def env_twt(d, eta):
        d = d[d.method.isin(ENVELOPE) & d.instance_id.isin(ids)]
        return float(d.groupby("method").twt.mean().min())

    structures = {}
    for c in CAMPUSES:
        crews = load_crews(CAP, c)
        order = [x["trade"] for x in
                 sorted(crews, key=lambda x: (-x["volume"], x["trade"]))]
        ovs = {
            "chain": build_overlay(c, crews, "chain", 1.0, 1.0, 0.6),
            "gen": build_overlay(c, crews, "generalist", None, 1.0, 0.6),
            "full": build_overlay(c, crews, "full", None, 1.0, 0.6),
            "pairs": rt.build_sigma_variant(
                c, crews, rt.sigma_pairs(order), 1.0, 0.6, order=order,
                struct_label="pairs", variant="pairs"),
            "feas": rt.build_sigma_variant(
                c, crews, rt.sigma_feas(rt.adjacency_order(crews)), 1.0,
                0.6, order=rt.adjacency_order(crews), struct_label="feas",
                variant="feas"),
        }
        for name, ov in ovs.items():
            memb = same = 0
            enrolled = 0
            for t in ov["technicians"]:
                sec = [s for s in t["skills"] if s != t["primary"]]
                if sec:
                    enrolled += 1
                memb += len(sec)
                same += sum(1 for s in sec if rt.cluster_of(s)
                            == rt.cluster_of(t["primary"]))
            s = structures.setdefault(name, {"B": 0, "E": 0, "same": 0})
            s["B"] += memb
            s["E"] += enrolled
            s["same"] += same
    for name, s in structures.items():
        s["cost_uniform"] = s["B"]
        s["cost_enrol_plus_marginal"] = s["E"] + s["B"]
        s["cost_family_discount"] = 2 * s["B"] - s["same"]
        s["same_cluster_share"] = s["same"] / s["B"]

    srcs = {
        "chain": lambda eta: t1[(t1.m == 0.6) & (t1.eta == eta)
                                & (t1.structure == "chain")
                                & (t1.phi == 1.0)],
        "gen": lambda eta: t2[(t2.m == 0.6) & (t2.eta == eta)
                              & (t2.structure == "generalist")],
        "full": lambda eta: t1[(t1.m == 0.6) & (t1.eta == eta)
                               & (t1.structure == "full")],
        "pairs": lambda eta: e7[(e7.variant == "pairs") & (e7.eta == eta)],
        "feas": lambda eta: e7[(e7.variant == "feas") & (e7.eta == eta)],
    }
    for name, fn in srcs.items():
        structures[name]["twt_eta10"] = env_twt(fn(1.0), 1.0)
        structures[name]["twt_eta08"] = env_twt(fn(0.8), 0.8)
    l0 = t1[(t1.m == 0.6) & (t1.eta == 1.0) & (t1.structure == "dedicated")]
    structures["l0"] = {"B": 0, "E": 0, "cost_uniform": 0,
                        "cost_enrol_plus_marginal": 0,
                        "cost_family_discount": 0,
                        "twt_eta10": env_twt(l0, 1.0),
                        "twt_eta08": env_twt(l0, 1.0)}
    out["cost_scenarios"] = structures


# ---------------------------------------------------------------------------
def patient_variants(out):
    ev = read("e6_patient", "results_variants.csv")
    if ev is None:
        return
    e6 = read("e6_patient")
    t1 = G.expand_l0(read("tier1"))
    blk = {}
    for (st, m), sub in ev.groupby(["structure", "m"]):
        cell = {}
        for rule, d in sub.groupby("method"):
            plain = rule.replace("_patient", "")
            pl = t1[(t1.m == m) & (t1.eta == 0.8) & (t1.structure == st)
                    & (t1.method == plain)]
            pl = pl[(pl.structure != "chain") | (pl.phi == 1.0)]
            n_inst = d.instance_id.nunique()
            cell[rule] = {
                "twt": float(d.twt.mean()),
                "twt_plain": float(pl.twt.mean()),
                "declines_per_instance": float(d.n_declines.mean()),
                "wait_bh_per_instance": float(d.deliberate_wait_bh.mean()),
                "primary_after_decline_share": float(
                    d.ran_primary_after_decline.sum()
                    / max(1, (d.ran_primary_after_decline
                              + d.ran_secondary_after_decline).sum()))}
        pe = e6[(e6.m == m) & (e6.structure == st)
                & (e6.method == "edd_patient")]
        ed = e6[(e6.m == m) & (e6.structure == st) & (e6.method == "edd")]
        cell["edd_patient"] = {"twt": float(pe.twt.mean()),
                               "twt_plain": float(ed.twt.mean()),
                               "declines_per_instance": float(
                                   pe.n_declines.mean()),
                               "wait_bh_per_instance": float(
                                   pe.deliberate_wait_bh.mean()),
                               "primary_after_decline_share": float(
                                   pe.ran_primary_after_decline.sum()
                                   / max(1, (pe.ran_primary_after_decline
                                             + pe.ran_secondary_after_decline
                                             ).sum()))}
        best = min(cell, key=lambda r: cell[r]["twt"])
        blk["%s_m%.1f" % (st, m)] = {"rules": cell, "best_patient": best,
                                     "best_patient_twt": cell[best]["twt"]}
    # patient-envelope capture in the penalised families, vs plain-EDD L0
    for m in (0.6, 0.8):
        l0 = float(t1[(t1.m == m) & (t1.eta == 0.8)
                      & (t1.structure == "dedicated")
                      & (t1.method == "edd")].twt.mean())
        row = {"l0_edd": l0}
        for st in ("chain", "full"):
            c = blk["%s_m%.1f" % (st, m)]
            row["div_best_patient_%s" % st] = l0 - c["best_patient_twt"]
        row["capture_best_patient"] = (row["div_best_patient_chain"]
                                       / row["div_best_patient_full"])
        blk["capture_m%.1f" % m] = row
    out["patient_variants"] = blk


# ---------------------------------------------------------------------------
def utilisation(out):
    """Two exact utilisation lenses from released artifacts.

    (a) Calibration-anchored: the crews were sized on p95 weekly labour
        hours, so p95-week utilisation = v_k / (40 bh x scaled crew) states
        what a 95th-percentile week demands of the scaled roster, per trade
        and campus-wide. This is the operational meaning of the crew
        multiplier m.
    (b) Instance release intensity: total released hours / (scaled headcount
        x window length) per replay test instance, the same statistic the
        matched-load generator fixes as U (variable-length replay windows;
        reported per size class; windows shorter than 8 bh release near-
        instantaneously and are summarised separately).
    """
    crews_all = {c: load_crews(CAP, c) for c in CAMPUSES}
    # Per-trade descriptive table shipped with the repository (referenced
    # from the deployment appendix).
    per_trade = ROOT / "results" / "workload" / "per_trade.csv"
    per_trade.parent.mkdir(parents=True, exist_ok=True)
    with open(per_trade, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["campus", "trade", "crew", "p95_weekly_hours",
                       "util_m1.0", "util_m0.8", "util_m0.6"])
        for c in CAMPUSES:
            for x in sorted(crews_all[c], key=lambda x: x["trade"]):
                row = [c, x["trade"], x["crew"], "%.2f" % x["volume"]]
                for m in (1.0, 0.8, 0.6):
                    sc = scaled_crews(crews_all[c], m)
                    row.append("%.3f" % (x["volume"]
                                         / (WEEK_BH * sc[x["trade"]])))
                wcsv.writerow(row)
    blk = {}
    for c in CAMPUSES:
        crews = crews_all[c]
        entry = {"trades": len(crews),
                 "headcount": sum(x["crew"] for x in crews),
                 "p95_weekly_hours_total": float(
                     sum(x["volume"] for x in crews))}
        for m in (1.0, 0.8, 0.6):
            sc = scaled_crews(crews, m)
            head = sum(sc.values())
            camp_u = entry["p95_weekly_hours_total"] / (WEEK_BH * head)
            per_trade = [x["volume"] / (WEEK_BH * sc[x["trade"]])
                         for x in crews]
            entry["m%.1f" % m] = {
                "p95_week_util_campus": float(camp_u),
                "p95_week_util_trade_median": float(np.median(per_trade)),
                "p95_week_util_trade_max": float(max(per_trade)),
                "share_trades_over_1": float(
                    np.mean([u > 1.0 for u in per_trade]))}
        blk["c%02d" % c] = entry

    with open(INST_ROOT / "index.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["track"] == "replay" and r["split"] == "test"
                and int(r["campus"]) in CAMPUSES
                and int(r["size_class"]) in (150, 400)]
    rel = {150: [], 400: []}
    short = 0
    for r in rows:
        inst = json.load(open(INST_ROOT / r["path"]))
        w = float(inst["meta"]["window_bh"])
        tot = sum(float(o["p_bh"]) for o in inst["work_orders"])
        head = sum(x["crew"] for x in crews_all[int(r["campus"])])
        if w < 8.0:
            short += 1
            continue
        rel[int(r["size_class"])].append(tot / (head * w))
    blk["release_intensity_m1"] = {
        str(sz): {"n": len(v), "median": float(np.median(v)),
                  "p90": float(np.percentile(v, 90))}
        for sz, v in rel.items() if v}
    blk["release_intensity_m1"]["n_windows_under_8bh"] = short
    out["utilisation"] = blk


# ---------------------------------------------------------------------------
def transfer_topology(out):
    """Held-out campuses 1-2: pairs/feas capture on the rules envelope,
    per campus and crew multiplier, vs the released e4 chain/full/L0."""
    e7 = read("e7_topology")
    e4 = read("e4")
    if e7 is None or e4 is None:
        return
    e4 = G.expand_l0(e4)
    blk = {}
    for c in (1, 2):
        for m in (0.6, 0.8):
            for eta in (1.0, 0.8):
                sub4 = e4[(e4.campus == c) & (e4.m == m) & (e4.eta == eta)]
                ids = set(sub4.instance_id)
                row = {}
                for st in ("dedicated", "chain", "full"):
                    d = sub4[(sub4.structure == st)
                             & sub4.method.isin(ENVELOPE)]
                    d = d[(d.structure != "chain") | (d.phi == 1.0)]
                    row[st] = float(d.groupby("method").twt.mean().min())
                for v in ("pairs", "feas"):
                    d = e7[(e7.campus == c) & (e7.m == m) & (e7.eta == eta)
                           & (e7.variant == v)
                           & e7.method.isin(ENVELOPE)
                           & e7.instance_id.isin(ids)]
                    if not len(d):
                        continue
                    row[v] = float(d.groupby("method").twt.mean().min())
                den = row["dedicated"] - row["full"]
                caps = {v: (row["dedicated"] - row[v]) / den
                        for v in ("chain", "pairs", "feas")
                        if v in row and abs(den) > 1e-9}
                blk["c%02d_m%.1f_eta%.1f" % (c, m, eta)] = {
                    "twt": row, "delta_full": den, "capture": caps}
    out["transfer_topology"] = blk


def patient_e4(out):
    """Patient rules on the held-out penalised cells."""
    pe = read("e6_patient", "results_e4.csv")
    pv = read("e6_patient", "results_variants_e4.csv")
    e4 = read("e4")
    if pe is None or e4 is None:
        return
    e4 = G.expand_l0(e4)
    blk = {}
    both = pd.concat([pe] + ([pv] if pv is not None else []),
                     ignore_index=True)
    for c in (1, 2):
        for (st, m), sub in both[both.campus == c].groupby(
                ["structure", "m"]):
            if st == "dedicated":
                continue
            cell = {}
            for rule, d in sub.groupby("method"):
                if not rule.endswith("_patient"):
                    continue
                plain = rule.replace("_patient", "")
                pl = e4[(e4.campus == c) & (e4.m == m) & (e4.eta == 0.8)
                        & (e4.structure == st) & (e4.method == plain)]
                pl = pl[(pl.structure != "chain") | (pl.phi == 1.0)]
                cell[rule] = {"twt": float(d.twt.mean()),
                              "twt_plain": float(pl.twt.mean())}
            if cell:
                best = min(cell, key=lambda r: cell[r]["twt"])
                blk["c%02d_%s_m%.1f" % (c, st, m)] = {
                    "rules": cell, "best_patient": best}
    out["patient_e4"] = blk


def optsigma(out):
    """Best-found one-secondary wiring: train objective and test capture."""
    files = sorted((RES / "e11_optsigma").glob("opt_sigma_c*.json"))
    if not files:
        return
    blk = {"per_campus": {json.load(open(f))["campus"]: {
        k: v for k, v in json.load(open(f)).items()
        if k in ("obj_train_edd", "best_start", "n_evals",
                 "descriptors")} for f in files}}
    e7 = read("e7_topology")
    t1 = G.expand_l0(read("tier1"))
    opt = verdict_e7(e7)[verdict_e7(e7).variant == "opt"] \
        if e7 is not None else None
    if opt is not None and len(opt):
        ids = sorted(set(opt[opt.eta == 1.0].instance_id))
        assert len(ids) == 763, "verdict opt set drifted: %d" % len(ids)
        test = {}
        for eta in (1.0, 0.8):
            row, row_e = {}, {}
            for st in ("dedicated", "chain", "full"):
                d = t1[(t1.m == 0.6) & (t1.eta == eta)
                       & (t1.structure == st)
                       & t1.instance_id.isin(ids)]
                d = d[(d.structure != "chain") | (d.phi == 1.0)]
                row[st] = float(d[d.method.isin(ENVELOPE)]
                                .groupby("method").twt.mean().min())
                row_e[st] = float(d[d.method == "edd"].twt.mean())
            d = opt[(opt.eta == eta) & opt.instance_id.isin(ids)]
            row["opt"] = float(d[d.method.isin(ENVELOPE)]
                               .groupby("method").twt.mean().min())
            row_e["opt"] = float(d[d.method == "edd"].twt.mean())
            den = row["dedicated"] - row["full"]
            den_e = row_e["dedicated"] - row_e["full"]
            test["eta%.1f" % eta] = {
                "twt": row, "delta_full": den,
                "capture_opt": (row["dedicated"] - row["opt"]) / den
                if abs(den) > 1e-9 else None,
                "capture_opt_edd": (row_e["dedicated"] - row_e["opt"]) / den_e
                if abs(den_e) > 1e-9 else None,
                "capture_chain": (row["dedicated"] - row["chain"]) / den
                if abs(den) > 1e-9 else None}
        blk["test"] = test
    out["optsigma"] = blk


def wait_policy(out):
    """E10 wait-action class vs released class, plain rules, and the
    idling-capable rules envelope, per Gate P scope."""
    ew = read("e10_wait")
    if ew is None:
        return
    t1 = G.expand_l0(read("tier1"))
    e6 = read("e6_patient")
    ev = read("e6_patient", "results_variants.csv")
    pat = pd.concat([x for x in (e6, ev) if x is not None],
                    ignore_index=True)
    pat = pat[pat.method.str.endswith("_patient")]
    mlp = sorted({m for m in t1.method.unique() if isinstance(m, str)
                  and m.startswith("v2mlp")})
    wmlp = sorted({m for m in ew.method.unique() if isinstance(m, str)
                   and m.startswith("v2wmlp")})

    def flex(df):
        d = df[(df.structure.isin(["chain", "full"]))
               & (df.m.isin([0.6, 0.8]))]
        return d[(d.structure != "chain") | (d.phi == 1.0)]

    def cellkey(d):
        return (d.structure + "|" + d.m.astype(str) + "|"
                + d.eta.astype(str))

    blk = {}
    for scope, msel in (("pooled", (0.6, 0.8)), ("m08", (0.8,)),
                        ("m06", (0.6,))):
        ft = flex(t1)
        ft = ft[ft.m.isin(msel)]
        fw = flex(ew)
        fw = fw[fw.m.isin(msel)]
        rules_pooled = {r: float(ft[ft.method == r].twt.mean())
                        for r in RANKED}
        best_rule = min(rules_pooled.values())
        rel_pool = float(ft[ft.method.isin(mlp)]
                         .groupby("method").twt.mean().mean())
        wait_pool = float(fw[fw.method.isin(wmlp)]
                          .groupby("method").twt.mean().mean())
        # Idling-capable envelope: per cell the best of (plain ranked rules,
        # patient rules); at eta = 1 patience is inert and only the plain
        # rules exist, which is exact (patient == plain there).
        env_vals = []
        for (st, m, eta), cell in ft.groupby(["structure", "m", "eta"]):
            best_plain = min(float(cell[cell.method == r].twt.mean())
                             for r in RANKED)
            pc = pat[(pat.structure == st) & (pat.m == m)]
            v = best_plain
            if eta == 0.8 and len(pc):
                v = min(v, float(pc.groupby("method").twt.mean().min()))
            n_cell = cell.instance_id.nunique()
            env_vals.append((v, n_cell))
        tot = sum(n for _v, n in env_vals)
        idling_env = sum(v * n for v, n in env_vals) / tot
        waits = fw[fw.method.isin(wmlp)]
        blk[scope] = {
            "wait_class_pooled": wait_pool,
            "released_class_pooled": rel_pool,
            "best_plain_rule": best_rule,
            "idling_envelope": float(idling_env),
            "wait_beats_released": bool(wait_pool < rel_pool),
            "wait_beats_best_rule": bool(wait_pool < best_rule),
            "wait_beats_idling_envelope": bool(wait_pool < idling_env),
            "mean_waits_per_episode": float(waits.waits.mean())
            if "waits" in waits else None,
            "mean_decisions": float(waits.decisions.mean())}
    out["wait_policy"] = blk


# ---------------------------------------------------------------------------
def _instance_starts():
    """instance_id -> (campus, window_start) for verdict replay test
    instances, read once from the Y1 instance index."""
    out = {}
    with open(INST_ROOT / "index.csv", newline="") as f:
        for r in csv.DictReader(f):
            if (r["track"] == "replay" and r["split"] == "test"
                    and int(r["campus"]) in CAMPUSES
                    and int(r["size_class"]) in (150, 400)):
                ws = json.load(open(INST_ROOT / r["path"]))["meta"][
                    "window_start"]
                out[r["id"]] = (int(r["campus"]), ws)
    return out


def calendar_cluster(out):
    """Calendar-start block bootstrap for the headline fixed-EDD capture
    ratio (m = 0.6, eta = 1.0). The base-instance key includes the size
    class, so a 150-order window and the 400-order window that share a
    calendar start are DISTINCT base instances and the base-instance
    bootstrap treats them as independent. This block instead resamples
    (campus, calendar-start) GROUPS within campus, binding the two sizes,
    the more conservative resampling unit."""
    t1 = G.expand_l0(read("tier1"))
    fam = t1[(t1.m == 0.6) & (t1.eta == 1.0) & (t1.method == "edd")]
    fam = fam[(fam.structure != "chain") | (fam.phi == 1.0)]
    piv = {}
    for st in ("dedicated", "chain", "full"):
        s = fam[fam.structure == st].copy()
        s["base"] = base_key(s)
        piv[st] = s.set_index("base")[["campus", "instance_id", "twt"]]
    common = piv["dedicated"].index
    for st in ("chain", "full"):
        common = common.intersection(piv[st].index)
    inst_ids = piv["dedicated"].loc[common, "instance_id"].to_numpy()
    camp = piv["dedicated"].loc[common, "campus"].to_numpy()
    v = {st: piv[st].loc[common, "twt"].to_numpy()
         for st in ("dedicated", "chain", "full")}
    starts = _instance_starts()
    grp = np.array([starts.get(i, (c, i))[1]
                    for i, c in zip(inst_ids, camp)])
    point = ((v["dedicated"] - v["chain"]).mean()
             / (v["dedicated"] - v["full"]).mean())
    # Group index lists per campus: each group = all rows sharing
    # (campus, calendar-start).
    groups_by_campus = {}
    for c in np.unique(camp):
        mask = camp == c
        idx = np.flatnonzero(mask)
        by_start = {}
        for i in idx:
            by_start.setdefault(grp[i], []).append(i)
        groups_by_campus[c] = [np.array(g) for g in by_start.values()]
    n_groups = sum(len(g) for g in groups_by_campus.values())
    rng = np.random.default_rng(BOOT_SEED)
    stats = []
    for _ in range(BOOT_B):
        take = []
        for c, groups in groups_by_campus.items():
            pick = rng.integers(0, len(groups), len(groups))
            for j in pick:
                take.append(groups[j])
        take = np.concatenate(take)
        num = (v["dedicated"][take] - v["chain"][take]).mean()
        den = (v["dedicated"][take] - v["full"][take]).mean()
        stats.append(num / den)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    out["calendar_cluster"] = {
        "rho_point": float(point), "ci": [float(lo), float(hi)],
        "n_instances": int(len(common)), "n_groups": int(n_groups),
        "B": BOOT_B, "seed": BOOT_SEED}


# ---------------------------------------------------------------------------
def windows(out):
    """Temporal structure of the replay test windows: within a size class
    every window has a distinct start; across size classes many share a
    start (the 400-order window extends the 150-order one)."""
    with open(INST_ROOT / "index.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["track"] == "replay" and r["split"] == "test"
                and int(r["campus"]) in CAMPUSES
                and int(r["size_class"]) in (150, 400)]
    starts = {}
    per_size = {"150": set(), "400": set()}
    for r in rows:
        meta = json.load(open(INST_ROOT / r["path"]))["meta"]
        key = (r["campus"], meta["window_start"])
        starts.setdefault(key, set()).add(r["size_class"])
        per_size[r["size_class"]].add(key)
    out["windows"] = {
        "n_instances": len(rows),
        "n_distinct_campus_starts": len(starts),
        "n_starts_shared_across_sizes": sum(
            1 for v in starts.values() if len(v) == 2),
        "distinct_starts_within_150": len(per_size["150"]),
        "distinct_starts_within_400": len(per_size["400"])}


# ---------------------------------------------------------------------------
def main():
    out = {}
    static_bounds(out)
    normalised(out)
    loco(out)
    holm6(out)
    weights_counts_phi(out)
    rolling(out)
    cost_scenarios(out)
    patient_variants(out)
    utilisation(out)
    windows(out)
    calendar_cluster(out)
    transfer_topology(out)
    patient_e4(out)
    optsigma(out)
    wait_policy(out)
    dst = Path(__file__).with_name("robustness_stats_out.json")
    json.dump(out, open(dst, "w"), indent=1, sort_keys=True)
    print("wrote", dst)
    for k in out:
        print("--", k, json.dumps(out[k])[:200])


if __name__ == "__main__":
    main()
