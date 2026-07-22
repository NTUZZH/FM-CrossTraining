"""Gate P / Gate C evaluation (protocol/Y2_protocol.md 2-3; pure functions).

Input: the tier1 tidy results table (results/tier1/results.csv) with the
L0 eta-reuse convention (dedicated rows carry eta = 1.0 and stand for both
eta values of their m family).

Gate P scopes: 'pooled' (all contended flexible cells), 'm08', 'm06'.
Contended flexible cells: m in {0.6, 0.8} x structure in {chain(1.0),
full} x eta in {1.0, 0.8}.

Policy pooling: per-instance-configuration mean over the verdict-class
seed pool, then pooled mean over configurations. Wilcoxon: paired
two-sided signed-rank on per-configuration (policy seed-mean - rule)
across common configurations; zero differences dropped ('wilcox'); Holm
correction across the seven ranked rules within each scope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RANKED = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
FLEX_STRUCTS = ("chain", "full")
CONTENDED_MS = (0.6, 0.8)
ETAS = (1.0, 0.8)


def config_key(df):
    """Unique instance-configuration key (instance x overlay cell)."""
    return (df["instance_id"].astype(str) + "|" + df["structure"].astype(str)
            + "|" + df["eta"].astype(str) + "|" + df["m"].astype(str))


def expand_l0(df):
    """Duplicate dedicated rows across both eta values (analysis reuse)."""
    l0 = df[df.structure == "dedicated"]
    out = [df[df.structure != "dedicated"]]
    for eta in ETAS:
        c = l0.copy()
        c["eta"] = eta
        out.append(c)
    return pd.concat(out, ignore_index=True)


def contended_flexible(df):
    return df[df.structure.isin(FLEX_STRUCTS) & df.m.isin(CONTENDED_MS)
              & df.eta.isin(ETAS)]


def _policy_seed_means(df, methods):
    """Per-configuration seed-mean TWT for a policy method pool."""
    pol = df[df.method.isin(methods)].copy()
    pol["cfg"] = config_key(pol)
    return pol.groupby("cfg").twt.mean()


def _rule_series(df, rule):
    r = df[df.method == rule].copy()
    r["cfg"] = config_key(r)
    return r.set_index("cfg").twt


def holm(pvals):
    """Holm step-down adjusted p-values (order preserved)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def gate_p_scope(df_scope, verdict_methods, per_seed_methods):
    """Evaluate one Gate P scope. Returns a dict with every criterion."""
    pol = _policy_seed_means(df_scope, verdict_methods)
    out = {"n_configs": int(pol.shape[0]),
           "policy_pooled_mean": float(pol.mean())}
    rules = {}
    pvals = []
    for rule in RANKED:
        rs = _rule_series(df_scope, rule)
        common = pol.index.intersection(rs.index)
        d_pol = pol.loc[common]
        d_rule = rs.loc[common]
        diff = (d_pol - d_rule).to_numpy()
        nz = diff[diff != 0.0]
        if len(nz) >= 1:
            try:
                stat, p = wilcoxon(nz, zero_method="wilcox",
                                   alternative="two-sided")
            except ValueError:
                stat, p = float("nan"), 1.0
        else:
            stat, p = float("nan"), 1.0
        rules[rule] = {"rule_pooled_mean": float(d_rule.mean()),
                       "policy_lower": bool(d_pol.mean() < d_rule.mean()),
                       "wilcoxon_p_raw": float(p),
                       "n_common": int(len(common)),
                       "n_nonzero": int(len(nz))}
        pvals.append(p)
    adj = holm(pvals)
    for rule, a in zip(RANKED, adj):
        rules[rule]["wilcoxon_p_holm"] = float(a)
        rules[rule]["significant_holm"] = bool(a < 0.05)
    out["rules"] = rules
    best_rule = min(RANKED, key=lambda r: rules[r]["rule_pooled_mean"])
    out["best_rule"] = best_rule
    out["best_rule_mean"] = rules[best_rule]["rule_pooled_mean"]

    # Seed-consistency: each seed's pooled mean vs the best rule.
    seed_beats = []
    for meth in per_seed_methods:
        s = df_scope[df_scope.method == meth]
        if len(s):
            seed_beats.append(float(s.twt.mean())
                              < out["best_rule_mean"])
    out["seeds_beating_best_rule"] = int(sum(seed_beats))
    out["n_seeds"] = len(seed_beats)

    out["mean_lower_all"] = bool(all(r["policy_lower"]
                                     for r in rules.values()))
    out["holm_all"] = bool(all(r["significant_holm"]
                               for r in rules.values()))
    out["seed_criterion"] = bool(out["seeds_beating_best_rule"] >= 8)
    out["pass"] = bool(out["mean_lower_all"] and out["holm_all"]
                       and out["seed_criterion"])
    return out


def gate_p(tier1, verdict_methods, per_seed_methods):
    df = expand_l0(tier1)
    flex = contended_flexible(df)
    scopes = {
        "pooled": flex,
        "m08": flex[flex.m == 0.8],
        "m06": flex[flex.m == 0.6],
    }
    res = {name: gate_p_scope(sub, verdict_methods, per_seed_methods)
           for name, sub in scopes.items()}
    res["pass"] = bool(all(res[s]["pass"] for s in ("pooled", "m08", "m06")))
    return res


def twt_best(df_cell, methods_full_cell):
    """Lowest pooled mean TWT over full-cell methods; policies enter as
    seed-mean pools (method prefix), rules as single methods."""
    best = None
    best_name = None
    for name, methods in methods_full_cell.items():
        sub = df_cell[df_cell.method.isin(methods)]
        if not len(sub):
            continue
        if len(methods) > 1:                      # seed pool -> per-cfg mean
            sub = sub.copy()
            sub["cfg"] = config_key(sub)
            val = float(sub.groupby("cfg").twt.mean().mean())
        else:
            val = float(sub.twt.mean())
        if best is None or val < best:
            best, best_name = val, name
    return best, best_name


def gate_c(tier1, methods_full_cell, guard_pct=0.02, threshold=0.70):
    """Gate C per protocol log 3. Returns dict with per-(m, eta) dividends,
    capture ratios, guard status, and verdict at eta = 1.0."""
    df = expand_l0(tier1)
    out = {"families": {}}
    for m in CONTENDED_MS:
        for eta in ETAS:
            fam = df[(df.m == m) & (df.eta == eta)]
            cells = {}
            for st in ("dedicated", "chain", "full"):
                sub = fam[fam.structure == st]
                val, name = twt_best(sub, methods_full_cell)
                cells[st] = {"twt_best": val, "best_method": name}
            d_chain = d_full = rho = None
            guard = None
            if all(cells[s]["twt_best"] is not None
                   for s in ("dedicated", "chain", "full")):
                l0v = cells["dedicated"]["twt_best"]
                d_chain = l0v - cells["chain"]["twt_best"]
                d_full = l0v - cells["full"]["twt_best"]
                guard = bool(d_full < guard_pct * l0v)
                rho = (d_chain / d_full) if (d_full and d_full > 0) else None
            out["families"]["m%s_eta%s" % (m, eta)] = {
                "cells": cells, "delta_chain": d_chain, "delta_full": d_full,
                "rho": rho, "guard_no_dividend": guard,
            }
    # Verdict at eta = 1.0 (pool the two m families? protocol: rho pooled
    # over verdict campuses and sizes PER crew-multiplier family; the
    # chaining-suffices criterion reads rho >= 0.70 at eta = 1.0. We report
    # per-family rho and evaluate the criterion on each family, with the
    # guard applied per family.)
    verdict = {}
    for m in CONTENDED_MS:
        fam = out["families"]["m%s_eta1.0" % m]
        if fam["guard_no_dividend"]:
            verdict["m%s" % m] = "not_evaluable_guard"
        elif fam["rho"] is None:
            verdict["m%s" % m] = "pending"
        else:
            verdict["m%s" % m] = ("pass" if fam["rho"] >= threshold
                                  else "fail")
    out["verdict_eta10"] = verdict
    out["threshold"] = threshold
    return out
