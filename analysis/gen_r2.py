"""Principal-regime, selection, adoption, decision-map and Frame U analyses
(analysis only, no new simulation).

Reads the released result files (results/tier1, tier2, e8_m07, e3), the
Frame U instance files (data/e3) and the released analysis code, and writes

  paper/sections/gen_regimes.tex   main-text principal-regime table
  paper/sections/gen_frameu.tex    supplement Frame U tables (trade loads;
                                   weighted tardiness and capture ratio)

plus the ``r2-`` keys of the numbers dictionary passed to generate(). It
never writes paper/numbers.json.

Items:

  A-BOOT  the headline Gate C capture-ratio interval with the envelope
          re-selected in every resample and with each structure's originally
          selected method held fixed.
  A-REG   principal-regime table: L0, CHAIN(1.0) and FULL pooled TWT, the
          dividends, the capture ratio with its interval and the 2% guard,
          on the Gate C estimator and under fixed EDD; deadline breaches.
  A-PHI   partial-adoption shares against two denominators, on three
          estimators.
  A-MAP   decision-map recount (unique winners versus ties).
  A-FU    Frame U: what is held constant, trade-level offered load, paired
          CHAIN - FULL difference, dedicated-anchored dividends and ratio,
          and the length-floor split of L0 tardiness under fixed EDD.

Estimators
  Gate C envelope  lowest pooled mean TWT over the seven ranked rules,
                   Random and both policy classes' per-instance seed-means
                   (analysis/gates.py gate_c). Ties go to the first method
                   in that order, as in gates.twt_best.
  rules envelope   the same over the seven ranked rules and Random (the
                   released partial-adoption estimator, build_all tier2).
  Frame U envelope the seven ranked rules and both policy classes (the
                   released frameu-m08-resolution pools).
  fixed EDD        EDD in every cell.

Bootstrap (10,000 resamples, seed 20260718, base instances resampled with
replacement within campus):
  stream A  the exact resample sets of notes/supplementary/cluster_stats.py
            item 2 (after the item-1 draws), used for every envelope
            interval on the 763 replay instances, so the m = 0.6 row
            reproduces cluster-rho-inst and cluster-r05-inst exactly;
  stream B  the exact resample sets of notes/supplementary/dividend_stats.py
            (edd_rho) and robustness_stats.py (normalised), used for every
            fixed-EDD interval on the replay instances, so rho-edd-ci and
            div-*-ci-units reproduce exactly;
  stream U  fresh draws for Frame U, stratified by campus x U.
Every envelope interval re-selects the best method inside each resample
unless it is labelled conditional.

Usage: PYTHONPATH=.:vendor python analysis/gen_r2.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

RANKED = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
COLS = RANKED + ["random", "policy_mlp", "policy_attn"]
MLP = ["v2mlp%d" % i for i in range(301, 311)]
ATTN = ["v2attn%d" % i for i in range(401, 411)]

GATEC = list(range(10))                       # rules + Random + both pools
RULES_RANDOM = list(range(8))                 # rules + Random
FRAMEU = list(range(7)) + [8, 9]              # rules + both pools
FIGU = list(range(7)) + [8]                   # Fig. 3 pools (rules + MLP)
EDD = [0]

PRETTY = {"edd": "EDD", "wspt": "WSPT", "atc": "ATC", "pfifo": "pFIFO",
          "mor": "LPT", "lfj_atc": "LFJ-ATC", "atc_eta": "ATC-$\\eta$",
          "random": "Random", "policy_mlp": "pair-MLP policy",
          "policy_attn": "pair-attention policy"}

BOOT_B, BOOT_SEED = 10_000, 20260718
GUARD, THRESHOLD, EPS = 0.02, 0.70, 1.0
VERDICT_CAMPUSES = (5, 9, 10, 12)
E3_WINDOW_BH = 80.0
U_LEVELS = (0.9, 1.0, 1.1, 1.3)

# (m, eta, row label) of the principal-regime table, in table order.
REGIMES = [(0.6, 1.0, ""), (0.6, 0.8, ""), (0.7, 1.0, ""), (0.8, 1.0, ""),
           (0.8, 0.8, ""), (1.0, 1.0, "")]


def _tag(m, eta):
    return "m%02de%02d" % (round(10 * m), round(10 * eta))


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
def _load(root, family):
    p = root / "results" / family / "results.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(
        root / "results" / family / "results.csv")


def _campus(iid):
    return int(str(iid).split("_")[0][1:])


def cell_matrix(df, structure, m, eta, master, phi=None):
    """(n_master, 10) per-instance TWT; columns COLS.

    Rules and Random carry one row per instance; a policy pool is collapsed
    to its per-instance seed-mean (all ten seeds present, asserted). L0 rows
    are eta-invariant and are not filtered on eta.
    """
    sub = df[(df.structure == structure) & (df.m == m)]
    if structure != "dedicated":
        sub = sub[sub.eta == eta]
    if phi is not None:
        sub = sub[sub.phi == phi]
    M = np.full((len(master), len(COLS)), np.nan)
    piv = sub[sub.method.isin(RANKED + ["random"])].pivot(
        index="instance_id", columns="method", values="twt").reindex(master)
    for j, meth in enumerate(RANKED + ["random"]):
        if meth in piv.columns:
            M[:, j] = piv[meth].to_numpy(float)
    for j, pool in ((8, MLP), (9, ATTN)):
        g = sub[sub.method.isin(pool)].groupby("instance_id").twt
        cnt = g.size().reindex(master)
        assert (cnt == len(pool)).all(), "incomplete policy pool"
        M[:, j] = g.mean().reindex(master).to_numpy(float)
    assert np.isfinite(M).all(), "missing rows in %s m=%s eta=%s" % (
        structure, m, eta)
    return M


def column(df, structure, m, eta, master, field, method="edd", phi=None):
    sub = df[(df.structure == structure) & (df.m == m)
             & (df.method == method)]
    if structure != "dedicated":
        sub = sub[sub.eta == eta]
    if phi is not None:
        sub = sub[sub.phi == phi]
    v = sub.set_index("instance_id")[field].reindex(master).to_numpy(float)
    assert np.isfinite(v).all()
    return v


# --------------------------------------------------------------------------- #
# Resample streams                                                             #
# --------------------------------------------------------------------------- #
def _counts(idx_list, n):
    W = np.zeros((len(idx_list), n), dtype=np.float64)
    for b, idx in enumerate(idx_list):
        W[b] = np.bincount(idx, minlength=n)
    return W


def _released_module(root, name):
    path = root / "notes" / "supplementary" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_rel_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stream_a(root, master):
    """cluster_stats.py item-2 instance-cluster resamples, replayed exactly.

    main() seeds one Generator, item 1 (the Gate P gap) consumes one call of
    make_resamplers over the same 763 instances (the same per-campus sizes),
    and item 2 draws the resamples the released Gate C interval uses.
    """
    cs = _released_module(root, "cluster_stats")
    campus_of = np.array([_campus(i) for i in master])
    rng = np.random.default_rng(cs.SEED)
    cs.make_resamplers(campus_of, rng, cs.NBOOT)          # item 1 draws
    inst_idx, _camp = cs.make_resamplers(campus_of, rng, cs.NBOOT)
    return _counts(inst_idx, len(master))


def stream_b(tier1, master):
    """dividend_stats.py edd_rho / robustness_stats.py boot_ci resamples.

    Rebuilds the released base-instance order exactly (EDD rows, m = 0.6,
    eta = 1.0, the intersection of the L0, CHAIN(1.0) and FULL keys) and
    replays the released draws, then maps them onto ``master``.
    """
    import analysis.gates as G
    t1 = G.expand_l0(tier1)
    fam = t1[(t1.m == 0.6) & (t1.eta == 1.0) & (t1.method == "edd")]
    fam = fam[(fam.structure != "chain") | (fam.phi == 1.0)]
    piv = {}
    for st in ("dedicated", "chain", "full"):
        s = fam[fam.structure == st].copy()
        s["base"] = (s.campus.astype(str) + "|" + s["size"].astype(str)
                     + "|" + s.track.astype(str) + "|"
                     + s.instance_id.astype(str))
        piv[st] = s.set_index("base")[["campus", "twt"]]
    common = piv["dedicated"].index
    for st in ("chain", "full"):
        common = common.intersection(piv[st].index)
    camp = piv["dedicated"].loc[common, "campus"].to_numpy()
    pos = {iid: k for k, iid in enumerate(master)}
    to_master = np.array([pos[b.split("|")[3]] for b in common])
    rng = np.random.default_rng(BOOT_SEED)
    idx_by_c = {c: np.flatnonzero(camp == c) for c in np.unique(camp)}
    idx = []
    for _ in range(BOOT_B):
        take = np.concatenate([rs[rng.integers(0, len(rs), len(rs))]
                               for rs in idx_by_c.values()])
        idx.append(to_master[take])
    return _counts(idx, len(master))


def stratified_weights(strata, n_boot=BOOT_B, seed=BOOT_SEED):
    """(n_boot, n) resample counts: units drawn with replacement within each
    stratum, strata visited in sorted order (tuples sort numerically), one
    Generator seeded once."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, len(strata)), dtype=np.float64)
    for s in sorted(set(strata)):
        idx = np.array([k for k, x in enumerate(strata) if x == s])
        k = len(idx)
        draw = rng.integers(0, k, size=(n_boot, k))
        flat = (np.arange(n_boot)[:, None] * k + draw).ravel()
        W[:, idx] += np.bincount(flat, minlength=n_boot * k).reshape(
            n_boot, k)
    return W


# --------------------------------------------------------------------------- #
# Estimators                                                                   #
# --------------------------------------------------------------------------- #
def best(M, cols, W=None):
    """Envelope over ``cols``: point value, selected column, and (with W) the
    re-selected and the conditional (original column held) resampled values
    plus the per-resample selected column."""
    sel = M[:, cols]
    means = sel.mean(axis=0)
    j = int(np.argmin(means))
    out = {"point": float(means[j]), "col": cols[j],
           "method": COLS[cols[j]]}
    if W is not None:
        mb = (W @ sel) / W.sum(axis=1, keepdims=True)
        out["resel"] = mb.min(axis=1)
        out["cond"] = mb[:, j]
        out["argmin"] = np.array(cols)[mb.argmin(axis=1)]
    return out


def ci(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def family(cells, cols, W=None, mode="resel"):
    """Dividends, capture ratio and guard for {'l0','chain','full'} cells."""
    b = {k: best(v, cols, W) for k, v in cells.items()}
    l0, ch, fu = b["l0"]["point"], b["chain"]["point"], b["full"]["point"]
    d_ch, d_fu = l0 - ch, l0 - fu
    out = {"l0": l0, "chain": ch, "full": fu, "d_chain": d_ch,
           "d_full": d_fu, "d_full_pct": 100.0 * d_fu / l0,
           "d_chain_pct": 100.0 * d_ch / l0,
           "rho": d_ch / d_fu if d_fu != 0 else float("nan"),
           "evaluable": bool(d_fu >= GUARD * l0),
           "sel": {k: v["method"] for k, v in b.items()}}
    if W is not None:
        for mode_ in ("resel", "cond"):
            L, C, F = (b["l0"][mode_], b["chain"][mode_], b["full"][mode_])
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (L - C) / (L - F)
            out["ci_rho_" + mode_] = ci(r)
            out["ci_dch_" + mode_] = ci(L - C)
            out["ci_dfu_" + mode_] = ci(L - F)
            out["ci_dfupct_" + mode_] = ci(100.0 * (L - F) / L)
        out["sel_freq"] = {
            k: {COLS[c]: float(np.mean(v["argmin"] == c))
                for c in np.unique(v["argmin"])}
            for k, v in b.items()}
    return out


# --------------------------------------------------------------------------- #
# A-REG (and A-BOOT, breaches)                                                 #
# --------------------------------------------------------------------------- #
def regimes(tier1, e8, master, WA, WB):
    res = {}
    src = {0.6: tier1, 0.8: tier1, 1.0: tier1, 0.7: e8}
    for m, eta, _lab in REGIMES:
        df = src[m]
        cells = {"l0": cell_matrix(df, "dedicated", m, eta, master),
                 "chain": cell_matrix(df, "chain", m, eta, master, phi=1.0),
                 "full": cell_matrix(df, "full", m, eta, master)}
        res[(m, eta, "env")] = family(cells, GATEC, WA)
        res[(m, eta, "edd")] = family(cells, EDD, WB)
    return res


def breaches(tier1, master, WB, m=0.6, eta=1.0):
    """Deadline breaches per 100 orders under fixed EDD (order-weighted)."""
    size = column(tier1, "dedicated", m, eta, master, "size")
    out = {"orders": float(size.sum())}
    cnt = {}
    for st, phi in (("dedicated", None), ("chain", 1.0), ("full", None)):
        bs = column(tier1, st, m, eta, master, "breach_share", phi=phi)
        cnt[st] = bs * size
        out[st] = float(100.0 * cnt[st].sum() / size.sum())
    for st in ("chain", "full"):
        d = cnt["dedicated"] - cnt[st]
        pt = float(100.0 * d.sum() / size.sum())
        bt = 100.0 * (WB @ d) / (WB @ size)
        out["avoided_" + st] = pt
        out["avoided_ci_" + st] = ci(bt)
        out["orders_per_avoided_" + st] = 100.0 / pt
        lo, hi = ci(bt)
        out["orders_per_avoided_ci_" + st] = (100.0 / hi, 100.0 / lo)
    return out


# --------------------------------------------------------------------------- #
# A-PHI                                                                         #
# --------------------------------------------------------------------------- #
def phi_shares(tier1, tier2, master, WA, WB, m=0.6, eta=1.0):
    both = pd.concat([tier1, tier2], ignore_index=True)
    cells = {"l0": cell_matrix(both, "dedicated", m, eta, master),
             "c025": cell_matrix(both, "chain", m, eta, master, phi=0.25),
             "c05": cell_matrix(both, "chain", m, eta, master, phi=0.5),
             "c10": cell_matrix(both, "chain", m, eta, master, phi=1.0),
             "full": cell_matrix(both, "full", m, eta, master)}
    out = {}
    for name, cols, W in (("rules", RULES_RANDOM, WA), ("gatec", GATEC, WA),
                          ("edd", EDD, WB)):
        b = {k: best(v, cols, W) for k, v in cells.items()}
        pt = {k: v["point"] for k, v in b.items()}
        rs = {k: v["resel"] for k, v in b.items()}
        d = {k: pt["l0"] - pt[k] for k in ("c025", "c05", "c10", "full")}
        db = {k: rs["l0"] - rs[k] for k in ("c025", "c05", "c10", "full")}
        r = {"sel": {k: v["method"] for k, v in b.items()},
             "twt": pt, "delta": d}
        for num in ("c05", "c025"):
            for den in ("c10", "full"):
                key = "%s_%s" % (num, den)
                r[key] = d[num] / d[den]
                with np.errstate(divide="ignore", invalid="ignore"):
                    r["ci_" + key] = ci(db[num] / db[den])
        r["c10_full"] = d["c10"] / d["full"]
        with np.errstate(divide="ignore", invalid="ignore"):
            r["ci_c10_full"] = ci(db["c10"] / db["full"])
        out[name] = r
    # The Discussion's "about 83%": the rules-envelope half-adoption share
    # multiplied by the Gate C capture ratio (two estimators combined).
    out["product_rules_share_x_gatec_rho"] = None
    return out


# --------------------------------------------------------------------------- #
# A-MAP                                                                        #
# --------------------------------------------------------------------------- #
CLASSIC = ["edd", "pfifo", "atc", "wspt", "mor"]
FLEXRULES = ["lfj_atc", "atc_eta"]


def decision_map(tier1):
    """Fig. S5 recount, the figure's exact convention plus a method-level
    variant. Returns per-cell records."""
    import analysis.gates as G
    df = G.expand_l0(tier1)
    recs = []
    for eta in (1.0, 0.8):
        d_eta = df[df.eta == eta]
        for c in VERDICT_CAMPUSES:
            sub_c = d_eta[d_eta.campus == c]
            for st in ("dedicated", "chain", "full"):
                for m in (1.0, 0.8, 0.6):
                    cell = sub_c[(sub_c.structure == st) & (sub_c.m == m)]
                    means = {x: float(cell[cell.method == x].twt.mean())
                             for x in CLASSIC + FLEXRULES + ["random"]}
                    for pname, pool in (("policy_mlp", MLP),
                                        ("policy_attn", ATTN)):
                        p = cell[cell.method.isin(pool)].copy()
                        p["cfg"] = G.config_key(p)
                        means[pname] = float(p.groupby("cfg").twt.mean()
                                             .mean())
                    fam = {"classic": min(means[x] for x in CLASSIC),
                           "flex": min(means[x] for x in FLEXRULES),
                           "policy": means["policy_mlp"]}
                    bst = min(fam.values())
                    win = [k for k, v in fam.items() if v <= bst + EPS]
                    # Method level: EDD and pFIFO produce identical schedules
                    # on the replay grid (asserted), so they count as one.
                    assert abs(means["edd"] - means["pfifo"]) < 1e-9
                    meth = {x: means[x] for x in
                            ["edd", "atc", "wspt", "mor", "lfj_atc",
                             "atc_eta", "policy_mlp"]}
                    bm = min(meth.values())
                    mwin = sorted((k for k, v in meth.items()
                                   if v <= bm + EPS), key=lambda k: meth[k])
                    # Variant: the learned-policy family takes the better of
                    # the two classes.
                    fam2 = dict(fam)
                    fam2["policy"] = min(means["policy_mlp"],
                                         means["policy_attn"])
                    b2 = min(fam2.values())
                    win2 = [k for k, v in fam2.items() if v <= b2 + EPS]
                    best_classic = min(CLASSIC, key=lambda x: means[x])
                    recs.append({
                        "eta": eta, "campus": c, "structure": st, "m": m,
                        "family_winners": win,
                        "family_label": win[0] if len(win) == 1 else "tie",
                        "family_label_both_pools": (win2[0] if len(win2) == 1
                                                    else "tie"),
                        "method_winners": mwin,
                        "best_classic": best_classic,
                        "means": {k: round(v, 3) for k, v in means.items()},
                        "contended_flexible": (st != "dedicated"
                                               and m in (0.6, 0.8))})
    return recs


def map_summary(recs):
    out = {}
    for eta in (1.0, 0.8):
        rr = [r for r in recs if r["eta"] == eta]
        cnt = {}
        for r in rr:
            cnt[r["family_label"]] = cnt.get(r["family_label"], 0) + 1
        cnt2 = {}
        for r in rr:
            cnt2[r["family_label_both_pools"]] = cnt2.get(
                r["family_label_both_pools"], 0) + 1
        mcnt = {"tie": 0}
        for r in rr:
            if len(r["method_winners"]) == 1:
                k = r["method_winners"][0]
                mcnt[k] = mcnt.get(k, 0) + 1
            else:
                mcnt["tie"] += 1
        out[eta] = {"n_cells": len(rr), "family": cnt,
                    "family_both_pools": cnt2, "method": mcnt}
    # Distinct cells: L0 cells are eta-invariant, so the eta = 0.8 row of the
    # map repeats the 12 L0 cells of the eta = 1 row.
    distinct = [r for r in recs if r["eta"] == 1.0] + [
        r for r in recs if r["eta"] == 0.8 and r["structure"] != "dedicated"]
    dc = {}
    for r in distinct:
        dc[r["family_label"]] = dc.get(r["family_label"], 0) + 1
    out["distinct"] = {"n_cells": len(distinct), "family": dc}
    return out


# --------------------------------------------------------------------------- #
# A-FU                                                                         #
# --------------------------------------------------------------------------- #
def frameu_trade_load(root):
    """Per-trade offered utilization under the dedicated roster, per
    instance: trade labor hours / (trade headcount x 80 bh)."""
    rows = []
    for p in sorted((root / "data" / "e3").glob("c*_e3_w80_u*.json")):
        inst = json.loads(p.read_text())
        meta = inst["meta"]
        crew = {}
        for t in inst["technicians"]:
            crew[t["trade"]] = crew.get(t["trade"], 0) + 1
        hours = {tr: 0.0 for tr in crew}
        for w in inst["work_orders"]:
            hours[w["trade"]] = hours.get(w["trade"], 0.0) + float(w["p_bh"])
        tot_h = sum(hours.values())
        head = sum(crew.values())
        util = {tr: hours[tr] / (crew[tr] * E3_WINDOW_BH) for tr in crew}
        uv = np.array(list(util.values()))
        hv = np.array([hours[tr] for tr in util])
        rows.append({
            "instance_id": meta["id"], "campus": meta["campus"],
            "u": float(meta["u_target"]),
            "arrival_multiplier": float(meta["arrival_multiplier"]),
            "u_agg": tot_h / (head * E3_WINDOW_BH),
            "n_orders": len(inst["work_orders"]),
            "headcount": head, "n_trades": len(util),
            "trade_mean": float(uv.mean()),
            "trade_hw": float((uv * hv).sum() / hv.sum()),
            "trade_max": float(uv.max()),
            "trade_min": float(uv.min()),
            "share_trades_gt1": float((uv > 1.0).mean()),
            "share_hours_gt1": float(hv[uv > 1.0].sum() / hv.sum()),
            "max_trade": max(util, key=util.get)})
    return pd.DataFrame(rows)


def frameu_floor(root, e3):
    """Length-floor split of Frame U tardiness under fixed EDD at eta = 1.

    The floor of an instance is sum_j w_j * max(0, r_j + p_j - d_j), the TWT
    no one-technician schedule can avoid (analysis/ecrew_floor.py applies the
    same definition to the replay instances). L0 TWT minus the floor is the
    tardiness caused by waiting for a technician. Pooled means per U level
    and over U >= 1.0."""
    floor = {}
    for p in sorted((root / "data" / "e3").glob("c*_e3_w80_u*.json")):
        inst = json.loads(p.read_text())
        floor[inst["meta"]["id"]] = sum(
            float(w["weight"]) * max(0.0, float(w["release_bh"])
                                     + float(w["p_bh"]) - float(w["due_bh"]))
            for w in inst["work_orders"])
    edd = e3[e3.method == "edd"]
    out = {}
    for name, us in [(u, (u,)) for u in U_LEVELS] + [("ge1", (1.0, 1.1, 1.3))]:
        sub = edd[edd.u_target.isin(us)]
        # L0 rows are eta-invariant and stored at eta = 1.0.
        l0 = sub[sub.structure == "dedicated"].set_index("instance_id").twt
        ch = sub[(sub.structure == "chain") & (sub.eta == 1.0)].set_index(
            "instance_id").twt
        fu = sub[(sub.structure == "full") & (sub.eta == 1.0)].set_index(
            "instance_id").twt
        ids = sorted(set(l0.index) & set(ch.index) & set(fu.index))
        L, C, F = (float(x[ids].mean()) for x in (l0, ch, fu))
        fl = float(np.mean([floor[i] for i in ids]))
        out[name] = {"n": len(ids), "l0": L, "floor": fl,
                     "floor_share": fl / L,
                     "full_removes_wait": (L - F) / (L - fl),
                     "chain_removes_wait": (L - C) / (L - fl)}
    return out


def frameu(e3):
    """Paired CHAIN - FULL difference and dedicated-anchored dividends."""
    # cell_matrix reads the eta-invariant L0 rows (stored at eta = 1.0) for
    # both eta values, so the released file is used without expand_l0.
    df = e3
    sets = [(u, (u,)) for u in U_LEVELS] + [("ge1", (1.0, 1.1, 1.3))]
    out = {}
    for name, us in sets:
        master = sorted(df[(df.structure == "dedicated")
                           & (df.u_target.isin(us))].instance_id.unique())
        strata = [(_campus(i), int(i.split("_u")[1][:3])) for i in master]
        W = stratified_weights(strata)
        sub = df[df.u_target.isin(us)]
        for eta in (1.0, 0.8):
            cells = {"l0": cell_matrix(sub, "dedicated", 1.0, eta, master),
                     "chain": cell_matrix(sub, "chain", 1.0, eta, master,
                                          phi=1.0),
                     "full": cell_matrix(sub, "full", 1.0, eta, master)}
            for est, cols in (("env", FRAMEU), ("edd", EDD)):
                f = family(cells, cols, W)
                bc = best(cells["chain"], cols, W)
                bf = best(cells["full"], cols, W)
                diff = bc["point"] - bf["point"]
                f["chain_minus_full"] = diff
                f["chain_minus_full_pct"] = 100.0 * diff / bf["point"]
                f["ci_cmf"] = ci(bc["resel"] - bf["resel"])
                f["ci_cmf_pct"] = ci(100.0 * (bc["resel"] - bf["resel"])
                                     / bf["resel"])
                f["n"] = len(master)
                out[(name, eta, est)] = f
        # Fig. 3 ribbons (Frame U figure): per U, pools as the figure draws.
        if name != "ge1":
            for eta in (1.0, 0.8):
                for st, phi in (("dedicated", None), ("chain", 1.0),
                                ("full", None)):
                    M = cell_matrix(sub, st, 1.0, eta, master, phi=phi)
                    b = best(M, FIGU, W)
                    out[("fig", name, eta, st)] = {
                        "point": b["point"], "method": b["method"],
                        "ci_resel": ci(b["resel"]), "ci_cond": ci(b["cond"])}
    return out


# --------------------------------------------------------------------------- #
# Formatting                                                                   #
# --------------------------------------------------------------------------- #
def _n(x, nd=2):
    t = "%.*f" % (nd, x)
    if t.startswith("-"):
        return t[1:] if float(t) == 0.0 else "$-$" + t[1:]
    return t


def _ci3(p, c, nd=3):
    return "%.*f (95\\%% CI %.*f--%.*f)" % (nd, p, nd, c[0], nd, c[1])


def _rng_txt(c, nd=1):
    """Interval text that stays readable when a bound is negative."""
    if c[0] < 0 or c[1] < 0:
        return "%s to %s" % (_n(c[0], nd), _n(c[1], nd))
    return "%s--%s" % (_n(c[0], nd), _n(c[1], nd))


def regimes_table(reg):
    lines = [
        "%% Generated by analysis/gen_r2.py from results/tier1 and "
        "results/e8_m07 -- do not edit.",
        "\\begin{table}[tbp]",
        "\\caption{Pooled weighted tardiness per instance, flexibility "
        "dividends, and capture ratio for each principal regime, on the "
        "pre-specified estimator (a) and with EDD dispatching every cell (b). "
        "The gates apply at $m \\in \\{0.6, 0.8\\}$. The level $m = 0.7$ is a "
        "robustness check, and $m = 1.0$ is the calibrated staffing. Every "
        "row pools the same 763 replay instances of the four verdict "
        "campuses. Intervals are 95\\% "
        "instance-cluster bootstrap intervals, a dash marks a regime below "
        "the 2\\% evaluability threshold, and no value is set in bold.}",
        "\\label{tab:regimes}", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular*}{\\tblwidth}{@{}l@{\\extracolsep{\\fill}}"
        "lrrrrrrrl@{}}",
        "\\toprule",
        "$m$ & $\\eta$ & L0 & CHAIN(1.0) & FULL & "
        "$\\Delta$(CHAIN) & $\\Delta$(FULL) & $\\Delta$(FULL)/L0 & "
        "$\\rho$ (95\\% CI) & Evaluable \\\\",
        "\\midrule"]
    for est, head in (("env", "(a) Best observed among the tested methods "
                       "(pre-specified Gate C estimator)"),
                      ("edd", "(b) Fixed EDD (no method selection)")):
        if est == "edd":
            lines += ["\\addlinespace[3pt]"]
        lines += ["\\multicolumn{10}{@{}l}{%s} \\\\" % head]
        for m, eta, lab in REGIMES:
            f = reg[(m, eta, est)]
            rho = ("%.3f (%.3f--%.3f)" % (f["rho"], f["ci_rho_resel"][0],
                                         f["ci_rho_resel"][1])
                   if f["evaluable"] else "--")
            lines.append(" & ".join([
                "%.1f" % m, "%.1f" % eta, _n(f["l0"]), _n(f["chain"]),
                _n(f["full"]), _n(f["d_chain"]), _n(f["d_full"]),
                "%s\\%%" % _n(f["d_full_pct"], 1), rho,
                "Yes" if f["evaluable"] else "No"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    return "\n".join(lines)


def _stack(*parts, align="r"):
    return "\\shortstack[%s]{%s}" % (align, "\\\\".join(parts))


def frameu_table(fu, load):
    """Two supplement tables: trade loads (tab:frameu-load), then weighted
    tardiness, the paired chain-minus-full difference and the
    dedicated-anchored ratio (tab:frameu)."""
    lines = [
        "%% Generated by analysis/gen_r2.py from results/e3 and data/e3 "
        "-- do not edit.",
        "\\begin{table}[!htb]",
        "\\caption{Offered load by trade in Frame U under the dedicated "
        "roster. A trade's load is its labor hours over its headcount "
        "times the 80~bh window, so a load above 1 exceeds the trade's own "
        "capacity. Each value is the mean over the 120 instances of a $U$ "
        "level, and the last two columns give the shares of trades, and of "
        "all labor hours, in trades loaded above 1.}",
        "\\label{tab:frameu-load}", "\\centering", "\\footnotesize",
        "\\begin{tabular*}{\\tblwidth}{@{}l@{\\extracolsep{\\fill}}"
        "rrrrrr@{}}",
        "\\toprule",
        " & ".join(["$U$", _stack("Aggregate", "load"),
                    _stack("Mean over", "trades"),
                    _stack("Hours-weighted", "mean"),
                    _stack("Most loaded", "trade"),
                    _stack("Trades", "above 1"),
                    _stack("Hours in trades", "above 1")])
        + " \\\\",
        "\\midrule"]
    for u in U_LEVELS:
        g = load[load.u == u]
        lines.append(" & ".join([
            "%.1f" % u, "%.2f" % g.u_agg.mean(),
            "%.2f" % g.trade_mean.mean(), "%.2f" % g.trade_hw.mean(),
            "%.2f" % g.trade_max.mean(),
            "%.0f\\%%" % (100 * g.share_trades_gt1.mean()),
            "%.0f\\%%" % (100 * g.share_hours_gt1.mean())])
            + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    lines += [
        "\\begin{table}[!htb]",
        "\\caption{Weighted tardiness in Frame U at matched aggregate "
        "offered load, as pooled mean TWT per instance with the paired "
        "chain-minus-full difference and the capture ratio against the "
        "dedicated roster. Panel (a) uses the best observed method and "
        "panel (b) fixed EDD. Intervals are 95\\% bootstrap intervals over "
        "instances resampled within campus and $U$. In panel (a) the best "
        "method is re-selected in every resample. A dash marks a family below the "
        "2\\% evaluability threshold, and no value is set in bold. $n$ = "
        "120 instances per $U$ level and 360 pooled.}",
        "\\label{tab:frameu}", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular*}{\\tblwidth}{@{}l@{\\extracolsep{\\fill}\\hspace{2\\tabcolsep}}"
        "lrrrrrrrl@{}}",
        "\\toprule",
        " & ".join(["$\\eta$", "$U$", "L0", "CHAIN(1.0)", "FULL",
                    _stack("CHAIN $-$ FULL", "(95\\% CI)"),
                    _stack("\\% of", "FULL"),
                    _stack("$\\Delta$(FULL)", "over L0"),
                    _stack("$\\rho$", "(95\\% CI)"), "Evaluable"])
        + " \\\\",
        "\\midrule"]
    for est, head in (("env", "(a) Best observed among the seven ranked "
                       "rules and both policy classes"),
                      ("edd", "(b) Fixed EDD (no method selection)")):
        if est == "edd":
            lines += ["\\addlinespace[3pt]"]
        lines += ["\\multicolumn{10}{@{}l}{%s} \\\\" % head]
        for eta in (1.0, 0.8):
            for i, name in enumerate(list(U_LEVELS) + ["ge1"]):
                f = fu[(name, eta, est)]
                ulab = ("$\\geq$1.0, pooled" if name == "ge1"
                        else "%.1f" % name)
                rho = ("%.2f (%.2f--%.2f)" % (f["rho"], f["ci_rho_resel"][0],
                                             f["ci_rho_resel"][1])
                       if f["evaluable"] else "--")
                lines.append(" & ".join([
                    "%.1f" % eta if i == 0 else "", ulab, _n(f["l0"], 0),
                    _n(f["chain"], 0), _n(f["full"], 0),
                    "%s (%s)" % (_n(f["chain_minus_full"], 0),
                                 _rng_txt(f["ci_cmf"], 0)),
                    "%s\\%%" % _n(f["chain_minus_full_pct"], 1),
                    "%s\\%%" % _n(f["d_full_pct"], 1), rho,
                    "Yes" if f["evaluable"] else "No"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def compute(root):
    root = Path(root)
    tier1 = _load(root, "tier1")
    tier2 = _load(root, "tier2")
    e8 = _load(root, "e8_m07")
    e3 = _load(root, "e3")
    master = sorted(tier1[(tier1.structure == "dedicated")
                          & tier1.campus.isin(VERDICT_CAMPUSES)]
                    .instance_id.unique())
    WA = stream_a(root, master)
    WB = stream_b(tier1, master)
    res = {"n_master": len(master)}
    res["reg"] = regimes(tier1, e8, master, WA, WB)
    res["breach"] = breaches(tier1, master, WB)
    res["phi"] = phi_shares(tier1, tier2, master, WA, WB)
    res["phi"]["product_rules_share_x_gatec_rho"] = (
        res["phi"]["rules"]["c05_c10"] * res["reg"][(0.6, 1.0, "env")]["rho"])
    res["map"] = decision_map(tier1)
    res["map_summary"] = map_summary(res["map"])
    res["fu"] = frameu(e3)
    res["fu_load"] = frameu_trade_load(root)
    res["fu_floor"] = frameu_floor(root, e3)
    return res


def _pct(x, nd=0):
    return "%.*f\\%%" % (nd, 100.0 * x)


def generate(numbers: dict, root: Path):
    root = Path(root)
    res = compute(root)
    reg, fu = res["reg"], res["fu"]

    (root / "paper/sections/gen_regimes.tex").write_text(regimes_table(reg))
    (root / "paper/sections/gen_frameu.tex").write_text(
        frameu_table(fu, res["fu_load"]))

    # ---- A-BOOT ----------------------------------------------------------
    h = reg[(0.6, 1.0, "env")]
    numbers["r2-boot-rho-resel"] = _ci3(h["rho"], h["ci_rho_resel"])
    numbers["r2-boot-rho-cond"] = _ci3(h["rho"], h["ci_rho_cond"])
    for k in ("l0", "chain", "full"):
        numbers["r2-boot-sel-%s" % k] = PRETTY[h["sel"][k]]
        numbers["r2-boot-selfreq-%s" % k] = "%.0f\\%%" % (
            100 * h["sel_freq"][k].get(h["sel"][k], 0.0))

    # ---- A-REG -----------------------------------------------------------
    for (m, eta, est), f in reg.items():
        t = "r2-reg-%s-%s" % (est, _tag(m, eta))
        numbers[t + "-l0"] = "%.2f" % f["l0"]
        numbers[t + "-chain"] = "%.2f" % f["chain"]
        numbers[t + "-full"] = "%.2f" % f["full"]
        numbers[t + "-dchain"] = "%.2f" % f["d_chain"]
        numbers[t + "-dfull"] = "%.2f" % f["d_full"]
        numbers[t + "-dfull-pct"] = "%.1f\\%%" % f["d_full_pct"]
        numbers[t + "-rho"] = (_ci3(f["rho"], f["ci_rho_resel"])
                               if f["evaluable"] else
                               "not evaluable (nominal %.3f)" % f["rho"])
        numbers[t + "-evaluable"] = "evaluable" if f["evaluable"] \
            else "not evaluable"
    b = res["breach"]
    numbers["r2-breach-l0-per100"] = "%.2f" % b["dedicated"]
    numbers["r2-breach-chain-per100"] = "%.2f" % b["chain"]
    numbers["r2-breach-full-per100"] = "%.2f" % b["full"]
    for st in ("chain", "full"):
        numbers["r2-breach-avoided-%s-per100" % st] = (
            "%.3f (95\\%% CI %.3f--%.3f)" % (
                b["avoided_" + st], *b["avoided_ci_" + st]))
        numbers["r2-breach-orders-per-avoided-%s" % st] = "%d" % (
            100 * round(b["orders_per_avoided_" + st] / 100))

    # ---- A-PHI -----------------------------------------------------------
    for est in ("rules", "gatec", "edd"):
        r = res["phi"][est]
        for key in ("c05_c10", "c05_full", "c025_c10", "c025_full",
                    "c10_full"):
            numbers["r2-phi-%s-%s" % (est, key.replace("_", "-"))] = (
                "%.0f\\%% (95\\%% CI %.0f--%.0f)" % (
                    100 * r[key], 100 * r["ci_" + key][0],
                    100 * r["ci_" + key][1]))
    numbers["r2-phi-product"] = "%.0f\\%%" % (
        100 * res["phi"]["product_rules_share_x_gatec_rho"])

    # ---- A-MAP -----------------------------------------------------------
    ms = res["map_summary"]
    for eta, tg in ((1.0, "eta10"), (0.8, "eta08")):
        s = ms[eta]
        numbers["r2-map-%s-cells" % tg] = str(s["n_cells"])
        for lab in ("tie", "classic", "flex", "policy"):
            numbers["r2-map-%s-%s" % (tg, lab)] = str(s["family"].get(lab, 0))
    numbers["r2-map-distinct-cells"] = str(ms["distinct"]["n_cells"])
    for lab in ("tie", "classic", "flex", "policy"):
        numbers["r2-map-distinct-%s" % lab] = str(
            ms["distinct"]["family"].get(lab, 0))

    # ---- A-FU ------------------------------------------------------------
    for (name, eta, est), f in ((k, v) for k, v in fu.items()
                                if k[0] != "fig"):
        ut = "u%03d" % round(100 * name) if name != "ge1" else "uge100"
        t = "r2-fu-%s-%s-e%02d" % (est, ut, round(10 * eta))
        numbers[t + "-cmf"] = "%s (95\\%% CI %s)" % (
            _n(f["chain_minus_full"], 1), _rng_txt(f["ci_cmf"], 1))
        numbers[t + "-cmf-pct"] = "%s\\%% (95\\%% CI %s)" % (
            _n(f["chain_minus_full_pct"], 2), _rng_txt(f["ci_cmf_pct"], 2))
        numbers[t + "-dfull-pct"] = "%.2f\\%%" % f["d_full_pct"]
        numbers[t + "-rho"] = (
            "%.2f (95\\%% CI %.2f--%.2f)" % (f["rho"], *f["ci_rho_resel"])
            if f["evaluable"] else "not evaluable")
        if f["evaluable"]:
            numbers[t + "-rho-pct"] = "%.0f\\%%" % (100 * f["rho"])
            numbers[t + "-rho-pctci"] = (
                "%.0f\\%% (95\\%% CI %.0f\\%%--%.0f\\%%)"
                % (100 * f["rho"], *(100 * x for x in f["ci_rho_resel"])))
    ld = res["fu_load"]
    for u in U_LEVELS:
        g = ld[ld.u == u]
        t = "r2-fu-load-u%03d" % round(100 * u)
        numbers[t + "-max"] = "%.2f" % g.trade_max.mean()
        numbers[t + "-hw"] = "%.2f" % g.trade_hw.mean()
        numbers[t + "-hours-gt1"] = "%.0f\\%%" % (100 * g.share_hours_gt1
                                                  .mean())
        numbers[t + "-trades-gt1"] = "%.0f\\%%" % (100 * g.share_trades_gt1
                                                   .mean())
    for name, f in res["fu_floor"].items():
        t = "r2-fu-floor-" + ("uge100" if name == "ge1"
                              else "u%03d" % round(100 * name))
        numbers[t + "-share"] = _pct(f["floor_share"], 1)
        numbers[t + "-full-removes-wait"] = _pct(f["full_removes_wait"], 1)
        numbers[t + "-chain-removes-wait"] = _pct(f["chain_removes_wait"], 1)
    return res


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "vendor"))
    numbers = json.load(open(root / "paper" / "numbers.json"))
    before = set(numbers)
    res = generate(numbers, root)
    for k in sorted(set(numbers) - before):
        print("%-44s %s" % (k, numbers[k]))
