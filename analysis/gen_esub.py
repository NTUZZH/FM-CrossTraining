"""E-SUB corrective-only and short-job robustness: table and numbers keys.

Reads results/esub/ (written by experiments/run_esub.py) and writes
paper/sections/gen_esub.tex plus the ``esub-`` keys of paper/numbers.json.

Two questions, one table.

(a) Decomposition. On the released instances and the released roster, the
    weighted tardiness of every work order is attributed to its class:
    corrective (``is_pm`` false) versus preventive, and a processing time of
    at most 8 business hours, one technician-day, versus more. The panel
    reports what share of the orders, of the labor hours, of the L0 weighted
    tardiness and of each dividend every class carries. The dispatcher is
    the cell's own envelope winner in the first block and EDD in the second.

(b) Matched-contention reruns. The preventive orders (respectively the
    orders above 8 bh) are dropped from every instance and the crews are
    recalibrated on the remaining stream, so the contention regime matches
    the reduced workload. The panel reports the dividend, the guard, the
    capture ratio and the penalized dividends for each reduced stream beside
    the full stream.

Definitions, dividends, the 2% guard, the capture ratio and the
instance-cluster interval follow analysis/gen_ecal.py, which follows the
released analysis.

Usage: the coordinator calls generate(numbers, root) from analysis/build_all.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.gen_ecal import (ENVELOPE, GUARD, THRESHOLD, VERDICT_CAMPUSES,
                               bootstrap_weights, cell_best, family_stats,
                               _num, _rng)

SHORT_BH = 8.0
DECOMP_M, DECOMP_ETA = 0.6, 1.0
SUBSET_MS = (1.0, 0.8, 0.6)
CLASSES = [("cm", "Corrective"), ("pm", "Preventive"),
           ("short", "$p_j \\leq 8$ bh"), ("long", "$p_j > 8$ bh")]
SUBSETS = [("full", "Full stream (released)"),
           ("cm", "Corrective only"), ("le8", "$p_j \\leq 8$ bh only")]
MTAG = {1.0: "m10", 0.8: "m08", 0.6: "m06"}


# --------------------------------------------------------------------------- #
# (a) Decomposition                                                            #
# --------------------------------------------------------------------------- #
def decomposition(root, m=DECOMP_M, eta=DECOMP_ETA):
    """Class components of the L0 total and of each dividend.

    For each cell the dispatcher is chosen first, by lowest pooled mean total
    weighted tardiness over the rules envelope (or fixed at EDD), and the
    class components are then read off that dispatcher's schedules.
    """
    d = pd.read_csv(root / "results/esub/decomp.csv")
    cl = pd.read_csv(root / "results/esub/classes.csv")
    cols = ["twt", "twt_cm", "twt_pm", "twt_short", "twt_long"]

    out = {"shares": {}, "env": {}, "edd": {}}
    n_orders = cl["n_orders"].sum()
    hours = cl["hours_total"].sum()
    for key, _name in CLASSES:
        out["shares"][key] = {
            "orders": float(cl["n_%s" % key].sum() / n_orders),
            "hours": float(cl["hours_%s" % key].sum() / hours)}
    out["n_orders"] = int(n_orders)
    out["n_instances"] = int(len(cl))

    for kind, methods in (("env", ENVELOPE), ("edd", ["edd"])):
        cells = {}
        for st, phi in (("dedicated", None), ("chain", 1.0), ("full", None)):
            sub = d[(d.structure == st) & (d.m == m)
                    & (d.method.isin(methods))]
            if st != "dedicated":
                sub = sub[sub.eta == eta]
            if phi is not None:
                sub = sub[sub.phi == phi]
            means = sub.groupby("method")[cols].mean()
            best = means["twt"].idxmin()
            cells[st] = {"method": best,
                         **{c: float(means.loc[best, c]) for c in cols}}
        blk = {"cells": cells, "l0": cells["dedicated"], "dividends": {}}
        for st in ("chain", "full"):
            div = {c: cells["dedicated"][c] - cells[st][c] for c in cols}
            blk["dividends"][st] = div
        out[kind] = blk
    return out


def class_share(block, key, which):
    """Share of one class in the L0 total or in one dividend."""
    if which == "l0":
        tot, part = block["l0"]["twt"], block["l0"]["twt_%s" % key]
    else:
        tot = block["dividends"][which]["twt"]
        part = block["dividends"][which]["twt_%s" % key]
    return (part / tot) if tot else float("nan")


# --------------------------------------------------------------------------- #
# (b) Matched-contention reruns                                                #
# --------------------------------------------------------------------------- #
def _matrix(df, structure, m, eta, master, phi=None):
    sub = df[(df.structure == structure) & (df.m == m)]
    if structure != "dedicated":
        sub = sub[sub.eta == eta]
    if phi is not None:
        sub = sub[sub.phi == phi]
    pos = {iid: k for k, iid in enumerate(master)}
    M = np.full((len(master), len(ENVELOPE)), np.nan)
    for j, meth in enumerate(ENVELOPE):
        s = sub[sub.method == meth]
        for iid, v in zip(s.instance_id, s.twt):
            if iid in pos:
                M[pos[iid], j] = v
    return M


def _families(df, master, W):
    env_cols = list(range(len(ENVELOPE)))
    edd_cols = [ENVELOPE.index("edd")]
    out = {}
    for m in SUBSET_MS:
        ded = _matrix(df, "dedicated", m, 1.0, master)
        for eta in (1.0, 0.8):
            cells = {"dedicated": ded,
                     "chain": _matrix(df, "chain", m, eta, master, phi=1.0),
                     "full": _matrix(df, "full", m, eta, master)}
            out[(m, eta, "env")] = family_stats(cells, env_cols, W)
            out[(m, eta, "edd")] = family_stats(cells, edd_cols, W)
    return out


def subsets(root):
    """Families for the full stream and for each reduced stream."""
    p = root / "results/esub/results.parquet"
    sub = pd.read_parquet(p) if p.exists() else pd.read_csv(
        root / "results/esub/results.csv")
    sub = sub[sub.method.isin(ENVELOPE)]

    q = root / "results/ecal/results.parquet"
    full = pd.read_parquet(q) if q.exists() else pd.read_csv(
        root / "results/ecal/results.csv")
    full = full[(full.calibration == "released")
                & full.method.isin(ENVELOPE)]

    out, counts = {}, {}
    master = sorted(full[full.structure == "dedicated"].instance_id.unique())
    campus_of = np.array([int(i.split("_")[0][1:]) for i in master])
    out["full"] = _families(full, master, bootstrap_weights(campus_of))
    counts["full"] = {"instances": len(master)}
    for kind in ("cm", "le8"):
        s = sub[sub.subset == kind]
        mm = sorted(s[s.structure == "dedicated"].instance_id.unique())
        co = np.array([int(i.split("_")[0][1:]) for i in mm])
        out[kind] = _families(s, mm, bootstrap_weights(co))
        counts[kind] = {"instances": len(mm)}
    return out, counts


def retained_counts(root):
    """Orders retained by each reduced stream."""
    cl = pd.read_csv(root / "results/esub/classes.csv")
    total = int(cl["n_orders"].sum())
    return {"full": total, "cm": int(cl["n_cm"].sum()),
            "le8": int(cl["n_short"].sum())}


# --------------------------------------------------------------------------- #
# Table                                                                        #
# --------------------------------------------------------------------------- #
def _pct(x, nd=1):
    return "%.*f\\%%" % (nd, 100.0 * x) if np.isfinite(x) else "--"


def _rho_cell(f):
    if f["guard"] or f["delta_full"] <= 0:
        return "--"
    if f["ci"] is None:
        return "%.2f" % f["rho"]
    return "%.2f (%.2f--%.2f)" % (f["rho"], f["ci"][0], f["ci"][1])


def build_table(dec, fam, counts, retained):
    n = dec["n_orders"]
    lines = ["%% Generated by analysis/gen_esub.py from results/esub/ "
             "-- do not edit.",
             "\\begin{table}[!htb]",
             "\\caption{Corrective work and single-technician work. "
             "Panel (a) attributes the weighted tardiness of the released "
             "cells at $m = 0.6$, $\\eta = 1.0$ to each order class: shares "
             "of the " + "{:,}".format(n) + " orders and of their labor "
             "hours, then of the L0 total and of each dividend, under the "
             "cell's envelope winner and under fixed EDD. Panel (b) drops "
             "one class from every instance, recalibrates the crews on the "
             "stream that remains and re-scores the ladder, so each reduced "
             "stream carries its own contention regime and its L0 level "
             "is not comparable across rows. A dash in a $\\rho$ column "
             "marks a family the 2\\% denominator guard declares not "
             "evaluable; a shown ratio carries a 95\\% instance-cluster "
             "bootstrap interval. $n$ = 763 instances per cell (762 on the "
             "corrective-only stream, whose one all-preventive instance is "
             "empty).}",
             "\\label{tab:esub}", "\\centering", "\\footnotesize",
             "\\begin{tabular*}{\\tblwidth}{@{}l@{\\extracolsep{\\fill}}"
             "rrrrrrrr@{}}",
             "\\toprule",
             "\\multicolumn{9}{@{}l}{(a) Where the tardiness and the "
             "dividend sit} \\\\",
             "& & & \\multicolumn{3}{c}{rules envelope} "
             "& \\multicolumn{3}{c}{fixed EDD} \\\\",
             "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}",
             "Order class & Orders & Hours & L0 & $\\Delta$(chain) & "
             "$\\Delta$(full) & L0 & $\\Delta$(chain) & $\\Delta$(full) "
             "\\\\",
             "\\midrule"]
    for key, name in CLASSES:
        row = [name, _pct(dec["shares"][key]["orders"]),
               _pct(dec["shares"][key]["hours"])]
        for kind in ("env", "edd"):
            row += [_pct(class_share(dec[kind], key, "l0")),
                    _pct(class_share(dec[kind], key, "chain")),
                    _pct(class_share(dec[kind], key, "full"))]
        lines.append(" & ".join(row) + " \\\\")

    lines += ["\\midrule",
              "\\addlinespace[2pt]",
              "\\multicolumn{9}{@{}l}{(b) Each stream with crews "
              "recalibrated on it} \\\\",
              "Stream & $m$ & Orders & L0 & $\\Delta$(FULL) & "
              "$\\rho$ (env.) & $\\rho$ (EDD) & "
              "\\multicolumn{2}{c}{$\\Delta$ at $\\eta{=}0.8$ (ch./full)} "
              "\\\\",
              "\\midrule"]
    for key, name in SUBSETS:
        for i, m in enumerate(SUBSET_MS):
            e = fam[key][(m, 1.0, "env")]
            d = fam[key][(m, 1.0, "edd")]
            e8 = fam[key][(m, 0.8, "env")]
            lines.append(" & ".join([
                name if i == 0 else "",
                "%.1f" % m,
                "{:,}".format(retained[key]) if i == 0 else "",
                _num(e["l0"]),
                "%s (%s\\%%)" % (_num(e["delta_full"]), _num(e["pct"])),
                _rho_cell(e),
                ("--" if (d["guard"] or d["delta_full"] <= 0)
                 else "%.2f" % d["rho"]),
                _num(e8["delta_chain"]), _num(e8["delta_full"]),
            ]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def generate(numbers: dict, root: Path):
    root = Path(root)
    res = root / "results/esub"
    if not (res / "decomp.csv").exists() or not (res / "results.csv").exists():
        return None
    dec = decomposition(root)
    fam, counts = subsets(root)
    retained = retained_counts(root)

    tex = build_table(dec, fam, counts, retained)
    out = root / "paper/sections/gen_esub.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)

    numbers["esub-n-orders"] = "{:,}".format(dec["n_orders"])
    numbers["esub-n-instances"] = "{:,}".format(dec["n_instances"])
    numbers["esub-short-bh"] = "8"

    # (a) class shares
    for key in ("cm", "pm", "short", "long"):
        tag = {"cm": "cm", "pm": "pm", "short": "short", "long": "long"}[key]
        numbers["esub-%s-share-orders" % tag] = _pct(
            dec["shares"][key]["orders"], 1)
        numbers["esub-%s-share-hours" % tag] = _pct(
            dec["shares"][key]["hours"], 1)
        for kind, suffix in (("env", ""), ("edd", "-edd")):
            numbers["esub-%s-share-l0twt%s" % (tag, suffix)] = _pct(
                class_share(dec[kind], key, "l0"), 1)
            numbers["esub-%s-share-div-chain%s" % (tag, suffix)] = _pct(
                class_share(dec[kind], key, "chain"), 1)
            numbers["esub-%s-share-div-full%s" % (tag, suffix)] = _pct(
                class_share(dec[kind], key, "full"), 1)

    # (b) subset families
    slug = {"full": "fullstream", "cm": "cm", "le8": "short"}
    for key in ("full", "cm", "le8"):
        s = slug[key]
        numbers["esub-%s-orders-retained" % s] = "{:,}".format(retained[key])
        numbers["esub-%s-instances" % s] = "{:,}".format(
            counts[key]["instances"])
        for m in SUBSET_MS:
            t = MTAG[m]
            e = fam[key][(m, 1.0, "env")]
            d = fam[key][(m, 1.0, "edd")]
            e8 = fam[key][(m, 0.8, "env")]
            numbers["esub-%s-l0-%s" % (s, t)] = "%.1f" % e["l0"]
            numbers["esub-%s-delta-full-%s" % (s, t)] = (
                "%.2f" % e["delta_full"])
            numbers["esub-%s-delta-full-pct-%s" % (s, t)] = (
                "%.1f\\%%" % e["pct"])
            numbers["esub-%s-guard-%s" % (s, t)] = (
                "fires" if e["guard"] else "clears")
            for kind, f in (("env", e), ("edd", d)):
                if not f["guard"] and f["delta_full"] > 0:
                    val = ("%.3f" % f["rho"] if f["ci"] is None
                           else "%.3f (95\\%% CI %.3f--%.3f)"
                           % (f["rho"], f["ci"][0], f["ci"][1]))
                    numbers["esub-%s-rho-%s-%s" % (s, kind, t)] = val
            numbers["esub-%s-eta08-order-%s" % (s, t)] = (
                "the chain delivers at least as much as full flexibility"
                if e8["delta_chain"] >= e8["delta_full"] else
                "full flexibility delivers more than the chain")

    # Headline reading: the corrective stream and the single-technician
    # stream are reported separately, because they answer different comments.
    def _range(key):
        vals = [fam[key][(m, 1.0, kind)]["rho"] for m in SUBSET_MS
                for kind in ("env", "edd")
                if not fam[key][(m, 1.0, kind)]["guard"]
                and fam[key][(m, 1.0, kind)]["delta_full"] > 0]
        return vals

    cm_rhos, short_rhos = _range("cm"), _range("le8")
    for key, vals in (("cm", cm_rhos), ("short", short_rhos)):
        if vals:
            numbers["esub-%s-rho-range" % key] = _rng(vals, "%.2f")
    cm_div = class_share(dec["env"], "cm", "chain")
    cm_hours = dec["shares"]["cm"]["hours"]
    long_l0 = class_share(dec["env"], "long", "l0")

    first = ("Corrective orders carry %s of the chain dividend and %s of the "
             "weighted tardiness at L0, against %s of the labor hours, and "
             "on the corrective-only stream with crews recalibrated on it the "
             "chain captures %s of what full flexibility delivers, so "
             "preventive work does not carry the result."
             % (_pct(cm_div, 0), _pct(class_share(dec["env"], "cm", "l0"), 1),
                _pct(cm_hours, 0),
                _rng(cm_rhos, "%.2f") if cm_rhos else "no evaluable share"))
    if short_rhos and min(short_rhos) >= THRESHOLD:
        second = ("On the stream of orders of at most 8 bh the chain captures "
                  "%s, so single-technician work supports the same reading."
                  % _rng(short_rhos, "%.2f"))
    elif short_rhos:
        second = ("On the stream of orders of at most 8 bh the chain captures "
                  "only %s, and that stream starts from %s weighted units of "
                  "tardiness at $m = 0.6$ against %s on the full stream, "
                  "because orders above 8 bh carry %s of the tardiness the "
                  "released cells accumulate."
                  % (_rng(short_rhos, "%.2f"),
                     _num(fam["le8"][(0.6, 1.0, "env")]["l0"]),
                     _num(fam["full"][(0.6, 1.0, "env")]["l0"]),
                     _pct(long_l0, 1)))
    else:
        second = ("On the stream of orders of at most 8 bh the dividend never "
                  "reaches the 2\\% guard, so the capture ratio is not "
                  "evaluable there.")
    numbers["esub-summary-sentence"] = first + " " + second

    with open(res / "analysis.json", "w") as f:
        json.dump({"decomposition": dec,
                   "subsets": {k: {"%s|%s|%s" % kk: vv
                                   for kk, vv in v.items()}
                               for k, v in fam.items()},
                   "retained": retained, "counts": counts}, f, indent=2)
    return str(out)


if __name__ == "__main__":
    n = {}
    print(generate(n, Path(__file__).resolve().parents[1]))
    for k in sorted(n):
        print("%-34s %s" % (k, n[k]))
