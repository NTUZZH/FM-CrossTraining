"""Size-resolved CP-SAT scale and runtime analysis (E-CP).

Reads the released static reference results/e1_static/ and resolves it by
instance size, so the exact model's cost can be read against the size of
the model it builds. Conventions are the ones the static-reference table
and Text S3 already use:

  * certified optimum = lowest validator TWT over the cpsat60 / cpsat300
    rows proved OPTIMAL for that (instance, structure, eta);
  * informative bound = strongest (largest) CP-SAT lower bound over the
    same two rows, counted only where it is non-zero against positive
    tardiness (a zero bound against positive tardiness brackets nothing);
  * gaps are means of the best ranked rule's distance, clipped at zero.

Model size is the number of optional intervals the CP-SAT model carries,
sum over orders of the count of technicians eligible for the order's
trade. It is computed exactly from the released instances and the overlay
builder, and does not depend on eta.

Writes paper/sections/gen_cpscale.tex and the "cpscale-" number keys.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
PAPER = ROOT / "paper"

from experiments.y1_root import y1_root                 # noqa: E402

Y1_ROOT = y1_root()

RANKED = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
SIZES = [50, 150, 400]
# (tag, structure, phi, eta, printed label)
CELLS = [("l0", "dedicated", None, 1.0, "L0"),
         ("c10", "chain", 1.0, 1.0, "CHAIN(1.0) (1.0)"),
         ("c08", "chain", 1.0, 0.8, "CHAIN(1.0) (0.8)"),
         ("f10", "full", None, 1.0, "FULL (1.0)"),
         ("f08", "full", None, 0.8, "FULL (0.8)")]
KEY = ["instance_id", "structure", "eta"]
CPSAT60_S = 60.0


# ---------------------------------------------------------------------------
def _read_e1():
    p = RES / "e1_static" / "results.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = RES / "e1_static" / "results.csv"
    return pd.read_csv(p) if p.exists() else None


def model_sizes(instance_ids):
    """Optional-interval count per (instance_id, structure).

    sum_j |M_j| with M_j = {u : trade(j) in S_u}, read off the overlay the
    runner builds for that campus and structure at m = 1.0.
    """
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "vendor") not in sys.path:
        sys.path.insert(0, str(ROOT / "vendor"))
    from overlays.build import build_overlay, load_crews

    cap = Y1_ROOT / "results/p1_calib/capacity.csv"
    inst_root = Y1_ROOT / "data/processed/instances"
    if not cap.exists() or not (inst_root / "index.csv").exists():
        return None

    index = {}
    with open(inst_root / "index.csv", newline="") as f:
        for r in csv.DictReader(f):
            index[r["id"]] = r

    overlays: dict = {}
    rows = []
    for iid in instance_ids:
        meta = index.get(iid)
        if meta is None:
            continue
        campus = int(meta["campus"])
        with open(inst_root / meta["path"]) as f:
            trades = [w["trade"] for w in json.load(f)["work_orders"]]
        for structure, phi in (("dedicated", None), ("chain", 1.0),
                               ("full", None)):
            k = (campus, structure, phi)
            if k not in overlays:
                ov = build_overlay(campus, load_crews(cap, campus),
                                   structure, phi, 1.0, 1.0)
                holders: dict = {}
                for t in ov["technicians"]:
                    for s in t["skills"]:
                        holders[s] = holders.get(s, 0) + 1
                overlays[k] = holders
            holders = overlays[k]
            rows.append({"instance_id": iid, "structure": structure,
                         "size": int(meta["size_class"]),
                         "n_orders": len(trades),
                         "intervals": sum(holders.get(g, 0)
                                          for g in trades)})
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
def _size_of(index_level):
    """Instance size class, read off the released instance id."""
    return index_level.map(lambda i: int(i.split("_")[2]))


def _block_stats(e1, ms, structure, eta, sizes):
    """Every quantity of one table row."""
    c60 = e1[(e1.method == "cpsat60") & (e1.structure == structure)
             & (e1.eta == eta) & (e1["size"].isin(sizes))]
    c300 = e1[(e1.method == "cpsat300") & (e1.structure == structure)
              & (e1.eta == eta) & (e1["size"].isin(sizes))]
    if not len(c60):
        return None
    n = len(c60)
    n60 = int((c60.proved_optimal == 1).sum())
    n300 = n60 + int((c300.proved_optimal == 1).sum())

    # Wall to proof: the 60 s run on its own where it proved, and the two
    # runs added where the tail proved it.
    walls = list(c60[c60.proved_optimal == 1].runtime_s.dropna())
    tail = c300[c300.proved_optimal == 1]
    if len(tail):
        base = c60.set_index("instance_id").runtime_s
        for iid, w in zip(tail.instance_id, tail.runtime_s):
            walls.append(float(base.get(iid, CPSAT60_S)) + float(w))
    walls = np.asarray(walls, dtype=float)

    # Gaps of the best ranked rule.
    sub = e1[e1["size"].isin(sizes)]
    opt = sub[(sub.method.isin(["cpsat60", "cpsat300"]))
              & (sub.proved_optimal == 1)].groupby(KEY).twt.min()
    bound = sub[sub.method.isin(["cpsat60", "cpsat300"])] \
        .groupby(KEY).best_bound_bh.max()
    rule = sub[sub.method.isin(RANKED)].groupby(KEY).twt.min()

    def _mask(frame):
        return ((frame.index.get_level_values("structure") == structure)
                & (frame.index.get_level_values("eta") == eta))

    j = pd.concat([opt, rule], axis=1).dropna()
    j.columns = ["opt", "rule"]
    j = j[_mask(j)]
    gap_opt = None
    if len(j):
        g = 100 * (j.rule - j.opt).clip(lower=0) / j.opt.replace(0, np.nan)
        gap_opt = float(g.mean())

    b = pd.concat([bound, rule], axis=1).dropna()
    b.columns = ["b", "rule"]
    b = b[_mask(b)]
    vac = (b.b <= 0) & (b.rule > 0)
    inf = b[~vac]
    gap_bnd = float(np.where(
        inf.rule > 0,
        100 * (inf.rule - inf.b).clip(lower=0) / inf.rule, 0.0).mean()) \
        if len(inf) else None

    lat = e1[(e1.method.isin(RANKED)) & (e1.structure == structure)
             & (e1.eta == eta) & (e1["size"].isin(sizes))]
    latency = float(lat.latency_ms_per_decision.median()) \
        if len(lat) else None

    intervals = None
    if ms is not None:
        mm = ms[(ms.structure == structure) & (ms["size"].isin(sizes))]
        if len(mm):
            intervals = float(mm.intervals.mean())

    return {
        "n": n, "n_certified": len(j),
        "p60": 100.0 * n60 / n, "p300": 100.0 * n300 / n,
        "median_wall": float(np.median(walls)) if len(walls) else None,
        "p90_wall": float(np.percentile(walls, 90)) if len(walls) else None,
        "gap_opt": gap_opt, "gap_bound": gap_bnd,
        "n_vacuous": int(vac.sum()), "latency_ms": latency,
        "intervals": intervals,
    }


def _pct(x, nd=0):
    return "--" if x is None else "%.*f\\%%" % (nd, x)


def _num(x, nd=1):
    return "--" if x is None else "%.*f" % (nd, x)


def _thousands(x):
    return "--" if x is None else "{:,}".format(int(round(x)))


# ---------------------------------------------------------------------------
def _table(blocks, n_per_row, note=None):
    lines = [
        "%% GENERATED by analysis/gen_cpscale.py from results/e1_static/"
        " -- do not edit.",
        "\\begin{table}[!htb]",
        "\\caption{Static reference resolved by instance size (E1$'$, "
        "$m = 1.0$): model size, proof rates, wall to proof, best-rule "
        "optimality gaps, and rule latency. $\\sum_j |M_j|$ counts the "
        "optional intervals of the CP-SAT model and is the mean over the "
        "instances of the row. Wall to proof covers the proved instances "
        "only, and adds the 60~s run to the 300~s re-run where the tail "
        "supplied the proof. Because it is measured to the certificate "
        "over proved instances, it differs from the median 60~s run wall "
        "of Table~\\ref{tab:static}, which is taken over every instance "
        "of the cell. The optimality gap is the best ranked rule's "
        "mean distance above the certified optimum on the certified "
        "subset; the bound gap is its mean distance above the strongest "
        "non-zero solver lower bound. A dash marks a row with no "
        "certified instance. Latency is the median per-decision wall of "
        "the ranked rules. Solver walls are as released, at two CP-SAT "
        "search workers per solve. $n$ = %d instances per row.}"
        % n_per_row,
        "\\label{tab:cpscale}", "\\centering", "\\footnotesize",
        "\\begin{tabular*}{\\tblwidth}{@{}l@{\\extracolsep{\\fill}}"
        "rrrrrrrr@{}}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{proved} "
        "& \\multicolumn{2}{c}{wall to proof (s), median / p90} "
        "& \\multicolumn{2}{c}{best-rule gap} & rule \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
        "$\\Lambda$ ($\\eta$) & $\\sum_j |M_j|$ & 60~s & 300~s "
        "& median & p90 & optimum & bound & ms/dec. \\\\",
        "\\midrule"]
    for si, size in enumerate(SIZES):
        if si:
            lines.append("\\midrule")
        lines.append("\\multicolumn{9}{@{}l}{\\emph{%d orders}} \\\\" % size)
        for tag, _st, _phi, _eta, label in CELLS:
            b = blocks.get((size, tag))
            if b is None:
                continue
            lines.append(
                "%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
                    label, _thousands(b["intervals"]),
                    _pct(b["p60"]), _pct(b["p300"]),
                    _num(b["median_wall"]), _num(b["p90_wall"]),
                    _pct(b["gap_opt"], 2), _pct(b["gap_bound"], 2),
                    _num(b["latency_ms"], 3)))
    lines += ["\\bottomrule", "\\end{tabular*}"]
    if note:
        lines.append("\\par\\smallskip {\\footnotesize %s}" % note)
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _summary(blocks):
    """Plain-English reading of how the two axes move."""
    l0_p60 = min(blocks[(s, "l0")]["p60"] for s in SIZES)
    c10_p60 = min(blocks[(s, "c10")]["p60"] for s in SIZES)
    f08_400 = blocks[(400, "f08")]
    iv_l0 = blocks[(400, "l0")]["intervals"]
    iv_fu = f08_400["intervals"]
    gaps_opt = [b["gap_opt"] for b in blocks.values()
                if b["gap_opt"] is not None]
    gaps_bnd = [b["gap_bound"] for b in blocks.values()
                if b["gap_bound"] is not None]
    lats = [b["latency_ms"] for b in blocks.values()
            if b["latency_ms"] is not None]
    return ("Certification cost follows eligibility overlap first and "
            "instance size second. With dedicated crews CP-SAT proves "
            "optimality on at least %.0f\\%% of instances within 60~s at "
            "every size from 50 to 400 orders, and the complete chain at "
            "full secondary speed holds the same rate. Overlapping "
            "eligibility enlarges the model and slows the proof together: "
            "at 400 orders full flexibility builds %s optional intervals "
            "against %s with dedicated crews, and under the 20\\%% "
            "efficiency penalty that block returns no certificate at all "
            "within 60~s or 300~s. Primal quality does not follow the "
            "certificate, because the best ranked rule stays within "
            "%.2f\\%% of the certified optimum and %.2f\\%% of the "
            "strongest non-zero lower bound in every block that certifies, "
            "and dispatches in %.3f to %.3f~ms per decision at every size."
            % (min(l0_p60, c10_p60), _thousands(iv_fu), _thousands(iv_l0),
               max(gaps_opt), max(gaps_bnd), min(lats), max(lats)))


# ---------------------------------------------------------------------------
def generate(numbers: dict, root: Path | None = None) -> dict:
    e1 = _read_e1()
    if e1 is None or "status" not in e1.columns:
        return numbers
    out_root = Path(root) if root is not None else ROOT

    ms = model_sizes(sorted(e1.instance_id.unique()))

    blocks = {}
    for size in SIZES:
        for tag, st, _phi, eta, _label in CELLS:
            b = _block_stats(e1, ms, st, eta, [size])
            if b is not None:
                blocks[(size, tag)] = b
    allsz = {}
    for tag, st, _phi, eta, _label in CELLS:
        b = _block_stats(e1, ms, st, eta, SIZES)
        if b is not None:
            allsz[tag] = b
    if not blocks:
        return numbers

    n_per_row = blocks[(SIZES[0], "l0")]["n"]

    # The one block whose lower bound is uninformative gets a table note,
    # so the zero in its bound column is not read as a measured gap.
    note = None
    worst = blocks.get((400, "f08"))
    if worst is not None and worst["n_vacuous"]:
        note = ("Note: at 400 orders under FULL (0.8) no instance carries "
                "a non-zero lower bound. Its bound-gap entry therefore "
                "covers only the %d instances whose best rule already "
                "reaches zero tardiness and is optimal for that reason; "
                "on the other %d the distance from the optimum is "
                "unknown." % (worst["n"] - worst["n_vacuous"],
                              worst["n_vacuous"]))
    with open(out_root / "paper" / "sections" / "gen_cpscale.tex", "w") as f:
        f.write(_table(blocks, n_per_row, note))

    # ---- number keys ----------------------------------------------------
    for size in SIZES:
        for tag, _st, _phi, _eta, _label in CELLS:
            b = blocks.get((size, tag))
            if b is None:
                continue
            p = "cpscale-%s%d" % (tag, size)
            numbers[p + "-p60"] = _pct(b["p60"])
            numbers[p + "-p300"] = _pct(b["p300"])
            numbers[p + "-wall"] = _num(b["median_wall"])
            numbers[p + "-wall-p90"] = _num(b["p90_wall"])
            numbers[p + "-model-size"] = _thousands(b["intervals"])
            if b["gap_bound"] is not None:
                numbers[p + "-bnd"] = _pct(b["gap_bound"], 2)
    for tag, _st, _phi, _eta, _label in CELLS:
        b = allsz.get(tag)
        if b is not None:
            numbers["cpscale-%s-p60-all-sizes" % tag] = _pct(b["p60"])
            numbers["cpscale-%s-p300-all-sizes" % tag] = _pct(b["p300"])

    # FULL at 400 orders, pooled over eta, the reading the main text uses
    # for its structure-level proof rates.
    f10, f08 = blocks[(400, "f10")], blocks[(400, "f08")]
    n_fu = f10["n"] + f08["n"]
    numbers["cpscale-full400-p60"] = "%.0f\\%%" % (
        (f10["p60"] * f10["n"] + f08["p60"] * f08["n"]) / n_fu)
    numbers["cpscale-full400-p300"] = "%.0f\\%%" % (
        (f10["p300"] * f10["n"] + f08["p300"] * f08["n"]) / n_fu)
    numbers["cpscale-model-size-full400"] = _thousands(f10["intervals"])
    numbers["cpscale-model-size-l0-400"] = _thousands(
        blocks[(400, "l0")]["intervals"])
    numbers["cpscale-model-size-chain400"] = _thousands(
        blocks[(400, "c10")]["intervals"])
    # At 400 orders FULL (0.8) certifies nothing, so there is no wall to
    # proof to report; the key carries the phrase the prose needs.
    if f08["median_wall"] is None:
        numbers["cpscale-median-wall-full08-400"] = (
            "no certificate within the 60~s and 300~s budgets")
        numbers["cpscale-p90-wall-full08-400"] = (
            "no certificate within the 60~s and 300~s budgets")
    else:
        numbers["cpscale-median-wall-full08-400"] = _num(f08["median_wall"])
        numbers["cpscale-p90-wall-full08-400"] = _num(f08["p90_wall"])
    numbers["cpscale-f08400-vacuous"] = str(f08["n_vacuous"])
    numbers["cpscale-f08400-informative"] = str(f08["n"] - f08["n_vacuous"])
    numbers["cpscale-l0-p60-all-sizes"] = _pct(allsz["l0"]["p60"])

    lats = [b["latency_ms"] for b in blocks.values()
            if b["latency_ms"] is not None]
    numbers["cpscale-rule-latency-range"] = "%.3f--%.3f~ms" % (min(lats),
                                                               max(lats))
    numbers["cpscale-n-per-row"] = str(n_per_row)
    numbers["cpscale-summary-sentence"] = _summary(blocks)
    return numbers


if __name__ == "__main__":
    nums = {}
    generate(nums, ROOT)
    for k in sorted(nums):
        print("%-38s %s" % (k, nums[k]))
