"""E-CREW hypothetical crew sizes for long orders: table and numbers keys.

Reads results/ecrew/ (written by experiments/run_ecrew.py) and writes
paper/sections/gen_ecrew.tex (Supplemental Table S20) plus the ``ecrew-`` keys
of paper/numbers.json.

Scenario: a work order longer than 8 bh needs
c_j = min(c_max, ceil(p_j / 8), k(g_j)) technicians at once, k(g) the primary
headcount of its trade at that crew multiplier; its elapsed time is its labor
hours over the crew's combined rate (1 per primary member, eta per secondary
member, rounded up to 0.01 bh). c_max = 1 is the released model and
reproduces results/tier1 bitwise (Supplemental Text S8).

Estimators, dividends, the 2% guard, the capture ratio and the
instance-cluster interval follow analysis/gen_ecal.py, which follows the
released analysis:

  fixed EDD        EDD dispatches every cell
  rules envelope   the lowest pooled mean over the seven ranked rules and
                   Random, re-selected per cell and inside every resample
  dividend         Delta(structure) = TWT(L0) - TWT(structure)
  capture ratio    rho = Delta(CHAIN(1.0)) / Delta(FULL)
  guard            not evaluable when Delta(FULL) < 2% of TWT(L0)
  interval         95% percentile interval, 10,000 resamples of base
                   instance ids within campus, seed 20260718

L0 runs at eta = 1.0 and stands for both eta values (tier1 convention).

Descriptives per (c_max, m), from results/ecrew/crew_sizes.csv and from the
L0 schedules of the L0 envelope winner (the rule whose pooled mean defines
the envelope's L0 level): share of orders that are crew orders, mean crew
size, share of L0 weighted tardiness carried by orders above 8 bh, mean wait
(start minus release) of crew orders and of single orders, technician-hours
held idle by reservations per instance.

Usage: the coordinator registers generate(numbers, root) in
analysis/build_all.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.gen_ecal import (ENVELOPE, GUARD, THRESHOLD, _num, _rng,
                               bootstrap_weights, family_stats)

CMAXES = (1, 2, 3, 4)
MS = (0.6, 0.8, 1.0)
ETAS = (1.0, 0.8)
MTAG = {1.0: "m10", 0.8: "m08", 0.6: "m06"}
LONG_BH = 8.0


def _etag(eta):
    return "" if eta == 1.0 else "-eta08"


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
def _load(root):
    p = root / "results/ecrew/results.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(
        root / "results/ecrew/results.csv")
    return df[df.method.isin(ENVELOPE)]


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
            M[pos[iid], j] = v
    if np.isnan(M).any():
        raise ValueError("missing rows in %s m=%s eta=%s"
                         % (structure, m, eta))
    return M


def families(df):
    """{(c_max, m, eta, kind): family_stats} for kind in env / edd."""
    master = sorted(df[(df.structure == "dedicated")
                       & (df.c_max == 1)].instance_id.unique())
    campus_of = np.array([int(i.split("_")[0][1:]) for i in master])
    W = bootstrap_weights(campus_of)
    env_cols = list(range(len(ENVELOPE)))
    edd_cols = [ENVELOPE.index("edd")]
    out = {}
    for c in CMAXES:
        sub = df[df.c_max == c]
        for m in MS:
            ded = _matrix(sub, "dedicated", m, 1.0, master)
            for eta in ETAS:
                cells = {"dedicated": ded,
                         "chain": _matrix(sub, "chain", m, eta, master,
                                          phi=1.0),
                         "full": _matrix(sub, "full", m, eta, master)}
                out[(c, m, eta, "env")] = family_stats(cells, env_cols, W)
                out[(c, m, eta, "edd")] = family_stats(cells, edd_cols, W)
    return out, len(master)


def descriptives(df, crews):
    """Per (c_max, m): crew shares from the audit file, and the L0 schedules
    of the L0 envelope winner and of EDD."""
    out = {}
    for c in CMAXES:
        for m in MS:
            cr = crews[(crews.c_max == c) & (crews.m == m)]
            n_orders = int(cr.n_orders.sum())
            n_crew = int(cr.n_size2.sum() + cr.n_size3.sum()
                         + cr.n_size4.sum())
            members = int(2 * cr.n_size2.sum() + 3 * cr.n_size3.sum()
                          + 4 * cr.n_size4.sum())
            n_long = int(cr.n_long.sum())
            d = {"n_orders": n_orders, "n_long": n_long, "n_crew": n_crew,
                 "crew_share": n_crew / n_orders,
                 "crew_share_of_long": n_crew / n_long,
                 "mean_crew_size": (members / n_crew) if n_crew else None,
                 "n_long_single": int(cr.n_long_single.sum()),
                 "size_counts": {k: int(cr["n_size%d" % k].sum())
                                 for k in (1, 2, 3, 4)}}
            l0 = df[(df.c_max == c) & (df.m == m)
                    & (df.structure == "dedicated")]
            means = l0.groupby("method").twt.mean()
            for tag, meth in (("env", means.idxmin()), ("edd", "edd")):
                s = l0[l0.method == meth]
                n_single = s.decisions - s.n_crew_orders
                wc = (s.mean_wait_crew.fillna(0) * s.n_crew_orders).sum()
                ws = (s.mean_wait_single.fillna(0) * n_single).sum()
                d[tag] = {
                    "method": meth,
                    "l0": float(s.twt.mean()),
                    "long_share_l0": float(s.twt_long.sum() / s.twt.sum()),
                    "wait_crew": (float(wc / s.n_crew_orders.sum())
                                  if s.n_crew_orders.sum() else None),
                    "wait_single": float(ws / n_single.sum()),
                    "held_bh": float(s.held_tech_bh.mean()),
                }
            out[(c, m)] = d
    return out


# --------------------------------------------------------------------------- #
# Table                                                                        #
# --------------------------------------------------------------------------- #
def _pct(x, nd=1):
    return "%.*f\\%%" % (nd, 100.0 * x)


def _rho_cell(f):
    if f["guard"] or f["delta_full"] <= 0:
        return "--"
    if f["ci"] is None:
        return _num(f["rho"], "%.2f")
    return "%s (%s--%s)" % (_num(f["rho"], "%.2f"), _num(f["ci"][0], "%.2f"),
                            _num(f["ci"][1], "%.2f"))


def _ci_key(f, nd=3):
    """gen_ecal's key format, with a typographic minus on negative bounds."""
    fmt = "%%.%df" % nd
    if f["ci"] is None:
        return _num(f["rho"], fmt)
    return "%s (95\\%% CI %s--%s)" % (_num(f["rho"], fmt),
                                       _num(f["ci"][0], fmt),
                                       _num(f["ci"][1], fmt))


def build_table(fam, desc, n_inst):
    lines = ["%% Generated by analysis/gen_ecrew.py from results/ecrew/ "
             "-- do not edit.",
             "\\begin{table}[!htb]",
             "\\caption{Capture ratio under hypothetical crews for orders "
             "longer than 8 bh (crew rule and reservation dispatch in Text "
             "S8). The crew-size cap $c_{\\max}$ limits the technicians an "
             "order may need at once, and $c_{\\max} = 1$ is the "
             "one-technician model of the main manuscript. Panel (a) "
             "describes the crew orders and the L0 schedules of the rule "
             "that sets the rules-envelope L0 level, including the mean wait "
             "from release to start and the technician-hours held idle per "
             "instance while crews assemble. In panels (b) and (c), L0, the "
             "dividends, and $\\Delta$(FULL) as a percentage of L0 are "
             "rules-envelope values. A "
             "dash in a $\\rho$ column marks a family below the 2\\% "
             "evaluability threshold on that estimator, and a shown $\\rho$ "
             "carries its 95\\% instance-cluster bootstrap interval. $n$ = "
             + str(n_inst) + " instances per cell.}",
             "\\label{tab:ecrew}", "\\centering", "\\footnotesize",
             "\\begin{tabular*}{\\tblwidth}{@{}l@{\\extracolsep{\\fill}}"
             "rrrrrrr@{}}",
             "\\toprule",
             "\\multicolumn{8}{@{}l}{(a) Crews and L0 schedules} \\\\",
             "& & \\multicolumn{2}{c}{Crew orders} & L0 TWT on "
             "& \\multicolumn{2}{c}{Mean wait (bh)} & Held (bh) \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){6-7}",
             "$m$ & $c_{\\max}$ & Share & Size & orders $>$ 8 bh "
             "& Crew & Single & per instance \\\\",
             "\\midrule"]
    for m in MS:
        for i, c in enumerate(CMAXES):
            d = desc[(c, m)]
            e = d["env"]
            lines.append(" & ".join([
                "%.1f" % m if i == 0 else "", str(c),
                _pct(d["crew_share"]) if d["n_crew"] else "--",
                "%.2f" % d["mean_crew_size"] if d["n_crew"] else "--",
                _pct(e["long_share_l0"]),
                "%.2f" % e["wait_crew"] if e["wait_crew"] is not None
                else "--",
                "%.2f" % e["wait_single"],
                "%.1f" % e["held_bh"]]) + " \\\\")
    for panel, eta in (("b", 1.0), ("c", 0.8)):
        lines += ["\\midrule", "\\addlinespace[2pt]",
                  "\\multicolumn{8}{@{}l}{(%s) Capture ratio, $\\eta = %.1f$}"
                  " \\\\" % (panel, eta),
                  "$m$ & $c_{\\max}$ & L0 & $\\Delta$(CHAIN) & "
                  "$\\Delta$(FULL) & \\% of L0 & $\\rho$ (EDD) & "
                  "$\\rho$ (envelope) \\\\",
                  "\\midrule"]
        for m in MS:
            for i, c in enumerate(CMAXES):
                e = fam[(c, m, eta, "env")]
                d = fam[(c, m, eta, "edd")]
                lines.append(" & ".join([
                    "%.1f" % m if i == 0 else "", str(c),
                    _num(e["l0"]), _num(e["delta_chain"], "%.2f"),
                    _num(e["delta_full"], "%.2f"),
                    _num(e["pct"]) + "\\%",
                    _rho_cell(d), _rho_cell(e)]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _evaluable(f):
    return (not f["guard"]) and f["delta_full"] > 0


def generate(numbers: dict, root: Path):
    root = Path(root)
    res = root / "results/ecrew"
    if not ((res / "results.parquet").exists()
            or (res / "results.csv").exists()) or \
            not (res / "crew_sizes.csv").exists():
        return None
    df = _load(root)
    crews = pd.read_csv(res / "crew_sizes.csv")
    fam, n_inst = families(df)
    desc = descriptives(df, crews)

    tex = build_table(fam, desc, n_inst)
    out = root / "paper/sections/gen_ecrew.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)

    d1 = desc[(1, 0.6)]
    numbers["ecrew-n-instances"] = "{:,}".format(n_inst)
    numbers["ecrew-long-bh"] = "%g" % LONG_BH
    numbers["ecrew-cmax-values"] = "1, 2, 3 and 4"
    numbers["ecrew-n-orders"] = "{:,}".format(d1["n_orders"])
    numbers["ecrew-n-long"] = "{:,}".format(d1["n_long"])
    numbers["ecrew-long-share-orders"] = _pct(d1["n_long"] / d1["n_orders"])

    # Per family: rho (both estimators), dividends, guard.
    for c in CMAXES:
        for m in MS:
            for eta in ETAS:
                sfx = "c%d-%s%s" % (c, MTAG[m], _etag(eta))
                for kind in ("env", "edd"):
                    f = fam[(c, m, eta, kind)]
                    numbers["ecrew-l0-%s-%s" % (kind, sfx)] = "%.1f" % f["l0"]
                    numbers["ecrew-dchain-%s-%s" % (kind, sfx)] = (
                        "%.2f" % f["delta_chain"])
                    numbers["ecrew-dfull-%s-%s" % (kind, sfx)] = (
                        "%.2f" % f["delta_full"])
                    numbers["ecrew-dfull-pct-%s-%s" % (kind, sfx)] = (
                        "%.1f\\%%" % f["pct"])
                    numbers["ecrew-guard-%s-%s" % (kind, sfx)] = (
                        "fires" if f["guard"] else "clears")
                    numbers["ecrew-rho-%s-%s" % (kind, sfx)] = (
                        _ci_key(f) if _evaluable(f) else "not evaluable")

    # Descriptives per (c_max, m).
    for c in CMAXES:
        for m in MS:
            d = desc[(c, m)]
            sfx = "c%d-%s" % (c, MTAG[m])
            numbers["ecrew-crew-share-%s" % sfx] = _pct(d["crew_share"])
            numbers["ecrew-crew-share-of-long-%s" % sfx] = _pct(
                d["crew_share_of_long"])
            numbers["ecrew-crew-size-%s" % sfx] = (
                "%.2f" % d["mean_crew_size"] if d["n_crew"] else "--")
            for tag in ("env", "edd"):
                e = d[tag]
                t = "" if tag == "env" else "-edd"
                numbers["ecrew-long-l0-share-%s%s" % (sfx, t)] = _pct(
                    e["long_share_l0"])
                numbers["ecrew-wait-crew-%s%s" % (sfx, t)] = (
                    "%.2f" % e["wait_crew"] if e["wait_crew"] is not None
                    else "--")
                numbers["ecrew-wait-single-%s%s" % (sfx, t)] = (
                    "%.2f" % e["wait_single"])
                numbers["ecrew-held-bh-%s%s" % (sfx, t)] = "%.1f" % e["held_bh"]

    # Ranges over the crew scenarios (c_max 2..4) for the evaluable families.
    def _vals(kinds, etas, ms=MS, cs=(2, 3, 4)):
        return [fam[(c, m, eta, k)]["rho"] for c in cs for m in ms
                for eta in etas for k in kinds
                if _evaluable(fam[(c, m, eta, k)])]

    def _count(kinds, etas, cs=(2, 3, 4)):
        fs = [fam[(c, m, eta, k)] for c in cs for m in MS for eta in etas
              for k in kinds]
        ev = [f for f in fs if _evaluable(f)]
        return len(fs), len(ev), sum(1 for f in ev if f["rho"] >= THRESHOLD)

    for kinds, ktag in ((("edd",), "edd"), (("env",), "env"),
                        (("edd", "env"), "both")):
        for etas, etag in (((1.0,), ""), ((0.8,), "-eta08")):
            v = _vals(kinds, etas)
            if v:
                numbers["ecrew-rho-range-%s%s" % (ktag, etag)] = _rng(v)
                numbers["ecrew-rho-range-%s%s-pct" % (ktag, etag)] = (
                    "%.0f\\%% to %.0f\\%%" % (100 * min(v), 100 * max(v)))
            n_all, n_ev, n_ok = _count(kinds, etas)
            numbers["ecrew-n-families-%s%s" % (ktag, etag)] = str(n_all)
            numbers["ecrew-n-evaluable-%s%s" % (ktag, etag)] = str(n_ev)
            numbers["ecrew-n-above-threshold-%s%s" % (ktag, etag)] = str(n_ok)
        v = _vals(kinds, (1.0,), ms=(0.6,))
        if v:
            numbers["ecrew-rho-range-%s-m06" % ktag] = _rng(v)

    v = [fam[(c, 0.6, 1.0, k)]["pct"] for c in (2, 3, 4)
         for k in ("env", "edd")]
    numbers["ecrew-dfull-pct-range-m06"] = _rng(v, "%.1f") + "\\%"
    v = [desc[(c, 0.6)]["env"]["long_share_l0"] for c in (2, 3, 4)]
    numbers["ecrew-long-l0-share-range-m06"] = (
        _rng([100 * x for x in v], "%.1f") + "\\%")
    v = [desc[(c, m)]["crew_share"] for c in (2, 3, 4) for m in MS]
    numbers["ecrew-crew-share-range"] = (
        _rng([100 * x for x in v], "%.1f") + "\\%")
    v = [fam[(c, m, 0.8, k)] for c in (2, 3, 4) for m in MS
         for k in ("env", "edd")]
    gt1 = [f for f in v if _evaluable(f) and f["rho"] > 1.0]
    numbers["ecrew-n-rho-above-one-eta08"] = str(len(gt1))

    # Length floor (analysis/ecrew_floor.py): the TWT no schedule can avoid.
    # L0 minus the floor is the tardiness caused by waiting for a technician,
    # the only part a workforce design can remove.
    fp = res / "floor.csv"
    if fp.exists():
        fl = pd.read_csv(fp).groupby(["m", "c_max"]).floor.mean()
        for c in CMAXES:
            for m in MS:
                floor = float(fl.loc[(m, c)])
                sfx = "c%d-%s" % (c, MTAG[m])
                numbers["ecrew-floor-%s" % sfx] = "%.1f" % floor
                for kind in ("env", "edd"):
                    f = fam[(c, m, 1.0, kind)]
                    wait = f["l0"] - floor
                    t = "" if kind == "env" else "-edd"
                    numbers["ecrew-floor-share-%s%s" % (sfx, t)] = _pct(
                        floor / f["l0"])
                    numbers["ecrew-wait-twt-%s%s" % (sfx, t)] = "%.1f" % wait
                    numbers["ecrew-full-removes-wait-%s%s" % (sfx, t)] = _pct(
                        f["delta_full"] / wait)
                    numbers["ecrew-chain-removes-wait-%s%s" % (sfx, t)] = _pct(
                        f["delta_chain"] / wait)

    with open(res / "analysis.json", "w") as f:
        json.dump({"families": {"c%d|%s|%s|%s" % k: v
                                for k, v in fam.items()},
                   "descriptives": {"c%d|%s" % k: v
                                    for k, v in desc.items()},
                   "n_instances": n_inst,
                   "guard": GUARD, "threshold": THRESHOLD}, f, indent=2)
    return str(out)


if __name__ == "__main__":
    n = {}
    print(generate(n, Path(__file__).resolve().parents[1]))
    for k in sorted(n):
        print("%-44s %s" % (k, n[k]))
