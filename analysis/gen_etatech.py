"""Per-technician secondary-skill efficiency: dividends, capture ratio and
the chain-versus-full ordering.

Reads results/e13_etatech/, where the efficiency of secondary work is drawn
independently for every (technician, secondary trade) pair instead of being
one shared scalar, and results/e13_etatech/baseline.csv, the comparator arm.
The comparator arm is the released engine with the scalar efficiency on the
same frozen Y1 corpus, and its flexible cells reproduce the released tier1
and tier2 rows bitwise. Its dedicated cells supply the L0 reference, which
is efficiency invariant because all dedicated work is primary.

Quantities follow the released definitions in analysis.gates: a cell's
envelope is the lowest pooled mean weighted tardiness over the seven ranked
rules and Random; the dividend of a structure is L0's envelope minus that
structure's; the capture ratio is the chain dividend over the full-
flexibility dividend, computed on pooled dividends and reported only where
the full dividend clears the 2% denominator guard. Fixed EDD repeats the
dividends on the single EDD method.

Usage: PYTHONPATH=.:vendor python analysis/gen_etatech.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e13_etatech"

METHODS = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta",
           "random"]
STRUCTS = ["chain", "generalist", "full"]
MS = [0.6, 0.8]
DIST_LABEL = {"u60": "Per technician $U[0.60, 1.00]$",
              "u80": "Per technician $U[0.80, 1.00]$"}
GUARD = 0.02
CACHE = RES / "summary.json"


def _envelope(sub):
    vals = {m: sub[sub.method == m].twt.mean() for m in METHODS}
    vals = {k: v for k, v in vals.items() if pd.notna(v)}
    best = min(vals, key=vals.get)
    return float(vals[best]), best


def _edd(sub):
    return float(sub[sub.method == "edd"].twt.mean())


def _arm(cells, l0_env, l0_edd, mean_eta):
    """Dividends of one draw arm against the L0 reference."""
    out = {"mean_eta": float(mean_eta)}
    for st in STRUCTS:
        env, name = _envelope(cells[st])
        out["env_%s" % st] = env
        out["env_%s_method" % st] = name
        out["delta_%s" % st] = l0_env - env
        out["edd_%s" % st] = _edd(cells[st])
        out["delta_edd_%s" % st] = l0_edd - _edd(cells[st])
    d_full = out["delta_full"]
    out["delta_full_pct"] = 100.0 * d_full / l0_env
    out["guard_clears"] = bool(d_full >= GUARD * l0_env)
    out["rho"] = (out["delta_chain"] / d_full) if d_full else None
    out["chain_ge_full_env"] = bool(out["delta_chain"] >= d_full)
    out["chain_ge_full_edd"] = bool(out["delta_edd_chain"]
                                    >= out["delta_edd_full"])
    out["chain_minus_full_pct"] = 100.0 * (out["delta_chain"] - d_full) / l0_env
    return out


def compute():
    het = pd.read_csv(RES / "results.csv")
    base = pd.read_csv(RES / "baseline.csv")
    assert bool((het.validator_ok == 1).all()), "infeasible rows in e13"
    assert bool((base.validator_ok == 1).all()), "infeasible rows in baseline"
    seeds = sorted(int(s) for s in het.draw_seed.unique())
    dists = sorted(het.dist.unique())

    out = {"n_rows": int(len(het)), "n_instances": int(het.instance_id.nunique()),
           "draw_seeds": seeds, "distributions": dists,
           "guard_pct": GUARD, "families": {}}
    for m in MS:
        l0 = base[(base.structure == "dedicated") & (base.m == m)]
        l0_env, l0_name = _envelope(l0)
        l0_edd = _edd(l0)
        fam = {"l0_env": l0_env, "l0_env_method": l0_name, "l0_edd": l0_edd,
               "arms": {}}
        uni = base[(base.structure != "dedicated") & (base.m == m)]
        fam["arms"]["uniform"] = _arm(
            {st: uni[uni.structure == st] for st in STRUCTS},
            l0_env, l0_edd, 0.8)
        for dist in dists:
            hs = het[(het.dist == dist) & (het.m == m)]
            for seed in seeds:
                s = hs[hs.draw_seed == seed]
                fam["arms"]["%s_%d" % (dist, seed)] = _arm(
                    {st: s[s.structure == st] for st in STRUCTS},
                    l0_env, l0_edd, s.mean_eta.mean())
            fam["arms"]["%s_pooled" % dist] = _arm(
                {st: hs[hs.structure == st] for st in STRUCTS},
                l0_env, l0_edd, hs.mean_eta.mean())
        out["families"]["m%s" % m] = fam

    # Ordering tallies over the six draws of the headline family.
    six = ["%s_%d" % (d, s) for d in dists for s in seeds]
    for m in MS:
        arms = out["families"]["m%s" % m]["arms"]
        out["families"]["m%s" % m]["ordering"] = {
            "n_draws": len(six),
            "chain_ge_full_env": sum(arms[k]["chain_ge_full_env"]
                                     for k in six),
            "chain_ge_full_edd": sum(arms[k]["chain_ge_full_edd"]
                                     for k in six),
            "guard_clears": sum(arms[k]["guard_clears"] for k in six),
            "env_exceptions": [k for k in six
                               if not arms[k]["chain_ge_full_env"]],
        }
    CACHE.write_text(json.dumps(out, indent=1))
    return out


INPUTS = [RES / "results.csv", RES / "baseline.csv"]


def _cache_fresh():
    if not CACHE.exists():
        return False
    stamp = CACHE.stat().st_mtime
    return all(not p.exists() or p.stat().st_mtime <= stamp for p in INPUTS)


def load_or_compute():
    if _cache_fresh():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            pass
    return compute()


# --------------------------------------------------------------------------- #
# Table                                                                       #
# --------------------------------------------------------------------------- #
def _z(v, nd=2):
    """Drop a negative sign that rounding alone produced."""
    return 0.0 if abs(v) < 0.5 * 10 ** (-nd) else v


def _row(label, a):
    rho = ("%.2f" % a["rho"]) if (a["guard_clears"] and a["rho"] is not None) \
        else "--"
    return ("%s & %.3f & %.2f & %.2f & %.2f & %.1f\\%% & %s & %.2f & %.2f \\\\"
            % (label, a["mean_eta"], _z(a["delta_chain"]),
               _z(a["delta_generalist"]), _z(a["delta_full"]),
               _z(a["delta_full_pct"], 1), rho, _z(a["delta_edd_chain"]),
               _z(a["delta_edd_full"])))


def _tex(res):
    dists = res["distributions"]
    seeds = res["draw_seeds"]
    lines = []
    lines.append("%% Generated by analysis/gen_etatech.py from "
                 "results/e13_etatech/ -- do not edit.")
    lines.append("\\begin{table}[!htb]")
    lines.append(
        "\\caption{Flexibility dividends when the secondary-skill efficiency "
        "is drawn per technician and secondary trade instead of being one "
        "shared number. Dividends are in weighted tardiness units against "
        "the dedicated baseline of the same crew multiplier, on the "
        "seven-rule-and-Random envelope and under fixed EDD; $\\bar{\\eta}$ "
        "is the realized mean of the draw. A capture ratio is printed only "
        "where the full-flexibility dividend clears the 2\\%% denominator "
        "guard, and a dash marks the cells where it does not. The first row "
        "of each panel is the scalar penalty of the main grid. "
        "$n$ = %d instances per cell.}" % res["n_instances"])
    lines.append("\\label{tab:etatech}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular*}{\\tblwidth}{@{}l@{\\extracolsep{\\fill}}"
                 "rrrrrrrr@{}}")
    lines.append("\\toprule")
    lines.append("& & \\multicolumn{5}{c}{Rules envelope} & "
                 "\\multicolumn{2}{c}{Fixed EDD} \\\\")
    lines.append("\\cmidrule(lr){3-7}\\cmidrule(l){8-9}")
    lines.append("Draw & $\\bar{\\eta}$ & CHAIN & GEN & FULL & FULL/L0 & "
                 "$\\rho$ & CHAIN & FULL \\\\")
    for i, m in enumerate(MS):
        fam = res["families"]["m%s" % m]
        lines.append("\\midrule")
        lines.append("\\multicolumn{9}{@{}l}{%s. $m = %s$, dedicated "
                     "baseline %.2f} \\\\"
                     % ("AB"[i], m, fam["l0_env"]))
        lines.append("\\midrule")
        lines.append(_row("Scalar $\\eta = 0.8$", fam["arms"]["uniform"]))
        for dist in dists:
            lines.append("\\multicolumn{9}{@{}l}{%s} \\\\" % DIST_LABEL[dist])
            for seed in seeds:
                lines.append(_row("\\quad seed %d" % seed,
                                  fam["arms"]["%s_%d" % (dist, seed)]))
            lines.append(_row("\\quad pooled",
                              fam["arms"]["%s_pooled" % dist]))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Numbers                                                                     #
# --------------------------------------------------------------------------- #
def _rng(vals, fmt="%.2f"):
    return (fmt + "--" + fmt) % (min(vals), max(vals))


def _word(k, n):
    names = {0: "in none", 1: "in one", 2: "in two", 3: "in three",
             4: "in four", 5: "in five", 6: "in six"}
    if k == n:
        return "in all %s draws" % {2: "two", 3: "three", 6: "six"}.get(n, n)
    return "%s of the %s draws" % (names.get(k, "in %d" % k),
                                   {6: "six", 3: "three"}.get(n, n))


def _numbers(res, numbers):
    seeds = res["draw_seeds"]
    dists = res["distributions"]
    six = ["%s_%d" % (d, s) for d in dists for s in seeds]
    f6 = res["families"]["m0.6"]
    f8 = res["families"]["m0.8"]
    a6 = f6["arms"]
    o6 = f6["ordering"]

    numbers["etatech-n-rows"] = "{:,}".format(res["n_rows"])
    numbers["etatech-seeds"] = ", ".join(str(s) for s in seeds)
    numbers["etatech-n-draws"] = str(len(six))

    for dist in dists:
        tag = dist
        per = [a6["%s_%d" % (dist, s)] for s in seeds]
        numbers["etatech-%s-meaneta-band" % tag] = _rng(
            [p["mean_eta"] for p in per], "%.3f")
        numbers["etatech-%s-dchain-range" % tag] = _rng(
            [p["delta_chain"] for p in per])
        numbers["etatech-%s-dfull-range" % tag] = _rng(
            [p["delta_full"] for p in per])
        numbers["etatech-%s-dgen-range" % tag] = _rng(
            [p["delta_generalist"] for p in per])
        numbers["etatech-%s-edd-dchain-range" % tag] = _rng(
            [p["delta_edd_chain"] for p in per])
        numbers["etatech-%s-edd-dfull-range" % tag] = _rng(
            [p["delta_edd_full"] for p in per])
        pool = a6["%s_pooled" % dist]
        numbers["etatech-%s-dchain-pooled" % tag] = "%.2f" % pool["delta_chain"]
        numbers["etatech-%s-dfull-pooled" % tag] = "%.2f" % pool["delta_full"]
        numbers["etatech-%s-dfull-pct" % tag] = ("%.1f\\%%"
                                                 % pool["delta_full_pct"])
        numbers["etatech-%s-guard-word" % tag] = (
            "clears the 2\\% guard" if pool["guard_clears"]
            else "stays below the 2\\% guard")
        if pool["guard_clears"]:
            numbers["etatech-%s-rho" % tag] = "%.3f" % pool["rho"]
            numbers["etatech-%s-rho-range" % tag] = _rng(
                [p["rho"] for p in per], "%.3f")

    numbers["etatech-uniform-dchain"] = "%.2f" % a6["uniform"]["delta_chain"]
    numbers["etatech-uniform-dfull"] = "%.2f" % a6["uniform"]["delta_full"]
    numbers["etatech-uniform-dgen"] = ("%.2f"
                                       % a6["uniform"]["delta_generalist"])

    numbers["etatech-chain-vs-full-ordering-word"] = _word(
        o6["chain_ge_full_edd"], o6["n_draws"])
    numbers["etatech-chain-vs-full-env-word"] = _word(
        o6["chain_ge_full_env"], o6["n_draws"])
    exc = o6["env_exceptions"]
    gaps = sorted(abs(a6[k]["chain_minus_full_pct"]) for k in exc)
    if exc:
        numbers["etatech-env-exception-pct"] = ", ".join(
            "%.2f\\%%" % g for g in gaps)
        numbers["etatech-env-exception-max-pct"] = "%.2f\\%%" % gaps[-1]
        numbers["etatech-env-exception-clause"] = (
            "the exception being a reversal of %s of the dedicated baseline"
            % numbers["etatech-env-exception-pct"] if len(exc) == 1 else
            "the %s exceptions being reversals of %s of the dedicated "
            "baseline" % ({2: "two", 3: "three"}.get(len(exc), len(exc)),
                          " and ".join("%.2f\\%%" % g for g in gaps)))
    else:
        numbers["etatech-env-exception-clause"] = "with no exception"
    numbers["etatech-m08-word"] = (
        "no draw at $m = 0.8$ clears the guard, so the family stays not "
        "evaluable there, as it is under the scalar penalty")
    numbers["etatech-m08-dfull-max-pct"] = "%.1f\\%%" % max(
        f8["arms"][k]["delta_full_pct"] for k in six)

    e9 = ROOT / "results" / "e9_etahet" / "summary.json"
    if e9.exists():
        d9 = json.loads(e9.read_text())["per_draw"]
        ch9 = [v["delta_env_chain"] for v in d9.values()]
        fu9 = [v["delta_env_full"] for v in d9.values()]
        numbers["etatech-vs-pairdraw-sentence"] = (
            "The released per-trade-pair draws from $[0.70, 0.95]$ leave a "
            "chain dividend of %s over three draws against the scalar "
            "penalty's %.2f, and a full-flexibility dividend of %s against "
            "%.2f. The full dividend rises in all three of those draws and "
            "the chain dividend straddles its scalar value. Per-technician "
            "draws with the same mean lower both, because a trade-pair draw "
            "gives every technician of a trade one speed on a secondary "
            "trade and so averages out within the trade."
            % (_rng(ch9), a6["uniform"]["delta_chain"], _rng(fu9),
               a6["uniform"]["delta_full"]))

    numbers["etatech-heterogeneity-cost-sentence"] = (
        "Spreading the efficiency across technicians at a fixed mean is "
        "itself costly. At $m = 0.6$ the scalar penalty $\\eta = 0.8$ leaves "
        "a chain dividend of %.2f and a full-flexibility dividend of %.2f. "
        "Draws from $U[0.60, 1.00]$, whose mean is the same 0.80, leave "
        "%.2f and %.2f. A realized duration is convex in the efficiency, so "
        "the slow half of a technician pool costs more than the fast half "
        "saves. A dispatching rule that does not look at which technician is "
        "fast cannot recover the difference."
        % (a6["uniform"]["delta_chain"], a6["uniform"]["delta_full"],
           a6["u60_pooled"]["delta_chain"], a6["u60_pooled"]["delta_full"]))

    u80 = a6["u80_pooled"]
    both = (numbers["etatech-chain-vs-full-ordering-word"] ==
            numbers["etatech-chain-vs-full-env-word"] == "in all six draws")
    if both:
        order = ("Over six draws from two distributions at $m = 0.6$, the "
                 "chain dividend is the larger of the two in all six, on the "
                 "rules envelope and under fixed EDD alike.")
    else:
        rev = ("The %s envelope reversal is %s of the dedicated baseline."
               % ("single", gaps and "%.2f\\%%" % gaps[0])) \
            if len(exc) == 1 else \
            ("The %s envelope reversals are %s of the dedicated baseline."
             % ({2: "two", 3: "three"}.get(len(exc), len(exc)),
                " and ".join("%.2f\\%%" % g for g in gaps)))
        order = ("Over six draws from two distributions at $m = 0.6$, the "
                 "chain dividend is the larger of the two under fixed EDD "
                 "%s and on the rules envelope %s. %s"
                 % (numbers["etatech-chain-vs-full-ordering-word"].replace(
                        "in all six draws", "in all six"),
                    numbers["etatech-chain-vs-full-env-word"].replace(
                        " of the six draws", ""), rev))
    numbers["etatech-summary-sentence"] = (
        "Drawing the secondary-skill efficiency separately for every "
        "technician and every secondary trade does not overturn the "
        "chain-matches-full reading. %s Under $U[0.80, 1.00]$ the full "
        "dividend reaches %.1f\\%% of the dedicated baseline and clears the "
        "denominator guard, and the chain captures %.0f\\%% of it. Spreading "
        "the efficiency at a fixed mean does cost absolute dividend, because "
        "$U[0.60, 1.00]$ leaves the chain %.2f against the scalar penalty's "
        "%.2f."
        % (order, u80["delta_full_pct"], 100 * u80["rho"],
           a6["u60_pooled"]["delta_chain"], a6["uniform"]["delta_chain"]))
    return numbers


def generate(numbers: dict, root: Path = ROOT) -> Path:
    res = load_or_compute()
    _numbers(res, numbers)
    dst = Path(root) / "paper" / "sections" / "gen_etatech.tex"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_tex(res))
    return dst


def main():
    res = compute()
    numbers = {}
    _numbers(res, numbers)
    dst = ROOT / "paper" / "sections" / "gen_etatech.tex"
    dst.write_text(_tex(res))
    print("wrote", dst)
    print("wrote", CACHE)
    for k in sorted(numbers):
        print("%-38s %s" % (k, numbers[k]))


if __name__ == "__main__":
    main()
