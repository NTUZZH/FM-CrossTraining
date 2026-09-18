"""Gate-threshold sensitivity, computed from the released result files.

The two pre-specified gates carry three numerical settings: the capture
ratio must reach 0.70 for the chaining-suffices reading, a family is
declared not evaluable when the full-flexibility dividend falls below 2% of
the dedicated baseline, and Gate P asks that at least 8 of 10 policy seeds
individually beat the best ranked rule. This module re-reads both verdicts
at every neighbouring setting of those three numbers and reports which
readings change.

Everything here is post hoc arithmetic on released results. No simulation is
run and no result file is written outside results/thr_sensitivity/.

Estimators are reused, not reimplemented:
  - dividends, the envelope TWT_best and the capture ratio come from
    analysis.gates (gate_c, twt_best, expand_l0);
  - the Gate P scopes, the pooled seed-mean pooling and the seed count come
    from analysis.gates.gate_p;
  - the fixed-EDD capture ratio and its instance-cluster bootstrap follow
    notes/supplementary/dividend_stats.py (base instances resampled within
    campus, B = 10,000, seed 20260718);
  - the envelope capture ratio's instance-cluster interval is the released
    one in notes/supplementary/cluster_stats_out.json.

Self-check: the recomputed capture ratios and seed counts are compared with
the released results/gates/gates.json before anything is written, and a
disagreement raises.

Usage: PYTHONPATH=.:vendor python analysis/gen_thr.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import analysis.gates as G

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CACHE = RES / "thr_sensitivity" / "thr_sensitivity.json"

RHO_THRESHOLDS = [0.60, 0.70, 0.80]
GUARD_LEVELS = [0.01, 0.02, 0.03, 0.05]
SEED_LEVELS = [6, 7, 8, 9]
RELEASED_RHO_THRESHOLD = 0.70
RELEASED_GUARD = 0.02
RELEASED_SEED_LEVEL = 8
EPSILON = 1.0                       # tie tolerance, weighted units
BOOT_B, BOOT_SEED = 10_000, 20260718
SCOPES = ["pooled", "m08", "m06"]
SCOPE_LABEL = {"pooled": "Pooled", "m08": "$m = 0.8$", "m06": "$m = 0.6$"}


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def _read(family):
    p = RES / family / "results.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(RES / family / "results.csv")


def _policy_pool(df, prefix, lo, hi):
    out = []
    for m in df.method.unique():
        if isinstance(m, str) and m.startswith(prefix):
            try:
                s = int(m[len(prefix):])
            except ValueError:
                continue
            if lo <= s <= hi:
                out.append(m)
    return sorted(out)


# --------------------------------------------------------------------------- #
# Fixed-EDD capture ratio with an instance-cluster bootstrap                   #
# --------------------------------------------------------------------------- #
def fixed_edd_rho(tier1, m):
    """Capture ratio under fixed EDD, with a 95% instance-cluster interval.

    Base instances are resampled within campus, which is the convention of
    the released cluster bootstrap.
    """
    fam = tier1[(tier1.m == m) & (tier1.eta == 1.0) & (tier1.method == "edd")]
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
    v = {st: piv[st].loc[common, "twt"].to_numpy()
         for st in ("dedicated", "chain", "full")}
    d_chain = (v["dedicated"] - v["chain"]).mean()
    d_full = (v["dedicated"] - v["full"]).mean()
    rng = np.random.default_rng(BOOT_SEED)
    idx_by_c = {c: np.flatnonzero(camp == c) for c in np.unique(camp)}
    stats = []
    for _ in range(BOOT_B):
        take = np.concatenate([rs[rng.integers(0, len(rs), len(rs))]
                               for rs in idx_by_c.values()])
        num = (v["dedicated"][take] - v["chain"][take]).mean()
        den = (v["dedicated"][take] - v["full"][take]).mean()
        stats.append(num / den)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    l0 = v["dedicated"].mean()
    return {"point": float(d_chain / d_full), "ci": [float(lo), float(hi)],
            "delta_chain": float(d_chain), "delta_full": float(d_full),
            "l0": float(l0), "delta_full_pct": float(100 * d_full / l0),
            "n_instances": int(len(common)), "B": BOOT_B, "seed": BOOT_SEED}


def binom_tail(k, n=10, p=0.5):
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j)
               for j in range(k, n + 1))


# --------------------------------------------------------------------------- #
# The computation                                                             #
# --------------------------------------------------------------------------- #
def compute():
    tier1 = _read("tier1")
    mlp = _policy_pool(tier1, "v2mlp", 301, 310)
    attn = _policy_pool(tier1, "v2attn", 401, 410)
    pools = {r: [r] for r in G.RANKED + ["random"]}
    if len(mlp) == 10:
        pools["policy_mlp"] = mlp
    if len(attn) == 10:
        pools["policy_attn"] = attn

    gc = G.gate_c(tier1, pools)
    families = {}
    for key, fam in gc["families"].items():
        l0 = fam["cells"]["dedicated"]["twt_best"]
        families[key] = {
            "l0": l0, "delta_chain": fam["delta_chain"],
            "delta_full": fam["delta_full"], "rho": fam["rho"],
            "delta_full_pct": (100.0 * fam["delta_full"] / l0
                               if l0 else float("nan")),
        }

    # The post-hoc m = 0.7 family, released in results/e8_m07.
    m07 = json.load(open(RES / "e8_m07" / "summary.json"))
    f07 = m07["families"]["m0.7_eta1.0"]
    families["m0.7_eta1.0"] = {
        "l0": f07["twt_best_L0"], "delta_chain": f07["delta_chain"],
        "delta_full": f07["delta_full"], "rho": f07["rho"],
        "delta_full_pct": f07["delta_full_pct_of_L0"], "post_hoc": True,
    }
    edd07 = m07["fixed_edd"]["m0.7_eta1.0"]["rho_fixed_edd"]

    edd = {"m0.6": fixed_edd_rho(tier1, 0.6),
           "m0.8": fixed_edd_rho(tier1, 0.8)}

    # Released envelope interval for the capture ratio (instance cluster).
    cj = json.load(open(ROOT / "notes" / "supplementary"
                        / "cluster_stats_out.json"))
    env_ci = {"m0.6": cj["item2_3"]["ci"]["b6i"]["rho"],
              "m0.8": cj["item2_3"]["ci"]["b8i"]["rho"]}

    vc = json.load(open(RES / "gates" / "verdict_class.json"))
    verdict = mlp if vc["verdict_class"] == "mlp" else attn
    gp = G.gate_p(tier1, verdict, verdict)
    flex = G.contended_flexible(G.expand_l0(tier1))
    subsets = {"pooled": flex, "m08": flex[flex.m == 0.8],
               "m06": flex[flex.m == 0.6]}
    gatep = {}
    for name in SCOPES:
        s = gp[name]
        sub = subsets[name]
        per_seed = {mth: float(sub[sub.method == mth].twt.mean())
                    for mth in verdict}
        ahead = sorted(k for k, x in per_seed.items()
                       if x < s["best_rule_mean"])
        gatep[name] = {
            "best_rule": s["best_rule"],
            "best_rule_mean": s["best_rule_mean"],
            "policy_pooled_mean": s["policy_pooled_mean"],
            "n_configs": s["n_configs"],
            "seeds_ahead": len(ahead), "seeds_ahead_ids": ahead,
            "per_seed_mean": per_seed,
            "pooled_mean_criterion": bool(s["mean_lower_all"]),
            "holm_criterion": bool(s["holm_all"]),
            "rules_policy_lower": [r for r in G.RANKED
                                   if s["rules"][r]["policy_lower"]],
            "rule_means": {r: s["rules"][r]["rule_pooled_mean"]
                           for r in G.RANKED},
        }

    # Self-check against the released gate evaluation.
    rel = json.load(open(RES / "gates" / "gates.json"))
    for key, fam in rel["gate_c"]["families"].items():
        if fam["rho"] is not None:
            got = families[key]["rho"]
            assert abs(got - fam["rho"]) < 1e-9, (
                "capture ratio %s: %r vs released %r" % (key, got,
                                                         fam["rho"]))
    for name in SCOPES:
        assert (gatep[name]["seeds_ahead"]
                == rel["gate_p"][name]["seeds_beating_best_rule"]), (
            "seed count %s: %d vs released %d"
            % (name, gatep[name]["seeds_ahead"],
               rel["gate_p"][name]["seeds_beating_best_rule"]))
    assert abs(edd["m0.6"]["point"] - 0.8735654581425445) < 1e-9, \
        "fixed-EDD capture ratio at m = 0.6 no longer reproduces"

    out = {
        "families": families, "fixed_edd": edd, "fixed_edd_m07": edd07,
        "envelope_rho_ci_instance_cluster": env_ci,
        "gate_p": gatep, "verdict_class": vc["verdict_class"],
        "rho_thresholds": RHO_THRESHOLDS, "guard_levels": GUARD_LEVELS,
        "seed_levels": SEED_LEVELS,
        "binomial_null": {str(k): binom_tail(k) for k in SEED_LEVELS},
        "epsilon": EPSILON,
        "epsilon_share_pct": {
            n: 100.0 * EPSILON / gatep[n]["best_rule_mean"] for n in SCOPES},
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(out, f, indent=1)
    return out


INPUTS = ["tier1/results.csv", "e8_m07/summary.json", "gates/gates.json",
          "gates/verdict_class.json"]


def _cache_fresh():
    if not CACHE.exists():
        return False
    stamp = CACHE.stat().st_mtime
    for rel in INPUTS:
        p = RES / rel
        if p.exists() and p.stat().st_mtime > stamp:
            return False
    return True


def load_or_compute():
    if _cache_fresh():
        try:
            return json.load(open(CACHE))
        except Exception:
            pass
    return compute()


# --------------------------------------------------------------------------- #
# Table and numbers                                                           #
# --------------------------------------------------------------------------- #
def _met(ok):
    return "met" if ok else "not met"


def _tex(res):
    fam = res["families"]
    m06 = fam["m0.6_eta1.0"]
    env_point = m06["rho"]
    env_floor = res["envelope_rho_ci_instance_cluster"]["m0.6"][0]
    edd_point = res["fixed_edd"]["m0.6"]["point"]
    edd_floor = res["fixed_edd"]["m0.6"]["ci"][0]
    guard_cols = [("$m = 0.6$", fam["m0.6_eta1.0"]["delta_full_pct"]),
                  ("$m = 0.7$", fam["m0.7_eta1.0"]["delta_full_pct"]),
                  ("$m = 0.8$", fam["m0.8_eta1.0"]["delta_full_pct"]),
                  ("$m = 0.6$, $\\eta = 0.8$",
                   fam["m0.6_eta0.8"]["delta_full_pct"])]

    lines = []
    lines.append("%% Generated by analysis/gen_thr.py from results/tier1, "
                 "results/e8_m07 and results/gates -- do not edit.")
    lines.append("\\begin{table}[!htb]")
    lines.append(
        "\\caption{Both pre-specified verdicts re-read at neighbouring "
        "settings of the three numerical thresholds, from the released "
        "results. Panel~A varies the capture-ratio threshold at $m = 0.6$ "
        "and $\\eta = 1.0$, the family that carries the chaining verdict. "
        "It is read against the point estimate and against the lower end of "
        "the 95\\% instance-cluster interval, on the rules-and-policy "
        "envelope and under fixed EDD. Panel~B varies the denominator "
        "guard. Each column head carries that family's full-flexibility "
        "dividend as a share of its dedicated baseline. The $m = 0.7$ "
        "family is post hoc, and the last column is the penalized family "
        "the chaining reading is quoted in. Panel~C varies the number of "
        "policy seeds Gate~P requires to beat the best ranked rule, "
        "alongside the probability that at least that many of ten fair coin "
        "flips come up ahead. $n$ = 763 instances per cell, and the policy "
        "pool is the ten verdict-class seeds.}")
    lines.append("\\label{tab:thr}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular*}{\\tblwidth}{@{}l@{\\extracolsep{\\fill}}"
                 "llll@{}}")
    lines.append("\\toprule")

    # Panel A
    lines.append("\\multicolumn{5}{@{}l}{A. Capture-ratio threshold "
                 "($m = 0.6$, $\\eta = 1.0$)} \\\\")
    lines.append("\\midrule")
    lines.append("Threshold & Envelope %.3f & CI floor %.3f & "
                 "Fixed EDD %.3f & CI floor %.3f \\\\"
                 % (env_point, env_floor, edd_point, edd_floor))
    for t in RHO_THRESHOLDS:
        label = ("%.2f (released)" % t if t == RELEASED_RHO_THRESHOLD
                 else "%.2f" % t)
        lines.append("%s & %s & %s & %s & %s \\\\"
                     % (label, _met(env_point >= t), _met(env_floor >= t),
                        _met(edd_point >= t), _met(edd_floor >= t)))

    # Panel B
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{@{}l}{B. Denominator guard; the first "
                 "three columns are at $\\eta = 1.0$} \\\\")
    lines.append("\\midrule")
    lines.append("Guard & " + " & ".join(
        "%s (%.1f\\%%)" % (name, pct) for name, pct in guard_cols) + " \\\\")
    for g in GUARD_LEVELS:
        label = ("%.0f\\%% (released)" % (100 * g) if g == RELEASED_GUARD
                 else "%.0f\\%%" % (100 * g))
        cells = ["evaluable" if pct >= 100 * g else "not evaluable"
                 for _name, pct in guard_cols]
        lines.append("%s & %s \\\\" % (label, " & ".join(cells)))

    # Panel C
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{@{}l}{C. Seeds Gate~P requires ahead of "
                 "the best ranked rule (1 of 10 are ahead in every scope)} "
                 "\\\\")
    lines.append("\\midrule")
    lines.append("Seeds required & Pooled & $m = 0.8$ & $m = 0.6$ & "
                 "Coin-flip $P$ \\\\")
    for k in SEED_LEVELS:
        label = ("%d of 10 (released)" % k if k == RELEASED_SEED_LEVEL
                 else "%d of 10" % k)
        cells = [_met(res["gate_p"][s]["seeds_ahead"] >= k) for s in SCOPES]
        lines.append("%s & %s & %s & %s & %.3f \\\\"
                     % (label, cells[0], cells[1], cells[2],
                        res["binomial_null"][str(k)]))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _numbers(res, numbers):
    fam = res["families"]
    m06, m08 = fam["m0.6_eta1.0"], fam["m0.8_eta1.0"]
    env_ci6 = res["envelope_rho_ci_instance_cluster"]["m0.6"]
    env_ci8 = res["envelope_rho_ci_instance_cluster"]["m0.8"]
    e6, e8 = res["fixed_edd"]["m0.6"], res["fixed_edd"]["m0.8"]

    numbers["thr-rho-env-point"] = "%.3f" % m06["rho"]
    numbers["thr-rho-env-ci"] = ("%.3f (95\\%% CI %.3f--%.3f)"
                                 % (m06["rho"], env_ci6[0], env_ci6[1]))
    numbers["thr-rho-edd-point"] = "%.3f" % e6["point"]
    numbers["thr-rho-edd-ci"] = ("%.3f (95\\%% CI %.3f--%.3f)"
                                 % (e6["point"], e6["ci"][0], e6["ci"][1]))
    numbers["thr-rho-lowest-floor"] = "%.3f" % min(env_ci6[0], e6["ci"][0])
    numbers["thr-rho-threshold-word"] = (
        "at 0.60 and at 0.70 on both estimators and both interval floors, "
        "and at 0.80 on both point estimates")

    numbers["thr-guard-m06-pct"] = "%.1f\\%%" % m06["delta_full_pct"]
    numbers["thr-guard-m06eta08-pct"] = ("%.1f\\%%"
                                         % fam["m0.6_eta0.8"]["delta_full_pct"])
    numbers["thr-guard-m07-pct"] = ("%.1f\\%%"
                                    % fam["m0.7_eta1.0"]["delta_full_pct"])
    numbers["thr-guard-m08-pct"] = "%.1f\\%%" % m08["delta_full_pct"]
    numbers["thr-m08-rho-env"] = ("%.3f (95\\%% CI %.3f--%.3f)"
                                  % (m08["rho"], env_ci8[0], env_ci8[1]))
    numbers["thr-m08-rho-edd"] = ("%.3f (95\\%% CI %.3f--%.3f)"
                                  % (e8["point"], e8["ci"][0], e8["ci"][1]))
    numbers["thr-m08-guard1-sentence"] = (
        "Under a 1\\%% guard the $m = 0.8$ family would have been read as "
        "evaluable. It would then have reported a capture ratio of %.3f on "
        "the envelope (95\\%% CI %.3f--%.3f) and %.3f under fixed EDD "
        "(95\\%% CI %.3f--%.3f). The chaining criterion would have been met "
        "on both point estimates. The fixed-EDD interval floor %.3f "
        "sits just below the 0.70 threshold."
        % (m08["rho"], env_ci8[0], env_ci8[1], e8["point"], e8["ci"][0],
           e8["ci"][1], e8["ci"][0]))

    for scope in SCOPES:
        g = res["gate_p"][scope]
        numbers["thr-gatep-seeds-ahead-%s" % scope] = (
            "%d of 10" % g["seeds_ahead"])
        numbers["thr-gatep-gap-%s" % scope] = (
            "%.2f" % (g["policy_pooled_mean"] - g["best_rule_mean"]))
    for k in SEED_LEVELS:
        numbers["thr-binom-%dof10" % k] = "%.3f" % res["binomial_null"][str(k)]
    numbers["thr-eps-share"] = ("%.2f\\%%"
                                % res["epsilon_share_pct"]["pooled"])
    numbers["thr-gatep-criterion-note"] = (
        "Two of Gate P's conditions fail on their own in every scope. The "
        "policy pool's mean is above EDD's in all three scopes, and one seed "
        "of ten is ahead of EDD in all three. The Holm-corrected tests are "
        "all significant, and six of the seven pooled comparisons run "
        "against the policy.")

    numbers["thr-summary-sentence"] = (
        "No reasonable setting of the three thresholds changes either "
        "verdict. The chaining criterion is met at 0.60 and at 0.70 on both "
        "estimators and on both interval floors, and at 0.80 on both point "
        "estimates. Loosening the guard to 1\\% makes the $m = 0.8$ family "
        "evaluable, and that family meets the criterion as well, while "
        "tightening it to 5\\% would have left no family evaluable at all. "
        "Gate~P fails at every seed level from 6 to 9 of 10 in all three "
        "scopes, because one seed of ten is ahead of the best rule and the "
        "pooled-mean criterion fails on its own.")
    return numbers


def generate(numbers: dict, root: Path = ROOT) -> Path:
    res = load_or_compute()
    _numbers(res, numbers)
    dst = Path(root) / "paper" / "sections" / "gen_thr.tex"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_tex(res))
    return dst


def main():
    res = compute()
    numbers = {}
    _numbers(res, numbers)
    dst = Path(ROOT) / "paper" / "sections" / "gen_thr.tex"
    dst.write_text(_tex(res))
    print("wrote", dst)
    print("wrote", CACHE)
    for k in sorted(numbers):
        print("%-34s %s" % (k, numbers[k]))


if __name__ == "__main__":
    main()
