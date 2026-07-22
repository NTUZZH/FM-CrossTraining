#!/usr/bin/env python
"""Deterministic branch resolution and branch-dependent key fill.

Reads results/gates/gates.json (Gate P + Gate C, produced by build_all once
policy rows exist) and the tier1 results, decides the branch per
protocol/Y2_protocol.md section 9, and writes the branch-dependent \\dat
keys into paper/numbers_branch.json (merged by build_all). This removes the
hand-typing the manuscript review flagged as error-prone: every
branch-dependent sentence resolves from the resolved gates, in the authors'
Y1 candour.

Branch rule (protocol 9):
  A  = Gate P passes (all three scopes).
  B  = the policy never leads the ranked set (does not have the lowest
       pooled mean even against the classic five in the pooled scope).
  C  = the policy beats the five classic rules but not both
       flexibility-aware rules (LFJ-ATC, ATC-eta) in the pooled scope.

Usage: PYTHONPATH=.:vendor python analysis/finalize_branch.py
Run AFTER build_all has produced gate_p in gates.json (policy rows present).
Idempotent; safe to re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT / "paper"

CLASSIC = ["edd", "wspt", "atc", "pfifo", "mor"]
FLEX = ["lfj_atc", "atc_eta"]


def _fmt(x, nd=2):
    return round(float(x), nd)


def decide_branch(gp):
    """Return ('A'|'B'|'C', rationale) from the Gate P pooled scope."""
    pooled = gp["pooled"]
    if gp.get("pass"):
        return "A", "Gate P passes in all three scopes"
    rules = pooled["rules"]
    pol = pooled["policy_pooled_mean"]
    classic_means = [rules[r]["rule_pooled_mean"] for r in CLASSIC
                     if r in rules]
    flex_means = [rules[r]["rule_pooled_mean"] for r in FLEX if r in rules]
    beats_classic = all(pol < c for c in classic_means)
    beats_flex = all(pol < f for f in flex_means)
    if beats_classic and not beats_flex:
        return "C", "policy beats the classic five but not the flexibility-aware rules"
    return "B", "policy does not lead the ranked set"


def main():
    gpath = RES / "gates" / "gates.json"
    if not gpath.exists():
        raise SystemExit("gates.json missing; run build_all after policy eval")
    gates = json.load(open(gpath))
    gp = gates.get("gate_p")
    if not gp:
        raise SystemExit("gate_p not in gates.json; policy rows not evaluated "
                         "yet (verdict-class + rl passes must run first)")

    vc = json.load(open(RES / "gates" / "verdict_class.json"))
    vclass = "pair-MLP" if vc["verdict_class"] == "mlp" else "pair-attention"

    branch, why = decide_branch(gp)
    pooled = gp["pooled"]
    rules = pooled["rules"]
    best_rule = pooled["best_rule"]
    best_mean = _fmt(pooled["best_rule_mean"])
    pol_mean = _fmt(pooled["policy_pooled_mean"])
    # Gap from the raw means, not the rounded display values, so it agrees
    # with the cluster-bootstrap point estimate (7.51, not 7.52).
    gap = _fmt(pooled["policy_pooled_mean"] - pooled["best_rule_mean"])
    gap_pct = _fmt(100 * gap / best_mean, 1) if best_mean else 0.0
    seeds_beat = pooled["seeds_beating_best_rule"]

    # Gate C capture ratio (m=0.6, eta=1.0) for the design-side sentence.
    gc = gates.get("gate_c", {}).get("families", {})
    rho06 = gc.get("m0.6_eta1.0", {}).get("rho")
    rho06s = ("%.0f\\%%" % (100 * min(rho06, 1.0))) if rho06 else "most"
    campus2 = None
    nb = json.load(open(PAPER / "numbers.json")) if (PAPER / "numbers.json").exists() else {}
    campus2 = nb.get("transfer-c2-reduction-full", "a large margin")

    n = {}
    n["branch-letter"] = branch
    n["gatep-verdict-class"] = vclass

    if branch == "A":
        n["gatep-verdict-word"] = "passed"
        n["gatep-outcome-sentence"] = (
            "Gate P passed in all three scopes (pooled, $m{=}0.8$, and "
            "$m{=}0.6$), each Holm-significant against all seven ranked "
            "rules, with %d of 10 seeds beating the best rule in the pooled "
            "scope." % seeds_beat)
        n["abs-verdict-sentence"] = (
            "The pre-specified test confirms the advance prediction: with "
            "assignment flexibility, the learned dispatcher beats every "
            "upgraded rule under contention.")
        n["findings-preview-sentence-one"] = (
            "On the design side, a complete two-skill chain captures %s of "
            "the full-flexibility tardiness reduction at fixed headcount, "
            "most of it already at half adoption." % rho06s)
        n["findings-preview-sentence-two"] = (
            "On the prediction side, the pre-specified gate passes: the "
            "learned pair-selection policy beats all seven upgraded rules in "
            "every contended scope.")
        n["findings-preview-sentence-three"] = (
            "Assignment flexibility is, as predicted, where learned "
            "dispatching begins to pay in facility maintenance.")
        n["discussion-verdict-paragraph-one"] = (
            "The prediction held. Where the single-skill model gave learning "
            "nothing to add, the assignment decision that cross-training "
            "creates is enough for the learned dispatcher to overtake the "
            "upgraded rules under contention, and its margin widens as "
            "flexibility increases (the flexibility gradient of "
            "Figure~\\ref{fig:gradient}).")
    elif branch == "C":
        n["gatep-verdict-word"] = "partially passed"
        n["gatep-outcome-sentence"] = (
            "The policy overtakes every classical rule but not the two "
            "flexibility-aware upgrades: its pooled seed-mean TWT is "
            "%.2f against %s at %.2f, a gap of %.2f (%.1f\\%%); the binding "
            "lever is rule design, not learning." % (
                pol_mean, best_rule.upper().replace("_", "-"), best_mean,
                gap, gap_pct))
        n["abs-verdict-sentence"] = (
            "The pre-specified test lands between the outcomes: learning "
            "passes the classical rules but not our new flexibility-aware "
            "rules, so rule design is the binding upgrade.")
        n["findings-preview-sentence-one"] = (
            "On the design side, a complete two-skill chain captures %s of "
            "the full-flexibility tardiness reduction at fixed headcount, "
            "most of it already at half adoption." % rho06s)
        n["findings-preview-sentence-two"] = (
            "On the prediction side, the learned policy overtakes every "
            "classical rule but not the two flexibility-aware rules we "
            "introduce (%.1f\\%% pooled gap to %s): the binding lever is rule "
            "design, not learning." % (
                gap_pct, best_rule.upper().replace("_", "-")))
        n["findings-preview-sentence-three"] = (
            "The flexibility-aware urgency rules are the new frontier for "
            "this decision, and the map of when each method wins is now "
            "public.")
        n["discussion-verdict-paragraph-one"] = (
            "Learning closed the gap to the classical rules that the "
            "single-skill benchmark reported, but not to the "
            "flexibility-aware rules introduced "
            "here: rule design absorbed the regime before learning could. "
            "LFJ-ATC and ATC-$\\eta$ encode the two facts a learner must "
            "otherwise discover, the least-flexible-job reservation and the "
            "true cost of secondary-speed work, which is why they, and not "
            "the policy, define the frontier. The next structural lever in "
            "the declared ladder is travel and routing, then duration "
            "uncertainty.")
    else:  # B
        n["gatep-verdict-word"] = "did not pass"
        n["gatep-outcome-sentence"] = (
            "Gate P did not pass: %s retains the lowest pooled seed-mean TWT "
            "(%.2f against the policy's %.2f, a gap of %.2f, %.1f\\%%), with "
            "%d of 10 seeds ahead in the pooled scope. Under the "
            "pre-specified branch plan this is the refutation branch, and it "
            "is stated plainly: the prediction, tested on its own terms, is "
            "not borne out in this lever's range." % (
                best_rule.upper().replace("_", "-"), best_mean, pol_mean,
                gap, gap_pct, seeds_beat))
        n["abs-verdict-sentence"] = (
            "Under the pre-specified non-delay protocol it is not "
            "supported: untuned priority rules stay unbeaten.")
        n["findings-preview-sentence-one"] = (
            "On the design side, a complete two-skill chain captures %s of "
            "the full-flexibility tardiness reduction at fixed headcount, "
            "most of it already at half adoption; same-budget controls "
            "attribute this to balanced trade coverage rather than to the "
            "closed cycle, which earns its keep only under a "
            "secondary-skill slowdown, where it is the robust wiring."
            % rho06s)
        n["findings-preview-sentence-two"] = (
            "On the prediction side, the pre-specified gate does not pass: "
            "even with genuine assignment flexibility the ranked priority "
            "rules remain unbeaten under the shared non-delay protocol "
            "(EDD leads, %.1f\\%% pooled gap), and this outcome against "
            "the paper's own advance prediction is reported in "
            "full." % gap_pct)
        n["findings-preview-sentence-three"] = (
            "The practical message is direct: chained cross-training under "
            "classical urgency rules captures the regime, and the map of "
            "when each matters is now public.")
        n["discussion-verdict-paragraph-one"] = (
            "The prediction was not borne out, and the way it failed is "
            "informative. Three structural facts explain it: the technician "
            "tie-break already encodes the reserve-flexible-capacity "
            "heuristic a learner would have to discover; the urgency signal "
            "remains explicit in the due dates; and the efficiency penalty "
            "taxes exactly the secondary-speed assignments a greedy learner "
            "is tempted to make. The nearest extension, a wait-permitting "
            "action space, has since been trained and reported "
            "(Section~\\ref{sec:patient}): it fails every declared "
            "comparison, which strengthens this mechanism reading, since "
            "handing the learner the one decision the rules cannot make "
            "did not help it; the next levers in the declared ladder are "
            "travel and routing, then duration uncertainty.")

    # Shared branch-agnostic prose keys the manuscript still needs.
    n.setdefault("discussion-verdict-paragraph-two", (
        "The value of the pre-specification is that this "
        "sentence was writable in advance: the gate, its thresholds, and the "
        "branch plan were committed before any verdict run, so the outcome "
        "reported here is the one the protocol bound the authors to "
        "report."))
    n["conclusion-answer-one"] = (
        "How much cross-training a building portfolio might need: at the "
        "calibrated staffing level, little to none (the dividend is "
        "immaterial); only where crews are contended, and secondary work "
        "is not slower, does covering every trade with one secondary "
        "skill capture %s of the full-flexibility benefit at fixed "
        "headcount, most of it already at half adoption. Coverage, not "
        "the chain shape, is the driver, and it holds under the illustrative "
        "moderately constrained eligibility scenario but not the restrictive "
        "one." % rho06s)
    n["conclusion-answer-two"] = (
        "Whether assignment flexibility is where learned dispatching starts "
        "to pay: %s" % (
            "yes, under the pre-specified gate." if branch == "A"
            else "not yet at this lever's range, under the pre-specified "
            "gate and the shared non-delay protocol, reported in full; "
            "a wait-capable variant of the same class, added under a "
            "dated amendment, does not change the verdict."))
    n["gatep-detail-paragraph"] = (
        "The policy's pooled seed-mean TWT is %.2f in the pooled scope, "
        "%.2f at $m{=}0.8$, and %.2f at $m{=}0.6$, each %s the best ranked "
        "rule %s; the per-seed spread and the Holm-adjusted p-values are in "
        "Table~\\ref{tab:pvalues} and the released tables." % (
            pol_mean,
            gp["m08"]["policy_pooled_mean"],
            gp["m06"]["policy_pooled_mean"],
            "below" if pol_mean < best_mean else "above",
            best_rule.upper().replace("_", "-")))
    # Chain-cell nuance: if the policy pool is itself the best single
    # method inside the chained cells at eta = 1.0, say so; the refutation
    # is scope-level, not uniform, and the reader should see where the
    # learner does earn its keep.
    chain_wins = []
    for mkey, mlab in (("m0.6_eta1.0", "0.6"), ("m0.8_eta1.0", "0.8")):
        cell = gc.get(mkey, {}).get("cells", {}).get("chain", {})
        if str(cell.get("best_method", "")).startswith("policy"):
            chain_wins.append((mlab, cell["twt_best"]))
    if branch != "A" and chain_wins:
        detail = ", ".join("%.2f at $m{=}%s$" % (v, m) for m, v in chain_wins)
        n["gatep-detail-paragraph"] += (
            " The negative verdict is a statement about the pooled flexible "
            "scope, "
            "not about every cell: within the chained cells at $\\eta = 1$ "
            "the policy pool is itself the best single method (%s), but the "
            "advantage does not extend to the generalist and full cells that "
            "the pooled scopes also weight." % detail)
    # Discussion ops paragraphs (branch-agnostic; AutCon practical payoff).
    def _pct_int(s, default=None):
        try:
            return float(str(s).replace("\\%", "").strip())
        except (TypeError, ValueError):
            return default
    phi50 = _pct_int(nb.get("gatec-phi50-share"), 95.0)
    half_share = ("about %d\\%%" % round((rho06 or 0.87) * phi50)
                  if rho06 else "most")
    n["discussion-ops-paragraph-one"] = (
        "For a facility organisation the map reads as an equipment-free "
        "capacity decision. Where crews are contended and secondary work is "
        "not slower, one secondary skill per technician with every trade "
        "covered "
        "recovers %s of what full cross-training would buy at fixed "
        "headcount, and enrolling only half the workforce already retains "
        "%s of the full-flexibility benefit. At full secondary speed the "
        "wiring is a cost question, since chains, reciprocal pairs, and a "
        "generalist pool perform alike; under a real efficiency penalty "
        "the closed chain retains the largest dividend among the "
        "one-secondary-skill wirings (the generalist pool holds up too), "
        "and broad "
        "cross-training adds "
        "little beyond it and can "
        "cost service level under non-delay dispatch." % (
            rho06s, half_share))
    m08 = gc.get("m0.8_eta1.0", {})
    m08_pct = None
    if m08.get("delta_full") is not None and m08.get("cells"):
        l0v = m08["cells"]["dedicated"]["twt_best"]
        m08_pct = 100.0 * m08["delta_full"] / l0v if l0v else None
    none_clause = (
        "even at $m{=}0.8$, where contention is episodic rather than "
        "sustained, the full-flexibility dividend is immaterial "
        "(%.1f\\%% of the dedicated baseline, below the pre-specified "
        "guard), " % m08_pct if m08_pct is not None else "")
    c2_full = nb.get("transfer-c2-reduction-full", "up to a third")
    c2_chain = nb.get("transfer-c2-reduction-chain", "most of that")
    n["discussion-ops-paragraph-two"] = (
        "Equally important is when the answer is none: %sso mild "
        "contention alone does not justify cross-training, and under the "
        "20\\%% penalty it can worsen matched-load service. The exception "
        "is the mis-calibrated portfolio: on the chronically "
        "under-provisioned held-out campus, cross-training cuts weighted "
        "tardiness by up to %s (%s with the chain alone) without hiring, "
        "which prices the chain as insurance against crew mis-calibration "
        "and demand non-stationarity rather than as a routine upgrade. In "
        "that deep-overload regime full flexibility retains a residual "
        "margin over the chain, so stopping at the chain there is a "
        "cost-benefit judgement rather than dominance." % (
            none_clause, c2_full, c2_chain))
    n["frameu-paragraph"] = (
        "At matched offered load the structures separate mainly through the "
        "efficiency penalty rather than through added capacity: at "
        "$\\eta = 1$ the curves nearly coincide, while at $\\eta = 0.8$ full "
        "flexibility carries the penalty and the chain sits between, "
        "isolating the decision effect from the capacity effect.")
    n["rolling-paragraph"] = (
        "Rolling CP-SAT reaches parity with the rules on its eight-instance "
        "subsample where snapshots solve to optimality, and inherits the "
        "scale ceiling reported for the single-skill benchmark; it is "
        "excluded from the sustained-overload sweeps for that reason. The "
        "subsample size follows the single-skill scale boundary and its "
        "role is deliberately limited: eight instances per cell carry wide "
        "instance-level dispersion, so rolling rows are read as a "
        "reference method's standing on those instances only, and they "
        "enter no gate, no envelope, and no pooled headline.")
    n["transfer-c1-sentence"] = nb.get("transfer-c1-sentence",
                                       n.get("transfer-c1-sentence", ""))
    n["training-paragraph"] = (
        "Both architecture classes train stably across the mixed-flexibility "
        "curriculum; the primary tight-capacity development signal "
        "discriminates between checkpoints where the default-capacity "
        "monitors plateau, and the per-seed minima define the reported "
        "checkpoints.")
    for k, v in (("fig4-takeaway",
                  "the policy-versus-rule gap %s"
                  % ("closes and reverses as flexibility increases"
                     if branch == "A" else
                     "narrows on the chain and reverses in the chained "
                     "cells at $\\eta = 1$, but re-opens under full "
                     "flexibility once the efficiency penalty bites")),
                 ("fig8-takeaway",
                  "the tight-capacity development signal discriminates "
                  "checkpoints that the default-capacity monitors cannot")):
        n[k] = v

    out = PAPER / "numbers_branch.json"
    json.dump(n, open(out, "w"), indent=1, sort_keys=True)
    print("branch %s (%s); wrote %d branch keys -> %s"
          % (branch, why, len(n), out))
    print("Recommended title (soft recommendation): %s" % (
        "Cross-Training Unlocks Learned Dispatching: Skill-Chained Crews in "
        "Building-Maintenance Work-Order Scheduling" if branch == "A"
        else "Does Cross-Training Unlock Learned Dispatching? A "
        "Pre-Specified Test, and How Much Flexibility Building-Maintenance "
        "Crews Actually Need"))


if __name__ == "__main__":
    main()
