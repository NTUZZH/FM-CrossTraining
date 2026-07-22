# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/cpsat.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""OR-Tools CP-SAT static exact model for the FM work-order scheduling benchmark.

Problem (notes/interfaces.md): unrelated parallel machines with trade
eligibility, release dates, weighted tardiness.  Because a technician serves one
trade and eligibility is exact trade-match, the machines are *identical within a
trade*; the model exploits that with a light symmetry break.

Model
-----
Per work order j:
  * one start var s_j and one end var e_j with e_j = s_j + p_j (centi-bh);
  * one assignment BoolVar x[j,u] per eligible technician u, AddExactlyOne;
  * one *optional* interval per eligible technician (presence x[j,u]) that shares
    s_j / e_j -- so a single pair of timing vars serves all its candidate techs;
  * AddNoOverlap over each technician's optional intervals (unary capacity);
  * tardiness T_j >= e_j - due_j, T_j >= 0; minimize sum_j w_j * T_j.

Symmetry break: identical technicians of a trade are ordered so that "used"
machines form a prefix (tech i may be used only if tech i-1 is used).  This is a
valid, objective-preserving break for identical parallel machines.  It is
DISABLED whenever ``tech_available`` is supplied, because per-technician
availability makes the machines no longer interchangeable (a busy technician and
a free one are not symmetric, so a prefix break could cut off the optimum).

warm_start: an existing schedule dict; its assignments become AddHint on the
assignment BoolVars and start vars (a warm start, never a hard constraint).

tech_available (rolling snapshots, notes/interfaces.md "Dynamic rolling track"):
an optional ``{tech_id: available_from_bh}`` map.  For each technician u with
``available_from_bh = a_u > 0`` a fixed dummy interval ``[0, a_u)`` (a constant
IntervalVar) is added to u's NoOverlap set, so u cannot process any work order
before ``a_u``.  This models a technician still busy on an in-progress job at the
snapshot instant without changing the schema.  Omitted / None => unchanged
behaviour (backward compatible; the static E1 path never sets it).

Rounding (documented)
---------------------
CP-SAT is integer, so bh are scaled to *centi-bh* (1 bh = 100 units).  To keep
the *reported* schedule feasible under the validator -- which requires
end_bh - start_bh == p_bh within 1e-6 and start_bh >= release_bh -- we report
start_bh on the centi grid but set end_bh = start_bh + p_bh with the *original*
float p_bh (so the duration check is exact).  So that this reported duration can
never create an overlap the model did not reserve, and so a start on the grid is
never earlier than the true release:

  * p_bh      -> ceil(p * 100)      (reserve at least the true processing time)
  * release   -> ceil(release * 100)(grid start >= true release)
  * due       -> round(due * 100)   (objective grid; nearest)

For values that are exact multiples of 0.01 (e.g. the hand-built fixture:
integers and quarter-hours) all three roundings are exact, so the model
objective, the reported schedule's WWT and the validator agree to the bit.  For
arbitrary floats ``objective_bh`` is the model's own (grid) objective and may
differ from the validator's WWT by the sub-centi discretization; both are
reported downstream.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

from ortools.sat.python import cp_model


def _centi_ceil(x: float) -> int:
    return int(math.ceil(float(x) * 100.0 - 1e-6))


def _centi_round(x: float) -> int:
    return int(round(float(x) * 100.0))


_STATUS_NAME = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}


def solve(
    instance: dict,
    time_limit_s: float = 60.0,
    workers: int = 8,
    warm_start: dict | None = None,
    tech_available: dict[str, float] | None = None,
    flow_tiebreak: bool = False,
) -> dict:
    """Solve the static instance exactly (to the time budget) with CP-SAT.

    Returns a schedule dict per notes/interfaces.md with extra keys
    ``status`` ('OPTIMAL'/'FEASIBLE'/...), ``objective_bh`` (float WWT of the
    model, or None if no solution) and ``best_bound_bh`` (float lower bound).
    """
    t_start = time.perf_counter()

    work_orders = instance["work_orders"]
    technicians = instance["technicians"]
    n = len(work_orders)

    p_c = [_centi_ceil(w["p_bh"]) for w in work_orders]
    rel_c = [_centi_ceil(w["release_bh"]) for w in work_orders]
    due_c = [_centi_round(w["due_bh"]) for w in work_orders]
    wt = [int(round(w["weight"])) for w in work_orders]

    # A safe horizon: latest of (release, technician availability) + total
    # processing (any WO finishes by then in any feasible schedule, since one
    # machine could in the worst case run the whole backlog after it frees).
    # tech_available is folded in so a job forced past a busy tech's free time
    # still fits inside the horizon.
    base = max(rel_c) if rel_c else 0
    if tech_available:
        avail_c = [_centi_ceil(a) for a in tech_available.values()
                   if a and float(a) > 0.0]
        if avail_c:
            base = max(base, max(avail_c))
    horizon = base + sum(p_c)
    horizon = max(horizon, 1)

    techs_by_trade = defaultdict(list)
    for u, tech in enumerate(technicians):
        techs_by_trade[tech["trade"]].append(u)

    model = cp_model.CpModel()

    s_vars = []
    e_vars = []
    for j in range(n):
        s = model.NewIntVar(rel_c[j], horizon, "s_%d" % j)
        e = model.NewIntVar(rel_c[j] + p_c[j], horizon, "e_%d" % j)
        model.Add(e == s + p_c[j])
        s_vars.append(s)
        e_vars.append(e)

    x = {}  # (j, u) -> BoolVar
    intervals_by_tech = defaultdict(list)
    for j, wo in enumerate(work_orders):
        eligible = techs_by_trade.get(wo["trade"], [])
        lits = []
        for u in eligible:
            b = model.NewBoolVar("x_%d_%d" % (j, u))
            x[(j, u)] = b
            iv = model.NewOptionalIntervalVar(
                s_vars[j], p_c[j], e_vars[j], b, "iv_%d_%d" % (j, u)
            )
            intervals_by_tech[u].append(iv)
            lits.append(b)
        # Exactly one eligible technician (infeasible model if eligible is empty,
        # which the spec forbids).
        model.AddExactlyOne(lits)

    # Per-technician availability (rolling snapshots): reserve [0, a_u) with a
    # constant dummy interval so a technician still busy on an in-progress job
    # cannot start any work order before it frees.  Added for every technician
    # with a_u > 0 (whether or not the solver ends up assigning it work).
    if tech_available:
        for u, tech in enumerate(technicians):
            a_u = float(tech_available.get(tech["id"], 0.0))
            if a_u > 0.0:
                a_c = _centi_ceil(a_u)
                if a_c > 0:
                    dummy = model.NewIntervalVar(0, a_c, a_c, "avail_%d" % u)
                    intervals_by_tech[u].append(dummy)

    for u, ivs in intervals_by_tech.items():
        model.AddNoOverlap(ivs)

    # Tardiness and objective.
    obj_terms = []
    for j in range(n):
        T = model.NewIntVar(0, horizon, "T_%d" % j)
        model.Add(T >= e_vars[j] - due_c[j])
        obj_terms.append(wt[j] * T)
    # flow_tiebreak (rolling snapshots only): lexicographic
    # (WWT, total completion time). Among WWT-equal plans prefer finishing
    # early -- without this, zero-tardiness snapshots have degenerate late-
    # start optima and the rolling executor "procrastinates" into the next
    # arrival burst (observed 2-9x WWT blowups; decisions.md 2026-07-05).
    # K = n*horizon + 1 > max possible sum(e_j) makes the order strict.
    flow_K = n * horizon + 1
    if flow_tiebreak:
        model.Minimize(sum(obj_terms) * flow_K + sum(e_vars))
    else:
        model.Minimize(sum(obj_terms))

    # Symmetry break: identical techs of a trade -> used machines form a prefix.
    # Only valid when technicians of a trade are truly interchangeable; with
    # per-technician availability (tech_available) they are not, so skip it.
    for trade, us in ({} if tech_available else techs_by_trade).items():
        if len(us) < 2:
            continue
        used = {}
        for u in us:
            lits = [x[(j, u)] for j in range(n) if (j, u) in x]
            if not lits:
                continue
            ub = model.NewBoolVar("used_%d" % u)
            model.AddMaxEquality(ub, lits)  # OR over the assignment bools
            used[u] = ub
        ordered = [used[u] for u in us if u in used]
        for a, b in zip(ordered, ordered[1:]):
            model.Add(b <= a)

    # Warm start (hints only).
    if warm_start:
        wo_index = {w["id"]: j for j, w in enumerate(work_orders)}
        tech_index = {t["id"]: u for u, t in enumerate(technicians)}
        for a in warm_start.get("assignments", []):
            j = wo_index.get(a.get("wo"))
            u = tech_index.get(a.get("tech"))
            if j is None:
                continue
            if u is not None and (j, u) in x:
                model.AddHint(x[(j, u)], 1)
            if a.get("start_bh") is not None:
                model.AddHint(s_vars[j], _centi_ceil(a["start_bh"]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    wall = time.perf_counter() - t_start

    status_name = _STATUS_NAME.get(status, str(status))
    if flow_tiebreak:
        # Extract the WWT component of the lexicographic objective.
        best_bound_bh = (solver.BestObjectiveBound() // flow_K) / 100.0
    else:
        best_bound_bh = solver.BestObjectiveBound() / 100.0

    assignments = []
    objective_bh = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if flow_tiebreak:
            objective_bh = (int(solver.ObjectiveValue()) // flow_K) / 100.0
        else:
            objective_bh = solver.ObjectiveValue() / 100.0
        for j, wo in enumerate(work_orders):
            assigned_u = None
            for u in techs_by_trade.get(wo["trade"], []):
                if solver.Value(x[(j, u)]) == 1:
                    assigned_u = u
                    break
            start_bh = solver.Value(s_vars[j]) / 100.0
            end_bh = start_bh + float(wo["p_bh"])  # exact duration for validator
            assignments.append(
                {
                    "wo": wo["id"],
                    "tech": technicians[assigned_u]["id"],
                    "start_bh": start_bh,
                    "end_bh": end_bh,
                }
            )

    return {
        "instance_id": instance["meta"]["id"],
        "method": "cpsat%d" % int(time_limit_s),
        "seed": 0,
        "wall_seconds": wall,
        "decisions": int(solver.NumBranches()),
        "assignments": assignments,
        "status": status_name,
        "objective_bh": objective_bh,
        "best_bound_bh": best_bound_bh,
    }
