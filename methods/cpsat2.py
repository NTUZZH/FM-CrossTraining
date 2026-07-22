"""CP-SAT static exact model v2: overlapping eligibility + pair durations
(Y1 fmwos.cpsat extended, conventions preserved).

Changes vs Y1: eligibility enlarges to M_j = {u : g_j in S_u}; each optional
interval's size is the PAIR duration p(j, u) in centi-bh (primary: nominal
p_j; secondary: ceil_grid(p_j / eta), already integral on the centi grid).
e_j = s_j + p(j, u_assigned) is enforced by the presence-linked optional
intervals (exactly one per order).

Symmetry breaking: the used-prefix ordering is valid only within groups of
technicians with identical (skill set, primary) signatures; group by
signature and break within groups. DISABLED whenever ``tech_available`` is
supplied (rolling snapshots), exactly as Y1.

Rounding (Y1 convention, documented in the module it extends): starts on the
centi grid; the REPORTED end_bh = start_bh + p(j,u) with the float pair
duration so the v2 validator's exact duration check passes.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

from ortools.sat.python import cp_model

from env.conventions import pair_p_bh


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


def solve(instance: dict, overlay: dict | None = None,
          time_limit_s: float = 60.0, workers: int = 8,
          warm_start: dict | None = None,
          tech_available: dict[str, float] | None = None,
          flow_tiebreak: bool = False) -> dict:
    """Solve the static v2 instance (instance + overlay) with CP-SAT."""
    t_start = time.perf_counter()

    work_orders = instance["work_orders"]
    techs_src = (overlay or instance)["technicians"]
    eta = float((overlay or {}).get("eta",
                instance.get("meta", {}).get("eta", 1.0)) or 1.0)

    technicians = []
    for tech in techs_src:
        if "skills" in tech:
            prim = tech.get("primary") or tech["skills"][0]
            skills = tuple(tech["skills"])
        else:
            prim = tech["trade"]
            skills = (prim,)
        technicians.append({"id": tech["id"], "primary": prim,
                            "skills": skills})
    n = len(work_orders)

    rel_c = [_centi_ceil(w["release_bh"]) for w in work_orders]
    due_c = [_centi_round(w["due_bh"]) for w in work_orders]
    wt = [int(round(w["weight"])) for w in work_orders]

    # Pair durations in centi-bh, per (order, eligible tech index).
    elig = []            # j -> list of tech indices
    p_pair_c = []        # j -> {u: centi duration}
    p_pair_f = []        # j -> {u: float duration for reporting}
    for w in work_orders:
        g = w["trade"]
        e_j, pc_j, pf_j = [], {}, {}
        for u, tech in enumerate(technicians):
            if g not in tech["skills"]:
                continue
            pf = pair_p_bh(w["p_bh"], tech["primary"] == g, eta)
            e_j.append(u)
            pf_j[u] = pf
            pc_j[u] = _centi_ceil(pf)
        elig.append(e_j)
        p_pair_c.append(pc_j)
        p_pair_f.append(pf_j)

    base = max(rel_c) if rel_c else 0
    if tech_available:
        avail_c = [_centi_ceil(a) for a in tech_available.values()
                   if a and float(a) > 0.0]
        if avail_c:
            base = max(base, max(avail_c))
    horizon = base + sum(max(pc.values()) if pc else 0 for pc in p_pair_c)
    horizon = max(horizon, 1)

    model = cp_model.CpModel()

    s_vars, e_vars = [], []
    for j in range(n):
        pmax = max(p_pair_c[j].values()) if p_pair_c[j] else 0
        pmin = min(p_pair_c[j].values()) if p_pair_c[j] else 0
        s = model.NewIntVar(rel_c[j], horizon, "s_%d" % j)
        e = model.NewIntVar(rel_c[j] + pmin, horizon + pmax, "e_%d" % j)
        s_vars.append(s)
        e_vars.append(e)

    x = {}
    intervals_by_tech = defaultdict(list)
    for j in range(n):
        lits = []
        for u in elig[j]:
            b = model.NewBoolVar("x_%d_%d" % (j, u))
            x[(j, u)] = b
            iv = model.NewOptionalIntervalVar(
                s_vars[j], p_pair_c[j][u], e_vars[j], b, "iv_%d_%d" % (j, u))
            intervals_by_tech[u].append(iv)
            lits.append(b)
        model.AddExactlyOne(lits)

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

    obj_terms = []
    for j in range(n):
        T = model.NewIntVar(0, horizon + max(p_pair_c[j].values() or [0]),
                            "T_%d" % j)
        model.Add(T >= e_vars[j] - due_c[j])
        obj_terms.append(wt[j] * T)
    flow_K = n * (horizon + 1) + 1
    if flow_tiebreak:
        model.Minimize(sum(obj_terms) * flow_K + sum(e_vars))
    else:
        model.Minimize(sum(obj_terms))

    # Signature-grouped symmetry break (v2): identical (skills, primary)
    # technicians are interchangeable (identical pair durations for every
    # order), so used machines may be forced to form a prefix WITHIN a
    # signature group. Skipped with tech_available (Y1 precedent).
    if not tech_available:
        groups = defaultdict(list)
        for u, tech in enumerate(technicians):
            groups[(tech["primary"], tech["skills"])].append(u)
        for sig, us in groups.items():
            if len(us) < 2:
                continue
            used = {}
            for u in us:
                lits = [x[(j, u)] for j in range(n) if (j, u) in x]
                if not lits:
                    continue
                ub = model.NewBoolVar("used_%d" % u)
                model.AddMaxEquality(ub, lits)
                used[u] = ub
            ordered = [used[u] for u in us if u in used]
            for a, b in zip(ordered, ordered[1:]):
                model.Add(b <= a)

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
            for u in elig[j]:
                if solver.Value(x[(j, u)]) == 1:
                    assigned_u = u
                    break
            start_bh = solver.Value(s_vars[j]) / 100.0
            end_bh = start_bh + float(p_pair_f[j][assigned_u])
            assignments.append({
                "wo": wo["id"],
                "tech": technicians[assigned_u]["id"],
                "start_bh": start_bh,
                "end_bh": end_bh,
            })

    return {
        "instance_id": instance["meta"]["id"],
        "overlay_id": (overlay or {}).get("overlay_id"),
        "method": "cpsat%d" % int(time_limit_s),
        "seed": 0,
        "wall_seconds": wall,
        "decisions": int(solver.NumBranches()),
        "assignments": assignments,
        "status": status_name,
        "objective_bh": objective_bh,
        "best_bound_bh": best_bound_bh,
    }
