"""Independent feasibility validator v2.

This module is the independent scorer for every reported v2 row. It is
deliberately
self-contained: it shares NO code with the engine, overlay builder, or any
method module, and re-derives every rule from the spec (including the
``ceil_grid`` eta convention, re-implemented locally). A bug in the benchmark
plumbing therefore cannot silently launder an infeasible schedule into a
"valid" one. Y1 heritage: fmwos.validator (vendored, untouched); v2 extends
it with overlay-aware skill legality and pair-realised durations.

Checks
------
(a) every work order assigned exactly once
(b) skill legality: the assigned technician u has g_j in S_u (overlay skills;
    an instance without overlay technicians falls back to Y1 trade equality)
(c) release respected: start_bh >= release_bh - 1e-9
(d) duration exact: end - start == p(j, u) within 1e-6, where
    p(j, u) = p_j if g_j == prim(u) else ceil_grid_local(p_j / eta)
    (eta = 1 -> p_j for every eligible technician)
(e) no overlap per technician
(f) schedule.instance_id matches instance.meta.id (and the overlay campus
    matches the instance campus)

Metrics: identical formulas and tolerances to Y1 (WWT, makespan, mean_flow,
breach shares per priority); at L0 the output is byte-identical to Y1's
validator on Y1 schedules (unit-tested).
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

_REL_TOL = 1e-9
_DUR_TOL = 1e-6
_OVL_TOL = 1e-9
_BREACH_TOL = 1e-9


def _ceil_grid_local(x_bh: float) -> float:
    """Independent re-derivation of the 0.01 bh round-up (spec 1.4)."""
    return math.ceil(x_bh * 100.0 - 1e-6) / 100.0


def _pair_p_local(p_bh: float, primary: bool, eta: float) -> float:
    if primary or eta >= 1.0:
        return float(p_bh)
    return _ceil_grid_local(float(p_bh) / float(eta))


def validate(instance, schedule, overlay=None):
    """Validate ``schedule`` against ``instance`` (+ optional ``overlay``)."""
    violations = []

    work_orders = instance.get("work_orders", []) or []
    wo_by_id = {wo["id"]: wo for wo in work_orders}

    # Technician skill map: overlay wins; else instance technicians (either
    # v2 dicts with skills or Y1 dicts with a single trade).
    techs = (overlay or instance).get("technicians", []) or []
    skills_by_id = {}
    prim_by_id = {}
    for tech in techs:
        tid = tech["id"]
        if "skills" in tech:
            prim = tech.get("primary") or tech["skills"][0]
            skills_by_id[tid] = set(tech["skills"])
        else:
            prim = tech["trade"]
            skills_by_id[tid] = {prim}
        prim_by_id[tid] = prim

    eta = 1.0
    if overlay is not None:
        eta = float(overlay.get("eta", 1.0))
        inst_campus = instance.get("meta", {}).get("campus")
        if int(overlay.get("campus", -1)) != int(inst_campus):
            violations.append(
                "(f) overlay campus %r does not match instance campus %r"
                % (overlay.get("campus"), inst_campus))
    elif "eta" in instance.get("meta", {}):
        eta = float(instance["meta"]["eta"] or 1.0)

    instance_id = instance.get("meta", {}).get("id")
    assignments = schedule.get("assignments", []) or []

    sched_instance_id = schedule.get("instance_id")
    if sched_instance_id != instance_id:
        violations.append(
            "(f) schedule.instance_id %r does not match instance.meta.id %r"
            % (sched_instance_id, instance_id))

    assign_counts = {}
    for a in assignments:
        wid = a.get("wo")
        assign_counts[wid] = assign_counts.get(wid, 0) + 1
    for wid in sorted(w for w in wo_by_id if w not in assign_counts):
        violations.append("(a) work order %r is never assigned (missing)"
                          % wid)
    for wid in sorted(w for w, c in assign_counts.items() if c > 1):
        violations.append("(a) work order %r is assigned %d times (duplicated)"
                          % (wid, assign_counts[wid]))
    for wid in sorted(w for w in assign_counts if w not in wo_by_id):
        violations.append("(a) assignment references work order %r which is "
                          "not in the instance" % wid)

    for a in assignments:
        wid = a.get("wo")
        tid = a.get("tech")
        wo = wo_by_id.get(wid)
        start = a.get("start_bh")
        end = a.get("end_bh")

        skills = skills_by_id.get(tid)
        if skills is None:
            violations.append(
                "(b) assignment for work order %r uses technician %r which "
                "does not exist" % (wid, tid))
        elif wo is not None and wo.get("trade") not in skills:
            violations.append(
                "(b) technician %r (skills %s) is not skill-eligible for "
                "work order %r (trade %r)"
                % (tid, sorted(skills), wid, wo.get("trade")))

        if wo is None:
            continue

        if start is not None and start < wo["release_bh"] - _REL_TOL:
            violations.append(
                "(c) work order %r starts at %s before its release_bh %s"
                % (wid, start, wo["release_bh"]))

        if start is not None and end is not None and skills is not None \
                and wo.get("trade") in skills:
            primary = (prim_by_id.get(tid) == wo.get("trade"))
            expect = _pair_p_local(wo["p_bh"], primary, eta)
            duration = end - start
            if abs(duration - expect) > _DUR_TOL:
                violations.append(
                    "(d) work order %r on technician %r has duration %s "
                    "(end %s - start %s) which does not equal p(j,u) %s "
                    "(p_bh %s, primary %s, eta %s)"
                    % (wid, tid, duration, end, start, expect,
                       wo["p_bh"], primary, eta))

    by_tech = {}
    for a in assignments:
        by_tech.setdefault(a.get("tech"), []).append(a)
    for tid, jobs in by_tech.items():
        ordered = sorted(jobs, key=lambda x: _num(x.get("start_bh")))
        for prev, cur in zip(ordered, ordered[1:]):
            if _num(cur.get("start_bh")) < _num(prev.get("end_bh")) - _OVL_TOL:
                violations.append(
                    "(e) technician %r: work order %r starts at %s before "
                    "work order %r ends at %s (overlap)"
                    % (tid, cur.get("wo"), cur.get("start_bh"),
                       prev.get("wo"), prev.get("end_bh")))

    metrics = _compute_metrics(schedule, wo_by_id)
    return {"feasible": len(violations) == 0,
            "violations": violations,
            "metrics": metrics}


def _num(x):
    return float(x) if x is not None else 0.0


def _compute_metrics(schedule, wo_by_id):
    """Y1 metric formulas, verbatim (fmwos.validator._compute_metrics)."""
    assignments = schedule.get("assignments", []) or []

    wwt = 0.0
    ends = []
    flows = []
    n = 0
    breaches = 0
    prio_total = {1: 0, 2: 0, 3: 0, 4: 0}
    prio_breach = {1: 0, 2: 0, 3: 0, 4: 0}

    for a in assignments:
        wo = wo_by_id.get(a.get("wo"))
        end = a.get("end_bh")
        if wo is None or end is None:
            continue
        end = float(end)
        weight = float(wo["weight"])
        due = float(wo["due_bh"])
        release = float(wo["release_bh"])
        priority = wo.get("priority")

        n += 1
        wwt += weight * max(0.0, end - due)
        ends.append(end)
        flows.append(end - release)

        breached = end > due + _BREACH_TOL
        if breached:
            breaches += 1
        if priority in prio_total:
            prio_total[priority] += 1
            if breached:
                prio_breach[priority] += 1

    makespan = float(np.max(ends)) if ends else 0.0
    mean_flow = float(np.mean(flows)) if flows else 0.0
    breach_share = (breaches / n) if n else 0.0

    per_priority_breach_share = {}
    for p in (1, 2, 3, 4):
        per_priority_breach_share[p] = (prio_breach[p] / prio_total[p]
                                        if prio_total[p] > 0 else None)

    return {
        "WWT": wwt,
        "makespan": makespan,
        "mean_flow": mean_flow,
        "breach_share": breach_share,
        "per_priority_breach_share": per_priority_breach_share,
        "wall_seconds": schedule.get("wall_seconds"),
        "decisions": schedule.get("decisions"),
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) not in (2, 3):
        sys.stderr.write("usage: python -m env.validator2 <instance.json> "
                         "<schedule.json> [overlay.json]\n")
        return 2
    with open(argv[0]) as f:
        instance = json.load(f)
    with open(argv[1]) as f:
        schedule = json.load(f)
    overlay = None
    if len(argv) == 3:
        with open(argv[2]) as f:
            overlay = json.load(f)
    result = validate(instance, schedule, overlay)
    print(json.dumps(result, indent=2))
    return 0 if result["feasible"] else 1


if __name__ == "__main__":
    sys.exit(main())
