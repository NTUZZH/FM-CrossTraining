# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/validator.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""Independent feasibility validator for the FM work-order scheduling benchmark.

This module is the *independent scorer*. It is deliberately self-contained: it shares no
code with the environment/solvers/io modules in ``fmwos`` and depends only on
the Python standard library and numpy. It re-derives every rule from the
authoritative spec in ``notes/interfaces.md`` so a bug in the benchmark plumbing
cannot silently launder an infeasible schedule into a "valid" one.

Public API
----------
validate(instance, schedule) -> {"feasible": bool,
                                 "violations": [str, ...],
                                 "metrics": {...}}

CLI
---
python -m fmwos.validator <instance.json> <schedule.json>
    prints the result dict as JSON to stdout; exit 0 iff feasible else 1.
"""

import json
import sys

import numpy as np

# Numerical tolerances, per spec (notes/interfaces.md, validator section).
_REL_TOL = 1e-9      # release respected:  start_bh >= release_bh - 1e-9
_DUR_TOL = 1e-6      # duration exact:      |(end - start) - p_bh| <= 1e-6
_OVL_TOL = 1e-9      # no overlap:          next.start >= prev.end - 1e-9
_BREACH_TOL = 1e-9   # SLA breach:          end_bh > due_bh + 1e-9


def validate(instance, schedule):
    """Validate ``schedule`` against ``instance`` and compute benchmark metrics.

    All feasibility checks are evaluated (we never stop at the first failure);
    every failure appends a distinct human-readable string to ``violations``.
    Metrics are always computed on the assignments as given, even when the
    schedule is infeasible.
    """
    violations = []

    # ---- index the instance -------------------------------------------------
    work_orders = instance.get("work_orders", []) or []
    wo_by_id = {}
    for wo in work_orders:
        wo_by_id[wo["id"]] = wo

    technicians = instance.get("technicians", []) or []
    tech_by_id = {t["id"]: t for t in technicians}

    instance_id = instance.get("meta", {}).get("id")
    assignments = schedule.get("assignments", []) or []

    # ---- (f) schedule.instance_id == instance.meta.id -----------------------
    sched_instance_id = schedule.get("instance_id")
    if sched_instance_id != instance_id:
        violations.append(
            "(f) schedule.instance_id {!r} does not match instance.meta.id "
            "{!r}".format(sched_instance_id, instance_id)
        )

    # ---- (a) every work order assigned exactly once -------------------------
    assign_counts = {}
    for a in assignments:
        wid = a.get("wo")
        assign_counts[wid] = assign_counts.get(wid, 0) + 1

    for wid in sorted(w for w in wo_by_id if w not in assign_counts):
        violations.append(
            "(a) work order {!r} is never assigned (missing)".format(wid)
        )
    for wid in sorted(w for w, c in assign_counts.items() if c > 1):
        violations.append(
            "(a) work order {!r} is assigned {} times (duplicated)".format(
                wid, assign_counts[wid]
            )
        )
    for wid in sorted(w for w in assign_counts if w not in wo_by_id):
        violations.append(
            "(a) assignment references work order {!r} which is not in the "
            "instance".format(wid)
        )

    # ---- per-assignment checks: (b) eligibility, (c) release, (d) duration --
    for a in assignments:
        wid = a.get("wo")
        tid = a.get("tech")
        wo = wo_by_id.get(wid)
        start = a.get("start_bh")
        end = a.get("end_bh")

        # (b) technician exists and its trade matches the work order's trade.
        tech = tech_by_id.get(tid)
        if tech is None:
            violations.append(
                "(b) assignment for work order {!r} uses technician {!r} which "
                "does not exist".format(wid, tid)
            )
        elif wo is not None and tech.get("trade") != wo.get("trade"):
            violations.append(
                "(b) technician {!r} (trade {!r}) is not eligible for work "
                "order {!r} (trade {!r})".format(
                    tid, tech.get("trade"), wid, wo.get("trade")
                )
            )

        if wo is None:
            # Cannot evaluate release/duration without the work order's data;
            # the missing/unknown WO is already reported under (a)/(b).
            continue

        # (c) start_bh >= release_bh - 1e-9.
        if start is not None and start < wo["release_bh"] - _REL_TOL:
            violations.append(
                "(c) work order {!r} starts at {} before its release_bh "
                "{}".format(wid, start, wo["release_bh"])
            )

        # (d) end_bh - start_bh == p_bh within 1e-6.
        if start is not None and end is not None:
            duration = end - start
            if abs(duration - wo["p_bh"]) > _DUR_TOL:
                violations.append(
                    "(d) work order {!r} has duration {} (end {} - start {}) "
                    "which does not equal p_bh {}".format(
                        wid, duration, end, start, wo["p_bh"]
                    )
                )

    # ---- (e) no overlap per technician (jobs sorted by start) ---------------
    by_tech = {}
    for a in assignments:
        by_tech.setdefault(a.get("tech"), []).append(a)
    for tid, jobs in by_tech.items():
        ordered = sorted(jobs, key=lambda x: _num(x.get("start_bh")))
        for prev, cur in zip(ordered, ordered[1:]):
            prev_end = _num(prev.get("end_bh"))
            cur_start = _num(cur.get("start_bh"))
            if cur_start < prev_end - _OVL_TOL:
                violations.append(
                    "(e) technician {!r}: work order {!r} starts at {} before "
                    "work order {!r} ends at {} (overlap)".format(
                        tid, cur.get("wo"), cur.get("start_bh"),
                        prev.get("wo"), prev.get("end_bh"),
                    )
                )

    metrics = _compute_metrics(schedule, wo_by_id)
    return {
        "feasible": len(violations) == 0,
        "violations": violations,
        "metrics": metrics,
    }


def _num(x):
    """Coerce a possibly-missing numeric field to a float for ordering/compare."""
    return float(x) if x is not None else 0.0


def _compute_metrics(schedule, wo_by_id):
    """Compute benchmark metrics on the assignments as given.

    Metrics join each assignment back to its work order for weight/due/release/
    priority. Assignments whose work order is unknown (or whose end_bh is
    missing) are skipped for metric purposes; those problems surface as
    feasibility violations instead.
    """
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
        # Weighted tardiness contribution for this WO.
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
        if prio_total[p] > 0:
            per_priority_breach_share[p] = prio_breach[p] / prio_total[p]
        else:
            per_priority_breach_share[p] = None  # class absent from instance

    return {
        "WWT": wwt,
        "makespan": makespan,
        "mean_flow": mean_flow,
        "breach_share": breach_share,
        "per_priority_breach_share": per_priority_breach_share,
        # Pass-through fields from the schedule (for latency/timing stats).
        "wall_seconds": schedule.get("wall_seconds"),
        "decisions": schedule.get("decisions"),
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        sys.stderr.write(
            "usage: python -m fmwos.validator <instance.json> <schedule.json>\n"
        )
        return 2
    with open(argv[0]) as f:
        instance = json.load(f)
    with open(argv[1]) as f:
        schedule = json.load(f)
    result = validate(instance, schedule)
    print(json.dumps(result, indent=2))
    return 0 if result["feasible"] else 1


if __name__ == "__main__":
    sys.exit(main())
