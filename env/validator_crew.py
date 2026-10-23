"""Independent validator for crew-order schedules (E-CREW).

Self-contained on purpose: it imports neither the crew engine
(``env/crew_engine.py``) nor the released engine, conventions, overlay builder
or any method module, and re-derives every rule from the crew-size specification (Supplemental
Text S8), including the 0.01 bh round-up. A bug in
the crew engine therefore cannot launder an infeasible schedule into a valid
one.

Input schedule: one assignment record per crew member,
``{"wo", "tech", "start_bh", "end_bh"}``.

Crew-size rule, re-derived:
    c_j = 1                                       if p_j <= long_bh (8 bh)
    c_j = min(c_max, ceil(p_j / long_bh), k(g_j)) otherwise,
    k(g) = technicians whose primary trade is g in the roster.

Elapsed-time rule, re-derived:
    |C| = 1 : p_j if the member is primary or eta >= 1, else
              ceil_0.01(p_j / eta)          (the released convention)
    |C| >= 2: ceil_0.01(p_j / sum r_u),  r_u = 1 if primary else eta

Checks
------
(a) every work order is scheduled exactly once: its records form one crew
    (no technician twice, one shared start and end), and no record names an
    unknown order
(g) crew size equals c_j
(b) every member exists and holds g_j
(s) all members share start and end
(d) end - start equals the elapsed-time rule within 1e-6
(c) start >= release_bh - 1e-9
(e) no technician overlaps itself
(f) schedule.instance_id matches instance.meta.id; overlay campus matches

Metrics, recomputed per ORDER (one completion time C_j per order, taken from
its records in order of first appearance, so that with every c_j = 1 the sums
run in the released validator's order and agree with it bitwise):
TWT = sum_j w_j max(0, C_j - d_j), makespan, mean flow, breach share and
per-priority breach shares (released formulas and tolerances), plus the
class split (p_j <= long_bh versus longer), crew-order counts and mean waits.
"""

from __future__ import annotations

import math

import numpy as np

_REL_TOL = 1e-9
_DUR_TOL = 1e-6
_OVL_TOL = 1e-9
_BREACH_TOL = 1e-9
LONG_BH = 8.0


def _ceil_grid(x_bh: float) -> float:
    return math.ceil(x_bh * 100.0 - 1e-6) / 100.0


def _single_p(p_bh: float, primary: bool, eta: float) -> float:
    if primary or eta >= 1.0:
        return float(p_bh)
    return _ceil_grid(float(p_bh) / float(eta))


def expected_crew_size(p_bh, k_g, c_max, long_bh=LONG_BH):
    if float(p_bh) <= long_bh:
        return 1
    if k_g < 1:
        return None                      # no primary technician: unstaffable
    return max(1, min(int(c_max), int(math.ceil(float(p_bh) / long_bh)),
                      int(k_g)))


def validate(instance, schedule, overlay=None, c_max=1, long_bh=LONG_BH):
    """Validate a crew schedule; returns {feasible, violations, metrics}."""
    viol = []
    work_orders = instance.get("work_orders", []) or []
    wo_by_id = {w["id"]: w for w in work_orders}

    techs = (overlay or instance).get("technicians", []) or []
    skills, prim = {}, {}
    for t in techs:
        if "skills" in t:
            p = t.get("primary") or t["skills"][0]
            skills[t["id"]] = set(t["skills"])
        else:
            p = t["trade"]
            skills[t["id"]] = {p}
        prim[t["id"]] = p
    k = {}
    for p in prim.values():
        k[p] = k.get(p, 0) + 1

    eta = 1.0
    if overlay is not None:
        eta = float(overlay.get("eta", 1.0))
        if int(overlay.get("campus", -1)) != int(
                instance.get("meta", {}).get("campus", -2)):
            viol.append("(f) overlay campus %r does not match instance "
                        "campus %r" % (overlay.get("campus"),
                                       instance.get("meta", {}).get("campus")))
    elif "eta" in instance.get("meta", {}):
        eta = float(instance["meta"]["eta"] or 1.0)

    if schedule.get("instance_id") != instance.get("meta", {}).get("id"):
        viol.append("(f) schedule.instance_id %r does not match "
                    "instance.meta.id %r"
                    % (schedule.get("instance_id"),
                       instance.get("meta", {}).get("id")))

    assignments = schedule.get("assignments", []) or []
    groups, order = {}, []
    for a in assignments:
        wid = a.get("wo")
        if wid not in groups:
            groups[wid] = []
            order.append(wid)
        groups[wid].append(a)

    for wid in sorted(w for w in wo_by_id if w not in groups):
        viol.append("(a) work order %r is never scheduled" % wid)
    for wid in order:
        if wid not in wo_by_id:
            viol.append("(a) record names work order %r which is not in the "
                        "instance" % wid)

    sizes = {}
    for wid in order:
        w = wo_by_id.get(wid)
        recs = groups[wid]
        if w is None:
            continue
        g = w["trade"]
        members = [r.get("tech") for r in recs]
        if len(set(members)) != len(members):
            viol.append("(a) work order %r lists a technician twice (%s): "
                        "scheduled more than once" % (wid, members))
        starts = {r.get("start_bh") for r in recs}
        ends = {r.get("end_bh") for r in recs}
        if len(starts) != 1 or len(ends) != 1:
            viol.append("(s) work order %r: crew members do not share start "
                        "and end (starts %s, ends %s)"
                        % (wid, sorted(starts), sorted(ends)))
        c_exp = expected_crew_size(w["p_bh"], k.get(g, 0), c_max, long_bh)
        sizes[wid] = len(members)
        if c_exp is None:
            viol.append("(g) work order %r: trade %r has no primary "
                        "technician" % (wid, g))
        elif len(members) != c_exp:
            viol.append("(g) work order %r has crew size %d, rule gives %d "
                        "(p_bh %s, k %s, c_max %s)"
                        % (wid, len(members), c_exp, w["p_bh"], k.get(g, 0),
                           c_max))
        ok_members = True
        for u in members:
            if u not in skills:
                viol.append("(b) work order %r uses technician %r which does "
                            "not exist" % (wid, u))
                ok_members = False
            elif g not in skills[u]:
                viol.append("(b) technician %r (skills %s) is not eligible "
                            "for work order %r (trade %r)"
                            % (u, sorted(skills[u]), wid, g))
                ok_members = False
        for r in recs:
            s = r.get("start_bh")
            if s is not None and s < w["release_bh"] - _REL_TOL:
                viol.append("(c) work order %r starts at %s before its "
                            "release %s" % (wid, s, w["release_bh"]))
                break
        if ok_members and len(starts) == 1 and len(ends) == 1:
            s, e = recs[0].get("start_bh"), recs[0].get("end_bh")
            if len(members) == 1:
                expect = _single_p(w["p_bh"], prim[members[0]] == g, eta)
            else:
                rate = math.fsum(1.0 if prim[u] == g else eta
                                 for u in members)
                expect = _ceil_grid(float(w["p_bh"]) / rate)
            if abs((e - s) - expect) > _DUR_TOL:
                viol.append("(d) work order %r: elapsed %s (end %s - start "
                            "%s) does not equal the rule's %s (p_bh %s, crew "
                            "%s, eta %s)" % (wid, e - s, e, s, expect,
                                             w["p_bh"], members, eta))

    by_tech = {}
    for a in assignments:
        by_tech.setdefault(a.get("tech"), []).append(a)
    for u, recs in by_tech.items():
        recs = sorted(recs, key=lambda r: float(r.get("start_bh") or 0.0))
        for prev, cur in zip(recs, recs[1:]):
            if float(cur["start_bh"]) < float(prev["end_bh"]) - _OVL_TOL:
                viol.append("(e) technician %r: work order %r starts at %s "
                            "before work order %r ends at %s (overlap)"
                            % (u, cur.get("wo"), cur.get("start_bh"),
                               prev.get("wo"), prev.get("end_bh")))

    metrics = _metrics(order, groups, wo_by_id, sizes, long_bh)
    return {"feasible": not viol, "violations": viol, "metrics": metrics}


def _metrics(order, groups, wo_by_id, sizes, long_bh):
    twt = twt_short = twt_long = 0.0
    ends, flows = [], []
    n = breaches = 0
    prio_total = {1: 0, 2: 0, 3: 0, 4: 0}
    prio_breach = {1: 0, 2: 0, 3: 0, 4: 0}
    n_long = n_crew = crew_members = 0
    wait_crew, wait_single = [], []
    for wid in order:
        w = wo_by_id.get(wid)
        rec = groups[wid][0]
        if w is None or rec.get("end_bh") is None:
            continue
        end = float(rec["end_bh"])
        start = float(rec["start_bh"])
        weight = float(w["weight"])
        due = float(w["due_bh"])
        release = float(w["release_bh"])
        n += 1
        t = weight * max(0.0, end - due)
        twt += t
        if float(w["p_bh"]) <= long_bh:
            twt_short += t
        else:
            twt_long += t
            n_long += 1
        ends.append(end)
        flows.append(end - release)
        breached = end > due + _BREACH_TOL
        if breached:
            breaches += 1
        pr = w.get("priority")
        if pr in prio_total:
            prio_total[pr] += 1
            if breached:
                prio_breach[pr] += 1
        c = sizes.get(wid, len(groups[wid]))
        if c >= 2:
            n_crew += 1
            crew_members += c
            wait_crew.append(start - release)
        else:
            wait_single.append(start - release)
    return {
        "WWT": twt,
        "twt_short": twt_short,
        "twt_long": twt_long,
        "makespan": float(np.max(ends)) if ends else 0.0,
        "mean_flow": float(np.mean(flows)) if flows else 0.0,
        "breach_share": (breaches / n) if n else 0.0,
        "per_priority_breach_share": {
            p: (prio_breach[p] / prio_total[p] if prio_total[p] else None)
            for p in (1, 2, 3, 4)},
        "n_orders": n,
        "n_long": n_long,
        "n_crew_orders": n_crew,
        "mean_crew_size": (crew_members / n_crew) if n_crew else None,
        "mean_wait_crew": (sum(wait_crew) / len(wait_crew)
                           if wait_crew else None),
        "mean_wait_single": (sum(wait_single) / len(wait_single)
                             if wait_single else None),
    }
