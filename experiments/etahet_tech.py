"""Per-technician secondary-skill efficiency injection.

The released benchmark applies one scalar efficiency ``eta`` to every
secondary-skill assignment. ``experiments/etahet.py`` relaxes that to one
value per ordered trade pair, so technicians of the same trade still share a
speed on a given secondary trade. This module relaxes it one step further:
each technician carries its own efficiency on each of its secondary trades,

    eta_{u,g},  drawn independently per technician u and secondary trade g.

The duration convention is unchanged; only the source of eta changes:

    p(j, u) = p_j                            if g_j == prim(u)
    p(j, u) = ceil_grid(p_j / eta_{u,g_j})   if g_j in S_u \\ {prim(u)}

eta is never applied to primary work, which returns p_j bit for bit.

Injection point. The engine computes every realised duration in one method,
``PairDispatchEnv.pair_p(job, tid)``. ``EtaTechEnv`` overrides that method
and nothing else, so the event stream, the tie-breaks, the non-delay
property and the seeding are inherited unchanged. ``validate_etatech`` is a
parallel feasibility checker that re-derives the same convention from the
per-technician table with its own arithmetic, so a plumbing bug in the
engine cannot launder an infeasible schedule. env/engine.py,
env/conventions.py, env/validator2.py, overlays/build.py,
experiments/run_dynamic.py and experiments/etahet.py are untouched.
"""
from __future__ import annotations

import math
import random

from env.engine import PairDispatchEnv
from env.conventions import pair_p_bh

# Draw support, locked before the run.
DISTRIBUTIONS = {
    "u60": (0.60, 1.00),          # mean 0.80, matches the main-grid penalty
    "u80": (0.80, 1.00),          # mean 0.90, milder penalty
}
DRAW_SEEDS = [20260901, 20260902, 20260903]


# --------------------------------------------------------------------------- #
# Per-technician eta table                                                    #
# --------------------------------------------------------------------------- #
def draw_eta_tech(technicians, trades, seed, campus, m, lo, hi) -> dict:
    """Draw eta(u, g) i.i.d. uniform[lo, hi] per technician and secondary trade.

    Deterministic given (seed, campus, m, lo, hi). The draw order is
    canonical: technicians in list order (T0, T1, ...), trades ascending,
    skipping the technician's own primary trade. The technician list depends
    on the crew multiplier but not on the structure, so the same table serves
    CHAIN, GEN and FULL at one (campus, m) and the three structures are
    compared under identical technician speeds.

    Returns {(technician_id, trade): eta}.
    """
    ts = sorted(trades)
    rng = random.Random((int(seed) * 1000 + int(campus)) * 1000
                        + int(round(float(m) * 100)))
    mat = {}
    for tech in technicians:
        prim = tech.get("primary") or tech["skills"][0]
        for g in ts:
            if g == prim:
                continue
            mat[(tech["id"], g)] = rng.uniform(lo, hi)
    return mat


def constant_eta_tech(technicians, trades, value: float = 0.8) -> dict:
    """Constant table, the regression control that reproduces scalar eta."""
    ts = sorted(trades)
    mat = {}
    for tech in technicians:
        prim = tech.get("primary") or tech["skills"][0]
        for g in ts:
            if g != prim:
                mat[(tech["id"], g)] = float(value)
    return mat


def table_mean(mat: dict) -> float:
    """Mean over every drawn entry; independent of the structure."""
    return sum(mat.values()) / len(mat) if mat else float("nan")


def active_mean(mat: dict, technicians) -> float:
    """Mean over the entries a structure actually uses.

    Only the pairs (u, g) with g in S_u \\ {prim(u)} can ever be exercised,
    so this is the efficiency the schedule really sees. CHAIN grants one
    secondary skill per cross-trained technician and FULL grants all of
    them, so the two structures have different active means from one table.
    """
    vals = []
    for tech in technicians:
        prim = tech.get("primary") or tech["skills"][0]
        for g in tech.get("skills", [prim]):
            if g != prim:
                vals.append(mat[(tech["id"], g)])
    return sum(vals) / len(vals) if vals else float("nan")


def table_to_records(mat: dict):
    """Serialise {(u, g): eta} -> sorted [[u, g, eta], ...] for JSON."""
    return [[u, g, mat[(u, g)]] for (u, g) in sorted(mat)]


def records_to_table(records) -> dict:
    return {(u, g): float(e) for u, g, e in records}


# --------------------------------------------------------------------------- #
# Engine subclass: override the single duration choke point                   #
# --------------------------------------------------------------------------- #
class EtaTechEnv(PairDispatchEnv):
    """PairDispatchEnv with a per-(technician, secondary trade) efficiency.

    Overrides only ``pair_p``. The rules and the Random floor drive the
    schedule through ``run_selector`` -> ``_driver`` -> ``pair_p``, so they
    see the heterogeneous durations everywhere. The policy path, which reads
    the scalar ``self.eta`` for its features and its lower bound, is not
    exercised here.
    """

    def __init__(self, instance, overlay, eta_tech, **kw):
        super().__init__(instance, overlay, **kw)
        self._eta_tech = dict(eta_tech)

    def pair_p(self, job, tid):
        trade = job["trade"]
        if self.prim_of[tid] == trade:
            # Primary work: p_j exactly, eta never applied.
            return float(job["p_bh"])
        return pair_p_bh(job["p_bh"], False, self._eta_tech[(tid, trade)])


# --------------------------------------------------------------------------- #
# Independent per-technician validator                                        #
# --------------------------------------------------------------------------- #
_REL_TOL = 1e-9
_DUR_TOL = 1e-6
_OVL_TOL = 1e-9


def _ceil_grid_local(x_bh: float) -> float:
    return math.ceil(x_bh * 100.0 - 1e-6) / 100.0


def _pair_p_tech_local(p_bh: float, primary: bool, eta: float) -> float:
    if primary or eta >= 1.0:
        return float(p_bh)
    return _ceil_grid_local(float(p_bh) / float(eta))


def validate_etatech(instance, schedule, overlay, eta_tech):
    """Independent feasibility checker for a per-technician-eta schedule.

    Mirrors env.validator2.validate, with the duration check (check d)
    re-derived from ``eta_tech[(technician_id, trade)]`` instead of a scalar.
    It shares no arithmetic with EtaTechEnv. Metrics come from validator2's
    own metric block, which never touches durations.
    """
    from env.validator2 import _compute_metrics

    violations = []
    work_orders = instance.get("work_orders", []) or []
    wo_by_id = {wo["id"]: wo for wo in work_orders}

    techs = (overlay or instance).get("technicians", []) or []
    skills_by_id, prim_by_id = {}, {}
    for tech in techs:
        tid = tech["id"]
        if "skills" in tech:
            prim = tech.get("primary") or tech["skills"][0]
            skills_by_id[tid] = set(tech["skills"])
        else:
            prim = tech["trade"]
            skills_by_id[tid] = {prim}
        prim_by_id[tid] = prim

    if overlay is not None:
        inst_campus = instance.get("meta", {}).get("campus")
        if int(overlay.get("campus", -1)) != int(inst_campus):
            violations.append("(f) overlay campus %r != instance campus %r"
                              % (overlay.get("campus"), inst_campus))

    instance_id = instance.get("meta", {}).get("id")
    if schedule.get("instance_id") != instance_id:
        violations.append("(f) schedule.instance_id %r != instance.meta.id %r"
                          % (schedule.get("instance_id"), instance_id))

    assignments = schedule.get("assignments", []) or []
    counts = {}
    for a in assignments:
        counts[a.get("wo")] = counts.get(a.get("wo"), 0) + 1
    for wid in sorted(w for w in wo_by_id if w not in counts):
        violations.append("(a) work order %r never assigned" % wid)
    for wid in sorted(w for w, c in counts.items() if c > 1):
        violations.append("(a) work order %r assigned %d times"
                          % (wid, counts[wid]))
    for wid in sorted(w for w in counts if w not in wo_by_id):
        violations.append("(a) assignment references unknown work order %r"
                          % wid)

    for a in assignments:
        wid, tid = a.get("wo"), a.get("tech")
        wo = wo_by_id.get(wid)
        start, end = a.get("start_bh"), a.get("end_bh")
        skills = skills_by_id.get(tid)
        if skills is None:
            violations.append("(b) work order %r uses unknown technician %r"
                              % (wid, tid))
        elif wo is not None and wo.get("trade") not in skills:
            violations.append("(b) technician %r not skill-eligible for %r"
                              % (tid, wid))
        if wo is None:
            continue
        if start is not None and start < wo["release_bh"] - _REL_TOL:
            violations.append("(c) work order %r starts %s before release %s"
                              % (wid, start, wo["release_bh"]))
        if (start is not None and end is not None and skills is not None
                and wo.get("trade") in skills):
            g = wo.get("trade")
            primary = (prim_by_id.get(tid) == g)
            eta = 1.0 if primary else float(eta_tech[(tid, g)])
            expect = _pair_p_tech_local(wo["p_bh"], primary, eta)
            dur = end - start
            if abs(dur - expect) > _DUR_TOL:
                violations.append(
                    "(d) work order %r on %r: duration %s != p(j,u) %s "
                    "(p_bh %s, primary %s, eta %s)"
                    % (wid, tid, dur, expect, wo["p_bh"], primary, eta))

    by_tech = {}
    for a in assignments:
        by_tech.setdefault(a.get("tech"), []).append(a)
    for tid, jobs in by_tech.items():
        ordered = sorted(jobs, key=lambda x: float(x.get("start_bh") or 0.0))
        for prev, cur in zip(ordered, ordered[1:]):
            if (float(cur.get("start_bh") or 0.0)
                    < float(prev.get("end_bh") or 0.0) - _OVL_TOL):
                violations.append("(e) technician %r overlap: %r before %r"
                                  % (tid, cur.get("wo"), prev.get("wo")))

    metrics = _compute_metrics(schedule, wo_by_id)
    return {"feasible": len(violations) == 0, "violations": violations,
            "metrics": metrics}
