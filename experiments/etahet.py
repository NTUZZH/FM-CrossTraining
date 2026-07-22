"""E-D Heterogeneous per-trade-pair efficiency (eta) injection.

The secondary-skill efficiency penalty in the released benchmark is a single
global scalar eta. Real cross-trade efficiency is heterogeneous. The paper
(Section model, "Exclusions") states that an
interface hook exists for technician-identity productivities beyond scalar
eta. This module IS that hook, exercised without touching any released source
file.

The hook, concretely
---------------------
The engine computes every pair-realised duration in exactly one method,
``PairDispatchEnv.pair_p(job, tid)``, and the independent validator mirrors it
in ``env.validator2._pair_p_local``. There is no ``eta_matrix`` data field the
released engine reads; the "hook" is this single choke point. Heterogeneous
eta is injected by (a) subclassing the engine to override ``pair_p`` with a
per-(primary, secondary) lookup, and (b) supplying a parallel INDEPENDENT
validator that keys the same convention on the ordered trade pair. Both are
new files; env/engine.py, env/conventions.py, env/validator2.py,
overlays/build.py and experiments/run_dynamic.py are untouched.

Convention (unchanged per pair, only the eta source changes):
    p(j, u) = p_j                         if g_j == prim(u)   (primary: exact)
    p(j, u) = ceil_grid(p_j / eta(g,h))   if g_j = h in S_u \\ {g=prim(u)}
where eta(g, h) is drawn i.i.d. uniform[0.70, 0.95] per ordered trade pair
(g -> h, g != h), per campus, per draw seed. eta is NEVER applied to primary
work: the primary branch returns p_j bit-for-bit.
"""
from __future__ import annotations

import math
import random

from env.engine import PairDispatchEnv
from env.conventions import pair_p_bh

# Draw support (pre-declared, locked).
ETA_LO = 0.70
ETA_HI = 0.95
DRAW_SEEDS = [20260831, 20260832, 20260833]


# --------------------------------------------------------------------------- #
# Heterogeneous eta matrix                                                    #
# --------------------------------------------------------------------------- #
def draw_eta_matrix(trades, seed, campus, lo: float = ETA_LO,
                    hi: float = ETA_HI) -> dict:
    """Draw eta(g -> h) i.i.d. uniform[lo, hi] for every ordered trade pair.

    Deterministic given (seed, campus). Draw order is canonical:
    sorted trades, outer g, inner h, skipping g == h. Returns
    {(g, h): eta} with g = primary trade, h = secondary trade.
    """
    ts = sorted(trades)
    rng = random.Random(int(seed) * 1000 + int(campus))
    mat = {}
    for g in ts:
        for h in ts:
            if g == h:
                continue
            mat[(g, h)] = rng.uniform(lo, hi)
    return mat


def matrix_mean(mat: dict) -> float:
    return sum(mat.values()) / len(mat) if mat else float("nan")


def matrix_to_records(mat: dict):
    """Serialise {(g,h): eta} -> sorted [[g, h, eta], ...] for JSON."""
    return [[g, h, mat[(g, h)]] for (g, h) in sorted(mat)]


def records_to_matrix(records) -> dict:
    return {(g, h): float(e) for g, h, e in records}


# --------------------------------------------------------------------------- #
# Engine subclass: override the single duration choke point                   #
# --------------------------------------------------------------------------- #
class EtaHetEnv(PairDispatchEnv):
    """PairDispatchEnv with per-(primary, secondary) eta.

    Overrides only ``pair_p``; every other engine behaviour (event stream,
    tie-breaks, non-delay property, seeding) is inherited byte-for-byte. Used
    for the rules / Random envelope only, which drive the schedule through
    ``run_selector`` -> ``_driver`` -> ``pair_p``; the RL lower-bound path
    (which reads the scalar self.eta) is not exercised here.
    """

    def __init__(self, instance, overlay, eta_matrix, **kw):
        super().__init__(instance, overlay, **kw)
        self._eta_matrix = dict(eta_matrix)

    def pair_p(self, job, tid):
        prim = self.prim_of[tid]
        trade = job["trade"]
        if prim == trade:
            # Primary work: p_j exactly, eta never applied.
            return float(job["p_bh"])
        eta = self._eta_matrix[(prim, trade)]
        return pair_p_bh(job["p_bh"], False, eta)


# --------------------------------------------------------------------------- #
# Independent het-aware validator (re-derives the convention locally)         #
# --------------------------------------------------------------------------- #
_REL_TOL = 1e-9
_DUR_TOL = 1e-6
_OVL_TOL = 1e-9


def _ceil_grid_local(x_bh: float) -> float:
    return math.ceil(x_bh * 100.0 - 1e-6) / 100.0


def _pair_p_het_local(p_bh: float, primary: bool, eta: float) -> float:
    if primary or eta >= 1.0:
        return float(p_bh)
    return _ceil_grid_local(float(p_bh) / float(eta))


def validate_etahet(instance, schedule, overlay, eta_matrix):
    """Independent feasibility checker for a heterogeneous-eta schedule.

    Mirrors env.validator2.validate but keys the duration check (check d) on
    eta(prim(u), g_j) from ``eta_matrix`` instead of a scalar. Shares no code
    with EtaHetEnv: a plumbing bug cannot launder an infeasible schedule.
    Metrics use the same Y1 formulas as validator2 (imported for the metric
    block only, not the duration check).
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
            prim = prim_by_id.get(tid)
            g = wo.get("trade")
            primary = (prim == g)
            eta = 1.0 if primary else float(eta_matrix[(prim, g)])
            expect = _pair_p_het_local(wo["p_bh"], primary, eta)
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
