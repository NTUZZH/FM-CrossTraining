# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/pdrs.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""Priority dispatching rules (PDRs) for the FM work-order scheduling benchmark.

All rules share one event-driven list-scheduling *dispatcher* (``dispatch``).
The dispatcher builds a **non-delay** schedule: a technician is never left idle
while a released, trade-matching work order is waiting.  This is the standard
list-scheduling protocol for R | elig, r_j | sum w_j T_j and is exactly what the
spec (notes/interfaces.md) asks for -- "when a technician is free and its trade
queue is non-empty, pick the job maximizing the rule's score".

Simulation model
----------------
Two kinds of events, held in a single time-ordered ``heapq``:

  * ``release`` -- a work order becomes available at ``release_bh``;
  * ``free``    -- a technician finishes a job (or is initially free at bh 0).

At each distinct event time we first drain *all* events at that time (so that
every job released at that instant is queued before any pick is made), then, for
each trade whose state changed, we greedily assign queued jobs to idle
technicians of that trade until one side runs out.  Ties between identical
technicians of a trade are broken by technician id (they are interchangeable in
v1, so this never affects the objective, only reproducibility).

The dispatcher never inserts idle time on purpose -- that is the defining
limitation of dispatching versus the CP-SAT solver, and it is *why* a solver can
beat every PDR (see tests/fixtures/tiny_instance.md).

Every choice is deterministic for a fixed ``seed`` (the seed only matters for the
'random' rule).  Complexity: O(E log E) for the event heap plus O(sum |queue|)
for the linear per-dispatch scans; for 400 jobs x dozens of technicians this is
well under a millisecond.

Output: a schedule dict per notes/interfaces.md (method = rule name,
``wall_seconds`` measured, ``decisions`` = number of pick decisions = number of
work orders, since every job is picked exactly once).
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
import time
from collections import defaultdict

# --------------------------------------------------------------------------- #
# Rules.  Each rule is a callable ``pick(queue, t, rng) -> job`` that returns
# the work order (a dict from instance["work_orders"]) to dispatch next from a
# single trade's ``queue`` at simulation time ``t``.  They are written as a
# ``min`` over a key so that a job-id final tiebreak makes every choice
# deterministic regardless of queue insertion order.
# --------------------------------------------------------------------------- #


def _pick_edd(queue, t, rng):
    """Earliest Due Date: smallest due_bh first (tie: job id)."""
    return min(queue, key=lambda j: (j["due_bh"], j["id"]))


def _pick_wspt(queue, t, rng):
    """Weighted Shortest Processing Time: largest weight/p_bh first."""
    return min(queue, key=lambda j: (-(j["weight"] / j["p_bh"]), j["id"]))


def _pick_atc(queue, t, rng, k=2.0):
    """Apparent Tardiness Cost (Vepsalainen & Morton), k = 2.

    score = (w / p) * exp( - max(0, due - t - p) / (k * pbar) )

    where ``pbar`` is the mean processing time of the jobs *currently* queued in
    this trade's queue (recomputed at every dispatch, so ATC is genuinely
    time- and queue-dependent).  The job of maximum score is dispatched.
    """
    pbar = sum(j["p_bh"] for j in queue) / len(queue)
    denom = k * pbar  # pbar > 0 because processing times are strictly positive

    def key(j):
        slack = max(0.0, j["due_bh"] - t - j["p_bh"])
        score = (j["weight"] / j["p_bh"]) * math.exp(-slack / denom)
        return (-score, j["id"])  # min of -score == max score

    return min(queue, key=key)


def _pick_pfifo(queue, t, rng):
    """Priority-FIFO: lowest priority class first (1 before 4); FIFO (earliest
    release_bh) within a class (tie: job id)."""
    return min(queue, key=lambda j: (j["priority"], j["release_bh"], j["id"]))


def _pick_mor(queue, t, rng):
    """MOR-backlog, single-trade interpretation.

    The classic 'Most Operations/Work Remaining' rule chooses *across* stages or
    resources -- but in this benchmark every technician serves exactly one trade,
    so there is no cross-trade choice to make (the spec calls this out).  We
    therefore implement the intent -- shrink the trade's backlog fastest -- as
    Longest Processing Time (LPT) *within* the technician's own queue: dispatch
    the largest-p_bh job first (tie: job id).
    """
    return min(queue, key=lambda j: (-j["p_bh"], j["id"]))


def _pick_random(queue, t, rng):
    """Uniformly random choice from the queue, drawn from the run's seeded RNG."""
    return queue[rng.randrange(len(queue))]


_RULES = {
    "edd": _pick_edd,
    "wspt": _pick_wspt,
    "atc": _pick_atc,
    "pfifo": _pick_pfifo,
    "mor": _pick_mor,
    "random": _pick_random,
}


def get_rule(name):
    """Return the ``pick(queue, t, rng)`` callable for ``name``.

    Raises ``ValueError`` for an unknown rule name.
    """
    try:
        return _RULES[name]
    except KeyError:
        raise ValueError(
            "unknown rule {!r}; valid rules: {}".format(name, sorted(_RULES))
        )


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

_KIND_FREE = 0     # a technician becomes available
_KIND_RELEASE = 1  # a work order is released


def dispatch(instance: dict, rule: str, seed: int = 0) -> dict:
    """Run the event-driven list-scheduling dispatcher under ``rule``.

    Parameters
    ----------
    instance : dict   an instance dict per notes/interfaces.md
    rule     : str    one of edd|wspt|atc|pfifo|mor|random
    seed     : int    RNG seed (only affects the 'random' rule)

    Returns
    -------
    dict  a schedule dict per notes/interfaces.md.
    """
    t_start = time.perf_counter()
    pick = get_rule(rule)
    rng = random.Random(seed)

    technicians = instance["technicians"]
    work_orders = instance["work_orders"]

    # Per-trade state.
    queue = defaultdict(list)  # trade -> list of released, unassigned WO dicts
    idle = defaultdict(list)   # trade -> heap of idle technician ids (strings)

    counter = itertools.count()  # unique tiebreak so heap never compares payloads
    events = []                  # heap of (time, seq, kind, payload...)

    # Every technician is available from bh 0 (shift structure is baked into the
    # bh axis, per interfaces.md).
    for tech in technicians:
        heapq.heappush(events, (0.0, next(counter), _KIND_FREE, tech["id"], tech["trade"]))
    # Release events.
    for wo in work_orders:
        heapq.heappush(events, (float(wo["release_bh"]), next(counter), _KIND_RELEASE, wo))

    assignments = []
    decisions = 0

    def try_dispatch(trade, now):
        nonlocal decisions
        q = queue[trade]
        free_techs = idle[trade]
        while free_techs and q:
            job = pick(q, now, rng)
            q.remove(job)                       # exact object; unique id
            tech_id = heapq.heappop(free_techs)  # smallest id -> deterministic
            start = float(now)
            end = start + float(job["p_bh"])     # travel = 0: end - start == p_bh
            assignments.append(
                {"wo": job["id"], "tech": tech_id, "start_bh": start, "end_bh": end}
            )
            decisions += 1
            heapq.heappush(events, (end, next(counter), _KIND_FREE, tech_id, trade))

    while events:
        now = events[0][0]
        touched = set()
        # Drain all events at this instant so every simultaneously-released job
        # is in the queue before any pick is made.
        while events and events[0][0] == now:
            _, _, kind, *payload = heapq.heappop(events)
            if kind == _KIND_FREE:
                tech_id, trade = payload
                heapq.heappush(idle[trade], tech_id)
                touched.add(trade)
            else:  # _KIND_RELEASE
                wo = payload[0]
                queue[wo["trade"]].append(wo)
                touched.add(wo["trade"])
        for trade in sorted(touched):
            try_dispatch(trade, now)

    return {
        "instance_id": instance["meta"]["id"],
        "method": rule,
        "seed": seed,
        "wall_seconds": time.perf_counter() - t_start,
        "decisions": decisions,
        "assignments": assignments,
    }
