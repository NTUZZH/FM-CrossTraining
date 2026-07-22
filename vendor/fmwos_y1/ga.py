# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/ga.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""Permutation-based Genetic Algorithm baseline for the FM work-order
scheduling benchmark.

Problem (notes/interfaces.md): identical technicians per trade, all available
from bh 0; exact trade-match eligibility; release dates ``release_bh``; no
preemption; travel = 0; minimise WWT = sum_j w_j * max(0, C_j - due_bh_j).

Locked defaults (docs/protocol.md): population 100, 60 s wall budget, OX (order)
crossover, swap mutation, PDR-seeded initial population.

Representation
--------------
A *genome* is a permutation of all work-order ids.  Internally we use integer
indices ``0..n-1`` that are bijective with the work-order ids (index j <-> the
j-th work order of ``instance["work_orders"]``); an index permutation is exactly
an id permutation and is faster to manipulate.

Decoder (serial schedule generation scheme, gap-aware)
------------------------------------------------------
Walk the permutation left to right.  For each work order, place it on the
*earliest-available* eligible technician.  Because technicians of a trade are
identical, "eligible" is every technician of the WO's trade.  Placement is
**gap-aware**: each technician keeps a time-sorted list of its already-placed
(busy) intervals, and a job is dropped at the earliest feasible point >= its
release_bh -- either an idle GAP between two earlier jobs that is wide enough, or
the tail after the last job.  Across the trade's technicians we take the globally
earliest such start (ties -> lowest technician index).  Since any start is
>= release_bh, the moment a technician offers start == release_bh we stop
scanning (it cannot be beaten), so the common "a free technician exists" case is
O(1); overall the decode is ~O(n * (techs_scanned + gaps_scanned)) with tiny
constants (well under 5 ms for 400 jobs on real instances).

This insertion decoder is strictly stronger than the non-delay PDR dispatcher
(pdrs.py): given the order in which a non-delay schedule starts its jobs, the
decoder can only match or improve every completion time, so it escapes the
"non-delay trap" that costs the PDRs on the tiny fixture (tests/fixtures).

Fitness
-------
WWT of the decoded schedule, computed inline with the exact validator definition
(sum of w_j * max(0, end_j - due_j)).  Lower is better.

Evolutionary loop
-----------------
* Init: the 5 PDR schedules (edd, wspt, atc, pfifo, mor) converted to
  permutations (jobs ordered by start_bh, tie-break wo id), padded to ``pop``
  with random permutations from the seeded RNG.
* Tournament selection (size 3, with replacement).
* OX order crossover -> one child per mating.
* Swap mutation: probability 0.2 per child, a single random transposition.
* Elitism: the best 2 genomes survive unchanged into the next generation.
* Generational replacement to size ``pop``.
* Stop at the wall-clock budget (checked between generations) OR after 200
  generations with no strict improvement of the incumbent.

Determinism
-----------
All genetic operators draw from ``random.Random(seed)`` and are fully
reproducible.  The *number of generations* depends on wall-clock speed, so for a
fixed budget the result may vary slightly across machines; the primary loop is
therefore generation-based and the budget is only checked between generations.
The generations actually completed are reported under the extra key
``"generations"``.

API
---
``solve_ga(instance, budget_s=60.0, seed=301, pop=100) -> schedule dict`` per
notes/interfaces.md, with ``method='ga'`` and the extra key ``generations``.
``decisions`` is the number of fitness evaluations (decode calls), for latency
stats consistent with the other baselines.
"""

from __future__ import annotations

import bisect
import time

from . import pdrs

_INF = float("inf")

# PDR rules used to seed the initial population (order fixed for reproducibility).
_SEED_RULES = ("edd", "wspt", "atc", "pfifo", "mor")

_ELITES = 2          # elitism: best-N carried unchanged into the next generation
_TOURNAMENT = 3      # tournament selection size
_MUT_PROB = 0.2      # per-child swap-mutation probability
_STALL_LIMIT = 200   # stop after this many generations with no improvement


# --------------------------------------------------------------------------- #
# Instance preparation (done once; the hot decode reads only flat lists).
# --------------------------------------------------------------------------- #
def _prepare(instance: dict) -> dict:
    """Flatten the instance into index-aligned lists for the decoder.

    Returns a dict with per-work-order arrays (``trade``, ``proc``, ``rel``,
    ``due``, ``wt``, ``wid``), the ``id_to_index`` map, and per-trade technician
    bookkeeping (``trade_tech_ids`` for output, ``trade_tech_count`` for the
    interval structure).  Only trades that actually carry work orders are given
    technician structures.
    """
    work_orders = instance["work_orders"]
    trade = [w["trade"] for w in work_orders]
    proc = [float(w["p_bh"]) for w in work_orders]
    rel = [float(w["release_bh"]) for w in work_orders]
    due = [float(w["due_bh"]) for w in work_orders]
    wt = [float(w["weight"]) for w in work_orders]
    wid = [w["id"] for w in work_orders]
    id_to_index = {w["id"]: j for j, w in enumerate(work_orders)}

    # Technician ids per trade, sorted so identical techs are deterministically
    # ordered (tie-break of gap-aware placement is the technician index here).
    trade_tech_ids: dict[str, list[str]] = {}
    for tech in instance["technicians"]:
        trade_tech_ids.setdefault(tech["trade"], []).append(tech["id"])
    for tr in trade_tech_ids:
        trade_tech_ids[tr] = sorted(trade_tech_ids[tr])

    # Only trades with at least one work order need an interval structure.
    trade_tech_count = {tr: len(trade_tech_ids.get(tr, ())) for tr in set(trade)}

    return {
        "n": len(work_orders),
        "trade": trade,
        "proc": proc,
        "rel": rel,
        "due": due,
        "wt": wt,
        "wid": wid,
        "id_to_index": id_to_index,
        "trade_tech_ids": trade_tech_ids,
        "trade_tech_count": trade_tech_count,
    }


def _earliest_start(intervals, r: float, p: float) -> float:
    """Earliest feasible start >= ``r`` for a job of length ``p`` on one
    technician whose busy ``intervals`` are a time-sorted, non-overlapping list
    of ``(start, end)`` tuples.  Returns the start of the first idle GAP wide
    enough, else the tail after the last busy interval.
    """
    candidate = r
    for bs, be in intervals:
        if be <= candidate:
            continue  # interval lies entirely before the candidate; skip
        # be > candidate: this interval could block the placement.
        if bs >= candidate + p:
            return candidate  # gap [candidate, bs) is wide enough
        candidate = be        # candidate is inside/too close; push past this job
    return candidate


def _decode(P: dict, genome, build: bool = False):
    """Decode a genome (index permutation) into a schedule.

    Returns the WWT (float).  If ``build`` is True, returns ``(wwt, assignments)``
    where ``assignments`` is the validator-ready list of dicts.
    """
    trade = P["trade"]
    proc = P["proc"]
    rel = P["rel"]
    due = P["due"]
    wt = P["wt"]

    # Fresh per-technician busy-interval lists for this decode.
    tech_iv = {tr: [[] for _ in range(cnt)] for tr, cnt in P["trade_tech_count"].items()}

    wwt = 0.0
    placed = [] if build else None  # (idx, trade, tech_index, start, end)

    for idx in genome:
        tr = trade[idx]
        r = rel[idx]
        p = proc[idx]
        techs = tech_iv[tr]

        best_start = _INF
        best_ti = 0
        for ti in range(len(techs)):
            st = _earliest_start(techs[ti], r, p)
            if st < best_start:
                best_start = st
                best_ti = ti
                if st <= r:  # no start can precede the release; this is optimal
                    break

        end = best_start + p
        bisect.insort(techs[best_ti], (best_start, end))

        d = due[idx]
        if end > d:
            wwt += wt[idx] * (end - d)
        if build:
            placed.append((idx, tr, best_ti, best_start, end))

    if not build:
        return wwt

    wid = P["wid"]
    tech_ids = P["trade_tech_ids"]
    assignments = [
        {
            "wo": wid[idx],
            "tech": tech_ids[tr][ti],
            "start_bh": start,
            "end_bh": end,
        }
        for (idx, tr, ti, start, end) in placed
    ]
    return wwt, assignments


# --------------------------------------------------------------------------- #
# Genetic operators.
# --------------------------------------------------------------------------- #
def _permutation_from_schedule(schedule: dict, id_to_index: dict) -> list:
    """Order the schedule's work orders by (start_bh, wo id) -> index permutation."""
    ordered = sorted(schedule["assignments"], key=lambda a: (a["start_bh"], a["wo"]))
    return [id_to_index[a["wo"]] for a in ordered]


def _tournament(fits, rng, size: int = _TOURNAMENT) -> int:
    """Return the index of the tournament winner (lowest fitness wins)."""
    best = rng.randrange(len(fits))
    for _ in range(size - 1):
        i = rng.randrange(len(fits))
        if fits[i] < fits[best]:
            best = i
    return best


def _ox(p1, p2, rng) -> list:
    """Order crossover (OX): copy a random contiguous slice of ``p1``, fill the
    rest with ``p2``'s genes in order (starting just after the slice, wrapping)."""
    n = len(p1)
    i = rng.randrange(n)
    j = rng.randrange(n)
    if i > j:
        i, j = j, i

    child = [None] * n
    seg = set()
    for k in range(i, j + 1):
        child[k] = p1[k]
        seg.add(p1[k])

    pos = (j + 1) % n
    for k in range(n):
        gene = p2[(j + 1 + k) % n]
        if gene not in seg:
            child[pos] = gene
            pos = (pos + 1) % n
    return child


def _swap_mutate(child, rng) -> None:
    """Single random transposition, in place."""
    n = len(child)
    a = rng.randrange(n)
    b = rng.randrange(n)
    child[a], child[b] = child[b], child[a]


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def solve_ga(
    instance: dict,
    budget_s: float = 60.0,
    seed: int = 301,
    pop: int = 100,
) -> dict:
    """Run the permutation GA and return a schedule dict per notes/interfaces.md.

    Parameters
    ----------
    instance : dict     an instance dict per the shared schema.
    budget_s : float    wall-clock budget in seconds (checked between generations).
    seed     : int      RNG seed for all genetic operators (and PDR seeding).
    pop      : int      population size.

    Returns
    -------
    dict  schedule dict with ``method='ga'``, ``seed``, ``wall_seconds``,
          ``decisions`` (number of fitness evaluations), ``assignments`` (the best
          genome decoded), and the extra key ``generations``.
    """
    import random

    t_start = time.perf_counter()
    P = _prepare(instance)
    n = P["n"]
    rng = random.Random(seed)

    # ---- initial population: PDR seeds + random fill ------------------------
    genomes: list[list] = []
    for rule in _SEED_RULES:
        if len(genomes) >= pop:
            break
        sched = pdrs.dispatch(instance, rule, seed=seed)
        genomes.append(_permutation_from_schedule(sched, P["id_to_index"]))

    base = list(range(n))
    while len(genomes) < pop:
        g = base[:]
        rng.shuffle(g)
        genomes.append(g)
    genomes = genomes[:pop]

    fits = [_decode(P, g) for g in genomes]
    evals = len(genomes)

    best_i = min(range(len(fits)), key=lambda i: fits[i])
    best_fit = fits[best_i]
    best_genome = genomes[best_i][:]

    generations = 0
    stall = 0

    # ---- evolution ----------------------------------------------------------
    # Budget is checked between generations (see module docstring): the primary
    # loop is generation-based so the run is reproducible up to how many
    # generations fit in the wall-clock window.
    while time.perf_counter() - t_start < budget_s and stall < _STALL_LIMIT:
        # Elites: the best _ELITES genomes survive unchanged.
        order = sorted(range(len(fits)), key=lambda i: fits[i])
        elite_ids = order[:_ELITES]
        new_genomes = [genomes[i][:] for i in elite_ids]
        new_fits = [fits[i] for i in elite_ids]

        while len(new_genomes) < pop:
            p1 = genomes[_tournament(fits, rng)]
            p2 = genomes[_tournament(fits, rng)]
            child = _ox(p1, p2, rng)
            if rng.random() < _MUT_PROB:
                _swap_mutate(child, rng)
            new_genomes.append(child)
            new_fits.append(_decode(P, child))
            evals += 1

        genomes = new_genomes
        fits = new_fits
        generations += 1

        gi = min(range(len(fits)), key=lambda i: fits[i])
        if fits[gi] < best_fit - 1e-9:
            best_fit = fits[gi]
            best_genome = genomes[gi][:]
            stall = 0
        else:
            stall += 1

    # ---- decode the incumbent for output ------------------------------------
    _, assignments = _decode(P, best_genome, build=True)

    return {
        "instance_id": instance["meta"]["id"],
        "method": "ga",
        "seed": seed,
        "wall_seconds": time.perf_counter() - t_start,
        "decisions": evals,
        "assignments": assignments,
        "generations": generations,
    }
