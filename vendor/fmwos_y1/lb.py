# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/lb.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""Admissible lower bound on remaining weighted tardiness (P3 reward shaping).

This module is a *pure* function used by ``fmwos.env`` to build the
potential-shaped reward (see notes/p3_spec.md §2 and notes/lemma_lb.md).  It
conditions ONLY on the currently-queued jobs and the current technician
availabilities (future arrivals are ignored -- that is what keeps the bound
admissible for the realized online episode, since arrivals can only add cost).

Public API
----------
``lb_remaining(queues, tech_busy_until, t) -> float``
    queues            : dict  trade -> list of (p, d, w) for queued jobs
    tech_busy_until   : dict  trade -> list of per-technician availability times
                        (the bh time each tech next becomes free; a currently
                        idle tech's value may be <= t and is clamped to t)
    t                 : float current bh time
Returns the summed per-trade lower bound (trades are independent -- separate
technician pools -- so the sum of admissible per-trade bounds is admissible).

Two per-trade bounds are taken and their max is used (both are valid lower
bounds, so the larger is too):

  (i)  per-job earliest-completion bound
         sum_j w_j * max(0, tau_min + p_j - d_j),  tau_min = min_i max(t, tau_i)
       C_j >= tau_min + p_j in any schedule, so each term lower-bounds w_j T_j
       and the sum lower-bounds sum_j w_j T_j.

  (ii) capacity-overflow bound (corrected, admissible form)
       Sort the queued jobs by due date.  For each distinct due d let
         D(d)   = sum_{j: d_j <= d} p_j                     (work due by d)
         cap(d) = sum_i max(0, d - max(t, tau_i))           (machine-time by d)
         O(d)   = max(0, D(d) - cap(d))                     (overflow work-hours)
       The overflow work necessarily finishes after d; the fluid "area" argument
       gives  sum_{j: d_j<=d} p_j * (C_j - d)^+  >=  O(d)^2 / (2 k).  Converting
       from this work-weighted area to the objective sum_j w_j (C_j - d_j)^+
       costs a factor rho_min = min_{j: d_j<=d} (w_j / p_j) (cheapest weight per
       unit work among the jobs due by d), so the admissible contribution is
         rho_min * O(d)^2 / (2 k).
       (NOTE: the P3 spec wrote this constant as ``w_min`` -- the cheapest queued
       *weight*.  That form bounds the fluid area itself, not the per-job
       objective, and OVER-estimates it whenever a job is large -- e.g. one job
       p=10, d=5, k=1 gives w_min*O^2/2k = 12.5 but the true tardiness is 5 --
       so it is NOT admissible.  ``rho_min = min w_j/p_j`` is the correct,
       provably-admissible constant; see notes/lemma_lb.md and decisions.md.)

The max of (i) and (ii) is the per-trade bound; trades are summed.
"""

from __future__ import annotations

import math

_EPS = 1e-12


def _lb_trade(queue, tau_list, t):
    """Admissible lower bound on the additional weighted tardiness of one trade.

    queue    : list of (p, d, w) for the trade's currently-queued jobs
    tau_list : list of per-technician availability times (bh)
    t        : current bh time
    """
    if not queue:
        return 0.0
    k = len(tau_list)
    if k == 0:
        # Spec guarantees every trade has >= 1 technician; degenerate guard.
        return 0.0

    # tau_i = max(t, free_at_i): an idle technician (free_at <= t) is available
    # from t; a busy one from its completion time.
    taus = [f if f > t else t for f in tau_list]
    tau_min = min(taus)

    # ---- (i) per-job earliest-completion bound ---------------------------- #
    bound_i = 0.0
    for (p, d, w) in queue:
        ec = tau_min + p          # earliest possible completion of job j
        if ec > d:
            bound_i += w * (ec - d)

    # ---- (ii) capacity-overflow (area) bound ------------------------------ #
    jobs = sorted(queue, key=lambda x: x[1])   # by due date d
    n = len(jobs)
    two_k = 2.0 * k
    d_work = 0.0                  # D(d): running total processing due by d
    rho_min = math.inf           # min_{d_j<=d} w_j / p_j
    bound_ii = 0.0
    i = 0
    while i < n:
        d = jobs[i][1]
        # Accumulate every job sharing this (distinct) due date before scoring,
        # so D(d) includes all jobs with d_j <= d.
        while i < n and jobs[i][1] == d:
            p, _dd, w = jobs[i]
            d_work += p
            r = w / p if p > _EPS else w / _EPS
            if r < rho_min:
                rho_min = r
            i += 1
        cap = 0.0
        for tau in taus:
            if d > tau:
                cap += d - tau
        overflow = d_work - cap
        if overflow > 0.0:
            term = rho_min * overflow * overflow / two_k
            if term > bound_ii:
                bound_ii = term

    return bound_i if bound_i > bound_ii else bound_ii


def lb_remaining(queues, tech_busy_until, t):
    """Summed admissible lower bound over all trades (see module docstring)."""
    total = 0.0
    for trade, q in queues.items():
        if not q:
            continue
        taus = tech_busy_until.get(trade, [])
        total += _lb_trade(q, taus, t)
    return total
