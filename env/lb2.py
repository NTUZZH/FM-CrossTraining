"""Admissible lower bound v2 on remaining weighted tardiness.

Skill overlap breaks Y1's per-trade decomposition; the v2 bound keeps Y1's
two per-group ingredients and re-derives their validity for overlapping
eligibility sets (proof in paper Appendix B):

(i)  Per-job earliest-completion bound. For queued job j with trade g,
     tau_min(g) = min over u with g in S_u of max(t, free_at_u). Any schedule
     has C_j >= tau_min(g_j) + p_j: the job starts no earlier than the first
     moment an eligible technician is available, and its realised duration is
     >= p_j (secondary assignments only inflate durations, eta <= 1). Hence
     LB_i(A) = sum_j w_j (tau_min(g_j) + p_j - d_j)^+ is admissible.

(ii) Group overflow bound for a trade subset T'. A(T', d) = queued jobs with
     g_j in T' and d_j <= d. Required machine time D >= sum nominal p_j.
     Available machine time to A is at most cap(T', d) = sum over technicians
     u with S_u intersecting T' of (d - tau_u)^+ (a deliberate over-count:
     such a technician may spend all its time on T' work, so counting it
     fully keeps the bound valid). Overflow O = (D - cap)^+ necessarily
     finishes after d; with k' = |{u : S_u cap T' != {}}| servers the
     remaining-work area is >= O^2 / (2 k'); converting area to objective
     needs a constant lower-bounding w_j / p~_j over every realised duration
     p~_j the job can take. p~_j is either p_j (primary) or
     ceil_grid(p_j / eta) (secondary), so the exactly-admissible constant is
       rho'_min(T', d) = min_{j in A} w_j / ceil_grid(p_j / eta)
     (= min w_j / p_j at eta = 1). An earlier draft constant
     w_j * eta / p_j ignores the 0.01 bh grid round-up and can overestimate
     by the grid slack (disclosed in Appendix B of the paper); the
     implemented constant is the corrected one.

Composition: job sets of trade-disjoint groups are disjoint, so their bounds
sum (capacity over-counting per group does not break validity). Implemented:
  LB(s) = max( (a) sum over singleton trades of max(LB_i, best-d LB_ii),
               (b) LB_ii over the global group T' = G  [flexible overlays only] )
At L0 (all skills singleton) the group bound (b) is skipped by construction,
so LB(s) reduces EXACTLY to Y1's per-trade scan (fmwos.lb), which the unit
tests verify numerically. Hook (c), the chain-segment DP, is reserved and
implemented only if training diagnostics show a loose-bound plateau.
"""

from __future__ import annotations

import math

from env.conventions import pair_p_bh

_EPS = 1e-12


def _rho(w, p, eta):
    """Exactly-admissible conversion constant: w / (worst realised duration).

    The worst (largest) realised duration of a job is its secondary-speed
    grid value ceil_grid(p / eta) (= p at eta = 1)."""
    worst = pair_p_bh(p, False, eta)
    return (w / worst) if worst > _EPS else (w / _EPS)


def _lb_group_ii(jobs, taus, eta):
    """Best-over-d overflow bound for one group.

    jobs : list of (p, d, w), the group's queued jobs (any order)
    taus : per-technician availability times max(t, free_at) of the group's
           technicians (over-counted set: every tech holding any group skill)
    eta  : the scalar efficiency penalty (enters rho'_min only)
    """
    if not jobs or not taus:
        return 0.0
    k = len(taus)
    jobs = sorted(jobs, key=lambda x: x[1])
    two_k = 2.0 * k
    d_work = 0.0
    rho_min = math.inf
    best = 0.0
    i, n = 0, len(jobs)
    while i < n:
        d = jobs[i][1]
        while i < n and jobs[i][1] == d:
            p, _d, w = jobs[i]
            d_work += p
            r = _rho(w, p, eta)
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
            if term > best:
                best = term
    return best


def lb_trade_v2(jobs, taus, tau_min, eta):
    """Per-trade term: max of the per-job bound and the trade overflow bound.

    jobs    : list of (p, d, w) queued in this trade
    taus    : availability times of ALL technicians holding this skill
    tau_min : min(taus) (passed in so the caller can cache it)
    """
    if not jobs:
        return 0.0
    bound_i = 0.0
    for (p, d, w) in jobs:
        ec = tau_min + p
        if ec > d:
            bound_i += w * (ec - d)
    bound_ii = _lb_group_ii(jobs, taus, eta)
    return bound_i if bound_i > bound_ii else bound_ii


def lb_remaining_v2(queues, tech_free_at, skills_of, t, eta,
                    any_overlap=None):
    """Summed admissible bound (see module docstring).

    queues       : {trade: [(p, d, w), ...]}
    tech_free_at : {tech_id: free_at_bh}
    skills_of    : {tech_id: iterable of trades}
    t            : current bh time
    eta          : scalar efficiency penalty in (0, 1]
    any_overlap  : optional precomputed flag (any |S_u| > 1); None -> derived
    """
    taus_by_trade: dict[str, list] = {}
    for tid, skills in skills_of.items():
        tau = tech_free_at[tid]
        if tau < t:
            tau = t
        for g in skills:
            taus_by_trade.setdefault(g, []).append(tau)

    total_a = 0.0
    for g, q in queues.items():
        if not q:
            continue
        taus = taus_by_trade.get(g, [])
        if not taus:
            continue
        total_a += lb_trade_v2(q, taus, min(taus), eta)

    if any_overlap is None:
        any_overlap = any(len(s) > 1 for s in skills_of.values())
    if not any_overlap:
        return total_a

    all_jobs = [j for q in queues.values() for j in q]
    all_taus = [max(t, f) for f in tech_free_at.values()]
    total_b = _lb_group_ii(all_jobs, all_taus, eta)
    return total_a if total_a > total_b else total_b
