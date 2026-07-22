"""Multi-skill dispatching rules v2.

Order selection: each rule scores the DISTINCT orders appearing in the
feasible-pair set P (rule score exactly as in Y1, using nominal p_j), picks
the argmax with Y1's deterministic tie-break (min over (key, job id)), then
the technician tie-break TB assigns the machine:

  TB: prefer primary-skill technicians; among them least-flexible first
  (smallest |S_u|); then lowest id (string compare, Y1 heap order).

The six Y1 rules (EDD, WSPT, ATC k=2, pFIFO, MOR, Random) are thereby
extended without changing their scores. Two flexibility-aware rules:

  LFJ-ATC : ATC score as primary key; ties within a relative tolerance of 1%
            (pre-declared constant, never tuned) broken by least-flexible-job
            first (fewest currently idle eligible technicians), then job id.
  ATC-eta : ATC computed with the pair-realised processing time
            p(j, u_best(j)), where u_best(j) is the TB-preferred idle
            technician for j; per-trade pbar uses the same realised times.
            Identical to ATC when eta = 1.

Random picks uniformly among the distinct orders of P, listed per sorted
trade in queue (release) order, via one rng.randrange draw per decision; at
L0 single-trade decision instants this consumes the seeded RNG exactly like
the v1 dispatcher (see the parity tests).

ATC's pbar is the mean nominal processing time of the trade's currently
queued orders (recomputed at every decision), exactly Y1's definition; every
order of a trade in P is queued in that trade, so the per-trade pbar over
P-orders equals Y1's queue pbar.
"""

from __future__ import annotations

import math

ATC_K = 2.0
LFJ_TIE_TOL = 0.01          # pre-declared, locked

RANKED_RULES = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
ALL_RULES = RANKED_RULES + ["random"]


def _tb(env, job):
    return env.tb_tech(job["trade"])


def _argmin_over_p(env, key_fn):
    """min over the distinct orders of P of (key_fn(trade, job), job id)."""
    best = None
    best_key = None
    for g in env.p_trades():
        for j in env.queue[g]:
            k = (key_fn(g, j), j["id"])
            if best_key is None or k < best_key:
                best_key = k
                best = j
    return best


def _sp(j):
    """Scoring processing time. A dispatcher scores on the processing time it
    can SEE at decision time. By default that is the true p_bh (released,
    deterministic behaviour preserved: no order carries p_score in any
    released run, so the parity tests are unaffected). The noisy-duration
    experiment (E12) attaches a per-order p_score = p_bh * exp(eps) so the
    rule scores on a noisy estimate while the engine still EXECUTES on the
    true p_bh (pair_p is untouched). EDD and pFIFO do not call this, so they
    are exactly duration-free and immune to the estimate."""
    return j.get("p_score", j["p_bh"])


def _sel_edd(env, rng):
    j = _argmin_over_p(env, lambda g, j: j["due_bh"])
    return _tb(env, j), j


def _sel_wspt(env, rng):
    j = _argmin_over_p(env, lambda g, j: -(j["weight"] / _sp(j)))
    return _tb(env, j), j


def _atc_scores(env, realised=False):
    """Per-order ATC scores (dict trade -> list of (score, job)).

    realised=True is the ATC-eta variant: p is the pair-realised
    p(j, u_best(j)) with u_best the TB-preferred idle technician; pbar is the
    per-trade mean of those realised times.
    """
    t = env._cur_now
    out = {}
    for g in env.p_trades():
        q = env.queue[g]
        if realised:
            u_best = env.tb_tech(g)     # TB key is order-independent per trade
            ps = [env.pair_p(j, u_best) for j in q]
        else:
            ps = [_sp(j) for j in q]
        pbar = sum(ps) / len(ps)
        denom = ATC_K * pbar
        scored = []
        for j, p in zip(q, ps):
            slack = max(0.0, j["due_bh"] - t - p)
            score = (j["weight"] / p) * math.exp(-slack / denom)
            scored.append((score, j))
        out[g] = scored
    return out


def _sel_atc(env, rng, realised=False):
    scores = _atc_scores(env, realised)
    best = None
    best_key = None
    for g, scored in scores.items():
        for score, j in scored:
            k = (-score, j["id"])
            if best_key is None or k < best_key:
                best_key = k
                best = j
    return _tb(env, best), best


def _sel_atc_eta(env, rng):
    return _sel_atc(env, rng, realised=True)


def _sel_lfj_atc(env, rng):
    scores = _atc_scores(env, realised=False)
    flat = [(score, g, j) for g, sc in scores.items() for score, j in sc]
    best_score = max(s for s, _g, _j in flat)
    tol = best_score * LFJ_TIE_TOL
    ties = [(g, j) for s, g, j in flat if s >= best_score - tol]
    j = min(ties, key=lambda gj: (env.n_idle_elig(gj[0]), gj[1]["id"]))[1]
    return _tb(env, j), j


def _sel_pfifo(env, rng):
    j = _argmin_over_p(env, lambda g, j: (j["priority"], j["release_bh"]))
    return _tb(env, j), j


def _sel_mor(env, rng):
    j = _argmin_over_p(env, lambda g, j: -_sp(j))
    return _tb(env, j), j


def _sel_random(env, rng):
    cands = [j for g in env.p_trades() for j in env.queue[g]]
    j = cands[rng.randrange(len(cands))]
    return _tb(env, j), j


_WAIT_EPS = 1e-9            # busy-technician guard (free_at strictly > now)


class PatientEDD:
    """Patient (idling-capable) EDD.

    Identical to plain EDD with the default primary-preferring technician
    tie-break, EXCEPT it may DECLINE to start an order on a NON-primary
    technician when a primary-skill technician for that order's trade is
    currently busy and frees soon enough that waiting beats the secondary-skill
    penalty. For the EDD-ordered orders of P, it serves the most urgent order
    whose chosen technician passes the test; an order fails (is declined) iff

        env.eta < 1  and  chosen tech is non-primary  and
        (primary_next_free - t) < (ceil(p_j/eta) - p_j).

    If every feasible order is declined it returns the DECLINE sentinel and the
    engine idles the crew until the next event (a decline requires a busy
    primary, so a completion event is guaranteed). At eta = 1 the break-even
    window is 0 and nothing is ever declined, so the schedule is identical to
    plain EDD; likewise at L0 (all technicians primary), the non-primary guard
    never fires.

    Stateful: create one instance per episode and read the counters afterwards
    (n_declines, deliberate_wait_bh, ran_primary_after_decline,
    ran_secondary_after_decline). Within-instant re-declines of the same order
    are counted once."""

    def __init__(self):
        self.n_declines = 0
        self.deliberate_wait_bh = 0.0
        self.declined_ids = set()
        self.ran_primary_after_decline = 0
        self.ran_secondary_after_decline = 0
        self._served_after_decline = set()
        self._instant = None
        self._declined_here = set()

    @staticmethod
    def _primary_next_free(env, g, t):
        best = None
        for u in env.holders[g]:
            if env.prim_of[u] == g and env.tech_free_at[u] > t + _WAIT_EPS:
                f = env.tech_free_at[u]
                if best is None or f < best:
                    best = f
        return best

    def __call__(self, env, rng):
        from env.engine import DECLINE
        from env.conventions import ceil_grid
        t = env._cur_now
        if t != self._instant:
            self._instant = t
            self._declined_here = set()

        # Distinct feasible orders in EDD order (due, id) -- same key/tie-break
        # as plain EDD's _argmin_over_p, extended to a full ordering.
        cands = sorted((j for g in env.p_trades() for j in env.queue[g]),
                       key=lambda j: (j["due_bh"], j["id"]))
        for j in cands:
            g = j["trade"]
            u = env.tb_tech(g)
            if env.eta < 1.0 and env.prim_of[u] != g:
                pnf = self._primary_next_free(env, g, t)
                if pnf is not None:
                    window = ceil_grid(j["p_bh"] / env.eta) - j["p_bh"]
                    if (pnf - t) < window:
                        jid = j["id"]
                        if jid not in self._declined_here:
                            self._declined_here.add(jid)
                            self.n_declines += 1
                            self.deliberate_wait_bh += (pnf - t)
                            self.declined_ids.add(jid)
                        continue
            jid = j["id"]
            if jid in self.declined_ids and jid not in self._served_after_decline:
                self._served_after_decline.add(jid)
                if env.prim_of[u] == g:
                    self.ran_primary_after_decline += 1
                else:
                    self.ran_secondary_after_decline += 1
            return u, j
        return DECLINE

    def counters(self):
        return {
            "n_declines": self.n_declines,
            "deliberate_wait_bh": self.deliberate_wait_bh,
            "ran_primary_after_decline": self.ran_primary_after_decline,
            "ran_secondary_after_decline": self.ran_secondary_after_decline,
        }


class _PatientRule:
    """Patient (idling-capable) variant of a ranked rule.

    Same decline mechanism as PatientEDD (which predates this base class and
    is kept verbatim for regression stability): serve the rule's most
    preferred order whose TB technician passes the break-even test; decline
    an order iff eta < 1, the TB technician is non-primary, and a busy
    primary technician frees within ceil(p/eta) - p. If every order is
    declined, return DECLINE and let the engine idle to the next event.
    At eta = 1 and at L0 the schedule is identical to the plain rule.

    Subclasses supply _pick(env, remaining): the rule's argmax over the
    remaining candidate orders, recomputed per call so score state (ATC
    queue means, idle-eligible counts) matches the plain rule exactly.
    """

    def __init__(self):
        self.n_declines = 0
        self.deliberate_wait_bh = 0.0
        self.declined_ids = set()
        self.ran_primary_after_decline = 0
        self.ran_secondary_after_decline = 0
        self._served_after_decline = set()
        self._instant = None
        self._declined_here = set()

    _primary_next_free = staticmethod(PatientEDD._primary_next_free)

    def _pick(self, env, remaining):
        raise NotImplementedError

    def __call__(self, env, rng):
        from env.engine import DECLINE
        from env.conventions import ceil_grid
        t = env._cur_now
        if t != self._instant:
            self._instant = t
            self._declined_here = set()

        remaining = [j for g in env.p_trades() for j in env.queue[g]]
        while remaining:
            j = self._pick(env, remaining)
            remaining.remove(j)
            g = j["trade"]
            u = env.tb_tech(g)
            if env.eta < 1.0 and env.prim_of[u] != g:
                pnf = self._primary_next_free(env, g, t)
                if pnf is not None:
                    window = ceil_grid(j["p_bh"] / env.eta) - j["p_bh"]
                    if (pnf - t) < window:
                        jid = j["id"]
                        if jid not in self._declined_here:
                            self._declined_here.add(jid)
                            self.n_declines += 1
                            self.deliberate_wait_bh += (pnf - t)
                            self.declined_ids.add(jid)
                        continue
            jid = j["id"]
            if (jid in self.declined_ids
                    and jid not in self._served_after_decline):
                self._served_after_decline.add(jid)
                if env.prim_of[u] == g:
                    self.ran_primary_after_decline += 1
                else:
                    self.ran_secondary_after_decline += 1
            return u, j
        return DECLINE

    counters = PatientEDD.counters


class PatientWSPT(_PatientRule):
    def _pick(self, env, remaining):
        return min(remaining,
                   key=lambda j: (-(j["weight"] / j["p_bh"]), j["id"]))


class _PatientATCBase(_PatientRule):
    REALISED = False

    def _scores(self, env):
        """id -> ATC score, recomputed per call (plain-rule state)."""
        return {j["id"]: s
                for _g, sc in _atc_scores(env, self.REALISED).items()
                for s, j in sc}


class PatientATC(_PatientATCBase):
    def _pick(self, env, remaining):
        sc = self._scores(env)
        return min(remaining, key=lambda j: (-sc[j["id"]], j["id"]))


class PatientATCeta(_PatientATCBase):
    REALISED = True

    def _pick(self, env, remaining):
        sc = self._scores(env)
        return min(remaining, key=lambda j: (-sc[j["id"]], j["id"]))


class PatientLFJATC(_PatientATCBase):
    """LFJ-ATC's tie band is relative to the best score of the remaining
    candidates, so it is re-evaluated after every decline."""

    def _pick(self, env, remaining):
        sc = self._scores(env)
        best = max(sc[j["id"]] for j in remaining)
        tol = best * LFJ_TIE_TOL
        ties = [j for j in remaining if sc[j["id"]] >= best - tol]
        return min(ties, key=lambda j: (env.n_idle_elig(j["trade"]), j["id"]))


PATIENT_RULES = {
    "edd_patient": PatientEDD,
    "wspt_patient": PatientWSPT,
    "atc_patient": PatientATC,
    "atc_eta_patient": PatientATCeta,
    "lfj_atc_patient": PatientLFJATC,
}


_SELECTORS = {
    "edd": _sel_edd,
    "wspt": _sel_wspt,
    "atc": _sel_atc,
    "pfifo": _sel_pfifo,
    "mor": _sel_mor,
    "random": _sel_random,
    "lfj_atc": _sel_lfj_atc,
    "atc_eta": _sel_atc_eta,
}


def get_selector(name: str):
    try:
        return _SELECTORS[name]
    except KeyError:
        raise ValueError("unknown rule %r; valid: %s"
                         % (name, sorted(_SELECTORS)))


def dispatch(instance: dict, overlay: dict | None, rule: str,
             seed: int = 0) -> dict:
    """Convenience: one full episode of ``rule`` through the pair engine."""
    from env.engine import PairDispatchEnv

    env = PairDispatchEnv(instance, overlay)
    return env.run_selector(get_selector(rule), method=rule, seed=seed)
