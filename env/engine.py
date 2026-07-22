"""Pair-selection dispatch engine v2.

Replaces v1's "technician frees -> pick an order from its trade queue" with
PAIR SELECTION: maintain the idle set and per-trade queues; the feasible-pair
set is P(t) = {(u, j) : u idle, g_j in S_u}. Whenever P(t) is non-empty
(checked after every arrival and completion event), the dispatcher selects
one pair, starts it, and re-checks. This preserves the non-delay property
(no eligible technician idles while eligible work waits) and reduces exactly
to v1's event protocol when skills are singletons.

Event machinery mirrors fmwos.pdrs.dispatch / fmwos.env.DispatchEnv
bit-for-bit: one time-ordered heap of (time, seq, kind, payload); FREE events
for all technicians pushed first at bh 0, then RELEASE events; at each
distinct event time ALL simultaneous events are drained before any pick;
end = start + p(j, u) as a plain float add. At L0 with a Y1 rule the produced
schedule is identical to the v1 dispatcher (parity tests in tests/).

Determinism: given (instance, overlay, dispatcher, seed) the schedule is
reproducible bitwise. All tie-breaks compare ids as strings (Y1's heap
order). The seed only matters for the random rule and RL sampling.

Three ways to run an episode, sharing ONE generator driver:
  * run_selector(selector, ...)  -- rules; selector(env, rng) -> (tech_id, job)
  * run_replay_y1(rule, ...)     -- the strict engine-parity harness: follows
    v1's per-instant semantics (smallest trade in P first, v1 rule pick within
    that trade's queue, TB technician) so that at L0 EVERY Y1 rule, including
    seeded random, reproduces fmwos.pdrs.dispatch draw-for-draw.
  * reset()/step(action)         -- the RL path: pair candidates (capped),
    38-dim pair features, potential-shaped reward with the v2 admissible
    bound (env.lb2).
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
import time

import numpy as np

from env.conventions import pair_p_bh
from env import lb2

F_ORDER = 12
F_PAIR = 6
F_CTX = 20
F_TOTAL = F_ORDER + F_PAIR + F_CTX          # 38 (v1.1 fix)
K_ORDERS_PER_TRADE = 64                      # candidate cap, locked default
K_PAIRS = 256                                # pair-slot tensor cap (padded);
                                             # truncation is slack-first and
                                             # deterministic, and rare (impl
                                             # constant, Appendix D)

_WEEK_BH = 40.0
_EWMA_HALFLIFE = 40.0
_LN2 = math.log(2.0)
_EPS = 1e-6
_TWO_PI = 2.0 * math.pi

_FREE = 0
_RELEASE = 1

_STRUCTURES = ("dedicated", "chain", "generalist", "full")

# Sentinel a selector may return to DECLINE all remaining feasible pairs at the
# current event instant (deliberate idling, e.g. the patient-EDD rule). The
# event loop then advances to the next event with no dispatch. This path is
# OFF by default: no built-in rule or the RL step() path ever returns it, so
# every existing selector/policy run is byte-identical to before.
DECLINE = object()


class PairDispatchEnv:
    """Online single-instance pair-selection episode (module docstring)."""

    def __init__(self, instance: dict, overlay: dict | None = None,
                 k_orders: int = K_ORDERS_PER_TRADE, reward_mode: str = "shaped",
                 check_nondelay: bool = False, tb_mode: str = "default",
                 instrument: bool = False, k_pairs: int = K_PAIRS,
                 allow_wait: bool = False):
        if tb_mode not in ("default", "random", "most_flexible"):
            raise ValueError("tb_mode must be default|random|most_flexible")
        self.tb_mode = tb_mode
        # Wait action (E10). OFF by default and provably inert: without it the
        # observation width, the action indexing, and every episode are
        # byte-identical to the released engine. When on, the observation
        # gains one extra token (feature index F_TOTAL = is-wait flag, the
        # context block shared) whose action sends the driver the DECLINE
        # sentinel; the token is masked unless at least one technician is
        # busy, so a completion event is always pending and the driver's
        # deadlock assert can never fire.
        self.allow_wait = bool(allow_wait)
        self._wait_idx = None
        self._n_waits = 0
        # Behavioural instrumentation (E-E). OFF by default and provably inert:
        # when self.instrument is False no counter is read and no shadow pass
        # runs, so every existing selector/policy episode is byte-identical.
        self.instrument = bool(instrument)
        # Pair-slot tensor cap. Defaults to the locked module constant K_PAIRS
        # (256), so default construction reproduces released inference exactly;
        # the E-E large-cap rerun raises it (weight-compatible, see policy2).
        self.k_pairs = int(k_pairs)
        if overlay is not None:
            from overlays.build import apply_overlay
            instance = apply_overlay(instance, overlay)
        self.instance = instance
        self.meta_id = instance["meta"]["id"]
        self.eta = float(instance["meta"].get("eta", 1.0))
        self.structure = instance["meta"].get("structure", "dedicated")
        self.phi = instance["meta"].get("phi") or 0.0
        self.k_orders = int(k_orders)
        self.reward_mode = reward_mode
        self.check_nondelay = bool(check_nondelay)

        # Technicians: accept v2 overlay dicts {id, primary, skills} or Y1
        # dicts {id, trade} (implicit L0).
        self.techs = []
        for tech in instance["technicians"]:
            if "skills" in tech:
                prim = tech.get("primary") or tech["skills"][0]
                skills = list(tech["skills"])
            else:
                prim = tech["trade"]
                skills = [prim]
            self.techs.append({"id": tech["id"], "primary": prim,
                               "skills": skills})
        self.prim_of = {t["id"]: t["primary"] for t in self.techs}
        self.skills_of = {t["id"]: tuple(t["skills"]) for t in self.techs}
        self.flex_of = {t["id"]: len(t["skills"]) for t in self.techs}
        self.n_techs = len(self.techs)
        self.any_overlap = any(v > 1 for v in self.flex_of.values())
        self.budget_b = sum(v - 1 for v in self.flex_of.values())

        # Trade universe: overlay skills + instance trades + WO trades.
        trades = set()
        for t in self.techs:
            trades.update(t["skills"])
        trades.update(instance.get("trades", []))
        for wo in instance["work_orders"]:
            trades.add(wo["trade"])
        self._all_trades = sorted(trades)
        self.holders = {g: [t["id"] for t in self.techs if g in t["skills"]]
                        for g in self._all_trades}
        self.k_primary = {g: sum(1 for t in self.techs if t["primary"] == g)
                          for g in self._all_trades}

        self._reset_state()

    # ------------------------------------------------------------------ #
    # State                                                              #
    # ------------------------------------------------------------------ #
    def _reset_state(self):
        self.queue = {g: [] for g in self._all_trades}
        self.idle = {}                        # tech_id -> True (set semantics)
        self.tech_free_at = {t["id"]: 0.0 for t in self.techs}
        self._wait_idx = None
        self._n_waits = 0
        self.assignments = []
        self._realized = 0.0
        self._n_idle_skill = {g: 0 for g in self._all_trades}

        self._lb_cache = {}
        self._lb_t = {}
        self._lb_dirty = set(self._all_trades)
        self._lb_global = 0.0
        self._lb_global_t = None

        self._ewma_s = 0.0                    # corrective arrivals, global
        self._ewma_last = 0.0

        self._cur_now = 0.0
        self._pairs = []
        self._done = True
        self._gen = None
        self._t_reset = time.perf_counter()

        # Instrumentation counters (per episode; only updated/read when
        # self.instrument). Initialised unconditionally: they are plain
        # accumulators that never enter the dispatch decision, so their
        # presence cannot change any schedule.
        self._instr = {
            "n_assign": 0, "n_secondary": 0,
            "pbh_total": 0.0, "pbh_secondary": 0.0,
            "rph_total": 0.0, "rph_secondary": 0.0,
            "decisions": 0, "order_cap_binds": 0,
            "pair_cap_binds": 0, "edd_excluded": 0,
        }

    # ------------------------------------------------------------------ #
    # Feasible-pair bookkeeping                                          #
    # ------------------------------------------------------------------ #
    def _set_idle(self, tid):
        self.idle[tid] = True
        for g in self.skills_of[tid]:
            self._n_idle_skill[g] += 1

    def _set_busy(self, tid):
        del self.idle[tid]
        for g in self.skills_of[tid]:
            self._n_idle_skill[g] -= 1

    def p_trades(self):
        """Sorted trades with queued work AND an eligible idle technician."""
        return [g for g in self._all_trades
                if self.queue[g] and self._n_idle_skill[g] > 0]

    def idle_techs_for(self, trade):
        """Idle technician ids holding ``trade`` (unsorted)."""
        return [tid for tid in self.idle if trade in self.skills_of[tid]]

    def n_idle_elig(self, trade):
        return self._n_idle_skill[trade]

    def tb_key(self, tid, trade):
        """Technician tie-break key: primary-skill first,
        least-flexible first, then lowest id (string compare, Y1 order)."""
        return (0 if self.prim_of[tid] == trade else 1, self.flex_of[tid], tid)

    def tb_tech(self, trade, rng=None):
        """TB-preferred idle technician for an order of ``trade``.

        tb_mode (E5' ablation): 'default' = primary-first,
        least-flexible, lowest id; 'most_flexible' = primary-first,
        MOST-flexible, lowest id; 'random' = uniform over the eligible idle
        set drawn from the episode's seeded rng (set in run_selector), so
        the ablation is deterministic per seed."""
        cands = self.idle_techs_for(trade)
        r = rng if rng is not None else getattr(self, "_tb_rng", None)
        if self.tb_mode == "random" and r is not None:
            return sorted(cands)[r.randrange(len(cands))]
        if self.tb_mode == "most_flexible":
            return min(cands, key=lambda tid: (
                0 if self.prim_of[tid] == trade else 1,
                -self.flex_of[tid], tid))
        return min(cands, key=lambda tid: self.tb_key(tid, trade))

    def pair_p(self, job, tid):
        return pair_p_bh(job["p_bh"], self.prim_of[tid] == job["trade"],
                         self.eta)

    # ------------------------------------------------------------------ #
    # Core event-loop driver                                             #
    # ------------------------------------------------------------------ #
    def _driver(self):
        """Generator: yields at each pair decision; caller sends (tid, job)."""
        seq = itertools.count()
        events = []
        for tech in self.techs:
            heapq.heappush(events, (0.0, next(seq), _FREE, tech["id"]))
        for wo in self.instance["work_orders"]:
            heapq.heappush(events, (float(wo["release_bh"]), next(seq),
                                    _RELEASE, wo))

        while events:
            now = events[0][0]
            while events and events[0][0] == now:
                _, _, kind, payload = heapq.heappop(events)
                if kind == _FREE:
                    self._set_idle(payload)
                else:
                    wo = payload
                    self.queue[wo["trade"]].append(wo)
                    self._on_arrival(wo, now)
            # Pair loop: dispatch while any feasible pair exists.
            while any(self.queue[g] and self._n_idle_skill[g] > 0
                      for g in self._all_trades):
                self._cur_now = now
                sel = yield
                if sel is DECLINE:
                    # Deliberate idle (patient rule): serve nothing more at
                    # this instant; the next event re-opens the decision. A
                    # decline is only legal while the rule waits for a busy
                    # technician to free, so a FREE completion event MUST be
                    # pending -- assert it, otherwise the relaxed loop could
                    # stall with queued work and no future event.
                    assert any(k == _FREE for _, _, k, _ in events), (
                        "decline with no pending completion event: the "
                        "non-delay-relaxed loop would deadlock")
                    break
                tid, job = sel
                trade = job["trade"]
                if trade not in self.skills_of[tid]:
                    raise RuntimeError("dispatcher picked an ineligible pair")
                self.queue[trade].remove(job)
                self._set_busy(tid)
                start = float(now)
                end = start + self.pair_p(job, tid)
                self.tech_free_at[tid] = end
                self.assignments.append(
                    {"wo": job["id"], "tech": tid,
                     "start_bh": start, "end_bh": end})
                self._realized += job["weight"] * max(0.0, end - job["due_bh"])
                if self.instrument:
                    ins = self._instr
                    pbh = job["p_bh"]
                    rph = end - start
                    ins["n_assign"] += 1
                    ins["pbh_total"] += pbh
                    ins["rph_total"] += rph
                    if self.prim_of[tid] != trade:      # secondary-skill use
                        ins["n_secondary"] += 1
                        ins["pbh_secondary"] += pbh
                        ins["rph_secondary"] += rph
                for g in self.skills_of[tid]:
                    self._lb_dirty.add(g)
                heapq.heappush(events, (end, next(seq), _FREE, tid))
            if self.check_nondelay:
                self._assert_nondelay()

    def _assert_nondelay(self):
        for g in self._all_trades:
            if self.queue[g] and self._n_idle_skill[g] > 0:
                raise AssertionError("non-delay invariant violated: trade %s "
                                     "has queued work and an eligible idle "
                                     "technician" % g)

    def _on_arrival(self, wo, now):
        if not wo.get("is_pm"):
            last = self._ewma_last
            decay = 0.5 ** ((now - last) / _EWMA_HALFLIFE) if now > last else 1.0
            self._ewma_s = self._ewma_s * decay + 1.0
            self._ewma_last = now
        self._lb_dirty.add(wo["trade"])

    # ------------------------------------------------------------------ #
    # Fast path: run with a selector callable                            #
    # ------------------------------------------------------------------ #
    def run_selector(self, selector, method: str = "rule", seed: int = 0) -> dict:
        """Run one episode; ``selector(env, rng) -> (tech_id, job)``."""
        t0 = time.perf_counter()
        self._reset_state()
        rng = random.Random(seed)
        self._tb_rng = rng          # random tie-break shares the episode rng
        gen = self._driver()
        try:
            next(gen)
            while True:
                # selector returns (tid, job) or the DECLINE sentinel; the
                # driver handles both. For every existing selector this is a
                # tuple, so behaviour is byte-identical to `gen.send((tid,job))`.
                gen.send(selector(self, rng))
        except StopIteration:
            pass
        return self._build_schedule(method, time.perf_counter() - t0, seed)

    def run_replay_y1(self, rule: str, seed: int = 0,
                      method: str | None = None) -> dict:
        """Strict v1-semantics harness (engine-parity test, module docstring).

        At each decision: take the LEXICOGRAPHICALLY SMALLEST trade in P,
        apply the vendored Y1 rule to that trade's queue (v1 signature and
        RNG consumption), assign via TB. At L0 this reproduces
        fmwos.pdrs.dispatch draw-for-draw for all six Y1 rules.
        """
        from fmwos_y1 import pdrs as _y1pdrs
        pick = _y1pdrs.get_rule(rule)

        def selector(env, rng):
            g = env.p_trades()[0]
            job = pick(env.queue[g], env._cur_now, rng)
            return env.tb_tech(g), job

        return self.run_selector(selector, method or ("y1replay_" + rule), seed)

    # ------------------------------------------------------------------ #
    # RL path: reset / step over candidate pairs                         #
    # ------------------------------------------------------------------ #
    def reset(self):
        self._reset_state()
        self._gen = self._driver()
        self.phi_prev = 0.0
        try:
            next(self._gen)
            self._done = False
        except StopIteration:
            self._done = True
        self._t_reset = time.perf_counter()
        return self._make_obs() if not self._done else self._zeros_obs(0)

    def step(self, action):
        if self._done:
            raise RuntimeError("step() on a finished episode; reset() first")
        n_pairs = len(self._pairs)
        if self._wait_idx is not None and int(action) == self._wait_idx:
            self._n_waits += 1
            payload = DECLINE
        else:
            payload = self._pairs[int(action)]
        try:
            self._gen.send(payload)
            done = False
        except StopIteration:
            done = True
        self._done = done
        reward = self._reward(done)
        obs = self._zeros_obs(0) if done else self._make_obs()
        info = {"n_pairs": n_pairs, "realized": self._realized}
        return obs, reward, done, info

    def _reward(self, done):
        if self.reward_mode == "terminal":
            return (-self._realized / 100.0) if done else 0.0
        if self.reward_mode == "realized":
            phi_now = self._realized
        else:
            phi_now = self._phi(self._cur_now)
        reward = (self.phi_prev - phi_now) / 100.0
        self.phi_prev = phi_now
        return reward

    # ------------------------------------------------------------------ #
    # Potential                                                          #
    # ------------------------------------------------------------------ #
    def _phi(self, t):
        total = self._realized + self._lb(t)
        return total

    def _lb(self, t):
        """v2 admissible bound with per-trade caching (Y1 idiom)."""
        total_a = 0.0
        for g in self._all_trades:
            q = self.queue[g]
            if not q:
                continue
            if (g in self._lb_dirty) or (self._lb_t.get(g) != t):
                taus = [max(t, self.tech_free_at[u]) for u in self.holders[g]]
                jobs = [(j["p_bh"], j["due_bh"], j["weight"]) for j in q]
                self._lb_cache[g] = (lb2.lb_trade_v2(jobs, taus, min(taus),
                                                     self.eta)
                                     if taus else 0.0)
                self._lb_t[g] = t
            total_a += self._lb_cache[g]
        if not self.any_overlap:
            self._lb_dirty.clear()
            return total_a
        if self._lb_dirty or self._lb_global_t != t:
            all_jobs = [(j["p_bh"], j["due_bh"], j["weight"])
                        for q in self.queue.values() for j in q]
            all_taus = [max(t, f) for f in self.tech_free_at.values()]
            self._lb_global = lb2._lb_group_ii(all_jobs, all_taus, self.eta)
            self._lb_global_t = t
        self._lb_dirty.clear()
        return max(total_a, self._lb_global)

    def phi(self):
        return self._phi(self._cur_now)

    # ------------------------------------------------------------------ #
    # Candidate pairs + observation                                      #
    # ------------------------------------------------------------------ #
    def candidate_pairs(self):
        """Capped feasible pairs: per trade in the union of idle skills, the
        ``k_orders`` smallest-slack orders; idle technicians deduplicated by
        (skill-set signature, next-available time), TB-min id representative.
        Deterministic order: trades sorted; orders by (slack, id); signatures
        by TB key of their representative."""
        t = self._cur_now
        # Signature = (primary, skill set); every candidate is idle, so its
        # next-available time is the current instant for all of them.
        sig_rep = {}
        for tid in self.idle:
            sig = (self.prim_of[tid], self.skills_of[tid])
            cur = sig_rep.get(sig)
            if cur is None or tid < cur:
                sig_rep[sig] = tid
        reps = list(sig_rep.values())
        pairs = []
        for g in self.p_trades():
            q = self.queue[g]
            if len(q) > self.k_orders:
                scored = sorted(((j["due_bh"] - t - j["p_bh"], j["id"], j)
                                 for j in q), key=lambda x: (x[0], x[1]))
                orders = [s[2] for s in scored[: self.k_orders]]
            else:
                orders = sorted(q, key=lambda j:
                                (j["due_bh"] - t - j["p_bh"], j["id"]))
            elig = sorted((tid for tid in reps if g in self.skills_of[tid]),
                          key=lambda tid: self.tb_key(tid, g))
            for j in orders:
                for tid in elig:
                    pairs.append((tid, j))
        return pairs

    def _make_obs(self):
        t = self._cur_now
        pairs = self.candidate_pairs()
        n_raw = len(pairs)
        if n_raw > self.k_pairs:
            # Deterministic slack-first truncation: keep the pairs of the most
            # urgent orders (slack, order id), TB-preferred technicians first.
            def _rank(pr):
                tid, j = pr
                return (j["due_bh"] - t - j["p_bh"], j["id"],
                        self.tb_key(tid, j["trade"]))
            pairs = sorted(pairs, key=_rank)[:self.k_pairs]
        f_obs = F_TOTAL + (1 if self.allow_wait else 0)
        wait_ok = self.allow_wait and len(self.idle) < self.n_techs
        if wait_ok and len(pairs) >= self.k_pairs:
            # Reserve the last slot for the wait token; the dropped pair is
            # the weakest-ranked one under the slack-first truncation above.
            pairs = pairs[:self.k_pairs - 1]
        self._pairs = pairs
        n = len(pairs)
        if self.instrument:
            self._instrument_decision(n_raw)
        feats = np.zeros((self.k_pairs, f_obs), dtype=np.float32)
        mask = np.zeros((self.k_pairs,), dtype=bool)
        mask[:n] = True
        qtw = {g: sum(j["p_bh"] for j in self.queue[g])
               for g in self._all_trades}
        ctx = self._ctx_features(t, qtw)
        for i, (tid, job) in enumerate(pairs):
            self._fill_order_features(feats[i, :F_ORDER], job, t,
                                      qtw[job["trade"]])
            self._fill_pair_features(feats[i, F_ORDER:F_ORDER + F_PAIR],
                                     tid, job, qtw)
            feats[i, F_ORDER + F_PAIR:F_TOTAL] = ctx
        if wait_ok:
            self._wait_idx = n
            feats[n, F_ORDER + F_PAIR:F_TOTAL] = ctx
            feats[n, F_TOTAL] = 1.0
            mask[n] = True
            n += 1
        else:
            self._wait_idx = None
        return {"pairs": feats, "mask": mask, "ctx": ctx, "n": n}

    def _zeros_obs(self, n):
        f_obs = F_TOTAL + (1 if self.allow_wait else 0)
        return {"pairs": np.zeros((self.k_pairs, f_obs), dtype=np.float32),
                "mask": np.zeros((self.k_pairs,), dtype=bool),
                "ctx": np.zeros((F_CTX,), dtype=np.float32),
                "n": n}

    # ------------------------------------------------------------------ #
    # Instrumentation (E-E; read-only shadow, never affects dispatch)    #
    # ------------------------------------------------------------------ #
    def _shadow_edd_pair(self):
        """(tech_id, order_id) that plain EDD with the default technician
        tie-break would dispatch at the current instant. Pure read: mirrors
        methods.rules._sel_edd (argmin over the distinct orders of P by
        (due_bh, id)) plus the default TB key (primary-first, least-flexible,
        lowest id), independent of self.tb_mode. It mutates nothing, so it
        cannot change the actual schedule."""
        best = None
        best_key = None
        for g in self.p_trades():
            for j in self.queue[g]:
                k = (j["due_bh"], j["id"])
                if best_key is None or k < best_key:
                    best_key = k
                    best = j
        if best is None:
            return None
        g = best["trade"]
        cands = self.idle_techs_for(g)
        tid = min(cands, key=lambda u: self.tb_key(u, g))
        return (tid, best["id"])

    def _instrument_decision(self, n_raw):
        """Per-policy-decision counters: did the per-trade order cap bind, did
        the pair-slot cap bind, and is EDD's default choice in the candidate
        set. Called from _make_obs only when self.instrument."""
        ins = self._instr
        ins["decisions"] += 1
        if any(len(self.queue[g]) > self.k_orders for g in self.p_trades()):
            ins["order_cap_binds"] += 1
        if n_raw > self.k_pairs:
            ins["pair_cap_binds"] += 1
        edd = self._shadow_edd_pair()
        if edd is not None:
            etid, ejid = edd
            present = any(tid == etid and job["id"] == ejid
                          for tid, job in self._pairs)
            if not present:
                ins["edd_excluded"] += 1

    @staticmethod
    def _fill_order_features(out, job, t, qtw):
        """Y1's 12 order features, verbatim (fmwos.env._fill_job_features)."""
        p = job["p_bh"]
        d = job["due_bh"]
        w = job["weight"]
        r = job["release_bh"]
        prio = job["priority"]
        out[0] = math.log1p(p)
        sd = (d - t - p) / 8.0
        out[1] = -30.0 if sd < -30.0 else (30.0 if sd > 30.0 else sd)
        out[2] = 1.0 if (t + p > d) else 0.0
        out[3] = w / 8.0
        if 1 <= prio <= 4:
            out[3 + prio] = 1.0
        out[8] = 1.0 if job.get("is_pm") else 0.0
        wd = (t - r) / 8.0
        out[9] = 0.0 if wd < 0.0 else (30.0 if wd > 30.0 else wd)
        out[10] = p / (qtw + _EPS)
        out[11] = math.log1p(w / p) if p > 0 else 0.0

    def _fill_pair_features(self, out, tid, job, qtw):
        g = job["trade"]
        primary = (self.prim_of[tid] == g)
        out[0] = 1.0 if primary else 0.0
        out[1] = math.log1p(self.pair_p(job, tid))
        out[2] = self.flex_of[tid] / 8.0
        skills = self.skills_of[tid]
        out[3] = sum(1 for s in skills if self.queue[s]) / len(skills)
        pg = self.prim_of[tid]
        kp = self.k_primary.get(pg, 1) or 1
        out[4] = qtw.get(pg, 0.0) / (8.0 * kp)
        others = any(u != tid and g in self.skills_of[u] for u in self.idle)
        out[5] = 0.0 if others else 1.0

    def _ctx_features(self, t, qtw):
        ctx = np.zeros((F_CTX,), dtype=np.float32)
        reach = sorted({g for tid in self.idle for g in self.skills_of[tid]})
        busiest = sorted((g for g in reach if self.queue[g]),
                         key=lambda g: (-qtw[g], g))[:3]
        for i, g in enumerate(busiest):
            k = len(self.holders[g]) or 1
            ctx[2 * i] = len(self.queue[g]) / 32.0
            ctx[2 * i + 1] = qtw[g] / (8.0 * k)
        rq = [j for g in reach for j in self.queue[g]]
        if rq:
            ctx[6] = min(j["due_bh"] - t - j["p_bh"] for j in rq) / 8.0
            ctx[7] = sum(1 for j in rq if j["priority"] in (1, 2)) / len(rq)
        ctx[8] = len(self.idle) / (self.n_techs or 1)
        last = self._ewma_last
        decay = 0.5 ** ((t - last) / _EWMA_HALFLIFE) if t > last else 1.0
        ctx[9] = (_LN2 / _EWMA_HALFLIFE) * self._ewma_s * decay * 8.0
        ang = _TWO_PI * ((t % _WEEK_BH) / _WEEK_BH)
        ctx[10] = math.sin(ang)
        ctx[11] = math.cos(ang)
        ctx[12] = t / _WEEK_BH
        ctx[13 + _STRUCTURES.index(self.structure)] = 1.0
        ctx[17] = float(self.phi or 0.0)
        ctx[18] = self.eta
        ctx[19] = self.budget_b / (self.n_techs or 1)
        return ctx

    # ------------------------------------------------------------------ #
    # Schedule output                                                    #
    # ------------------------------------------------------------------ #
    def _build_schedule(self, method, wall, seed=0):
        out = {
            "instance_id": self.meta_id,
            "overlay_id": self.instance["meta"].get("overlay_id"),
            "method": method,
            "seed": seed,
            "wall_seconds": wall,
            "decisions": len(self.assignments),
            "assignments": list(self.assignments),
        }
        if self.instrument:
            out.update(("instr_" + k, v) for k, v in self._instr.items())
        if self.allow_wait:
            out["waits"] = self._n_waits
        return out

    def to_schedule(self, method: str, seed: int = 0) -> dict:
        return self._build_schedule(method,
                                    time.perf_counter() - self._t_reset, seed)
