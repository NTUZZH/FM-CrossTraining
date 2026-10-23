"""Crew-order dispatch engine (E-CREW robustness study).

A subclass of the released pair-selection engine (``env/engine.py``, left
untouched) in which a long work order may need several technicians at once.
It exists to answer one question: does the capture ratio of the flexibility
ladder survive when the labor hours of long orders are read as the work of a
small crew rather than of one technician? The crew sizes are explicitly
hypothetical scenarios, not observed practice.

Crew-size rule
--------------
    c_j = 1                                         if p_j <= LONG_BH (8 bh)
    c_j = min(c_max, ceil(p_j / LONG_BH), k(g_j))   otherwise

with k(g) the number of technicians whose PRIMARY trade is g in the roster
(overlay) of the episode. Structures add skills and never move headcount, so
c_j is the same under L0, CHAIN and FULL at a given (instance, m). c_max = 1
reproduces the released model.

Elapsed time (labor content preserved)
--------------------------------------
    crew of one:  p(j, u) = env.conventions.pair_p_bh(p_j, primary, eta)
    crew C, |C| >= 2:  ceil_0.01( p_j / sum_{u in C} r_u ),
                       r_u = 1 if prim(u) == g_j else eta
All members start together and stay busy for the whole elapsed time; every
member must hold g_j.

Dispatch protocol: priority list scheduling with reservation
------------------------------------------------------------
At each event instant, after all simultaneous events are drained (as in the
released driver), holds are cleared. Then, repeatedly, the rule selects by
its own unchanged score one order among the orders that have at least one
idle, unheld, eligible technician and are not blocked at this instant. If at
least c_j such technicians exist, the order starts with the first c_j of them
in the released technician tie-break order (primary first, least flexible,
lowest id; the selector's technician leads). Otherwise those technicians are
HELD for the order (idle, unavailable to any other order until the next event
instant) and the order is BLOCKED for this instant. When nothing selectable
remains the driver advances to the next event.

Implementation: during an instant, held technicians are removed from the idle
set and blocked orders from their trade queue, and both are restored when the
instant ends (queues back in arrival order). Every selector of
``methods/rules.py`` therefore runs unchanged. With every c_j = 1 nothing is
ever held or blocked and the driver executes the released code path, so the
schedule equals ``PairDispatchEnv``'s bitwise (tests/test_crew_engine.py).

Rule scores: the job dict a selector reads carries ``p_bh = p_j / c_j`` (the
crew's elapsed time at full speed; p_j itself when c_j = 1); the engine keeps
the logged labor hours under ``p_labor`` and computes elapsed time from them.
EDD and pFIFO never read processing time.

Schedule output: one assignment record PER CREW MEMBER,
``{"wo", "tech", "start_bh", "end_bh"}``, members of one order consecutive and
led by the selector's technician. With every c_j = 1 this is the released
schedule format record for record.
"""

from __future__ import annotations

import heapq
import itertools
import math

from env.conventions import ceil_grid
from env.engine import DECLINE, PairDispatchEnv, _FREE, _RELEASE

LONG_BH = 8.0          # one technician-day; the pre-declared short-order proxy


def primary_counts(technicians) -> dict:
    """k(g): technicians whose primary trade is g (overlay or Y1 dicts)."""
    k = {}
    for t in technicians:
        prim = (t.get("primary") or t["skills"][0]) if "skills" in t \
            else t["trade"]
        k[prim] = k.get(prim, 0) + 1
    return k


def crew_size(p_bh: float, k_g: int, c_max: int,
              long_bh: float = LONG_BH) -> int:
    """c_j of one order (module docstring)."""
    if float(p_bh) <= long_bh:
        return 1
    if k_g < 1:
        raise ValueError("trade has no primary technician")
    return max(1, min(int(c_max), int(math.ceil(float(p_bh) / long_bh)),
                      int(k_g)))


def crew_sizes(work_orders, technicians, c_max: int,
               long_bh: float = LONG_BH) -> dict:
    """{order id: c_j} for one instance under one roster."""
    k = primary_counts(technicians)
    return {w["id"]: crew_size(w["p_bh"], k.get(w["trade"], 0), c_max,
                               long_bh)
            for w in work_orders}


class CrewDispatchEnv(PairDispatchEnv):
    """Pair engine with crew orders and reservation (module docstring)."""

    def __init__(self, instance: dict, overlay: dict | None = None,
                 c_max: int = 1, long_bh: float = LONG_BH, **kw):
        if int(c_max) < 1:
            raise ValueError("c_max must be >= 1")
        super().__init__(instance, overlay, **kw)
        self.c_max = int(c_max)
        self.long_bh = float(long_bh)
        # self.instance is a fresh shallow copy when an overlay is applied,
        # and the caller's dict otherwise; copy so the caller's instance and
        # its work-order dicts are never mutated.
        self.instance = dict(self.instance)
        sizes = crew_sizes(self.instance["work_orders"],
                           self.instance["technicians"], self.c_max,
                           self.long_bh)
        jobs = []
        for w in self.instance["work_orders"]:
            c = sizes[w["id"]]
            j = dict(w)
            j["p_labor"] = w["p_bh"]
            j["crew"] = c
            if c > 1:
                j["p_bh"] = float(w["p_bh"]) / c     # scenario score time
            jobs.append(j)
        self.instance["work_orders"] = jobs
        self.crew_of = sizes
        self._reset_state()

    # ------------------------------------------------------------------ #
    def _reset_state(self):
        super()._reset_state()
        self._arr_seq = {}
        self.held_bh = 0.0             # technician-hours held idle
        self.n_blocks = 0              # order blocks (one per order-instant)
        self.n_hold_instants = 0       # event instants that ended with a hold

    def crew_elapsed(self, job, members):
        """Elapsed time of ``job`` on crew ``members`` (module docstring)."""
        if len(members) == 1:
            return self.pair_p(job, members[0])
        rate = 0.0
        for u in members:
            rate += 1.0 if self.prim_of[u] == job["trade"] else self.eta
        return ceil_grid(float(job["p_labor"]) / rate)

    # ------------------------------------------------------------------ #
    def _driver(self):
        """Released driver plus crew starts, holds and blocks."""
        seq = itertools.count()
        arr = itertools.count()
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
                    self._arr_seq[wo["id"]] = next(arr)
                    self.queue[wo["trade"]].append(wo)
                    self._on_arrival(wo, now)
            held = []                  # technicians held at this instant
            blocked = []               # orders blocked at this instant
            while any(self.queue[g] and self._n_idle_skill[g] > 0
                      for g in self._all_trades):
                self._cur_now = now
                sel = yield
                if sel is DECLINE:
                    assert any(k == _FREE for _, _, k, _ in events), (
                        "decline with no pending completion event: the "
                        "non-delay-relaxed loop would deadlock")
                    break
                tid, job = sel
                trade = job["trade"]
                if trade not in self.skills_of[tid]:
                    raise RuntimeError("dispatcher picked an ineligible pair")
                c = job["crew"]
                if c == 1:
                    # Released code path, line for line.
                    self.queue[trade].remove(job)
                    self._set_busy(tid)
                    start = float(now)
                    end = start + self.pair_p(job, tid)
                    self.tech_free_at[tid] = end
                    self.assignments.append(
                        {"wo": job["id"], "tech": tid,
                         "start_bh": start, "end_bh": end})
                    self._realized += job["weight"] * max(0.0,
                                                          end - job["due_bh"])
                    if self.instrument:
                        ins = self._instr
                        pbh = job["p_bh"]
                        rph = end - start
                        ins["n_assign"] += 1
                        ins["pbh_total"] += pbh
                        ins["rph_total"] += rph
                        if self.prim_of[tid] != trade:
                            ins["n_secondary"] += 1
                            ins["pbh_secondary"] += pbh
                            ins["rph_secondary"] += rph
                    for g in self.skills_of[tid]:
                        self._lb_dirty.add(g)
                    heapq.heappush(events, (end, next(seq), _FREE, tid))
                    continue
                elig = [u for u in self.idle if trade in self.skills_of[u]]
                if len(elig) >= c:
                    rest = sorted((u for u in elig if u != tid),
                                  key=lambda u: self.tb_key(u, trade))
                    members = [tid] + rest[:c - 1]
                    self.queue[trade].remove(job)
                    for u in members:
                        self._set_busy(u)
                    start = float(now)
                    end = start + self.crew_elapsed(job, members)
                    for u in members:
                        self.tech_free_at[u] = end
                        self.assignments.append(
                            {"wo": job["id"], "tech": u,
                             "start_bh": start, "end_bh": end})
                        for g in self.skills_of[u]:
                            self._lb_dirty.add(g)
                    self._realized += job["weight"] * max(0.0,
                                                          end - job["due_bh"])
                    for u in members:
                        heapq.heappush(events, (end, next(seq), _FREE, u))
                else:
                    # Reserve: hold every idle eligible technician for this
                    # order and block it until the next event instant.
                    for u in elig:
                        self._set_busy(u)
                        held.append(u)
                    self.queue[trade].remove(job)
                    blocked.append(job)
                    self.n_blocks += 1
            if blocked or held:
                assert any(k == _FREE for _, _, k, _ in events), (
                    "blocked crew order with no pending completion event: "
                    "the reservation loop would deadlock")
                nxt = events[0][0]
                self.held_bh += len(held) * (nxt - now)
                self.n_hold_instants += 1
                for u in held:
                    self._set_idle(u)
                touched = set()
                for job in blocked:
                    self.queue[job["trade"]].append(job)
                    touched.add(job["trade"])
                for g in touched:
                    self.queue[g].sort(key=lambda j: self._arr_seq[j["id"]])
            if self.check_nondelay and not blocked:
                self._assert_nondelay()

    def _build_schedule(self, method, wall, seed=0):
        out = super()._build_schedule(method, wall, seed)
        out["decisions"] = len({a["wo"] for a in self.assignments})
        out["c_max"] = self.c_max
        out["held_tech_bh"] = self.held_bh
        out["n_blocks"] = self.n_blocks
        out["n_hold_instants"] = self.n_hold_instants
        return out
