# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/rolling.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""Rolling-horizon CP-SAT policy for dynamic dispatch episodes (E2 baseline).

This is the dynamic-track counterpart of the static ``fmwos.cpsat`` solver: it
runs ONE instance as an online episode (work orders hidden until ``release_bh``)
and, on every arrival event, re-solves a *static snapshot* of the currently
released-but-unstarted jobs, then executes the resulting incumbent until the next
arrival.  It is the strong "solver that may insert idle time" baseline the
dispatching rules (``fmwos.pdrs``) and the learned policy are measured against
(notes/interfaces.md, "Dynamic rolling track"; notes/p3_spec.md §1).

Protocol (spec-locked)
----------------------
* Replan trigger: new arrival(s).  Arrivals within ``BATCH_BH`` (0.1 bh) of the
  first arrival of a batch are folded into ONE replan (a bounded number of
  simultaneous / near-simultaneous releases -> one solve).
* Snapshot model: the released-and-unstarted jobs (the queue); technicians are
  available from their ``busy_until`` time (a tech running an in-progress job is
  unavailable until it finishes -- in-progress jobs are FIXED, never re-solved).
  The snapshot is expressed in a frame shifted so the replan instant is 0, so all
  queued jobs have release ``max(0, r_j - t)`` and due ``d_j - t``; technician
  availability ``max(0, busy_until - t)`` is passed to ``cpsat.solve`` via its
  ``tech_available`` hook (a dummy [0, a_u) interval per busy tech).
* Budget: ``budget_s`` (default 2.0 s) wall per replan, ``workers=2``,
  warm-started from the current incumbent mapped into the shifted frame (a hint,
  never a constraint).
* Execution between replans: each technician follows its incumbent's job ORDER;
  when a technician becomes free it starts its next incumbent job, and it may
  deliberately IDLE until that job's incumbent start time (the advantage over a
  non-delay rule).  Solves are synchronous (simulated time is frozen during the
  solve), so no job can arrive mid-solve; the only way a queued job could be
  missing from the incumbent is a replan whose CP-SAT returned NO solution
  within the budget -- those jobs are covered by a deterministic EDD-appended
  fallback plan (re-optimised at the next replan) so execution always completes.

Output: a schedule dict per notes/interfaces.md with ``method='rollcp2'``,
``decisions`` = number of replans, ``wall_seconds`` = total wall (solve time
included) and the extra key ``mean_replan_s`` (mean CP-SAT wall per replan).

Determinism (documented residual nondeterminism)
------------------------------------------------
CP-SAT's ``random_seed`` is fixed (0) inside ``cpsat.solve``; technician
tie-order is the sorted technician id; the event heap uses a unique sequence
counter.  Two residual nondeterminism sources remain, both inside CP-SAT:

* *time-budget truncation*: if a snapshot is not proved OPTIMAL within
  ``budget_s`` the incumbent is CP-SAT's best-so-far at the cutoff, which
  varies with machine load;
* *parallel search* (``workers=2``, spec-locked): CP-SAT's workers share
  bounds asynchronously, so even a solve proved OPTIMAL may return a
  *different optimal* schedule run-to-run.

Consequence: the reported WWT is stable whenever snapshots are solved to
optimality (observed on replay instances), but tie-equivalent plans can shift
technician-free times and therefore the arrival-batch boundaries -- so
``decisions`` (replan count), ``mean_replan_s`` and the exact assignment
schedule may vary slightly across runs.  Documented rather than eliminated
(single-worker determinism would deviate from the spec's workers=2 budget).
"""

from __future__ import annotations

import heapq
import time

from . import cpsat

BATCH_BH = 0.1          # arrivals within this window fold into one replan
REPLAN_EVERY_BH = 4.0   # periodic trigger: bound plan staleness even with no
                        # arrivals (big-bang releases produced single budget-
                        # starved solves that were never revisited; see
                        # decisions.md 2026-07-05 "rolling trigger v2")
_EPS = 1e-6

# Event kinds (ordered so the heap tuple never has to compare payloads: the
# unique sequence counter breaks every tie before the kind field is reached).
_FREE = 0    # a technician finished its job
_WAKE = 1    # a deliberate-idle period ended; re-evaluate the technician
_REL = 2     # a work order was released


class _RollingSim:
    """Event-driven simulator that re-solves a CP-SAT snapshot on each arrival."""

    def __init__(self, instance: dict, budget_s: float = 2.0):
        self.instance = instance
        self.budget_s = float(budget_s)

        self.jobs = {wo["id"]: wo for wo in instance["work_orders"]}
        self.technicians = list(instance["technicians"])
        self.sorted_techs = sorted(t["id"] for t in self.technicians)
        self.techs_of_trade = {}
        for t in self.technicians:
            self.techs_of_trade.setdefault(t["trade"], []).append(t["id"])
        for tr in self.techs_of_trade:
            self.techs_of_trade[tr].sort()

        # Per-technician runtime state.
        self.tech_free_at = {t["id"]: 0.0 for t in self.technicians}
        self.tech_busy = {t["id"]: None for t in self.technicians}   # job id or None
        self.tech_wake = {}                                          # tid -> pending wake bh

        # Per-job state machine: 'unreleased' -> 'queued' -> 'in_progress' -> 'done'.
        self.state = {jid: "unreleased" for jid in self.jobs}

        # Incumbent plan: tech id -> [(job_id, planned_start_abs), ...] for the
        # still-queued jobs assigned to that tech, sorted by planned start.
        self.incumbent = {t["id"]: [] for t in self.technicians}

        self.assignments = []
        self.n_replans = 0
        self.replan_walls = []
        self.last_replan_at = None   # bh of the most recent replan (None = never)

        self._seq = 0
        self._events = []
        for wo in instance["work_orders"]:
            self._push(float(wo["release_bh"]), _REL, wo["id"])

    # ------------------------------------------------------------------ #
    def _push(self, t, kind, payload):
        heapq.heappush(self._events, (float(t), self._seq, kind, payload))
        self._seq += 1

    # ------------------------------------------------------------------ #
    def _start_job(self, tid, jid, now):
        wo = self.jobs[jid]
        start = now if now >= wo["release_bh"] else float(wo["release_bh"])
        end = start + float(wo["p_bh"])
        self.assignments.append(
            {"wo": jid, "tech": tid, "start_bh": start, "end_bh": end}
        )
        self.tech_busy[tid] = jid
        self.tech_free_at[tid] = end
        self.state[jid] = "in_progress"
        self.tech_wake.pop(tid, None)
        self._push(end, _FREE, tid)

    # ------------------------------------------------------------------ #
    def _dispatch(self, now):
        """Start (or deliberately idle) every free technician per the incumbent."""
        for tid in self.sorted_techs:
            if self.tech_busy[tid] is not None:
                continue                          # busy on an in-progress job
            if self.tech_wake.get(tid) is not None:
                continue                          # already committed to an idle wait
            nxt = None
            for (jid, ps) in self.incumbent.get(tid, ()):
                if self.state[jid] == "queued":
                    nxt = (jid, ps)
                    break
            if nxt is None:
                continue                          # nothing assigned -> stay idle
            jid, ps = nxt
            rel = float(self.jobs[jid]["release_bh"])
            start = max(now, ps, rel)
            if start <= now + _EPS:
                self._start_job(tid, jid, now)
            else:
                # Deliberate idle: wait until the incumbent's planned start.
                self.tech_wake[tid] = start
                self._push(start, _WAKE, tid)

    # ------------------------------------------------------------------ #
    def _replan(self, now):
        queued = [jid for jid, st in self.state.items() if st == "queued"]
        if not queued:
            return
        queued.sort()   # deterministic snapshot order

        # Snapshot instance in a frame shifted so `now` == 0.
        snap_wos = []
        for jid in queued:
            wo = self.jobs[jid]
            rel = max(0.0, float(wo["release_bh"]) - now)
            snap_wos.append({
                "id": jid,
                "trade": wo["trade"],
                "p_bh": float(wo["p_bh"]),
                "release_bh": rel,
                "due_bh": float(wo["due_bh"]) - now,
                "priority": wo.get("priority", 3),
                "weight": float(wo["weight"]),
            })
        snapshot = {
            "meta": {"id": "%s_snap_%d" % (self.instance["meta"]["id"], self.n_replans)},
            "trades": self.instance.get("trades", []),
            "technicians": self.technicians,
            "work_orders": snap_wos,
        }
        tech_avail = {tid: max(0.0, self.tech_free_at[tid] - now)
                      for tid in self.tech_free_at}

        # Warm start: map the current incumbent for still-queued jobs into the
        # shifted frame (a hint; jobs no longer queued are dropped).
        qset = set(queued)
        warm_assign = []
        for tid, plan in self.incumbent.items():
            for (jid, ps) in plan:
                if jid in qset:
                    warm_assign.append({"wo": jid, "tech": tid,
                                        "start_bh": max(0.0, ps - now)})
        warm = {"assignments": warm_assign} if warm_assign else None

        sol = cpsat.solve(snapshot, time_limit_s=self.budget_s, workers=2,
                          warm_start=warm, tech_available=tech_avail,
                          flow_tiebreak=True)
        self.last_replan_at = now
        self.n_replans += 1
        self.replan_walls.append(float(sol.get("wall_seconds", 0.0)))

        # Rebuild the incumbent from the snapshot solution (shift back to abs).
        new_inc = {t["id"]: [] for t in self.technicians}
        for a in sol.get("assignments", []):
            jid = a["wo"]
            tid = a["tech"]
            new_inc[tid].append((jid, now + float(a["start_bh"])))
        # Safety net (rare budget-truncation edge case): if CP-SAT returned no
        # solution within the budget, some queued jobs are uncovered -- and if no
        # further arrival triggers a replan they would be stranded.  Cover them
        # with a deterministic EDD list plan appended after the planned work;
        # the next replan (if any) re-optimises them, and the executor's own
        # clamps keep whatever plan we hand it feasible.
        covered = {jid for plan in new_inc.values() for (jid, _ps) in plan}
        missing = [jid for jid in queued if jid not in covered]
        if missing:
            plan_end = {}
            for t in self.technicians:
                tid = t["id"]
                end = max(now, self.tech_free_at[tid])
                for (jid, ps) in new_inc[tid]:
                    end = max(end, ps) + float(self.jobs[jid]["p_bh"])
                plan_end[tid] = end
            for jid in sorted(missing,
                              key=lambda j: (self.jobs[j]["due_bh"], j)):
                wo = self.jobs[jid]
                cands = self.techs_of_trade.get(wo["trade"])
                if not cands:
                    continue  # spec forbids trades without techs; defensive
                tid = min(cands, key=lambda x: (plan_end[x], x))
                start = max(plan_end[tid], float(wo["release_bh"]))
                new_inc[tid].append((jid, start))
                plan_end[tid] = start + float(wo["p_bh"])

        for tid in new_inc:
            new_inc[tid].sort(key=lambda x: (x[1], x[0]))
        self.incumbent = new_inc
        # The plan changed: void any pending idle commitments so free techs
        # re-evaluate against the new incumbent in the following _dispatch.
        self.tech_wake.clear()

    # ------------------------------------------------------------------ #
    def run(self):
        ev = self._events
        while ev:
            now = ev[0][0]
            frees, rels = [], []
            # Drain everything at this exact instant.
            while ev and ev[0][0] == now:
                _, _, kind, payload = heapq.heappop(ev)
                if kind == _FREE:
                    frees.append(payload)
                elif kind == _REL:
                    rels.append(payload)
                else:  # _WAKE: release the idle commitment so _dispatch acts.
                    w = self.tech_wake.get(payload)
                    if w is not None and w <= now + _EPS:
                        self.tech_wake.pop(payload, None)
                    # A wake whose commitment is later than now is STALE (its
                    # plan was replaced by a replan and a newer wake exists);
                    # leave the newer commitment for its own event.
            # Fold near-simultaneous releases (within BATCH_BH) into this replan,
            # but only extend past releases -- an intervening FREE/WAKE (which
            # the heap would surface first, being earlier in time) stops the
            # batch so global time ordering is preserved.
            if rels:
                while ev and ev[0][2] == _REL and ev[0][0] < now + BATCH_BH:
                    _, _, _, payload = heapq.heappop(ev)
                    rels.append(payload)

            for tid in frees:
                jid = self.tech_busy[tid]
                self.tech_busy[tid] = None
                if jid is not None:
                    self.state[jid] = "done"
            for jid in rels:
                if self.state[jid] == "unreleased":
                    self.state[jid] = "queued"

            stale = (
                self.last_replan_at is not None
                and now - self.last_replan_at >= REPLAN_EVERY_BH - _EPS
                and any(st == "queued" for st in self.state.values())
            )
            if rels or stale:
                self._replan(now)
            self._dispatch(now)

    # ------------------------------------------------------------------ #
    def to_schedule(self, wall):
        n = self.n_replans
        mean_replan = (sum(self.replan_walls) / n) if n else 0.0
        return {
            "instance_id": self.instance["meta"]["id"],
            "method": "rollcp2",
            "seed": 0,
            "wall_seconds": wall,
            "decisions": n,                 # number of replans
            "assignments": list(self.assignments),
            "mean_replan_s": mean_replan,
        }


def roll_cpsat(instance: dict, budget_s: float = 2.0) -> dict:
    """Run the rolling CP-SAT policy on ``instance`` (see module docstring).

    Returns a schedule dict (method ``'rollcp2'``) with ``decisions`` = number of
    replans, ``wall_seconds`` = total wall (solve time included) and the extra
    key ``mean_replan_s`` = mean CP-SAT wall per replan.
    """
    t0 = time.perf_counter()
    sim = _RollingSim(instance, budget_s=budget_s)
    sim.run()
    return sim.to_schedule(time.perf_counter() - t0)
