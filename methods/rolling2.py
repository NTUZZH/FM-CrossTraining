"""Rolling-horizon CP-SAT v2 (Y1 fmwos.rolling extended).

As Y1: 2 s budget, 2 workers, warm start from the incumbent, replan on
arrivals (0.1 bh batch) plus the periodic 4 bh staleness trigger, flow-time
tiebreak, deliberate idling allowed between replans. v2 differences: the
snapshot solver is methods.cpsat2 (overlay eligibility, pair durations,
skill-signature symmetry break OFF via the tech_available path), the
executor consumes p(j, u) pair durations, and the no-solution fallback
assigns EDD-ordered jobs to the eligible technician with the earliest plan
end (tie: TB key, then id).
"""

from __future__ import annotations

import heapq
import time

from env.conventions import pair_p_bh
from methods import cpsat2

BATCH_BH = 0.1
REPLAN_EVERY_BH = 4.0
_EPS = 1e-6

_FREE = 0
_WAKE = 1
_REL = 2


class _RollingSim:
    def __init__(self, instance: dict, overlay: dict | None,
                 budget_s: float = 2.0):
        self.instance = instance
        self.overlay = overlay
        self.budget_s = float(budget_s)
        self.eta = float((overlay or {}).get("eta",
                         instance.get("meta", {}).get("eta", 1.0)) or 1.0)

        techs_src = (overlay or instance)["technicians"]
        self.techs = []
        for tech in techs_src:
            if "skills" in tech:
                prim = tech.get("primary") or tech["skills"][0]
                skills = set(tech["skills"])
            else:
                prim = tech["trade"]
                skills = {prim}
            self.techs.append({"id": tech["id"], "primary": prim,
                               "skills": skills, "flex": len(skills)})
        self.skills_of = {t["id"]: t["skills"] for t in self.techs}
        self.prim_of = {t["id"]: t["primary"] for t in self.techs}
        self.flex_of = {t["id"]: t["flex"] for t in self.techs}
        self.sorted_techs = sorted(t["id"] for t in self.techs)

        self.jobs = {wo["id"]: wo for wo in instance["work_orders"]}
        self.tech_free_at = {t["id"]: 0.0 for t in self.techs}
        self.tech_busy = {t["id"]: None for t in self.techs}
        self.tech_wake = {}
        self.state = {jid: "unreleased" for jid in self.jobs}
        self.incumbent = {t["id"]: [] for t in self.techs}

        self.assignments = []
        self.n_replans = 0
        self.replan_walls = []
        self.last_replan_at = None

        self._seq = 0
        self._events = []
        for wo in instance["work_orders"]:
            self._push(float(wo["release_bh"]), _REL, wo["id"])

    def _push(self, t, kind, payload):
        heapq.heappush(self._events, (float(t), self._seq, kind, payload))
        self._seq += 1

    def _pair_p(self, jid, tid):
        wo = self.jobs[jid]
        return pair_p_bh(wo["p_bh"], self.prim_of[tid] == wo["trade"],
                         self.eta)

    def _start_job(self, tid, jid, now):
        wo = self.jobs[jid]
        start = now if now >= wo["release_bh"] else float(wo["release_bh"])
        end = start + self._pair_p(jid, tid)
        self.assignments.append(
            {"wo": jid, "tech": tid, "start_bh": start, "end_bh": end})
        self.tech_busy[tid] = jid
        self.tech_free_at[tid] = end
        self.state[jid] = "in_progress"
        self.tech_wake.pop(tid, None)
        self._push(end, _FREE, tid)

    def _dispatch(self, now):
        for tid in self.sorted_techs:
            if self.tech_busy[tid] is not None:
                continue
            if self.tech_wake.get(tid) is not None:
                continue
            nxt = None
            for (jid, ps) in self.incumbent.get(tid, ()):
                if self.state[jid] == "queued":
                    nxt = (jid, ps)
                    break
            if nxt is None:
                continue
            jid, ps = nxt
            rel = float(self.jobs[jid]["release_bh"])
            start = max(now, ps, rel)
            if start <= now + _EPS:
                self._start_job(tid, jid, now)
            else:
                self.tech_wake[tid] = start
                self._push(start, _WAKE, tid)

    def _replan(self, now):
        queued = [jid for jid, st in self.state.items() if st == "queued"]
        if not queued:
            return
        queued.sort()

        snap_wos = []
        for jid in queued:
            wo = self.jobs[jid]
            snap_wos.append({
                "id": jid, "trade": wo["trade"], "p_bh": float(wo["p_bh"]),
                "release_bh": max(0.0, float(wo["release_bh"]) - now),
                "due_bh": float(wo["due_bh"]) - now,
                "priority": wo.get("priority", 3),
                "weight": float(wo["weight"]),
            })
        snapshot = {
            "meta": {"id": "%s_snap_%d" % (self.instance["meta"]["id"],
                                           self.n_replans),
                     "campus": self.instance["meta"].get("campus")},
            "trades": self.instance.get("trades", []),
            "technicians": (self.overlay or self.instance)["technicians"],
            "work_orders": snap_wos,
        }
        tech_avail = {tid: max(0.0, self.tech_free_at[tid] - now)
                      for tid in self.tech_free_at}

        qset = set(queued)
        warm_assign = []
        for tid, plan in self.incumbent.items():
            for (jid, ps) in plan:
                if jid in qset:
                    warm_assign.append({"wo": jid, "tech": tid,
                                        "start_bh": max(0.0, ps - now)})
        warm = {"assignments": warm_assign} if warm_assign else None

        sol = cpsat2.solve(snapshot, overlay=self.overlay,
                           time_limit_s=self.budget_s, workers=2,
                           warm_start=warm, tech_available=tech_avail,
                           flow_tiebreak=True)
        self.last_replan_at = now
        self.n_replans += 1
        self.replan_walls.append(float(sol.get("wall_seconds", 0.0)))

        new_inc = {t["id"]: [] for t in self.techs}
        for a in sol.get("assignments", []):
            new_inc[a["tech"]].append((a["wo"], now + float(a["start_bh"])))

        covered = {jid for plan in new_inc.values() for (jid, _ps) in plan}
        missing = [jid for jid in queued if jid not in covered]
        if missing:
            plan_end = {}
            for t in self.techs:
                tid = t["id"]
                end = max(now, self.tech_free_at[tid])
                for (jid, ps) in new_inc[tid]:
                    end = max(end, ps) + self._pair_p(jid, tid)
                plan_end[tid] = end
            for jid in sorted(missing,
                              key=lambda j: (self.jobs[j]["due_bh"], j)):
                wo = self.jobs[jid]
                cands = [t["id"] for t in self.techs
                         if wo["trade"] in t["skills"]]
                if not cands:
                    continue
                tid = min(cands, key=lambda x: (
                    plan_end[x],
                    0 if self.prim_of[x] == wo["trade"] else 1,
                    self.flex_of[x], x))
                start = max(plan_end[tid], float(wo["release_bh"]))
                new_inc[tid].append((jid, start))
                plan_end[tid] = start + self._pair_p(jid, tid)

        for tid in new_inc:
            new_inc[tid].sort(key=lambda x: (x[1], x[0]))
        self.incumbent = new_inc
        self.tech_wake.clear()

    def run(self):
        ev = self._events
        while ev:
            now = ev[0][0]
            frees, rels = [], []
            while ev and ev[0][0] == now:
                _, _, kind, payload = heapq.heappop(ev)
                if kind == _FREE:
                    frees.append(payload)
                elif kind == _REL:
                    rels.append(payload)
                else:
                    w = self.tech_wake.get(payload)
                    if w is not None and w <= now + _EPS:
                        self.tech_wake.pop(payload, None)
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

    def to_schedule(self, wall):
        n = self.n_replans
        return {
            "instance_id": self.instance["meta"]["id"],
            "overlay_id": (self.overlay or {}).get("overlay_id"),
            "method": "rollcp2",
            "seed": 0,
            "wall_seconds": wall,
            "decisions": n,
            "assignments": list(self.assignments),
            "mean_replan_s": (sum(self.replan_walls) / n) if n else 0.0,
        }


def roll_cpsat(instance: dict, overlay: dict | None = None,
               budget_s: float = 2.0) -> dict:
    t0 = time.perf_counter()
    sim = _RollingSim(instance, overlay, budget_s=budget_s)
    sim.run()
    return sim.to_schedule(time.perf_counter() - t0)
