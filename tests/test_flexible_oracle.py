"""Flexible-cell regression oracle (closes the top residual regression risk).

The L0 regression anchor cross-checks the pair engine against the
predecessor ONLY at singleton skills. This test gives chain/full/eta<1
cells the analogous independent oracle: a from-scratch re-implementation of
the pair-selection event loop (different data structures, no import from
env/engine.py) that, for each deterministic rule, must reproduce the
engine's schedule bitwise on hundreds of small flexible instances.

Independence: the oracle shares NO code with env/engine.py. It implements
the same well-specified dispatch semantics (non-delay pair selection, the
rule's order key, the technician tie-break, and the eta pair-duration grid)
in an independent event loop, so a match confirms the engine's queue
bookkeeping, event ordering, non-delay loop, pair-duration application, and
tie-break are all correct at flexible cells, not just at L0.

Run: PYTHONPATH=.:vendor python tests/test_flexible_oracle.py [n_instances]
"""
import glob
import json
import math
import os
import random
import sys
from pathlib import Path

from env.engine import PairDispatchEnv
from env.validator2 import validate as validate2
from methods.rules import get_selector
from overlays.build import build_overlay, load_crews

_REPO = Path(__file__).resolve().parents[1]
Y1_ROOT = Path(os.environ.get("FMWOS_Y1_ROOT", _REPO.parent / "FM-Scheduling"))
CAP = str(Y1_ROOT / "results/p1_calib/capacity.csv")
INST_GLOB = str(Y1_ROOT / "data/processed/instances/c*/replay/50/*.json")

# Deterministic rules with simple, independently-implementable order keys.
# (ATC/LFJ-ATC/ATC-eta are exercised by the feasibility + no-worse tests in
# tests/test_methods2.py; here we want rules whose key we can re-derive with
# zero shared logic.)
ORDER_KEY = {
    "edd": lambda j, t: (j["due_bh"], j["id"]),
    "wspt": lambda j, t: (-(j["weight"] / j["p_bh"]), j["id"]),
    "pfifo": lambda j, t: (j["priority"], j["release_bh"], j["id"]),
    "mor": lambda j, t: (-j["p_bh"], j["id"]),
}


def _ceil_grid(x):
    return math.ceil(x * 100.0 - 1e-6) / 100.0


def _pair_p(p, primary, eta):
    return p if (primary or eta >= 1.0) else _ceil_grid(p / eta)


def oracle_schedule(instance, overlay, rule):
    """Independent from-scratch pair-selection simulator for one rule.

    Uses a plain time-sorted event list (no heap), a dict of per-trade
    queues, and an explicit non-delay inner loop, all written independently
    of env/engine.py.
    """
    key = ORDER_KEY[rule]
    eta = float(overlay["eta"])
    techs = {t["id"]: {"primary": t["primary"], "skills": set(t["skills"]),
                       "flex": len(t["skills"]), "free": 0.0}
             for t in overlay["technicians"]}
    trades = sorted({g for t in overlay["technicians"] for g in t["skills"]}
                    | {w["trade"] for w in instance["work_orders"]})
    queue = {g: [] for g in trades}

    # Event list: (time, kind, payload). kind 0 = tech free, 1 = release.
    # Process strictly in (time, kind) order, draining ties.
    events = [(0.0, 0, tid) for tid in techs]
    for w in instance["work_orders"]:
        events.append((float(w["release_bh"]), 1, w))
    # A monotone counter breaks payload comparison; sort by (time, kind, seq).
    events = [(e[0], e[1], i, e[2]) for i, e in enumerate(events)]
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    idle = set()
    assignments = []
    ei = 0
    n = len(events)
    completions = []          # (time, tid) pending, kept sorted lazily

    def eligible_idle(trade):
        return [tid for tid in idle if trade in techs[tid]["skills"]]

    def tb(tid, trade):
        return (0 if techs[tid]["primary"] == trade else 1,
                techs[tid]["flex"], tid)

    # Merge release events and completion events on one timeline.
    pending_rel = events
    ri = 0
    while ri < n or completions:
        # Next event time = min of next release and next completion.
        next_rel = pending_rel[ri][0] if ri < n else math.inf
        completions.sort()
        next_comp = completions[0][0] if completions else math.inf
        now = min(next_rel, next_comp)
        if now == math.inf:
            break
        # Drain all completions at `now`.
        while completions and completions[0][0] == now:
            _, tid = completions.pop(0)
            techs[tid]["free"] = now
            idle.add(tid)
        # Drain all releases at `now`.
        while ri < n and pending_rel[ri][0] == now:
            _, kind, _seq, payload = pending_rel[ri]
            ri += 1
            if kind == 0:
                idle.add(payload)
            else:
                queue[payload["trade"]].append(payload)
        # Non-delay inner loop: dispatch while any feasible pair exists.
        while True:
            best_trade = None
            best_job = None
            best_k = None
            for g in trades:
                if not queue[g] or not eligible_idle(g):
                    continue
                for j in queue[g]:
                    k = key(j, now)
                    if best_k is None or k < best_k:
                        best_k = k
                        best_job = j
                        best_trade = g
            if best_job is None:
                break
            # Technician tie-break.
            cands = eligible_idle(best_trade)
            u = min(cands, key=lambda tid: tb(tid, best_trade))
            queue[best_trade].remove(best_job)
            idle.discard(u)
            p = _pair_p(best_job["p_bh"], techs[u]["primary"] == best_trade,
                        eta)
            start = now
            end = start + p
            techs[u]["free"] = end
            assignments.append((best_job["id"], u, round(start, 9),
                                round(end, 9)))
            completions.append((end, u))
    return sorted(assignments)


def engine_canon(sched):
    return sorted((a["wo"], a["tech"], round(a["start_bh"], 9),
                   round(a["end_bh"], 9)) for a in sched["assignments"])


def load_sample(n):
    paths = sorted(glob.glob(INST_GLOB))
    rng = random.Random(20260708)
    return [json.load(open(p)) for p in rng.sample(paths, min(n, len(paths)))]


def test_flexible_oracle(instances):
    cells = [("chain", 1.0, 0.8, 0.6), ("chain", 1.0, 1.0, 0.6),
             ("full", None, 0.8, 0.6), ("full", None, 0.75, 0.8),
             ("chain", 0.5, 0.9, 0.6), ("generalist", None, 0.8, 0.6),
             ("full", None, 0.8, 1.0)]
    n_checked = 0
    for inst in instances:
        campus = inst["meta"]["campus"]
        crews = load_crews(CAP, campus)
        for (st, phi, eta, m) in cells:
            ov = build_overlay(campus, crews, st, phi, eta, m)
            env = PairDispatchEnv(inst, ov, check_nondelay=True)
            for rule in ORDER_KEY:
                eng = env.run_selector(get_selector(rule), method=rule,
                                       seed=301)
                assert validate2(inst, eng, ov)["feasible"], (rule, st)
                orc = oracle_schedule(inst, ov, rule)
                assert engine_canon(eng) == orc, (
                    "ORACLE MISMATCH rule=%s cell=(%s,phi=%s,eta=%s,m=%s) "
                    "inst=%s\n  engine has %d assigns, oracle %d"
                    % (rule, st, phi, eta, m, inst["meta"]["id"],
                       len(eng["assignments"]), len(orc)))
                n_checked += 1
    print("  %d (instance x cell x rule) bitwise oracle checks passed"
          % n_checked)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    instances = load_sample(n)
    test_flexible_oracle(instances)
    print("PASS test_flexible_oracle (%d flexible instances)" % n)
    print("OK flexible oracle: chain/full/gen x eta{1.0,0.9,0.8,0.75} "
          "cross-checked against an independent event loop")
