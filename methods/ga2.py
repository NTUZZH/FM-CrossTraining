"""Permutation GA v2 (Y1 fmwos.ga extended to overlays).

Genome remains an order permutation. The gap-aware serial decoder now places
each order at the earliest feasible point over ALL eligible technicians
(g_j in S_u) using the pair duration p(j, u); across eligible technicians it
prefers EARLIEST COMPLETION (start + p(j,u)), tie-broken by TB (primary
first, least-flexible, lowest id). Same GA hyperparameters as Y1: population
100, 60 s wall budget, OX crossover, swap mutation 0.2, tournament 3,
elitism 2, stall limit 200; PDR-seeded initial population (the five ranked
deterministic v2 rules).
"""

from __future__ import annotations

import random
import time

from env.conventions import pair_p_bh
from env.engine import PairDispatchEnv
from methods.rules import get_selector

_SEED_RULES = ("edd", "wspt", "atc", "pfifo", "mor")
_ELITES = 2
_TOURNAMENT = 3
_MUT_PROB = 0.2
_STALL_LIMIT = 200


def _prepare(instance, overlay):
    work_orders = instance["work_orders"]
    techs_src = (overlay or instance)["technicians"]
    eta = float((overlay or {}).get("eta",
                instance.get("meta", {}).get("eta", 1.0)) or 1.0)
    techs = []
    for tech in techs_src:
        if "skills" in tech:
            prim = tech.get("primary") or tech["skills"][0]
            skills = set(tech["skills"])
        else:
            prim = tech["trade"]
            skills = {prim}
        techs.append({"id": tech["id"], "primary": prim, "skills": skills,
                      "flex": len(skills)})
    elig = []      # j -> list of (tb_key, tech_index, p(j,u))
    for w in work_orders:
        g = w["trade"]
        row = []
        for u, tech in enumerate(techs):
            if g in tech["skills"]:
                p = pair_p_bh(w["p_bh"], tech["primary"] == g, eta)
                tb = (0 if tech["primary"] == g else 1, tech["flex"],
                      tech["id"])
                row.append((tb, u, p))
        row.sort()
        elig.append(row)
    return {
        "n": len(work_orders),
        "rel": [float(w["release_bh"]) for w in work_orders],
        "due": [float(w["due_bh"]) for w in work_orders],
        "wt": [float(w["weight"]) for w in work_orders],
        "wid": [w["id"] for w in work_orders],
        "id_to_index": {w["id"]: j for j, w in enumerate(work_orders)},
        "techs": techs,
        "elig": elig,
    }


def _earliest_start(intervals, r, p):
    """Earliest feasible start >= r on one technician (sorted busy list)."""
    if not intervals:
        return r
    t = r
    for (s, e) in intervals:
        if t + p <= s:
            return t
        if e > t:
            t = e
    return t


def _decode(prep, perm, want_assignments=False):
    """Serial gap-aware decode; returns (wwt, assignments|None)."""
    busy = [[] for _ in prep["techs"]]
    wwt = 0.0
    out = [] if want_assignments else None
    rel, due, wt = prep["rel"], prep["due"], prep["wt"]
    for j in perm:
        r = rel[j]
        best = None      # (end, tb, u, start, p)
        for (tb, u, p) in prep["elig"][j]:
            s = _earliest_start(busy[u], r, p)
            e = s + p
            key = (e, tb)
            if best is None or key < best[0]:
                best = (key, u, s, p)
        (e, _tb), u, s, p = best[0], best[1], best[2], best[3]
        # insert into the busy list keeping it sorted by start
        lst = busy[u]
        lo, hi = 0, len(lst)
        while lo < hi:
            mid = (lo + hi) // 2
            if lst[mid][0] < s:
                lo = mid + 1
            else:
                hi = mid
        lst.insert(lo, (s, e))
        wwt += wt[j] * max(0.0, e - due[j])
        if want_assignments:
            out.append({"wo": prep["wid"][j],
                        "tech": prep["techs"][u]["id"],
                        "start_bh": s, "end_bh": e})
    return wwt, out


def _perm_from_schedule(prep, schedule):
    order = sorted(schedule["assignments"],
                   key=lambda a: (a["start_bh"], a["wo"]))
    return [prep["id_to_index"][a["wo"]] for a in order]


def _ox(rng, a, b):
    n = len(a)
    i, j = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[i:j + 1] = a[i:j + 1]
    fill = [g for g in b if g not in set(child[i:j + 1])]
    k = 0
    for idx in list(range(0, i)) + list(range(j + 1, n)):
        child[idx] = fill[k]
        k += 1
    return child


def solve_ga(instance: dict, overlay: dict | None = None,
             budget_s: float = 60.0, seed: int = 301, pop: int = 100) -> dict:
    t0 = time.perf_counter()
    prep = _prepare(instance, overlay)
    n = prep["n"]
    rng = random.Random(seed)

    genomes = []
    env = PairDispatchEnv(instance, overlay)
    for rule in _SEED_RULES:
        sched = env.run_selector(get_selector(rule), method=rule, seed=seed)
        genomes.append(_perm_from_schedule(prep, sched))
    while len(genomes) < pop:
        g = list(range(n))
        rng.shuffle(g)
        genomes.append(g)

    evals = 0

    def fitness(g):
        nonlocal evals
        evals += 1
        return _decode(prep, g)[0]

    fits = [fitness(g) for g in genomes]
    best_i = min(range(len(genomes)), key=lambda i: fits[i])
    best_g, best_f = list(genomes[best_i]), fits[best_i]

    generations = 0
    stall = 0
    while time.perf_counter() - t0 < budget_s and stall < _STALL_LIMIT:
        order = sorted(range(len(genomes)), key=lambda i: fits[i])
        new = [list(genomes[i]) for i in order[:_ELITES]]
        while len(new) < pop:
            cand = rng.sample(range(len(genomes)), _TOURNAMENT)
            pa = genomes[min(cand, key=lambda i: fits[i])]
            cand = rng.sample(range(len(genomes)), _TOURNAMENT)
            pb = genomes[min(cand, key=lambda i: fits[i])]
            child = _ox(rng, pa, pb)
            if rng.random() < _MUT_PROB and n >= 2:
                i, j = rng.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]
            new.append(child)
        genomes = new
        fits = [fitness(g) for g in genomes]
        generations += 1
        gi = min(range(len(genomes)), key=lambda i: fits[i])
        if fits[gi] < best_f - 1e-12:
            best_f = fits[gi]
            best_g = list(genomes[gi])
            stall = 0
        else:
            stall += 1

    _wwt, assignments = _decode(prep, best_g, want_assignments=True)
    return {
        "instance_id": instance["meta"]["id"],
        "overlay_id": (overlay or {}).get("overlay_id"),
        "method": "ga",
        "seed": seed,
        "wall_seconds": time.perf_counter() - t0,
        "decisions": evals,
        "assignments": assignments,
        "generations": generations,
    }
