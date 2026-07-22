"""Trade-adjacency and permutation chain overlays.

Three variants of the workload-adjacent chain, all built with the EXACT
construction conventions of overlays/build.py (same crew scaling, same Y1
technician numbering, same skill canonicalisation, same budget accounting).
Only the two topology choices vary:

  1. chain_adj  : the chain ORDER follows an expert trade-skill adjacency
                  grouping instead of descending p95 weekly hours. phi = 1.0.
  2. permutation: the chain order is a seeded random cycle (build.py already
                  supports this via perm_seed; kept here only for symmetry /
                  documentation -- the runner calls build_overlay directly).
  3. tech-select: at partial adoption (phi = 0.5) the technicians that
                  receive the second skill are drawn at random within each
                  trade, instead of the released "lowest-indexed" rule.

Trade codes are Uniformat II Level-2 SystemCodes (data/raw/FMUCD.csv):

    B20 Exterior Enclosure      D30 HVAC              E10 Equipment
    B30 Roofing                 D40 Fire Protection   E20 Furnishings
    C10 Interior Construction   D50 Electrical        MISC merged-small
    C30 Interior Finishes       D90 General bldg svc  UNK  unknown
    D10 Conveying (elevators)   D20 Plumbing

Expert adjacency clusters (a maintenance training department's cross-skill
families -- a technician cross-trained between two trades in the same cluster
is operationally realistic):

    mechanical  : D20 plumbing, D30 HVAC, D40 fire protection (wet-pipe
                  suppression), D90 general building services  -- pipefitting,
                  valves/pumps, thermal and wet systems.
    electrical  : D50 electrical, D10 conveying (elevators) -- power and
                  electro-mechanical controls.
    envelope    : B20 exterior enclosure, B30 roofing -- weatherproofing.
    interiors   : C10 interior construction, C30 interior finishes --
                  framing/drywall and paint/flooring finish trades.
    general     : E10 equipment, E20 furnishings, MISC, UNK -- general
                  maintenance / handyman and the merged small/unknown trades.
"""
from __future__ import annotations

import hashlib
import math

from overlays.build import (chain_order, load_crews, overlay_id,  # noqa
                            scaled_crews, _technicians, OVERLAY_SEED)

TRADE_LEGEND = {
    "B20": "Exterior Enclosure", "B30": "Roofing",
    "C10": "Interior Construction", "C30": "Interior Finishes",
    "D10": "Conveying (elevators)", "D20": "Plumbing", "D30": "HVAC",
    "D40": "Fire Protection", "D50": "Electrical",
    "D90": "General building services", "E10": "Equipment",
    "E20": "Furnishings", "MISC": "Merged small trades", "UNK": "Unknown",
}

# Expert adjacency clusters, in a fixed reference order. Cluster membership is
# by trade code; a campus uses only the codes present in its crew table.
CLUSTERS: dict[str, tuple[str, ...]] = {
    "mechanical": ("D20", "D30", "D40", "D90"),
    "electrical": ("D50", "D10"),
    "envelope": ("B20", "B30"),
    "interiors": ("C10", "C30"),
    "general": ("E10", "E20", "MISC", "UNK"),
}


def cluster_of(trade: str) -> str:
    for name, members in CLUSTERS.items():
        if trade in members:
            return name
    return "general"          # any unforeseen code lands in the catch-all


def adjacency_order(crews) -> list[str]:
    """Trade cycle that keeps same-cluster trades contiguous.

    Clusters are ordered by descending total p95 weekly hours; within a
    cluster, trades by descending p95 hours (tie: trade name ascending) --
    the released workload convention, applied inside each skill family. A
    contiguous-block cycle over C clusters has exactly C inter-cluster edges,
    the minimum possible, so adjacency is maximised.
    """
    vol = {c["trade"]: c["volume"] for c in crews}
    trades = list(vol)
    buckets: dict[str, list[str]] = {}
    for t in trades:
        buckets.setdefault(cluster_of(t), []).append(t)
    cluster_vol = {name: sum(vol[t] for t in ts) for name, ts in buckets.items()}
    order: list[str] = []
    for name in sorted(buckets, key=lambda n: (-cluster_vol[n], n)):
        order += sorted(buckets[name], key=lambda t: (-vol[t], t))
    return order


def _chosen_pool(pool, n_flex, tech_seed):
    """Which technicians in a trade pool receive the secondary skill.

    tech_seed is None -> released convention (lowest-indexed, pool[:n_flex]).
    Otherwise a per-(seed, trade) deterministic random draw (stable across
    processes: no salted built-in hash).
    """
    if tech_seed is None:
        return pool[:n_flex]
    trade = pool[0]["primary"] if pool else ""
    h = hashlib.sha256(("%d:%s" % (int(tech_seed), trade)).encode()).hexdigest()
    import random
    rng = random.Random(int(h[:16], 16))
    idx = sorted(rng.sample(range(len(pool)), n_flex))
    return [pool[i] for i in idx]


def build_chain_variant(campus, crews, phi, eta, m, *, order=None,
                        tech_seed=None, struct_label="chain",
                        variant=None, perm_seed=None) -> dict:
    """Build one chain-variant overlay, byte-for-byte in build.py's style.

    order       : explicit trade cycle (chain_adj). None -> workload order
                  (optionally shuffled by perm_seed, build.py convention).
    tech_seed   : random technician selection within trade (else lowest-idx).
    struct_label: value written to the overlay 'structure' field.
    """
    crew_of = scaled_crews(crews, m)
    if order is None:
        order = chain_order(crews, perm_seed)
    K = len(order)
    techs = _technicians(crew_of)
    by_trade: dict[str, list] = {}
    for t in techs:
        by_trade.setdefault(t["primary"], []).append(t)   # T-index order

    if K > 1:
        for k, trade in enumerate(order):
            succ = order[(k + 1) % K]
            pool = by_trade[trade]
            n_flex = math.ceil(float(phi) * len(pool))
            for t in _chosen_pool(pool, n_flex, tech_seed):
                if succ not in t["skills"]:
                    t["skills"] = t["skills"] + [succ]

    # Canonicalise skills exactly as build.build_overlay does.
    for t in techs:
        rest = sorted(s for s in set(t["skills"]) if s != t["primary"])
        t["skills"] = [t["primary"]] + rest

    budget = sum(len(t["skills"]) - 1 for t in techs)
    oid = variant_overlay_id(campus, struct_label, phi, eta, m,
                             perm_seed=perm_seed, tech_seed=tech_seed)
    return {
        "overlay_id": oid,
        "campus": int(campus),
        "structure": struct_label,
        "phi": float(phi),
        "eta": float(eta),
        "crew_multiplier": float(m),
        "chain_order": order,
        "budget_B": int(budget),
        "headcount": len(techs),
        "technicians": techs,
        "variant": variant,
        "selection": ("lowest_indexed" if tech_seed is None
                      else "random_seed_%d" % int(tech_seed)),
        "provenance": {
            "crews": "[C] Y1 p95 calibration (capacity.csv)",
            "skills": "[D] topology overlay, base seed %d%s%s" % (
                OVERLAY_SEED,
                "" if perm_seed is None else ", chain perm %d" % perm_seed,
                "" if tech_seed is None else ", tech-select %d" % tech_seed),
        },
    }


# --------------------------------------------------------------------------- #
# Same-budget sparse-topology controls. Every technician receives exactly ONE
# secondary skill, so B = headcount = B(CHAIN(1.0)) by construction; only the
# trade-level secondary map sigma (trade -> secondary trade) varies. These are
# the classical comparison structures for the chain: disconnected reciprocal
# pairs, an unconstrained random one-secondary graph, an uneven hub, and a
# licence-constrained chain.

LICENSED_TRADES = ("D10", "D20", "D40", "D50")
# Licence-gated building trades: conveying/elevators (D10), plumbing (D20),
# fire protection (D40), electrical (D50). A technician of another trade
# cannot take these up as a cross-trained secondary, so arcs INTO them are
# prohibited; arcs OUT of them are allowed (a licensed electrician may learn
# an unlicensed trade).
# A milder "normal qualification" regime gates only the two trades whose
# certification is hardest to obtain, electrical (D50) and elevators (D10);
# plumbing and fire protection are treated as ordinarily cross-trainable.
LICENSED_NORMAL = ("D50", "D10")


def sigma_pairs(order):
    """Disjoint reciprocal trade pairs in the given cycle order.

    Odd K: the last three trades form one 3-cycle, so every trade still has
    exactly one secondary and the membership budget stays B = headcount."""
    K = len(order)
    sig = {}
    if K < 2:
        return sig
    end = K if K % 2 == 0 else K - 3
    for i in range(0, end, 2):
        a, b = order[i], order[i + 1]
        sig[a], sig[b] = b, a
    if K % 2 == 1:
        a, b, c = order[K - 3], order[K - 2], order[K - 1]
        sig[a], sig[b], sig[c] = b, c, a
    return sig


def sigma_rand1(order, seed):
    """Seeded uniform random secondary per trade; connectivity unconstrained.

    Stable across processes (sha-derived rng, no salted built-in hash)."""
    import random
    sig = {}
    for t in order:
        others = [g for g in order if g != t]
        if not others:
            continue
        h = hashlib.sha256(("rand1:%d:%s"
                            % (int(seed), t)).encode()).hexdigest()
        sig[t] = random.Random(int(h[:16], 16)).choice(others)
    return sig


def sigma_star(order):
    """Hub topology: every trade's secondary is the largest-workload trade;
    the hub trade's own secondary is the second-largest."""
    sig = {}
    if len(order) < 2:
        return sig
    hub = order[0]
    for t in order[1:]:
        sig[t] = hub
    sig[hub] = order[1]
    return sig


def sigma_feas(order, licensed=LICENSED_TRADES):
    """Licence-constrained chain: each trade's secondary is the NEXT trade in
    cyclic order that is not licence-gated. Licence-gated trades receive no
    in-arcs (their crews get no help); they still give one out-arc. Raises if
    a trade has no legal target (never happens on the released campuses)."""
    K = len(order)
    sig = {}
    if K < 2:
        return sig
    for k, t in enumerate(order):
        for step in range(1, K):
            cand = order[(k + step) % K]
            if cand != t and cand not in licensed:
                sig[t] = cand
                break
        if t not in sig:
            raise ValueError("no licence-legal secondary for trade %r" % t)
    return sig


def build_sigma_variant(campus, crews, sigma, eta, m, *, order,
                        struct_label, variant) -> dict:
    """Overlay from a trade-level secondary map, in build.py's exact style
    (same crew scaling, technician numbering, skill canonicalisation)."""
    crew_of = scaled_crews(crews, m)
    techs = _technicians(crew_of)
    for t in techs:
        succ = sigma.get(t["primary"])
        if succ is not None and succ not in t["skills"]:
            t["skills"] = t["skills"] + [succ]
    for t in techs:
        rest = sorted(s for s in set(t["skills"]) if s != t["primary"])
        t["skills"] = [t["primary"]] + rest
    budget = sum(len(t["skills"]) - 1 for t in techs)
    oid = "c%02d_%s_eta%03d_m%03d" % (int(campus), struct_label,
                                      int(round(float(eta) * 100)),
                                      int(round(float(m) * 100)))
    return {
        "overlay_id": oid, "campus": int(campus), "structure": struct_label,
        "phi": 1.0, "eta": float(eta), "crew_multiplier": float(m),
        "chain_order": list(order), "sigma": dict(sigma),
        "budget_B": int(budget), "headcount": len(techs),
        "technicians": techs, "variant": variant,
        "provenance": {
            "crews": "[C] Y1 p95 calibration (capacity.csv)",
            "skills": "[D] sparse-topology control %r, base seed %d"
                      % (struct_label, OVERLAY_SEED)},
    }


def topology_descriptors(sigma, order, crew_of):
    """Connectedness, coverage, degree, and added-eligibility capacity of the
    trade-level secondary digraph (reported per control topology)."""
    trades = list(order)
    indeg = {t: 0 for t in trades}
    for _s, d in sigma.items():
        indeg[d] += 1
    adj = {t: set() for t in trades}
    for s, d in sigma.items():
        adj[s].add(d)
        adj[d].add(s)
    seen: set[str] = set()
    comps = 0
    for t in trades:
        if t in seen:
            continue
        comps += 1
        stack = [t]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj[u] - seen)
    covered = [t for t in trades if indeg[t] > 0]
    helpers = {t: sum(crew_of[s] for s, d in sigma.items() if d == t)
               for t in trades}
    return {
        "n_trades": len(trades),
        "n_arcs": len(sigma),
        "weak_components": comps,
        "coverage_share": (len(covered) / len(trades)) if trades else 0.0,
        "uncovered_trades": [t for t in trades if indeg[t] == 0],
        "in_degree_max": max(indeg.values()) if trades else 0,
        "in_degree": indeg,
        "added_eligible_techs": helpers,
    }


def variant_overlay_id(campus, struct_label, phi, eta, m, *,
                       perm_seed=None, tech_seed=None) -> str:
    tag = {"chain_adj": "chainadj", "chain": "chain"}.get(struct_label,
                                                          struct_label)
    parts = ["c%02d" % int(campus), tag]
    if tag in ("chain", "chainadj"):
        parts.append("phi%03d" % int(round(float(phi) * 100)))
    parts.append("eta%03d" % int(round(float(eta) * 100)))
    parts.append("m%03d" % int(round(float(m) * 100)))
    if perm_seed is not None:
        parts.append("perm%d" % perm_seed)
    if tech_seed is not None:
        parts.append("tsel%d" % tech_seed)
    return "_".join(parts)
