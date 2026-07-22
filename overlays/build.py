"""Resource-overlay generator: the flexibility ladder.

An overlay is a deterministic function of the Y1-calibrated crews for a campus
(trade g_k, crew size c_k, weekly labour volume), the structure label, phi,
eta, the crew multiplier m, and (for chain-permutation sensitivity only) a
seed. Headcount never changes with structure; only skill sets do.

Technician construction (matches Y1 exactly):
crews are scaled per trade as max(1, int(round(c_k * m))) (fmwos.tightness
convention, banker's rounding via Python round), technicians are numbered
T0.. sequentially over trades in SORTED-name order, so the L0 overlay's
technician list is byte-identical to Y1's instance technicians at m = 1.0 and
to tightness.scale_crew(instance, m) at m < 1. Tie-breaks downstream compare
technician ids as strings (Y1's heap order).

Chain order: trades sorted by DESCENDING p95_weekly_hours from
the released Y1 capacity table (tie: trade name ascending); the chain is the
cycle t_1 -> t_2 -> ... -> t_K -> t_1 ("workload-adjacent chaining").

Structures:
  dedicated  : S_u = {prim(u)}  (L0; identical to v1)
  chain(phi) : per chain-position k, the ceil(phi * c_k) lowest-indexed
               technicians of trade t_k additionally receive skill t_{k+1 mod K}
  generalist : budget-matched pool; B_target = headcount (= B(CHAIN(1.0)));
               n_gen = round(B_target / (K - 1)) technicians become full
               generalists (S_u = G), apportioned over trades proportionally
               to crew size by largest remainder (ties: larger crew first,
               then trade name), lowest-indexed technicians within a trade
  full       : S_u = G for all u

Degenerate topologies (unit-tested): K = 1 -> chain/generalist collapse to L0
(no secondary skill exists); K = 2 -> CHAIN(1.0) coincides with FULL.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OVERLAY_SEED = 20260707          # recorded provenance seed (base ladder is
                                 # fully deterministic; the seed matters only
                                 # for chain permutations in E5')
STRUCTURES = ("dedicated", "chain", "generalist", "full")


def load_crews(capacity_csv: str | Path, campus: int):
    """Read the Y1 capacity table -> list of dicts {trade, crew, volume}."""
    import csv

    rows = []
    with open(capacity_csv, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["campus"]) == int(campus):
                rows.append({"trade": r["trade"], "crew": int(r["crew"]),
                             "volume": float(r["p95_weekly_hours"])})
    if not rows:
        raise ValueError("campus %r not in capacity table" % campus)
    return rows


def chain_order(crews, perm_seed: int | None = None):
    """Workload-adjacent chain order (descending volume; tie: trade name).

    ``perm_seed`` (E5' sensitivity only) applies a seeded random permutation
    of the cycle instead.
    """
    trades = [c["trade"] for c in
              sorted(crews, key=lambda c: (-c["volume"], c["trade"]))]
    if perm_seed is not None:
        import random
        rng = random.Random(perm_seed)
        rng.shuffle(trades)
    return trades


def scaled_crews(crews, m: float):
    """Apply the crew multiplier FIRST, Y1 convention."""
    return {c["trade"]: max(1, int(round(c["crew"] * m))) for c in crews}


def _technicians(crew_of: dict[str, int]):
    """Y1-convention technician list: T0.. over sorted trade names."""
    techs = []
    tid = 0
    for trade in sorted(crew_of):
        for _ in range(crew_of[trade]):
            techs.append({"id": "T%d" % tid, "primary": trade,
                          "skills": [trade]})
            tid += 1
    return techs


def build_overlay(campus: int, crews, structure: str, phi, eta: float,
                  m: float, perm_seed: int | None = None) -> dict:
    """Build one overlay dict."""
    if structure not in STRUCTURES:
        raise ValueError("structure must be one of %r" % (STRUCTURES,))
    crew_of = scaled_crews(crews, m)
    order = chain_order(crews, perm_seed)
    K = len(order)
    techs = _technicians(crew_of)
    by_trade = {}
    for t in techs:
        by_trade.setdefault(t["primary"], []).append(t)   # T-index order

    if structure == "chain":
        if phi is None:
            raise ValueError("chain needs phi")
        if K > 1:
            for k, trade in enumerate(order):
                succ = order[(k + 1) % K]
                pool = by_trade[trade]
                n_flex = math.ceil(float(phi) * len(pool))
                for t in pool[:n_flex]:
                    if succ not in t["skills"]:
                        t["skills"] = t["skills"] + [succ]
    elif structure == "generalist":
        if K > 1:
            all_trades = sorted(crew_of)
            headcount = sum(crew_of.values())
            b_target = headcount                       # = B(CHAIN(1.0))
            n_gen = int(round(b_target / (K - 1)))
            n_gen = min(n_gen, headcount)
            # Largest-remainder apportionment over trades by crew size.
            quotas = {g: n_gen * crew_of[g] / headcount for g in all_trades}
            base = {g: int(math.floor(quotas[g])) for g in all_trades}
            rem = n_gen - sum(base.values())
            frac_order = sorted(all_trades,
                                key=lambda g: (-(quotas[g] - base[g]),
                                               -crew_of[g], g))
            for g in frac_order[:rem]:
                base[g] += 1
            for g in all_trades:
                take = min(base[g], crew_of[g])
                for t in by_trade[g][:take]:
                    t["skills"] = list(all_trades)     # full generalist
    elif structure == "full":
        all_trades = sorted(crew_of)
        for t in techs:
            t["skills"] = list(all_trades)

    # Canonicalise skills: primary first, then the rest sorted.
    for t in techs:
        rest = sorted(s for s in set(t["skills"]) if s != t["primary"])
        t["skills"] = [t["primary"]] + rest

    budget = sum(len(t["skills"]) - 1 for t in techs)
    return {
        "overlay_id": overlay_id(campus, structure, phi, eta, m, perm_seed),
        "campus": int(campus),
        "structure": structure,
        "phi": (float(phi) if structure == "chain" else None),
        "eta": float(eta),
        "crew_multiplier": float(m),
        "chain_order": order,
        "budget_B": int(budget),
        "headcount": len(techs),
        "technicians": techs,
        "provenance": {"crews": "[C] Y1 p95 calibration (capacity.csv)",
                       "skills": "[D] designed ladder, seed %d%s" % (
                           OVERLAY_SEED,
                           "" if perm_seed is None
                           else ", chain perm seed %d" % perm_seed)},
    }


def overlay_id(campus, structure, phi, eta, m, perm_seed=None):
    tag = {"dedicated": "l0", "chain": "chain", "generalist": "gen",
           "full": "full"}[structure]
    parts = ["c%02d" % int(campus), tag]
    if structure == "chain":
        parts.append("phi%03d" % int(round(float(phi) * 100)))
    parts.append("eta%03d" % int(round(float(eta) * 100)))
    parts.append("m%03d" % int(round(float(m) * 100)))
    if perm_seed is not None:
        parts.append("perm%d" % perm_seed)
    return "_".join(parts)


def apply_overlay(instance: dict, overlay: dict) -> dict:
    """Return a SHALLOW-copied instance whose technicians come from the
    overlay (work orders and meta shared; meta extended with overlay refs).

    The instance's own technician list is replaced entirely; overlay
    construction guarantees every instance trade has >= 1 primary technician
    (same trade universe as Y1's capacity table).
    """
    if int(overlay["campus"]) != int(instance["meta"]["campus"]):
        raise ValueError("overlay campus %r != instance campus %r"
                         % (overlay["campus"], instance["meta"]["campus"]))
    inst = dict(instance)
    inst["technicians"] = overlay["technicians"]
    meta = dict(instance["meta"])
    meta["overlay_id"] = overlay["overlay_id"]
    meta["structure"] = overlay["structure"]
    meta["phi"] = overlay["phi"]
    meta["eta"] = overlay["eta"]
    meta["crew_multiplier"] = overlay["crew_multiplier"]
    inst["meta"] = meta
    return inst


def save_overlay(overlay: dict, root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p = root / (overlay["overlay_id"] + ".json")
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(overlay, f, indent=1)
    import os
    os.replace(tmp, p)
    return p
