# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/tightness.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""Crew-tightness transform for the contention regimes (E3 / Gate B amendment).

``scale_crew(instance, m)`` returns a DEEP COPY of ``instance`` whose per-trade
technician count is scaled by ``m`` (``max(1, round(count * m))``), keeping every
work order untouched.  This turns any workload -- replay or generator -- into a
more (m < 1) or less (m > 1) capacity-contended variant, which is what makes the
dynamic dispatch problem discriminative (notes/decisions.md 2026-07-05, Gate B
protocol amendment: contended regimes use crew_multiplier in {0.6, 0.8}).

The copy's ``meta.crew_multiplier`` records ``m`` and its ``meta.id`` is suffixed
``_m<m>`` so scaled instances never collide with the originals in an index or a
results shard.  Technicians are renumbered ``T0..`` grouped by sorted trade so the
result is deterministic and self-consistent regardless of the source ids.
"""

from __future__ import annotations

import copy
from collections import defaultdict


def scale_crew(instance: dict, m: float) -> dict:
    """Deep-copy ``instance`` with per-trade crew counts scaled by ``m``.

    Each trade keeps ``max(1, round(original_count * m))`` technicians (never
    dropping a trade to zero).  Returns the new instance dict.
    """
    inst = copy.deepcopy(instance)

    by_trade: dict[str, int] = defaultdict(int)
    for tech in inst["technicians"]:
        by_trade[tech["trade"]] += 1

    new_techs = []
    tid = 0
    for trade in sorted(by_trade):
        count = by_trade[trade]
        scaled = max(1, int(round(count * m)))
        for _ in range(scaled):
            new_techs.append({"id": "T%d" % tid, "trade": trade})
            tid += 1
    inst["technicians"] = new_techs

    meta = dict(inst.get("meta", {}))
    meta["crew_multiplier"] = float(m)
    meta["id"] = "%s_m%s" % (meta.get("id", "inst"), m)
    inst["meta"] = meta
    return inst
