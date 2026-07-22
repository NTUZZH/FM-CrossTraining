# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/sensitivity.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""Instance-level perturbation transforms for the E5 sensitivity study
(SLA and capacity robustness; docs/protocol.md locked defaults: SLA sweep +-50%,
capacity sweep +-25%).

Two knobs, one transform each, both PURE functions on an instance dict:

* SLA tightness -- ``scale_sla(instance, f)`` shrinks (f < 1) or stretches
  (f > 1) every work order's due window about its release: the slack
  ``due_bh - release_bh`` is multiplied by ``f`` while ``release_bh`` (and the
  priority / weight / processing time) are left untouched.  f in {0.5, 1.5}
  realises the Appendix B +-50% SLA sweep.

* Capacity tightness -- REUSE ``fmwos.tightness.scale_crew(instance, m)`` (no
  new code): it scales every trade's technician count by ``m`` (>= 1 kept),
  m in {0.75, 1.25} realising the Appendix B +-25% capacity sweep.  It is
  re-exported here so the sensitivity runner imports a single module; see
  ``fmwos/tightness.py`` for the implementation and its ``_m<m>`` id suffix.

Both transforms deep-copy their input, annotate ``meta`` with the multiplier and
suffix ``meta.id`` (``_sla<f>`` here; ``_m<m>`` in ``scale_crew``) so a perturbed
instance never collides with the original in an index or a results shard.  The
identity cases -- ``scale_sla(inst, 1.0)`` / ``scale_crew(inst, 1.0)`` -- still
return an annotated deep copy, but the runner represents the (f=1, m=1) baseline
by the untouched base instance itself (no transform, no suffix), so the baseline
id is exactly the base id.
"""

from __future__ import annotations

import copy

# Re-export so callers get SLA and capacity transforms from one module.  Capacity
# robustness needs NO new code: fmwos.tightness.scale_crew already deep-copies,
# scales per-trade crew by max(1, round(count*m)), sets meta.crew_multiplier and
# suffixes meta.id with ``_m<m>`` -- exactly the contract scale_sla mirrors below.
from .tightness import scale_crew  # noqa: F401  (re-exported for the E5 runner)


def scale_sla(instance: dict, f: float) -> dict:
    """Deep-copy ``instance`` with every due window scaled by ``f`` about release.

    For every work order::

        due_bh := release_bh + f * (due_bh - release_bh)

    ``release_bh`` is preserved exactly (the arrival stream is unchanged), and
    ``priority``, ``weight`` and ``p_bh`` are never touched -- only the deadline
    slack is stretched (f > 1, looser SLA) or shrunk (f < 1, tighter SLA).  The
    copy records ``meta.sla_multiplier = f`` and suffixes ``meta.id`` with
    ``_sla<f>``.  Returns the new instance dict.
    """
    inst = copy.deepcopy(instance)

    for wo in inst["work_orders"]:
        release = float(wo["release_bh"])
        due = float(wo["due_bh"])
        wo["due_bh"] = release + float(f) * (due - release)

    meta = dict(inst.get("meta", {}))
    meta["sla_multiplier"] = float(f)
    meta["id"] = "%s_sla%s" % (meta.get("id", "inst"), f)
    inst["meta"] = meta
    return inst
