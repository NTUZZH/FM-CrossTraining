"""Shared v2 conventions: the eta processing-time grid.

p(j, u) = p_j                  if g_j == prim(u)      (exact float, untouched)
p(j, u) = ceil_grid(p_j / eta) if g_j in S_u \\ {prim(u)}
infeasible                     otherwise

``ceil_grid`` rounds UP to the 0.01 bh grid. It is applied identically in the
simulator, the CP-SAT model, and (re-derived independently) the validator, so
all three agree exactly.

Numerical note: p_j lives on a 1e-4 bh grid (Y1 instances round to 4 decimals)
and eta is in {1.0, 0.9, 0.8, 0.75}. p_j/eta * 100 is a rational N/(100*eta)
whose distance from any integer, when not exactly integral, is at least 1/90
~ 0.011, so a 1e-6 snap guard cleanly separates float noise (<= ~1e-9 here)
from genuine non-integrality.
"""

from __future__ import annotations

import math

GRID_BH = 0.01          # CP-SAT integer grid, in business hours
_SNAP = 1e-6            # float-noise guard for exact-integer quotients


def ceil_grid(x_bh: float) -> float:
    """Round ``x_bh`` UP to the 0.01 bh grid (with the snap guard)."""
    return math.ceil(x_bh * 100.0 - _SNAP) / 100.0


def pair_p_bh(p_bh: float, primary: bool, eta: float) -> float:
    """Realised processing time of a job on a technician."""
    if primary or eta >= 1.0:
        return float(p_bh)
    return ceil_grid(float(p_bh) / float(eta))
