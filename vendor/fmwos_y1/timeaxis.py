# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/timeaxis.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""Business-hour (bh) time axis for FM work-order scheduling.

The scheduling axis is a *continuous* line of business hours: each weekday
contributes exactly 8 contiguous hours (the shift 08:00-16:00 local naive time)
and weekends contribute nothing. Consecutive weekday shifts are glued together,
so bh 8 (end of Monday's shift) is the same point as bh 8 = start of Tuesday's
shift. This removes calendars from the scheduling stack (see notes/interfaces.md
and decisions.md 2026-07-04).

Public API
----------
- ``SLA_BH``   : {priority -> due-window length in bh}
- ``WEIGHT``   : {priority -> tardiness weight}
- ``to_bh(ts, t0)``        : bh offset of a single timestamp from anchor ``t0``.
- ``to_bh_series(ts, t0)`` : vectorised version for a pandas Series / array.
- ``abs_bh(ts)`` / ``abs_bh_series(ts)`` : absolute bh from a fixed epoch
  (building block; ``to_bh`` is just a difference of these).

Conventions
-----------
- ``t0`` must be a weekday at 08:00 (a shift start); ``bh(t0) == 0``.
- Timestamps that fall inside a weekend or outside the 08:00-16:00 shift map
  *forward* to the next shift start (they are released at the next moment a
  technician could touch them). A timestamp exactly at 16:00 maps to the next
  business day's 08:00.
- All processing times p_j (LaborHours) are consumed 1:1 on this axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# SLA due-window length per priority class, in business hours.
# (Appendix B calendar values converted 24 cal-h -> 8 bh, 7 cal-d -> 40 bh.)
SLA_BH: dict[int, float] = {1: 8.0, 2: 24.0, 3: 80.0, 4: 171.4}

# Priority-weighted tardiness weights.
WEIGHT: dict[int, float] = {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}

_SHIFT_START_H = 8.0   # 08:00 local naive
_SHIFT_END_H = 16.0    # 16:00 local naive
_SHIFT_LEN_H = _SHIFT_END_H - _SHIFT_START_H  # 8 bh / weekday
_EPOCH = np.datetime64("1970-01-01", "D")     # a Thursday (weekday index 3)


def abs_bh_series(ts) -> np.ndarray:
    """Absolute business hours from ``_EPOCH`` for each timestamp in ``ts``.

    Off-shift / weekend timestamps are clamped *forward* to the next shift
    start before counting. Returns a float64 numpy array (monotone
    non-decreasing in calendar time).
    """
    values = pd.to_datetime(pd.Series(ts).reset_index(drop=True)).to_numpy(
        dtype="datetime64[us]"
    )
    return _abs_bh_ndarray(values)


def _abs_bh_ndarray(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="datetime64[us]")
    dates = values.astype("datetime64[D]")
    # Fractional hours into the calendar day.
    hf = (values - dates) / np.timedelta64(1, "h")
    days = dates.astype("int64")            # days since 1970-01-01
    wd = (days + 3) % 7                      # 0=Mon .. 6=Sun (epoch is Thu)

    weekend = wd >= 5
    on_weekday = ~weekend
    after = on_weekday & (hf >= _SHIFT_END_H)
    inside = on_weekday & (hf >= _SHIFT_START_H) & (hf < _SHIFT_END_H)
    # `before` (weekday, hf < 8) and `inside` keep their own date.

    into = np.where(inside, hf - _SHIFT_START_H, 0.0)

    eff = dates.copy()
    if after.any():
        # after the shift on a weekday -> next business day 08:00
        eff[after] = np.busday_offset(dates[after], 1, roll="forward")
    if weekend.any():
        # weekend -> roll forward to the next business day 08:00
        eff[weekend] = np.busday_offset(dates[weekend], 0, roll="forward")

    bdays = np.busday_count(_EPOCH, eff).astype("float64")
    return bdays * _SHIFT_LEN_H + into


def abs_bh(ts) -> float:
    """Absolute bh from ``_EPOCH`` for a single timestamp."""
    values = np.array([np.datetime64(pd.Timestamp(ts), "us")], dtype="datetime64[us]")
    return float(_abs_bh_ndarray(values)[0])


def to_bh(ts, t0) -> float:
    """bh offset of timestamp ``ts`` from anchor ``t0`` (``bh(t0) == 0``).

    ``t0`` is expected to be a weekday 08:00 shift start. Off-shift/weekend
    ``ts`` map forward to the next shift start.
    """
    return abs_bh(ts) - abs_bh(t0)


def to_bh_series(ts, t0) -> np.ndarray:
    """Vectorised :func:`to_bh` over a pandas Series / array of timestamps."""
    return abs_bh_series(ts) - abs_bh(t0)
