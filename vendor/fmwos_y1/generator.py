# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/generator.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""Calibrated parametric instance generator (track [C]).

Complements the replay track (``fmwos.instances``): instead of slicing real
release windows out of FMUCD, it fits a small per-campus parametric model on the
*train* years and then samples fresh, statistically-comparable instances. This
supports controlled sweeps over (instance size x crew tightness x PM/CM ratio)
that the finite replay corpus cannot cover.

Fitting (``fit_params``) is done on the CLEANED dataframe (``fmwos.io.clean``),
per campus, reusing the calibration primitives in ``fmwos.calib`` unchanged:
the SAME MISC trade merge, the SAME v2 priority mapping, and the SAME crew
capacities. Per merged trade it records

  * ``arrival_rate_per_bh`` = (# train WOs of the trade) / (business hours in the
    campus's train span); the train span is measured on the bh axis, so this is
    a mean arrival intensity, 40 bh per week;
  * a lognormal (mu, sigma) for log(LaborHours), fitted on the train rows BEFORE
    any per-campus cap is applied (the generator clips its draws at the campus
    p99.5 labor cap recorded in the pack);
  * ``pm_share`` (fraction of train rows that are PPM);
  * two priority-class distributions: PM rows are always class 4 (rule R5a),
    CM rows use the empirical class distribution over 1..4;
  * per-quarter (Q1..Q4) arrival multipliers (a seasonal option, unused by the
    default generator path).

Campus-level the pack carries the crew capacity dict (trade -> crew) and the
p99.5 labor cap.

Generation (``generate``) is a homogeneous Poisson superposition: per trade an
independent Poisson process at rate ``arrival_rate * arrival_multiplier`` is
grown over an expanding window until at least ``size`` arrivals exist, the
``size`` earliest are kept, and ``window_bh`` is set to the last kept release
(min 8). Every draw comes from a single ``numpy.random.Generator(seed)`` so the
whole instance is a deterministic function of (params, size, seed, knobs).

Fixed-window generation (``generate_window``) is the storm-v2 / utilization-sweep
variant (track ``storm2``): the same Poisson superposition is drawn over a FIXED
window ``[0, window_bh)`` and EVERY arrival is kept, so the work-order count
``n`` is itself random and scales with ``arrival_multiplier`` -- a rate-scaled
workload against fixed capacity, which (unlike first-N sampling) actually
produces sustained overload (notes/phase4_report.md "E3 axes status").
``base_utilization`` reports the pack's offered-load / capacity ratio ``u0`` at
``arrival_multiplier = 1`` (using the clipped-lognormal mean processing time, so
it is consistent with the labor-cap clipping the generator applies), hence
drawing at ``arrival_multiplier = u_target / u0`` targets a realized utilization
of ``u_target``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import calib
from . import timeaxis as ta

TRAIN_END = pd.Timestamp("2017-12-31 23:59:59")
_WEEK_BH = 40.0
LABOR_CAP_Q = 0.995        # per-campus p99.5 cap for generated draws
P_BH_FLOOR = 0.05          # smallest processing time a generated WO may take
WINDOW_MIN_BH = 8.0
_OVERSHOOT = 2.0           # initial arrival window aims for ~2x the target size
_MAX_GROW = 40             # safety cap on window-doubling iterations


# --------------------------------------------------------------------------- #
# Fitting                                                                     #
# --------------------------------------------------------------------------- #
def _campus_frame(clean_df: pd.DataFrame, campus: int) -> pd.DataFrame:
    """Rows of ``clean_df`` for ``campus`` (no-op if already pre-filtered)."""
    mask = clean_df["UniversityID"].astype("int64") == int(campus)
    return clean_df[mask]


def _quarter_bdays(first: pd.Timestamp, last: pd.Timestamp) -> np.ndarray:
    """Business-day count per calendar quarter (Q1..Q4) over [first, last]."""
    out = np.zeros(4, dtype="float64")
    if pd.isna(first) or pd.isna(last) or last < first:
        return out
    bdays = pd.bdate_range(first.normalize(), last.normalize())
    if len(bdays) == 0:
        return out
    q = bdays.quarter.to_numpy()
    for k in range(1, 5):
        out[k - 1] = float((q == k).sum())
    return out


def fit_params(clean_df: pd.DataFrame, campus: int) -> dict:
    """Fit the per-campus generator parameter pack.

    ``clean_df`` may be the full multi-campus cleaned frame or one already
    filtered to ``campus`` (fast path for tests). The MISC merge and crew
    capacities are computed on ALL years of the campus (as in calib.py); the
    arrival rate, lognormal, pm_share and priority distributions use train-year
    rows only (WOStartDate <= 2017-12-31).
    """
    campus = int(campus)
    cdf = _campus_frame(clean_df, campus)

    # ---- MISC merge + capacities (all-year, reused from calib) ------------- #
    tmap = calib.trade_merge_map(cdf)
    trade_m = calib.apply_trade_merge(cdf, tmap)
    capacity = calib.build_capacity(cdf, trade_m)
    cap_dict = {str(r.trade): int(r.crew) for r in capacity.itertuples()}

    # ---- v2 priority mapping (all-year, reused from calib) ----------------- #
    mapping = calib.build_priority_mapping(cdf)
    priority = calib.priority_class_series(cdf, mapping)

    # ---- assemble a working frame ----------------------------------------- #
    work = pd.DataFrame({
        "trade_m": trade_m.to_numpy(),
        "hours": cdf["LaborHours"].to_numpy(dtype="float64"),
        "is_pm": cdf["is_pm"].fillna(False).astype(bool).to_numpy(),
        "priority": priority.to_numpy().astype("int64"),
        "start": cdf["WOStartDate"].to_numpy(),
    })
    work["start"] = pd.to_datetime(work["start"])

    # campus p99.5 labor cap (for clipping generated draws)
    labor_cap = float(np.quantile(work["hours"].to_numpy(), LABOR_CAP_Q)) \
        if len(work) else 1.0

    # ---- train subset ------------------------------------------------------ #
    train = work[work["start"] <= TRAIN_END].copy()
    n_train_total = int(len(train))

    # campus train span, in business hours (40 bh / week)
    if n_train_total > 0:
        t_first = pd.Timestamp(train["start"].min())
        t_last = pd.Timestamp(train["start"].max())
        span_bh = float(ta.abs_bh(t_last) - ta.abs_bh(t_first))
    else:
        t_first = t_last = pd.NaT
        span_bh = 0.0
    span_bh = max(span_bh, _WEEK_BH)  # guard tiny/degenerate spans

    q_bdays = _quarter_bdays(t_first, t_last)
    q_bdays_total = float(q_bdays.sum())

    # ---- per-trade parameters --------------------------------------------- #
    all_trades = sorted(cap_dict.keys())
    trades: dict[str, dict] = {}
    for tr in all_trades:
        sub = train[train["trade_m"] == tr]
        n = int(len(sub))
        arrival_rate = n / span_bh if span_bh > 0 else 0.0

        # lognormal of LaborHours (fit BEFORE any per-campus cap)
        hrs = sub["hours"].to_numpy(dtype="float64")
        hrs = hrs[hrs > 0]
        if len(hrs) >= 2:
            logs = np.log(hrs)
            mu = float(logs.mean())
            sigma = float(logs.std(ddof=0))
        elif len(hrs) == 1:
            mu, sigma = float(np.log(hrs[0])), 0.0
        else:
            mu, sigma = 0.0, 0.0
        sigma = max(sigma, 0.0)

        pm = sub["is_pm"].to_numpy()
        pm_share = float(pm.mean()) if n else 0.0

        # CM empirical priority-class distribution over 1..4
        cm = sub[~sub["is_pm"]]
        cm_dist = {str(c): 0.0 for c in (1, 2, 3, 4)}
        if len(cm) > 0:
            vc = cm["priority"].value_counts(normalize=True)
            for c in (1, 2, 3, 4):
                cm_dist[str(c)] = float(vc.get(c, 0.0))
            s = sum(cm_dist.values())
            if s > 0:
                cm_dist = {k: v / s for k, v in cm_dist.items()}
            else:
                cm_dist = {"1": 0.0, "2": 0.0, "3": 1.0, "4": 0.0}
        else:
            cm_dist = {"1": 0.0, "2": 0.0, "3": 1.0, "4": 0.0}  # R5d default

        # per-quarter arrival multipliers (seasonal option; default unused)
        q_mult = {f"Q{k}": 1.0 for k in (1, 2, 3, 4)}
        if n > 0 and q_bdays_total > 0:
            sq = sub["start"].dt.quarter.to_numpy()
            for k in (1, 2, 3, 4):
                cq = float((sq == k).sum())
                obs_share = cq / n
                exp_share = q_bdays[k - 1] / q_bdays_total
                q_mult[f"Q{k}"] = float(obs_share / exp_share) \
                    if exp_share > 0 else 0.0

        trades[tr] = {
            "arrival_rate_per_bh": arrival_rate,
            "logn_mu": mu,
            "logn_sigma": sigma,
            "pm_share": pm_share,
            "cm_priority_dist": cm_dist,
            "pm_priority_dist": {"4": 1.0},   # R5a: PM is always class 4
            "quarter_mult": q_mult,
            "n_train": n,
        }

    return {
        "campus": campus,
        "trades": trades,
        "capacity": cap_dict,
        "labor_cap": labor_cap,
        "train_span_bh": span_bh,
        "n_train_total": n_train_total,
        "fit": {
            "train_end": str(TRAIN_END.date()),
            "labor_cap_quantile": LABOR_CAP_Q,
            "week_bh": _WEEK_BH,
            "p_bh_floor": P_BH_FLOOR,
        },
    }


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #
def _technicians(cap_dict: dict[str, int], crew_multiplier: float):
    """(trades, technicians) with crew = max(1, round(crew * multiplier))."""
    trades = sorted(cap_dict.keys())
    techs: list[dict] = []
    tid = 0
    for tr in trades:
        n = max(1, int(round(cap_dict[tr] * crew_multiplier)))
        for _ in range(n):
            techs.append({"id": f"T{tid}", "trade": tr})
            tid += 1
    return trades, techs


def _draw_arrivals(rng, draw_trades, rates, size):
    """Superposed homogeneous Poisson arrivals; grow window until >= size.

    Returns (times, trade_idx) arrays for the ``size`` earliest arrivals,
    already sorted by time (stable). ``rates`` are per-trade rates aligned to
    ``draw_trades``.
    """
    total_rate = float(np.sum(rates))
    if total_rate <= 0.0:
        # Degenerate: no positive rate. Fall back to unit-rate uniform arrivals
        # spread over a size-length window so the instance is still well-formed.
        times = np.arange(size, dtype="float64") + 1.0
        idx = np.zeros(size, dtype="int64")
        return times, idx

    window = max(WINDOW_MIN_BH, size / total_rate * _OVERSHOOT)
    for _ in range(_MAX_GROW):
        all_times: list[np.ndarray] = []
        all_idx: list[np.ndarray] = []
        for j, rate in enumerate(rates):
            if rate <= 0.0:
                continue
            n = int(rng.poisson(rate * window))
            if n > 0:
                all_times.append(rng.uniform(0.0, window, size=n))
                all_idx.append(np.full(n, j, dtype="int64"))
        total = int(sum(len(a) for a in all_times))
        if total >= size:
            times = np.concatenate(all_times)
            idx = np.concatenate(all_idx)
            order = np.argsort(times, kind="stable")
            times = times[order][:size]
            idx = idx[order][:size]
            return times, idx
        window *= 2.0

    # Should not happen for realistic rates; return whatever we have, padded.
    if all_times:
        times = np.concatenate(all_times)
        idx = np.concatenate(all_idx)
    else:
        times = np.array([], dtype="float64")
        idx = np.array([], dtype="int64")
    order = np.argsort(times, kind="stable")
    times, idx = times[order], idx[order]
    if len(times) < size:  # pad deterministically
        pad = size - len(times)
        last = float(times[-1]) if len(times) else 0.0
        extra_t = last + np.arange(1, pad + 1, dtype="float64")
        times = np.concatenate([times, extra_t])
        idx = np.concatenate([idx, np.zeros(pad, dtype="int64")])
    return times[:size], idx[:size]


def generate(params: dict, size: int, seed: int, crew_multiplier: float = 1.0,
             pm_share_override: float | None = None,
             arrival_multiplier: float = 1.0) -> dict:
    """Sample one synthetic instance from a fitted parameter pack.

    Deterministic in (params, size, seed, crew_multiplier, pm_share_override,
    arrival_multiplier). Returns an instance dict in the interfaces.md schema
    with meta.track='generator', provenance 'C', window_start='synthetic'.
    """
    size = int(size)
    campus = int(params["campus"])
    cap_dict = {str(k): int(v) for k, v in params["capacity"].items()}
    labor_cap = float(params["labor_cap"])
    trade_params = params["trades"]

    rng = np.random.default_rng(int(seed))

    # trades eligible to receive arrivals (positive fitted rate), deterministic
    draw_trades = sorted(
        t for t, p in trade_params.items()
        if float(p.get("arrival_rate_per_bh", 0.0)) > 0.0
    )
    rates = np.array(
        [float(trade_params[t]["arrival_rate_per_bh"]) * float(arrival_multiplier)
         for t in draw_trades],
        dtype="float64",
    )

    times, idx = _draw_arrivals(rng, draw_trades, rates, size)
    releases = np.round(times, 4)

    # per-WO attributes, drawn in release order (deterministic)
    classes = np.array([1, 2, 3, 4])
    work_orders: list[dict] = []
    for k in range(size):
        tr = draw_trades[int(idx[k])] if len(draw_trades) else \
            (sorted(cap_dict)[0] if cap_dict else "MISC")
        tp = trade_params.get(tr, {})

        pm_p = float(pm_share_override) if pm_share_override is not None \
            else float(tp.get("pm_share", 0.0))
        is_pm = bool(rng.random() < pm_p)

        if is_pm:
            prio = 4  # R5a
        else:
            dist = tp.get("cm_priority_dist", {"3": 1.0})
            probs = np.array([float(dist.get(str(c), 0.0)) for c in (1, 2, 3, 4)],
                             dtype="float64")
            s = probs.sum()
            probs = probs / s if s > 0 else np.array([0.0, 0.0, 1.0, 0.0])
            prio = int(rng.choice(classes, p=probs))

        mu = float(tp.get("logn_mu", 0.0))
        sigma = float(tp.get("logn_sigma", 0.0))
        p_bh = float(np.exp(rng.normal(mu, sigma))) if sigma > 0 else float(np.exp(mu))
        p_bh = float(min(labor_cap, max(P_BH_FLOOR, p_bh)))

        rb = float(releases[k])
        work_orders.append({
            "id": f"W{k}",
            "trade": str(tr),
            "p_bh": round(p_bh, 4),
            "release_bh": round(rb, 4),
            "due_bh": round(rb + ta.SLA_BH[prio], 4),
            "priority": int(prio),
            "weight": ta.WEIGHT[prio],
            "building": None,
            "is_pm": is_pm,
        })

    window_bh = max(WINDOW_MIN_BH, float(releases[-1]) if size > 0 else WINDOW_MIN_BH)

    trades, technicians = _technicians(cap_dict, crew_multiplier)

    inst = {
        "meta": {
            "id": f"c{campus:02d}_gen_{size}_s{int(seed)}",
            "campus": campus,
            "track": "generator",
            "size_class": size,
            "window_start": "synthetic",
            "window_bh": round(float(window_bh), 4),
            "provenance": "C",
            "seed": int(seed),
            "crew_multiplier": float(crew_multiplier),
            "pm_share_override": (None if pm_share_override is None
                                  else float(pm_share_override)),
            "arrival_multiplier": float(arrival_multiplier),
        },
        "trades": list(trades),
        "technicians": list(technicians),
        "work_orders": work_orders,
    }
    return inst


# --------------------------------------------------------------------------- #
# Utilization (storm-v2 mapping)                                              #
# --------------------------------------------------------------------------- #
def _clipped_lognormal_mean(mu: float, sigma: float,
                            floor: float, cap: float) -> float:
    """Mean of ``clip(X, floor, cap)`` for ``X ~ Lognormal(mu, sigma)``.

    The generator clips every ``p_bh`` draw to ``[P_BH_FLOOR, labor_cap]``, so
    the mean processing time a trade actually contributes is this *clipped*
    lognormal mean, NOT the raw ``exp(mu + sigma^2/2)`` (the labor cap chops the
    heavy upper tail; the floor lifts a little mass at the bottom). Using the
    clipped mean is what keeps the utilization mapping self-consistent with the
    realized workload -- see :func:`base_utilization`. Exact closed form via the
    standard-normal cdf (``math.erf``; no scipy dependency)::

        E[clip] = floor*Phi(za) + m*(Phi(zb-s) - Phi(za-s)) + cap*(1-Phi(zb))

    with ``m = exp(mu+s^2/2)``, ``za=(ln floor-mu)/s``, ``zb=(ln cap-mu)/s``.
    """
    if not (cap > floor):
        return float(max(floor, min(cap, math.exp(mu))))
    if sigma <= 0.0:
        # degenerate lognormal collapses to the point exp(mu), then clipped
        return float(min(cap, max(floor, math.exp(mu))))

    def _phi(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    la, lb = math.log(floor), math.log(cap)
    za = (la - mu) / sigma
    zb = (lb - mu) / sigma
    m = math.exp(mu + 0.5 * sigma * sigma)
    body = m * (_phi(zb - sigma) - _phi(za - sigma))
    return float(floor * _phi(za) + body + cap * (1.0 - _phi(zb)))


def base_utilization(params: dict, crew_multiplier: float = 1.0) -> float:
    """Base utilization ``u0`` of a fitted pack at ``arrival_multiplier = 1``.

    ``u0 = (sum_trade arrival_rate_per_bh * mean_p_bh) / total_crew`` where
    ``mean_p_bh`` is the clipped-lognormal mean (:func:`_clipped_lognormal_mean`,
    floor ``P_BH_FLOOR``, cap ``labor_cap``) and ``total_crew`` is the technician
    count actually built for the instance (:func:`_technicians`, i.e.
    ``max(1, round(crew*crew_multiplier))`` per trade -- the SAME denominator as
    the realized-utilization column). Each technician supplies one business-hour
    of capacity per business hour, so ``u0`` is the long-run offered-load /
    capacity ratio; because the numerator is a per-bh work rate and the
    denominator a capacity, the window length cancels and
    ``E[realized_util] = arrival_multiplier * u0``. Hence drawing at
    ``arrival_multiplier = u_target / u0`` centers realized utilization on
    ``u_target``. Deterministic in the pack.
    """
    cap_dict = {str(k): int(v) for k, v in params["capacity"].items()}
    labor_cap = float(params["labor_cap"])
    trade_params = params["trades"]
    work_rate = 0.0
    for tp in trade_params.values():
        rate = float(tp.get("arrival_rate_per_bh", 0.0))
        if rate <= 0.0:
            continue
        mu = float(tp.get("logn_mu", 0.0))
        sigma = float(tp.get("logn_sigma", 0.0))
        work_rate += rate * _clipped_lognormal_mean(mu, sigma, P_BH_FLOOR,
                                                    labor_cap)
    _, techs = _technicians(cap_dict, crew_multiplier)
    total_crew = len(techs)
    return float(work_rate / total_crew) if total_crew > 0 else 0.0


def generate_window(params: dict, window_bh: float, seed: int,
                    crew_multiplier: float = 1.0,
                    arrival_multiplier: float = 1.0,
                    pm_share_override: float | None = None) -> dict:
    """Sample one synthetic instance over a FIXED business-hour window.

    Unlike :func:`generate` (which draws until a target *count* is reached and
    lets the window float), this draws the homogeneous Poisson superposition
    over the FIXED window ``[0, window_bh)`` and keeps EVERY arrival: the
    work-order count ``n`` is random and scales with ``arrival_multiplier`` (and
    the fitted per-trade rates). This is the storm-v2 / utilization-sweep track
    -- a rate-scaled workload against fixed capacity that produces sustained
    overload.

    Everything else matches :func:`generate`: ``p_bh`` is a per-trade lognormal
    draw clipped to ``[P_BH_FLOOR, labor_cap]``, priority follows the PM (R5a) /
    CM rules, ``due = release + SLA[priority]``, and crew is scaled by
    ``crew_multiplier``.

    Deterministic in (params, window_bh, seed, crew_multiplier,
    arrival_multiplier, pm_share_override). Guard: if a draw yields 0 arrivals
    over the whole window it is redrawn once with ``seed + 500000`` and
    ``meta['redrawn']`` is set True. Returns an instance dict in the
    interfaces.md schema with meta.track='storm2', provenance 'C',
    window_start='synthetic', and ``meta.window_bh`` = the fixed window (NOT the
    last release). ``meta.size_class`` = the realized ``n``. The ``id`` is a
    default the caller is expected to overwrite.
    """
    window_bh = float(window_bh)
    campus = int(params["campus"])
    cap_dict = {str(k): int(v) for k, v in params["capacity"].items()}
    labor_cap = float(params["labor_cap"])
    trade_params = params["trades"]
    base_seed = int(seed)

    # trades eligible to receive arrivals (positive fitted rate), deterministic
    draw_trades = sorted(
        t for t, p in trade_params.items()
        if float(p.get("arrival_rate_per_bh", 0.0)) > 0.0
    )
    rates = np.array(
        [float(trade_params[t]["arrival_rate_per_bh"]) * float(arrival_multiplier)
         for t in draw_trades],
        dtype="float64",
    )

    # ---- draw arrivals over the FIXED window (redraw ONCE if empty) -------- #
    # A single Generator produces the arrivals AND (below) the per-WO
    # attributes, so the whole instance is one deterministic stream. On the
    # (practically impossible for a realistic pack + 80 bh window) event of an
    # empty draw we restart the stream from seed+500000 and flag it.
    redrawn = False
    seed_used = base_seed
    times_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    rng = np.random.default_rng(seed_used)
    for _attempt in range(2):
        rng = np.random.default_rng(seed_used)
        times_parts, idx_parts = [], []
        for j, rate in enumerate(rates):
            if rate <= 0.0:
                continue
            n_j = int(rng.poisson(rate * window_bh))
            if n_j > 0:
                times_parts.append(rng.uniform(0.0, window_bh, size=n_j))
                idx_parts.append(np.full(n_j, j, dtype="int64"))
        if sum(len(a) for a in times_parts) > 0:
            break
        redrawn = True
        seed_used = base_seed + 500000

    if times_parts:
        times = np.concatenate(times_parts)
        idx = np.concatenate(idx_parts)
        order = np.argsort(times, kind="stable")
        times = times[order]
        idx = idx[order]
    else:
        times = np.array([], dtype="float64")
        idx = np.array([], dtype="int64")
    releases = np.round(times, 4)
    n = int(len(releases))

    classes = np.array([1, 2, 3, 4])
    work_orders: list[dict] = []
    for k in range(n):
        tr = draw_trades[int(idx[k])]
        tp = trade_params.get(tr, {})

        pm_p = float(pm_share_override) if pm_share_override is not None \
            else float(tp.get("pm_share", 0.0))
        is_pm = bool(rng.random() < pm_p)

        if is_pm:
            prio = 4  # R5a
        else:
            dist = tp.get("cm_priority_dist", {"3": 1.0})
            probs = np.array([float(dist.get(str(c), 0.0)) for c in (1, 2, 3, 4)],
                             dtype="float64")
            s = probs.sum()
            probs = probs / s if s > 0 else np.array([0.0, 0.0, 1.0, 0.0])
            prio = int(rng.choice(classes, p=probs))

        mu = float(tp.get("logn_mu", 0.0))
        sigma = float(tp.get("logn_sigma", 0.0))
        p_bh = float(np.exp(rng.normal(mu, sigma))) if sigma > 0 else float(np.exp(mu))
        p_bh = float(min(labor_cap, max(P_BH_FLOOR, p_bh)))

        rb = float(releases[k])
        work_orders.append({
            "id": f"W{k}",
            "trade": str(tr),
            "p_bh": round(p_bh, 4),
            "release_bh": round(rb, 4),
            "due_bh": round(rb + ta.SLA_BH[prio], 4),
            "priority": int(prio),
            "weight": ta.WEIGHT[prio],
            "building": None,
            "is_pm": is_pm,
        })

    trades, technicians = _technicians(cap_dict, crew_multiplier)

    inst = {
        "meta": {
            "id": f"c{campus:02d}_storm2_w{int(round(window_bh))}_s{base_seed}",
            "campus": campus,
            "track": "storm2",
            "size_class": n,
            "window_start": "synthetic",
            "window_bh": round(float(window_bh), 4),
            "provenance": "C",
            "seed": base_seed,
            "crew_multiplier": float(crew_multiplier),
            "pm_share_override": (None if pm_share_override is None
                                  else float(pm_share_override)),
            "arrival_multiplier": float(arrival_multiplier),
            "redrawn": bool(redrawn),
        },
        "trades": list(trades),
        "technicians": list(technicians),
        "work_orders": work_orders,
    }
    return inst
