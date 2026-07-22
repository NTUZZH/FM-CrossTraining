#!/usr/bin/env python
"""E-C: widen the Gate C evaluable base with crew multiplier m=0.7.

Post-hoc robustness ONLY. The pre-registered Gate C verdict (rho >= 0.70,
observed 0.870 at m=0.6, eta=1.0) stands as registered. These cells land in
results/e8_m07/ and MUST NOT be merged into results/tier1/. analysis.gates /
analysis.build_all read tier1 by explicit name and never glob results/*, so
e8_m07 cannot be swept into the registered gate computation.

One variable changes versus tier1: the crew multiplier m = 0.7 (tier1 ran
{0.6, 0.8, 1.0}). Everything else is the released tier1 dynamic-eval code
path, reused verbatim from experiments/run_dynamic.py: same instances, same
overlay construction (overlays.build.build_overlay), same pair engine, same
independent validator2 checker, same RULE_SEED, same greedy policy rollouts.

Methods per cell mirror the tier1 full-cell pool that analysis.gates.gate_c
draws from: the 7 ranked rules + Random (seed 301, deterministic, one run
each) and BOTH policy pools by INFERENCE ONLY with the released checkpoints
(pair-MLP seeds 301-310, pair-attention seeds 401-410; greedy rollouts).
No training. No specialist checkpoints (ablation pool, never the verdict pool).

Cells per instance (tier1 L0 eta-reuse convention):
  dedicated (L0)  eta = 1.0, once           <- eta-invariant
  chain(phi=1.0)  eta in {1.0, 0.8}
  full            eta in {1.0, 0.8}
= 5 cells per instance.

In-runner asserts (drift dies in seconds, not hours):
  A1  every overlay's per-trade crew headcount equals the released
      max(1, round(0.7 * c_k)) convention AND sits between the released
      m=0.6 and m=0.8 crews for that trade.
  A2  every policy checkpoint loads with k_pairs = 256 and the released
      parameter count (pair_mlp 40834, pair_attn 75138).
  A3  the instance-id set equals the tier1 instance-id set exactly.
  A4  dedicated (L0) cells are eta-invariant: emitted once at eta = 1.0,
      exactly as tier1 (inherited from run_dynamic.cells_for).

Usage:
  PYTHONPATH=.:vendor python experiments/run_ec_m07.py --smoke [--merge]
  PYTHONPATH=.:vendor python experiments/run_ec_m07.py --methods rules,rl \
      --workers 22 [--campus 5] [--merge]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

# Reuse the released tier1 runner verbatim; we only reconfigure it.
import experiments.run_dynamic as rd            # noqa: E402
from overlays.build import build_overlay, save_overlay, scaled_crews  # noqa: E402

M07 = 0.7
OVERLAY_ROOT = ROOT / "overlays" / "generated_overlays" / "m07"
OUT_DIR = ROOT / "results" / "e8_m07"

# Released architecture invariants (loaded-checkpoint reference; all released
# seeds agree, and no training summary records the count, so the checkpoints
# themselves are the reference the runner pins to).
PARAM_REF = {"pair_mlp": 40834, "pair_attn": 75138}

# Verdict-pool seeds only (exclude specialist ablation checkpoints).
MLP_SEEDS = list(range(301, 311))
ATTN_SEEDS = list(range(401, 411))

# Pre-scaled released crew tables per campus, for the A1 monotonicity check.
_CREWS_BY_CAMPUS: dict = {}
_SCALED_CACHE: dict = {}


def _scaled(campus):
    if campus not in _SCALED_CACHE:
        crews = _CREWS_BY_CAMPUS.get(campus)
        if crews is None:
            crews = rd.load_crews(rd.CAP, campus)
            _CREWS_BY_CAMPUS[campus] = crews
        _SCALED_CACHE[campus] = {
            0.6: scaled_crews(crews, 0.6),
            0.7: scaled_crews(crews, 0.7),
            0.8: scaled_crews(crews, 0.8),
        }
    return _SCALED_CACHE[campus]


# --------------------------------------------------------------------------- #
# Patched overlay builder: build via the released code path, assert, save.
# rd.run_config resolves overlay_for / _get_policy as module globals, so
# patching them on rd (before fork) makes every worker use the checked path.
# --------------------------------------------------------------------------- #
def _assert_a1(ov, campus):
    """A1: overlay headcount per trade == released max(1,round(0.7*c)) and
    sits in [m0.6, m0.8]."""
    from collections import Counter
    by_trade = Counter(t["primary"] for t in ov["technicians"])
    s = _scaled(campus)
    for trade, cnt in by_trade.items():
        assert cnt == s[0.7][trade], (
            "E-C A1: overlay %s trade %s headcount %d != scaled %d"
            % (ov["overlay_id"], trade, cnt, s[0.7][trade]))
        assert s[0.6][trade] <= cnt <= s[0.8][trade], (
            "E-C A1: overlay %s trade %s %d not in [%d,%d]"
            % (ov["overlay_id"], trade, cnt, s[0.6][trade], s[0.8][trade]))


def overlay_for(campus, structure, phi, eta, m):
    assert abs(m - M07) < 1e-12, "E-C runs only m=0.7 (got %r)" % m
    key = (campus, structure, phi, eta, m)
    ov = rd._OVERLAYS.get(key)
    if ov is None:
        from overlays.build import overlay_id
        disk = OVERLAY_ROOT / (overlay_id(campus, structure, phi, eta, m)
                               + ".json")
        if disk.exists():
            # Already materialised (atomic os.replace guarantees complete);
            # load it, do not re-save. Removes the multi-worker save race.
            with open(disk) as f:
                ov = json.load(f)
            _assert_a1(ov, campus)
        else:
            crews = _CREWS_BY_CAMPUS.get(campus)
            if crews is None:
                crews = rd.load_crews(rd.CAP, campus)
                _CREWS_BY_CAMPUS[campus] = crews
            ov = build_overlay(campus, crews, structure, phi, eta, m)
            _assert_a1(ov, campus)
            save_overlay(ov, OVERLAY_ROOT)
        rd._OVERLAYS[key] = ov
    return ov


def _get_policy(path):
    pol = rd._POLICIES.get(path)
    if pol is None:
        import torch
        torch.set_num_threads(1)
        from methods.policy2 import load_policy
        pol = load_policy(path, map_location="cpu")
        pol.eval()
        # A2: architecture invariants.
        arch = getattr(pol, "ARCH", "pair_mlp")
        assert pol.k_pairs == 256, "E-C A2: %s k_pairs=%r" % (path, pol.k_pairs)
        n = sum(p.numel() for p in pol.parameters())
        assert n == PARAM_REF[arch], (
            "E-C A2: %s param count %d != released %d"
            % (path, n, PARAM_REF[arch]))
        rd._POLICIES[path] = pol
    return pol


# --------------------------------------------------------------------------- #
def discover_verdict_rl():
    specs = [s for s in rd.discover_rl()
             if (s[0].startswith("v2mlp") and int(s[0][5:]) in MLP_SEEDS)
             or (s[0].startswith("v2attn") and int(s[0][6:]) in ATTN_SEEDS)]
    return specs


def build_m07_configs():
    """Reuse rd.build_configs('tier1') with the m grid reduced to {0.7}."""
    saved = rd.TIER1_MS
    rd.TIER1_MS = [M07]
    try:
        cfgs = rd.build_configs("tier1")
    finally:
        rd.TIER1_MS = saved
    return cfgs


def _tier1_instance_ids():
    import pandas as pd
    p = ROOT / "results" / "tier1" / "results.parquet"
    if p.exists():
        return set(pd.read_parquet(p, columns=["instance_id"]).instance_id)
    return set(r["id"] for r in rd._replay_rows(rd.VERDICT_CAMPUSES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="rules,rl",
                    help="comma subset of rules,rl")
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--campus", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    # Wire rd to our output dir, patched builders, and method set.
    rd.OUT_DIR = OUT_DIR
    rd.overlay_for = overlay_for
    rd._get_policy = _get_policy
    rd._SPEC_SPECS = []                       # never the specialist pool

    if args.merge:
        rd.merge()
        return

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    rd.METHODS = methods
    if "rl" in methods:
        rd._RL_SPECS = discover_verdict_rl()
        assert len(rd._RL_SPECS) == 20, (
            "E-C: expected 20 verdict checkpoints, found %d"
            % len(rd._RL_SPECS))

    # A3: instance-id set equals tier1's.
    configs = build_m07_configs()
    cfg_iids = set(c["instance_id"] for c in configs)
    tier1_iids = _tier1_instance_ids()
    assert cfg_iids == tier1_iids, (
        "E-C A3: instance-id set differs from tier1 (%d vs %d; sym-diff %d)"
        % (len(cfg_iids), len(tier1_iids),
           len(cfg_iids ^ tier1_iids)))

    if args.campus:
        keep = {int(c) for c in args.campus.split(",")}
        configs = [c for c in configs if c["campus"] in keep]

    if args.smoke:
        # 2 campus-5 instances (one 150, one 400) x {L0, CHAIN(1.0), FULL},
        # eta = 1.0 only; rules + Random + ONE policy seed (v2mlp301).
        c5 = [c for c in configs if c["campus"] == 5]
        i150 = sorted({c["instance_id"] for c in c5 if c["size"] == 150})[0]
        i400 = sorted({c["instance_id"] for c in c5 if c["size"] == 400})[0]
        keep_iids = {i150, i400}
        configs = [c for c in configs
                   if c["instance_id"] in keep_iids and c["eta"] == 1.0]
        if "rl" in methods:
            rd._RL_SPECS = [s for s in rd._RL_SPECS if s[0] == "v2mlp301"]
        print("SMOKE: instances %s ; %d cells ; rl_specs=%s"
              % (sorted(keep_iids), len(configs),
                 [s[0] for s in rd._RL_SPECS]))

    # A4 is structural: rd.cells_for emits dedicated only at eta=1.0. Verify.
    for c in configs:
        if c["structure"] == "dedicated":
            assert c["eta"] == 1.0, "E-C A4: dedicated cell with eta!=1.0"

    if args.limit:
        configs = configs[:args.limit]

    pending = []
    for c in configs:
        dst = rd._shard_path(c)
        have = set()
        if dst.exists():
            try:
                with open(dst) as f:
                    have = set(json.load(f).get("rows", {}))
            except Exception:
                pass
        if not set(rd._expected(c)) <= have:
            pending.append(c)

    print("E-C m=0.7  methods=%s  configs=%d  pending=%d  workers=%d  out=%s"
          % (methods, len(configs), len(pending), args.workers, OUT_DIR),
          flush=True)
    if not pending:
        rd.merge()
        return

    t0 = time.time()
    done = errs = 0
    workers = 1 if args.smoke else args.workers
    if workers <= 1:
        for c in pending:
            res = rd.run_config(c)
            done += 1
            if not res.get("ok"):
                errs += 1
                print("[ERR] %s: %s" % (res["config_id"], res.get("error")),
                      flush=True)
            elif res.get("infeasible"):
                print("[INFEASIBLE] %s: %s"
                      % (res["config_id"], res["infeasible"]), flush=True)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            for res in pool.imap_unordered(rd.run_config, pending):
                done += 1
                if not res.get("ok"):
                    errs += 1
                    print("[ERR] %s: %s"
                          % (res["config_id"], res.get("error")), flush=True)
                elif res.get("infeasible"):
                    print("[INFEASIBLE] %s: %s"
                          % (res["config_id"], res["infeasible"]), flush=True)
                if done % 200 == 0 or done == len(pending):
                    el = time.time() - t0
                    print("  %d/%d  %.0fs  eta %.0fs  (%d err)"
                          % (done, len(pending), el,
                             el / done * (len(pending) - done), errs),
                          flush=True)
    rd.merge()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump({"family": "e8_m07", "m": M07, "methods": list(methods),
                   "note": "post-hoc robustness; NOT merged into tier1",
                   "elapsed_s": time.time() - t0, "n_configs": len(configs),
                   "n_run": done, "n_errors": errs,
                   "rl_specs": [s[0] for s in rd._RL_SPECS],
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print("E-C done: %d run, %d errors -> %s" % (done, errs, OUT_DIR))


if __name__ == "__main__":
    main()
