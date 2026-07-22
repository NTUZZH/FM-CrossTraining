#!/usr/bin/env python
"""E10 wait-action policy evaluation (protocol log Amendment A2).

Evaluates the 10 wait-action checkpoints (results/train/wait_mlp_seed7XX)
greedily on the Tier-1 replay verdict cells, through the SAME pair engine,
overlays, and validator as the released tier1 run; the only differences are
the policy class (wait token, f_pair 39), allow_wait=True at evaluation,
and the output directory results/e10_wait/ (NEVER merged into tier1).
Per-episode wait counts are recorded in a `waits` column.

In-runner guards:
  - every checkpoint must load with f_pair = F_TOTAL + 1 and carry exactly
    released-parameter-count + 128 parameters;
  - config identity vs tier1: same instance set, overlays, cells, engine
    defaults (asserted via the shared run_dynamic builders).

Usage (CPU only):
  PYTHONPATH=.:vendor python experiments/run_e10_wait.py [--workers 20]
      [--smoke] [--merge]
"""
from __future__ import annotations

import argparse
import csv
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

from env.engine import F_TOTAL, PairDispatchEnv          # noqa: E402
from env.validator2 import validate as validate2         # noqa: E402
from experiments import run_dynamic as RD                # noqa: E402

OUT_DIR = ROOT / "results" / "e10_wait"
TRAIN_DIR = ROOT / "results" / "train"
SEEDS = list(range(701, 711))

_POLICIES: dict = {}


def discover_wait():
    specs = []
    for s in SEEDS:
        p = TRAIN_DIR / ("wait_mlp_seed%d" % s) / "best.pt"
        if p.exists():
            specs.append(("v2wmlp%d" % s, str(p), s))
    return specs


def _get_policy(path):
    pol = _POLICIES.get(path)
    if pol is None:
        import torch
        torch.set_num_threads(1)
        from methods.policy2 import load_policy, make_policy
        pol = load_policy(path, map_location="cpu")
        assert pol.f_pair == F_TOTAL + 1, (
            "wait checkpoint %s has f_pair %d" % (path, pol.f_pair))
        base = sum(p.numel() for p in make_policy("mlp").parameters())
        got = sum(p.numel() for p in pol.parameters())
        assert got == base + pol.hidden, (
            "parameter drift in %s: %d != %d + %d"
            % (path, got, base, pol.hidden))
        pol.eval()
        _POLICIES[path] = pol
    return pol


def run_config(args):
    cfg, specs = args
    t0 = time.perf_counter()
    try:
        dst = OUT_DIR / "shards" / (cfg["config_id"] + ".json")
        old_rows = {}
        if dst.exists():
            try:
                with open(dst) as f:
                    old_rows = json.load(f).get("rows", {}) or {}
            except Exception:
                pass
        todo = [s for s in specs if s[0] not in old_rows]
        if not todo:
            return {"config_id": cfg["config_id"], "ok": True,
                    "skipped": True}

        with open(cfg["path"]) as f:
            inst = json.load(f)
        ov = RD.overlay_for(cfg["campus"], cfg["structure"], cfg["phi"],
                            cfg["eta"], cfg["m"])
        out_rows = {}
        for meth, path, seed in todo:
            pol = _get_policy(path)
            env = PairDispatchEnv(inst, ov, allow_wait=True)
            t1 = time.perf_counter()
            obs = env.reset()
            done = env._done
            while not done:
                a, _, _, _ = pol.act(obs, greedy=True, device="cpu")
                obs, _r, done, _i = env.step(a)
            sched = env.to_schedule(meth, seed=seed)
            sched["wall_seconds"] = time.perf_counter() - t1
            res = validate2(inst, sched, ov)
            r = RD._row(cfg, meth, seed, sched, res)
            r["waits"] = sched.get("waits", 0)
            out_rows[meth] = r

        out_rows = {**old_rows, **out_rows}
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"config_id": cfg["config_id"], "rows": out_rows}, f)
        os.replace(tmp, dst)
        bad = [k for k, r in out_rows.items() if not r.get("validator_ok")]
        return {"config_id": cfg["config_id"], "ok": True,
                "infeasible": bad, "wall": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"config_id": cfg["config_id"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


FIELDS = RD.FIELDS + ["waits"]


def merge():
    rows = []
    n_bad = 0
    shard_dir = OUT_DIR / "shards"
    for p in sorted(shard_dir.glob("*.json")) if shard_dir.exists() else []:
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        for _meth, r in sorted(d.get("rows", {}).items()):
            rows.append(r)
            if not r.get("validator_ok"):
                n_bad += 1
    rows.sort(key=lambda r: (r["campus"], r["size"], r["instance_id"],
                             r["structure"], str(r["phi"]), r["eta"],
                             r["m"], r["method"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(OUT_DIR / "results.parquet",
                                      index=False)
    except Exception:
        pass
    print("merged %d rows (%d infeasible) -> %s"
          % (len(rows), n_bad, OUT_DIR / "results.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--smoke", action="store_true",
                    help="2 campus-5 instances, first available checkpoint")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        merge()
        return

    specs = discover_wait()
    if not specs:
        sys.exit("no wait checkpoints under results/train/wait_mlp_seed7XX")
    configs = RD.build_configs("tier1")
    if args.smoke:
        ids = sorted({c["instance_id"] for c in configs
                      if c["campus"] == 5 and c["size"] == 150})[:1]
        ids += sorted({c["instance_id"] for c in configs
                       if c["campus"] == 5 and c["size"] == 400})[:1]
        configs = [c for c in configs if c["instance_id"] in ids]
        specs = specs[:1]

    pending = []
    for c in configs:
        dst = OUT_DIR / "shards" / (c["config_id"] + ".json")
        have = set()
        if dst.exists():
            try:
                with open(dst) as f:
                    have = set(json.load(f).get("rows", {}))
            except Exception:
                pass
        if not {s[0] for s in specs} <= have:
            pending.append((c, specs))

    print("e10_wait: checkpoints=%d configs=%d pending=%d workers=%d "
          "smoke=%s" % (len(specs), len(configs), len(pending),
                        args.workers, args.smoke), flush=True)
    if not pending:
        merge()
        return

    t0 = time.time()
    done = errs = 0
    if args.smoke or args.workers <= 1:
        results = (run_config(x) for x in pending)
        pool = None
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(args.workers)
        results = pool.imap_unordered(run_config, pending)
    for res in results:
        done += 1
        if not res.get("ok"):
            errs += 1
            print("[ERR] %s: %s\n%s" % (res["config_id"], res.get("error"),
                                        res.get("traceback", "")),
                  flush=True)
        elif res.get("infeasible"):
            print("[INFEASIBLE] %s %s" % (res["config_id"],
                                          res["infeasible"]), flush=True)
        if done % 500 == 0 or done == len(pending):
            print("  %d/%d %.0fs (%d err)"
                  % (done, len(pending), time.time() - t0, errs), flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    merge()


if __name__ == "__main__":
    main()
