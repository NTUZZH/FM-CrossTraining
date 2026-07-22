#!/usr/bin/env python
"""E11: best-found one-secondary wiring under the same membership budget.

Searches, per verdict campus, over trade-level secondary maps sigma
(every technician of trade t receives secondary sigma[t], so B = headcount
exactly as CHAIN(1.0)) for the map minimising mean fixed-EDD TWT on the
released TRAINING replay windows (window_start <= 2017-12-31, the released
curriculum's split) at m = 0.6, eta = 1.0. First-improvement hill climbing
with multi-restart (workload chain, pairs, two seeded random maps), eval
budget per campus fixed in advance. The search never touches a test
window; the best-found map is evaluated ONCE on the test set by
run_e7_topology --variants opt (kind "opt" reads the JSON written here).

Reported as best-found, not certified optimal.

Usage: PYTHONPATH=.:vendor python experiments/run_e11_optsigma.py
       [--budget 3000] [--workers 4]
"""
from __future__ import annotations

import argparse
import glob
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
Y1 = Path(os.environ.get("FMWOS_Y1_ROOT", ROOT.parent / "FM-Scheduling"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from env.engine import PairDispatchEnv                   # noqa: E402
from methods.rules import get_selector                   # noqa: E402
from overlays.build import chain_order, load_crews       # noqa: E402
from overlays import topology_overlays as rt             # noqa: E402

CAP = Y1 / "results/p1_calib/capacity.csv"
INST_ROOT = Y1 / "data/processed/instances"
OUT_DIR = ROOT / "results" / "e11_optsigma"
CAMPUSES = [5, 9, 10, 12]
SIZES = [150, 400]
TRAIN_ANCHOR_MAX = "2017-12-31"          # released curriculum split
M, ETA = 0.6, 1.0
RULE_SEED = 301
RAND_STARTS = (20261001, 20261002)


def train_instances(campus):
    files = []
    for s in SIZES:
        pat = str(INST_ROOT / ("c%02d" % campus) / "replay" / str(s)
                  / "*.json")
        for f in sorted(glob.glob(pat)):
            try:
                with open(f) as fh:
                    inst = json.load(fh)
            except Exception:
                continue
            if inst["meta"]["window_start"] <= TRAIN_ANCHOR_MAX:
                files.append(inst)
    return files


def objective(campus, crews, order, sigma, insts, cache):
    key = tuple(sorted(sigma.items()))
    if key in cache:
        return cache[key]
    ov = rt.build_sigma_variant(campus, crews, sigma, ETA, M, order=order,
                                struct_label="opt", variant="opt")
    assert ov["budget_B"] == ov["headcount"]
    tot = 0.0
    sel = get_selector("edd")
    for inst in insts:
        env = PairDispatchEnv(inst, ov)
        env.run_selector(sel, method="edd", seed=RULE_SEED)
        tot += env._realized
    val = tot / len(insts)
    cache[key] = val
    return val


def search_campus(campus, budget):
    t0 = time.time()
    crews = load_crews(CAP, campus)
    order = chain_order(crews)
    insts = train_instances(campus)
    if not insts:
        raise RuntimeError("no train windows for campus %d" % campus)
    cache: dict = {}
    starts = [("chain", {t: order[(i + 1) % len(order)]
                         for i, t in enumerate(order)}),
              ("pairs", rt.sigma_pairs(order))]
    for sd in RAND_STARTS:
        starts.append(("rand%d" % sd, rt.sigma_rand1(order, sd)))

    best_sigma, best_val, best_start = None, float("inf"), None
    for name, sigma in starts:
        sigma = dict(sigma)
        val = objective(campus, crews, order, sigma, insts, cache)
        improved = True
        while improved and len(cache) < budget:
            improved = False
            for t in order:
                if len(cache) >= budget:
                    break
                for cand in order:
                    if cand == t or cand == sigma.get(t):
                        continue
                    trial = dict(sigma)
                    trial[t] = cand
                    v = objective(campus, crews, order, trial, insts, cache)
                    if v < val - 1e-9:
                        sigma, val = trial, v
                        improved = True
                        break
        if val < best_val:
            best_sigma, best_val, best_start = sigma, val, name
        print("[c%02d] start=%-10s val=%.4f evals=%d"
              % (campus, name, val, len(cache)), flush=True)

    d = rt.topology_descriptors(best_sigma, order,
                                {c["trade"]: c["crew"] for c in crews})
    out = {"campus": campus, "sigma": best_sigma, "order": order,
           "obj_train_edd": best_val, "best_start": best_start,
           "n_evals": len(cache), "budget": budget,
           "n_train_instances": len(insts),
           "m": M, "eta": ETA, "objective": "fixed-EDD mean TWT",
           "descriptors": {k: v for k, v in d.items()
                           if k != "added_eligible_techs"},
           "wall_s": time.time() - t0}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / ("opt_sigma_c%02d.json" % campus), "w") as f:
        json.dump(out, f, indent=1)
    print("[c%02d] BEST start=%s train-EDD=%.4f evals=%d comps=%d "
          "coverage=%.2f (%.0fs)"
          % (campus, best_start, best_val, len(cache),
             d["weak_components"], d["coverage_share"], out["wall_s"]),
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            pool.starmap(search_campus,
                         [(c, args.budget) for c in CAMPUSES])
    else:
        for c in CAMPUSES:
            search_campus(c, args.budget)


if __name__ == "__main__":
    main()
