#!/usr/bin/env python
"""E1' static-reference runner (protocol log 7).

Grid: verdict campuses x sizes {50,150,400} x tracks {replay-test,
generator-test}, FIRST 15 instances per (campus, size, track) sorted by id;
overlays {L0, CHAIN(1.0), FULL} x eta {1.0, 0.8} at m = 1.0 (5 effective
cells, L0 eta-invariant).

Per config (Y1 p2_e1 conventions):
  1. 7 ranked rules + random through the pair engine (dynamic non-delay
     references; also the warm-start source).
  2. cpsat60: methods.cpsat2, workers=2, warm-started from the best ranked
     rule schedule.
  3. cpsat300 ONLY if cpsat60 did not prove OPTIMAL, warm-started from the
     cpsat60 incumbent.
  4. GA (60 s, seed 301, pop 100).
  5. Policy greedy rollouts are added later via --methods rl (incremental
     shards), once training has finished.

Rows carry the dynamic FIELDS plus status/objective_bh/best_bound_bh/
proved_optimal for the solver methods (solver diagnostics live in the shard
and a solver.csv mirror).

Usage:
  PYTHONPATH=.:vendor python experiments/run_static.py \
      [--methods rules,cpsat,ga] [--workers 11] [--limit N] [--merge]
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

from env.engine import PairDispatchEnv                  # noqa: E402
from env.validator2 import validate as validate2       # noqa: E402
from methods.rules import get_selector, RANKED_RULES    # noqa: E402
from overlays.build import build_overlay, load_crews    # noqa: E402

CAP = Y1 / "results/p1_calib/capacity.csv"
INST_ROOT = Y1 / "data/processed/instances"
TRAIN_DIR = ROOT / "results/train"
OUT_DIR = ROOT / "results/e1_static"

RULE_SEED = 301
GA_SEED = 301
ALL_RULES = RANKED_RULES + ["random"]
CAMPUSES = [5, 9, 10, 12]
SIZES = [50, 150, 400]
N_PER_CELL = 15
CPSAT60_S = 60.0
CPSAT300_S = 300.0
CPSAT_WORKERS = 2

CELLS = [("dedicated", None, 1.0), ("chain", 1.0, 1.0), ("chain", 1.0, 0.8),
         ("full", None, 1.0), ("full", None, 0.8)]

FIELDS = ["instance_id", "campus", "size", "track", "structure", "phi",
          "eta", "m", "u_target", "u_realized", "method", "seed", "twt",
          "makespan", "mean_flow", "breach_share", "breach_p1", "breach_p2",
          "breach_p3", "breach_p4", "decisions", "latency_ms_per_decision",
          "replans", "validator_ok", "runtime_s",
          "status", "objective_bh", "best_bound_bh", "proved_optimal"]

_OVERLAYS: dict = {}
_POLICIES: dict = {}
_RL_SPECS: list = []
METHODS = ("rules", "cpsat", "ga")


def overlay_for(campus, structure, phi, eta):
    key = (campus, structure, phi, eta)
    ov = _OVERLAYS.get(key)
    if ov is None:
        ov = build_overlay(campus, load_crews(CAP, campus), structure, phi,
                           eta, 1.0)
        _OVERLAYS[key] = ov
    return ov


def build_configs():
    import csv
    by_cell = {}
    with open(INST_ROOT / "index.csv", newline="") as f:
        for r in csv.DictReader(f):
            if (r["split"] == "test" and int(r["campus"]) in CAMPUSES
                    and r["track"] in ("replay", "generator")
                    and int(r["size_class"]) in SIZES):
                key = (int(r["campus"]), int(r["size_class"]), r["track"])
                by_cell.setdefault(key, []).append(r)
    configs = []
    for key in sorted(by_cell):
        rows = sorted(by_cell[key], key=lambda r: r["id"])[:N_PER_CELL]
        for r in rows:
            for (st, phi, eta) in CELLS:
                from overlays.build import overlay_id
                configs.append({
                    "config_id": "%s__%s" % (
                        r["id"], overlay_id(int(r["campus"]), st, phi, eta,
                                            1.0)),
                    "instance_id": r["id"],
                    "path": str(INST_ROOT / r["path"]),
                    "campus": int(r["campus"]), "size": int(r["size_class"]),
                    "track": r["track"], "structure": st, "phi": phi,
                    "eta": eta, "m": 1.0, "u_target": None,
                    "u_realized": None,
                })
    return configs


def discover_rl():
    import re
    specs = []
    for arch in ("mlp", "attn"):
        for d in sorted(glob.glob(str(TRAIN_DIR / ("%s_seed*" % arch)))):
            m = re.match(r".*seed(\d+)$", d)
            if m and os.path.exists(os.path.join(d, "best.pt")):
                specs.append(("v2%s%s" % (arch, m.group(1)), arch,
                              os.path.join(d, "best.pt"), int(m.group(1))))
    return specs


def _get_policy(path):
    pol = _POLICIES.get(path)
    if pol is None:
        import torch
        torch.set_num_threads(1)
        from methods.policy2 import load_policy
        pol = load_policy(path, map_location="cpu")
        pol.eval()
        _POLICIES[path] = pol
    return pol


def _row(cfg, method, seed, sched, res, solver=None):
    m = res["metrics"]
    pp = m["per_priority_breach_share"]
    decisions = sched.get("decisions")
    wall = sched.get("wall_seconds")
    mean_ms = None
    if decisions and wall is not None and decisions > 0:
        mean_ms = 1000.0 * float(wall) / float(decisions)
    r = {
        "instance_id": cfg["instance_id"], "campus": cfg["campus"],
        "size": cfg["size"], "track": cfg["track"],
        "structure": cfg["structure"], "phi": cfg["phi"], "eta": cfg["eta"],
        "m": cfg["m"], "u_target": None, "u_realized": None,
        "method": method, "seed": seed,
        "twt": m["WWT"], "makespan": m["makespan"],
        "mean_flow": m["mean_flow"], "breach_share": m["breach_share"],
        "breach_p1": pp.get(1), "breach_p2": pp.get(2),
        "breach_p3": pp.get(3), "breach_p4": pp.get(4),
        "decisions": decisions, "latency_ms_per_decision": mean_ms,
        "replans": None, "validator_ok": int(bool(res["feasible"])),
        "runtime_s": wall,
        "status": None, "objective_bh": None, "best_bound_bh": None,
        "proved_optimal": None,
    }
    if solver:
        r["status"] = solver.get("status")
        r["objective_bh"] = solver.get("objective_bh")
        r["best_bound_bh"] = solver.get("best_bound_bh")
        r["proved_optimal"] = int(solver.get("status") == "OPTIMAL")
    return r


def _solver_row(cfg, method, sol, inst, ov):
    """CP-SAT row; a budget-exhausted NO-SOLUTION run is recorded with null
    metrics and null validator_ok (an RQ4 hardness data point), never as an
    infeasible schedule."""
    sched = dict(sol, method=method)
    if sol.get("assignments"):
        res = validate2(inst, sched, ov)
        return _row(cfg, method, 0, sched, res, solver=sol)
    r = {k: None for k in FIELDS}
    r.update({
        "instance_id": cfg["instance_id"], "campus": cfg["campus"],
        "size": cfg["size"], "track": cfg["track"],
        "structure": cfg["structure"], "phi": cfg["phi"], "eta": cfg["eta"],
        "m": cfg["m"], "method": method, "seed": 0,
        "decisions": sol.get("decisions"),
        "runtime_s": sol.get("wall_seconds"),
        "status": sol.get("status"), "objective_bh": None,
        "best_bound_bh": sol.get("best_bound_bh"), "proved_optimal": 0,
    })
    return r


def _expected(cfg):
    exp = []
    if "rules" in METHODS:
        exp += ALL_RULES
    if "cpsat" in METHODS:
        exp += ["cpsat60"]          # cpsat300 appears conditionally
    if "ga" in METHODS:
        exp += ["ga"]
    if "rl" in METHODS:
        exp += [s[0] for s in _RL_SPECS]
    return exp


def run_config(cfg):
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
        with open(cfg["path"]) as f:
            inst = json.load(f)
        ov = overlay_for(cfg["campus"], cfg["structure"], cfg["phi"],
                         cfg["eta"])
        expected = _expected(cfg)
        todo = [meth for meth in expected if meth not in old_rows]
        if not todo:
            return {"config_id": cfg["config_id"], "ok": True,
                    "skipped": True}

        out_rows = {}
        env = PairDispatchEnv(inst, ov)
        rule_scheds = {}
        for meth in ALL_RULES:
            if meth in old_rows or meth in todo:
                sched = env.run_selector(get_selector(meth), method=meth,
                                         seed=RULE_SEED)
                rule_scheds[meth] = sched
                if meth in todo:
                    res = validate2(inst, sched, ov)
                    out_rows[meth] = _row(cfg, meth, RULE_SEED, sched, res)

        if "cpsat60" in todo:
            from methods import cpsat2
            best_rule = min(
                (meth for meth in RANKED_RULES),
                key=lambda meth: validate2(inst, rule_scheds[meth],
                                           ov)["metrics"]["WWT"])
            c60 = cpsat2.solve(inst, ov, time_limit_s=CPSAT60_S,
                               workers=CPSAT_WORKERS,
                               warm_start=rule_scheds[best_rule])
            out_rows["cpsat60"] = _solver_row(cfg, "cpsat60", c60, inst, ov)
            if c60.get("status") != "OPTIMAL":
                # No solution within 60 s falls back to the rule warm start.
                warm = c60 if c60.get("assignments") \
                    else rule_scheds[best_rule]
                c300 = cpsat2.solve(inst, ov, time_limit_s=CPSAT300_S,
                                    workers=CPSAT_WORKERS, warm_start=warm)
                out_rows["cpsat300"] = _solver_row(cfg, "cpsat300", c300,
                                                   inst, ov)

        if "ga" in todo:
            from methods import ga2
            sched = ga2.solve_ga(inst, ov, budget_s=60.0, seed=GA_SEED,
                                 pop=100)
            res = validate2(inst, sched, ov)
            out_rows["ga"] = _row(cfg, "ga", GA_SEED, sched, res)

        for meth in todo:
            if meth.startswith("v2"):
                spec = next(s for s in _RL_SPECS if s[0] == meth)
                pol = _get_policy(spec[2])
                penv = PairDispatchEnv(inst, ov)
                t1 = time.perf_counter()
                obs = penv.reset()
                done = penv._done
                while not done:
                    a, _, _, _ = pol.act(obs, greedy=True, device="cpu")
                    obs, _r, done, _i = penv.step(a)
                sched = penv.to_schedule(meth, seed=spec[3])
                sched["wall_seconds"] = time.perf_counter() - t1
                res = validate2(inst, sched, ov)
                out_rows[meth] = _row(cfg, meth, spec[3], sched, res)

        out_rows = {**old_rows, **out_rows}
        shard = {"config_id": cfg["config_id"], "rows": out_rows,
                 "wall_seconds_total": time.perf_counter() - t0}
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(shard, f)
        os.replace(tmp, dst)
        bad = [k for k, r in out_rows.items() if r.get("validator_ok") == 0]
        return {"config_id": cfg["config_id"], "ok": True, "infeasible": bad,
                "wall": shard["wall_seconds_total"]}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"config_id": cfg["config_id"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


def merge(verbose=True):
    import csv
    rows = []
    shard_dir = OUT_DIR / "shards"
    n_bad = 0
    for p in sorted(shard_dir.glob("*.json")) if shard_dir.exists() else []:
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        for meth, r in sorted(d.get("rows", {}).items()):
            rows.append(r)
            if not r.get("validator_ok"):
                n_bad += 1
    rows.sort(key=lambda r: (r["campus"], r["size"], r["track"],
                             r["structure"], r["phi"] or 0, r["eta"],
                             r["instance_id"], r["method"]))
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
    if verbose:
        print("merged %d rows (%d infeasible) -> %s"
              % (len(rows), n_bad, OUT_DIR / "results.csv"))


def main():
    global METHODS, _RL_SPECS
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="rules,cpsat,ga")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    METHODS = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    if args.merge:
        merge()
        return
    if "rl" in METHODS:
        _RL_SPECS = discover_rl()
        if not _RL_SPECS:
            sys.exit("no RL checkpoints under %s" % TRAIN_DIR)

    configs = build_configs()
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
        if not set(_expected(c)) <= have:
            pending.append(c)
    if args.limit:
        pending = pending[:args.limit]
    print("e1_static: configs=%d pending=%d methods=%s workers=%d"
          % (len(configs), len(pending), METHODS, args.workers), flush=True)
    if not pending:
        merge()
        return

    t0 = time.time()
    done = errs = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for res in pool.imap_unordered(run_config, pending):
            done += 1
            if not res.get("ok"):
                errs += 1
                print("[ERR] %s: %s" % (res["config_id"], res.get("error")),
                      flush=True)
            elif res.get("infeasible"):
                print("[INFEASIBLE] %s: %s"
                      % (res["config_id"], res["infeasible"]), flush=True)
            if done % 100 == 0 or done == len(pending):
                el = time.time() - t0
                print("  %d/%d  %.0fs  eta %.0fs  (%d err)"
                      % (done, len(pending), el,
                         el / done * (len(pending) - done), errs),
                      flush=True)
    merge()
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump({"methods": list(METHODS), "elapsed_s": time.time() - t0,
                   "n_configs": len(configs), "n_run": done,
                   "n_errors": errs,
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  indent=2)


if __name__ == "__main__":
    main()
