#!/usr/bin/env python
"""E5' sensitivity runner (protocol log 7, designated cells).

Designated family: replay TEST, verdict campuses {5,9,10,12}, size 150,
FIRST 30 instances per campus (sorted id). Base cell: (CHAIN(1.0),
eta = 0.8, m = 0.6). Variants (one axis at a time, everything else base):

  base       : the base cell itself (in-family reference for tau)
  eta075/090 : eta in {0.75, 0.9}
  sla050/150 : SLA windows scaled x0.5 / x1.5 (d' = r + 0.5/1.5 (d - r))
  crew075/125: crew scale composed with m (overlay m = 0.6*0.75 / 0.6*1.25)
  w4321      : weight vector (4,3,2,1) by priority class
  w27931     : weight vector (27,9,3,1)
  perm1/2/3  : chain permutation seeds 20260708/09/10
  tbrand/tbflex        : TB ablation on the base cell
  tbrand_full/tbflex_full : TB ablation on (m = 0.8, FULL, eta = 1.0)
  full_base  : (m = 0.8, FULL, eta = 1.0) default-TB reference for the TB
               ablation pair
  cap256     : policy-only candidate-cap ablation (added with --methods rl)

Methods: 7 ranked rules + random (seed 301) now; policies join via
--methods rl after training (incremental shards, run_dynamic idiom).
Output: results/e5/results.csv with a `variant` column.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
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
OUT_DIR = ROOT / "results/e5"

RULE_SEED = 301
ALL_RULES = RANKED_RULES + ["random"]
CAMPUSES = [5, 9, 10, 12]
SIZE = 150
N_PER_CAMPUS = 30
PERM_SEEDS = {"perm1": 20260708, "perm2": 20260709, "perm3": 20260710}

BASE = {"structure": "chain", "phi": 1.0, "eta": 0.8, "m": 0.6}
FULLC = {"structure": "full", "phi": None, "eta": 1.0, "m": 0.8}

FIELDS = ["variant", "instance_id", "campus", "size", "track", "structure",
          "phi", "eta", "m", "u_target", "u_realized", "method", "seed",
          "twt", "makespan", "mean_flow", "breach_share", "breach_p1",
          "breach_p2", "breach_p3", "breach_p4", "decisions",
          "latency_ms_per_decision", "replans", "validator_ok", "runtime_s"]

METHODS = ("rules",)
_RL_SPECS: list = []
_POLICIES: dict = {}


def variants():
    """variant -> (cell dict, instance transform, tb_mode, perm_seed,
                   k_orders)"""
    v = {}
    v["base"] = (dict(BASE), None, "default", None, 64)
    v["eta075"] = (dict(BASE, eta=0.75), None, "default", None, 64)
    v["eta090"] = (dict(BASE, eta=0.9), None, "default", None, 64)
    v["sla050"] = (dict(BASE), ("sla", 0.5), "default", None, 64)
    v["sla150"] = (dict(BASE), ("sla", 1.5), "default", None, 64)
    v["crew075"] = (dict(BASE, m=0.6 * 0.75), None, "default", None, 64)
    v["crew125"] = (dict(BASE, m=0.6 * 1.25), None, "default", None, 64)
    v["w4321"] = (dict(BASE), ("w", (4.0, 3.0, 2.0, 1.0)), "default", None, 64)
    v["w27931"] = (dict(BASE), ("w", (27.0, 9.0, 3.0, 1.0)), "default",
                   None, 64)
    for name, seed in PERM_SEEDS.items():
        v[name] = (dict(BASE), None, "default", seed, 64)
    v["tbrand"] = (dict(BASE), None, "random", None, 64)
    v["tbflex"] = (dict(BASE), None, "most_flexible", None, 64)
    v["full_base"] = (dict(FULLC), None, "default", None, 64)
    v["tbrand_full"] = (dict(FULLC), None, "random", None, 64)
    v["tbflex_full"] = (dict(FULLC), None, "most_flexible", None, 64)
    v["cap256"] = (dict(BASE), None, "default", None, 256)   # rl-only
    return v


def transform_instance(inst, tf):
    if tf is None:
        return inst
    kind, arg = tf
    out = copy.deepcopy(inst)
    if kind == "sla":
        for w in out["work_orders"]:
            w["due_bh"] = round(w["release_bh"]
                                + arg * (w["due_bh"] - w["release_bh"]), 4)
    elif kind == "w":
        for w in out["work_orders"]:
            pr = int(w["priority"])
            if 1 <= pr <= 4:
                w["weight"] = float(arg[pr - 1])
    out["meta"] = dict(out["meta"])
    out["meta"]["e5_transform"] = "%s:%s" % (kind, arg)
    return out


def family_instances():
    import csv
    by_campus = {}
    with open(INST_ROOT / "index.csv", newline="") as f:
        for r in csv.DictReader(f):
            if (r["track"] == "replay" and r["split"] == "test"
                    and int(r["campus"]) in CAMPUSES
                    and int(r["size_class"]) == SIZE):
                by_campus.setdefault(int(r["campus"]), []).append(r)
    rows = []
    for c in sorted(by_campus):
        rows += sorted(by_campus[c], key=lambda r: r["id"])[:N_PER_CAMPUS]
    return rows


def discover_rl():
    specs = []
    for arch in ("mlp", "attn"):
        for d in sorted(glob.glob(str(TRAIN_DIR / ("%s_seed*" % arch)))):
            mm = re.match(r".*seed(\d+)$", d)
            if mm and os.path.exists(os.path.join(d, "best.pt")):
                specs.append(("v2%s%s" % (arch, mm.group(1)), arch,
                              os.path.join(d, "best.pt"), int(mm.group(1))))
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


def build_configs():
    configs = []
    insts = family_instances()
    for vname, (cell, tf, tb, perm, kord) in variants().items():
        for r in insts:
            configs.append({
                "config_id": "%s__%s" % (r["id"], vname),
                "variant": vname, "instance_id": r["id"],
                "path": str(INST_ROOT / r["path"]),
                "campus": int(r["campus"]), "size": SIZE,
                "track": "replay", "cell": cell, "tf": tf, "tb": tb,
                "perm": perm, "k_orders": kord,
            })
    return configs


def _expected(cfg):
    exp = []
    if cfg["variant"] != "cap256" and "rules" in METHODS:
        exp += ALL_RULES
    if "rl" in METHODS:
        exp += [s[0] for s in _RL_SPECS]
    return exp


def _row(cfg, method, seed, sched, res):
    m = res["metrics"]
    pp = m["per_priority_breach_share"]
    decisions = sched.get("decisions")
    wall = sched.get("wall_seconds")
    mean_ms = (1000.0 * wall / decisions) if (decisions and wall) else None
    cell = cfg["cell"]
    return {"variant": cfg["variant"], "instance_id": cfg["instance_id"],
            "campus": cfg["campus"], "size": cfg["size"], "track": "replay",
            "structure": cell["structure"], "phi": cell["phi"],
            "eta": cell["eta"], "m": cell["m"], "u_target": None,
            "u_realized": None, "method": method, "seed": seed,
            "twt": m["WWT"], "makespan": m["makespan"],
            "mean_flow": m["mean_flow"], "breach_share": m["breach_share"],
            "breach_p1": pp.get(1), "breach_p2": pp.get(2),
            "breach_p3": pp.get(3), "breach_p4": pp.get(4),
            "decisions": decisions, "latency_ms_per_decision": mean_ms,
            "replans": None, "validator_ok": int(bool(res["feasible"])),
            "runtime_s": wall}


def run_config(cfg):
    try:
        dst = OUT_DIR / "shards" / (cfg["config_id"] + ".json")
        old_rows = {}
        if dst.exists():
            try:
                with open(dst) as f:
                    old_rows = json.load(f).get("rows", {}) or {}
            except Exception:
                pass
        expected = _expected(cfg)
        todo = [meth for meth in expected if meth not in old_rows]
        if not todo:
            return {"config_id": cfg["config_id"], "ok": True,
                    "skipped": True}

        with open(cfg["path"]) as f:
            inst = json.load(f)
        inst = transform_instance(inst, cfg["tf"])
        cell = cfg["cell"]
        ov = build_overlay(cfg["campus"], load_crews(CAP, cfg["campus"]),
                           cell["structure"], cell["phi"], cell["eta"],
                           cell["m"], perm_seed=cfg["perm"])
        out_rows = {}
        env = PairDispatchEnv(inst, ov, tb_mode=cfg["tb"],
                              k_orders=cfg["k_orders"])
        for meth in todo:
            if meth in ALL_RULES:
                sched = env.run_selector(get_selector(meth), method=meth,
                                         seed=RULE_SEED)
                res = validate2(inst, sched, ov)
                out_rows[meth] = _row(cfg, meth, RULE_SEED, sched, res)
            else:
                spec = next(s for s in _RL_SPECS if s[0] == meth)
                pol = _get_policy(spec[2])
                penv = PairDispatchEnv(inst, ov, tb_mode="default",
                                       k_orders=cfg["k_orders"])
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
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"config_id": cfg["config_id"], "rows": out_rows}, f)
        os.replace(tmp, dst)
        bad = [k for k, r in out_rows.items() if r.get("validator_ok") == 0]
        return {"config_id": cfg["config_id"], "ok": True, "infeasible": bad}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"config_id": cfg["config_id"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


def merge():
    import csv
    rows = []
    n_bad = 0
    shard_dir = OUT_DIR / "shards"
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
    rows.sort(key=lambda r: (r["variant"], r["campus"], r["instance_id"],
                             r["method"], r["seed"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    print("merged %d rows (%d infeasible) -> %s"
          % (len(rows), n_bad, OUT_DIR / "results.csv"))


def main():
    global METHODS, _RL_SPECS
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="rules")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    METHODS = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    if args.merge:
        merge()
        return
    if "rl" in METHODS:
        _RL_SPECS = discover_rl()
        if not _RL_SPECS:
            sys.exit("no RL checkpoints")
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
    print("e5: configs=%d pending=%d methods=%s"
          % (len(configs), len(pending), METHODS), flush=True)
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
                print("[INFEASIBLE] %s %s" % (res["config_id"],
                                              res["infeasible"]), flush=True)
            if done % 200 == 0 or done == len(pending):
                print("  %d/%d %.0fs (%d err)"
                      % (done, len(pending), time.time() - t0, errs),
                      flush=True)
    merge()


if __name__ == "__main__":
    main()
