#!/usr/bin/env python
"""E-E instrumentation + large-cap runner.

Two parts, selected with --part:

  instrument  Rerun EDD + both policy pools (all 10+10 seeds) on the penalized
              FULL cells (eta=0.8, m in {0.6, 0.8}) AND the CHAIN(1.0) cells at
              the same eta/m (for contrast), at the RELEASED caps
              (k_orders=64, k_pairs=256), with engine instrumentation ON. The
              instrumentation is a read-only shadow (env.engine, OFF by
              default), so every produced schedule is byte-identical to tier1;
              the runner ASSERTS twt == released tier1 twt per (instance, cell,
              method, seed) and aborts on any mismatch. Output: adds the
              behavioural columns (secondary-assignment share, cap-bind
              frequencies, EDD-choice-excluded frequency).
              -> results/ee_instrument/

  bigcap      Large-candidate policy rerun on the penalized FULL cells only
              (eta=0.8, m in {0.6, 0.8}), both policy pools, all seeds, at TWO
              cap settings recorded in a `cap` column:
                std : k_orders=64,  k_pairs=256   (released caps; MUST
                      reproduce tier1 bitwise -- asserted)
                big : k_orders=256, k_pairs=2048  (the large-candidate rerun)
              The pair scorer is applied per pair (permutation-equivariant; no
              weight tied to the slot count), so the released checkpoints are
              weight-compatible with a larger pair tensor -- the `big` rows
              differ from `std` only where a cap bound.
              -> results/ee_bigcap/

Config parity with tier1: the ONLY intended differences from the released
tier1 dynamic run are (1) instrument=True, (2) for `bigcap.big` the two cap
values, (3) the output path. Same instances, same overlays, same seeds, same
validator. In-runner asserts enforce it.

Naming: results/ee_instrument/ and results/ee_bigcap/ are NOT globbed by any
analysis module (analysis.build_all._read and analysis.figures read families
by explicit name; only results/train/<arch>_seed* is globbed). Findable only
by a dedicated E-E reader.

Usage:
  PYTHONPATH=.:vendor OMP_NUM_THREADS=1 python experiments/run_ee_instrument.py \
      --part instrument --workers 22
  PYTHONPATH=.:vendor OMP_NUM_THREADS=1 python experiments/run_ee_instrument.py \
      --part bigcap --workers 22
  # smoke: 2 instances, EDD + one mlp + one attn seed, FULL eta0.8 m0.6
  ... --part instrument --smoke
  ... --part bigcap --smoke
  ... --merge          (rebuild results.csv from shards for the chosen --part)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
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

from env.engine import PairDispatchEnv, K_ORDERS_PER_TRADE, K_PAIRS  # noqa: E402
from env.validator2 import validate as validate2                     # noqa: E402
from methods.rules import get_selector                               # noqa: E402
from overlays.build import build_overlay, load_crews, overlay_id     # noqa: E402

CAP = Y1 / "results/p1_calib/capacity.csv"
INST_ROOT = Y1 / "data/processed/instances"
TRAIN_DIR = ROOT / "results/train"
TIER1_SHARDS = ROOT / "results/tier1/shards"

RULE_SEED = 301
VERDICT_CAMPUSES = [5, 9, 10, 12]
SIZES = [150, 400]

# Expected released parameter counts (guards against silent checkpoint drift;
# same values asserted for the E-C run).
PARAM_COUNTS = {"pair_mlp": 40834, "pair_attn": 75138}

# Cells. Penalized FULL target + CHAIN(1.0) contrast.
FULL_CELLS = [("full", None, 0.8, 0.6), ("full", None, 0.8, 0.8)]
CHAIN_CELLS = [("chain", 1.0, 0.8, 0.6), ("chain", 1.0, 0.8, 0.8)]

# Large-cap settings for the bigcap part.
CAP_SETTINGS = {"std": (K_ORDERS_PER_TRADE, K_PAIRS),   # 64 / 256 (released)
                "big": (256, 2048)}                     # large-candidate rerun

INSTR_KEYS = ["instr_n_assign", "instr_n_secondary", "instr_pbh_total",
              "instr_pbh_secondary", "instr_rph_total", "instr_rph_secondary",
              "instr_decisions", "instr_order_cap_binds",
              "instr_pair_cap_binds", "instr_edd_excluded"]

BASE_FIELDS = ["instance_id", "campus", "size", "track", "structure", "phi",
               "eta", "m", "method", "seed", "twt", "makespan", "mean_flow",
               "decisions", "validator_ok", "released_twt", "twt_matches",
               "runtime_s"]

_POLICIES: dict = {}


# --------------------------------------------------------------------------- #
def _replay_rows():
    rows = []
    with open(INST_ROOT / "index.csv", newline="") as f:
        for r in csv.DictReader(f):
            if (r["track"] == "replay" and r["split"] == "test"
                    and int(r["campus"]) in VERDICT_CAMPUSES
                    and int(r["size_class"]) in SIZES):
                rows.append(r)
    rows.sort(key=lambda r: (int(r["campus"]), int(r["size_class"]), r["id"]))
    return rows


def discover_policies():
    """(method, arch, path, seed) for the 10 mlp + 10 attn verdict pools."""
    specs = []
    for arch, pref in (("mlp", "v2mlp"), ("attn", "v2attn")):
        for d in sorted(glob.glob(str(TRAIN_DIR / ("%s_seed*" % arch)))):
            mm = re.match(r".*seed(\d+)$", d)
            if mm and os.path.exists(os.path.join(d, "best.pt")):
                specs.append(("%s%s" % (pref, mm.group(1)), arch,
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


def _released_twt(config_id, method):
    """Released tier1 twt for (config_id, method), or None if absent."""
    p = TIER1_SHARDS / (config_id + ".json")
    if not p.exists():
        return None
    try:
        with open(p) as f:
            rows = json.load(f).get("rows", {})
    except Exception:
        return None
    r = rows.get(method)
    return None if r is None else r.get("twt")


# --------------------------------------------------------------------------- #
def build_configs(part):
    cells = FULL_CELLS + CHAIN_CELLS if part == "instrument" else FULL_CELLS
    rows = _replay_rows()
    configs = []
    for r in rows:
        campus = int(r["campus"])
        for (st, phi, eta, m) in cells:
            tag = overlay_id(campus, st, phi, eta, m)
            configs.append({
                "config_id": "%s__%s" % (r["id"], tag),
                "instance_id": r["id"], "path": str(INST_ROOT / r["path"]),
                "campus": campus, "size": int(r["size_class"]),
                "structure": st, "phi": phi, "eta": eta, "m": m,
            })
    return configs


def _base_row(cfg, method, seed, sched, res, released, cap_orders, cap_pairs):
    m = res["metrics"]
    twt = m["WWT"]
    matches = None if released is None else int(twt == released)
    row = {
        "instance_id": cfg["instance_id"], "campus": cfg["campus"],
        "size": cfg["size"], "track": "replay", "structure": cfg["structure"],
        "phi": cfg["phi"], "eta": cfg["eta"], "m": cfg["m"],
        "method": method, "seed": seed, "twt": twt, "makespan": m["makespan"],
        "mean_flow": m["mean_flow"], "decisions": sched.get("decisions"),
        "validator_ok": int(bool(res["feasible"])),
        "released_twt": released, "twt_matches": matches,
        "runtime_s": sched.get("wall_seconds"),
    }
    for k in INSTR_KEYS:
        row[k] = sched.get(k)
    row["cap"] = None
    row["cap_k_orders"] = cap_orders
    row["cap_k_pairs"] = cap_pairs
    return row


def _run_policy(inst, ov, spec, k_orders, k_pairs):
    pol = _get_policy(spec[2])
    penv = PairDispatchEnv(inst, ov, tb_mode="default", instrument=True,
                           k_orders=k_orders, k_pairs=k_pairs)
    t1 = time.perf_counter()
    obs = penv.reset()
    done = penv._done
    while not done:
        a, _, _, _ = pol.act(obs, greedy=True, device="cpu")
        obs, _r, done, _i = penv.step(a)
    sched = penv.to_schedule(spec[0], seed=spec[3])
    sched["wall_seconds"] = time.perf_counter() - t1
    return sched


def run_instrument(cfg):
    """EDD + all policies, released caps, instrument ON, assert twt==tier1."""
    try:
        with open(cfg["path"]) as f:
            inst = json.load(f)
        ov = build_overlay(cfg["campus"], load_crews(CAP, cfg["campus"]),
                           cfg["structure"], cfg["phi"], cfg["eta"], cfg["m"])
        rows = {}
        mism = []
        # EDD (rule): candidate-cap columns are N/A (rules do not use the
        # capped candidate set), so decisions/cap fields stay 0.
        env = PairDispatchEnv(inst, ov, instrument=True)
        sched = env.run_selector(get_selector("edd"), method="edd",
                                 seed=RULE_SEED)
        res = validate2(inst, sched, ov)
        rel = _released_twt(cfg["config_id"], "edd")
        rows["edd"] = _base_row(cfg, "edd", RULE_SEED, sched, res, rel,
                                K_ORDERS_PER_TRADE, K_PAIRS)
        if rel is not None and res["metrics"]["WWT"] != rel:
            mism.append(("edd", rel, res["metrics"]["WWT"]))
        # Policies
        for spec in _SPECS:
            sched = _run_policy(inst, ov, spec, K_ORDERS_PER_TRADE, K_PAIRS)
            res = validate2(inst, sched, ov)
            rel = _released_twt(cfg["config_id"], spec[0])
            rows[spec[0]] = _base_row(cfg, spec[0], spec[3], sched, res, rel,
                                      K_ORDERS_PER_TRADE, K_PAIRS)
            if rel is not None and res["metrics"]["WWT"] != rel:
                mism.append((spec[0], rel, res["metrics"]["WWT"]))
        if mism:
            return {"config_id": cfg["config_id"], "ok": False,
                    "error": "TWT MISMATCH vs tier1: %r" % mism[:3]}
        _write_shard(OUT_DIR, cfg["config_id"], rows)
        bad = [k for k, r in rows.items() if r["validator_ok"] == 0]
        return {"config_id": cfg["config_id"], "ok": True, "infeasible": bad}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"config_id": cfg["config_id"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


def run_bigcap(cfg):
    """Policies at std (64/256) and big (256/2048); assert std==tier1."""
    try:
        with open(cfg["path"]) as f:
            inst = json.load(f)
        ov = build_overlay(cfg["campus"], load_crews(CAP, cfg["campus"]),
                           cfg["structure"], cfg["phi"], cfg["eta"], cfg["m"])
        rows = {}
        mism = []
        for spec in _SPECS:
            for capname, (ko, kp) in CAP_SETTINGS.items():
                sched = _run_policy(inst, ov, spec, ko, kp)
                res = validate2(inst, sched, ov)
                rel = (_released_twt(cfg["config_id"], spec[0])
                       if capname == "std" else None)
                row = _base_row(cfg, spec[0], spec[3], sched, res, rel, ko, kp)
                row["cap"] = capname
                rows["%s|%s" % (spec[0], capname)] = row
                if (capname == "std" and rel is not None
                        and res["metrics"]["WWT"] != rel):
                    mism.append((spec[0], rel, res["metrics"]["WWT"]))
        if mism:
            return {"config_id": cfg["config_id"], "ok": False,
                    "error": "std-cap TWT MISMATCH vs tier1: %r" % mism[:3]}
        _write_shard(OUT_DIR, cfg["config_id"], rows)
        bad = [k for k, r in rows.items() if r["validator_ok"] == 0]
        return {"config_id": cfg["config_id"], "ok": True, "infeasible": bad}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"config_id": cfg["config_id"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


def _expected_keys(part):
    """Row keys a complete shard must contain, for resume/skip."""
    if part == "instrument":
        return ["edd"] + [s[0] for s in _SPECS]
    return ["%s|%s" % (s[0], cap) for s in _SPECS for cap in CAP_SETTINGS]


def _shard_complete(out_dir, config_id, expected):
    p = out_dir / "shards" / (config_id + ".json")
    if not p.exists():
        return False
    try:
        with open(p) as f:
            have = set(json.load(f).get("rows", {}))
    except Exception:
        return False
    return set(expected) <= have


def _write_shard(out_dir, config_id, rows):
    dst = out_dir / "shards" / (config_id + ".json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"config_id": config_id, "rows": rows}, f)
    os.replace(tmp, dst)


def merge(out_dir, part):
    fields = list(BASE_FIELDS) + INSTR_KEYS
    if part == "bigcap":
        fields += ["cap", "cap_k_orders", "cap_k_pairs"]
    else:
        fields += ["cap_k_orders", "cap_k_pairs"]
    rows = []
    n_bad = 0
    shard_dir = out_dir / "shards"
    for p in sorted(shard_dir.glob("*.json")) if shard_dir.exists() else []:
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        for _k, r in sorted(d.get("rows", {}).items()):
            rows.append(r)
            if not r.get("validator_ok"):
                n_bad += 1
    rows.sort(key=lambda r: (r["campus"], r["size"], r["m"], r["structure"],
                             r["eta"], r["instance_id"], r["method"],
                             r.get("cap") or ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_mismatch = sum(1 for r in rows if r.get("twt_matches") == 0)
    print("merged %d rows (%d infeasible, %d twt-mismatch) -> %s"
          % (len(rows), n_bad, n_mismatch, out_dir / "results.csv"))
    return {"rows": len(rows), "infeasible": n_bad, "mismatch": n_mismatch}


# --------------------------------------------------------------------------- #
def assert_launch_invariants(configs):
    """One-variable-changed guards; abort in seconds on drift."""
    ids = {c["instance_id"] for c in configs}
    assert len(ids) == 763, ("expected 763 verdict replay instances, got %d"
                             % len(ids))
    assert len(_SPECS) == 20, ("expected 10 mlp + 10 attn checkpoints, got %d"
                               % len(_SPECS))
    import torch
    for spec in _SPECS:
        pol = _get_policy(spec[2])
        n = sum(p.numel() for p in pol.parameters())
        exp = PARAM_COUNTS[pol.ARCH]
        assert n == exp, ("param-count drift %s: %d != %d"
                          % (spec[0], n, exp))
        assert pol.k_pairs == K_PAIRS, ("checkpoint k_pairs %d != %d"
                                        % (pol.k_pairs, K_PAIRS))
    del torch
    print("launch invariants OK: 763 instances, 20 checkpoints, param counts "
          "%s, k_pairs=256" % PARAM_COUNTS)


OUT_DIR = None
_SPECS: list = []


def main():
    global OUT_DIR, _SPECS
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["instrument", "bigcap"],
                    required=True)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--smoke", action="store_true",
                    help="2 instances, FULL eta0.8 m0.6, few methods")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    OUT_DIR = ROOT / "results" / ("ee_instrument" if args.part == "instrument"
                                  else "ee_bigcap")
    if args.merge:
        merge(OUT_DIR, args.part)
        return

    _SPECS = discover_policies()
    configs = build_configs(args.part)

    if args.smoke:
        OUT_DIR = OUT_DIR.with_name(OUT_DIR.name + "_smoke")
        smoke_ids = ["c12_replay_400_0070", "c12_replay_400_0096"]
        configs = [c for c in configs if c["instance_id"] in smoke_ids
                   and c["structure"] == "full" and c["eta"] == 0.8
                   and c["m"] == 0.6]
        _SPECS = [s for s in _SPECS if s[0] in ("v2mlp301", "v2attn401")]
        print("SMOKE: %d configs, methods=%s"
              % (len(configs), ["edd"] * (args.part == "instrument")
                 + [s[0] for s in _SPECS]))
    else:
        assert_launch_invariants(configs)

    # Resume: skip configs whose shard is already complete (idempotent; lets
    # the sweep run in chunks and be re-invoked safely).
    if not args.smoke:
        exp = _expected_keys(args.part)
        n_all = len(configs)
        configs = [c for c in configs
                   if not _shard_complete(OUT_DIR, c["config_id"], exp)]
        print("resume: %d/%d configs already complete, %d pending"
              % (n_all - len(configs), n_all, len(configs)))

    if args.limit:
        configs = configs[:args.limit]

    if not configs:
        merge(OUT_DIR, args.part)
        return

    fn = run_instrument if args.part == "instrument" else run_bigcap
    print("part=%s configs=%d workers=%d out=%s"
          % (args.part, len(configs), args.workers, OUT_DIR), flush=True)

    t0 = time.time()
    done = errs = 0
    if args.workers <= 1 or len(configs) <= 2:
        results = (fn(c) for c in configs)
        for res in results:
            done += 1
            if not res.get("ok"):
                errs += 1
                print("[ERR] %s: %s" % (res["config_id"], res.get("error")),
                      flush=True)
                if res.get("traceback"):
                    print(res["traceback"], flush=True)
            elif res.get("infeasible"):
                print("[INFEASIBLE] %s %s"
                      % (res["config_id"], res["infeasible"]), flush=True)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            for res in pool.imap_unordered(fn, configs):
                done += 1
                if not res.get("ok"):
                    errs += 1
                    print("[ERR] %s: %s"
                          % (res["config_id"], res.get("error")), flush=True)
                elif res.get("infeasible"):
                    print("[INFEASIBLE] %s %s"
                          % (res["config_id"], res["infeasible"]), flush=True)
                if done % 200 == 0 or done == len(configs):
                    el = time.time() - t0
                    print("  %d/%d %.0fs eta %.0fs (%d err)"
                          % (done, len(configs), el,
                             el / done * (len(configs) - done), errs),
                          flush=True)
    print("ran %d configs, %d errors, %.1fs" % (done, errs, time.time() - t0))
    if errs == 0:
        merge(OUT_DIR, args.part)
    else:
        print("NOT merging: %d errors (asserts fired) -- inspect above" % errs)


if __name__ == "__main__":
    main()
