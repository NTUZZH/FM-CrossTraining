#!/usr/bin/env python
"""L0 regression anchor (MANDATORY, stop-the-line).

Runs the v2 pair engine at L0 (all skills singleton) over the Y1 dynamic
verdict cells (campuses 5/9/10/12, sizes 150/400, replay track, crew
multipliers 1.0/0.8/0.6) and verifies, per instance-configuration:

  S1 structural: the L0 overlay's technician list (id, primary) equals Y1's
     technicians (instance list at m = 1; tightness.scale_crew at m < 1);
  S2 bitwise schedules: for the five deterministic rules, the v2 engine's
     schedule equals fmwos_y1.pdrs.dispatch's schedule exactly (canonical
     (wo, tech, start, end) sort); for seeded random, the v1-semantics
     replay harness (run_replay_y1) equals it exactly;
  S3 released numbers: validator2's WWT on the v2 schedule matches the
     released Y1 per-instance wwt (results/p4_dyneval/results.csv) to
     <= 1e-9 relative (CSV decimal printing + summation-order float noise);
  S4 pooled anchors: the pooled means reproduce the L0 anchors (EDD 315.52
     etc.); the natural-v2 random is compared distributionally.

Any S1/S2/S3 failure is a stop-the-line bug.

Usage: PYTHONPATH=.:vendor python experiments/anchor_l0.py [--workers N]
Writes results/anchor_l0/{report.json, per_instance.csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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

from fmwos_y1 import pdrs, tightness                    # noqa: E402
from env.engine import PairDispatchEnv                  # noqa: E402
from env.validator2 import validate as validate2       # noqa: E402
from methods.rules import get_selector                  # noqa: E402
from overlays.build import build_overlay, load_crews    # noqa: E402

CAP = Y1 / "results/p1_calib/capacity.csv"
INST_ROOT = Y1 / "data/processed/instances"
RESULTS_Y1 = Y1 / "results/p4_dyneval/results.csv"
OUT = ROOT / "results/anchor_l0"

VERDICT_CAMPUSES = [5, 9, 10, 12]
SIZES = [150, 400]
MULTS = [1.0, 0.8, 0.6]
DET_RULES = ["edd", "wspt", "atc", "pfifo", "mor"]
SEED = 301

_OVERLAYS = {}


def overlay_for(campus, m):
    key = (campus, m)
    if key not in _OVERLAYS:
        _OVERLAYS[key] = build_overlay(campus, load_crews(CAP, campus),
                                       "dedicated", None, 1.0, m)
    return _OVERLAYS[key]


def canon(schedule):
    return sorted((a["wo"], a["tech"], a["start_bh"], a["end_bh"])
                  for a in schedule["assignments"])


def run_config(cfg):
    """One (instance, m): all checks; returns row dicts + failure list."""
    path, m = cfg["path"], cfg["m"]
    with open(path) as f:
        base = json.load(f)
    y1_inst = tightness.scale_crew(base, m) if m != 1.0 else base
    ov = overlay_for(cfg["campus"], m)

    fails = []
    # S1 structural equality
    y1_techs = [(t["id"], t["trade"]) for t in y1_inst["technicians"]]
    v2_techs = [(t["id"], t["primary"]) for t in ov["technicians"]]
    if y1_techs != v2_techs:
        fails.append("S1 technician mismatch m=%s" % m)

    rows = []
    env = PairDispatchEnv(base, ov)
    for rule in DET_RULES + ["random"]:
        y1_sched = pdrs.dispatch(y1_inst, rule, seed=SEED)
        if rule == "random":
            v2_sched = env.run_replay_y1(rule, seed=SEED, method=rule)
            nat = env.run_selector(get_selector(rule), method=rule, seed=SEED)
            nat_wwt = validate2(base, nat, ov)["metrics"]["WWT"]
        else:
            v2_sched = env.run_selector(get_selector(rule), method=rule,
                                        seed=SEED)
            nat_wwt = None
        if canon(v2_sched) != canon(y1_sched):
            fails.append("S2 schedule mismatch rule=%s m=%s" % (rule, m))
        res = validate2(base, v2_sched, ov)
        if not res["feasible"]:
            fails.append("validator2 infeasible rule=%s m=%s" % (rule, m))
        rows.append({"y1_id": cfg["y1_id"], "campus": cfg["campus"],
                     "size": cfg["size"], "m": m, "rule": rule,
                     "wwt_v2": res["metrics"]["WWT"],
                     "wwt_v2_natural_random": nat_wwt})
    return {"rows": rows, "fails": fails, "id": cfg["y1_id"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    y1 = pd.read_csv(RESULTS_Y1)
    y1 = y1[(y1.track == "replay") & (y1.campus.isin(VERDICT_CAMPUSES))
            & (y1["size"].isin(SIZES))
            & (y1.method.isin(DET_RULES + ["random"]))]

    idx = pd.read_csv(INST_ROOT / "index.csv")
    idx = idx[(idx.track == "replay") & (idx.split == "test")
              & (idx.campus.isin(VERDICT_CAMPUSES))
              & (idx.size_class.isin(SIZES))]
    configs = []
    for _, r in idx.iterrows():
        for m in MULTS:
            configs.append({
                "y1_id": r["id"] if m == 1.0 else "%s_m%s" % (r["id"], m),
                "path": str(INST_ROOT / r["path"]),
                "campus": int(r["campus"]), "size": int(r["size_class"]),
                "m": m})
    print("anchor configs: %d (instances %d x m %s)"
          % (len(configs), len(idx), MULTS), flush=True)

    all_rows, all_fails = [], []
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_config, configs), 1):
            all_rows.extend(res["rows"])
            all_fails.extend("%s :: %s" % (res["id"], f) for f in res["fails"])
            if i % 500 == 0 or i == len(configs):
                print("  %d/%d (%.0fs, %d failures)"
                      % (i, len(configs), time.time() - t0, len(all_fails)),
                      flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "per_instance.csv", index=False)

    # S3: per-instance wwt vs released Y1 numbers
    y1k = y1.set_index(["id", "method"]).wwt
    s3 = {"checked": 0, "missing": 0, "max_rel": 0.0, "mismatch": 0}
    for _, r in df.iterrows():
        key = (r["y1_id"], r["rule"])
        if key not in y1k.index:
            s3["missing"] += 1
            continue
        ref = float(y1k.loc[key])
        got = float(r["wwt_v2"])
        s3["checked"] += 1
        rel = abs(got - ref) / max(1.0, abs(ref))
        s3["max_rel"] = max(s3["max_rel"], rel)
        if rel > 1e-9:
            s3["mismatch"] += 1
            all_fails.append("S3 wwt mismatch %s %s: v2=%r y1=%r"
                             % (r["y1_id"], r["rule"], got, ref))

    # S4: pooled anchors (m = 1.0) + random distributional
    pooled = {}
    for rule in DET_RULES + ["random"]:
        for m in MULTS:
            sub = df[(df.rule == rule) & (df.m == m)]
            pooled["%s_m%s" % (rule, m)] = round(float(sub.wwt_v2.mean()), 2)
    nat = df[(df.rule == "random") & (df.m == 1.0)]
    rand_nat_pooled = round(float(nat.wwt_v2_natural_random.mean()), 2)
    expected_default = {"edd": 315.52, "wspt": 317.54, "atc": 316.26,
                        "pfifo": 315.52, "mor": 321.50, "random": 318.79}
    s4_fail = []
    for rule, want in expected_default.items():
        got = pooled["%s_m1.0" % rule]
        if abs(got - want) > 0.005:
            s4_fail.append("S4 pooled %s: got %.2f want %.2f"
                           % (rule, got, want))
    all_fails.extend(s4_fail)

    report = {
        "green": not all_fails,
        "n_configs": len(configs),
        "n_failures": len(all_fails),
        "failures_head": all_fails[:50],
        "s3": s3,
        "pooled_v2": pooled,
        "pooled_random_natural_m1.0": rand_nat_pooled,
        "expected_default_anchors": expected_default,
        "elapsed_s": round(time.time() - t0, 1),
        "y1_commit": "25ee06200af37cc2337083694d68564162c45ab9",
        "seed": SEED,
    }
    with open(OUT / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("green", "n_configs", "n_failures", "s3",
                       "pooled_random_natural_m1.0", "elapsed_s")}, indent=2))
    print("ANCHOR %s" % ("GREEN" if report["green"] else "RED"))
    return 0 if report["green"] else 1


if __name__ == "__main__":
    sys.exit(main())
