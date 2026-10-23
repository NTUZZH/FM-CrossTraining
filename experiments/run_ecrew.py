#!/usr/bin/env python
"""E-CREW runner: hypothetical crew sizes for long work orders.

Re-scores the flexibility ladder (L0, CHAIN(1.0), FULL) on the 763 verdict
replay test instances when a long order (p_j > 8 bh) needs a crew of
c_j = min(c_max, ceil(p_j / 8), k(g_j)) technicians at once, its elapsed time
shrinks with the crew's combined rate, and its labor content is preserved
(env/crew_engine.py; specification in Supplemental Text S8). c_max = 1 is the released model and is
checked bitwise against results/tier1/results.csv before any other scenario
runs.

Grid: c_max in {1, 2, 3, 4} x structures L0 / CHAIN(1.0) / FULL x
m in {1.0, 0.8, 0.6} x eta in {1.0, 0.8} (L0 once per m at eta = 1.0, standing
for both eta values, as in tier1 and E-SUB) x the seven ranked rules plus
Random (seed 301). Learned policies are not run: they never saw a crew order.

Every schedule is scored by the independent crew validator
(env/validator_crew.py); at c_max = 1 it is also scored by the released
validator, whose TWT must equal the crew validator's bitwise. Any failing row
stops the sweep.

Outputs (results/ecrew/)
  results.csv      tier1 schema plus c_max and the crew columns (+ .parquet)
  crew_sizes.csv   per (instance, m, c_max): realized crew-size counts, the
                   long orders the k(g) cap reduced, crew labor hours
  summary.json     counts, regression result, wall time
  smoke/           two-instance check output, kept apart

Usage (CPU only; one thread per worker; cores 0-11):
  PYTHONPATH=.:vendor taskset -c 0-11 python experiments/run_ecrew.py --phase regress --workers 12
  PYTHONPATH=.:vendor taskset -c 0-11 python experiments/run_ecrew.py --phase smoke
  PYTHONPATH=.:vendor taskset -c 0-11 python experiments/run_ecrew.py --phase sweep --workers 12
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

import experiments.run_dynamic as rd                        # noqa: E402
from experiments.recalibrate import resolve_y1_root          # noqa: E402
from env.crew_engine import LONG_BH, CrewDispatchEnv, primary_counts  # noqa
from env.validator2 import validate as validate2            # noqa: E402
from env.validator_crew import validate as validate_crew    # noqa: E402
from methods.rules import get_selector                      # noqa: E402
from overlays.build import build_overlay, load_crews        # noqa: E402

OUT_DIR = ROOT / "results" / "ecrew"
SMOKE_DIR = OUT_DIR / "smoke"

Y1_ROOT = resolve_y1_root()
rd.Y1 = Y1_ROOT
rd.CAP = Y1_ROOT / "results/p1_calib/capacity.csv"
rd.INST_ROOT = Y1_ROOT / "data/processed/instances"

RULE_SEED = rd.RULE_SEED
ALL_RULES = rd.ALL_RULES
VERDICT_CAMPUSES = rd.VERDICT_CAMPUSES
INST_ROOT = rd.INST_ROOT
RELEASED_CAP = rd.CAP


def corpus_fingerprint():
    """Same digest as experiments/run_esub.py: calibration table plus the
    instance index, so a run against another corpus never reuses shards."""
    h = hashlib.sha256()
    for f in (RELEASED_CAP, INST_ROOT / "index.csv"):
        h.update(f.read_bytes() if f.exists() else b"missing")
    return h.hexdigest()[:8]


FINGERPRINT = corpus_fingerprint()
SHARDS = OUT_DIR / ("shards_" + FINGERPRINT)

CMAXES = [1, 2, 3, 4]
MS = [1.0, 0.8, 0.6]
ETAS = [1.0, 0.8]
STRUCTS = [("dedicated", None), ("chain", 1.0), ("full", None)]
SMOKE_INSTANCES = ["c05_replay_150_0100", "c05_replay_400_0100"]

EXTRA = ["n_long", "n_crew_orders", "mean_crew_size", "twt_short",
         "twt_long", "mean_wait_crew", "mean_wait_single", "held_tech_bh",
         "n_blocks"]
FIELDS = ["c_max"] + list(rd.FIELDS) + EXTRA
CREW_FIELDS = ["instance_id", "campus", "size", "m", "c_max", "n_orders",
               "n_long", "n_size1", "n_size2", "n_size3", "n_size4",
               "n_long_single", "n_long_capped_by_k", "hours_long",
               "hours_crew"]

_OVERLAYS: dict = {}
_CREWS: dict = {}


def overlay_for(campus, structure, phi, eta, m):
    key = (campus, structure, phi, eta, m)
    ov = _OVERLAYS.get(key)
    if ov is None:
        if campus not in _CREWS:
            _CREWS[campus] = load_crews(RELEASED_CAP, campus)
        ov = build_overlay(campus, _CREWS[campus], structure, phi, eta, m)
        _OVERLAYS[key] = ov
    return ov


def cells():
    out = []
    for (st, phi) in STRUCTS:
        for m in MS:
            if st == "dedicated":
                out.append((st, phi, 1.0, m))
            else:
                for eta in ETAS:
                    out.append((st, phi, eta, m))
    return out


def crew_row(inst, technicians, sizes, m, c_max, base):
    k = primary_counts(technicians)
    n = {1: 0, 2: 0, 3: 0, 4: 0}
    n_long = long_single = capped = 0
    hours_long = hours_crew = 0.0
    for w in inst["work_orders"]:
        c = sizes[w["id"]]
        n[c] = n.get(c, 0) + 1
        p = float(w["p_bh"])
        if p > LONG_BH:
            n_long += 1
            hours_long += p
            if c == 1:
                long_single += 1
            if min(c_max, int(math.ceil(p / LONG_BH))) > k.get(w["trade"], 0):
                capped += 1
        if c >= 2:
            hours_crew += p
    return {**base, "m": m, "c_max": c_max,
            "n_orders": len(inst["work_orders"]), "n_long": n_long,
            "n_size1": n[1], "n_size2": n[2], "n_size3": n[3],
            "n_size4": n[4], "n_long_single": long_single,
            "n_long_capped_by_k": capped, "hours_long": hours_long,
            "hours_crew": hours_crew}


def run_instance(cfg):
    t0 = time.perf_counter()
    iid = cfg["instance_id"]
    try:
        shard_dir = Path(cfg["shard_dir"])
        n_expect = len(cells()) * len(ALL_RULES)
        with open(cfg["path"]) as f:
            inst = json.load(f)
        base = {"instance_id": iid, "campus": cfg["campus"],
                "size": cfg["size"]}
        done_any = False
        for c_max in cfg["c_maxes"]:
            dst = shard_dir / ("%s__c%d.json" % (iid, c_max))
            if dst.exists():
                try:
                    with open(dst) as f:
                        old = json.load(f)
                    if len(old.get("rows", [])) == n_expect:
                        continue
                except Exception:   # noqa: BLE001
                    pass
            rows, crews, bad = [], [], []
            sizes_by_m = {}
            for (st, phi, eta, m) in cells():
                ov = overlay_for(cfg["campus"], st, phi, eta, m)
                env = CrewDispatchEnv(inst, ov, c_max=c_max)
                if m not in sizes_by_m:
                    sizes_by_m[m] = env.crew_of
                    crews.append(crew_row(inst, ov["technicians"],
                                          env.crew_of, m, c_max, base))
                elif env.crew_of != sizes_by_m[m]:
                    raise AssertionError("crew sizes differ across "
                                         "structures at m = %s" % m)
                for meth in ALL_RULES:
                    sched = env.run_selector(get_selector(meth), method=meth,
                                             seed=RULE_SEED)
                    res = validate_crew(inst, sched, ov, c_max=c_max)
                    ok = bool(res["feasible"])
                    mt = res["metrics"]
                    if c_max == 1:
                        r2 = validate2(inst, sched, ov)
                        if not (r2["feasible"]
                                and r2["metrics"]["WWT"] == mt["WWT"]):
                            ok = False
                            bad.append("%s|%s|%s|%s|released validator "
                                       "disagrees" % (st, eta, m, meth))
                    if not ok:
                        bad.append("%s|%s|%s|%s|%s" % (
                            st, eta, m, meth, res["violations"][:2]))
                    pp = mt["per_priority_breach_share"]
                    decisions = sched.get("decisions")
                    wall = sched.get("wall_seconds")
                    rows.append({
                        "c_max": c_max, **base, "track": "replay",
                        "structure": st, "phi": phi, "eta": eta, "m": m,
                        "u_target": None, "u_realized": None,
                        "method": meth, "seed": RULE_SEED,
                        "twt": mt["WWT"], "makespan": mt["makespan"],
                        "mean_flow": mt["mean_flow"],
                        "breach_share": mt["breach_share"],
                        "breach_p1": pp.get(1), "breach_p2": pp.get(2),
                        "breach_p3": pp.get(3), "breach_p4": pp.get(4),
                        "decisions": decisions,
                        "latency_ms_per_decision": (
                            1000.0 * wall / decisions
                            if decisions and wall is not None else None),
                        "replans": None,
                        "validator_ok": int(ok),
                        "runtime_s": wall,
                        "n_long": mt["n_long"],
                        "n_crew_orders": mt["n_crew_orders"],
                        "mean_crew_size": mt["mean_crew_size"],
                        "twt_short": mt["twt_short"],
                        "twt_long": mt["twt_long"],
                        "mean_wait_crew": mt["mean_wait_crew"],
                        "mean_wait_single": mt["mean_wait_single"],
                        "held_tech_bh": sched["held_tech_bh"],
                        "n_blocks": sched["n_blocks"]})
            if bad:
                return {"config_id": "%s c_max=%d" % (iid, c_max),
                        "ok": False, "error": "validation failed",
                        "traceback": "\n".join(bad[:10])}
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump({"instance_id": iid, "c_max": c_max, "rows": rows,
                           "crews": crews}, f)
            os.replace(tmp, dst)
            done_any = True
        return {"config_id": iid, "ok": True, "skipped": not done_any,
                "wall": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"config_id": iid, "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


def drive(configs, workers, label):
    t0 = time.time()
    done = 0
    if workers <= 1:
        results = (run_instance(c) for c in configs)
        pool = None
    else:
        pool = mp.get_context("fork").Pool(workers)
        results = pool.imap_unordered(run_instance, configs)
    for res in results:
        done += 1
        if not res.get("ok"):
            print("[STOP] %s: %s\n%s" % (res["config_id"], res.get("error"),
                                         res.get("traceback", "")),
                  flush=True)
            if pool is not None:
                pool.terminate()
            sys.exit("STOP: a row failed validation or raised in %s" % label)
        if done % 100 == 0 or done == len(configs):
            el = time.time() - t0
            print("  %s %d/%d %.0fs eta %.0fs" % (
                label, done, len(configs), el,
                el / done * (len(configs) - done)), flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    return time.time() - t0


# --------------------------------------------------------------------------- #
def merge(shard_dir, out_dir, c_maxes, verbose=True):
    rows, crews = [], []
    for p in sorted(Path(shard_dir).glob("*.json")):
        with open(p) as f:
            d = json.load(f)
        if d["c_max"] not in c_maxes:
            continue
        rows += d["rows"]
        crews += d["crews"]
    rows.sort(key=lambda r: (r["c_max"], r["campus"], r["size"], r["m"],
                             r["structure"], r["phi"] or 0, r["eta"],
                             r["instance_id"], r["method"]))
    crews.sort(key=lambda r: (r["c_max"], r["campus"], r["size"], r["m"],
                              r["instance_id"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    with open(out_dir / "crew_sizes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CREW_FIELDS)
        w.writeheader()
        for r in crews:
            w.writerow({k: r.get(k) for k in CREW_FIELDS})
    try:
        import pandas as pd
        pd.DataFrame(rows, columns=FIELDS).to_parquet(
            out_dir / "results.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        print("parquet not written: %s" % e)
    n_bad = sum(1 for r in rows if not r.get("validator_ok"))
    if verbose:
        print("merged %d rows (%d failing validation), %d crew rows -> %s"
              % (len(rows), n_bad, len(crews), out_dir), flush=True)
    return rows, crews, n_bad


def load_tier1():
    out = {}
    with open(ROOT / "results/tier1/results.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] in ALL_RULES:
                out[(r["instance_id"], r["structure"], r["phi"], r["eta"],
                     r["m"], r["method"])] = r
    return out


def regression_check(rows, verbose=True):
    """c_max = 1 rows against the released tier1 rows, TWT bitwise; the other
    metric columns are compared too and reported."""
    ref = load_tier1()
    stats = {"compared": 0, "compared_m06_m08": 0, "compared_m10": 0,
             "twt_mismatch": 0, "missing": 0, "other_metric_mismatch": 0}
    details = []
    for r in rows:
        if r["c_max"] != 1:
            continue
        key = (r["instance_id"], r["structure"],
               "" if r["phi"] is None else "%s" % r["phi"],
               "%s" % r["eta"], "%s" % r["m"], r["method"])
        t = ref.get(key)
        if t is None:
            stats["missing"] += 1
            if len(details) < 10:
                details.append(("MISSING", key))
            continue
        stats["compared"] += 1
        stats["compared_m10" if r["m"] == 1.0 else "compared_m06_m08"] += 1
        if float(r["twt"]) != float(t["twt"]):
            stats["twt_mismatch"] += 1
            if len(details) < 10:
                details.append(("TWT", key, r["twt"], t["twt"]))
        for c in ("makespan", "mean_flow", "breach_share", "breach_p1",
                  "breach_p2", "breach_p3", "breach_p4", "decisions"):
            a, b = r[c], t[c]
            if (a is None) != (b == ""):
                stats["other_metric_mismatch"] += 1
                break
            if a is not None and float(a) != float(b):
                stats["other_metric_mismatch"] += 1
                break
    n_expect_rows = 763 * len(cells()) * len(ALL_RULES)
    stats["expected"] = n_expect_rows
    ok = (stats["compared"] == n_expect_rows and stats["missing"] == 0
          and stats["twt_mismatch"] == 0)
    if verbose:
        print("[gate C] tier1 bitwise TWT: %d compared (%d at m in {0.6, "
              "0.8}, %d at m = 1.0), %d mismatch, %d missing; other metric "
              "columns differ on %d row(s)"
              % (stats["compared"], stats["compared_m06_m08"],
                 stats["compared_m10"], stats["twt_mismatch"],
                 stats["missing"], stats["other_metric_mismatch"]),
              flush=True)
        for d in details:
            print("   ", d)
    return ok, stats


# --------------------------------------------------------------------------- #
def configs_for(ids, c_maxes, shard_dir):
    rows_idx = rd._replay_rows(VERDICT_CAMPUSES)
    if ids is not None:
        rows_idx = [r for r in rows_idx if r["id"] in ids]
    # Large instances first so the pool's tail is short.
    rows_idx.sort(key=lambda r: (-int(r["size_class"]), int(r["campus"]),
                                 r["id"]))
    return [{"instance_id": r["id"], "path": str(INST_ROOT / r["path"]),
             "campus": int(r["campus"]), "size": int(r["size_class"]),
             "c_maxes": c_maxes, "shard_dir": str(shard_dir)}
            for r in rows_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["regress", "smoke", "sweep", "merge"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    print("e-crew phase=%s, Y1 root %s (corpus %s), cells=%d, rules=%d, "
          "workers=%d" % (args.phase, Y1_ROOT.name, FINGERPRINT, len(cells()),
                          len(ALL_RULES), args.workers), flush=True)

    summary_p = OUT_DIR / "summary.json"
    summary = json.load(open(summary_p)) if summary_p.exists() else {}

    if args.phase == "smoke":
        cfgs = configs_for(set(SMOKE_INSTANCES), CMAXES, SMOKE_DIR / "shards")
        el = drive(cfgs, 1, "smoke")
        rows, crews, n_bad = merge(SMOKE_DIR / "shards", SMOKE_DIR, CMAXES)
        if n_bad or len(rows) != len(cfgs) * len(CMAXES) * len(cells()) * \
                len(ALL_RULES):
            sys.exit("SMOKE FAILED")
        print("smoke PASS: %d rows, %d instances, %.1fs" % (len(rows),
                                                           len(cfgs), el))
        return

    if args.phase == "regress":
        cfgs = configs_for(None, [1], SHARDS)
        if len(cfgs) != 763:
            sys.exit("expected 763 verdict instances, found %d" % len(cfgs))
        el = drive(cfgs, args.workers, "c_max=1")
        rows, _cr, n_bad = merge(SHARDS, OUT_DIR, [1])
        ok, stats = regression_check(rows)
        summary.update({"regress": {"finished": time.strftime(
            "%Y-%m-%d %H:%M:%S"), "elapsed_s": el, "rows": len(rows),
            "failing_validation": n_bad, **stats, "pass": ok}})
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(summary_p, "w"), indent=2)
        if not ok or n_bad:
            sys.exit("GATE C FAILED: c_max = 1 does not reproduce tier1")
        print("gate C PASS")
        return

    if args.phase == "sweep":
        if not summary.get("regress", {}).get("pass"):
            sys.exit("gate C has not passed; run --phase regress first")
        cfgs = configs_for(None, [2, 3, 4], SHARDS)
        el = drive(cfgs, args.workers, "c_max=2..4")
        summary["sweep"] = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "elapsed_s": el}

    rows, crews, n_bad = merge(SHARDS, OUT_DIR, CMAXES)
    ok, stats = regression_check(rows, verbose=True)
    per_c = {c: sum(1 for r in rows if r["c_max"] == c) for c in CMAXES}
    summary.update({
        "n_instances": len({r["instance_id"] for r in rows}),
        "rows": len(rows), "rows_per_cmax": per_c,
        "failing_validation": n_bad, "crew_rows": len(crews),
        "y1_root": Y1_ROOT.name, "corpus_fingerprint": FINGERPRINT,
        "cells": [list(c) for c in cells()], "c_max": CMAXES,
        "rule_set": ALL_RULES, "rule_seed": RULE_SEED, "long_bh": LONG_BH,
        "regression_recheck": stats})
    json.dump(summary, open(summary_p, "w"), indent=2)
    if n_bad or not ok:
        sys.exit("STOP: failing rows or regression drift after merge")
    print("done", per_c)


if __name__ == "__main__":
    main()
