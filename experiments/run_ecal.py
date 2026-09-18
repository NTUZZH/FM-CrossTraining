#!/usr/bin/env python
"""E-CAL roster-calibration sensitivity runner.

The released roster comes from the 95th percentile of weekly labor hours on
the Y1 train years. This runner rebuilds the flexibility ladder on alternative
crew tables (``experiments/recalibrate.py``) and scores it with the SAME v2
pair engine, validator and ranked rules used for tier1, so every row is
directly comparable to the released cells.

Calibrations (``calibration`` column): the quantile variants q in {0.90, 0.95,
0.975, 0.99} on the train window, and q = 0.95 on all recorded years. The
q = 0.95 / train label is the released roster; it is re-run rather than reused
because reproducing the released TWT bitwise is the regression check on the
whole path.

Grid per calibration: the 763 verdict-campus replay test instances (campuses
5, 9, 10, 12; sizes 150 and 400), structures L0 / CHAIN(1.0) / FULL, crew
multiplier m in {1.0, 0.8, 0.6}, eta in {1.0, 0.8}, seven ranked rules plus
Random at seed 301. L0 is eta-invariant and is run once per m at eta = 1.0,
the released convention.

This runner touches no released file. It reuses ``experiments.run_dynamic``
for read-only helpers (instance selection, constants, FIELDS, rule set) so the
instance set is identical to the released sweep, and
``overlays.build.build_overlay`` for the ladder, so only the crew column
differs from the released overlays.

Output: results/ecal/results.csv (+ .parquet), tier1 schema plus a
``calibration`` column, and results/ecal/summary.json.

Usage (CPU only; one thread per worker):
  PYTHONPATH=.:vendor taskset -c 0-9 python experiments/run_ecal.py --smoke
  PYTHONPATH=.:vendor taskset -c 0-9 python experiments/run_ecal.py --workers 10
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

import experiments.run_dynamic as rd                        # noqa: E402
from experiments.recalibrate import resolve_y1_root          # noqa: E402
from env.engine import PairDispatchEnv                      # noqa: E402
from env.validator2 import validate as validate2            # noqa: E402
from methods.rules import get_selector                      # noqa: E402
from overlays.build import build_overlay, load_crews, overlay_id  # noqa: E402

OUT_DIR = ROOT / "results" / "ecal"

# Instances and the released calibration table come from one resolved corpus
# root, so the two can never be mixed. Rebinding them on the read-only helper
# module keeps its instance selection on the same root.
Y1_ROOT = resolve_y1_root()
rd.Y1 = Y1_ROOT
rd.CAP = Y1_ROOT / "results/p1_calib/capacity.csv"
rd.INST_ROOT = Y1_ROOT / "data/processed/instances"

RULE_SEED = rd.RULE_SEED
ALL_RULES = rd.ALL_RULES
VERDICT_CAMPUSES = rd.VERDICT_CAMPUSES
INST_ROOT = rd.INST_ROOT


def corpus_fingerprint():
    """Short digest of the calibration table and the instance index.

    Episodes are cached per instance, so a run against a different corpus
    must not reuse them. The digest goes in the shard directory name, which
    makes stale reuse impossible instead of merely unlikely, and leaves the
    earlier run's shards in place.
    """
    h = hashlib.sha256()
    for f in (rd.CAP, INST_ROOT / "index.csv"):
        h.update(f.read_bytes() if f.exists() else b"missing")
    return h.hexdigest()[:8]


FINGERPRINT = corpus_fingerprint()
SHARD_DIR = OUT_DIR / ("shards_" + FINGERPRINT)

# "released" is the calibration table the released results were produced with,
# copied through by experiments/recalibrate.py; the rest are rebuilt from the
# raw corpus and form a self-consistent family.
CALIBRATIONS = ["released", "q090_train", "q095_train", "q0975_train",
                "q099_train", "q095_all"]
RELEASED_LABEL = "released"            # reproduces results/tier1 bitwise

MS = [1.0, 0.8, 0.6]
ETAS = [1.0, 0.8]
STRUCTS = [("dedicated", None), ("chain", 1.0), ("full", None)]

SMOKE_INSTANCES = ["c05_replay_150_0100", "c05_replay_400_0100"]

FIELDS = ["calibration"] + list(rd.FIELDS)

_CREWS: dict = {}
_OVERLAYS: dict = {}


def capacity_path(label):
    return OUT_DIR / ("capacity_%s.csv" % label)


def crews_for(label, campus):
    key = (label, campus)
    if key not in _CREWS:
        _CREWS[key] = load_crews(capacity_path(label), campus)
    return _CREWS[key]


def overlay_for(label, campus, structure, phi, eta, m):
    key = (label, campus, structure, phi, eta, m)
    ov = _OVERLAYS.get(key)
    if ov is None:
        ov = build_overlay(campus, crews_for(label, campus), structure, phi,
                           eta, m)
        _OVERLAYS[key] = ov
    return ov


def cells():
    """(structure, phi, eta, m) with the released L0 eta-dedup."""
    out = []
    for (st, phi) in STRUCTS:
        for m in MS:
            if st == "dedicated":
                out.append((st, phi, 1.0, m))
            else:
                for eta in ETAS:
                    out.append((st, phi, eta, m))
    return out


# --------------------------------------------------------------------------- #
def build_configs(labels, smoke=False):
    """One config = (calibration, instance); every cell runs inside it."""
    rows = rd._replay_rows(VERDICT_CAMPUSES)
    if smoke:
        rows = [r for r in rows if r["id"] in SMOKE_INSTANCES]
    rows.sort(key=lambda r: (int(r["campus"]), int(r["size_class"]), r["id"]))
    configs = []
    for label in labels:
        for r in rows:
            configs.append({
                "calibration": label, "instance_id": r["id"],
                "path": str(INST_ROOT / r["path"]),
                "campus": int(r["campus"]), "size": int(r["size_class"]),
                "config_id": "%s__%s" % (label, r["id"]),
            })
    return configs


def _row(cfg, st, phi, eta, m, method, sched, res):
    mt = res["metrics"]
    pp = mt["per_priority_breach_share"]
    decisions = sched.get("decisions")
    wall = sched.get("wall_seconds")
    mean_ms = (1000.0 * wall / decisions
               if decisions and wall is not None and decisions > 0 else None)
    return {"calibration": cfg["calibration"],
            "instance_id": cfg["instance_id"], "campus": cfg["campus"],
            "size": cfg["size"], "track": "replay", "structure": st,
            "phi": phi, "eta": eta, "m": m, "u_target": None,
            "u_realized": None, "method": method, "seed": RULE_SEED,
            "twt": mt["WWT"], "makespan": mt["makespan"],
            "mean_flow": mt["mean_flow"], "breach_share": mt["breach_share"],
            "breach_p1": pp.get(1), "breach_p2": pp.get(2),
            "breach_p3": pp.get(3), "breach_p4": pp.get(4),
            "decisions": decisions, "latency_ms_per_decision": mean_ms,
            "replans": None, "validator_ok": int(bool(res["feasible"])),
            "runtime_s": wall}


def run_config(cfg):
    t0 = time.perf_counter()
    try:
        dst = SHARD_DIR / (cfg["config_id"] + ".json")
        if dst.exists():
            try:
                with open(dst) as f:
                    old = json.load(f)
                if len(old.get("rows", [])) == len(cells()) * len(ALL_RULES):
                    return {"config_id": cfg["config_id"], "ok": True,
                            "skipped": True}
            except Exception:
                pass

        with open(cfg["path"]) as f:
            inst = json.load(f)
        rows, bad = [], []
        for (st, phi, eta, m) in cells():
            ov = overlay_for(cfg["calibration"], cfg["campus"], st, phi, eta,
                             m)
            env = PairDispatchEnv(inst, ov)
            for meth in ALL_RULES:
                sched = env.run_selector(get_selector(meth), method=meth,
                                         seed=RULE_SEED)
                res = validate2(inst, sched, ov)
                r = _row(cfg, st, phi, eta, m, meth, sched, res)
                rows.append(r)
                if r["validator_ok"] == 0:
                    bad.append("%s|%s" % (overlay_id(cfg["campus"], st, phi,
                                                     eta, m), meth))
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"config_id": cfg["config_id"], "rows": rows}, f)
        os.replace(tmp, dst)
        return {"config_id": cfg["config_id"], "ok": True, "infeasible": bad,
                "wall": time.perf_counter() - t0, "n_rows": len(rows)}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"config_id": cfg["config_id"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- #
# Regression check 2: the released calibration must reproduce tier1 bitwise   #
# --------------------------------------------------------------------------- #
def load_tier1_rule_twt():
    """(instance_id, structure, phi, eta, m, method) -> twt string."""
    out = {}
    with open(ROOT / "results/tier1/results.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] in ALL_RULES:
                out[(r["instance_id"], r["structure"], r["phi"], r["eta"],
                     r["m"], r["method"])] = float(r["twt"])
    return out


def _tier1_key(row):
    phi = "" if row["phi"] is None else "%s" % row["phi"]
    return (row["instance_id"], row["structure"], phi, "%s" % row["eta"],
            "%s" % row["m"], row["method"])


def load_drift_record():
    """Instances whose released tier1 rows no longer reproduce.

    The Y1 instance corpus on disk was rebuilt after the tier1 evaluation
    (the Y1 report results/r4_revision/corpus_diff.md records the two
    construction corrections), so a small, enumerated set of instances scores
    differently under the current files. The record pins that set and its row
    count, so the check still fails on any new drift.
    """
    p = ROOT / "notes" / "revision_r1" / "tier1_corpus_drift.json"
    if not p.exists():
        return None
    return json.load(open(p))


def regression_check(rows, verbose=True):
    """Every released-calibration row must equal the tier1 rule row bitwise,
    except on the enumerated corpus-drift instances."""
    ref = load_tier1_rule_twt()
    drift = load_drift_record() or {}
    allowed = set(drift.get("instances", []))
    checked = mism = missing = unexplained = 0
    bad_instances = set()
    details = []
    for r in rows:
        if r["calibration"] != RELEASED_LABEL:
            continue
        key = _tier1_key(r)
        if key not in ref:
            missing += 1
            if len(details) < 12:
                details.append(("MISSING", key))
            continue
        checked += 1
        if float(r["twt"]) != ref[key]:
            mism += 1
            bad_instances.add(r["instance_id"])
            if r["instance_id"] not in allowed:
                unexplained += 1
                if len(details) < 12:
                    details.append(("MISMATCH", key, float(r["twt"]),
                                    ref[key]))
    expected = drift.get("mismatch_rows")
    if verbose:
        print("[check 2] tier1 bitwise: %d compared, %d mismatch in %d "
              "instance(s), %d missing, %d outside the recorded corpus drift "
              "(recorded: %s rows)"
              % (checked, mism, len(bad_instances), missing, unexplained,
                 expected), flush=True)
        for d in details:
            print("   ", d)
    # A rerun on the corpus the released results were produced with has no
    # mismatch at all; a rerun on a drifted corpus must match the recorded
    # set exactly. Anything else fails.
    ok = (checked > 0 and missing == 0 and unexplained == 0
          and (mism == 0 or expected is None or mism == expected))
    return checked, mism, missing, unexplained, ok


# --------------------------------------------------------------------------- #
def read_shards():
    rows = []
    if SHARD_DIR.exists():
        for p in sorted(SHARD_DIR.glob("*.json")):
            with open(p) as f:
                rows += json.load(f).get("rows", [])
    return rows


def merge(verbose=True):
    rows = read_shards()
    rows.sort(key=lambda r: (r["calibration"], r["campus"], r["size"], r["m"],
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
        pd.DataFrame(rows).to_parquet(OUT_DIR / "results.parquet", index=False)
    except Exception:
        pass
    n_bad = sum(1 for r in rows if not r.get("validator_ok"))
    if verbose:
        print("merged %d rows (%d infeasible) -> %s"
              % (len(rows), n_bad, OUT_DIR / "results.csv"), flush=True)
    return rows, n_bad


def write_summary(rows, n_bad, elapsed, n_configs):
    import collections
    per_cal = collections.Counter(r["calibration"] for r in rows)
    heads = {}
    for label in CALIBRATIONS:
        heads[label] = {}
        for c in VERDICT_CAMPUSES:
            crews = load_crews(capacity_path(label), c)
            heads[label]["c%02d" % c] = {
                "%s" % m: build_overlay(c, crews, "dedicated", None, 1.0,
                                        m)["headcount"] for m in MS}
    checked, mism, missing, unexplained, _ok = regression_check(rows,
                                                                verbose=False)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "elapsed_s": elapsed, "n_configs": n_configs,
                   "n_rows": len(rows), "n_infeasible": n_bad,
                   "calibrations": CALIBRATIONS,
                   "y1_root": Y1_ROOT.name,
                   "corpus_fingerprint": FINGERPRINT,
                   "rows_per_calibration": dict(per_cal),
                   "rule_set": ALL_RULES, "rule_seed": RULE_SEED,
                   "cells": [list(c) for c in cells()],
                   "headcount_by_calibration": heads,
                   "regression_tier1": {
                       "compared": checked, "mismatches": mism,
                       "missing": missing,
                       "outside_recorded_drift": unexplained,
                       "drift_record": "notes/revision_r1/"
                                       "tier1_corpus_drift.json"}},
                  f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--calibrations", default=None,
                    help="comma list (default: every calibration)")
    ap.add_argument("--smoke", action="store_true",
                    help="two campus-5 instances, every calibration and cell")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    print("Y1 root: %s (corpus %s)" % (Y1_ROOT.name, FINGERPRINT),
          flush=True)
    if args.merge:
        rows, n_bad = merge()
        regression_check(rows)
        return

    labels = ([s.strip() for s in args.calibrations.split(",")]
              if args.calibrations else CALIBRATIONS)
    for label in labels:
        if not capacity_path(label).exists():
            sys.exit("missing crew table %s; run experiments/recalibrate.py "
                     "--all first" % capacity_path(label).name)

    configs = build_configs(labels, smoke=args.smoke)
    print("e-cal: calibrations=%s configs=%d cells=%d rules=%d workers=%d "
          "smoke=%s" % (labels, len(configs), len(cells()), len(ALL_RULES),
                        args.workers, args.smoke), flush=True)

    t0 = time.time()
    done = errs = 0
    if args.smoke or args.workers <= 1:
        results = (run_config(c) for c in configs)
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(args.workers)
        results = pool.imap_unordered(run_config, configs)
    for res in results:
        done += 1
        if not res.get("ok"):
            errs += 1
            print("[ERR] %s: %s\n%s" % (res["config_id"], res.get("error"),
                                        res.get("traceback", "")), flush=True)
        elif res.get("infeasible"):
            print("[INFEASIBLE] %s %s" % (res["config_id"],
                                          res["infeasible"]), flush=True)
        if done % 200 == 0 or done == len(configs):
            el = time.time() - t0
            print("  %d/%d %.0fs eta %.0fs (%d err)"
                  % (done, len(configs), el, el / done * (len(configs) - done),
                     errs), flush=True)
    if not (args.smoke or args.workers <= 1):
        pool.close()
        pool.join()
    elapsed = time.time() - t0

    if args.smoke:
        rows = []
        for c in configs:
            with open(SHARD_DIR / (c["config_id"] + ".json")) as f:
                rows += json.load(f).get("rows", [])
        n_bad = sum(1 for r in rows if not r.get("validator_ok"))
        print("smoke: %d rows, %d infeasible, %.1fs" % (len(rows), n_bad,
                                                        elapsed))
        _c, _m, _mi, _u, chk_ok = regression_check(rows)
        ok = (chk_ok and n_bad == 0)
        print("smoke %s" % ("PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)

    rows, n_bad = merge()
    _c, _m, _mi, _u, chk_ok = regression_check(rows)
    if not chk_ok:
        sys.exit("REGRESSION CHECK FAILED: the released calibration does not "
                 "reproduce results/tier1/results.csv outside the recorded "
                 "corpus drift")
    if n_bad:
        sys.exit("STOP: %d infeasible row(s)" % n_bad)
    write_summary(rows, n_bad, elapsed, len(configs))
    print("done in %.0fs" % elapsed)


if __name__ == "__main__":
    main()
