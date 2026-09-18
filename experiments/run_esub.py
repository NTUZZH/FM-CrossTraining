#!/usr/bin/env python
"""E-SUB runner: corrective-only and one-technician-plausible robustness.

Part (i) decomposition
----------------------
Reruns the ranked rules plus Random on the released verdict cells (m in
{0.6, 0.8}, structures L0 / CHAIN(1.0) / FULL, eta in {1.0, 0.8}) with the
released crew table, and records weighted tardiness per work-order class as
well as per instance. Classes: corrective (``is_pm`` false) versus preventive,
and p_j <= 8 bh (one technician-day, the pre-declared proxy for work one
technician can plausibly finish alone) versus p_j > 8 bh. Per-order weighted
tardiness is ``w_j * max(0, C_j - d_j)`` with C_j the completion time of the
validator-accepted schedule, which is the quantity ``env/validator2.py`` sums
into the reported TWT; the per-instance total is compared against the released
tier1 row bitwise.

Part (ii) matched-contention reruns
-----------------------------------
Builds two subset instance sets from the same 763 verdict instances: one that
drops the preventive orders (ids suffixed ``_cm``) and one that drops orders
longer than 8 bh (ids suffixed ``_le8``). Everything else in the instance is
carried through unchanged. Crews are recalibrated on the subset's own stream
(``experiments/recalibrate.py``), so the contention regime matches the reduced
workload rather than the full one, and the rules run on L0 / CHAIN(1.0) /
FULL, m in {1.0, 0.8, 0.6}, eta in {1.0, 0.8}.

Outputs (results/esub/)
  classes.csv     per instance: order counts and labor hours per class
  decomp.csv      part (i): per (instance, cell, method) total TWT and the
                  four class components (+ .parquet)
  results.csv     part (ii): tier1 schema plus a ``subset`` column
                  (+ .parquet)
  summary.json    counts, regression-check result, wall time

Usage (CPU only; one thread per worker):
  PYTHONPATH=.:vendor taskset -c 0-9 python experiments/run_esub.py --smoke
  PYTHONPATH=.:vendor taskset -c 0-9 python experiments/run_esub.py --all --workers 10
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
from overlays.build import build_overlay, load_crews        # noqa: E402

OUT_DIR = ROOT / "results" / "esub"
DATA_DIR = ROOT / "data" / "esub"

# Instances and the released calibration table come from one resolved corpus
# root (see experiments/recalibrate.py), so the two can never be mixed.
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
    """Short digest of the calibration table and the instance index, so a run
    against a different corpus cannot reuse the episodes of an earlier one."""
    h = hashlib.sha256()
    for f in (RELEASED_CAP, INST_ROOT / "index.csv"):
        h.update(f.read_bytes() if f.exists() else b"missing")
    return h.hexdigest()[:8]


FINGERPRINT = corpus_fingerprint()
DECOMP_SHARDS = OUT_DIR / ("shards_decomp_" + FINGERPRINT)
SUBSET_SHARDS = OUT_DIR / ("shards_subset_" + FINGERPRINT)

SHORT_BH = 8.0                      # one technician-day
DECOMP_MS = [0.8, 0.6]              # the contended released families
SUBSET_MS = [1.0, 0.8, 0.6]
ETAS = [1.0, 0.8]
STRUCTS = [("dedicated", None), ("chain", 1.0), ("full", None)]

SUBSETS = {"cm": ("_cm", "cm_q095_train"),
           "le8": ("_le8", "le8_q095_train")}

SMOKE_INSTANCES = ["c05_replay_150_0100", "c05_replay_400_0100"]

CLASS_FIELDS = ["instance_id", "campus", "size", "n_orders", "n_cm", "n_pm",
                "n_short", "n_long", "hours_total", "hours_cm", "hours_pm",
                "hours_short", "hours_long"]
DECOMP_FIELDS = ["instance_id", "campus", "size", "structure", "phi", "eta",
                 "m", "method", "twt", "twt_cm", "twt_pm", "twt_short",
                 "twt_long", "validator_ok"]
SUBSET_FIELDS = ["subset"] + list(rd.FIELDS)

_OVERLAYS: dict = {}
_CREWS: dict = {}


def capacity_path(label):
    return OUT_DIR / ("capacity_%s.csv" % label)


def crews_for(label, campus):
    key = (label, campus)
    if key not in _CREWS:
        path = RELEASED_CAP if label == "released" else capacity_path(label)
        _CREWS[key] = load_crews(path, campus)
    return _CREWS[key]


def overlay_for(label, campus, structure, phi, eta, m):
    key = (label, campus, structure, phi, eta, m)
    ov = _OVERLAYS.get(key)
    if ov is None:
        ov = build_overlay(campus, crews_for(label, campus), structure, phi,
                           eta, m)
        _OVERLAYS[key] = ov
    return ov


def cells(ms):
    out = []
    for (st, phi) in STRUCTS:
        for m in ms:
            if st == "dedicated":
                out.append((st, phi, 1.0, m))
            else:
                for eta in ETAS:
                    out.append((st, phi, eta, m))
    return out


# --------------------------------------------------------------------------- #
# Subset instance construction                                                 #
# --------------------------------------------------------------------------- #
def subset_instance(inst, kind):
    """Drop one order class; carry everything else through unchanged."""
    if kind == "cm":
        keep = [w for w in inst["work_orders"] if not w.get("is_pm")]
    elif kind == "le8":
        keep = [w for w in inst["work_orders"]
                if float(w["p_bh"]) <= SHORT_BH]
    else:
        raise ValueError(kind)
    out = dict(inst)
    out["work_orders"] = keep
    meta = dict(inst["meta"])
    meta["base_instance_id"] = inst["meta"]["id"]
    meta["id"] = inst["meta"]["id"] + SUBSETS[kind][0]
    meta["subset"] = kind
    meta["n_wos_retained"] = len(keep)
    out["meta"] = meta
    return out


def subset_path(kind, instance_id):
    return DATA_DIR / kind / (instance_id + SUBSETS[kind][0] + ".json")


def build_subset_instances(rows, verbose=True):
    """Write both subset instance sets; returns per-kind retained counts.

    An instance whose orders are all of the dropped class has no subset to
    schedule, so it is left out of that arm and counted. Every cell of one
    arm then runs on the same instance list, which is what the L0 / chain /
    full comparison within that arm requires.
    """
    stats = {k: {"n_instances": 0, "n_orders": 0, "n_orders_base": 0,
                 "min_orders": None, "dropped_empty": []} for k in SUBSETS}
    for r in rows:
        with open(INST_ROOT / r["path"]) as f:
            inst = json.load(f)
        for kind in SUBSETS:
            sub = subset_instance(inst, kind)
            if not sub["work_orders"]:
                stats[kind]["dropped_empty"].append(r["id"])
                continue
            p = subset_path(kind, r["id"])
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(sub, f, separators=(",", ":"))
            os.replace(tmp, p)
            s = stats[kind]
            s["n_instances"] += 1
            s["n_orders"] += len(sub["work_orders"])
            s["n_orders_base"] += len(inst["work_orders"])
            s["min_orders"] = (len(sub["work_orders"]) if s["min_orders"]
                               is None else min(s["min_orders"],
                                                len(sub["work_orders"])))
    if verbose:
        for kind, s in stats.items():
            print("subset %-4s %d instances, %d of %d orders retained "
                  "(%.1f%%), smallest %d, %d instance(s) empty and dropped"
                  % (kind, s["n_instances"], s["n_orders"],
                     s["n_orders_base"],
                     100.0 * s["n_orders"] / s["n_orders_base"],
                     s["min_orders"], len(s["dropped_empty"])), flush=True)
    return stats


# --------------------------------------------------------------------------- #
# Part (i): per-order decomposition on the released cells                      #
# --------------------------------------------------------------------------- #
def class_row(inst):
    """Per-instance order counts and labor hours per class."""
    n = {"n_orders": 0, "n_cm": 0, "n_pm": 0, "n_short": 0, "n_long": 0}
    h = {"hours_total": 0.0, "hours_cm": 0.0, "hours_pm": 0.0,
         "hours_short": 0.0, "hours_long": 0.0}
    for w in inst["work_orders"]:
        p = float(w["p_bh"])
        n["n_orders"] += 1
        h["hours_total"] += p
        if w.get("is_pm"):
            n["n_pm"] += 1
            h["hours_pm"] += p
        else:
            n["n_cm"] += 1
            h["hours_cm"] += p
        if p <= SHORT_BH:
            n["n_short"] += 1
            h["hours_short"] += p
        else:
            n["n_long"] += 1
            h["hours_long"] += p
    return {"instance_id": inst["meta"]["id"],
            "campus": int(inst["meta"]["campus"]),
            "size": int(inst["meta"]["size_class"]), **n, **h}


def decompose(inst, sched):
    """Total and per-class weighted tardiness of one schedule.

    The total accumulates in the assignment order the validator uses, so it
    equals the released TWT bitwise; the class components accumulate
    separately and sum to the total up to floating-point association.
    """
    wo = {w["id"]: w for w in inst["work_orders"]}
    total = 0.0
    parts = {"cm": 0.0, "pm": 0.0, "short": 0.0, "long": 0.0}
    for a in sched["assignments"]:
        w = wo[a["wo"]]
        t = float(w["weight"]) * max(0.0, float(a["end_bh"])
                                     - float(w["due_bh"]))
        total += t
        parts["pm" if w.get("is_pm") else "cm"] += t
        parts["short" if float(w["p_bh"]) <= SHORT_BH else "long"] += t
    return total, parts


def run_decomp(cfg):
    t0 = time.perf_counter()
    try:
        dst = DECOMP_SHARDS / (cfg["instance_id"] + ".json")
        n_expect = len(cells(DECOMP_MS)) * len(ALL_RULES)
        if dst.exists():
            try:
                with open(dst) as f:
                    old = json.load(f)
                if len(old.get("rows", [])) == n_expect:
                    return {"config_id": cfg["instance_id"], "ok": True,
                            "skipped": True}
            except Exception:
                pass
        with open(cfg["path"]) as f:
            inst = json.load(f)
        rows, bad = [], []
        for (st, phi, eta, m) in cells(DECOMP_MS):
            ov = overlay_for("released", cfg["campus"], st, phi, eta, m)
            env = PairDispatchEnv(inst, ov)
            for meth in ALL_RULES:
                sched = env.run_selector(get_selector(meth), method=meth,
                                         seed=RULE_SEED)
                res = validate2(inst, sched, ov)
                total, parts = decompose(inst, sched)
                rows.append({"instance_id": cfg["instance_id"],
                             "campus": cfg["campus"], "size": cfg["size"],
                             "structure": st, "phi": phi, "eta": eta, "m": m,
                             "method": meth, "twt": total,
                             "twt_cm": parts["cm"], "twt_pm": parts["pm"],
                             "twt_short": parts["short"],
                             "twt_long": parts["long"],
                             "validator_ok": int(bool(res["feasible"]))})
                if not res["feasible"]:
                    bad.append("%s|%s|%s" % (st, eta, meth))
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"instance_id": cfg["instance_id"], "rows": rows,
                       "classes": class_row(inst)}, f)
        os.replace(tmp, dst)
        return {"config_id": cfg["instance_id"], "ok": True,
                "infeasible": bad, "wall": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"config_id": cfg["instance_id"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- #
# Part (ii): subset reruns                                                     #
# --------------------------------------------------------------------------- #
def run_subset(cfg):
    t0 = time.perf_counter()
    try:
        dst = SUBSET_SHARDS / (cfg["config_id"] + ".json")
        n_expect = len(cells(SUBSET_MS)) * len(ALL_RULES)
        if dst.exists():
            try:
                with open(dst) as f:
                    old = json.load(f)
                if len(old.get("rows", [])) == n_expect:
                    return {"config_id": cfg["config_id"], "ok": True,
                            "skipped": True}
            except Exception:
                pass
        with open(cfg["path"]) as f:
            inst = json.load(f)
        label = SUBSETS[cfg["subset"]][1]
        rows, bad = [], []
        for (st, phi, eta, m) in cells(SUBSET_MS):
            ov = overlay_for(label, cfg["campus"], st, phi, eta, m)
            env = PairDispatchEnv(inst, ov)
            for meth in ALL_RULES:
                sched = env.run_selector(get_selector(meth), method=meth,
                                         seed=RULE_SEED)
                res = validate2(inst, sched, ov)
                mt = res["metrics"]
                pp = mt["per_priority_breach_share"]
                decisions = sched.get("decisions")
                wall = sched.get("wall_seconds")
                rows.append({
                    "subset": cfg["subset"],
                    "instance_id": cfg["instance_id"],
                    "campus": cfg["campus"], "size": cfg["size"],
                    "track": "replay", "structure": st, "phi": phi,
                    "eta": eta, "m": m, "u_target": None,
                    "u_realized": None, "method": meth, "seed": RULE_SEED,
                    "twt": mt["WWT"], "makespan": mt["makespan"],
                    "mean_flow": mt["mean_flow"],
                    "breach_share": mt["breach_share"],
                    "breach_p1": pp.get(1), "breach_p2": pp.get(2),
                    "breach_p3": pp.get(3), "breach_p4": pp.get(4),
                    "decisions": decisions,
                    "latency_ms_per_decision": (
                        1000.0 * wall / decisions
                        if decisions and wall is not None and decisions > 0
                        else None),
                    "replans": None,
                    "validator_ok": int(bool(res["feasible"])),
                    "runtime_s": wall})
                if not res["feasible"]:
                    bad.append("%s|%s|%s" % (st, eta, meth))
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"config_id": cfg["config_id"], "rows": rows}, f)
        os.replace(tmp, dst)
        return {"config_id": cfg["config_id"], "ok": True, "infeasible": bad,
                "wall": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"config_id": cfg["config_id"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- #
# Regression check: per-instance sums equal the released tier1 TWT bitwise     #
# --------------------------------------------------------------------------- #
def load_tier1_rule_twt():
    out = {}
    with open(ROOT / "results/tier1/results.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] in ALL_RULES:
                out[(r["instance_id"], r["structure"], r["phi"], r["eta"],
                     r["m"], r["method"])] = float(r["twt"])
    return out


def load_drift_instances():
    """Instances whose released tier1 rows no longer reproduce.

    The Y1 instance corpus on disk was rebuilt after the tier1 evaluation, so
    an enumerated set of instances scores differently under the current files
    (notes/revision_r1/tier1_corpus_drift.json records the set and the cause).
    Any mismatch outside that set fails the check.
    """
    p = ROOT / "notes" / "revision_r1" / "tier1_corpus_drift.json"
    if not p.exists():
        return set()
    return set(json.load(open(p)).get("instances", []))


def regression_check(rows, verbose=True):
    ref = load_tier1_rule_twt()
    allowed = load_drift_instances()
    checked = mism = missing = unexplained = 0
    max_split = 0.0
    drifted = set()
    details = []
    for r in rows:
        key = (r["instance_id"], r["structure"],
               "" if r["phi"] is None else "%s" % r["phi"],
               "%s" % r["eta"], "%s" % r["m"], r["method"])
        if key not in ref:
            missing += 1
            if len(details) < 12:
                details.append(("MISSING", key))
            continue
        checked += 1
        if float(r["twt"]) != ref[key]:
            mism += 1
            drifted.add(r["instance_id"])
            if r["instance_id"] not in allowed:
                unexplained += 1
                if len(details) < 12:
                    details.append(("MISMATCH", key, float(r["twt"]),
                                    ref[key]))
        for a, b in (("twt_cm", "twt_pm"), ("twt_short", "twt_long")):
            max_split = max(max_split, abs(r[a] + r[b] - r["twt"]))
    if verbose:
        print("[check] tier1 bitwise: %d compared, %d mismatch in %d "
              "instance(s), %d missing, %d outside the recorded corpus "
              "drift; max |class sum - total| = %.3e"
              % (checked, mism, len(drifted), missing, unexplained,
                 max_split), flush=True)
        for d in details:
            print("   ", d)
    ok = checked > 0 and missing == 0 and unexplained == 0
    return checked, mism, missing, max_split, ok


# --------------------------------------------------------------------------- #
def merge_decomp(verbose=True):
    rows, classes = [], []
    if DECOMP_SHARDS.exists():
        for p in sorted(DECOMP_SHARDS.glob("*.json")):
            with open(p) as f:
                d = json.load(f)
            rows += d.get("rows", [])
            if d.get("classes"):
                classes.append(d["classes"])
    rows.sort(key=lambda r: (r["campus"], r["size"], r["m"], r["structure"],
                             r["phi"] or 0, r["eta"], r["instance_id"],
                             r["method"]))
    classes.sort(key=lambda r: (r["campus"], r["size"], r["instance_id"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "decomp.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DECOMP_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in DECOMP_FIELDS})
    with open(OUT_DIR / "classes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLASS_FIELDS)
        w.writeheader()
        for r in classes:
            w.writerow({k: r.get(k) for k in CLASS_FIELDS})
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(OUT_DIR / "decomp.parquet", index=False)
    except Exception:
        pass
    if verbose:
        print("decomp: %d rows, %d instances -> %s"
              % (len(rows), len(classes), OUT_DIR / "decomp.csv"), flush=True)
    return rows, classes


def merge_subset(verbose=True):
    rows = []
    if SUBSET_SHARDS.exists():
        for p in sorted(SUBSET_SHARDS.glob("*.json")):
            with open(p) as f:
                rows += json.load(f).get("rows", [])
    rows.sort(key=lambda r: (r["subset"], r["campus"], r["size"], r["m"],
                             r["structure"], r["phi"] or 0, r["eta"],
                             r["instance_id"], r["method"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUBSET_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in SUBSET_FIELDS})
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(OUT_DIR / "results.parquet",
                                      index=False)
    except Exception:
        pass
    n_bad = sum(1 for r in rows if not r.get("validator_ok"))
    if verbose:
        print("subset: %d rows (%d infeasible) -> %s"
              % (len(rows), n_bad, OUT_DIR / "results.csv"), flush=True)
    return rows, n_bad


# --------------------------------------------------------------------------- #
def drive(fn, configs, workers, label):
    t0 = time.time()
    done = errs = 0
    if workers <= 1:
        results = (fn(c) for c in configs)
        pool = None
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(workers)
        results = pool.imap_unordered(fn, configs)
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
            print("  %s %d/%d %.0fs eta %.0fs (%d err)"
                  % (label, done, len(configs), el,
                     el / done * (len(configs) - done), errs), flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    if errs:
        sys.exit("STOP: %d error(s) in %s" % (errs, label))
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--part", default="both", choices=["decomp", "subset",
                                                       "both"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        rows, _cl = merge_decomp()
        regression_check(rows)
        merge_subset()
        return

    rows_idx = rd._replay_rows(VERDICT_CAMPUSES)
    if args.smoke:
        rows_idx = [r for r in rows_idx if r["id"] in SMOKE_INSTANCES]
    rows_idx.sort(key=lambda r: (int(r["campus"]), int(r["size_class"]),
                                 r["id"]))
    print("e-sub: %d instance(s), decomp cells=%d, subset cells=%d, rules=%d, "
          "workers=%d, smoke=%s" % (len(rows_idx), len(cells(DECOMP_MS)),
                                    len(cells(SUBSET_MS)), len(ALL_RULES),
                                    args.workers, args.smoke), flush=True)

    print("Y1 root: %s (corpus %s)" % (Y1_ROOT.name, FINGERPRINT),
          flush=True)
    elapsed = {}
    subset_stats = {}
    if args.part in ("decomp", "both"):
        configs = [{"instance_id": r["id"],
                    "path": str(INST_ROOT / r["path"]),
                    "campus": int(r["campus"]),
                    "size": int(r["size_class"])} for r in rows_idx]
        elapsed["decomp"] = drive(run_decomp, configs,
                                  1 if args.smoke else args.workers, "decomp")
        drows, _cl = merge_decomp()
        _c, _m, _mi, _ms, chk_ok = regression_check(drows)
        if not chk_ok:
            sys.exit("REGRESSION CHECK FAILED: per-instance sums do not "
                     "reproduce results/tier1/results.csv outside the "
                     "recorded corpus drift")

    if args.part in ("subset", "both"):
        for label in {v[1] for v in SUBSETS.values()}:
            if not capacity_path(label).exists():
                sys.exit("missing crew table %s; run "
                         "experiments/recalibrate.py --all first"
                         % capacity_path(label).name)
        subset_stats = build_subset_instances(rows_idx)
        configs = []
        for kind in SUBSETS:
            empty = set(subset_stats[kind]["dropped_empty"])
            for r in rows_idx:
                if r["id"] in empty:
                    continue
                iid = r["id"] + SUBSETS[kind][0]
                configs.append({"subset": kind, "instance_id": iid,
                                "path": str(subset_path(kind, r["id"])),
                                "campus": int(r["campus"]),
                                "size": int(r["size_class"]),
                                "config_id": iid})
        elapsed["subset"] = drive(run_subset, configs,
                                  1 if args.smoke else args.workers, "subset")
        srows, n_bad = merge_subset()
        if n_bad:
            sys.exit("STOP: %d infeasible subset row(s)" % n_bad)

    if args.smoke:
        print("smoke PASS")
        return

    drows, classes = merge_decomp()
    checked, mism, missing, max_split, _ok = regression_check(
        drows, verbose=False)
    srows, n_bad = merge_subset()
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "elapsed_s": elapsed,
                   "n_instances": len(rows_idx),
                   "y1_root": Y1_ROOT.name,
                   "corpus_fingerprint": FINGERPRINT,
                   "decomp": {"rows": len(drows),
                              "cells": [list(c) for c in cells(DECOMP_MS)],
                              "regression_compared": checked,
                              "regression_mismatches": mism,
                              "drift_record": "notes/revision_r1/"
                                              "tier1_corpus_drift.json",
                              "max_class_sum_deviation": max_split},
                   "subset": {"rows": len(srows), "infeasible": n_bad,
                              "cells": [list(c) for c in cells(SUBSET_MS)],
                              "subsets": {k: SUBSETS[k][1] for k in SUBSETS},
                              "instances": subset_stats},
                   "rule_set": ALL_RULES, "rule_seed": RULE_SEED,
                   "short_bh": SHORT_BH}, f, indent=2)
    print("done", elapsed)


if __name__ == "__main__":
    main()
