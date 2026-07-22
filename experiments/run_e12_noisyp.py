#!/usr/bin/env python
"""E12: duration-estimate sensitivity (look-ahead robustness).

The dispatcher is assumed to see the true labour duration at dispatch. Real
CMMS record actual duration only at completion; at dispatch a rule would see
an ESTIMATE. This runner gives each rule a noisy estimate p_hat = p * exp(eps),
eps ~ N(0, sigma), seeded per (instance, sigma, order), while the engine still
EXECUTES on the true p (pair_p untouched) and the independent validator scores
the true-duration schedule. It measures how the method ranking, the chain and
FULL dividends, and Gate C move as the estimate degrades.

The point the experiment isolates: EDD and pFIFO score on due date and
priority, not processing time, so they are exactly duration-free and their
schedules are IDENTICAL at every sigma; only WSPT/ATC/MOR/LFJ-ATC/ATC-eta use
the estimate. sigma = 0 must reproduce the released tier1 rule TWT bitwise
(identity guard).

Verdict cells: m = 0.6, eta in {1.0, 0.8}, structures dedicated / CHAIN(1.0) /
FULL, replay verdict campuses. Output results/e12_noisyp/.

Usage (CPU): PYTHONPATH=.:vendor python experiments/run_e12_noisyp.py \
    [--workers 18] [--check] [--merge]
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

import multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
Y1 = Path(os.environ.get("FMWOS_Y1_ROOT", ROOT.parent / "FM-Scheduling"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from env.engine import PairDispatchEnv                       # noqa: E402
from env.validator2 import validate as validate2            # noqa: E402
from methods.rules import get_selector, RANKED_RULES         # noqa: E402
from overlays.build import build_overlay, load_crews         # noqa: E402

CAP = Y1 / "results/p1_calib/capacity.csv"
INST_ROOT = Y1 / "data/processed/instances"
OUT_DIR = ROOT / "results" / "e12_noisyp"
RULE_SEED = 301
VERDICT = [5, 9, 10, 12]
SIZES = [150, 400]
M = 0.6
CELLS = [("dedicated", None, 1.0), ("dedicated", None, 0.8),
         ("chain", 1.0, 1.0), ("chain", 1.0, 0.8),
         ("full", None, 1.0), ("full", None, 0.8)]
SIGMAS = [0.0, 0.1, 0.2, 0.4]
RULES = RANKED_RULES + ["random"]

FIELDS = ["instance_id", "campus", "size", "structure", "phi", "eta", "m",
          "sigma", "method", "twt", "validator_ok"]


def _replay_rows(campuses):
    rows = []
    with open(INST_ROOT / "index.csv", newline="") as f:
        for r in csv.DictReader(f):
            if (r["track"] == "replay" and r["split"] == "test"
                    and int(r["campus"]) in campuses
                    and int(r["size_class"]) in SIZES):
                rows.append(r)
    return rows


def _noise(instance_id, sigma, order_id):
    """Deterministic per-order log-normal multiplier (stable across procs)."""
    if sigma <= 0.0:
        return 1.0
    h = hashlib.sha256(("%s|%.3f|%s" % (instance_id, sigma, order_id))
                       .encode()).hexdigest()
    # two 52-bit draws -> standard normal via Box-Muller
    u1 = (int(h[:13], 16) + 1) / (2 ** 52 + 1)
    u2 = (int(h[13:26], 16) + 1) / (2 ** 52 + 1)
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return math.exp(sigma * z)


def run_config(cfg):
    t0 = time.perf_counter()
    try:
        dst = OUT_DIR / "shards" / (cfg["config_id"] + ".json")
        if dst.exists():
            return {"config_id": cfg["config_id"], "ok": True, "skipped": True}
        crews = load_crews(CAP, cfg["campus"])
        ov = build_overlay(cfg["campus"], crews, cfg["structure"],
                           cfg["phi"], cfg["eta"], cfg["m"])
        with open(cfg["path"]) as f:
            inst = json.load(f)
        rows = []
        for sigma in SIGMAS:
            # Attach the per-order scoring estimate; the engine still executes
            # on the true p_bh (pair_p reads p_bh, never p_score).
            for wo in inst["work_orders"]:
                if sigma > 0.0:
                    wo["p_score"] = wo["p_bh"] * _noise(cfg["instance_id"],
                                                        sigma, wo["id"])
                else:
                    wo.pop("p_score", None)
            for meth in RULES:
                env = PairDispatchEnv(inst, ov)
                sched = env.run_selector(get_selector(meth), method=meth,
                                         seed=RULE_SEED)
                res = validate2(inst, sched, ov)
                rows.append({
                    "instance_id": cfg["instance_id"], "campus": cfg["campus"],
                    "size": cfg["size"], "structure": cfg["structure"],
                    "phi": cfg["phi"], "eta": cfg["eta"], "m": cfg["m"],
                    "sigma": sigma, "method": meth,
                    "twt": res["metrics"]["WWT"],
                    "validator_ok": int(bool(res["feasible"]))})
            for wo in inst["work_orders"]:
                wo.pop("p_score", None)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"config_id": cfg["config_id"], "rows": rows}, f)
        os.replace(tmp, dst)
        return {"config_id": cfg["config_id"], "ok": True,
                "wall": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"config_id": cfg["config_id"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


def build_configs():
    configs = []
    for r in _replay_rows(VERDICT):
        for (st, phi, eta) in CELLS:
            tag = "%s_phi%s_eta%03d" % (st, phi, int(round(eta * 100)))
            configs.append({
                "config_id": "%s__%s" % (r["id"], tag),
                "instance_id": r["id"],
                "path": str(INST_ROOT / r["path"]),
                "campus": int(r["campus"]), "size": int(r["size_class"]),
                "structure": st, "phi": phi, "eta": eta, "m": M})
    return configs


def check():
    """sigma=0 must reproduce released tier1 rule TWT (identity guard)."""
    import pandas as pd
    t1 = pd.read_csv(ROOT / "results" / "tier1" / "results.csv")
    cfgs = build_configs()
    # one nonzero-TWT spot per structure
    seen, spot = set(), []
    for c in cfgs:
        k = c["structure"]
        if k not in seen:
            seen.add(k)
            spot.append(c)
    ok = True
    for c in spot:
        crews = load_crews(CAP, c["campus"])
        ov = build_overlay(c["campus"], crews, c["structure"], c["phi"],
                           c["eta"], c["m"])
        inst = json.load(open(c["path"]))
        for wo in inst["work_orders"]:
            wo.pop("p_score", None)          # sigma = 0
        for meth in ("edd", "wspt", "atc"):
            env = PairDispatchEnv(inst, ov)
            sched = env.run_selector(get_selector(meth), method=meth,
                                     seed=RULE_SEED)
            got = validate2(inst, sched, ov)["metrics"]["WWT"]
            sub = t1[(t1.instance_id == c["instance_id"])
                     & (t1.structure == c["structure"])
                     & (t1.eta == c["eta"]) & (t1.m == c["m"])
                     & (t1.method == meth)]
            if c["phi"] is None:
                sub = sub[sub.phi.isna()]
            else:
                sub = sub[sub.phi == c["phi"]]
            want = float(sub.iloc[0].twt) if len(sub) else None
            match = want is not None and abs(got - want) <= 1e-6
            ok = ok and match
            print("  %-9s %-4s %s: got %.4f want %s %s"
                  % (c["structure"], meth, c["instance_id"], got, want,
                     "OK" if match else "MISMATCH"))
    print("IDENTITY GUARD:", "PASS" if ok else "FAIL")
    return ok


def merge():
    rows = []
    d = OUT_DIR / "shards"
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            rows.extend(json.load(open(p))["rows"])
        except Exception:
            continue
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    print("merged %d rows -> %s" % (len(rows), OUT_DIR / "results.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        return
    if args.merge:
        merge()
        return
    cfgs = [c for c in build_configs()
            if not (OUT_DIR / "shards" / (c["config_id"] + ".json")).exists()]
    print("e12_noisyp: %d configs pending (%d sigmas x %d rules each)"
          % (len(cfgs), len(SIGMAS), len(RULES)), flush=True)
    if not cfgs:
        merge()
        return
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        done = errs = 0
        for res in pool.imap_unordered(run_config, cfgs):
            done += 1
            if not res.get("ok"):
                errs += 1
                print("[ERR] %s: %s" % (res["config_id"], res.get("error")),
                      flush=True)
            if done % 400 == 0 or done == len(cfgs):
                print("  %d/%d %.0fs (%d err)"
                      % (done, len(cfgs), time.time() - t0, errs), flush=True)
    merge()


if __name__ == "__main__":
    main()
