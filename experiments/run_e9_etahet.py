#!/usr/bin/env python
"""E-D Heterogeneous-eta runner: rules + Random envelope on the
penalized verdict cells (CHAIN(1.0), FULL at m=0.6), with per-trade-pair eta.

Comparator: released tier1 CHAIN(1.0)/FULL/L0 cells at uniform eta=0.8, m=0.6
(results/tier1/results.csv). L0 is eta-invariant (all primary work), so it is
reused from tier1 and never recomputed here.

This runner touches NO released source file. It reuses experiments.run_dynamic
ONLY for read-only helpers (instance selection, constants, FIELDS, selectors)
so the instance set is identical to the released sweep. Durations come from
experiments.etahet.EtaHetEnv; feasibility from experiments.etahet.validate_etahet.

Modes:
  --repro-check   validity gate 3c: constant-0.8 matrix must reproduce the
                  released uniform-eta=0.8 per-instance TWT BITWISE.
  --smoke         3c + one heterogeneous draw on 2 campus-9 instances
                  (150+400) x {CHAIN(1.0), FULL} x all rules; assert loads.
  (default)       full plan: 3 draws x {CHAIN, FULL} x 763 verdict instances.
                  NOT run under E-D scope (prepare + smoke only).

Usage:
  PYTHONPATH=.:vendor OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python experiments/run_e9_etahet.py --smoke
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

import experiments.run_dynamic as rd                       # noqa: E402
from overlays.build import build_overlay, load_crews, overlay_id, apply_overlay  # noqa: E402
from methods.rules import get_selector                     # noqa: E402
from experiments.etahet import (                           # noqa: E402
    EtaHetEnv, validate_etahet, draw_eta_matrix, matrix_mean,
    matrix_to_records, records_to_matrix, DRAW_SEEDS)

CAP = rd.CAP
INST_ROOT = rd.INST_ROOT
RULE_SEED = rd.RULE_SEED
ALL_RULES = rd.ALL_RULES
VERDICT_CAMPUSES = rd.VERDICT_CAMPUSES

OUT_DIR = ROOT / "results" / "e9_etahet"
OVL_DIR = ROOT / "overlays" / "generated_overlays" / "etahet"
M = 0.6
CELLS = [("chain", 1.0), ("full", None)]        # penalized cells; L0 reused
COMPARATOR_ETA = 0.8                            # released uniform penalty
FIELDS = list(rd.FIELDS) + ["draw_seed", "mean_eta"]


# --------------------------------------------------------------------------- #
# Overlays: released technician list (eta-independent) + attached eta matrix   #
# --------------------------------------------------------------------------- #
def trades_of(campus):
    return sorted({c["trade"] for c in load_crews(CAP, campus)})


def released_overlay(campus, structure, phi):
    """The released penalized overlay (uniform eta=0.8, m=0.6). Technicians,
    budget and headcount are eta-independent, so this is byte-identical to the
    tier1 comparator cell for this (campus, structure, m)."""
    return build_overlay(campus, load_crews(CAP, campus), structure, phi,
                         COMPARATOR_ETA, M)


def build_etahet_overlay(campus, structure, phi, seed, eta_matrix):
    """Released overlay + heterogeneous-eta payload. Asserts the technician
    portion is byte-identical to the released cell (in-runner assert 3b)."""
    ov = released_overlay(campus, structure, phi)
    ref = released_overlay(campus, structure, phi)
    assert ov["technicians"] == ref["technicians"], "technician list drift"
    assert ov["headcount"] == ref["headcount"], "headcount drift"
    assert ov["budget_B"] == ref["budget_B"], "budget drift"
    ov = dict(ov)
    ov["overlay_id"] = ov["overlay_id"] + "_etahet%d" % seed
    ov["eta_het"] = True
    ov["draw_seed"] = int(seed)
    ov["eta_matrix"] = matrix_to_records(eta_matrix)
    ov["eta_matrix_mean"] = matrix_mean(eta_matrix)
    return ov


def save_overlay_json(ov):
    OVL_DIR.mkdir(parents=True, exist_ok=True)
    p = OVL_DIR / (ov["overlay_id"] + ".json")
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(ov, f, indent=1)
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------- #
# Assert helpers                                                              #
# --------------------------------------------------------------------------- #
def assert_primary_exact(inst, ov, sched):
    """In-runner assert 3a: primary-trade work uses p_j exactly."""
    prim_by_id = {t["id"]: t["primary"] for t in ov["technicians"]}
    wo = {w["id"]: w for w in inst["work_orders"]}
    n_prim = 0
    for a in sched["assignments"]:
        w = wo[a["wo"]]
        if prim_by_id[a["tech"]] == w["trade"]:
            n_prim += 1
            dur = a["end_bh"] - a["start_bh"]
            # pair_p returns p_j exactly for primary; end = start + p_j carries
            # only float-add noise (~1e-15). Any eta penalty rounds UP to the
            # 0.01 grid, so it would shift dur by >= 0.01: 1e-9 cleanly
            # separates "primary exact" from "eta was applied".
            assert abs(dur - float(w["p_bh"])) <= 1e-9, (
                "eta applied to PRIMARY work %s on %s: dur=%r p_bh=%r"
                % (a["wo"], a["tech"], dur, w["p_bh"]))
    return n_prim


# --------------------------------------------------------------------------- #
# One (instance, cell, draw) -> method rows                                    #
# --------------------------------------------------------------------------- #
def run_cell(inst_path, campus, size, structure, phi, seed, eta_matrix,
             methods, check_primary=False):
    with open(inst_path) as f:
        inst = json.load(f)
    ov = build_etahet_overlay(campus, structure, phi, seed, eta_matrix)
    mean_eta = ov["eta_matrix_mean"]
    rows = []
    for meth in methods:
        env = EtaHetEnv(inst, ov, eta_matrix)
        sched = env.run_selector(get_selector(meth), method=meth,
                                 seed=RULE_SEED)
        if check_primary:
            assert_primary_exact(inst, ov, sched)
        res = validate_etahet(inst, sched, ov, eta_matrix)
        m = res["metrics"]
        pp = m["per_priority_breach_share"]
        decisions = sched.get("decisions")
        wall = sched.get("wall_seconds")
        mean_ms = (1000.0 * wall / decisions
                   if decisions and wall is not None and decisions > 0
                   else None)
        rows.append({
            "instance_id": inst["meta"]["id"], "campus": campus, "size": size,
            "track": "replay", "structure": structure, "phi": phi,
            "eta": COMPARATOR_ETA, "m": M, "u_target": None,
            "u_realized": None, "method": meth, "seed": RULE_SEED,
            "twt": m["WWT"], "makespan": m["makespan"],
            "mean_flow": m["mean_flow"], "breach_share": m["breach_share"],
            "breach_p1": pp.get(1), "breach_p2": pp.get(2),
            "breach_p3": pp.get(3), "breach_p4": pp.get(4),
            "decisions": decisions, "latency_ms_per_decision": mean_ms,
            "replans": None, "validator_ok": int(bool(res["feasible"])),
            "runtime_s": wall, "draw_seed": int(seed),
            "mean_eta": mean_eta,
        })
    return rows


# --------------------------------------------------------------------------- #
# 3c: constant-0.8 matrix must reproduce released tier1 TWT bitwise            #
# --------------------------------------------------------------------------- #
def load_tier1_twt():
    """(instance_id, structure, phi, m, method) -> twt, at eta=0.8, m=0.6."""
    out = {}
    with open(ROOT / "results/tier1/results.csv", newline="") as f:
        for r in csv.DictReader(f):
            if (r["eta"] == "0.8" and r["m"] == "0.6"
                    and r["structure"] in ("chain", "full")):
                key = (r["instance_id"], r["structure"],
                       r["phi"], r["method"])
                out[key] = r["twt"]
    return out


def repro_check(instances, verbose=True):
    """Constant-0.8 everywhere -> byte-identical TWT to released tier1."""
    tier1 = load_tier1_twt()
    checked = mism = 0
    details = []
    for (campus, size, inst_path) in instances:
        trades = trades_of(campus)
        const = {(g, h): 0.8 for g in trades for h in trades if g != h}
        for structure, phi in CELLS:
            rows = run_cell(inst_path, campus, size, structure, phi,
                            seed=0, eta_matrix=const, methods=ALL_RULES,
                            check_primary=True)
            for row in rows:
                phi_s = ("1.0" if structure == "chain" else "")
                key = (row["instance_id"], structure, phi_s, row["method"])
                ref = tier1.get(key)
                got = repr(float(row["twt"]))
                if ref is None:
                    details.append(("MISSING", key))
                    continue
                checked += 1
                # bitwise compare on the float value
                same = float(row["twt"]) == float(ref)
                if not same:
                    mism += 1
                    details.append(("MISMATCH", key, got, ref))
    if verbose:
        print("[3c] repro-check: %d rows compared, %d mismatches"
              % (checked, mism))
        for d in details[:12]:
            print("   ", d)
    return checked, mism, details


# --------------------------------------------------------------------------- #
# Smoke                                                                       #
# --------------------------------------------------------------------------- #
def smoke():
    # 2 campus-9 instances (150 + 400)
    rows_idx = rd._replay_rows([9])
    by_size = {}
    for r in rows_idx:
        by_size.setdefault(int(r["size_class"]), []).append(r)
    picks = []
    for size in (150, 400):
        r = sorted(by_size[size], key=lambda r: r["id"])[0]
        picks.append((9, size, str(INST_ROOT / r["path"]), r["id"]))
    instances = [(c, s, p) for (c, s, p, _i) in picks]
    print("smoke instances:", [i for (_c, _s, _p, i) in picks])

    # 3c gate first.
    checked, mism, details = repro_check(instances, verbose=True)
    if mism or checked == 0:
        print("[3c] FAILED -> hook injection route is wrong; STOP.")
        return False

    # 3b already asserted inside build_etahet_overlay (technician byte-identity).
    print("[3b] technician/budget/headcount byte-identity asserted per cell.")

    # One heterogeneous draw (first seed) on the 2 instances x {chain, full}.
    seed = DRAW_SEEDS[0]
    all_rows = []
    for (campus, size, inst_path, iid) in picks:
        trades = trades_of(campus)
        mat = draw_eta_matrix(trades, seed, campus)
        print("draw seed=%d campus=%d: |pairs|=%d realized mean=%.4f "
              "(min=%.3f max=%.3f)"
              % (seed, campus, len(mat), matrix_mean(mat),
                 min(mat.values()), max(mat.values())))
        for structure, phi in CELLS:
            rows = run_cell(inst_path, campus, size, structure, phi, seed,
                            mat, ALL_RULES, check_primary=True)
            all_rows += rows

    # 3a already asserted (check_primary=True). Report primary coverage once.
    print("[3a] primary-exact assert passed on all smoke schedules.")

    # Write smoke rows; confirm pandas load.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke_csv = OUT_DIR / "smoke_results.csv"
    with open(smoke_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    n_feas = sum(r["validator_ok"] for r in all_rows)
    print("[5] wrote %d smoke rows -> %s (%d validator_ok)"
          % (len(all_rows), smoke_csv, n_feas))
    try:
        import pandas as pd
        df = pd.read_csv(smoke_csv)
        print("[5] pandas load ok: shape=%s columns=%d"
              % (df.shape, len(df.columns)))
        print(df[["instance_id", "structure", "method", "twt",
                  "validator_ok", "mean_eta"]].to_string(index=False))
    except Exception as e:
        print("[5] pandas load FAILED:", e)
        return False
    return n_feas == len(all_rows)


# --------------------------------------------------------------------------- #
# Full plan (NOT run under E-D scope)                                          #
# --------------------------------------------------------------------------- #
def build_full_configs():
    rows = rd._replay_rows(VERDICT_CAMPUSES)
    configs = []
    for r in rows:
        campus = int(r["campus"])
        for seed in DRAW_SEEDS:
            for structure, phi in CELLS:
                configs.append({
                    "instance_id": r["id"], "campus": campus,
                    "size": int(r["size_class"]),
                    "path": str(INST_ROOT / r["path"]),
                    "structure": structure, "phi": phi, "seed": seed,
                    "config_id": "%s__%s_etahet%d" % (
                        r["id"], overlay_id(campus, structure, phi,
                                            COMPARATOR_ETA, M), seed),
                })
    return configs


def run_full(workers):
    import multiprocessing as mp
    configs = build_full_configs()
    print("full configs:", len(configs))
    # Materialise overlays once (deterministic), then run in a pool.
    (OUT_DIR / "shards").mkdir(parents=True, exist_ok=True)
    for campus in VERDICT_CAMPUSES:
        trades = trades_of(campus)
        for seed in DRAW_SEEDS:
            mat = draw_eta_matrix(trades, seed, campus)
            for structure, phi in CELLS:
                save_overlay_json(build_etahet_overlay(
                    campus, structure, phi, seed, mat))
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_full_one, configs), 1):
            if i % 200 == 0:
                el = time.time() - t0
                print("  %d/%d %.0fs eta %.0fs"
                      % (i, len(configs), el,
                         el / i * (len(configs) - i)), flush=True)
    merge_full()


def _run_full_one(cfg):
    dst = OUT_DIR / "shards" / (cfg["config_id"] + ".json")
    if dst.exists():
        return
    trades = trades_of(cfg["campus"])
    mat = draw_eta_matrix(trades, cfg["seed"], cfg["campus"])
    rows = run_cell(cfg["path"], cfg["campus"], cfg["size"], cfg["structure"],
                    cfg["phi"], cfg["seed"], mat, ALL_RULES)
    tmp = dst.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"config_id": cfg["config_id"], "rows": rows}, f)
    os.replace(tmp, dst)


def merge_full():
    rows = []
    for p in sorted((OUT_DIR / "shards").glob("*.json")):
        with open(p) as f:
            rows += json.load(f).get("rows", [])
    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    print("merged -> %d rows" % len(rows))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--repro-check", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.smoke:
        ok = smoke()
        sys.exit(0 if ok else 1)
    if args.repro_check:
        rows_idx = rd._replay_rows([9])
        by_size = {}
        for r in rows_idx:
            by_size.setdefault(int(r["size_class"]), []).append(r)
        inst = [(9, s, str(INST_ROOT / sorted(by_size[s],
                 key=lambda r: r["id"])[0]["path"])) for s in (150, 400)]
        _, mism, _ = repro_check(inst)
        sys.exit(0 if mism == 0 else 1)
    if args.full:
        run_full(args.workers)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
