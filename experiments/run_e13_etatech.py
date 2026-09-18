#!/usr/bin/env python
"""Per-technician secondary-skill efficiency: rules and Random envelope on
the penalized verdict cells, with eta drawn per (technician, secondary trade).

Cells: verdict campuses 5/9/10/12, the 763 replay test instances, structures
CHAIN(1.0), GEN and FULL, crew multipliers 0.6 and 0.8. Methods: the seven
ranked rules and Random, seed 301.

Inputs come from the Y1 root that experiments/y1_root.py resolves, which is
this project's frozen v1.0 corpus. A comparator arm (--baseline) runs the
released engine with the scalar efficiency on that root and writes
baseline.csv. It carries the eta-invariant L0 cells, which supply the
dividend reference, and the uniform eta = 0.8 cell of every structure. Its
flexible cells reproduce the released result files bitwise, so the arm is a
check as well as the reference: GEN at m = 0.8 is the one cell that was
never released and has no counterpart to check against.

Two draw distributions, three fixed seeds each:
  u60  Uniform[0.60, 1.00], mean 0.80, the main-grid penalty's mean
  u80  Uniform[0.80, 1.00], mean 0.90

Durations come from experiments.etahet_tech.EtaTechEnv; feasibility from
experiments.etahet_tech.validate_etatech, which re-derives every duration
from the per-technician table. No released source file and no released
result file is touched. experiments.run_dynamic is imported for read-only
helpers so the instance set matches the released sweep exactly.

Modes:
  --baseline      the comparator arm described above.
  --repro-check   a constant per-technician table of 0.8 must reproduce the
                  scalar-eta rows bitwise, both against the comparator arm
                  (the new code path) and against the released result files
                  (CHAIN(1.0) and FULL at m = 0.6 and m = 0.8 in tier1, GEN
                  at m = 0.6 in tier2).
  --repro-all     both of those checks over all 763 instances. Any mismatch
                  in either check fails the run.
  --smoke         the repro check on two instances, the validator
                  independence check, and one heterogeneous draw per
                  distribution on those two instances.
  --full          the full plan.

Usage:
  PYTHONPATH=.:vendor OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    taskset -c 10-15 python experiments/run_e13_etatech.py --full --workers 6
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
from experiments.y1_root import y1_root, is_frozen         # noqa: E402
from overlays.build import build_overlay, load_crews, overlay_id  # noqa: E402
from methods.rules import get_selector                     # noqa: E402
from experiments.etahet_tech import (                      # noqa: E402
    EtaTechEnv, validate_etatech, draw_eta_tech, constant_eta_tech,
    table_mean, active_mean, DISTRIBUTIONS, DRAW_SEEDS)

Y1_ROOT = y1_root()
CAP = rd.CAP
INST_ROOT = rd.INST_ROOT
assert Path(CAP).resolve().is_relative_to(Path(Y1_ROOT).resolve()), \
    "the runner and the dynamic runner resolved different Y1 roots"
RULE_SEED = rd.RULE_SEED
ALL_RULES = rd.ALL_RULES
VERDICT_CAMPUSES = rd.VERDICT_CAMPUSES
N_VERDICT_INSTANCES = 763

OUT_DIR = ROOT / "results" / "e13_etatech"
SHARD_DIR = OUT_DIR / "shards"
MS = [0.6, 0.8]
STRUCTS = [("chain", 1.0), ("generalist", None), ("full", None)]
COMPARATOR_ETA = 0.8                       # released uniform penalty
FIELDS = list(rd.FIELDS) + ["dist", "eta_lo", "eta_hi", "draw_seed",
                            "mean_eta", "mean_eta_active"]

# Cells that exist in the released sweep at uniform eta = 0.8, with the file
# they live in. GEN at m = 0.8 was never released and has no comparator.
RELEASED_CELLS = [
    ("chain", 1.0, 0.6, "tier1"), ("full", None, 0.6, "tier1"),
    ("chain", 1.0, 0.8, "tier1"), ("full", None, 0.8, "tier1"),
    ("generalist", None, 0.6, "tier2"),
]

# Comparator arm, run with the released engine on the current inputs: the
# eta-invariant L0 cells and the uniform eta = 0.8 cell of every structure.
# The L0 cells are run at eta = 1.0 because all L0 work is primary.
BASELINE_CELLS = ([("dedicated", None, 1.0, m) for m in MS]
                  + [(st, phi, COMPARATOR_ETA, m)
                     for (st, phi) in STRUCTS for m in MS])
BASELINE_CSV = OUT_DIR / "baseline.csv"
BASELINE_SHARDS = OUT_DIR / "baseline_shards"


_TRADES: dict = {}
_OVERLAYS: dict = {}


def trades_of(campus):
    t = _TRADES.get(campus)
    if t is None:
        t = sorted({c["trade"] for c in load_crews(CAP, campus)})
        _TRADES[campus] = t
    return t


def released_overlay(campus, structure, phi, m):
    """The released penalized overlay for this cell (uniform eta = 0.8).

    Technicians, headcount and skill budget do not depend on eta, so this is
    the same technician list the released comparator cell used. Memoized per
    process; callers copy before adding the heterogeneous payload.
    """
    key = (campus, structure, phi, m)
    ov = _OVERLAYS.get(key)
    if ov is None:
        ov = build_overlay(campus, load_crews(CAP, campus), structure, phi,
                           COMPARATOR_ETA, m)
        _OVERLAYS[key] = ov
    return ov


def build_etatech_overlay(campus, structure, phi, m, tag, eta_tech):
    """Released overlay plus the per-technician payload."""
    ov = released_overlay(campus, structure, phi, m)
    ref = rd.overlay_for(campus, structure, phi, COMPARATOR_ETA, m)
    assert ov["technicians"] == ref["technicians"], "technician list drift"
    assert ov["headcount"] == ref["headcount"], "headcount drift"
    assert ov["budget_B"] == ref["budget_B"], "budget drift"
    ov = dict(ov)
    ov["overlay_id"] = ov["overlay_id"] + "_etatech_" + tag
    ov["eta_per_technician"] = True
    return ov


def assert_primary_exact(inst, ov, sched):
    """Primary-trade work must use p_j exactly; eta never touches it."""
    prim_by_id = {t["id"]: t["primary"] for t in ov["technicians"]}
    wo = {w["id"]: w for w in inst["work_orders"]}
    n_prim = 0
    for a in sched["assignments"]:
        w = wo[a["wo"]]
        if prim_by_id[a["tech"]] == w["trade"]:
            n_prim += 1
            dur = a["end_bh"] - a["start_bh"]
            # Any eta penalty rounds up to the 0.01 grid, so it would shift
            # the duration by at least 0.01; 1e-9 separates float-add noise
            # from a penalty that was applied.
            assert abs(dur - float(w["p_bh"])) <= 1e-9, (
                "eta applied to PRIMARY work %s on %s: dur=%r p_bh=%r"
                % (a["wo"], a["tech"], dur, w["p_bh"]))
    return n_prim


# --------------------------------------------------------------------------- #
# One (instance, cell, draw) -> method rows                                    #
# --------------------------------------------------------------------------- #
def run_cell(inst, campus, size, structure, phi, m, tag, eta_tech,
             methods, dist=None, seed=None, check_primary=False):
    ov = build_etatech_overlay(campus, structure, phi, m, tag, eta_tech)
    lo, hi = DISTRIBUTIONS.get(dist, (None, None))
    mean_all = table_mean(eta_tech)
    mean_act = active_mean(eta_tech, ov["technicians"])
    rows = []
    for meth in methods:
        env = EtaTechEnv(inst, ov, eta_tech)
        sched = env.run_selector(get_selector(meth), method=meth,
                                 seed=RULE_SEED)
        if check_primary:
            assert_primary_exact(inst, ov, sched)
        res = validate_etatech(inst, sched, ov, eta_tech)
        mt = res["metrics"]
        pp = mt["per_priority_breach_share"]
        decisions = sched.get("decisions")
        wall = sched.get("wall_seconds")
        mean_ms = (1000.0 * wall / decisions
                   if decisions and wall is not None and decisions > 0
                   else None)
        rows.append({
            "instance_id": inst["meta"]["id"], "campus": campus, "size": size,
            "track": "replay", "structure": structure, "phi": phi,
            "eta": COMPARATOR_ETA, "m": m, "u_target": None,
            "u_realized": None, "method": meth, "seed": RULE_SEED,
            "twt": mt["WWT"], "makespan": mt["makespan"],
            "mean_flow": mt["mean_flow"], "breach_share": mt["breach_share"],
            "breach_p1": pp.get(1), "breach_p2": pp.get(2),
            "breach_p3": pp.get(3), "breach_p4": pp.get(4),
            "decisions": decisions, "latency_ms_per_decision": mean_ms,
            "replans": None, "validator_ok": int(bool(res["feasible"])),
            "runtime_s": wall, "dist": dist, "eta_lo": lo, "eta_hi": hi,
            "draw_seed": (int(seed) if seed is not None else None),
            "mean_eta": mean_all, "mean_eta_active": mean_act,
        })
    return rows


# --------------------------------------------------------------------------- #
# Regression check: constant 0.8 must reproduce the released rows bitwise     #
# --------------------------------------------------------------------------- #
_RELEASED_TWT: dict = {}


def load_released_twt():
    """(instance_id, structure, m, method) -> twt at uniform eta = 0.8.

    Cached per process: the released tables are large and the regression
    check reads them once per worker, not once per instance.
    """
    if _RELEASED_TWT:
        return _RELEASED_TWT
    out = _RELEASED_TWT
    for family in ("tier1", "tier2"):
        with open(ROOT / "results" / family / "results.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r["eta"] != "0.8":
                    continue
                st = r["structure"]
                if st == "chain" and r["phi"] not in ("1.0", "1"):
                    continue
                if st not in ("chain", "full", "generalist"):
                    continue
                out[(r["instance_id"], st, float(r["m"]), r["method"])] = \
                    float(r["twt"])
    return out


_BASELINE_TWT: dict = {}


def load_baseline_twt():
    """(instance_id, structure, m, method) -> twt from the comparator arm.

    The comparator arm is the released engine with the scalar eta, run on the
    inputs this experiment uses. Comparing against it tests the new code path
    and nothing else.
    """
    if _BASELINE_TWT:
        return _BASELINE_TWT
    if not BASELINE_CSV.exists():
        raise SystemExit("comparator arm missing; run --baseline first")
    with open(BASELINE_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["structure"] == "dedicated":
                continue
            _BASELINE_TWT[(r["instance_id"], r["structure"], float(r["m"]),
                           r["method"])] = float(r["twt"])
    return _BASELINE_TWT


def repro_check(instances, verbose=True, against="baseline"):
    """Constant-0.8 per-technician table must reproduce the scalar-eta rows.

    ``against='baseline'`` compares with the released engine run on the same
    inputs: this is the regression gate on the new code path. ``against=
    'released'`` compares with the released result files, which is a check on
    the inputs rather than on the code.
    """
    ref_tbl = (load_released_twt() if against == "released"
               else load_baseline_twt())
    cells = (RELEASED_CELLS if against == "released"
             else [(st, phi, m, "baseline")
                   for (st, phi) in STRUCTS for m in MS])
    checked = mism = missing = 0
    details = []
    for (campus, size, inst_path) in instances:
        with open(inst_path) as f:
            inst = json.load(f)
        trades = trades_of(campus)
        for (structure, phi, m, _fam) in cells:
            ov = released_overlay(campus, structure, phi, m)
            const = constant_eta_tech(ov["technicians"], trades, 0.8)
            rows = run_cell(inst, campus, size, structure, phi, m,
                            "const080", const, ALL_RULES, check_primary=True)
            for row in rows:
                key = (row["instance_id"], structure, m, row["method"])
                ref = ref_tbl.get(key)
                if ref is None:
                    missing += 1
                    details.append(("MISSING", key))
                    continue
                checked += 1
                if float(row["twt"]) != ref:
                    mism += 1
                    details.append(("MISMATCH", key, repr(float(row["twt"])),
                                    repr(ref)))
    if verbose:
        print("[regression vs %s] %d rows compared, %d mismatches, %d missing"
              % (against, checked, mism, missing))
        for d in details[:12]:
            print("   ", d)
    return checked, mism, missing, details


# --------------------------------------------------------------------------- #
# Comparator arm: the released engine with the scalar eta, current inputs     #
# --------------------------------------------------------------------------- #
def _baseline_shard(cfg):
    dst = BASELINE_SHARDS / (cfg["instance_id"] + ".json")
    if dst.exists():
        return 0
    from env.engine import PairDispatchEnv
    from env.validator2 import validate as validate2
    with open(cfg["path"]) as f:
        inst = json.load(f)
    campus = cfg["campus"]
    rows = []
    for (structure, phi, eta, m) in BASELINE_CELLS:
        ov = rd.overlay_for(campus, structure, phi, eta, m)
        for meth in ALL_RULES:
            env = PairDispatchEnv(inst, ov)
            sched = env.run_selector(get_selector(meth), method=meth,
                                     seed=RULE_SEED)
            res = validate2(inst, sched, ov)
            mt = res["metrics"]
            rows.append({
                "instance_id": cfg["instance_id"], "campus": campus,
                "size": cfg["size"], "structure": structure, "phi": phi,
                "eta": eta, "m": m, "method": meth, "twt": mt["WWT"],
                "makespan": mt["makespan"], "mean_flow": mt["mean_flow"],
                "breach_share": mt["breach_share"],
                "validator_ok": int(bool(res["feasible"])),
            })
    tmp = dst.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(rows, f)
    os.replace(tmp, dst)
    return len(rows)


def run_baseline(workers):
    import multiprocessing as mp
    cfgs = [{"instance_id": r["id"], "campus": int(r["campus"]),
             "size": int(r["size_class"]),
             "path": str(INST_ROOT / r["path"])} for r in instance_index()]
    BASELINE_SHARDS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_baseline_shard, cfgs), 1):
            if i % 200 == 0:
                print("  %d/%d %.0fs" % (i, len(cfgs), time.time() - t0),
                      flush=True)
    rows = []
    for p in sorted(BASELINE_SHARDS.glob("*.json")):
        with open(p) as f:
            rows += json.load(f)
    fields = ["instance_id", "campus", "size", "structure", "phi", "eta", "m",
              "method", "twt", "makespan", "mean_flow", "breach_share",
              "validator_ok"]
    with open(BASELINE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    n_bad = sum(1 for r in rows if not r["validator_ok"])
    print("comparator arm: %d rows -> %s (%d infeasible), %.0fs"
          % (len(rows), BASELINE_CSV, n_bad, time.time() - t0))
    return len(rows)


def validator_independence_check(campus, size, inst_path, verbose=True):
    """The checker must re-derive durations from the per-technician table.

    A schedule that validates under its own table must be rejected when one
    entry of that table is changed, otherwise the checker is not reading it.
    """
    with open(inst_path) as f:
        inst = json.load(f)
    trades = trades_of(campus)
    ov = released_overlay(campus, "full", None, 0.6)
    lo, hi = DISTRIBUTIONS["u60"]
    tab = draw_eta_tech(ov["technicians"], trades, DRAW_SEEDS[0], campus,
                        0.6, lo, hi)
    ov2 = build_etatech_overlay(campus, "full", None, 0.6, "probe", tab)
    env = EtaTechEnv(inst, ov2, tab)
    sched = env.run_selector(get_selector("edd"), method="edd",
                             seed=RULE_SEED)
    ok = validate_etatech(inst, sched, ov2, tab)
    if not ok["feasible"]:
        print("[validator] baseline schedule reported INFEASIBLE:",
              ok["violations"][:3])
        return False
    # Perturb the entry of a technician that actually ran secondary work.
    prim = {t["id"]: t["primary"] for t in ov2["technicians"]}
    wo = {w["id"]: w for w in inst["work_orders"]}
    target = None
    for a in sched["assignments"]:
        g = wo[a["wo"]]["trade"]
        if prim[a["tech"]] != g:
            target = (a["tech"], g)
            break
    if target is None:
        print("[validator] no secondary assignment in the probe schedule")
        return False
    bad = dict(tab)
    bad[target] = bad[target] * 0.5
    res = validate_etatech(inst, sched, ov2, bad)
    if verbose:
        print("[validator] perturbing eta(%s, %s) %.4f -> %.4f: feasible=%s, "
              "%d violations" % (target[0], target[1], tab[target],
                                 bad[target], res["feasible"],
                                 len(res["violations"])))
    return (not res["feasible"]) and len(res["violations"]) > 0


# --------------------------------------------------------------------------- #
# Full plan                                                                   #
# --------------------------------------------------------------------------- #
def instance_index():
    rows = rd._replay_rows(VERDICT_CAMPUSES)
    assert len(rows) == N_VERDICT_INSTANCES, (
        "expected %d verdict instances, got %d"
        % (N_VERDICT_INSTANCES, len(rows)))
    return rows


def build_shards():
    """One shard per (instance, crew multiplier): 18 cells, 144 rows."""
    shards = []
    for r in instance_index():
        for m in MS:
            shards.append({
                "instance_id": r["id"], "campus": int(r["campus"]),
                "size": int(r["size_class"]),
                "path": str(INST_ROOT / r["path"]), "m": m,
                "shard_id": "%s__m%03d" % (r["id"], int(round(m * 100))),
            })
    return shards


def _run_shard(cfg):
    dst = SHARD_DIR / (cfg["shard_id"] + ".json")
    if dst.exists():
        return 0
    with open(cfg["path"]) as f:
        inst = json.load(f)
    campus, m = cfg["campus"], cfg["m"]
    trades = trades_of(campus)
    rows = []
    for dist, (lo, hi) in sorted(DISTRIBUTIONS.items()):
        for seed in DRAW_SEEDS:
            ref_ov = released_overlay(campus, "chain", 1.0, m)
            tab = draw_eta_tech(ref_ov["technicians"], trades, seed, campus,
                                m, lo, hi)
            tag = "%s_s%d" % (dist, seed)
            for structure, phi in STRUCTS:
                rows += run_cell(inst, campus, cfg["size"], structure, phi,
                                 m, tag, tab, ALL_RULES, dist=dist,
                                 seed=seed)
    tmp = dst.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"shard_id": cfg["shard_id"], "rows": rows}, f)
    os.replace(tmp, dst)
    return len(rows)


def run_full(workers):
    import multiprocessing as mp
    shards = build_shards()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    print("shards: %d (expect %d rows)"
          % (len(shards), len(shards) * len(DISTRIBUTIONS) * len(DRAW_SEEDS)
             * len(STRUCTS) * len(ALL_RULES)))
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_shard, shards), 1):
            if i % 100 == 0:
                el = time.time() - t0
                print("  %d/%d %.0fs eta %.0fs"
                      % (i, len(shards), el, el / i * (len(shards) - i)),
                      flush=True)
    elapsed = time.time() - t0
    n = merge()
    meta = {
        "family": "e13_etatech",
        "note": "per-technician secondary-skill efficiency; post-hoc "
                "sensitivity, not merged into tier1",
        "distributions": {k: list(v) for k, v in DISTRIBUTIONS.items()},
        "draw_seeds": DRAW_SEEDS,
        "structures": [s for s, _ in STRUCTS], "crew_multipliers": MS,
        "methods": ALL_RULES, "rule_seed": RULE_SEED,
        "n_instances": N_VERDICT_INSTANCES, "n_rows": n,
        "y1_corpus": "frozen v1.0" if is_frozen() else "sibling repository",
        "elapsed_s": elapsed, "workers": workers,
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("elapsed %.0fs, %d rows" % (elapsed, n))


def merge():
    rows = []
    for p in sorted(SHARD_DIR.glob("*.json")):
        with open(p) as f:
            rows += json.load(f).get("rows", [])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    print("merged -> %d rows -> %s" % (len(rows), OUT_DIR / "results.csv"))
    return len(rows)


# --------------------------------------------------------------------------- #
def smoke_instances(n=2):
    rows_idx = rd._replay_rows([9])
    by_size = {}
    for r in rows_idx:
        by_size.setdefault(int(r["size_class"]), []).append(r)
    picks = []
    for size in (150, 400)[:n]:
        r = sorted(by_size[size], key=lambda x: x["id"])[0]
        picks.append((9, size, str(INST_ROOT / r["path"])))
    return picks


def smoke():
    picks = smoke_instances()
    print("smoke instances:", [Path(p).name for (_c, _s, p) in picks])
    checked, mism, missing, _ = repro_check(picks)
    if mism or missing or checked == 0:
        print("[regression] FAILED; STOP.")
        return False
    if not validator_independence_check(*picks[0]):
        print("[validator] independence check FAILED; STOP.")
        return False
    all_rows = []
    for (campus, size, inst_path) in picks:
        with open(inst_path) as f:
            inst = json.load(f)
        trades = trades_of(campus)
        for dist, (lo, hi) in sorted(DISTRIBUTIONS.items()):
            seed = DRAW_SEEDS[0]
            ref_ov = released_overlay(campus, "chain", 1.0, 0.6)
            tab = draw_eta_tech(ref_ov["technicians"], trades, seed, campus,
                                0.6, lo, hi)
            print("draw %s seed=%d campus=%d m=0.6: |entries|=%d mean=%.4f "
                  "(min=%.3f max=%.3f)"
                  % (dist, seed, campus, len(tab), table_mean(tab),
                     min(tab.values()), max(tab.values())))
            for structure, phi in STRUCTS:
                all_rows += run_cell(inst, campus, size, structure, phi, 0.6,
                                     "%s_s%d" % (dist, seed), tab, ALL_RULES,
                                     dist=dist, seed=seed, check_primary=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke_csv = OUT_DIR / "smoke_results.csv"
    with open(smoke_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    n_feas = sum(r["validator_ok"] for r in all_rows)
    print("[smoke] wrote %d rows -> %s (%d validator_ok)"
          % (len(all_rows), smoke_csv, n_feas))
    import pandas as pd
    df = pd.read_csv(smoke_csv)
    print("[smoke] pandas load ok: shape=%s" % (df.shape,))
    print(df[["instance_id", "structure", "dist", "method", "twt",
              "validator_ok", "mean_eta", "mean_eta_active"]]
          .head(20).to_string(index=False))
    return n_feas == len(all_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--repro-check", action="store_true")
    ap.add_argument("--repro-all", action="store_true",
                    help="regression check over all 763 instances")
    ap.add_argument("--baseline", action="store_true",
                    help="released engine, scalar eta, current inputs")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.smoke:
        sys.exit(0 if smoke() else 1)
    if args.repro_check:
        picks = smoke_instances()
        _, mism, missing, _ = repro_check(picks)
        sys.exit(0 if (mism == 0 and missing == 0) else 1)
    if args.baseline:
        run_baseline(args.workers)
        return
    if args.repro_all:
        sys.exit(0 if repro_all(args.workers) else 1)
    if args.full:
        run_full(args.workers)
        return
    if args.merge:
        merge()
        return
    ap.print_help()


def _repro_one_baseline(cfg):
    c, m_, ms_, det = repro_check([(cfg["campus"], cfg["size"], cfg["path"])],
                                  verbose=False, against="baseline")
    return c, m_, ms_, det


def _repro_one_released(cfg):
    c, m_, ms_, det = repro_check([(cfg["campus"], cfg["size"], cfg["path"])],
                                  verbose=False, against="released")
    return c, m_, ms_, det


def repro_all(workers):
    """Both checks over all 763 instances: the code path against the
    comparator arm, and the current inputs against the released files."""
    import multiprocessing as mp
    cfgs = [{"campus": int(r["campus"]), "size": int(r["size_class"]),
             "path": str(INST_ROOT / r["path"])} for r in instance_index()]
    ctx = mp.get_context("fork")
    out = {"constant_eta": 0.8, "n_instances": len(cfgs),
           "y1_corpus": "frozen v1.0" if is_frozen() else "sibling repository"}
    for name, fn in (("baseline", _repro_one_baseline),
                     ("released", _repro_one_released)):
        t0 = time.time()
        tot = mis = mss = 0
        diverging = []
        with ctx.Pool(workers) as pool:
            for i, (c, m_, ms_, det) in enumerate(
                    pool.imap_unordered(fn, cfgs), 1):
                tot += c
                mis += m_
                mss += ms_
                diverging += [list(map(str, d)) for d in det]
                if i % 300 == 0:
                    print("  [%s] %d/%d %.0fs"
                          % (name, i, len(cfgs), time.time() - t0), flush=True)
        print("[regression vs %s] ALL instances: %d rows compared, "
              "%d mismatches, %d missing, %.0fs"
              % (name, tot, mis, mss, time.time() - t0))
        out[name] = {"rows_compared": tot, "mismatches": mis,
                     "missing": mss, "detail": diverging}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "regression_check.json", "w") as f:
        json.dump(out, f, indent=1)
    return all(out[k]["mismatches"] == 0 and out[k]["missing"] == 0
               for k in ("baseline", "released"))


if __name__ == "__main__":
    main()
