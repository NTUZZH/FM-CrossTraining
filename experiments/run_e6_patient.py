#!/usr/bin/env python
"""E-A / E6-patient runner: the patient (idling-capable) EDD variant.

The engine's non-delay protocol (no eligible technician idles while eligible
work waits) could confound the "flexibility can cost TWT under a secondary-skill
penalty" finding, because deliberate idling (waiting for a faster primary
technician) is never allowed. This runner relaxes exactly that, only for a
targeted patient rule, and measures the effect.

Comparator: plain EDD on the SAME cells (results/tier1/results.csv). Cells and
overlays are built by reusing experiments.run_dynamic verbatim, so the ONLY
intended differences from the tier1 EDD run are the rule (edd_patient vs edd)
and the output path (results/e6_patient/ vs results/tier1/).

Modes (never launches the full sweep by itself):
  --check       run the in-runner asserts + config diff only
  --smoke       run the 2-instance end-to-end smoke and print a table
  --full        the full E-A sweep (GATED; refuses unless --i-have-approval)

Usage:
  PYTHONPATH=.:vendor python experiments/run_e6_patient.py --check --smoke
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

from experiments import run_dynamic as RD            # noqa: E402
from env.engine import PairDispatchEnv, K_ORDERS_PER_TRADE   # noqa: E402
from env.validator2 import validate as validate2     # noqa: E402
from methods.rules import (PATIENT_RULES, get_selector,       # noqa: E402
                           PatientEDD)

# Patient variants of the other strong ranked rules (edd_patient is the
# released E-A run in results.csv; these are the same cells with only the
# rule changed, written to results_variants.csv).
VARIANT_RULES = ["wspt_patient", "atc_patient", "lfj_atc_patient",
                 "atc_eta_patient"]

OUT_DIR = ROOT / "results" / "e6_patient"
RULE_SEED = RD.RULE_SEED                              # 301, same as tier1

# E-A sweep cells (penalized, flexible): eta=0.8, m in {0.6,0.8},
# CHAIN(phi=1.0) and FULL.
SWEEP_CELLS = {
    ("chain", 1.0, 0.8, 0.6),
    ("chain", 1.0, 0.8, 0.8),
    ("full", None, 0.8, 0.6),
    ("full", None, 0.8, 0.8),
}
# Assert-only cells.
ETA1_CELL = ("chain", 1.0, 1.0, 0.6)      # break-even window == 0 -> == EDD
L0_CELL = ("dedicated", None, 1.0, 0.6)   # all primary -> == EDD

COUNTER_FIELDS = ["n_declines", "deliberate_wait_bh",
                  "ran_primary_after_decline", "ran_secondary_after_decline"]
FIELDS = RD.FIELDS + COUNTER_FIELDS


# --------------------------------------------------------------------------- #
def _load_inst(cfg):
    with open(cfg["path"]) as f:
        return json.load(f)


def _overlay(cfg):
    return RD.overlay_for(cfg["campus"], cfg["structure"], cfg["phi"],
                          cfg["eta"], cfg["m"])


def _cell_of(cfg):
    return (cfg["structure"], cfg["phi"], cfg["eta"], cfg["m"])


def _run_edd(inst, ov):
    env = PairDispatchEnv(inst, ov)
    return env.run_selector(get_selector("edd"), method="edd", seed=RULE_SEED)


def _run_patient(inst, ov):
    env = PairDispatchEnv(inst, ov)
    sel = PatientEDD()
    sched = env.run_selector(sel, method="edd_patient", seed=RULE_SEED)
    return sched, sel


def _row(cfg, method, sched, res, counters=None):
    r = RD._row(cfg, method, RULE_SEED, sched, res)
    for k in COUNTER_FIELDS:
        r[k] = (counters or {}).get(k)
    return r


FAMILY = "tier1"          # set from --family; "e4" = held-out campuses 1/2


def configs(cells):
    """Family configs restricted to the requested set of cells."""
    return [c for c in RD.build_configs(FAMILY) if _cell_of(c) in cells]


# --------------------------------------------------------------------------- #
# Check 3: in-runner asserts
# --------------------------------------------------------------------------- #
def _assignments_equal(a, b):
    ka = sorted((x["wo"], x["tech"], round(x["start_bh"], 6),
                 round(x["end_bh"], 6)) for x in a["assignments"])
    kb = sorted((x["wo"], x["tech"], round(x["start_bh"], 6),
                 round(x["end_bh"], 6)) for x in b["assignments"])
    return ka == kb


def check_asserts():
    print("== CHECK 3: in-runner asserts ==")

    # (a) eta = 1.0: patient == plain EDD (break-even window is zero).
    c = configs({ETA1_CELL})[0]
    inst, ov = _load_inst(c), _overlay(c)
    edd = _run_edd(inst, ov)
    pat, sel = _run_patient(inst, ov)
    assert _assignments_equal(edd, pat), \
        "(a) eta=1 patient schedule != plain EDD on %s" % c["config_id"]
    assert sel.n_declines == 0, "(a) eta=1 must have 0 declines"
    print("  (a) eta=1.0 identical to EDD  [%s]  declines=%d  OK"
          % (c["instance_id"], sel.n_declines))

    # (b) L0 (all primary): patient == plain EDD.
    c = configs({L0_CELL})[0]
    inst, ov = _load_inst(c), _overlay(c)
    edd = _run_edd(inst, ov)
    pat, sel = _run_patient(inst, ov)
    assert _assignments_equal(edd, pat), \
        "(b) L0 patient schedule != plain EDD on %s" % c["config_id"]
    assert sel.n_declines == 0, "(b) L0 must have 0 declines"
    print("  (b) L0 identical to EDD        [%s]  declines=%d  OK"
          % (c["instance_id"], sel.n_declines))

    # (c) regression: plain-EDD path still reproduces the released tier1
    #     per-instance TWT on 2 spot instances (engine change inert when off).
    tier1 = _read_tier1()
    # Prefer spot instances whose tier1 EDD TWT is nonzero, so reproduction is
    # a real test (not 0.0 == 0.0).
    allc = configs(SWEEP_CELLS)
    nz = [c for c in allc if (_tier1_twt(tier1, c, "edd") or 0.0) > 0.0]
    spot = (nz[:2] if len(nz) >= 2 else allc[:2])
    for c in spot:
        inst, ov = _load_inst(c), _overlay(c)
        edd = _run_edd(inst, ov)
        res = validate2(inst, edd, ov)
        got = res["metrics"]["WWT"]
        want = _tier1_twt(tier1, c, "edd")
        assert want is not None, "no tier1 edd row for %s" % c["config_id"]
        assert abs(got - want) <= 1e-6, \
            "(c) edd TWT drift on %s: got %.6f want %.6f" \
            % (c["config_id"], got, want)
        print("  (c) EDD reproduces tier1 TWT   [%s %s eta%.1f m%.1f]  "
              "%.4f == %.4f  OK"
              % (c["instance_id"], c["structure"], c["eta"], c["m"],
                 got, want))
    print("  ALL ASSERTS PASS\n")


def _read_tier1():
    import pandas as pd
    p = ROOT / "results" / "tier1" / "results.csv"
    return pd.read_csv(p)


def _tier1_twt(df, cfg, method):
    phi = cfg["phi"]
    sub = df[(df.instance_id == cfg["instance_id"])
             & (df.structure == cfg["structure"])
             & (df.eta == cfg["eta"]) & (df.m == cfg["m"])
             & (df.method == method)]
    if phi is None:
        sub = sub[sub.phi.isna()]
    else:
        sub = sub[sub.phi == phi]
    if len(sub) == 0:
        return None
    return float(sub.iloc[0].twt)


# --------------------------------------------------------------------------- #
# Check 2: config diff vs the tier1 EDD run for the same cell
# --------------------------------------------------------------------------- #
def _resolved_config(cfg, method):
    """The configuration a run actually resolves for one cell: cfg fields, the
    resolved overlay identity/shape, and the env/selector parameters."""
    ov = _overlay(cfg)
    return {
        "instance_id": cfg["instance_id"],
        "instance_path": cfg["path"],
        "campus": cfg["campus"], "size": cfg["size"],
        "track": cfg["track"], "structure": cfg["structure"],
        "phi": cfg["phi"], "eta": cfg["eta"], "m": cfg["m"],
        "overlay_id": ov["overlay_id"], "budget_B": ov["budget_B"],
        "headcount": ov["headcount"], "chain_order": ov["chain_order"],
        "n_technicians": len(ov["technicians"]),
        "k_orders": K_ORDERS_PER_TRADE, "reward_mode": "shaped",
        "tb_mode": "default", "check_nondelay": False,
        "seed": RULE_SEED,
        "rule": method,
    }


def config_diff():
    print("== CHECK 2: config diff (patient cell vs tier1 EDD cell) ==")
    c = configs(SWEEP_CELLS)[0]
    a = _resolved_config(c, "edd")           # what tier1 resolves
    b = _resolved_config(c, "edd_patient")   # what e6 resolves
    diffs = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    print("  cell: %s" % c["config_id"])
    print("  resolved differences (tier1 -> e6):")
    for k, (av, bv) in sorted(diffs.items()):
        print("    %-14s %r -> %r" % (k, av, bv))
    # Output path is not part of the resolved config; state it explicitly.
    print("    %-14s %r -> %r" % ("out_dir",
                                  str(ROOT / "results" / "tier1"),
                                  str(OUT_DIR)))
    only = set(diffs) | {"out_dir"}
    assert only == {"rule", "out_dir"}, \
        "UNEXPECTED config drift: %s" % (only - {"rule", "out_dir"})
    print("  ONLY the rule and the output path differ.  OK\n")
    # Dump both resolved configs for the record.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "config_diff.json", "w") as f:
        json.dump({"tier1_edd": a, "e6_patient": b,
                   "differences": {k: {"tier1": av, "e6": bv}
                                   for k, (av, bv) in diffs.items()}},
                  f, indent=2)


# --------------------------------------------------------------------------- #
# Check 5: smoke
# --------------------------------------------------------------------------- #
SMOKE_CELLS = {("chain", 1.0, 0.8, 0.6), ("full", None, 0.8, 0.6)}


def smoke(write=True):
    print("== CHECK 5: smoke (2 instances x {CHAIN(1.0), FULL} x eta0.8 m0.6) ==")
    allc = configs(SMOKE_CELLS)
    # one campus-5 replay-150 and one replay-400.
    by_size = {}
    for c in allc:
        if c["campus"] == 5:
            by_size.setdefault(c["size"], []).append(c)
    inst_ids = []
    for sz in (150, 400):
        ids = sorted({c["instance_id"] for c in by_size.get(sz, [])})
        if ids:
            inst_ids.append(ids[0])
    pick = [c for c in allc if c["instance_id"] in inst_ids]
    pick.sort(key=lambda c: (c["size"], c["instance_id"], c["structure"]))

    rows = []
    t0 = time.perf_counter()
    print("  %-22s %-6s %-5s %-4s | %10s %10s %8s | %4s %8s %4s %4s"
          % ("instance", "struct", "eta", "m", "twt_edd", "twt_pat",
             "d_twt%", "ndec", "wait_bh", "pri", "sec"))
    for c in pick:
        inst, ov = _load_inst(c), _overlay(c)
        edd = _run_edd(inst, ov)
        res_e = validate2(inst, edd, ov)
        pat, sel = _run_patient(inst, ov)
        res_p = validate2(inst, pat, ov)
        assert res_e["feasible"] and res_p["feasible"], \
            "infeasible smoke schedule on %s" % c["config_id"]
        twt_e = res_e["metrics"]["WWT"]
        twt_p = res_p["metrics"]["WWT"]
        cnt = sel.counters()
        dpct = (100.0 * (twt_p - twt_e) / twt_e) if twt_e > 0 else 0.0
        print("  %-22s %-6s %-5.1f %-4.1f | %10.3f %10.3f %+7.2f | "
              "%4d %8.2f %4d %4d"
              % (c["instance_id"], c["structure"], c["eta"], c["m"],
                 twt_e, twt_p, dpct, cnt["n_declines"],
                 cnt["deliberate_wait_bh"], cnt["ran_primary_after_decline"],
                 cnt["ran_secondary_after_decline"]))
        rows.append(_row(c, "edd", edd, res_e))
        rows.append(_row(c, "edd_patient", pat, res_p, cnt))
    dt = time.perf_counter() - t0
    print("  smoke wall: %.2fs for %d (instance,cell) pairs (%d schedules)"
          % (dt, len(pick), 2 * len(pick)))

    if write:
        _write_csv(rows, OUT_DIR / "smoke_results.csv")
        # Load it back with pandas (check the intended analysis can read it).
        import pandas as pd
        df = pd.read_csv(OUT_DIR / "smoke_results.csv")
        print("  wrote %s ; pandas loaded %d rows, %d cols; methods=%s"
              % (OUT_DIR / "smoke_results.csv", len(df), df.shape[1],
                 sorted(df.method.unique())))
        pcols = [c for c in COUNTER_FIELDS if c in df.columns]
        print("  counter columns present: %s" % pcols)
    print()
    return rows


def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})


# --------------------------------------------------------------------------- #
def full_sweep(workers):
    cells = set(SWEEP_CELLS)
    pend = configs(cells)
    suffix = "" if FAMILY == "tier1" else "_%s" % FAMILY
    print("FULL sweep: %d (instance,cell) configs -> %d schedules"
          % (len(pend), 2 * len(pend)))
    rows = []
    for i, c in enumerate(pend, 1):
        inst, ov = _load_inst(c), _overlay(c)
        edd = _run_edd(inst, ov)
        res_e = validate2(inst, edd, ov)
        pat, sel = _run_patient(inst, ov)
        res_p = validate2(inst, pat, ov)
        rows.append(_row(c, "edd", edd, res_e))
        rows.append(_row(c, "edd_patient", pat, res_p, sel.counters()))
        if i % 200 == 0:
            print("  %d/%d" % (i, len(pend)), flush=True)
    _write_csv(rows, OUT_DIR / ("results%s.csv" % suffix))
    print("wrote %s (%d rows)" % (OUT_DIR / ("results%s.csv" % suffix),
                                  len(rows)))


def variants_sweep():
    """Patient variants of the other strong ranked rules on the E-A cells.
    With --family e4 the same cells run on the held-out campuses.

    Comparators: each variant's PLAIN rule rows already exist in tier1 for
    the same cells; edd_patient is the released results.csv. Identity
    asserts (eta = 1 and L0 equal the plain rule for every variant) are
    covered by tests/test_patient_rules.py; here we assert the resolved
    config differs from the released e6 run only in the rule name.
    """
    c0 = configs(SWEEP_CELLS)[0]
    for rule in VARIANT_RULES:
        a = _resolved_config(c0, "edd_patient")
        b = _resolved_config(c0, rule)
        diff = {k for k in a if a[k] != b[k]}
        assert diff == {"rule"}, "config drift for %s: %r" % (rule, diff)

    pend = configs(set(SWEEP_CELLS))
    suffix = "" if FAMILY == "tier1" else "_%s" % FAMILY
    print("VARIANTS sweep (%s): %d (instance,cell) configs x %d rules"
          % (FAMILY, len(pend), len(VARIANT_RULES)))
    rows = []
    for i, c in enumerate(pend, 1):
        inst, ov = _load_inst(c), _overlay(c)
        for rule in VARIANT_RULES:
            env = PairDispatchEnv(inst, ov)
            sel = PATIENT_RULES[rule]()
            sched = env.run_selector(sel, method=rule, seed=RULE_SEED)
            res = validate2(inst, sched, ov)
            assert res["feasible"], "infeasible %s on %s" % (rule,
                                                            c["config_id"])
            rows.append(_row(c, rule, sched, res, sel.counters()))
        if i % 200 == 0:
            print("  %d/%d" % (i, len(pend)), flush=True)
    _write_csv(rows, OUT_DIR / ("results_variants%s.csv" % suffix))
    print("wrote %s (%d rows)"
          % (OUT_DIR / ("results_variants%s.csv" % suffix), len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--variants", action="store_true",
                    help="patient variants of the other strong rules "
                         "(results_variants.csv)")
    ap.add_argument("--i-have-approval", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--family", default="tier1", choices=["tier1", "e4"])
    args = ap.parse_args()

    global FAMILY
    FAMILY = args.family

    if args.check:
        config_diff()
        check_asserts()
    if args.smoke:
        smoke()
    if args.full:
        if not args.i_have_approval:
            sys.exit("REFUSED: --full requires --i-have-approval.")
        full_sweep(args.workers)
    if args.variants:
        variants_sweep()
    if not (args.check or args.smoke or args.full or args.variants):
        ap.error("pick at least one of --check / --smoke / --full "
                 "/ --variants")


if __name__ == "__main__":
    main()
