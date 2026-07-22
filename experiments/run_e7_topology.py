#!/usr/bin/env python
"""E-B topology sensitivity runner.

Tests three otherwise-arbitrary choices in the released workload-adjacent
CHAIN overlay. This runner rebuilds the chain under alternatives and scores it
with the SAME v2 pair engine + validator + ranked rules used for tier1, so
every number is directly comparable to the released CHAIN(1.0) cell at
m = 0.6, eta in {1.0, 0.8}, verdict campuses {5, 9, 10, 12}.

Variants (variant column in the output):
  chain_adj      : chain order follows expert trade-skill adjacency clusters
                   (overlays.topology_overlays) instead of descending p95 hours.
                   phi = 1.0; eta in {1.0, 0.8}; m = 0.6. structure "chain_adj".
  perm<seed>     : 10 seeded random chain cycles, seeds 20260801..20260810.
                   phi = 1.0; eta = 1.0; m = 0.6. structure "chain".
  tsel<seed>     : CHAIN(0.5) where the technicians receiving the second skill
                   are drawn at random within trade (not lowest-indexed),
                   seeds 20260821..20260823. eta = 1.0; m = 0.6. structure
                   "chain".

Same-budget sparse-topology controls (one secondary skill per technician, so
B = headcount = B(CHAIN(1.0)) by construction; only the trade-level secondary
map varies):
  pairs          : disjoint reciprocal trade pairs in workload order (odd K:
                   the last three trades form one 3-cycle). eta in {1.0, 0.8}.
  star           : hub topology; every trade's secondary is the largest-
                   workload trade, the hub's is the second. eta in {1.0, 0.8}.
  feas           : licence-constrained chain over the expert adjacency order;
                   arcs into licence-gated trades (D10 D20 D40 D50) are
                   prohibited. eta in {1.0, 0.8}.
  rand1<seed>    : seeded uniform random one-secondary graph per trade,
                   connectivity unconstrained, seeds 20260901..20260903.
                   eta = 1.0.

Comparator for every row: the released overlay built by overlays.build for the
same (campus, structure, phi, eta, m). In-runner asserts (abort loudly) pin the
one-variable-changed invariant per config:
  (a) headcount == released headcount for the same (campus, m);
  (b) budget_B  == released CHAIN(1.0) [adj/perm] resp. CHAIN(0.5) [tsel] B;
  (c) every technician's primary trade unchanged.

Output: results/e7_topology/results.csv (+ .parquet), tier1 schema plus a
`variant` column. Nothing in analysis/ globs this path (checked at prep time);
a dedicated e7 analysis must read it explicitly.

Usage (CPU only; one thread per worker):
  PYTHONPATH=.:vendor OMP_NUM_THREADS=1 python experiments/run_e7_topology.py \
      [--variants chain_adj,perm,tsel] [--workers 22] [--smoke] [--merge]
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

import multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
Y1 = Path(os.environ.get("FMWOS_Y1_ROOT", ROOT.parent / "FM-Scheduling"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from env.engine import PairDispatchEnv                       # noqa: E402
from env.validator2 import validate as validate2            # noqa: E402
from methods.rules import get_selector, RANKED_RULES, ALL_RULES  # noqa: E402
from overlays.build import build_overlay, chain_order, load_crews  # noqa: E402
from overlays import topology_overlays as rt                 # noqa: E402

CAP = Y1 / "results/p1_calib/capacity.csv"
INST_ROOT = Y1 / "data/processed/instances"
OUT_DIR = ROOT / "results" / "e7_topology"

RULE_SEED = 301
VERDICT_CAMPUSES = [5, 9, 10, 12]
SIZES = [150, 400]
M = 0.6

CHAIN_ADJ_ETAS = [1.0, 0.8]
PERM_SEEDS = list(range(20260801, 20260811))   # 10 permutations
PERM_ETA = 1.0
TSEL_SEEDS = [20260821, 20260822, 20260823]    # 3 technician-selection draws
TSEL_PHI = 0.5
TSEL_ETA = 1.0

SPARSE_ETAS = [1.0, 0.8]                       # pairs / star / feas
RAND1_SEEDS = [20260901, 20260902, 20260903]   # 3 random one-secondary graphs
RAND1_ETA = 1.0
SPARSE_KINDS = ("pairs", "star", "feas", "feasnorm", "rand1")

SMOKE_CAMPUS = 5
SMOKE_INSTANCES = ["c05_replay_150_0100", "c05_replay_400_0100"]

FIELDS = ["variant", "instance_id", "campus", "size", "track", "structure",
          "phi", "eta", "m", "u_target", "u_realized", "method", "seed",
          "twt", "makespan", "mean_flow", "breach_share", "breach_p1",
          "breach_p2", "breach_p3", "breach_p4", "decisions",
          "latency_ms_per_decision", "replans", "validator_ok", "runtime_s"]

# set in main()
RULE_SET = ALL_RULES


# --------------------------------------------------------------------------- #
def _replay_rows(campuses):
    rows = []
    with open(INST_ROOT / "index.csv", newline="") as f:
        for r in csv.DictReader(f):
            if (r["track"] == "replay" and r["split"] == "test"
                    and int(r["campus"]) in campuses
                    and int(r["size_class"]) in SIZES):
                rows.append(r)
    return rows


def build_configs(variants, smoke=False, campuses=None, ms=None):
    """One config = (variant, instance, eta cell, m). ``campuses``/``ms``
    override the defaults (held-out transfer runs); shard names carry an
    m tag only when m != 0.6, so the released shards keep their ids."""
    campuses = [SMOKE_CAMPUS] if smoke else (campuses or VERDICT_CAMPUSES)
    ms = ms or [M]
    rows = _replay_rows(campuses)
    if smoke:
        rows = [r for r in rows if r["id"] in SMOKE_INSTANCES]
    rows.sort(key=lambda r: (int(r["campus"]), int(r["size_class"]), r["id"]))

    configs = []
    for r in rows:
      for m_val in ms:
        mtag = "" if abs(m_val - 0.6) < 1e-9 else "_m%03d" % round(m_val * 100)
        base = {"instance_id": r["id"], "path": str(INST_ROOT / r["path"]),
                "campus": int(r["campus"]), "size": int(r["size_class"]),
                "track": "replay", "m": m_val}
        if "chain_adj" in variants:
            etas = [1.0] if smoke else CHAIN_ADJ_ETAS
            for eta in etas:
                configs.append({**base, "variant": "chain_adj", "kind": "adj",
                                "structure": "chain_adj", "phi": 1.0,
                                "eta": eta, "perm_seed": None,
                                "tech_seed": None,
                                "config_id": "%s__chain_adj_eta%03d%s"
                                % (r["id"], int(round(eta * 100)), mtag)})
        if "perm" in variants and not smoke:
            for s in PERM_SEEDS:
                configs.append({**base, "variant": "perm%d" % s, "kind": "perm",
                                "structure": "chain", "phi": 1.0,
                                "eta": PERM_ETA, "perm_seed": s,
                                "tech_seed": None,
                                "config_id": "%s__perm%d%s" % (r["id"], s, mtag)})
        if "tsel" in variants and not smoke:
            for s in TSEL_SEEDS:
                configs.append({**base, "variant": "tsel%d" % s, "kind": "tsel",
                                "structure": "chain", "phi": TSEL_PHI,
                                "eta": TSEL_ETA, "perm_seed": None,
                                "tech_seed": s,
                                "config_id": "%s__tsel%d%s" % (r["id"], s, mtag)})
        for kind in ("pairs", "star", "feas", "feasnorm"):
            if kind in variants:
                etas = [1.0] if smoke else SPARSE_ETAS
                for eta in etas:
                    configs.append({**base, "variant": kind, "kind": kind,
                                    "structure": kind, "phi": 1.0, "eta": eta,
                                    "perm_seed": None, "tech_seed": None,
                                    "config_id": "%s__%s_eta%03d%s"
                                    % (r["id"], kind, int(round(eta * 100)),
                                       mtag)})
        if "opt" in variants:
            for eta in ([1.0] if smoke else SPARSE_ETAS):
                configs.append({**base, "variant": "opt", "kind": "opt",
                                "structure": "opt", "phi": 1.0, "eta": eta,
                                "perm_seed": None, "tech_seed": None,
                                "config_id": "%s__opt_eta%03d%s"
                                % (r["id"], int(round(eta * 100)), mtag)})
        if "rand1" in variants:
            seeds = RAND1_SEEDS[:1] if smoke else RAND1_SEEDS
            for s in seeds:
                configs.append({**base, "variant": "rand1_%d" % s,
                                "kind": "rand1", "structure": "rand1",
                                "phi": 1.0, "eta": RAND1_ETA,
                                "perm_seed": None, "tech_seed": s,
                                "config_id": "%s__rand1_%d%s" % (r["id"], s, mtag)})
    return configs


def _build_variant_overlay(cfg, crews):
    """Build the variant overlay for this config."""
    if cfg["kind"] == "adj":
        order = rt.adjacency_order(crews)
        return rt.build_chain_variant(cfg["campus"], crews, cfg["phi"],
                                      cfg["eta"], cfg["m"], order=order,
                                      struct_label="chain_adj",
                                      variant=cfg["variant"])
    if cfg["kind"] == "perm":
        return rt.build_chain_variant(cfg["campus"], crews, cfg["phi"],
                                      cfg["eta"], cfg["m"],
                                      perm_seed=cfg["perm_seed"],
                                      struct_label="chain",
                                      variant=cfg["variant"])
    if cfg["kind"] == "tsel":
        return rt.build_chain_variant(cfg["campus"], crews, cfg["phi"],
                                      cfg["eta"], cfg["m"],
                                      tech_seed=cfg["tech_seed"],
                                      struct_label="chain",
                                      variant=cfg["variant"])
    if cfg["kind"] in ("pairs", "star", "rand1"):
        order = chain_order(crews)                 # released workload order
        sigma = {"pairs": rt.sigma_pairs,
                 "star": rt.sigma_star}.get(cfg["kind"])
        sigma = (sigma(order) if sigma is not None
                 else rt.sigma_rand1(order, cfg["tech_seed"]))
        return rt.build_sigma_variant(cfg["campus"], crews, sigma, cfg["eta"],
                                      cfg["m"], order=order,
                                      struct_label=cfg["structure"],
                                      variant=cfg["variant"])
    if cfg["kind"] == "opt":
        f = ROOT / "results" / "e11_optsigma" / (
            "opt_sigma_c%02d.json" % cfg["campus"])
        d = json.load(open(f))
        return rt.build_sigma_variant(cfg["campus"], crews, d["sigma"],
                                      cfg["eta"], cfg["m"],
                                      order=d["order"],
                                      struct_label="opt", variant="opt")
    if cfg["kind"] in ("feas", "feasnorm"):
        order = rt.adjacency_order(crews)          # expert adjacency clusters
        lic = (rt.LICENSED_NORMAL if cfg["kind"] == "feasnorm"
               else rt.LICENSED_TRADES)
        sigma = rt.sigma_feas(order, licensed=lic)
        return rt.build_sigma_variant(cfg["campus"], crews, sigma, cfg["eta"],
                                      cfg["m"], order=order,
                                      struct_label=cfg["kind"],
                                      variant=cfg["variant"])
    raise ValueError(cfg["kind"])


def _assert_invariants(cfg, ov, crews):
    """One-variable-changed guard. Comparator = released overlay for the same
    (campus, structure, phi, eta, m). Aborts loudly on any drift."""
    ref_phi = TSEL_PHI if cfg["kind"] == "tsel" else 1.0
    ref = build_overlay(cfg["campus"], crews, "chain", ref_phi, cfg["eta"],
                        cfg["m"])
    c, m = cfg["campus"], cfg["m"]
    assert ov["headcount"] == ref["headcount"], (
        "HEADCOUNT drift c%d m%s: %d != released %d"
        % (c, m, ov["headcount"], ref["headcount"]))
    assert ov["budget_B"] == ref["budget_B"], (
        "BUDGET drift c%d m%s %s: B=%d != released CHAIN(%.2f) B=%d"
        % (c, m, cfg["variant"], ov["budget_B"], ref_phi, ref["budget_B"]))
    if cfg["kind"] in ("adj", "perm", "opt") + SPARSE_KINDS:
        assert ov["budget_B"] == ov["headcount"], (
            "B != headcount at phi=1.0 c%d m%s: B=%d hc=%d"
            % (c, m, ov["budget_B"], ov["headcount"]))
    if cfg["kind"] in SPARSE_KINDS + ("opt",):
        assert all(len(t["skills"]) == 2 for t in ov["technicians"]), (
            "sparse control %s c%d m%s: a technician does not hold exactly "
            "one secondary skill" % (cfg["variant"], c, m))
    prim_ov = {t["id"]: t["primary"] for t in ov["technicians"]}
    prim_ref = {t["id"]: t["primary"] for t in ref["technicians"]}
    assert prim_ov == prim_ref, (
        "PRIMARY-TRADE drift c%d m%s %s" % (c, m, cfg["variant"]))


def _row(cfg, method, seed, sched, res):
    m = res["metrics"]
    pp = m["per_priority_breach_share"]
    decisions = sched.get("decisions")
    wall = sched.get("wall_seconds")
    mean_ms = (1000.0 * wall / decisions) if (decisions and wall) else None
    return {"variant": cfg["variant"], "instance_id": cfg["instance_id"],
            "campus": cfg["campus"], "size": cfg["size"], "track": "replay",
            "structure": cfg["structure"], "phi": cfg["phi"],
            "eta": cfg["eta"], "m": cfg["m"], "u_target": None,
            "u_realized": None, "method": method, "seed": seed,
            "twt": m["WWT"], "makespan": m["makespan"],
            "mean_flow": m["mean_flow"], "breach_share": m["breach_share"],
            "breach_p1": pp.get(1), "breach_p2": pp.get(2),
            "breach_p3": pp.get(3), "breach_p4": pp.get(4),
            "decisions": decisions, "latency_ms_per_decision": mean_ms,
            "replans": None, "validator_ok": int(bool(res["feasible"])),
            "runtime_s": wall}


def run_config(cfg):
    t0 = time.perf_counter()
    try:
        dst = OUT_DIR / "shards" / (cfg["config_id"] + ".json")
        old_rows = {}
        if dst.exists():
            try:
                with open(dst) as f:
                    old_rows = json.load(f).get("rows", {}) or {}
            except Exception:
                pass
        todo = [meth for meth in RULE_SET if meth not in old_rows]
        if not todo:
            return {"config_id": cfg["config_id"], "ok": True, "skipped": True}

        crews = load_crews(CAP, cfg["campus"])
        ov = _build_variant_overlay(cfg, crews)
        _assert_invariants(cfg, ov, crews)       # abort loudly on drift

        with open(cfg["path"]) as f:
            inst = json.load(f)
        env = PairDispatchEnv(inst, ov)
        out_rows = {}
        for meth in todo:
            sched = env.run_selector(get_selector(meth), method=meth,
                                     seed=RULE_SEED)
            res = validate2(inst, sched, ov)
            out_rows[meth] = _row(cfg, meth, RULE_SEED, sched, res)

        out_rows = {**old_rows, **out_rows}
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"config_id": cfg["config_id"], "rows": out_rows,
                       "budget_B": ov["budget_B"], "headcount": ov["headcount"],
                       "chain_order": ov["chain_order"]}, f)
        os.replace(tmp, dst)
        bad = [k for k, r in out_rows.items() if r.get("validator_ok") == 0]
        return {"config_id": cfg["config_id"], "ok": True, "infeasible": bad,
                "wall": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"config_id": cfg["config_id"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


def check_sparse(variants):
    """Pre-launch guard for the sparse controls: build every overlay, run the
    comparator asserts, and record topology descriptors + the resolved-config
    diff against the released CHAIN(1.0) overlay."""
    from overlays.build import scaled_crews
    out = {}
    kinds = [v for v in variants if v in SPARSE_KINDS] or list(SPARSE_KINDS)
    for campus in VERDICT_CAMPUSES:
        crews = load_crews(CAP, campus)
        crew_of = scaled_crews(crews, M)
        ref = build_overlay(campus, crews, "chain", 1.0, 1.0, M)
        for kind in kinds:
            builds = []
            if kind == "rand1":
                order = chain_order(crews)
                for s in RAND1_SEEDS:
                    builds.append(("rand1_%d" % s,
                                   rt.sigma_rand1(order, s), order))
            elif kind == "feas":
                order = rt.adjacency_order(crews)
                builds.append(("feas", rt.sigma_feas(order), order))
            else:
                order = chain_order(crews)
                fn = {"pairs": rt.sigma_pairs, "star": rt.sigma_star}[kind]
                builds.append((kind, fn(order), order))
            for name, sigma, order in builds:
                ov = rt.build_sigma_variant(campus, crews, sigma, 1.0, M,
                                            order=order, struct_label=kind,
                                            variant=name)
                cfg = {"kind": kind, "variant": name, "campus": campus,
                       "m": M, "eta": 1.0}
                _assert_invariants(cfg, ov, crews)
                d = rt.topology_descriptors(sigma, order, crew_of)
                out.setdefault("c%02d" % campus, {})[name] = {
                    "headcount": ov["headcount"],
                    "headcount_released_chain": ref["headcount"],
                    "budget_B": ov["budget_B"],
                    "budget_released_chain": ref["budget_B"],
                    "primary_map_identical": True,   # asserted above
                    "sigma": sigma, **d}
                print("c%02d %-14s B=%3d (ref %3d) comps=%d coverage=%.2f "
                      "indeg_max=%d uncovered=%s"
                      % (campus, name, ov["budget_B"], ref["budget_B"],
                         d["weak_components"], d["coverage_share"],
                         d["in_degree_max"], ",".join(d["uncovered_trades"])
                         or "-"), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "sparse_config_check.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote %s" % (OUT_DIR / "sparse_config_check.json"))


def merge():
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
    rows.sort(key=lambda r: (r["variant"], r["campus"], r["size"],
                             r["instance_id"], r["eta"], r["method"]))
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
    print("merged %d rows (%d infeasible) -> %s"
          % (len(rows), n_bad, OUT_DIR / "results.csv"))


def main():
    global RULE_SET
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="chain_adj,perm,tsel")
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--smoke", action="store_true",
                    help="chain_adj only, 2 campus-5 instances, ranked rules")
    ap.add_argument("--ranked-only", action="store_true",
                    help="7 ranked rules (no random)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--campuses", default=None,
                    help="comma list overriding the verdict campuses "
                         "(e.g. 1,2 for the held-out transfer run)")
    ap.add_argument("--ms", default=None,
                    help="comma list of crew multipliers (default 0.6)")
    ap.add_argument("--check", action="store_true",
                    help="build every sparse-control overlay, run the "
                         "comparator asserts, and write topology descriptors "
                         "(no episodes)")
    args = ap.parse_args()

    if args.merge:
        merge()
        return

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.check:
        check_sparse(variants)
        return
    if args.smoke:
        variants = [v for v in variants
                    if v in ("chain_adj",) + SPARSE_KINDS] or ["chain_adj"]
        RULE_SET = RANKED_RULES
    elif args.ranked_only:
        RULE_SET = RANKED_RULES

    campuses = ([int(x) for x in args.campuses.split(",")]
                if args.campuses else None)
    ms = ([float(x) for x in args.ms.split(",")] if args.ms else None)
    configs = build_configs(variants, smoke=args.smoke, campuses=campuses,
                            ms=ms)
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
        if not set(RULE_SET) <= have:
            pending.append(c)

    print("e7_topology: variants=%s configs=%d pending=%d rules=%d workers=%d "
          "smoke=%s" % (variants, len(configs), len(pending), len(RULE_SET),
                        args.workers, args.smoke), flush=True)
    if not pending:
        merge()
        return

    t0 = time.time()
    done = errs = 0
    if args.smoke or args.workers <= 1:
        results = (run_config(c) for c in pending)
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(args.workers)
        results = pool.imap_unordered(run_config, pending)
    walls = []
    for res in results:
        done += 1
        if not res.get("ok"):
            errs += 1
            print("[ERR] %s: %s\n%s" % (res["config_id"], res.get("error"),
                                        res.get("traceback", "")), flush=True)
        elif res.get("infeasible"):
            print("[INFEASIBLE] %s %s" % (res["config_id"],
                                          res["infeasible"]), flush=True)
        if res.get("wall"):
            walls.append(res["wall"])
        if done % 200 == 0 or done == len(pending):
            print("  %d/%d %.0fs (%d err)"
                  % (done, len(pending), time.time() - t0, errs), flush=True)
    if not (args.smoke or args.workers <= 1):
        pool.close()
        pool.join()
    if walls:
        walls.sort()
        print("per-config wall: min=%.3fs median=%.3fs max=%.3fs (n=%d)"
              % (walls[0], walls[len(walls) // 2], walls[-1], len(walls)),
              flush=True)
    merge()


if __name__ == "__main__":
    main()
