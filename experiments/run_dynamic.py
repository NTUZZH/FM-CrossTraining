#!/usr/bin/env python
"""Dynamic-evaluation runner v2: families tier1 / tier2 / e3 / e4
(grids pre-declared in protocol/Y2_protocol.md 7).

One task = one (instance, overlay cell) x the requested method set. Shards
are incremental and resumable (Y1 p4_dyneval idiom): a shard accumulates
method rows across invocations, so rules can run before training finishes
and policies/rolling are added later without recomputing anything.

Methods
-------
  rules  : 7 ranked (edd wspt atc pfifo mor lfj_atc atc_eta) + random,
           seed 301, through the v2 pair engine.
  rl     : every checkpoint results/train/<arch>_seed<t>/best.pt, greedy,
           method names v2mlp<t> / v2attn<t> (torch CPU, 1 thread/worker).
  rollcp : rolling CP-SAT v2, tier1 only, FIRST 8 instance ids per
           (campus, size, overlay cell).
Scoring: env.validator2 ONLY (independent checker). Every row records
validator_ok; infeasible rows are retained and flagged (they would be a
stop-the-line bug).

L0 eta-invariance: dedicated cells run once per m with eta = 1.0; the
analysis layer reuses them across eta (protocol log 6).

Usage:
  PYTHONPATH=.:vendor python experiments/run_dynamic.py --family tier1 \
      [--methods rules,rl,rollcp] [--workers 22] [--campus 5,9] [--merge]
"""
from __future__ import annotations

import argparse
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

from env.engine import PairDispatchEnv                  # noqa: E402
from env.validator2 import validate as validate2       # noqa: E402
from methods.rules import get_selector, RANKED_RULES    # noqa: E402
from overlays.build import build_overlay, load_crews    # noqa: E402

CAP = Y1 / "results/p1_calib/capacity.csv"
INST_ROOT = Y1 / "data/processed/instances"
PARAM_ROOT = Y1 / "results/p2_generator"
TRAIN_DIR = ROOT / "results/train"
E3_DATA = ROOT / "data/e3"

RULE_SEED = 301
ALL_RULES = RANKED_RULES + ["random"]

VERDICT_CAMPUSES = [5, 9, 10, 12]
HELDOUT_CAMPUSES = [1, 2]
SIZES = [150, 400]

TIER1_STRUCTS = [("dedicated", None), ("chain", 1.0), ("full", None)]
TIER2_STRUCTS = [("chain", 0.25), ("chain", 0.5), ("generalist", None)]
ETAS = [1.0, 0.8]
TIER1_MS = [1.0, 0.8, 0.6]
TIER2_MS = [1.0, 0.6]

E3_US = [0.9, 1.0, 1.1, 1.3]
E3_WINDOW_BH = 80.0
E3_N = 30
E3_SEED_BASE = 90000

ROLLCP_PER_CELL = 8

FIELDS = ["instance_id", "campus", "size", "track", "structure", "phi",
          "eta", "m", "u_target", "u_realized", "method", "seed", "twt",
          "makespan", "mean_flow", "breach_share", "breach_p1", "breach_p2",
          "breach_p3", "breach_p4", "decisions", "latency_ms_per_decision",
          "replans", "validator_ok", "runtime_s"]

_OVERLAYS: dict = {}
_POLICIES: dict = {}
_RL_SPECS: list = []          # [(method, arch, path)] set in parent
_SPEC_SPECS: list = []        # specialist ablation checkpoints


def overlay_for(campus, structure, phi, eta, m):
    key = (campus, structure, phi, eta, m)
    ov = _OVERLAYS.get(key)
    if ov is None:
        ov = build_overlay(campus, load_crews(CAP, campus), structure, phi,
                           eta, m)
        _OVERLAYS[key] = ov
    return ov


def cells_for(family):
    """(structure, phi, eta, m) cells with L0 eta-dedup."""
    structs = TIER1_STRUCTS if family in ("tier1", "e4") else TIER2_STRUCTS
    ms = TIER1_MS if family in ("tier1", "e4") else TIER2_MS
    cells = []
    for (st, phi) in structs:
        for m in ms:
            if st == "dedicated":
                cells.append((st, phi, 1.0, m))       # eta-invariant
            else:
                for eta in ETAS:
                    cells.append((st, phi, eta, m))
    return cells


def e3_cells():
    cells = []
    for (st, phi) in TIER1_STRUCTS:
        if st == "dedicated":
            cells.append((st, phi, 1.0, 1.0))
        else:
            for eta in ETAS:
                cells.append((st, phi, eta, 1.0))
    return cells


# --------------------------------------------------------------------------- #
# E3' instance generation (fixed 80 bh window utilisation sweep; own data dir)
# --------------------------------------------------------------------------- #
def _e3_id(campus, u, i):
    return "c%02d_e3_w80_u%03d_%04d" % (campus, int(round(u * 100)), i)


def _e3_seed(campus, u, i):
    ci = VERDICT_CAMPUSES.index(campus) * len(E3_US) + E3_US.index(u)
    return E3_SEED_BASE + ci * 1000 + i


def ensure_e3_instances():
    from fmwos_y1 import generator
    E3_DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for campus in VERDICT_CAMPUSES:
        with open(PARAM_ROOT / ("params_c%d.json" % campus)) as f:
            params = json.load(f)
        u0 = generator.base_utilization(params, crew_multiplier=1.0)
        for u in E3_US:
            am = float(u / u0) if u0 > 0 else 1.0
            for i in range(E3_N):
                iid = _e3_id(campus, u, i)
                path = E3_DATA / (iid + ".json")
                if not path.exists():
                    inst = generator.generate_window(
                        params, window_bh=E3_WINDOW_BH,
                        seed=_e3_seed(campus, u, i),
                        crew_multiplier=1.0, arrival_multiplier=am)
                    inst["meta"]["id"] = iid
                    inst["meta"]["track"] = "e3"
                    inst["meta"]["split"] = "test"
                    inst["meta"]["u_target"] = float(u)
                    tmp = path.with_suffix(".json.tmp")
                    with open(tmp, "w") as f:
                        json.dump(inst, f, separators=(",", ":"))
                    os.replace(tmp, path)
                with open(path) as f:
                    inst = json.load(f)
                total_p = sum(float(w["p_bh"]) for w in inst["work_orders"])
                u_real = total_p / (len(inst["technicians"]) * E3_WINDOW_BH)
                rows.append({"id": iid, "campus": campus, "u": u,
                             "path": path.name,
                             "n": len(inst["work_orders"]),
                             "u_realized": u_real})
    with open(E3_DATA / "index.json", "w") as f:
        json.dump(rows, f)
    return rows


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #
def _replay_rows(campuses):
    import csv
    rows = []
    with open(INST_ROOT / "index.csv", newline="") as f:
        for r in csv.DictReader(f):
            if (r["track"] == "replay" and r["split"] == "test"
                    and int(r["campus"]) in campuses
                    and int(r["size_class"]) in SIZES):
                rows.append(r)
    return rows


def build_configs(family):
    configs = []
    if family in ("tier1", "tier2", "e4"):
        campuses = HELDOUT_CAMPUSES if family == "e4" else VERDICT_CAMPUSES
        for r in _replay_rows(campuses):
            for (st, phi, eta, m) in cells_for(family):
                configs.append({
                    "config_id": "%s__%s" % (
                        r["id"],
                        _cell_tag(int(r["campus"]), st, phi, eta, m)),
                    "instance_id": r["id"],
                    "path": str(INST_ROOT / r["path"]),
                    "campus": int(r["campus"]), "size": int(r["size_class"]),
                    "track": "replay", "structure": st, "phi": phi,
                    "eta": eta, "m": m, "u_target": None, "u_realized": None,
                    "family": family,
                })
    elif family == "e3":
        rows = ensure_e3_instances()
        for r in rows:
            for (st, phi, eta, m) in e3_cells():
                configs.append({
                    "config_id": "%s__%s" % (
                        r["id"], _cell_tag(r["campus"], st, phi, eta, m)),
                    "instance_id": r["id"],
                    "path": str(E3_DATA / Path(r["path"]).name),
                    "campus": r["campus"], "size": r["n"],
                    "track": "e3", "structure": st, "phi": phi,
                    "eta": eta, "m": m,
                    "u_target": r["u"], "u_realized": r["u_realized"],
                    "family": family,
                })
    else:
        raise ValueError(family)
    configs.sort(key=lambda c: (c["campus"], c["size"], c["m"],
                                c["structure"], c["phi"] or 0, c["eta"],
                                c["instance_id"]))
    return configs


def _cell_tag(campus, st, phi, eta, m):
    from overlays.build import overlay_id
    return overlay_id(campus, st, phi, eta, m)


def assign_rollcp(configs, per_cell):
    from collections import defaultdict
    cells = defaultdict(list)
    for c in configs:
        key = (c["campus"], c["size"], c["structure"], c["phi"], c["eta"],
               c["m"])
        cells[key].append(c)
    for key, group in cells.items():
        group.sort(key=lambda c: c["instance_id"])
        for j, c in enumerate(group):
            c["rollcp"] = j < per_cell
    return configs


# --------------------------------------------------------------------------- #
# RL checkpoints
# --------------------------------------------------------------------------- #
def discover_rl():
    specs = []
    for d in sorted(glob.glob(str(TRAIN_DIR / "mlp_seed*"))):
        m = re.match(r".*mlp_seed(\d+)$", d)
        if m and os.path.exists(os.path.join(d, "best.pt")):
            specs.append(("v2mlp%s" % m.group(1), "mlp",
                          os.path.join(d, "best.pt"), int(m.group(1))))
    for d in sorted(glob.glob(str(TRAIN_DIR / "attn_seed*"))):
        m = re.match(r".*attn_seed(\d+)$", d)
        if m and os.path.exists(os.path.join(d, "best.pt")):
            specs.append(("v2attn%s" % m.group(1), "attn",
                          os.path.join(d, "best.pt"), int(m.group(1))))
    return specs


def discover_specialists():
    """Specialist ablation checkpoints -> (method, structure, path, seed).

    Evaluated ONLY on tier1 cells matching their training structure
    (protocol log 6: ablation, never the verdict pool)."""
    out = []
    for kind, st in (("chain", "chain"), ("full", "full")):
        pat = str(TRAIN_DIR / ("spec_%s_mlp_seed*" % kind))
        for d in sorted(glob.glob(pat)):
            m = re.match(r".*seed(\d+)$", d)
            if m and os.path.exists(os.path.join(d, "best.pt")):
                out.append(("v2spec%s%s" % (kind, m.group(1)), st,
                            os.path.join(d, "best.pt"), int(m.group(1))))
    return out


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


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _row(cfg, method, seed, sched, res, extra_replans=None):
    m = res["metrics"]
    pp = m["per_priority_breach_share"]
    decisions = sched.get("decisions")
    wall = sched.get("wall_seconds")
    mean_ms = None
    if decisions and wall is not None and decisions > 0:
        mean_ms = 1000.0 * float(wall) / float(decisions)
    return {
        "instance_id": cfg["instance_id"], "campus": cfg["campus"],
        "size": cfg["size"], "track": cfg["track"],
        "structure": cfg["structure"], "phi": cfg["phi"], "eta": cfg["eta"],
        "m": cfg["m"], "u_target": cfg["u_target"],
        "u_realized": cfg["u_realized"],
        "method": method, "seed": seed,
        "twt": m["WWT"], "makespan": m["makespan"],
        "mean_flow": m["mean_flow"], "breach_share": m["breach_share"],
        "breach_p1": pp.get(1), "breach_p2": pp.get(2),
        "breach_p3": pp.get(3), "breach_p4": pp.get(4),
        "decisions": decisions, "latency_ms_per_decision": mean_ms,
        "replans": extra_replans,
        "validator_ok": int(bool(res["feasible"])),
        "runtime_s": wall,
    }


def _shard_path(cfg):
    return OUT_DIR / "shards" / (cfg["config_id"] + ".json")


def _expected(cfg):
    exp = []
    if "rules" in METHODS:
        exp += ALL_RULES
    if "rl" in METHODS:
        exp += [s[0] for s in _RL_SPECS]
    if "specs" in METHODS:
        exp += [s[0] for s in _SPEC_SPECS if s[1] == cfg["structure"]]
    if "rollcp" in METHODS and cfg.get("rollcp"):
        exp.append("rollcp2")
    return exp


def run_config(cfg):
    t0 = time.perf_counter()
    try:
        dst = _shard_path(cfg)
        old_rows, old_expected = {}, []
        if dst.exists():
            try:
                with open(dst) as f:
                    old = json.load(f)
                old_rows = old.get("rows", {}) or {}
                old_expected = list(old.get("methods_expected", []) or [])
            except Exception:
                pass

        with open(cfg["path"]) as f:
            inst = json.load(f)
        ov = overlay_for(cfg["campus"], cfg["structure"], cfg["phi"],
                         cfg["eta"], cfg["m"])
        expected = _expected(cfg)
        todo = [meth for meth in expected if meth not in old_rows]
        if not todo:
            return {"config_id": cfg["config_id"], "ok": True, "skipped": True}

        out_rows = {}
        bad = []
        env = PairDispatchEnv(inst, ov)
        for meth in todo:
            if meth in ALL_RULES:
                sched = env.run_selector(get_selector(meth), method=meth,
                                         seed=RULE_SEED)
                res = validate2(inst, sched, ov)
                out_rows[meth] = _row(cfg, meth, RULE_SEED, sched, res)
            elif meth == "rollcp2":
                from methods import rolling2
                sched = rolling2.roll_cpsat(inst, ov, budget_s=2.0)
                res = validate2(inst, sched, ov)
                r = _row(cfg, meth, 0, sched, res,
                         extra_replans=sched.get("decisions"))
                r["latency_ms_per_decision"] = None
                r["replans"] = sched.get("decisions")
                out_rows[meth] = r
            else:
                spec = next(s for s in _RL_SPECS + _SPEC_SPECS
                            if s[0] == meth)
                pol = _get_policy(spec[2])
                penv = PairDispatchEnv(inst, ov)
                t1 = time.perf_counter()
                obs = penv.reset()
                done = penv._done
                while not done:
                    a, _, _, _ = pol.act(obs, greedy=True, device="cpu")
                    obs, _r, done, _i = penv.step(a)
                sched = penv.to_schedule(meth, seed=spec[3])
                sched["wall_seconds"] = time.perf_counter() - t1
                res = validate2(inst, sched, ov)
                out_rows[meth] = _row(cfg, meth, spec[3], sched, res)
            if out_rows[meth]["validator_ok"] == 0:
                bad.append(meth)

        out_rows = {**old_rows, **out_rows}
        shard = {"config_id": cfg["config_id"], "rows": out_rows,
                 "methods_expected": sorted(set(expected) | set(old_expected)),
                 "wall_seconds_total": time.perf_counter() - t0}
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(shard, f)
        os.replace(tmp, dst)
        return {"config_id": cfg["config_id"], "ok": True, "infeasible": bad,
                "wall": shard["wall_seconds_total"]}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"config_id": cfg["config_id"], "ok": False,
                "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def merge(verbose=True):
    import csv
    rows = []
    n_shards = n_partial = n_bad = 0
    shard_dir = OUT_DIR / "shards"
    for p in sorted(shard_dir.glob("*.json")) if shard_dir.exists() else []:
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            n_partial += 1
            continue
        n_shards += 1
        for meth, r in sorted(d.get("rows", {}).items()):
            rows.append(r)
            if not r.get("validator_ok"):
                n_bad += 1
    rows.sort(key=lambda r: (r["campus"], r["size"], r["m"], r["structure"],
                             r["phi"] or 0, r["eta"], r["instance_id"],
                             r["method"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(OUT_DIR / "results.parquet",
                                      index=False)
    except Exception:
        pass
    if verbose:
        print("merged %d shards -> %d rows -> %s (%d infeasible, %d partial)"
              % (n_shards, len(rows), out_csv, n_bad, n_partial))
    return {"rows": len(rows), "infeasible": n_bad}


# --------------------------------------------------------------------------- #
OUT_DIR = None
METHODS = ("rules",)


def main():
    global OUT_DIR, METHODS, _RL_SPECS, _SPEC_SPECS
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=["tier1", "tier2", "e3", "e4"])
    ap.add_argument("--methods", default="rules",
                    help="comma subset of rules,rl,rollcp")
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--campus", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    OUT_DIR = ROOT / "results" / args.family
    METHODS = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    if args.merge:
        merge()
        return

    if "rl" in METHODS:
        _RL_SPECS = discover_rl()
        print("rl checkpoints: %d (%s...)"
              % (len(_RL_SPECS), ", ".join(s[0] for s in _RL_SPECS[:4])))
        if not _RL_SPECS:
            sys.exit("no RL checkpoints found under %s" % TRAIN_DIR)
    if "specs" in METHODS:
        _SPEC_SPECS = discover_specialists()
        print("specialist checkpoints: %d" % len(_SPEC_SPECS))

    configs = build_configs(args.family)
    if args.family == "tier1":
        configs = assign_rollcp(configs, ROLLCP_PER_CELL)
    if args.campus:
        keep = {int(c) for c in args.campus.split(",")}
        configs = [c for c in configs if c["campus"] in keep]

    pending = []
    for c in configs:
        dst = _shard_path(c)
        have = set()
        if dst.exists():
            try:
                with open(dst) as f:
                    have = set(json.load(f).get("rows", {}))
            except Exception:
                pass
        if not set(_expected(c)) <= have:
            pending.append(c)
    if args.limit:
        pending = pending[:args.limit]

    print("family=%s methods=%s configs=%d pending=%d workers=%d"
          % (args.family, METHODS, len(configs), len(pending), args.workers),
        flush=True)
    if not pending:
        merge()
        return

    t0 = time.time()
    done = errs = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for res in pool.imap_unordered(run_config, pending):
            done += 1
            if not res.get("ok"):
                errs += 1
                print("[ERR] %s: %s" % (res["config_id"], res.get("error")),
                      flush=True)
            elif res.get("infeasible"):
                print("[INFEASIBLE] %s: %s"
                      % (res["config_id"], res["infeasible"]), flush=True)
            if done % 200 == 0 or done == len(pending):
                el = time.time() - t0
                print("  %d/%d  %.0fs  eta %.0fs  (%d err)"
                      % (done, len(pending), el,
                         el / done * (len(pending) - done), errs), flush=True)
    merge()
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump({"family": args.family, "methods": list(METHODS),
                   "elapsed_s": time.time() - t0, "n_configs": len(configs),
                   "n_run": done, "n_errors": errs,
                   "rl_specs": [s[0] for s in _RL_SPECS],
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f,
                  indent=2)


if __name__ == "__main__":
    main()
