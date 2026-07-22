"""E-F post-hoc evaluator: score an arbitrary pair-MLP checkpoint on the
penalized-FULL verdict cells and the Gate P contended-flexible scope, using the
EXACT tier1 inference path (run_dynamic.overlay_for + PairDispatchEnv greedy +
validate2 WWT). Read-only: writes nothing except what the caller saves.

Reproduces the released tier1 policy numbers when handed results/train/
mlp_seed*/best.pt (verified against results/tier1/results.parquet).

Scopes:
  full08 : structure=full, eta=0.8, m in {0.6,0.8}   (the penalized-FULL failure cells)
  gatep  : structure in {chain,full}, m in {0.6,0.8}, eta in {1.0,0.8}
           (the contended-flexible Gate P scope; L0 is not in-scope)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

import multiprocessing as mp  # noqa: E402
import pandas as pd  # noqa: E402

import experiments.run_dynamic as rd  # noqa: E402  (globs are lazy; safe)
from env.engine import PairDispatchEnv  # noqa: E402
from env.validator2 import validate as validate2  # noqa: E402
from methods.policy2 import load_policy  # noqa: E402

_POL = None
_CKPT = None


def _cells(scope):
    if scope == "full08":
        return [("full", None, 0.8, 0.6), ("full", None, 0.8, 0.8)]
    if scope == "gatep":
        cells = []
        for st, phi in (("chain", 1.0), ("full", None)):
            for m in (0.6, 0.8):
                for eta in (1.0, 0.8):
                    cells.append((st, phi, eta, m))
        return cells
    raise ValueError(scope)


def _configs(scope):
    want = set(_cells(scope))
    out = []
    for cfg in rd.build_configs("tier1"):
        if (cfg["structure"], cfg["phi"], cfg["eta"], cfg["m"]) in want:
            out.append(cfg)
    return out


def _init(ckpt):
    global _POL, _CKPT
    import torch
    torch.set_num_threads(1)
    _POL = load_policy(ckpt, map_location="cpu")
    _POL.eval()
    _CKPT = ckpt


def _run(cfg):
    import json
    with open(cfg["path"]) as f:
        inst = json.load(f)
    ov = rd.overlay_for(cfg["campus"], cfg["structure"], cfg["phi"],
                        cfg["eta"], cfg["m"])
    env = PairDispatchEnv(inst, ov)
    obs = env.reset()
    done = env._done
    while not done:
        a, _, _, _ = _POL.act(obs, greedy=True, device="cpu")
        obs, _r, done, _i = env.step(a)
    sched = env.to_schedule("probe")
    res = validate2(inst, sched, ov)
    return {"instance_id": cfg["instance_id"], "campus": cfg["campus"],
            "size": cfg["size"], "structure": cfg["structure"],
            "phi": cfg["phi"], "eta": cfg["eta"], "m": cfg["m"],
            "twt": res["metrics"]["WWT"],
            "validator_ok": int(bool(res["feasible"]))}


def evaluate(ckpt, method_label, scope="full08", workers=12):
    cfgs = _configs(scope)
    with mp.Pool(workers, initializer=_init, initargs=(ckpt,)) as pool:
        rows = pool.map(_run, cfgs, chunksize=16)
    df = pd.DataFrame(rows)
    df["method"] = method_label
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--scope", default="full08", choices=["full08", "gatep"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    df = evaluate(args.ckpt, args.label, args.scope, args.workers)
    df.to_parquet(args.out)
    print("wrote %d rows -> %s  (mean twt=%.2f, infeasible=%d)"
          % (len(df), args.out, df.twt.mean(), int((df.validator_ok == 0).sum())))


if __name__ == "__main__":
    main()
