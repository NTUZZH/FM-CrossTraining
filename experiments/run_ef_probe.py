"""E-F training-artifact probe: extended-budget PPO on the SAME
mixed-flexibility curriculum as the released verdict pool, to test whether the
penalized-FULL policy failure is a training-budget artifact.

Post-hoc robustness ONLY. The pre-registered gate verdict stands as registered.
The single deliberate change vs the released pair-MLP pool (mlp_seed301-310) is
the update budget: 1,200 -> 3,600 (x3). Seeds are non-registered ids
(v2mlp901-903) so no released analysis can sweep them (results/train_probe/ is a
sibling of results/train/ and is matched by NO analysis glob; verified at prep).

This runner does NOT modify methods/train2.py. It reproduces train2's own
deterministic setup (same seed, same module constants) to ASSERT the
one-variable-changed invariant BEFORE the long loop, then delegates training to
methods.train2.train unchanged. Any drift dies in seconds, not hours.

Modes:
  --preflight            run the asserts + config diff only (no training)
  --smoke                asserts + 2 real updates on GPU to a scratch dir,
                         reload checkpoint, then diff resolved config
  --test-assert          build a WRONG-width net and confirm the param assert
                         aborts (negative control)
  --launch --seed S      the real probe run (3,600 updates -> results/train_probe)
                         GUARDED: refuses unless Y2_EF_APPROVED=1 is set.

Nothing here starts a full run without explicit approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "vendor"))

import torch  # noqa: E402

from methods import train2  # noqa: E402
from methods.policy2 import make_policy, load_policy  # noqa: E402

# ---- the invariant constants of the released verdict pool ------------------ #
RELEASED_PARAM = 40834          # pair-MLP, hidden=128, k_pairs=256
RELEASED_KPAIRS = 256
RELEASED_UPDATES = 1200
PROBE_UPDATES = 3600            # the ONE deliberate change (x3 budget)
RELEASED_N_REPLAY = 760         # list_replay_train_files, campuses 5/9/10/12
RELEASED_N_DEV = 32
RELEASED_EVAL_EVERY = 20
REF_CONFIG = _REPO / "results/train/mlp_seed301/config.json"
OUT_ROOT = _REPO / "results/train_probe"

# keys of the resolved config that ARE allowed to differ from the comparator
ALLOWED_DIFF_KEYS = {"seed", "updates"}   # output dir is not stored in config


def _sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _build_reference_setup(seed, sizes=(150, 400)):
    """Reproduce train2's deterministic sampler + dev set for the given seed
    (train2.train builds these identically from the same seed/constants)."""
    sampler = train2.CurriculumSampler(train2.CAMPUSES, list(sizes), seed,
                                       structures=None)
    dev = train2.load_dev_set(train2.CAMPUSES, list(sizes), RELEASED_N_DEV)
    return sampler, dev


def preflight(seed, verbose=True):
    """The five-check in-runner asserts. Returns a dict of evidence."""
    ev = {}

    # 1. parameter count + k_pairs (identical architecture) ------------------ #
    pol = make_policy("mlp")
    n_param = sum(p.numel() for p in pol.parameters())
    assert n_param == RELEASED_PARAM, (
        "PARAM COUNT DRIFT: %d != released %d -- architecture changed, ABORT"
        % (n_param, RELEASED_PARAM))
    assert pol.k_pairs == RELEASED_KPAIRS, (
        "K_PAIRS DRIFT: %d != %d, ABORT" % (pol.k_pairs, RELEASED_KPAIRS))
    ev["param_count"] = n_param
    ev["k_pairs"] = pol.k_pairs

    # 2. curriculum instance set identical to the released training runs ----- #
    sampler, dev = _build_reference_setup(seed)
    n_replay = len(sampler.replay_files)
    assert n_replay == RELEASED_N_REPLAY, (
        "REPLAY SET DRIFT: %d files != released %d -- Y1 instance root changed,"
        " ABORT" % (n_replay, RELEASED_N_REPLAY))
    assert len(dev) == RELEASED_N_DEV, (
        "DEV SET DRIFT: %d != %d, ABORT" % (len(dev), RELEASED_N_DEV))
    ev["n_replay_train_files"] = n_replay
    ev["n_dev"] = len(dev)
    # hash of the resolved instance list (order-sensitive) + dev list
    ev["replay_list_hash"] = _sha([os.path.relpath(p, train2.INST_ROOT)
                                   for p in sampler.replay_files])
    ev["dev_list_hash"] = _sha(sorted(
        i["meta"].get("instance_id", i["meta"].get("window_start", ""))
        for i in dev))

    # 3. curriculum mix identical (structures / m / eta / dev specs) ---------- #
    assert sampler.structures == train2.STRUCT_CHOICES, "STRUCTURE MIX DRIFT"
    assert train2.M_CHOICES == [0.5, 0.6, 0.8, 1.0], "M GRID DRIFT"
    assert train2.ETA_CHOICES == [1.0, 0.8], "ETA GRID DRIFT"
    assert train2.DEV_PRIMARY == {"structure": "chain", "phi": 1.0,
                                  "eta": 0.8, "m": 0.6}, "DEV PRIMARY DRIFT"
    ev["structures"] = sampler.structures
    ev["dev_primary"] = train2.DEV_PRIMARY

    if verbose:
        print("[preflight] PASS  param=%d k_pairs=%d replay=%d dev=%d"
              % (n_param, pol.k_pairs, n_replay, len(dev)))
        print("[preflight] replay_list_hash=%s dev_list_hash=%s"
              % (ev["replay_list_hash"], ev["dev_list_hash"]))
    return ev


def config_diff(resolved_path, verbose=True):
    """Diff a resolved probe config.json against the released comparator.
    Every difference must be an intended one (ALLOWED_DIFF_KEYS)."""
    with open(REF_CONFIG) as fh:
        ref = json.load(fh)
    with open(resolved_path) as fh:
        got = json.load(fh)
    diffs = {}
    for k in sorted(set(ref) | set(got)):
        if ref.get(k) != got.get(k):
            diffs[k] = (ref.get(k), got.get(k))
    unexpected = {k: v for k, v in diffs.items() if k not in ALLOWED_DIFF_KEYS}
    if verbose:
        print("[config-diff] vs %s" % REF_CONFIG)
        for k, (a, b) in diffs.items():
            tag = "OK  " if k in ALLOWED_DIFF_KEYS else "BAD "
            print("  [%s] %-24s released=%r  probe=%r" % (tag, k, a, b))
        if not diffs:
            print("  (identical)")
    assert not unexpected, ("UNEXPECTED CONFIG DIFFS (not one-variable): %s"
                            % list(unexpected))
    return diffs


def run_smoke(device="cuda"):
    """Asserts + 2 real updates on GPU; reload checkpoint; config diff."""
    scratch = _REPO / "results/train_probe/_smoke_v2mlp901"
    print("== preflight ==")
    preflight(901)
    print("== 2-update smoke on %s -> %s ==" % (device, scratch))
    train2.train(901, 2, str(scratch), arch="mlp", smoke=False, device=device)
    # checkpoint reload + param assert on the reloaded net
    m = load_policy(str(scratch / "best.pt"))
    n = sum(p.numel() for p in m.parameters())
    assert n == RELEASED_PARAM and m.k_pairs == RELEASED_KPAIRS
    print("[smoke] checkpoint reloaded OK, param=%d k_pairs=%d" % (n, m.k_pairs))
    print("== config diff (resolved smoke config vs released) ==")
    # smoke config has updates=2; the real probe sets 3600. Both are 'updates'
    # (an ALLOWED key), so the diff proves everything ELSE is identical.
    config_diff(str(scratch / "config.json"))
    print("[smoke] DONE")


def test_assert():
    """Negative control: a wrong-width net must trip the param assert."""
    import torch.nn as nn
    from methods.policy2 import PairMLP
    bad = PairMLP(hidden=64)          # deliberately wrong width
    n = sum(p.numel() for p in bad.parameters())
    try:
        assert n == RELEASED_PARAM, (
            "PARAM COUNT DRIFT: %d != released %d" % (n, RELEASED_PARAM))
    except AssertionError as e:
        print("[test-assert] PASS: wrong-width net (hidden=64, param=%d) "
              "correctly aborted: %s" % (n, e))
        return
    raise SystemExit("[test-assert] FAIL: wrong-width net did NOT trip assert")


def launch(seed, device="cuda"):
    if os.environ.get("Y2_EF_APPROVED") != "1":
        raise SystemExit(
            "REFUSED: E-F full run is gated. Re-run with Y2_EF_APPROVED=1 "
            "to confirm.")
    out = OUT_ROOT / ("v2mlp%d" % seed)
    print("== preflight (seed %d) ==" % seed)
    preflight(seed)
    os.makedirs(out, exist_ok=True)

    # Also retain the dev_full-SELECTED checkpoint (best_full.pt) alongside
    # train2's dev_primary best.pt, so the selection-target-mismatch hypothesis
    # is testable post-hoc without retraining. train2 only saves on
    # dev_primary improvement; we wrap its module-level eval_dev so that each
    # time the FULL monitor hits a new minimum we snapshot the (cpu-clone)
    # policy. No train2.py edit; restored in finally.
    _orig_eval = train2.eval_dev
    _best_full = {"v": float("inf")}

    def _wrapped_eval(policy, dev, bank, spec, device):
        val = _orig_eval(policy, dev, bank, spec, device)
        if spec is train2.DEV_MON_FULL and val < _best_full["v"]:
            _best_full["v"] = val
            policy.save(str(out / "best_full.pt"))
        return val

    train2.eval_dev = _wrapped_eval
    print("== LAUNCH probe: %d updates -> %s ==" % (PROBE_UPDATES, out))
    try:
        train2.train(seed, PROBE_UPDATES, str(out), arch="mlp", smoke=False,
                     device=device)
    finally:
        train2.eval_dev = _orig_eval
    with open(out / "best_full_dev.json", "w") as fh:
        json.dump({"best_dev_full": _best_full["v"]}, fh)
    print("[launch] best_full dev(WWT)=%.4f -> %s/best_full.pt"
          % (_best_full["v"], out))


def main(argv=None):
    ap = argparse.ArgumentParser(description="E-F training-artifact probe")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--test-assert", dest="test_assert", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--seed", type=int, default=901)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args(argv)
    if args.test_assert:
        test_assert()
    elif args.preflight:
        preflight(args.seed)
    elif args.smoke:
        run_smoke(device=args.device)
    elif args.launch:
        launch(args.seed, device=args.device)
    else:
        ap.error("choose one of --preflight/--smoke/--test-assert/--launch")


if __name__ == "__main__":
    main()
