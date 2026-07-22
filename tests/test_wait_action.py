"""Unit tests for the wait action (E10).

Invariants:
  1. allow_wait=False is inert: observation width F_TOTAL, no wait slot,
     schedules identical to before the change (covered by the released
     engine-parity suite plus the width check here).
  2. The wait token appears only while at least one technician is busy, so
     the driver's decline assert can never fire and episodes terminate even
     under a wait-whenever-legal adversary.
  3. The shaped return still telescopes to -TWT/100 with waits in the
     trajectory.
  4. A released checkpoint, zero-padded to the wait width, reproduces its
     own schedule exactly when the wait slot is masked out: the plumbing
     changes nothing until the new action is actually taken.
"""
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from env.engine import F_TOTAL, PairDispatchEnv
from methods.policy2 import load_policy, make_policy
from overlays.build import build_overlay, load_crews

ROOT = Path(__file__).resolve().parents[1]
Y1 = Path(os.environ.get("FMWOS_Y1_ROOT",
                         ROOT.parent / "FM-Scheduling"))
CAP = str(Y1 / "results/p1_calib/capacity.csv")
INST_GLOB = str(Y1 / "data/processed/instances/c05/replay/150/*.json")
RELEASED_CKPT = ROOT / "results/train/mlp_seed301/best.pt"


@pytest.fixture(scope="module")
def inst():
    return json.load(open(sorted(glob.glob(INST_GLOB))[0]))


@pytest.fixture(scope="module")
def ov(inst):
    campus = inst["meta"]["campus"]
    return build_overlay(campus, load_crews(CAP, campus), "chain", 1.0,
                         0.8, 0.6)


def test_default_is_inert(inst, ov):
    env = PairDispatchEnv(inst, ov)
    obs = env.reset()
    assert obs["pairs"].shape[1] == F_TOTAL
    assert env._wait_idx is None


def test_wait_slot_only_when_someone_is_busy(inst, ov):
    env = PairDispatchEnv(inst, ov, allow_wait=True)
    obs = env.reset()
    # At t = 0 every technician is idle: no wait slot.
    assert env._wait_idx is None
    assert obs["pairs"].shape[1] == F_TOTAL + 1
    # Dispatch one pair; if a decision follows while that technician is
    # busy, the wait slot must be present and carry the is-wait flag.
    obs, _r, done, _i = env.step(0)
    seen_wait = False
    while not done:
        if env._wait_idx is not None:
            w = env._wait_idx
            assert obs["mask"][w]
            assert obs["pairs"][w, F_TOTAL] == 1.0
            assert obs["pairs"][w, :F_TOTAL - 20].sum() == 0.0
            seen_wait = True
            break
        obs, _r, done, _i = env.step(0)
    assert seen_wait, "no busy-technician decision instant reached"


def test_telescoping_with_waits(inst, ov):
    env = PairDispatchEnv(inst, ov, allow_wait=True)
    obs = env.reset()
    total = 0.0
    done = env._done
    rng = random.Random(7)
    waits = 0
    while not done:
        if env._wait_idx is not None and rng.random() < 0.5:
            a = env._wait_idx
            waits += 1
        else:
            a = rng.randrange(len(env._pairs))
        obs, r, done, _info = env.step(a)
        total += r
    assert waits > 0, "trajectory exercised no waits"
    assert abs(total - (-env._realized / 100.0)) < 1e-6
    assert env._n_waits == waits


def test_wait_adversary_terminates(inst, ov):
    """Always wait when legal: the episode must still complete and pass the
    validator (waiting only defers work, never drops it)."""
    from env.validator2 import validate as validate2
    env = PairDispatchEnv(inst, ov, allow_wait=True)
    env.reset()
    done = env._done
    steps = 0
    while not done:
        a = env._wait_idx if env._wait_idx is not None else 0
        _obs, _r, done, _i = env.step(a)
        steps += 1
        assert steps < 500_000
    sched = env.to_schedule("wait_adversary")
    assert sched["waits"] == env._n_waits > 0
    res = validate2(inst, sched, ov)
    assert res["feasible"]


@pytest.mark.skipif(not RELEASED_CKPT.exists(),
                    reason="released checkpoint not present")
def test_released_checkpoint_parity_with_wait_masked(inst, ov):
    released = load_policy(str(RELEASED_CKPT))
    padded = make_policy("mlp", f_pair=F_TOTAL + 1)
    sd = {k: v.clone() for k, v in padded.state_dict().items()}
    for k, v in released.state_dict().items():
        if k == "enc1.weight":
            sd[k] = torch.zeros_like(sd[k])
            sd[k][:, :F_TOTAL] = v
        else:
            sd[k] = v
    padded.load_state_dict(sd)
    padded.eval()
    released.eval()

    def run(policy, allow_wait, mask_wait):
        env = PairDispatchEnv(inst, ov, allow_wait=allow_wait)
        obs = env.reset()
        done = env._done
        while not done:
            if mask_wait and env._wait_idx is not None:
                obs = dict(obs)
                m = obs["mask"].copy()
                m[env._wait_idx] = False
                obs["mask"] = m
                obs["n"] = int(m.sum())
            a, _, _, _ = policy.act(obs, greedy=True, device="cpu")
            obs, _r, done, _i = env.step(a)
        return sorted((x["wo"], x["tech"], round(x["start_bh"], 9))
                      for x in env.assignments)

    ref = run(released, allow_wait=False, mask_wait=False)
    got = run(padded, allow_wait=True, mask_wait=True)
    assert ref == got
