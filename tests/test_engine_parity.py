"""Pair-engine v2 unit tests: v1 parity, determinism, non-delay.

Run: PYTHONPATH=.:vendor python tests/test_engine_parity.py [n_instances]

Parity (fairness protocol, mandatory): with singleton skills the v2 pair
engine reproduces the Y1 standalone dispatcher's schedule EXACTLY:
  (1) the five deterministic Y1 rules through the natural v2 extension
      (global argmax over P + TB) -- assignment-set equality, bitwise on
      (wo, tech, start, end) after canonical sort;
  (2) ALL six Y1 rules, including seeded random, through the strict
      v1-semantics replay harness (run_replay_y1) -- draw-for-draw equality.
"""
import glob
import json
import os
import random
import sys
from pathlib import Path

from fmwos_y1 import pdrs
from env.engine import PairDispatchEnv
from methods.rules import get_selector

_REPO = Path(__file__).resolve().parents[1]
Y1_ROOT = Path(os.environ.get("FMWOS_Y1_ROOT", _REPO.parent / "FM-Scheduling"))
INST_GLOB = str(Y1_ROOT / "data/processed/instances/c*/replay/*/*.json")
DET_RULES = ["edd", "wspt", "atc", "pfifo", "mor"]


def canon(schedule):
    return sorted((a["wo"], a["tech"], round(a["start_bh"], 9),
                   round(a["end_bh"], 9)) for a in schedule["assignments"])


def load_sample(n):
    paths = sorted(glob.glob(INST_GLOB))
    rng = random.Random(20260707)
    take = rng.sample(paths, min(n, len(paths)))
    return [json.load(open(p)) for p in take]


def test_parity(instances):
    n_multi = 0
    for inst in instances:
        y1 = {r: pdrs.dispatch(inst, r, seed=301) for r in DET_RULES + ["random"]}
        env = PairDispatchEnv(inst, check_nondelay=True)
        for r in DET_RULES:
            v2 = env.run_selector(get_selector(r), method=r, seed=301)
            assert canon(v2) == canon(y1[r]), (
                "parity FAIL rule=%s inst=%s" % (r, inst["meta"]["id"]))
        # strict engine parity incl. random: v1-semantics replay harness
        for r in DET_RULES + ["random"]:
            v2r = env.run_replay_y1(r, seed=301)
            assert canon(v2r) == canon(y1[r]), (
                "replay parity FAIL rule=%s inst=%s" % (r, inst["meta"]["id"]))
        # natural v2 random: exact match not guaranteed at multi-trade
        # instants; count agreement for the record.
        v2rand = env.run_selector(get_selector("random"), method="random",
                                  seed=301)
        if canon(v2rand) != canon(y1["random"]):
            n_multi += 1
    print("  natural-v2 random differed on %d/%d instances "
          "(expected only at multi-trade decision instants)"
          % (n_multi, len(instances)))


def test_determinism(instances):
    inst = instances[0]
    env = PairDispatchEnv(inst)
    a = env.run_selector(get_selector("atc"), method="atc", seed=301)
    b = env.run_selector(get_selector("atc"), method="atc", seed=301)
    assert canon(a) == canon(b)
    r1 = env.run_selector(get_selector("random"), method="random", seed=7)
    r2 = env.run_selector(get_selector("random"), method="random", seed=7)
    assert canon(r1) == canon(r2)


def test_flexible_smoke(instances):
    """Flexible overlay: engine runs, validator passes, non-delay holds."""
    sys.path.insert(0, ".")
    from overlays.build import build_overlay, load_crews
    from env.validator2 import validate as validate2
    cap = str(Y1_ROOT / "results/p1_calib/capacity.csv")
    for inst in instances[:3]:
        campus = inst["meta"]["campus"]
        crews = load_crews(cap, campus)
        for st, phi, eta in (("chain", 1.0, 0.8), ("full", None, 0.8),
                             ("generalist", None, 1.0)):
            ov = build_overlay(campus, crews, st, phi, eta, 0.6)
            env = PairDispatchEnv(inst, ov, check_nondelay=True)
            for rule in ("edd", "atc", "lfj_atc", "atc_eta"):
                sched = env.run_selector(get_selector(rule), method=rule,
                                         seed=301)
                res = validate2(inst, sched, ov)
                assert res["feasible"], (rule, st, res["violations"][:2])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    instances = load_sample(n)
    test_parity(instances)
    print("PASS test_parity (%d instances x (5 det + 6 replay) rules)" % n)
    test_determinism(instances)
    print("PASS test_determinism")
    test_flexible_smoke(instances)
    print("PASS test_flexible_smoke")
    print("OK engine tests")
