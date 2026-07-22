"""Validator v2 unit tests.

Run: PYTHONPATH=.:vendor python tests/test_validator2.py
"""
import copy
import glob
import json
import os
import random
from pathlib import Path

from fmwos_y1 import pdrs
from fmwos_y1.validator import validate as validate_y1
from env.validator2 import validate as validate2
from env.engine import PairDispatchEnv
from methods.rules import get_selector
from overlays.build import build_overlay, load_crews

_REPO = Path(__file__).resolve().parents[1]
Y1_ROOT = Path(os.environ.get("FMWOS_Y1_ROOT", _REPO.parent / "FM-Scheduling"))
CAP = str(Y1_ROOT / "results/p1_calib/capacity.csv")
INST_GLOB = str(Y1_ROOT / "data/processed/instances/c*/replay/*/*.json")


def sample_instances(n):
    paths = sorted(glob.glob(INST_GLOB))
    rng = random.Random(20260709)
    return [json.load(open(p)) for p in rng.sample(paths, n)]


def test_l0_byte_equality_vs_y1():
    """L0 validation reproduces Y1 validation byte-for-byte on Y1 outputs."""
    for inst in sample_instances(8):
        for rule in ("edd", "atc", "random"):
            sched = pdrs.dispatch(inst, rule, seed=301)
            r1 = validate_y1(inst, sched)
            r2 = validate2(inst, sched, overlay=None)
            assert r1["feasible"] == r2["feasible"]
            assert r1["violations"] == r2["violations"]
            assert json.dumps(r1["metrics"], sort_keys=True) == \
                   json.dumps(r2["metrics"], sort_keys=True), rule


def _flex_setup():
    inst = sample_instances(1)[0]
    campus = inst["meta"]["campus"]
    ov = build_overlay(campus, load_crews(CAP, campus), "chain", 1.0, 0.8, 1.0)
    env = PairDispatchEnv(inst, ov)
    sched = env.run_selector(get_selector("atc_eta"), method="atc_eta",
                             seed=301)
    return inst, ov, sched


def test_skill_illegality_rejected():
    inst, ov, sched = _flex_setup()
    assert validate2(inst, sched, ov)["feasible"]
    bad = copy.deepcopy(sched)
    # find an assignment and force it onto a technician lacking the skill
    skills = {t["id"]: set(t["skills"]) for t in ov["technicians"]}
    wo_trade = {w["id"]: w["trade"] for w in inst["work_orders"]}
    done = False
    for a in bad["assignments"]:
        g = wo_trade[a["wo"]]
        for tid, sk in skills.items():
            if g not in sk:
                a["tech"] = tid
                done = True
                break
        if done:
            break
    assert done
    res = validate2(inst, bad, ov)
    assert not res["feasible"]
    assert any(v.startswith("(b)") for v in res["violations"])


def test_eta_duration_mismatch_rejected():
    inst, ov, sched = _flex_setup()
    prim = {t["id"]: t["primary"] for t in ov["technicians"]}
    wo_trade = {w["id"]: w["trade"] for w in inst["work_orders"]}
    bad = copy.deepcopy(sched)
    hit = False
    for a in bad["assignments"]:
        if prim[a["tech"]] != wo_trade[a["wo"]]:      # a secondary assignment
            a["end_bh"] = a["start_bh"] + (a["end_bh"] - a["start_bh"]) * 0.8
            hit = True
            break
    assert hit, "schedule has no secondary assignment to corrupt"
    res = validate2(inst, bad, ov)
    assert not res["feasible"]
    assert any(v.startswith("(d)") for v in res["violations"])


def test_eta_one_secondary_keeps_nominal_p():
    """eta = 1.0: secondary assignments consume exactly p_j (no grid)."""
    inst = sample_instances(1)[0]
    campus = inst["meta"]["campus"]
    ov = build_overlay(campus, load_crews(CAP, campus), "full", None, 1.0, 0.6)
    env = PairDispatchEnv(inst, ov)
    sched = env.run_selector(get_selector("edd"), method="edd", seed=301)
    res = validate2(inst, sched, ov)
    assert res["feasible"], res["violations"][:3]
    p_of = {w["id"]: w["p_bh"] for w in inst["work_orders"]}
    for a in sched["assignments"]:
        assert abs((a["end_bh"] - a["start_bh"]) - p_of[a["wo"]]) < 1e-9


if __name__ == "__main__":
    test_l0_byte_equality_vs_y1()
    print("PASS test_l0_byte_equality_vs_y1")
    test_skill_illegality_rejected()
    print("PASS test_skill_illegality_rejected")
    test_eta_duration_mismatch_rejected()
    print("PASS test_eta_duration_mismatch_rejected")
    test_eta_one_secondary_keeps_nominal_p()
    print("PASS test_eta_one_secondary_keeps_nominal_p")
    print("OK validator2 tests")
