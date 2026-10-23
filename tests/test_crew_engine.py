"""Unit tests for the crew-order engine and its independent validator (E-CREW).

(i)   c_max = 1 parity: the crew engine reproduces PairDispatchEnv's schedule
      record for record, on replay instances, every structure and every rule,
      and the crew validator's TWT equals the released validator's bitwise.
(ii)  A hand-built instance where a two-person order waits for its second
      technician while a single order is held off, against a hand-computed
      schedule; and a mixed crew (primary + secondary) whose elapsed time
      follows ceil_0.01(p / (1 + eta)).
(iii) The validator rejects a wrong crew size, an ineligible member, an
      overlap and a short elapsed time.
(iv)  No deadlock on stress instances (long orders on small rosters,
      c_max = 4, every structure and rule), and crew sizes identical across
      structures at a given crew multiplier.

Run: PYTHONPATH=.:vendor python -m pytest tests/test_crew_engine.py -q
"""
import copy
import csv
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.crew_engine import CrewDispatchEnv, crew_sizes  # noqa: E402
from env.engine import PairDispatchEnv  # noqa: E402
from env.validator2 import validate as validate2  # noqa: E402
from env.validator_crew import validate as validate_crew  # noqa: E402
from experiments.y1_root import y1_root  # noqa: E402
from methods.rules import ALL_RULES, get_selector  # noqa: E402
from overlays.build import build_overlay, load_crews  # noqa: E402

Y1 = y1_root()
CAP = Y1 / "results/p1_calib/capacity.csv"
INST = Y1 / "data/processed/instances"
SEED = 301

PARITY_IDS = ["c05_replay_150_0100", "c09_replay_400_0100",
              "c10_replay_150_0100", "c12_replay_400_0100",
              "c12_replay_150_0150", "c10_replay_400_0150"]
CELLS = [("dedicated", None, 1.0, 0.6), ("chain", 1.0, 0.8, 0.6),
         ("full", None, 1.0, 0.8), ("chain", 1.0, 1.0, 1.0),
         ("full", None, 0.8, 0.6)]


def _index():
    with open(INST / "index.csv", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)
                if r["track"] == "replay" and r["split"] == "test"}


def _load(iid, idx=None):
    idx = idx or _index()
    return json.load(open(INST / idx[iid]["path"]))


def _overlay(inst, st, phi, eta, m):
    campus = int(inst["meta"]["campus"])
    return build_overlay(campus, load_crews(CAP, campus), st, phi, eta, m)


# --------------------------------------------------------------------------- #
# (i) c_max = 1 parity                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("iid", PARITY_IDS)
def test_cmax1_parity(iid):
    idx = _index()
    if iid not in idx:
        pytest.skip("instance %s not in the corpus" % iid)
    inst = _load(iid, idx)
    snapshot = copy.deepcopy(inst)
    for st, phi, eta, m in CELLS:
        ov = _overlay(inst, st, phi, eta, m)
        for meth in ALL_RULES:
            a = PairDispatchEnv(inst, ov).run_selector(
                get_selector(meth), method=meth, seed=SEED)
            b = CrewDispatchEnv(inst, ov, c_max=1).run_selector(
                get_selector(meth), method=meth, seed=SEED)
            assert a["assignments"] == b["assignments"], (iid, st, eta, meth)
            r2 = validate2(inst, a, ov)
            rc = validate_crew(inst, b, ov, c_max=1)
            assert r2["feasible"] and rc["feasible"], rc["violations"][:3]
            assert rc["metrics"]["WWT"] == r2["metrics"]["WWT"]
            assert rc["metrics"]["makespan"] == r2["metrics"]["makespan"]
            assert rc["metrics"]["mean_flow"] == r2["metrics"]["mean_flow"]
            assert rc["metrics"]["breach_share"] == \
                r2["metrics"]["breach_share"]
            assert b["held_tech_bh"] == 0.0 and b["n_blocks"] == 0
    assert inst == snapshot, "the crew engine mutated the caller's instance"


def test_cmax1_parity_without_overlay():
    inst = _load("c05_replay_150_0100")
    for meth in ALL_RULES:
        a = PairDispatchEnv(inst).run_selector(get_selector(meth),
                                               method=meth, seed=SEED)
        b = CrewDispatchEnv(inst, c_max=1).run_selector(
            get_selector(meth), method=meth, seed=SEED)
        assert a["assignments"] == b["assignments"], meth


# --------------------------------------------------------------------------- #
# (ii) hand-built instances                                                    #
# --------------------------------------------------------------------------- #
def _tiny(orders, techs, eta=1.0, structure="dedicated"):
    inst = {"meta": {"id": "tiny", "campus": 99}, "trades": ["A", "B"],
            "technicians": [], "work_orders": orders}
    ov = {"overlay_id": "tiny", "campus": 99, "structure": structure,
          "phi": None, "eta": eta, "crew_multiplier": 1.0,
          "technicians": techs}
    return inst, ov


def _wo(wid, trade, p, rel, due, w=1.0, prio=3):
    return {"id": wid, "trade": trade, "p_bh": p, "release_bh": rel,
            "due_bh": due, "priority": prio, "weight": w, "is_pm": False}


def _by_order(sched):
    out = {}
    for a in sched["assignments"]:
        out.setdefault(a["wo"], []).append((a["tech"], a["start_bh"],
                                            a["end_bh"]))
    return out


def test_crew_waits_and_single_order_is_held_off():
    """Two A technicians. J1 (4 bh) runs on T0 from 0 to 4. J2 (12 bh,
    crew of 2, due 5) arrives at 1 and finds only T1 idle: T1 is held and
    J2 blocked. J3 (2 bh, due 50) arrives at 2; EDD selects J2 again, T1 is
    held again, so J3 is held off. At 4 T0 frees and J2 starts on {T0, T1}
    for ceil(12 / 2) = 6 bh, ending at 10; J3 then runs from 10 to 12.
    Held time: T1 over [1, 2] and [2, 4], 3 technician-hours."""
    orders = [_wo("J1", "A", 4.0, 0.0, 100.0), _wo("J2", "A", 12.0, 1.0, 5.0),
              _wo("J3", "A", 2.0, 2.0, 50.0)]
    techs = [{"id": "T0", "primary": "A", "skills": ["A"]},
             {"id": "T1", "primary": "A", "skills": ["A"]},
             {"id": "T2", "primary": "B", "skills": ["B"]}]
    inst, ov = _tiny(orders, techs)
    env = CrewDispatchEnv(inst, ov, c_max=2)
    sched = env.run_selector(get_selector("edd"), method="edd", seed=SEED)
    got = _by_order(sched)
    assert got == {"J1": [("T0", 0.0, 4.0)],
                   "J2": [("T0", 4.0, 10.0), ("T1", 4.0, 10.0)],
                   "J3": [("T0", 10.0, 12.0)]}
    assert sched["held_tech_bh"] == 3.0
    assert sched["n_blocks"] == 2
    res = validate_crew(inst, sched, ov, c_max=2)
    assert res["feasible"], res["violations"]
    assert res["metrics"]["WWT"] == 5.0            # J2: 1.0 * (10 - 5)
    assert res["metrics"]["n_crew_orders"] == 1
    assert res["metrics"]["mean_wait_crew"] == 3.0
    # Without crews the released engine starts J2 on T1 at 1 and J3 waits.
    base = PairDispatchEnv(inst, ov).run_selector(get_selector("edd"),
                                                  method="edd", seed=SEED)
    assert _by_order(base)["J2"] == [("T1", 1.0, 13.0)]


def test_mixed_crew_elapsed_time():
    """A crew of one primary and one secondary member (eta = 0.8) takes
    ceil_0.01(12 / 1.8) = 6.67 bh; the selector's technician leads."""
    orders = [_wo("J1", "A", 4.0, 0.0, 2.0), _wo("J2", "A", 12.0, 0.0, 50.0)]
    techs = [{"id": "T0", "primary": "A", "skills": ["A"]},
             {"id": "T1", "primary": "A", "skills": ["A"]},
             {"id": "T2", "primary": "B", "skills": ["B", "A"]}]
    inst, ov = _tiny(orders, techs, eta=0.8, structure="chain")
    env = CrewDispatchEnv(inst, ov, c_max=4)
    assert env.crew_of == {"J1": 1, "J2": 2}       # k(A) = 2 caps ceil(12/8)
    sched = env.run_selector(get_selector("edd"), method="edd", seed=SEED)
    got = _by_order(sched)
    assert got["J1"] == [("T0", 0.0, 4.0)]
    assert got["J2"] == [("T1", 0.0, 6.67), ("T2", 0.0, 6.67)]
    res = validate_crew(inst, sched, ov, c_max=4)
    assert res["feasible"], res["violations"]


def test_selectors_score_on_crew_elapsed_time():
    orders = [_wo("J1", "A", 20.0, 0.0, 100.0), _wo("J2", "A", 6.0, 0.0, 9.0)]
    techs = [{"id": "T%d" % i, "primary": "A", "skills": ["A"]}
             for i in range(3)]
    inst, ov = _tiny(orders, techs)
    env = CrewDispatchEnv(inst, ov, c_max=3)
    jobs = {j["id"]: j for j in env.instance["work_orders"]}
    assert jobs["J1"]["crew"] == 3 and jobs["J1"]["p_labor"] == 20.0
    assert jobs["J1"]["p_bh"] == 20.0 / 3
    assert jobs["J2"]["crew"] == 1 and jobs["J2"]["p_bh"] == 6.0
    assert inst["work_orders"][0]["p_bh"] == 20.0   # caller untouched


# --------------------------------------------------------------------------- #
# (iii) validator rejections                                                   #
# --------------------------------------------------------------------------- #
def _valid_case():
    orders = [_wo("J1", "A", 4.0, 0.0, 100.0), _wo("J2", "A", 12.0, 1.0, 5.0),
              _wo("J3", "A", 2.0, 2.0, 50.0), _wo("J4", "B", 3.0, 0.0, 9.0)]
    techs = [{"id": "T0", "primary": "A", "skills": ["A"]},
             {"id": "T1", "primary": "A", "skills": ["A"]},
             {"id": "T2", "primary": "B", "skills": ["B"]}]
    inst, ov = _tiny(orders, techs)
    sched = CrewDispatchEnv(inst, ov, c_max=2).run_selector(
        get_selector("edd"), method="edd", seed=SEED)
    assert validate_crew(inst, sched, ov, c_max=2)["feasible"]
    return inst, ov, sched


def _codes(res):
    return {v[:3] for v in res["violations"]}


def test_validator_rejects_wrong_crew_size():
    inst, ov, sched = _valid_case()
    bad = copy.deepcopy(sched)
    drop = next(i for i, a in enumerate(bad["assignments"])
                if a["wo"] == "J2" and a["tech"] == "T1")
    del bad["assignments"][drop]
    res = validate_crew(inst, bad, ov, c_max=2)
    assert not res["feasible"] and "(g)" in _codes(res)
    # The same schedule under c_max = 1 has a crew where the rule gives one.
    res1 = validate_crew(inst, sched, ov, c_max=1)
    assert not res1["feasible"] and "(g)" in _codes(res1)


def test_validator_rejects_ineligible_member():
    inst, ov, sched = _valid_case()
    bad = copy.deepcopy(sched)
    for a in bad["assignments"]:
        if a["wo"] == "J2" and a["tech"] == "T1":
            a["tech"] = "T2"
    res = validate_crew(inst, bad, ov, c_max=2)
    assert not res["feasible"] and "(b)" in _codes(res)


def test_validator_rejects_overlap():
    inst, ov, sched = _valid_case()
    bad = copy.deepcopy(sched)
    for a in bad["assignments"]:
        if a["wo"] == "J3":                     # 10 -> 12 on T0; pull to 9
            a["start_bh"] -= 1.0
            a["end_bh"] -= 1.0
    res = validate_crew(inst, bad, ov, c_max=2)
    assert not res["feasible"] and "(e)" in _codes(res)


def test_validator_rejects_short_elapsed_time():
    inst, ov, sched = _valid_case()
    bad = copy.deepcopy(sched)
    for a in bad["assignments"]:
        if a["wo"] == "J2":                     # both members, end 10 -> 9.5
            a["end_bh"] -= 0.5
    res = validate_crew(inst, bad, ov, c_max=2)
    assert not res["feasible"] and "(d)" in _codes(res)


def test_validator_rejects_unshared_end():
    inst, ov, sched = _valid_case()
    bad = copy.deepcopy(sched)
    for a in bad["assignments"]:
        if a["wo"] == "J2" and a["tech"] == "T1":
            a["end_bh"] += 0.5
    res = validate_crew(inst, bad, ov, c_max=2)
    assert not res["feasible"] and "(s)" in _codes(res)


# --------------------------------------------------------------------------- #
# (iv) stress: no deadlock                                                     #
# --------------------------------------------------------------------------- #
def _stress_instance(seed, n=160):
    rng = random.Random(seed)
    trades = ["A", "B", "C"]
    orders = []
    for i in range(n):
        g = rng.choice(trades)
        p = rng.choice([0.5, 2.0, 7.9, 8.0, 8.01, 15.5, 23.7, 40.0, 96.0])
        rel = float(rng.randrange(0, 60))
        orders.append(_wo("W%03d" % i, g, p, rel, rel + rng.choice(
            [4.0, 16.0, 80.0]), w=float(rng.choice([1, 2, 4, 8])),
            prio=rng.choice([1, 2, 3, 4])))
    # k(A) = 4, k(B) = 2, k(C) = 1
    techs = []
    tid = 0
    for g, k in (("A", 4), ("B", 2), ("C", 1)):
        for _ in range(k):
            techs.append({"id": "T%d" % tid, "primary": g, "skills": [g]})
            tid += 1
    return orders, techs


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_stress_no_deadlock(seed):
    orders, techs = _stress_instance(seed)
    for structure in ("dedicated", "chain", "full"):
        tt = copy.deepcopy(techs)
        if structure == "chain":
            succ = {"A": "B", "B": "C", "C": "A"}
            for t in tt:
                t["skills"] = [t["primary"], succ[t["primary"]]]
        elif structure == "full":
            for t in tt:
                t["skills"] = [t["primary"]] + sorted(
                    {"A", "B", "C"} - {t["primary"]})
        for eta in (1.0, 0.8):
            inst, ov = _tiny(copy.deepcopy(orders), tt, eta=eta,
                             structure=structure)
            inst["trades"] = ["A", "B", "C"]
            for c_max in (1, 2, 3, 4):
                for meth in ALL_RULES:
                    env = CrewDispatchEnv(inst, ov, c_max=c_max)
                    sched = env.run_selector(get_selector(meth), method=meth,
                                             seed=SEED)
                    res = validate_crew(inst, sched, ov, c_max=c_max)
                    assert res["feasible"], (structure, eta, c_max, meth,
                                             res["violations"][:3])


def test_stress_real_instance_and_structure_invariant_sizes():
    inst = _load("c12_replay_400_0100")
    for m in (0.6, 1.0):
        sizes = None
        for st, phi, eta in (("dedicated", None, 1.0), ("chain", 1.0, 0.8),
                             ("full", None, 0.8)):
            ov = _overlay(inst, st, phi, eta, m)
            s = crew_sizes(inst["work_orders"], ov["technicians"], 4)
            sizes = sizes or s
            assert s == sizes, "crew sizes differ across structures"
            for meth in ALL_RULES:
                sched = CrewDispatchEnv(inst, ov, c_max=4).run_selector(
                    get_selector(meth), method=meth, seed=SEED)
                res = validate_crew(inst, sched, ov, c_max=4)
                assert res["feasible"], res["violations"][:3]
        assert max(sizes.values()) == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
