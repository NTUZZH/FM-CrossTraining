"""Method-suite v2 tests: cpsat2 / ga2 / rolling2.

Run: PYTHONPATH=.:vendor python tests/test_methods2.py
"""
import glob
import json
import os
import random
from pathlib import Path

from fmwos_y1 import cpsat as cpsat_y1
from env import lb2
from env.engine import PairDispatchEnv
from env.validator2 import validate as validate2
from methods import cpsat2, ga2, rolling2
from methods.rules import get_selector
from overlays.build import build_overlay, load_crews

_REPO = Path(__file__).resolve().parents[1]
Y1_ROOT = Path(os.environ.get("FMWOS_Y1_ROOT", _REPO.parent / "FM-Scheduling"))
CAP = str(Y1_ROOT / "results/p1_calib/capacity.csv")


def small_instances(n, size=50):
    paths = sorted(glob.glob(
        str(Y1_ROOT / ("data/processed/instances/c*/replay/%d/*.json" % size))))
    rng = random.Random(20260710)
    return [json.load(open(p)) for p in rng.sample(paths, n)]


def test_cpsat2_l0_matches_y1():
    """At L0 the v2 model must agree with Y1's solver (objective + status)."""
    for inst in small_instances(4):
        campus = inst["meta"]["campus"]
        ov = build_overlay(campus, load_crews(CAP, campus), "dedicated",
                           None, 1.0, 1.0)
        a = cpsat_y1.solve(inst, time_limit_s=20.0, workers=4)
        b = cpsat2.solve(inst, ov, time_limit_s=20.0, workers=4)
        assert a["status"] == b["status"] == "OPTIMAL", (a["status"],
                                                         b["status"])
        assert abs(a["objective_bh"] - b["objective_bh"]) < 1e-9
        res = validate2(inst, b, ov)
        assert res["feasible"], res["violations"][:3]


def test_cpsat2_flexible_feasible_and_no_worse():
    """Flexible optima are validator-feasible and never worse than L0."""
    for inst in small_instances(3):
        campus = inst["meta"]["campus"]
        crews = load_crews(CAP, campus)
        l0 = cpsat2.solve(inst, build_overlay(campus, crews, "dedicated",
                                              None, 1.0, 0.6),
                          time_limit_s=30.0, workers=4)
        fu_ov = build_overlay(campus, crews, "full", None, 0.8, 0.6)
        fu = cpsat2.solve(inst, fu_ov, time_limit_s=30.0, workers=4)
        res = validate2(inst, fu, fu_ov)
        assert res["feasible"], res["violations"][:3]
        if l0["status"] == "OPTIMAL" and fu["status"] == "OPTIMAL":
            assert fu["objective_bh"] <= l0["objective_bh"] + 1e-6
        # validator WWT within centi-grid slack of the model objective
        assert res["metrics"]["WWT"] <= fu["objective_bh"] + 0.02 * len(
            inst["work_orders"]) * 8


def test_lb_admissible_vs_cpsat(n_snapshots=50):
    """LB <= CP-SAT optimum on small static snapshots."""
    rng = random.Random(20260711)
    checked = 0
    insts = small_instances(10)
    while checked < n_snapshots:
        inst = insts[checked % len(insts)]
        campus = inst["meta"]["campus"]
        crews = load_crews(CAP, campus)
        st, phi, eta = rng.choice([("chain", 1.0, 0.8), ("full", None, 0.8),
                                   ("chain", 0.5, 1.0),
                                   ("dedicated", None, 1.0)])
        ov = build_overlay(campus, crews, st, phi, eta, 0.6)
        # random static snapshot: subset of orders, all released at 0
        k = rng.randint(3, 12)
        wos = rng.sample(inst["work_orders"], min(k, len(inst["work_orders"])))
        snap = {"meta": {"id": "lbsnap%d" % checked, "campus": campus},
                "trades": inst["trades"],
                "technicians": ov["technicians"],
                "work_orders": [dict(w, release_bh=0.0) for w in wos]}
        sol = cpsat2.solve(snap, ov, time_limit_s=15.0, workers=4)
        if sol["status"] != "OPTIMAL":
            checked += 1
            continue
        queues = {}
        for w in snap["work_orders"]:
            queues.setdefault(w["trade"], []).append(
                (w["p_bh"], w["due_bh"], w["weight"]))
        skills = {t["id"]: tuple(t["skills"]) for t in ov["technicians"]}
        free = {t["id"]: 0.0 for t in ov["technicians"]}
        lb = lb2.lb_remaining_v2(queues, free, skills, 0.0, eta)
        # centi-grid slack: model tardiness uses grid starts/durations
        assert lb <= sol["objective_bh"] + 0.03 * len(wos) * 8 + 1e-6, (
            checked, lb, sol["objective_bh"])
        checked += 1
    print("  %d LB-vs-CPSAT snapshots checked" % checked)


def test_ga2_feasible_and_beats_seed_rules():
    for inst in small_instances(2):
        campus = inst["meta"]["campus"]
        ov = build_overlay(campus, load_crews(CAP, campus), "chain", 1.0,
                           0.8, 0.6)
        sched = ga2.solve_ga(inst, ov, budget_s=6.0, seed=301, pop=40)
        res = validate2(inst, sched, ov)
        assert res["feasible"], res["violations"][:3]
        env = PairDispatchEnv(inst, ov)
        edd = env.run_selector(get_selector("edd"), method="edd", seed=301)
        edd_wwt = validate2(inst, edd, ov)["metrics"]["WWT"]
        assert res["metrics"]["WWT"] <= edd_wwt + 1e-9


def test_rolling2_feasible():
    for inst in small_instances(2):
        campus = inst["meta"]["campus"]
        ov = build_overlay(campus, load_crews(CAP, campus), "full", None,
                           0.8, 0.6)
        sched = rolling2.roll_cpsat(inst, ov, budget_s=1.0)
        res = validate2(inst, sched, ov)
        assert res["feasible"], res["violations"][:3]
        assert sched["decisions"] >= 1


if __name__ == "__main__":
    test_cpsat2_l0_matches_y1()
    print("PASS test_cpsat2_l0_matches_y1")
    test_cpsat2_flexible_feasible_and_no_worse()
    print("PASS test_cpsat2_flexible_feasible_and_no_worse")
    test_lb_admissible_vs_cpsat()
    print("PASS test_lb_admissible_vs_cpsat")
    test_ga2_feasible_and_beats_seed_rules()
    print("PASS test_ga2_feasible_and_beats_seed_rules")
    test_rolling2_feasible()
    print("PASS test_rolling2_feasible")
    print("OK methods2 tests")
