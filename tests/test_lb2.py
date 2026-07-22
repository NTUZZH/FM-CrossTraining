"""Reward-bound v2 unit tests.

Run: PYTHONPATH=.:vendor python tests/test_lb2.py

- exact reduction to Y1's LB at L0 with eta = 1 (numerical, vs fmwos_y1.lb)
- telescoping of the shaped return to -TWT/100 at machine precision
- admissibility vs exhaustive-optimal TWT on tiny random flexible instances
  (brute-force optimum over pair-permutation schedules; the CP-SAT
  cross-check on 50 static snapshots is added with methods/cpsat2)
"""
import glob
import itertools
import json
import os
import random
from pathlib import Path

from fmwos_y1 import lb as lb_y1
from env import lb2
from env.engine import PairDispatchEnv
from env.conventions import pair_p_bh
from methods.rules import get_selector
from overlays.build import build_overlay

_REPO = Path(__file__).resolve().parents[1]
Y1_ROOT = Path(os.environ.get("FMWOS_Y1_ROOT", _REPO.parent / "FM-Scheduling"))
INST_GLOB = str(Y1_ROOT / "data/processed/instances/c*/replay/150/*.json")


def test_l0_reduction_vs_y1():
    rng = random.Random(1)
    for _ in range(300):
        n_trades = rng.randint(1, 4)
        queues = {}
        tech_free = {}
        skills = {}
        tid = 0
        t = rng.uniform(0, 100)
        for g in range(n_trades):
            gname = "G%d" % g
            queues[gname] = [(rng.uniform(0.1, 8), t + rng.uniform(-20, 60),
                              rng.choice([1, 2, 4, 8]))
                             for _ in range(rng.randint(0, 6))]
            for _ in range(rng.randint(1, 3)):
                tech_free["T%d" % tid] = t + rng.uniform(-5, 30)
                skills["T%d" % tid] = (gname,)
                tid += 1
        v2 = lb2.lb_remaining_v2(queues, tech_free, skills, t, 1.0)
        y1 = lb_y1.lb_remaining(
            {g: q for g, q in queues.items()},
            {g: [tech_free[u] for u in tech_free if skills[u] == (g,)]
             for g in queues},
            t)
        assert abs(v2 - y1) < 1e-9, (v2, y1)


def test_telescoping():
    paths = sorted(glob.glob(INST_GLOB))
    rng = random.Random(2)
    from overlays.build import load_crews
    cap = str(Y1_ROOT / "results/p1_calib/capacity.csv")
    for p in rng.sample(paths, 4):
        inst = json.load(open(p))
        campus = inst["meta"]["campus"]
        ov = build_overlay(campus, load_crews(cap, campus), "chain", 1.0,
                           0.8, 0.6)
        for mode in ("shaped", "realized", "terminal"):
            env = PairDispatchEnv(inst, ov, reward_mode=mode)
            obs = env.reset()
            total = 0.0
            done = env._done
            rng2 = random.Random(3)
            while not done:
                a = rng2.randrange(obs["n"])
                obs, r, done, info = env.step(a)
                total += r
            twt = env._realized
            assert abs(total - (-twt / 100.0)) < 1e-6, (mode, total, twt)


def _brute_force_optimum(inst, ov):
    """Exhaustive best TWT over all non-delay pair-selection policies."""
    env = PairDispatchEnv(inst, ov)

    best = [float("inf")]

    def rec(env_state_actions):
        # depth-first over action sequences using fresh episodes (tiny n)
        env2 = PairDispatchEnv(inst, ov)
        obs = env2.reset()
        for a in env_state_actions:
            obs, _, done, _ = env2.step(a)
        if env2._done:
            best[0] = min(best[0], env2._realized)
            return
        for a in range(obs["n"]):
            rec(env_state_actions + [a])

    rec([])
    return best[0]


def test_admissibility_tiny_flexible():
    """LB at t=0 (empty realized) never exceeds the true optimal TWT.

    Note: the brute force explores non-delay schedules; the LB is admissible
    for ALL schedules, so LB <= optimal non-delay TWT is implied.
    """
    rng = random.Random(4)
    for trial in range(25):
        trades = ["A", "B"]
        wos = []
        for i in range(rng.randint(2, 5)):
            p = round(rng.uniform(0.5, 4.0), 2)
            r = 0.0
            pr = rng.choice([1, 2, 3, 4])
            sla = {1: 2.0, 2: 6.0, 3: 20.0, 4: 42.0}[pr]
            wos.append({"id": "w%d" % i, "trade": rng.choice(trades),
                        "p_bh": p, "release_bh": r, "due_bh": r + sla,
                        "priority": pr,
                        "weight": {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}[pr],
                        "building": None, "is_pm": False})
        inst = {"meta": {"id": "tiny%d" % trial, "campus": 5},
                "trades": trades,
                "technicians": [{"id": "T0", "trade": "A"},
                                {"id": "T1", "trade": "B"}],
                "work_orders": wos}
        crews = [{"trade": "A", "crew": 1, "volume": 2.0},
                 {"trade": "B", "crew": 1, "volume": 1.0}]
        ov = build_overlay(5, crews, "full", None, 0.8, 1.0)
        opt = _brute_force_optimum(inst, ov)
        queues = {}
        for w in wos:
            queues.setdefault(w["trade"], []).append(
                (w["p_bh"], w["due_bh"], w["weight"]))
        skills = {"T0": ("A", "B"), "T1": ("A", "B")}
        free = {"T0": 0.0, "T1": 0.0}
        lb = lb2.lb_remaining_v2(queues, free, skills, 0.0, 0.8)
        assert lb <= opt + 1e-9, (trial, lb, opt)


def test_rho_constant_is_admissible_on_grid():
    """The conversion constant lower-bounds w/p~ for BOTH realised durations.

    (This test previously encoded a draft constant eta/p and correctly
    failed: ceil_grid rounds up, so w*eta/p_j can exceed
    w/ceil_grid(p_j/eta).)"""
    for p in (0.01, 0.37, 1.2345, 7.99):
        for eta in (0.75, 0.8, 0.9, 1.0):
            worst = pair_p_bh(p, False, eta)
            rho = 1.0 / worst
            for realised in (p, worst):        # primary / secondary
                assert rho <= 1.0 / realised + 1e-12
            if eta == 1.0:
                assert abs(rho - 1.0 / p) < 1e-12   # Y1 reduction intact


if __name__ == "__main__":
    test_l0_reduction_vs_y1()
    print("PASS test_l0_reduction_vs_y1 (300 random states)")
    test_telescoping()
    print("PASS test_telescoping (3 reward modes x 4 flexible episodes)")
    test_admissibility_tiny_flexible()
    print("PASS test_admissibility_tiny_flexible (25 brute-forced instances)")
    test_rho_constant_is_admissible_on_grid()
    print("PASS test_rho_constant_is_admissible_on_grid")
    print("OK lb2 tests")
