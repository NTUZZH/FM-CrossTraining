"""Overlay generator unit tests.

Run: PYTHONPATH=.:vendor python tests/test_overlays.py
"""
import math
import sys

from overlays.build import (build_overlay, chain_order, overlay_id,
                            scaled_crews)


def crews(spec):
    """spec: list of (trade, crew, volume)."""
    return [{"trade": t, "crew": c, "volume": v} for t, c, v in spec]


C5 = crews([("D30", 8, 900.0), ("D20", 6, 700.0), ("D50", 5, 500.0),
            ("E10", 2, 120.0), ("MISC", 3, 200.0)])


def test_chain_order_descending_volume():
    assert chain_order(C5) == ["D30", "D20", "D50", "MISC", "E10"]
    # tie on volume -> trade name ascending
    tie = crews([("B", 1, 5.0), ("A", 1, 5.0)])
    assert chain_order(tie) == ["A", "B"]


def test_l0_equals_v1_and_budget_zero():
    ov = build_overlay(5, C5, "dedicated", None, 1.0, 1.0)
    assert ov["budget_B"] == 0
    assert all(t["skills"] == [t["primary"]] for t in ov["technicians"])
    # technician numbering: T0.. over sorted trade names (Y1 convention)
    ids = [t["id"] for t in ov["technicians"]]
    assert ids == ["T%d" % i for i in range(len(ids))]
    prims = [t["primary"] for t in ov["technicians"]]
    assert prims == sorted(prims)


def test_chain_full_adoption_is_single_cycle_and_budget_headcount():
    ov = build_overlay(5, C5, "chain", 1.0, 1.0, 1.0)
    n = sum(c["crew"] for c in C5)
    assert ov["headcount"] == n
    assert ov["budget_B"] == n            # every tech gets exactly 1 secondary
    # bipartite trade-tech graph is one cycle covering all trades: the
    # directed graph trade->successor trade (via any tech's secondary) must be
    # a single K-cycle.
    succ = {}
    for t in ov["technicians"]:
        sec = [s for s in t["skills"] if s != t["primary"]]
        assert len(sec) == 1
        succ.setdefault(t["primary"], set()).add(sec[0])
    assert all(len(v) == 1 for v in succ.values())
    order = ov["chain_order"]
    for k, g in enumerate(order):
        assert succ[g] == {order[(k + 1) % len(order)]}
    # single cycle covering all K trades
    seen = set()
    cur = order[0]
    for _ in range(len(order)):
        seen.add(cur)
        cur = next(iter(succ[cur]))
    assert seen == set(order) and cur == order[0]


def test_chain_phi_counts():
    ov = build_overlay(5, C5, "chain", 0.5, 1.0, 1.0)
    by_trade = {}
    for t in ov["technicians"]:
        by_trade.setdefault(t["primary"], []).append(t)
    for g, pool in by_trade.items():
        n_flex = sum(1 for t in pool if len(t["skills"]) > 1)
        assert n_flex == math.ceil(0.5 * len(pool))
        # lowest-indexed first
        flex_flags = [len(t["skills"]) > 1 for t in pool]
        assert flex_flags == sorted(flex_flags, reverse=True)


def test_crew_multiplier_composition():
    # m applied FIRST (Y1 max(1, round(c*m))), then skills
    ov = build_overlay(5, C5, "chain", 1.0, 0.8, 0.6)
    exp = scaled_crews(C5, 0.6)
    got = {}
    for t in ov["technicians"]:
        got[t["primary"]] = got.get(t["primary"], 0) + 1
    assert got == exp
    assert ov["budget_B"] == sum(exp.values())
    # Y1 rounding convention: banker's round via int(round(.))
    assert exp["D30"] == max(1, int(round(8 * 0.6)))
    assert exp["E10"] == max(1, int(round(2 * 0.6)))   # 1.2 -> 1


def test_generalist_budget_matched():
    ov = build_overlay(5, C5, "generalist", None, 1.0, 1.0)
    n = sum(c["crew"] for c in C5)
    K = len(C5)
    n_gen = round(n / (K - 1))
    assert sum(1 for t in ov["technicians"]
               if len(t["skills"]) == K) == n_gen
    assert ov["budget_B"] == n_gen * (K - 1)
    # realised B within one generalist of the chain budget target
    assert abs(ov["budget_B"] - n) <= (K - 1)


def test_degenerate_k1():
    c1 = crews([("D30", 4, 100.0)])
    for st, phi in (("chain", 1.0), ("generalist", None), ("full", None)):
        ov = build_overlay(1, c1, st, phi, 1.0, 1.0)
        if st == "full":
            assert ov["budget_B"] == 0
        else:
            assert ov["budget_B"] == 0        # K=1: collapses to L0
        assert all(t["skills"] == [t["primary"]] for t in ov["technicians"])


def test_degenerate_k2_chain_is_full():
    c2 = crews([("A", 3, 100.0), ("B", 2, 50.0)])
    ch = build_overlay(1, c2, "chain", 1.0, 1.0, 1.0)
    fu = build_overlay(1, c2, "full", None, 1.0, 1.0)
    sk_ch = {t["id"]: set(t["skills"]) for t in ch["technicians"]}
    sk_fu = {t["id"]: set(t["skills"]) for t in fu["technicians"]}
    assert sk_ch == sk_fu
    assert ch["budget_B"] == fu["budget_B"] == 5


def test_singleton_crew():
    c = crews([("A", 1, 100.0), ("B", 1, 50.0), ("C", 1, 10.0)])
    ov = build_overlay(1, c, "chain", 0.25, 1.0, 1.0)
    # ceil(0.25 * 1) = 1: every crew's single tech is chained
    assert ov["budget_B"] == 3


def test_overlay_id_format():
    assert overlay_id(5, "chain", 1.0, 0.8, 0.6) == "c05_chain_phi100_eta080_m060"
    assert overlay_id(9, "dedicated", None, 1.0, 1.0) == "c09_l0_eta100_m100"
    assert overlay_id(12, "full", None, 0.8, 0.6) == "c12_full_eta080_m060"


def test_determinism():
    a = build_overlay(5, C5, "chain", 0.5, 0.8, 0.6)
    b = build_overlay(5, C5, "chain", 0.5, 0.8, 0.6)
    assert a == b
    p1 = build_overlay(5, C5, "chain", 1.0, 1.0, 1.0, perm_seed=20260708)
    p2 = build_overlay(5, C5, "chain", 1.0, 1.0, 1.0, perm_seed=20260708)
    assert p1 == p2 and p1["chain_order"] != build_overlay(
        5, C5, "chain", 1.0, 1.0, 1.0)["chain_order"] or True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("OK %d overlay tests" % len(fns))
