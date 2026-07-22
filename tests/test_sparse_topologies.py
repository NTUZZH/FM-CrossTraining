"""Unit tests for the same-budget sparse-topology controls.

Every control must give each technician exactly one secondary skill
(B = headcount = B(CHAIN(1.0))), keep primaries untouched, and realise its
declared trade-level topology.
"""
import pytest

from overlays.build import build_overlay, chain_order, scaled_crews
from overlays import topology_overlays as rt


def crews(k, crew=3):
    return [{"trade": "G%02d" % i, "crew": crew, "volume": 100.0 - i}
            for i in range(k)]


LIC_CREWS = [
    {"trade": "D50", "crew": 4, "volume": 90.0},   # licensed
    {"trade": "D30", "crew": 3, "volume": 80.0},
    {"trade": "D20", "crew": 3, "volume": 70.0},   # licensed
    {"trade": "C10", "crew": 2, "volume": 60.0},
    {"trade": "B30", "crew": 2, "volume": 50.0},
]


def _build(cs, sigma, order, label="x"):
    return rt.build_sigma_variant(1, cs, sigma, 1.0, 1.0, order=order,
                                  struct_label=label, variant=label)


@pytest.mark.parametrize("k", [2, 3, 4, 5, 11, 13])
def test_budget_matches_chain_for_every_control(k):
    cs = crews(k)
    order = chain_order(cs)
    ref = build_overlay(1, cs, "chain", 1.0, 1.0, 1.0)
    sigmas = {"pairs": rt.sigma_pairs(order),
              "star": rt.sigma_star(order),
              "rand1": rt.sigma_rand1(order, 20260901)}
    for name, sig in sigmas.items():
        ov = _build(cs, sig, order, name)
        assert ov["budget_B"] == ref["budget_B"] == ov["headcount"]
        assert all(len(t["skills"]) == 2 for t in ov["technicians"])
        prim = {t["id"]: t["primary"] for t in ov["technicians"]}
        prim_ref = {t["id"]: t["primary"] for t in ref["technicians"]}
        assert prim == prim_ref


def test_pairs_reciprocal_even_k():
    order = [c["trade"] for c in crews(6)]
    sig = rt.sigma_pairs(order)
    assert all(sig[sig[t]] == t for t in order)
    # disjoint 2-cycles: three components on six trades
    assert rt.topology_descriptors(
        sig, order, {t: 1 for t in order})["weak_components"] == 3


def test_pairs_odd_k_has_single_triple():
    order = [c["trade"] for c in crews(7)]
    sig = rt.sigma_pairs(order)
    trip = order[-3:]
    assert sig[trip[0]] == trip[1] and sig[trip[1]] == trip[2] \
        and sig[trip[2]] == trip[0]
    assert all(sig[sig[t]] == t for t in order[:4])


def test_pairs_k3_is_three_cycle_and_k1_empty():
    order3 = [c["trade"] for c in crews(3)]
    sig = rt.sigma_pairs(order3)
    assert sorted(sig) == sorted(order3) and len(set(sig.values())) == 3
    assert rt.sigma_pairs(["G00"]) == {}


def test_star_concentrates_indegree_on_hub():
    cs = crews(5)
    order = chain_order(cs)
    sig = rt.sigma_star(order)
    d = rt.topology_descriptors(sig, order, scaled_crews(cs, 1.0))
    assert d["in_degree_max"] == len(order) - 1
    assert d["weak_components"] == 1
    # only the hub and its own target receive help
    assert len(d["uncovered_trades"]) == len(order) - 2


def test_rand1_deterministic_and_unconstrained():
    order = [c["trade"] for c in crews(13)]
    a = rt.sigma_rand1(order, 20260901)
    b = rt.sigma_rand1(order, 20260901)
    assert a == b
    assert all(a[t] != t for t in order)
    assert rt.sigma_rand1(order, 20260902) != a


def test_feas_prohibits_arcs_into_licensed_trades():
    order = rt.adjacency_order(LIC_CREWS)
    sig = rt.sigma_feas(order)
    assert set(sig) == set(order)          # every trade still gives one arc
    assert not set(sig.values()) & set(rt.LICENSED_TRADES)
    d = rt.topology_descriptors(sig, order, scaled_crews(LIC_CREWS, 1.0))
    assert set(d["uncovered_trades"]) == {"D50", "D20"}
    ov = _build(LIC_CREWS, sig, order, "feas")
    assert ov["budget_B"] == ov["headcount"]


def test_feas_raises_when_no_legal_target():
    cs = [{"trade": "D50", "crew": 2, "volume": 9.0},
          {"trade": "D20", "crew": 2, "volume": 8.0}]
    with pytest.raises(ValueError):
        rt.sigma_feas(rt.adjacency_order(cs))
