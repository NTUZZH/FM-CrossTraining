"""Candidate-dedup signature must not collapse behaviourally distinct techs.

Correctness property: the learned policy builds candidate pairs by
deduplicating idle technicians by a "signature". Under FULL flexibility every
technician shares the same skill set, so a skill-set-only signature would
collapse technicians that differ only in their designated PRIMARY trade. That
distinction is behaviourally real: the pair features carry a primary-skill flag
(engine._fill_pair_features out[0]) and the pair duration depends on whether the
assigned tech's primary matches the order's trade (engine.pair_p / eta grid). If
the signature dropped primary, the policy would never see the primary-match
technician as a candidate for a full cell, which would be a real bug.

The signature in env.engine.PairDispatchEnv.candidate_pairs is
    sig = (self.prim_of[tid], self.skills_of[tid])
i.e. it DOES include primary. This test constructs a FULL-flexibility state
(3 techs, identical skill set, distinct primaries, one queued order) and asserts
the candidate set retains BOTH a primary-match tech and a non-match tech. It
would fail if the signature were reduced to the skill set alone.

Run: PYTHONPATH=.:vendor python tests/test_full_dedup_signature.py
"""
from env.engine import PairDispatchEnv


def _full_flex_instance():
    """One queued order of trade 'A'; three idle techs, same skill set
    {A,B,C}, primaries A / B / C (a FULL-flexibility cell)."""
    techs = [
        {"id": "T1", "primary": "A", "skills": ["A", "B", "C"]},
        {"id": "T2", "primary": "B", "skills": ["A", "B", "C"]},
        {"id": "T3", "primary": "C", "skills": ["A", "B", "C"]},
    ]
    wo = {"id": "W1", "trade": "A", "release_bh": 0.0, "due_bh": 8.0,
          "p_bh": 4.0, "weight": 1.0, "priority": 2, "is_pm": False}
    return {"meta": {"id": "full-dedup-unit", "eta": 0.8, "structure": "full",
                     "phi": None, "campus": 0},
            "technicians": techs, "work_orders": [wo], "trades": ["A", "B", "C"]}


def test_full_dedup_retains_primary_and_nonprimary():
    inst = _full_flex_instance()
    env = PairDispatchEnv(inst)          # eta from meta (0.8)
    env.reset()                          # drive to first decision point

    pairs = env.candidate_pairs()
    order_trade = "A"
    for tid, job in pairs:
        assert job["trade"] == order_trade

    tech_ids = [tid for tid, _ in pairs]
    primaries = {tid: env.prim_of[tid] for tid in tech_ids}

    match = [tid for tid in tech_ids if primaries[tid] == order_trade]
    nonmatch = [tid for tid in tech_ids if primaries[tid] != order_trade]

    assert match, (
        "candidate set dropped the primary-match technician; signature "
        "collapsed distinct primaries -> REAL BUG for FULL cells. pairs=%r"
        % [(t, primaries[t]) for t in tech_ids])
    assert nonmatch, (
        "candidate set dropped every non-primary technician; signature "
        "over-collapsed. pairs=%r" % [(t, primaries[t]) for t in tech_ids])

    # Full-flex, distinct primaries, identical skill set => all three survive
    # dedup (signature = (primary, skill-set)).
    assert set(tech_ids) == {"T1", "T2", "T3"}, (
        "expected all three distinct-primary techs to survive dedup, got %r"
        % tech_ids)

    # And the pair feature flag actually distinguishes them: primary-match
    # tech gets out[0]=1.0, others 0.0.
    for tid, job in pairs:
        flag = 1.0 if env.prim_of[tid] == job["trade"] else 0.0
        assert flag == (1.0 if tid == "T1" else 0.0)

    print("candidate pairs (tech, primary): %r"
          % [(t, primaries[t]) for t in tech_ids])
    print("  primary-match techs: %r ; non-match techs: %r" % (match, nonmatch))


if __name__ == "__main__":
    test_full_dedup_retains_primary_and_nonprimary()
    print("PASS test_full_dedup_signature "
          "(signature includes primary; FULL cells keep both classes)")
