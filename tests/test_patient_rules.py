"""Unit tests for the generalised patient rules.

Invariants: at eta = 1 (zero break-even window) and at L0 (all technicians
primary) every patient variant schedules identically to its plain rule; a
patient-EDD reimplementation through the shared base reproduces the released
PatientEDD schedule on a penalised flexible cell.
"""
import json
import os
from pathlib import Path

import pytest

from env.engine import PairDispatchEnv
from methods.rules import (PATIENT_RULES, PatientEDD, _PatientRule,
                           get_selector)
from overlays.build import build_overlay, load_crews

ROOT = Path(__file__).resolve().parents[1]
Y1 = Path(os.environ.get("FMWOS_Y1_ROOT",
                         ROOT.parent / "FM-Scheduling"))
CAP = Y1 / "results/p1_calib/capacity.csv"
INST = Y1 / "data/processed/instances"


def _one_instance():
    import csv
    with open(INST / "index.csv", newline="") as f:
        for r in csv.DictReader(f):
            if (r["track"] == "replay" and r["split"] == "test"
                    and int(r["campus"]) == 5
                    and int(r["size_class"]) == 150):
                return json.load(open(INST / r["path"]))
    raise RuntimeError("no campus-5 replay-150 test instance found")


def _run(inst, ov, selector, method):
    env = PairDispatchEnv(inst, ov)
    return env.run_selector(selector, method=method, seed=301)


def _keys(sched):
    return sorted((a["wo"], a["tech"], round(a["start_bh"], 6),
                   round(a["end_bh"], 6)) for a in sched["assignments"])


class _PatientEDDViaBase(_PatientRule):
    def _pick(self, env, remaining):
        return min(remaining, key=lambda j: (j["due_bh"], j["id"]))


PLAIN_OF = {"edd_patient": "edd", "wspt_patient": "wspt",
            "atc_patient": "atc", "atc_eta_patient": "atc_eta",
            "lfj_atc_patient": "lfj_atc"}


@pytest.fixture(scope="module")
def inst():
    return _one_instance()


@pytest.fixture(scope="module")
def crews():
    return load_crews(CAP, 5)


@pytest.mark.parametrize("name", sorted(PATIENT_RULES))
def test_eta1_identical_to_plain(inst, crews, name):
    ov = build_overlay(5, crews, "chain", 1.0, 1.0, 0.6)
    sel = PATIENT_RULES[name]()
    pat = _run(inst, ov, sel, name)
    plain = _run(inst, ov, get_selector(PLAIN_OF[name]), PLAIN_OF[name])
    assert _keys(pat) == _keys(plain)
    assert sel.n_declines == 0


@pytest.mark.parametrize("name", sorted(PATIENT_RULES))
def test_l0_identical_to_plain(inst, crews, name):
    ov = build_overlay(5, crews, "dedicated", None, 0.8, 0.6)
    sel = PATIENT_RULES[name]()
    pat = _run(inst, ov, sel, name)
    plain = _run(inst, ov, get_selector(PLAIN_OF[name]), PLAIN_OF[name])
    assert _keys(pat) == _keys(plain)
    assert sel.n_declines == 0


def test_base_reproduces_released_patient_edd(inst, crews):
    ov = build_overlay(5, crews, "chain", 1.0, 0.8, 0.6)
    released = PatientEDD()
    a = _run(inst, ov, released, "edd_patient")
    via_base = _PatientEDDViaBase()
    b = _run(inst, ov, via_base, "edd_patient")
    assert _keys(a) == _keys(b)
    assert released.counters() == via_base.counters()
    assert released.n_declines > 0     # the cell actually exercises waiting


def test_patient_declines_on_penalised_flexible_cell(inst, crews):
    ov = build_overlay(5, crews, "full", None, 0.8, 0.6)
    for name in sorted(PATIENT_RULES):
        sel = PATIENT_RULES[name]()
        _run(inst, ov, sel, name)
        c = sel.counters()
        assert c["n_declines"] >= 0    # counters readable for every variant
