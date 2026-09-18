"""Resolve the Y1 corpus root that every runner and analysis module reads.

The released results were computed against the Y1 inputs as they stood in
July 2026: the replay and generator instances under
``data/processed/instances``, the crew calibration
``results/p1_calib/capacity.csv``, the generator parameter packs under
``results/p2_generator``, and the Y1 dynamic evaluation
``results/p4_dyneval/results.csv`` that the L0 anchor checks against. That
corpus was later rebuilt in the sibling repository, so this project carries
its own frozen copy at ``data/y1_frozen_v10`` and reads it by default.

Resolution order:

1. an explicit override passed by the caller;
2. the ``FMWOS_Y1_ROOT`` environment variable;
3. this project's frozen copy, when it is installed;
4. the sibling repository.

Instances, the calibration table and the parameter packs always come from
the same root, so the two corpus versions can never be mixed inside one run.
The raw FMUCD corpus is the exception handled by
``experiments/recalibrate.py``: it is identical in both versions, so the
frozen root need not carry the 1.4 GB file.
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
FROZEN = _PROJECT / "data" / "y1_frozen_v10"
SIBLING = _PROJECT.parent / "PaperY-FMScheduling"

# The file whose presence marks an installed frozen root.
_MARKER = "results/p1_calib/capacity.csv"


def y1_root(override: str | os.PathLike | None = None) -> Path:
    """Return the Y1 root to read, following the order in the module doc."""
    if override:
        return Path(override)
    env = os.environ.get("FMWOS_Y1_ROOT")
    if env:
        return Path(env)
    if (FROZEN / _MARKER).exists():
        return FROZEN
    return SIBLING


def is_frozen(root: str | os.PathLike | None = None) -> bool:
    """True when the resolved root is this project's frozen copy."""
    return Path(root or y1_root()).resolve() == FROZEN.resolve()


Y1_ROOT = y1_root()
