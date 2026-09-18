#!/bin/bash
# One-command rerun of the roster-calibration sensitivity (E-CAL) and the
# corrective-only / single-technician robustness (E-SUB), end to end:
# crew tables, episodes, regression checks, supplement tables and numbers keys.
#
# The corpus root is resolved by experiments/recalibrate.py: FMWOS_Y1_ROOT if
# set, else data/y1_frozen_v10 when it exists, else the sibling repository.
# Episode shards are named after a digest of that root's calibration table and
# instance index, so a run against a different corpus never reuses the
# episodes of an earlier one.
#
#   experiments/run_ecal_esub.sh [workers]
#   FMWOS_Y1_ROOT=<corpus root> ECAL_CORES=0-9 PYTHON=<interpreter> \
#       experiments/run_ecal_esub.sh 10
#
# Every stage stops the pipeline on a failed regression check.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKERS="${1:-10}"
PY="${PYTHON:-python}"
export PYTHONPATH=".:vendor"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Pin the workers to a core set when one is asked for (ECAL_CORES=0-9) and
# taskset is available; otherwise run unpinned.
PIN=()
if [[ -n "${ECAL_CORES:-}" ]] && command -v taskset >/dev/null 2>&1; then
  PIN=(taskset -c "$ECAL_CORES")
fi

run() { echo; echo "=== $* ==="; "${PIN[@]}" "$PY" "$@"; }

run experiments/recalibrate.py --rebuild --all
run experiments/run_ecal.py --workers "$WORKERS"
run experiments/run_esub.py --all --workers "$WORKERS"
run analysis/gen_ecal.py
run analysis/gen_esub.py

echo
echo "=== done: results/ecal, results/esub, paper/sections/gen_ecal.tex, "
echo "    paper/sections/gen_esub.tex ==="
