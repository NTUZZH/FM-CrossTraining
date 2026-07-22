#!/bin/bash
# Extend the per-structure specialist policies from 3 seeds to the full 10,
# matching the verdict pool's seed budget (addresses the "specialists only 3
# seeds" fairness concern). Adds CHAIN(1.0) seeds 504-510 and FULL seeds
# 604-610; the existing 501-503 / 601-603 are reused. Same locked stack as
# launch_training.sh.
#   bash experiments/launch_specialists_full.sh [device] [parallel]
set -u
DEVICE=${1:-cuda}
PAR=${2:-4}
PY=${PY:-python}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/train logs

run_one() {
  local seed=$1 out=$2 struct=$3
  if [ -f "$out/done.json" ]; then echo "[skip] $out already done"; return 0; fi
  echo "[start $(date +%H:%M:%S)] spec $struct seed $seed -> $out"
  PYTHONPATH=.:vendor Y2_TORCH_THREADS=2 "$PY" -m methods.train2 \
    --arch mlp --seed "$seed" --updates 1200 --device "$DEVICE" \
    --structures="$struct" --out "$out" > "logs/train_spec_${seed}.log" 2>&1
  echo "[done  $(date +%H:%M:%S)] spec $struct seed $seed (exit $?)"
}

JOBS=()
for s in 504 505 506 507 508 509 510; do
  JOBS+=("$s results/train/spec_chain_mlp_seed$s chain1.0")
done
for s in 604 605 606 607 608 609 610; do
  JOBS+=("$s results/train/spec_full_mlp_seed$s full")
done

for j in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 15; done
  # shellcheck disable=SC2086
  run_one $j &
  sleep 3
done
wait
echo "ALL_SPECIALISTS_DONE $(date)"
