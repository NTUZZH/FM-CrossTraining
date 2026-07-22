#!/bin/bash
# E10 wait-action training fleet (protocol log Amendment A2). Mirrors
# launch_training.sh exactly (same $PAR job-keeping, same env caps); the
# ONLY differences per job are --wait, the seed range 701-710, and the
# output directories results/train/wait_mlp_seed7XX.
#   bash experiments/launch_wait_training.sh [device] [parallel]
set -u
DEVICE=${1:-cuda}
PAR=${2:-8}
PY=${PY:-python}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/train logs

run_one() {
  local seed=$1 out=$2
  if [ -f "$out/done.json" ]; then echo "[skip] $out already done"; return 0; fi
  echo "[start $(date +%H:%M:%S)] wait mlp seed $seed -> $out"
  PYTHONPATH=.:vendor Y2_TORCH_THREADS=2 Y2_MICROBATCH=0 "$PY" -m methods.train2 \
    --arch mlp --seed "$seed" --updates 1200 --device "$DEVICE" --wait \
    --out "$out" > "logs/train_wait_${seed}.log" 2>&1
  local rc=$?
  echo "[done  $(date +%H:%M:%S)] wait mlp seed $seed (exit $rc)"
  if [ $rc -ne 0 ]; then echo "[FAIL] wait $seed rc=$rc (no done.json -> retried on relaunch)"; fi
}

for s in 701 702 703 704 705 706 707 708 709 710; do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 15; done
  run_one "$s" "results/train/wait_mlp_seed$s" &
  sleep 3
done
wait
echo "ALL_WAIT_TRAINING_DONE $(date)"
