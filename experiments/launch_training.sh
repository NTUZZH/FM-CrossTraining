#!/bin/bash
# v2 training fleet (protocol log 6). Runs inside tmux; keeps $PAR jobs live.
#   bash experiments/launch_training.sh [device] [parallel]
set -u
DEVICE=${1:-cuda}
PAR=${2:-8}
PY=${PY:-python}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/train logs

run_one() {
  local arch=$1 seed=$2 out=$3 extra=${4:-}
  if [ -f "$out/done.json" ]; then echo "[skip] $out already done"; return 0; fi
  local mb=0
  if [ "$arch" = "attn" ]; then mb=256; fi   # grad-accum chunk (OOM guard)
  echo "[start $(date +%H:%M:%S)] $arch seed $seed -> $out"
  PYTHONPATH=.:vendor Y2_TORCH_THREADS=2 Y2_MICROBATCH=$mb "$PY" -m methods.train2 \
    --arch "$arch" --seed "$seed" --updates 1200 --device "$DEVICE" \
    --out "$out" $extra > "logs/train_${arch}_${seed}.log" 2>&1
  local rc=$?
  echo "[done  $(date +%H:%M:%S)] $arch seed $seed (exit $rc)"
  if [ $rc -ne 0 ]; then echo "[FAIL] $arch $seed rc=$rc (no done.json -> retried on relaunch)"; fi
}

JOBS=()
for s in 301 302 303 304 305 306 307 308 309 310; do
  JOBS+=("mlp $s results/train/mlp_seed$s")
done
for s in 401 402 403 404 405 406 407 408 409 410; do
  JOBS+=("attn $s results/train/attn_seed$s")
done
for s in 501 502 503; do
  JOBS+=("mlp $s results/train/spec_chain_mlp_seed$s --structures=chain1.0")
done
for s in 601 602 603; do
  JOBS+=("mlp $s results/train/spec_full_mlp_seed$s --structures=full")
done

for j in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 15; done
  # shellcheck disable=SC2086
  run_one $j &
  sleep 3
done
wait
echo "ALL_TRAINING_DONE $(date)"
