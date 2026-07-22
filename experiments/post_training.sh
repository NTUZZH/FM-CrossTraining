#!/bin/bash
# Post-training pipeline (idempotent).
# 1. verdict-class amendment (refuses if fleet unfinished) + git commit
# 2. policy evaluation on every family + rolling on tier1 + static rl
# 3. rebuild numbers + gates + figures
set -e
PY=${PY:-python}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

n_main=$(ls results/train/mlp_seed*/done.json results/train/attn_seed*/done.json 2>/dev/null | grep -Ec "(mlp_seed3|attn_seed4)")
echo "main-pool jobs done: $n_main/20"
if [ "$n_main" -lt 20 ]; then
  echo "main pools unfinished; abort (idempotent, rerun later)"; exit 1
fi

if [ ! -f results/gates/verdict_class.json ]; then
  PYTHONPATH=.:vendor $PY experiments/verdict_class.py
  git add protocol/Y2_protocol.md results/gates/verdict_class.json
  git commit -q -m "Amendment A1: verdict class recorded (pre-registered rule) before test eval"
  echo "verdict-class amendment committed"
fi

mkdir -p logs
for fam in tier1 tier2 e4 e3; do
  PYTHONPATH=.:vendor $PY experiments/run_dynamic.py --family $fam \
    --methods rules,rl --workers 22 > "logs/${fam}_rl.log" 2>&1
  echo "$fam rl pass done"
done
PYTHONPATH=.:vendor $PY experiments/run_e5.py --methods rules,rl \
  --workers 22 > logs/e5_rl.log 2>&1
echo "e5 rl pass done"
PYTHONPATH=.:vendor $PY experiments/run_dynamic.py --family tier1 \
  --methods rules,rl,rollcp --workers 12 > logs/tier1_rollcp.log 2>&1
echo "tier1 rollcp pass done"
PYTHONPATH=.:vendor $PY experiments/run_static.py --methods rules,cpsat,ga,rl \
  --workers 11 > logs/e1_rl.log 2>&1
echo "e1 rl pass done"

n_spec=$(ls results/train/spec_*_seed*/done.json 2>/dev/null | wc -l)
if [ "$n_spec" -ge 6 ]; then
  PYTHONPATH=.:vendor $PY experiments/run_dynamic.py --family tier1 \
    --methods rules,rl,specs --workers 22 > logs/tier1_specs.log 2>&1
  echo "tier1 specialist pass done"
else
  echo "specialists $n_spec/6 done; specialist pass deferred (rerun me)"
fi

PYTHONPATH=.:vendor $PY analysis/build_all.py --figures
# Resolve the branch from the gates and fill branch-dependent prose keys,
# then rebuild so numbers_branch.json merges into numbers.tex.
PYTHONPATH=.:vendor $PY analysis/finalize_branch.py | tee logs/branch.log
PYTHONPATH=.:vendor $PY analysis/build_all.py --figures
echo "POST_TRAINING_DONE"
