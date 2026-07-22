# FM-CrossTraining: a skill-overlay framework for counterfactual workforce design from CMMS work orders

Framework code, benchmark overlays, methods, and results for the paper
*"FM-CrossTraining: a skill-overlay framework for counterfactual workforce design from CMMS work orders"*.

Cross-training is released as deterministic **resource overlays** on the
instances of the open single-skill dispatching benchmark
([FM-Scheduling](https://github.com/NTUZZH/FM-Scheduling)): the workload never
changes, only the skill structure of the crew does. The release includes the
flexibility-ladder overlay generator (L0 / CHAIN(&phi;) / GEN / FULL), a
pair-selection dispatch engine with an independent validator as sole scorer,
an upgraded rule and solver suite, a pair-scoring learned dispatcher, the
dated pre-specified protocol log, and the scored result files behind every
figure and table in the paper.

## Layout

- `overlays/` -- the flexibility-ladder generator, unit-tested;
  `generate_all.py` materialises every released overlay bit-for-bit from the
  released crew calibration and a recorded seed.
- `env/` -- pair-selection engine (`engine.py`), independent validator
  (`validator2.py`), admissible reward lower bound for overlapping skill
  pools (`lb2.py`), grid conventions (`conventions.py`).
- `methods/` -- priority rules with the technician tie-break and the two
  flexibility-aware rules LFJ-ATC and ATC-&eta; (`rules.py`), CP-SAT exact
  and rolling references (`cpsat2.py`, `rolling2.py`), genetic algorithm
  (`ga2.py`), pair policies (`policy2.py`), PPO trainer (`train2.py`).
- `experiments/` -- runners: `anchor_l0.py` (machine-precision regression
  anchor against the single-skill release), `run_static.py`,
  `run_dynamic.py`, `run_e5.py` (sensitivity), `launch_training.sh`,
  `post_training.sh`.
- `analysis/` -- `gates.py` (the pre-specified Gate P / Gate C tests),
  `build_all.py` (every number in the paper), `figures.py` (every exhibit).
- `protocol/Y2_protocol.md` -- the dated protocol log: gates, thresholds,
  seeds, and the reporting plan, committed before any verdict experiment ran.
- `results/` -- aggregated per-method scored result files (`results.csv` per
  experiment family) and the gate outputs (`gates/`).
- `tests/` -- unit and parity tests (overlay accounting, engine parity,
  validator, reward-bound admissibility).

## Reproduce

```bash
# Python >= 3.10 with torch, ortools, numpy, pandas, scipy
export PYTHONPATH=.:vendor

# 1. regression anchor (must be GREEN before trusting any result)
python experiments/anchor_l0.py --workers 24

# 2. unit tests (pytest suites + the two module-style parity harnesses)
python -m pytest tests/ -q --ignore=tests/test_engine_parity.py \
    --ignore=tests/test_flexible_oracle.py
python tests/test_engine_parity.py && python tests/test_flexible_oracle.py

# 3. rule + solver passes
python experiments/run_dynamic.py --family tier1 --methods rules --workers 20
python experiments/run_dynamic.py --family tier2 --methods rules --workers 20
python experiments/run_dynamic.py --family e3    --methods rules --workers 20
python experiments/run_dynamic.py --family e4    --methods rules --workers 20
python experiments/run_e5.py                     --methods rules --workers 20
python experiments/run_static.py --methods rules,cpsat,ga --workers 11

# 4. training (GPU; 10 MLP + 10 attention + 6 specialist seeds)
bash experiments/launch_training.sh cuda 8

# 5. after training: verdict class, policy evaluation, gates, exhibits
bash experiments/post_training.sh
python analysis/build_all.py --figures

# 6. controls and supplementary statistics (CPU)
python experiments/run_e6_patient.py --check --smoke     # then --full / --variants
python experiments/run_e6_patient.py --family e4 --full --i-have-approval  # held-out
python experiments/run_e7_topology.py --check --variants pairs,star,feas,rand1
python experiments/run_e7_topology.py --variants chain_adj,perm,tsel,pairs,star,feas,rand1
python experiments/run_e7_topology.py --variants pairs,feas --campuses 1,2 --ms 0.6,0.8
python experiments/run_e11_optsigma.py && python experiments/run_e7_topology.py --variants opt
python notes/supplementary/sparse_topology_stats.py
python notes/supplementary/robustness_stats.py

# 7. wait-action policy class (E10, protocol amendment A2; GPU then CPU)
bash experiments/launch_wait_training.sh cuda 8
python experiments/run_e10_wait.py --workers 20
```

## Data

Instances are the unchanged single-skill release of
[FM-Scheduling](https://github.com/NTUZZH/FM-Scheduling). The raw FMUCD
dataset is on Mendeley Data (DOI 10.17632/cb8d2nsjss.1, CC BY-NC 4.0); its
SHA-256 is verified before use. Overlays are deterministic functions of the
released crew calibration and are also shipped as files.

## Licence

CC BY-NC 4.0, inherited from FMUCD. Non-commercial.
