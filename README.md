# FM-CrossTraining: computational framework and open benchmark for multiskill maintenance dispatching

Framework code, benchmark overlays, methods, and results for the paper
*"FM-CrossTraining: Computational Framework and Open Benchmark for Multiskill
Maintenance Dispatching"* (submitted to the ASCE Journal of Computing in
Civil Engineering).

Cross-training is released as deterministic **resource overlays** on the
instances of the open single-skill dispatching benchmark
([FM-Scheduling](https://github.com/NTUZZH/FM-Scheduling)): the workload never
changes, only the skill structure of the crew does. The release includes the
flexibility-ladder overlay generator (L0 / CHAIN(&phi;) / GEN / FULL), a
pair-selection dispatch engine with an independent validator as sole scorer,
an upgraded rule and solver suite, a pair-scoring learned dispatcher, the
dated pre-specified protocol log, and the scored result files behind every
figure and table in the paper.

## Release v1.2.1

Version 1.2.1 adds one re-analysis of released results. `analysis/gen_r2.py`
also splits the dedicated-roster weighted tardiness of every Frame U instance
into the length floor and the waiting-induced remainder under fixed EDD, and
it reports how much of that remainder the chain and full flexibility remove
(Supplemental Text S8). The split reads only released instance and result
files. `analysis/build_all.py` drops the prose fields of the numbers
dictionary that no manuscript source cites. No released result row changed.

## Release v1.2

Version 1.2 adds a crew-size analysis for long work orders, the length floor
of weighted tardiness, and re-analyses of released results. The v1.1 runners
and result files are unchanged, and `analysis/build_all.py` rebuilds every
released number from the shipped result files.

**Hypothetical crew sizes** (Supplemental Table S20 and Text S8). The
work-order records hold no technician count per order, so the main model
reads the labor hours of an order as one technician's uninterrupted work.
`env/crew_engine.py` replaces this reading for orders longer than 8 business
hours. Such an order needs c = min(c_max, ceil(p / 8), k) technicians at
once, where p is its labor hours and k is the primary headcount of its trade.
Its elapsed time is p over the crew's combined rate, rounded up to the
0.01-hour grid, so the order keeps its labor content. The dispatcher keeps
the released rules and holds idle technicians for the highest-ranked crew
order until the crew is complete. `experiments/run_ecrew.py` re-scores L0,
CHAIN(1.0) and FULL with the seven ranked rules and Random for c_max = 1 to 4
on the 763 verdict instances. `env/validator_crew.py` scores every schedule
and imports neither engine. At c_max = 1 the crew engine reproduces all
91,560 rule rows of `results/tier1` bitwise, and the runner stops on any
mismatch.

**Length floor** (Supplemental Text S8). An order that would end after its
due date even if it started at release is late in every schedule.
`analysis/ecrew_floor.py` sums this unavoidable weighted tardiness per
instance and crew scenario, so the tardiness above it is the part that
waiting for a technician causes.

**Re-analyses of released results** (Table 2 of the paper and Supplemental
Tables S21 and S22). `analysis/gen_r2.py` reads only released result and
instance files. It builds the principal-regime table, repeats the Gate C
bootstrap both with and without re-selecting the best observed method in each
resample, counts unique winners and ties in the decision map, and compares
the structures in Frame U at matched aggregate offered load, with the
trade-level loads and the paired chain-minus-full difference.

Results go to `results/ecrew/`: `results.csv` with every scored row, the same
data as `results.parquet`, the realized crew sizes (`crew_sizes.csv`), the
length floor per instance (`floor.csv`), and the summary and analysis JSON.
The resumable shard cache is not shipped.

## Release v1.1

Version 1.1 adds the frozen benchmark inputs and five sensitivity analyses.
The v1.0 runners are unchanged except for the line that locates their input
corpus, which now goes through one shared resolver, and they reproduce the
released gate outputs byte for byte. `analysis/build_all.py` and
`analysis/figures.py` additionally carry the five new tables and the revised
exhibit styling.

**Frozen input corpus.** The single-skill repository re-released its corpus
after the results here were computed, under two construction corrections, so
its current instance files and crew calibration no longer reproduce these
numbers. The inputs as they stood for this work therefore ship with the
release, as `data/y1_frozen_v10/instances_v10.tar.zst` (13.6 MB, 4,986
instance files plus the crew calibration, the priority mapping, the six
generator parameter packs, and the single-skill dynamic evaluation the L0
anchor checks against). Unpack it in place before running anything:

```bash
cd data/y1_frozen_v10 && tar --zstd -xf instances_v10.tar.zst && cd ../..
```

`data/y1_frozen_v10/PROVENANCE.md` records how the corpus was rebuilt, the
checksums, and the verification: the rebuilt calibration is byte-identical to
the table the released sweeps used, the eight dispatching rules reproduce
219,768 released result rows with zero mismatches, and the L0 anchor matches
the single-skill numbers to 2.2e-16 relative. `experiments/y1_root.py`
resolves the corpus root for every runner, analysis module and test, in the
order: explicit argument, `FMWOS_Y1_ROOT`, this frozen copy, the sibling
repository. The raw 1.4 GB FMUCD file is not duplicated here; it is identical
in both corpus versions, and only `experiments/recalibrate.py` reads it.

**Five new analyses** (Supplemental Tables S15--S19), each of them post hoc
and outside the two pre-specified gates:

| analysis | runner | generator | table |
|:--|:--|:--|:--|
| roster-calibration sensitivity: crews rebuilt at four weekly-hours quantiles and on all recorded years | `experiments/recalibrate.py`, `experiments/run_ecal.py` | `analysis/gen_ecal.py` | S15 |
| corrective-only and single-technician-plausible work, as a decomposition and as matched-contention reruns | `experiments/run_esub.py` | `analysis/gen_esub.py` | S16 |
| secondary-skill efficiency drawn per technician and secondary trade instead of one shared scalar | `experiments/etahet_tech.py`, `experiments/run_e13_etatech.py` | `analysis/gen_etatech.py` | S17 |
| both verdicts re-read at neighboring settings of the three gate thresholds | none (arithmetic on released results) | `analysis/gen_thr.py` | S18 |
| the exact static reference resolved by instance size: model size, proof rate, wall to proof | none (re-reads `results/e1_static`) | `analysis/gen_cpscale.py` | S19 |

Results go to `results/ecal/`, `results/esub/`, `results/e13_etatech/` and
`results/thr_sensitivity/`. Each family carries `results.csv` with every
scored row, the same data as `results.parquet`, the crew tables it was run
on, and the summary, analysis and regression-check JSON. The resumable shard
caches of these four families were pruned and are not shipped, unlike the
v1.0 families, which keep theirs; a rerun rebuilds them.

**Regression checks.** Every new code path reproduces the released numbers
where the settings coincide, and each runner fails on a mismatch:
`run_ecal.py` re-runs the released calibration instead of reusing the
released rows, `run_esub.py` compares its per-instance totals against the
released tier1 rows, and `run_e13_etatech.py --repro-all` requires a constant
per-technician efficiency of 0.8 to reproduce the scalar-efficiency rows
bitwise, both against its own comparator arm and against the released result
files.

## Layout

- `overlays/` -- the flexibility-ladder generator, unit-tested;
  `generate_all.py` materializes every released overlay bit-for-bit from the
  released crew calibration and a recorded seed.
- `env/` -- pair-selection engine (`engine.py`), independent validator
  (`validator2.py`), admissible reward lower bound for overlapping skill
  pools (`lb2.py`), grid conventions (`conventions.py`), and the crew-order
  engine and its independent validator (`crew_engine.py`,
  `validator_crew.py`).
- `methods/` -- priority rules with the technician tie-break and the two
  flexibility-aware rules LFJ-ATC and ATC-&eta; (`rules.py`), CP-SAT exact
  and rolling references (`cpsat2.py`, `rolling2.py`), genetic algorithm
  (`ga2.py`), pair policies (`policy2.py`), PPO trainer (`train2.py`).
- `experiments/` -- runners: `y1_root.py` (input-corpus resolver),
  `anchor_l0.py` (machine-precision regression anchor against the
  single-skill release), `run_static.py`, `run_dynamic.py`, `run_e5.py`
  (sensitivity), `launch_training.sh`, `post_training.sh`, and the v1.1
  runners `recalibrate.py`, `run_ecal.py`, `run_esub.py`,
  `run_ecal_esub.sh`, `etahet_tech.py`, `run_e13_etatech.py`, and the v1.2
  crew-size runner `run_ecrew.py`.
- `analysis/` -- `gates.py` (the pre-specified Gate P / Gate C tests),
  `build_all.py` (every number in the paper), `figures.py` (every exhibit),
  and the per-analysis table generators `gen_ecal.py`, `gen_esub.py`,
  `gen_etatech.py`, `gen_thr.py`, `gen_cpscale.py`, `gen_ecrew.py`,
  `gen_r2.py`, and `ecrew_floor.py`.
- `protocol/Y2_protocol.md` -- the dated protocol log: gates, thresholds,
  seeds, and the reporting plan, committed before any verdict experiment ran.
- `results/` -- aggregated per-method scored result files (`results.csv` per
  experiment family) and the gate outputs (`gates/`).
- `data/y1_frozen_v10/` -- the frozen input corpus archive and its
  provenance note.
- `supplemental/` -- the Supplemental Materials PDF (Texts S1--S8,
  Tables S1--S22, Figs. S1--S6).
- `tests/` -- unit and parity tests (overlay accounting, engine parity,
  validator, reward-bound admissibility).

## Reproduce

```bash
# Python >= 3.10 with torch, ortools, numpy, pandas, scipy, pyarrow
export PYTHONPATH=.:vendor

# 0. unpack the frozen input corpus (see Release v1.1)
cd data/y1_frozen_v10 && tar --zstd -xf instances_v10.tar.zst && cd ../..

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

# 8. v1.1 sensitivity analyses (CPU)
#    roster calibration and work-order subsets, end to end; the crew rebuild
#    reads the raw FMUCD corpus from the sibling repository (see Data)
ECAL_CORES=0-9 bash experiments/run_ecal_esub.sh 10
#    per-technician secondary-skill efficiency
python experiments/run_e13_etatech.py --baseline --workers 6
python experiments/run_e13_etatech.py --repro-all --workers 6
python experiments/run_e13_etatech.py --full --workers 6
python analysis/gen_etatech.py
#    gate thresholds and the size-resolved static reference, from released files
python analysis/gen_thr.py && python analysis/gen_cpscale.py

# 9. v1.2 crew-size analysis and length floor (CPU)
python experiments/run_ecrew.py --phase regress --workers 12
python experiments/run_ecrew.py --phase sweep --workers 12
python analysis/ecrew_floor.py
python analysis/build_all.py
```

`analysis/build_all.py` and the `gen_*.py` modules write their tables and
number keys under `paper/`, which this release does not carry, and create
`paper/sections/` when it is missing.

## Data

Instances, the crew calibration and the generator parameter packs are the
single-skill release of
[FM-Scheduling](https://github.com/NTUZZH/FM-Scheduling) as it stood when
these results were computed, shipped here as the frozen archive described
above. The raw FMUCD dataset is on Mendeley Data (DOI 10.17632/cb8d2nsjss.1,
CC BY-NC 4.0); its SHA-256 is verified before use, and it is the one input
the frozen archive does not carry, so `experiments/recalibrate.py` reads it
from a checkout of the single-skill repository or from `FMWOS_Y1_ROOT`.
Overlays are deterministic functions of the released crew calibration and are
also shipped as files.

## License

CC BY-NC 4.0, inherited from FMUCD. Non-commercial.
