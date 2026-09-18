# Frozen Y1 input corpus, version 1.0

This directory holds the Y1 benchmark inputs as they stood when the results in
`results/` were computed (2026-07-18 onwards). The sibling repository rebuilt
its corpus on 2026-08-19/21 under two construction corrections, so its current
files no longer reproduce those results. Every runner and analysis module in
this project resolves its Y1 root through `experiments/y1_root.py` and reads
this directory unless `FMWOS_Y1_ROOT` says otherwise.

## Contents

| path | what it is |
|:--|:--|
| `data/processed/instances/` | 3,186 replay instances and 1,800 generator instances, with `index.csv` |
| `results/p1_calib/capacity.csv` | per-(campus, trade) crew sizes, the input to every overlay |
| `results/p1_calib/priority_mapping.csv` | the priority-class mapping behind the instances' `priority` field |
| `results/p2_generator/params_c*.json` | the six per-campus generator parameter packs |
| `results/p4_dyneval/results.csv` | the Y1 dynamic evaluation the L0 anchor checks against |
| `instances_v10.tar.zst` | all of the above in one archive, for the public release |

Instance counts by cell: replay test 600/600/501 and replay train 517/506/462
at sizes 50/150/400; generator test 600 at each size. 4,986 instance files,
149,339,972 bytes uncompressed.

The raw corpus `data/raw/FMUCD.csv` is deliberately absent. It is identical in
both corpus versions (SHA-256
`4464648252c4bdca2a6deba9d467e94aec7568d675f51e06d6d343b3c09f006a`, verified
2026-09-19), so `experiments/recalibrate.py` reads the 1.4 GB file from the
sibling repository instead of duplicating it here.

## How it was rebuilt

Source repository: `PaperY-FMScheduling`, commit
`04e43b83555ea91f8aa56221a269b495871fda8b`, checked out into a detached git
worktree so the repository's own working tree was never written to. Its two
builders carry an explicit corpus switch (added in Y1 commit `6121df31`), and
`--corpus v10` selects the first release's behaviour: the legacy R7
dominant-line sort and the priority mapping fitted on all years.

```
python scripts/p1_instances.py --corpus v10     # calibration + replay instances
python scripts/p2_generator.py --corpus v10     # parameter packs + generator instances
```

`results/p4_dyneval/results.csv` was not rebuilt. It is the Y1 evaluation file
as released, copied from the sibling repository's own v1.0 snapshot
(`results/_v10_archive/p4_dyneval/results.csv`, identical to the file at Y1
commit `a5721f16^`).

### Environment

Python 3.11.15, numpy 1.26.4, pandas 3.0.3, the versions pinned in the Y1
`environment.yml`. The environment is load-bearing, not incidental: the
v1.0 R7 rule breaks a tie between labor lines of equal hours through numpy's unstable
quicksort, so the winning line depends on the numpy version. Rebuilding under
numpy 2.4.6 produced a third corpus that matched neither v1.0 nor v1.1, with
three crew p95 values and roughly 60 trade assignments off.

## Verification

Each check below compares the rebuild with an artifact that was created before
the Y1 rebuild and never edited since.

**Calibration.** `results/p1_calib/capacity.csv` and `priority_mapping.csv` are
byte-identical to the tables at Y1 commit `eefa01eb`, the last commit that
wrote them before the rebuild.

**Generator parameter packs.** All six `params_c*.json` are byte-identical to
the packs at Y1 commit `a5721f16^`.

**Pipeline control.** The same worktree and environment, run with `--corpus
v11`, reproduce the sibling repository's current corpus byte for byte: 3,186 of
3,186 replay instances and `capacity.csv`. Only the corpus switch separates the
two rebuilds.

**Released results.** Re-running the eight dispatching rules through the
project's own runners on this root reproduces every released rule row, on all
four recorded metrics (weighted tardiness, makespan, mean flow, breach share):

| family | rows compared | mismatches | mismatches on the sibling's current corpus |
|:--|---:|---:|---:|
| tier1 | 91,560 | 0 | 2,054 |
| tier2 | 73,248 | 0 | not measured |
| e4 | 40,560 | 0 | not measured |
| e1_static (rules) | 14,400 | 0 | 3,624 |

**L0 anchor.** `experiments/anchor_l0.py` on this root is green with 2,289
configurations, 0 failures, and 13,734 of 13,734 per-instance weighted
tardiness values matching the released Y1 numbers to 2.2e-16 relative. Every
field of its report equals the released `results/anchor_l0/report.json`, and
its per-instance table matches row for row.

**Stored overlays.** `overlays/generated` rebuilds byte-identically from this
root, 648 of 648 files. From the sibling's current calibration, 108 campus-10
overlays differ, because three p95 values moved and re-ordered that campus's
trade chain.

**Generator-driven instances.** All 480 instances in `data/e3` regenerate
exactly from the parameter packs here; 240 of them do not from the sibling's
current packs.

## Checksums (MD5)

```
bf1c78181ed3adb2d237856750770ead  results/p1_calib/capacity.csv
aa9004e29a15e46c6953b5ca5b51a95d  results/p1_calib/priority_mapping.csv
e72ce9f1bccc30d4c033c22cab610ef5  results/p4_dyneval/results.csv
0645ef0696867d3584665481c762dee3  instances_v10.tar.zst
```

`instances_v10.tar.zst` is 13,645,612 bytes and extracts to the four directories
listed above with every file byte-identical.

## What separates this corpus from the sibling's current one

The two corrections the sibling adopted are recorded in its own report
`results/r4_revision/corpus_diff.md`: the priority mapping is refit on rows up
to 2017-12-31 and applied unchanged afterwards, and the R7 dominant-line
selection uses a stable sort. Measured file by file on the 3,186 replay
instances, 3,008 are identical, 134 carry a changed work-order field, 40 differ
only in the order of their work-order list, and 4 differ in which work orders
fall inside the sampling window. The sampling itself is unchanged: `index.csv`
is identical in every column, so the two versions describe the same 3,186
windows.
