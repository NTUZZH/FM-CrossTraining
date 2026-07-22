# Paper Y2 pre-specified protocol log

Dated: 2026-07-07 (+08). Committed BEFORE any verdict experiment is run.
Author pre-approval: work order v1.1 (2026-07-07) pre-approves Gate P as
amended, Gate C rho >= 0.70 with the dividend guard, the verdict-class rule,
and the locked defaults below (work order 0.6.6); that pre-approval, with
its date, is recorded here in lieu of a fresh sign-off. Any later change is
a dated amendment appended at the end and disclosed in the manuscript.

State of the repository at commit time: L0 regression anchor GREEN
(results/anchor_l0/report.json: 2,289 configs, 13,734 per-instance WWT
checks against Y1's released results, max relative deviation 2.2e-16, all
schedules bitwise; random covered by the exact v1-semantics replay harness
plus distributional agreement, decision D4). No experiment beyond the
anchor and unit tests has been run.

## 1. Scope and splits

- Verdict campuses {5, 9, 10, 12}; held-out campuses {1, 2} appear ONLY in
  E4' (transfer). Replay TEST split only for evaluation; policy training and
  dev sets use replay TRAIN (window_start <= 2017-12-31) and the generator.
- Sizes: dynamic {150, 400}; static {50, 150, 400}.
- Instances are Y1's, byte-identical; overlays are the only new input.

## 2. Gate P (prediction test; primary verdict)

- Scope: replay track, verdict campuses, sizes {150, 400}, contended
  flexible cells = m in {0.6, 0.8} x Lambda in {CHAIN(1.0), FULL} x eta in
  {1.0, 0.8} (8 overlay cells; v1.1 reconciliation: CHAIN(0.5) and GEN live
  in Tier-2 and are excluded from the verdict).
- Rule set to beat (7 ranked): EDD, WSPT, ATC (k=2), pFIFO, MOR, LFJ-ATC,
  ATC-eta, each with the default technician tie-break TB. Random is the
  unranked floor and is not part of the gate.
- Policy pool: the pre-designated verdict class (Section 4), 10 seeds,
  per-instance seed-mean TWT.
- Pass criterion, evaluated in each of THREE scopes (pooled contended
  flexible cells; the m = 0.8 family; the m = 0.6 family):
  (a) the policy's pooled seed-mean TWT is lower than every ranked rule's
      pooled mean TWT in that scope; and
  (b) the paired Wilcoxon signed-rank test (two-sided, per-instance policy
      seed-mean vs rule, zero differences dropped) against EACH of the seven
      ranked rules is significant at p < 0.05 after Holm correction across
      the seven comparisons within that scope.
  ALL THREE scopes must pass (deliberate conjunction, stated as such in the
  manuscript; stricter than any cross-scope correction).
  Robustness condition: in each scope, at least 8 of 10 seeds individually
  beat the best rule (the ranked rule with the lowest pooled mean TWT in
  that scope) on pooled mean TWT.
- Secondary descriptive (not a gate): policy-vs-best-rule gap as a function
  of Lambda (the flexibility gradient), and win/tie/loss tallies with tie
  tolerance epsilon = 1.0 weighted unit.

## 3. Gate C (chaining test)

- Per contended crew-multiplier family (m in {0.6, 0.8}, eta fixed), the
  flexibility dividend Delta(Lambda) = TWT_best(L0) - TWT_best(Lambda),
  where TWT_best(Lambda) is the LOWEST pooled mean TWT over the full-cell
  dynamic methods (7 ranked rules + Random + both policy classes'
  seed-means; rolling CP-SAT excluded: it runs on an 8-id subsample and
  would not be a full-cell estimator). Pooling is over verdict campuses and
  both sizes (plain instance pooling, replay test).
- Capture ratio rho = Delta(CHAIN(1.0)) / Delta(FULL), computed on POOLED
  dividends (pool first, then divide; never a mean of per-cell ratios).
- Chaining-suffices criterion: rho >= 0.70 at eta = 1.0 (pre-approved
  v1.1); the eta = 0.8 value is reported alongside.
- Denominator guard: if pooled Delta(FULL) < 2% of pooled TWT_best(L0) in a
  crew-multiplier family, declare "no material flexibility dividend" for
  that family, report Gate C as not evaluable there, and apply contingency
  R1: Frame U (E3', u >= 1.0 cells) becomes the pre-specified fallback
  arena for the Gate P mechanism claim; optionally add m = 0.5 as a dated
  amendment.
- Also reported (descriptive): the phi-curve (CHAIN(0.25/0.5/1.0)) and the
  CHAIN(1.0)-vs-GEN comparison at matched budget B.

## 4. Verdict-class rule (pre-registered)

Both classes train with identical curriculum and PPO settings:
pair-MLP seeds 301-310, pair-attention seeds 401-410, 1,200 updates each.
The class with the LOWER MEAN PER-SEED DEVELOPMENT MINIMUM of the primary
dev signal becomes the Gate P pool; the other is reported as an ablation.
- Primary dev signal: mean validator WWT of the greedy policy on the fixed
  32-instance cell-stratified replay-TRAIN dev set under overlay
  (m = 0.6, CHAIN(1.0), eta = 0.8), evaluated every 20 updates; the
  per-seed dev minimum is min over evaluations (checkpoint = that minimum,
  saved as best.pt).
- Monitors (reported, never used for selection): replay-default (L0,
  m = 1.0) and (m = 0.6, FULL, eta = 0.8).
The choice is recorded HERE as a dated amendment BEFORE any test-set
evaluation of either class.

## 5. Statistics

Paired two-sided Wilcoxon signed-rank on common instance ids, alpha = 0.05;
zero differences dropped. Holm correction for the Gate P family (the seven
rule comparisons within each scope). Tie tolerance epsilon = 1.0 weighted
unit for descriptive win/tie/loss tallies only. Seed means only; no
best-of-seeds anywhere. Per-family p-value tables released with the
repository. Every table caption states its n.

## 6. Locked defaults (Appendix D v2; work order 2.7)

Carried verbatim from Y1: SLA windows 8/24/80/171.4 bh; weights 8/4/2/1;
crew estimator p95 weekly labour hours / 40, >= 1; MISC threshold 1,000;
CP-SAT budgets 60 s (+300 s hard tail), rolling 2 s / 2 workers / 4 bh
periodic trigger + arrival trigger + flow-time tiebreak; GA population 100,
60 s, OX, swap 0.2, tournament 3, elitism 2, stall 200; PPO lr 3e-4,
gamma 1.0, GAE lambda 0.98, clip 0.2, 4 epochs, minibatch 1024, entropy
0.01, value coef 0.5, grad clip 0.5, 16 parallel envs x 512 steps.

Y2 additions locked NOW, before any verdict run:
- eta grid: main {1.0, 0.8}; sensitivity {0.9, 0.75}. eta = 1.0 means
  p(j,u) = p_j exactly for every eligible technician (decision D6);
  ceil_grid (0.01 bh round-up) engages only for eta < 1.
- phi grid: {0.25, 0.5, 1.0}.
- Chain-order rule: campus trades sorted by DESCENDING p95_weekly_hours
  from the released Y1 capacity table (tie: trade name ascending);
  the chain is the cycle t_1 -> ... -> t_K -> t_1 (decision D2).
- Overlay construction seed: 20260707 (recorded; base ladder is fully
  deterministic). Chain-permutation seeds (E5'): 20260708, 20260709,
  20260710.
- Technician tie-break TB: primary-skill first, then least-flexible
  (smallest |S_u|), then lowest id (string compare, Y1 heap order).
- LFJ-ATC tie tolerance: 1% relative (never tuned). ATC k = 2 untuned.
- Candidate cap: 64 smallest-slack orders per trade (E5' ablation: 256);
  technicians deduplicated by (skill-set signature, next-available time);
  pair-slot tensor cap 256 with deterministic slack-first truncation
  (implementation constant; rules are exempt and scan the full pair set).
- Policy updates: 1,200 per seed (doubled vs Y1's 600). Seeds: pair-MLP
  301-310, pair-attention 401-410. Curriculum (uniform mixing): track ~
  U{replay-train, generator}; m ~ U{0.5, 0.6, 0.8, 1.0}; structure ~
  U{L0, CHAIN(0.5), CHAIN(1.0), FULL}; eta ~ U{1.0, 0.8}; generator
  arrival_multiplier 1.0; instances materialised at m = 1.0 workload with
  the overlay carrying m. Reward: shaped, with the v2 admissible bound and
  the grid-exact conversion constant (decision D5).
- Specialist ablation (not verdict): 3 seeds each (501-503 CHAIN(1.0),
  601-603 FULL), trained at (that structure) x m ~ U{0.5,0.6,0.8,1.0} x
  eta ~ U{1.0,0.8}, other settings identical.
- L0 is eta-invariant: run once per m, reuse rows across eta in analysis.

## 7. Experiment grid (pre-declared; work order 3.1)

- E1' static: verdict campuses x sizes {50,150,400} x tracks {replay-test,
  generator-test}, FIRST 15 instances per (campus, size, track) in sorted
  id order; overlays {L0, CHAIN(1.0), FULL} x eta {1.0, 0.8} at m = 1.0
  (5 effective cells); methods: 7 ranked rules + Random, CP-SAT 60 s
  (300 s re-run on every instance not proved OPTIMAL at 60 s), GA (60 s,
  seed 301), policy greedy rollout (after training; both classes).
- E2' Tier-1 (dynamic): replay test, verdict campuses, sizes {150, 400},
  ALL test instances per cell (<= 100/cell by Y1 construction); overlays
  {L0, CHAIN(1.0), FULL} x eta {1.0, 0.8} x m {1.0, 0.8, 0.6} (15
  effective cells after L0 eta-reuse). Methods per cell: 7 ranked + Random
  (seed 301), policy verdict class 10 seeds + other class 10 seeds,
  rolling CP-SAT on the FIRST 8 instance ids per (campus, size, overlay
  cell). VERDICT CELLS (Gate P) = the m in {0.6, 0.8} x {CHAIN(1.0),
  FULL} x eta subset; they run only after this log is committed.
- E2' Tier-2 (dynamic): overlays {CHAIN(0.25), CHAIN(0.5), GEN} x eta
  {1.0, 0.8} x m {1.0, 0.6}; same replay cells and methods minus rolling.
- E3' Frame U: generator packs, fixed 80 bh window, u in {0.9, 1.0, 1.1,
  1.3}, crew multiplier 1.0, 30 instances per (campus, u); arrival
  multiplier = u / u0 with u0 = base_utilization(pack) (Y1 storm2
  machinery); seeds 90000 + cell_index*1000 + i, cell_index =
  campus_idx * 4 + u_idx (disjoint from every Y1 range); overlays {L0,
  CHAIN(1.0), FULL} x eta {1.0, 0.8}; methods: 7 ranked + Random + both
  policy classes (10 seeds each).
- E4' transfer: campuses 1 and 2, Tier-1 grid and methods (rolling
  excluded); campus-2 framing per work order 1.5/RQ5.
- E5' sensitivity, designated family = replay test, verdict campuses,
  size 150, m = 0.6, CHAIN(1.0) (FULL alongside where the variant needs an
  envelope), eta = 0.8 unless the variant varies eta, FIRST 30 instances
  per campus in sorted id order. Variants: eta in {0.75, 0.9}; SLA x0.5 /
  x1.5; crew x0.75 / x1.25 (composed with m = 0.6); weight vectors
  (4,3,2,1) and (27,9,3,1); 3 chain permutations (seeds above); TB
  ablation (default vs random-idle-eligible vs most-flexible-first) on the
  same family plus (m = 0.8, FULL, eta = 1.0); candidate-cap 256 (policy
  only). Outputs: Kendall tau_b of method rankings vs the locked default,
  Y1 Figure-10 idiom.

## 8. Provenance and fairness

- Provenance tags: [R] replay, [C] calibrated generator, [D] designed
  skill structures (new in v2, defined in manuscript Section 4).
- Fairness protocol: rules and policies run through the SAME pair engine
  and event stream; v1 parity and the L0 anchor are the exactness
  evidence; every reported row is validator2-feasible; latency accounted
  per decision (rules, policies) and per replan (rolling).
- No rule parameter is tuned on test data. Campuses 1/2 never enter
  training, curriculum, or checkpoint selection.

## 9. Branch plan (title/abstract emphasis follows the branch)

- Branch A (Gate P passes): headline "assignment flexibility is where
  learning starts to pay"; design map second act.
- Branch B (policy never beats the ranked set): headline "chained crews
  plus flexibility-aware rules suffice; the refutation of our own
  published prediction reported in full".
- Branch C (policy beats the five classic rules but not LFJ-ATC/ATC-eta):
  headline "the binding lever is rule design, not learning".
All three are publishable; Gate C's design result is co-equal in every
branch. The title follows the resolved branch.

## Amendments

### Amendment A1 (2026-07-10): verdict class recorded
Applying the pre-registered rule of Section 4 to the completed training pools, BEFORE any test-set evaluation of either class: mean per-seed development minimum (primary signal) pair-MLP = 421.0172, pair-attention = 422.3694. The verdict class is **pair-mlp**; the other class is reported as an ablation. Per-seed minima in results/gates/verdict_class.json.


### Amendment A2 (2026-07-21): wait-action policy class (E10), declared before any training run
Motivated by the observation that the registered non-idling protocol
binds the learned class, while a break-even waiting rule improves the
penalised cells (Section 7.5). Declared BEFORE any E10 training or
evaluation:

- Variant: pair-MLP with ONE change, an extra observation token whose
  action declines all pairs at the instant (engine DECLINE path, legal
  only while at least one technician is busy). Token features: is-wait
  flag plus the shared context block; f_pair 38 -> 39; parameter count
  asserted equal to the released class + 128 (the extra encoder column).
  Everything else is the locked stack: PPO defaults, curriculum, dev
  sets, checkpoint rule (per-seed primary-signal minimum), 1,200
  updates, 10 seeds (701-710), greedy evaluation.
- Evaluation: Tier-1 replay verdict cells, seed-mean pooling as
  released; waits counted per episode. Output results/e10_wait/ (never
  merged into tier1).
- Declared comparisons, ALL reported regardless of outcome, none a
  registered gate (this is a post-protocol addition and will be labelled
  as such in the manuscript):
  (i)  vs the released non-idling verdict class (does the wait action
       help the learner?);
  (ii) vs the best plain ranked rule per Gate P scope (the original
       prediction re-read with a wait-capable class);
  (iii) vs the best patient rule on the penalised cells (the
       idling-capable baseline; the fair arena for the wait variant).
- Reading rules, fixed now: "the wait-capable class beats the patient
  rules" requires lower pooled seed-mean in the penalised scopes with
  the same Wilcoxon/Holm machinery as Gate P applied descriptively;
  anything else is reported as "does not close the gap". No checkpoint
  reselection, no post-hoc seed filtering.
