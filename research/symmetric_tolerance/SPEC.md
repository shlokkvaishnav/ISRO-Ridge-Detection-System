# SPEC: Does Arm C's advantage over the classical arms survive when all three are scored at the same 1-2 px tolerance?

Copied verbatim from [issue #14](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/issues/14),
per research/AGENT_PIPELINE.md's Implementer instructions.

**Type:** analysis (re-scoring an existing result: CPU inference of committed checkpoints on the existing 6 benchmark tiles, no training, no new tile extraction)

**Research question**
The headline result (README ESTABLISHED: Arm C 40/57 = 70.2% vs Arm A 23/57 = 40.4% and Arm B 25/57 = 43.9%) is scored with the exact-pixel hit rule: a vertex counts only if the mask covers its own pixel. Arms A and B work on slope and characteristically mark the ridge *flanks*, not the crest (DECISION_LOG 2026-09-18), while Arm C was trained on polyline-buffer labels centred on the catalog line. When **all three arms** are scored at the same 1–2 px tolerance, and each is compared with a chance baseline built at that same tolerance, does Arm C still catch more benchmark vertices than each classical arm?

**Hypothesis**
The gap narrows but survives at 2 px: Arm C still beats both A and B on a paired per-vertex test, and its lift over tolerance-matched chance is at least as large as theirs. Why: at 2 px the A/B shared-miss set shrinks from 21 to 12, but the primary Arm C checkpoint still hits 8 of those 12 at exact pixel (`research/three_way_vertex_map/SPEC.md`), and tolerance can only add hits to Arm C too. Against it: A and B each gain 12 and 10 hits between exact pixel and 2 px (23→35, 25→35), and their fragmented masks may gain more from dilation than Arm C's contiguous one.

**Null / alternative hypothesis**
Null: at 2 px, Arm C's per-vertex hits are no better than each classical arm's. Among vertices where C and a classical arm disagree, each is equally likely to be the one that hits (paired sign test). Under this null the exact-pixel gap was largely the flank-vs-crest metric artefact that DECISION_LOG 2026-09-18 warned the comparison would need to handle.

A second way the hypothesis can fail: C still beats A and B on raw hits at 2 px, but only because its tolerance-dilated mask covers more of each tile. Its lift over tolerance-matched chance is then lower than the classical arms', and the raw win is a coverage effect.

**Motivation**
DECISION_LOG 2026-09-18 recorded that slope-domain Arm A finds ridge flanks, not the crest, and said that any comparison against Arm B/C "needs to define ridge-match tolerance in terms of close to the true ridge structure ... record this in the eventual comparison SPEC rather than silently normalizing the outputs". The Arm C head-to-head (#3) and every analysis since (#5, #7, #10) used exact pixel only. #7 then scored A and B at 1–2 px as a descriptive check and found the classical shared-miss set shrinks by 43% at 2 px. It left this explicit follow-up: "Score Arm C at the same 1-2 px tolerance as A/B, so the mask-shape confound is tested symmetrically." Its interpretation says "the comparison is not yet symmetric". No issue has taken it up.

This is the right next question for three reasons:
- It tests the project's headline ESTABLISHED claim, not a detail of it.
- It has to be settled before more compute goes into the current metric. Extending the benchmark with more tiles (to fix the power problem #10 hit), or running seed replicates, would otherwise scale up a possible metric artefact.
- It is cheap. The A/B tolerance hits are already committed (`A_tol1/A_tol2/B_tol1/B_tol2` in `results/three_way_vertex_map/vertices.json`). Both Arm C checkpoints are committed (`results/arm_c/model.pt`, `results/arm_c_early_stopping/model_best.pt`). `research/three_way_vertex_map/vertex_map.py` already has CPU inference (`arm_c_mask`) and the tolerance rule (`within_tolerance`). The only data that isn't in git is the 6 benchmark tiles (gitignored). #7 used the same ones, and `scripts/build_tile_dataset.py` rebuilds them.

**Experimental design**
- **Data:** the 57 truth vertices on the 6 benchmark tiles (3674, 1851, 748, 5017, 4098, 3461), at the native row/col already recorded in `vertices.json`. Masks:
  - A and B: the committed `A_tolN` / `B_tolN` columns (their masks are regenerated only for the chance baseline below).
  - C (primary): `results/arm_c/model.pt` at threshold 0.5.
  - C2 (secondary, descriptive): `results/arm_c_early_stopping/model_best.pt` at threshold 0.5.
  All run in the repo's Docker image, CPU only.
- **Tolerance rule:** exactly `within_tolerance` in `vertex_map.py`: a hit if any mask pixel lies within Euclidean distance N of the vertex pixel. **Primary N = 2 px. Secondary N = 1 px.** Exact pixel (N = 0) is reported as the reference.
- **Validation gate, before any C-vs-classical joint statistic is read:**
  1. C and C2 at exact pixel reproduce the per-tile hits in `results/arm_c/eval.json` (5/6/2/6/12/9) and `results/arm_c_early_stopping/eval_best.json` (5/8/4/7/11/5).
  2. Regenerated A and B masks reproduce the committed `A`, `B`, `A_tol1`, `A_tol2`, `B_tol1`, `B_tol2` columns vertex for vertex.
  3. Per-tile coverage matches `vertices.json`.
  If any check fails, stop and report the discrepancy.
- **Power check, from marginals only, before reading the joint.** Record each arm's total hits at 1 and 2 px. For each comparison (C vs A, C vs B), the smallest achievable one-sided sign-test p is 0.5^(h_C − h_X), reached if every discordant vertex favours C. Record it, and the number of C-favouring discordants needed to reject at the observed marginals.
  - **Reachability is already guaranteed at 2 px.** Tolerance can only add hits, so h_C ≥ 40, while h_A = h_B = 35 (committed). The gap is therefore ≥ 5, and the smallest achievable p ≤ 0.5^5 = 0.031 < 0.05.
  - **Non-rejection is also reachable.** For example, gap 5 with 7 classical-favouring discordants gives 12 vs 7, p ≈ 0.18.
  - So every pre-registered outcome below is reachable. If the recorded p_min is above 0.05 anyway, report outcome (e).
- **Held constant:** threshold 0.5 for C and C2, default A (shape-filter) and B (frangi) settings, the 6 tiles. Nothing is tuned; no other tolerance is tried for the decision.

**Metrics**
Deciding metrics, primary checkpoint C, at 2 px:
1. **Paired sign test, C vs A and C vs B separately.** Count discordant vertices (C hits and X misses, X hits and C misses), then compute the exact one-sided binomial p (p = 0.5). "C beats both" requires p < 0.05 for **each** comparison. This is an intersection-union test, so no multiplicity correction is needed.
2. **Lift over tolerance-matched chance**, for each arm: observed hits / expected hits. Expected hits = Σ over vertices of the tile's *dilated coverage*, meaning the fraction of that tile's pixels that have at least one mask pixel within N px (the chance that a uniformly random vertex position would score a hit under the same rule). Compare lift(C) with lift(A) and lift(B). This is a magnitude comparison, not a test, and it decides only between outcomes (a) and (b).

Descriptive only, not deciding:
- the same statistics at 1 px and at exact pixel (the exact-pixel lift must reproduce #7's coverage-matched numbers)
- per-tile hits and dilated coverage for every arm at 0/1/2 px
- C2 at the same tolerances, with its own sign tests and lift
- leave-one-tile-out sign-test p-values at 2 px
- for #7's residual sets: at 2 px, how many of the 12 A/B shared misses C and C2 hit, and which vertices every arm and checkpoint misses at 2 px

**Baselines / controls**
- **Chance, pre-registered:** the tolerance-matched random-position baseline in metric 2 (per-tile dilated coverage under the same Euclidean rule). At exact pixel this reduces to #7's coverage-matched chance, which is checked as part of the gate. This matters because a fragmented mask gains far more area from dilation than a contiguous one of equal coverage, so raw recall at tolerance is not comparable across arms without it.
- **Reference:** the exact-pixel comparison (the current headline), computed under the same sign test so the change caused by tolerance is visible on one scale.

**Expected outcomes**
(a) At 2 px, C beats both A and B (each sign-test p < 0.05), and lift(C) ≥ lift(A) and ≥ lift(B). The headline advantage survives symmetric tolerance and is not a coverage effect.
(b) At 2 px, C beats both on the sign test, but lift(C) is below at least one classical arm's lift. C's raw advantage at tolerance is explained by its dilated coverage.
(c) At 2 px, C does not beat at least one classical arm (p ≥ 0.05). The exact-pixel gap to that arm is not distinguishable from the flank-vs-crest metric artefact at this n.
(d) Any of (a)–(c), but the verdict flips across p = 0.05 when one tile is dropped, or 1 px and 2 px give different outcomes. The result then depends on one tile or on the exact tolerance.
(e) The validation gate fails, or the recorded smallest achievable p is above 0.05.

**Interpretation plan**
(a) README: the headline claim gains a qualifier. It holds under symmetric 2 px tolerance with a tolerance-matched chance baseline. #7's caveat "Arm C was not itself scored at tolerance" is resolved. Keep the 6-tile, 2-checkpoint scope.
(b) README: restate the headline as coverage-dependent under tolerance. "Beats the classical arms" holds at exact pixel. At tolerance it holds only in raw hits, not per unit of dilated mask area. Add to DO NOT CLAIM.
(c) README: the headline ESTABLISHED entry is downgraded. The exact-pixel gap to the named arm cannot be separated from the flank-vs-crest metric artefact on this benchmark. State the power from the pre-registered check ("not detected", not "no difference"). Add a DECISION_LOG entry, since this changes the project's main claim. Any later benchmark extension or seed study should then pre-register a tolerance-based metric.
(d) Report per tile and per tolerance. Generalize nothing from a single-tile or single-tolerance effect. README states which of (a)–(c) holds at which tolerance.
(e) Report the gate discrepancy or the lack of power as the finding, with a DECISION_LOG entry. The headline is unchanged, and the asymmetry caveat stays open.

For every outcome, report C2 next to C without letting it decide, as #7 did.

**Confounds considered**
- **Non-independence along the ridge.** The sign test treats vertices as independent. Masks are contiguous, so adjacent vertices co-vary, and the p-values are optimistic. Leave-one-tile-out (outcome d) guards against single-tile effects; it does not correct for this.
- **Tolerance choice.** 2 px (200 m at 100 m/px) is chosen here, before C is scored at tolerance, to match #7's largest descriptive tolerance and the flank offset seen on synthetic ridges. It is not tuned. 1 px is reported alongside, and a disagreement between them triggers (d).
- **Dilation helps fragmented masks more.** Raw recall at tolerance rewards scattered masks. That is why the tolerance-matched chance baseline (metric 2) decides between (a) and (b), and why raw hits alone never decide.
- **Arm C's supervision.** Arm C's labels are centred on the catalog line, so tolerance may add fewer hits to C than to A/B. That is the effect being measured, not a confound, but it means (c) would say something about the metric, not about which method "sees" more ridge.
- **Checkpoint specificity.** #10 found the two Arm C checkpoints' per-vertex hits are not detectably shared beyond chance. The result therefore describes checkpoint C. C2 is reported but does not decide.
- **Selection on the benchmark.** C2's epoch was selected by val loss on these 6 tiles (#5). C (epoch 60, run 1) was not selected that way, which is one reason C is primary.
- **Catalog or registration error.** Tolerance also absorbs small catalog misplacements, for every arm equally in distance terms. It cannot separate "misregistered vertex" from "flank-offset detection", and the report should not claim to.
- **Small n.** 57 vertices in 6 tiles. A non-rejection is "not detected at this n", stated with the power recorded before the joint was read.

**Compute:** CPU only, in the existing Docker image. Inference of two committed checkpoints plus Arms A/B on the 6 existing benchmark tiles, re-extracted with `scripts/build_tile_dataset.py` if they are not present locally. **No Kaggle GPU training, no new tiles.**


## Results

Script: `research/symmetric_tolerance/tolerance_rescore.py`. Raw per-vertex output (A, B, C
and C2 at 0/1/2 px, row/col, and per-tile dilated coverage with its Monte Carlo check) is in
`results/symmetric_tolerance/vertices.json`. The gate and power check are in
`results/symmetric_tolerance/gate.json`. Every statistic below is in
`results/symmetric_tolerance/analysis.json`, which `--analyze` recomputes from
`vertices.json` alone. Everything ran in the repo's Docker image, CPU only.

**Implementation notes**
- **Shared code.** The masks and the tolerance rule come from `vertex_map.py` (#7), imported
  rather than copied: `arm_c_mask`, `within_tolerance`, the A/B calls and the constants.
- **Dilated coverage.** This is the mean of `binary_dilation(mask, disk)`. The disk is the
  same offset set `within_tolerance` scans, and pixels outside the tile count as not-mask.
  The script checks that `dilated[r, c] == within_tolerance(mask, r, c, N)` at every vertex,
  arm and tolerance, and all cells agree.
- **Chance-baseline check.** As an independent test, 20,000 uniformly random positions per
  tile were scored with `within_tolerance` itself. Across the 72 tile x arm x tolerance cells,
  the Monte Carlo rate matches the analytic dilated coverage to max |z| = 2.39 (max absolute
  difference 0.008). That is about what 72 draws of noise give.
- **Sign tests.** Each is an exact one-sided binomial, P(Bin(n_C + n_X, 0.5) >= n_C),
  cross-checked by 200,000 Monte Carlo draws. The two agree to about 1e-3.

### Validation gate: passed (`gate.json`, committed in `f97aa11` before `analysis.json`)

| Check | Reference | Observed |
|---|---|---|
| 1. C per-tile hits, exact pixel | `arm_c/eval.json`: 5/6/2/6/12/9 | 5/6/2/6/12/9 |
| 1. C2 per-tile hits, exact pixel | `eval_best.json`: 5/8/4/7/11/5 | 5/8/4/7/11/5 |
| 2. A, B, A_tol1, A_tol2, B_tol1, B_tol2 vs #7's `vertices.json` | vertex for vertex | 0 mismatches (C, C2: also 0) |
| 2. A / B per tile (DECISION_LOG) | A 2/1/3/4/6/7, B 3/3/2/2/5/10 | identical |
| 2. #7's tolerance figures | A 23/27/35, B 25/30/35, A&B miss 21/16/12 at 0/1/2 px | identical |
| 3. Per-tile coverage vs #7's `vertices.json`, all four arms | | max abs diff 0.0 |
| Exact-pixel chance = #6/#7 coverage-matched expectation | C 17.72 expected, lift 2.26; C2 20.42, 1.96 | C 17.717, 2.258; C2 20.421, 1.959 |

Tile order everywhere: 3674/1851/748/5017/4098/3461.

### Power check from marginals only (in `gate.json`, recorded before the joint was read)

| Tolerance | h_C | h_A | h_B | gap C-A / C-B | p_min | Rejects iff classical-favouring discordants <= |
|---|---|---|---|---|---|---|
| **2 px (deciding)** | 43 | 35 | 35 | 8 / 8 | 0.0039 / 0.0039 | 5 / 5 |
| 1 px | 41 | 27 | 30 | 14 / 11 | 6.1e-5 / 4.9e-4 | 21 / 13 |

Every pre-registered outcome was reachable (p_min < 0.05 at 2 px), so (e) does not apply.
As in PR #9, the committed gate file set already includes the joint per-vertex table
(`vertices.json`). `--analyze` was run only after `f97aa11` was committed. Git can show the
commit order but cannot prove the run order.

### Deciding metrics: primary checkpoint C at 2 px

| Comparison | both | C only | X only | neither | exact one-sided p | MC p |
|---|---|---|---|---|---|---|
| C vs A | 28 | 15 | 7 | 7 | **0.0669** | 0.0676 |
| C vs B | 28 | 15 | 7 | 7 | **0.0669** | 0.0668 |

Neither comparison reaches p < 0.05. At these marginals each could reject with at most 5
classical-favouring discordants, and each had 7.

| Arm | Hits at 2 px | Expected (tolerance-matched chance) | Lift |
|---|---|---|---|
| A | 35 | 28.94 | 1.21 |
| B | 35 | 30.10 | 1.16 |
| **C** | 43 | 23.61 | **1.82** |
| C2 (descriptive) | 44 | 24.33 | 1.81 |

Lift(C) is above both classical lifts. Under the issue's rule, though, lift only separates (a)
from (b), and both of those require the sign tests to reject first. It does not decide here.

**Base outcome at 2 px: (c).**

### Pre-registered secondary: 1 px

| Comparison | both | C only | X only | neither | p |
|---|---|---|---|---|---|
| C vs A | 21 | 20 | 6 | 10 | 0.0047 |
| C vs B | 24 | 17 | 6 | 10 | 0.0173 |

Lifts at 1 px are A 27/23.98 = 1.13, B 30/25.06 = 1.20 and C 41/20.82 = 1.97 (C2: 41/22.47 =
1.82). Both sign tests reject and lift(C) is the largest, so **the 1 px outcome is (a)**. The
1 px and 2 px outcomes disagree.

### Reference: exact pixel (the current headline, on the same sign-test scale)

| Comparison | both | C only | X only | neither | p |
|---|---|---|---|---|---|
| C vs A | 17 | 23 | 6 | 11 | 0.0012 |
| C vs B | 19 | 21 | 6 | 11 | 0.0030 |

Lifts at 0 px are A 23/18.99 = 1.21, B 25/19.98 = 1.25, C 40/17.72 = 2.26 and C2 40/20.42 =
1.96. The outcome rule gives (a) at 0 px.

### Descriptive: what tolerance does to each arm

| Arm | Hits 0 / 1 / 2 px | Expected 0 / 1 / 2 px | Lift 0 / 1 / 2 px |
|---|---|---|---|
| A | 23 / 27 / 35 | 18.99 / 23.98 / 28.94 | 1.21 / 1.13 / 1.21 |
| B | 25 / 30 / 35 | 19.98 / 25.06 / 30.10 | 1.25 / 1.20 / 1.16 |
| C | 40 / 41 / 43 | 17.72 / 20.82 / 23.61 | 2.26 / 1.97 / 1.82 |
| C2 | 40 / 41 / 44 | 20.42 / 22.47 / 24.33 | 1.96 / 1.82 / 1.81 |

- **Hits.** From 0 to 2 px, A gains 12 hits and B gains 10. C gains 3 and C2 gains 4.
- **Chance.** Each classical arm's expected hits grow by about 10, and C's by 5.9. As the issue
  anticipated, the fragmented classical masks gain more area from dilation.
- **Lift.** The classical arms stay flat at about 1.2. C falls from 2.26 to 1.82 but stays well
  above both.
- **Against own chance (descriptive only).** At 2 px, the Poisson-binomial p of each arm's hits
  against its own tolerance-matched chance is 5.1e-8 for C, 6.4e-9 for C2, 0.069 for A and
  0.12 for B.

### Outcome (d) checks

**Leave one tile out, primary C at 2 px** (pre-registered descriptive; feeds (d)):

| Dropped | C vs A (C only / A only, p) | C vs B (C only / B only, p) | Lifts A / B / C | Rule |
|---|---|---|---|---|
| 3674 | 14 / 7, 0.095 | 15 / 7, 0.067 | 1.17 / 1.10 / 1.88 | (c) |
| 1851 | 10 / 7, 0.31 | 10 / 6, 0.23 | 1.30 / 1.26 / 1.84 | (c) |
| **748** | 15 / 4, **0.0096** | 14 / 5, **0.032** | 1.22 / 1.24 / 1.89 | **(a)** |
| 5017 | 12 / 6, 0.12 | 11 / 7, 0.24 | 1.23 / 1.23 / 1.93 | (c) |
| 4098 | 12 / 7, 0.18 | 11 / 6, 0.17 | 1.19 / 1.18 / 1.53 | (c) |
| **3461** | 12 / 4, **0.038** | 14 / 4, **0.015** | 1.13 / 0.96 / 1.86 | **(a)** |

- Dropping 748 or 3461 flips both comparisons across p = 0.05.
- Dropping 1851, 5017 or 4098 moves p further from rejection.
- Lift(C) stays above both classical lifts in every drop.

**Outcome: (d), base (c).** Both (d) triggers fire:
- the verdict flips across p = 0.05 when one tile (748 or 3461) is dropped;
- 1 px gives (a) and 2 px gives (c).

### Descriptive: per-tile hits at 0 / 1 / 2 px (dilated coverage in brackets)

| Tile | n | A | B | C | C2 |
|---|---|---|---|---|---|
| 3674 | 5 | 2/2/4 (.32/.41/.50) | 3/3/5 (.38/.47/.55) | 5/5/5 (.54/.61/.67) | 5/5/5 (.62/.66/.70) |
| 1851 | 8 | 1/1/2 (.30/.37/.45) | 3/3/3 (.39/.49/.58) | 6/6/7 (.36/.43/.50) | 8/8/8 (.57/.61/.65) |
| 748 | 8 | 3/5/5 (.36/.45/.54) | 2/2/3 (.36/.44/.52) | 2/2/2 (.15/.20/.25) | 4/5/7 (.35/.41/.47) |
| 5017 | 7 | 4/4/4 (.35/.45/.54) | 2/2/2 (.31/.39/.48) | 6/6/6 (.48/.56/.63) | 7/7/7 (.64/.67/.70) |
| 4098 | 15 | 6/7/10 (.36/.44/.53) | 5/9/10 (.40/.50/.60) | 12/13/13 (.20/.23/.26) | 11/11/12 (.25/.28/.31) |
| 3461 | 14 | 7/8/10 (.31/.40/.48) | 10/11/12 (.28/.36/.44) | 9/9/10 (.32/.37/.42) | 5/5/5 (.13/.15/.17) |

- **748** is C's weakest tile: 2 of 8 at every tolerance, at 15-25% dilated coverage. It is
  also where A gains the most.
- **3461:** B (12) beats C (10) at 2 px.
- These are the two tiles whose removal makes C win.

### Descriptive: C2 (secondary checkpoint)

| Tolerance | C2 vs A (C2 only / A only, p) | C2 vs B (C2 only / B only, p) | Rule |
|---|---|---|---|
| 0 px | 25 / 8, 0.0023 | 21 / 6, 0.0030 | (a) |
| 1 px | 22 / 8, 0.0081 | 21 / 10, 0.035 | (a) |
| 2 px | 17 / 8, 0.054 | 19 / 10, 0.068 | (c) |

C2 follows C: (a) at 0 and 1 px, not detected at 2 px. Leaving one tile out at 2 px, C2
rejects both comparisons only when 3461 is dropped (p = 0.0013 and 0.0004).

### Descriptive: #7's residual sets at 2 px

- **Classical shared misses.** A and B both miss 12 vertices at 2 px. C hits 8 of them at
  exact pixel (as #7 recorded) and 9 at 2 px. C2 hits 9 at exact pixel and 10 at 2 px.
- **Missed by everything.** Only one vertex is missed by every arm and both checkpoints at
  2 px: **748 v2** (row 51, col 35). At exact pixel, #7 listed four (748 v2/v5/v7, 3461 v0).

## Interpretation

**Outcome: (d), with base outcome (c) at the primary 2 px tolerance.**
- At the pre-registered 2 px, the primary checkpoint does not beat either classical arm on the
  paired sign test. Both comparisons give 15 vs 7 discordant vertices, p = 0.067.
- At 1 px it beats both (p = 0.0047 and 0.017) with the larger lift, which is outcome (a).
- At 2 px, dropping tile 748 or tile 3461 flips the verdict to "beats both".
- The result therefore depends on the exact tolerance and on single tiles. Per the
  pre-registered rule, it is reported per tolerance and per tile, and nothing is generalized
  from it.

**Against the stated hypothesis.** The hypothesis was that the gap narrows but survives on the
paired test at 2 px, and that lift(C) is at least as large as the classical lifts.
- **The gap narrows.** For C vs A, the discordant split goes from 23 vs 6 at exact pixel to
  15 vs 7 at 2 px, because A gains 12 hits and C gains 3.
- **"Survives on the paired test" is not supported** at the primary tolerance (p = 0.067). This
  means "not detected at this n", not "no difference". The power check, recorded before the
  joint was read, allowed at most 5 classical-favouring discordants, and there were 7.
- **The lift half holds.** Lift(C) is 1.82 against 1.21 and 1.16, and C stays ahead in every
  leave-one-tile-out drop. The issue's second failure route, a raw win that is only a coverage
  effect, is not what happened. C's per-area advantage survives tolerance better than its
  raw-hit advantage does.
- **The null's reading is not shown either.** The null says the exact-pixel gap was largely the
  flank-vs-crest artefact. That is not established: at 2 px, C-favouring discordants still
  outnumber classical-favouring ones by about 2 to 1. The test cannot separate that from
  chance with 57 vertices and this few discordants.

**What this establishes (6 tiles, 57 vertices, primary checkpoint, pre-registered rule)**
- **At 1 px, the headline holds.** C beats A and B under a symmetric rule with a
  tolerance-matched chance baseline.
- **At 2 px it is not detected** on the paired test, and detection depends on whether tiles
  748 and 3461 are included.
- **Most of the exact-pixel gap depends on the hit tolerance.** A and B gain 12 and 10 hits
  within 2 px, and C gains 3. This fits the flank-detection account in DECISION_LOG
  2026-09-18. This analysis cannot separate flank offset from catalog misregistration.
- **Per unit of dilated mask area,** C's lift stays about 1.5x the classical arms' at every
  tolerance and in every drop. This is a magnitude comparison, not a test.

**What it does not establish**
- **That C and the classical arms are equivalent at 2 px.** A non-rejection on 22 discordant
  vertices is not evidence of no difference.
- **That the lift ordering is significant.** It was pre-registered as a magnitude comparison
  that separates (a) from (b), not as a test.
- **Anything about tolerances other than 0/1/2 px.** No other tolerance was tried.
- **Checkpoint generality.** C2 shows the same per-tolerance pattern: (a) at 0 and 1 px, (c) at
  2 px. But by #10 the two checkpoints' per-vertex hits are not detectably shared, so this is
  two observations, not a property of Arm C.
- **Independence.** The sign test treats vertices as independent. Adjacent vertices co-vary
  along the ridge, so every p-value here is optimistic, including the rejections at 1 px.

**Per interpretation plan (d)**, which says to state which of (a)-(c) holds at each tolerance.
(c) holds at the primary tolerance, so (c)'s plan items are applied as well:
- **README:** the ESTABLISHED Arm C entry keeps its exact-pixel numbers and now states the
  outcome at each tolerance: (a) at 1 px, and (c) at 2 px (p = 0.067), which flips when 748 or
  3461 is dropped.
  - #7's caveat "Arm C was not itself scored at tolerance" is resolved and replaced with this
    result.
  - DO NOT CLAIM gains a line: do not claim Arm C beats the classical arms independently of
    the hit tolerance.
- **DECISION_LOG:** gets an entry, because the main claim's wording changes and (c)'s plan asks
  for one when the primary tolerance does not reject. Per (c)'s plan, any later benchmark
  extension or seed study should pre-register a tolerance-based metric.
- Nothing is generalized from the single-tile flips or from the 1 px result alone.

## Decision

**Implementer's self-assessment: MERGE.** This is an analysis result, outcome (d) with base
(c): a qualification of the headline, not a new method.
- **Gate.** Every item reproduced exactly: both checkpoints per tile, A/B vertex for vertex at
  0/1/2 px, #7's tolerance figures, coverage (max diff 0.0), and the #6/#7 chance
  expectations. `gate.json`, including the marginals-only power check, was committed
  (`f97aa11`) before `analysis.json`.
- **Statistics.** The statistics are exact and cross-checked by Monte Carlo, both the sign
  tests and the chance baseline (random positions scored with `within_tolerance`).
- **Labelling.** The outcome is labelled against the issue's (a)-(e) definitions. The part of
  the hypothesis the data contradict (the paired test at 2 px) and the part they support (the
  lift ordering) are both stated.
- **Scope.** No shared `src/` code was changed, and no other tolerance, threshold or tile set
  was tried.

**Natural follow-ups** (not done here; each would be a new issue):
- **More tiles.** The deciding comparison is limited by n: 22 discordant vertices at 2 px. A
  benchmark extension (which would also fix the power problem #10 hit) should pre-register a
  tolerance-based metric and this tolerance-matched chance baseline.
- **Tile 748.** C (at 15-25% dilated coverage) misses 6 of 8 vertices on 748 at every
  tolerance, while C2 reaches 7 of 8 at 2 px. Why? 748 v2 is the only vertex every arm misses
  at 2 px.
- **Distance profile.** A per-arm profile of distance to the catalog line (mask offset vs
  crest) would separate the flank-offset account from misregistration better than a hit
  tolerance can.
