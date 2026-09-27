# SPEC: Do the two Arm C checkpoints catch the same benchmark vertices, beyond within-tile chance?

Copied verbatim from [issue #10](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/issues/10),
per research/AGENT_PIPELINE.md's Implementer instructions.

**Type:** analysis (analysis of an existing result, no new data collection)

**Research question**
The two Arm C checkpoints scored on the benchmark come from different training runs of identical code: `results/arm_c/model.pt` (run 1, epoch 60) and `results/arm_c_early_stopping/model_best.pt` (run 2, epoch 7). Each gets 40/57, but their per-tile hits differ (e.g. 3461: 9 vs 5, 748: 2 vs 4). Within each tile, and given how many vertices each checkpoint catches there, do the two checkpoints catch the *same* truth vertices more often than two independent selections would? In short: is Arm C's per-vertex attribution a property of the method, or of the training run?

**Hypothesis**
The two checkpoints agree above within-tile chance, but only partially. Why: they share architecture, inputs, weak labels and training tiles, so vertices with strong ridge signal should be caught by both. But the same code has also produced a near-chance epoch-60 checkpoint (21.1%, `research/arm_c_early_stopping/SPEC.md`), and per-tile hits already differ, so run-to-run variation in *which* vertices are caught is expected too.

**Null / alternative hypothesis**
Null: within each tile, the two checkpoints' hit sets behave like independent random subsets of that tile's vertices, with each checkpoint's observed per-tile hit count held fixed. Their overlap is then what the per-tile hit counts alone predict, and "which vertices Arm C catches" is run-specific on this evidence.

The opposite would also contradict the hypothesis: the two hit sets are nearly identical (overlap close to the maximum the per-tile counts allow). Run-to-run variation would then change only *how many* vertices are caught, not which.

**Motivation**
`research/RELATED_WORK.md`'s novelty verdict names per-instance attribution across method families as the project's load-bearing, unclaimed contribution. PR #9 (issue #7) computed the first three-way per-vertex map and named specific vertices as "missed by all three arms". The PR and its review both list checkpoint instability as a remaining confound: the map describes two checkpoints, not "Arm C" in general. README's OPEN tier already lists Arm C's seed stability, but only for aggregate recall. Aggregate recall can match across runs (40/57 and 40/57) while the per-vertex hits differ. If they differ beyond chance-level overlap, the Arm C column of the per-vertex map can't be attributed to the method family, and neither can the residual-set claims built on it.

This needs no training, no tile extraction and no model inference. Everything is in committed data: the `C` and `C2` columns of `results/three_way_vertex_map/vertices.json` (from PR #9, approved and pending merge; branch from `master` after it merges), plus the per-tile marginals in `results/arm_c/eval.json` and `results/arm_c_early_stopping/eval_best.json`. The aggregate-recall version of the seed-stability question would need new Kaggle runs. This per-vertex version doesn't.

**Experimental design**
- **Data:** the 57 rows of `results/three_way_vertex_map/vertices.json`, columns `tile`, `vertex`, `C`, `C2` (and `A`, `B` for the reference contrast). No masks, tiles or checkpoints are loaded.
- **Validation gate, before any joint C×C2 statistic is read:** the per-tile sums of `C` and `C2` must equal the per-tile hits in `results/arm_c/eval.json` and `results/arm_c_early_stopping/eval_best.json`. Those are 5/6/2/6/12/9 and 5/8/4/7/11/5 across tiles 3674/1851/748/5017/4098/3461. If they don't match, stop and report the discrepancy.
- **Informative tiles:** where either checkpoint catches all or none of a tile's vertices, the overlap is fixed and carries no information. From the committed marginals alone, that rules out 3674 (C 5/5), 1851 (C2 8/8) and 5017 (C2 7/7). The test is therefore decided by 748, 4098 and 3461 (37 vertices). The uninformative tiles are reported, not dropped silently.
- **Power check, from the marginals only, before reading the joint:** compute the smallest achievable one-sided p-value, i.e. the p-value if the overlap on every informative tile reached its maximum, min(a, b). Record it in the SPEC before computing the observed overlap. If it is above 0.05, the test cannot reject by construction: report outcome (e) and stop.
- **Held constant:** the committed hit table. Nothing is recomputed, tuned or re-thresholded.

**Metrics**
Deciding metrics, C (primary checkpoint) vs C2 (secondary checkpoint):
1. **Total overlap S** (vertices hit by both checkpoints) across informative tiles, compared with its within-tile chance distribution. Per tile, the overlap of two independent random subsets of sizes a_t and b_t drawn from n_t vertices is hypergeometric. The total's distribution is the convolution across tiles. Report E[S], the observed S and the exact one-sided p = P(S ≥ S_obs).
2. **Normalized agreement index** (S_obs − E[S]) / (S_max − E[S]), where S_max = Σ min(a_t, b_t) over informative tiles. 0 means chance-level agreement. 1 means the maximum overlap the per-tile counts allow.

Descriptive only, not deciding:
- the 2×2 C×C2 table, per tile and pooled
- the list of vertices where the two checkpoints disagree, by tile
- leave-one-informative-tile-out p-values and index values
- the A/B reference contrast below

**Baselines / controls**
- **Chance, pre-registered:** the within-tile stratified hypergeometric null above, conditional on each checkpoint's per-tile hit count. A coverage-matched random-mask null is deliberately *not* used. Both checkpoints catch vertices at about twice their coverage-matched rate (PR #9), so any two good detectors would "agree" beyond random-mask chance. The question is only which vertices each one picks, given how many it catches on that tile.
- **Adjacency null (secondary, pre-registered):** vertices along one polyline are ordered, and contiguous masks catch runs of adjacent vertices. As a second null, circularly shift C2's hit sequence along the polyline vertex order within each informative tile, independently per tile. Enumerate all shift combinations exactly, or use at least 20,000 random draws. This keeps each checkpoint's run structure. Report its p-value alongside the hypergeometric one.
- **Reference contrast (descriptive, not deciding):** the same statistic and index for Arm A vs Arm B on the same 57 vertices, over the tiles informative for that pair. This calibrates what agreement between two *different* method families looks like under the same null. README's claim that A and B are "genuinely complementary" was made from raw counts (12 both-hit of 57), not against a chance baseline.

**Expected outcomes**
(a) S is above chance (p < 0.05) and the index is ≥ 0.5. The two runs largely agree on which vertices they catch.
(b) S is above chance (p < 0.05), but the index is < 0.5. Agreement is real but partial (the stated hypothesis).
(c) S cannot be told apart from within-tile chance (p ≥ 0.05). Which vertices Arm C catches is not detectably shared across runs at this n.
(d) Any of (a)–(c), but the result flips across p = 0.05 when one informative tile is dropped, or the hypergeometric and adjacency nulls fall on opposite sides of p = 0.05. The effect is then a single-tile or run-structure effect.
(e) The validation gate fails, or the smallest achievable p-value is above 0.05, so the test has no power by construction.

**Interpretation plan**
(a) README: on this benchmark, the Arm C column of the three-way map is largely shared between two independent runs. PR #9's per-vertex attribution can be read as a property of Arm C for those vertices. Qualify it: n = 2 runs, 3 informative tiles, and the runs also differ in epoch.
(b) README: report the three-way map's Arm C column per checkpoint, and name the vertices where the checkpoints disagree. Claims about specific vertices ("missed by all three arms", "caught only by C") hold only where both checkpoints agree. For example, PR #9 lists 4 vertices missed by both checkpoints and both classical arms.
(c) README: Arm C's per-vertex attribution is run-specific on this evidence. The three-way map's C column describes checkpoints, not the method family, and the per-vertex residual-set claims are restated as checkpoint-specific. This is "not detected", not "independent": state the power from the pre-registered check. Extend the existing OPEN seed-stability item: the per-vertex question needs more runs, not only aggregate recall.
(d) Report per tile and per null. Generalize nothing from a single-tile or run-structure effect.
(e) Report the gate discrepancy or the lack of power as the finding, and add a DECISION_LOG entry. Make no per-vertex stability claim.

For every outcome, report the A/B reference contrast next to the C/C2 result, without letting it decide the outcome.

**Confounds considered**
- **Run vs epoch.** C is run 1 at epoch 60, and C2 is run 2 at epoch 7. Disagreement can come from the different run, the different epoch, or both, and this analysis cannot separate them. A within-run pair (run 2 epoch 7 vs run 2 epoch 60, `results/arm_c_early_stopping/model.pt`) would isolate epoch. But that checkpoint scores 12/57, about chance, and is not in `vertices.json`, so it is excluded here, as in #7.
- **Different coverage.** C2's mask coverage is higher than C's on every tile (`vertices.json`, `coverage`). Conditioning on each checkpoint's per-tile hit count absorbs the difference in *how many* vertices are caught. It does not absorb differences in *where* the extra mask area falls.
- **Small n, few informative tiles.** Only 3 tiles (37 vertices) carry information, and tile 4098 has driven results before (PR #9's leave-one-tile-out). That is why leave-one-informative-tile-out is reported.
- **Non-independence along the ridge.** The hypergeometric null treats a tile's vertices as exchangeable. Contiguous masks make adjacent vertices co-vary, so its p-value is likely optimistic. That is why the adjacency null is pre-registered alongside it.
- **Selection on the benchmark.** C2's epoch was selected by val loss on these same 6 tiles (PR #6). That could make C2 more similar to whatever fits these tiles, and so to C, than an unselected checkpoint would be.
- **Exact-pixel metric.** Both checkpoints are scored with the exact-pixel hit rule, so a vertex one pixel outside one checkpoint's mask counts as disagreement. No tolerance is applied, because `vertices.json` records C and C2 at exact pixel only.



## Results

All numbers come from `research/arm_c_checkpoint_agreement/agreement.py`, which reads only
committed JSON (`results/three_way_vertex_map/vertices.json` and the two eval files). Raw
outputs: `results/arm_c_checkpoint_agreement/gate.json` and `analysis.json`.

### Validation gate and power check: passed (`gate.json`)

`gate.json` and this power section were committed (`c21b06a`) before `analysis.json`
(`4c6e4c1`). The `--analyze` code already existed at `c21b06a`, and git cannot show when the
joint was first computed; the order claim is about the committed outputs. This step reads
each checkpoint's per-tile hit counts separately.

- **Gate:** the per-tile sums of `C` and `C2` in `vertices.json` equal the per-tile hits in
  `results/arm_c/eval.json` (5/6/2/6/12/9) and `results/arm_c_early_stopping/eval_best.json`
  (5/8/4/7/11/5). The per-tile vertex counts equal the eval files' truth counts
  (5/8/8/7/15/14). Each tile's `vertex` indices run 0..n-1, as the adjacency null needs.
- **Informative tiles:** 748, 4098 and 3461 (37 vertices), exactly as the issue predicted.
  Uninformative, reported not dropped: 3674 (C 5/5 and C2 5/5), 1851 (C2 8/8), 5017 (C2 7/7).
- **Null over the informative tiles:** E[S] = 13.01, S_max = 18. The null support is
  S = 8..18 (tile 4098 forces an overlap of at least 12 + 11 - 15 = 8).
- **Smallest achievable one-sided p** (every informative tile at min(a, b)): 1.2e-4, well
  below 0.05. The test can reject.
- **Critical value:** P(S >= 16) = 0.027 and P(S >= 15) = 0.126, so the test rejects at
  0.05 only if S >= 16.

**Consequence for the outcome categories, noted before the joint was read.** S = 16 means an
agreement index of (16 - 13.01) / (18 - 13.01) = 0.60. So **any rejection at p < 0.05 has an
index >= 0.60, and outcome (b), the stated hypothesis, cannot occur on this data by
construction.** The only reachable base outcomes are (a) (S >= 16) and (c) (S <= 15), with
(d) as a possible qualifier. "Partial agreement" in the hypothesis's sense (index between 0
and 0.6) is not distinguishable from chance at this n. Under (c), this is the power
statement the interpretation plan asks for.

Leave-one-informative-tile-out power, from the marginals only:

| Dropped tile | E[S] | S_max | Smallest achievable p | Rejects at 0.05 only if S >= | Index at that S |
|---|---|---|---|---|---|
| 748 | 12.01 | 16 | 5.5e-4 | 15 | 0.75 |
| 4098 | 4.21 | 7 | 0.013 | 7 (= S_max) | 1.00 |
| 3461 | 9.80 | 13 | 1.9e-3 | 12 | 0.69 |

Dropping 4098 leaves a test that rejects only if the two checkpoints agree perfectly on 748
and 3461. A (d) "flip" on that drop would therefore be close to a flip by construction; it
will be reported as such.

### Deciding metrics: C (run 1, epoch 60) vs C2 (run 2, epoch 7), informative tiles only

| Metric | Value |
|---|---|
| Observed overlap S (vertices both checkpoints hit) | **15** |
| E[S] under the within-tile hypergeometric null | 13.01 |
| S_max = Σ min(a_t, b_t) | 18 |
| Exact one-sided p = P(S >= 15) | **0.126** |
| Monte Carlo cross-check (200,000 draws) | p = 0.126, mean S = 13.01 |
| Normalized agreement index (S − E) / (S_max − E) | **0.40** |

`scipy.stats.hypergeom` convolved independently gives the same p (0.12600) and E[S].

### Pre-registered secondary: adjacency (circular-shift) null

C2's hit sequence was circularly shifted along polyline vertex order, independently on each
informative tile. All 8 × 15 × 14 = 1,680 shift combinations were enumerated exactly.

| | Value |
|---|---|
| E[S] under the shift null | 13.01 |
| Exact one-sided p = P(S >= 15) | **0.179** |
| Convolution of the per-tile shift distributions (check) | 0.179 |
| Monte Carlo (20,000 draws) | 0.179 |

Both nulls are on the same side of 0.05 (neither rejects).

### Per tile (informative tiles)

| Tile | n | C hits | C2 hits | Both | E[both] | min–max(both) | Where it sits | Hypergeometric P (same direction) | Adjacency P (same direction, n shifts) |
|---|---|---|---|---|---|---|---|---|---|
| 748 | 8 | 2 | 4 | 2 | 1.00 | 0–2 | maximum: C's hits are a subset of C2's | P(both ≥ 2) = 0.214 | 0.250 (8) |
| 4098 | 15 | 12 | 11 | 8 | 8.80 | 8–11 | minimum possible (12 + 11 − 15 = 8) | P(both ≤ 8) = 0.363 | 0.400 (15) |
| 3461 | 14 | 9 | 5 | 5 | 3.21 | 0–5 | maximum: C2's hits are a subset of C's | P(both ≥ 5) = 0.063 | 0.214 (14) |

The two P columns are **descriptive and not pre-registered** (added in review round 1;
`analysis.json` → `descriptive.per_tile_nulls_not_preregistered`, which also has both tails).
**None of the three per-tile patterns is distinguishable from within-tile chance on its own.**
Nesting on 748 happens by chance about 1 time in 5. On 4098 the "minimum possible" overlap
of 8 is also the single most likely value under the null (P(both = 8) = 0.363). The per-tile
pictures look opposite, but they are what small per-tile counts produce by chance.

### Outcome (d) check: leave one informative tile out

| Dropped | S | E[S] | S_max | p (hypergeometric) | index | p (adjacency, descriptive) |
|---|---|---|---|---|---|---|
| 748 | 13 | 12.01 | 16 | 0.330 | 0.25 | 0.357 |
| **4098** | **7** | 4.21 | 7 | **0.013** | 1.00 | 0.054 |
| 3461 | 10 | 9.80 | 13 | 0.611 | 0.06 | 0.600 |

Dropping 4098 flips the hypergeometric p across 0.05 (0.126 → 0.013), so the pre-registered
(d) rule fires. As recorded in the power check before the joint was read, 7 is that subset's
only rejectable value: p < 0.05 there requires perfect nesting on both remaining tiles. The
adjacency null on the same subset gives p = 0.054, on the other side of 0.05 (descriptive;
the (d) rule compares the two nulls on the full informative set only).

### Descriptive: 2×2 C × C2, all 6 tiles

| Tile | Both | C only | C2 only | Neither |
|---|---|---|---|---|
| 3674 | 5 | 0 | 0 | 0 |
| 1851 | 6 | 0 | 2 | 0 |
| 748 | 2 | 0 | 2 | 4 |
| 5017 | 6 | 0 | 1 | 0 |
| 4098 | 8 | 4 | 3 | 0 |
| 3461 | 5 | 4 | 0 | 5 |
| **Pooled** | **32** | **8** | **8** | **9** |

Pooled raw agreement is 41/57 (72%), but 3674, 1851 and 5017 contribute agreement fixed by
the counts alone.

### Descriptive: vertices where the checkpoints disagree (tile vN, polyline order)

- **C only (8):** 4098 v0, v1, v2, v12; 3461 v1, v3, v11, v12.
- **C2 only (8):** 1851 v0, v6; 748 v0, v4; 5017 v2; 4098 v4, v13, v14.
- **Among the 21 vertices both classical arms miss:** C hits 14, C2 hits 15, both hit 12,
  either hits 17. The checkpoints disagree on 5: 4098 v2 and 3461 v12 (C only); 1851 v0,
  748 v4 and 4098 v13 (C2 only). The 4 vertices missed by A, B, C and C2 are 748 v2, v5, v7
  and 3461 v0.

### Descriptive: A/B reference contrast (same statistic and index)

All 6 tiles are informative for the A/B pair (neither arm hits all or none of any tile).

| Pair | Tiles | S | E[S] | S_max | p (hypergeometric) | MC p | index | p (adjacency) |
|---|---|---|---|---|---|---|---|---|
| C vs C2 | 748, 4098, 3461 | 15 | 13.01 | 18 | 0.126 | 0.126 | 0.40 | 0.179 |
| A vs B | all 6 | 12 | 10.47 | 19 | 0.275 | 0.276 | 0.18 | 0.264 |

Against a chance baseline, A/B agreement is also indistinguishable from within-tile chance
(index 0.18). README's "genuinely complementary" reading of the 12/57 both-hit count is
consistent with that: the two classical arms overlap about as much as their per-tile hit
counts predict. The two C checkpoints sit higher on the index (0.40 vs 0.18) but neither
pair is detectably above chance. The contrast is descriptive and does not decide the outcome.

### Descriptive: a correction to the issue's coverage confound

The issue says C2's mask coverage is higher than C's on every tile. `vertices.json` shows it
is not: on 3461, C2 covers 13.1% against C's 32.2%. The other five tiles do have C2 > C
(4098: 24.8% vs 20.1%). The nesting follows coverage on both nested tiles: on 748 the
higher-coverage checkpoint (C2, 34.7% vs 15.2%) catches a superset of C's hits, and on
3461 the higher-coverage checkpoint (C) catches a superset of C2's. This is noticed after
the fact and not tested.

## Interpretation

**Outcome: (d), with base outcome (c).** Pooled over the three informative tiles, the two
checkpoints' overlap cannot be told apart from within-tile chance (S = 15 against 13.01,
p = 0.126; adjacency null p = 0.179). The (c) result flips to p = 0.013 when tile 4098 is
dropped, so per the pre-registered rule this is a single-tile-dependent result and is
reported per tile, not generalized.

**Against the stated hypothesis.** The hypothesis predicted (b): agreement above chance but
partial. (b) was unreachable at this n by construction, as the power check recorded before
the joint was read: any rejection needed S >= 16, an index of at least 0.60. The observed
index (0.40) is in the "partial" range the hypothesis describes, but it is not
distinguishable from 0. The data therefore neither support nor refute the hypothesis. They
do rule out the opposite extreme the issue names (hit sets nearly identical, index near 1)
on the pooled informative tiles.

**Per tile, which is what the (d) rule asks for:**
- The patterns look opposite: nested on 748 and 3461 (the checkpoint with less mask area on
  the tile catches a subset of the other's hits), and minimal overlap on 4098 (C only: v0-v2,
  v12; C2 only: v4, v13, v14).
- **None of them is distinguishable from within-tile chance on its own:** hypergeometric
  P = 0.21 (748), 0.063 (3461) and P(overlap ≤ 8) = 0.36 (4098); adjacency P = 0.25, 0.21
  and 0.40 (descriptive, not pre-registered).
- So the per-tile view is not evidence of per-tile stability on some tiles or instability on
  others. Like the pooled test, it detects nothing at this n.

**What this establishes (6 tiles, 57 vertices, 2 checkpoints, exact-pixel metric)**
- The Arm C column of the three-way map (PR #9) is not detectably shared between the two
  checkpoints beyond what their per-tile hit counts predict. Per-vertex attribution to
  "Arm C" rather than to a checkpoint is not supported on this evidence.
- It is "not detected", not "independent". The pooled test could only reject at an index
  of 0.60 or more. The per-tile patterns (nested on two tiles, minimal overlap on one) are
  each within chance too.
- Claims about specific vertices hold at checkpoint level. Among the 21 classical shared
  misses, the checkpoints agree on 16 (12 caught by both, 4 missed by both) and disagree on 5.

**What it does not establish**
- **Run vs epoch.** C is run 1 at epoch 60 and C2 is run 2 at epoch 7. Disagreement could
  come from either or both.
- **That the nesting on 748/3461 means anything.** It is two tiles, neither distinguishable
  from chance (P = 0.21 and 0.063). It follows which checkpoint has more mask area on that
  tile, which was noticed after the fact.
- **Anything about more than two runs.** n = 2 checkpoints.
- **A/B.** The reference contrast is descriptive; it suggests the classical arms' overlap is
  also at chance level, but that was not the question here.

**Per interpretation plan (c), qualified by (d):**
- README's three-way map entry is scoped to checkpoints: the Arm C column describes the two
  checkpoints, and per-vertex claims are stated per checkpoint (or where both agree).
- README's OPEN seed-stability item is extended: the per-vertex question needs more runs,
  not only aggregate recall.
- Nothing is generalized from the tile-4098 dependence or from the nesting on 748/3461.

## Decision

**Implementer's self-assessment: MERGE, as an analysis result (a null, qualified per tile).**
- The validation gate reproduced both eval files exactly. `gate.json` and the power section,
  including the consequence that (b) was unreachable, were committed (`c21b06a`) before
  `analysis.json` (`4c6e4c1`). The `--analyze` code already existed at `c21b06a`; git cannot
  show when the joint was first computed.
- The deciding statistic is exact, cross-checked by Monte Carlo and by `scipy`. The
  adjacency null is exhaustively enumerated.
- The outcome is labelled against the pre-registered (a)-(e) definitions, including the (d)
  trigger and why that trigger sits at the edge of the subset's power.
- One factual error in the issue (C2 coverage "higher on every tile") is corrected in the
  descriptive section, not silently.

README.md is updated per interpretation plan (c), qualified by (d). No DECISION_LOG.md
entry: the plan calls for one only under outcome (e).

**Natural follow-ups** (not done here; each would be a new issue):
- More Arm C runs (different seeds, same epoch-selection rule) to estimate per-vertex
  stability directly, rather than from two checkpoints that also differ in epoch.
- The within-run pair (run 2, epoch 7 vs epoch 60) to isolate epoch from run, if a
  non-degenerate epoch-60 checkpoint from run 2 is ever available.
- Whether the coverage-nesting pattern on 748/3461 holds on more tiles: does the checkpoint
  with more mask area on a tile consistently catch a superset of the other's hits?

## Amendments

**Review round 1 (PR #12).** The reviewer found an unsupported per-tile claim: the
Interpretation read "stable across the two runs on two tiles and run-specific on the third"
from per-tile patterns that are each within chance. That sentence is replaced with a
chance-calibrated statement. Per-tile hypergeometric and adjacency P-values were added to
`agreement.py` and `analysis.json` as descriptive, not pre-registered. Every other number is
unchanged: the deciding metrics, the outcome label ((d), base (c)) and the README
conclusions do not change. The commit-order claim is reworded to what git can show.
