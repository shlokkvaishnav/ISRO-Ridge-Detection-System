# SPEC: NMS + hysteresis thresholding vs. shape-filtering

Copied verbatim from [issue #1](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/issues/1),
per research/AGENT_PIPELINE.md's Implementer instructions.

**Type:** method (a new methodological component — threshold scheme)

**Research question**
Does replacing the current percentile-threshold + shape-filter scheme (research/DECISION_LOG.md, 2026-09-22) with orientation-based non-maximum suppression + hysteresis (double) thresholding — the actual missing steps versus a proper Canny-style ridge pipeline — reduce real-tile mask clutter (currently 30-36% of tile area) without losing the recall gain the shape-filter fix already achieved (21/52 → aim to hold or beat this)?

**Hypothesis**
NMS along the local ridge-normal direction will thin the current blob-shaped response regions into single-pixel-wide candidate lines before any thresholding happens, and hysteresis thresholding (high threshold seeds definite ridge pixels, low threshold only extends along already-seeded structure) should then reject isolated clutter that never connects to a strong seed — both mechanisms the current shape-filter approach doesn't have, since it filters blob shape *after* a flat single-threshold cut, not before.

**Null / alternative hypothesis**
NMS+hysteresis performs no better (or worse) than the current shape-filter scheme on the same recall-vs-coverage tradeoff — i.e. for any hysteresis high/low threshold pair tested, either recall drops below 21/52 at matched or lower mask coverage, or mask coverage stays above 30% at matched or higher recall. This would mean the clutter problem isn't a thresholding-scheme artifact and further strengthens the terrain-roughness hypothesis already in DECISION_LOG.md.

**Motivation**
Directly continues the classical-arm thresholding work already in progress (research/DECISION_LOG.md's two 2026-09-22 entries). If this closes most of the gap between Arm A/B's current partial recall and a clean ridge trace, Arm C may be less urgent than the current README's HYPOTHESIS section suggests. If it doesn't help, that's a stronger, more specific data point for the terrain-roughness hypothesis than the current A/B comparison alone.

**Experimental design**
Implement NMS (using the same log-Gabor/Hessian orientation estimate already computed by each arm's filter) + hysteresis thresholding as a new option in src/classical/morphology.py, applied before the existing shape-filter step. Run on the same 6 tiles already used for the Arm A/B comparison (3674, 1851, 748, 5017, 4098, 3461), same ground-truth-vertex-recall metric, same mask-coverage metric. Sweep a small grid of (high, low) hysteresis threshold pairs. Compare against the current shape-filter baseline on the same tiles — not a new tile set, so results are directly comparable to existing numbers.

**Metrics**
True-ridge-vertex recall (same definition as existing DECISION_LOG.md entries: fraction of catalog polyline vertices covered by the final mask) and mask coverage (% of tile area). Both matter — the outcome decides on a recall-at-matched-coverage (or coverage-at-matched-recall) basis, not recall alone.

**Baselines / controls**
The existing shape-filter pipeline's numbers on the same 6 tiles (21/52 recall, 30-36% coverage) is the baseline this is compared against directly — already committed, no need to rerun.

**Expected outcomes**
(a) NMS+hysteresis beats shape-filtering on both axes — best case, likely to replace the current default. (b) It improves one axis at the cost of the other — a real tradeoff, document which regime is preferable for the eventual arm comparison. (c) No meaningful difference — strengthens the terrain-roughness hypothesis. (d) It helps on some tiles and not others in a pattern correlated with ridge length/degradation — would itself be a finding worth its own bucket in the eventual per-morphology-class comparison.

**Interpretation plan**
(a) → adopt as the new default for both Arm A and Arm B (test on Arm B too before fully committing, since it reuses Arm A's post-processing). (b) → keep both options, document the tradeoff, let the eventual comparison pick per use case. (c) → strong additional evidence for Arm C being necessary, not just tuning. (d) → flag for the per-morphology-class/degradation-bucket analysis plan.md already calls for.

**Confounds considered**
Orientation estimation itself may be unreliable on noisy real terrain (the same terrain-roughness problem that broke simple thresholding could break orientation estimation too) — if so, NMS could suppress real ridge pixels as often as clutter. This needs checking directly (e.g. does the estimated orientation at true ridge-line pixels look locally coherent, similar to the existing phase-symmetry-response-at-truth-vertices check already done for the original clutter finding) rather than assumed to work from the synthetic-DEM tests alone, since those don't have real terrain-roughness noise.

---

## Results

**Confound check (orientation coherence at true ridge vertices, segment 3461, 14 vertices):**
mixed. Local 3x3-window circular std of orientation ranged 3.5-39.3 degrees across vertices.
Notably, the highest-symmetry vertices (0.28-0.51) trended toward *less* coherent local
orientation (26-39 deg std) than low-symmetry vertices (3.5-10 deg std) — the opposite of
what would make NMS unambiguously safe. Not a clean abort signal, but not a clean pass
either; proceeded to the actual experiment rather than over-interpreting one diagnostic.

**Sweep, aggregate over all 6 tiles (57 true vertices), best point found:**

| Config | Recall | Coverage |
|---|---|---|
| Baseline (shape-filter, `detect_ridges` defaults) | 23/57 (40.4%) | 33.2% |
| NMS+hysteresis, low=65 high=88 | 24/57 (42.1%) | 30.8% |
| NMS+hysteresis, low=70 high=88 | 24/57 (42.1%) | 30.8% |
| NMS+hysteresis, low=60 high=85 | 27/57 (47.4%) | 41.8% |
| NMS+hysteresis, low=70 high=90 | 20/57 (35.1%) | 23.9% |
| NMS+hysteresis, low=75 high=95 | 5/57 (8.8%) | 10.9% |

**Per-tile breakdown at the aggregate-best point (low=65, high=88):**

| Tile | Baseline hits | Baseline coverage | NMS+hyst hits | NMS+hyst coverage |
|---|---|---|---|---|
| 3674 | 2/5 | 32% | 1/5 | 31% |
| 1851 | 1/8 | 30% | 5/8 | 30% |
| 748 | 3/8 | 36% | 3/8 | 33% |
| 5017 | 4/7 | 35% | 4/7 | 28% |
| 4098 | 6/15 | 36% | 4/15 | 32% |
| 3461 | 7/14 | 31% | 7/14 | 30% |

**Interpretation:** this is outcome (d), not outcome (a). The aggregate numbers alone (42.1%
vs 40.4% recall, 30.8% vs 33.2% coverage) look like a clean win, but the per-tile breakdown
shows it's driven substantially by one tile (1851: 5 vs 1 hits) while two others get
strictly worse (3674: 1 vs 2; 4098: 4 vs 6) and three are flat or nearly so. Reporting the
aggregate alone here would have overstated the result exactly the way
`research/GIT_WORKFLOW.md`'s reviewer instructions warn against.

**What this does establish:** NMS+hysteresis is a real, working alternative with a genuine
(if inconsistent) effect, and its coverage is consistently at-or-below the shape-filter
baseline across every tile tested — even where it loses on recall, it isn't spending more
mask area to do so. That coverage consistency is worth keeping as a documented option.

**What this does NOT establish:** that NMS+hysteresis should replace shape-filtering as the
default, or that it reliably helps. Whatever makes it help dramatically on segment 1851 and
hurt on 3674/4098 is not identified — this needs the per-morphology-class/degradation-bucket
breakdown `plan.md` already calls for, not a single aggregate-recall verdict.

**Confounds that remain:** the mixed orientation-coherence finding above may partly explain
the per-tile inconsistency (tiles where orientation happens to be locally coherent near the
true ridge may be exactly where NMS helps) — not tested directly per-tile, only in aggregate
on one segment.

**Decision: MERGE as an additional documented option, not a new default.** The
implementation is correct and tested (6 new tests, all passing), the coverage-consistency
finding is real and worth keeping available, and the mixed-per-tile result is itself a
legitimate, honestly-reported finding — not a broken feature. `detect_ridges` (shape-filter)
remains the default; `detect_ridges_nms_hysteresis` is available as an alternative for the
eventual per-morphology-class comparison to weigh in on.
