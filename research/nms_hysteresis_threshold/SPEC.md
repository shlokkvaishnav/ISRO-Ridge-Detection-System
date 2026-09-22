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
