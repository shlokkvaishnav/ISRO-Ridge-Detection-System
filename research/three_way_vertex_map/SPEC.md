# SPEC: Does Arm C catch the 21 benchmark vertices both classical arms miss, or share their blind spot?

Copied verbatim from [issue #7](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/issues/7),
per research/AGENT_PIPELINE.md's Implementer instructions.

**Type:** analysis (analysis of an existing result, no new data collection)

**Research question**
On the 6-tile, 57-vertex benchmark, 21 true-ridge vertices are missed by both classical arms (`research/DECISION_LOG.md`, 2026-09-22 complementarity entry). How many of those 21 does Arm C catch? After adjusting for Arm C's per-tile mask coverage, is its hit rate on them different from its hit rate on the 36 vertices at least one classical arm already catches?

**Hypothesis**
Arm C catches a meaningful share of the 21 shared-miss vertices, clearly above what a random mask of equal per-tile coverage would catch. Its hit rate on them is still lower than on the 36 classically-caught vertices. Why: Arm C's aggregate gain (40/57, against 23/57 and 25/57) is too large to come only from re-finding the 36 classically-caught vertices. The per-tile counts on 3674 (C 5/5, A 2, B 3) and 4098 (C 12/15, A 6, B 5) already imply that some shared-miss vertices are caught. But README.md's terrain-roughness HYPOTHESIS predicts that some ridge signal at 100m/px is weak for any method, so a residual set that all three arms miss is expected.

**Null / alternative hypothesis**
Arm C's hits on the 21 shared-miss vertices are about what a coverage-matched random mask would give. That expectation is the sum, over those vertices, of Arm C's coverage fraction on each vertex's tile: about 6–7 of 21 at ~31% coverage. Arm C's advantage would then come almost entirely from the 36 vertices the classical arms already reach, and all three method families would share the blind spot. The opposite would also contradict the hypothesis: if Arm C's hit rate on the 21 shared-miss vertices is as high as or higher than on the 36 caught ones, it has no measurable blind spot in common with the classical arms on this benchmark.

**Motivation**
This is the repo's central research question: does each method family have its own blind spot, or do they share one? `research/RELATED_WORK.md`'s novelty verdict names per-instance attribution across method families as the project's load-bearing, unclaimed contribution. That attribution exists only pairwise so far (A vs B). Arm C has only aggregate and per-tile numbers, so the three-way map the project is built around has never been computed. Either answer changes the README:
- If Arm C reaches what the classical filters cannot, that counts against the terrain-roughness hypothesis as a limit on every method family.
- If there is a residual set all three arms miss, that directly supports the hypothesis. It also names the specific vertices a future per-morphology breakdown should look at first.

No training is needed. The analysis uses the committed checkpoints (`results/arm_c/model.pt`, `results/arm_c_early_stopping/model_best.pt`), the existing evaluator's metric, and the deterministic Arm A/B pipelines. It runs on the 6 benchmark tiles that `scripts/build_tile_dataset.py` already produces.

**Experimental design**
- **Tiles:** the 6 benchmark tiles only (3674, 1851, 748, 5017, 4098, 3461; `results/arm_c/val_ids.json`), all held out of Arm C training. The tiles are gitignored; if they are not already local, regenerate these 6 with the existing script. No other tile extraction.
- **Arms:**
  - Arm A: the default `detect_ridges` (shape-filter), with the same parameters as the 2026-09-22 benchmark.
  - Arm B: frangi, with the same post-processing.
  - Arm C: `results/arm_c/model.pt` (the merged headline checkpoint) is primary, and `results/arm_c_early_stopping/model_best.pt` (epoch 7) is secondary. Both run at the pre-registered 0.5 threshold, with the same resize-to-native procedure as `src/deep/evaluate.py`.
- **Per-vertex table:** for each of the 57 truth vertices, record hit/miss for A, B and C, plus the tile and each arm's per-tile coverage. A hit uses the existing rule: `mask[r, c]` at the vertex's reprojected native pixel.
- **Validation gate, before any Arm C result is read:**
  - The regenerated A/B per-vertex table must reproduce the recorded 12 / 11 / 13 / 21 split (both / A-only / B-only / neither) and the per-tile A/B hits in DECISION_LOG.md.
  - The Arm C per-tile hits must exactly reproduce `results/arm_c/eval.json` and `results/arm_c_early_stopping/eval_best.json`.
  - If either check fails, stop and report the discrepancy. Do not continue on a baseline that does not reproduce.
- **Held constant:** tiles, truth vertices, reprojection code and thresholds. Nothing is tuned.

**Metrics**
Deciding metrics, on the primary checkpoint (`results/arm_c/model.pt`):
1. Arm C hits among the 21 vertices both classical arms miss, compared with the coverage-matched chance expectation: the sum over those 21 vertices of Arm C's coverage fraction on each vertex's tile. Report an exact one-sided p-value from a Poisson-binomial with those per-vertex probabilities.
2. Arm C's hit rate on those 21 vertices compared with its hit rate on the 36 vertices at least one classical arm catches. Express each rate as lift over its own coverage-matched expectation.

Descriptive only, not deciding:
- the full 8-cell A×B×C per-vertex table
- the list of vertices all three arms miss, with tile IDs
- the same table for `model_best.pt`
- a per-tile breakdown of everything above

**Baselines / controls**
- A coverage-matched random mask per tile. This is the same analytic expectation as `random_mask_expected_hits` in `results/arm_c_early_stopping/threshold_sweep.json`, applied to each vertex subset rather than to the tile total.
- Arm C's own hit rate on the 36 classically-caught vertices is the internal control for harder versus easier vertices.
- No synthetic-DEM baseline is needed. The question is about which real-tile instances each arm catches, and each arm's behaviour on synthetic DEMs is already established.

**Expected outcomes**
(a) C catches the shared-miss set well above chance (p < 0.05 on metric 1), with a lift roughly equal to its lift on the 36 caught vertices. Arm C does not share the classical blind spot on this benchmark.
(b) C catches the shared-miss set above chance but with a clearly lower lift than on the caught set, and a non-trivial set of vertices all three arms miss remains. The blind spot is partly shared (the stated hypothesis).
(c) C's hits on the shared-miss set cannot be told apart from chance (p ≥ 0.05). All three method families share the blind spot, and Arm C's gain comes from vertices the classical arms already reach.
(d) Any of (a)–(c), but the vertices all three arms miss, or the vertices only C catches, concentrate in one or two tiles. The apparent per-vertex effect is then really a per-tile effect.
(e) The validation gate fails: the A/B per-vertex split or the C per-tile numbers do not reproduce.

**Interpretation plan**
(a) Add the three-way map to README's ESTABLISHED tier, qualified as "on this benchmark." Weaken the terrain-roughness HYPOTHESIS: the shared-miss vertices are not undetectable at 100m/px from DEM-derived inputs. This does NOT show that Arm C has no blind spots in general (n=21, 6 tiles).
(b) Report the three-way map and name the vertices all three arms miss. This fits the terrain-roughness hypothesis but does not confirm it, because those vertices could also reflect catalog or registration error. The natural next question is to inspect those specific vertices (morphology, relief, catalog placement).
(c) Strong support, on this benchmark, for a limit that applies to all three method families. Arm C's headline should then read "catches classically-reachable vertices more reliably," not "catches what classical methods miss." This does NOT show the three families are equivalent overall.
(d) Report per tile, and do not generalize a per-vertex claim from a one-tile effect. This points to the existing OPEN item about extending the benchmark beyond 6 tiles.
(e) Report the discrepancy as the finding and add a DECISION_LOG entry. Make no three-way claim until it is resolved.

For every outcome, report the secondary checkpoint alongside the primary. If the two land in different outcome categories, say so rather than choosing one.

**Confounds considered**
- **Mask shape vs the exact-pixel metric.** Arms A and B work on slope and characteristically mark the ridge flanks, not the crest (DECISION_LOG 2026-09-18). Arm C was trained on polyline-buffer labels that sit on the catalog line itself. So C might "catch" a shared-miss vertex because its output is centered on the catalog line, not because it detects signal the classical filters cannot. The primary metric does not control for this. As a descriptive check, also score A and B at the same vertices with a small pixel tolerance (e.g. a hit if any mask pixel lies within 1–2 px). Say explicitly whether the set both classical arms miss shrinks substantially under that tolerance. If it does, part of the "blind spot" is an artefact of the metric.
- **Non-independence.** The 57 vertices sit in 6 tiles, and vertices on the same ridge are correlated. The Poisson-binomial p-value treats them as independent, so it is optimistic. Per-tile reporting (outcome d) guards against this; it does not correct for it.
- **Small n.** With 21 shared-miss vertices, only large differences are detectable. State the minimum detectable difference rather than reading a null result as "no difference."
- **Checkpoint instability.** Epoch-60 checkpoints from identical code scored 70.2% and 21.1% (`research/arm_c_early_stopping/SPEC.md`). Any three-way map therefore describes a specific checkpoint, not "Arm C" in general, which is why two checkpoints are reported. The 21.1% checkpoint is excluded: it is near chance, so it tells us nothing about which vertices Arm C catches. The epoch-7 checkpoint was selected by val loss on these same 6 tiles, which is mild selection on the benchmark and is already recorded as a confound in PR #6.
- **Threshold.** Fixed at 0.5, as pre-registered everywhere else; no threshold is tuned for this analysis. The epoch-7 checkpoint's recall is sensitive to the threshold, so its secondary result carries that caveat.
- **Catalog or registration error.** A vertex all three arms miss may be a misplaced catalog vertex rather than a genuinely hard ridge. This analysis cannot tell the two apart, so the report must present catalog error as a live alternative to the terrain-roughness reading.


## Amendments

Dated notes only. The sections above are the issue body, unchanged.
