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


