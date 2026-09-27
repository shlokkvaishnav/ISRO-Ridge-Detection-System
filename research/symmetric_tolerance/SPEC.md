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

_Pending: filled in after the validation gate and the pre-registered analysis run._

## Interpretation

_Pending._

## Decision

_Pending._
