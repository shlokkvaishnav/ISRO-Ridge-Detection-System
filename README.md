# ISRO-Ridge-Detection-System

A lunar wrinkle ridge detection system built and evaluated across three approaches —
classical phase-symmetry morphology, Hessian-based ridge filtering, and deep learning —
compared head-to-head on the same DEM tiles to find out which ridges each one catches,
which it misses, and why.

See [`plan.md`](plan.md) for the full plan: the three arms being implemented, data sources
(LOLA DEM, LROC WAC, Chandrayaan-2 TMC-2), comparison methodology, sequencing, and
alternate approaches considered.

## Status

**All three arms implemented. Arm C (deep learning) substantially beats both classical arms
on the exact same 6-tile benchmark (70.2% recall vs 40.4%/43.9%) — a real result, matched
at the loss-optimal (early-stopped) checkpoint at the fixed 0.5 threshold, with
qualifiers, not yet a settled win (see below).** 120-tile
pilot pulled directly from the remote LROC WAC + GLD100 mosaics (no full download, see
`research/DECISION_LOG.md`), stratified across the Thompson et al. 2017 ridge catalog.

**ESTABLISHED**
> On synthetic DEMs, both Arm A and Arm B correctly detect ridges only when present and
> track a curved ridge's centerline to within a few pixels on average (11 tests total,
> `tests/test_classical_pipeline.py` + `tests/test_hessian_pipeline.py`, all passing).
>
> On real GLD100 tiles, Arm A's original percentile-threshold-then-close approach failed
> (1/14 true vertices on segment 3461); shape-filtering fragments *before* gap-linking
> roughly tripled aggregate recall (6/52 -> 21/52 across 6 tiles). Arm B (Hessian/frangi),
> reusing Arm A's exact post-processing so the comparison isolates the filter itself, gets a
> close but not clearly better result: 25/57 (43.9%) vs Arm A's 23/57 (40.4%) aggregate
> recall, with per-tile wins split roughly evenly between the two arms, and both landing in
> the same 28-40% mask-coverage range on every tile. Full numbers in `research/DECISION_LOG.md`.
>
> The two arms are genuinely complementary, not noisy variants of the same signal: per-
> vertex breakdown shows both hit 12, A-only 11, B-only 13, neither 21 (of 57). Taking the
> union lifts recall to 63.2% — but not for free: mask coverage rises proportionally too,
> from ~28-40% (either arm alone) to ~48-58% of tile area. This is a real recall/precision
> tradeoff, not a free ensemble win.
>
> An alternative Arm A post-processing (non-maximum suppression + hysteresis thresholding,
> instead of shape-filtering) was tried and merged as an additional option, not a new
> default — mixed result, driven substantially by one tile, two others got strictly worse.
> Coverage was consistently at-or-below the shape-filter baseline on every tile tested
> though, which is the one part worth keeping. Full numbers in `research/DECISION_LOG.md`.
>
> **Arm C (dual-branch DEM+aspect CNN, DBR-Net-inspired) beats both classical arms on the
> same 6-tile, 57-vertex benchmark**: 40/57 (70.2%) recall at 30.8% coverage, vs Arm A's
> 40.4%/33.2% and Arm B's 43.9%. Trained on Kaggle's GPU (114 of 120 tiles, the 6 benchmark
> tiles held out of training entirely — not just a random split, which a first attempt used
> and which leaked 5 of those 6 tiles into training before being caught). Architecture
> deliberately smaller than DBR-Net's own (ResNet-34-per-branch would overfit ~9x less
> training data); weak polyline-buffer labels instead of DBR-Net's hand-labeled masks, a
> real, acknowledged supervision-quality gap. **A real confound attached to this number**:
> training history shows clear overfitting past epoch ~4-10 (val loss bottoms out at epoch 4,
> rises 2-6x by epoch 60 while train loss keeps falling), no early stopping was used, and
> only the final (overfit) checkpoint was saved. **A retrain that saved both checkpoints
> ([#5](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/issues/5)) partly
> settles this.** At the fixed 0.5 threshold, the loss-optimal checkpoint (epoch 7) gives
> the same aggregate recall (40/57) at similar pixel-weighted coverage (31.8%), so ~70%
> recall at ~31% coverage is reachable without training into the overfit regime. The match
> is threshold-sensitive (epoch-7 recall runs 56 to 12 of 57 over thresholds 0.3–0.7) and
> per-tile hits differ. After adjusting for coverage, epoch 7 is somewhat weaker (1.96× vs
> 2.26× lift over a random mask). The epoch-60 checkpoint scored 70.2% and 21.1% across two
> runs of identical code (`research/arm_c_early_stopping/SPEC.md`). Full numbers, per-tile
> breakdown, and the five infrastructure bugs hit getting a Kaggle GPU run working at all:
> `research/arm_c_deep_learning/SPEC.md`.
>
> **Three-way per-vertex map, on this benchmark
> ([#7](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/issues/7),
> `research/three_way_vertex_map/SPEC.md`):**
> - **Reproduction first.** The A/B 12/11/13/21 split and both Arm C checkpoints' per-tile
>   scores were reproduced exactly before anything was read.
> - **Arm C catches most of the classical arms' shared misses.** Of the 21 vertices both
>   classical arms miss, the merged Arm C checkpoint catches 14. A random mask with the same
>   per-tile coverage would catch 6.0 (exact one-sided p = 1.9e-4, vertices treated as
>   independent, so optimistic).
> - **No lower hit rate on the hard set.** Its lift over chance there (2.32x) is about the
>   same as on the 36 classically-caught vertices (2.23x). Only a hit-rate gap of about 35 pp
>   or more was detectable at this n.
> - **The epoch-7 checkpoint agrees less cleanly:** 15/21 against 8.3 expected, p = 0.0016,
>   but with a 12% lower lift on the shared-miss set. That result depends on tile 4098.
> - **The residual set is small and concentrated.** 7 vertices are missed by all three arms
>   (with the merged checkpoint as Arm C), 4 of them on tile 748. Only 4 are missed by both
>   classical arms and both Arm C checkpoints: 748 v2/v5/v7 and 3461 v0.
> - **The Arm C column describes two checkpoints, not the method family
>   ([#10](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/issues/10),
>   `research/arm_c_checkpoint_agreement/SPEC.md`).** Within each tile, the two
>   checkpoints' overlap is not detectably above what their per-tile hit counts predict (15 vs 13.0
>   expected on the 3 informative tiles, p = 0.13, agreement index 0.40; the test could only
>   reject at index >= 0.60). Per tile it is mixed: fully nested on 748 and 3461, minimal
>   overlap on 4098. Per-vertex claims hold per checkpoint, or where both agree: of the 21
>   classical shared misses, both catch 12, both miss 4, and they disagree on 5.
> - **Caveat: part of the classical "blind spot" is the exact-pixel metric.** At a 2 px
>   tolerance, the set both A and B miss shrinks from 21 to 12. Arm C still hits 8 of those
>   12, but Arm C was not itself scored at tolerance.
> - **Scope.** This holds on this benchmark only: 6 tiles, 57 vertices, two checkpoints. It
>   does not show that Arm C has no blind spots in general.

**HYPOTHESIS**
> That the real-tile clutter problem reflects something more fundamental than either
> specific ridge-detection filter — most likely that 100m/px GLD100 terrain roughness
> genuinely competes with real wrinkle-ridge signal at the same spatial scale, which neither
> phase symmetry nor Hessian-eigenvalue filtering can fully separate from a slope map alone.
> The A/B complementarity finding above strengthens this further: two structurally different
> filters each catch a different partial subset and still leave 37% of true vertices uncaught
> by either — consistent with a genuinely weak/ambiguous signal at this resolution, not
> either filter being tuned wrong. Strongest evidence yet for needing Arm C rather than
> continued classical-arm tuning — Arm C's result below is consistent with this, not
> conclusive proof given its overfitting confound. Still needs the per-morphology-class/
> degradation-bucket breakdown `plan.md` calls for, not just an aggregate number.
>
> **Weakened by the three-way map (#7):** as a limit on *every* method family, this
> hypothesis now has evidence against it on this benchmark. Arm C reaches 14 of the 21
> vertices both classical arms miss, well above chance, so those vertices are mostly not
> undetectable from DEM-derived inputs at 100m/px. The hypothesis survives only as an
> account of why the *classical* filters fail. The small residual set that all three arms
> miss (7 vertices, 4 on one tile) is just as compatible with catalog or registration error.

**OPEN**
> Whether `meijering`/`sato` (Arm B's other two methods, not yet tried) perform
> meaningfully differently from `frangi` on the same tiles.
>
> Whether a smarter A/B combination (local-neighborhood agreement, confidence-weighted)
> could capture more of the union's recall gain without its full coverage cost — only the
> plain OR-union was tested, since it was the simplest way to answer whether the two arms
> are complementary at all.
>
> Whether the flat-tile false-positive risk from percentile-based grayscale normalization
> is a real problem on actual tiles, or only synthetic near-zero-relief inputs.
>
> Whether these findings hold across all 120 pilot tiles or just the 6 used for the
> head-to-head benchmark so far.
>
> Whether Arm C's result is stable across seeds. Across two runs of identical code, the
> epoch-60 checkpoint scored 70.2% and 21.1%. The loss-optimal checkpoint has one
> evaluation, and its epoch is chosen by (noisy) val loss on the benchmark tiles
> themselves. That is mild selection on the test set; a disjoint early-stopping split would
> remove it. Stability is also open per vertex, not only in aggregate: the two 40/57
> checkpoints (different runs and epochs) do not detectably agree on *which* vertices they
> catch beyond within-tile chance (#10, p = 0.13 on 3 informative tiles; "not detected",
> not "shown independent"). Answering that needs more runs, not only more aggregate scores.

**DO NOT CLAIM**
> That Arm C is a validated, production-ready detector, or that its 70.2% recall is
> guaranteed to reproduce on a rerun. It was matched once at the loss-optimal checkpoint,
> at one threshold, with best-epoch selection on the benchmark tiles, and an epoch-60
> checkpoint from the same code scored 21.1%. Do not claim the two checkpoints behave
> alike: they tie only at 0.5.
> "Beats the classical arms in this run" is the supported claim, not "is a better detector"
> unqualified.
>
> That either classical arm "works" on real lunar data in a usable sense, or that Arm B is better/worse
> than Arm A — the aggregate numbers are close and the per-tile pattern is mixed. That an
> ensemble of A and B is a good practical detector — the plain union costs roughly as much in
> added mask area as it gains in recall. Do not claim the terrain-roughness hypothesis above
> is confirmed; it's the best-supported explanation so far, not a tested one.

See [`plan.md`](plan.md) for the full plan, `research/DECISION_LOG.md` for implementation
decisions and full findings (why `phasepack` wasn't used, flank-detection, flat-tile,
real-data clutter/shape-filtering, Arm A vs Arm B comparison, A/B complementarity, and the
Arm C headline result), and `research/arm_c_deep_learning/SPEC.md` for Arm C's full writeup
including the overfitting confound and infrastructure incidents getting a Kaggle GPU run
working at all.

## Repository structure

- `research/` — spec, related-work positioning, decision log
- `src/` — the three detection arms (classical, hessian, deep) plus shared preprocessing
- `scripts/` — run/compare/report scripts
- `data/` — raw tiles and the public ridge catalog (large files gitignored, download
  scripts committed instead)
- `tests/` — correctness tests for preprocessing and each arm
- `docs/` — architecture and reproduction notes
