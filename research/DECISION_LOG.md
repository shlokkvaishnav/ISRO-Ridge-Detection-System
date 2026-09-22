# Decision log

Newest first.

## 2026-09-22 — Arm A and Arm B are complementary, but the union isn't free

Follow-up to the head-to-head entry below, answering its own OPEN item: do Arm A and Arm B
catch the same true-ridge vertices or different ones?

**ESTABLISHED**
> Per-vertex breakdown across the same 6 tiles (57 true vertices total): both arms hit 12,
> Arm A only 11, Arm B only 13, neither 21. The two filters are genuinely complementary, not
> noisy variants of the same signal -- roughly half of all correctly-detected vertices (24 of
> 36) are caught by only one of the two arms. Taking the union (A OR B) lifts aggregate
> recall from 40.4%/43.9% (either alone) to 63.2% -- a real, substantial jump.
>
> This is not free: mask coverage also rises with the union, from ~28-40% (either arm alone)
> to ~48-58% of tile area. The union's coverage growth roughly tracks its recall growth
> (e.g. segment 3461: A=31%/B=28% -> union=48%, alongside 7/14 and 10/14 -> 12/14 hits) --
> the "extra" hits from combining arms come from genuinely different response regions, not
> from one arm being a cheap superset of the other. An ensemble is a real recall/precision
> tradeoff, not a strictly-better option.

**HYPOTHESIS**
> That this complementarity itself is further evidence for the terrain-roughness hypothesis
> in the entry below: if two structurally different filters each catch a different partial
> subset of the true ridge and still leave 21/57 (37%) uncaught by either, that's consistent
> with the ridge signal being genuinely weak/ambiguous relative to background terrain at
> 100m/px in a way no local-filter-response approach fully resolves, rather than either
> filter simply being tuned wrong.

**OPEN**
> Whether a smarter combination than a plain OR-union (e.g. requiring agreement in a local
> neighborhood, or weighting by local response confidence) could get more of the recall gain
> without the full coverage cost. Not tried -- the plain union was checked first since it's
> the simplest test of "are these complementary at all," which was the actual open question.

**DO NOT CLAIM**
> That an ensemble of Arm A and Arm B is a good practical detector -- the plain union costs
> roughly as much in added mask area as it gains in recall. This finding answers whether the
> two arms are complementary (yes), not whether combining them is a good idea as a shipped
> approach.

## 2026-09-22 — Arm B (Hessian filters) implemented; head-to-head with Arm A on real tiles

## 2026-09-22 — Arm B (Hessian filters) implemented; head-to-head with Arm A on real tiles

`src/hessian/ridge_filter.py` wraps scikit-image's `frangi`/`meijering`/`sato` directly
(no from-scratch reimplementation -- these are already well-tested library functions, unlike
phase symmetry which needed vendoring, see the `phasepack` entry below). `src/hessian/pipeline.py`
reuses Arm A's exact threshold/shape-filter/gap-link morphology
(`src/classical/morphology.py`) and slope preprocessing unchanged, so any difference between
arms is attributable to the ridge-detection filter itself, not different post-processing.

**ESTABLISHED**
> Head-to-head on the same 6 real tiles used for Arm A's tuning (see the two entries below),
> same truth-vertex-recall metric: Arm B (frangi) gets 25/57 (43.9%) aggregate recall vs
> Arm A's 23/57 (40.4%) -- close, not a clear win for either. Per-tile, neither arm
> dominates: Arm B wins on 3674 (3 vs 2), 1851 (3 vs 1), 3461 (10 vs 7); Arm A wins on 748
> (3 vs 2), 5017 (4 vs 2), 4098 (6 vs 5). Both arms' mask coverage lands in the same 28-40%
> of tile area range on every tile.

**HYPOTHESIS** — strengthened, not yet confirmed:
> That the real-tile clutter problem (both arms landing in the same 30-40% mask-coverage,
> partial-recall range, despite using different ridge-detection filters entirely) reflects
> something more fundamental than either specific filter -- most likely that 100m/px GLD100
> terrain roughness genuinely competes with real wrinkle-ridge signal at the same spatial
> scale, which neither a phase-symmetry nor a Hessian-eigenvalue filter can fully separate
> from a slope map alone. This is the strongest evidence yet for needing Arm C (deep
> learning, which can learn texture/context cues beyond local ridge-shaped-response filters)
> rather than continued classical-arm tuning -- but not proven; still needs the per-
> morphology-class/degradation-bucket breakdown the original plan.md calls for, not just an
> aggregate number.

**OPEN**
> Whether `meijering` or `sato` (not yet tried, only `frangi`) perform meaningfully
> differently from each other on the same tiles -- Arm B currently only tested with one of
> its three available methods.
>
> Whether the two arms are catching the *same* vertices or complementary ones -- an
> aggregate recall comparison doesn't show this. If Arm A and Arm B's hits are substantially
> non-overlapping, an ensemble might outperform either alone even without deep learning;
> untested.

**DO NOT CLAIM**
> That Arm B is "better" than Arm A, or vice versa -- the aggregate numbers are close and
> the per-tile pattern is mixed. Do not claim the terrain-roughness hypothesis above is
> confirmed -- it's the best-supported explanation so far, not a tested one.

## 2026-09-22 — Shape-filtered thresholding: real, partial improvement, not a fix

## 2026-09-22 — Shape-filtered thresholding: real, partial improvement, not a fix

Follow-up to the same-day entry below. Tuned against real tiles (not assumed): the fix is
to shape-filter individual thresholded fragments (favoring elongated components over blobs,
via `skimage.measure.regionprops` eccentricity + major-axis length) *before* gap-linking,
not after -- gap-linking first fuses most of a real tile's thresholded pixels into one
sprawling connected blob, at which point per-component shape discrimination has nothing
left to discriminate between. New defaults: `threshold_percentile=75` (loosened from 90,
so more of the true ridge's moderate response values are admitted),
`min_length_px=8, min_eccentricity=0.85` (rejects blob-shaped clutter),
`gap_link_radius=4`. See `src/classical/morphology.py` and
`tests/test_classical_pipeline.py::TestCleanRidgeMask::test_blob_shaped_clutter_is_rejected`
for the mechanism directly under test.

**ESTABLISHED**
> Across all 6 previously-inspected real tiles (3674, 1851, 748, 5017, 4098, 3461), true-
> ridge-vertex recall roughly tripled in aggregate (6/52 -> 21/52 true vertices covered by
> the mask) after reordering shape-filtering before gap-linking and loosening the threshold.
> Per-tile: 3674 (2/5), 1851 (1/8), 748 (3/8), 5017 (4/7), 4098 (6/15), 3461 (7/14).

**Still true, not fixed**:
> The mask covers 30-36% of tile area on every tile tested -- visually, this is large
> amorphous white patches, not a clean line tracing the ridge (see
> `results/real_tiles_v2/3461/classical_arm_stages.png` for a representative example). This
> is a real, measured, reproducible improvement in recall, not a solved detector -- calling
> it "working" would overstate what changed. Most of the mask is still not the ridge.

**OPEN**
> Whether further gains need a fundamentally different signal (orientation coherence across
> neighboring pixels, an absolute physical-units floor rather than any percentile scheme,
> or accepting that slope-domain classical detection has a real ceiling on 100m/px terrain
> that motivates Arm C) rather than continued threshold/shape-parameter tuning. Not yet
> tested: whether Arm B (Hessian filters) on the same slope maps has a different failure
> mode or the same one.

**DO NOT CLAIM**
> That Arm A now "works" on real data, or that this is a finished thresholding scheme --
> it is a measured, partial improvement (recall) with a real, unresolved cost (mask area /
> precision) that has not been optimized or even measured in this pass.

## 2026-09-22 — On real lunar tiles, percentile thresholding fails to isolate the mapped ridge from terrain-roughness clutter

First real-data test of Arm A, on 30 tiles extracted from LROC WAC + GLD100 (see
`scripts/build_tile_dataset.py`), stratified by ridge length across the Thompson et al.
2017 catalog. This lifts the "DO NOT CLAIM anything about real lunar ridges yet" caveat
below -- with a negative result, not a positive one.

**ESTABLISHED**
> On real GLD100 tiles, the ridge masks are visually dominated by scattered, disconnected
> blobs tracing general terrain roughness, not a clean trace of the mapped ridge -- unlike
> the clean single/double-line output on synthetic Gaussian ridges. Quantified directly on
> segment 3461 (31.7km ridge, 14-vertex polyline, tile shape 117x317): sampling the phase-
> symmetry response at the true ridge line's own pixel coordinates gives values of
> [0.0, 0.318, 0.366, 0.0, 0.0, 0.510, 0.379, 0.396, 0.282, 0.044, 0.0, 0.0, 0.0, 0.067] --
> mostly at or below the tile's own mean response (0.173), with only **one of fourteen**
> vertices clearing the 90th-percentile cutoff (0.466) the mask's threshold uses. Meanwhile
> the mask still marks exactly the top 10% of the tile regardless, because percentile
> thresholding has no absolute concept of "ridge" -- it returns the top decile of whatever
> is in the tile, ridge or not. On real, naturally rough 100m/px terrain, that decile is
> dominated by ordinary terrain texture (small craters, general roughness), not the mapped
> ridge specifically. This directly explains the scattered, non-ridge-tracing masks seen on
> tiles 1851, 3461, and (by pixel-count pattern, not yet overlay-confirmed) 748/5017/4098/3674.

**Root cause, as far as tested**: not (necessarily) that phase symmetry fails to respond to
the real ridge -- the ridge line does show locally elevated response at several vertices
(0.3-0.5) relative to the tile's low points (0.0) -- but that plain percentile thresholding
cannot separate "elevated because it's the ridge" from "elevated because it's ordinary
terrain roughness that happens to be locally high everywhere." The threshold step, not
necessarily the phase-symmetry filter itself, is the current failure point -- but this is
not yet confirmed by testing an alternative thresholding scheme, only inferred from one
segment's response profile.

**OPEN**
> Whether a fix belongs in the threshold/morphology stage (e.g. an absolute response floor,
> ridge-orientation coherence across neighboring pixels, minimum-length connected-component
> filtering) or whether this reflects a more fundamental limit of slope-domain phase
> symmetry on real, degraded, 100m/px terrain -- which would match the literature's own
> stated motivation for building deep-learning detectors (HL-YOLOv8) specifically because
> classical methods miss degraded/eroded ridges. Not yet distinguished.
>
> Whether this pattern holds across all 30 pilot tiles or is specific to the two inspected
> in detail (1851, 3461) -- only one segment (3461) was checked quantitatively against its
> true polyline; the rest were only checked by eye (mask visually resembling scattered
> noise) or by pixel-count order of magnitude, not confirmed by overlay.

**DO NOT CLAIM**
> That Arm A "works" on real lunar data -- it does not, in its current threshold form, on
> the tiles tested so far. Also do not claim the phase-symmetry filter itself is broken --
> the response does show local structure near the true ridge; the threshold step is the
> better-supported culprit but not proven to be the sole one.

## 2026-09-18 — Slope-based phase symmetry detects ridge flanks, not a single crest line

On a synthetic Gaussian-cross-section ridge, Arm A's output is **two** parallel lines
(tracking the ridge's two flanks) rather than one line along the crest. This is inherent
to building the pipeline on a slope map: a Gaussian ridge's slope is highest on both
flanks and passes through ~zero at the exact crest (and at the base), so a slope-domain
symmetry filter finds the flanks, not the peak.

This is not being treated as a bug to fix by switching to elevation-domain filtering —
the reference literature (Frontiers 2023, ISPRS 2019) also operates on slope maps, so this
is expected, reproducible behavior of the established method, not an implementation error.
It does mean any later comparison against Arm B/Arm C needs to define ridge-match tolerance
in terms of "close to the true ridge structure" (either flank, or within some distance of
the crest) rather than expecting a single centerline output from every arm — record this in
the eventual comparison SPEC rather than silently normalizing the outputs.

## 2026-09-18 — Flat-tile false positives from percentile-based grayscale contrast stretch

`dem_to_slope.slope_to_grayscale` normalizes slope values against their own 0.5th/99.5th
percentile range. On a real ridge-bearing tile this is the right behavior — it makes the
pipeline robust to varying absolute relief across tiles. But on a **tile with no real ridge
signal** (e.g. flat mare interior, or numerical noise on an otherwise flat surface), this
normalization stretches whatever tiny variation exists to fill the full 0-255 range,
which the phase symmetry filter and percentile-based thresholding downstream can then
mistake for real structure — an early version of `tests/test_classical_pipeline.py`
reproduced this directly: a flat DEM with small Gaussian noise (std=0.01) produced ~29% of
pixels marked "ridge."

**Fix applied so far**: `threshold_response` now requires a strict `> cutoff` against a
cutoff floored at `1e-6`, which correctly rejects the degenerate all-zero-response case (a
genuinely flat, noiseless tile). This does **not** fully solve the noisy-flat-tile case
above — it only fixes the exact-zero case. The noisy-flat-tile risk is real and open: a
tile that is flat except for small numeric/sensor noise can still produce a stretched,
apparently-structured grayscale image that the phase symmetry filter responds to.

**OPEN**: whether real LOLA/TMC-2 tiles ever have this little genuine relief in a full
tile (unlikely at typical tile sizes, since some real topography is nearly always present),
and if they do, whether an absolute (not just relative/percentile) relief floor should gate
the pipeline before it even attempts detection on a tile. Not implemented yet — flagged
here rather than silently left for a real tile to surface later.

## 2026-09-18 — `phasepack` not used; monogenic phase symmetry implemented directly

`phasepack` (the existing Python port of Kovesi's phase symmetry) was the first choice per
the implementation plan, but it's unmaintained since ~2015 and a quick check of its source
shows reliance on numpy APIs removed in recent numpy releases. Rather than fighting an
unmaintained dependency, `src/classical/phase_symmetry.py` implements the monogenic
phase-symmetry formulation (log-Gabor filter bank + Riesz transform) directly, matching
Kovesi's published method. This also keeps the implementation inspectable, consistent with
this account's general preference (see `Replica-Recall-Divergence`'s `nano-db`) for owning
the parts of a pipeline that matter for understanding *why* a result holds, rather than
depending on an opaque or fragile third-party implementation for the core mechanism being
studied.
