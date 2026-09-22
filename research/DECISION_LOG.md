# Decision log

Newest first.

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
