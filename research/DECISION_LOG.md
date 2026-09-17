# Decision log

Newest first.

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
