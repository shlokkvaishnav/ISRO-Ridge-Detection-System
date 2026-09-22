# ISRO-Ridge-Detection-System

A lunar wrinkle ridge detection system built and evaluated across three approaches —
classical phase-symmetry morphology, Hessian-based ridge filtering, and deep learning —
compared head-to-head on the same DEM tiles to find out which ridges each one catches,
which it misses, and why.

See [`plan.md`](plan.md) for the full plan: the three arms being implemented, data sources
(LOLA DEM, LROC WAC, Chandrayaan-2 TMC-2), comparison methodology, sequencing, and
alternate approaches considered.

## Status

**Arm A implemented; a shape-filtering fix roughly tripled real-ridge recall on real GLD100
tiles, but the mask is still ~30-36% of tile area — a partial, measured improvement, not a
solved detector.** 30-tile pilot pulled directly from the remote LROC WAC + GLD100 mosaics
(no full download, see `research/DECISION_LOG.md`), stratified across the Thompson et al.
2017 ridge catalog. Arms B (Hessian filters) and C (deep learning) not started.

**ESTABLISHED**
> On synthetic DEMs (straight ridge, curved ridge, flat/no-ridge control), Arm A correctly
> detects ridges only when present and tracks a curved ridge's true centerline to within a
> few pixels on average. `tests/test_classical_pipeline.py` (6 tests, all passing).
>
> On real GLD100 tiles, the original percentile-threshold-then-close approach failed: only
> 1 of 14 true ridge vertices (segment 3461) cleared the mask's own 90th-percentile cutoff.
> Reordering to shape-filter individual fragments (favoring elongated components over blobs)
> *before* gap-linking, plus loosening the threshold to the 75th percentile, roughly tripled
> aggregate recall across all 6 inspected tiles (6/52 -> 21/52 true vertices covered). The
> mask still covers 30-36% of tile area on every tile tested — visually large amorphous
> patches, not a clean ridge trace. Full numbers and mechanism in `research/DECISION_LOG.md`.

**HYPOTHESIS** — carried into the eventual arm comparison, not yet tested against B/C:
> That the "two flanks, not one crest line" output shape (see
> `research/DECISION_LOG.md`) is specific to slope-domain phase symmetry and that Arm B
> (Hessian filters, run on the same slope map) will show the same shape, while Arm C (if
> trained against a catalog's single-centerline ground truth) will not.
>
> Whether further real-tile gains need a fundamentally different signal (orientation
> coherence, an absolute physical-units floor) or reflect a real ceiling on slope-domain
> classical detection at 100m/px that motivates Arm C — not yet distinguished from continued
> threshold/shape-parameter tuning.

**OPEN**
> Whether the flat-tile false-positive risk from percentile-based grayscale normalization
> (separate from the terrain-roughness finding above) is a real problem on actual tiles, or
> only synthetic near-zero-relief inputs.
>
> Whether the shape-filtering improvement holds across all 30 pilot tiles or just the 6
> inspected in detail so far.

**DO NOT CLAIM**
> That Arm A "works" on real lunar data, or that the current thresholding scheme is
> finished — it's a measured, partial recall improvement with a real, unmeasured precision
> cost (30-36% mask coverage). Do not claim the phase-symmetry filter itself is broken; the
> response does show local structure near the true ridge, the threshold/shape step is the
> better-supported bottleneck but not the confirmed sole one.

See [`plan.md`](plan.md) for the full plan and `research/DECISION_LOG.md` for implementation
decisions and full findings (why `phasepack` wasn't used, flank-detection, flat-tile, and
real-data clutter/shape-filtering findings).

## Repository structure

- `research/` — spec, related-work positioning, decision log
- `src/` — the three detection arms (classical, hessian, deep) plus shared preprocessing
- `scripts/` — run/compare/report scripts
- `data/` — raw tiles and the public ridge catalog (large files gitignored, download
  scripts committed instead)
- `tests/` — correctness tests for preprocessing and each arm
- `docs/` — architecture and reproduction notes
