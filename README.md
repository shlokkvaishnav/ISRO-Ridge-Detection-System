# ISRO-Ridge-Detection-System

A lunar wrinkle ridge detection system built and evaluated across three approaches —
classical phase-symmetry morphology, Hessian-based ridge filtering, and deep learning —
compared head-to-head on the same DEM tiles to find out which ridges each one catches,
which it misses, and why.

See [`plan.md`](plan.md) for the full plan: the three arms being implemented, data sources
(LOLA DEM, LROC WAC, Chandrayaan-2 TMC-2), comparison methodology, sequencing, and
alternate approaches considered.

## Status

**Arm A implemented; passes on synthetic ridges, fails to cleanly isolate real ridges from
terrain-roughness clutter on real GLD100 tiles.** 30-tile pilot pulled directly from the
remote LROC WAC + GLD100 mosaics (no full download, see `research/DECISION_LOG.md`),
stratified across the Thompson et al. 2017 ridge catalog. Arms B (Hessian filters) and C
(deep learning) not started.

**ESTABLISHED**
> On synthetic DEMs (straight ridge, curved ridge, flat/no-ridge control), Arm A correctly
> detects ridges only when present and tracks a curved ridge's true centerline to within a
> few pixels on average. `tests/test_classical_pipeline.py` (5 tests, all passing).
>
> On real GLD100 tiles, this does not transfer: percentile-based thresholding produces
> masks dominated by scattered terrain-roughness clutter, not a clean ridge trace. Quantified
> on segment 3461 (31.7km ridge): only 1 of 14 true ridge-line vertices clears the mask's own
> 90th-percentile threshold; most score at or below the tile's mean response. Full writeup
> and root-cause analysis in `research/DECISION_LOG.md`.

**HYPOTHESIS** — carried into the eventual arm comparison, not yet tested against B/C:
> That the "two flanks, not one crest line" output shape (see
> `research/DECISION_LOG.md`) is specific to slope-domain phase symmetry and that Arm B
> (Hessian filters, run on the same slope map) will show the same shape, while Arm C (if
> trained against a catalog's single-centerline ground truth) will not.
>
> That the real-tile thresholding failure is fixable at the threshold/morphology stage
> (absolute response floor, orientation coherence, connected-component length filtering)
> rather than being an inherent limit of slope-domain phase symmetry on degraded, 100m/px
> terrain. Not yet distinguished — see `research/DECISION_LOG.md`.

**OPEN**
> Whether the flat-tile false-positive risk from percentile-based grayscale normalization
> (separate from the terrain-roughness finding above) is a real problem on actual tiles, or
> only synthetic near-zero-relief inputs.
>
> Whether the real-tile clutter finding holds across all 30 pilot tiles or just the two
> inspected in detail so far (1851, 3461 quantitatively; the rest only by eye/pixel count).

**DO NOT CLAIM**
> That Arm A works on real lunar data in its current threshold form — it does not, on every
> tile inspected so far. Do not claim the phase-symmetry filter itself is broken; the
> response does show local structure near the true ridge, the threshold step is the
> better-supported culprit but not the confirmed sole one.

See [`plan.md`](plan.md) for the full plan and `research/DECISION_LOG.md` for implementation
decisions and full findings (why `phasepack` wasn't used, flank-detection, flat-tile, and
real-data clutter findings).

## Repository structure

- `research/` — spec, related-work positioning, decision log
- `src/` — the three detection arms (classical, hessian, deep) plus shared preprocessing
- `scripts/` — run/compare/report scripts
- `data/` — raw tiles and the public ridge catalog (large files gitignored, download
  scripts committed instead)
- `tests/` — correctness tests for preprocessing and each arm
- `docs/` — architecture and reproduction notes
