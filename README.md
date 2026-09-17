# ISRO-Ridge-Detection-System

A lunar wrinkle ridge detection system built and evaluated across three approaches —
classical phase-symmetry morphology, Hessian-based ridge filtering, and deep learning —
compared head-to-head on the same DEM tiles to find out which ridges each one catches,
which it misses, and why.

See [`plan.md`](plan.md) for the full plan: the three arms being implemented, data sources
(LOLA DEM, LROC WAC, Chandrayaan-2 TMC-2), comparison methodology, sequencing, and
alternate approaches considered.

## Status

**Arm A (classical phase-symmetry + morphology) implemented and passing on synthetic
ridges.** Arms B (Hessian filters) and C (deep learning) not started. No real lunar DEM
tiles used yet — everything so far is validated against synthetic ground-truth ridges.

**ESTABLISHED**
> On synthetic DEMs (straight ridge, curved ridge, flat/no-ridge control), Arm A correctly
> detects ridges only when present and tracks a curved ridge's true centerline to within a
> few pixels on average. `tests/test_classical_pipeline.py` (5 tests, all passing).

**HYPOTHESIS** — carried into the eventual arm comparison, not yet tested against B/C:
> That the "two flanks, not one crest line" output shape (see
> `research/DECISION_LOG.md`) is specific to slope-domain phase symmetry and that Arm B
> (Hessian filters, run on the same slope map) will show the same shape, while Arm C (if
> trained against a catalog's single-centerline ground truth) will not.

**OPEN**
> Whether the flat-tile false-positive risk from percentile-based grayscale normalization
> (`research/DECISION_LOG.md`) is a real problem on actual LOLA/TMC-2 tiles, or only shows
> up on synthetic near-zero-relief inputs. Untested against real data.

**DO NOT CLAIM**
> Anything about real lunar ridges yet — every result above is on synthetic data. No claim
> here has been tested against LOLA, LROC WAC, or Chandrayaan-2 TMC-2 imagery.

See [`plan.md`](plan.md) for the full plan and `research/DECISION_LOG.md` for implementation
decisions (why `phasepack` wasn't used, the flank-detection and flat-tile findings above).

## Repository structure

- `research/` — spec, related-work positioning, decision log
- `src/` — the three detection arms (classical, hessian, deep) plus shared preprocessing
- `scripts/` — run/compare/report scripts
- `data/` — raw tiles and the public ridge catalog (large files gitignored, download
  scripts committed instead)
- `tests/` — correctness tests for preprocessing and each arm
- `docs/` — architecture and reproduction notes
