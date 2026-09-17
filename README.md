# ISRO-Ridge-Detection-System

A lunar wrinkle ridge detection system built and evaluated across three approaches —
classical phase-symmetry morphology, Hessian-based ridge filtering, and deep learning —
compared head-to-head on the same DEM tiles to find out which ridges each one catches,
which it misses, and why.

See [`plan.md`](plan.md) for the full plan: the three arms being implemented, data sources
(LOLA DEM, LROC WAC, Chandrayaan-2 TMC-2), comparison methodology, sequencing, and
alternate approaches considered.

## Status

Planning stage — no experiments run yet. Findings will be recorded here with
ESTABLISHED / HYPOTHESIS / OPEN / DO-NOT-CLAIM tiers as they come in, same discipline as
this account's other research repos.

## Repository structure

- `research/` — spec, related-work positioning, decision log
- `src/` — the three detection arms (classical, hessian, deep) plus shared preprocessing
- `scripts/` — run/compare/report scripts
- `data/` — raw tiles and the public ridge catalog (large files gitignored, download
  scripts committed instead)
- `tests/` — correctness tests for preprocessing and each arm
- `docs/` — architecture and reproduction notes
