# ISRO-Ridge-Detection-System

A lunar wrinkle ridge detection system built and evaluated across three approaches —
classical phase-symmetry morphology, Hessian-based ridge filtering, and deep learning —
compared head-to-head on the same DEM tiles to find out which ridges each one catches,
which it misses, and why.

See [`plan.md`](plan.md) for the full plan: the three arms being implemented, data sources
(LOLA DEM, LROC WAC, Chandrayaan-2 TMC-2), comparison methodology, sequencing, and
alternate approaches considered.

## Status

**Arms A and B both implemented; head-to-head on real GLD100 tiles they perform comparably
(~40-44% aggregate ridge-vertex recall, 28-40% mask coverage), which is the strongest
evidence yet that 100m/px terrain roughness — not either specific filter — is the real
bottleneck.** 30-tile pilot pulled directly from the remote LROC WAC + GLD100 mosaics (no
full download, see `research/DECISION_LOG.md`), stratified across the Thompson et al. 2017
ridge catalog. Arm C (deep learning) not started.

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

**HYPOTHESIS**
> That the real-tile clutter problem reflects something more fundamental than either
> specific ridge-detection filter — most likely that 100m/px GLD100 terrain roughness
> genuinely competes with real wrinkle-ridge signal at the same spatial scale, which neither
> phase symmetry nor Hessian-eigenvalue filtering can fully separate from a slope map alone.
> The A/B complementarity finding above strengthens this further: two structurally different
> filters each catch a different partial subset and still leave 37% of true vertices uncaught
> by either — consistent with a genuinely weak/ambiguous signal at this resolution, not
> either filter being tuned wrong. Strongest evidence yet for needing Arm C rather than
> continued classical-arm tuning — not proven, and still needs the per-morphology-class/
> degradation-bucket breakdown `plan.md` calls for, not just an aggregate number.
>
> That the "two flanks, not one crest line" output shape (see `research/DECISION_LOG.md`)
> is specific to slope-domain filtering and that Arm C (if trained against the catalog's
> single-centerline ground truth) will not reproduce it. Not yet tested.

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
> Whether these findings hold across all 30 pilot tiles or just the 6 inspected in detail
> so far.

**DO NOT CLAIM**
> That either arm "works" on real lunar data in a usable sense, or that Arm B is better/worse
> than Arm A — the aggregate numbers are close and the per-tile pattern is mixed. That an
> ensemble of A and B is a good practical detector — the plain union costs roughly as much in
> added mask area as it gains in recall. Do not claim the terrain-roughness hypothesis above
> is confirmed; it's the best-supported explanation so far, not a tested one.

See [`plan.md`](plan.md) for the full plan and `research/DECISION_LOG.md` for implementation
decisions and full findings (why `phasepack` wasn't used, flank-detection, flat-tile,
real-data clutter/shape-filtering, Arm A vs Arm B comparison, and A/B complementarity findings).

## Repository structure

- `research/` — spec, related-work positioning, decision log
- `src/` — the three detection arms (classical, hessian, deep) plus shared preprocessing
- `scripts/` — run/compare/report scripts
- `data/` — raw tiles and the public ridge catalog (large files gitignored, download
  scripts committed instead)
- `tests/` — correctness tests for preprocessing and each arm
- `docs/` — architecture and reproduction notes
