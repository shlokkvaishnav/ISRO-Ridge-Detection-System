# Plan — ISRO-Ridge-Detection-System

## Research question

Do the three main approaches to detecting lunar wrinkle ridges — classical phase-symmetry
morphology, Hessian-based ridge filtering, and deep learning — actually agree on which
ridges exist, or does each one have a characteristic blind spot that the others don't
share? Specifically: **does each method fail on a different, characterizable subclass of
ridge (by morphology type — parallel / isolated / concentric — and by degradation
severity), such that no single method is a strict superset of the others?**

This is not "build a ridge detector." It's "do the three published/standard ways of doing
this actually catch the same ridges, and if not, which fails where, and why." Answered
honestly, with negative/inconclusive results kept, not hidden.

## Why this matters

The literature already tells us classical phase-symmetry morphology struggles on eroded/
degraded ridges (that's explicitly why deep learning approaches like HL-YOLOv8 were built).
But nobody publishing a deep-learning detector systematically shows *what a Hessian-based
ridge filter would have caught that the deep model missed*, or vice versa — comparisons in
the literature are almost always "our method vs. the classical baseline it's meant to beat,"
not a genuine three-way failure-mode map. That gap is the actual contribution here.

## Data

- **DEM**: LRO LOLA (public, NASA PDS / USGS Astrogeology) — primary elevation source used
  in the reference literature (Frontiers 2023 automated LWR paper, ISPRS 2019).
- **Optical**: LROC WAC imagery, co-registered with the DEM, for the deep-learning arm and
  for visual sanity-checking.
- **ISRO angle**: Chandrayaan-2 TMC-2 (Terrain Mapping Camera) DEM as a second, independently-
  sourced elevation dataset — useful both for your ISRO framing and as a genuine cross-
  instrument robustness check (does a method's failure mode replicate across two different
  DEM sources, or is it an artifact of LOLA's specific noise characteristics?).
- **Ground truth**: existing published Lunar Wrinkle Ridge global-mapping catalogs (e.g. the
  Global Mapping and Analysis of Lunar Wrinkle Ridges dataset) as the labeled reference set —
  don't hand-annotate from scratch; a vetted public catalog exists and using it is more
  defensible than a self-made ground truth.

## Three arms, evaluated on the same ridges

### Arm A — Classical: slope map + phase symmetry + morphology (the literature baseline)
1. DEM → slope map (grayscale)
2. Phase symmetry filter (tunable wavelength/scale) to get illumination-invariant ridge
   candidates
3. Regional thresholding
4. Morphological closing (bridge gaps in a fragmented ridge line), opening (remove noise
   blobs), edge linking (connect fragments into continuous ridge lines)

This reproduces the field-standard pipeline (Frontiers 2023 / ISPRS 2019) — it's the
reference arm the other two are measured against, not a novel contribution on its own.

### Arm B — Hessian-based ridge filters (Frangi / Meijering / Sato)
1. Same slope-map input as Arm A, for a fair comparison
2. Apply `skimage.filters.frangi` / `meijering` / `sato` — Hessian-eigenvalue-based ridge
   detectors, same family used for vessel detection in medical imaging
3. Compare directly against phase symmetry on the *same* preprocessing — isolates whether
   the ridge-detection filter itself, not the preprocessing pipeline, is where methods
   diverge

This arm is cheap to run (all in scikit-image already) and is the natural first thing to
build — no training required, just apply and compare against Arm A.

### Arm C — Deep learning (YOLOv8-family or a segmentation model)
1. Fine-tune an attention-augmented YOLOv8-style detector (per HL-YOLOv8 in the recent
   literature) or a segmentation model (U-Net-style) on DEM/slope-map tiles, using the
   public ridge catalog as labels
2. This is the arm most likely to catch the fine-grained, weak-texture ridges that both
   classical methods (A and B) miss — per the literature's own stated motivation for
   building it
3. Also the arm to interrogate hardest for false positives — a model trained to be
   sensitive to faint ridges may hallucinate ridge-like structures where there are none

## Comparison methodology

- Run all three arms on the same tiles, against the same public ridge catalog as ground
  truth.
- For every ridge in the catalog, record per-arm: detected / missed, and if detected,
  positional/length accuracy.
- Bucket every ridge by its published morphology class (parallel / isolated / concentric)
  and by a degradation proxy (ridge height/elevation-offset from the catalog, as a rough
  proxy for erosion severity).
- The actual finding to report: a confusion-style matrix of {method} x {ridge class x
  degradation bucket}, showing where each method's detection rate drops — not a single
  aggregate F1/IoU number, since (per the `Replica-Recall-Divergence` lesson) an aggregate
  metric is exactly what hides a method-specific blind spot.
- Keep an honest findings section in the README with ESTABLISHED / HYPOTHESIS / OPEN /
  DO-NOT-CLAIM tiers, same discipline as the other repos in this portfolio — including
  publishing negative or inconclusive comparisons, not just the flattering ones.

## Sequence

1. **Arm B first** (Hessian filters) — cheapest, no training, immediately comparable to the
   literature's Arm A description. Gets a real result on the board fastest.
2. **Arm A** (reproduce the classical phase-symmetry pipeline) — needed as the actual
   baseline, not just described from the papers. Do this concretely so the comparison is
   real, not assumed from reading someone else's numbers.
3. **Cross-instrument check**: rerun A and B on the Chandrayaan-2 TMC-2 DEM tile(s) to see
   whether either method's failure pattern is DEM-source-specific.
4. **Arm C** (deep learning) last — highest setup cost (needs labeled tiles curated from
   the public catalog, training compute, and the most engineering), and by this point you
   already know from A/B where the interesting gaps are, so training/evaluation can target
   those specific failure regions instead of a blind full-map run.
5. **Comparison report**: the confusion-style matrix and honest findings writeup.

## Alternate approaches considered (and why not chosen as primary)

- **Pure gradient/edge detection (Sobel/Canny) as a fourth arm**: rejected as a serious
  contender — the literature is explicit that gradient-based edges fail on the exact
  ridges this project cares about (faint, degraded). Worth one quick sanity-check run to
  document *why* it's excluded, not worth a full arm.
- **SIFT-based DEM reconstruction**: relevant to *building* a DEM from stereo imagery, not
  to detecting ridges from an existing DEM — out of scope unless the ISRO angle later
  needs a from-scratch DEM rather than the public LOLA/TMC-2 products.
- **Full 3-way ensemble/voting detector as the deliverable**: tempting, but that's a product,
  not a research question — building "the best possible detector" isn't the goal here; the
  goal is characterizing where each method disagrees and why. An ensemble could be a
  follow-up once the failure-mode map exists.

## Repository structure (mirrors the existing portfolio pattern)

```
research/
  SPEC.md              per-experiment spec, filled before implementation (question, metric,
                        interpretation, pre-registered before running)
  RELATED_WORK.md       positioning against the Frontiers/ISPRS/HL-YOLOv8 papers, and what
                        this project must not claim that they already own
  DECISION_LOG.md       why things are the way they are, newest first
src/                    the three arms: classical/, hessian/, deep/, plus shared preprocessing
                        (dem_to_slope, tiling, catalog loading)
scripts/                run_arm.py, compare_arms.py, build_confusion_matrix.py
data/                   raw DEM/WAC tiles (gitignored — large binary files, documented
                        download script instead) and the public ridge catalog
tests/                  correctness tests for preprocessing and each arm's wrapper
docs/                   architecture notes, how to reproduce
```

## Next step

Start with Arm B (Hessian filters via scikit-image) against a small hand-picked set of DEM
tiles with known catalog ridges, just to get a real first comparison on the board before
committing to the full pipeline.
