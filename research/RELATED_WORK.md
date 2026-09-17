# Related work and positioning

## What already exists

- **Automated LWR detection with LROC-WAC + LOLA** (Frontiers, 2023) — the reference
  classical pipeline: slope map from DEM, phase symmetry filtering, regional thresholding,
  morphological closing/opening/edge-linking. This project's Arm A reproduces this pipeline
  as a real baseline rather than citing its reported numbers secondhand.
- **Automatic detection and characterization of Lunar Wrinkle ridges** (ScienceDirect, 2020)
  — establishes the three morphology classes (parallel, isolated, concentric) used here as
  the bucketing scheme for the comparison.
- **AUTOMATED DETECTION OF LUNAR RIDGES BASED ON DEM DATA** (ISPRS, 2019) — DEM-based
  detection, methodological precedent for Arm A.
- **HL-YOLOv8** (2025/2026) — attention-augmented YOLOv8 for lunar linear tectonic features,
  motivated explicitly by classical methods missing fine-grained/weak-texture ridges. This
  project's Arm C follows this direction, but the contribution here is not the detector
  itself — it's characterizing where it agrees/disagrees with the classical arms.
- **Global mapping and analysis of lunar wrinkle ridges** — the public catalog used as
  ground truth for all three arms.

## What this project must not claim

- **"First systematic comparison of classical vs. deep-learning ridge detection"** — verify
  this claim against the full HL-YOLOv8 paper and any comparison tables it already includes
  before asserting novelty; if HL-YOLOv8 already benchmarks against a classical baseline,
  the novel part of this project is the *three-way* comparison including Hessian filters and
  the per-morphology-class/degradation-bucket failure breakdown, not the mere existence of a
  comparison.
- **"Hessian-based ridge filters have never been applied to lunar terrain"** — unverified;
  check the geomorphology literature (Frangi/Meijering/Sato are widely used in general
  terrain/DEM ridge extraction, per the kernel-pattern-modeling and plan-curvature
  literature) before claiming novelty for Arm B in isolation. The likely honest framing is
  that Hessian filters are established for terrain ridges generally, and this project's
  contribution is comparing them specifically against phase symmetry and deep learning on
  the *same* lunar tiles and catalog.
- **"This generalizes to other planetary bodies"** — untested; scope is the Moon (LOLA +
  TMC-2) unless and until Mars/Mercury data is added.

## Open verification tasks

- Read the HL-YOLOv8 paper in full to check what baselines it already compares against.
- Read at least one general (non-lunar) Hessian-ridge-on-DEM paper to confirm Arm B isn't
  itself a "first" claim.
