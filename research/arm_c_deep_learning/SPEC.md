# SPEC: Arm C, DBR-Net-inspired

Copied verbatim from [issue #3](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/issues/3),
per research/AGENT_PIPELINE.md's Implementer instructions.

**Type:** research (substantial investigation, spans several experiments)

**Research question**
Can a dual-branch (DEM + aspect) CNN with attention-based feature fusion — the architecture family Lu et al. 2025 (DBR-Net, Research in Astronomy and Astrophysics, DOI 10.1088/1674-4527/ade352) validated for lunar wrinkle ridges — be adapted to this project's actual constraints (30-ish pilot tiles vs. their 1,069 hand-labeled tiles; catalog polylines as weak labels, not hand-labeled pixel masks; CPU-only training, no GPU on this machine) and produce a real, if smaller, Arm C to compare against Arm A/B using the existing true-ridge-vertex-recall/mask-coverage metric?

**Hypothesis**
A lighter dual-branch architecture (not full ResNet-34 per branch — CPU training on ~30-150 tiles cannot support that depth without overfitting), trained on weak labels (polylines buffered to an estimated ridge width), will learn *something* better than chance but will underperform DBR-Net's reported 78.4% recall / 71.6% IoU by a wide margin, primarily because of the ~10-30x smaller dataset and much weaker (line-buffer, not hand-drawn polygon) label quality — not because the architecture family is wrong for this problem.

**Null / alternative hypothesis**
The model fails to learn anything better than the shape-filter/Hessian arms' current recall (40-44%) even after correcting for the data/label gap as far as practical here — i.e. Arm C is not yet a useful addition at this data scale, and scaling the tile pipeline (which already exists and is cheap, per `scripts/build_tile_dataset.py`) needs to happen before a deep-learning arm is worth maintaining, not just training longer or tuning hyperparameters on the current 30-tile pilot.

**Motivation**
Directly answers the open question in `research/README.md` ("Arm C — Not started") and the HYPOTHESIS entries in the top-level README pointing at deep learning as the strongest remaining candidate for closing the real-tile recall gap both classical arms hit. DBR-Net (read in full, see `research/RELATED_WORK.md`) is the closest existing prior work and the right architecture family to adapt rather than inventing one from scratch — but it cannot be used directly (no released code/weights, and its data/compute assumptions don't hold here).

**Experimental design**
1. Scale the tile pilot (existing `scripts/build_tile_dataset.py`, cheap remote reads, no new infra needed) to a larger sample -- exact N to be decided based on how long remote extraction takes, logged in SPEC.md's amendments.
2. Generate weak segmentation-mask labels by buffering each catalog polyline by an estimated ridge half-width in pixels (needs its own small calibration step against a few tiles' visible ridge width).
3. Compute an aspect-data input analogous to DBR-Net's (direction of steepest slope from the DEM), reusing `src/preprocessing/dem_to_slope.py`'s existing gradient computation rather than reimplementing it.
4. Build a dual-branch (DEM slope-grayscale branch + aspect branch) small CNN with an ACFF-style fusion module (maximize -> 1x1 conv -> weighted concat, per the paper's description), sized for CPU training on this project's actual data volume -- explicitly NOT a ResNet-34 clone.
5. Train on CPU, evaluate with the same true-ridge-vertex-recall and mask-coverage metric already used for Arm A/B, on the same held-out tiles where possible for a fair three-way comparison.

**Metrics**
True-ridge-vertex recall and mask coverage (same definitions as the existing A/B comparison in `research/DECISION_LOG.md`) -- not DBR-Net's own precision/recall/F1/IoU, since our ground truth (polyline vertices) and theirs (hand-labeled polygon masks) aren't the same measurement, and reporting our number next to theirs without that caveat would misrepresent both.

**Baselines / controls**
Arm A (40.4% recall, 33.2% coverage) and Arm B (43.9% recall, ~similar coverage range) on the same 6-tile evaluation set already used throughout `DECISION_LOG.md`.

**Expected outcomes**
(a) Arm C beats both classical arms even at this reduced scale -- would be a strong, surprising result given the data/compute gap, worth extra scrutiny before trusting it. (b) Arm C underperforms both classical arms -- expected given the hypothesis above, and the useful output is characterizing *how much* data would likely be needed to close the gap, not a working detector yet. (c) Arm C is complementary (catches different vertices than A/B, like the A/B relationship itself) even while underperforming in aggregate -- would be a genuinely interesting finding in the same shape as the A/B complementarity result.

**Interpretation plan**
(a) -> scrutinize hard for a labeling/evaluation bug before believing it, then scale up data collection to validate it holds. (b) -> report the gap honestly, and treat "how many tiles would Arm C plausibly need" as the next research question rather than abandoning the direction. (c) -> run the same per-vertex overlap analysis already used for A/B, extended to three arms.

**Confounds considered**
Weak (polyline-buffer) labels are a real, different supervision signal from DBR-Net's hand-labeled masks -- any recall/IoU shortfall could be a label-quality artifact rather than a data-volume or architecture artifact, and this project's current setup cannot cleanly separate those two causes without a small hand-labeled validation subset, which does not exist yet. Flagged as an explicit limitation, not solved by this experiment.
