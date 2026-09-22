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

---

## Amendments (dated notes, per research/GIT_WORKFLOW.md's spec discipline)

**2026-09-22 -- infrastructure incidents en route, before any training result:**

- `build_tile_dataset.py` originally overwrote `manifest.json` wholesale on every
  run. Caught before it destroyed the original 30-tile reference set (stopped a
  run mid-execution once noticed); fixed to be additive/idempotent.
- The same script then only wrote `manifest.json` once, at the end of the whole
  extraction loop. Stopping a run partway through (to reconsider the target
  tile count given how slow serial extraction was) lost 11 tiles' worth of
  manifest entries even though the actual GeoTIFF files were already written
  to disk -- real network cost paid, not reflected in the manifest. Recovered
  10 of 11 manually (the 11th was mid-write, discarded); fixed to write after
  every tile.
- Serial extraction was too slow to reach a meaningful tile count in
  reasonable time (non-tiled source rasters mean large-bbox segments require
  many scattered `/vsicurl/` strip reads). Parallelized with a thread pool
  (I/O-bound, not CPU-bound) -- real speedup, not parallel-for-its-own-sake.
  Scaled the pilot 30 -> 120 tiles this way (still far short of DBR-Net's
  1,069 hand-labeled tiles, see the confounds section above).
- The GitHub repo was found to be private when the Kaggle training kernel's
  anonymous `git clone` failed with an auth error -- unclear whether it was
  switched from public at some point or a state neither party fully tracked.
  Asked the user rather than assuming either direction; they confirmed
  restoring it to public (its original state) was correct, done via
  `gh repo edit --visibility public`.

None of these affect the experiment's actual hypothesis or design -- they're
tooling/infrastructure issues encountered while building toward the first
training run, recorded here because research/GIT_WORKFLOW.md's spec
discipline says amendments get dated notes, not silently folded into a
rewritten history.

- Manifest paths were written with `os.path.relpath` on this Windows dev
  machine, producing backslash-separated strings. Training on Kaggle
  (Linux) failed with `RasterioIOError` because Linux doesn't treat `\` as
  a path separator -- fixed at the source (explicit forward-slash paths)
  and repaired all 120 existing entries; verified Arm A still reproduces
  its on-record number after the fix.
- The dataset mount path assumption (`/kaggle/input/<slug>/`) was wrong for
  this account -- actual layout nests under
  `/kaggle/input/datasets/<owner>/<slug>/`. Fixed by searching for the
  directory containing `manifest.json` instead of hardcoding either layout.

## Results

**First run (random 20% val split, before the holdout fix above):** 129/198
(65.2%) recall, 25.6% coverage. Preserved in `results/arm_c_random_split/`
but **not used as the headline number** -- only 1 of the 6 tiles Arm A/B
were compared on happened to land in this random val split; the other 5
were in Arm C's training set, so this number isn't a fair three-way
comparison. Kept for the record, not cited as the result.

**Holdout run (the 6 Arm A/B benchmark tiles excluded from training
entirely, per the `holdout_ids` fix):**

| Arm | Recall (57 vertices, 6 tiles) | Coverage |
|---|---|---|
| A (phase symmetry, shape-filtered) | 23/57 = 40.4% | 33.2% |
| B (Hessian/frangi) | 25/57 = 43.9% | ~similar range |
| **C (this run)** | **40/57 = 70.2%** | **30.8%** |

Per-tile: 3674 (5/5), 1851 (6/8), 748 (2/8), 5017 (6/7), 4098 (12/15), 3461
(9/14) -- recall independently recomputed from the raw per-tile numbers
(`5+6+2+6+12+9=40`, `5+8+8+7+15+14=57`), matches exactly.

**A real confound, not yet resolved: the training history shows clear
overfitting past epoch ~4-10.** Validation loss (on the held-out 6 tiles)
reaches its minimum at epoch 4 (0.545), then rises to 1.1-3.3 by epoch 60
while train loss keeps falling (0.67 -> 0.16). No early stopping was used;
`model.pt` is the final (epoch 60), not best-val-loss, checkpoint, and only
the final checkpoint was saved -- the epoch-4 checkpoint's recall/coverage
is unknown and cannot be recovered without retraining. This does not
automatically invalidate the 70.2% recall number: BCE loss measures
per-pixel probability calibration, which is a different objective from
threshold-then-count-vertices recall, and coverage staying at a reasonable
30.8% (not saturating toward marking the whole tile positive) argues
against the crudest overfitting failure mode. But an overfit-loss-curve
model producing a good recall number on the same 6 tiles used to tune
`epochs`/architecture choices earlier in this SPEC is exactly the kind of
result that needs independent scrutiny before being trusted as a clean
win, per this SPEC's own interpretation plan ("(a) -> scrutinize hard for a
labeling/evaluation bug before believing it").

**What this establishes:** Arm C, in this specific run, substantially
outperforms both classical arms on the exact same 6-tile benchmark, using
comparable-to-lower mask coverage than Arm A. The result is reproducible
in the sense that the arithmetic checks out and the metric has no leakage
(evaluate.py scores against the real catalog polylines directly, never
touching the weak-label buffers used for training).

**What this does NOT establish:** That this specific recall number is
stable/repeatable -- single run, single seed, no repeated-seed variance
reported (unlike the classical arms' 6-tile comparisons, which were at
least deterministic given fixed algorithms). Whether early stopping at the
loss-based optimum would give a similar, better, or worse recall number --
untested. Whether the result holds with a larger/more tiles, given the
persistent gap to DBR-Net's 1,069-tile training set. Whether the weak-label
supervision (vs. DBR-Net's hand-labeled masks) is doing something that
happens to correlate well with this specific evaluation metric in a way
that wouldn't hold for a genuinely independent, hand-labeled test set --
still an open, undissolved confound from this SPEC's original confounds
section.

**Recommended immediate follow-up, not done in this pass:** rerun with
early stopping (save the best-val-loss checkpoint, not just the final one)
and compare its recall/coverage against this run's 70.2%/30.8% -- the
single most direct way to check whether the overfitting confound above
actually matters for this metric or not.
