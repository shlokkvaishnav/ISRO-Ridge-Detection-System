# SPEC: Does Arm C hold at the loss-optimal checkpoint?

Copied verbatim from [issue #5](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/issues/5),
per research/AGENT_PIPELINE.md's Implementer instructions.

**Type:** experiment (one specific, narrowly-scoped experiment)

**Research question**
Arm C's headline 40/57 (70.2%) recall / 30.8% coverage result (research/arm_c_deep_learning/SPEC.md, research/DECISION_LOG.md) came from the final (epoch 60) checkpoint of a training run whose validation loss was actually best at epoch 4 and got 2-6x worse by the end. Does the loss-optimal checkpoint give a similar, better, or worse recall/coverage on the same 6-tile benchmark?

**Hypothesis**
The loss-optimal checkpoint will show lower coverage (less confident/aggressive predictions, since it hasn't overfit toward the training weak-labels' specific shape) and plausibly lower recall too, since the current 70.2% may be partly propped up by the model becoming overconfident on shapes similar to training-set weak labels. If so, the "true" generalizable recall is somewhere between the classical arms' ~40-44% and the current overfit number's 70.2%, not the full 70.2%.

**Null / alternative hypothesis**
The loss-optimal checkpoint's recall/coverage is statistically indistinguishable from the final checkpoint's (within the ~57-vertex sample's inherent noise) -- would mean the overfitting-by-loss doesn't meaningfully affect this particular metric, and the current headline number can be trusted more or less as-is.

**Motivation**
This is the single most direct, already-identified open question about Arm C's own headline result (research/DECISION_LOG.md's HYPOTHESIS section names it explicitly). Answering it either strengthens the existing 70.2% claim considerably or substantially revises it -- both outcomes matter to how Arm C gets described in any future write-up of this project.

**Experimental design**
Retrain with the exact same setup as the merged run (src/deep/train.py, same holdout_ids, same architecture/hyperparameters) but add checkpoint saving at every epoch (or at minimum whenever val_loss improves), not just the final epoch. Evaluate the best-val-loss checkpoint with the existing src/deep/evaluate.py against the same 6-tile benchmark, using the same metric already on record.

**Metrics**
True-ridge-vertex recall and mask coverage (same definitions as every prior arm comparison) at the loss-optimal checkpoint, compared directly against the already-committed 70.2%/30.8% final-checkpoint numbers in results/arm_c/eval.json.

**Baselines / controls**
The already-merged final-checkpoint result (results/arm_c/eval.json) is the baseline this is compared against -- no need to rerun that arm, it's already committed.

**Expected outcomes**
(a) Loss-optimal checkpoint recall is similar or better, with lower coverage -- strengthens the existing claim, and the loss-optimal checkpoint becomes the new default artifact. (b) Loss-optimal checkpoint recall is meaningfully worse -- the honest recall number for Arm C is lower than currently reported, requiring an update to README.md/DECISION_LOG.md's findings tiers. (c) Recall is similar but coverage differs substantially either direction -- a real but more nuanced finding about what continued training past the loss optimum actually does to this model.

**Interpretation plan**
(a) -> update DECISION_LOG.md to note the confound is resolved favorably, keep 70.2% as the reported number but note it's now corroborated. (b) -> revise the headline claim down, this becomes the number actually reported going forward, and the current 70.2% gets an explicit "superseded" note per research/GIT_WORKFLOW.md's retired-claims discipline. (c) -> report both numbers with their respective coverage tradeoffs, similar framing to the NMS+hysteresis result.

**Confounds considered**
Random seed/initialization is not controlled between this run and the original merged run beyond what src/deep/train.py already does (documented as an open reproducibility gap in the merged PR) -- any difference found could be seed variance, not purely an early-stopping effect. Ideally this experiment would also report a second same-seed rerun of the ORIGINAL (final-checkpoint) setup to establish how much of any difference is attributable to seed noise vs. the early-stopping change itself, but that doubles the Kaggle compute cost for this single experiment -- flagged as a scope decision to make explicitly, not silently skipped.

## Amendments

- **2026-09-22, design change before the run:** instead of a separate retrain compared
  against the merged run, `src/deep/train.py` now tracks the best-val-loss epoch and saves
  `model_best.pt` alongside the final `model.pt` **within one training run**, and
  `kaggle_train.py` evaluates both. Both checkpoints then share one seed and one trajectory,
  so the best-vs-final comparison is free of the seed-variance confound the issue flagged.
  Same architecture, hyperparameters, 60 epochs and the same 6 held-out benchmark tiles
  (`val_ids.json` is identical to `results/arm_c/val_ids.json`).

## Results

Kaggle GPU run of commit `9b37874` (the kernel clones this branch; the clone's HEAD is
recorded in `results/arm_c_early_stopping/kaggle_run.log`). All outputs are in
`results/arm_c_early_stopping/`.

Validation loss (on the 6 held-out benchmark tiles) bottoms out at **epoch 7 (0.553)**, then
rises to 1.451 by epoch 60. The merged run's minimum was epoch 4 (0.545), rising to 1.139,
so the two trajectories are similar in shape.

| Checkpoint | Recall (57 vertices, 6 tiles) | Coverage | File |
|---|---|---|---|
| Merged run, final (epoch 60), earlier run, baseline | 40/57 = 70.2% | 30.8% | `results/arm_c/eval.json` |
| **This run, best val loss (epoch 7)** | **40/57 = 70.2%** | **31.8%** | `eval_best.json` |
| This run, final (epoch 60) | 12/57 = 21.1% | 18.9% | `eval.json` |

Per-tile hits (truth in brackets), recomputed by hand from the JSON files:

| Tile | 3674 (5) | 1851 (8) | 748 (8) | 5017 (7) | 4098 (15) | 3461 (14) | Sum |
|---|---|---|---|---|---|---|---|
| Merged final | 5 | 6 | 2 | 6 | 12 | 9 | 40 |
| This run, best (ep 7) | 5 | 8 | 4 | 7 | 11 | 5 | 40 |
| This run, final (ep 60) | 2 | 3 | 0 | 2 | 1 | 4 | 12 |

## Interpretation

- **The loss-optimal checkpoint matches the headline recall** (40/57 vs 40/57), with
  coverage essentially unchanged (31.8% vs 30.8%). This is outcome (a) from the issue:
  similar recall, and the pre-registered hypothesis that coverage would be *lower* is
  **not supported**.
- **The overfit final checkpoint is not reliable.** Within one run, going from epoch 7 to
  epoch 60 cut recall from 70.2% to 21.1%. The merged run's epoch-60 checkpoint scored
  70.2%, so the epoch-60 result swings from 21% to 70% between two runs of identical code.
  The merged headline came from an unstable checkpoint that happened to land well. The
  number itself survives, but only because the loss-optimal checkpoint reproduces it.
- **The aggregate match partly hides per-tile differences.** Tile 3461 drops from 9/14 to
  5/14 and 748 rises from 2/8 to 4/8. With 57 vertices, a tie at 40 should be read as
  "about the same level", not as an exact replication.
- **New confound, disclosed rather than hidden:** the "best" epoch is chosen by val loss
  on the same 6 tiles used to score recall. That is selection on the benchmark, so the
  best-checkpoint number is mildly optimistic. It is mild because selection uses BCE loss,
  not recall, and picks one of 60 epochs. A clean fix needs a separate validation split for
  early stopping, disjoint from the 6 benchmark tiles.

## Decision

Implementer's self-assessment: **MERGE**. The question is answered: the loss-optimal
checkpoint gives 70.2% recall at 31.8% coverage, and it replaces the final checkpoint as
the reported Arm C artifact (`results/arm_c_early_stopping/model_best.pt`). Findings tiers
are updated in `README.md` and `research/DECISION_LOG.md`. The early-stopping selection
confound and seed variance of the best checkpoint are left open as follow-up questions,
not claimed as resolved.
