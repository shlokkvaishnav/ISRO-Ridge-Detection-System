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
- **2026-09-27, after review round 1 on PR #6:** added a threshold sweep, per-tile mean
  coverage, and lift over a random mask
  (`threshold_sweep.py` -> `results/arm_c_early_stopping/threshold_sweep.json`). These are
  post-hoc diagnostics requested in review. The pre-registered metric is still recall and
  coverage at the fixed 0.5 threshold.

## Results

**Provenance.** The Kaggle kernel clones the branch `experiment/arm-c-early-stopping`, not
a pinned commit. The log records only `Cloning into ...`, not a SHA. The log records only time since kernel start. The Kaggle API reports the kernel's last run
at 2026-09-22 16:22:56 UTC, and `9b37874` was committed at 16:21:38 UTC. `9b37874` was
therefore the branch head at run time, so the clone was presumably that commit. The only later
commit touches no code. The committed checkpoints re-evaluate on CPU to exactly the
committed `eval.json`/`eval_best.json`, per tile, checked independently by the reviewer and
by `threshold_sweep.py`. `kaggle_train.py` now prints the cloned `HEAD` so future logs
record it. All outputs are in `results/arm_c_early_stopping/`.

Validation loss (on the 6 held-out benchmark tiles) bottoms out at **epoch 7 (0.553)**, then
rises to 1.451 by epoch 60. The merged run's minimum was epoch 4 (0.545), rising to 1.139.
Val loss is noisy near the minimum (0.583 at ep 5, 0.553 at ep 7, 0.566 at ep 14), so which
epoch counts as "best" is itself fragile.

**Pre-registered metric (threshold 0.5):**

| Checkpoint | Recall (57 vertices) | Coverage, pixel-weighted | Coverage, mean per tile | File |
|---|---|---|---|---|
| Merged run, final (ep 60), baseline | 40/57 = 70.2% | 30.8% | 34.4% | `results/arm_c/eval.json` |
| This run, best val loss (ep 7) | 40/57 = 70.2% | 31.8% | 42.5% | `eval_best.json` |
| This run, final (ep 60) | 12/57 = 21.1% | 18.9% | 20.5% | `eval.json` |

Per-tile hits (truth in brackets), recomputed from the JSON files:

| Tile | 3674 (5) | 1851 (8) | 748 (8) | 5017 (7) | 4098 (15) | 3461 (14) | Sum |
|---|---|---|---|---|---|---|---|
| Merged final | 5 | 6 | 2 | 6 | 12 | 9 | 40 |
| This run, best (ep 7) | 5 | 8 | 4 | 7 | 11 | 5 | 40 |
| This run, final (ep 60) | 2 | 3 | 0 | 2 | 1 | 4 | 12 |

For the best checkpoint, 3 tiles (3674, 1851, 5017) are at 57–64% coverage and supply 20 of
its 40 hits.

**Threshold sweep (post-hoc).** Each cell is recall at pixel-weighted coverage, with lift
over a random mask of the same per-tile coverage in brackets. Expected random hits are
Σ truth_i × coverage_i.

| Checkpoint | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| Merged final (ep 60) | 41/57, 36.6% (1.96×) | 41, 33.6% (2.13×) | 40, 30.8% (2.26×) | 36, 27.9% (2.23×) | 26, 24.8% (1.81×) |
| This run, best (ep 7) | 56/57, 56.1% (1.66×) | 49, 43.0% (1.84×) | 40, 31.8% (1.96×) | 24, 20.2% (1.77×) | 12, 8.9% (1.87×) |
| This run, final (ep 60) | 15/57, 24.9% (1.07×) | 13, 21.6% (1.06×) | 12, 18.9% (1.12×) | 11, 16.2% (1.19×) | 9, 13.5% (1.17×) |

## Interpretation

- **Against the pre-registration, this is the null result, not outcome (a).** Outcome (a)
  required similar recall *with lower coverage*. At the pre-registered 0.5 threshold,
  recall is identical (40/57) and pixel-weighted coverage is similar (31.8% vs 30.8%). The
  hypothesis that the loss-optimal checkpoint would show lower coverage is not supported.
  Following the null/(a) action, 70.2% stays as the reported number, with the qualifiers
  below.
- **What the match does show:** about 70% recall at about 31% pixel-weighted coverage is
  reachable at the loss-optimal epoch, without training into the overfit regime.
- **What it does not show: that the two models behave alike.** The epoch-7 model's
  probabilities are bunched near 0.5. Its recall runs from 56/57 to 12/57 over thresholds
  0.3–0.7, while the merged model's runs from 41 to 26. The tie at 40 is where two curves
  with very different slopes cross at 0.5. Mean per-tile coverage is higher (42.5% vs
  34.4%), and per-tile hits differ (3461: 9 to 5, 748: 2 to 4).
- **After adjusting for coverage, the epoch-7 checkpoint is somewhat weaker than the
  merged headline model at 4 of the 5 thresholds tested** (0.3–0.6; e.g. 1.96× vs 2.26×
  lift at 0.5), and about equal at 0.7 (1.87× vs 1.81×). So early stopping does not
  simply remove an overfitting artifact from the headline model's discrimination; over
  most of the threshold range the overfit merged checkpoint discriminates slightly better.
- **The epoch-60 checkpoint varies widely between runs.** It scored 70.2% (lift 2.26×) in
  the merged run and 21.1% (lift 1.12×, about chance) in this one, with identical code.
  With n=2, neither run can be called the typical one. Training to epoch 60 without early
  stopping does not reliably give a useful model.
- **Selection confound, disclosed:** the "best" epoch is chosen by val loss on the same 6
  tiles used to score recall. That is mild selection on the benchmark (a BCE criterion,
  one epoch out of 60). Removing it needs an early-stopping split disjoint from the
  benchmark tiles.

## Decision

Implementer's self-assessment: **MERGE**. The pre-registered question is answered with raw
evidence committed and independently re-evaluated: at the fixed 0.5 threshold, the null
holds. The 70.2% headline stays, qualified. It is threshold-sensitive at the loss-optimal
epoch, somewhat weaker after adjusting for coverage, and epoch-60 training is unreliable
across runs. Findings tiers are updated in `README.md` and `research/DECISION_LOG.md`.
The early-stopping selection confound and seed variance stay OPEN as follow-up questions.
