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
