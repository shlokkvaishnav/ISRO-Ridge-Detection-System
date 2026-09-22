<!--
Fill this in before requesting review. It mirrors research/GIT_WORKFLOW.md's
"mini peer review before merging" section — the reviewer session checks your
answers against research/GIT_WORKFLOW.md's actual merge criteria, not just
whether the code works. Don't merge this yourself; see the human-merge gate
in research/AGENT_PIPELINE.md.
-->

Closes #<issue number>

## What question did this branch answer?

## What was the hypothesis?

(Copy from the issue/SPEC.md if unchanged. If the work became exploratory partway through, say so here explicitly rather than presenting a post-hoc hypothesis as the original one.)

## What evidence was collected?

(Link to raw results / commits, not just a summary. Recall/coverage numbers should be reproducible from the tile manifest and ground-truth catalog directly, not only from this branch's own analysis script.)

## What does the result actually establish?

## What does it explicitly NOT establish?

## What confounds remain?

## What assumptions were made?

## Could another researcher reproduce this?

(Yes/no, and what's missing if not.)

## If this adds a `research/<name>/` directory, is it in the experiment index?

(`research/README.md`'s table.)

## Did the implementation introduce any unintended changes elsewhere?

(Check: existing arm parameters, the tile manifest, other branches' conclusions — see `research/GIT_WORKFLOW.md`'s isolation rule.)

## Does this change the research thesis in the top-level README.md?

## Implementer's self-assessed decision

MERGE / ARCHIVE / REVISE / ABANDON / REPRODUCE — and why, against `research/GIT_WORKFLOW.md`'s merge criteria. The reviewer will confirm or override this, not rubber-stamp it.

---

<!-- Reviewer: post your own MERGE/ARCHIVE/REVISE/ABANDON/REPRODUCE recommendation as a PR comment, not here. You review and recommend; the user merges. -->
