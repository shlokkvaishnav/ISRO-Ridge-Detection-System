# Git workflow for this research

`master` is the validated research state, not just working code. Branches are where
uncertainty gets explored. This document says how those two things stay separate on purpose.

A branch is not merged because it works or produces a better number. It is merged when it is
validated, reproducible, documented, and useful to the research — and a branch that
disproves a hypothesis, rules out a mechanism, or produces a negative result can satisfy all
four of those without the detector improving at all.

## Branch types

| Prefix | For | Example |
|---|---|---|
| `research/<topic>` | A substantial research investigation, usually spanning several experiments | `research/deep-learning-arm` |
| `experiment/<name>` | One specific, narrowly-scoped experiment | `experiment/meijering-vs-frangi` |
| `analysis/<name>` | Analysis of an existing result — no new data collection | `analysis/ab-vertex-overlap` |
| `method/<name>` | A new methodological component (a filter, a metric, a threshold scheme) | `method/shape-filtered-threshold` |
| `reproduction/<target>` | Reproducing an external paper or published method | `reproduction/hl-yolov8` |

Don't create a branch for a cosmetic or purely-editorial change — those go straight to
`master` via normal review. Do create one whenever a change could affect an experimental
conclusion (see **Isolation**, below).

**SETUP GOES STRAIGHT TO `master`.** No branch, no PR, no review round. *Setup* is the
machinery rather than the findings: tooling and checkers (`research/check_index.py` and its
tests), CI config, this document and `AGENT_PIPELINE.md`, hooks, and data-acquisition
scaffolding that produces no claim. **Research still takes the full path** — spec, branch,
PR, review, manual merge.

**If a change touches both setup and a research claim, it is research.** Branch it.

## Before writing code: the spec

A substantial branch (`research/*`, most `experiment/*`) starts with a filled-out copy of
[`SPEC_TEMPLATE.md`](SPEC_TEMPLATE.md), committed before the implementation that answers it.
Committing the spec first timestamps the hypothesis — a hypothesis written after seeing the
result is not a hypothesis, it's a caption. If a branch turns exploratory partway through (an
unexpected observation redirects it), say so explicitly in the spec and in the final
writeup; do not backfill a clean hypothesis once the answer is known.

A small `experiment/*` branch (e.g. re-running an existing tile set at a different
threshold parameter) can skip the full template and state the one-line question + expected
outcome in the first commit message instead — use judgement, but when in doubt, write the
spec.

## Isolation

If a change touches any of: which arm's filter is used, the threshold/morphology
parameters, the DEM tile set, the ridge catalog subset, the evaluation metric, or the
comparison methodology — put it on its own branch. Mixing two of these in one branch makes
"what caused the change" unanswerable, which defeats the point of running the experiment at
all. This project already has one concrete example: the 2026-09-22 thresholding fix touched
both the threshold percentile *and* the fragment-ordering logic in one pass — acceptable
there because it was setup-stage tuning before this workflow existed, not something to repeat
now that it does.

**Formatting and research must never share a branch.** Process, repo-hygiene, and
documentation-structure changes go straight to `master`; research goes through a PR. The two
must not ride together — separating them afterward is manual surgery.

The test to apply before committing: *would this line belong in the PR description of an
experiment?* A README typo fix would not.

## Lifecycle

1. Research question → 2. Literature check → 3. Hypothesis → 4. Experimental design →
5. Implementation → 6. Validation → 7. Experiment → 8. Analysis → 9. Interpretation →
10. **Decision**

The decision is one of:

- **MERGE** — sufficiently validated; becomes part of the main research codebase.
- **ARCHIVE** — scientifically useful, doesn't belong in the main implementation. Keep the
  branch (or a documented summary + the branch ref) rather than deleting it.
- **REVISE** — promising, but the experiment or implementation needs another pass.
- **ABANDON** — the question or approach is no longer useful. Still not deleted without
  inspection — see below.
- **REPRODUCE** — should be repeated under better controls before a decision can be made.

"Not merged" is not a synonym for "failed." A branch that cleanly rules out an approach
(e.g. confirms a filter's clutter problem is fundamental, not tunable) is a successful branch
that ends in ARCHIVE, not a failed one.

## Merge criteria

Before merging into `master`, check all nine — not every one needs to be perfect, but any
real weakness has to be written down, not glossed over:

**Scientific relevance** (addresses an approved research question) · **Correctness**
(implementation does what it claims) · **Experimental validity** (controls/baselines/metrics
are appropriate — e.g. real tiles vs synthetic, matched threshold parameters across arms) ·
**Reproducibility** (another researcher could rerun it) · **Documentation** (purpose and
methodology are written down) · **Interpretation** (we know what the result does and doesn't
establish) · **Research integrity** (negative results and limitations are honestly recorded,
not smoothed over) · **Integration** (doesn't make the codebase harder to understand without
justification) · **Evidence** (enough of it to justify moving from "branch" to "validated
state")

A positive result is not a merge criterion. A negative result is not a merge blocker — see
this project's own `research/DECISION_LOG.md` entries on the original thresholding failure
and the union-isn't-free finding, both real, both merged as findings.

## When *not* to merge

Irreproducible. Unstable implementation. Fundamentally flawed methodology. Result depends on
an uncontrolled confound. Multiple important variables changed at once with no way to
attribute the effect. Result uninterpretable. Substantial technical debt with no research
justification. Purely exploratory with no validated conclusion yet. Contradicts the
established methodology without an approved reason. Based on a cherry-picked tile subset.
Makes an unsupported scientific claim.

Any of these → archive the branch and write down what was learned instead of merging or
deleting.

## Negative results

Don't hide them. A hypothesis being false, a filter not separating ridge from clutter, an
expected recall gain not appearing, a threshold scheme failing under real terrain — these are
findings, and they get recorded the same way a positive result would. A negative-result
branch merges into `master` only if the *implementation itself* becomes validated
infrastructure. Otherwise the branch and its writeup are preserved, not deleted, and the
conclusion is what travels forward.

## Commit messages

Prefix by what changed, and say so explicitly when a commit changes methodology, not just
implementation:

```
research: add deep-learning arm scaffolding
experiment: sweep threshold percentile 60-90
analysis: compute A/B per-vertex overlap
method: add meijering ridge filter
fix: correct CRS transform for GLD100 tiles
docs: document real-tile validation protocol
```

Not: `stuff`, `changes`, `final`, `final-final`, `working`, `update`, `fixed`.

## Mini peer review before merging

For any substantial branch, answer these before merging — in the PR description, not just in
your head:

What question did this branch answer? What was the hypothesis? What evidence was collected?
What does the result actually establish — and what does it *not* establish? What confounds
remain? What assumptions were made? Could another researcher reproduce it? Did the
implementation introduce any unintended changes elsewhere? Does this change the research
thesis in `README.md`? Merge / archive / revise / abandon / reproduce?

## Decision log

Significant research decisions — why an experiment was designed a certain way, why a
threshold was chosen, why a hypothesis was rejected, why a branch was merged or archived —
are recorded in [`DECISION_LOG.md`](DECISION_LOG.md). The repository should not depend on
anyone's memory of why a call was made.

## What this is not

This is not a tool for making the repository look cleaner. Failed experiments, wrong
hypotheses, bugs, dead ends, negative results, and abandoned methods are expected to exist in
the history and on preserved branches — that *is* the research record. History does not get
rewritten to make the project look more linear than it was, and a branch is not deleted
without first checking whether it holds unique work (`git log <branch> --oneline`,
`git diff master...<branch>`).
