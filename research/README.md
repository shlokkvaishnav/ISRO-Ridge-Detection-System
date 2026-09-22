# Research index

This directory is the research. `src/`, `scripts/`, and `tests/` are the system it runs on —
infrastructure, not the contribution.

## The contract

**Research problem.** No published comparison shows which lunar wrinkle ridges a classical
phase-symmetry detector, a Hessian-eigenvalue detector, and a deep-learning detector each
catch and miss on the same real terrain — comparisons in the literature are almost always
"our method vs. the classical baseline it's meant to beat," not a genuine multi-way
failure-mode map.

**Research question.** Do classical phase-symmetry morphology, Hessian-based ridge
filtering, and deep learning agree on which lunar wrinkle ridges exist on real terrain — or
does each have a characteristic blind spot the others don't share?

**Scope.** LROC WAC + GLD100 (100m/px), Thompson et al. 2017 catalog (5,999 segments),
120-tile stratified-by-length pilot (grown from an initial 30). Chandrayaan-2 TMC-2 planned
as a cross-instrument check, not yet run.

Full contract and current findings: see the top-level [`README.md`](../README.md#status).
Prior-art positioning and what NOT to claim because someone else already owns it:
[`RELATED_WORK.md`](RELATED_WORK.md).

## Experiment index

| Investigation | Location | Type | Status |
|---|---|---|---|
| Arm A (classical phase-symmetry + morphology), synthetic validation | [`../src/classical/`](../src/classical/) | Method | Closed — passes on synthetic straight/curved/flat DEMs, 6 tests |
| Arm A on real GLD100 tiles: does the synthetic-DEM validation transfer? | See `DECISION_LOG.md`, 2026-09-22 entries | Confirmatory (exploratory framing — noticed, not pre-registered) | Closed — no. Original percentile-threshold-then-close approach: 1/14 true vertices on segment 3461. Root cause: percentile thresholding has no absolute concept of "ridge" vs terrain roughness |
| Fix: shape-filter ridge fragments before gap-linking, not after | See `DECISION_LOG.md`, 2026-09-22 | Method (exploratory — tuned against real tiles, not pre-registered) | Closed — partial improvement. Aggregate recall 6/52 → 21/52 across 6 tiles; mask still covers 30-36% of tile area. Not a solved detector |
| Arm B (Hessian ridge filters: frangi/meijering/sato) | [`../src/hessian/`](../src/hessian/) | Method | Closed — implemented, reuses Arm A's exact post-processing so the comparison isolates the filter |
| Arm A vs Arm B head-to-head on real tiles | See `DECISION_LOG.md`, 2026-09-22 | Confirmatory (exploratory framing) | Closed — close, no clear winner. Arm B (frangi) 43.9% aggregate recall vs Arm A 40.4%, per-tile wins split evenly, both in the same 28-40% mask-coverage range |
| Do Arm A and Arm B catch the same true-ridge vertices? | See `DECISION_LOG.md`, 2026-09-22 | Analysis (zero new tile extraction — reused existing masks) | Closed — no, genuinely complementary (both hit 12, A-only 11, B-only 13, neither 21 of 57). Union recall 63.2%, but mask coverage rises proportionally (~48-58%) — not a free ensemble win |
| Does NMS + hysteresis thresholding beat shape-filtering? | [`nms_hysteresis_threshold/`](nms_hysteresis_threshold/) | Method (pre-registered, [#1](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/issues/1)) | **Merged** ([PR #2](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/pull/2)) as an additional option, not a new default. Mixed result, not a clean win: aggregate looks better (42.1% vs 40.4% recall, 30.8% vs 33.2% coverage) but driven substantially by one tile, two others get strictly worse. Coverage consistently at-or-below baseline on every tile. `detect_ridges` (shape-filter) remains the default arm |
| Arm C (DBR-Net-inspired dual-branch DEM+aspect CNN) | [`arm_c_deep_learning/`](arm_c_deep_learning/) | Research (pre-registered, [#3](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/issues/3)) | **Merged** ([PR #4](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/pull/4)) — 40/57 (70.2%) recall at 30.8% coverage on the Arm A/B benchmark, beating both classical arms, trained on Kaggle GPU with the 6 benchmark tiles held out of training entirely. Real unresolved confound merged alongside the result, not hidden: training history shows overfitting past epoch ~4-10, no early stopping used, only the final checkpoint saved. Early-stopping follow-up is the natural next issue, not yet filed |
| Chandrayaan-2 TMC-2 cross-instrument check | — | — | Not started — blocked on PRADAN authenticated access, see `DECISION_LOG.md` |

**Confirmatory** means the question and comparison design were fixed before the relevant run.
**Exploratory** means the finding was noticed first and investigated after — reported as such
rather than reframed as planned. Everything above predates this project's adoption of the
full researcher/implementer/reviewer pipeline (`AGENT_PIPELINE.md`) and was run directly in
one working session — logged honestly as exploratory rather than backfilling a pre-
registration that didn't happen. Investigations filed from here forward go through the full
issue → SPEC.md → PR → review path.

## Working on this research

Branches isolate uncertainty; `master` holds only validated state. Before starting a
substantial new branch, copy [`SPEC_TEMPLATE.md`](SPEC_TEMPLATE.md) and fill in the research
question, hypothesis, and design *before* implementing or looking at results. Full policy —
branch naming, isolation rules, merge criteria, negative-results handling, commit
conventions: [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md). Why past decisions were made the way they
were: [`DECISION_LOG.md`](DECISION_LOG.md). Running this as a researcher → implementer →
reviewer pipeline, coordinated through GitHub issues/PRs/labels:
[`AGENT_PIPELINE.md`](AGENT_PIPELINE.md).
