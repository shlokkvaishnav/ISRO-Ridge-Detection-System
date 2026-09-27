# SPEC: Does Arm C catch the 21 benchmark vertices both classical arms miss, or share their blind spot?

Copied verbatim from [issue #7](https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System/issues/7),
per research/AGENT_PIPELINE.md's Implementer instructions.

**Type:** analysis (analysis of an existing result, no new data collection)

**Research question**
On the 6-tile, 57-vertex benchmark, 21 true-ridge vertices are missed by both classical arms (`research/DECISION_LOG.md`, 2026-09-22 complementarity entry). How many of those 21 does Arm C catch? After adjusting for Arm C's per-tile mask coverage, is its hit rate on them different from its hit rate on the 36 vertices at least one classical arm already catches?

**Hypothesis**
Arm C catches a meaningful share of the 21 shared-miss vertices, clearly above what a random mask of equal per-tile coverage would catch. Its hit rate on them is still lower than on the 36 classically-caught vertices. Why: Arm C's aggregate gain (40/57, against 23/57 and 25/57) is too large to come only from re-finding the 36 classically-caught vertices. The per-tile counts on 3674 (C 5/5, A 2, B 3) and 4098 (C 12/15, A 6, B 5) already imply that some shared-miss vertices are caught. But README.md's terrain-roughness HYPOTHESIS predicts that some ridge signal at 100m/px is weak for any method, so a residual set that all three arms miss is expected.

**Null / alternative hypothesis**
Arm C's hits on the 21 shared-miss vertices are about what a coverage-matched random mask would give. That expectation is the sum, over those vertices, of Arm C's coverage fraction on each vertex's tile: about 6–7 of 21 at ~31% coverage. Arm C's advantage would then come almost entirely from the 36 vertices the classical arms already reach, and all three method families would share the blind spot. The opposite would also contradict the hypothesis: if Arm C's hit rate on the 21 shared-miss vertices is as high as or higher than on the 36 caught ones, it has no measurable blind spot in common with the classical arms on this benchmark.

**Motivation**
This is the repo's central research question: does each method family have its own blind spot, or do they share one? `research/RELATED_WORK.md`'s novelty verdict names per-instance attribution across method families as the project's load-bearing, unclaimed contribution. That attribution exists only pairwise so far (A vs B). Arm C has only aggregate and per-tile numbers, so the three-way map the project is built around has never been computed. Either answer changes the README:
- If Arm C reaches what the classical filters cannot, that counts against the terrain-roughness hypothesis as a limit on every method family.
- If there is a residual set all three arms miss, that directly supports the hypothesis. It also names the specific vertices a future per-morphology breakdown should look at first.

No training is needed. The analysis uses the committed checkpoints (`results/arm_c/model.pt`, `results/arm_c_early_stopping/model_best.pt`), the existing evaluator's metric, and the deterministic Arm A/B pipelines. It runs on the 6 benchmark tiles that `scripts/build_tile_dataset.py` already produces.

**Experimental design**
- **Tiles:** the 6 benchmark tiles only (3674, 1851, 748, 5017, 4098, 3461; `results/arm_c/val_ids.json`), all held out of Arm C training. The tiles are gitignored; if they are not already local, regenerate these 6 with the existing script. No other tile extraction.
- **Arms:**
  - Arm A: the default `detect_ridges` (shape-filter), with the same parameters as the 2026-09-22 benchmark.
  - Arm B: frangi, with the same post-processing.
  - Arm C: `results/arm_c/model.pt` (the merged headline checkpoint) is primary, and `results/arm_c_early_stopping/model_best.pt` (epoch 7) is secondary. Both run at the pre-registered 0.5 threshold, with the same resize-to-native procedure as `src/deep/evaluate.py`.
- **Per-vertex table:** for each of the 57 truth vertices, record hit/miss for A, B and C, plus the tile and each arm's per-tile coverage. A hit uses the existing rule: `mask[r, c]` at the vertex's reprojected native pixel.
- **Validation gate, before any Arm C result is read:**
  - The regenerated A/B per-vertex table must reproduce the recorded 12 / 11 / 13 / 21 split (both / A-only / B-only / neither) and the per-tile A/B hits in DECISION_LOG.md.
  - The Arm C per-tile hits must exactly reproduce `results/arm_c/eval.json` and `results/arm_c_early_stopping/eval_best.json`.
  - If either check fails, stop and report the discrepancy. Do not continue on a baseline that does not reproduce.
- **Held constant:** tiles, truth vertices, reprojection code and thresholds. Nothing is tuned.

**Metrics**
Deciding metrics, on the primary checkpoint (`results/arm_c/model.pt`):
1. Arm C hits among the 21 vertices both classical arms miss, compared with the coverage-matched chance expectation: the sum over those 21 vertices of Arm C's coverage fraction on each vertex's tile. Report an exact one-sided p-value from a Poisson-binomial with those per-vertex probabilities.
2. Arm C's hit rate on those 21 vertices compared with its hit rate on the 36 vertices at least one classical arm catches. Express each rate as lift over its own coverage-matched expectation.

Descriptive only, not deciding:
- the full 8-cell A×B×C per-vertex table
- the list of vertices all three arms miss, with tile IDs
- the same table for `model_best.pt`
- a per-tile breakdown of everything above

**Baselines / controls**
- A coverage-matched random mask per tile. This is the same analytic expectation as `random_mask_expected_hits` in `results/arm_c_early_stopping/threshold_sweep.json`, applied to each vertex subset rather than to the tile total.
- Arm C's own hit rate on the 36 classically-caught vertices is the internal control for harder versus easier vertices.
- No synthetic-DEM baseline is needed. The question is about which real-tile instances each arm catches, and each arm's behaviour on synthetic DEMs is already established.

**Expected outcomes**
(a) C catches the shared-miss set well above chance (p < 0.05 on metric 1), with a lift roughly equal to its lift on the 36 caught vertices. Arm C does not share the classical blind spot on this benchmark.
(b) C catches the shared-miss set above chance but with a clearly lower lift than on the caught set, and a non-trivial set of vertices all three arms miss remains. The blind spot is partly shared (the stated hypothesis).
(c) C's hits on the shared-miss set cannot be told apart from chance (p ≥ 0.05). All three method families share the blind spot, and Arm C's gain comes from vertices the classical arms already reach.
(d) Any of (a)–(c), but the vertices all three arms miss, or the vertices only C catches, concentrate in one or two tiles. The apparent per-vertex effect is then really a per-tile effect.
(e) The validation gate fails: the A/B per-vertex split or the C per-tile numbers do not reproduce.

**Interpretation plan**
(a) Add the three-way map to README's ESTABLISHED tier, qualified as "on this benchmark." Weaken the terrain-roughness HYPOTHESIS: the shared-miss vertices are not undetectable at 100m/px from DEM-derived inputs. This does NOT show that Arm C has no blind spots in general (n=21, 6 tiles).
(b) Report the three-way map and name the vertices all three arms miss. This fits the terrain-roughness hypothesis but does not confirm it, because those vertices could also reflect catalog or registration error. The natural next question is to inspect those specific vertices (morphology, relief, catalog placement).
(c) Strong support, on this benchmark, for a limit that applies to all three method families. Arm C's headline should then read "catches classically-reachable vertices more reliably," not "catches what classical methods miss." This does NOT show the three families are equivalent overall.
(d) Report per tile, and do not generalize a per-vertex claim from a one-tile effect. This points to the existing OPEN item about extending the benchmark beyond 6 tiles.
(e) Report the discrepancy as the finding and add a DECISION_LOG entry. Make no three-way claim until it is resolved.

For every outcome, report the secondary checkpoint alongside the primary. If the two land in different outcome categories, say so rather than choosing one.

**Confounds considered**
- **Mask shape vs the exact-pixel metric.** Arms A and B work on slope and characteristically mark the ridge flanks, not the crest (DECISION_LOG 2026-09-18). Arm C was trained on polyline-buffer labels that sit on the catalog line itself. So C might "catch" a shared-miss vertex because its output is centered on the catalog line, not because it detects signal the classical filters cannot. The primary metric does not control for this. As a descriptive check, also score A and B at the same vertices with a small pixel tolerance (e.g. a hit if any mask pixel lies within 1–2 px). Say explicitly whether the set both classical arms miss shrinks substantially under that tolerance. If it does, part of the "blind spot" is an artefact of the metric.
- **Non-independence.** The 57 vertices sit in 6 tiles, and vertices on the same ridge are correlated. The Poisson-binomial p-value treats them as independent, so it is optimistic. Per-tile reporting (outcome d) guards against this; it does not correct for it.
- **Small n.** With 21 shared-miss vertices, only large differences are detectable. State the minimum detectable difference rather than reading a null result as "no difference."
- **Checkpoint instability.** Epoch-60 checkpoints from identical code scored 70.2% and 21.1% (`research/arm_c_early_stopping/SPEC.md`). Any three-way map therefore describes a specific checkpoint, not "Arm C" in general, which is why two checkpoints are reported. The 21.1% checkpoint is excluded: it is near chance, so it tells us nothing about which vertices Arm C catches. The epoch-7 checkpoint was selected by val loss on these same 6 tiles, which is mild selection on the benchmark and is already recorded as a confound in PR #6.
- **Threshold.** Fixed at 0.5, as pre-registered everywhere else; no threshold is tuned for this analysis. The epoch-7 checkpoint's recall is sensitive to the threshold, so its secondary result carries that caveat.
- **Catalog or registration error.** A vertex all three arms miss may be a misplaced catalog vertex rather than a genuinely hard ridge. This analysis cannot tell the two apart, so the report must present catalog error as a live alternative to the terrain-roughness reading.


## Amendments

Dated notes only. The sections above are the issue body, unchanged.

**2026-09-27 (implementer).** The secondary checkpoint and its reference numbers
(`results/arm_c_early_stopping/model_best.pt`, `eval_best.json`) were merged to `master`
with PR #6 before this branch was cut, so this branch reads them from `master` and does not
depend on an unmerged branch. No change to the design above.

**2026-09-27 (implementer).** Two descriptive checks were added that the issue does not
pre-register: a leave-one-tile-out recomputation of metrics 1-2 (to judge outcome (d)),
and a cross-tab of the 21 shared-miss vertices against the 1-2 px A/B tolerance check (to
size the mask-shape confound). Both are labelled `not_preregistered_descriptive` in
`results/three_way_vertex_map/analysis.json`. Neither decides the outcome.

## Results

Everything below can be recomputed from `results/three_way_vertex_map/vertices.json`. That
file has one row per truth vertex (tile, vertex index, native row/col, hit/miss for A, B, C
and C2, and the A/B 1-2 px tolerance hits) plus the per-tile coverage of every arm. The
script is `research/three_way_vertex_map/vertex_map.py`, run under the repo's `Dockerfile`
(CPU only). Step 1 builds the table and runs the gate. Step 2 (`--analyze`) refuses to run
unless the gate passed.

### Validation gate: passed (`results/three_way_vertex_map/gate.json`)

Checked before any Arm C statistic was computed:
- **Arm A and B per-tile hits** reproduce DECISION_LOG.md exactly: A = 2/1/3/4/6/7 and
  B = 3/3/2/2/5/10 on 3674/1851/748/5017/4098/3461.
- **A/B per-vertex split** reproduces exactly: both 12, A-only 11, B-only 13, neither 21.
- **Arm C primary** (`results/arm_c/model.pt`) per-tile hits reproduce
  `results/arm_c/eval.json` exactly (5/6/2/6/12/9 = 40/57). Per-tile coverage matches to 0.0.
- **Arm C secondary** (`results/arm_c_early_stopping/model_best.pt`) reproduces
  `results/arm_c_early_stopping/eval_best.json` exactly (5/8/4/7/11/5 = 40/57). Coverage
  matches to within 1e-14.
- All 57 vertices fall inside their tiles.

### Deciding metrics, primary checkpoint (`results/arm_c/model.pt`, threshold 0.5)

| Vertex set | n | Arm C hits | Coverage-matched expectation | Lift | Exact one-sided p (Poisson-binomial) |
|---|---|---|---|---|---|
| Shared miss (A and B both miss) | 21 | **14** (66.7%) | 6.04 | **2.32x** | **1.9e-4** |
| Caught by A or B | 36 | 26 (72.2%) | 11.68 | 2.23x | 4.8e-7 |

- **Metric 1:** Arm C catches 14 of the 21 shared-miss vertices, against a chance
  expectation of 6.04 (p = 1.9e-4).
- **Metric 2:** Arm C's lift is about the same on the shared-miss set (2.32x) as on the
  caught set (2.23x). The ratio is 1.04. The raw hit rates are 66.7% and 72.2%.
- **Minimum detectable difference:** about 35 percentage points in raw hit rate between the
  two sets. This uses a normal approximation at the pooled rate, two-sided alpha 0.05 and
  80% power, and it assumes independent vertices, so it is optimistic. Any true gap smaller
  than that, in either direction, cannot be excluded. "About the same lift" means no large
  difference was seen, not that there is none.

### Secondary checkpoint (`results/arm_c_early_stopping/model_best.pt`, epoch 7, threshold 0.5)

| Vertex set | n | Arm C hits | Expectation | Lift | p |
|---|---|---|---|---|---|
| Shared miss | 21 | 15 (71.4%) | 8.29 | 1.81x | 1.6e-3 |
| Caught by A or B | 36 | 25 (69.4%) | 12.14 | 2.06x | 1.5e-6 |

- The lift ratio (shared-miss over caught) is 0.88.
- The raw hit rate is slightly *higher* on the shared-miss set.
- The lower lift comes from this checkpoint's higher coverage on the tiles where most shared
  misses sit (1851 and 5017).

### Descriptive: 8-cell A x B x C table (vertex counts)

| A | B | C primary | C secondary |
|---|---|---|---|
| + | + | C+ 10 / C- 2 | C+ 9 / C- 3 |
| + | - | C+ 7 / C- 4 | C+ 6 / C- 5 |
| - | + | C+ 9 / C- 4 | C+ 10 / C- 3 |
| - | - | C+ **14** / C- **7** | C+ **15** / C- **6** |

### Descriptive: vertices all three arms miss (tile vN (native row, col))

- **Primary (7):** 1851 v0 (20,22); 748 v2 (51,35); 748 v4 (76,49); 748 v5 (95,48);
  748 v7 (136,52); 4098 v13 (162,41); 3461 v0 (85,297).
- **Secondary (6):** 748 v2, 748 v5, 748 v7, 4098 v2, 3461 v0, 3461 v12.
- **In both lists (4):** 748 v2, 748 v5, 748 v7, 3461 v0.
- **Tile concentration:** 4 of the primary's 7 are on tile 748. That is also where the
  primary's coverage is lowest (15.2%).
- **Polyline endpoints:** 3 of the 7 are endpoints (1851 v0, 748 v7, 3461 v0). This is an
  observation only; it was not tested.

### Descriptive: per tile (primary)

| Tile | n | C cov | Shared miss n | C hits / expected | Caught n | C hits / expected | All-three miss |
|---|---|---|---|---|---|---|---|
| 3674 | 5 | 54.3% | 1 | 1 / 0.54 | 4 | 4 / 2.17 | 0 |
| 1851 | 8 | 35.9% | 4 | 3 / 1.44 | 4 | 3 / 1.44 | 1 |
| 748 | 8 | 15.2% | 5 | 1 / 0.76 | 3 | 1 / 0.46 | 4 |
| 5017 | 7 | 48.5% | 3 | 3 / 1.45 | 4 | 3 / 1.94 | 0 |
| 4098 | 15 | 20.1% | 6 | 5 / 1.20 | 9 | 7 / 1.81 | 1 |
| 3461 | 14 | 32.2% | 2 | 1 / 0.65 | 12 | 8 / 3.87 | 1 |

- The secondary checkpoint's per-tile table is in `analysis.json` (`secondary_C2.per_tile`).
- The 14 shared-miss vertices the primary checkpoint catches come from all 6 tiles:
  5 from 4098, 3 each from 1851 and 5017, and 1 each from 3674, 748 and 3461.

### Descriptive: leave-one-tile-out (not pre-registered)

- **Primary:** dropping any single tile keeps metric 1 at p < 0.05. The worst case is
  dropping 4098: 9/15 against 4.84 expected, p = 0.019. The lift ratio stays between 0.97
  and 1.10.
- **Secondary:** dropping 4098 raises metric 1 to p = 0.066 (10/15 against 6.79 expected).
  Every other drop stays below p = 0.01. The lift ratio ranges from 0.77 to 0.99.

### Descriptive: mask-shape confound (A/B at 1-2 px tolerance)

| Tolerance | A hits | B hits | Vertices both A and B miss |
|---|---|---|---|
| exact pixel (the metric) | 23 | 25 | 21 |
| 1 px | 27 | 30 | 16 |
| 2 px | 35 | 35 | 12 |

- **The shared-miss set shrinks substantially under tolerance:** by 24% at 1 px and by 43%
  at 2 px.
- **Where Arm C's hits fall:**
  - A or B reaches 9 of the 21 shared-miss vertices within 2 px. The primary Arm C hits 6
    of those 9.
  - 12 stay missed by both A and B at 2 px. The primary Arm C still hits 8 of them (the
    secondary hits 9).
- **Arm C was not scored at tolerance.** The issue asks only for A and B.

## Interpretation

**Outcome: the primary checkpoint is (a), with a (d) qualifier on the residual set. The
secondary checkpoint sits on the (a)/(b) boundary and is less robust across tiles. The two
checkpoints do not land cleanly in the same category.**

**Primary checkpoint**
- **Metric 1:** Arm C catches the shared-miss set well above chance (14/21 against 6.04,
  p = 1.9e-4).
- **Metric 2:** its lift on the shared-miss set (2.32x) is about equal to its lift on the
  classically-caught set (2.23x). That is outcome (a) as pre-registered.
- **Against the stated hypothesis:** the hypothesis predicted (b): above chance, but with a
  *lower* hit rate or lift on the shared-miss set. The first half is supported. The second
  is not: no lower lift was observed. At n = 21 against n = 36, only a gap of roughly 35 pp
  or more would have been detectable, so a moderate gap is not ruled out.
- **Not a one-tile effect:** the C-only hits span all 6 tiles, and every leave-one-tile-out
  drop stays significant.
- **The residual set does concentrate:** 4 of the 7 vertices all three arms miss are on
  tile 748. Per the (d) rule, the residual is reported per tile. It is not generalized into
  a per-vertex "shared blind spot" claim.

**Secondary checkpoint (epoch 7)**
- **Metric 1:** also well above chance (15/21 against 8.29, p = 0.0016).
- **Metric 2:** its lift on the shared-miss set is 12% lower than on the caught set (1.81x
  against 2.06x), which leans toward (b). But its raw hit rate is higher on the shared-miss
  set, and the gap is far inside the minimum detectable difference. This analysis cannot
  place it in (a) or in (b).
- **Tile fragility:** its metric-1 result depends on tile 4098; dropping that tile gives
  p = 0.066. This is a (d)-type fragility the primary does not have.
- Per the interpretation plan, this is reported, not resolved.

**What this establishes, on this benchmark (6 tiles, 57 vertices, exact-pixel metric)**
- Most of the 21 vertices both classical arms miss are not undetectable from DEM-derived
  inputs at 100m/px. Both Arm C checkpoints reach 14-15 of them, at about twice the rate of
  a coverage-matched random mask.
- The three method families do not share one blind spot. The classical arms' shared misses
  are mostly reachable by Arm C.

**What it does not establish**
- **That Arm C has no blind spots in general.** This is n = 21 on 6 tiles, from two
  specific checkpoints. The same training code has also produced a near-chance (21.1%)
  checkpoint, which is excluded here.
- **That the 7 (or 6) vertices all three arms miss are genuinely hard ridges.** Catalog or
  registration error is an equally live explanation. The concentration on tile 748 is also
  compatible with Arm C's low coverage there (15%).
- **That Arm C "sees signal the classical filters cannot."**
  - Part of the classical blind spot is an artefact of the exact-pixel metric. Under a 2 px
    tolerance the shared-miss set shrinks from 21 to 12. That fits the known
    flank-not-crest behaviour of the slope-domain arms.
  - Arm C's polyline-buffer training labels may favour it under an exact-pixel metric.
  - Arm C still hits 8 of the 12 vertices that survive tolerance, so the headline does not
    vanish under this check. But Arm C was not itself scored at tolerance, so the
    comparison is not yet symmetric.
- **Accurate significance.** The p-values treat vertices as independent, so they are
  optimistic: vertices on the same ridge are correlated.

**Terrain-roughness hypothesis.** Per interpretation plan (a), it is weakened: the 21
classical shared misses are mostly not undetectable at 100m/px. This counts against it as a
limit on *every* method family. It does not refute it as an account of why the *classical*
filters fail.

## Decision

**Implementer's self-assessment: MERGE, as an analysis result.**
- The validation gate reproduced exactly.
- The metrics are the pre-registered ones.
- The raw per-vertex data and the script are committed.
- The outcome is labelled against the pre-registration, including where it contradicts the
  stated hypothesis (b).
- The checkpoint disagreement, the concentration of the residual set on one tile, and the
  metric-shape confound are all reported.

Per interpretation plan (a), `README.md` is updated:
- The three-way map is added to ESTABLISHED, qualified as "on this benchmark."
- The terrain-roughness HYPOTHESIS is weakened.

No DECISION_LOG.md entry is added: the interpretation plan calls for one only under
outcome (e).

**Natural follow-ups** (not done here; each would be a new issue):
- Score Arm C at the same 1-2 px tolerance as A/B, so the mask-shape confound is tested
  symmetrically.
- Inspect the 4 vertices every arm and checkpoint misses (748 v2/v5/v7, 3461 v0): their
  morphology, relief and catalog placement.
- Extend the benchmark beyond 6 tiles.
