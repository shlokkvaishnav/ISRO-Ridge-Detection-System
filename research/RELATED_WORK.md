# Related work: what is actually unclaimed

This document positions the finding in this repo against published work. It exists to keep
the project honest: the claims that first suggested themselves (see the original stub,
preserved in git history) turn out to be narrower once checked, and the last section lists
what must **not** be said because someone else already owns it.

**Refreshed 2026-09-22.** The original version (2026-09-18) was written at repo scaffolding
time and left three open verification tasks unresolved: whether HL-YOLOv8 already benchmarks
against a classical baseline, whether Hessian ridge filters have been applied to lunar
terrain specifically, and (implicitly) whether the three-way comparison this project runs is
actually novel. This pass resolves all three by literature search, and finds a fourth,
closer collision (DBR-Net) the original version didn't know to look for.

**Status of the citations.** Gathered by web search on 2026-09-22 (queries logged in §Scooping
check below), not independently re-verified against full paper text the way
`Replica-Recall-Divergence`'s equivalent document does — that project's own history shows
search-summary claims can misquote or overstate a source, so treat the numbers below as
"as reported by search," not "read in the original PDF," until someone does that pass.
**One suspected contamination is flagged and excluded, not cited**: a search for Hessian-vs-
deep-learning lunar comparisons returned an AI-generated summary sentence that reads almost
verbatim like this project's own public README (which now appears in general web search
results, since the repo is public). That sentence is not treated as independent prior art
anywhere below.

---

## The claim

No published work runs classical phase-symmetry morphology, Hessian-eigenvalue ridge
filtering, and a deep-learning detector against the *same* lunar tiles and the *same*
ground-truth catalog, then reports which individual true-ridge vertices each one catches and
misses. Existing comparisons in this space are architecture papers that benchmark a new
detector's aggregate metric (precision/recall/mAP) against whatever baseline it's built to
beat — not a genuine multi-method failure-mode map.

Measured so far, classical arms only (Arm C not yet built): Arm A (phase symmetry, shape-
filtered) 40.4% aggregate true-vertex recall; Arm B (Hessian/frangi) 43.9%; union 63.2% but
at proportionally higher mask-coverage cost — see `research/DECISION_LOG.md` for full
numbers. The three-way comparison this project is built around is not yet complete (Arm C
not started), so the claim below is about what's *unclaimed*, not yet about a finished result.

---

## Per-area findings

### 1. Automated lunar wrinkle ridge detection (classical) — **COVERS, this project's own baseline**

- **Analysis and mapping of lunar wrinkle ridges (LWRs) using automated LWRs detection
  process with LROC-WAC and LOLA data** (Frontiers in Astronomy and Space Sciences, 2023).
  The reference classical pipeline this project's Arm A reproduces as a real, run baseline:
  slope map → phase symmetry → regional thresholding → morphological closing/opening/edge-
  linking.
- **AUTOMATED DETECTION OF LUNAR RIDGES BASED ON DEM DATA** (ISPRS Archives, 2019). DEM-based
  detection, methodological precedent for the same pipeline shape.
- **Global mapping and analysis of lunar wrinkle ridges** (Thompson, Robinson, Watters,
  Johnson — LPSC 2017 abstract; the shapefile product this project uses as ground truth,
  5,999 segments). Not a detection *method* paper — the catalog itself.

These three establish that Arm A's classical pipeline is a faithful reproduction of the
field's own standard approach, not a novel method in its own right. Nothing here compares
against Hessian filters or deep learning.

### 2. Deep learning for lunar wrinkle ridges specifically — **COVERS, closer than the original stub knew**

Two direct prior works, both closer collisions than the 2026-09-18 version of this document
was aware of:

- **HL-YOLOv8** — an attention-augmented YOLOv8 (multiscale lightweight channel attention +
  multi-head self-attention) for lunar linear tectonic features (wrinkle ridges, grabens,
  lobate scarps). As reported: baseline YOLOv8 on wrinkle ridges scores 53.7% precision /
  61.0% recall / 53.4% mAP@0.5; HL-YOLOv8 improves this to 67.4% / 70.5% / 72.7%. This
  resolves the original stub's open question: HL-YOLOv8's own comparison is
  architecture-vs-architecture (YOLOv8 variants against each other), **not** against a
  classical phase-symmetry or Hessian-filter baseline. The seam this project sits in — a
  classical-vs-classical-vs-deep comparison — is still open on this evidence.
- **DBR-Net** (Dual-Branch Ridge Detection Network) — found in the same search pass, not in
  the original stub. Purpose-built for lunar wrinkle ridges specifically (not the broader
  "linear tectonic features" class HL-YOLOv8 targets), uses aspect data as an edge-
  information branch plus an Attention Complementary Feature Fusion (ACFF) module, reports
  pixel-level extraction and 6 newly-detected candidate ridges from applying it to lunar mare
  regions. This is the single closest existing work to this project's planned Arm C — closer
  than HL-YOLOv8, since it targets wrinkle ridges specifically rather than the broader
  tectonic-feature class.

**What this means for Arm C, not yet built:** this project cannot claim to be first to apply
deep learning to lunar wrinkle ridge detection. DBR-Net already exists and is closer to what
Arm C would be than HL-YOLOv8 is. Full DBR-Net paper (venue, publication details, whether it
reports a classical baseline comparison) not yet read in full — flagged as an open
verification task below, and load-bearing before Arm C's own related-work framing is written.

### 3. Hessian-eigenvalue ridge filters on terrain/DEM data — **PARTIAL: established generally, not confirmed on lunar/planetary terrain**

Resolves the original stub's second open question directly:

- Frangi/Meijering/Sato-family filters are well-established generally for tube/ridge-like
  structure detection (their canonical use is vessel segmentation in medical imaging), and
  reviewed as a class in **Ridge Detection by Image Filtering Techniques: A Review and an
  Objective Analysis** (Pattern Recognition and Image Analysis, Springer) — a general
  comparative survey of ridge-detection filtering techniques, not planetary-specific.
- On terrestrial DEM/geomorphology specifically, related but structurally different
  approaches exist: the **Ridge-Drainage Index / continuum filter** (characterizing terrain
  on a ridge-to-drainage continuum via local openness angles, not Hessian eigenvalues
  directly) and classical **plan-curvature-based ridge/channel classification** (smooth the
  DEM, classify positive/negative plan curvature as ridge/channel). These solve an adjacent
  problem (ridge vs. drainage classification generally) with different mathematical
  machinery, not the same Hessian-eigenvalue mechanism this project's Arm B uses.
- **No result found applying Frangi/Meijering/Sato specifically to lunar, Mars, Mercury, or
  Europa terrain data.** This is a "we could not find," not a proof of absence — the search
  was not exhaustive and planetary-science venues are not fully indexed by general web
  search. Stated as an absence, carefully, per the same caution
  `Replica-Recall-Divergence/research/RELATED_WORK.md` applies to its own absence claims.

**What this means for Arm B:** the honest framing is "Hessian ridge filters are established
generally, and for terrestrial DEM ridge/channel classification via different curvature-based
methods, but this project could not find them applied via the Frangi/Meijering/Sato
eigenvalue mechanism specifically to lunar or other planetary terrain" — not "first to use
Hessian filters on terrain" (false, terrestrial precedent exists) and not "first to use them
on the Moon" (unverified, not "first" — "not found," which is weaker and should be stated as
such).

### 4. Canny-style NMS + hysteresis thresholding on planetary terrain — **COVERS a related target (craters), GAP on ridges**

A direct, close prior use of exactly this technique family on lunar imagery: a lunar crater
detection methodology using **Canny edge detection to enhance crater rims** plus a modified
Hough transform for elliptical localization, with the same NMS + hysteresis thresholding
structure this project's `nms_hysteresis_threshold` experiment implements
(`research/DECISION_LOG.md`, 2026-09-22).

**Distinguish, don't conflate:** that work targets crater rims (roughly circular, high-
contrast boundaries against a flat interior) — a much more favorable target for Canny-style
edge detection than a degraded, low-relief wrinkle ridge embedded in naturally rough mare
terrain, which is exactly the harder case this project's own real-tile findings document.
Confirms the technique family has planetary precedent generally, not that it's expected to
work as well on this project's specific, harder target.

### 5. Structurally analogous problem: double ridges on Europa — **Adjacent domain, real evidence on the precision/recall tradeoff shape**

**LineaMapper** (Mask R-CNN instance segmentation of linear surface features on Europa,
including "double ridges" — a feature class structurally similar to lunar wrinkle ridges,
different formation mechanism) is the most directly comparable *cross-domain* precedent
found: a deep-learning detector for a ridge-like linear planetary feature, with an honestly
reported precision/recall tradeoff (higher precision than recall overall; highest precision
specifically on double ridges, highest recall on the broader "ridge complex" class). This is
useful adjacent evidence that recall/precision tradeoffs across feature sub-classes are a
known, real issue in this general feature family — not specific evidence about wrinkle
ridges or about classical-vs-deep-learning comparisons, since LineaMapper is deep-learning-
only with no classical baseline reported in what was found.

### 6. Cross-method (classical vs. classical vs. deep-learning) comparison with per-instance failure attribution — **GAP, this project's actual claim**

Nothing found runs a phase-symmetry detector, a Hessian-eigenvalue detector, and a deep-
learning detector against the same tiles and ground truth and reports which *individual*
catalog instances each one catches or misses (as opposed to reporting only aggregate
precision/recall/mAP per architecture). HL-YOLOv8 and DBR-Net each report their own
architecture's aggregate metrics against other deep-learning baselines; neither, as far as
found, runs a genuine three-family comparison with instance-level attribution. This is the
actual seam this project sits in, and it survives the two direct-collision checks in §2
because those papers are architecture-improvement papers, not comparison-methodology papers.

---

## Novelty verdict

Ranked by how much of the field-standard literature has already been checked, strongest
survivor first:

1. **Per-instance failure-mode attribution across method families, not just aggregate
   metrics.** Unclaimed as far as found (§6). This is the load-bearing contribution — the
   A/B complementarity finding already in `DECISION_LOG.md` (both arms hit 12 of 57 vertices,
   A-only 11, B-only 13, neither 21) is a concrete instance of exactly this kind of
   attribution, and nothing found in HL-YOLOv8, DBR-Net, or LineaMapper does the equivalent.
2. **The three-way method-family comparison itself** (classical phase symmetry vs. Hessian
   eigenvalue vs. deep learning), on the same tiles and catalog. Weaker than item 1 alone
   because pairwise comparisons (deep learning vs. its own predecessor architecture) are
   common in the field — what's missing specifically is the *breadth* of comparing across
   method *families*, not just across deep-learning generations.
3. **Hessian-eigenvalue filters applied to lunar terrain.** Plausible gap (§3), stated
   carefully as "not found," not "first" — terrestrial DEM ridge/channel work exists with
   different curvature-based machinery, and the search was not exhaustive.

**Already well-trodden — cite, do not claim:** deep learning applied to lunar wrinkle ridge
/ linear tectonic feature detection generally (HL-YOLOv8, DBR-Net — both exist, both closer
than this project's original scaffolding assumed); NMS + hysteresis thresholding on lunar
imagery generally (the crater-rim Canny application); Hessian ridge filters as a general
technique class (medical imaging, general ridge-detection surveys); ridge/channel
classification from DEM curvature generally (Ridge-Drainage Index, plan-curvature methods).

---

## Must-cite

| # | Reference | Why |
|---|---|---|
| 1 | Analysis and mapping of lunar wrinkle ridges (LWRs) using automated LWRs detection process with LROC-WAC and LOLA data, Frontiers in Astronomy and Space Sciences, 2023 | Arm A's reference classical pipeline |
| 2 | AUTOMATED DETECTION OF LUNAR RIDGES BASED ON DEM DATA, ISPRS Archives, 2019 | DEM-based detection precedent |
| 3 | Thompson, Robinson, Watters, Johnson, Global mapping and analysis of lunar wrinkle ridges, LPSC 2017 | Ground-truth catalog (5,999 segments) used by all arms |
| 4 | HL-YOLOv8 (Automated Detection and Classification of Lunar Linear Tectonic Features Using a Deep Learning Method) | Closest deep-learning prior work for the broader tectonic-feature class; reports 67.4%/70.5%/72.7% (P/R/mAP@0.5) on wrinkle ridges specifically; compares only against other YOLOv8 variants, not classical methods — the seam this project sits in |
| 5 | DBR-Net (dual-branch, aspect-data, ACFF module, lunar-wrinkle-ridge-specific) | **Closest existing deep-learning work to this project's planned Arm C** — read in full before writing Arm C's own related-work section |
| 6 | LineaMapper (Mask R-CNN, Europa linear features incl. double ridges) | Adjacent-domain evidence for precision/recall tradeoffs in ridge-like linear feature detection; no classical baseline reported |
| 7 | Automated lunar crater detection with Canny edge detection + Hough transform | Direct precedent for NMS+hysteresis on lunar imagery, different target (crater rims vs. degraded ridges) |
| 8 | Ridge Detection by Image Filtering Techniques: A Review and an Objective Analysis, Pattern Recognition and Image Analysis | General survey of the Hessian/Frangi-family filter class, not planetary-specific |

**Deliberately not cited:** an AI-search-summary claim that classical phase-symmetry and
Hessian filtering are "compared against deep learning methods for lunar wrinkle ridge
detection" in some unnamed existing study. This could not be traced to an independent source
and reads like a paraphrase of this project's own public README — treated as contamination,
not evidence, per the standing caution in §Scooping check.

---

## Framing risks — what NOT to claim

1. **Do not claim to be first to apply deep learning to lunar wrinkle ridges.** DBR-Net and
   HL-YOLOv8 both exist and both predate this project's Arm C (not yet built). Claim instead:
   first to compare a deep-learning detector against classical phase-symmetry *and* Hessian-
   eigenvalue detectors on the same tiles with instance-level attribution.
2. **Do not claim Hessian ridge filters are new to terrain analysis.** They're an established
   general technique (medical imaging origin, reviewed as a class). The narrower, carefully-
   stated claim is "not found applied via this specific eigenvalue mechanism to lunar/
   planetary terrain" — a "not found," not a "first."
3. **Do not claim NMS+hysteresis is novel for lunar imagery.** It's already used for crater-
   rim detection. Distinguish: applied here to a much harder target (degraded ridges in rough
   terrain, not high-contrast crater rims against flat interior) — and this project's own
   real-tile results (`DECISION_LOG.md`) already document that the harder target produces a
   mixed, not clean, result — consistent with, not contradicting, this distinction.
4. **Do not claim this generalizes beyond the Moon.** LineaMapper's Europa double-ridge work
   is adjacent evidence the general feature class matters elsewhere, not evidence this
   project's specific findings transfer. Scope stays LOLA/WAC (+ planned Chandrayaan-2
   TMC-2) unless and until other-body data is added.
5. **Do not cite the "compared against deep learning" AI-summary sentence found during this
   search pass.** Flagged above as likely self-referential contamination from this project's
   own public repo, not independently verified prior art.

---

## Scooping check — what was actually searched, and when

**2026-09-22**, web search, six queries:

1. `comparison classical machine learning deep learning geomorphological feature detection DEM benchmark which method catches which`
2. `Hessian ridge filter terrain DEM geomorphology application Frangi Meijering Sato landform`
3. `Canny non-maximum suppression hysteresis thresholding lunar planetary terrain feature extraction prior work`
4. `HL-YOLOv8 lunar linear tectonic features baseline comparison classical methods recall`
5. `ridge-drainage continuum filter digital elevation model ridge channel classification method`
6. `"wrinkle ridge" deep learning detection lunar Mars Mercury CNN automated mapping 2024 2025`
7. `DBR-Net dual-branch ridge detection network lunar wrinkle ridge aspect attention complementary feature fusion`
8. `LineaMapper lineament mapping deep learning Europa wrinkle ridge retrain`
9. `DBR-Net HL-YOLOv8 lunar wrinkle ridge baseline comparison phase symmetry classical method Frangi`
10. `"double ridges" Europa detection method comparison classical filter deep learning which features missed`

**Found and now cited:** DBR-Net (closer Arm C collision than previously known), HL-YOLOv8's
actual comparison scope (architecture-vs-architecture, not vs. classical), LineaMapper
(adjacent-domain precision/recall evidence), the lunar-crater Canny precedent, and the
general Hessian-filter-review + terrestrial ridge-drainage literature.

**Not found:** a three-method-family comparison with per-instance attribution for any
planetary ridge/lineament feature; Hessian/Frangi-family filters applied specifically to
lunar or other planetary terrain.

**Standing caveat:** this is "we could not find," not "there is none." Web search is not a
substitute for a full literature-database search (ADS, Web of Science) before any external
submission, and none of the sources above were read past their search-result summaries —
verify against the actual paper text before relying on a specific reported number (e.g.
HL-YOLOv8's 70.5% recall figure) in anything citing this document.

## Open verification tasks

- Read the DBR-Net paper in full — venue, exact architecture, whether it reports any
  classical baseline comparison. This is now the load-bearing unread source, more urgent
  than the original stub's HL-YOLOv8 task (now resolved, see §2).
- Read the HL-YOLOv8 paper in full to confirm the reported 53.7/61.0/53.4 → 67.4/70.5/72.7
  numbers and check for any classical-method comparison row not surfaced by search summaries.
- A proper ADS/Web of Science search before any external submission — this pass used general
  web search only, per the standing caveat above.
