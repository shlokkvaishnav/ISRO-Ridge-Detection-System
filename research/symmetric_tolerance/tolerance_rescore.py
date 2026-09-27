"""Does Arm C's advantage survive when A, B and C are all scored at the same 1-2 px tolerance?

Pre-registered in issue #14 (research/symmetric_tolerance/SPEC.md). Two steps:

1. ``python research/symmetric_tolerance/tolerance_rescore.py``
   Regenerates the Arm A (shape-filter), Arm B (frangi) and Arm C (C = results/arm_c/model.pt,
   C2 = results/arm_c_early_stopping/model_best.pt, threshold 0.5) masks on the 6 benchmark
   tiles with the exact code paths of research/three_way_vertex_map/vertex_map.py (imported,
   not copied). For every truth vertex and every arm it records the hit at N = 0, 1, 2 px
   under ``within_tolerance`` (N = 0 is ``mask[r, c]``), and for every tile and arm the
   *dilated coverage* at N: the fraction of the tile's pixels that have at least one mask
   pixel within Euclidean distance N (the chance that a uniformly random vertex position
   would score a hit under the same rule). Writes results/symmetric_tolerance/vertices.json.
   Then runs the validation gate (issue #14, items 1-3, plus the per-tile A/B references in
   DECISION_LOG and the exact-pixel coverage-matched expectations of #6/#7) and the power
   check from marginals only, and writes results/symmetric_tolerance/gate.json. It computes
   no joint C-vs-classical statistic.

2. ``python research/symmetric_tolerance/tolerance_rescore.py --analyze``
   Refuses to run unless gate.json says the gate passed and every deciding comparison's
   smallest achievable p is <= 0.05. Computes, from vertices.json alone, the pre-registered
   deciding metrics at 2 px (exact one-sided sign tests C vs A and C vs B; lift over
   tolerance-matched chance), the pre-registered descriptive set (1 px, 0 px, per tile, C2,
   leave-one-tile-out, residual sets), Monte Carlo cross-checks, and the outcome label
   against the issue's (a)-(e). Writes results/symmetric_tolerance/analysis.json.

Needs data/tiles/<id>/ and data/raw/wrinkle_ridges_shapefile/ locally (gitignored).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "vertex_map", os.path.join(HERE, "..", "three_way_vertex_map", "vertex_map.py"))
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)

OUT_DIR = "results/symmetric_tolerance"
VERTICES_JSON = os.path.join(OUT_DIR, "vertices.json")
GATE_JSON = os.path.join(OUT_DIR, "gate.json")
ANALYSIS_JSON = os.path.join(OUT_DIR, "analysis.json")
REF_VERTICES = "results/three_way_vertex_map/vertices.json"
REF_SWEEP = "results/arm_c_early_stopping/threshold_sweep.json"
REF_ANALYSIS_7 = "results/three_way_vertex_map/analysis.json"
BENCHMARK = vm.BENCHMARK
ARMS = ["A", "B", "C", "C2"]
CLASSICAL = ["A", "B"]
TOLS = [0, 1, 2]
PRIMARY_TOL = 2
SECONDARY_TOL = 1
ALPHA = 0.05
MC_DRAWS = 200_000
MC_POSITIONS = 20_000
SEED = 20260927


def hit_key(arm: str, tol: int) -> str:
    return arm if tol == 0 else f"{arm}_tol{tol}"


def disk_footprint(tol: int) -> np.ndarray:
    """Offsets (dr, dc) with dr^2 + dc^2 <= tol^2: the same set within_tolerance scans."""
    d = np.arange(-tol, tol + 1)
    return (d[:, None] ** 2 + d[None, :] ** 2) <= tol * tol


def dilate(mask: np.ndarray, tol: int) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    if tol == 0:
        return mask.copy()
    # Symmetric footprint, outside-tile pixels count as not-mask (border_value=0), so
    # dilated[r, c] == within_tolerance(mask, r, c, tol) for every in-tile pixel.
    return binary_dilation(mask, structure=disk_footprint(tol), border_value=0)


# --------------------------------------------------------------------------
# Step 1: masks -> per-vertex hits at 0/1/2 px + dilated coverage
# --------------------------------------------------------------------------

def build_table() -> dict:
    import rasterio
    import rasterio.transform
    import rasterio.warp
    import torch

    from src.classical.pipeline import detect_ridges as detect_a
    from src.data.ridge_catalog import load_ridge_catalog
    from src.data.tile_extraction import RIDGE_CATALOG_CRS
    from src.deep.model import DualBranchRidgeNet
    from src.hessian.pipeline import detect_ridges as detect_b

    with open(vm.MANIFEST) as f:
        manifest = {e["id"]: e for e in json.load(f)}
    segments = {s.id: s for s in load_ridge_catalog(vm.SHP)}
    models = {}
    for name, path in vm.CHECKPOINTS.items():
        m = DualBranchRidgeNet()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        models[name] = m

    vertices, tiles, dilation_consistency = [], [], []
    rng = np.random.default_rng(SEED)
    for tid in BENCHMARK:
        with rasterio.open(os.path.join(vm.TILES_DIR, manifest[tid]["dem_path"])) as src:
            elevation = src.read(1).astype(np.float64)
            transform, crs = src.transform, src.crs
        masks = {
            "A": detect_a(elevation)["ridge_mask"].astype(bool),
            "B": detect_b(elevation, method="frangi")["ridge_mask"].astype(bool),
        }
        for name, model in models.items():
            masks[name] = vm.arm_c_mask(model, elevation, vm.THRESHOLD)
        dil = {(k, t): dilate(masks[k], t) for k in ARMS for t in TOLS}
        # Monte Carlo cross-check of the chance baseline: uniformly random positions scored
        # with within_tolerance itself, independent of the dilation code path.
        rr = rng.integers(0, elevation.shape[0], MC_POSITIONS)
        cc = rng.integers(0, elevation.shape[1], MC_POSITIONS)
        mc = {str(t): {k: float(np.mean([vm.within_tolerance(masks[k], int(a), int(b), t)
                                          for a, b in zip(rr, cc)])) for k in ARMS} for t in TOLS}

        seg = segments[tid]
        xs, ys = rasterio.warp.transform(RIDGE_CATALOG_CRS, crs,
                                         [p[0] for p in seg.points], [p[1] for p in seg.points])
        rows, cols = rasterio.transform.rowcol(transform, xs, ys)
        h, w = elevation.shape
        tiles.append({
            "tile": tid,
            "shape": [h, w],
            "coverage": {k: float(masks[k].sum() / masks[k].size) for k in ARMS},
            "dilated_coverage": {str(t): {k: float(dil[(k, t)].mean()) for k in ARMS} for t in TOLS},
            "dilated_coverage_monte_carlo": {"positions": MC_POSITIONS, "seed": SEED, **mc},
        })
        for i, (r, c) in enumerate(zip(rows, cols)):
            r, c = int(r), int(c)
            inside = 0 <= r < h and 0 <= c < w
            v = {"tile": tid, "vertex": i, "row": r, "col": c, "in_tile": inside}
            for k in ARMS:
                v[k] = bool(inside and masks[k][r, c])
                for t in TOLS[1:]:
                    v[hit_key(k, t)] = vm.within_tolerance(masks[k], r, c, t)
                    if inside:
                        dilation_consistency.append(bool(dil[(k, t)][r, c]) == v[hit_key(k, t)])
            vertices.append(v)

    return {
        "description": "Per-vertex hit/miss for Arms A, B, C (primary) and C2 (secondary) on the 6-tile "
                       "benchmark at 0/1/2 px. Key 'X' = exact pixel mask[row, col]; 'X_tolN' = "
                       "vertex_map.within_tolerance(mask, row, col, N) (any mask pixel within Euclidean "
                       "distance N). tiles[].dilated_coverage[N][X] = fraction of tile pixels with a mask "
                       "pixel within N px (tolerance-matched chance hit probability).",
        "arms": {
            "A": "src.classical.pipeline.detect_ridges, defaults (shape-filter)",
            "B": "src.hessian.pipeline.detect_ridges, method=frangi, defaults",
            "C": f"{vm.CHECKPOINTS['C']} @ threshold {vm.THRESHOLD}",
            "C2": f"{vm.CHECKPOINTS['C2']} @ threshold {vm.THRESHOLD}",
        },
        "dilation_matches_within_tolerance_at_every_vertex": all(dilation_consistency),
        "tiles": tiles,
        "vertices": vertices,
    }


# --------------------------------------------------------------------------
# exact statistics
# --------------------------------------------------------------------------

def sign_test_p(n_c: int, n_x: int) -> float:
    """Exact one-sided sign test: P(Binomial(n_c + n_x, 0.5) >= n_c)."""
    n = n_c + n_x
    return sum(comb(n, k) for k in range(n_c, n + 1)) / 2 ** n if n else 1.0


def power_from_marginals(h_c: int, h_x: int) -> dict:
    """What the sign test can do given only the two arms' total hits.
    With gap g = h_c - h_x, the discordant counts satisfy n_c = n_x + g. The smallest
    achievable p is 0.5**g (n_x = 0). p grows with n_x at fixed g, so rejection holds for
    n_x <= max_classical_discordants_to_reject."""
    g = h_c - h_x
    if g <= 0:
        return {"gap": g, "p_min": 1.0, "reachable": False,
                "max_classical_favouring_discordants_to_reject": None,
                "max_C_favouring_discordants_to_reject": None}
    p_min = 0.5 ** g
    n_x_max = None
    n_x = 0
    while n_x + g + n_x <= 57:  # at most 57 discordants
        if sign_test_p(n_x + g, n_x) < ALPHA:
            n_x_max = n_x
        n_x += 1
    return {
        "gap": g,
        "p_min": p_min,
        "reachable": p_min < ALPHA,
        "max_classical_favouring_discordants_to_reject": n_x_max,
        "max_C_favouring_discordants_to_reject": None if n_x_max is None else n_x_max + g,
        "rule": "reject (p < 0.05) iff n_classical_favouring <= max_classical_favouring_discordants_to_reject",
    }


def poisson_binomial_upper(ps, k: int) -> float:
    return float(vm.poisson_binomial_pmf(ps)[k:].sum())


def cov_lookup(tiles) -> dict:
    return {t["tile"]: t["dilated_coverage"] for t in tiles}


def lift(V, tiles, arm: str, tol: int) -> dict:
    cov = cov_lookup(tiles)
    ps = [cov[v["tile"]][str(tol)][arm] for v in V]
    hits = sum(v[hit_key(arm, tol)] for v in V)
    exp = float(sum(ps))
    return {
        "hits": hits, "n": len(V), "expected_hits_tolerance_matched": round(exp, 4),
        "lift": round(hits / exp, 4) if exp else None,
        "descriptive_p_one_sided_ge_poisson_binomial": poisson_binomial_upper(ps, hits),
    }


def sign_test(V, c_arm: str, x_arm: str, tol: int) -> dict:
    kc, kx = hit_key(c_arm, tol), hit_key(x_arm, tol)
    n_c = sum(v[kc] and not v[kx] for v in V)
    n_x = sum(v[kx] and not v[kc] for v in V)
    return {
        f"{c_arm}_hits": sum(v[kc] for v in V), f"{x_arm}_hits": sum(v[kx] for v in V),
        "both": sum(v[kc] and v[kx] for v in V), "neither": sum(not v[kc] and not v[kx] for v in V),
        f"{c_arm}_only": n_c, f"{x_arm}_only": n_x,
        "p_one_sided": sign_test_p(n_c, n_x),
    }


def classify(st: dict, lifts: dict, c_arm: str = "C") -> str:
    """Issue #14 (a)-(c) at a single tolerance, from the sign tests and lifts."""
    beats = all(st[x]["p_one_sided"] < ALPHA for x in CLASSICAL)
    if not beats:
        return "c"
    if all(lifts[c_arm]["lift"] >= lifts[x]["lift"] for x in CLASSICAL):
        return "a"
    return "b"


# --------------------------------------------------------------------------
# Step 1b: validation gate + power check (marginals only)
# --------------------------------------------------------------------------

def run_gate(table: dict) -> dict:
    V, tiles = table["vertices"], table["tiles"]
    with open(REF_VERTICES) as f:
        ref = json.load(f)
    RV, RT = ref["vertices"], ref["tiles"]
    g = {}

    # 1. C and C2 exact-pixel per-tile hits reproduce the eval files.
    for name, path in vm.C_REFERENCE.items():
        with open(path) as f:
            e = json.load(f)
        refpt = {t["id"]: {"hits": t["hits"], "truth": t["truth"]} for t in e["per_tile"]}
        obs = {t: {"hits": sum(v[name] for v in V if v["tile"] == t),
                   "truth": sum(1 for v in V if v["tile"] == t)} for t in BENCHMARK}
        g[f"1_{name}_exact_per_tile"] = {
            "reference_file": path,
            "reference": {str(k): v for k, v in refpt.items()},
            "observed": {str(k): v for k, v in obs.items()},
            "passed": refpt == obs,
        }

    # 2. Regenerated A/B reproduce the committed A, B, A_tol1/2, B_tol1/2 vertex for vertex.
    cols = ["A", "B", "A_tol1", "A_tol2", "B_tol1", "B_tol2"]
    same_positions = [(v["tile"], v["vertex"], v["row"], v["col"]) for v in V] == \
                     [(v["tile"], v["vertex"], v["row"], v["col"]) for v in RV]
    mism = [{"tile": a["tile"], "vertex": a["vertex"], "column": k, "observed": a[k], "committed": b[k]}
            for a, b in zip(V, RV) for k in cols if a[k] != b[k]]
    per_tile_ab = {t: {k: sum(v[k] for v in V if v["tile"] == t) for k in CLASSICAL} for t in BENCHMARK}
    g["2_AB_vertex_for_vertex"] = {
        "reference_file": REF_VERTICES,
        "columns": cols,
        "vertex_positions_identical": same_positions,
        "mismatches": mism,
        "per_tile_exact_reference_DECISION_LOG": {str(k): v for k, v in vm.AB_REFERENCE_PER_TILE.items()},
        "per_tile_exact_observed": {str(k): v for k, v in per_tile_ab.items()},
        "totals_observed": {k: sum(v[k] for v in V) for k in cols},
        "passed": same_positions and not mism and per_tile_ab == vm.AB_REFERENCE_PER_TILE,
    }
    # Also C/C2 exact columns vertex for vertex vs #7 (not required by #14, stricter than item 1).
    cmism = [{"tile": a["tile"], "vertex": a["vertex"], "column": k} for a, b in zip(V, RV)
             for k in ("C", "C2") if a[k] != b[k]]
    g["2b_C_C2_exact_vertex_for_vertex_vs_7"] = {"mismatches": cmism, "passed": not cmism}

    # 7's tolerance figures for A/B (shared-miss 21 -> ? -> 12; A 23 -> 35, B 25 -> 35).
    tol7 = {}
    for t in TOLS:
        tol7[f"{t}px"] = {
            "A_hits": sum(v[hit_key("A", t)] for v in V),
            "B_hits": sum(v[hit_key("B", t)] for v in V),
            "A_and_B_miss": sum(not v[hit_key("A", t)] and not v[hit_key("B", t)] for v in V),
        }
    with open(REF_ANALYSIS_7) as f:
        a7 = json.load(f)["tolerance_check_A_B"]
    ref_tol7 = {"0px": {"A_or_B_misses": a7["exact"]["A_or_B_misses"]}}
    for t in (1, 2):
        ref_tol7[f"{t}px"] = a7[f"tol{t}px"]
    tol_ok = tol7["0px"]["A_and_B_miss"] == ref_tol7["0px"]["A_or_B_misses"] and all(
        tol7[f"{t}px"]["A_hits"] == ref_tol7[f"{t}px"]["A_hits"]
        and tol7[f"{t}px"]["B_hits"] == ref_tol7[f"{t}px"]["B_hits"]
        and tol7[f"{t}px"]["A_and_B_miss"] == ref_tol7[f"{t}px"]["A_or_B_misses"] for t in (1, 2))
    g["2c_issue7_tolerance_figures"] = {"reference": ref_tol7, "observed": tol7, "passed": tol_ok}

    # 3. Per-tile coverage matches vertices.json (exact-pixel mask area, all four arms).
    rcov = {t["tile"]: t["coverage"] for t in RT}
    diffs = {str(t["tile"]): {k: abs(t["coverage"][k] - rcov[t["tile"]][k]) for k in ARMS} for t in tiles}
    maxdiff = max(d for x in diffs.values() for d in x.values())
    zero_eq = all(t["dilated_coverage"]["0"][k] == t["coverage"][k] for t in tiles for k in ARMS)
    g["3_per_tile_coverage"] = {
        "reference_file": REF_VERTICES, "abs_diff": diffs, "max_abs_diff": maxdiff,
        "dilated_coverage_at_0px_equals_coverage": zero_eq,
        "passed": maxdiff < 1e-12 and zero_eq,
    }

    # Exact-pixel chance baseline reduces to #6/#7's coverage-matched expectation.
    with open(REF_SWEEP) as f:
        sw = json.load(f)
    with open(REF_ANALYSIS_7) as f:
        a7full = json.load(f)
    chk = {}
    for arm, sk, ak in (("C", "merged_final_ep60", "primary_C"), ("C2", "this_run_best_ep7", "secondary_C2")):
        L = lift(V, tiles, arm, 0)
        r7 = a7full[ak]
        exp7 = r7["metric1_shared_miss_vs_chance"]["expected_hits_coverage_matched"] + \
            r7["metric2_caught_set"]["expected_hits_coverage_matched"]
        chk[arm] = {
            "expected_0px": L["expected_hits_tolerance_matched"], "lift_0px": L["lift"],
            "issue7_expected_sum_shared_miss_plus_caught": round(exp7, 4),
            "issue6_random_mask_expected_hits": sw[sk]["0.5"]["random_mask_expected_hits"],
            "issue6_lift": sw[sk]["0.5"]["lift_over_random_mask"],
            "passed": abs(L["expected_hits_tolerance_matched"] - exp7) < 1e-3
            and round(L["expected_hits_tolerance_matched"], 2) == sw[sk]["0.5"]["random_mask_expected_hits"]
            and round(L["lift"], 2) == sw[sk]["0.5"]["lift_over_random_mask"],
        }
    g["3b_exact_pixel_chance_reproduces_6_7"] = {**chk, "passed": all(c["passed"] for c in chk.values())}

    g["dilation_matches_within_tolerance_at_every_vertex"] = {
        "passed": table["dilation_matches_within_tolerance_at_every_vertex"]}

    gate_passed = all(c["passed"] for c in g.values())

    # Power check from marginals only (no joint read).
    marg = {f"{t}px": {k: sum(v[hit_key(k, t)] for v in V) for k in ARMS} for t in TOLS}
    power = {}
    for t in (PRIMARY_TOL, SECONDARY_TOL):
        m = marg[f"{t}px"]
        power[f"{t}px"] = {f"C_vs_{x}": power_from_marginals(m["C"], m[x]) for x in CLASSICAL}
    deciding_reachable = all(power[f"{PRIMARY_TOL}px"][f"C_vs_{x}"]["reachable"] for x in CLASSICAL)

    return {
        "checks": g,
        "gate_passed": gate_passed,
        "marginal_hits": marg,
        "power_check_marginals_only": power,
        "deciding_2px_p_min_below_0.05": deciding_reachable,
        "analysis_allowed": gate_passed and deciding_reachable,
    }


# --------------------------------------------------------------------------
# Step 2: pre-registered analysis
# --------------------------------------------------------------------------

def mc_sign_test(n_c: int, n_x: int, rng) -> float:
    n = n_c + n_x
    if n == 0:
        return 1.0
    return float((rng.binomial(n, 0.5, MC_DRAWS) >= n_c).mean())


def analyze(table: dict, gate: dict) -> dict:
    V, tiles = table["vertices"], table["tiles"]
    rng = np.random.default_rng(SEED)
    out = {"alpha": ALPHA, "primary_tol_px": PRIMARY_TOL, "secondary_tol_px": SECONDARY_TOL}

    per_tol = {}
    for t in TOLS:
        lifts = {k: lift(V, tiles, k, t) for k in ARMS}
        st_c = {x: sign_test(V, "C", x, t) for x in CLASSICAL}
        st_c2 = {x: sign_test(V, "C2", x, t) for x in CLASSICAL}
        for st in list(st_c.values()) + list(st_c2.values()):
            keys = [k for k in st if k.endswith("_only")]
            st["p_one_sided_monte_carlo"] = mc_sign_test(st[keys[0]], st[keys[1]], rng)
        per_tol[f"{t}px"] = {
            "lift_over_tolerance_matched_chance": lifts,
            "sign_tests_C": st_c,
            "sign_tests_C2_descriptive": st_c2,
            "outcome_rule_C": classify(st_c, lifts, "C"),
            "outcome_rule_C2_descriptive": classify(st_c2, lifts, "C2"),
            "sign_test_A_vs_B_descriptive": sign_test(V, "A", "B", t),
        }
    out["by_tolerance"] = per_tol

    # Monte Carlo cross-check of the chance baseline (random positions scored with
    # within_tolerance vs the analytic dilated coverage). |z| is in binomial SEs.
    zs = []
    for t in tiles:
        for tol in TOLS:
            for k in ARMS:
                p = t["dilated_coverage"][str(tol)][k]
                q = t["dilated_coverage_monte_carlo"][str(tol)][k]
                se = (p * (1 - p) / MC_POSITIONS) ** 0.5
                zs.append({"tile": t["tile"], "tol": tol, "arm": k, "exact": round(p, 5),
                           "mc": round(q, 5), "z": round((q - p) / se, 3) if se else 0.0})
    out["chance_baseline_monte_carlo_check"] = {
        "positions_per_tile": MC_POSITIONS,
        "max_abs_z": max(abs(z["z"]) for z in zs),
        "max_abs_diff": max(abs(z["mc"] - z["exact"]) for z in zs),
        "cells": zs,
    }

    # Leave-one-tile-out at 2 px (pre-registered descriptive; feeds outcome (d)).
    loto = {}
    for drop in BENCHMARK:
        vs = [v for v in V if v["tile"] != drop]
        ts = [t for t in tiles if t["tile"] != drop]
        entry = {}
        for c_arm in ("C", "C2"):
            st = {x: sign_test(vs, c_arm, x, PRIMARY_TOL) for x in CLASSICAL}
            L = {k: lift(vs, ts, k, PRIMARY_TOL) for k in ARMS}
            entry[c_arm] = {
                **{f"vs_{x}": {"C_only": st[x][f"{c_arm}_only"], "X_only": st[x][f"{x}_only"],
                               "p": st[x]["p_one_sided"]} for x in CLASSICAL},
                "lifts": {k: L[k]["lift"] for k in ARMS},
                "outcome_rule": classify(st, L, c_arm),
            }
        loto[str(drop)] = entry
    out["leave_one_tile_out_2px"] = loto

    # Per tile, every arm, 0/1/2 px.
    cov = cov_lookup(tiles)
    out["per_tile"] = [{
        "tile": tid, "n": sum(1 for v in V if v["tile"] == tid),
        "hits": {f"{t}px": {k: sum(v[hit_key(k, t)] for v in V if v["tile"] == tid) for k in ARMS} for t in TOLS},
        "dilated_coverage": {f"{t}px": {k: round(cov[tid][str(t)][k], 4) for k in ARMS} for t in TOLS},
    } for tid in BENCHMARK]

    # Residual sets (#7): A/B shared misses at 2 px, and vertices every arm misses at 2 px.
    k2 = lambda k: hit_key(k, PRIMARY_TOL)  # noqa: E731
    sm2 = [v for v in V if not v[k2("A")] and not v[k2("B")]]
    out["residual_sets_2px"] = {
        "AB_shared_miss_2px_n": len(sm2),
        "AB_shared_miss_2px_vertices": [{"tile": v["tile"], "vertex": v["vertex"],
                                         **{k: v[k] for k in ("C", "C2")},
                                         "C_tol2": v["C_tol2"], "C2_tol2": v["C2_tol2"]} for v in sm2],
        "C_hits_exact": sum(v["C"] for v in sm2), "C_hits_2px": sum(v["C_tol2"] for v in sm2),
        "C2_hits_exact": sum(v["C2"] for v in sm2), "C2_hits_2px": sum(v["C2_tol2"] for v in sm2),
        "missed_by_every_arm_and_checkpoint_2px": [
            {"tile": v["tile"], "vertex": v["vertex"], "row": v["row"], "col": v["col"]}
            for v in V if not any(v[k2(k)] for k in ARMS)],
    }

    # Outcome against the pre-registration.
    base2 = per_tol[f"{PRIMARY_TOL}px"]["outcome_rule_C"]
    base1 = per_tol[f"{SECONDARY_TOL}px"]["outcome_rule_C"]
    beats2 = {x: per_tol[f"{PRIMARY_TOL}px"]["sign_tests_C"][x]["p_one_sided"] < ALPHA for x in CLASSICAL}
    flips = {tile: [x for x in CLASSICAL if (e["C"][f"vs_{x}"]["p"] < ALPHA) != beats2[x]]
             for tile, e in loto.items()}
    flips = {k: v for k, v in flips.items() if v}
    d = bool(flips) or base1 != base2
    out["outcome"] = {
        "gate_passed": gate["gate_passed"],
        "p_min_reachable": gate["deciding_2px_p_min_below_0.05"],
        "base_outcome_2px": base2,
        "base_outcome_1px": base1,
        "loto_verdict_flips_2px": flips,
        "one_px_and_two_px_differ": base1 != base2,
        "outcome_d": d,
        "label": ("d (base " + base2 + ")") if d else base2,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.analyze:
        table = build_table()
        with open(VERTICES_JSON, "w") as f:
            json.dump(table, f, indent=1)
        gate = run_gate(table)
        with open(GATE_JSON, "w") as f:
            json.dump(gate, f, indent=2)
        print(json.dumps(gate, indent=2))
        return

    with open(GATE_JSON) as f:
        gate = json.load(f)
    if not gate["analysis_allowed"]:
        raise SystemExit("Validation gate failed or p_min > 0.05: outcome (e); the analysis must not be run (issue #14).")
    with open(VERTICES_JSON) as f:
        table = json.load(f)
    result = analyze(table, gate)
    with open(ANALYSIS_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
