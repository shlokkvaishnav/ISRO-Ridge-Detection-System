"""Three-way (Arm A x Arm B x Arm C) per-vertex map on the 6-tile benchmark.

Pre-registered in issue #7 (research/three_way_vertex_map/SPEC.md). Two steps:

1. ``python research/three_way_vertex_map/vertex_map.py``
   Runs Arm A (default shape-filter ``detect_ridges``), Arm B (frangi, same
   post-processing) and Arm C (both checkpoints, threshold 0.5, same
   resize-to-native procedure as src/deep/evaluate.py) on the 6 benchmark
   tiles, and writes every truth vertex's hit/miss per arm to
   results/three_way_vertex_map/vertices.json (raw data), plus per-tile
   coverage per arm. It then runs the validation gate and writes
   results/three_way_vertex_map/gate.json. It does not compute any Arm C
   statistic.

2. ``python research/three_way_vertex_map/vertex_map.py --analyze``
   Refuses to run unless gate.json says both checks passed. Computes the
   pre-registered metrics from vertices.json alone and writes
   results/three_way_vertex_map/analysis.json.

Hit rule (unchanged from every prior arm comparison): ``mask[r, c]`` at the
truth vertex's reprojected native pixel; out-of-tile vertices are misses.
Needs data/tiles/<id>/ and data/raw/wrinkle_ridges_shapefile/ locally.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from itertools import product

import numpy as np

OUT_DIR = "results/three_way_vertex_map"
MANIFEST = "data/tiles/manifest.json"
TILES_DIR = "data/tiles"
SHP = "data/raw/wrinkle_ridges_shapefile/WRINKLE_RIDGES_180.SHP"
BENCHMARK = [3674, 1851, 748, 5017, 4098, 3461]
THRESHOLD = 0.5
CHECKPOINTS = {
    "C": "results/arm_c/model.pt",  # primary: merged headline checkpoint (epoch 60)
    "C2": "results/arm_c_early_stopping/model_best.pt",  # secondary: epoch 7
}
C_REFERENCE = {
    "C": "results/arm_c/eval.json",
    "C2": "results/arm_c_early_stopping/eval_best.json",
}
# research/DECISION_LOG.md, 2026-09-22 Arm B head-to-head + complementarity entries.
AB_REFERENCE_PER_TILE = {
    3674: {"A": 2, "B": 3},
    1851: {"A": 1, "B": 3},
    748: {"A": 3, "B": 2},
    5017: {"A": 4, "B": 2},
    4098: {"A": 6, "B": 5},
    3461: {"A": 7, "B": 10},
}
AB_REFERENCE_SPLIT = {"both": 12, "A_only": 11, "B_only": 13, "neither": 21}
TOLERANCES_PX = [1, 2]


# --------------------------------------------------------------------------
# Step 1: masks -> per-vertex raw data
# --------------------------------------------------------------------------

def arm_c_mask(model, elevation: np.ndarray, threshold: float, target_size: int = 128) -> np.ndarray:
    """Exactly the per-tile mask computation in src/deep/evaluate.py::evaluate."""
    import torch
    from skimage.transform import resize

    from src.deep.aspect import elevation_to_aspect_variance
    from src.preprocessing.dem_to_slope import elevation_to_slope, slope_to_grayscale

    native_shape = elevation.shape
    slope = elevation_to_slope(elevation)
    dem_gray = slope_to_grayscale(slope).astype(np.float64) / 255.0
    aspect_var = elevation_to_aspect_variance(elevation)
    dem_r = resize(dem_gray, (target_size, target_size), anti_aliasing=True)
    aspect_r = resize(aspect_var, (target_size, target_size), anti_aliasing=True)
    dem_t = torch.from_numpy(dem_r).float().unsqueeze(0).unsqueeze(0)
    aspect_t = torch.from_numpy(aspect_r).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = torch.sigmoid(model(dem_t, aspect_t)).cpu().numpy()[0, 0]
    pred_native = resize(pred, native_shape, anti_aliasing=True)
    return pred_native >= threshold


def within_tolerance(mask: np.ndarray, r: int, c: int, tol: int) -> bool:
    """True if any mask pixel lies within Euclidean distance ``tol`` of (r, c).
    Descriptive confound check only (issue #7, 'Mask shape vs the exact-pixel
    metric'); the deciding metric uses the exact pixel."""
    h, w = mask.shape
    for dr in range(-tol, tol + 1):
        for dc in range(-tol, tol + 1):
            if dr * dr + dc * dc > tol * tol:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                return True
    return False


def build_vertex_table() -> dict:
    import rasterio
    import rasterio.transform
    import rasterio.warp
    import torch

    from src.classical.pipeline import detect_ridges as detect_a
    from src.data.ridge_catalog import load_ridge_catalog
    from src.data.tile_extraction import RIDGE_CATALOG_CRS
    from src.deep.model import DualBranchRidgeNet
    from src.hessian.pipeline import detect_ridges as detect_b

    with open(MANIFEST) as f:
        manifest = {e["id"]: e for e in json.load(f)}
    segments = {s.id: s for s in load_ridge_catalog(SHP)}

    models = {}
    for name, path in CHECKPOINTS.items():
        m = DualBranchRidgeNet()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        models[name] = m

    vertices, tiles = [], []
    for tid in BENCHMARK:
        with rasterio.open(os.path.join(TILES_DIR, manifest[tid]["dem_path"])) as src:
            elevation = src.read(1).astype(np.float64)
            transform, crs = src.transform, src.crs

        masks = {
            "A": detect_a(elevation)["ridge_mask"].astype(bool),
            "B": detect_b(elevation, method="frangi")["ridge_mask"].astype(bool),
        }
        for name, model in models.items():
            masks[name] = arm_c_mask(model, elevation, THRESHOLD)

        seg = segments[tid]
        lons = [p[0] for p in seg.points]
        lats = [p[1] for p in seg.points]
        xs, ys = rasterio.warp.transform(RIDGE_CATALOG_CRS, crs, lons, lats)
        rows, cols = rasterio.transform.rowcol(transform, xs, ys)

        h, w = elevation.shape
        tiles.append({
            "tile": tid,
            "shape": [h, w],
            "coverage": {k: float(m.sum() / m.size) for k, m in masks.items()},
        })
        for i, (r, c) in enumerate(zip(rows, cols)):
            r, c = int(r), int(c)
            inside = 0 <= r < h and 0 <= c < w
            v = {"tile": tid, "vertex": i, "row": r, "col": c, "in_tile": inside}
            for k, m in masks.items():
                v[k] = bool(inside and m[r, c])
            for k in ("A", "B"):
                for tol in TOLERANCES_PX:
                    v[f"{k}_tol{tol}"] = within_tolerance(masks[k], r, c, tol)
            vertices.append(v)

    return {
        "description": "Per-vertex hit/miss for Arms A, B, C (primary) and C2 (secondary) on the "
                       "6-tile benchmark. Hit = mask[row, col] at the reprojected native pixel. "
                       "A_tolN/B_tolN: any A/B mask pixel within N px (Euclidean), descriptive only.",
        "arms": {
            "A": "src.classical.pipeline.detect_ridges, defaults (shape-filter)",
            "B": "src.hessian.pipeline.detect_ridges, method=frangi, defaults",
            "C": f"{CHECKPOINTS['C']} @ threshold {THRESHOLD}",
            "C2": f"{CHECKPOINTS['C2']} @ threshold {THRESHOLD}",
        },
        "tiles": tiles,
        "vertices": vertices,
    }


def run_gate(table: dict) -> dict:
    V = table["vertices"]
    checks = {}

    per_tile = {t: {k: sum(v[k] for v in V if v["tile"] == t) for k in ("A", "B")} for t in BENCHMARK}
    split = {
        "both": sum(v["A"] and v["B"] for v in V),
        "A_only": sum(v["A"] and not v["B"] for v in V),
        "B_only": sum(v["B"] and not v["A"] for v in V),
        "neither": sum(not v["A"] and not v["B"] for v in V),
    }
    checks["AB"] = {
        "reference_per_tile": {str(k): v for k, v in AB_REFERENCE_PER_TILE.items()},
        "observed_per_tile": {str(k): v for k, v in per_tile.items()},
        "reference_split": AB_REFERENCE_SPLIT,
        "observed_split": split,
        "passed": per_tile == AB_REFERENCE_PER_TILE and split == AB_REFERENCE_SPLIT,
    }

    for name, ref_path in C_REFERENCE.items():
        with open(ref_path) as f:
            ref = json.load(f)
        ref_tiles = {t["id"]: {"hits": t["hits"], "truth": t["truth"]} for t in ref["per_tile"]}
        obs_tiles = {
            t: {"hits": sum(v[name] for v in V if v["tile"] == t),
                "truth": sum(1 for v in V if v["tile"] == t)}
            for t in BENCHMARK
        }
        cov_ref = {t["id"]: t["coverage_pct"] for t in ref["per_tile"]}
        cov_obs = {t["tile"]: 100 * t["coverage"][name] for t in table["tiles"]}
        checks[name] = {
            "reference_file": ref_path,
            "reference_per_tile": {str(k): v for k, v in ref_tiles.items()},
            "observed_per_tile": {str(k): v for k, v in obs_tiles.items()},
            "coverage_pct_max_abs_diff": max(abs(cov_ref[t] - cov_obs[t]) for t in BENCHMARK),
            "passed": ref_tiles == obs_tiles,
        }

    checks["all_passed"] = all(c["passed"] for k, c in checks.items() if isinstance(c, dict))
    return checks


# --------------------------------------------------------------------------
# Step 2: pre-registered metrics, from vertices.json alone
# --------------------------------------------------------------------------

def poisson_binomial_pmf(ps) -> np.ndarray:
    pmf = np.zeros(len(ps) + 1)
    pmf[0] = 1.0
    for p in ps:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0] *= 1 - p
    return pmf


def subset_stats(vs, arm: str, coverage: dict) -> dict:
    ps = [coverage[v["tile"]][arm] for v in vs]
    hits = sum(v[arm] for v in vs)
    expected = float(sum(ps))
    pmf = poisson_binomial_pmf(ps)
    return {
        "n": len(vs),
        "hits": hits,
        "hit_rate": hits / len(vs) if vs else None,
        "expected_hits_coverage_matched": round(expected, 4),
        "lift": round(hits / expected, 4) if expected else None,
        "p_one_sided_ge": float(pmf[hits:].sum()),
    }


def mdd_two_proportions(n1: int, n2: int, p: float, z_a: float = 1.959964, z_b: float = 0.841621) -> float:
    """Approximate minimum detectable difference in hit rate between two groups
    (two-sided alpha 0.05, power 0.8, normal approximation at pooled rate p)."""
    return (z_a + z_b) * math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


def analyze_arm(V, tiles, arm: str) -> dict:
    coverage = {t["tile"]: t["coverage"] for t in tiles}
    shared_miss = [v for v in V if not v["A"] and not v["B"]]
    caught = [v for v in V if v["A"] or v["B"]]
    m1 = subset_stats(shared_miss, arm, coverage)
    m2 = subset_stats(caught, arm, coverage)

    cells = {}
    for a, b, c in product([True, False], repeat=3):
        key = f"A{'+' if a else '-'}B{'+' if b else '-'}C{'+' if c else '-'}"
        cells[key] = sum(v["A"] == a and v["B"] == b and v[arm] == c for v in V)

    triple_miss = [{"tile": v["tile"], "vertex": v["vertex"], "row": v["row"], "col": v["col"]}
                   for v in shared_miss if not v[arm]]
    c_only = [{"tile": v["tile"], "vertex": v["vertex"]} for v in shared_miss if v[arm]]

    per_tile = []
    for t in BENCHMARK:
        tv = [v for v in V if v["tile"] == t]
        sm = [v for v in tv if not v["A"] and not v["B"]]
        ca = [v for v in tv if v["A"] or v["B"]]
        per_tile.append({
            "tile": t,
            "n": len(tv),
            "coverage": {k: round(coverage[t][k], 4) for k in ("A", "B", arm)},
            "shared_miss_n": len(sm),
            "shared_miss_C_hits": sum(v[arm] for v in sm),
            "shared_miss_expected": round(len(sm) * coverage[t][arm], 3),
            "caught_n": len(ca),
            "caught_C_hits": sum(v[arm] for v in ca),
            "caught_expected": round(len(ca) * coverage[t][arm], 3),
            "triple_miss": len(sm) - sum(v[arm] for v in sm),
        })

    pooled = (m1["hits"] + m2["hits"]) / (m1["n"] + m2["n"])
    return {
        "metric1_shared_miss_vs_chance": m1,
        "metric2_caught_set": m2,
        "lift_ratio_shared_miss_over_caught": round(m1["lift"] / m2["lift"], 4) if m1["lift"] and m2["lift"] else None,
        "raw_hit_rate_difference_caught_minus_shared_miss": round(m2["hit_rate"] - m1["hit_rate"], 4),
        "approx_minimum_detectable_hit_rate_difference": round(mdd_two_proportions(m1["n"], m2["n"], pooled), 4),
        "eight_cell_table": cells,
        "triple_miss_vertices": triple_miss,
        "shared_miss_caught_by_C": c_only,
        "per_tile": per_tile,
    }


def tolerance_check(V) -> dict:
    out = {"exact": {"A_or_B_misses": sum(not v["A"] and not v["B"] for v in V)}}
    for tol in TOLERANCES_PX:
        out[f"tol{tol}px"] = {
            "A_hits": sum(v[f"A_tol{tol}"] for v in V),
            "B_hits": sum(v[f"B_tol{tol}"] for v in V),
            "A_or_B_misses": sum(not v[f"A_tol{tol}"] and not v[f"B_tol{tol}"] for v in V),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not args.analyze:
        table = build_vertex_table()
        with open(os.path.join(OUT_DIR, "vertices.json"), "w") as f:
            json.dump(table, f, indent=1)
        gate = run_gate(table)
        with open(os.path.join(OUT_DIR, "gate.json"), "w") as f:
            json.dump(gate, f, indent=2)
        print(json.dumps(gate, indent=2))
        return

    with open(os.path.join(OUT_DIR, "gate.json")) as f:
        if not json.load(f)["all_passed"]:
            raise SystemExit("Validation gate failed; the analysis must not be run (issue #7).")
    with open(os.path.join(OUT_DIR, "vertices.json")) as f:
        table = json.load(f)
    V, tiles = table["vertices"], table["tiles"]
    result = {
        "primary_C": analyze_arm(V, tiles, "C"),
        "secondary_C2": analyze_arm(V, tiles, "C2"),
        "tolerance_check_A_B": tolerance_check(V),
    }
    with open(os.path.join(OUT_DIR, "analysis.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
