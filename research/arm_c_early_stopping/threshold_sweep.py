"""Threshold sweep for the three Arm C checkpoints on the 6-tile benchmark.

Added in response to review round 1 on PR #6: the fixed 0.5 threshold is the
pre-registered metric, but the best-vs-final comparison is only meaningful if
we know how sensitive each checkpoint is to it. Also reports the mean of
per-tile coverage and recall lift over a random mask of equal coverage
(expected hits = sum(truth_i * coverage_i)).

Run from the repo root (needs data/tiles and the ridge shapefile locally):
    python research/arm_c_early_stopping/threshold_sweep.py
"""
import json

from src.deep.evaluate import evaluate

MANIFEST = "data/tiles/manifest.json"
TILES = "data/tiles"
SHP = "data/raw/wrinkle_ridges_shapefile/WRINKLE_RIDGES_180.SHP"
BENCHMARK = [3674, 1851, 748, 5017, 4098, 3461]
CHECKPOINTS = {
    "merged_final_ep60": "results/arm_c/model.pt",
    "this_run_best_ep7": "results/arm_c_early_stopping/model_best.pt",
    "this_run_final_ep60": "results/arm_c_early_stopping/model.pt",
}
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

out = {}
for name, path in CHECKPOINTS.items():
    out[name] = {}
    for thr in THRESHOLDS:
        r = evaluate(model_path=path, manifest_path=MANIFEST, tiles_dir=TILES,
                     shp_path=SHP, tile_ids=BENCHMARK, threshold=thr)
        tiles = r["per_tile"]
        chance = sum(t["truth"] * t["coverage_pct"] / 100 for t in tiles)
        hits = sum(t["hits"] for t in tiles)
        out[name][str(thr)] = {
            "recall_frac": r["recall_frac"],
            "coverage_pct_pixel_weighted": round(r["coverage_pct"], 2),
            "coverage_pct_mean_per_tile": round(sum(t["coverage_pct"] for t in tiles) / len(tiles), 2),
            "random_mask_expected_hits": round(chance, 2),
            "lift_over_random_mask": round(hits / chance, 2) if chance else None,
        }
        print(name, thr, out[name][str(thr)])

with open("results/arm_c_early_stopping/threshold_sweep.json", "w") as f:
    json.dump(out, f, indent=2)
