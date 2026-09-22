"""Evaluate a trained Arm C model with the same true-ridge-vertex-recall /
mask-coverage metric used for Arm A/B (research/DECISION_LOG.md), so the
three arms are directly comparable rather than comparing Arm C's own
loss/IoU against Arm A/B's recall/coverage numbers, which would not be an
apples-to-apples comparison.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import rasterio
import rasterio.warp
import torch
from skimage.transform import resize

from src.data.ridge_catalog import load_ridge_catalog
from src.data.tile_extraction import RIDGE_CATALOG_CRS
from src.deep.aspect import elevation_to_aspect_variance
from src.deep.model import DualBranchRidgeNet
from src.preprocessing.dem_to_slope import elevation_to_slope, slope_to_grayscale


def evaluate(
    model_path: str,
    manifest_path: str,
    tiles_dir: str,
    shp_path: str,
    tile_ids,
    threshold: float = 0.5,
    target_size: int = 128,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchRidgeNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(manifest_path) as f:
        manifest = {entry["id"]: entry for entry in json.load(f)}
    segments_by_id = {s.id: s for s in load_ridge_catalog(shp_path)}

    total_hits = total_truth = total_pixels = total_area = 0
    per_tile = []

    for tid in tile_ids:
        entry = manifest[tid]
        seg = segments_by_id[tid]
        dem_path = os.path.join(tiles_dir, entry["dem_path"])

        with rasterio.open(dem_path) as src:
            elevation = src.read(1).astype(np.float64)
            transform = src.transform
            crs = src.crs

        native_shape = elevation.shape
        slope = elevation_to_slope(elevation)
        dem_gray = slope_to_grayscale(slope).astype(np.float64) / 255.0
        aspect_var = elevation_to_aspect_variance(elevation)

        dem_r = resize(dem_gray, (target_size, target_size), anti_aliasing=True)
        aspect_r = resize(aspect_var, (target_size, target_size), anti_aliasing=True)
        dem_t = torch.from_numpy(dem_r).float().unsqueeze(0).unsqueeze(0).to(device)
        aspect_t = torch.from_numpy(aspect_r).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(dem_t, aspect_t)).cpu().numpy()[0, 0]

        pred_native = resize(pred, native_shape, anti_aliasing=True)
        mask = pred_native >= threshold

        lons = [p[0] for p in seg.points]
        lats = [p[1] for p in seg.points]
        xs, ys = rasterio.warp.transform(RIDGE_CATALOG_CRS, crs, lons, lats)
        rows, cols = rasterio.transform.rowcol(transform, xs, ys)
        truth = list(zip(rows, cols))

        hits = sum(1 for r, c in truth if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c])
        total_hits += hits
        total_truth += len(truth)
        total_pixels += mask.sum()
        total_area += elevation.size

        per_tile.append(
            {"id": tid, "hits": hits, "truth": len(truth), "coverage_pct": 100 * mask.sum() / elevation.size}
        )

    return {
        "recall": total_hits / total_truth if total_truth else 0.0,
        "recall_frac": f"{total_hits}/{total_truth}",
        "coverage_pct": 100 * total_pixels / total_area if total_area else 0.0,
        "per_tile": per_tile,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="results/arm_c/model.pt")
    parser.add_argument("--manifest", default="data/tiles/manifest.json")
    parser.add_argument("--tiles-dir", default="data/tiles")
    parser.add_argument("--shp", default="data/raw/wrinkle_ridges_shapefile/WRINKLE_RIDGES_180.SHP")
    parser.add_argument("--val-ids", default="results/arm_c/val_ids.json")
    parser.add_argument("--out", default="results/arm_c/eval.json")
    args = parser.parse_args()

    with open(args.val_ids) as f:
        tile_ids = json.load(f)

    result = evaluate(args.model, args.manifest, args.tiles_dir, args.shp, tile_ids)
    print(f"Recall: {result['recall_frac']} ({100*result['recall']:.1f}%)")
    print(f"Coverage: {result['coverage_pct']:.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
