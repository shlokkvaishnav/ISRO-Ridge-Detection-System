"""Build the pilot tile dataset: select ridge segments from the Thompson et
al. 2017 catalog via stratified sampling on length, extract a co-registered
DEM + WAC image tile pair for each directly from the remote global mosaics
(no full download -- see research/DECISION_LOG.md), and write a manifest.

Usage:
    python scripts/build_tile_dataset.py --n 30
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ridge_catalog import load_ridge_catalog  # noqa: E402
from src.data.tile_extraction import (  # noqa: E402
    extract_tile_pair,
    open_gld100_dem,
    open_wac_mosaic,
)

SHP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "raw",
    "wrinkle_ridges_shapefile",
    "WRINKLE_RIDGES_180.SHP",
)
TILES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tiles"
)


def stratified_sample_by_length(segments, n: int, seed: int = 0):
    """Sample n segments spread across quartiles of the length distribution,
    so the pilot isn't accidentally all-short or all-long ridges."""
    rng = random.Random(seed)
    lengths = sorted(s.length_km for s in segments)
    q = [lengths[int(p * (len(lengths) - 1))] for p in (0.0, 0.25, 0.5, 0.75, 1.0)]

    buckets = [[] for _ in range(4)]
    for s in segments:
        for i in range(4):
            lo, hi = q[i], q[i + 1]
            if lo <= s.length_km <= hi:
                buckets[i].append(s)
                break

    per_bucket = n // 4
    remainder = n - per_bucket * 4
    sample = []
    for i, bucket in enumerate(buckets):
        take = per_bucket + (1 if i < remainder else 0)
        take = min(take, len(bucket))
        sample.extend(rng.sample(bucket, take))
    return sample


def save_geotiff(path: str, data: np.ndarray, transform, crs, dtype) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data.astype(dtype), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=30,
        help="target TOTAL tile count. Existing tiles in the manifest are kept "
        "untouched and count toward this total -- only the shortfall is newly "
        "sampled and extracted, so re-running with a larger --n grows the "
        "dataset in place instead of overwriting it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad-km", type=float, default=2.0)
    args = parser.parse_args()

    if not os.path.exists(SHP_PATH):
        raise SystemExit(
            f"{SHP_PATH} not found. Run scripts/fetch_lroc_globals.py first "
            "(only the small shapefile needs downloading -- the rasters are "
            "read remotely)."
        )

    os.makedirs(TILES_DIR, exist_ok=True)
    manifest_path = os.path.join(TILES_DIR, "manifest.json")
    manifest = []
    existing_ids = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        existing_ids = {entry["id"] for entry in manifest}
        print(f"Found existing manifest with {len(existing_ids)} tiles -- keeping them untouched")

    n_needed = max(args.n - len(existing_ids), 0)
    if n_needed == 0:
        print(f"Already have {len(existing_ids)} >= requested {args.n} tiles, nothing to do")
        return

    print(f"Loading ridge catalog from {SHP_PATH}")
    segments = load_ridge_catalog(SHP_PATH)
    print(f"  {len(segments)} segments loaded")

    candidates = [s for s in segments if s.id not in existing_ids]
    sample = stratified_sample_by_length(candidates, n_needed, seed=args.seed)
    print(f"  sampled {len(sample)} NEW segments across length quartiles (excluding {len(existing_ids)} already extracted)")

    print("Opening remote rasters (no full download, windowed reads only)...")
    with open_gld100_dem() as dem_src, open_wac_mosaic() as wac_src:
        for seg in sample:
            tile_dir = os.path.join(TILES_DIR, str(seg.id))
            os.makedirs(tile_dir, exist_ok=True)

            tiles = extract_tile_pair(dem_src, wac_src, seg.bbox, pad_km=args.pad_km)

            dem_path = os.path.join(tile_dir, "dem.tif")
            wac_path = os.path.join(tile_dir, "wac.tif")
            save_geotiff(dem_path, tiles["dem"], tiles["dem_transform"], tiles["dem_crs"], "float32")
            save_geotiff(wac_path, tiles["wac"], tiles["wac_transform"], tiles["wac_crs"], "uint8")

            manifest.append(
                {
                    "id": seg.id,
                    "length_km": seg.length_km,
                    "bbox": seg.bbox,
                    "n_vertices": len(seg.points),
                    "dem_path": os.path.relpath(dem_path, TILES_DIR),
                    "wac_path": os.path.relpath(wac_path, TILES_DIR),
                    "dem_shape": list(tiles["dem"].shape),
                    "wac_shape": list(tiles["wac"].shape),
                }
            )
            print(f"  segment {seg.id}: length={seg.length_km:.2f}km, dem={tiles['dem'].shape}, wac={tiles['wac'].shape}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path} ({len(manifest)} tiles total, {len(sample)} newly added)")


if __name__ == "__main__":
    main()
