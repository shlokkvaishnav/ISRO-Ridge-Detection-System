"""Stage data/tiles/ + the ground-truth shapefile into a folder shaped for
upload as a Kaggle Dataset (research/arm_c_deep_learning/SPEC.md) -- Arm C
trains on Kaggle's GPU, since this machine has none.

Only dem.tif is needed (Arm C's inputs are DEM + derived aspect, not the
WAC image), so wac.tif is skipped to keep the upload smaller.

Usage:
    python scripts/package_kaggle_dataset.py --out staging/isro-ridge-tiles
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-dir", default="data/tiles")
    parser.add_argument("--shp-dir", default="data/raw/wrinkle_ridges_shapefile")
    parser.add_argument("--out", default="staging/isro-ridge-tiles")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    manifest_path = os.path.join(args.tiles_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    for entry in manifest:
        tid = str(entry["id"])
        dst_dir = os.path.join(args.out, tid)
        os.makedirs(dst_dir, exist_ok=True)
        src_dem = os.path.join(args.tiles_dir, entry["dem_path"])
        dst_dem = os.path.join(dst_dir, "dem.tif")
        shutil.copyfile(src_dem, dst_dem)

    shutil.copyfile(manifest_path, os.path.join(args.out, "manifest.json"))

    shp_dst = os.path.join(args.out, "shapefile")
    os.makedirs(shp_dst, exist_ok=True)
    for fname in os.listdir(args.shp_dir):
        if fname.upper().startswith("WRINKLE_RIDGES_180"):
            shutil.copyfile(os.path.join(args.shp_dir, fname), os.path.join(shp_dst, fname))

    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(args.out)
        for f in fs
    )
    print(f"Staged {len(manifest)} tiles to {args.out} ({total_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
