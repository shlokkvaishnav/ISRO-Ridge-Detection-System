"""PyTorch Dataset for Arm C: loads each tile's DEM, computes the
aspect-variance channel and a weak polyline-buffer label mask, resizes
everything to a fixed size for batching (tiles vary in native size; DBR-Net
sidesteps this with a fixed sliding-window crop over a much larger training
image -- this project's tiles are already individually cropped per ridge
segment, so a resize is the simpler fit here, at the cost of some aspect-
ratio distortion on non-square tiles, noted as a limitation in SPEC.md).
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import rasterio
import rasterio.warp
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

from src.data.ridge_catalog import RidgeSegment, load_ridge_catalog
from src.data.tile_extraction import RIDGE_CATALOG_CRS
from src.deep.aspect import elevation_to_aspect_variance
from src.deep.weak_labels import polyline_to_mask
from src.preprocessing.dem_to_slope import elevation_to_slope, slope_to_grayscale


class RidgeTileDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        tiles_dir: str,
        shp_path: str,
        target_size: int = 128,
        half_width_px: int = 18,
    ):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.tiles_dir = tiles_dir
        self.target_size = target_size
        self.half_width_px = half_width_px

        segments_by_id = {s.id: s for s in load_ridge_catalog(shp_path)}
        self.segments: List[RidgeSegment] = [segments_by_id[entry["id"]] for entry in self.manifest]

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        entry = self.manifest[idx]
        seg = self.segments[idx]
        dem_path = os.path.join(self.tiles_dir, entry["dem_path"])

        with rasterio.open(dem_path) as src:
            elevation = src.read(1).astype(np.float64)
            transform = src.transform
            crs = src.crs

        aspect_var = elevation_to_aspect_variance(elevation)
        slope = elevation_to_slope(elevation)
        dem_gray = slope_to_grayscale(slope).astype(np.float64) / 255.0

        lons = [p[0] for p in seg.points]
        lats = [p[1] for p in seg.points]
        xs, ys = rasterio.warp.transform(RIDGE_CATALOG_CRS, crs, lons, lats)
        rows, cols = rasterio.transform.rowcol(transform, xs, ys)
        pixel_points = list(zip(rows, cols))
        label = polyline_to_mask(elevation.shape, pixel_points, half_width_px=self.half_width_px)

        dem_r = resize(dem_gray, (self.target_size, self.target_size), anti_aliasing=True)
        aspect_r = resize(aspect_var, (self.target_size, self.target_size), anti_aliasing=True)
        label_r = resize(
            label.astype(np.float64), (self.target_size, self.target_size), order=0, anti_aliasing=False
        )

        dem_t = torch.from_numpy(dem_r).float().unsqueeze(0)
        aspect_t = torch.from_numpy(aspect_r).float().unsqueeze(0)
        label_t = torch.from_numpy(label_r).float().unsqueeze(0)

        return dem_t, aspect_t, label_t, entry["id"]
