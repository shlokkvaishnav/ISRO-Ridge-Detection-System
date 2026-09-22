"""Extract co-registered DEM + WAC image tiles for a ridge segment's bounding
box, reading directly from the remote global mosaics via GDAL's /vsicurl/
(no full download -- see research/DECISION_LOG.md).

The WAC mosaic and GLD100 DEM use *different* map projections (different
central meridian: 0 deg vs 180 deg -- confirmed by inspecting both rasters'
CRS directly, not assumed). This module always converts the shapefile's
lon/lat bbox into each raster's own CRS independently via
`rasterio.warp.transform`, and never assumes the two rasters share pixel
space.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import rasterio
import rasterio.warp
from rasterio.crs import CRS
from rasterio.windows import Window

MOON_RADIUS_KM = 1737.4

# The shapefile's own CRS (see its .PRJ: a plain geographic lon/lat on a
# sphere of radius 1737400m, Greenwich prime meridian).
RIDGE_CATALOG_CRS = CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")

WAC_MOSAIC_URL = "/vsicurl/https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
GLD100_DEM_URL = "/vsicurl/https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_WAC_GLD100_DTM_79S79N_100m_v1.1.tif"

MIN_TILE_PX = 32


def open_wac_mosaic() -> rasterio.DatasetReader:
    return rasterio.open(WAC_MOSAIC_URL)


def open_gld100_dem() -> rasterio.DatasetReader:
    return rasterio.open(GLD100_DEM_URL)


def _pad_bbox_deg(
    bbox: Tuple[float, float, float, float], pad_km: float
) -> Tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    mean_lat = (min_lat + max_lat) / 2
    km_per_deg_lat = MOON_RADIUS_KM * math.pi / 180
    km_per_deg_lon = km_per_deg_lat * max(math.cos(math.radians(mean_lat)), 1e-6)

    pad_lat = pad_km / km_per_deg_lat
    pad_lon = pad_km / km_per_deg_lon
    return (min_lon - pad_lon, min_lat - pad_lat, max_lon + pad_lon, max_lat + pad_lat)


def _bbox_to_window(
    bbox_deg: Tuple[float, float, float, float], src: rasterio.DatasetReader
) -> Window:
    min_lon, min_lat, max_lon, max_lat = bbox_deg
    xs, ys = rasterio.warp.transform(
        RIDGE_CATALOG_CRS, src.crs, [min_lon, max_lon], [min_lat, max_lat]
    )
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    row_top, col_left = src.index(x0, y1)
    row_bottom, col_right = src.index(x1, y0)
    row0, row1 = sorted((row_top, row_bottom))
    col0, col1 = sorted((col_left, col_right))

    if row1 - row0 < MIN_TILE_PX:
        deficit = MIN_TILE_PX - (row1 - row0)
        row0 -= deficit // 2
        row1 += deficit - deficit // 2
    if col1 - col0 < MIN_TILE_PX:
        deficit = MIN_TILE_PX - (col1 - col0)
        col0 -= deficit // 2
        col1 += deficit - deficit // 2

    row0 = max(row0, 0)
    col0 = max(col0, 0)
    row1 = min(row1, src.height)
    col1 = min(col1, src.width)

    return Window(col0, row0, col1 - col0, row1 - row0)


def extract_tile_pair(
    dem_src: rasterio.DatasetReader,
    wac_src: rasterio.DatasetReader,
    bbox: Tuple[float, float, float, float],
    pad_km: float = 2.0,
) -> dict:
    """Returns a dict with `dem`, `wac` (numpy arrays) and each one's own
    `dem_transform`/`wac_transform` + `dem_crs`/`wac_crs`, extracted
    independently in each raster's own CRS."""
    padded = _pad_bbox_deg(bbox, pad_km)

    dem_window = _bbox_to_window(padded, dem_src)
    wac_window = _bbox_to_window(padded, wac_src)

    dem_data = dem_src.read(1, window=dem_window).astype(np.float64)
    wac_data = wac_src.read(1, window=wac_window)

    return {
        "dem": dem_data,
        "dem_transform": dem_src.window_transform(dem_window),
        "dem_crs": dem_src.crs,
        "wac": wac_data,
        "wac_transform": wac_src.window_transform(wac_window),
        "wac_crs": wac_src.crs,
    }
