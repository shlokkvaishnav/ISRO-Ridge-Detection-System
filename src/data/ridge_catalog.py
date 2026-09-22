"""Loads the Thompson et al. 2017 lunar wrinkle ridge shapefile.

Uses `pyshp` directly rather than `geopandas`: the shapefile's geometry is
plain polylines and its only attributes are start/end lon/lat (confirmed by
inspecting WRINKLE_RIDGES_README.TXT and the shapefile itself -- there is no
morphology-class field, so the OPEN item flagged in
research/DECISION_LOG.md is resolved: length is the stratification proxy,
not morphology class, because the class field doesn't exist in this product).
`geopandas` would pull in a full GDAL/fiona/shapely stack for functionality
(spatial joins, reprojection) this project doesn't need -- `pyshp` plus a
plain haversine length calculation is enough.

The shapefile's CRS is a plain geographic lon/lat on a sphere of radius
1737400m (see the .PRJ file), matching its own start_lon/end_lon/start_lat/
end_lat fields directly.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import shapefile

MOON_RADIUS_KM = 1737.4


@dataclass
class RidgeSegment:
    id: int
    points: List[Tuple[float, float]]  # (lon, lat) degrees, in order
    length_km: float
    bbox: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * MOON_RADIUS_KM * math.asin(math.sqrt(a))


def _polyline_length_km(points: List[Tuple[float, float]]) -> float:
    return sum(
        _haversine_km(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        for i in range(len(points) - 1)
    )


def load_ridge_catalog(shp_path: str) -> List[RidgeSegment]:
    """Load every ridge segment from the shapefile at `shp_path`
    (a .SHP file; the matching .DBF/.SHX must sit alongside it)."""
    if not os.path.exists(shp_path):
        raise FileNotFoundError(shp_path)

    sf = shapefile.Reader(shp_path)
    segments = []
    for i, shape in enumerate(sf.shapes()):
        points = [(float(x), float(y)) for x, y in shape.points]
        if len(points) < 2:
            continue  # degenerate, no real line to detect

        lons = [p[0] for p in points]
        lats = [p[1] for p in points]
        bbox = (min(lons), min(lats), max(lons), max(lats))

        segments.append(
            RidgeSegment(
                id=i,
                points=points,
                length_km=_polyline_length_km(points),
                bbox=bbox,
            )
        )
    return segments
