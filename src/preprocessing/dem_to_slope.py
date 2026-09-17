"""Convert a DEM elevation array into a slope map.

All three detection arms (classical, Hessian, deep) consume the same slope
map, so that comparisons isolate the ridge-detection step itself rather than
differences in preprocessing.
"""
from __future__ import annotations

import numpy as np


def elevation_to_slope(
    elevation: np.ndarray,
    pixel_size: float = 1.0,
) -> np.ndarray:
    """Compute a slope-magnitude map from an elevation array.

    Parameters
    ----------
    elevation:
        2D array of elevation values (meters), e.g. loaded from a DEM GeoTIFF.
    pixel_size:
        Ground distance per pixel (meters), used to scale the gradient into a
        true slope rather than a raw per-pixel difference. Defaults to 1.0
        for synthetic/test data where the scale doesn't matter.

    Returns
    -------
    Slope magnitude array, same shape as ``elevation``, in units of
    (elevation unit) per (pixel_size unit) horizontally.
    """
    if elevation.ndim != 2:
        raise ValueError(f"expected a 2D elevation array, got shape {elevation.shape}")

    dy, dx = np.gradient(elevation, pixel_size)
    slope = np.sqrt(dx**2 + dy**2)
    return slope


def slope_to_grayscale(slope: np.ndarray) -> np.ndarray:
    """Normalize a slope map to uint8 grayscale (0-255) for filters that
    expect an image rather than raw physical units."""
    lo, hi = np.percentile(slope, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(slope, dtype=np.uint8)
    clipped = np.clip(slope, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    return (normalized * 255).astype(np.uint8)
