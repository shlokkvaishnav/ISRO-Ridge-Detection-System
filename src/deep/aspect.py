"""Aspect (slope direction) + a local circular-variance signal, as the
second input channel for the dual-branch model -- Arm C's analogue of
DBR-Net's "aspect data ... after applying variance filtering" (Lu et al.
2025, Sec. 2; see research/RELATED_WORK.md and
research/arm_c_deep_learning/SPEC.md).

The paper describes computing aspect from the DEM then applying a GDAL
variance filter, without giving the exact filter window/parameters -- this
module is this project's own interpretation of that description (local
circular variance of the aspect angle, via a uniform filter over cos/sin),
not a verbatim reproduction. Circular variance is the natural choice here
because aspect is an angle: a naive linear variance of the raw angle value
would spuriously blow up right at the 0/360-degree wraparound, which has
nothing to do with real terrain roughness.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def elevation_to_aspect(elevation: np.ndarray, pixel_size: float = 1.0) -> np.ndarray:
    """Direction of steepest ascent at each pixel, in radians (-pi, pi],
    via the same gradient computation dem_to_slope.elevation_to_slope uses
    for magnitude."""
    dy, dx = np.gradient(elevation, pixel_size)
    return np.arctan2(dy, dx)


def aspect_circular_variance(aspect: np.ndarray, window: int = 3) -> np.ndarray:
    """Local circular variance of the aspect field: 0 where nearby aspect
    values all point the same direction (smooth terrain), approaching 1
    where they're locally incoherent/discontinuous (ridge/valley boundaries,
    roughness) -- the edge-information signal DBR-Net's variance-filtered
    aspect branch is meant to provide.
    """
    cos_mean = uniform_filter(np.cos(aspect), size=window)
    sin_mean = uniform_filter(np.sin(aspect), size=window)
    resultant_length = np.sqrt(cos_mean**2 + sin_mean**2)
    return 1.0 - resultant_length


def elevation_to_aspect_variance(
    elevation: np.ndarray, pixel_size: float = 1.0, window: int = 3
) -> np.ndarray:
    """Convenience wrapper: elevation straight to the variance-filtered
    aspect signal used as the model's second input channel."""
    aspect = elevation_to_aspect(elevation, pixel_size=pixel_size)
    return aspect_circular_variance(aspect, window=window)
