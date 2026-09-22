"""Arm B: Hessian ridge filter (Frangi/Meijering/Sato) + morphology.

Reuses Arm A's threshold/shape-filter/gap-link morphology
(src/classical/morphology.py) and slope preprocessing
(src/preprocessing/dem_to_slope.py) unchanged -- the comparison this repo
is built for is only meaningful if both arms share every stage except the
ridge-detection filter itself.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.classical.morphology import clean_ridge_mask, threshold_response
from src.hessian.ridge_filter import hessian_ridge_response
from src.preprocessing.dem_to_slope import elevation_to_slope, slope_to_grayscale


def detect_ridges(
    elevation: np.ndarray,
    pixel_size: float = 1.0,
    method: str = "frangi",
    sigmas: range = range(1, 6),
    threshold_percentile: float = 75.0,
    denoise_max_size: int = 5,
    min_length_px: float = 8.0,
    min_eccentricity: float = 0.85,
    gap_link_radius: int = 4,
) -> dict:
    """Run the full Hessian-filter pipeline on a raw elevation array.

    Same threshold/shape/gap-link defaults as Arm A (see
    src/classical/pipeline.py, tuned 2026-09-22 against real tiles) -- kept
    identical on purpose so any difference in the eventual comparison is
    attributable to the ridge-detection filter, not to different
    post-processing parameters.
    """
    slope = elevation_to_slope(elevation, pixel_size=pixel_size)
    grayscale = slope_to_grayscale(slope)

    response = hessian_ridge_response(grayscale, method=method, sigmas=sigmas)

    binary = threshold_response(response, percentile=threshold_percentile)
    ridge_mask = clean_ridge_mask(
        binary,
        denoise_max_size=denoise_max_size,
        min_length_px=min_length_px,
        min_eccentricity=min_eccentricity,
        gap_link_radius=gap_link_radius,
    )

    return {
        "slope": slope,
        "grayscale": grayscale,
        "hessian_response": response,
        "ridge_mask": ridge_mask,
    }
