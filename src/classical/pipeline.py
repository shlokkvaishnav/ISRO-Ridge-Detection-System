"""Arm A: classical phase-symmetry + morphology ridge detection pipeline.

This is the reference/baseline arm — the field-standard approach (DEM ->
slope -> phase symmetry -> morphology) that Arm B (Hessian filters) and
Arm C (deep learning) are compared against.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.classical.morphology import clean_ridge_mask, threshold_response
from src.classical.phase_symmetry import monogenic_phase_symmetry
from src.preprocessing.dem_to_slope import elevation_to_slope, slope_to_grayscale


def detect_ridges(
    elevation: np.ndarray,
    pixel_size: float = 1.0,
    phase_symmetry_kwargs: Optional[dict] = None,
    threshold_percentile: float = 90.0,
    closing_radius: int = 2,
    opening_min_size: int = 8,
    gap_link_radius: int = 3,
) -> dict:
    """Run the full classical pipeline on a raw elevation array.

    Returns every intermediate stage (not just the final mask) so a caller
    can inspect or visualize the pipeline for debugging and for the
    cross-arm comparison this repo is ultimately built for.
    """
    slope = elevation_to_slope(elevation, pixel_size=pixel_size)
    grayscale = slope_to_grayscale(slope)

    ps_kwargs = phase_symmetry_kwargs or {}
    phase_symmetry = monogenic_phase_symmetry(grayscale.astype(np.float64), **ps_kwargs)

    binary = threshold_response(phase_symmetry, percentile=threshold_percentile)
    ridge_mask = clean_ridge_mask(
        binary,
        closing_radius=closing_radius,
        opening_min_size=opening_min_size,
        gap_link_radius=gap_link_radius,
    )

    return {
        "slope": slope,
        "grayscale": grayscale,
        "phase_symmetry": phase_symmetry,
        "ridge_mask": ridge_mask,
    }
