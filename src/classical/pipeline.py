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
    threshold_percentile: float = 75.0,
    denoise_max_size: int = 5,
    min_length_px: float = 8.0,
    min_eccentricity: float = 0.85,
    gap_link_radius: int = 4,
) -> dict:
    """Run the full classical pipeline on a raw elevation array.

    Defaults updated 2026-09-22 (see research/DECISION_LOG.md) after real-
    tile testing showed the original 90th-percentile-threshold-then-close
    approach missed most of a real ridge's own response values. The current
    defaults trade a looser threshold for shape-based filtering (elongation)
    doing the actual clutter-vs-ridge discrimination -- a real, still
    partial improvement (recall roughly 2-4x across tested tiles), not a
    solved problem; see the decision log for exact numbers.

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
        denoise_max_size=denoise_max_size,
        min_length_px=min_length_px,
        min_eccentricity=min_eccentricity,
        gap_link_radius=gap_link_radius,
    )

    return {
        "slope": slope,
        "grayscale": grayscale,
        "phase_symmetry": phase_symmetry,
        "ridge_mask": ridge_mask,
    }
