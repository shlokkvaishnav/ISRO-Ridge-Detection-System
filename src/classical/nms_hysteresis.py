"""Non-maximum suppression + hysteresis thresholding — an alternative to
`morphology.threshold_response` + `morphology.filter_by_shape`, evaluated in
research/nms_hysteresis_threshold/ (see SPEC.md and PR for the question this
answers, and research/DECISION_LOG.md for the result).

Standard Canny-style mechanism, applied to a ridge-response map (phase
symmetry or a Hessian filter) instead of a gradient-magnitude edge map:
thin the response to single-pixel-wide candidate lines by suppressing
non-maximal pixels along the local ridge-normal direction, then use a
high/low threshold pair so isolated pixels need to connect to a strong seed
to survive, rather than surviving on a flat single cutoff.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import apply_hysteresis_threshold


def non_maximum_suppression(response: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Suppress pixels that are not a local maximum along the direction
    given by `orientation` (radians, e.g. from
    `phase_symmetry.monogenic_phase_symmetry(..., return_orientation=True)`).

    Quantizes orientation into 4 bins (0/45/90/135 degrees), the same
    approach Canny edge detection uses, rather than interpolating along the
    exact angle -- simpler and robust enough at this resolution.
    """
    if response.shape != orientation.shape:
        raise ValueError("response and orientation must have the same shape")

    angle_deg = np.rad2deg(orientation) % 180.0
    suppressed = np.zeros_like(response)

    def shifted(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
        return np.roll(np.roll(arr, dr, axis=0), dc, axis=1)

    bin_masks_and_shifts = [
        ((angle_deg < 22.5) | (angle_deg >= 157.5), (0, 1)),   # ~0 deg: horizontal neighbors
        ((angle_deg >= 22.5) & (angle_deg < 67.5), (1, 1)),    # ~45 deg: diagonal
        ((angle_deg >= 67.5) & (angle_deg < 112.5), (1, 0)),   # ~90 deg: vertical neighbors
        ((angle_deg >= 112.5) & (angle_deg < 157.5), (1, -1)), # ~135 deg: anti-diagonal
    ]

    for mask, (dr, dc) in bin_masks_and_shifts:
        n1 = shifted(response, dr, dc)
        n2 = shifted(response, -dr, -dc)
        keep = mask & (response >= n1) & (response >= n2)
        suppressed[keep] = response[keep]

    # np.roll wraps around at the border, so border neighbors are invalid --
    # never let a border pixel survive NMS on a wrapped comparison.
    suppressed[0, :] = 0
    suppressed[-1, :] = 0
    suppressed[:, 0] = 0
    suppressed[:, -1] = 0

    return suppressed


def hysteresis_threshold(
    response: np.ndarray, low_percentile: float = 60.0, high_percentile: float = 85.0
) -> np.ndarray:
    """Percentile-parameterized wrapper around
    `skimage.filters.apply_hysteresis_threshold` (which takes absolute
    values): pixels above `high_percentile` seed a region; pixels above
    `low_percentile` extend an already-seeded region but cannot start one
    on their own.
    """
    low = np.percentile(response, low_percentile)
    high = max(np.percentile(response, high_percentile), low + 1e-6)
    return apply_hysteresis_threshold(response, low, high)
