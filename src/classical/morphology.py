"""Regional thresholding + morphological cleanup for the classical ridge arm.

Turns a continuous phase-symmetry response map into a binary ridge mask:
threshold -> close small gaps -> remove small noise blobs -> link nearby
fragments into continuous ridge lines.
"""
from __future__ import annotations

import numpy as np
from skimage import morphology


def threshold_response(response: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """Regional threshold: keep pixels above the given percentile of the
    response map, rather than a single global constant, so the cutoff
    adapts to the response range of a given tile.

    Uses a strict ``>`` against a cutoff floored at a small epsilon, not
    ``>=`` against the raw percentile: on a response map with no real signal
    (e.g. a flat DEM), every value is 0, the 90th percentile is 0, and
    ``response >= 0`` would mark the entire tile as ridge -- a degenerate
    case a non-strict comparison on real, noisy tiles wouldn't obviously
    surface.
    """
    cutoff = max(np.percentile(response, percentile), 1e-6)
    return response > cutoff


def clean_ridge_mask(
    binary_mask: np.ndarray,
    closing_radius: int = 2,
    opening_min_size: int = 8,
    gap_link_radius: int = 3,
) -> np.ndarray:
    """Morphological cleanup: close small gaps in ridge lines, remove small
    noise blobs, then link nearby fragments across a larger gap.

    - closing (dilation -> erosion) bridges pixel-scale gaps in a ridge line
    - remove_small_objects (the opening step) strips isolated noise specks
    - a second, larger closing pass is the edge-linking step: it connects
      ridge fragments separated by a small break without over-merging
      unrelated nearby structure (kept smaller than closing_radius would be
      wrong for; gap_link_radius should be tuned to the expected fragment
      gap, not the ridge width)
    """
    closed = morphology.closing(binary_mask, morphology.disk(closing_radius))
    opened = morphology.remove_small_objects(closed, max_size=opening_min_size - 1)
    linked = morphology.closing(opened, morphology.disk(gap_link_radius))
    return linked
