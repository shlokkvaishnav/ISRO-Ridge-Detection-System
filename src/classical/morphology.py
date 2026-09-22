"""Regional thresholding + morphological cleanup for the classical ridge arm.

Turns a continuous phase-symmetry response map into a binary ridge mask.

The pipeline order matters and was chosen empirically, not assumed (see
research/DECISION_LOG.md): denoise -> shape-filter individual small
fragments (favoring elongated, line-like components over blobs) -> only
THEN gap-link. Filtering shape before gap-linking is required -- the
opposite order (gap-link first) fuses most of a real, noisy tile's
thresholded pixels into one sprawling connected blob before shape filtering
ever runs, at which point per-component shape discrimination can't do
anything (it's all one component).
"""
from __future__ import annotations

import numpy as np
from skimage import morphology
from skimage.measure import label, regionprops


def threshold_response(response: np.ndarray, percentile: float = 75.0) -> np.ndarray:
    """Regional threshold: keep pixels above the given percentile of the
    response map, rather than a single global constant, so the cutoff
    adapts to the response range of a given tile.

    Default lowered from 90th to 75th percentile (see
    research/DECISION_LOG.md, 2026-09-22): on real terrain, the true ridge's
    own response values are often only moderately elevated, not top-decile,
    so a 90th-percentile cutoff missed most of the true ridge. The looser
    75th-percentile cutoff admits far more clutter too -- shape filtering in
    `clean_ridge_mask` is what actually separates ridge from clutter, not
    this threshold alone.

    Uses a strict ``>`` against a cutoff floored at a small epsilon, not
    ``>=`` against the raw percentile: on a response map with no real signal
    (e.g. a flat DEM), every value is 0, the Nth percentile is 0, and
    ``response >= 0`` would mark the entire tile as ridge.
    """
    cutoff = max(np.percentile(response, percentile), 1e-6)
    return response > cutoff


def filter_by_shape(
    mask: np.ndarray, min_length_px: float = 8.0, min_eccentricity: float = 0.85
) -> np.ndarray:
    """Keep only connected components shaped like a ridge fragment: long and
    elongated (high eccentricity), not blob-like terrain-roughness clutter.

    Must run on the *raw*, not-yet-gap-linked mask -- gap-linking first
    would merge many small components (some ridge, mostly clutter on real
    terrain) into one large sprawling shape that this function can no
    longer discriminate.
    """
    labeled = label(mask, connectivity=2)
    keep = np.zeros_like(mask)
    for region in regionprops(labeled):
        if region.axis_major_length >= min_length_px and region.eccentricity >= min_eccentricity:
            keep[labeled == region.label] = True
    return keep


def clean_ridge_mask(
    binary_mask: np.ndarray,
    denoise_max_size: int = 5,
    min_length_px: float = 8.0,
    min_eccentricity: float = 0.85,
    gap_link_radius: int = 4,
) -> np.ndarray:
    """Denoise -> shape-filter -> gap-link, in that order (see module
    docstring for why the order matters).

    - `remove_small_objects` strips single/few-pixel noise specks before
      shape filtering even has to consider them
    - `filter_by_shape` keeps only elongated, ridge-fragment-shaped
      components, run on still-separate (not yet gap-linked) components
    - a final closing pass links the surviving ridge-shaped fragments
      across small gaps into continuous lines
    """
    denoised = morphology.remove_small_objects(binary_mask, max_size=denoise_max_size)
    shaped = filter_by_shape(denoised, min_length_px=min_length_px, min_eccentricity=min_eccentricity)
    linked = morphology.closing(shaped, morphology.disk(gap_link_radius))
    return linked
