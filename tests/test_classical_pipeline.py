"""Tests for the Arm A classical pipeline, on synthetic DEMs with a known
ground-truth ridge (or no ridge at all), so correctness can be checked
directly rather than only visually inspected.
"""
from __future__ import annotations

import numpy as np
import pytest
from skimage.measure import label

from src.classical.morphology import clean_ridge_mask
from src.classical.pipeline import detect_ridges


def make_flat_dem(size: int = 128, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """A genuinely flat DEM (no relief at all). Note: this deliberately does
    NOT add realistic sensor noise -- see
    research/DECISION_LOG.md#flat-tile-false-positives for why small numeric
    noise on a flat tile is a known false-positive risk for this pipeline
    (percentile-based grayscale contrast-stretch turns tiny noise into an
    apparently full-range signal), not something this test should paper
    over by picking a noise level that happens to pass.
    """
    if noise == 0.0:
        return np.zeros((size, size))
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, noise, size=(size, size))


def make_straight_ridge_dem(
    size: int = 128, ridge_row: int = 64, width: float = 3.0, height: float = 10.0
) -> np.ndarray:
    y, _x = np.mgrid[0:size, 0:size]
    return height * np.exp(-((y - ridge_row) ** 2) / (2 * width**2))


def make_curved_ridge_dem(
    size: int = 128, amplitude: float = 15.0, width: float = 3.0, height: float = 10.0
):
    y, x = np.mgrid[0:size, 0:size]
    centerline = size / 2 + amplitude * np.sin(2 * np.pi * x[0] / size)
    dem = height * np.exp(-((y - centerline[np.newaxis, :]) ** 2) / (2 * width**2))
    return dem, centerline


class TestDetectRidges:
    def test_flat_dem_produces_no_ridge(self):
        dem = make_flat_dem()
        result = detect_ridges(dem)
        assert result["ridge_mask"].sum() == 0

    def test_straight_ridge_is_detected(self):
        dem = make_straight_ridge_dem()
        result = detect_ridges(dem)
        mask = result["ridge_mask"]
        assert mask.sum() > 0
        # every detected row should be near the true ridge row (64 +/- a few px)
        rows_with_ridge = np.where(mask.any(axis=1))[0]
        assert len(rows_with_ridge) > 0
        assert abs(rows_with_ridge.mean() - 64) < 5

    def test_curved_ridge_centerline_is_tracked(self):
        dem, centerline = make_curved_ridge_dem()
        result = detect_ridges(dem)
        mask = result["ridge_mask"]
        assert mask.sum() > 0

        deviations = []
        for col in range(mask.shape[1]):
            rows_hit = np.where(mask[:, col])[0]
            if len(rows_hit) == 0:
                continue
            detected_row = rows_hit.mean()
            deviations.append(abs(detected_row - centerline[col]))

        assert len(deviations) > mask.shape[1] * 0.5, (
            "ridge should be detected across at least half the columns"
        )
        assert np.mean(deviations) < 6.0


class TestCleanRidgeMask:
    def test_closing_bridges_a_gapped_ridge_line(self):
        # A 5px-thick line, not 1px: a disk structuring element erodes away
        # anything thinner than its own radius regardless of what dilation
        # bridged first, so a 1px fixture would fail for a reason that has
        # nothing to do with gap-linking. Real ridge-mask fragments (from
        # thresholded phase-symmetry response) are never 1px thick, so this
        # matches what the pipeline actually produces. Thickness/gap/radius
        # chosen from an empirical check (see PR discussion / run log), not
        # hand-derived disk geometry -- the pinch-point math for a closing
        # bridge is sensitive to discretization.
        size = 64
        mask = np.zeros((size, size), dtype=bool)
        mask[28:33, 5:25] = True
        mask[28:33, 30:55] = True  # a 5px gap at columns 25-29

        assert label(mask, connectivity=2).max() == 2, "fixture should start as two components"

        cleaned = clean_ridge_mask(mask, closing_radius=1, opening_min_size=1, gap_link_radius=5)
        assert label(cleaned, connectivity=2).max() == 1, "gap should be bridged into one component"

    def test_opening_removes_small_noise_blobs(self):
        size = 64
        mask = np.zeros((size, size), dtype=bool)
        mask[10, 10] = True  # single-pixel noise speck
        mask[40:45, 40] = True  # a real 5px ridge fragment

        cleaned = clean_ridge_mask(mask, closing_radius=1, opening_min_size=4, gap_link_radius=1)
        assert not cleaned[10, 10]
        assert cleaned[40:45, 40].any()
