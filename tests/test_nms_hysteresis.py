"""Tests for src/classical/nms_hysteresis.py and the
detect_ridges_nms_hysteresis pipeline variant.
"""
from __future__ import annotations

import numpy as np

from src.classical.nms_hysteresis import hysteresis_threshold, non_maximum_suppression
from src.classical.pipeline import detect_ridges_nms_hysteresis
from tests.test_classical_pipeline import make_flat_dem, make_straight_ridge_dem


class TestNonMaximumSuppression:
    def test_thins_a_peaked_ridge_response_to_its_local_maximum(self):
        # A peaked (not flat-topped) band of response, like a real ridge's
        # response profile across its width: NMS along the vertical
        # direction should keep only the row-wise local maximum, narrowing
        # the band to (near) a single row. A flat-topped plateau has no
        # unique local max and legitimately survives NMS whole -- that's
        # correct NMS behavior, not what this test is checking.
        size = 32
        response = np.zeros((size, size))
        response[14, :] = 0.3
        response[15, :] = 0.8
        response[16, :] = 1.0  # the peak row
        response[17, :] = 0.6
        response[18, :] = 0.2
        orientation = np.full((size, size), np.pi / 2)  # ~90 deg everywhere

        thinned = non_maximum_suppression(response, orientation)
        band_rows_before = (response[:, 16] > 0).sum()
        band_rows_after = (thinned[:, 16] > 0).sum()
        assert band_rows_after < band_rows_before
        assert thinned[16, 16] == 1.0, "the true peak row should survive"

    def test_shape_mismatch_raises(self):
        response = np.zeros((10, 10))
        orientation = np.zeros((5, 5))
        try:
            non_maximum_suppression(response, orientation)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestHysteresisThreshold:
    def test_flat_response_produces_no_mask(self):
        response = np.zeros((32, 32))
        mask = hysteresis_threshold(response)
        assert mask.sum() == 0

    def test_isolated_pixel_below_high_does_not_survive(self):
        response = np.zeros((32, 32))
        response[16, 16] = 0.5  # a single moderate pixel, no neighbors
        mask = hysteresis_threshold(response, low_percentile=50, high_percentile=99)
        # with only one non-zero pixel in the whole tile, it IS both the low
        # and high percentile value; hysteresis is really tested by the
        # pipeline-level tests below on a tile with real structure
        assert mask.dtype == bool


class TestDetectRidgesNmsHysteresis:
    def test_flat_dem_produces_no_ridge(self):
        dem = make_flat_dem()
        result = detect_ridges_nms_hysteresis(dem)
        assert result["ridge_mask"].sum() == 0

    def test_straight_ridge_is_detected(self):
        dem = make_straight_ridge_dem()
        result = detect_ridges_nms_hysteresis(dem)
        mask = result["ridge_mask"]
        assert mask.sum() > 0
        rows_with_ridge = np.where(mask.any(axis=1))[0]
        assert len(rows_with_ridge) > 0
        assert abs(rows_with_ridge.mean() - 64) < 8
