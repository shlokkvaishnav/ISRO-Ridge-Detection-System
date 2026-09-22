"""Tests for the Arm B Hessian-filter pipeline, mirroring
tests/test_classical_pipeline.py's synthetic-DEM approach so both arms are
held to the same correctness bar.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.hessian.pipeline import detect_ridges
from src.hessian.ridge_filter import hessian_ridge_response
from tests.test_classical_pipeline import (
    make_curved_ridge_dem,
    make_flat_dem,
    make_straight_ridge_dem,
)


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
        rows_with_ridge = np.where(mask.any(axis=1))[0]
        assert len(rows_with_ridge) > 0
        assert abs(rows_with_ridge.mean() - 64) < 8

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
            deviations.append(abs(rows_hit.mean() - centerline[col]))

        assert len(deviations) > mask.shape[1] * 0.3, (
            "ridge should be detected across a meaningful fraction of columns"
        )
        assert np.mean(deviations) < 8.0


class TestHessianRidgeResponse:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            hessian_ridge_response(np.zeros((10, 10)), method="not-a-real-method")

    def test_response_is_normalized(self):
        dem = make_straight_ridge_dem()
        response = hessian_ridge_response(dem.astype(np.float64))
        assert response.max() <= 1.0
        assert response.min() >= 0.0
