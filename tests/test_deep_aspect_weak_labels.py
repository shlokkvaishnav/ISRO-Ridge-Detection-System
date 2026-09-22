"""Tests for src/deep/aspect.py and src/deep/weak_labels.py -- the
non-torch parts of Arm C (see research/arm_c_deep_learning/SPEC.md), which
can be tested on this machine without a GPU.
"""
from __future__ import annotations

import numpy as np

from src.deep.aspect import aspect_circular_variance, elevation_to_aspect, elevation_to_aspect_variance
from src.deep.weak_labels import polyline_to_mask


class TestAspect:
    def test_flat_dem_has_uniform_aspect_variance(self):
        # A perfectly flat DEM has an undefined/degenerate gradient
        # everywhere (all zero), so aspect is arbitrary but constant --
        # circular variance should be ~0 (fully coherent, even if the
        # "direction" itself is meaningless on a flat plane).
        dem = np.zeros((32, 32))
        var = elevation_to_aspect_variance(dem)
        assert var.shape == dem.shape
        assert np.allclose(var, 0.0, atol=1e-6)

    def test_aspect_variance_is_bounded(self):
        rng = np.random.default_rng(0)
        dem = rng.normal(0, 1, size=(32, 32))
        var = elevation_to_aspect_variance(dem)
        assert var.min() >= -1e-9
        assert var.max() <= 1.0 + 1e-9

    def test_uniform_slope_has_low_aspect_variance(self):
        # A DEM that's a pure tilted plane has the same aspect everywhere
        # -> circular variance should be near zero, unlike random noise.
        y, x = np.mgrid[0:32, 0:32]
        dem = 0.1 * x + 0.05 * y
        var = elevation_to_aspect_variance(dem)
        assert var[2:-2, 2:-2].mean() < 0.05


class TestWeakLabels:
    def test_straight_line_produces_a_buffered_band(self):
        shape = (64, 64)
        points = [(32, 5), (32, 58)]
        mask = polyline_to_mask(shape, points, half_width_px=5)
        assert mask[32, 30]
        assert mask[32 + 4, 30]
        assert not mask[32 + 10, 30]

    def test_out_of_bounds_points_are_clipped_not_erroring(self):
        shape = (32, 32)
        points = [(-5, -5), (40, 40)]
        mask = polyline_to_mask(shape, points, half_width_px=2)
        assert mask.shape == shape

    def test_zero_half_width_gives_thin_line_only(self):
        shape = (32, 32)
        points = [(16, 2), (16, 29)]
        mask = polyline_to_mask(shape, points, half_width_px=0)
        assert mask[16, 15]
        assert not mask[16 + 3, 15]
