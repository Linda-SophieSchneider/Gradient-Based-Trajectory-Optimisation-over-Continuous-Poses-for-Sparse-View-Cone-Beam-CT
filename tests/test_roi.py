"""Tests for differentiable_coverage.roi."""

import mlx.core as mx

from differentiable_coverage import (
    roi_from_bbox,
    roi_from_mask,
    roi_from_points,
    sample_unit_sphere,
    saturated_coverage,
    ScoreConfig,
)


class TestROIFromPoints:
    def test_center_radius_and_shape(self):
        pts = mx.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=mx.float32,
        )
        roi = roi_from_points(pts)
        mx.eval(roi.center, roi.points, roi.weights)

        assert roi.points.shape == (4, 3)
        assert roi.weights.shape == (4,)
        assert float(mx.sum(mx.abs(roi.center - mx.array([0.5, 0.5, 0.5])))) < 1e-6
        assert roi.radius > 0.0

    def test_downsampling_cap(self):
        pts = mx.arange(30, dtype=mx.float32).reshape(10, 3)
        roi = roi_from_points(pts, max_points=4)
        assert roi.points.shape[0] == 4


class TestROIFromBBox:
    def test_bbox_center_and_point_count(self):
        roi = roi_from_bbox(
            min_corner=(0.0, 0.0, 0.0),
            max_corner=(4.0, 6.0, 8.0),
            voxel_spacing=2.0,
            origin=(1.0, 1.0, 1.0),
            points_per_axis=3,
        )
        mx.eval(roi.center, roi.points, roi.weights)

        expected_center = mx.array([5.0, 7.0, 9.0], dtype=mx.float32)
        assert float(mx.sum(mx.abs(roi.center - expected_center))) < 1e-6
        assert roi.points.shape == (27, 3)
        assert float(mx.sum(roi.weights)) == 1.0


class TestROIFromMask:
    def test_mask_center_and_downsampling(self):
        mask = mx.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            ],
            dtype=mx.float32,
        )
        roi = roi_from_mask(mask, voxel_spacing=1.0, origin=(0.0, 0.0, 0.0), max_points=4)
        mx.eval(roi.center, roi.points, roi.weights)

        assert roi.points.shape[0] == 4
        assert roi.weights.shape == (4,)
        assert roi.radius > 0.0

    def test_mask_selection_works_with_coverage(self):
        mask = mx.array(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=mx.float32,
        )
        roi = roi_from_mask(mask, max_points=8)

        sources = mx.array(
            [
                [200.0, 0.0, 0.0],
                [0.0, 200.0, 0.0],
            ],
            dtype=mx.float32,
        )
        radon = sample_unit_sphere(16)
        nu = mx.ones((2,), dtype=mx.float32)
        cfg = ScoreConfig(tau=0.07)

        cov = saturated_coverage(
            sources,
            roi.center,
            radon,
            nu,
            cfg,
            roi_points=roi.points,
            roi_weights=roi.weights,
        )
        mx.eval(cov)
        assert float(cov) >= 0.0
