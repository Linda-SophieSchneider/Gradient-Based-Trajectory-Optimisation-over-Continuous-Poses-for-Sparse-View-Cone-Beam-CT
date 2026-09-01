from __future__ import annotations

import numpy as np

from experiments.run import (
    _method_kwargs_with_roi,
    _method_optimizes_roi,
    _method_roi_center,
    _resolve_roi_context,
)


class _FakeMX:
    float32 = np.float32

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def array(x, dtype=np.float32):
        return np.array(x, dtype=dtype)

    @staticmethod
    def zeros_like(x):
        return np.zeros_like(np.asarray(x))

    @staticmethod
    def ones(shape, dtype=np.float32):
        return np.ones(shape, dtype=dtype)


def _stack():
    return {"np": np, "mx": _FakeMX()}


def test_support_com_sphere_roi_is_deterministic_and_off_center():
    vol_np = np.zeros((9, 9, 9), dtype=np.float32)
    vol_np[6, 5, 7] = 4.0
    vol_np[6, 5, 6] = 2.0

    cfg = {
        "roi": {
            "type": "support_com_sphere",
            "support_threshold_frac": 0.05,
            "radius_scale": 0.25,
            "radius_min_mm": 1.0,
            "radius_max_mm": 3.0,
            "point_grid": 5,
        }
    }
    geometry = {"voxel_pitch": 1.0}

    roi_a = _resolve_roi_context(cfg, vol_np, geometry, _stack(), want_mask=True)
    roi_b = _resolve_roi_context(cfg, vol_np, geometry, _stack(), want_mask=True)

    center = np.asarray(roi_a["center"])
    assert np.allclose(center, np.asarray(roi_b["center"]))
    assert center[0] > 2.0
    assert center[1] > 0.0
    assert center[2] > 1.0
    assert roi_a["type"] == "support_com_sphere"
    assert float(roi_a["radius_mm"]) == 1.0
    assert roi_a["mask"].any()
    assert roi_a["points"].shape[0] > 0


def test_support_com_sphere_component_filter_picks_local_component():
    vol_np = np.zeros((11, 11, 11), dtype=np.float32)
    vol_np[2:5, 2:5, 2:5] = 1.0
    vol_np[7:10, 7:10, 7:10] = 2.0

    cfg = {
        "roi": {
            "type": "support_com_sphere",
            "support_threshold_frac": 0.05,
            "component_threshold_frac": 0.4,
            "component_select": "mass",
            "radius_scale": 0.25,
            "radius_min_mm": 1.0,
            "radius_max_mm": 3.0,
            "point_grid": 5,
        }
    }
    geometry = {"voxel_pitch": 1.0}

    roi = _resolve_roi_context(cfg, vol_np, geometry, _stack(), want_mask=True)
    center = np.asarray(roi["center"])
    assert np.all(center > 1.5)
    assert np.all(center < 5.5)
    assert "component=mass@0.400" in roi["summary"]


def test_support_com_sphere_max_clearance_moves_center_off_wall():
    vol_np = np.zeros((15, 15, 15), dtype=np.float32)
    vol_np[3:12, 3:12, 3:12] = 1.0
    vol_np[3:12, 3:12, 3:5] = 2.0  # brighter wall should not attract the ROI

    cfg = {
        "roi": {
            "type": "support_com_sphere",
            "support_threshold_frac": 0.05,
            "component_threshold_frac": 0.2,
            "component_select": "largest",
            "center_mode": "max_clearance",
            "radius_scale": 0.25,
            "radius_min_mm": 1.0,
            "radius_max_mm": 3.0,
            "point_grid": 5,
        }
    }
    geometry = {"voxel_pitch": 1.0}

    roi = _resolve_roi_context(cfg, vol_np, geometry, _stack(), want_mask=True)
    center = np.asarray(roi["center"])
    assert np.all(np.abs(center) < 1.0)
    assert "center_mode=max_clearance" in roi["summary"]


def test_vcls_keeps_roi_eval_only_policy():
    roi_ctx = {
        "center": np.array([3.0, -2.0, 1.0], dtype=np.float32),
        "radius_mm": 6.0,
        "points": np.ones((7, 3), dtype=np.float32),
        "weights": np.ones(7, dtype=np.float32) / 7.0,
        "mask": np.ones((3, 3, 3), dtype=bool),
        "type": "sphere",
    }
    geometry = {"voxel_pitch": 0.5}

    assert not _method_optimizes_roi("vcls")
    center = _method_roi_center("vcls", roi_ctx, _stack())
    assert np.allclose(np.asarray(center), np.zeros(3, dtype=np.float32))

    kwargs = _method_kwargs_with_roi(
        "vcls", None, roi_ctx, geometry, optimize_roi=False,
    )
    assert "roi_points" not in kwargs
    assert "roi_weights" not in kwargs
    assert "bundle_cfg" not in kwargs


def test_bundle_method_receives_shared_roi_inputs():
    roi_ctx = {
        "center": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "radius_mm": 4.5,
        "points": np.ones((5, 3), dtype=np.float32),
        "weights": np.ones(5, dtype=np.float32) / 5.0,
        "mask": None,
        "type": "sphere",
    }
    geometry = {"voxel_pitch": 0.8}

    assert _method_optimizes_roi("vcls_adam_bundle")
    center = _method_roi_center("vcls_adam_bundle", roi_ctx, _stack())
    assert np.allclose(np.asarray(center), np.asarray(roi_ctx["center"]))

    kwargs = _method_kwargs_with_roi(
        "vcls_adam_bundle", None, roi_ctx, geometry, optimize_roi=True,
    )
    assert kwargs["roi_points"].shape == (5, 3)
    assert kwargs["roi_weights"].shape == (5,)
    assert float(kwargs["bundle_cfg"].roi_radius) == 4.5
