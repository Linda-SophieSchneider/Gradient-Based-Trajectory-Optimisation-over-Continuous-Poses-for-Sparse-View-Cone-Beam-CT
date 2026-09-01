from __future__ import annotations

import math

import numpy as np

from differentiable_coverage.eval.metrics import (
    psnr,
    roi_hfen,
    roi_psnr,
    roi_rmse,
    roi_ssim,
)


def test_roi_rmse_uses_explicit_mask():
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    rec = ref.copy()
    rec[2:6, 2:6, 2:6] = 1.0

    mask = np.zeros_like(ref, dtype=bool)
    mask[2:6, 2:6, 2:6] = True

    assert math.isclose(roi_rmse(rec, ref, mask=mask), 1.0)


def test_roi_psnr_ignores_outside_error_when_masked():
    ref = np.zeros((8, 8, 8), dtype=np.float32)
    rec = ref.copy()
    rec[0, 0, 0] = 1.0

    mask = np.zeros_like(ref, dtype=bool)
    mask[2:6, 2:6, 2:6] = True

    assert roi_psnr(rec, ref, peak=1.0, mask=mask) == float("inf")
    assert psnr(rec, ref, peak=1.0) < float("inf")


def test_roi_ssim_and_hfen_are_finite_on_mask():
    ref = np.zeros((12, 12, 12), dtype=np.float32)
    ref[3:9, 3:9, 3:9] = 1.0
    rec = ref.copy()
    rec[4:8, 4:8, 4:8] = 0.7

    mask = np.zeros_like(ref, dtype=bool)
    mask[2:10, 2:10, 2:10] = True

    ssim_val = roi_ssim(rec, ref, mask=mask)
    hfen_val = roi_hfen(rec, ref, mask=mask)

    assert 0.0 <= ssim_val <= 1.0
    assert hfen_val >= 0.0
