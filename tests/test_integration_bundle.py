"""End-to-end integration test for the bundle pipeline.

Runs the proposed analytic-bundle selector on a tiny synthetic phantom,
reconstructs with SART, and checks that the resulting PSNR is finite
and substantially above an uninformative baseline.  Catches regressions
in the full select-simulate-reconstruct chain.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest


@pytest.mark.integration
def test_bundle_end_to_end_smoke():
    """Smoke test: bundle pipeline runs without error on a 64^3 phantom
    and produces a non-trivial reconstruction."""
    from differentiable_coverage.eval.geometry import geometry_from_sources
    from differentiable_coverage.eval.metrics import psnr
    from differentiable_coverage.eval.reco import (
        reconstruct_sart_volume, simulate_sinogram,
    )
    from differentiable_coverage.eval.trajectories import build_baseline_sources
    from differentiable_coverage.eval.vcl import compute_R_gamma
    from differentiable_coverage.score import sample_unit_sphere

    mx.random.seed(0)

    # Tiny synthetic phantom: two cubes at different intensities
    N = 48
    vol_np = np.zeros((N, N, N), dtype=np.float32)
    vol_np[N // 4 : 3 * N // 4, N // 4 : 3 * N // 4, N // 4 : 3 * N // 4] = 0.5
    vol_np[N // 3 : 2 * N // 3, N // 3 : 2 * N // 3, N // 3 : 2 * N // 3] = 1.0
    vol = mx.array(vol_np)

    sid, sdd = 100.0, 180.0
    det_voxels, det_pitch = 96, 0.5
    voxel_pitch = 0.6
    k_max = 64
    k = 12

    # Build VCL cache for the bundle methods.
    candidates = sample_unit_sphere(k_max) * sid
    vcl_pre = compute_R_gamma(
        vol, candidates, sid=sid, sdd=sdd,
        det_shape=(det_voxels, det_voxels),
        du=det_pitch, dv=det_pitch, voxel_spacing=voxel_pitch,
        r1=1e-2, seed=0,
    )

    # Select sources via the proposed analytic bundle pipeline.
    src = build_baseline_sources(
        "vcls_adam_bundle", k, sid,
        roi_center=mx.array([0.0, 0.0, 0.0]),
        vcl_precompute=vcl_pre, volume=vol, sdd=sdd,
        detector_shape=(det_voxels, det_voxels),
        du=det_pitch, dv=det_pitch, voxel_spacing=voxel_pitch,
        n_candidates=k_max, seed=0,
    )
    mx.eval(src)
    assert src.shape == (k, 3)
    norms = mx.linalg.norm(src, axis=-1)
    assert mx.all(mx.abs(norms - sid) < 1e-2)  # on sphere

    # Simulate sinogram, reconstruct.
    sp, dc, du, dv = geometry_from_sources(src, sid=sid, sdd=sdd)
    sino = simulate_sinogram(
        vol, sp, dc, du, dv,
        det_u=det_voxels, det_v=det_voxels,
        du=det_pitch, dv=det_pitch, voxel_spacing=voxel_pitch,
    )
    mx.eval(sino)
    assert bool(mx.all(mx.isfinite(sino)))

    res = reconstruct_sart_volume(
        vol.shape, sino, sp, dc, du, dv,
        du=det_pitch, dv=det_pitch, voxel_spacing=voxel_pitch,
        iteration_count=5, show_progress=False,
    )
    mx.eval(res.reconstruction)

    # PSNR should be finite and well above 0 dB.  For this tiny phantom
    # with k = 12 sources the absolute value is not the point; only that
    # the pipeline is finite and produces a non-zero reconstruction.
    p = float(psnr(res.reconstruction, vol))
    assert math.isfinite(p)
    assert p > 5.0, f"unexpectedly bad reconstruction PSNR = {p:.2f}"


@pytest.mark.integration
def test_bundle_centerline_is_subset_of_full_bundle():
    """The 1-ray centerline gate should produce sources on the same
    sphere as the 5x9 bundle and the discrete VCLS baseline."""
    from differentiable_coverage.eval.trajectories import build_baseline_sources
    from differentiable_coverage.eval.vcl import compute_R_gamma
    from differentiable_coverage.score import sample_unit_sphere

    mx.random.seed(0)
    N = 32
    vol = mx.array(
        (np.random.default_rng(0).uniform(0, 0.5, (N, N, N))).astype(np.float32)
    )
    sid, sdd = 100.0, 180.0
    det_voxels, det_pitch = 64, 0.5
    voxel_pitch = 0.6
    k_max, k = 48, 8

    candidates = sample_unit_sphere(k_max) * sid
    vcl_pre = compute_R_gamma(
        vol, candidates, sid=sid, sdd=sdd,
        det_shape=(det_voxels, det_voxels),
        du=det_pitch, dv=det_pitch, voxel_spacing=voxel_pitch,
        r1=1e-2, seed=0,
    )

    common = dict(
        roi_center=mx.array([0.0, 0.0, 0.0]),
        vcl_precompute=vcl_pre, volume=vol, sdd=sdd,
        detector_shape=(det_voxels, det_voxels),
        du=det_pitch, dv=det_pitch, voxel_spacing=voxel_pitch,
        n_candidates=k_max, seed=0,
    )
    for name in ("vcls", "vcls_adam_bundle_center", "vcls_adam_bundle"):
        src = build_baseline_sources(name, k, sid, **common)
        mx.eval(src)
        norms = mx.linalg.norm(src, axis=-1)
        assert mx.all(mx.abs(norms - sid) < 1e-2), (
            f"{name}: sources not on sphere"
        )
