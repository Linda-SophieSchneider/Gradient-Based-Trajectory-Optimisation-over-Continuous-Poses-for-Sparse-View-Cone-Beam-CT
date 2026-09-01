"""Tests for the Fourier-weighted coverage covariance score.

`coverage_covariance_information` already accepts a per-direction weighting
``direction_weights``.  This test file verifies the contract holds when
those weights come from :func:`fourier_radon_weights`, plus the
end-to-end differentiability we rely on for the new
``greedy_adam_icov_fft`` baseline.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from differentiable_coverage.fourier_radon import fourier_radon_weights
from differentiable_coverage.score import (
    ScoreConfig,
    coverage_covariance_information,
    sample_unit_sphere,
)


@pytest.fixture
def setup_small():
    """Tiny but full-rank configuration for analytical tests."""
    sid = 500.0
    sources = sample_unit_sphere(8, seed=0) * sid
    roi_center = mx.array([0.0, 0.0, 0.0])
    radon_normals = sample_unit_sphere(32)
    nu = mx.ones(8)
    cfg = ScoreConfig(tau=0.1)
    return dict(
        sources=sources, roi_center=roi_center,
        radon_normals=radon_normals, nu=nu, cfg=cfg,
    )


@pytest.fixture
def z_cylinder_volume():
    N = 32
    zz, yy, xx = np.meshgrid(
        np.arange(N), np.arange(N), np.arange(N), indexing="ij"
    )
    cy, cx = (N - 1) / 2, (N - 1) / 2
    r2_xy = (yy - cy) ** 2 + (xx - cx) ** 2
    vol = (r2_xy < (N / 6) ** 2).astype(np.float32) * 0.5
    return mx.array(vol)


# ---------------------------------------------------------------------------
# Contract: weighted I_cov matches the uniform-weights default
# ---------------------------------------------------------------------------

def test_explicit_uniform_weights_match_default_up_to_normalisation(setup_small):
    """The default `direction_weights=None` uses ``γ_F = sum(F)``; passing
    ``direction_weights=ones(z)`` makes ``γ_F = F·1`` which is the same
    vector.  So the two calls should yield bit-equal results."""
    s = setup_small
    z = s["radon_normals"].shape[0]
    ones_w = mx.ones(z, dtype=mx.float32)

    icov_default = float(coverage_covariance_information(
        s["sources"], s["roi_center"], s["radon_normals"], s["nu"], s["cfg"],
    ))
    icov_ones = float(coverage_covariance_information(
        s["sources"], s["roi_center"], s["radon_normals"], s["nu"], s["cfg"],
        direction_weights=ones_w,
    ))
    assert math.isclose(icov_default, icov_ones, rel_tol=1e-5), (
        f"uniform-ones weights should reproduce the default; "
        f"default={icov_default}, ones={icov_ones}"
    )


def test_fourier_weights_change_icov_vs_uniform(setup_small, z_cylinder_volume):
    """Volume-aware weights must yield a different (and non-trivial)
    `I_cov` than the uniform fallback on an anisotropic phantom."""
    s = setup_small
    w_fft = fourier_radon_weights(z_cylinder_volume, s["radon_normals"])
    z = s["radon_normals"].shape[0]
    # Use ones/z for uniform so the magnitudes are comparable.
    icov_fft = float(coverage_covariance_information(
        s["sources"], s["roi_center"], s["radon_normals"], s["nu"], s["cfg"],
        direction_weights=w_fft,
    ))
    icov_uni = float(coverage_covariance_information(
        s["sources"], s["roi_center"], s["radon_normals"], s["nu"], s["cfg"],
        direction_weights=mx.ones(z) / z,
    ))
    assert abs(icov_fft - icov_uni) > 1e-8, (
        f"Fourier weights produced identical I_cov to uniform on an "
        f"anisotropic phantom; uni={icov_uni}, fft={icov_fft}"
    )


def test_weights_scale_yields_quadratic_icov(setup_small, z_cylinder_volume):
    """`γ_F = F·w` is linear in w → `I_cov = γ^T R^{-1} γ` is quadratic in w."""
    s = setup_small
    w = fourier_radon_weights(z_cylinder_volume, s["radon_normals"])
    icov_1 = float(coverage_covariance_information(
        s["sources"], s["roi_center"], s["radon_normals"], s["nu"], s["cfg"],
        direction_weights=w,
    ))
    icov_2 = float(coverage_covariance_information(
        s["sources"], s["roi_center"], s["radon_normals"], s["nu"], s["cfg"],
        direction_weights=2.0 * w,
    ))
    ratio = icov_2 / max(icov_1, 1e-12)
    assert math.isclose(ratio, 4.0, rel_tol=5e-2), (
        f"expected ratio 4.0 for 2× weights; got {ratio:.4f}"
    )


# ---------------------------------------------------------------------------
# Gradient flow — the whole point of the new method
# ---------------------------------------------------------------------------

def test_gradient_w_r_t_sources_is_finite_and_nonzero(
        setup_small, z_cylinder_volume):
    s = setup_small
    w_fft = fourier_radon_weights(z_cylinder_volume, s["radon_normals"])

    def loss(sources):
        return coverage_covariance_information(
            sources, s["roi_center"], s["radon_normals"], s["nu"], s["cfg"],
            direction_weights=w_fft,
        )

    grad_fn = mx.grad(loss)
    g = grad_fn(s["sources"])
    mx.eval(g)
    g_np = np.asarray(g)

    assert g.shape == s["sources"].shape, "gradient shape mismatch"
    assert np.all(np.isfinite(g_np)), "non-finite gradient entries"
    assert float(np.linalg.norm(g_np)) > 1e-6, (
        "gradient norm is zero — no source-dependence detected"
    )


def test_gradient_consistent_under_seed(setup_small, z_cylinder_volume):
    """Calling the function twice with the same inputs must yield identical
    gradients (no hidden randomness — we rely on this for reproducibility)."""
    s = setup_small
    w_fft = fourier_radon_weights(z_cylinder_volume, s["radon_normals"])

    def loss(sources):
        return coverage_covariance_information(
            sources, s["roi_center"], s["radon_normals"], s["nu"], s["cfg"],
            direction_weights=w_fft,
        )

    g1 = mx.grad(loss)(s["sources"]); mx.eval(g1)
    g2 = mx.grad(loss)(s["sources"]); mx.eval(g2)
    rel = float(mx.max(mx.abs(g1 - g2)) / mx.max(mx.abs(g1)))
    assert rel < 1e-5, f"non-reproducible gradient; rel diff = {rel}"


# ---------------------------------------------------------------------------
# End-to-end: gradient step lowers a sensible loss
# ---------------------------------------------------------------------------

def test_one_gradient_step_decreases_neg_icov(setup_small, z_cylinder_volume):
    """Negative-`I_cov` is what the maximisation problem reduces to in our
    optimiser.  A single step in the negative-gradient direction must
    strictly decrease it, otherwise the Adam-driven baseline cannot make
    progress."""
    s = setup_small
    w_fft = fourier_radon_weights(z_cylinder_volume, s["radon_normals"])

    def neg_icov(sources):
        return -coverage_covariance_information(
            sources, s["roi_center"], s["radon_normals"], s["nu"], s["cfg"],
            direction_weights=w_fft,
        )

    loss_before = float(neg_icov(s["sources"]))
    g = mx.grad(neg_icov)(s["sources"])
    mx.eval(g)
    # Small step in the descent direction.
    eta = 1.0
    new_sources = s["sources"] - eta * g
    loss_after = float(neg_icov(new_sources))
    assert loss_after < loss_before, (
        f"negative-I_cov did not decrease after one gradient step; "
        f"before={loss_before:.4f}, after={loss_after:.4f}"
    )
