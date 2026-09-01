"""Tests for the Fourier-slice Radon importance weights.

Verifies the volume-aware weighting that lifts the additive soft-Tuy
objective with phantom-specific information, without needing the SART
forward operator (no per-phantom precomputation).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from differentiable_coverage.fourier_radon import fourier_radon_weights
from differentiable_coverage.score import sample_unit_sphere


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def constant_volume():
    """Constant μ-volume: only the DC bin is non-zero in Fourier space, so
    every line through the centre carries identical energy regardless of
    direction."""
    N = 32
    return mx.array(np.full((N, N, N), 0.5, dtype=np.float32))


@pytest.fixture
def isotropic_ball_volume():
    """Centred solid ball — rotationally symmetric, so all Radon normals
    should receive (approximately) equal weight."""
    N = 32
    zz, yy, xx = np.meshgrid(
        np.arange(N), np.arange(N), np.arange(N), indexing="ij"
    )
    cz, cy, cx = (N - 1) / 2, (N - 1) / 2, (N - 1) / 2
    r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    vol = (r2 < (N / 4) ** 2).astype(np.float32) * 0.5
    return mx.array(vol)


@pytest.fixture
def z_cylinder_volume():
    """Long cylinder elongated along the z-axis.

    Smooth in z (only DC in z), structured in xy.  Fourier energy is
    therefore concentrated in the (k_y, k_x) plane, so Radon-normal
    directions along x or y receive much higher weight than along z.
    """
    N = 32
    zz, yy, xx = np.meshgrid(
        np.arange(N), np.arange(N), np.arange(N), indexing="ij"
    )
    cy, cx = (N - 1) / 2, (N - 1) / 2
    r2_xy = (yy - cy) ** 2 + (xx - cx) ** 2
    vol = (r2_xy < (N / 6) ** 2).astype(np.float32) * 0.5
    return mx.array(vol)


# ---------------------------------------------------------------------------
# Basic shape / API sanity
# ---------------------------------------------------------------------------

def test_output_shape_matches_input(z_cylinder_volume):
    normals = sample_unit_sphere(50)
    w = fourier_radon_weights(z_cylinder_volume, normals)
    assert w.shape == (50,)


def test_weights_are_non_negative(z_cylinder_volume):
    normals = sample_unit_sphere(64)
    w = fourier_radon_weights(z_cylinder_volume, normals)
    mx.eval(w)
    assert float(mx.min(w)) >= 0.0


def test_normalized_weights_sum_to_one(z_cylinder_volume):
    normals = sample_unit_sphere(40)
    w = fourier_radon_weights(z_cylinder_volume, normals, normalize=True)
    mx.eval(w)
    assert math.isclose(float(mx.sum(w)), 1.0, abs_tol=1e-4)


def test_raw_weights_can_be_returned(z_cylinder_volume):
    normals = sample_unit_sphere(10)
    w = fourier_radon_weights(z_cylinder_volume, normals, normalize=False)
    mx.eval(w)
    # Raw energies are positive and clearly above unit-normalised range for a
    # 32^3 cylinder with μ ≈ 0.5.
    assert float(mx.min(w)) > 0.0
    assert float(mx.max(w)) > 1.0


def test_rejects_wrong_input_shape(z_cylinder_volume):
    bad = mx.array([0.0, 0.0, 1.0])           # 1-D, not (z, 3)
    with pytest.raises(ValueError, match="shape"):
        fourier_radon_weights(z_cylinder_volume, bad)


# ---------------------------------------------------------------------------
# Physical / mathematical correctness
# ---------------------------------------------------------------------------

def test_constant_volume_gives_uniform_weights(constant_volume):
    """All directional energy lives in the DC bin → every Fourier line through
    the origin sees the same value, regardless of direction."""
    normals = sample_unit_sphere(64)
    w = fourier_radon_weights(constant_volume, normals, normalize=False)
    mx.eval(w)
    w_np = np.asarray(w)
    rel_std = float(w_np.std() / w_np.mean())
    assert rel_std < 5e-2, (
        f"constant volume should give uniform weights; rel_std={rel_std}"
    )


def test_isotropic_ball_gives_approximately_uniform(isotropic_ball_volume):
    """A rotationally symmetric phantom: weights should be ≈ uniform across
    quasi-uniformly sampled normals."""
    normals = sample_unit_sphere(120)
    w = fourier_radon_weights(isotropic_ball_volume, normals, normalize=False)
    mx.eval(w)
    w_np = np.asarray(w)
    rel_std = float(w_np.std() / w_np.mean())
    # Looser bound because (a) finite-resolution discretisation and
    # (b) finite Fibonacci-lattice sphere sampling break exact isotropy.
    assert rel_std < 0.25, (
        f"isotropic ball not uniform enough; rel_std={rel_std}"
    )


def test_z_cylinder_peaks_perpendicular_to_axis(z_cylinder_volume):
    """For a z-elongated cylinder, the Fourier line along k_z has very low
    energy (only DC), while lines along k_x and k_y carry the disk-FT
    energy.  Verifies that the Fourier weights track the elongation
    structure of the phantom — the very property that should let our
    soft-Tuy lift its view-importance proxy."""
    mu_z = mx.array([[0.0, 0.0, 1.0]])    # world (x, y, z) — Radon normal along z
    mu_x = mx.array([[1.0, 0.0, 0.0]])
    mu_y = mx.array([[0.0, 1.0, 0.0]])

    w_z = float(
        fourier_radon_weights(z_cylinder_volume, mu_z, normalize=False)[0]
    )
    w_x = float(
        fourier_radon_weights(z_cylinder_volume, mu_x, normalize=False)[0]
    )
    w_y = float(
        fourier_radon_weights(z_cylinder_volume, mu_y, normalize=False)[0]
    )

    assert w_x > 2.0 * w_z, (
        f"k_x line should carry significantly more energy than k_z; "
        f"got w_x={w_x:.3e}, w_z={w_z:.3e}"
    )
    assert w_y > 2.0 * w_z, (
        f"k_y line should carry significantly more energy than k_z; "
        f"got w_y={w_y:.3e}, w_z={w_z:.3e}"
    )
    # Cylinder is symmetric in xy → w_x ≈ w_y.
    sym_err = abs(w_x - w_y) / max(w_x, w_y)
    assert sym_err < 0.2, (
        f"k_x and k_y should be symmetric for a circular cylinder; "
        f"got w_x={w_x:.3e}, w_y={w_y:.3e}, rel_err={sym_err:.3f}"
    )


def test_quadratic_in_volume_amplitude(z_cylinder_volume):
    """|F|² is quadratic in V → scaling V by c scales the weights by c²."""
    normals = sample_unit_sphere(16)
    w_1 = fourier_radon_weights(
        z_cylinder_volume, normals, normalize=False,
    )
    w_3 = fourier_radon_weights(
        3.0 * z_cylinder_volume, normals, normalize=False,
    )
    mx.eval(w_1); mx.eval(w_3)
    ratio = float(mx.mean(w_3 / mx.maximum(w_1, 1e-12)))
    assert math.isclose(ratio, 9.0, rel_tol=0.05), (
        f"expected w(3V) = 9·w(V); got ratio={ratio:.3f}"
    )


def test_antipodal_symmetry(z_cylinder_volume):
    """V is real → |F3[V]|² is even, so w(μ) = w(-μ)."""
    normals = sample_unit_sphere(32)
    neg_normals = -normals
    w_pos = fourier_radon_weights(z_cylinder_volume, normals, normalize=False)
    w_neg = fourier_radon_weights(z_cylinder_volume, neg_normals, normalize=False)
    mx.eval(w_pos); mx.eval(w_neg)
    rel_err = float(
        mx.max(mx.abs(w_pos - w_neg)) / mx.max(mx.abs(w_pos))
    )
    assert rel_err < 1e-4, (
        f"antipodal symmetry violated; max rel diff = {rel_err:.3e}"
    )


def test_normalized_invariant_to_scale(z_cylinder_volume):
    """When ``normalize=True`` the weights become scale-invariant."""
    normals = sample_unit_sphere(20)
    w_a = fourier_radon_weights(z_cylinder_volume, normals, normalize=True)
    w_b = fourier_radon_weights(
        7.5 * z_cylinder_volume, normals, normalize=True,
    )
    mx.eval(w_a); mx.eval(w_b)
    rel_err = float(
        mx.max(mx.abs(w_a - w_b)) / mx.max(mx.abs(w_a))
    )
    assert rel_err < 1e-4, f"normalised weights not scale-invariant; got {rel_err}"
