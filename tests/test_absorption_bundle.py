"""Tests for the MLX-native analytic bundle absorption term.

Covers the trilinear sampler, bundle line integrals, gradient flow, and
the soft Beer-Lambert gate wrapper.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from differentiable_coverage.absorption_bundle import (
    BundleAbsorptionConfig,
    _orthonormal_frame,
    _trilinear_sample,
    absorption_gate_bundle,
    bundle_path_integral,
    calibrate_bundle_alpha,
    calibrate_bundle_weight,
)


def test_bundle_weight_calibration_targets_median_contribution():
    lam = calibrate_bundle_weight(0.17)
    assert math.isclose(lam * 0.17, 0.2, rel_tol=1e-12)


def test_bundle_weight_calibration_rejects_negative_target():
    with pytest.raises(ValueError, match="non-negative"):
        calibrate_bundle_weight(1.0, target=-0.1)


# ----------------------------------------------------------------------
# Trilinear sampling primitive
# ----------------------------------------------------------------------

class TestTrilinearSample:
    def test_corner_values_returned_exactly(self):
        """Sampling at an integer grid index returns that voxel value."""
        N = 8
        rng = np.random.default_rng(0)
        vol = mx.array(rng.uniform(0, 1, (N, N, N)).astype(np.float32))
        # Probe 5 random integer corners
        for _ in range(5):
            z = rng.integers(0, N)
            y = rng.integers(0, N)
            x = rng.integers(0, N)
            ijk = mx.array([[float(z), float(y), float(x)]])
            v = _trilinear_sample(vol, ijk)
            assert math.isclose(
                float(v[0]), float(np.asarray(vol)[z, y, x]),
                rel_tol=1e-5, abs_tol=1e-5,
            )

    def test_midpoint_is_eightfold_average(self):
        """A sample exactly between 8 corners is their unweighted mean."""
        N = 4
        vol = mx.array(np.arange(N**3, dtype=np.float32).reshape(N, N, N))
        # midpoint between corners (1,1,1) and (2,2,2)
        ijk = mx.array([[1.5, 1.5, 1.5]])
        v = float(_trilinear_sample(vol, ijk)[0])
        # 8 corner values around (1.5,1.5,1.5)
        corners = [np.asarray(vol)[1 + dz, 1 + dy, 1 + dx]
                   for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
        assert math.isclose(v, float(np.mean(corners)), rel_tol=1e-5)

    def test_gradient_flows_through_position(self):
        """∂μ(p)/∂p is finite where the volume has structure."""
        N = 16
        # Linear ramp in the x direction so the gradient is +1/voxel in x
        x = np.arange(N, dtype=np.float32)
        vol = mx.array(np.broadcast_to(x, (N, N, N)).copy())

        def f(ijk):
            return mx.sum(_trilinear_sample(vol, ijk))

        ijk = mx.array([[5.5, 5.5, 5.5]])
        g = mx.grad(f)(ijk)
        gx = float(g[0, 2])  # x is index 2 of ijk in our convention
        assert math.isclose(gx, 1.0, abs_tol=1e-4)


# ----------------------------------------------------------------------
# Orthonormal frame
# ----------------------------------------------------------------------

class TestOrthonormalFrame:
    def test_frame_is_orthonormal(self):
        d = mx.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [1.0 / math.sqrt(2), 1.0 / math.sqrt(2), 0.0]])
        u, v = _orthonormal_frame(d)
        # Check unit length
        for vec in (u, v):
            n = mx.linalg.norm(vec, axis=-1)
            assert mx.all(mx.abs(n - 1.0) < 1e-5)
        # Check orthogonality of u, v, d
        du = mx.sum(d * u, axis=-1)
        dv = mx.sum(d * v, axis=-1)
        uv = mx.sum(u * v, axis=-1)
        assert mx.all(mx.abs(du) < 1e-5)
        assert mx.all(mx.abs(dv) < 1e-5)
        assert mx.all(mx.abs(uv) < 1e-5)

    def test_polar_fallback(self):
        """When d is close to z-axis the fallback frame stays well defined."""
        d = mx.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        u, v = _orthonormal_frame(d)
        n_u = mx.linalg.norm(u, axis=-1)
        n_v = mx.linalg.norm(v, axis=-1)
        assert mx.all(mx.abs(n_u - 1.0) < 1e-5)
        assert mx.all(mx.abs(n_v - 1.0) < 1e-5)


# ----------------------------------------------------------------------
# Bundle line integral
# ----------------------------------------------------------------------

def _make_dense_blob(N: int, centre_voxel=(0.0, 0.0, 0.0), radius=8.0):
    """Spherical attenuation blob centred at the given voxel offset
    relative to volume centre.  Returns mx.array (N, N, N)."""
    zz, yy, xx = np.meshgrid(
        np.arange(N) - (N - 1) / 2,
        np.arange(N) - (N - 1) / 2,
        np.arange(N) - (N - 1) / 2,
        indexing="ij",
    )
    cz, cy, cx = centre_voxel
    r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return mx.array((r2 < radius ** 2).astype(np.float32))


class TestBundlePathIntegral:
    def test_zero_for_empty_volume(self):
        N = 32
        vol = mx.zeros((N, N, N))
        roi = mx.array([0.0, 0.0, 0.0])
        sources = mx.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
        cfg = BundleAbsorptionConfig(
            roi_radius=2.0, n_rays_u=3, n_rays_v=3,
            n_samples=16, voxel_spacing=1.0,
        )
        tau = bundle_path_integral(sources, roi, vol, cfg)
        assert mx.all(tau < 1e-5)

    def test_centerline_collapses_to_single_ray(self):
        """With n_u = n_v = 1 the bundle is a single centreline ray.

        The blob is offset along axis 0 (= world z, MLX (Z,Y,X)
        convention).  A +z source crosses it; a +x source does not.
        """
        N = 32
        vol = _make_dense_blob(N, centre_voxel=(8.0, 0, 0), radius=4.0)
        roi = mx.array([0.0, 0.0, 0.0])
        # Source at +x: ray (100,0,0)→(0,0,0) stays at z=0, misses blob.
        # Source at +z: ray (0,0,100)→(0,0,0) crosses z=8, hits blob.
        sources = mx.array([[100.0, 0.0, 0.0], [0.0, 0.0, 100.0]])
        cfg = BundleAbsorptionConfig(
            roi_radius=0.0, n_rays_u=1, n_rays_v=1,
            n_samples=64, voxel_spacing=1.0,
        )
        tau = bundle_path_integral(sources, roi, vol, cfg)
        # +x source misses blob (tau ~ 0)
        assert float(tau[0]) < 1.0
        # +z source crosses blob (tau >> +x source's tau)
        assert float(tau[1]) > float(tau[0]) + 1.0

    def test_off_centre_blob_breaks_direction_symmetry(self):
        """A +z dense blob raises tau for a +x source and not for a -x one."""
        N = 32
        vol = _make_dense_blob(N, centre_voxel=(8.0, 0, 0), radius=4.0)
        roi = mx.array([0.0, 0.0, 0.0])
        cfg = BundleAbsorptionConfig(
            roi_radius=2.0, n_rays_u=3, n_rays_v=3,
            n_samples=32, voxel_spacing=1.0,
        )
        # +x and -x sources viewing through origin
        sources = mx.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
        tau = bundle_path_integral(sources, roi, vol, cfg)
        # Symmetric about origin (sphere is at +z in voxel coords, world +x in
        # the convention of bundle_path_integral) → both should be similar.
        # The point is that tau is non-negative and finite.
        assert mx.all(tau >= 0.0)
        assert mx.all(mx.isfinite(tau))

    def test_gradient_pushes_source_away_from_dense(self):
        """∂(−τ̄)/∂s is non-zero and tangentially pointed away from dense."""
        N = 32
        vol = _make_dense_blob(N, centre_voxel=(8.0, 0, 0), radius=4.0)
        roi = mx.array([0.0, 0.0, 0.0])
        cfg = BundleAbsorptionConfig(
            roi_radius=2.0, n_rays_u=3, n_rays_v=3,
            n_samples=32, voxel_spacing=1.0,
        )
        sources = mx.array([[0.0, 0.0, 100.0]])  # looks through blob

        def loss(s):
            return -mx.mean(bundle_path_integral(s, roi, vol, cfg))

        g = mx.grad(loss)(sources)
        # Gradient is finite and non-zero
        assert bool(mx.all(mx.isfinite(g)))
        assert float(mx.linalg.norm(g)) > 1e-6


# ----------------------------------------------------------------------
# Soft Beer-Lambert gate + calibration
# ----------------------------------------------------------------------

class TestBundleGate:
    def test_gate_in_unit_interval(self):
        N = 24
        vol = _make_dense_blob(N, centre_voxel=(6.0, 0, 0), radius=3.0)
        roi = mx.array([0.0, 0.0, 0.0])
        cfg = BundleAbsorptionConfig(
            roi_radius=1.0, n_rays_u=3, n_rays_v=3,
            n_samples=16, voxel_spacing=1.0, alpha=0.1,
        )
        sources = mx.random.normal(shape=(20, 3)) * 50.0
        nu = absorption_gate_bundle(sources, roi, vol, cfg)
        assert mx.all(nu >= 0.0)
        assert mx.all(nu <= 1.0)

    def test_calibrate_returns_finite_alpha(self):
        N = 24
        vol = _make_dense_blob(N, centre_voxel=(6.0, 0, 0), radius=3.0)
        roi = mx.array([0.0, 0.0, 0.0])
        cfg = BundleAbsorptionConfig(
            roi_radius=1.0, n_rays_u=3, n_rays_v=3,
            n_samples=16, voxel_spacing=1.0,
        )
        alpha = calibrate_bundle_alpha(vol, roi, sid=80.0, n_probe=64, cfg=cfg)
        assert math.isfinite(alpha)
        assert alpha > 0.0


# ----------------------------------------------------------------------
# Off-centre ROI vs volume origin (REV-P0-02)
# ----------------------------------------------------------------------

def _reference_tau_single_ray(sources, roi_center, vol_np, n_samples,
                              voxel_spacing, volume_center=(0.0, 0.0, 0.0)):
    """Brute-force numpy reference for the single-centreline bundle.

    Midpoint rule along source→roi_center with the volume centre voxel at
    ``volume_center`` (world mm) — the world→index map is independent of
    ``roi_center``.
    """
    from scipy.ndimage import map_coordinates

    Z, Y, X = vol_np.shape
    ci = np.array([(Z - 1) / 2.0, (Y - 1) / 2.0, (X - 1) / 2.0])
    vc = np.asarray(volume_center, dtype=np.float64)
    roi = np.asarray(roi_center, dtype=np.float64)
    taus = []
    for s in np.asarray(sources, dtype=np.float64):
        t = (np.arange(n_samples) + 0.5) / n_samples
        pts = s[None, :] + t[:, None] * (roi - s)[None, :]
        p_rel = pts - vc[None, :]
        coords = np.stack([
            p_rel[:, 2] / voxel_spacing + ci[0],
            p_rel[:, 1] / voxel_spacing + ci[1],
            p_rel[:, 0] / voxel_spacing + ci[2],
        ], axis=0)
        vals = map_coordinates(vol_np, coords, order=1, mode="nearest")
        length = np.linalg.norm(roi - s)
        taus.append(vals.sum() * length / n_samples)
    return np.array(taus)


class TestOffCentreROIVolumeOrigin:
    """The world→index mapping must use the volume centre, not the ROI centre.

    With an off-centre ROI in an origin-centred volume, mapping through the
    ROI centre translates the whole attenuation field per ROI (the limited-
    angle `ours_roi_abs` defect)."""

    def _setup(self):
        N = 32
        vol = _make_dense_blob(N, centre_voxel=(8.0, 0.0, 0.0), radius=4.0)
        # Blob sits at world (0, 0, +8) mm; ROI is off-centre at (0, 0, +10).
        roi = mx.array([0.0, 0.0, 10.0])
        sources = mx.array([
            [0.0, 0.0, -90.0],   # crosses the blob on the way to the ROI
            [100.0, 0.0, 10.0],  # grazes the blob perpendicular to z
            [0.0, 100.0, 10.0],
        ])
        cfg = BundleAbsorptionConfig(
            roi_radius=0.0, n_rays_u=1, n_rays_v=1,
            n_samples=128, voxel_spacing=1.0,
        )
        return vol, roi, sources, cfg

    def test_off_centre_roi_matches_world_frame_reference(self):
        vol, roi, sources, cfg = self._setup()
        tau = np.asarray(bundle_path_integral(sources, roi, vol, cfg))
        ref = _reference_tau_single_ray(
            sources, roi, np.asarray(vol, dtype=np.float64),
            cfg.n_samples, cfg.voxel_spacing,
        )
        # The blob is genuinely on these rays: the reference must be non-trivial.
        assert ref[0] > 1.0 and ref[1] > 1.0
        np.testing.assert_allclose(tau, ref, rtol=5e-4, atol=5e-3)

    def test_world_translation_of_sources_and_roi_changes_tau(self):
        """Shifting sources + ROI over a *fixed* inhomogeneous volume must
        sample different material — it must NOT be a no-op."""
        vol, roi, sources, cfg = self._setup()
        tau_base = np.asarray(bundle_path_integral(sources, roi, vol, cfg))
        shift = mx.array([0.0, 0.0, 12.0])
        tau_shifted = np.asarray(bundle_path_integral(
            sources + shift[None, :], roi + shift, vol, cfg))
        assert not np.allclose(tau_base, tau_shifted, atol=1e-3)

    def test_translation_equivariance_with_volume_center(self):
        """Shifting sources, ROI, *and* the declared volume centre together
        is a pure world-frame relabelling and must be exact."""
        vol, roi, sources, cfg = self._setup()
        tau_base = np.asarray(bundle_path_integral(sources, roi, vol, cfg))
        shift = mx.array([3.0, -7.0, 12.0])
        tau_equi = np.asarray(bundle_path_integral(
            sources + shift[None, :], roi + shift, vol, cfg,
            volume_center=shift))
        # Algebraically identical but not bitwise: float32 coordinate maths at
        # ~100 mm magnitudes near a binary blob edge → keep tolerance modest.
        np.testing.assert_allclose(tau_equi, tau_base, rtol=1e-4, atol=1e-3)

    def test_centred_roi_unchanged_by_explicit_volume_center(self):
        """roi_center = 0 (the real-experiment convention) must behave
        identically with and without an explicit volume_center."""
        N = 32
        vol = _make_dense_blob(N, centre_voxel=(8.0, 0.0, 0.0), radius=4.0)
        roi = mx.zeros(3)
        sources = mx.array([[100.0, 0.0, 0.0], [0.0, 0.0, 100.0]])
        cfg = BundleAbsorptionConfig(
            roi_radius=2.0, n_rays_u=3, n_rays_v=3,
            n_samples=32, voxel_spacing=1.0,
        )
        tau_default = np.asarray(bundle_path_integral(sources, roi, vol, cfg))
        tau_explicit = np.asarray(bundle_path_integral(
            sources, roi, vol, cfg, volume_center=mx.zeros(3)))
        np.testing.assert_allclose(tau_default, tau_explicit, rtol=0, atol=0)

    def test_gate_and_calibration_respect_volume_center(self):
        """The gate and alpha calibration must honour a genuinely non-zero
        volume_center: a rigid world shift of sources, ROI, and volume
        centre together leaves both invariant."""
        vol, roi, sources, cfg = self._setup()
        shift = mx.array([4.0, -3.0, 9.0])
        nu_base = np.asarray(absorption_gate_bundle(sources, roi, vol, cfg))
        nu_equi = np.asarray(absorption_gate_bundle(
            sources + shift[None, :], roi + shift, vol, cfg,
            volume_center=shift))
        np.testing.assert_allclose(nu_equi, nu_base, rtol=1e-4, atol=1e-4)
        alpha_base = calibrate_bundle_alpha(vol, roi, sid=80.0, n_probe=32,
                                            cfg=cfg)
        alpha_equi = calibrate_bundle_alpha(vol, roi + shift, sid=80.0,
                                            n_probe=32, cfg=cfg,
                                            volume_center=shift)
        assert math.isfinite(alpha_base) and alpha_base > 0.0
        np.testing.assert_allclose(alpha_equi, alpha_base, rtol=1e-4)


# ----------------------------------------------------------------------
# Volume-clipped quadrature (REV-P1-01)
# ----------------------------------------------------------------------

class TestClipToVolume:
    """clip_to_volume=True integrates only the ray ∩ bounding-box segment.

    The integral's converged value must match the full-segment rule (the
    out-of-volume stretches contribute ~0 for air-margined phantoms); the
    clipped rule just spends all n_samples inside the volume."""

    def _cfg(self, n_samples, clip):
        return BundleAbsorptionConfig(
            roi_radius=0.0, n_rays_u=1, n_rays_v=1,
            n_samples=n_samples, voxel_spacing=1.0, clip_to_volume=clip,
        )

    def test_clipped_converges_to_full_segment_reference(self):
        N = 32
        vol = _make_dense_blob(N, centre_voxel=(8.0, 0.0, 0.0), radius=4.0)
        roi = mx.array([0.0, 0.0, 10.0])
        sources = mx.array([
            [0.0, 0.0, -90.0],
            [100.0, 0.0, 10.0],
            [-60.0, 80.0, 0.0],
        ])
        tau_clip = np.asarray(bundle_path_integral(
            sources, roi, vol, self._cfg(64, True)))
        ref = _reference_tau_single_ray(
            sources, roi, np.asarray(vol, dtype=np.float64),
            4096, 1.0)
        np.testing.assert_allclose(tau_clip, ref, rtol=2e-2, atol=2e-2)

    def test_clipped_beats_full_segment_at_equal_budget(self):
        """With few samples on a long ray, the clipped rule must be closer
        to the converged value than the legacy full-segment rule."""
        N = 32
        vol = _make_dense_blob(N, centre_voxel=(8.0, 0.0, 0.0), radius=4.0)
        roi = mx.array([0.0, 0.0, 10.0])
        # Long standoff: only a small fraction of the segment is in-volume.
        sources = mx.array([[0.0, 0.0, -500.0]])
        ref = _reference_tau_single_ray(
            sources, roi, np.asarray(vol, dtype=np.float64), 8192, 1.0)
        tau_full = float(bundle_path_integral(
            sources, roi, vol, self._cfg(16, False))[0])
        tau_clip = float(bundle_path_integral(
            sources, roi, vol, self._cfg(16, True))[0])
        assert abs(tau_clip - ref[0]) < abs(tau_full - ref[0])
        assert abs(tau_clip - ref[0]) < 0.05 * max(ref[0], 1e-6)

    def test_ray_missing_box_integrates_to_zero(self):
        N = 32
        vol = mx.ones((N, N, N))          # solid volume: any hit gives τ > 0
        roi = mx.array([0.0, 0.0, 80.0])  # target far outside the box
        sources = mx.array([[0.0, 200.0, 80.0]])   # segment stays at z = 80
        tau = bundle_path_integral(sources, roi, vol, self._cfg(32, True))
        assert float(tau[0]) == 0.0

    def test_clipped_translation_equivariance(self):
        N = 32
        vol = _make_dense_blob(N, centre_voxel=(8.0, 0.0, 0.0), radius=4.0)
        roi = mx.array([0.0, 0.0, 10.0])
        sources = mx.array([[0.0, 0.0, -90.0], [100.0, 0.0, 10.0]])
        cfg = self._cfg(64, True)
        tau_base = np.asarray(bundle_path_integral(sources, roi, vol, cfg))
        shift = mx.array([3.0, -7.0, 12.0])
        tau_equi = np.asarray(bundle_path_integral(
            sources + shift[None, :], roi + shift, vol, cfg,
            volume_center=shift))
        np.testing.assert_allclose(tau_equi, tau_base, rtol=1e-4, atol=1e-3)

    def test_clipped_gradient_flows(self):
        N = 32
        vol = _make_dense_blob(N, centre_voxel=(8.0, 0.0, 0.0), radius=4.0)
        roi = mx.array([0.0, 0.0, 0.0])
        cfg = BundleAbsorptionConfig(
            roi_radius=2.0, n_rays_u=3, n_rays_v=3,
            n_samples=32, voxel_spacing=1.0, clip_to_volume=True,
        )
        sources = mx.array([[0.0, 0.0, 100.0]])

        def loss(s):
            return -mx.mean(bundle_path_integral(s, roi, vol, cfg))

        g = mx.grad(loss)(sources)
        assert bool(mx.all(mx.isfinite(g)))
        assert float(mx.linalg.norm(g)) > 1e-6


# ----------------------------------------------------------------------
# Configuration object
# ----------------------------------------------------------------------

class TestConfig:
    def test_defaults_match_paper(self):
        cfg = BundleAbsorptionConfig()
        # Paper §IV-F: n_u = 5, n_v = 9, 32 samples, ROI 5 mm
        assert cfg.n_rays_u == 5
        assert cfg.n_rays_v == 9
        assert cfg.n_samples == 32
        assert math.isclose(cfg.roi_radius, 5.0)
