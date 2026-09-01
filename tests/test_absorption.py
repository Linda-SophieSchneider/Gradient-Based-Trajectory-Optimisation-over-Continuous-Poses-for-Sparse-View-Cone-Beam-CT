"""Tests for differentiable_coverage.absorption."""

import math
import mlx.core as mx
import pytest

from differentiable_coverage import AbsorptionConfig, absorption_gate
from differentiable_coverage.absorption import (
    _detector_containment_gate,
    _detector_frame,
    _soft_footprint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return AbsorptionConfig(
        alpha=4.0, eta=0.05, roi_radius=12.0,
        sid=200.0, sdd=400.0,
        det_u=16, det_v=16, du=1.0, dv=1.0,
        voxel_spacing=1.0,
    )


@pytest.fixture
def small_volume():
    import diffct_mlx
    return diffct_mlx.shepp_logan_3d(32, 32, 32) * 0.02


@pytest.fixture
def sources_2():
    """Two sources on a circle at radius 200 in xy-plane."""
    return mx.array([
        [200.0, 0.0, 0.0],
        [0.0, 200.0, 0.0],
    ])


@pytest.fixture
def roi_center():
    return mx.zeros(3)


# ---------------------------------------------------------------------------
# _detector_frame
# ---------------------------------------------------------------------------

class TestDetectorFrame:
    def test_output_shapes(self, sources_2, roi_center, cfg):
        det_center, det_u_vec, det_v_vec = _detector_frame(sources_2, roi_center, cfg.sdd)
        assert det_center.shape == sources_2.shape
        assert det_u_vec.shape == sources_2.shape
        assert det_v_vec.shape == sources_2.shape

    def test_det_u_vec_unit(self, sources_2, roi_center, cfg):
        _, det_u_vec, _ = _detector_frame(sources_2, roi_center, cfg.sdd)
        norms = mx.linalg.norm(det_u_vec, axis=-1)
        mx.eval(norms)
        assert float(mx.max(mx.abs(norms - 1.0))) < 1e-5

    def test_det_v_vec_unit(self, sources_2, roi_center, cfg):
        _, _, det_v_vec = _detector_frame(sources_2, roi_center, cfg.sdd)
        norms = mx.linalg.norm(det_v_vec, axis=-1)
        mx.eval(norms)
        assert float(mx.max(mx.abs(norms - 1.0))) < 1e-5

    def test_detector_on_opposite_side(self, roi_center, cfg):
        """Detector center must lie sdd away from source along the ray."""
        source = mx.array([[200.0, 0.0, 0.0]])
        det_center, _, _ = _detector_frame(source, roi_center, cfg.sdd)
        mx.eval(det_center)
        # Source is at +x=200; detector must be at +x=200 - sdd = -200
        expected_x = 200.0 - cfg.sdd
        assert abs(float(det_center[0, 0]) - expected_x) < 1e-3


# ---------------------------------------------------------------------------
# _soft_footprint
# ---------------------------------------------------------------------------

class TestSoftFootprint:
    def test_shape_fallback(self, cfg):
        w = _soft_footprint(cfg)
        assert w.shape == (cfg.det_u, cfg.det_v)

    def test_shape_view_dependent(self, cfg, sources_2, roi_center):
        w = _soft_footprint(cfg, sources_2, roi_center)
        assert w.shape == (sources_2.shape[0], cfg.det_u, cfg.det_v)

    def test_range(self, cfg, sources_2, roi_center):
        w = _soft_footprint(cfg, sources_2, roi_center)
        mx.eval(w)
        assert float(mx.min(w)) >= cfg.footprint_floor
        assert float(mx.max(w)) <= 1.0 + 1e-6

    def test_center_is_max(self, cfg, sources_2, roi_center):
        w = _soft_footprint(cfg, sources_2, roi_center)
        mx.eval(w)
        center_u = cfg.det_u // 2
        center_v = cfg.det_v // 2
        center_val = float(w[0, center_u, center_v])
        assert center_val == pytest.approx(float(mx.max(w)), rel=1e-4)


# ---------------------------------------------------------------------------
# _detector_containment_gate
# ---------------------------------------------------------------------------

class TestContainmentGate:
    def test_large_roi_reduces_gate(self, sources_2, roi_center, cfg):
        small_roi_cfg = cfg
        large_roi_cfg = AbsorptionConfig(
            alpha=cfg.alpha,
            eta=cfg.eta,
            roi_radius=60.0,
            sid=cfg.sid,
            sdd=cfg.sdd,
            det_u=cfg.det_u,
            det_v=cfg.det_v,
            du=cfg.du,
            dv=cfg.dv,
            voxel_spacing=cfg.voxel_spacing,
            beta_pixel=cfg.beta_pixel,
            beta_frac=cfg.beta_frac,
            footprint_floor=cfg.footprint_floor,
        )
        gate_small = _detector_containment_gate(sources_2, roi_center, small_roi_cfg)
        gate_large = _detector_containment_gate(sources_2, roi_center, large_roi_cfg)
        mx.eval(gate_small, gate_large)
        assert float(mx.min(gate_large)) < float(mx.min(gate_small))
        assert float(mx.max(gate_large)) < 0.5


# ---------------------------------------------------------------------------
# absorption_gate
# ---------------------------------------------------------------------------

class TestAbsorptionGate:
    def test_output_shape(self, sources_2, roi_center, small_volume, cfg):
        nu = absorption_gate(sources_2, roi_center, small_volume, cfg)
        assert nu.shape == (sources_2.shape[0],)

    def test_range(self, sources_2, roi_center, small_volume, cfg):
        nu = absorption_gate(sources_2, roi_center, small_volume, cfg)
        mx.eval(nu)
        assert float(mx.min(nu)) > 0.0
        assert float(mx.max(nu)) < 1.0

    def test_gradient_exists_and_nonzero(self, roi_center):
        """The absorption gate must propagate a non-zero gradient to sources.

        The gradient path here is through the containment gate, which is placed
        in its active sigmoid regime: sources at SID=200, roi_radius=12 mm, and
        a det_u=48 detector give half_u=23.5 mm while the projected ROI reaches
        ±24 mm — barely outside, so worst_over_run ≈ +0.02 and
        containment_nu ≈ sigmoid(−0.21) ≈ 0.45.  The product
        absorption_nu * d(containment_nu)/d(sources) is then ≈ 1e-2, well
        above the 1e-5 threshold.  Threshold is loose to tolerate float32.
        """
        import diffct_mlx
        phantom = mx.array(diffct_mlx.shepp_logan_3d(32, 32, 32) * 0.02, dtype=mx.float32)
        close_sources = mx.array([[200.0, 0.0, 0.0], [0.0, 200.0, 0.0]], dtype=mx.float32)
        sensitive_cfg = AbsorptionConfig(
            alpha=4.0, eta=0.05, roi_radius=12.0,
            sid=200.0, sdd=400.0,
            det_u=48, det_v=48, du=1.0, dv=1.0,
            voxel_spacing=1.0, beta_pixel=4.0, beta_frac=20.0,
        )

        def obj(srcs):
            nu = absorption_gate(srcs, roi_center, phantom, sensitive_cfg)
            return mx.sum(nu)

        grad = mx.grad(obj)(close_sources)
        mx.eval(grad)
        assert grad.shape == close_sources.shape
        assert float(mx.sum(mx.abs(grad))) > 1e-5, (
            "Absorption gate returned near-zero gradient w.r.t. sources — "
            "check that the containment gate VJP is active."
        )

    def test_gradient_shape_matches_sources(self, roi_center, small_volume, cfg):
        k = 4
        sources = mx.array([
            [200.0, 0.0, 0.0],
            [0.0, 200.0, 0.0],
            [-200.0, 0.0, 0.0],
            [0.0, -200.0, 0.0],
        ])

        def obj(srcs):
            return mx.sum(absorption_gate(srcs, roi_center, small_volume, cfg))

        grad = mx.grad(obj)(sources)
        mx.eval(grad)
        assert grad.shape == (k, 3)
