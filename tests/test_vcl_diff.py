"""Tests for the continuous VCL information score.

Covers the custom VJP of the quadratic form γᵀR⁻¹γ against autograd on a
small explicit matrix inverse, the complete row-wise geometry VJP, and the
value range of `vcl_loss_continuous`.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from differentiable_coverage.score import _gamma_inv_quad_form
from differentiable_coverage.vcl_diff import (
    FULL_FINITE_DIFFERENCE_VJP,
    LEGACY_AUTODIFF_VJP,
    _rowwise_finite_difference_vjp,
)


def _make_spd(k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((k, k)).astype(np.float32)
    R = A @ A.T + 0.1 * np.eye(k, dtype=np.float32)
    return mx.array(R)


class TestQuadraticForm:
    @pytest.mark.parametrize("k", [4, 8])
    def test_value_matches_naive_inverse(self, k):
        """f(γ, R) = γᵀR⁻¹γ matches a direct numpy inverse for SPD R."""
        R = _make_spd(k, seed=1)
        gamma = mx.array(np.random.default_rng(2).standard_normal(k).astype(np.float32))
        f = float(_gamma_inv_quad_form(R, gamma))
        R_np = np.asarray(R); g_np = np.asarray(gamma)
        expected = float(g_np @ np.linalg.inv(R_np) @ g_np)
        assert math.isclose(f, expected, rel_tol=1e-3)

    @pytest.mark.parametrize("k", [4, 6])
    def test_gradient_with_respect_to_gamma(self, k):
        """∂f/∂γ = 2 R⁻¹ γ — checked numerically against the custom VJP."""
        R = _make_spd(k, seed=1)

        def f(gamma):
            return _gamma_inv_quad_form(R, gamma)

        # placeholder line ensures Edit replace_all is unambiguous

        gamma = mx.array(np.random.default_rng(3).standard_normal(k).astype(np.float32))
        g = mx.grad(f)(gamma)
        expected = 2 * np.linalg.solve(np.asarray(R), np.asarray(gamma))
        assert np.allclose(np.asarray(g), expected, atol=1e-3)

    @pytest.mark.parametrize("k", [4, 6])
    def test_gradient_with_respect_to_R(self, k):
        """∂f/∂R = -u uᵀ with u = R⁻¹ γ — the matrix VJP claim of the paper."""
        R = _make_spd(k, seed=1)
        gamma = mx.array(np.random.default_rng(4).standard_normal(k).astype(np.float32))

        def f(R_in):
            return _gamma_inv_quad_form(R_in, gamma)

        g = mx.grad(f)(R)
        u = np.linalg.solve(np.asarray(R), np.asarray(gamma))
        expected = -np.outer(u, u)
        assert np.allclose(np.asarray(g), expected, atol=1e-3)


class TestBasisGeometryVJP:
    def test_rowwise_finite_difference_vjp_matches_analytic_derivative(self):
        """The six-evaluation VJP returns a separate gradient for every view."""
        sources_np = np.array(
            [[0.4, -0.2, 0.7], [-0.5, 0.3, 0.2]],
            dtype=np.float32,
        )
        cotangent_np = np.array(
            [[0.8, -0.4, 0.2], [-0.1, 0.6, -0.7]],
            dtype=np.float32,
        )
        sources = mx.array(sources_np)
        cotangent = mx.array(cotangent_np)

        def basis_fn(s):
            x, y, z = s[:, 0], s[:, 1], s[:, 2]
            return mx.stack([x * x + y * z, mx.sin(y), x * z], axis=1)

        actual = np.asarray(
            _rowwise_finite_difference_vjp(
                basis_fn, sources, cotangent, step=1e-3
            )
        )

        x, y, z = sources_np[:, 0], sources_np[:, 1], sources_np[:, 2]
        c0, c1, c2 = (
            cotangent_np[:, 0],
            cotangent_np[:, 1],
            cotangent_np[:, 2],
        )
        expected = np.stack(
            [
                c0 * (2.0 * x) + c2 * z,
                c0 * z + c1 * np.cos(y),
                c0 * y + c2 * x,
            ],
            axis=1,
        )
        assert np.allclose(actual, expected, rtol=2e-3, atol=2e-4)

    def test_rowwise_vjp_uses_six_basis_evaluations_independent_of_k(self):
        calls = 0

        def basis_fn(s):
            nonlocal calls
            calls += 1
            return mx.stack([s[:, 0] + s[:, 1], s[:, 2]], axis=1)

        sources = mx.zeros((37, 3), dtype=mx.float32)
        cotangent = mx.ones((37, 2), dtype=mx.float32)
        gradient = _rowwise_finite_difference_vjp(
            basis_fn, sources, cotangent, step=0.5
        )
        mx.eval(gradient)
        assert calls == 6
        assert gradient.shape == (37, 3)


class TestVCLLossContinuous:
    """End-to-end smoke test: vcl_loss_continuous evaluates and back-props
    on a small synthetic phantom without crashing."""

    @pytest.mark.slow
    def test_footprint_projector_and_backprojector_are_adjoint(self):
        """The identity used by the VCL operator holds for the matched pair."""
        from diffct_mlx import (
            cone_backward_footprint,
            cone_forward_footprint,
        )

        from differentiable_coverage.eval.geometry import geometry_from_sources

        rng = np.random.default_rng(12)
        volume = mx.array(
            rng.normal(size=(8, 8, 8)).astype(np.float32)
        )
        sinogram = mx.array(
            rng.normal(size=(1, 8, 8)).astype(np.float32)
        )
        sources = mx.array(
            np.array([[0.0, 80.0, 5.0]], dtype=np.float32)
        )
        src, det_c, det_u, det_v = geometry_from_sources(
            sources, sid=80.0, sdd=140.0
        )
        projected = cone_forward_footprint(
            volume, src, det_c, det_u, det_v, 8, 8, 2.0, 2.0, 1.0
        )
        backprojected = cone_backward_footprint(
            sinogram,
            src,
            det_c,
            det_u,
            det_v,
            8,
            8,
            8,
            2.0,
            2.0,
            1.0,
        )
        forward_inner_product = float(mx.sum(projected * sinogram))
        adjoint_inner_product = float(mx.sum(volume * backprojected))
        assert math.isclose(
            forward_inner_product,
            adjoint_inner_product,
            rel_tol=2e-5,
            abs_tol=2e-5,
        )

    @pytest.mark.slow
    def test_smoke(self):
        from differentiable_coverage.vcl_diff import (
            build_vcl_context, vcl_loss_continuous,
        )
        rng = np.random.default_rng(0)
        vol = mx.array(rng.uniform(0, 1, (32, 32, 32)).astype(np.float32))
        ctx = build_vcl_context(
            vol, sid=100.0, sdd=180.0,
            det_shape=(16, 16),
            target_shape=(32, 32, 32), r1=1e-2,
        )
        assert ctx.geometry_vjp_mode == FULL_FINITE_DIFFERENCE_VJP
        sources = mx.array(rng.standard_normal((4, 3)).astype(np.float32)) * 100.0

        def f(s):
            return vcl_loss_continuous(s, ctx)

        v = float(f(sources))
        g = mx.grad(f)(sources)
        # 0 ≤ vcl_loss ≤ 1 by construction (= 1 - I_vcl)
        assert 0.0 <= v <= 1.0 + 1e-3
        assert bool(mx.all(mx.isfinite(g)))

    @pytest.mark.slow
    def test_complete_geometry_vjp_matches_scalar_directional_difference(self):
        """The default VJP differentiates the evaluated VCL value, end to end."""
        from differentiable_coverage.vcl_diff import (
            build_vcl_context,
            vcl_loss_continuous,
        )

        n = 12
        z, y, x = np.mgrid[:n, :n, :n]
        centre = (n - 1) / 2.0
        vol_np = np.exp(
            -(
                (x - centre) ** 2
                + 1.3 * (y - centre) ** 2
                + 0.7 * (z - centre) ** 2
            )
            / (2.0 * 2.2**2)
        ).astype(np.float32)
        ctx = build_vcl_context(
            mx.array(vol_np),
            sid=100.0,
            sdd=180.0,
            det_shape=(12, 12),
            target_shape=(12, 12, 12),
            r1=0.25,
            seed=0,
            prefer_sparse_backprojection=False,
        )
        assert ctx.geometry_fd_step == 0.5
        sources_np = np.array(
            [[0.0, 100.0, 8.0], [-92.0, 15.0, -12.0]],
            dtype=np.float32,
        )
        sources_np *= 100.0 / np.linalg.norm(
            sources_np, axis=1, keepdims=True
        )
        direction = np.array(
            [[0.7, -0.2, 0.4], [-0.3, 0.5, 0.6]],
            dtype=np.float32,
        )
        # Use feasible first-order directions on the viewing sphere.
        unit_sources = sources_np / np.linalg.norm(
            sources_np, axis=1, keepdims=True
        )
        direction -= (
            np.sum(direction * unit_sources, axis=1, keepdims=True)
            * unit_sources
        )
        direction /= np.linalg.norm(direction)

        def loss(s):
            return vcl_loss_continuous(s, ctx)

        sources = mx.array(sources_np)
        gradient = mx.grad(loss)(sources)
        autodiff_directional = float(
            mx.sum(gradient * mx.array(direction))
        )
        eps = 0.1
        finite_difference_directional = (
            float(loss(mx.array(sources_np + eps * direction)))
            - float(loss(mx.array(sources_np - eps * direction)))
        ) / (2.0 * eps)

        assert np.isclose(
            autodiff_directional,
            finite_difference_directional,
            rtol=0.12,
            atol=2e-4,
        ), (
            f"complete VCL VJP {autodiff_directional:.6g} does not match "
            f"scalar finite difference {finite_difference_directional:.6g}"
        )

    @pytest.mark.slow
    def test_geometry_vjp_mode_changes_gradient_not_value(self):
        """The legacy partial VJP remains available only as a regression mode."""
        from differentiable_coverage.vcl_diff import (
            build_vcl_context,
            vcl_loss_continuous,
        )

        rng = np.random.default_rng(7)
        vol = mx.array(rng.uniform(0, 1, (10, 10, 10)).astype(np.float32))
        common = dict(
            sid=80.0,
            sdd=140.0,
            det_shape=(10, 10),
            target_shape=(10, 10, 10),
            r1=0.2,
            seed=1,
            prefer_sparse_backprojection=False,
        )
        full_ctx = build_vcl_context(
            vol,
            geometry_vjp_mode=FULL_FINITE_DIFFERENCE_VJP,
            geometry_fd_step=0.1,
            **common,
        )
        legacy_ctx = build_vcl_context(
            vol,
            geometry_vjp_mode=LEGACY_AUTODIFF_VJP,
            **common,
        )
        sources = mx.array(
            np.array(
                [[0.0, 80.0, 5.0], [-75.0, 12.0, -8.0]],
                dtype=np.float32,
            )
        )
        full_value = float(vcl_loss_continuous(sources, full_ctx))
        legacy_value = float(vcl_loss_continuous(sources, legacy_ctx))
        assert math.isclose(full_value, legacy_value, rel_tol=1e-6, abs_tol=1e-6)

        full_grad = np.asarray(
            mx.grad(lambda s: vcl_loss_continuous(s, full_ctx))(sources)
        )
        legacy_grad = np.asarray(
            mx.grad(lambda s: vcl_loss_continuous(s, legacy_ctx))(sources)
        )
        assert not np.allclose(full_grad, legacy_grad, rtol=1e-2, atol=1e-5)
