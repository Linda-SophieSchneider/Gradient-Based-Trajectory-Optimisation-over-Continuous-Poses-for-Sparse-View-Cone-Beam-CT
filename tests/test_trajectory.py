"""Tests for differentiable_coverage.trajectory."""

import math
import mlx.core as mx
import pytest

from differentiable_coverage import (
    CArmTwoAxisGantry, CircularArc, Free3D, Helix, TwoAxisGantry,
)


class TestCircularArc:
    def test_output_shape(self):
        arc = CircularArc(sid=200.0)
        theta = mx.linspace(0.0, math.pi, 8)
        out = arc(theta)
        assert out.shape == (8, 3)

    def test_radius(self):
        sid = 150.0
        arc = CircularArc(sid=sid)
        theta = mx.linspace(0.0, 2 * math.pi, 16)
        out = arc(theta)
        radii = mx.sqrt(out[:, 0] ** 2 + out[:, 1] ** 2)
        mx.eval(radii)
        assert float(mx.max(mx.abs(radii - sid))) < 1e-4

    def test_fixed_z(self):
        z_val = 5.0
        arc = CircularArc(sid=100.0, z=z_val)
        theta = mx.linspace(0.0, math.pi, 4)
        out = arc(theta)
        mx.eval(out)
        assert float(mx.max(mx.abs(out[:, 2] - z_val))) < 1e-6

    def test_gradient_flows(self):
        arc = CircularArc(sid=200.0)

        def obj(theta):
            sources = arc(theta)
            return mx.sum(sources)

        theta = mx.linspace(0.0, math.pi, 4)
        grad = mx.grad(obj)(theta)
        mx.eval(grad)
        assert grad.shape == theta.shape
        assert float(mx.sum(mx.abs(grad))) > 0.0


class TestHelix:
    def test_output_shape(self):
        helix = Helix(sid=200.0, pitch=50.0)
        theta = mx.linspace(0.0, 2 * math.pi, 10)
        out = helix(theta)
        assert out.shape == (10, 3)

    def test_xy_radius(self):
        sid = 120.0
        helix = Helix(sid=sid, pitch=30.0)
        theta = mx.linspace(0.0, 2 * math.pi, 20)
        out = helix(theta)
        radii = mx.sqrt(out[:, 0] ** 2 + out[:, 1] ** 2)
        mx.eval(radii)
        assert float(mx.max(mx.abs(radii - sid))) < 1e-4

    def test_z_linear_in_theta(self):
        pitch = 40.0
        helix = Helix(sid=100.0, pitch=pitch)
        theta = mx.array([0.0, 2 * math.pi])
        out = helix(theta)
        mx.eval(out)
        assert abs(float(out[1, 2]) - pitch) < 1e-4

    def test_gradient_flows(self):
        helix = Helix(sid=200.0, pitch=50.0)

        def obj(theta):
            sources = helix(theta)
            return mx.sum(sources[:, 2])  # gradient w.r.t. z component

        theta = mx.linspace(0.0, math.pi, 6)
        grad = mx.grad(obj)(theta)
        mx.eval(grad)
        assert grad.shape == theta.shape
        assert float(mx.sum(mx.abs(grad))) > 0.0


class TestTwoAxisGantry:
    def test_output_shape(self):
        gantry = TwoAxisGantry(sid=200.0)
        params = mx.zeros((6, 2))
        out = gantry(params)
        assert out.shape == (6, 3)

    def test_radius(self):
        sid = 150.0
        gantry = TwoAxisGantry(sid=sid)
        # Zero elevation: sources lie in xy-plane
        params = mx.stack([mx.linspace(0.0, math.pi, 8), mx.zeros(8)], axis=-1)
        out = gantry(params)
        radii = mx.linalg.norm(out, axis=-1)
        mx.eval(radii)
        assert float(mx.max(mx.abs(radii - sid))) < 1e-4

    def test_elevation_axis(self):
        """At θ=0, φ=π/2: source should be at (0, 0, sid)."""
        gantry = TwoAxisGantry(sid=100.0)
        params = mx.array([[0.0, math.pi / 2]])
        out = gantry(params)
        mx.eval(out)
        assert abs(float(out[0, 2]) - 100.0) < 1e-4

    def test_gradient_flows(self):
        gantry = TwoAxisGantry(sid=200.0)

        def obj(params):
            sources = gantry(params)
            return mx.sum(sources)

        params = mx.zeros((4, 2))
        grad = mx.grad(obj)(params)
        mx.eval(grad)
        assert grad.shape == params.shape
        assert float(mx.sum(mx.abs(grad))) > 0.0


class TestCArmTwoAxisGantry:
    def test_clamps_to_limits(self):
        gantry = CArmTwoAxisGantry(sid=100.0)
        params = mx.array([
            [10.0, 10.0],
            [-10.0, -10.0],
        ])
        clamped = gantry.clamp(params)
        mx.eval(clamped)
        assert float(mx.max(clamped[:, 0])) <= gantry.theta_max + 1e-6
        assert float(mx.min(clamped[:, 0])) >= gantry.theta_min - 1e-6
        assert float(mx.max(clamped[:, 1])) <= gantry.phi_max + 1e-6
        assert float(mx.min(clamped[:, 1])) >= gantry.phi_min - 1e-6

    def test_output_uses_clamped_angles(self):
        gantry = CArmTwoAxisGantry(sid=100.0)
        params = mx.array([[10.0, 10.0]])
        out_a = gantry(params)
        out_b = gantry(gantry.clamp(params))
        mx.eval(out_a, out_b)
        assert float(mx.max(mx.abs(out_a - out_b))) < 1e-6


class TestFree3D:
    def test_identity(self):
        free = Free3D()
        sources = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = free(sources)
        mx.eval(out)
        assert float(mx.sum(mx.abs(out - sources))) < 1e-9

    def test_gradient_flows(self):
        free = Free3D()

        def obj(srcs):
            return mx.sum(free(srcs))

        sources = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        grad = mx.grad(obj)(sources)
        mx.eval(grad)
        assert grad.shape == sources.shape
        assert float(mx.sum(mx.abs(grad - mx.ones_like(sources)))) < 1e-6
