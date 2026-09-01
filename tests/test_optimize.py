"""Tests for differentiable_coverage.optimize."""

import math
import mlx.core as mx
import pytest

from differentiable_coverage import adam_ascent, anneal, gradient_ascent


# ---------------------------------------------------------------------------
# anneal
# ---------------------------------------------------------------------------

class TestAnneal:
    def test_geometric_start(self):
        assert anneal(0, 10, start=1.0, end=0.01) == pytest.approx(1.0)

    def test_geometric_end(self):
        assert anneal(9, 10, start=1.0, end=0.01) == pytest.approx(0.01)

    def test_linear_midpoint(self):
        val = anneal(5, 11, start=0.0, end=1.0, kind="linear")
        assert val == pytest.approx(0.5)

    def test_geometric_monotone_decreasing(self):
        vals = [anneal(i, 20, start=1.0, end=0.001) for i in range(20)]
        for a, b in zip(vals, vals[1:]):
            assert a >= b

    def test_single_step(self):
        assert anneal(0, 1, start=5.0, end=2.0) == pytest.approx(2.0)

    def test_geometric_product_rule(self):
        """Ratio between consecutive geometric steps must be constant."""
        vals = [anneal(i, 5, start=8.0, end=1.0) for i in range(5)]
        ratios = [vals[i + 1] / vals[i] for i in range(4)]
        for r in ratios:
            assert r == pytest.approx(ratios[0], rel=1e-5)


# ---------------------------------------------------------------------------
# gradient_ascent
# ---------------------------------------------------------------------------

class TestGradientAscent:
    def _quadratic_fn(self, optimum=3.0):
        """Returns a concave objective f(x) = -(x - opt)^2, max at x=opt."""
        def fn(params, step):
            return -mx.sum((params - optimum) ** 2)
        return fn

    def test_converges_to_optimum(self):
        fn = self._quadratic_fn(optimum=3.0)
        params = mx.array([0.0])
        final, history = gradient_ascent(fn, params, lr=0.1, n_steps=100)
        mx.eval(final)
        assert abs(float(final[0]) - 3.0) < 0.05

    def test_history_length(self):
        fn = self._quadratic_fn()
        params = mx.array([0.0])
        _, history = gradient_ascent(fn, params, lr=0.1, n_steps=50)
        assert len(history) == 50

    def test_history_increases(self):
        fn = self._quadratic_fn(optimum=5.0)
        params = mx.array([0.0])
        _, history = gradient_ascent(fn, params, lr=0.05, n_steps=30)
        # objective should increase over the first steps
        assert history[-1] > history[0]

    def test_callback_called(self):
        fn = self._quadratic_fn()
        calls = []
        def cb(step, params, val):
            calls.append(step)
        params = mx.array([0.0])
        gradient_ascent(fn, params, lr=0.1, n_steps=10, callback=cb)
        assert calls == list(range(10))

    def test_multidim_params(self):
        """Works with 2D parameter arrays."""
        def fn(params, step):
            return -mx.sum(params ** 2)
        params = mx.array([[2.0, -1.0], [0.5, 3.0]])
        final, _ = gradient_ascent(fn, params, lr=0.1, n_steps=50)
        mx.eval(final)
        assert float(mx.max(mx.abs(final))) < 0.5

    def test_step_index_passed(self):
        """The step index must count up correctly in coverage_fn."""
        received = []
        def fn(params, step):
            received.append(step)
            return mx.sum(params)
        params = mx.array([1.0])
        gradient_ascent(fn, params, lr=0.0, n_steps=5)
        assert received == [0, 1, 2, 3, 4]

    def test_tol_grad_stops_early(self):
        """tol_grad stops optimization once gradient is tiny."""
        fn = self._quadratic_fn(optimum=0.0)  # already at optimum → grad≈0
        params = mx.array([0.0])
        _, history = gradient_ascent(fn, params, lr=0.1, n_steps=100, tol_grad=1e-3)
        assert len(history) < 100

    def test_tol_rel_stops_early(self):
        """tol_rel stops when improvement per step is negligible."""
        fn = self._quadratic_fn(optimum=0.0)
        params = mx.array([0.0])
        _, history = gradient_ascent(fn, params, lr=0.1, n_steps=100, tol_rel=1e-6)
        assert len(history) < 100


# ---------------------------------------------------------------------------
# adam_ascent
# ---------------------------------------------------------------------------

class TestAdamAscent:
    def _quadratic_fn(self, optimum=3.0):
        def fn(params, step):
            return -mx.sum((params - optimum) ** 2)
        return fn

    def test_converges_to_optimum(self):
        fn = self._quadratic_fn(optimum=3.0)
        params = mx.array([0.0])
        final, history = adam_ascent(fn, params, lr=0.1, n_steps=200)
        mx.eval(final)
        assert abs(float(final[0]) - 3.0) < 0.1

    def test_history_length(self):
        fn = self._quadratic_fn()
        params = mx.array([0.0])
        _, history = adam_ascent(fn, params, lr=0.1, n_steps=30)
        assert len(history) == 30

    def test_improves_objective(self):
        fn = self._quadratic_fn(optimum=5.0)
        params = mx.array([0.0])
        _, history = adam_ascent(fn, params, lr=0.05, n_steps=50)
        assert history[-1] > history[0]

    def test_return_best_pairs_value_with_evaluated_iterate(self):
        """Best tracking must not attach a pre-update value to new params."""
        fn = self._quadratic_fn(optimum=0.0)
        start = mx.array([1.0])
        final, history = adam_ascent(fn, start, lr=10.0, n_steps=2)
        mx.eval(final)

        assert history[0] == pytest.approx(-1.0)
        assert float(final[0]) == pytest.approx(1.0)

    def test_callback_pairs_value_with_evaluated_iterate(self):
        fn = self._quadratic_fn(optimum=0.0)
        seen = []

        def cb(step, params, value):
            mx.eval(params)
            seen.append((step, float(params[0]), value))

        adam_ascent(
            fn,
            mx.array([1.0]),
            lr=10.0,
            n_steps=2,
            callback=cb,
            return_best=False,
        )

        assert seen[0] == pytest.approx((0, 1.0, -1.0))

    def test_faster_than_sgd_on_ill_conditioned(self):
        """Adam should reach the optimum faster than plain SGD on a scaled problem."""
        def fn(params, step):
            return -(params[0] ** 2 * 0.01 + params[1] ** 2 * 100.0)

        start = mx.array([5.0, 5.0])
        _, hist_adam = adam_ascent(fn, start, lr=0.01, n_steps=50)
        _, hist_sgd = gradient_ascent(fn, start, lr=0.01, n_steps=50)
        assert hist_adam[-1] > hist_sgd[-1]

    def test_tol_grad_stops_early(self):
        fn = self._quadratic_fn(optimum=0.0)
        params = mx.array([0.0])
        _, history = adam_ascent(fn, params, lr=0.1, n_steps=100, tol_grad=1e-3)
        assert len(history) < 100


# ----------------------------------------------------------------------
# Matched central-FD gradient mode (REV-P0-01 derivative-only ablation)
# ----------------------------------------------------------------------

class TestCentralFDGradMode:
    def test_fd_grad_matches_analytic_on_smooth_objective(self):
        """Central differences are exact for quadratics up to float noise."""
        from differentiable_coverage.optimize import central_fd_grad
        import numpy as np

        c = mx.array([0.3, -1.2, 2.0, 0.7])

        def f(p):
            return -mx.sum((p - c) ** 2)

        p0 = mx.array([1.0, 1.0, -1.0, 0.0])
        g_an = mx.grad(lambda p: f(p))(p0)
        g_fd = central_fd_grad(f, p0, fd_step=1e-3)
        np.testing.assert_allclose(np.asarray(g_fd), np.asarray(g_an),
                                   rtol=1e-3, atol=1e-4)

    def test_adam_trajectories_identical_when_fd_is_exact(self):
        """On a quadratic objective, fd_central and analytic must produce
        the same optimisation trajectory: the ONLY difference between the
        modes is the gradient estimator."""
        import numpy as np

        c = mx.array([0.5, -0.25])

        def fn(p, _step):
            return -mx.sum((p - c) ** 2)

        start = mx.array([2.0, 2.0])
        p_an, h_an = adam_ascent(fn, start, lr=0.1, n_steps=25,
                                 grad_mode="analytic")
        p_fd, h_fd = adam_ascent(fn, start, lr=0.1, n_steps=25,
                                 grad_mode="fd_central", fd_step=1e-3)
        np.testing.assert_allclose(np.asarray(p_fd), np.asarray(p_an),
                                   rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(h_fd, h_an, rtol=1e-3, atol=1e-4)

    def test_unknown_grad_mode_raises(self):
        def fn(p, _step):
            return -mx.sum(p ** 2)

        with pytest.raises(ValueError, match="grad_mode"):
            adam_ascent(fn, mx.array([1.0]), n_steps=1, grad_mode="bogus")

    def test_fd_grad_tracks_bundle_objective(self):
        """On the (rougher) bundle objective with the frozen clipped rule,
        the FD gradient must at least point the same way as autograd."""
        import numpy as np
        from differentiable_coverage.absorption_bundle import (
            BundleAbsorptionConfig, bundle_path_integral,
        )
        from differentiable_coverage.optimize import central_fd_grad

        N = 32
        zz, yy, xx = mx.meshgrid(*(mx.arange(N) - (N - 1) / 2.0,) * 3,
                                 indexing="ij")
        r2 = zz * zz + yy * yy + xx * xx
        # Smooth Gaussian blob (avoids binary-edge kink noise in this test).
        vol = mx.exp(-r2 / (2.0 * 5.0 ** 2))
        roi = mx.zeros(3)
        cfg = BundleAbsorptionConfig(
            roi_radius=2.0, n_rays_u=3, n_rays_v=3, n_samples=128,
            voxel_spacing=1.0, clip_to_volume=True,
        )
        sources = mx.array([[30.0, 40.0, 20.0], [-50.0, 10.0, -30.0]])

        def f(s):
            return -mx.mean(bundle_path_integral(s, roi, vol, cfg))

        g_an = np.asarray(mx.grad(f)(sources), dtype=np.float64)
        g_fd = np.asarray(central_fd_grad(f, sources, fd_step=0.25),
                          dtype=np.float64)
        cos = (np.sum(g_an * g_fd)
               / max(np.linalg.norm(g_an) * np.linalg.norm(g_fd), 1e-30))
        assert cos > 0.95
        assert 0.5 < np.linalg.norm(g_fd) / np.linalg.norm(g_an) < 2.0


# ----------------------------------------------------------------------
# CMA-ES comparator (REV-P1-08 budget-matched derivative-free arm)
# ----------------------------------------------------------------------

class TestCMAESAscent:
    def test_reaches_quadratic_optimum(self):
        import numpy as np
        from differentiable_coverage.optimize import cmaes_ascent

        c = mx.array([1.0, -2.0, 0.5])

        def fn(p, _step):
            return -mx.sum((p - c) ** 2)

        start = mx.array([5.0, 5.0, 5.0])
        best, hist = cmaes_ascent(fn, start, sigma0=1.0, budget_evals=2000,
                                  seed=0)
        np.testing.assert_allclose(np.asarray(best), np.asarray(c),
                                   atol=1e-2)
        # best-so-far history is monotone non-decreasing
        assert all(b >= a - 1e-12 for a, b in zip(hist, hist[1:]))

    def test_respects_projection(self):
        import numpy as np
        from differentiable_coverage.optimize import cmaes_ascent

        sid = 10.0

        def project(p):
            n = mx.linalg.norm(p, axis=-1, keepdims=True)
            return p * (sid / mx.maximum(n, 1e-6))

        target = mx.array([[0.0, 0.0, sid]])

        def fn(p, _step):
            return -mx.sum((p - target) ** 2)

        start = project(mx.array([[5.0, 5.0, 5.0]]))
        best, _ = cmaes_ascent(fn, start, sigma0=1.0, budget_evals=1500,
                               seed=0, project_fn=project)
        n = float(mx.linalg.norm(best, axis=-1)[0])
        assert math.isclose(n, sid, rel_tol=1e-4)
        assert float(best[0, 2]) > 0.9 * sid

    def test_deterministic_for_fixed_seed(self):
        import numpy as np
        from differentiable_coverage.optimize import cmaes_ascent

        def fn(p, _step):
            return -mx.sum(p ** 2)

        start = mx.array([3.0, -3.0])
        b1, h1 = cmaes_ascent(fn, start, sigma0=0.5, budget_evals=400, seed=3)
        b2, h2 = cmaes_ascent(fn, start, sigma0=0.5, budget_evals=400, seed=3)
        np.testing.assert_array_equal(np.asarray(b1), np.asarray(b2))
        assert h1 == h2

    def test_selector_optimizer_switch_end_to_end(self):
        """greedy_adam_vcl_continuous(optimizer='cmaes') runs the SAME
        objective/init/projection path and returns on-manifold sources."""
        import numpy as np
        from differentiable_coverage.eval.trajectories import (
            greedy_adam_vcl_continuous,
        )

        N = 24
        rng = np.random.default_rng(0)
        vol = mx.array((rng.uniform(0, 1, (N, N, N)) > 0.97)
                       .astype(np.float32))
        k, sid = 6, 80.0
        common = dict(
            roi_center=mx.zeros(3), volume=vol, sdd=140.0,
            lambda_cov=1.0, lambda_vcl=0.0, lambda_path=0.0,
            lambda_bundle=0.5, init_method="greedy_tuy",
            n_candidates=40, n_normals=60, n_steps=5, seed=0,
        )
        src = greedy_adam_vcl_continuous(
            k, sid, optimizer="cmaes", optimizer_budget_evals=120, **common)
        assert src.shape == (k, 3)
        radii = np.asarray(mx.linalg.norm(src, axis=-1))
        np.testing.assert_allclose(radii, sid, rtol=1e-4)

    def test_unknown_optimizer_raises(self):
        import numpy as np
        import pytest as _pytest
        from differentiable_coverage.eval.trajectories import (
            greedy_adam_vcl_continuous,
        )

        vol = mx.zeros((8, 8, 8))
        with _pytest.raises(ValueError, match="optimizer"):
            greedy_adam_vcl_continuous(
                4, 50.0, roi_center=mx.zeros(3), volume=vol,
                optimizer="bogus", n_candidates=10, n_normals=20,
                n_steps=2, seed=0, lambda_vcl=0.0, lambda_cov=1.0,
                init_method="greedy_tuy")
