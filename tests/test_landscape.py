"""Tests for differentiable_coverage.landscape."""

import mlx.core as mx
import pytest

from differentiable_coverage import (
    ScoreConfig,
    adam_ascent,
    coverage_stats,
    gradient_ascent,
    multi_restart,
    random_sphere_sources,
    saturated_coverage,
    sample_unit_sphere,
)
from differentiable_coverage.landscape import RestartResult


# ---------------------------------------------------------------------------
# random_sphere_sources
# ---------------------------------------------------------------------------

class TestRandomSphereSources:
    def test_shape(self):
        sources = random_sphere_sources(8, sid=200.0, seed=0)
        assert sources.shape == (8, 3)

    def test_radius(self):
        sid = 150.0
        sources = random_sphere_sources(32, sid=sid, seed=1)
        norms = mx.linalg.norm(sources, axis=-1)
        mx.eval(norms)
        assert float(mx.max(mx.abs(norms - sid))) < 1e-3

    def test_different_seeds_differ(self):
        s0 = random_sphere_sources(4, 200.0, seed=0)
        s1 = random_sphere_sources(4, 200.0, seed=1)
        mx.eval(s0, s1)
        diff = float(mx.sum(mx.abs(s0 - s1)))
        assert diff > 1e-3

    def test_k_equals_one(self):
        sources = random_sphere_sources(1, sid=100.0, seed=42)
        assert sources.shape == (1, 3)
        norm = float(mx.linalg.norm(sources[0]))
        assert abs(norm - 100.0) < 1e-3


# ---------------------------------------------------------------------------
# multi_restart
# ---------------------------------------------------------------------------

class TestMultiRestart:
    """Uses a tiny 2D toy objective to keep tests fast."""

    @pytest.fixture
    def _setup(self):
        radon_normals = sample_unit_sphere(30)
        cfg = ScoreConfig(tau=0.07)
        roi_center = mx.zeros(3)
        return radon_normals, cfg, roi_center

    def test_returns_n_results(self, _setup):
        radon_normals, cfg, roi_center = _setup

        def coverage_fn(params, step):
            nu = mx.ones(params.shape[0])
            return saturated_coverage(params, roi_center, radon_normals, nu, cfg)

        def init_fn(seed):
            return random_sphere_sources(3, 200.0, seed=seed)

        def optimizer_fn(fn, params):
            return gradient_ascent(fn, params, lr=5.0, n_steps=5)

        results, best = multi_restart(coverage_fn, init_fn, optimizer_fn, n_restarts=4)
        assert len(results) == 4

    def test_best_is_highest(self, _setup):
        radon_normals, cfg, roi_center = _setup

        def coverage_fn(params, step):
            nu = mx.ones(params.shape[0])
            return saturated_coverage(params, roi_center, radon_normals, nu, cfg)

        def init_fn(seed):
            return random_sphere_sources(3, 200.0, seed=seed)

        def optimizer_fn(fn, params):
            return gradient_ascent(fn, params, lr=5.0, n_steps=5)

        results, best = multi_restart(coverage_fn, init_fn, optimizer_fn, n_restarts=4)
        for r in results:
            assert best.coverage >= r.coverage - 1e-9

    def test_seeds_recorded(self, _setup):
        radon_normals, cfg, roi_center = _setup

        def coverage_fn(params, step):
            nu = mx.ones(params.shape[0])
            return saturated_coverage(params, roi_center, radon_normals, nu, cfg)

        def init_fn(seed):
            return random_sphere_sources(2, 200.0, seed=seed)

        def optimizer_fn(fn, params):
            return gradient_ascent(fn, params, lr=5.0, n_steps=3)

        results, _ = multi_restart(coverage_fn, init_fn, optimizer_fn, n_restarts=3)
        assert [r.seed for r in results] == [0, 1, 2]

    def test_curves_recorded(self, _setup):
        radon_normals, cfg, roi_center = _setup

        def coverage_fn(params, step):
            nu = mx.ones(params.shape[0])
            return saturated_coverage(params, roi_center, radon_normals, nu, cfg)

        def init_fn(seed):
            return random_sphere_sources(2, 200.0, seed=seed)

        n_steps = 4

        def optimizer_fn(fn, params):
            return gradient_ascent(fn, params, lr=5.0, n_steps=n_steps)

        results, _ = multi_restart(coverage_fn, init_fn, optimizer_fn, n_restarts=2)
        for r in results:
            assert len(r.curve) == n_steps

    def test_best_final_coverage_non_trivial(self, _setup):
        radon_normals, cfg, roi_center = _setup

        def coverage_fn(params, step):
            nu = mx.ones(params.shape[0])
            return saturated_coverage(params, roi_center, radon_normals, nu, cfg)

        def init_fn(seed):
            return random_sphere_sources(4, 200.0, seed=seed)

        def optimizer_fn(fn, params):
            return gradient_ascent(fn, params, lr=10.0, n_steps=20)

        _, best = multi_restart(coverage_fn, init_fn, optimizer_fn, n_restarts=4)
        assert best.coverage > 0.0

    def test_adam_ascent_compatible(self, _setup):
        radon_normals, cfg, roi_center = _setup

        def coverage_fn(params, step):
            nu = mx.ones(params.shape[0])
            return saturated_coverage(params, roi_center, radon_normals, nu, cfg)

        def init_fn(seed):
            return random_sphere_sources(3, 200.0, seed=seed)

        def optimizer_fn(fn, params):
            return adam_ascent(fn, params, lr=5.0, n_steps=5)

        results, best = multi_restart(coverage_fn, init_fn, optimizer_fn, n_restarts=3)
        assert len(results) == 3
        assert best.coverage >= 0.0


# ---------------------------------------------------------------------------
# coverage_stats
# ---------------------------------------------------------------------------

class TestCoverageStats:
    def _make_results(self, coverages):
        return [
            RestartResult(params=mx.zeros(3), coverage=c, curve=[c], seed=i)
            for i, c in enumerate(coverages)
        ]

    def test_basic_keys(self):
        results = self._make_results([0.3, 0.5, 0.7])
        stats = coverage_stats(results)
        for key in ("mean", "std", "min", "max", "median", "best"):
            assert key in stats

    def test_min_max(self):
        results = self._make_results([0.1, 0.5, 0.9])
        stats = coverage_stats(results)
        assert abs(stats["min"] - 0.1) < 1e-9
        assert abs(stats["max"] - 0.9) < 1e-9

    def test_mean(self):
        results = self._make_results([0.0, 1.0])
        stats = coverage_stats(results)
        assert abs(stats["mean"] - 0.5) < 1e-9

    def test_single_element(self):
        results = self._make_results([0.7])
        stats = coverage_stats(results)
        assert abs(stats["min"] - 0.7) < 1e-9
        assert abs(stats["max"] - 0.7) < 1e-9
        assert abs(stats["mean"] - 0.7) < 1e-9
