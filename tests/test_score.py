"""Tests for differentiable_coverage.score."""

import math
import mlx.core as mx
import pytest

from differentiable_coverage import (
    ScoreConfig,
    accumulated_coverage,
    greedy_source_init,
    orthogonality_kernel,
    ray_directions,
    sample_unit_sphere,
    saturated_coverage,
)


# ---------------------------------------------------------------------------
# sample_unit_sphere
# ---------------------------------------------------------------------------

class TestSampleUnitSphere:
    def test_shape(self):
        pts = sample_unit_sphere(100)
        assert pts.shape == (100, 3)

    def test_unit_norm(self):
        pts = sample_unit_sphere(64)
        norms = mx.linalg.norm(pts, axis=-1)
        mx.eval(norms)
        assert float(mx.max(mx.abs(norms - 1.0))) < 1e-5

    def test_single_point(self):
        pts = sample_unit_sphere(1)
        assert pts.shape == (1, 3)

    def test_coverage_hemisphere(self):
        """Fibonacci lattice should cover both hemispheres."""
        pts = sample_unit_sphere(200)
        mx.eval(pts)
        has_pos_y = float(mx.sum(pts[:, 1] > 0)) > 0
        has_neg_y = float(mx.sum(pts[:, 1] < 0)) > 0
        assert has_pos_y and has_neg_y


# ---------------------------------------------------------------------------
# ray_directions
# ---------------------------------------------------------------------------

class TestRayDirections:
    def test_unit_length(self, small_sources, roi_center):
        dirs, dists = ray_directions(small_sources, roi_center)
        norms = mx.linalg.norm(dirs, axis=-1)
        mx.eval(norms)
        assert float(mx.max(mx.abs(norms - 1.0))) < 1e-5

    def test_distance_equals_radius(self, roi_center):
        sid = 150.0
        sources = mx.array([[sid, 0.0, 0.0]])
        _, dists = ray_directions(sources, roi_center)
        mx.eval(dists)
        assert abs(float(dists[0]) - sid) < 1e-4

    def test_direction_along_source_axis(self, roi_center):
        # ray_directions returns (sources - roi_center) / norm, i.e. the ray axis
        # sign doesn't matter for the orthogonality score (it uses dot^2)
        sources = mx.array([[100.0, 0.0, 0.0]])
        dirs, _ = ray_directions(sources, roi_center)
        mx.eval(dirs)
        assert abs(abs(float(dirs[0, 0])) - 1.0) < 1e-5  # aligned with x-axis
        assert abs(float(dirs[0, 1])) < 1e-6
        assert abs(float(dirs[0, 2])) < 1e-6

    def test_shape(self, small_sources, roi_center):
        dirs, dists = ray_directions(small_sources, roi_center)
        assert dirs.shape == small_sources.shape
        assert dists.shape == (small_sources.shape[0],)


# ---------------------------------------------------------------------------
# orthogonality_kernel
# ---------------------------------------------------------------------------

class TestOrthogonalityKernel:
    def test_shape(self, small_sources, roi_center, radon_normals):
        dirs, _ = ray_directions(small_sources, roi_center)
        psi = orthogonality_kernel(dirs, radon_normals, sigma=0.1)
        assert psi.shape == (small_sources.shape[0], radon_normals.shape[0])

    def test_range(self, small_sources, roi_center, radon_normals):
        dirs, _ = ray_directions(small_sources, roi_center)
        psi = orthogonality_kernel(dirs, radon_normals, sigma=0.1)
        mx.eval(psi)
        assert float(mx.min(psi)) >= 0.0
        assert float(mx.max(psi)) <= 1.0 + 1e-6

    def test_max_when_orthogonal(self):
        """ψ = 1 when ray is orthogonal to Radon normal (dot product = 0)."""
        ray = mx.array([[1.0, 0.0, 0.0]])
        normal = mx.array([[0.0, 1.0, 0.0]])  # orthogonal to ray
        psi = orthogonality_kernel(ray, normal, sigma=0.5)
        mx.eval(psi)
        assert abs(float(psi[0, 0]) - 1.0) < 1e-6

    def test_decreases_when_parallel(self):
        """ψ << 1 when ray is parallel to Radon normal."""
        ray = mx.array([[1.0, 0.0, 0.0]])
        normal = mx.array([[1.0, 0.0, 0.0]])  # parallel
        psi = orthogonality_kernel(ray, normal, sigma=0.1)
        mx.eval(psi)
        assert float(psi[0, 0]) < 0.01


# ---------------------------------------------------------------------------
# accumulated_coverage
# ---------------------------------------------------------------------------

class TestAccumulatedCoverage:
    def test_shape(self, small_sources, roi_center, radon_normals):
        dirs, _ = ray_directions(small_sources, roi_center)
        psi = orthogonality_kernel(dirs, radon_normals, sigma=0.1)
        nu = mx.ones(small_sources.shape[0])
        cov = accumulated_coverage(psi, nu)
        assert cov.shape == (radon_normals.shape[0],)

    def test_range(self, small_sources, roi_center, radon_normals):
        dirs, _ = ray_directions(small_sources, roi_center)
        psi = orthogonality_kernel(dirs, radon_normals, sigma=0.1)
        nu = mx.ones(small_sources.shape[0])
        cov = accumulated_coverage(psi, nu)
        mx.eval(cov)
        assert float(mx.min(cov)) >= 0.0
        assert float(mx.max(cov)) <= 1.0 + 1e-6

    def test_zero_nu_gives_zero_coverage(self, small_sources, roi_center, radon_normals):
        dirs, _ = ray_directions(small_sources, roi_center)
        psi = orthogonality_kernel(dirs, radon_normals, sigma=0.1)
        nu = mx.zeros(small_sources.shape[0])
        cov = accumulated_coverage(psi, nu)
        mx.eval(cov)
        assert float(mx.max(mx.abs(cov))) < 1e-6

    def test_monotone_in_nu(self, small_sources, roi_center, radon_normals):
        dirs, _ = ray_directions(small_sources, roi_center)
        psi = orthogonality_kernel(dirs, radon_normals, sigma=0.1)
        cov_low = accumulated_coverage(psi, 0.5 * mx.ones(small_sources.shape[0]))
        cov_high = accumulated_coverage(psi, mx.ones(small_sources.shape[0]))
        mx.eval(cov_low, cov_high)
        assert float(mx.mean(cov_high)) >= float(mx.mean(cov_low))


# ---------------------------------------------------------------------------
# saturated_coverage
# ---------------------------------------------------------------------------

class TestSaturatedCoverage:
    def test_scalar_output(self, small_sources, roi_center, radon_normals, score_cfg, ones_nu):
        cov = saturated_coverage(small_sources, roi_center, radon_normals, ones_nu, score_cfg)
        mx.eval(cov)
        assert cov.shape == ()

    def test_range(self, small_sources, roi_center, radon_normals, score_cfg, ones_nu):
        cov = saturated_coverage(small_sources, roi_center, radon_normals, ones_nu, score_cfg)
        mx.eval(cov)
        val = float(cov)
        assert 0.0 <= val <= 1.0

    def test_gradient_exists(self, small_sources, roi_center, radon_normals, score_cfg, ones_nu):
        """mx.grad must return a gradient of the same shape as sources."""
        def obj(srcs):
            return saturated_coverage(srcs, roi_center, radon_normals, ones_nu, score_cfg)

        grad = mx.grad(obj)(small_sources)
        mx.eval(grad)
        assert grad.shape == small_sources.shape

    def test_gradient_nonzero(self, small_sources, roi_center, radon_normals, score_cfg, ones_nu):
        """Gradient must be non-zero for a non-degenerate configuration."""
        def obj(srcs):
            return saturated_coverage(srcs, roi_center, radon_normals, ones_nu, score_cfg)

        grad = mx.grad(obj)(small_sources)
        mx.eval(grad)
        assert float(mx.sum(mx.abs(grad))) > 1e-8

    def test_numerical_gradient_close(self, roi_center, radon_normals, score_cfg, ones_nu):
        """Autodiff gradient should match finite differences within tolerance."""
        sources = mx.array([[200.0, 0.0, 0.0], [0.0, 200.0, 0.0]])
        nu = mx.ones(2)

        def obj(srcs):
            return saturated_coverage(srcs, roi_center, radon_normals, nu, score_cfg)

        grad_auto = mx.grad(obj)(sources)
        mx.eval(grad_auto)

        eps = 1.0
        grad_fd = mx.zeros_like(sources)
        for i in range(sources.shape[0]):
            for j in range(3):
                delta = mx.zeros_like(sources)
                delta_vals = delta.tolist()
                delta_vals[i][j] = eps
                delta = mx.array(delta_vals)
                f_plus = float(obj(sources + delta))
                f_minus = float(obj(sources - delta))
                fd_ij = (f_plus - f_minus) / (2 * eps)
                grad_fd_list = grad_fd.tolist()
                grad_fd_list[i][j] = fd_ij
                grad_fd = mx.array(grad_fd_list)

        mx.eval(grad_fd)
        rel_err = float(mx.mean(mx.abs(grad_auto - grad_fd))) / (
            float(mx.mean(mx.abs(grad_fd))) + 1e-10
        )
        assert rel_err < 0.05, f"Relative error {rel_err:.4f} exceeds 5%"


# ---------------------------------------------------------------------------
# greedy_source_init
# ---------------------------------------------------------------------------

class TestGreedySourceInit:
    def test_output_shape(self, roi_center, radon_normals, score_cfg):
        """Should return exactly k sources with shape (k, 3)."""
        candidates = mx.array([
            [200.0, 0.0, 0.0], [0.0, 200.0, 0.0],
            [-200.0, 0.0, 0.0], [0.0, -200.0, 0.0],
            [141.0, 141.0, 0.0], [-141.0, 141.0, 0.0],
        ])
        selected = greedy_source_init(candidates, roi_center, radon_normals, k=3, cfg=score_cfg)
        mx.eval(selected)
        assert selected.shape == (3, 3)

    def test_k_equals_n(self, roi_center, radon_normals, score_cfg):
        """k == len(candidates) should return all candidates."""
        candidates = mx.array([
            [200.0, 0.0, 0.0], [0.0, 200.0, 0.0], [-200.0, 0.0, 0.0],
        ])
        selected = greedy_source_init(candidates, roi_center, radon_normals, k=3, cfg=score_cfg)
        mx.eval(selected)
        assert selected.shape == (3, 3)

    def test_selected_are_from_candidates(self, roi_center, radon_normals, score_cfg):
        """Every returned source must be one of the candidates."""
        candidates = mx.array([
            [200.0, 0.0, 0.0], [0.0, 200.0, 0.0],
            [-200.0, 0.0, 0.0], [0.0, -200.0, 0.0],
        ])
        selected = greedy_source_init(candidates, roi_center, radon_normals, k=2, cfg=score_cfg)
        mx.eval(selected)
        cand_list = candidates.tolist()
        sel_list = selected.tolist()
        for s in sel_list:
            assert any(
                all(abs(s[j] - c[j]) < 1e-4 for j in range(3)) for c in cand_list
            ), f"Selected source {s} not found in candidates"

    def test_coverage_beats_single_source(self, roi_center, radon_normals, score_cfg):
        """Greedy-init k=2 should have higher coverage than the best single source."""
        candidates = mx.array([
            [200.0, 0.0, 0.0], [0.0, 200.0, 0.0],
            [-200.0, 0.0, 0.0], [0.0, -200.0, 0.0],
        ])
        selected_2 = greedy_source_init(candidates, roi_center, radon_normals, k=2, cfg=score_cfg)
        nu_2 = mx.ones(2)
        cov_2 = float(saturated_coverage(selected_2, roi_center, radon_normals, nu_2, score_cfg))

        best_single = max(
            float(saturated_coverage(
                mx.expand_dims(candidates[i], axis=0),
                roi_center, radon_normals, mx.ones(1), score_cfg,
            ))
            for i in range(candidates.shape[0])
        )
        assert cov_2 >= best_single - 1e-6


# ---------------------------------------------------------------------------
# ScoreConfig
# ---------------------------------------------------------------------------

class TestScoreConfig:
    def test_gaussian_sigma_explicit(self):
        cfg = ScoreConfig(tau=0.1, sigma=0.05)
        assert cfg.gaussian_sigma() == 0.05

    def test_gaussian_sigma_derived(self):
        cfg = ScoreConfig(tau=0.1)
        sigma = cfg.gaussian_sigma()
        assert sigma > 0.0
        # check: exp(-(tau/sigma)^2 / 2) == epsilon_at_tau
        reconstructed_eps = math.exp(-(cfg.tau / sigma) ** 2 / 2.0)
        assert abs(reconstructed_eps - cfg.epsilon_at_tau) < 1e-10

    def test_sigma_smaller_than_tau(self):
        cfg = ScoreConfig(tau=0.1, epsilon_at_tau=1e-3)
        assert cfg.gaussian_sigma() < cfg.tau
