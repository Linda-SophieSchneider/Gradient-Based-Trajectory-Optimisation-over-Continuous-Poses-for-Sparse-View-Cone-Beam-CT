"""Tests for the three exploration mechanisms added to the continuous
Adam loop in Sec. III-G and ablated in Sec. VI-C of the paper.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from differentiable_coverage.optimize import adam_ascent


# ----------------------------------------------------------------------
# Langevin noise hook in adam_ascent
# ----------------------------------------------------------------------

class TestLangevinNoise:
    def test_noise_fn_perturbs_iterates(self):
        """When a noise_fn is supplied, the iterate is no longer the
        deterministic Adam trajectory."""
        target = mx.array([1.0, 1.0])

        def obj(p, _step):
            return -mx.sum((p - target) ** 2)

        x0 = mx.zeros(2)
        # Deterministic Adam
        x_det, _ = adam_ascent(obj, x0, lr=0.1, n_steps=20)
        # Adam + tangent Gaussian noise
        rng_state = {"key": mx.random.key(0)}

        def noise_fn(p, _step, lr_t):
            rng_state["key"], sub = mx.random.split(rng_state["key"])
            return 0.05 * mx.random.normal(shape=p.shape, key=sub)

        x_noisy, _ = adam_ascent(
            obj, x0, lr=0.1, n_steps=20, noise_fn=noise_fn,
            return_best=False,
        )
        # Noise should produce a different iterate
        d = float(mx.linalg.norm(x_det - x_noisy))
        assert d > 1e-3

    def test_noise_fn_zero_recovers_deterministic(self):
        target = mx.array([1.0, 1.0])

        def obj(p, _step):
            return -mx.sum((p - target) ** 2)

        x0 = mx.zeros(2)
        x_a, _ = adam_ascent(obj, x0, lr=0.1, n_steps=10)
        x_b, _ = adam_ascent(
            obj, x0, lr=0.1, n_steps=10,
            noise_fn=lambda p, s, lr: mx.zeros_like(p),
        )
        assert float(mx.linalg.norm(x_a - x_b)) < 1e-6


# ----------------------------------------------------------------------
# Sigma-annealing hook in greedy_adam_vcl_continuous
# ----------------------------------------------------------------------

class TestSigmaAnnealing:
    def test_sigma_schedule_is_a_no_op_when_constant(self):
        """When cov_sigma_schedule='constant', the sigma used inside the
        coverage loss equals the default cfg.gaussian_sigma() at every step.
        We verify this indirectly: the function should still produce a
        finite source set with the same shape as the input k."""
        from differentiable_coverage.eval.trajectories import (
            greedy_adam_vcl_continuous,
        )
        rng = np.random.default_rng(0)
        vol = mx.array(rng.uniform(0, 0.5, (16, 16, 16)).astype(np.float32))
        sources = greedy_adam_vcl_continuous(
            k=8, sid=100.0,
            roi_center=mx.array([0.0, 0.0, 0.0]),
            volume=vol, sdd=180.0,
            lambda_cov=1.0, lambda_vcl=0.0, lambda_path=0.0,
            n_steps=5, init_method="greedy_tuy",
            n_candidates=24, n_normals=24,
            cov_sigma_schedule="constant",
        )
        assert sources.shape == (8, 3)
        # All sources on or near the sphere of radius sid
        norms = mx.linalg.norm(sources, axis=-1)
        assert mx.all(mx.abs(norms - 100.0) < 1e-2)

    def test_sigma_cosine_schedule_returns_valid_solution(self):
        from differentiable_coverage.eval.trajectories import (
            greedy_adam_vcl_continuous,
        )
        rng = np.random.default_rng(0)
        vol = mx.array(rng.uniform(0, 0.5, (16, 16, 16)).astype(np.float32))
        sources = greedy_adam_vcl_continuous(
            k=8, sid=100.0,
            roi_center=mx.array([0.0, 0.0, 0.0]),
            volume=vol, sdd=180.0,
            lambda_cov=1.0, lambda_vcl=0.0, lambda_path=0.0,
            n_steps=5, init_method="greedy_tuy",
            n_candidates=24, n_normals=24,
            cov_sigma_schedule="cosine", cov_sigma_mult=4.0,
        )
        assert sources.shape == (8, 3)
        assert bool(mx.all(mx.isfinite(sources)))


# ----------------------------------------------------------------------
# Repulsive ensemble
# ----------------------------------------------------------------------

class TestEnsemble:
    def test_ensemble_returns_single_member(self):
        """`greedy_ensemble_vcl_continuous` selects one of N members."""
        from differentiable_coverage.eval.trajectories import (
            greedy_ensemble_vcl_continuous,
        )
        rng = np.random.default_rng(0)
        vol = mx.array(rng.uniform(0, 0.5, (16, 16, 16)).astype(np.float32))
        best = greedy_ensemble_vcl_continuous(
            k=6, sid=100.0,
            roi_center=mx.array([0.0, 0.0, 0.0]),
            volume=vol, sdd=180.0,
            n_ensemble=4, repulsion_weight=0.01,
            init_jitter=0.05,
            lambda_cov=1.0, lambda_vcl=0.0, lambda_path=0.0,
            n_steps=3, init_method="greedy_tuy",
            n_candidates=24, n_normals=24,
        )
        assert best.shape == (6, 3)
        norms = mx.linalg.norm(best, axis=-1)
        assert mx.all(mx.abs(norms - 100.0) < 1e-2)

    def test_diverse_init_strategy(self):
        """`init_strategy='diverse'` uses VCLS + greedy + random for the four
        members.  Smoke test: does not error out, returns one valid member."""
        from differentiable_coverage.eval.trajectories import (
            greedy_ensemble_vcl_continuous,
        )
        rng = np.random.default_rng(0)
        vol = mx.array(rng.uniform(0, 0.5, (16, 16, 16)).astype(np.float32))
        best = greedy_ensemble_vcl_continuous(
            k=6, sid=100.0,
            roi_center=mx.array([0.0, 0.0, 0.0]),
            volume=vol, sdd=180.0,
            n_ensemble=4, init_strategy="diverse",
            repulsion_weight=0.1, repulsion_schedule="cosine",
            lambda_cov=1.0, lambda_vcl=0.0, lambda_path=0.0,
            n_steps=3,
            n_candidates=24, n_normals=24,
        )
        assert best.shape == (6, 3)
        assert bool(mx.all(mx.isfinite(best)))
