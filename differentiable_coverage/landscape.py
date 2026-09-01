"""Landscape analysis: multiple random restarts and coverage-curve recording (§6).

Empirically characterises the non-convexity of the saturated coverage
objective by running the same optimiser from many random initialisations and
collecting per-restart coverage curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx


@dataclass
class RestartResult:
    """Outcome of a single optimisation restart."""

    params: mx.array
    coverage: float
    curve: list[float]
    seed: int


def random_sphere_sources(
    k: int,
    sid: float,
    *,
    seed: int = 0,
    dtype=mx.float32,
) -> mx.array:
    """k uniform-random source positions on a sphere of radius *sid*.

    Uses a Box-Muller / Gaussian-normalise construction so each point is
    drawn from the uniform distribution on S^2.

    Parameters
    ----------
    k : int
        Number of sources.
    sid : float
        Source-to-isocenter distance (sphere radius).
    seed : int
        MLX random seed (each call resets the global seed).
    dtype :
        Output dtype.

    Returns
    -------
    sources : ``(k, 3)``
    """
    mx.random.seed(seed)
    raw = mx.random.normal(shape=(k, 3)).astype(dtype)
    norms = mx.linalg.norm(raw, axis=-1, keepdims=True)
    unit = raw / mx.maximum(norms, 1e-9)
    sources = sid * unit
    mx.eval(sources)
    return sources


def multi_restart(
    coverage_fn: Callable[[mx.array, int], mx.array],
    init_fn: Callable[[int], mx.array],
    optimizer_fn: Callable[
        [Callable[[mx.array, int], mx.array], mx.array],
        tuple[mx.array, list[float]],
    ],
    n_restarts: int,
) -> tuple[list[RestartResult], RestartResult]:
    """Run *optimizer_fn* from *n_restarts* independent initialisations.

    Parameters
    ----------
    coverage_fn : callable(params, step) -> scalar
        The differentiable objective to maximise.  Same signature as the
        first argument of :func:`gradient_ascent` / :func:`adam_ascent`.
    init_fn : callable(seed) -> params
        Returns an initial parameter array for integer seed *seed*.  Called
        once per restart with ``seed = 0, 1, ..., n_restarts - 1``.
    optimizer_fn : callable(coverage_fn, init_params) -> (params, history)
        Wraps one of the optimisers from :mod:`optimize`.  Must match the
        return contract ``(final_params, list[float])``.
    n_restarts : int
        Number of independent restarts.

    Returns
    -------
    results : list[RestartResult]
        One entry per restart, in seed order.
    best : RestartResult
        The restart with the highest final coverage value.
    """
    results: list[RestartResult] = []
    for seed in range(n_restarts):
        init_params = init_fn(seed)
        final_params, curve = optimizer_fn(coverage_fn, init_params)
        final_coverage = curve[-1] if curve else 0.0
        results.append(
            RestartResult(
                params=final_params,
                coverage=final_coverage,
                curve=curve,
                seed=seed,
            )
        )
    best = max(results, key=lambda r: r.coverage)
    return results, best


def coverage_stats(results: list[RestartResult]) -> dict[str, float]:
    """Summary statistics over a list of restart results.

    Returns a dict with keys ``mean``, ``std``, ``min``, ``max``,
    ``median``, and ``best`` (same as ``max``).
    """
    values = sorted(r.coverage for r in results)
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    std = variance ** 0.5
    median = values[n // 2] if n % 2 == 1 else (values[n // 2 - 1] + values[n // 2]) / 2.0
    return {
        "mean": mean,
        "std": std,
        "min": values[0],
        "max": values[-1],
        "median": median,
        "best": values[-1],
    }
