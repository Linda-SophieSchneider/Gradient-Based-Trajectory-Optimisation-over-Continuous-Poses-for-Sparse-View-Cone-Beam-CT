"""Joint reconstruction-trajectory optimisation loop (§2, §6).

Alternates between two steps:

1. **Reconstruction step** — lightweight SIRT that updates the attenuation-
   volume estimate using the *current* trajectory.
2. **Trajectory step**    — Adam gradient ascent on the full absorption-aware
   saturated-coverage objective using the *updated* volume.

This implements the self-consistent loop described in §2 of
``future_work_differentiable_coverage.md``.  The volume gradient is blocked
between the two phases (``mx.stop_gradient``) so each step is an independent
optimisation rather than a bi-level problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import diffct_mlx

from .absorption import AbsorptionConfig, absorption_gate, _detector_frame
from .optimize import adam_ascent
from .score import ScoreConfig, saturated_coverage


@dataclass
class JointLoopConfig:
    """Hyperparameters for the joint reconstruction-trajectory loop."""

    n_outer: int = 10
    n_traj_steps: int = 50
    n_recon_iters: int = 5
    lr_traj: float = 5.0
    sirt_step_size: float = 0.1


@dataclass
class JointLoopResult:
    """Outputs of :func:`joint_loop`."""

    sources: mx.array
    mu_volume: mx.array
    traj_history: list[float]
    outer_coverage: list[float]


def _sirt_update(
    mu: mx.array,
    sources: mx.array,
    measurements: mx.array,
    roi_center: mx.array,
    cfg: AbsorptionConfig,
    *,
    n_iters: int,
    step_size: float,
) -> mx.array:
    """``n_iters`` SIRT iterations updating the attenuation-volume estimate.

    Computes the forward projection for all views, forms the residual against
    *measurements*, backprojects, and applies a gradient-ascent update with
    non-negativity clamp.

    Parameters
    ----------
    mu : ``(D, H, W)``
    sources : ``(k, 3)``
    measurements : ``(k, det_u, det_v)``
    roi_center : ``(3,)``
    cfg : AbsorptionConfig
    n_iters :
        Number of SIRT iterations per outer step.
    step_size :
        SIRT relaxation parameter (learning rate).

    Returns
    -------
    mu_updated : ``(D, H, W)``
    """
    det_center, det_u_vec, det_v_vec = _detector_frame(sources, roi_center, cfg.sdd)
    D, H, W = mu.shape

    for _ in range(n_iters):
        proj = diffct_mlx.cone_forward(
            mu, sources, det_center, det_u_vec, det_v_vec,
            cfg.det_u, cfg.det_v, cfg.du, cfg.dv, cfg.voxel_spacing,
        )
        mx.eval(proj)
        residual = measurements - proj
        delta = diffct_mlx.cone_backward(
            residual, sources, det_center, det_u_vec, det_v_vec,
            D, H, W, cfg.du, cfg.dv, cfg.voxel_spacing,
        )
        mx.eval(delta)
        mu = mx.maximum(mu + step_size * delta, 0.0)
        mx.eval(mu)

    return mu


def joint_loop(
    sources: mx.array,
    measurements: mx.array,
    mu_init: mx.array,
    roi_center: mx.array,
    radon_normals: mx.array,
    cfg_score: ScoreConfig,
    cfg_abs: AbsorptionConfig,
    loop_cfg: JointLoopConfig | None = None,
    *,
    callback: Callable[[int, mx.array, mx.array, float], None] | None = None,
) -> JointLoopResult:
    """Jointly optimise the acquisition trajectory and the attenuation-volume estimate.

    Parameters
    ----------
    sources : ``(k, 3)``
        Initial source positions.
    measurements : ``(k, det_u, det_v)``
        Measured projections.  For a simulation study, generate these via
        ``diffct_mlx.cone_forward`` on the ground-truth volume.
    mu_init : ``(D, H, W)``
        Initial volume estimate (e.g. zeros, FBP, or ground-truth).
    roi_center : ``(3,)``
        ROI centre used for ray directions and detector placement.
    radon_normals : ``(z, 3)``
        Sampled Radon plane normals.
    cfg_score : ScoreConfig
        Smoothing parameters for the coverage objective.
    cfg_abs : AbsorptionConfig
        Geometry and smoothing temperatures for the absorption gate.
    loop_cfg : JointLoopConfig or None
        Outer-loop hyperparameters (defaults used when ``None``).
    callback : callable(outer_iter, sources, mu, coverage) or None
        Called at the end of each outer iteration with the current state.

    Returns
    -------
    JointLoopResult
    """
    if loop_cfg is None:
        loop_cfg = JointLoopConfig()

    mu = mx.array(mu_init, dtype=mx.float32)
    mx.eval(mu)

    traj_history: list[float] = []
    outer_coverage: list[float] = []

    for outer in range(loop_cfg.n_outer):
        # --- Reconstruction step -------------------------------------------
        mu = _sirt_update(
            mu, sources, measurements, roi_center, cfg_abs,
            n_iters=loop_cfg.n_recon_iters,
            step_size=loop_cfg.sirt_step_size,
        )

        # Detach the volume from the trajectory optimisation so that
        # the two phases remain independent (no bi-level coupling).
        mu_snap = mx.stop_gradient(mu)

        # --- Trajectory step -----------------------------------------------
        def coverage_fn(srcs: mx.array, step: int) -> mx.array:
            nu = absorption_gate(srcs, roi_center, mu_snap, cfg_abs)
            return saturated_coverage(srcs, roi_center, radon_normals, nu, cfg_score)

        sources, step_history = adam_ascent(
            coverage_fn,
            sources,
            lr=loop_cfg.lr_traj,
            n_steps=loop_cfg.n_traj_steps,
        )
        traj_history.extend(step_history)

        cov = step_history[-1] if step_history else 0.0
        outer_coverage.append(cov)

        if callback is not None:
            callback(outer, sources, mu, cov)

    return JointLoopResult(
        sources=sources,
        mu_volume=mu,
        traj_history=traj_history,
        outer_coverage=outer_coverage,
    )
