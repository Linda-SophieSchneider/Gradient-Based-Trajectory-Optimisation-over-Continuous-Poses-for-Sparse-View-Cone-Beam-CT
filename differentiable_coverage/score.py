"""Differentiable soft near-orthogonality coverage score.

Implements §1 of `future_work_differentiable_coverage.md`:
the saturated coverage objective made differentiable in the source
positions via smooth replacements for the hinge score and the saturation cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class ScoreConfig:
    """Smoothing parameters for the differentiable coverage score (§1)."""

    tau: float
    sigma: float | None = None
    epsilon_at_tau: float = 1e-2

    def gaussian_sigma(self) -> float:
        if self.sigma is not None:
            return float(self.sigma)
        return float(self.tau) / math.sqrt(2.0 * math.log(1.0 / self.epsilon_at_tau))


def sample_unit_sphere(
    n: int, *, seed: int | None = None, dtype=mx.float32,
) -> mx.array:
    """Fibonacci-lattice samples of Radon plane normals on the unit sphere.

    Parameters
    ----------
    n :
        Number of points.
    seed :
        If ``None`` (default), returns the deterministic Fibonacci
        lattice; bit-for-bit reproducible across calls.  If an
        integer, a uniformly random ``SO(3)`` rotation derived from
        the seed is applied to the lattice, so different seeds
        produce different (but still quasi-uniform) candidate sets.
        This is the physically meaningful randomisation for
        multi-seed reproducibility studies of view selection.

    The default ``seed=None`` keeps the historical behaviour, so all
    callers that do not opt in to seeded sampling see the original
    lattice and the existing test results remain reproducible.
    """
    import numpy as _np

    phi = math.pi * (3.0 - math.sqrt(5.0))
    k = mx.arange(n, dtype=dtype)
    y = 1.0 - 2.0 * k / max(n - 1, 1)
    r = mx.sqrt(mx.maximum(0.0, 1.0 - y * y))
    theta = phi * k
    pts = mx.stack([mx.cos(theta) * r, y, mx.sin(theta) * r], axis=1)
    if seed is None:
        return pts
    # Uniform random unit quaternion → SO(3) rotation matrix.
    rng = _np.random.default_rng(int(seed))
    q = rng.standard_normal(4).astype(_np.float64)
    q /= _np.linalg.norm(q)
    qw, qx, qy, qz = q
    R = _np.array(
        [[1 - 2 * (qy * qy + qz * qz),  2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
         [2 * (qx * qy + qz * qw),      1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
         [2 * (qx * qz - qy * qw),      2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)]],
        dtype=_np.float32,
    )
    return pts @ mx.array(R.T)


def ray_directions(sources: mx.array, roi_center: mx.array) -> tuple[mx.array, mx.array]:
    """Unit ray vectors from each source to the ROI center (§1 Steps 1-2)."""
    r = sources - roi_center
    rho = mx.linalg.norm(r, axis=-1, keepdims=True)
    return r / mx.maximum(rho, 1e-9), mx.squeeze(rho, axis=-1)


def orthogonality_kernel(
    ray_dirs: mx.array, radon_normals: mx.array, sigma: float
) -> mx.array:
    """Gaussian smooth replacement of the hinge score (§1 Step 5).

    Returns
    -------
    psi : mx.array of shape ``(k, z)``
        ``exp(-(mu_j . d_i)^2 / (2 sigma^2))``, smooth and ``C^infty``.
    """
    g = ray_dirs @ radon_normals.T
    return mx.exp(-(g * g) / (2.0 * sigma * sigma))


def accumulated_coverage(psi: mx.array, nu: mx.array) -> mx.array:
    """Per-direction smoothed saturated coverage ``1 - exp(-Sigma_j)`` (§1 Step 6)."""
    weighted = mx.expand_dims(nu, axis=-1) * psi
    sigma_j = mx.sum(weighted, axis=0)
    return 1.0 - mx.exp(-sigma_j)


@mx.custom_function
def _gamma_inv_quad_form(R_F: mx.array, gamma_F: mx.array) -> mx.array:
    """Compute ``γ^T R^{-1} γ`` with an explicit VJP.

    mlx's ``linalg.solve`` does not yet have a VJP defined, so we wrap the
    Cholesky/LU solve in a custom function and supply the closed-form
    gradient:

      ∂(γ^T R^{-1} γ) / ∂R = -u u^T,  ∂(γ^T R^{-1} γ) / ∂γ = 2 u,
      with u = R^{-1} γ.

    Solve runs on the CPU stream because the GPU solver isn't enabled yet.
    """
    with mx.stream(mx.cpu):
        u = mx.linalg.solve(R_F, gamma_F[:, None])[:, 0]
    return mx.sum(gamma_F * u)


@_gamma_inv_quad_form.vjp
def _gamma_inv_quad_form_vjp(primals, cotangent, _out):
    R_F, gamma_F = primals
    with mx.stream(mx.cpu):
        u = mx.linalg.solve(R_F, gamma_F[:, None])[:, 0]
    # cotangent is a scalar
    grad_R = -cotangent * (u[:, None] * u[None, :])
    grad_g = cotangent * 2.0 * u
    return grad_R, grad_g


def coverage_covariance_information(
    sources: mx.array,
    roi_center: mx.array,
    radon_normals: mx.array,
    nu: mx.array,
    cfg: ScoreConfig,
    *,
    lambda_R: float = 1e-3,
    direction_weights: mx.array | None = None,
) -> mx.array:
    """VCL-inspired information score $I_{\\mathrm{cov}}$ from §VII of the paper.

    Each source contributes a coverage *fingerprint*
    $f_i = \\nu_i\\,[\\psi(\\mu_1^\\top d_i), \\dots, \\psi(\\mu_z^\\top d_i)]$ to a row
    matrix $F \\in \\mathbb{R}^{k \\times z}$.  The intra-view covariance is
    $R_F = F F^\\top + \\lambda_R I$ and the informativeness vector is
    $\\gamma_F = F w$ for a direction weighting $w$ (uniform by default).
    The information score

    $$I_{\\mathrm{cov}} = \\gamma_F^\\top R_F^{-1} \\gamma_F$$

    rewards sources that cover directions where coverage is most needed
    while penalising sources whose fingerprints are linearly predictable from
    the others (through $R_F^{-1}$).

    ``lambda_R`` acts both as a Tikhonov regulariser and as a soft prior that
    ``R_F \\approx \\lambda_R I`` — preventing the inverse from blowing up
    when two sources are near-identical.
    """
    sigma = cfg.gaussian_sigma()
    d, _ = ray_directions(sources, roi_center)
    psi = orthogonality_kernel(d, radon_normals, sigma)  # (k, z)
    F = mx.expand_dims(nu, axis=-1) * psi  # (k, z)

    if direction_weights is None:
        gamma_F = mx.sum(F, axis=-1)  # uniform w, shape (k,)
    else:
        gamma_F = F @ direction_weights

    k = sources.shape[0]
    R_F = F @ mx.transpose(F) + lambda_R * mx.eye(k, dtype=F.dtype)
    return _gamma_inv_quad_form(R_F, gamma_F)


def greedy_source_init(
    candidates: mx.array,
    roi_center: mx.array,
    radon_normals: mx.array,
    k: int,
    cfg: ScoreConfig,
    *,
    roi_points: mx.array | None = None,
    roi_weights: mx.array | None = None,
) -> mx.array:
    """Greedy sequential selection of ``k`` sources from *candidates* (§3.1 warm-start).

    Selects sources one by one, each time picking the candidate that maximises
    the marginal gain in saturated coverage given the already-selected set.
    Provides a warm start that is typically much better than a uniform or random
    initialisation for subsequent gradient refinement.

    Parameters
    ----------
    candidates : ``(n, 3)``
        Pool of candidate source positions to select from.
    roi_center : ``(3,)``
        ROI centre.
    radon_normals : ``(z, 3)``
        Sampled Radon plane normals.
    k : int
        Number of sources to select.
    cfg : ScoreConfig
        Smoothing parameters.

    Returns
    -------
    selected : ``(k, 3)``
        Greedily selected source positions in the order they were chosen.
    """
    sigma = cfg.gaussian_sigma()
    n = candidates.shape[0]

    if roi_points is None:
        # Precompute per-candidate psi rows (no grad needed here)
        d_all, _ = ray_directions(candidates, roi_center)  # (n, 3)
        psi_all = orthogonality_kernel(d_all, radon_normals, sigma)  # (n, z)
        mx.eval(psi_all)
        psi_np = psi_all.tolist()

        selected_indices: list[int] = []
        sigma_j = mx.zeros(radon_normals.shape[0])

        for _ in range(k):
            best_idx = -1
            best_gain = -1.0
            for i in range(n):
                if i in selected_indices:
                    continue
                psi_i = mx.array(psi_np[i])
                new_sigma_j = sigma_j + psi_i
                cov = float(mx.mean(1.0 - mx.exp(-new_sigma_j)))
                gain = cov - float(mx.mean(1.0 - mx.exp(-sigma_j)))
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            selected_indices.append(best_idx)
            psi_sel = mx.array(psi_np[best_idx])
            sigma_j = sigma_j + psi_sel
            mx.eval(sigma_j)

        return mx.stack([candidates[i] for i in selected_indices])

    cand_np = np.asarray(candidates, dtype=np.float32)
    normals_np = np.asarray(radon_normals, dtype=np.float32)
    points_np = np.asarray(roi_points, dtype=np.float32)
    if roi_weights is None:
        weights_np = np.ones(points_np.shape[0], dtype=np.float32) / max(points_np.shape[0], 1)
    else:
        weights_np = np.asarray(roi_weights, dtype=np.float32)
        weights_np = weights_np / max(float(weights_np.sum()), 1e-9)

    m = points_np.shape[0]
    z = normals_np.shape[0]
    psi_all = np.empty((m, n, z), dtype=np.float32)
    for j, point in enumerate(points_np):
        r = cand_np - point[None, :]
        rho = np.linalg.norm(r, axis=-1, keepdims=True)
        d = r / np.maximum(rho, 1e-9)
        g = d @ normals_np.T
        psi_all[j] = np.exp(-(g * g) / (2.0 * sigma * sigma)).astype(np.float32)

    selected_indices: list[int] = []
    sigma_j = np.zeros((m, z), dtype=np.float32)
    cov_prev = np.mean(1.0 - np.exp(-sigma_j), axis=1)

    for _ in range(k):
        best_idx = -1
        best_gain = -1.0
        for i in range(n):
            if i in selected_indices:
                continue
            new_sigma_j = sigma_j + psi_all[:, i, :]
            cov_point = np.mean(1.0 - np.exp(-new_sigma_j), axis=1)
            gain = float(np.sum(weights_np * (cov_point - cov_prev)))
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        selected_indices.append(best_idx)
        sigma_j = sigma_j + psi_all[:, best_idx, :]
        cov_prev = np.mean(1.0 - np.exp(-sigma_j), axis=1)

    return mx.stack([candidates[i] for i in selected_indices])


def saturated_coverage(
    sources: mx.array,
    roi_center: mx.array,
    radon_normals: mx.array,
    nu: mx.array,
    cfg: ScoreConfig,
    *,
    roi_points: mx.array | None = None,
    roi_weights: mx.array | None = None,
) -> mx.array:
    """Full differentiable saturated coverage objective ``C_sat`` (§1).

    Parameters
    ----------
    sources : ``(k, 3)``
        Continuous source positions.
    roi_center : ``(3,)``
        ROI centre.  Used as the single evaluation point unless *roi_points*
        is provided.
    radon_normals : ``(z, 3)``
        Sampled Radon plane normals (unit vectors).
    nu : ``(k,)``
        Per-source validity weight in ``[0, 1]`` (from
        `absorption.absorption_gate`).
    roi_points : ``(m, 3)`` or ``None``
        Optional set of ROI sample points for per-voxel coverage aggregation.
        When provided, the coverage is computed for each point and averaged
        (weighted by *roi_weights* if given).  Enables richer coverage
        metrics beyond the single-centre approximation.
    roi_weights : ``(m,)`` or ``None``
        Non-negative weights for each ROI point.  Normalised internally.
        Uniform weights are used when ``None``.
    """
    if roi_points is None:
        d, _ = ray_directions(sources, roi_center)
        psi = orthogonality_kernel(d, radon_normals, cfg.gaussian_sigma())
        return mx.mean(accumulated_coverage(psi, nu))

    # Per-voxel aggregation: average coverage over all roi_points
    sigma = cfg.gaussian_sigma()
    if roi_weights is None:
        w = mx.ones(roi_points.shape[0], dtype=roi_points.dtype) / roi_points.shape[0]
    else:
        w = roi_weights / mx.maximum(mx.sum(roi_weights), 1e-9)

    # coverage_m: shape (m,) — mean coverage for each ROI point
    coverages = []
    for i in range(roi_points.shape[0]):
        d_i, _ = ray_directions(sources, roi_points[i])
        psi_i = orthogonality_kernel(d_i, radon_normals, sigma)
        coverages.append(mx.mean(accumulated_coverage(psi_i, nu)))
    coverage_m = mx.stack(coverages)  # (m,)
    return mx.sum(w * coverage_m)
