"""Continuous View Covariance Loss for source-trajectory refinement.

Implements the VCL functional of Lin et al. (TPAMI 2025) as an MLX function of
continuous source positions:

  L_VCL(s_1, …, s_k) = 1 − γ_Ω(s)^⊤ R_Ω(s)^{-1} γ_Ω(s)

with

  T_i(s_i) = Aᵗ_{s_i} H A_{s_i} x          (filtered back-projection basis),
  T_i^sub  = S T_i / ‖S T_i‖                (subsampled, unit-normalised),
  R_{ij}   = ⟨T_i^sub, T_j^sub⟩ + λ δ_{ij},
  γ_i      = ⟨T_i^sub, S x / ‖S x‖⟩.

The quadratic form has the exact custom VJP derived in the paper.  The matched
footprint projector does not expose a complete geometry VJP for its adjoint,
however.  The default ``"full_finite_difference"`` mode therefore
finite-differences the *complete normalised per-view basis* rather than silently
dropping the adjoint-geometry term.  Forward projection, detector geometry,
filtering, backprojection, voxel sampling, and row normalisation are all inside
the same central difference.

Each basis row depends only on its corresponding source.  All views can thus be
perturbed along one coordinate simultaneously, so the reverse pass needs six
additional basis evaluations (±ε in three coordinates), independent of the
number of views.  ``"legacy_autodiff"`` preserves the earlier partial
projector VJP only for controlled regressions and must not be used to claim the
gradient of the evaluated VCL score.

To keep the optimiser tractable we downsample the reference volume to a
modest grid (default ``128³``); the geometric coverage objective continues
to evaluate at full resolution because it does not touch the volume.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from diffct_mlx import cone_forward_footprint

from . import _torch_bridge
from .eval.geometry import geometry_from_sources
from .ramp_filter import ramp_filter_3d_mlx
from .score import _gamma_inv_quad_form
from .vcl_backprojection import (
    backproject_single_view,
    extract_sampled_backprojection_mx,
    sparse_backprojection_supported,
)

FULL_FINITE_DIFFERENCE_VJP = "full_finite_difference"
LEGACY_AUTODIFF_VJP = "legacy_autodiff"
_VALID_GEOMETRY_VJP_MODES = {
    FULL_FINITE_DIFFERENCE_VJP,
    LEGACY_AUTODIFF_VJP,
}


@dataclass
class VCLContext:
    """Cached static data for the differentiable VCL loss.

    Computed once per ``(volume, ROI)`` pair and reused across Adam steps.
    """

    volume: mx.array                # ``(D, H, W)`` downsampled reference
    volume_shape: tuple[int, int, int]
    sample_indices: np.ndarray      # flat indices of subsampled voxels
    x_sub_normalised: mx.array      # ``S x / ‖S x‖`` as 1D mx.array
    sid: float
    sdd: float
    det_shape: tuple[int, int]
    du: float
    dv: float
    voxel_spacing: float
    ridge: float = 1e-3             # Tikhonov stabiliser on R + λ I
    prefer_sparse_backprojection: bool = True
    geometry_vjp_mode: str = FULL_FINITE_DIFFERENCE_VJP
    geometry_fd_step: float = 0.5   # mm; complete basis central difference


def build_vcl_context(
    volume: mx.array,
    *,
    sid: float,
    sdd: float,
    det_shape: tuple[int, int] = (128, 128),
    du: float = 2.0,
    dv: float = 2.0,
    voxel_spacing: float = 1.0,
    target_shape: tuple[int, int, int] = (128, 128, 128),
    r1: float = 1e-3,
    seed: int = 0,
    ridge: float = 1e-3,
    roi_mask: np.ndarray | None = None,
    prefer_sparse_backprojection: bool = True,
    geometry_vjp_mode: str = FULL_FINITE_DIFFERENCE_VJP,
    geometry_fd_step: float = 0.5,
) -> VCLContext:
    """Downsample the volume and subsample voxels for the VCL term.
    
    Parameters
    ----------
    roi_mask
        Optional boolean mask. If provided, samples only from ROI voxels.
    prefer_sparse_backprojection
        If True, request sparse/sample-only backprojection when the installed
        ``diffct_mlx`` exposes a compatible API.
    geometry_vjp_mode
        ``"full_finite_difference"`` (default) differentiates the complete
        normalised basis and therefore includes both projector geometry paths.
        ``"legacy_autodiff"`` retains the historical partial VJP for regression
        comparisons only.
    geometry_fd_step
        Central-difference step in the source-coordinate unit (millimetres in
        the paper experiments).
    """
    from .eval.datasets.ornl_nozzle import _trilinear_to_shape

    if geometry_vjp_mode not in _VALID_GEOMETRY_VJP_MODES:
        valid = ", ".join(sorted(_VALID_GEOMETRY_VJP_MODES))
        raise ValueError(
            f"Unknown geometry_vjp_mode {geometry_vjp_mode!r}; expected one of: {valid}."
        )
    if not np.isfinite(geometry_fd_step) or geometry_fd_step <= 0.0:
        raise ValueError("geometry_fd_step must be a finite positive number.")

    # Never up-sample: pick min(source dim, target dim) per axis.
    target_shape = tuple(min(s, t) for s, t in zip(volume.shape, target_shape))
    if tuple(volume.shape) != target_shape:
        vol_np = np.asarray(volume, dtype=np.float32)
        vol_small = _trilinear_to_shape(vol_np, target_shape)
    else:
        vol_small = np.asarray(volume, dtype=np.float32)

    # Handle ROI masking if provided
    if roi_mask is not None:
        roi_mask_np = np.asarray(roi_mask, dtype=np.float32)
        if tuple(roi_mask_np.shape) != tuple(volume.shape):
            raise ValueError(
                "roi_mask must have the same shape as the input volume before downsampling."
            )
        if tuple(volume.shape) != target_shape:
            roi_mask_small = _trilinear_to_shape(roi_mask_np, target_shape) > 0.5
        else:
            roi_mask_small = roi_mask_np > 0.5
        roi_indices = np.flatnonzero(roi_mask_small.ravel())
        if roi_indices.size == 0:
            raise ValueError("roi_mask does not contain any selectable voxels after downsampling.")
        n_keep = max(1, int(r1 * len(roi_indices)))
        rng = np.random.default_rng(seed)
        indices = rng.choice(roi_indices, size=n_keep, replace=False).astype(np.int64)
    else:
        n_total = int(np.prod(target_shape))
        n_keep = max(1, int(r1 * n_total))
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_total, size=n_keep, replace=False).astype(np.int64)

    x_sub = vol_small.reshape(-1)[indices]
    norm = float(np.linalg.norm(x_sub))
    if norm <= 0.0:
        raise ValueError("Subsampled reference volume has zero norm.")
    x_sub_normalised = mx.array((x_sub / norm).astype(np.float32))

    return VCLContext(
        volume=mx.array(vol_small),
        volume_shape=target_shape,
        sample_indices=indices,
        x_sub_normalised=x_sub_normalised,
        sid=sid, sdd=sdd,
        det_shape=det_shape,
        du=du, dv=dv,
        voxel_spacing=voxel_spacing,
        ridge=ridge,
        prefer_sparse_backprojection=prefer_sparse_backprojection,
        geometry_vjp_mode=geometry_vjp_mode,
        geometry_fd_step=float(geometry_fd_step),
    )


def vcl_loss_continuous(
    sources: mx.array,
    ctx: VCLContext,
) -> mx.array:
    """Differentiable VCL loss ``1 − γ^⊤ R^{-1} γ``.

    Parameters
    ----------
    sources
        ``(k, 3)`` current source positions (Adam variable).
    ctx
        Precomputed :class:`VCLContext`.

    Returns
    -------
    Scalar mx.array.  Lower is better — minimise during Adam.
    """
    T_mat = view_basis_matrix(sources, ctx)
    k = int(sources.shape[0])

    R = T_mat @ mx.transpose(T_mat)
    R = R + ctx.ridge * mx.eye(k, dtype=R.dtype)
    gamma = T_mat @ ctx.x_sub_normalised

    info = _gamma_inv_quad_form(R, gamma)
    return 1.0 - info


def _rowwise_finite_difference_vjp(
    basis_fn: Callable[[mx.array], mx.array],
    sources: mx.array,
    cotangent: mx.array,
    step: float,
) -> mx.array:
    """Central-difference VJP for a row-separable view-basis function.

    ``basis_fn(sources)[i]`` must depend only on ``sources[i]``.  Perturbing
    every row along the same coordinate therefore yields all per-view partial
    derivatives in one paired evaluation.  Contracting row-wise with the
    incoming cotangent returns one 3-D gradient per source.
    """
    if step <= 0.0:
        raise ValueError("Finite-difference step must be positive.")

    sources_sg = mx.stop_gradient(sources)
    cotangent_sg = mx.stop_gradient(cotangent)
    axis_grads: list[mx.array] = []
    for axis in range(3):
        delta_np = np.zeros((1, 3), dtype=np.float32)
        delta_np[0, axis] = step
        delta = mx.array(delta_np, dtype=sources_sg.dtype)
        basis_plus = mx.stop_gradient(basis_fn(sources_sg + delta))
        basis_minus = mx.stop_gradient(basis_fn(sources_sg - delta))
        derivative = (basis_plus - basis_minus) / (2.0 * step)
        axis_grads.append(mx.sum(cotangent_sg * derivative, axis=1))
    return mx.stack(axis_grads, axis=-1)


def view_basis_matrix(sources: mx.array, ctx: VCLContext) -> mx.array:
    """Per-view filtered-back-projection bases ``T`` of shape ``(k, NS)``.

    Each row ``T_i`` is the subsampled, L2-normalised filtered back-projection
    of view ``i`` (the VCL view basis).  Shared by the VCL loss and the OED
    information objective so both see the identical differentiable bases.
    """
    if ctx.geometry_vjp_mode == LEGACY_AUTODIFF_VJP:
        return _view_basis_matrix_impl(sources, ctx)

    @mx.custom_function
    def _basis_with_complete_vjp(sources_in: mx.array) -> mx.array:
        return _view_basis_matrix_impl(sources_in, ctx)

    @_basis_with_complete_vjp.vjp
    def _basis_vjp(primals, cotangent, _output):
        # MLX passes a single-input custom function's primal as the array
        # itself rather than as a one-element tuple.
        sources_in = primals
        grad_sources = _rowwise_finite_difference_vjp(
            lambda perturbed: _view_basis_matrix_impl(perturbed, ctx),
            sources_in,
            cotangent,
            ctx.geometry_fd_step,
        )
        return grad_sources

    return _basis_with_complete_vjp(sources)


def _view_basis_matrix_impl(sources: mx.array, ctx: VCLContext) -> mx.array:
    """Evaluate the normalised matched-footprint basis without a custom VJP."""
    src_pos, det_c, det_u_v, det_v_v = geometry_from_sources(
        sources, sid=ctx.sid, sdd=ctx.sdd
    )
    _forward = (
        _torch_bridge.bridged_cone_forward_footprint
        if _torch_bridge.is_torch_backend() else cone_forward_footprint
    )
    sino = _forward(
        ctx.volume,
        src_pos, det_c, det_u_v, det_v_v,
        ctx.det_shape[0], ctx.det_shape[1],
        ctx.du, ctx.dv, ctx.voxel_spacing,
    )
    sino_filtered = ramp_filter_3d_mlx(sino)
    D, H, W = ctx.volume_shape
    k = int(sources.shape[0])

    # Torch backend + sparse backprojection: do the whole per-view loop in one
    # MLX<->Torch crossing instead of k of them (see bridged function's
    # docstring -- per-call bridging overhead, not compute, dominates at k~400
    # otherwise: ~10s/step measured vs ~0.1s/step batched).
    if ctx.prefer_sparse_backprojection and _torch_bridge.is_torch_backend() \
            and sparse_backprojection_supported():
        T_mat = _torch_bridge.bridged_per_view_sparse_backprojections(
            sino_filtered, src_pos, det_c, det_u_v, det_v_v,
            D, H, W, ctx.du, ctx.dv, ctx.voxel_spacing, ctx.sample_indices,
        )
        norms = mx.linalg.norm(T_mat, axis=1, keepdims=True)
        return T_mat / (norms + 1e-9)

    T_rows: list[mx.array] = []
    for i in range(k):
        single_sino = sino_filtered[i:i + 1]  # (1, n_u, n_v)
        T_i_back, _used_sparse = backproject_single_view(
            single_sino,
            src_pos[i:i + 1], det_c[i:i + 1],
            det_u_v[i:i + 1], det_v_v[i:i + 1],
            output_shape=(D, H, W),
            du=ctx.du, dv=ctx.dv, voxel_spacing=ctx.voxel_spacing,
            sample_indices=ctx.sample_indices,
            prefer_sparse=ctx.prefer_sparse_backprojection,
        )
        T_sub = extract_sampled_backprojection_mx(
            T_i_back, ctx.sample_indices, (D, H, W)
        )
        T_sub = T_sub / (mx.linalg.norm(T_sub) + 1e-9)
        T_rows.append(T_sub)
    return mx.stack(T_rows, axis=0)  # (k, NS)
