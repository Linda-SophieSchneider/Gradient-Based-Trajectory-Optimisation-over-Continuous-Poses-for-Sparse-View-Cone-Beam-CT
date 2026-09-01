"""Shared helpers for dense or sparse VCL backprojection.

These helpers centralise runtime detection of optional sparse-output support in
``diffct_mlx.cone_backward_footprint`` so the discrete and differentiable VCL
paths stay aligned.
"""

from __future__ import annotations

from functools import lru_cache
import inspect

import mlx.core as mx
import numpy as np

from diffct_mlx import cone_backward_footprint

from . import _torch_bridge


@lru_cache(maxsize=1)
def _backprojection_signature() -> inspect.Signature | None:
    try:
        return inspect.signature(cone_backward_footprint)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def sparse_backprojection_mode() -> str | None:
    """Return the supported sparse-output mode of ``cone_backward_footprint``.

    Returns
    -------
    ``"indices"``
        The projector accepts an explicit sample-index argument.
    ``"roi_mask"``
        The projector accepts an ROI/mask argument but no explicit indices.
    ``None``
        No known sparse-output parameter was detected.
    """
    signature = _backprojection_signature()
    if signature is None:
        return None
    params = signature.parameters
    names = {name.lower(): name for name in params}

    for candidate in ("indices", "sample_indices", "voxel_indices"):
        if candidate in names:
            return "indices"
    for candidate in ("roi_mask", "mask"):
        if candidate in names:
            return "roi_mask"
    return None


def sparse_backprojection_supported() -> bool:
    return sparse_backprojection_mode() is not None


def sparse_backprojection_kwargs(
    sample_indices: np.ndarray,
    output_shape: tuple[int, int, int],
) -> dict[str, object] | None:
    """Build sparse-output kwargs for the installed projector, if supported."""
    mode = sparse_backprojection_mode()
    if mode == "indices":
        signature = _backprojection_signature()
        if signature is None:
            return None
        params = signature.parameters
        for candidate in ("indices", "sample_indices", "voxel_indices"):
            if candidate in params:
                return {candidate: sample_indices}
    if mode == "roi_mask":
        mask = np.zeros(output_shape, dtype=np.uint8)
        mask.reshape(-1)[sample_indices] = 1
        signature = _backprojection_signature()
        if signature is None:
            return None
        params = signature.parameters
        for candidate in ("roi_mask", "mask"):
            if candidate in params:
                return {candidate: mask}
    return None


def backproject_single_view(
    sino_single: mx.array,
    src_single: mx.array,
    det_center_single: mx.array,
    det_u_single: mx.array,
    det_v_single: mx.array,
    *,
    output_shape: tuple[int, int, int],
    du: float,
    dv: float,
    voxel_spacing: float,
    sample_indices: np.ndarray | None = None,
    prefer_sparse: bool = True,
) -> tuple[mx.array, bool]:
    """Backproject one filtered view, optionally asking the projector for sparse output."""
    D, H, W = output_shape
    # NOTE: the shape/spacing arguments are passed *positionally*, not as
    # keywords.  ``cone_backward_footprint`` is an ``mx.custom_function``; when
    # it appears inside a differentiated graph MLX forwards any *keyword*
    # arguments on to the registered VJP, whose signature is
    # ``(primals, cotangent, _)`` and rejects them (``TypeError: ... unexpected
    # keyword argument 'D'``).  Captured positionally the same values live in
    # ``primals`` and the VJP reads them back, so the differentiable VCL / OED
    # path works.
    base_args = (
        sino_single,
        src_single,
        det_center_single,
        det_u_single,
        det_v_single,
        D,
        H,
        W,
        du,
        dv,
        voxel_spacing,
    )

    # On diffct_mlx's torch backend (no Apple Metal on this machine),
    # cone_backward_footprint expects torch tensors, not mx arrays -- bridge
    # through _torch_bridge instead. The raw backprojector is data-only and
    # therefore omits its geometry derivative. vcl_diff's default complete
    # basis VJP differentiates the whole A_s^T H A_s x construction by central
    # differences, so this low-level omission is not silently inherited by the
    # optimiser.
    _backward = (
        _torch_bridge.bridged_cone_backward_footprint
        if _torch_bridge.is_torch_backend() else cone_backward_footprint
    )

    if prefer_sparse and sample_indices is not None:
        sparse_kwargs = sparse_backprojection_kwargs(sample_indices, output_shape)
        if sparse_kwargs is not None:
            # The installed projector takes the sparse selector as its 12th
            # positional argument (``indices``); pass it positionally too so the
            # autograd VJP forwarding rule above keeps holding.
            sparse_values = tuple(sparse_kwargs.values())
            try:
                back = _backward(*base_args, *sparse_values)
                return back, True
            except TypeError:
                # Signature inspection can still be fooled by stale wrappers or
                # incompatible argument types; fall back to the dense path.
                pass

    back = _backward(*base_args)
    return back, False


def extract_sampled_backprojection_np(
    backprojection: mx.array,
    sample_indices: np.ndarray,
    output_shape: tuple[int, int, int],
) -> np.ndarray:
    """Convert a projector output to a sampled 1D float32 numpy vector."""
    back_np = np.asarray(backprojection, dtype=np.float32)
    n_samples = int(sample_indices.size)
    full_size = int(np.prod(output_shape))

    if back_np.size == n_samples:
        return back_np.reshape(-1)
    if back_np.size == full_size:
        return back_np.reshape(-1)[sample_indices]
    if back_np.ndim > 1 and back_np.shape[0] == 1 and back_np.reshape(-1).size == n_samples:
        return back_np.reshape(-1)
    raise ValueError(
        f"Unexpected backprojection output shape {tuple(back_np.shape)} for "
        f"{n_samples} samples and full shape {output_shape}."
    )


def extract_sampled_backprojection_mx(
    backprojection: mx.array,
    sample_indices: np.ndarray,
    output_shape: tuple[int, int, int],
) -> mx.array:
    """Convert a projector output to a sampled 1D MLX vector without breaking gradients."""
    n_samples = int(sample_indices.size)
    full_size = int(np.prod(output_shape))
    flat = backprojection.reshape(-1)

    if flat.shape[0] == n_samples:
        return flat
    if flat.shape[0] == full_size:
        sample_idx = mx.array(sample_indices.astype(np.int64))
        return flat[sample_idx]
    raise ValueError(
        f"Unexpected backprojection output size {flat.shape[0]} for "
        f"{n_samples} samples and full shape {output_shape}."
    )
