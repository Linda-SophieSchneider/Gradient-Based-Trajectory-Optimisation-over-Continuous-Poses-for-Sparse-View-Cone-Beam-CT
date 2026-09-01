"""MLX<->Torch tensor bridge for ``diffct_mlx``'s torch/CUDA backend.

``vcl_diff.py``/``vcl_backprojection.py`` call ``diffct_mlx.cone_forward_footprint``/
``cone_backward_footprint`` with plain MLX arrays, which is only valid when
``diffct_mlx.backend == "mlx"`` (real Apple Metal hardware). On the ``torch``
backend (this machine: no Metal, CUDA via numba/torch instead) those functions
expect PyTorch tensors. This module bridges the two, wrapped in
``mx.custom_function`` so the VJP recomputes the torch forward pass and pulls
gradients back out via ``torch.autograd.grad`` -- mirroring the recompute-in-vjp
pattern already used by ``score._gamma_inv_quad_form``/``oed._logdet_spd``.

Only used when :func:`is_torch_backend` is true; the native ``mlx`` backend
path is untouched and does not import this module.
"""
from __future__ import annotations

import os

import numpy as np
import mlx.core as mx

# Which torch forward projector the VCL term differentiates through:
#   "footprint" (default) -- matched voxel-footprint operator, FD-based
#                            geometry VJP (diffct >= 2.0.1)
#   "siddon"              -- diffct_mlx.cone_forward, ANALYTIC geometry
#                            gradients (diffct >= 2.1.0, ~4x faster VJP,
#                            validated against exact autograd cos > 0.998)
VCL_FORWARD_IMPL = os.environ.get("DIFFCOV_VCL_FORWARD", "footprint").strip().lower()


def is_torch_backend() -> bool:
    import diffct_mlx
    # diffct-mlx < 2 exposes only the native MLX implementation and has no
    # public ``backend`` attribute.  Treat that legacy API as MLX so current
    # figure/reconstruction code remains compatible with the project venv.
    return getattr(diffct_mlx, "backend", "mlx") == "torch"


def _mx_to_torch(arr, requires_grad: bool = False):
    import torch
    t = torch.from_numpy(np.array(arr, dtype=np.float32).copy())
    if torch.cuda.is_available():
        t = t.cuda()
    if requires_grad:
        t.requires_grad_(True)
    return t


def _torch_to_mx(t) -> mx.array:
    return mx.array(t.detach().cpu().numpy().astype(np.float32))


def to_backend(a):
    """MLX array -> backend-native array for a raw diffct_mlx footprint call.

    Identity on the native ``mlx`` backend (Apple Metal), so that path is
    untouched.  On the ``torch`` backend the raw footprint projectors
    (``cone_forward_footprint``/``cone_backward_footprint``) require torch
    tensors -- unlike the analytic/reconstruction helpers, which coerce inputs
    via ``as_array``/``xp`` and accept MLX arrays directly.  Use this only to
    feed those raw projectors; convert the result back with :func:`from_backend`
    so the surrounding MLX/NumPy code is unchanged.  Evaluation-only path (no
    autograd through it), so no ``mx.custom_function`` VJP is needed.
    """
    return _mx_to_torch(a) if is_torch_backend() else a


def from_backend(a) -> mx.array:
    """Inverse of :func:`to_backend` (identity on the ``mlx`` backend)."""
    return _torch_to_mx(a) if is_torch_backend() else a


def bridged_cone_forward_footprint(
    volume, src_pos, det_c, det_u_v, det_v_v,
    det_u, det_v, du, dv, voxel_spacing,
):
    """Torch-backed cone forward projector, MLX in/out, MLX-differentiable.

    Dispatches to the footprint or Siddon torch projector per
    :data:`VCL_FORWARD_IMPL` (env ``DIFFCOV_VCL_FORWARD``).
    """
    if VCL_FORWARD_IMPL == "siddon":
        from diffct_mlx import cone_forward as _torch_forward
    else:
        from diffct_mlx import cone_forward_footprint as _torch_forward

    def _run(volume_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx):
        vol_t = _mx_to_torch(volume_mx)
        src_t = _mx_to_torch(src_pos_mx, requires_grad=True)
        detc_t = _mx_to_torch(det_c_mx, requires_grad=True)
        du_t = _mx_to_torch(det_u_v_mx, requires_grad=True)
        dv_t = _mx_to_torch(det_v_v_mx, requires_grad=True)
        sino_t = _torch_forward(vol_t, src_t, detc_t, du_t, dv_t,
                                 det_u, det_v, du, dv, voxel_spacing)
        return sino_t, (src_t, detc_t, du_t, dv_t)

    @mx.custom_function
    def _fwd(volume_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx):
        sino_t, _ = _run(volume_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx)
        return _torch_to_mx(sino_t)

    @_fwd.vjp
    def _fwd_vjp(primals, cotangent, _out):
        import torch
        volume_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx = primals
        sino_t, (src_t, detc_t, du_t, dv_t) = _run(
            volume_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx
        )
        cot_t = _mx_to_torch(cotangent)
        grads = torch.autograd.grad(
            sino_t, [src_t, detc_t, du_t, dv_t], grad_outputs=cot_t, allow_unused=True
        )

        def g(grad, ref):
            return _torch_to_mx(grad) if grad is not None else mx.zeros_like(ref)

        return (
            mx.zeros_like(volume_mx),
            g(grads[0], src_pos_mx), g(grads[1], det_c_mx),
            g(grads[2], det_u_v_mx), g(grads[3], det_v_v_mx),
        )

    return _fwd(volume, src_pos, det_c, det_u_v, det_v_v)


def bridged_cone_backward_footprint(
    sino, src_pos, det_c, det_u_v, det_v_v,
    D, H, W, du, dv, voxel_spacing, indices=None,
):
    """Torch-backed ``cone_backward_footprint``, MLX in/out, MLX-differentiable.

    Note: this projector's torch/CUDA implementation only returns a gradient
    w.r.t. ``sino`` (data-only backprojector, per diffct-mlx's own changelog).
    This bridge therefore exposes no standalone geometry derivative. The
    default VCL path supplies the missing dependence by finite-differencing the
    complete matched basis in :mod:`differentiable_coverage.vcl_diff`.
    """
    from diffct_mlx import cone_backward_footprint as _torch_backward
    import torch

    def _run(sino_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx):
        sino_t = _mx_to_torch(sino_mx, requires_grad=True)
        src_t = _mx_to_torch(src_pos_mx, requires_grad=True)
        detc_t = _mx_to_torch(det_c_mx, requires_grad=True)
        du_t = _mx_to_torch(det_u_v_mx, requires_grad=True)
        dv_t = _mx_to_torch(det_v_v_mx, requires_grad=True)
        args = (sino_t, src_t, detc_t, du_t, dv_t, D, H, W, du, dv, voxel_spacing)
        if indices is not None:
            idx_t = torch.from_numpy(np.asarray(indices))
            if torch.cuda.is_available():
                idx_t = idx_t.cuda()
            vol_t = _torch_backward(*args, idx_t)
        else:
            vol_t = _torch_backward(*args)
        return vol_t, (sino_t, src_t, detc_t, du_t, dv_t)

    @mx.custom_function
    def _bwd(sino_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx):
        vol_t, _ = _run(sino_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx)
        return _torch_to_mx(vol_t)

    @_bwd.vjp
    def _bwd_vjp(primals, cotangent, _out):
        import torch
        sino_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx = primals
        vol_t, (sino_t, src_t, detc_t, du_t, dv_t) = _run(
            sino_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx
        )
        cot_t = _mx_to_torch(cotangent)
        grads = torch.autograd.grad(
            vol_t, [sino_t, src_t, detc_t, du_t, dv_t], grad_outputs=cot_t, allow_unused=True
        )

        def g(grad, ref):
            return _torch_to_mx(grad) if grad is not None else mx.zeros_like(ref)

        return tuple(g(gr, ref) for gr, ref in zip(grads, primals))

    return _bwd(sino, src_pos, det_c, det_u_v, det_v_v)


def bridged_per_view_sparse_backprojections(
    sino_filtered, src_pos, det_c, det_u_v, det_v_v,
    D, H, W, du, dv, voxel_spacing, sample_indices,
):
    """Batched per-view sparse backprojection: one MLX<->Torch crossing, not k.

    ``view_basis_matrix`` needs one independent sparse-sampled backprojection
    ``T_i`` per view (not a combined multi-view reconstruction), so it must call
    ``cone_backward_footprint`` once per view -- but doing that k times through
    :func:`bridged_cone_backward_footprint` means k separate MLX<->Torch
    round-trips (numpy copy, ``.cuda()`` transfer, ``mx.custom_function``
    dispatch) per Adam step. Measured on this machine: the raw sparse kernel
    call is ~0.3ms, yet k=400 of those *individually bridged* cost ~10s total --
    almost entirely per-call bridging overhead, not compute (confirmed by
    timing the plain torch loop alone: ~1ms/view with no python<->mx crossing
    in between). This function does the whole per-view loop *inside* one torch
    region -- k calls to the torch backprojector back-to-back, no bridging in
    between -- and crosses back to MLX exactly once, before and after.

    Returns
    -------
    mx.array of shape ``(k, len(sample_indices))``. The raw bridge is
    differentiable w.r.t. ``sino_filtered``; the default VCL basis-level VJP
    supplies the complete source-position derivative around this operation.
    """
    from diffct_mlx import cone_backward_footprint as _torch_backward
    import torch

    def _run(sino_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx):
        sino_t = _mx_to_torch(sino_mx, requires_grad=True)
        src_t = _mx_to_torch(src_pos_mx, requires_grad=True)
        detc_t = _mx_to_torch(det_c_mx, requires_grad=True)
        du_t = _mx_to_torch(det_u_v_mx, requires_grad=True)
        dv_t = _mx_to_torch(det_v_v_mx, requires_grad=True)
        idx_t = torch.from_numpy(np.asarray(sample_indices))
        if torch.cuda.is_available():
            idx_t = idx_t.cuda()
        k = sino_t.shape[0]
        rows = [
            _torch_backward(
                sino_t[i:i + 1], src_t[i:i + 1], detc_t[i:i + 1],
                du_t[i:i + 1], dv_t[i:i + 1],
                D, H, W, du, dv, voxel_spacing, idx_t,
            )
            for i in range(k)
        ]
        T_mat_t = torch.stack(rows, dim=0)  # (k, len(sample_indices))
        return T_mat_t, (sino_t, src_t, detc_t, du_t, dv_t)

    @mx.custom_function
    def _bwd_batch(sino_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx):
        T_mat_t, _ = _run(sino_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx)
        return _torch_to_mx(T_mat_t)

    @_bwd_batch.vjp
    def _bwd_batch_vjp(primals, cotangent, _out):
        import torch
        sino_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx = primals
        T_mat_t, (sino_t, src_t, detc_t, du_t, dv_t) = _run(
            sino_mx, src_pos_mx, det_c_mx, det_u_v_mx, det_v_v_mx
        )
        cot_t = _mx_to_torch(cotangent)
        grads = torch.autograd.grad(
            T_mat_t, [sino_t, src_t, detc_t, du_t, dv_t], grad_outputs=cot_t, allow_unused=True
        )

        def g(grad, ref):
            return _torch_to_mx(grad) if grad is not None else mx.zeros_like(ref)

        return tuple(g(gr, ref) for gr, ref in zip(grads, primals))

    return _bwd_batch(sino_filtered, src_pos, det_c, det_u_v, det_v_v)
