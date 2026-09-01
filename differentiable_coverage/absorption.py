"""Differentiable absorption-aware signal-quality gate.

Implements §2 of `future_work_differentiable_coverage.md`: replaces the binary
detectability mask of the MILP paper with a smooth gate ``nu_i in [0, 1]``
computed via DiffCT's differentiable forward projector through a known
attenuation volume ``mu(x)``.

Gradient status
---------------
The public DiffCT cone-beam VJP currently differentiates w.r.t. ``src_pos``
but not w.r.t. ``det_center / det_u_vec / det_v_vec``.  Because those detector
arrays are derived from ``sources`` here, we wrap the projector in a local
source-geometry custom VJP that finite-differences the *full* map
``sources -> detector frame -> projections``.  This restores the missing source
dependence without requiring a patch to DiffCT itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import diffct_mlx


@dataclass(frozen=True)
class AbsorptionConfig:
    """Geometry plus smoothing temperatures for the gate (§2)."""

    alpha: float
    eta: float
    sid: float
    sdd: float
    det_u: int
    det_v: int
    du: float = 1.0
    dv: float = 1.0
    voxel_spacing: float = 1.0
    roi_radius: float = 10.0
    beta_pixel: float = 4.0
    beta_frac: float = 20.0
    footprint_floor: float = 1e-3


import os as _os
# Step size for the finite-difference VJP through cone_forward (mm).
# Configurable via the DIFFCT_FD_EPS environment variable so the R6
# review-response sensitivity sweep can probe alternative values
# without rebuilding the package.
_FD_EPS_SOURCE_GEOMETRY: float = float(_os.environ.get("DIFFCT_FD_EPS", "0.5"))


def _cross3(a: mx.array, b: mx.array) -> mx.array:
    return mx.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def _detector_frame(
    sources: mx.array, roi_center: mx.array, sdd: float
) -> tuple[mx.array, mx.array, mx.array]:
    """Detector placed opposite the source through the ROI, world-z as up reference."""
    direction = roi_center - sources
    ray = direction / mx.maximum(mx.linalg.norm(direction, axis=-1, keepdims=True), 1e-9)
    det_center = sources + sdd * ray
    up = mx.broadcast_to(mx.array([0.0, 0.0, 1.0], dtype=sources.dtype), ray.shape)
    det_u_vec = _cross3(up, ray)
    det_u_vec = det_u_vec / mx.maximum(
        mx.linalg.norm(det_u_vec, axis=-1, keepdims=True), 1e-9
    )
    det_v_vec = _cross3(ray, det_u_vec)
    return det_center, det_u_vec, det_v_vec


def _detector_pixel_grid(cfg: AbsorptionConfig, *, dtype=mx.float32) -> tuple[mx.array, mx.array]:
    """Detector-plane coordinates in physical units."""
    u = (mx.arange(cfg.det_u, dtype=dtype) - 0.5 * (cfg.det_u - 1)) * cfg.du
    v = (mx.arange(cfg.det_v, dtype=dtype) - 0.5 * (cfg.det_v - 1)) * cfg.dv
    return mx.meshgrid(u, v, indexing="ij")


def _roi_boundary_points(roi_center: mx.array, roi_radius: float) -> mx.array:
    """Sample points on the ROI boundary for footprint and containment tests."""
    dtype = roi_center.dtype
    axes = mx.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=dtype,
    )
    corner_scale = 1.0 / math.sqrt(3.0)
    corners = mx.array(
        [
            [corner_scale, corner_scale, corner_scale],
            [corner_scale, corner_scale, -corner_scale],
            [corner_scale, -corner_scale, corner_scale],
            [corner_scale, -corner_scale, -corner_scale],
            [-corner_scale, corner_scale, corner_scale],
            [-corner_scale, corner_scale, -corner_scale],
            [-corner_scale, -corner_scale, corner_scale],
            [-corner_scale, -corner_scale, -corner_scale],
        ],
        dtype=dtype,
    )
    offsets = mx.concatenate([axes, corners], axis=0)
    return roi_center[None, :] + roi_radius * offsets


def _project_points_to_detector(
    sources: mx.array,
    roi_center: mx.array,
    points: mx.array,
    cfg: AbsorptionConfig,
) -> tuple[mx.array, mx.array]:
    """Project 3-D points into detector-plane coordinates for each source view."""
    det_center, det_u_vec, det_v_vec = _detector_frame(sources, roi_center, cfg.sdd)
    normal = det_center - sources
    normal = normal / mx.maximum(mx.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)

    rays = points[None, :, :] - sources[:, None, :]
    denom = mx.sum(rays * normal[:, None, :], axis=-1)
    t = cfg.sdd / mx.maximum(denom, 1e-9)
    hits = sources[:, None, :] + t[..., None] * rays
    rel = hits - det_center[:, None, :]

    p_u = mx.sum(rel * det_u_vec[:, None, :], axis=-1)
    p_v = mx.sum(rel * det_v_vec[:, None, :], axis=-1)
    return p_u, p_v


def _cone_forward_with_source_geometry_impl(
    mu_volume: mx.array,
    sources: mx.array,
    roi_center: mx.array,
    sdd: float,
    det_u: int,
    det_v: int,
    du: float,
    dv: float,
    voxel_spacing: float,
) -> mx.array:
    """Forward projection with detector geometry derived from the sources."""
    det_center, det_u_vec, det_v_vec = _detector_frame(sources, roi_center, sdd)
    return diffct_mlx.cone_forward(
        mu_volume,
        sources,
        det_center,
        det_u_vec,
        det_v_vec,
        det_u,
        det_v,
        du,
        dv,
        voxel_spacing,
    )


@mx.custom_function
def _cone_forward_with_source_geometry(
    mu_volume: mx.array,
    sources: mx.array,
    roi_center: mx.array,
    sdd: float,
    det_u: int,
    det_v: int,
    du: float = 1.0,
    dv: float = 1.0,
    voxel_spacing: float = 1.0,
) -> mx.array:
    """Forward projector with a total VJP w.r.t. source positions."""
    return _cone_forward_with_source_geometry_impl(
        mu_volume, sources, roi_center, sdd, det_u, det_v, du, dv, voxel_spacing
    )


@_cone_forward_with_source_geometry.vjp
def _cone_forward_with_source_geometry_vjp(primals, cotangent, _):
    mu_volume, sources, roi_center = primals[:3]
    sdd = primals[3]
    det_u = primals[4]
    det_v = primals[5]
    du = primals[6] if len(primals) > 6 else 1.0
    dv = primals[7] if len(primals) > 7 else 1.0
    voxel_spacing = primals[8] if len(primals) > 8 else 1.0

    mu_volume = mx.array(mu_volume, dtype=mx.float32)
    sources = mx.array(sources, dtype=mx.float32)
    roi_center = mx.array(roi_center, dtype=mx.float32)
    cotangent = mx.array(cotangent, dtype=mx.float32)

    grads = []
    for axis in range(3):
        delta_values = [[0.0, 0.0, 0.0]]
        delta_values[0][axis] = _FD_EPS_SOURCE_GEOMETRY
        delta = mx.array(delta_values, dtype=sources.dtype)

        f_plus = _cone_forward_with_source_geometry_impl(
            mu_volume,
            sources + delta,
            roi_center,
            sdd,
            det_u,
            det_v,
            du,
            dv,
            voxel_spacing,
        )
        f_minus = _cone_forward_with_source_geometry_impl(
            mu_volume,
            sources - delta,
            roi_center,
            sdd,
            det_u,
            det_v,
            du,
            dv,
            voxel_spacing,
        )
        df_daxis = (f_plus - f_minus) / (2.0 * _FD_EPS_SOURCE_GEOMETRY)
        grads.append(mx.sum(cotangent * df_daxis, axis=(1, 2)))

    grad_sources = mx.stack(grads, axis=-1)
    return (
        mx.zeros_like(mu_volume),
        grad_sources,
        mx.zeros_like(roi_center),
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _soft_footprint(
    cfg: AbsorptionConfig,
    sources: mx.array | None = None,
    roi_center: mx.array | None = None,
) -> mx.array:
    """Gaussian footprint matched to the projected ROI extent (§2 Step B)."""
    uu, vv = _detector_pixel_grid(cfg)

    if sources is None or roi_center is None:
        magnification = cfg.sdd / cfg.sid
        sigma_u = max(cfg.roi_radius * magnification, cfg.du)
        sigma_v = max(cfg.roi_radius * magnification, cfg.dv)
        w = mx.exp(-0.5 * (uu * uu / (sigma_u * sigma_u) + vv * vv / (sigma_v * sigma_v)))
        return mx.maximum(w, cfg.footprint_floor)

    roi_center = mx.array(roi_center, dtype=sources.dtype)
    boundary = _roi_boundary_points(roi_center, cfg.roi_radius)
    boundary_u, boundary_v = _project_points_to_detector(sources, roi_center, boundary, cfg)
    center_u, center_v = _project_points_to_detector(
        sources, roi_center, roi_center[None, :], cfg
    )
    center_u = mx.squeeze(center_u, axis=1)
    center_v = mx.squeeze(center_v, axis=1)

    sigma_u = mx.maximum(mx.max(mx.abs(boundary_u - center_u[:, None]), axis=1), cfg.du)
    sigma_v = mx.maximum(mx.max(mx.abs(boundary_v - center_v[:, None]), axis=1), cfg.dv)

    du_rel = uu[None, :, :] - center_u[:, None, None]
    dv_rel = vv[None, :, :] - center_v[:, None, None]
    w = mx.exp(
        -0.5 * (
            du_rel * du_rel / (sigma_u[:, None, None] * sigma_u[:, None, None])
            + dv_rel * dv_rel / (sigma_v[:, None, None] * sigma_v[:, None, None])
        )
    )
    return mx.maximum(w, cfg.footprint_floor)


def _detector_containment_gate(
    sources: mx.array,
    roi_center: mx.array,
    cfg: AbsorptionConfig,
    *,
    beta_contain: float = 10.0,
) -> mx.array:
    """Soft gate that checks projected ROI boundary points against the detector.

    Uses the worst signed over-run across sampled ROI boundary points:

        over_u = |p_u| / half_u - 1
        over_v = |p_v| / half_v - 1
        over_run = max(over_u, over_v)

    Gate:

        sigma(-beta_contain * max_point over_run)

    This preserves the boundary-sample logic from the note without the
    multiplicative shrinkage of a product over many interior points.
    """
    boundary = _roi_boundary_points(mx.array(roi_center, dtype=sources.dtype), cfg.roi_radius)
    p_u, p_v = _project_points_to_detector(sources, roi_center, boundary, cfg)
    half_u = 0.5 * (cfg.det_u - 1) * cfg.du
    half_v = 0.5 * (cfg.det_v - 1) * cfg.dv

    over_u = mx.abs(p_u) / max(half_u, 1e-9) - 1.0
    over_v = mx.abs(p_v) / max(half_v, 1e-9) - 1.0
    over_run = mx.maximum(over_u, over_v)
    worst_over_run = mx.max(over_run, axis=1)
    return mx.sigmoid(-beta_contain * worst_over_run)


def _make_fd_delta(k: int, axis: int, eps: float, dtype) -> mx.array:
    """(k, 3) array with ``eps`` on column ``axis``, zero elsewhere."""
    row = [0.0, 0.0, 0.0]
    row[axis] = eps
    return mx.broadcast_to(mx.array([row], dtype=dtype), (k, 3))


@mx.custom_function
def _absorption_gate_tangential_custom(
    sources,     # (k, 3)
    roi_center,  # (3,)
    mu_sg,       # (D, H, W) — already stop_gradiented
    alpha, eta, sdd,
    det_u, det_v, du, dv, voxel_spacing,
    roi_radius, beta_pixel, beta_frac, footprint_floor,
    beta_contain,
):
    """Forward pass: identical to absorption_gate."""
    cfg = AbsorptionConfig(
        alpha=float(alpha), eta=float(eta), sid=float(sdd), sdd=float(sdd),
        det_u=int(det_u), det_v=int(det_v),
        du=float(du), dv=float(dv), voxel_spacing=float(voxel_spacing),
        roi_radius=float(roi_radius), beta_pixel=float(beta_pixel),
        beta_frac=float(beta_frac), footprint_floor=float(footprint_floor),
    )
    return absorption_gate(sources, roi_center, mu_sg, cfg, beta_contain=float(beta_contain))


@_absorption_gate_tangential_custom.vjp
def _absorption_gate_tangential_vjp(primals, cotangent, _):
    sources, roi_center, mu_sg = primals[0], primals[1], primals[2]
    alpha, eta, sdd = primals[3], primals[4], primals[5]
    det_u, det_v = primals[6], primals[7]
    du, dv, voxel_spacing = primals[8], primals[9], primals[10]
    roi_radius, beta_pixel, beta_frac, footprint_floor = (
        primals[11], primals[12], primals[13], primals[14]
    )
    beta_contain = primals[15]

    cfg = AbsorptionConfig(
        alpha=float(alpha), eta=float(eta), sid=float(sdd), sdd=float(sdd),
        det_u=int(det_u), det_v=int(det_v),
        du=float(du), dv=float(dv), voxel_spacing=float(voxel_spacing),
        roi_radius=float(roi_radius), beta_pixel=float(beta_pixel),
        beta_frac=float(beta_frac), footprint_floor=float(footprint_floor),
    )

    k = sources.shape[0]
    eps = _FD_EPS_SOURCE_GEOMETRY

    # FD gradient: d(nu_i)/d(s_i_axis) for all sources simultaneously.
    # Since nu_i depends only on sources[i], simultaneous perturbation is exact.
    grad_axes = []
    for axis in range(3):
        delta = _make_fd_delta(k, axis, eps, sources.dtype)
        nu_p = absorption_gate(sources + delta, roi_center, mu_sg, cfg,
                               beta_contain=float(beta_contain))
        nu_m = absorption_gate(sources - delta, roi_center, mu_sg, cfg,
                               beta_contain=float(beta_contain))
        grad_axes.append(cotangent * (nu_p - nu_m) / (2.0 * eps))  # (k,)
    grad_sources = mx.stack(grad_axes, axis=-1)  # (k, 3)

    # Tangential projection: remove radial (viewing-direction) component.
    r = sources - roi_center  # (k, 3) - (3,) broadcasts to (k, 3)
    rho = mx.linalg.norm(r, axis=-1, keepdims=True)
    d = r / mx.maximum(rho, 1e-9)
    dot = mx.sum(d * grad_sources, axis=-1, keepdims=True)
    grad_tangential = grad_sources - dot * d

    return (
        grad_tangential,
        mx.zeros_like(roi_center),
        mx.zeros_like(mu_sg),
        None, None, None,
        None, None, None, None, None,
        None, None, None, None,
        None,
    )


def compute_absorption_gate(
    sources: mx.array,
    roi_center: mx.array,
    mu_volume,
    cfg: AbsorptionConfig,
    *,
    grad_mode: str = "none",
    beta_contain: float = 10.0,
    gate_type: str = "fraction",
) -> mx.array:
    """Absorption gate ``nu_i`` with configurable gradient mode.

    Parameters
    ----------
    grad_mode : ``"none"`` | ``"fd_src"`` | ``"tangential"`` | ``"analytical_experimental"``

        * ``"none"`` — stop_gradient; only the geometric coverage gradient flows
          (recommended default for Paper 1 baseline).
        * ``"fd_src"`` — FD VJP through the full cone_forward + detector-frame map;
          6 extra forward passes per step.
        * ``"tangential"`` — FD gradient projected onto the tangent plane of the
          viewing sphere (removes radial components; 6 extra passes like fd_src).
        * ``"analytical_experimental"`` — requires ``DIFFCT_GEOMETRY_VJP=1``;
          prints a warning otherwise and falls back to ``fd_src``.
    """
    if gate_type == "path":
        gate_fn = absorption_gate_path
    elif gate_type == "mean":
        gate_fn = absorption_gate_mean
    else:
        gate_fn = absorption_gate
    if grad_mode == "none":
        return mx.stop_gradient(
            gate_fn(sources, roi_center, mu_volume, cfg, beta_contain=beta_contain)
        )
    if grad_mode == "fd_src":
        return gate_fn(sources, roi_center, mu_volume, cfg, beta_contain=beta_contain)
    if grad_mode == "tangential":
        mu_sg = mx.stop_gradient(mx.array(mu_volume, dtype=mx.float32))
        roi_c = mx.array(roi_center, dtype=mx.float32)
        return _absorption_gate_tangential_custom(
            sources, roi_c, mu_sg,
            cfg.alpha, cfg.eta, cfg.sdd,
            cfg.det_u, cfg.det_v,
            cfg.du, cfg.dv, cfg.voxel_spacing,
            cfg.roi_radius, cfg.beta_pixel, cfg.beta_frac, cfg.footprint_floor,
            beta_contain,
        )
    if grad_mode == "analytical_experimental":
        import os
        import warnings
        if os.getenv("DIFFCT_GEOMETRY_VJP", "0") != "1":
            warnings.warn(
                "analytical_experimental requires DIFFCT_GEOMETRY_VJP=1; "
                "falling back to fd_src",
                stacklevel=2,
            )
        return absorption_gate(sources, roi_center, mu_volume, cfg, beta_contain=beta_contain)
    raise ValueError(
        f"Unknown grad_mode {grad_mode!r}. "
        "Choose from 'none', 'fd_src', 'tangential', 'analytical_experimental'."
    )


def _footprint_forward_through_sources(
    quantity: mx.array,
    sources: mx.array,
    roi_center: mx.array,
    cfg: AbsorptionConfig,
) -> mx.array:
    """Forward-project ``quantity`` through the *matched footprint* projector
    using detector geometry consistent with the rest of the eval pipeline.

    This wrapper exists so the calibration and the absorption gate see exactly
    the same numerical values — the legacy custom-VJP wrapper
    ``_cone_forward_with_source_geometry`` uses a Siddon projector with a
    different detector convention which mismatches the calibration probes.
    Because the gate is meant to be used with ``stop_gradient``, we do not
    need a custom VJP here.
    """
    from .eval.geometry import geometry_from_sources
    from ._torch_bridge import is_torch_backend, bridged_cone_forward_footprint

    src_pos, det_center, det_u_vec, det_v_vec = geometry_from_sources(
        sources, sid=cfg.sid, sdd=cfg.sdd
    )
    if is_torch_backend():
        # On the torch/CUDA backend the raw diffct_mlx footprint projector
        # expects torch tensors and is not MLX-differentiable.  Route through the
        # MLX<->Torch bridge so the projector's FD-based geometry VJP still flows
        # back to the source positions -- required for ``grad_mode="fd_src"``
        # (the path-divergence gate), where dropping it would silently collapse
        # the absorption term to a constant.  ``DIFFCOV_VCL_FORWARD`` is unset in
        # paper runs, so the bridge dispatches to the matching footprint operator.
        return bridged_cone_forward_footprint(
            quantity,
            src_pos, det_center, det_u_vec, det_v_vec,
            cfg.det_u, cfg.det_v,
            cfg.du, cfg.dv, cfg.voxel_spacing,
        )
    return diffct_mlx.cone_forward_footprint(
        quantity,
        src_pos, det_center, det_u_vec, det_v_vec,
        cfg.det_u, cfg.det_v,
        cfg.du, cfg.dv, cfg.voxel_spacing,
    )


def absorption_gate_path(
    sources: mx.array,
    roi_center: mx.array,
    mu_volume: mx.array,
    cfg: AbsorptionConfig,
    *,
    beta_contain: float = 10.0,
    mask_threshold: float | None = None,
) -> mx.array:
    """Pure-geometry **path-length** gate.

    Forward-projects a binary mask of the object (``μ > threshold``) and
    gates on the ROI-footprint-weighted mean of the resulting *path length
    through material*.  Because the integrand is binary, the gate signal
    depends only on *how much material the ray crosses*, not on local
    attenuation strength — so it captures shape-driven directional variation
    (e.g. hexagonal cross-section of the ORNL nozzle) that a Beer-Lambert
    line integral averages out.

        ν_i = σ(β_frac · (α - ⟨L_p⟩_ROI))   with   L_p = ∫ 𝟙{μ(x) > τ} dℓ.

    ``α`` is set per-phantom (see :func:`differentiable_coverage.eval.trajectories._calibrate_alpha_path`)
    to the median of per-view ROI-footprint-weighted mean path lengths so the
    sigmoid sits in its sensitive region.
    """
    mu_mlx = mx.stop_gradient(mx.array(mu_volume, dtype=mx.float32))
    if mask_threshold is None:
        mask_threshold = 0.05 * float(mx.max(mu_mlx))
    object_mask = mx.stop_gradient(
        (mu_mlx > mask_threshold).astype(mx.float32)
    )
    projections = _footprint_forward_through_sources(
        object_mask, sources, roi_center, cfg
    )
    weights = _soft_footprint(cfg, sources, roi_center)
    w_sum = mx.sum(weights, axis=(1, 2)) + 1e-9
    mean_path = mx.sum(weights * projections, axis=(1, 2)) / w_sum
    absorption_nu = mx.sigmoid(cfg.beta_frac * (cfg.alpha - mean_path))
    containment_nu = _detector_containment_gate(
        sources, roi_center, cfg, beta_contain=beta_contain
    )
    return absorption_nu * containment_nu


def absorption_gate_mean(
    sources: mx.array,
    roi_center: mx.array,
    mu_volume: mx.array,
    cfg: AbsorptionConfig,
    *,
    beta_contain: float = 10.0,
) -> mx.array:
    """Single-sigmoid mean-attenuation gate.

    Unlike :func:`absorption_gate` which thresholds the *fraction* of unusable
    pixels (sigmoid-of-sigmoid → near-binary), this variant gates on the
    ROI-footprint-weighted *mean* line integral:

        ν_i = σ(β_frac · (α - ⟨a_p⟩_ROI))   with   ⟨a_p⟩_ROI = Σ_p w_p a_p / Σ_p w_p

    A single sigmoid keeps the response smooth and direction-sensitive even
    for high-Z objects whose pixel-level line integrals saturate.  The
    detector-containment gate is multiplied in unchanged.
    """
    mu_mlx = mx.stop_gradient(mx.array(mu_volume, dtype=mx.float32))
    projections = _cone_forward_with_source_geometry(
        mu_mlx, sources, roi_center,
        cfg.sdd, cfg.det_u, cfg.det_v,
        cfg.du, cfg.dv, cfg.voxel_spacing,
    )
    weights = _soft_footprint(cfg, sources, roi_center)
    w_sum = mx.sum(weights, axis=(1, 2)) + 1e-9
    mean_attenuation = mx.sum(weights * projections, axis=(1, 2)) / w_sum
    absorption_nu = mx.sigmoid(cfg.beta_frac * (cfg.alpha - mean_attenuation))
    containment_nu = _detector_containment_gate(
        sources, roi_center, cfg, beta_contain=beta_contain
    )
    return absorption_nu * containment_nu


def absorption_gate(
    sources: mx.array,
    roi_center: mx.array,
    mu_volume: mx.array,
    cfg: AbsorptionConfig,
    *,
    beta_contain: float = 10.0,
) -> mx.array:
    """Differentiable signal-quality gate ``nu_i`` (§2 Steps A-C).

    Combines:
    * **Absorption gate** (§2): logistic gate on the per-view unusable-pixel
      fraction derived from the DiffCT forward projection.
    * **Containment gate**: soft penalisation of sources whose ROI projection
      lies outside the detector footprint (prevents the optimiser from pushing
      sources to geometrically infeasible positions).

    Returns
    -------
    nu : mx.array of shape ``(k,)`` with entries in ``(0, 1)``.
    """
    # MLX's custom_function VJP indexes primals by MLX-array arguments only.
    # If mu_volume were a NumPy array the VJP return tuple would be offset,
    # causing incorrect gradients.  We always convert to MLX and detach from
    # the gradient tape (we are optimising sources, not the attenuation map).
    mu_mlx = mx.stop_gradient(mx.array(mu_volume, dtype=mx.float32))
    projections = _cone_forward_with_source_geometry(
        mu_mlx,
        sources,
        roi_center,
        cfg.sdd,
        cfg.det_u,
        cfg.det_v,
        cfg.du,
        cfg.dv,
        cfg.voxel_spacing,
    )
    pixel_usability = mx.sigmoid(cfg.beta_pixel * (cfg.alpha - projections))
    weights = _soft_footprint(cfg, sources, roi_center)
    weighted_loss = weights * (1.0 - pixel_usability)
    unusable_fraction = mx.sum(weighted_loss, axis=(1, 2)) / mx.sum(weights, axis=(1, 2))
    absorption_nu = mx.sigmoid(cfg.beta_frac * (cfg.eta - unusable_fraction))
    containment_nu = _detector_containment_gate(sources, roi_center, cfg, beta_contain=beta_contain)
    return absorption_nu * containment_nu
