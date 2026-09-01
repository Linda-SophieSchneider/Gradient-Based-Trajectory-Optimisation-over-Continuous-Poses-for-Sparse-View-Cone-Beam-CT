"""MLX-native bundle-integral absorption gate.

A fully autograd-friendly path-integral gate that does **not** go through
``cone_forward`` and therefore needs no finite-difference VJP.  For every
source $s_i$ we integrate $\\mu$ along a small bundle of rays that land on
a disk of radius $r_\\text{ROI}$ centred at the reconstruction ROI.  The
integration uses trilinear volume sampling whose gradient with respect to
the sample coordinates flows analytically through MLX — exactly like
:func:`vcl_loss_continuous` does for the VCLS surrogate.

The resulting gate value $\\nu_i \\in [0,1]$ down-weights views whose
bundle has to cross a lot of material, in a way that has a real
end-to-end differentiable gradient on the source positions.

Formally,

    $$\\bar\\tau_i = \\frac{1}{|\\mathcal R|}
        \\sum_{r\\in\\mathcal R}\\int_0^{L_{i,r}} \\mu\\bigl(s_i + t\\,d_{i,r}\\bigr)\\,dt,$$

    $$\\nu_i = \\sigma\\!\\bigl(\\beta\\,(\\alpha - \\bar\\tau_i)\\bigr),$$

where $\\mathcal R$ is the set of bundle rays (default $5\\times 9 = 45$)
covering a ROI disk of radius $r_\\text{ROI}$ perpendicular to the
viewing axis, $L_{i,r}$ is the source-to-target distance along ray $r$,
and $\\alpha, \\beta$ control the sigmoid soft-threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


# Dimensionless target contribution of the bundle penalty at the median
# probe direction.  Keeping this in one place prevents individual studies
# from silently changing the relative objective scale.
DEFAULT_BUNDLE_PENALTY_TARGET = 0.2


def calibrate_bundle_weight(
    median_tau: float,
    target: float = DEFAULT_BUNDLE_PENALTY_TARGET,
    *,
    eps: float = 1e-6,
) -> float:
    """Return ``lambda_bundle`` for a target median penalty.

    The additive objective uses ``-lambda_bundle * mean(tau_bar)``.  Setting
    ``lambda_bundle = target / median_tau`` makes its typical contribution
    dimensionless and comparable across volumes, resolutions, and scanner
    manifolds.
    """
    if target < 0.0:
        raise ValueError("bundle penalty target must be non-negative")
    return float(target) / max(float(median_tau), eps)


@dataclass(frozen=True)
class BundleAbsorptionConfig:
    """Configuration for :func:`absorption_gate_bundle`.

    ``voxel_spacing`` converts world (mm) coordinates to volume indices.
    The volume centre voxel sits at the world position given by the
    ``volume_center`` argument of the bundle functions (default: the world
    origin / isocentre).  ``roi_center`` is only the bundle *target* and is
    independent of the volume placement.
    """

    roi_radius: float = 5.0          # mm — radius of bundle target disk
    n_rays_u: int = 5
    n_rays_v: int = 9
    n_samples: int = 32              # samples per ray
    voxel_spacing: float = 1.0       # mm/voxel
    # When True, the n_samples midpoint quadrature covers only the
    # ray ∩ volume-bounding-box segment instead of the full source→target
    # segment.  Out-of-volume stretches contribute nothing physically;
    # sampling them wastes quadrature resolution and reads clamped border
    # voxels.  Default False = the legacy full-segment rule used by the
    # published runs (see the bundle_quadrature_convergence study).
    clip_to_volume: bool = False
    # Beer-Lambert-style soft gate: nu = exp(-alpha * tau_bar).  alpha is
    # chosen so that nu(median path) = 0.5, i.e. alpha = ln(2) / median(τ).
    # This avoids sigmoid saturation when the dense-direction τ̄ is far
    # above the median — gradients stay non-zero everywhere.
    alpha: float = 0.05              # 1 / characteristic path integral
    eps: float = 1e-9


def _orthonormal_frame(d: mx.array) -> tuple[mx.array, mx.array]:
    """Return two unit vectors orthogonal to ``d`` (shape ``(k, 3)``).

    Uses a robust world-up reference: cross with $\\hat z$, falling back to
    $\\hat y$ when ``d`` is too parallel.  Returned ``u``, ``v`` are unit
    vectors with ``(u, v, d)`` right-handed (up to numerical accuracy).
    """
    k = d.shape[0]
    z_hat = mx.broadcast_to(mx.array([0.0, 0.0, 1.0]), (k, 3))
    y_hat = mx.broadcast_to(mx.array([0.0, 1.0, 0.0]), (k, 3))

    # If |d · z_hat| > 0.95, d is nearly parallel to z_hat → use y_hat instead.
    dot_z = mx.abs(mx.sum(d * z_hat, axis=-1, keepdims=True))   # (k, 1)
    use_y = mx.broadcast_to(dot_z > 0.95, (k, 3))
    ref = mx.where(use_y, y_hat, z_hat)                          # (k, 3)

    # u = normalize(d × ref); v = d × u
    u_raw = mx.stack([
        d[:, 1] * ref[:, 2] - d[:, 2] * ref[:, 1],
        d[:, 2] * ref[:, 0] - d[:, 0] * ref[:, 2],
        d[:, 0] * ref[:, 1] - d[:, 1] * ref[:, 0],
    ], axis=-1)
    u_norm = mx.maximum(mx.linalg.norm(u_raw, axis=-1, keepdims=True), 1e-9)
    u = u_raw / u_norm

    v_raw = mx.stack([
        d[:, 1] * u[:, 2] - d[:, 2] * u[:, 1],
        d[:, 2] * u[:, 0] - d[:, 0] * u[:, 2],
        d[:, 0] * u[:, 1] - d[:, 1] * u[:, 0],
    ], axis=-1)
    # v_raw is already (close to) unit by construction; normalise for safety.
    v = v_raw / mx.maximum(mx.linalg.norm(v_raw, axis=-1, keepdims=True), 1e-9)
    return u, v


def _trilinear_sample(vol: mx.array, ijk: mx.array) -> mx.array:
    """Differentiable trilinear interpolation.

    Parameters
    ----------
    vol :
        Reference volume of shape ``(Z, Y, X)``.  ``stop_gradient`` should
        be applied by the caller — gradients are taken w.r.t. ``ijk``.
    ijk :
        Sample positions in **volume index coordinates** with shape
        ``(..., 3)``, ordering ``(z, y, x)``.

    Returns
    -------
    values :
        Sampled values with the leading shape of ``ijk``.
    """
    Z, Y, X = vol.shape
    z = mx.clip(ijk[..., 0], 0.0, float(Z - 1))
    y = mx.clip(ijk[..., 1], 0.0, float(Y - 1))
    x = mx.clip(ijk[..., 2], 0.0, float(X - 1))

    z0 = mx.floor(z).astype(mx.int32)
    y0 = mx.floor(y).astype(mx.int32)
    x0 = mx.floor(x).astype(mx.int32)
    z1 = mx.minimum(z0 + 1, mx.array(Z - 1, dtype=mx.int32))
    y1 = mx.minimum(y0 + 1, mx.array(Y - 1, dtype=mx.int32))
    x1 = mx.minimum(x0 + 1, mx.array(X - 1, dtype=mx.int32))

    fz = z - z0.astype(mx.float32)
    fy = y - y0.astype(mx.float32)
    fx = x - x0.astype(mx.float32)

    # Gather 8 corner values.  vol[z, y, x] indexing with arrays broadcasts.
    c000 = vol[z0, y0, x0]
    c001 = vol[z0, y0, x1]
    c010 = vol[z0, y1, x0]
    c011 = vol[z0, y1, x1]
    c100 = vol[z1, y0, x0]
    c101 = vol[z1, y0, x1]
    c110 = vol[z1, y1, x0]
    c111 = vol[z1, y1, x1]

    # Blend along x, then y, then z.  The integer indices act as constants;
    # the smooth (fx, fy, fz) carry the gradient.
    c00 = c000 * (1.0 - fx) + c001 * fx
    c01 = c010 * (1.0 - fx) + c011 * fx
    c10 = c100 * (1.0 - fx) + c101 * fx
    c11 = c110 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c01 * fy
    c1 = c10 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


def bundle_path_integral(
    sources: mx.array,        # (k, 3) world (mm)
    roi_center: mx.array,     # (3,)   world (mm)
    mu_volume: mx.array,      # (Z, Y, X)
    cfg: BundleAbsorptionConfig | None = None,
    *,
    volume_center: mx.array | None = None,
) -> mx.array:
    """Bundle-mean line integral ``τ̄_i = ⟨∫ μ dℓ⟩_bundle``.

    Returns a ``(k,)`` vector of dimensionless optical depths when attenuation
    is expressed in mm⁻¹ and world coordinates in mm.
    Differentiable in ``sources`` through trilinear sampling — no FD, no
    cone_forward.  This is the *raw* path integral; use as an additive
    penalty $-\\lambda \\cdot \\overline{τ̄}$ in the objective to avoid
    saturation problems that arise when wrapping it in a bounded gate.

    ``volume_center`` is the world position (mm) of the volume centre voxel
    and defaults to the world origin / isocentre.  It is deliberately
    separate from ``roi_center``: an off-centre bundle target must not
    translate the attenuation field.
    """
    cfg = cfg or BundleAbsorptionConfig()
    return _bundle_tau_bar(sources, roi_center, mu_volume, cfg,
                           volume_center=volume_center)


def _bundle_tau_bar(
    sources: mx.array,
    roi_center: mx.array,
    mu_volume: mx.array,
    cfg: BundleAbsorptionConfig,
    *,
    volume_center: mx.array | None = None,
) -> mx.array:
    """Internal helper that computes τ̄_i for one (sources, mu) pair.

    Factored out so both :func:`bundle_path_integral` and
    :func:`absorption_gate_bundle` share the same trilinear pipeline.
    """
    mu_sg = mx.stop_gradient(mx.array(mu_volume, dtype=mx.float32))
    Z, Y, X = mu_sg.shape
    centre_idx = mx.array(
        [(Z - 1) / 2.0, (Y - 1) / 2.0, (X - 1) / 2.0], dtype=mx.float32
    )
    if volume_center is None:
        vol_c = None                       # volume centred on the world origin
    else:
        vol_c = mx.array(volume_center, dtype=mx.float32).reshape(3)

    r_vec = roi_center[None, :] - sources                 # (k, 3)
    rho = mx.linalg.norm(r_vec, axis=-1, keepdims=True)
    d = r_vec / mx.maximum(rho, cfg.eps)
    u_hat, v_hat = _orthonormal_frame(d)

    # Single-ray (centerline) case: grid collapses to a single 0 offset so
    # the bundle reduces to a single ray from source to roi_center.
    gu = mx.linspace(-1.0, 1.0, cfg.n_rays_u) if cfg.n_rays_u > 1 else mx.array([0.0])
    gv = mx.linspace(-1.0, 1.0, cfg.n_rays_v) if cfg.n_rays_v > 1 else mx.array([0.0])
    GU, GV = mx.meshgrid(gu, gv, indexing="ij")
    GU = GU.reshape(-1)
    GV = GV.reshape(-1)
    R = cfg.roi_radius

    offsets = R * (
        GU[None, :, None] * u_hat[:, None, :]
        + GV[None, :, None] * v_hat[:, None, :]
    )
    targets = roi_center[None, None, :] + offsets

    t_grid = (mx.arange(cfg.n_samples, dtype=mx.float32) + 0.5) / cfg.n_samples
    ray_dir = targets - sources[:, None, :]               # (k, R, 3)

    if cfg.clip_to_volume:
        # Slab intersection of each ray with the volume bounding box in
        # world (x, y, z); quadrature covers only the in-box sub-segment.
        # Piecewise differentiable in sources through the min/max chain.
        half = mx.array([(X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0],
                        dtype=mx.float32) * cfg.voxel_spacing
        box_c = mx.zeros(3, dtype=mx.float32) if vol_c is None else vol_c
        lo = (box_c - half)[None, None, :]
        hi = (box_c + half)[None, None, :]
        s = sources[:, None, :]                            # (k, 1, 3)
        d_safe = mx.where(mx.abs(ray_dir) < 1e-9,
                          mx.where(ray_dir >= 0.0, 1e-9, -1e-9), ray_dir)
        ta = (lo - s) / d_safe
        tb = (hi - s) / d_safe
        t_enter = mx.clip(mx.max(mx.minimum(ta, tb), axis=-1), 0.0, 1.0)
        t_exit = mx.clip(mx.min(mx.maximum(ta, tb), axis=-1), 0.0, 1.0)
        seg = mx.maximum(t_exit - t_enter, 0.0)            # (k, R)
        t = t_enter[:, :, None] + t_grid[None, None, :] * seg[:, :, None]
        pts = s[:, :, None, :] + t[..., None] * ray_dir[:, :, None, :]
        seg_frac = seg
    else:
        t = t_grid.reshape(1, 1, cfg.n_samples, 1)
        pts = sources[:, None, None, :] + t * ray_dir[:, :, None, :]
        seg_frac = None

    p_rel = pts if vol_c is None else pts - vol_c[None, None, None, :]
    ijk = mx.stack([
        p_rel[..., 2] / cfg.voxel_spacing + centre_idx[0],
        p_rel[..., 1] / cfg.voxel_spacing + centre_idx[1],
        p_rel[..., 0] / cfg.voxel_spacing + centre_idx[2],
    ], axis=-1)

    mu_samples = _trilinear_sample(mu_sg, ijk)
    ray_len = mx.linalg.norm(ray_dir, axis=-1)
    if seg_frac is None:
        dl = ray_len / cfg.n_samples
    else:
        dl = ray_len * seg_frac / cfg.n_samples
    line_int = mx.sum(mu_samples, axis=-1) * dl
    return mx.mean(line_int, axis=-1)                     # τ̄_i, (k,)


def absorption_gate_bundle(
    sources: mx.array,        # (k, 3) world (mm)
    roi_center: mx.array,     # (3,)   world (mm)
    mu_volume: mx.array,      # (Z, Y, X)
    cfg: BundleAbsorptionConfig | None = None,
    *,
    volume_center: mx.array | None = None,
) -> mx.array:
    """Analytic bundle-integral absorption gate.

    Returns a ``(k,)`` vector ``nu`` of gate values in $[0, 1]$, where
    *lower* values mark sources whose viewing bundle traverses heavy
    absorption (and which should therefore be de-prioritised).  The
    gradient with respect to ``sources`` is computed end-to-end through
    trilinear sampling — no finite differences, no cone_forward.

    The volume centre voxel sits at ``volume_center`` (world mm, default:
    the world origin / isocentre); ``roi_center`` is only the bundle
    target.  Pass a calibrated ``cfg.alpha`` (e.g. the median bundle mean
    over the candidate set) so the sigmoid sits in its sensitive region
    for the phantom.
    """
    cfg = cfg or BundleAbsorptionConfig()
    tau_bar = _bundle_tau_bar(sources, roi_center, mu_volume, cfg,
                              volume_center=volume_center)
    # nu_i = exp(-alpha * tau_bar_i) ∈ (0, 1].  Saturates as τ̄ grows;
    # for optimisation prefer :func:`bundle_path_integral` with an
    # additive `-λ · mean(τ̄)` penalty, which does not saturate.
    return mx.exp(-cfg.alpha * tau_bar)


def calibrate_bundle_alpha(
    mu_volume: mx.array,
    roi_center: mx.array,
    sid: float,
    *,
    n_probe: int = 256,
    cfg: BundleAbsorptionConfig | None = None,
    seed: int = 0,
    volume_center: mx.array | None = None,
) -> float:
    """Pick ``alpha`` so ``nu = exp(-alpha · τ̄)`` returns 0.5 for the
    median-direction source on the SID sphere.

    Specifically, ``alpha = ln 2 / median(τ̄)`` over ``n_probe`` uniform
    sphere samples.  This places the median source in the sensitive
    region of the exponential gate, while ensuring gradients stay
    non-zero even for high-absorption directions (Beer-Lambert never
    saturates above zero).
    """
    from .score import sample_unit_sphere
    import math as _math

    cfg = cfg or BundleAbsorptionConfig()
    probes = sample_unit_sphere(n_probe) * sid           # (n_probe, 3)
    probes_w = probes + roi_center[None, :]

    # τ̄ directly (cfg.alpha is irrelevant to the raw path integral); no
    # gate/exp/log round trip, no clamp on dense phantoms.
    tau = _bundle_tau_bar(probes_w, roi_center, mu_volume, cfg,
                          volume_center=volume_center)
    med_tau = float(mx.median(tau))
    if med_tau <= 0.0:
        # Phantom is essentially empty — return a neutral alpha.
        return 1.0
    return _math.log(2.0) / med_tau
