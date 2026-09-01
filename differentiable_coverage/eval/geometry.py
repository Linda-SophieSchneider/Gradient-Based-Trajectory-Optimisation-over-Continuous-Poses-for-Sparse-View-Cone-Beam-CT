"""Detector geometry from arbitrary source positions.

Given an ``(n_views, 3)`` source-position array on a sphere of radius
``sid``, build the detector triple ``(det_center, det_u_vec, det_v_vec)``
following the same convention as ``diffct_mlx.custom_trajectory_3d``:

* The detector centre sits on the ray ``source -> isocenter -> det_center``
  at distance ``sdd - sid`` behind the isocenter.
* The detector's ``u`` axis is the unit vector in the xy-plane orthogonal
  to the source's projection on xy (the "azimuthal" tangent).
* The detector's ``v`` axis is the cross product ``(source/||source||) x u``,
  which becomes ``[0, 0, 1]`` for sources in the xy-plane and degrades
  gracefully for off-axis sources.

This makes the geometry well-defined for any source on a sphere, not only
for the circular orbit.
"""

from __future__ import annotations

import mlx.core as mx


def geometry_from_sources(
    src_pos: mx.array,
    *,
    sid: float,
    sdd: float,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Build a complete cone-beam geometry from arbitrary source positions.

    Parameters
    ----------
    src_pos
        ``(n_views, 3)`` array of source positions; ``||src_pos[i]|| ≈ sid``
        is assumed but not enforced (we normalize internally).
    sid, sdd
        Source-to-isocenter and source-to-detector distance.

    Returns
    -------
    src_pos, det_center, det_u_vec, det_v_vec
        Each ``(n_views, 3)``, in the diffct_mlx convention.
    """
    src_pos = mx.array(src_pos, dtype=mx.float32)
    if src_pos.ndim != 2 or src_pos.shape[-1] != 3:
        raise ValueError(f"src_pos must be (n_views, 3); got {tuple(src_pos.shape)}")

    src_norm = mx.linalg.norm(src_pos, axis=1, keepdims=True)
    src_unit = src_pos / mx.maximum(src_norm, eps)

    # Detector centre: opposite of source, behind isocenter
    det_center = -src_unit * (sdd - sid)

    # u-axis: in xy-plane, perpendicular to the source's xy projection.
    # For a source at (-sid sinθ, sid cosθ, *), this should be (cosθ, sinθ, 0)
    # to match diffct_mlx.circular_trajectory_3d exactly when src_z = 0.
    sx = src_unit[:, 0]
    sy = src_unit[:, 1]
    # Build u purely in xy; renormalize to handle north/south pole degeneracy.
    u_x = -sy
    u_y = sx
    u_z = mx.zeros_like(u_x)
    u_raw = mx.stack([u_x, u_y, u_z], axis=-1)
    u_norm = mx.linalg.norm(u_raw, axis=1, keepdims=True)
    fallback = mx.array([[1.0, 0.0, 0.0]], dtype=mx.float32)
    det_u_vec = mx.where(u_norm > eps, u_raw / mx.maximum(u_norm, eps), fallback)

    # v-axis: cross(src_unit, u) — note sign so that v ≈ +z for equatorial sources.
    v_raw = mx.stack(
        [
            det_u_vec[:, 1] * src_unit[:, 2] - det_u_vec[:, 2] * src_unit[:, 1],
            det_u_vec[:, 2] * src_unit[:, 0] - det_u_vec[:, 0] * src_unit[:, 2],
            det_u_vec[:, 0] * src_unit[:, 1] - det_u_vec[:, 1] * src_unit[:, 0],
        ],
        axis=-1,
    )
    # The sign convention of diffct_mlx puts v along +z for circular orbits.
    # cross(u, src_unit) gives that orientation; cross(src_unit, u) gives the
    # opposite. We pick whichever matches +z for the equatorial case.
    sign = mx.sign(v_raw[:, 2:3] + 1e-12)
    # If v_z ~ 0 (off-equator), just keep v_raw — the absolute orientation is
    # consistent across views by construction.
    det_v_vec = mx.where(mx.abs(v_raw[:, 2:3]) > eps, v_raw * sign, v_raw)
    v_norm = mx.linalg.norm(det_v_vec, axis=1, keepdims=True)
    det_v_vec = det_v_vec / mx.maximum(v_norm, eps)

    return src_pos, det_center, det_u_vec, det_v_vec
