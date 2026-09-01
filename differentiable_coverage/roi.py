"""Lightweight helpers for defining a region of interest (ROI).

The current coverage code expects three pieces of ROI information:

- ``roi_center`` for the main geometry,
- optional ``roi_points`` / ``roi_weights`` for richer coverage aggregation,
- ``roi_radius`` for the absorption-aware detector footprint and containment gate.

This module provides small, readable utilities to build those values from
either a bounding box, a binary mask, or an explicit point cloud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class ROISelection:
    """Minimal ROI bundle used by the rest of the package."""

    center: mx.array
    radius: float
    points: mx.array
    weights: mx.array


def _as_world_vector(value, *, dtype=mx.float32) -> mx.array:
    """Convert a scalar or length-3 vector into a world-space 3-vector."""
    arr = mx.array(value, dtype=dtype)
    if arr.ndim == 0:
        return mx.array([arr, arr, arr], dtype=dtype)
    if arr.shape != (3,):
        raise ValueError(f"Expected scalar or shape (3,), got {arr.shape}")
    return arr


def _uniform_weights(n: int, *, dtype=mx.float32) -> mx.array:
    """Return normalized uniform weights for ``n`` ROI points."""
    if n <= 0:
        raise ValueError("ROI must contain at least one point")
    return mx.ones((n,), dtype=dtype) / float(n)


def _radius_from_points(points: mx.array, center: mx.array) -> float:
    """Max Euclidean distance of any point from the center."""
    dists = mx.linalg.norm(points - center[None, :], axis=1)
    mx.eval(dists)
    return float(mx.max(dists))


def _downsample_points(points: mx.array, max_points: int) -> mx.array:
    """Deterministic stride-based downsampling for human-readable behavior."""
    n = int(points.shape[0])
    if n <= max_points:
        return points
    step = max(1, math.ceil(n / max_points))
    return points[::step][:max_points]


def roi_from_points(
    points,
    *,
    max_points: int | None = None,
    dtype=mx.float32,
) -> ROISelection:
    """Build an ROI from explicit world-space points.

    Parameters
    ----------
    points :
        Array-like of shape ``(m, 3)`` in world coordinates.
    max_points :
        Optional deterministic downsampling cap for ``roi_points``.
    """
    pts = mx.array(points, dtype=dtype)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points with shape (m, 3), got {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError("ROI point set must not be empty")

    if max_points is not None:
        pts = _downsample_points(pts, max_points)

    center = mx.mean(pts, axis=0)
    mx.eval(center)
    radius = _radius_from_points(pts, center)
    weights = _uniform_weights(int(pts.shape[0]), dtype=dtype)
    return ROISelection(center=center, radius=radius, points=pts, weights=weights)


def roi_from_bbox(
    min_corner,
    max_corner,
    *,
    voxel_spacing=1.0,
    origin=(0.0, 0.0, 0.0),
    points_per_axis: int = 3,
    dtype=mx.float32,
) -> ROISelection:
    """Build an ROI from an axis-aligned bounding box in voxel coordinates.

    ``min_corner`` and ``max_corner`` are interpreted as voxel indices.
    Sample points are placed uniformly inside the box and then mapped to
    world coordinates via ``origin + voxel_spacing * index``.
    """
    if points_per_axis < 2:
        raise ValueError("points_per_axis must be at least 2")

    spacing = _as_world_vector(voxel_spacing, dtype=dtype)
    world_origin = _as_world_vector(origin, dtype=dtype)
    min_idx = mx.array(min_corner, dtype=dtype)
    max_idx = mx.array(max_corner, dtype=dtype)

    if min_idx.shape != (3,) or max_idx.shape != (3,):
        raise ValueError("Bounding-box corners must have shape (3,)")

    center_idx = 0.5 * (min_idx + max_idx)
    center = world_origin + spacing * center_idx

    axes = [
        mx.linspace(float(min_idx[i]), float(max_idx[i]), points_per_axis, dtype=dtype)
        for i in range(3)
    ]
    xx, yy, zz = mx.meshgrid(*axes, indexing="ij")
    points_idx = mx.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1)
    points = world_origin[None, :] + points_idx * spacing[None, :]

    radius = _radius_from_points(points, center)
    weights = _uniform_weights(int(points.shape[0]), dtype=dtype)
    return ROISelection(center=center, radius=radius, points=points, weights=weights)


def roi_from_mask(
    mask,
    *,
    voxel_spacing=1.0,
    origin=(0.0, 0.0, 0.0),
    max_points: int = 256,
    dtype=mx.float32,
) -> ROISelection:
    """Build an ROI from a binary mask of shape ``(D, H, W)``.

    Notes
    -----
    MLX does not currently offer a small, dependency-free equivalent of NumPy's
    full ROI tooling, so this helper intentionally keeps the implementation
    simple and readable:

    - active voxels are extracted deterministically via ``mask.tolist()``,
    - voxel centers are converted to world coordinates,
    - points are stride-downsampled if the mask is large.
    """
    mask_mx = mx.array(mask)
    if mask_mx.ndim != 3:
        raise ValueError(f"Expected mask with shape (D, H, W), got {mask_mx.shape}")

    active_indices: list[list[float]] = []
    mask_list = mask_mx.tolist()
    for z, plane in enumerate(mask_list):
        for y, row in enumerate(plane):
            for x, value in enumerate(row):
                if value:
                    active_indices.append([float(z), float(y), float(x)])

    if not active_indices:
        raise ValueError("ROI mask does not contain any active voxels")

    points_idx = mx.array(active_indices, dtype=dtype)
    spacing = _as_world_vector(voxel_spacing, dtype=dtype)
    world_origin = _as_world_vector(origin, dtype=dtype)

    # Use voxel centers rather than voxel corners.
    points = world_origin[None, :] + (points_idx + 0.5) * spacing[None, :]
    points = _downsample_points(points, max_points)

    center = mx.mean(points, axis=0)
    mx.eval(center)
    radius = _radius_from_points(points, center)
    weights = _uniform_weights(int(points.shape[0]), dtype=dtype)
    return ROISelection(center=center, radius=radius, points=points, weights=weights)
