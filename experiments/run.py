"""Single-entry-point reproduction runner.

Reads a YAML config that describes one experiment cell of the paper
and produces the matching CSV (and optionally figure).  Paper-1 configs
live under ``experiments/configs/paper1/main`` and
``experiments/configs/paper1/ablation``.

Example
-------
::

    python experiments/run.py --config experiments/configs/paper1/main/milp_384.yaml
    python experiments/run.py --config experiments/configs/paper1/main/ornl_512.yaml

Phantom data
------------
The MILP and ORNL reference volumes are *not* shipped with this
repository.  Place the downloaded files at the paths configured in each
config (default ``./data/...``) before running.  See the project README
for download instructions.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# ----------------------------------------------------------------------
# Lazy imports so help text and config validation work without MLX.
# ----------------------------------------------------------------------

def _load_mlx_stack():
    from differentiable_coverage.runtime import configure_runtime

    runtime = configure_runtime()
    import mlx.core as mx
    import numpy as np
    from differentiable_coverage.eval.geometry import geometry_from_sources
    from differentiable_coverage.eval.metrics import (
        psnr, ssim, nrmse, hfen, rmse,
        roi_psnr, roi_ssim, roi_nrmse, roi_hfen, roi_rmse,
    )
    from differentiable_coverage.eval.reco import (
        reconstruct_sart_volume, simulate_sinogram,
    )
    from differentiable_coverage.eval.trajectories import (
        build_baseline_sources, last_optim_terms, reset_optim_terms,
    )
    from differentiable_coverage.eval.vcl import compute_R_gamma
    from differentiable_coverage.score import sample_unit_sphere

    return {
        "mx": mx,
        "np": np,
        "geometry_from_sources": geometry_from_sources,
        "psnr": psnr, "ssim": ssim, "nrmse": nrmse, "hfen": hfen, "rmse": rmse,
        "roi_psnr": roi_psnr, "roi_ssim": roi_ssim,
        "roi_nrmse": roi_nrmse, "roi_hfen": roi_hfen,
        "roi_rmse": roi_rmse,
        "reconstruct_sart_volume": reconstruct_sart_volume,
        "simulate_sinogram": simulate_sinogram,
        "build_baseline_sources": build_baseline_sources,
        "last_optim_terms": last_optim_terms,
        "reset_optim_terms": reset_optim_terms,
        "compute_R_gamma": compute_R_gamma,
        "sample_unit_sphere": sample_unit_sphere,
        "runtime": runtime,
    }


# ----------------------------------------------------------------------
# Phantom loaders
# ----------------------------------------------------------------------

def _apply_r5_env_overrides(spec: dict) -> dict:
    """Merge R5 prior-mismatch env-var overrides into the phantom spec.

    Archived prior-mismatch configs can sweep over noise/blur levels by
    setting ``DIFFCT_R5_SIGMA`` / ``DIFFCT_R5_BLUR`` between invocations;
    the YAML carries a documentation-only default that must not silently
    shadow the env var.  Returns a shallow copy of *spec* with the
    overrides applied.
    """
    out = dict(spec)
    sigma_env = os.environ.get("DIFFCT_R5_SIGMA")
    blur_env = os.environ.get("DIFFCT_R5_BLUR")
    if sigma_env is not None:
        out["noise_sigma_pct"] = float(sigma_env)
    if blur_env is not None:
        out["blur_voxels"] = float(blur_env)
    return out


def _has_prior_degradation(spec: dict) -> bool:
    """True iff the spec requests any R5-style prior degradation."""
    return (
        float(spec.get("noise_sigma_pct", 0.0)) > 0.0
        or float(spec.get("blur_voxels", 0.0)) > 0.0
    )


def _load_phantom_pair(spec: dict, stack: dict):
    """Load (vol_gt, vol_prior) for one phantom spec.

    *vol_gt* is the unaltered phantom used for sinogram simulation and
    metric computation; *vol_prior* is what the bundle/VCL terms see.
    When no R5 degradation is configured (the common case),
    ``vol_gt is vol_prior``.  Env vars ``DIFFCT_R5_SIGMA`` /
    ``DIFFCT_R5_BLUR`` override the corresponding YAML fields, so the
    R5 driver loop can sweep noise/blur levels without rewriting YAML.

    Supported types: ``ornl_nozzle`` (HDF5 from the Lin et al. release),
    ``milp_npy`` (cubic ``.npy`` attenuation maps), and ``carm_real``
    (our own real C-arm acquisition, reconstructed on demand and cached).
    """
    mx = stack["mx"]; np = stack["np"]
    spec = _apply_r5_env_overrides(spec)

    phantom_type = spec["type"]
    path = Path(spec.get("path", "")).expanduser()
    # The ORNL and carm_real loaders resolve their actual data via env
    # vars (or bundled/cached defaults) and ignore the YAML ``path``
    # field, so we skip the existence check for those phantom types.
    if phantom_type not in ("ornl_nozzle", "carm_real") and not path.exists():
        raise FileNotFoundError(
            f"Phantom file {path} not found.  See the project README for "
            "download instructions."
        )

    if phantom_type == "ornl_nozzle":
        from differentiable_coverage.eval.datasets.ornl_nozzle import (
            load_as_phantom,
        )
        section = spec.get("section", "L")
        n = int(spec["resolution"])
        vol = load_as_phantom(section, (n, n, n))
        return vol, vol

    if phantom_type == "carm_real":
        from differentiable_coverage.eval.datasets.carm_real import (
            load_as_phantom as load_carm_phantom,
        )
        n = int(spec["resolution"])
        data = load_carm_phantom(
            spec.get("object_name", "pigeon"),
            spec.get("tilt", "P75"),
            volume_shape=(n, n, n),
            voxel_spacing_mm=float(spec.get("voxel_spacing_mm", 0.5)),
            downsample=int(spec.get("downsample", 4)),
        )
        vol = mx.array(data.reconstruction)
        return vol, vol

    if phantom_type == "milp_npy":
        v = np.load(path).astype(np.float32)
        if v.ndim != 3 or len(set(v.shape)) != 1:
            raise ValueError(
                f"milp_npy phantom must be cubic 3-D, got shape {v.shape}"
            )
        source = int(v.shape[0])
        target = int(spec.get("resolution", source))
        if target < source and source % target == 0:
            f = source // target
            v = v.reshape(target, f, target, f, target, f).mean(axis=(1, 3, 5))
        elif target != source:
            raise ValueError(
                f"Cannot block-average {path} from {source}^3 to {target}^3; "
                "source resolution must be divisible by target resolution."
            )
        vol_gt = mx.array(v)
        if not _has_prior_degradation(spec):
            return vol_gt, vol_gt

        # Build the degraded prior volume on top of the clean copy.
        v_prior = v.copy()
        if float(spec.get("noise_sigma_pct", 0.0)) > 0.0:
            rng = np.random.default_rng(int(spec.get("noise_seed", 0)))
            sigma = float(spec["noise_sigma_pct"]) / 100.0 * float(v.max())
            v_prior = v_prior + rng.normal(
                0.0, sigma, size=v_prior.shape,
            ).astype(np.float32)
            v_prior = np.maximum(v_prior, 0.0)
        if float(spec.get("blur_voxels", 0.0)) > 0.0:
            from scipy.ndimage import gaussian_filter
            v_prior = gaussian_filter(v_prior, sigma=float(spec["blur_voxels"]))
        return vol_gt, mx.array(v_prior)

    raise ValueError(f"Unknown phantom type {phantom_type!r}")


def _load_phantom(spec: dict, stack: dict):
    """Backward-compat single-volume loader.

    Returns the *prior* volume (degraded under R5, identical to GT
    otherwise).  External callers — notably
    ``experiments.precompute_caches`` — build the (R, gamma) cache from
    the prior, so this is the right default.
    """
    _, vol_prior = _load_phantom_pair(spec, stack)
    return vol_prior


# ----------------------------------------------------------------------
# Geometry presets (Lin et al.\ ORNL geometry + the MILP cone-beam setup
# we use in the paper).  Everything matches what is documented in
# §IV-B of the paper.
# ----------------------------------------------------------------------

GEOMETRY_PRESETS = {
    # ORNL Lin et al.\ TPAMI 2025 native geometry, voxel pitch scaled
    # to the chosen reconstruction grid.
    "ornl": {
        "sid": 243.307, "sdd": 808.508,
        "det_voxels": 1024, "det_pitch": 0.2,
        "voxel_pitch_native": 0.06, "native_resolution": 1008,
    },
    # Standard cone-beam geometry used for the MILP phantom benchmark
    # (Sec. IV-B).  Voxel pitch depends on the reconstruction grid.
    "milp": {
        "sid": 500.0, "sdd": 900.0,
        "det_voxels": 256, "det_pitch": 0.5,
        "voxel_pitch_native": 0.3, "native_resolution": 384,
    },
    # Real C-arm acquisition (2025-06 campaign), pigeon/P75.  sid/sdd are
    # the mean of the 217 per-view GeoKit-decomposed values (near-constant
    # across the real trajectory); det_voxels/det_pitch match the 4x
    # detector downsample used to build the reference reconstruction.
    # Only used to synthesise NEW candidate-view sinograms against that
    # reference volume, exactly as the ORNL preset does -- not the real
    # per-view geometry itself, which only feeds the one-time reference
    # reconstruction in differentiable_coverage.eval.datasets.carm_real.
    "carm_pigeon": {
        "sid": 539.85, "sdd": 1001.6,
        "det_voxels": 360, "det_pitch": 1.2,
        "voxel_pitch_native": 0.5, "native_resolution": 512,
    },
}


def _resolve_geometry(name: str, resolution: int) -> dict:
    """Return a geometry dict for the given preset and recon resolution.

    Voxel pitch is scaled so that the physical field of view matches the
    native acquisition resolution.
    """
    p = GEOMETRY_PRESETS[name]
    return {
        "sid": p["sid"], "sdd": p["sdd"],
        "det_voxels": p["det_voxels"], "det_pitch": p["det_pitch"],
        "voxel_pitch": (
            p["voxel_pitch_native"] * p["native_resolution"] / resolution
        ),
    }


def _centered_world_axes(shape: tuple[int, int, int],
                         voxel_spacing: float):
    import numpy as np

    nz, ny, nx = (int(s) for s in shape)
    z = (np.arange(nz, dtype=np.float32) - 0.5 * (nz - 1)) * float(voxel_spacing)
    y = (np.arange(ny, dtype=np.float32) - 0.5 * (ny - 1)) * float(voxel_spacing)
    x = (np.arange(nx, dtype=np.float32) - 0.5 * (nx - 1)) * float(voxel_spacing)
    return x, y, z


_ROI_EVAL_ONLY_METHODS = {
    "vcls",
    "vcls_circle",
    "vcls_adam_vcl",
    "vcls_adam_geo",
    "vcls_adam_vcl_two_axis",
    "vcls_adam_vcl_carm",
    "vcls_adam_anneal",
    "vcls_adam_langevin",
    "vcls_adam_ensemble",
}


def _roi_grid_shape(spec: dict) -> tuple[int, int, int]:
    grid = spec.get("point_grid", 5)
    if isinstance(grid, int):
        if grid < 1:
            raise ValueError("roi.point_grid must be >= 1")
        return (grid, grid, grid)
    vals = tuple(int(v) for v in grid)
    if len(vals) != 3 or any(v < 1 for v in vals):
        raise ValueError("roi.point_grid must be an int or a 3-tuple of ints >= 1")
    return vals


def _linspace_mm(lo: float, hi: float, n: int, np) -> Any:
    if n <= 1 or abs(hi - lo) < 1e-9:
        return np.array([0.5 * (lo + hi)], dtype=np.float32)
    return np.linspace(lo, hi, n, dtype=np.float32)


def _sphere_mask(shape: tuple[int, int, int], x, y, z, center, radius_mm, np):
    yy2 = (y[:, None] - center[1]) ** 2
    xx2 = (x[None, :] - center[0]) ** 2
    r2_xy = yy2 + xx2
    mask = np.empty(shape, dtype=bool)
    rad2 = float(radius_mm) ** 2
    for iz, zc in enumerate(z):
        mask[iz] = (r2_xy + (zc - center[2]) ** 2) <= rad2
    return mask


def _sample_sphere_points(center, radius_mm: float, grid_shape, np):
    xs = _linspace_mm(center[0] - radius_mm, center[0] + radius_mm, grid_shape[0], np)
    ys = _linspace_mm(center[1] - radius_mm, center[1] + radius_mm, grid_shape[1], np)
    zs = _linspace_mm(center[2] - radius_mm, center[2] + radius_mm, grid_shape[2], np)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    pts = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=-1)
    keep = np.sum((pts - center[None, :]) ** 2, axis=1) <= float(radius_mm) ** 2 + 1e-6
    pts = pts[keep]
    if pts.size == 0:
        pts = center[None, :]
    weights = np.ones(pts.shape[0], dtype=np.float32) / pts.shape[0]
    return pts.astype(np.float32), weights


def _sample_bbox_points(lo, hi, grid_shape, np):
    xs = _linspace_mm(float(lo[0]), float(hi[0]), grid_shape[0], np)
    ys = _linspace_mm(float(lo[1]), float(hi[1]), grid_shape[1], np)
    zs = _linspace_mm(float(lo[2]), float(hi[2]), grid_shape[2], np)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    pts = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=-1)
    weights = np.ones(pts.shape[0], dtype=np.float32) / pts.shape[0]
    return pts.astype(np.float32), weights


def _select_support_component(
    support: Any,
    vol_np: Any,
    *,
    component_threshold_frac: float | None,
    component_select: str,
    np,
):
    if component_threshold_frac is None:
        return support, None
    from scipy import ndimage as ndi

    vmax = float(vol_np.max())
    comp_mask = vol_np > float(component_threshold_frac) * vmax
    labels, n_comp = ndi.label(comp_mask)
    if n_comp == 0:
        raise ValueError("support_com_sphere component filter produced no component")

    sizes = np.bincount(labels.ravel())[1:]
    masses = np.array(
        [float(vol_np[labels == (i + 1)].sum()) for i in range(n_comp)],
        dtype=np.float64,
    )
    means = np.array(
        [float(vol_np[labels == (i + 1)].mean()) for i in range(n_comp)],
        dtype=np.float64,
    )

    if component_select == "largest":
        idx = int(np.argmax(sizes))
    elif component_select == "mass":
        idx = int(np.argmax(masses))
    elif component_select == "mean":
        idx = int(np.argmax(means))
    else:
        raise ValueError(
            "roi.component_select must be one of: largest, mass, mean"
        )
    selected = labels == (idx + 1)
    return support & selected, {
        "threshold_frac": float(component_threshold_frac),
        "select": component_select,
        "size": int(sizes[idx]),
        "mass": float(masses[idx]),
        "mean": float(means[idx]),
    }


def _support_center_mm(
    support: Any,
    vol_np: Any,
    x,
    y,
    z,
    *,
    center_mode: str,
    np,
):
    coords = np.argwhere(support)
    if coords.size == 0:
        raise ValueError("support_com_sphere center selection got empty support")

    if center_mode == "com":
        support_weights = vol_np[support].astype(np.float64)
        wsum = float(np.sum(support_weights))
        wx = x[coords[:, 2]]
        wy = y[coords[:, 1]]
        wz = z[coords[:, 0]]
        center = np.array([
            np.sum(wx * support_weights) / max(wsum, 1e-9),
            np.sum(wy * support_weights) / max(wsum, 1e-9),
            np.sum(wz * support_weights) / max(wsum, 1e-9),
        ], dtype=np.float32)
        return center, {
            "mode": "com",
        }

    if center_mode == "max_clearance":
        from scipy import ndimage as ndi

        dist = ndi.distance_transform_edt(support.astype(np.uint8))
        max_dist = float(dist[support].max())
        best = support & (dist >= max_dist - 1e-6)
        best_coords = np.argwhere(best)
        best_weights = vol_np[best].astype(np.float64)
        if best_weights.size == 0 or float(best_weights.sum()) <= 0.0:
            best_weights = np.ones(best_coords.shape[0], dtype=np.float64)
        wx = x[best_coords[:, 2]]
        wy = y[best_coords[:, 1]]
        wz = z[best_coords[:, 0]]
        center = np.array([
            np.sum(wx * best_weights) / max(float(best_weights.sum()), 1e-9),
            np.sum(wy * best_weights) / max(float(best_weights.sum()), 1e-9),
            np.sum(wz * best_weights) / max(float(best_weights.sum()), 1e-9),
        ], dtype=np.float32)
        return center, {
            "mode": "max_clearance",
            "max_clearance_vox": max_dist,
            "n_peak_vox": int(best_coords.shape[0]),
        }

    raise ValueError("roi.center_mode must be one of: com, max_clearance")


def _method_optimizes_roi(name: str) -> bool:
    return name not in _ROI_EVAL_ONLY_METHODS


def _method_roi_center(name: str, roi_ctx: dict, stack: dict):
    if _method_optimizes_roi(name):
        return roi_ctx["center"]
    return stack["mx"].zeros_like(roi_ctx["center"])


def _resolve_roi_context(cfg: dict, vol, geometry: dict, stack: dict,
                         *, want_mask: bool = False) -> dict:
    np = stack["np"]; mx = stack["mx"]
    spec = cfg.get("roi")
    if not spec:
        return {
            "center": mx.zeros(3, dtype=mx.float32),
            "radius_mm": None,
            "mask": None,
            "points": None,
            "weights": None,
            "type": "none",
            "summary": "ROI default: isocenter only",
        }

    shape = tuple(int(s) for s in vol.shape)
    voxel_spacing = float(geometry["voxel_pitch"])
    x, y, z = _centered_world_axes(shape, voxel_spacing)
    mask = None
    points = None
    weights = None
    grid_shape = _roi_grid_shape(spec)

    roi_type = str(spec.get("type", "sphere"))
    if roi_type in ("sphere", "center_radius"):
        center = np.asarray(
            spec.get("center_mm", [0.0, 0.0, 0.0]), dtype=np.float32,
        )
        if center.shape != (3,):
            raise ValueError("roi.center_mm must have shape (3,)")
        radius_mm = float(spec["radius_mm"])
        points, weights = _sample_sphere_points(center, radius_mm, grid_shape, np)
        summary = (
            f"ROI sphere center={center.tolist()} mm radius={radius_mm:.3f} mm "
            f"points={points.shape[0]}"
        )
        if want_mask:
            mask = _sphere_mask(shape, x, y, z, center, radius_mm, np)
    elif roi_type == "bbox":
        if "min_mm" in spec and "max_mm" in spec:
            lo = np.asarray(spec["min_mm"], dtype=np.float32)
            hi = np.asarray(spec["max_mm"], dtype=np.float32)
        elif "min_vox" in spec and "max_vox" in spec:
            min_vox = np.asarray(spec["min_vox"], dtype=np.float32)
            max_vox = np.asarray(spec["max_vox"], dtype=np.float32)
            if min_vox.shape != (3,) or max_vox.shape != (3,):
                raise ValueError("roi min/max voxel corners must have shape (3,)")
            x0 = -0.5 * (shape[2] - 1) * voxel_spacing
            y0 = -0.5 * (shape[1] - 1) * voxel_spacing
            z0 = -0.5 * (shape[0] - 1) * voxel_spacing
            lo = np.array([
                x0 + min_vox[0] * voxel_spacing,
                y0 + min_vox[1] * voxel_spacing,
                z0 + min_vox[2] * voxel_spacing,
            ], dtype=np.float32)
            hi = np.array([
                x0 + max_vox[0] * voxel_spacing,
                y0 + max_vox[1] * voxel_spacing,
                z0 + max_vox[2] * voxel_spacing,
            ], dtype=np.float32)
        else:
            raise ValueError(
                "roi bbox needs either min_mm/max_mm or min_vox/max_vox"
            )
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("roi min/max corners must have shape (3,)")
        lo0, hi0 = lo, hi
        lo = np.minimum(lo0, hi0)
        hi = np.maximum(lo0, hi0)
        center = 0.5 * (lo + hi)
        radius_mm = float(np.linalg.norm(0.5 * (hi - lo)))
        points, weights = _sample_bbox_points(lo, hi, grid_shape, np)
        summary = (
            f"ROI bbox min={lo.tolist()} mm max={hi.tolist()} mm "
            f"(center={center.tolist()} mm points={points.shape[0]})"
        )
        if want_mask:
            mask_x = (x >= lo[0]) & (x <= hi[0])
            mask_y = (y >= lo[1]) & (y <= hi[1])
            mask_z = (z >= lo[2]) & (z <= hi[2])
            mask = mask_z[:, None, None] & mask_y[None, :, None] & mask_x[None, None, :]
    elif roi_type == "support_com_sphere":
        vol_np = np.asarray(vol, dtype=np.float32)
        vmax = float(vol_np.max())
        threshold_frac = float(spec.get("support_threshold_frac", 0.05))
        support = vol_np > threshold_frac * vmax
        component_info = None
        component_threshold_frac = spec.get("component_threshold_frac")
        if component_threshold_frac is not None:
            support, component_info = _select_support_component(
                support, vol_np,
                component_threshold_frac=float(component_threshold_frac),
                component_select=str(spec.get("component_select", "largest")),
                np=np,
            )
        if not bool(support.any()):
            raise ValueError("support_com_sphere produced an empty support mask")
        center, center_info = _support_center_mm(
            support, vol_np, x, y, z,
            center_mode=str(spec.get("center_mode", "com")),
            np=np,
        )
        coords = np.argwhere(support)
        wx = x[coords[:, 2]]
        wy = y[coords[:, 1]]
        wz = z[coords[:, 0]]
        bbox_lo = np.array([wx.min(), wy.min(), wz.min()], dtype=np.float32)
        bbox_hi = np.array([wx.max(), wy.max(), wz.max()], dtype=np.float32)
        bbox_diag = float(np.linalg.norm(bbox_hi - bbox_lo))
        radius_scale = float(spec.get("radius_scale", 0.25))
        radius_mm = float(np.clip(
            radius_scale * bbox_diag,
            float(spec.get("radius_min_mm", 8.0)),
            float(spec.get("radius_max_mm", 15.0)),
        ))
        points, weights = _sample_sphere_points(center, radius_mm, grid_shape, np)
        summary = "ROI support_com_sphere "
        if component_info is not None:
            summary += (
                f"component={component_info['select']}@{component_info['threshold_frac']:.3f} "
                f"size={component_info['size']} "
            )
        summary += f"center_mode={center_info['mode']} "
        summary += (
            f"center={center.tolist()} mm radius={radius_mm:.3f} mm "
            f"support_thr={threshold_frac:.3f} points={points.shape[0]}"
        )
        if want_mask:
            mask = _sphere_mask(shape, x, y, z, center, radius_mm, np)
    else:
        raise ValueError(f"Unknown roi.type {roi_type!r}")

    if want_mask and not bool(mask.any()):
        raise ValueError("ROI mask is empty for the loaded phantom")
    return {
        "center": mx.array(center, dtype=mx.float32),
        "radius_mm": radius_mm,
        "mask": mask,
        "points": mx.array(points, dtype=mx.float32) if points is not None else None,
        "weights": mx.array(weights, dtype=mx.float32) if weights is not None else None,
        "type": roi_type,
        "summary": summary,
    }


def _method_kwargs_with_roi(
    name: str, method_kwargs: dict | None, roi_ctx: dict, geometry: dict,
    *, optimize_roi: bool,
):
    kwargs = dict(method_kwargs or {})
    if not optimize_roi:
        return kwargs
    if roi_ctx.get("points") is not None and "roi_points" not in kwargs:
        kwargs["roi_points"] = roi_ctx["points"]
    if roi_ctx.get("weights") is not None and "roi_weights" not in kwargs:
        kwargs["roi_weights"] = roi_ctx["weights"]
    if (
        "bundle" not in name
        or roi_ctx.get("radius_mm") is None
        or "bundle_cfg" in kwargs
    ):
        return kwargs
    from differentiable_coverage.absorption_bundle import BundleAbsorptionConfig

    kwargs["bundle_cfg"] = BundleAbsorptionConfig(
        roi_radius=float(roi_ctx["radius_mm"]),
        n_rays_u=1 if name.endswith("_center") else 5,
        n_rays_v=1 if name.endswith("_center") else 9,
        n_samples=32,
        voxel_spacing=float(geometry["voxel_pitch"]),
    )
    return kwargs


# ----------------------------------------------------------------------
# Per-step trace and surrogate term columns
# ----------------------------------------------------------------------

# Set from ``--out`` in :func:`main` when ``DIFFCT_TRACE`` is truthy, so the
# per-step trace does not depend on the operator remembering a path.  Left at
# ``None`` otherwise, which reproduces the env-var-only behaviour exactly.
_TRACE_DIR_DEFAULT: str | None = None


def _resolve_trace_dir(trace_tag: str) -> str | None:
    """Directory for this cell's per-step trace, or ``None`` to skip it.

    ``DIFFCT_TRACE_DIR`` keeps its existing meaning and still wins;
    ``DIFFCT_TRACE=1`` derives ``<output>.trace/`` from ``--out`` instead;
    ``DIFFCT_TRACE_MATCH`` filters by tag exactly as before.
    """
    trace_dir = os.environ.get("DIFFCT_TRACE_DIR") or _TRACE_DIR_DEFAULT
    trace_match = os.environ.get("DIFFCT_TRACE_MATCH")
    if trace_dir and trace_match and trace_match not in trace_tag:
        return None
    return trace_dir


def _final_term_columns(terms: dict) -> dict:
    """Surrogate terms of the selection that produced a row, as columns.

    Prefixed with ``final_`` and appended by :func:`_write_csv` after every
    existing column, so a config's established columns keep their names,
    order and values.  A selector that runs no continuous objective returns
    an empty dict and its rows gain no columns at all.
    """
    return {f"final_{name}": value for name, value in terms.items()}


# ----------------------------------------------------------------------
# Per-cell evaluation
# ----------------------------------------------------------------------

def _compute_metrics(vol, recon, peak: float, metrics: list, stack: dict,
                     roi_mask=None) -> dict:
    out = {}
    for m in metrics:
        f = stack[m]
        kwargs = {}
        if m in ("psnr", "roi_psnr"):
            kwargs["peak"] = peak
        if m.startswith("roi_") and roi_mask is not None:
            kwargs["mask"] = roi_mask
        out[m] = float(f(recon, vol, **kwargs))
    return out


def _eval_cell(
    *, name: str, k: int, vol_gt, vol_prior, vcl_pre, geometry: dict,
    sart_iter: int, k_max: int, metrics: list, stack: dict,
    cfg: dict,
    seed: int = 0,
    roi_ctx: dict | None = None,
    method_kwargs: dict | None = None,
) -> dict:
    """Run one (method, k) cell: select, simulate, reconstruct, score.

    *vol_prior* is fed to the selector (bundle / VCL terms see it);
    *vol_gt* drives sinogram simulation and metric computation.  For
    configs without R5 prior degradation the two are the same array.
    """
    mx = stack["mx"]
    roi_ctx = roi_ctx or {
        "center": mx.zeros(3, dtype=mx.float32),
        "radius_mm": None,
        "mask": None,
        "points": None,
        "weights": None,
        "type": "none",
    }

    trace_tag = f"{name}_k{k}_seed{seed}_kmax{k_max}"
    trace_dir = _resolve_trace_dir(trace_tag)

    optimize_roi = _method_optimizes_roi(name)
    # A method may override the candidate-pool size via ``kwargs.n_candidates``.
    # This lets the dense VCLS baseline keep the full K_max pool while our
    # continuous methods select from a coarser pool (they refine off-grid, so
    # they do not need the dense candidate set the discrete swap search does).
    method_kwargs = dict(method_kwargs or {})
    n_cand = int(method_kwargs.pop("n_candidates", k_max))
    method_kwargs = _method_kwargs_with_roi(
        name, method_kwargs, roi_ctx, geometry,
        optimize_roi=optimize_roi,
    )
    selector_roi_center = _method_roi_center(name, roi_ctx, stack)
    stack["reset_optim_terms"]()
    t = time.time()
    src = stack["build_baseline_sources"](
        name, k, geometry["sid"],
        roi_center=selector_roi_center,
        vcl_precompute=vcl_pre, volume=vol_prior, sdd=geometry["sdd"],
        detector_shape=(geometry["det_voxels"], geometry["det_voxels"]),
        du=geometry["det_pitch"], dv=geometry["det_pitch"],
        voxel_spacing=geometry["voxel_pitch"], n_candidates=n_cand,
        prefer_sparse_backprojection=_prefer_sparse_backprojection(cfg),
        seed=seed, trace_dir=trace_dir, trace_tag=trace_tag,
        method_kwargs=method_kwargs,
    )
    mx.eval(src)
    t_sel = time.time() - t

    sp, dc, du, dv = stack["geometry_from_sources"](
        src, sid=geometry["sid"], sdd=geometry["sdd"]
    )
    sino = stack["simulate_sinogram"](
        vol_gt, sp, dc, du, dv,
        det_u=geometry["det_voxels"], det_v=geometry["det_voxels"],
        du=geometry["det_pitch"], dv=geometry["det_pitch"],
        voxel_spacing=geometry["voxel_pitch"],
    )
    mx.eval(sino)

    t = time.time()
    res = stack["reconstruct_sart_volume"](
        vol_gt.shape, sino, sp, dc, du, dv,
        du=geometry["det_pitch"], dv=geometry["det_pitch"],
        voxel_spacing=geometry["voxel_pitch"],
        iteration_count=sart_iter, show_progress=False,
    )
    mx.eval(res.reconstruction)
    t_rec = time.time() - t

    peak = float(vol_gt.max())
    m = _compute_metrics(
        vol_gt, res.reconstruction, peak, metrics, stack,
        roi_mask=roi_ctx.get("mask"),
    )
    roi_center_np = stack["np"].asarray(roi_ctx["center"], dtype=stack["np"].float32)
    m.update({
        "method": name,
        "k": k,
        "sel_s": t_sel,
        "rec_s": t_rec,
        "roi_type": roi_ctx.get("type", "none"),
        "roi_center_x_mm": float(roi_center_np[0]),
        "roi_center_y_mm": float(roi_center_np[1]),
        "roi_center_z_mm": float(roi_center_np[2]),
        "roi_radius_mm": (
            float(roi_ctx["radius_mm"])
            if roi_ctx.get("radius_mm") is not None else ""
        ),
        "roi_role": "optimization+evaluation" if optimize_roi else "evaluation_only",
    })
    # Surrogate values of the returned iterate, read back from the selector.
    # They come from the evaluation the optimiser had already made, so this
    # costs no objective call and cannot move any number above.
    m.update(_final_term_columns(stack["last_optim_terms"]()))
    return m


# ----------------------------------------------------------------------
# Experiment runners (one per experiment_type)
# ----------------------------------------------------------------------

# Methods that read the discrete K_max x K_max (R, gamma) cache.  These
# are the VCLS warm-start variants (init_method == "vcls"), which seed
# Adam from the discrete swap-search pick.  The greedy_* / uniform_*
# variants initialise from Tuy-greedy or a uniform sphere and use only
# the cheap per-step k x k continuous VCL context, never the discrete
# cache, so they are deliberately absent.  This single set is the source
# of truth for both the cache build and the per-method cost attribution.
_VCL_CACHE_METHODS = frozenset({
    "vcls", "vcls_adam_vcl", "vcls_adam_geo",
    "vcls_adam_bundle_center", "vcls_adam_bundle",
    "vcls_adam_bundle_fd", "vcls_cmaes_bundle",
    "vcls_adam_vcl_two_axis", "vcls_adam_bundle_two_axis",
    "vcls_adam_vcl_carm", "vcls_adam_bundle_carm",
    "vcls_adam_icov_fft",
    "vcls_adam_oed",
})


def _method_needs_vcl_cache(method: dict) -> bool:
    """True iff this method reads the discrete (R, gamma) cache.

    Dropping the cache for the greedy_* / uniform_* methods removes a
    wasted O(K_max^2) precompute with bit-identical results, which is
    what gives the cold-start bundle path its wall-clock advantage.
    """
    return (
        method["name"] in _VCL_CACHE_METHODS
        or method.get("base_kwargs", {}).get("init_method") == "vcls"
    )


def _prefer_sparse_backprojection(cfg: dict) -> bool:
    """Config switch for sparse/sample-only backprojection in VCL code paths."""
    vcl_cfg = cfg.get("vcl", {})
    return bool(vcl_cfg.get("prefer_sparse_backprojection", True))


def _build_vcl_cache_if_needed(cfg, vol, geometry, stack, seed: int = 0):
    """The discrete VCLS baselines need a (R, gamma) cache.  Also build
    the cache when any method's ``base_kwargs.init_method == "vcls"``,
    so that parameter sweeps don't recompute it for every step.

    If a precomputed cache for the (phantom, k_max, seed) combination
    exists in ``data/cache/``, it is loaded from disk instead of being
    rebuilt; see ``experiments/precompute_caches.py``.

    Set ``force_rebuild_cache: true`` in the config to bypass the disk
    fast-path and always build the matrix from scratch.  This is what the
    cost benchmark uses, so that ``cache_s`` reflects the true per-object
    precompute that VCLS must pay on every new scan, rather than a
    millisecond disk load of a memoised result.
    """
    if not any(_method_needs_vcl_cache(m) for m in cfg["methods"]):
        return None

    # Try disk-cache fast path (skipped when force_rebuild_cache is set).
    phantom_tag = _phantom_tag_for_cache(cfg)
    if phantom_tag is not None and not cfg.get("force_rebuild_cache", False):
        try:
            from experiments.precompute_caches import load_vcl_cache
            cached = load_vcl_cache(phantom_tag, cfg["k_max"], seed)
            if cached is not None:
                print(f"    using disk cache for {phantom_tag} seed={seed}",
                      flush=True)
                return cached
        except Exception as e:
            print(f"    disk-cache load failed ({e}); rebuilding", flush=True)

    candidates = stack["sample_unit_sphere"](cfg["k_max"], seed=seed) * geometry["sid"]
    return stack["compute_R_gamma"](
        vol, candidates, sid=geometry["sid"], sdd=geometry["sdd"],
        det_shape=(geometry["det_voxels"], geometry["det_voxels"]),
        du=geometry["det_pitch"], dv=geometry["det_pitch"],
        voxel_spacing=geometry["voxel_pitch"],
        r1=cfg.get("voxel_subsample_r1", 1e-3), seed=seed,
        prefer_sparse_backprojection=_prefer_sparse_backprojection(cfg),
    )


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _normalize_key_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def _row_key(row: dict[str, Any], cols: list[str]) -> tuple[str, ...]:
    return tuple(_normalize_key_value(row.get(c, "")) for c in cols)


def _load_reusable_rows(cfg: dict, out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_read_csv_rows(out))
    for entry in cfg.get("reuse_from", []):
        path = Path(entry["path"])
        defaults = dict(entry.get("defaults", {}))
        for row in _read_csv_rows(path):
            merged = dict(defaults)
            merged.update(row)
            rows.append(merged)
    return rows


def _phantom_tag_for_cache(cfg: dict) -> str | None:
    """Map a YAML phantom spec to the tag used in
    ``experiments.precompute_caches.PHANTOM_REGISTRY``."""
    spec = cfg.get("phantom", {})
    p_type = spec.get("type")
    path = str(spec.get("path", ""))
    res = int(spec.get("resolution", 0))
    if p_type == "milp_npy":
        if "synthetic_metal_dataset" in path:
            if "mild" in path and res == 256:     return "synthetic_metal_mild"
            if "moderate" in path and res == 256: return "synthetic_metal_moderate"
            if "hard" in path and res == 256:     return "synthetic_metal_hard"
        if "mild" in path and res == 384:     return "milp_mild"
        if "moderate" in path and res == 384: return "milp_moderate"
        if "mild" in path and res == 192:     return "milp_mild_192"
        return None
    if p_type == "ornl_nozzle":
        if res == 512: return "ornl"
        return None
    return None


def _run_method_grid(cfg: dict, out: Path, stack: dict,
                     seeds: list[int] | None = None) -> None:
    """Standard experiment: one phantom, one geometry, list of methods,
    each method evaluated at every k in `k_values`.  Writes one CSV row
    per (method, k, seed) cell.  Reproduces Tables I and II of the paper.

    If ``seeds`` is provided, the cache is built once and the method
    grid is repeated for each seed; the seed is added as a column in
    the output CSV.  This is the efficient path for review-response
    multi-seed runs.
    """
    mx = stack["mx"]
    print(f"[{cfg['name']}] loading phantom {cfg['phantom']['path']} ...",
          flush=True)
    vol_gt, vol_prior = _load_phantom_pair(cfg["phantom"], stack)
    mx.eval(vol_gt); mx.eval(vol_prior)
    if vol_gt is not vol_prior:
        print(f"  R5 degradation active; "
              f"sigma_pct={cfg['phantom'].get('noise_sigma_pct', 0.0)} "
              f"blur_vox={cfg['phantom'].get('blur_voxels', 0.0)}",
              flush=True)
    print(f"  shape={vol_gt.shape} peak={float(vol_gt.max()):.3f}", flush=True)

    rows: list[dict] = []
    metrics = cfg.get("metrics", ["psnr", "ssim", "nrmse", "hfen"])
    geometry = _resolve_geometry(
        cfg["geometry"], int(cfg["phantom"]["resolution"]),
    )
    roi_ctx = _resolve_roi_context(
        cfg, vol_gt, geometry, stack,
        want_mask=any(m.startswith("roi_") for m in metrics),
    )
    print(f"  geometry: sid={geometry['sid']} sdd={geometry['sdd']} "
          f"voxel_pitch={geometry['voxel_pitch']:.3f} mm", flush=True)
    print(f"  {roi_ctx['summary']}", flush=True)

    seed_list = seeds if seeds is not None else [cfg.get("seed", 0)]
    # The (R, gamma) cache depends on the seeded candidate rotation, so
    # we rebuild (or reload from disk) per seed.  Loading from disk is
    # millisecond-fast when the cache file exists.
    for seed in seed_list:
        print(f"\n=== seed = {seed} ===", flush=True)
        print(f"  acquiring VCL cache K_max={cfg['k_max']} (seed={seed}) ...",
              flush=True)
        t0 = time.time()
        vcl_pre = _build_vcl_cache_if_needed(
            cfg, vol_prior, geometry, stack, seed=seed,
        )
        # Shared (R, gamma) build time for this seed.  It is charged only
        # to the methods that actually read the cache (the VCLS warm-start
        # variants); cold-start greedy_* / uniform_* methods get cache_s=0
        # so the cost table reflects their true precompute-free wall-clock.
        cache_s = time.time() - t0
        print(f"    done ({cache_s:.1f}s)", flush=True)
        for method in cfg["methods"]:
            for k in cfg["k_values"]:
                tag = f"{method['name']} k={k} seed={seed}"
                print(f"\n[{tag}] ...", flush=True)
                row = _eval_cell(
                    name=method["name"], k=k,
                    vol_gt=vol_gt, vol_prior=vol_prior, vcl_pre=vcl_pre,
                    geometry=geometry, sart_iter=cfg["sart_iterations"],
                    k_max=cfg["k_max"], metrics=metrics, stack=stack, cfg=cfg,
                    seed=seed,
                    roi_ctx=roi_ctx,
                    method_kwargs=method.get("kwargs"),
                )
                row["seed"] = seed
                row["cache_s"] = cache_s if _method_needs_vcl_cache(method) else 0.0
                row["total_s"] = row["cache_s"] + row["sel_s"] + row["rec_s"]
                print(f"  PSNR={row['psnr']:.3f}  SSIM={row['ssim']:.4f}  "
                      f"NRMSE={row['nrmse']:.4f}  HFEN={row['hfen']:.2f}",
                      flush=True)
                rows.append(row)

    _write_csv(rows, out, metrics, extra_cols=["seed"])


def _run_param_sweep(cfg: dict, out: Path, stack: dict,
                     seeds: list[int] | None = None) -> None:
    """Sweep one hyperparameter (e.g. learning rate, Langevin temperature)
    on a single (phantom, method, k) configuration.  Used for Tables IV,
    V and Figure 3.
    """
    mx = stack["mx"]
    vol_gt, vol_prior = _load_phantom_pair(cfg["phantom"], stack)
    mx.eval(vol_gt); mx.eval(vol_prior)
    metrics = cfg.get("metrics", ["psnr", "ssim", "nrmse", "hfen"])
    geometry = _resolve_geometry(
        cfg["geometry"], int(cfg["phantom"]["resolution"]),
    )
    roi_ctx = _resolve_roi_context(
        cfg, vol_gt, geometry, stack,
        want_mask=any(m.startswith("roi_") for m in metrics),
    )

    # The sweep always evaluates the same method; only one entry in the
    # methods list is allowed for a parameter sweep.
    if len(cfg["methods"]) != 1:
        raise ValueError("parameter sweep needs exactly one method entry")
    method = cfg["methods"][0]
    k = int(cfg["k_values"][0])

    print(f"[{cfg['name']}] phantom shape={vol_gt.shape}", flush=True)
    print(f"  {roi_ctx['summary']}", flush=True)
    # ``--seeds`` repeats the whole sweep once per seed.  Without it the
    # selector is called exactly as before, so the archived single-seed
    # sweep CSVs reproduce byte for byte.
    seed_list = seeds if seeds is not None else [None]

    sweep = cfg["sweep"]
    param_name = sweep["param"]
    param_values = sweep["values"]
    extra_kwargs = method.get("kwargs", {})

    rows: list[dict] = []

    from differentiable_coverage.absorption_bundle import (
        BundleAbsorptionConfig, bundle_path_integral,
        calibrate_bundle_weight,
    )
    from differentiable_coverage.eval.trajectories import (
        greedy_adam_vcl_continuous,
    )

    # Bundle-related sweep parameters translate into BundleAbsorptionConfig
    # overrides plus an auto-calibrated lambda_bundle, rather than into a
    # direct kwarg of greedy_adam_vcl_continuous.
    BUNDLE_CFG_PARAMS = {
        "bundle_roi_radius":  "roi_radius",
        "bundle_n_rays_u":    "n_rays_u",
        "bundle_n_rays_v":    "n_rays_v",
        "bundle_n_samples":   "n_samples",
    }

    base_kwargs = method.get("base_kwargs", {})
    base_kwargs_bundle = base_kwargs.get("lambda_bundle", None) is not None
    sweep_is_bundle_geom = param_name in BUNDLE_CFG_PARAMS
    sweep_is_lambda_bundle = (param_name == "lambda_bundle")
    needs_bundle = (
        sweep_is_bundle_geom or sweep_is_lambda_bundle or base_kwargs_bundle
    )

    # When sweeping K_max (=n_candidates) itself, the (R, gamma) cache
    # must be rebuilt per value because the matrix size changes.  For
    # all other sweeps we build the cache once with the cfg default.
    sweep_is_kmax = (param_name == "n_candidates")

    for run_seed in seed_list:
        sweep_seed = (
            int(cfg.get("seed", 0)) if run_seed is None else int(run_seed)
        )
        if run_seed is not None:
            print(f"\n=== seed = {sweep_seed} ===", flush=True)
        if sweep_is_kmax:
            print(f"  K_max sweep: cache rebuilt per value "
                  f"(seed={sweep_seed})", flush=True)
            vcl_pre = None
        else:
            print(f"  acquiring VCL cache (seed={sweep_seed}) ...",
                  flush=True)
            vcl_pre = _build_vcl_cache_if_needed(
                cfg, vol_prior, geometry, stack, seed=sweep_seed,
            )

        for v in param_values:
            # Build a per-iteration kwargs dict, plus optionally an
            # explicit bundle_cfg derived from the sweep value.
            all_kwargs = {**base_kwargs, **extra_kwargs}
            bcfg = None

            # For the K_max sweep we override cfg["k_max"] per value and
            # (re)build/(re)load the matching (R, gamma) cache.
            if sweep_is_kmax:
                cfg["k_max"] = int(v)
                print(f"  loading/building VCL cache for K_max={v} "
                      f"(seed={sweep_seed}) ...", flush=True)
                vcl_pre = _build_vcl_cache_if_needed(
                    cfg, vol_prior, geometry, stack, seed=sweep_seed,
                )
            if needs_bundle:
                # Start from the paper default and apply the sweep override
                # if it touches a bundle geometric parameter.
                bcfg_kwargs = {
                    "voxel_spacing": geometry["voxel_pitch"],
                    "roi_radius": (
                        float(roi_ctx["radius_mm"])
                        if roi_ctx.get("radius_mm") is not None else 5.0
                    ),
                    "n_rays_u": 5, "n_rays_v": 9,
                    "n_samples": 32,
                }
                if sweep_is_bundle_geom:
                    bcfg_kwargs[BUNDLE_CFG_PARAMS[param_name]] = int(v) if (
                        param_name != "bundle_roi_radius"
                    ) else float(v)
                bcfg = BundleAbsorptionConfig(**bcfg_kwargs)
                # Decide lambda_bundle: explicit sweep value wins, else
                # auto-calibrate to 0.2 / median(tau) at the median sphere
                # direction.
                if sweep_is_lambda_bundle:
                    all_kwargs["lambda_bundle"] = float(v)
                else:
                    probes = stack["sample_unit_sphere"](128) * geometry["sid"]
                    med_tau = float(mx.median(bundle_path_integral(
                        probes, roi_ctx["center"], vol_prior, bcfg,
                    )))
                    all_kwargs["lambda_bundle"] = calibrate_bundle_weight(med_tau)
                all_kwargs["bundle_cfg"] = bcfg
            # Non-bundle sweep parameter: forward directly (except
            # n_candidates, which we pass explicitly below to avoid a
            # duplicate-kwarg conflict).
            if (not sweep_is_bundle_geom and not sweep_is_lambda_bundle
                    and not sweep_is_kmax):
                all_kwargs[param_name] = v
            if run_seed is not None:
                # Only when --seeds is given, so the default path still calls
                # the selector with its own defaults.  One seed drives both
                # the initialisation and the Langevin stream, as in
                # build_baseline_sources, unless the config pins them.
                all_kwargs.setdefault("seed", int(run_seed))
                all_kwargs.setdefault("noise_seed", int(run_seed))
            optimize_roi = _method_optimizes_roi(method["name"])
            all_kwargs = _method_kwargs_with_roi(
                method["name"], all_kwargs, roi_ctx, geometry,
                optimize_roi=optimize_roi,
            )
            selector_roi_center = _method_roi_center(method["name"], roi_ctx, stack)

            n_cand = int(v) if sweep_is_kmax else cfg["k_max"]
            trace_tag = (
                f"{method['name']}_{param_name}{v}_k{k}_seed{sweep_seed}"
            )
            trace_dir = _resolve_trace_dir(trace_tag)
            print(f"\n[{method['name']} {param_name}={v}] ...", flush=True)
            stack["reset_optim_terms"]()
            t0 = time.time()
            src = greedy_adam_vcl_continuous(
                k, sid=geometry["sid"], sdd=geometry["sdd"],
                roi_center=selector_roi_center,
                volume=vol_prior,
                detector_shape=(geometry["det_voxels"], geometry["det_voxels"]),
                du=geometry["det_pitch"], dv=geometry["det_pitch"],
                voxel_spacing=geometry["voxel_pitch"],
                n_candidates=n_cand, vcls_precompute=vcl_pre,
                trace_dir=trace_dir, trace_tag=trace_tag,
                **all_kwargs,
            )
            mx.eval(src)
            t_sel = time.time() - t0

            sp, dc, du, dv = stack["geometry_from_sources"](
                src, sid=geometry["sid"], sdd=geometry["sdd"]
            )
            sino = stack["simulate_sinogram"](
                vol_gt, sp, dc, du, dv,
                det_u=geometry["det_voxels"], det_v=geometry["det_voxels"],
                du=geometry["det_pitch"], dv=geometry["det_pitch"],
                voxel_spacing=geometry["voxel_pitch"],
            )
            mx.eval(sino)
            res = stack["reconstruct_sart_volume"](
                vol_gt.shape, sino, sp, dc, du, dv,
                du=geometry["det_pitch"], dv=geometry["det_pitch"],
                voxel_spacing=geometry["voxel_pitch"],
                iteration_count=cfg["sart_iterations"], show_progress=False,
            )
            mx.eval(res.reconstruction)

            peak = float(vol_gt.max())
            m = _compute_metrics(
                vol_gt, res.reconstruction, peak, metrics, stack,
                roi_mask=roi_ctx.get("mask"),
            )
            m.update({param_name: v, "sel_s": t_sel})
            if run_seed is not None:
                m["seed"] = int(run_seed)
            m.update(_final_term_columns(stack["last_optim_terms"]()))
            print(f"  PSNR={m['psnr']:.3f}  HFEN={m['hfen']:.2f}", flush=True)
            rows.append(m)

    _write_csv(rows, out, metrics, extra_cols=[param_name, "seed"])


# ----------------------------------------------------------------------
# CSV writer
# ----------------------------------------------------------------------

def _write_csv(rows: list[dict], path: Path, metrics: list,
               extra_cols: list | None = None,
               quiet: bool = False) -> None:
    if not rows:
        return
    extra_cols = extra_cols or []
    term_cols = _collect_term_cols(rows)
    fields = (
        ["method", "k"] + metrics + ["cache_s", "sel_s", "rec_s", "total_s"]
        + [
            "roi_type", "roi_center_x_mm", "roi_center_y_mm",
            "roi_center_z_mm", "roi_radius_mm", "roi_role",
        ]
        + extra_cols
        # Appended last so every established column keeps its position.
        + term_cols
    )
    # Keep only fields that are present.
    fields = [f for f in fields if any(f in r for r in rows)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    if not quiet:
        print(f"\nWrote {path}")
    _write_terms_csv(rows, path, metrics, extra_cols, term_cols, quiet=quiet)


def _collect_term_cols(rows: list[dict]) -> list[str]:
    """The ``final_*`` surrogate columns present in *rows*, first seen first."""
    term_cols: list[str] = []
    for r in rows:
        for name in r:
            if name.startswith("final_") and name not in term_cols:
                term_cols.append(name)
    return term_cols


def _write_terms_csv(rows: list[dict], path: Path, metrics: list,
                     extra_cols: list, term_cols: list,
                     quiet: bool = False) -> None:
    """Companion ``<output>.terms.csv`` pairing surrogate with image quality.

    Same rows as the result CSV, restricted to the identity columns, the
    image-quality metrics and the ``final_*`` terms, so a surrogate-against-
    quality scatter can be built from any run without re-deriving the result
    CSV's per-config column set.  Written only when at least one method
    evaluated a continuous objective.
    """
    if not term_cols:
        return
    fields = ["method", "k"] + list(extra_cols) + list(metrics) + term_cols
    fields = [f for f in fields if any(f in r for r in rows)]
    terms_path = path.with_suffix(".terms.csv")
    with terms_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    if not quiet:
        print(f"Wrote {terms_path}")


def _run_kmax_method_grid(cfg: dict, out: Path, stack: dict,
                          seeds: list[int] | None = None) -> None:
    """Method grid over an explicit list of K_max values.

    Each cell is keyed by (method, k, seed, n_candidates). Existing rows
    in *out* or configured ``reuse_from`` CSVs are reused and skipped.
    The selector cache build time is recorded separately as ``cache_s``,
    and ``total_s`` is defined as ``cache_s + sel_s + rec_s``.
    """
    mx = stack["mx"]
    print(f"[{cfg['name']}] loading phantom {cfg['phantom']['path']} ...",
          flush=True)
    vol_gt, vol_prior = _load_phantom_pair(cfg["phantom"], stack)
    mx.eval(vol_gt); mx.eval(vol_prior)
    print(f"  shape={vol_gt.shape} peak={float(vol_gt.max()):.3f}", flush=True)

    geometry = _resolve_geometry(
        cfg["geometry"], int(cfg["phantom"]["resolution"]),
    )
    metrics = cfg.get("metrics", ["psnr", "ssim", "nrmse", "hfen"])
    roi_ctx = _resolve_roi_context(
        cfg, vol_gt, geometry, stack,
        want_mask=any(m.startswith("roi_") for m in metrics),
    )
    print(f"  geometry: sid={geometry['sid']} sdd={geometry['sdd']} "
          f"voxel_pitch={geometry['voxel_pitch']:.3f} mm", flush=True)
    print(f"  {roi_ctx['summary']}", flush=True)
    rows = _read_csv_rows(out)
    key_cols = ["method", "k", "seed", "n_candidates"]
    existing_keys = {_row_key(r, key_cols) for r in rows}

    seed_list = seeds if seeds is not None else [cfg.get("seed", 0)]
    kmax_values = [int(v) for v in cfg["kmax_values"]]

    for seed in seed_list:
        print(f"\n=== seed = {seed} ===", flush=True)
        for kmax in kmax_values:
            print(f"\n=== K_max = {kmax} ===", flush=True)
            cache_rows = []
            for method in cfg["methods"]:
                for k in cfg["k_values"]:
                    key = _row_key(
                        {
                            "method": method["name"],
                            "k": k,
                            "seed": seed,
                            "n_candidates": kmax,
                        },
                        key_cols,
                    )
                    if key not in existing_keys:
                        cache_rows.append((method, k, key))
            if not cache_rows:
                print("  all cells already available; skipping", flush=True)
                continue

            cfg_k = dict(cfg)
            cfg_k["k_max"] = kmax
            print(f"  acquiring VCL cache K_max={kmax} (seed={seed}) ...",
                  flush=True)
            t0 = time.time()
            vcl_pre = _build_vcl_cache_if_needed(
                cfg_k, vol_prior, geometry, stack, seed=seed,
            )
            cache_s = time.time() - t0
            print(f"    done ({cache_s:.1f}s)", flush=True)

            for method, k, key in cache_rows:
                tag = f"{method['name']} k={k} seed={seed} K_max={kmax}"
                print(f"\n[{tag}] ...", flush=True)
                row = _eval_cell(
                    name=method["name"], k=k,
                    vol_gt=vol_gt, vol_prior=vol_prior, vcl_pre=vcl_pre,
                    geometry=geometry, sart_iter=cfg["sart_iterations"],
                    k_max=kmax, metrics=metrics, stack=stack, cfg=cfg_k,
                    seed=seed,
                    roi_ctx=roi_ctx,
                    method_kwargs=method.get("kwargs"),
                )
                row["seed"] = seed
                row["n_candidates"] = kmax
                row["cache_s"] = cache_s
                row["total_s"] = cache_s + row["sel_s"] + row["rec_s"]
                print(f"  PSNR={row['psnr']:.3f}  SSIM={row['ssim']:.4f}  "
                      f"NRMSE={row['nrmse']:.4f}  HFEN={row['hfen']:.2f}",
                      flush=True)
                rows.append(row)
                existing_keys.add(key)
                _write_csv(
                    rows, out, metrics,
                    extra_cols=["seed", "n_candidates"], quiet=True,
                )

    _write_csv(rows, out, metrics, extra_cols=["seed", "n_candidates"])


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Reproduce one paper experiment from a YAML config."
    )
    p.add_argument("--config", required=True,
                   help="Path to an experiment YAML config.")
    p.add_argument("--out", default=None,
                   help="Output CSV path; defaults to the config 'output' field.")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated list of seeds; overrides config.seed "
                        "and runs the method grid (or the parameter sweep) "
                        "once per seed.")
    args = p.parse_args(argv)
    seeds = None
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text())

    out = Path(args.out or cfg.get("output", f"results/{cfg_path.stem}.csv"))

    global _TRACE_DIR_DEFAULT
    if os.environ.get("DIFFCT_TRACE", "").lower() not in ("", "0", "false", "no"):
        _TRACE_DIR_DEFAULT = str(out.with_suffix(".trace"))
        print(f"per-step optimisation traces -> {_TRACE_DIR_DEFAULT}")

    stack = _load_mlx_stack()

    exp_type = cfg.get("experiment_type", "method_grid")
    if exp_type == "method_grid":
        _run_method_grid(cfg, out, stack, seeds=seeds)
    elif exp_type == "kmax_method_grid":
        _run_kmax_method_grid(cfg, out, stack, seeds=seeds)
    elif exp_type == "param_sweep":
        _run_param_sweep(cfg, out, stack, seeds=seeds)
    elif exp_type == "figure_gradient_field":
        # Delegate to the dedicated script.  The script consumes the
        # same YAML phantom path; everything else is hard-coded in the
        # figure source.
        os.environ.setdefault(
            "GRADIENT_FIELD_PHANTOM", str(cfg["phantom"]["path"])
        )
        from differentiable_coverage.figures import gradient_field as gf
        gf.main()
    elif exp_type == "figure_milp_slices":
        raise SystemExit(
            "use experiments/render_milp_slices.py for this config"
        )
    else:
        raise ValueError(f"Unknown experiment_type {exp_type!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
