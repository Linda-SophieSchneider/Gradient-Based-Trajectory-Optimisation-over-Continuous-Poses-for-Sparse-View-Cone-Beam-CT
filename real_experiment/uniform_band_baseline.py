"""Assemble a REAL "uniform-on-band" baseline from already-acquired projections.

Motivation (user, 2026-07-21): the circular-subset baseline is not on the
same kinematic manifold as the planned trajectories (it sits at phi=0, using
none of the +-30deg elevation freedom) AND shares the reference scan's own
session (no registration, no geometry-frame reconstruction penalty) -- an
unfair comparison in two directions at once. The fair baseline is a
trajectory UNIFORMLY sampled on the SAME two-axis band manifold, so the only
remaining variable is the selection algorithm (uniform vs. our optimiser).

Rather than simulate this (which would reintroduce the sim-to-real gap this
whole section exists to close), we ASSEMBLE it from real measured
projections already on disk: pool every view acquired on the band manifold
across the whole project (bundle k=100/400 + all3 k=100/400, ~998 views
total), generate a Fibonacci-uniform target set of k directions restricted
to |phi|<=30deg, and optimally match each target direction to the nearest
unused real view (linear_sum_assignment on angular distance). The result is
a genuinely measured, on-manifold, selection-only baseline.

Reconstruction: each selected view keeps its own origin scan's registered
pose (shared yaw/roll + per-scan T from the geometry-frame registration of
2026-07-21; see geoframe_*_pose.json / project memory) so the mixed-origin
geometry lands in the reference frame with zero volume interpolation,
exactly like the planned arms. Same canonical ASD-POCS (outer=50) + FOV mask
protocol, same LS intensity match + four metrics against the FDK-1200
reference.

Output: uniform_band_N0100/, uniform_band_N0400/ (reconstruction.rek +
slice PNG), printed per-scan view-count breakdown and metrics.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Differentiable-Coverage"))

import reconstruct_measured_cuda as rec                   # noqa: E402
import diffct_mlx as dct                                  # noqa: E402
from diffct_mlx.backend import active as _b               # noqa: E402
from diffct_mlx.reconstruction_algorithms.cases import (  # noqa: E402
    _build_sensitivity_support_mask,
    _build_leap_style_circular_fov_mask,
)
from scanner_io.rek2py import rek2py                      # noqa: E402
from differentiable_coverage.eval import metrics as M     # noqa: E402
from sart_circular_baseline import (                      # noqa: E402
    ASD_OUTER, ASD_REG_ITERS, ASD_ALPHA, ASD_BETA_RED, SART_RELAX,
)

xp = _b.xp
REFERENCE = HERE / "reference_reconstructions" / "output_circular1200_fdk_quant" / "reconstruction_FDK.rek"
PHI_MAX_DEG = 30.0

# Shared session pose (2026-07-21 geometry-frame registration; corr 0.986).
YAW, ROLL = -89.75, -0.50
# Per-scan translation refinement (mm), all within ~0.5mm of each other.
SCAN_T = {
    "bundle_N0100": np.array([-6.16, -2.83, 13.33]),
    "bundle_N0400": np.array([-6.66, -2.83, 13.33]),
    "all3_N0100": np.array([-6.1648, -2.3324, 13.8296]),
    "all3_N0400": np.array([-6.16, -2.83, 13.83]),
}
SCAN_DIRS = {
    "bundle_N0100": "bundle_2026_07_17_N0100",
    "bundle_N0400": "bundle_2026_07_17_N0400",
    "all3_N0100": "all3_2026_07_17_N0100",
    "all3_N0400": "all3_2026_07_17_N0400",
}


def rot_z(deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rot_x(deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


R_SHARED = rot_x(ROLL) @ rot_z(YAW)


def load_pool():
    """Load geometry + log-normalised sinogram for all four measured scans."""
    pool = []  # list of dicts: scan, idx, theta, phi, src, det_c, det_u, det_v
    sinos = {}
    geoms = {}
    kamera_root = Path(os.environ.get(
        "KAMERA_DATA_DIR", "/ssd_data/diffct_scratch/TrajektorienOptimierung/Kamera"))
    for tag, dirname in SCAN_DIRS.items():
        data_dir = kamera_root / dirname
        sino_raw, geom_raw, meta = rec.load_measured_dataset(data_dir, rec.DETECTOR_BIN, 1)
        src, det_c, det_u_v, det_v_v, iso = rec.build_geometry(geom_raw)
        sino = rec.log_normalize(sino_raw)
        sinos[tag] = sino
        geoms[tag] = dict(src=src, det_c=det_c, det_u_v=det_u_v, det_v_v=det_v_v,
                          du=meta["du"], dv=meta["dv"])
        sid = np.mean(np.linalg.norm(src, axis=1))
        theta = np.arctan2(-src[:, 0], src[:, 1])
        phi = np.arcsin(np.clip(src[:, 2] / sid, -1, 1))
        for i in range(len(src)):
            pool.append(dict(scan=tag, idx=i, theta=theta[i], phi=phi[i]))
        print(f"  {tag}: {len(src)} views, phi range [{np.degrees(phi.min()):.1f}, "
              f"{np.degrees(phi.max()):.1f}] deg")
    return pool, sinos, geoms


def uniform_band_directions(k, phi_max_deg):
    """Fibonacci-lattice directions on the unit sphere, restricted to the band."""
    phi_max = np.radians(phi_max_deg)
    frac_in_band = np.sin(phi_max)  # fraction of sphere surface with |phi|<=phi_max
    n_total = int(np.ceil(k / frac_in_band * 1.15))
    while True:
        i = np.arange(n_total) + 0.5
        z = 1 - 2 * i / n_total          # uniform in z = sin(phi)
        phi = np.arcsin(z)
        golden = np.pi * (3 - np.sqrt(5))
        theta = golden * np.arange(n_total)
        mask = np.abs(phi) <= phi_max
        if mask.sum() >= k:
            phi, theta = phi[mask][:k], theta[mask][:k]
            break
        n_total = int(n_total * 1.3)
    x = np.cos(phi) * np.cos(theta)   # match TwoAxisGantry's (theta,phi)->dir convention
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    return np.stack([-x, y, z], axis=1)  # matches src = sid*(-cos phi sin th, cos phi cos th, sin phi)... see note below


def select_uniform_subset(pool, k):
    targets = uniform_band_directions(k, PHI_MAX_DEG)  # (k, 3) unit vectors
    real_dirs = np.array([[np.cos(p["phi"]) * (-np.sin(p["theta"])),
                           np.cos(p["phi"]) * np.cos(p["theta"]),
                           np.sin(p["phi"])] for p in pool])  # (n_pool, 3)
    cost = 1.0 - targets @ real_dirs.T   # (k, n_pool), lower = closer
    row, col = linear_sum_assignment(cost)
    chosen = [pool[c] for c in col]
    mean_ang = np.degrees(np.arccos(np.clip(1 - cost[row, col], -1, 1))).mean()
    print(f"  matched {k} targets, mean angular residual {mean_ang:.2f} deg")
    from collections import Counter
    print(f"  scan breakdown: {dict(Counter(c['scan'] for c in chosen))}")
    return chosen


def reconstruct(chosen, k, ref):
    n = len(chosen)
    src = np.empty((n, 3)); det_c = np.empty((n, 3))
    det_u_v = np.empty((n, 3)); det_v_v = np.empty((n, 3))
    du = dv = None
    rows = []
    for j, c in enumerate(chosen):
        g = GEOMS[c["scan"]]; s = SINOS[c["scan"]]
        T = SCAN_T[c["scan"]]
        src[j] = g["src"][c["idx"]] @ R_SHARED.T + T
        det_c[j] = g["det_c"][c["idx"]] @ R_SHARED.T + T
        det_u_v[j] = g["det_u_v"][c["idx"]] @ R_SHARED.T
        det_v_v[j] = g["det_v_v"][c["idx"]] @ R_SHARED.T
        rows.append(s[c["idx"]])
        du, dv = g["du"], g["dv"]
    sino = np.stack(rows, axis=0)

    shape = ref.shape
    voxel_mm = 0.2777
    det_u_count, det_v_count = sino.shape[1], sino.shape[2]
    fwd1, back1, back_all = dct.make_cone_3d_operators(
        xp.array(src), xp.array(det_c), xp.array(det_u_v), xp.array(det_v_v),
        volume_shape=shape, detector_shape=(det_u_count, det_v_count),
        du=du, dv=dv, voxel_spacing=voxel_mm, projector_mode="footprint")
    sino_d = xp.array(sino, dtype=_b.float32)
    views = [sino_d[i] for i in range(n)]

    sens = _build_sensitivity_support_mask(back_all, (n, det_u_count, det_v_count))
    fov = _build_leap_style_circular_fov_mask(shape, voxel_mm, (det_u_count, det_v_count),
                                              du, xp.array(src), xp.array(det_c))
    mask = xp.array(sens & fov)

    params = dct.SARTParameters(
        volume_shape=shape, iteration_count=ASD_OUTER, sart_iteration_count=1,
        normalized_sart_relaxation=SART_RELAX, enforce_positivity=True,
        shuffle_projection_order=True, projection_order_seed=0,
        volume_support_mask=mask, volume_support_mask_mode="always")
    reg = dct.ASDPOCSParameters(reg_iteration_count=ASD_REG_ITERS,
                                alpha=ASD_ALPHA, beta_red=ASD_BETA_RED)
    t0 = time.time()
    vol = dct.reconstruct_asd_pocs(views, fwd1, back1, params, reg, show_progress=True)
    v_np = np.asarray(_b.to_numpy(vol), np.float32)
    print(f"  ASD-POCS done in {time.time()-t0:.0f}s, range [{v_np.min():.4f}, {v_np.max():.4f}]")

    out = HERE / f"uniform_band_N{k:04d}"
    out.mkdir(exist_ok=True)
    rec.save_rek(v_np, out / "reconstruction.rek", voxel_mm * 1000.0)
    rec.save_slice_pngs(v_np, out / "reconstruction")

    scale = float((v_np * ref).sum() / (v_np * v_np).sum())
    v_s = v_np * scale
    print(f"  k={k}: LS={scale:.3f}  PSNR={M.psnr(v_s, ref):.2f}  "
          f"SSIM={M.ssim(v_s, ref):.4f}  NRMSE={M.nrmse(v_s, ref):.4f}  "
          f"HFEN={M.hfen(v_s, ref):.2f}", flush=True)


def main():
    import os
    global GEOMS, SINOS
    ks = tuple(int(x) for x in os.environ.get("UB_KS", "100,400").split(","))
    print("=== loading pool of all real band-manifold acquisitions ===")
    pool, SINOS, GEOMS = load_pool()
    print(f"total pool: {len(pool)} views")

    _, ref = rek2py(str(REFERENCE), switch_order=True)
    ref = np.asarray(ref, np.float32)

    for k in ks:
        print(f"\n=== uniform-on-band k={k} ===")
        chosen = select_uniform_subset(pool, k)
        reconstruct(chosen, k, ref)


if __name__ == "__main__":
    main()
