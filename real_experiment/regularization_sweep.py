"""Per-(arm, k) TV-strength sweep, selection by DATA RESIDUAL (not by
comparison to the reference).

Motivation (user, 2026-07-22): a single fixed ASD-POCS setting (outer=50,
TV 40 steps x alpha=0.3), tuned once on the all3-k=100 arm, was applied
unchanged to every arm/k. Two independent, untuned trajectories (bundle,
uniform-on-band) got WORSE from k=100 to k=400 despite their real angular
coverage/uniformity strictly improving with k (verified: uniform-on-band's
own coverage score goes 0.857 -> 0.999, mean NN gap 9.86deg -> 3.65deg, yet
PSNR drops 45.53 -> 45.15) -- the reconstruction protocol, not the
trajectory, is the moving part. all3, the one arm the settings were tuned
on, is the only one that behaves as expected. This sweeps TV alpha in
{0.15, 0.2, 0.3} at fixed outer=50 for every (arm, k), and selects the
alpha with the LOWEST fractional data residual ||Ax-b||/||b|| -- a
reference-free, Morozov-discrepancy-style criterion (never picks the alpha
that happens to score best against the ground truth, which would be
circular reasoning) -- then regenerates the final volume/metrics with the
winning alpha.

Arm types (each -> unified (src, det_c, det_u_v, det_v_v, sino, du, dv,
voxel_mm, shape)):
  circular  -- equidistant subset of circular_1200, native reference frame
  bundle/all3 -- single measured scan, geometry-frame registered (2026-07-21
                 pose: yaw=-89.75, roll=-0.5, per-scan T)
  uniform   -- real views pooled across all 4 measured scans, matched to a
               Fibonacci band-uniform target set (2026-07-21, see
               uniform_band_baseline.py), geometry-frame registered per view

Env: SWEEP_ARM in {circular,bundle,all3,uniform}, SWEEP_K in {100,400}.
Output: sweeps/sweep_<arm>_k<k>_a<alpha>.rek + printed residual/TV per alpha;
final winning volume copied to results_final/<arm>_final_k<k>/reconstruction.rek.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

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
from sart_circular_baseline import ASD_OUTER, ASD_REG_ITERS, ASD_BETA_RED, SART_RELAX  # noqa: E402
import uniform_band_baseline as ub                        # noqa: E402

xp = _b.xp
REFERENCE = HERE / "reference_reconstructions" / "output_circular1200_fdk_quant" / "reconstruction_FDK.rek"
VOXEL_MM = 0.2777
YAW, ROLL = -89.75, -0.50
SCAN_T = ub.SCAN_T
KAMERA = Path(os.environ.get(
    "KAMERA_DATA_DIR", "/ssd_data/diffct_scratch/TrajektorienOptimierung/Kamera"))
ALPHAS = tuple(float(a) for a in os.environ.get("SWEEP_ALPHAS", "0.15,0.20,0.30").split(","))


def rot_z(deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rot_x(deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


R_SHARED = rot_x(ROLL) @ rot_z(YAW)


def load_arm(arm, k, ref_shape):
    if arm == "circular":
        sino_raw, geom_raw, meta = rec.load_measured_dataset(
            KAMERA / "circular_1200",
            rec.DETECTOR_BIN, view_stride=1200 // k)
        src, det_c, det_u_v, det_v_v, iso = rec.build_geometry(geom_raw)
        sino = rec.log_normalize(sino_raw)
        du, dv = meta["du"], meta["dv"]
        return src, det_c, det_u_v, det_v_v, sino, du, dv

    if arm in ("bundle", "all3"):
        tag = f"{arm}_N0{k:03d}"
        data_dir = KAMERA / f"{arm}_2026_07_17_N0{k:03d}"
        sino_raw, geom_raw, meta = rec.load_measured_dataset(data_dir, rec.DETECTOR_BIN, 1)
        src, det_c, det_u_v, det_v_v, iso = rec.build_geometry(geom_raw)
        sino = rec.log_normalize(sino_raw)
        T = SCAN_T[tag]
        src = src @ R_SHARED.T + T
        det_c = det_c @ R_SHARED.T + T
        det_u_v = det_u_v @ R_SHARED.T
        det_v_v = det_v_v @ R_SHARED.T
        return src, det_c, det_u_v, det_v_v, sino, meta["du"], meta["dv"]

    if arm == "uniform":
        pool, sinos, geoms = ub.load_pool()
        chosen = ub.select_uniform_subset(pool, k)
        n = len(chosen)
        src = np.empty((n, 3)); det_c = np.empty((n, 3))
        det_u_v = np.empty((n, 3)); det_v_v = np.empty((n, 3))
        rows = []
        du = dv = None
        for j, c in enumerate(chosen):
            g = geoms[c["scan"]]; s = sinos[c["scan"]]
            T = SCAN_T[c["scan"]]
            src[j] = g["src"][c["idx"]] @ R_SHARED.T + T
            det_c[j] = g["det_c"][c["idx"]] @ R_SHARED.T + T
            det_u_v[j] = g["det_u_v"][c["idx"]] @ R_SHARED.T
            det_v_v[j] = g["det_v_v"][c["idx"]] @ R_SHARED.T
            rows.append(s[c["idx"]]); du, dv = g["du"], g["dv"]
        sino = np.stack(rows, axis=0)
        return src, det_c, det_u_v, det_v_v, sino, du, dv

    raise ValueError(arm)


def residual(v_np, fwd, n_views, meas, norm_b):
    v = xp.array(v_np); r = 0.0
    for i in range(n_views):
        sim = np.asarray(_b.to_numpy(fwd(v, i)), np.float32)
        r += float(np.sum((sim - meas[i]) ** 2))
    return np.sqrt(r) / norm_b


def tv_norm(v):
    g = np.sqrt(sum(np.square(np.gradient(v, axis=a)) for a in range(3)))
    return float(g.sum()) / v.size * 1e3


def write_sweep_summary(path: Path, results, best):
    """Persist the reference-free selection evidence for one arm and budget.

    The reconstruction volumes are too large to version, but this small CSV
    records every tested TV weight, its fractional data residual, and the
    selected row.  It is deliberately written next to the driver scripts so
    it can be committed with a rerun without exposing raw projections.
    """
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=("alpha", "fractional_data_residual", "tv_norm_x1e3",
                        "reference_psnr_diagnostic", "selected"),
        )
        writer.writeheader()
        for alpha, res, tv, _volume, psnr in results:
            writer.writerow({
                "alpha": f"{alpha:.6g}",
                "fractional_data_residual": f"{res:.8f}",
                "tv_norm_x1e3": f"{tv:.8f}",
                # This diagnostic is retained for audit only.  It is never
                # used to choose alpha.
                "reference_psnr_diagnostic": f"{psnr:.6f}",
                "selected": int(alpha == best[0]),
            })


def main():
    arm = os.environ.get("SWEEP_ARM", "bundle")
    k = int(os.environ.get("SWEEP_K", "100"))
    print(f"=== sweep {arm} k={k} ===")

    _, ref = rek2py(str(REFERENCE), switch_order=True)
    ref = np.asarray(ref, np.float32)
    shape = ref.shape

    src, det_c, det_u_v, det_v_v, sino, du, dv = load_arm(arm, k, shape)
    n_views, det_u_count, det_v_count = sino.shape
    norm_b = float(np.linalg.norm(sino))

    fwd1, back1, back_all = dct.make_cone_3d_operators(
        xp.array(src), xp.array(det_c), xp.array(det_u_v), xp.array(det_v_v),
        volume_shape=shape, detector_shape=(det_u_count, det_v_count),
        du=du, dv=dv, voxel_spacing=VOXEL_MM, projector_mode="footprint")
    sino_d = xp.array(sino, dtype=_b.float32)
    views = [sino_d[i] for i in range(n_views)]

    sens = _build_sensitivity_support_mask(back_all, (n_views, det_u_count, det_v_count))
    fov = _build_leap_style_circular_fov_mask(shape, VOXEL_MM, (det_u_count, det_v_count),
                                              du, xp.array(src), xp.array(det_c))
    mask = xp.array(sens & fov)

    # SWEEP_REUSE=1: score existing per-alpha volumes from a previous sweep
    # instead of reconstructing them again (summary-CSV regeneration only).
    reuse = os.environ.get("SWEEP_REUSE", "0") == "1"

    results = []
    for alpha in ALPHAS:
        t0 = time.time()
        cached = HERE / "sweeps" / f"sweep_{arm}_k{k:04d}_a{alpha:.2f}" / "reconstruction.rek"
        if reuse and cached.exists():
            _, v_np = rek2py(str(cached), switch_order=True)
            v_np = np.asarray(v_np, np.float32)
        else:
            params = dct.SARTParameters(
                volume_shape=shape, iteration_count=ASD_OUTER, sart_iteration_count=1,
                normalized_sart_relaxation=SART_RELAX, enforce_positivity=True,
                shuffle_projection_order=True, projection_order_seed=0,
                volume_support_mask=mask, volume_support_mask_mode="always")
            reg = dct.ASDPOCSParameters(reg_iteration_count=ASD_REG_ITERS, alpha=alpha,
                                        beta_red=ASD_BETA_RED)
            vol = dct.reconstruct_asd_pocs(views, fwd1, back1, params, reg, show_progress=False)
            v_np = np.asarray(_b.to_numpy(vol), np.float32)
        res = residual(v_np, fwd1, n_views, sino, norm_b)
        tv = tv_norm(v_np)
        scale = float((v_np * ref).sum() / (v_np * v_np).sum())
        v_s = v_np * scale
        psnr = M.psnr(v_s, ref)
        dt = time.time() - t0
        print(f"  alpha={alpha:.2f}: {dt:5.0f}s  resid={res:.4f}  tv={tv:.4f}  "
              f"(reference-PSNR={psnr:.2f}, NOT used for selection)", flush=True)
        if not (reuse and cached.exists()):
            out = HERE / "sweeps" / f"sweep_{arm}_k{k:04d}_a{alpha:.2f}"
            out.mkdir(exist_ok=True)
            rec.save_rek(v_np, out / "reconstruction.rek", VOXEL_MM * 1000.0)
        results.append((alpha, res, tv, v_np, psnr))

    best = min(results, key=lambda r: r[1])
    summary_path = HERE / "sweeps" / f"sweep_{arm}_k{k:04d}_summary.csv"
    write_sweep_summary(summary_path, results, best)
    print(f"  WINNER by data residual: alpha={best[0]:.2f} (resid={best[1]:.4f}, "
          f"reference-PSNR={best[4]:.2f})")
    print(f"  wrote selection evidence: {summary_path.name}")
    if not reuse:
        out = HERE / "results_final" / f"{arm}_final_k{k:04d}"
        out.mkdir(exist_ok=True)
        rec.save_rek(best[3], out / "reconstruction.rek", VOXEL_MM * 1000.0)
        rec.save_slice_pngs(best[3], out / "reconstruction")
    print(f"SWEEP_{arm.upper()}_K{k}_DONE")


if __name__ == "__main__":
    main()
