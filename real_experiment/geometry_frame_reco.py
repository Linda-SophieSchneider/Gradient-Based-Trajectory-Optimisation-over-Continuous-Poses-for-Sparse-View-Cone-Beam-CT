"""Geometry-frame registration + reconstruction-protocol variants.

Instead of warping reconstructed volumes (two trilinear passes -> resolution
floor, k=400 scoring like k=100), apply the rigid object pose to the SCAN
GEOMETRY and reconstruct directly in the reference frame: measurement
m_i = int mu_S(p) dl  =  int mu_R(F(p)) dl, so reconstructing with the
transformed geometry (src' = R@src + T, det' = R@det + T, axes rotated)
yields mu_R natively -- zero interpolation on any volume.

Pose (R, T): initialised from the volume registration (yaw ~ +/-89.5 deg
about world z, roll ~ +/-1 deg about world x, shift [48,-12,-24] voxels in
array (z,y,x) order). Sign/axis conventions are NOT derived by hand: all
sign combinations are scored by correlating measured projections with
reference reprojections through the candidate geometry, then the winner is
refined by coordinate descent (yaw/roll 0.25 deg, T 0.5 mm).

Variants reconstructed on the test arm (env-selectable):
  V1_geo25        canonical tuned ASD-POCS (sart update, outer=25)
  V2_nsart_w25    + iterative_update_method='normalized_sart' with
                    trapezoidal quadrature weights (the 'sart' path ignores
                    projection_weights; normalized_sart consumes them)
  V3_geo50        canonical, outer=50
Each variant: data residual, LS match to the native reference, PSNR/SSIM/
NRMSE/HFEN (no warping anywhere), volume + slice PNG saved.

Env: EZRT_DATA_DIR (measured scan), GEO_OUT tag prefix, GEO_VARIANTS
(comma list, default V1_geo25,V2_nsart_w25,V3_geo50).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Differentiable-Coverage"))

import reconstruct_ezrt_cuda as rec                       # noqa: E402
import diffct_mlx as dct                                  # noqa: E402
from diffct_mlx.backend import active as _b               # noqa: E402
from diffct_mlx.reconstruction_algorithms.cases import (  # noqa: E402
    _build_sensitivity_support_mask,
    _build_leap_style_circular_fov_mask,
    _trajectory_quadrature_weights,
)
from EZRT_Helpers.rek2py import rek2py                    # noqa: E402
from differentiable_coverage.eval import metrics as M     # noqa: E402
from sart_circular_baseline import (                      # noqa: E402
    ASD_OUTER, ASD_REG_ITERS, ASD_ALPHA, ASD_BETA_RED, SART_RELAX,
)

xp = _b.xp
REFERENCE = HERE / "reference_reconstructions" / "output_circular1200_fdk_quant" / "reconstruction_FDK.rek"
OUT_TAG = os.environ.get("GEO_OUT", "geoframe_all3_N0100")
VARIANTS = os.environ.get("GEO_VARIANTS", "V1_geo25,V2_nsart_w25,V3_geo50").split(",")

# volume-registration initialisation (register_and_score.py result)
YAW0 = 89.5          # deg, about world z (array axes 1,2)
ROLL0 = 1.0          # deg, about world x (array axes 0,1)
SHIFT_VOX = np.array([48.0, -12.0, -24.0])   # array (z, y, x)
VOXEL_MM = 0.2777


def rot_z(deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rot_x(deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def main() -> None:
    _, ref = rek2py(str(REFERENCE), switch_order=True)
    ref = np.asarray(ref, np.float32)

    sino_raw, geom_raw, meta = rec.load_ezrt_dataset(rec.DATA_DIR, rec.DETECTOR_BIN, 1)
    n_views, det_u, det_v = sino_raw.shape
    du, dv = meta["du"], meta["dv"]
    src, det_c, det_u_v, det_v_v, iso = rec.build_geometry(geom_raw)
    meas = rec.log_normalize(sino_raw); del sino_raw
    fod = float(np.mean(np.linalg.norm(src, axis=1)))
    fdd = float(np.mean(np.linalg.norm(det_c - src, axis=1)))
    voxel_mm = du / (fdd / fod)
    shape = ref.shape
    print(f"{rec.DATA_DIR.name}: {n_views} views; ref={shape}")

    ref_d = xp.array(ref)
    probe_views = list(range(0, n_views, max(1, n_views // 10)))[:10]
    pose_file = HERE / "registration" / os.environ.get("GEO_POSE_FILE", f"{OUT_TAG}_pose.json")

    def geo(Rw, T):
        return (src @ Rw.T + T, det_c @ Rw.T + T, det_u_v @ Rw.T, det_v_v @ Rw.T)

    def score(Rw, T, views):
        s2, d2, u2, v2 = geo(Rw, T)
        fwd1, _, _ = dct.make_cone_3d_operators(
            xp.array(s2), xp.array(d2), xp.array(u2), xp.array(v2),
            volume_shape=shape, detector_shape=(det_u, det_v),
            du=du, dv=dv, voxel_spacing=voxel_mm, projector_mode="siddon")
        cs = []
        for i in views:
            sim = np.asarray(_b.to_numpy(fwd1(ref_d, i)), np.float32)
            cs.append(float(np.corrcoef(sim.ravel(), meas[i].ravel())[0, 1]))
        return float(np.median(cs))

    # world translation candidates from the array-space shift (z,y,x)->(x,y,z)
    T_arr = np.array([SHIFT_VOX[2], SHIFT_VOX[1], SHIFT_VOX[0]]) * VOXEL_MM

    if pose_file.exists():
        pj = json.loads(pose_file.read_text())
        yaw, roll, T, inv = pj["yaw"], pj["roll"], np.array(pj["T"], float), pj["inv"]
        print(f"loaded cached pose from {pose_file.name}: yaw={yaw:+.2f} "
              f"roll={roll:+.2f} T={np.round(T, 2)} (corr {pj['corr']:.4f})")

        def R_of(yw, rl):
            R = rot_x(rl) @ rot_z(yw)
            return R.T if inv else R

        Rw = R_of(yaw, roll)
        # per-arm translation touch-up: arms share the session pose but have
        # mm-level isocentre-estimate differences between scans
        cur = score(Rw, T, probe_views)
        for ax in range(3):
            for dt in (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5):
                T2 = T.copy(); T2[ax] += dt
                s = score(Rw, T2, probe_views)
                if s > cur:
                    T, cur = T2, s
        print(f"per-arm T refinement: T={np.round(T, 2)} mm  corr={cur:.4f}")
    else:
        # --- sign/convention search ------------------------------------
        best = (None, -2.0)
        for ys in (+1, -1):
            for rs in (+1, -1):
                for ts in (+1, -1):
                    Rw = rot_x(rs * ROLL0) @ rot_z(ys * YAW0)
                    for inv in (False, True):
                        Rc = Rw.T if inv else Rw
                        Tc = (ts * T_arr) @ (Rc.T if inv else np.eye(3)) * (-1 if inv else 1)
                        sc = score(Rc, Tc, probe_views[:5])
                        if sc > best[1]:
                            best = ((Rc.copy(), Tc.copy(), ys, rs, ts, inv), sc)
        (Rw, T, ys, rs, ts, inv), sc0 = best
        print(f"convention search: yaw_sign={ys} roll_sign={rs} t_sign={ts} inverse={inv} "
              f"-> median corr {sc0:.4f}")

        yaw, roll = ys * YAW0, rs * ROLL0
        T = T.copy()

        def R_of(yw, rl):
            R = rot_x(rl) @ rot_z(yw)
            return R.T if inv else R

        cur = score(R_of(yaw, roll), T, probe_views)
        for _round in range(2):
            for dy in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
                s = score(R_of(yaw + dy, roll), T, probe_views)
                if s > cur: yaw, cur = yaw + dy, s
            for dr in (-1.0, -0.5, 0.5, 1.0):
                s = score(R_of(yaw, roll + dr), T, probe_views)
                if s > cur: roll, cur = roll + dr, s
            for ax in range(3):
                for dt in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
                    T2 = T.copy(); T2[ax] += dt
                    s = score(R_of(yaw, roll), T2, probe_views)
                    if s > cur: T, cur = T2, s
        Rw = R_of(yaw, roll)
        print(f"refined pose: yaw={yaw:+.2f} roll={roll:+.2f} T={np.round(T, 2)} mm "
              f"-> median reprojection corr {cur:.4f}")
        pose_file.write_text(json.dumps(dict(yaw=float(yaw), roll=float(roll),
                                             T=[float(x) for x in T],
                                             inv=bool(inv), corr=float(cur))))
        print(f"saved pose to {pose_file.name}")

    # --- reconstruction variants --------------------------------------------
    s2, d2, u2, v2 = geo(Rw, T)
    fwd1f, back1f, back_all = dct.make_cone_3d_operators(
        xp.array(s2), xp.array(d2), xp.array(u2), xp.array(v2),
        volume_shape=shape, detector_shape=(det_u, det_v),
        du=du, dv=dv, voxel_spacing=voxel_mm, projector_mode="footprint")
    sino_d = xp.array(meas, dtype=_b.float32)
    views = [sino_d[i] for i in range(n_views)]
    norm_b = float(np.linalg.norm(meas))

    sens = _build_sensitivity_support_mask(back_all, (n_views, det_u, det_v))
    fov = _build_leap_style_circular_fov_mask(shape, voxel_mm, (det_u, det_v), du,
                                              xp.array(s2), xp.array(d2))
    mask = xp.array(sens & fov)
    weights = _trajectory_quadrature_weights(xp.array(s2))

    def residual(v_np):
        v = xp.array(v_np); r = 0.0
        fwd_s, _, _ = dct.make_cone_3d_operators(
            xp.array(s2), xp.array(d2), xp.array(u2), xp.array(v2),
            volume_shape=shape, detector_shape=(det_u, det_v),
            du=du, dv=dv, voxel_spacing=voxel_mm, projector_mode="footprint")
        for i in range(n_views):
            sim = np.asarray(_b.to_numpy(fwd_s(v, i)), np.float32)
            r += float(np.sum((sim - meas[i]) ** 2))
        return np.sqrt(r) / norm_b

    def run_variant(name, outer, nsart, use_weights):
        kw = dict(volume_shape=shape, iteration_count=outer, sart_iteration_count=1,
                  normalized_sart_relaxation=SART_RELAX, enforce_positivity=True,
                  shuffle_projection_order=True, projection_order_seed=0,
                  volume_support_mask=mask, volume_support_mask_mode="always")
        if nsart:
            kw["iterative_update_method"] = "normalized_sart"
        if use_weights:
            kw["projection_weights"] = weights
        t0 = time.time()
        vol = dct.reconstruct_asd_pocs(
            views, fwd1f, back1f, dct.SARTParameters(**kw),
            dct.ASDPOCSParameters(reg_iteration_count=ASD_REG_ITERS,
                                  alpha=ASD_ALPHA, beta_red=ASD_BETA_RED))
        v_np = np.asarray(_b.to_numpy(vol), np.float32)
        dt = time.time() - t0
        scale = float((v_np * ref).sum() / (v_np * v_np).sum())
        v_s = v_np * scale
        res = residual(v_np)
        print(f"{name}: {dt:5.0f}s  resid={res:.4f}  LS={scale:.3f}  "
              f"PSNR={M.psnr(v_s, ref):.2f}  SSIM={M.ssim(v_s, ref):.4f}  "
              f"NRMSE={M.nrmse(v_s, ref):.4f}  HFEN={M.hfen(v_s, ref):.2f}", flush=True)
        out = HERE / "registration" / f"{OUT_TAG}_{name}"
        out.mkdir(exist_ok=True)
        rec.save_rek(v_np, out / "reconstruction.rek", voxel_mm * 1000.0)
        rec.save_slice_pngs(v_np, out / "reconstruction")

    specs = {
        "V1_geo25": (ASD_OUTER, False, False),
        "V2_nsart_w25": (ASD_OUTER, True, True),
        "V3_geo50": (50, False, False),
    }
    for name in VARIANTS:
        outer, nsart, w = specs[name.strip()]
        run_variant(name.strip(), outer, nsart, w)


if __name__ == "__main__":
    main()
