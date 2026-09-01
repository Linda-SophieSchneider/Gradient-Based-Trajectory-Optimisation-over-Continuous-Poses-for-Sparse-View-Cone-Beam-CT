"""Validate + reconstruct a measured planned-trajectory scan.

Pipeline for each newly acquired scan of a planned trajectory:
  1. load the raw scanner dataset (4x binning) with its per-view AGV geometry
  2. match the measured source positions against a planned trajectory CSV
     (nearest-neighbour after mean-removal -> executed-series check)
  3. projection-consistency check against the prescan FDK prior
     (per-view Pearson + phase-correlation shift; a rigid object offset
     shows up as a constant v-shift + sinusoidal u-shift, see the
     2026-07-17 bundle-scan investigation)
  4. reconstruct with the canonical tuned ASD-POCS protocol of
     sart_circular_baseline.py (incl. per-trajectory FOV support mask)

Env:
  SCAN_DATA_DIR       measured scan directory (required)
  MEAS_OUT_DIR        output directory (default measured_<basename of data dir>)
  MEAS_PLANNED_CSV    planned trajectory_coords.csv to compare against (optional)
  MEAS_SKIP_CHECKS=1  reconstruct only
"""
from __future__ import annotations

import csv as csvmod
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import reconstruct_measured_cuda as rec                   # noqa: E402
import diffct_mlx as dct                                  # noqa: E402
from diffct_mlx.backend import active as _b               # noqa: E402
from diffct_mlx.reconstruction_algorithms.cases import (  # noqa: E402
    _build_sensitivity_support_mask,
    _build_leap_style_circular_fov_mask,
)
from scanner_io.rek2py import rek2py                      # noqa: E402
from sart_circular_baseline import (                      # noqa: E402
    ASD_OUTER, ASD_REG_ITERS, ASD_ALPHA, ASD_BETA_RED, SART_RELAX,
)

xp = _b.xp
OUT_DIR = Path(os.environ.get("MEAS_OUT_DIR",
                              str(HERE / f"measured_{rec.DATA_DIR.name}")))
PLANNED_CSV = os.environ.get("MEAS_PLANNED_CSV", "")
SKIP_CHECKS = os.environ.get("MEAS_SKIP_CHECKS", "") == "1"
PRESCAN_FDK = HERE / "reference_reconstructions" / "output_prescan" / "reconstruction_FDK.rek"


def main() -> None:
    sino_raw, geom_raw, meta = rec.load_measured_dataset(rec.DATA_DIR, rec.DETECTOR_BIN, 1)
    n_views, det_u, det_v = sino_raw.shape
    du, dv = meta["du"], meta["dv"]
    src, det_c, det_u_v, det_v_v, iso = rec.build_geometry(geom_raw)
    meas = rec.log_normalize(sino_raw); del sino_raw
    fod = float(np.mean(np.linalg.norm(src, axis=1)))
    fdd = float(np.mean(np.linalg.norm(det_c - src, axis=1)))
    voxel_mm = du / (fdd / fod)
    shape = (det_u,) * 3
    print(f"{rec.DATA_DIR.name}: {n_views} views, isocentre={np.round(iso, 1)}, "
          f"FOD={fod:.2f}, voxel={voxel_mm:.4f}")

    if PLANNED_CSV and not SKIP_CHECKS:
        with open(PLANNED_CSV, newline="") as fh:
            rows = list(csvmod.DictReader(fh))
        planned = np.array([[float(r["src_x_mm"]), float(r["src_y_mm"]),
                             float(r["src_z_mm"])] for r in rows])
        # build_geometry() output and planned CSVs are BOTH isocentre-centred:
        # compare directly. (Mean-removal is WRONG here — with missing views
        # the subset mean biases every NN distance, e.g. 35mm instead of the
        # true 8mm on the 88-view all3 scan.)
        d = np.linalg.norm(src[:, None, :] - planned[None, :, :], axis=2)
        nn = d.min(axis=1)
        print(f"vs planned ({Path(PLANNED_CSV).parent.parent.name}): "
              f"median NN = {np.median(nn):.2f} mm  max = {nn.max():.2f} mm")

    if not SKIP_CHECKS and PRESCAN_FDK.exists():
        _, vol = rek2py(str(PRESCAN_FDK), switch_order=True)
        vol = np.asarray(vol, np.float32)
        fwd1, _, _ = dct.make_cone_3d_operators(
            xp.array(src), xp.array(det_c), xp.array(det_u_v), xp.array(det_v_v),
            volume_shape=vol.shape, detector_shape=(det_u, det_v),
            du=du, dv=dv, voxel_spacing=voxel_mm, projector_mode="siddon")
        vol_d = xp.array(vol)
        corrs, shifts = [], []
        for i in range(0, n_views, 4):
            sim = np.asarray(_b.to_numpy(fwd1(vol_d, i)), np.float32)
            mv = meas[i]
            corrs.append(float(np.corrcoef(sim.ravel(), mv.ravel())[0, 1]))
            F = np.fft.fft2(mv) * np.conj(np.fft.fft2(sim)); F /= np.abs(F) + 1e-12
            pc = np.abs(np.fft.ifft2(F))
            pu, pv = np.unravel_index(np.argmax(pc), pc.shape)
            if pu > det_u // 2: pu -= det_u
            if pv > det_v // 2: pv -= det_v
            shifts.append((pu * du, pv * dv))
        corrs = np.array(corrs); shifts = np.array(shifts)
        print(f"projection consistency: median corr={np.median(corrs):.4f} "
              f"min={corrs.min():.4f} | shift median=({np.median(shifts[:,0]):+.2f},"
              f"{np.median(shifts[:,1]):+.2f})mm std=({shifts[:,0].std():.2f},"
              f"{shifts[:,1].std():.2f})")

    # canonical tuned ASD-POCS protocol (identical to the circular baselines)
    fwd1f, back1f, back_all = dct.make_cone_3d_operators(
        xp.array(src), xp.array(det_c), xp.array(det_u_v), xp.array(det_v_v),
        volume_shape=shape, detector_shape=(det_u, det_v),
        du=du, dv=dv, voxel_spacing=voxel_mm, projector_mode="footprint")
    sino_d = xp.array(meas, dtype=_b.float32)
    views = [sino_d[i] for i in range(n_views)]

    sens = _build_sensitivity_support_mask(back_all, (n_views, det_u, det_v))
    fov = _build_leap_style_circular_fov_mask(shape, voxel_mm, (det_u, det_v), du,
                                              xp.array(src), xp.array(det_c))
    mask = xp.array(sens & fov)

    params = dct.SARTParameters(
        volume_shape=shape, iteration_count=ASD_OUTER, sart_iteration_count=1,
        normalized_sart_relaxation=SART_RELAX, enforce_positivity=True,
        shuffle_projection_order=True, projection_order_seed=0,
        volume_support_mask=mask, volume_support_mask_mode="always")
    reg = dct.ASDPOCSParameters(reg_iteration_count=ASD_REG_ITERS,
                                alpha=ASD_ALPHA, beta_red=ASD_BETA_RED)
    t0 = time.time()
    volume = dct.reconstruct_asd_pocs(views, fwd1f, back1f, params, reg,
                                      show_progress=True)
    vol_np = np.asarray(_b.to_numpy(volume), dtype=np.float32)
    print(f"ASD-POCS ({ASD_OUTER} outer, TV {ASD_REG_ITERS}x{ASD_ALPHA}) done in "
          f"{time.time() - t0:.1f}s, range [{vol_np.min():.5f}, {vol_np.max():.5f}]")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec.save_rek(vol_np, OUT_DIR / "reconstruction_ASDPOCS_final.rek", voxel_mm * 1000.0)
    rec.save_slice_pngs(vol_np, OUT_DIR / "reconstruction_ASDPOCS_final")


if __name__ == "__main__":
    main()
