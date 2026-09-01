"""Circular-subset SART baseline for the real-world camera experiment.

For each view budget k in {100, 400}, take every (1200/k)-th projection of the
measured 1200-view circular scan (equidistant angles), reconstruct with the
tuned ASD-POCS protocol (see the protocol block below), and score against the
1200-view reference reconstruction with the paper's four metrics
(PSNR, SSIM, NRMSE, HFEN).

The scoring reference is the QUANTITATIVE 1200-view FDK reconstruction
(diffct-mlx >= 2.1.0 physical path, mu in 1/mm; built by
reconstruct_measured_cuda.py into output_circular1200_fdk_quant/) -- the user's
explicit decision (2026-07-15): the SART-1200 volume carries too much
iteration noise / too many artefacts to serve as ground-truth proxy.
Because the SART arms converge to attenuation-per-voxel units (mu *
voxel_spacing, diffct units convention) while the FDK reference is in 1/mm,
each arm is intensity-matched to the reference by a single per-arm
least-squares factor before scoring (recorded in the CSV).

Known property of this cross-algorithm comparison (measured 2026-07-15,
twice: with legacy-calibrated AND quantitative FDK): more-converged SART
(higher k) is sharper/noisier and can drift L2-further from the smooth FBP
volume even as structure improves, so global intensity metrics (PSNR/SSIM/
NRMSE) may not rank k monotonically; HFEN tracks the structural improvement.
Never score against the pre-2.1.0 hand-rolled FDK (radiometrically off by a
geometry-dependent factor, see fdk_scale_check.py).

Everything heavy is reused:
  - loading / geometry / -log normalisation / .rek export:
      ``reconstruct_measured_cuda`` (unchanged, same 4x detector binning -> 768^3)
  - SART parameters: mirror ``differentiable_coverage.eval.reco``
    (the paper's reconstruction pipeline), executed through the same
    ``diffct_mlx`` operators as the reference reconstruction
  - metrics: ``differentiable_coverage.eval.metrics`` (paper defaults)

Both the SART volumes and the FDK reference are round-tripped through the
same .rek writer/reader so the voxel-order convention is identical on both
sides of every metric.

Env overrides: SCAN_DATA_DIR (default circular_1200), BASELINE_REFERENCE_REK,
BASELINE_OUT_DIR.
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

import reconstruct_measured_cuda as rec                    # noqa: E402
import diffct_mlx as dct                                   # noqa: E402
from diffct_mlx.backend import active as _b                # noqa: E402
from scanner_io.rek2py import rek2py                       # noqa: E402
from differentiable_coverage.eval import metrics as M      # noqa: E402

xp = _b.xp

OUT_ROOT = Path(os.environ.get("BASELINE_OUT_DIR", str(HERE / "reference_reconstructions" / "baseline_circular_sart")))
# Default reference: quantitative 1200-view FDK (user's decision, see
# docstring). Build it with reconstruct_measured_cuda.py if missing:
#   SCAN_DATA_DIR=.../circular_1200 SCAN_OUT_DIR=.../output_circular1200_fdk_quant
REFERENCE_REK = Path(os.environ.get(
    "BASELINE_REFERENCE_REK",
    str(HERE / "reference_reconstructions" / "output_circular1200_fdk_quant" / "reconstruction_FDK.rek"),
))
VIEW_COUNTS = (100, 400)

# Reconstruction protocol for ALL sparse-view arms of the real-world study
# (user decision 2026-07-17, tuned on the measured bundle-N0100 scan):
# ASD-POCS (TV-regularised SART) + per-trajectory FOV support mask.
# Tuning evidence (fraction data residual / TV-norm on the measured scan):
#   SART-15: 0.294/2.26 | ASD default: 0.264/1.05 | tuned: 0.232/0.91 |
#   tuned+mask: 0.210/0.81 (Pareto-best on both axes).
# NOTE: this deviates from the paper's simulated-study protocol (plain
# SART-15); documented as such in paper sec:real. projection_weights are NOT
# used: the 'sart' update path ignores them (only 'normalized_sart' consumes
# them -- verified empirically and in _core.py).
ASD_OUTER = 50   # raised from 25 on 2026-07-21: +0.17dB / better on all four
                 # metrics on the measured all3-k=100 arm, flattening beyond
ASD_REG_ITERS = 40
ASD_ALPHA = 0.30
ASD_BETA_RED = 0.99
SART_RELAX = 0.9  # relaxation of the SART data step inside ASD-POCS


def sart_reconstruct(n_views_target: int) -> tuple[Path, int]:
    """Load the equidistant circular subset and reconstruct via tuned ASD-POCS."""
    from diffct_mlx.reconstruction_algorithms.cases import (
        _build_sensitivity_support_mask,
        _build_leap_style_circular_fov_mask,
    )

    out_dir = OUT_ROOT / f"N{n_views_target:04d}"
    rek_path = out_dir / "reconstruction_ASDPOCS.rek"
    if rek_path.exists():
        print(f"[k={n_views_target}] reusing existing {rek_path}")
        return rek_path, n_views_target

    sino_raw, geom_raw, meta = rec.load_measured_dataset(
        rec.DATA_DIR, rec.DETECTOR_BIN, view_stride=1200 // n_views_target
    )
    n_views, det_u_count, det_v_count = sino_raw.shape
    du, dv = meta["du"], meta["dv"]
    src, det_c, det_u_vec, det_v_vec, iso = rec.build_geometry(geom_raw)

    fod = float(np.mean(np.linalg.norm(src, axis=1)))
    fdd = float(np.mean(np.linalg.norm(det_c - src, axis=1)))
    voxel_mm = du / (fdd / fod)
    volume_shape = (det_u_count,) * 3
    print(f"[k={n_views_target}] {n_views} views, isocentre={np.round(iso, 1)}, "
          f"FOD={fod:.2f} FDD={fdd:.2f}, voxel={voxel_mm:.4f} mm, volume={volume_shape}")

    sino = rec.log_normalize(sino_raw)
    del sino_raw

    forward_single, back_single, back_all = dct.make_cone_3d_operators(
        xp.array(src), xp.array(det_c), xp.array(det_u_vec), xp.array(det_v_vec),
        volume_shape=volume_shape, detector_shape=(det_u_count, det_v_count),
        du=du, dv=dv, voxel_spacing=voxel_mm,
        projector_mode="footprint",
    )
    sino_d = xp.array(sino, dtype=_b.float32)
    measured_views = [sino_d[i] for i in range(n_views)]

    # Per-trajectory FOV support mask (part of the reconstruction operator,
    # like the geometry itself): sensitivity AND circular-FOV.
    sens = _build_sensitivity_support_mask(back_all, (n_views, det_u_count, det_v_count))
    fov = _build_leap_style_circular_fov_mask(
        volume_shape, voxel_mm, (det_u_count, det_v_count), du,
        xp.array(src), xp.array(det_c))
    mask = xp.array(sens & fov)

    params = dct.SARTParameters(
        volume_shape=volume_shape,
        iteration_count=ASD_OUTER,
        sart_iteration_count=1,
        normalized_sart_relaxation=SART_RELAX,
        enforce_positivity=True,
        shuffle_projection_order=True,
        projection_order_seed=0,
        volume_support_mask=mask,
        volume_support_mask_mode="always",
    )
    reg = dct.ASDPOCSParameters(
        reg_iteration_count=ASD_REG_ITERS, alpha=ASD_ALPHA, beta_red=ASD_BETA_RED,
    )
    t0 = time.time()
    volume = dct.reconstruct_asd_pocs(
        measured_views, forward_single, back_single, params, reg, show_progress=True
    )
    vol_np = np.asarray(_b.to_numpy(volume), dtype=np.float32)
    print(f"[k={n_views_target}] ASD-POCS ({ASD_OUTER} outer, TV {ASD_REG_ITERS}x{ASD_ALPHA}) "
          f"done in {time.time() - t0:.1f}s, range [{vol_np.min():.5f}, {vol_np.max():.5f}]")

    out_dir.mkdir(parents=True, exist_ok=True)
    rec.save_rek(vol_np, rek_path, voxel_mm * 1000.0)
    rec.save_slice_pngs(vol_np, out_dir / "reconstruction_ASDPOCS")
    return rek_path, n_views


def main() -> None:
    print(f"diffct backend: {dct.backend}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not REFERENCE_REK.exists():
        if REFERENCE_REK == OUT_ROOT / "N1200" / "reconstruction_SART.rek":
            print("reference (1200-view SART) missing -- building it first")
            sart_reconstruct(1200)
        else:
            raise FileNotFoundError(f"reference reconstruction not found: {REFERENCE_REK}")

    rek_paths = {k: sart_reconstruct(k)[0] for k in VIEW_COUNTS}

    # Metrics: load reference and arms through the identical reader/convention.
    print(f"\nreference: {REFERENCE_REK}")
    _, ref = rek2py(str(REFERENCE_REK), switch_order=True)
    ref = np.asarray(ref, dtype=np.float32)

    rows = []
    for k, rek_path in rek_paths.items():
        _, recon = rek2py(str(rek_path), switch_order=True)
        recon = np.asarray(recon, dtype=np.float32)
        assert recon.shape == ref.shape, (recon.shape, ref.shape)
        # Per-arm LS intensity matching: SART arms are in mu*voxel units, the
        # FDK reference in 1/mm; a single free scale per arm removes the unit
        # and amplitude-convergence mismatch before scoring.
        scale = float((recon * ref).sum() / (recon * recon).sum())
        recon = recon * scale
        row = dict(
            trajectory="circular_subset", k=k, ls_scale=round(scale, 4),
            psnr=M.psnr(recon, ref),
            ssim=M.ssim(recon, ref),
            nrmse=M.nrmse(recon, ref),
            hfen=M.hfen(recon, ref),
        )
        rows.append(row)
        print(f"  k={k:4d}: LS scale={scale:.4f}  PSNR={row['psnr']:.2f} dB  "
              f"SSIM={row['ssim']:.4f}  NRMSE={row['nrmse']:.4f}  HFEN={row['hfen']:.2f}")

    csv_path = OUT_ROOT / "metrics_circular_baseline.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDone. Metrics written to {csv_path}")


if __name__ == "__main__":
    main()
