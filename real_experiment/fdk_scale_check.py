"""Quantify the radiometric FDK scale of the reconstruct_measured_cuda pipeline.

Forward-project a known-mu cylinder with Siddon (exact line integrals in mm),
then reconstruct with EXACTLY the script's FDK path (cosine weights, diffct
ramp filter, Siddon-adjoint backprojector, normalization pi*SID/(2*SDD*N)).
The ratio recon/mu_true measures the radiometric scale of that pipeline for
each geometry convention. SART on the same sinogram is the control (measured
exact: 0.9999 of true mu).

Baseline measured with diffct-mlx 2.0.1 (2026-07-15), BEFORE the FDK fixes:
  unit geometry (du=voxel=1, M=1.5):        FDK/mu = 0.74
  real geometry (du=.556, vox=.278, M=2.0): FDK/mu = 0.45
  object-size dependence (unpadded ramp!):  0.94 / 0.74 / 0.40
                                            at 25/50/75% FOV width
RE-RUN THIS after any diffct-mlx FDK update (ramp padding / normalization /
U^2 backprojection): all cases should move to ~1.0 and become object-size
independent before an FDK volume is trusted as a quantitative metric
reference. Object-size sweep: adjust the phantom radius (n/4 below) or add
runs at n/8 and 3*n/8 as in the 2026-07-15 investigation.
"""
import math
import sys
import numpy as np

import diffct_mlx as dct
from diffct_mlx.backend import active as _b
from diffct_mlx.geometry import circular_trajectory_3d

xp = _b.xp

MU = 0.05  # mm^-1


def run_quantitative(name, *, n, det_n, num_views, du, voxel, sid, sdd):
    """diffct >= 2.1.0 quantitative FDK path (physical ramp |f|/du, trapezoidal
    angular weights, voxel-driven (sid/U)^2 gather BP). Expected ~= 1.0."""
    from diffct_mlx.reconstruction_algorithms.cases import _quantitative_fdk_operators

    src, det_c, det_u_v, det_v_v = circular_trajectory_3d(num_views, sid, sdd)
    zz, yy, xx = np.meshgrid(*(np.arange(n) - (n - 1) / 2,) * 3, indexing="ij")
    vol = ((xx**2 + yy**2 < (n / 4) ** 2) & (np.abs(zz) < n / 4)).astype(np.float32) * MU
    core = (xx**2 + yy**2 < (n / 6) ** 2) & (np.abs(zz) < n / 6)

    fwd_1, _, _ = dct.make_cone_3d_operators(
        src, det_c, det_u_v, det_v_v, volume_shape=(n,) * 3,
        detector_shape=(det_n, det_n), du=du, dv=du, voxel_spacing=voxel,
        projector_mode="siddon")
    sino = xp.stack([fwd_1(xp.array(vol), i) for i in range(num_views)])

    # Siddon output integrates in voxel units -> sinogram_scale=voxel converts
    # to physical mu*mm line integrals (pass 1.0 for measured -log data).
    q_weight, q_filter, q_back = _quantitative_fdk_operators(
        src, det_c, det_u_v, det_v_v,
        volume_shape=(n,) * 3, detector_shape=(det_n, det_n),
        du=du, dv=du, voxel_spacing=voxel, sinogram_scale=voxel)
    if q_back is None:
        print(f"{name}: quantitative path unavailable on this backend")
        return
    fdk = dct.reconstruct_fdk(
        sino, q_back,
        dct.FDKParameters(normalization_scale=1.0, enforce_positivity=True),
        weight_projections=q_weight, filter_projections=q_filter)
    fdk_np = np.asarray(_b.to_numpy(fdk))
    print(f"{name}: du={du} voxel={voxel} M={sdd/sid:.3f} "
          f"-> quantitative FDK core mean/mu = {float(fdk_np[core].mean() / MU):.4f}")


def run_case(name, *, n, det_n, num_views, du, voxel, sid, sdd, sart_iters=0):
    src, det_c, det_u_v, det_v_v = circular_trajectory_3d(num_views, sid, sdd)

    # cylinder phantom, radius n/4 voxels, height n/2
    zz, yy, xx = np.meshgrid(*(np.arange(n) - (n - 1) / 2,) * 3, indexing="ij")
    vol = ((xx**2 + yy**2 < (n / 4) ** 2) & (np.abs(zz) < n / 4)).astype(np.float32) * MU
    core = (xx**2 + yy**2 < (n / 6) ** 2) & (np.abs(zz) < n / 6)  # eroded core

    fwd_1, back_1, _ = dct.make_cone_3d_operators(
        src, det_c, det_u_v, det_v_v, volume_shape=(n,) * 3,
        detector_shape=(det_n, det_n), du=du, dv=du, voxel_spacing=voxel,
        projector_mode="siddon")
    _, _, back_all = dct.make_cone_3d_operators(
        src, det_c, det_u_v, det_v_v, volume_shape=(n,) * 3,
        detector_shape=(det_n, det_n), du=du, dv=du, voxel_spacing=voxel,
        projector_mode="siddon")

    vol_d = xp.array(vol)
    sino = xp.stack([fwd_1(vol_d, i) for i in range(num_views)])

    # --- FDK exactly as in reconstruct_measured_cuda.py ---
    u = (xp.arange(det_n) - (det_n - 1) / 2.0) * du
    w = sdd / xp.sqrt(sdd**2 + u.reshape(1, det_n, 1) ** 2 + u.reshape(1, 1, det_n) ** 2)
    norm = (math.pi * sid) / (2.0 * sdd * num_views)
    fdk = dct.reconstruct_fdk(
        sino, back_all,
        dct.FDKParameters(voxel_spacing=voxel, enforce_positivity=True,
                          normalization_scale=norm, filter_axis=1),
        weight_projections=lambda raw: raw * w)
    fdk_np = np.asarray(_b.to_numpy(fdk))
    scale_fdk = float(fdk_np[core].mean() / MU)

    msg = (f"{name}: du={du} voxel={voxel} M={sdd/sid:.3f} N={num_views} "
           f"-> FDK core mean/mu = {scale_fdk:.4f}")
    if sart_iters:
        params = dct.SARTParameters(volume_shape=(n,) * 3, iteration_count=sart_iters,
                                    normalized_sart_relaxation=0.9,
                                    enforce_positivity=True,
                                    shuffle_projection_order=True)
        sart = dct.reconstruct_sart([sino[i] for i in range(num_views)],
                                    fwd_1, back_1, params, show_progress=False)
        sart_np = np.asarray(_b.to_numpy(sart))
        msg += f" | SART core mean/mu = {float(sart_np[core].mean() / MU):.4f}"
    print(msg, flush=True)
    return scale_fdk


# B: diffct's synthetic unit-spacing convention (constant calibrated here)
run_case("B unit ", n=128, det_n=128, num_views=180, du=1.0, voxel=1.0,
         sid=600.0, sdd=900.0)
# A: our real-scan convention (real-like spacings, magnification 2)
run_case("A real ", n=128, det_n=128, num_views=180, du=0.5556, voxel=0.2777,
         sid=997.07, sdd=1995.64, sart_iters=6)
# C: like A but voxel doubled -> tests the voxel-size dependence
run_case("C vox2x", n=128, det_n=128, num_views=180, du=0.5556, voxel=0.5556,
         sid=997.07, sdd=1995.64)

# Quantitative path (diffct >= 2.1.0): all three cases must be ~1.0.
run_quantitative("Q unit ", n=128, det_n=128, num_views=180, du=1.0, voxel=1.0,
                 sid=600.0, sdd=900.0)
run_quantitative("Q real ", n=128, det_n=128, num_views=180, du=0.5556,
                 voxel=0.2777, sid=997.07, sdd=1995.64)
run_quantitative("Q vox2x", n=128, det_n=128, num_views=180, du=0.5556,
                 voxel=0.5556, sid=997.07, sdd=1995.64)
