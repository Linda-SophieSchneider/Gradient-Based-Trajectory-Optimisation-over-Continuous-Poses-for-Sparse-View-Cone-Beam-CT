#!/usr/bin/env python
"""Plan cone-beam trajectories for 50/100/200/300/400 views, each an independent
continuous-optimisation run within a 360 deg (azimuth) x +-PHI_MAX (elevation)
band (currently +-30 deg; was 35 until 2026-07-15, reduced per the machine
operator's envelope limit).

Object model: the FDK reconstruction at ``FDK_PATH``.  Scan geometry
(FOD/FDD/isocentre): re-derived from a handful of real views of circular_1200 by
reusing ``reconstruct_ezrt_cuda.load_ezrt_dataset``/``build_geometry`` unchanged.

For each target view count k in {50,100,200,300,400}: k (theta, phi) source
positions are drawn at random from the band, then ALL k positions
are moved jointly by Adam gradient ascent on
``differentiable_coverage.saturated_coverage`` -- continuous, gradient-based view
optimisation (as opposed to discrete/combinatorial view selection). Azimuth
uses ``differentiable_coverage.TwoAxisGantry`` unconstrained (periodic via
sin/cos, needs no clamp); elevation is kept in +-PHI_MAX by a smooth
``phi = PHI_MAX * tanh(phi_raw)`` reparametrisation rather than a hard clamp --
empirically this reaches the same coverage as clamping
(``CArmTwoAxisGantry``'s own approach) but without ~1/3 of the sources getting
stuck exactly on the boundary rail (clamping's gradient is zero past the
boundary, so a source that overshoots during optimisation can never move back
inward; tanh's gradient never vanishes). Each k is its own independent
optimisation; there is no shared curve across the five view counts.

The active objective is selected per run via ``PLAN_OBJECTIVE`` (see the
config block): coverage only, coverage + bundle-absorption penalty
(``differentiable_coverage.absorption_bundle``), or coverage + VCL info
(``L = C_geo + lambda_info * I_vcl``, the default). ``I_vcl`` is ``differentiable_coverage.vcl_diff``'s continuous
View Covariance Loss surrogate (Lin et al., extended to continuous source
positions): ``1 - vcl_loss_continuous(sources, ctx)``, built once per FDK
volume via ``build_vcl_context`` (downsampled reference + fixed voxel
sub-sample, matching the paper's r1=1e-3 default).

VCL needs this machine's diffct_mlx **torch** backend (no Apple Metal here) to
actually run the forward/backprojection it's built on; that backend passes
plain MLX arrays straight into torch calls, which crashes without a bridge.
``differentiable_coverage._torch_bridge`` supplies that bridge (MLX<->Torch via
an ``mx.custom_function`` whose vjp recomputes the torch forward pass and
pulls the gradient out via ``torch.autograd.grad``), wired into
``vcl_diff.view_basis_matrix``/``vcl_backprojection.backproject_single_view``
behind an ``is_torch_backend()`` check -- the native Apple ``mlx`` backend path
is untouched. Caveat carried over from diffct-mlx's own changelog: its torch
cone-beam **backprojector** is data-only (gradient only w.r.t. the sinogram,
not source/detector geometry), so VCL's source-position gradient here comes
entirely through the forward-projection step, not the backprojection step --
a real (verified non-zero, monotone in k) but structurally partial gradient,
not the fully analytic one Apple-Metal hardware would give.

Per-view source/detector geometry is built via ``EzrtHeader`` (so units/sign
conventions match ``reconstruct_ezrt_cuda.build_geometry`` exactly -- metres,
negated orientation vectors) and written as one consolidated
``trajectory_headers.txt`` per view count, with every view's fields listed
one after another, rather than one binary .raw file per view.

Run with the repo venv (no CUDA_HOME needed -- this script pins mlx to CPU):
    /home/schneider/TrajectoryOptimization/.venv/bin/python \
        RealWorldExample_ConOpt/plan_trajectory.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

# This machine's mlx-cuda build (a preview CUDA port of Apple's MLX) silently
# returns wrong results for this workload and crashes on mx.value_and_grad.
# All arrays here are tiny (<=few hundred elements/axis), so pin to the
# mature CPU backend instead of chasing the experimental CUDA one.
mx.set_default_device(mx.cpu)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # for reconstruct_ezrt_cuda.py and EZRT_Helpers

import reconstruct_ezrt_cuda as rec                              # noqa: E402
from EZRT_Helpers.rek2py import rek2py                            # noqa: E402
from EZRT_Helpers.ezrt_header import EzrtHeader                   # noqa: E402
from differentiable_coverage import (                             # noqa: E402
    TwoAxisGantry, ScoreConfig, sample_unit_sphere, saturated_coverage, adam_ascent,
)
from differentiable_coverage.absorption import _detector_frame    # noqa: E402
from differentiable_coverage.absorption_bundle import (           # noqa: E402
    BundleAbsorptionConfig, bundle_path_integral,
)
from differentiable_coverage.vcl_diff import (                    # noqa: E402
    build_vcl_context, vcl_loss_continuous,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# PLAN_FDK_PATH / PLAN_OUT_DIR / EZRT_DATA_DIR (the last one read by
# reconstruct_ezrt_cuda itself) let this script be re-pointed at a different
# scan/reconstruction without editing the file -- same convention reconstruct_ezrt_cuda.py
# already uses for its own EZRT_DATA_DIR/EZRT_OUT_DIR.
FDK_PATH = Path(os.environ.get(
    "PLAN_FDK_PATH",
    "/home/schneider/DiffCT_CUDA_Development/TestReconstructions/output/reconstruction_FDK.rek",
))
OUT_DIR = Path(os.environ.get("PLAN_OUT_DIR", str(HERE / "planned_trajectory")))

PHI_MAX = math.radians(30.0)   # elevation half-band -> 60 deg-wide band total
                               # (was 35 deg until 2026-07-15; colleague: 35 is
                               # beyond the rig's safe envelope)

# Active objective, switchable per run via PLAN_OBJECTIVE:
#   "coverage" -> C_geo only
#   "bundle"   -> C_geo + lambda_bundle * L_bundle   (absorption penalty)
#   "vcl"      -> C_geo + lambda_info * I_vcl        (default, as before)
#   "all"      -> C_geo + lambda_info * I_vcl + lambda_bundle * L_bundle
#                 (the paper's full composite, eq. loss_combined)
OBJECTIVE = os.environ.get("PLAN_OBJECTIVE", "vcl").strip().lower()
if OBJECTIVE not in ("coverage", "bundle", "vcl", "all"):
    raise SystemExit(f"PLAN_OBJECTIVE must be coverage|bundle|vcl|all, got {OBJECTIVE!r}")
N_RADON = int(os.environ.get("PLAN_N_RADON", "256"))
                               # candidate Radon-plane normals for the coverage
                               # score. Matching rule (empirically confirmed
                               # 2026-07-15): the optimiser overfits the lattice
                               # unless dgamma >= 0.5*sqrt(4*pi/z); z=256 with
                               # 4 deg tolerance overestimated coverage by up to
                               # ~8% at k=50 (re-scored on z=2000). z=2000 is
                               # safely matched for dgamma >= ~2.5 deg.
PLAN_TAU = float(os.environ.get("PLAN_TAU", "0.07"))
                               # coverage kernel: tau acts as sin(dgamma) in
                               # ScoreConfig -> 0.07 = 4 deg tolerance.
ROI_MAX_POINTS = 120           # FDK-intensity-weighted ROI points (kept small: Python-loop cost)
ADAM_STEPS = 150
RANDOM_SEED = 0
VIEW_COUNTS = (50, 100, 200, 300, 400)

VCL_TARGET_SHAPE = 128         # paper default: VCL builds its view-basis model on a 128^3 volume
LAMBDA_INFO = 1.0              # I_vcl is already bounded in [0,1] like C_geo -- no calibration needed


# --------------------------------------------------------------------------- #
# Real scan geometry (reuses reconstruct_ezrt_cuda.py verbatim)
# --------------------------------------------------------------------------- #
def real_scan_geometry():
    """FOD / FDD from a sparse sample of real circular_1200 views."""
    _, geom_raw, _ = rec.load_ezrt_dataset(rec.DATA_DIR, detector_bin=32, view_stride=100)
    src, det_c, _, _, iso = rec.build_geometry(geom_raw)
    fod = float(np.mean(np.linalg.norm(src, axis=1)))
    fdd = float(np.mean(np.linalg.norm(det_c - src, axis=1)))
    return fod, fdd, iso


# --------------------------------------------------------------------------- #
# ROI (coverage target) from the FDK reconstruction
# --------------------------------------------------------------------------- #
def load_mu_volume(path: Path):
    """Load a .rek attenuation volume as (mx.array (nz,ny,nx), voxel_mm)."""
    header, vol = rek2py(path, switch_order=True)
    voxel_mm = header.voxel_size_x_in_um / 1000.0
    return mx.array(vol, dtype=mx.float32), voxel_mm


def roi_from_fdk(path: Path, stride: int = 16, max_points: int = ROI_MAX_POINTS):
    """Intensity-weighted ROI point cloud (world mm, isocentre-centred) from the FDK volume."""
    header, vol = rek2py(path, switch_order=True)   # (nz, ny, nx)
    voxel_mm = header.voxel_size_x_in_um / 1000.0
    n = vol.shape[0]

    small = vol[::stride, ::stride, ::stride]
    thresh = np.percentile(small, 95.0)
    iz, iy, ix = np.nonzero(small > thresh)
    weights = small[iz, iy, ix].astype(np.float64)

    origin = -0.5 * n * voxel_mm
    spacing = stride * voxel_mm
    pts = np.stack([ix, iy, iz], axis=1) * spacing + origin  # (x, y, z) world mm

    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(int)
        pts, weights = pts[idx], weights[idx]
    weights = weights / weights.sum()
    center = (pts * weights[:, None]).sum(axis=0)
    return (
        mx.array(pts, dtype=mx.float32),
        mx.array(weights, dtype=mx.float32),
        mx.array(center, dtype=mx.float32),
    )


# --------------------------------------------------------------------------- #
# Bundle absorption penalty (paper Sec. "Analytic Bundle Absorption Penalty")
# --------------------------------------------------------------------------- #
def calibrate_lambda_bundle(mu_volume, sid: float, bundle_cfg: BundleAbsorptionConfig,
                             n_probe: int = 256, seed: int = 0) -> float:
    """lambda_bundle = 0.2 / median(tau_bar) over n_probe uniform sphere probes.

    Matches the paper's auto-calibration recipe exactly (Sec. "Hyperparameters"):
    picks lambda so the bundle penalty's typical magnitude matches the (bounded
    [0,1]) coverage term, using the same probe-sampling idea as the library's own
    ``calibrate_bundle_alpha`` (median over a sphere of candidate sources), just
    solved for the additive penalty's weight instead of the saturating gate's alpha.
    """
    mx.random.seed(seed)
    probes = sample_unit_sphere(n_probe) * sid  # centred on the world origin/isocentre
    tau = bundle_path_integral(probes, mx.zeros(3), mu_volume, bundle_cfg)
    med_tau = float(mx.median(tau))
    return 0.2 / med_tau if med_tau > 0.0 else 1.0


def plot_overview(results, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3.6), sharey=True)
    for ax, (n_views, theta, phi) in zip(axes, results):
        ax.scatter(np.degrees(np.asarray(theta)) % 360.0, np.degrees(np.asarray(phi)), s=10, color="C0")
        phi_deg = math.degrees(PHI_MAX)
        ax.axhline(phi_deg, color="r", ls="--", lw=0.8)
        ax.axhline(-phi_deg, color="r", ls="--", lw=0.8)
        ax.set_xlim(0, 360)
        ax.set_title(f"N={n_views}")
        ax.set_xlabel("theta (deg)")
    axes[0].set_ylabel("phi (deg)")
    fig.suptitle(f"Independently optimised sources per view count "
                 f"(+-{math.degrees(PHI_MAX):.0f} deg band)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# EZRT export
# --------------------------------------------------------------------------- #
def export_ezrt(n_views, theta, phi, sources, det_center, det_u, det_v):
    view_dir = OUT_DIR / f"N{n_views:04d}"
    view_dir.mkdir(parents=True, exist_ok=True)

    src_np = np.asarray(sources)
    det_c_np = np.asarray(det_center)
    det_u_np = np.asarray(det_u)
    det_v_np = np.asarray(det_v)
    theta_np, phi_np = np.asarray(theta), np.asarray(phi)

    def _fmt3(v):
        return f"({float(v[0]):.6f}, {float(v[1]):.6f}, {float(v[2]):.6f})"

    rows = []
    blocks = []
    for i in range(n_views):
        h = EzrtHeader()  # image_width=image_height=0, number_of_images=1 -> pure geometry header
        # Dataset-specific AGV convention (see reconstruct_ezrt_cuda.build_geometry):
        # positions stored in METRES, orientation vectors stored NEGATED.
        h.agv_source_position = tuple((src_np[i] / 1000.0).tolist())
        h.agv_detector_center_position = tuple((det_c_np[i] / 1000.0).tolist())
        h.agv_detector_line_direction = tuple((-det_u_np[i]).tolist())
        h.agv_detector_col_direction = tuple((-det_v_np[i]).tolist())
        h.focus_object_distance_in_mm = float(np.linalg.norm(src_np[i]))
        h.focus_detector_distance_in_mm = float(np.linalg.norm(det_c_np[i] - src_np[i]))
        h.number_projection_angles = n_views

        blocks.append(
            f"=== view {i:04d} ===\n"
            f"theta_deg = {math.degrees(theta_np[i]) % 360.0:.4f}\n"
            f"phi_deg = {math.degrees(phi_np[i]):.4f}\n"
            f"agv_source_position_m = {_fmt3(h.agv_source_position)}\n"
            f"agv_detector_center_position_m = {_fmt3(h.agv_detector_center_position)}\n"
            f"agv_detector_line_direction = {_fmt3(h.agv_detector_line_direction)}\n"
            f"agv_detector_col_direction = {_fmt3(h.agv_detector_col_direction)}\n"
            f"focus_object_distance_in_mm = {h.focus_object_distance_in_mm:.4f}\n"
            f"focus_detector_distance_in_mm = {h.focus_detector_distance_in_mm:.4f}\n"
            f"number_projection_angles = {h.number_projection_angles}\n"
        )
        rows.append((i, math.degrees(theta_np[i]) % 360.0, math.degrees(phi_np[i]), *src_np[i]))

    (view_dir / "trajectory_headers.txt").write_text("\n".join(blocks))

    np.savetxt(
        view_dir / "trajectory_coords.csv", rows, delimiter=",",
        header="view,theta_deg,phi_deg,src_x_mm,src_y_mm,src_z_mm", comments="",
        fmt=("%d", "%.4f", "%.4f", "%.4f", "%.4f", "%.4f"),
    )
    print(f"  N={n_views:4d}: wrote trajectory_headers.txt ({n_views} views) + trajectory_coords.csv -> {view_dir}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=== real scan geometry (circular_1200) ===")
    fod, fdd, iso = real_scan_geometry()
    print(f"FOD={fod:.2f} mm  FDD={fdd:.2f} mm  isocentre(mm)={iso}")

    print("\n=== ROI from FDK reconstruction ===")
    roi_points, roi_weights, roi_center = roi_from_fdk(FDK_PATH)
    print(f"ROI points={roi_points.shape[0]}  weighted center(mm)={roi_center.tolist()}")

    mu_volume, voxel_mm = load_mu_volume(FDK_PATH)

    # Bundle-absorption penalty: always calibrated (cheap, useful diagnostic),
    # active in the objective only for PLAN_OBJECTIVE=bundle.
    # Quadrature: clip_to_volume=True, n_samples=512 (corrected rule). The
    # camera-era exports used the legacy full-segment@32 rule, which the
    # bundle_quadrature convergence study showed converges to the wrong value
    # (31% value / 88 deg gradient error on the camera prior, lambda_bundle
    # ~1.7x too weak); clip@512 is the recommendation of that study for
    # future planning (clip@256 marginally missed the 5 deg gradient
    # tolerance on real-recon data). First used for the Pinterguss plans.
    bundle_cfg = BundleAbsorptionConfig(
        voxel_spacing=voxel_mm, n_samples=512, clip_to_volume=True
    )
    lambda_bundle = calibrate_lambda_bundle(mu_volume, fod, bundle_cfg)
    bundle_tag = "active" if OBJECTIVE in ("bundle", "all") else "inactive"
    print(f"[{bundle_tag}] lambda_bundle={lambda_bundle:.4f} (0.2 / median tau_bar over 256 sphere probes)")

    vcl_ctx = None
    if OBJECTIVE in ("vcl", "all"):
        print("\n=== VCL info term (paper eq. I_vcl) ===")
        # Downsampled reference volume + fixed voxel sub-sample (r1=1e-3 default),
        # scaled to match our real geometry's magnification at the smaller resolution.
        n_full = mu_volume.shape[0]
        mag = fdd / fod
        voxel_vcl = (n_full * voxel_mm) / VCL_TARGET_SHAPE
        du_vcl = dv_vcl = voxel_vcl * mag
        vcl_ctx = build_vcl_context(
            mu_volume, sid=fod, sdd=fdd,
            det_shape=(VCL_TARGET_SHAPE, VCL_TARGET_SHAPE), du=du_vcl, dv=dv_vcl,
            voxel_spacing=voxel_vcl, target_shape=(VCL_TARGET_SHAPE,) * 3,
        )
        print(f"VCL context built: {VCL_TARGET_SHAPE}^3 reference volume, "
              f"lambda_info={LAMBDA_INFO} (I_vcl is already in [0,1], like C_geo -- no calibration)")
    print(f"\nactive objective (PLAN_OBJECTIVE={OBJECTIVE}): "
          + {"coverage": "C_geo only",
             "bundle": "C_geo + lambda_bundle * L_bundle",
             "vcl": "C_geo + lambda_info * I_vcl",
             "all": "C_geo + lambda_info * I_vcl + lambda_bundle * L_bundle"}[OBJECTIVE])

    # Azimuth: full free rotation, no constraint needed (periodic via sin/cos).
    # Elevation: kept in +-35 deg by a smooth tanh squashing rather than a hard
    # clamp -- same achievable coverage, no sources stuck exactly on the rail
    # (see module docstring).
    gantry = TwoAxisGantry(sid=fod)
    radon = sample_unit_sphere(N_RADON)
    score_cfg = ScoreConfig(tau=PLAN_TAU)
    mx.random.seed(RANDOM_SEED)

    def sources_from_params(params_raw):
        theta = params_raw[:, 0]
        phi = PHI_MAX * mx.tanh(params_raw[:, 1])
        return gantry(mx.stack([theta, phi], axis=-1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== per-k random init + continuous optimisation, EZRT export {VIEW_COUNTS} ===")
    results = []
    for n_views in VIEW_COUNTS:
        theta0 = mx.random.uniform(shape=(n_views,)) * (2.0 * math.pi)
        phi0_raw = mx.random.uniform(shape=(n_views,)) * 2.0 - 1.0
        params0 = mx.stack([theta0, phi0_raw], axis=-1)
        nu = mx.ones((n_views,))

        def coverage_fn(params_raw, step):
            sources = sources_from_params(params_raw)
            cov = saturated_coverage(sources, roi_center, radon, nu, score_cfg,
                                      roi_points=roi_points, roi_weights=roi_weights)
            obj = cov
            if OBJECTIVE in ("vcl", "all"):
                # I_vcl = 1 - vcl_loss_continuous(...); the paper's "clean" (non-
                # noise-weighted) View Covariance Loss information score, in [0,1].
                obj = obj + LAMBDA_INFO * (1.0 - vcl_loss_continuous(sources, vcl_ctx))
            if OBJECTIVE in ("bundle", "all"):
                # Raw (unbounded) bundle-mean attenuation penalty -- NOT the
                # saturating gate (its gradient vanishes on high-absorption
                # directions, see the paper). Volume is isocentre-centred ->
                # bundle target must be mx.zeros(3), not the weighted roi_center.
                tau_bar = bundle_path_integral(sources, mx.zeros(3), mu_volume, bundle_cfg)
                obj = obj + lambda_bundle * (-mx.mean(tau_bar))
            return obj

        init_cov = float(coverage_fn(params0, 0))
        params_opt, _ = adam_ascent(coverage_fn, params0, lr=0.05, n_steps=ADAM_STEPS)
        final_cov = float(coverage_fn(params_opt, 0))

        theta = params_opt[:, 0]
        phi = PHI_MAX * mx.tanh(params_opt[:, 1])
        sources = sources_from_params(params_opt)
        det_center, det_u, det_v = _detector_frame(sources, mx.zeros(3), fdd)
        # Diagnostics: bundle-mean always (cheap); I_vcl only when its context exists.
        tau_bar_final = float(mx.mean(bundle_path_integral(sources, mx.zeros(3), mu_volume, bundle_cfg)))
        vcl_note = ""
        if vcl_ctx is not None:
            vcl_note = f"I_vcl={float(1.0 - vcl_loss_continuous(sources, vcl_ctx)):.4f}, "

        print(f"  N={n_views:4d}: objective random-init={init_cov:.4f}  ->  optimised={final_cov:.4f}"
              f"   ({vcl_note}mean tau_bar={tau_bar_final:.4f} [{bundle_tag}])")
        export_ezrt(n_views, theta, phi, sources, det_center, det_u, det_v)
        results.append((n_views, theta, phi))

    plot_overview(results, OUT_DIR / "trajectory_overview.png")
    print(f"\nDone. Planned trajectories written to {OUT_DIR}")


if __name__ == "__main__":
    main()
