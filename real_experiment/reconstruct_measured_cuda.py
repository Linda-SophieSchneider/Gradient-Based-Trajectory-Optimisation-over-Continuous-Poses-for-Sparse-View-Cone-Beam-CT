#!/usr/bin/env python
"""Cone-beam reconstruction of raw scanner (.raw) projections with diffct-mlx on CUDA.

Dataset: TrajektorienOptimierung / Kamera / circular_1200
    * 1200 projections, 3072 x 3072, 16-bit, vendor raw format with per-view
      arbitrary-geometry vectors (AGV) stored in the 2048-byte header.

Two dataset-specific corrections were provided by the data owner and are applied
below (see build_geometry):
    1. AGV positions are stored in METRES, not micrometres -> scale by 1000 to mm.
    2. Both detector orientation vectors (line & column) must be NEGATED.

Pipeline:
    load .raw + AGV geometry  ->  -log normalisation  ->  recenter to isocentre
      ->  (A) FDK  (cone-beam FBP, analytic)      -> saved as .rek
      ->  (B) SIRT (iterative, footprint model)   -> saved as .rek

Run with the CUDA test venv, e.g.:
    /tmp/.../ct_venv/bin/python TestReconstructions/reconstruct_measured_cuda.py
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent

# Location of the extracted projection folder (contains projection_XXXX.raw).
DATA_DIR = Path(
    os.environ.get(
        "SCAN_DATA_DIR",
        "/ssd_data/diffct_scratch/TrajektorienOptimierung/Kamera/circular_1200",
    )
)

# Where the reconstructed .rek volumes and preview PNGs are written.
OUT_DIR = Path(os.environ.get("SCAN_OUT_DIR", str(Path(__file__).resolve().parent / "output")))

# Detector down-sampling (mean pooling). 4 -> 768x768 detector, ~0.28 mm voxels,
# full 213 mm field of view fits a 768^3 volume comfortably on a 96 GB GPU.
DETECTOR_BIN = int(os.environ.get("SCAN_DETECTOR_BIN", "4"))

# Use every Nth view. 1 = all 1200 views.
VIEW_STRIDE = int(os.environ.get("SCAN_VIEW_STRIDE", "1"))

# Reconstruction volume (nz, ny, nx) = (z/height, y, x). None -> cube sized to
# the down-sampled detector width.
VOLUME_SHAPE = None

# SIRT outer iterations. Default 0 = skip: plain SIRT is severely
# under-converged at feasible iteration counts (30 iters -> blooming halos,
# max/p99.9 only 1.7 vs FDK's 6.0 on the prescan) and no longer plays a role
# in the pipeline (FDK is the reference/prior, SART reconstructs the arms).
SIRT_ITERS = int(os.environ.get("SCAN_SIRT_ITERS", "0"))

# Dataset-specific corrections (from the data owner).
POSITIONS_IN_METRES = True          # scale AGV positions by 1000 -> mm
NEGATE_DETECTOR_ORIENTATIONS = True  # negate both line & column direction vectors

# raw2py returns each projection indexed [v, u] = [row/vertical, col/horizontal]
# (verified: axis0 is stationary across views = rotation axis = det_v; axis1
# oscillates = in-plane = det_u).  The projector expects (det_u, det_v), so each
# projection must be transposed.  This mirrors transpose_uv=True in the library's
# MeasuredConeDataConfig.
TRANSPOSE_UV = True

# --------------------------------------------------------------------------- #
# Imports that depend on the environment
# --------------------------------------------------------------------------- #
# Vendor-specific raw-projection I/O helpers (raw2py, header, py2rek) are not
# included in this release; provide your own `scanner_io` module exposing
# equivalent functions/classes for your scanner's raw format.
sys.path.insert(0, str(REPO_ROOT))  # for the scanner_io helpers

from scanner_io.raw2py import raw2py              # noqa: E402
from scanner_io.header import ScanHeader          # noqa: E402
from scanner_io.py2rek import py2rek              # noqa: E402

import diffct_mlx as dct                          # noqa: E402
from diffct_mlx.backend import active as _b       # noqa: E402
from diffct_mlx.real_measured_data_helper import estimate_cone_isocenter  # noqa: E402

xp = _b.xp


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.findall(r"\d+|\D+", path.name)]


def _bin_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool a 2D image by an integer factor (centre-cropped to a multiple)."""
    if factor <= 1:
        return image.astype(np.float32, copy=False)
    h, w = image.shape
    eh, ew = (h // factor) * factor, (w // factor) * factor
    oh, ow = (h - eh) // 2, (w - ew) // 2
    image = image[oh:oh + eh, ow:ow + ew]
    return image.reshape(eh // factor, factor, ew // factor, factor).mean(axis=(1, 3), dtype=np.float32)


def _load_projection(path, factor):
    """Load one .raw, bin it, and return it as (det_u, det_v)."""
    header, image = raw2py(path)          # image is (v, u) = (vertical/row, horizontal/col)
    binned = _bin_mean(image, factor)
    if TRANSPOSE_UV:
        binned = np.ascontiguousarray(binned.T)   # -> (det_u, det_v)
    return header, binned


def load_measured_dataset(data_dir: Path, detector_bin: int, view_stride: int):
    """Load raw scanner projections + per-view AGV geometry.

    Returns
    -------
    sino_raw : (n_views, det_u, det_v) float32   -- raw (binned) intensities
    geom     : dict with src_pos/det_center/det_u_vec/det_v_vec  (n_views, 3)
    meta     : dict with pixel pitch, detector shape, etc.
    """
    raw_files = sorted(data_dir.glob("*.raw"), key=_natural_key)
    if not raw_files:
        raise FileNotFoundError(f"No .raw projections found in {data_dir}")
    raw_files = raw_files[::view_stride]
    n_views = len(raw_files)

    # Probe the first header for detector geometry.
    h0, img0 = _load_projection(raw_files[0], detector_bin)  # (det_u, det_v)
    det_u_count, det_v_count = img0.shape

    # Detector pixel pitch [mm].  pixel_width_in_um is 0 in these headers, so
    # derive it from the physical detector width.
    if h0.pixel_width_in_um and h0.pixel_width_in_um > 0.0:
        pitch_um = float(h0.pixel_width_in_um)
    else:
        pitch_um = float(h0.detector_width_in_um) / float(h0.number_horizontal_pixels)
    du = dv = (pitch_um / 1000.0) * detector_bin  # mm

    sino_raw = np.empty((n_views, det_u_count, det_v_count), dtype=np.float32)
    src_pos = np.empty((n_views, 3), dtype=np.float64)
    det_center = np.empty((n_views, 3), dtype=np.float64)
    det_line = np.empty((n_views, 3), dtype=np.float64)   # -> detector u axis
    det_col = np.empty((n_views, 3), dtype=np.float64)    # -> detector v axis

    print(f"Loading {n_views} projections from {data_dir}")
    t0 = time.time()
    for i, path in enumerate(raw_files):
        header, image = _load_projection(path, detector_bin)
        sino_raw[i] = image
        src_pos[i] = np.asarray(header.agv_source_position, dtype=np.float64)
        det_center[i] = np.asarray(header.agv_detector_center_position, dtype=np.float64)
        det_line[i] = np.asarray(header.agv_detector_line_direction, dtype=np.float64)
        det_col[i] = np.asarray(header.agv_detector_col_direction, dtype=np.float64)
        if (i + 1) % 200 == 0 or i == n_views - 1:
            print(f"  {i + 1}/{n_views}  ({time.time() - t0:.1f}s)")

    geom = dict(src_pos=src_pos, det_center=det_center, det_line=det_line, det_col=det_col)
    meta = dict(
        detector_shape=(det_u_count, det_v_count),
        du=du, dv=dv,
        raw_detector_shape=(int(h0.number_horizontal_pixels), int(h0.number_vertical_pixels)),
        detector_bin=detector_bin,
    )
    return sino_raw, geom, meta


def build_geometry(geom: dict):
    """Apply the dataset-specific corrections and recenter to the isocentre.

    Corrections
    -----------
    * positions metres -> mm  (x1000)
    * negate both detector orientation vectors (line & column)
    """
    src = geom["src_pos"].copy()
    det_c = geom["det_center"].copy()
    det_u = geom["det_line"].copy()   # line direction -> detector u axis
    det_v = geom["det_col"].copy()    # column direction -> detector v axis

    if POSITIONS_IN_METRES:
        src *= 1000.0
        det_c *= 1000.0

    if NEGATE_DETECTOR_ORIENTATIONS:
        det_u = -det_u
        det_v = -det_v

    # Normalise orientation vectors (they are already ~unit, but be safe).
    det_u /= np.linalg.norm(det_u, axis=1, keepdims=True)
    det_v /= np.linalg.norm(det_v, axis=1, keepdims=True)

    # Recenter so the rotation isocentre sits at the volume origin (0,0,0),
    # which is where diffct reconstructs.
    iso = estimate_cone_isocenter(src.astype(np.float32), det_c.astype(np.float32)).astype(np.float64)
    src -= iso
    det_c -= iso

    return (
        src.astype(np.float32),
        det_c.astype(np.float32),
        det_u.astype(np.float32),
        det_v.astype(np.float32),
        iso,
    )


def log_normalize(sino_raw: np.ndarray, air_border_px: int = 24) -> np.ndarray:
    """Convert transmission intensities to line integrals: p = -log(I / I0).

    I0 is estimated per view from the bright detector border (air) region.
    """
    stack = sino_raw
    n = stack.shape[0]
    b = air_border_px
    border = np.concatenate(
        [
            stack[:, :b, :].reshape(n, -1),
            stack[:, -b:, :].reshape(n, -1),
            stack[:, :, :b].reshape(n, -1),
            stack[:, :, -b:].reshape(n, -1),
        ],
        axis=1,
    )
    i0 = np.percentile(border, 99.0, axis=1).reshape(n, 1, 1)
    i0 = np.maximum(i0, 1.0)
    p = -np.log(np.clip(stack / i0, 1e-6, 1.0))
    return p.astype(np.float32)


def save_rek(volume_np: np.ndarray, path: Path, voxel_size_um: float):
    """Save a (nz, ny, nx) float32 volume as a .rek file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = volume_np.shape
    vol = np.ascontiguousarray(volume_np, dtype=np.float32)
    header = ScanHeader(
        image_width=nx,
        image_height=ny,
        bit_depth=ScanHeader.convert_to_bitdepth(np.float32),
        number_of_images=nz,
        number_voxels=(nx, ny, nz),
    )
    header.voxel_size_x_in_um = float(voxel_size_um)
    header.voxel_size_z_in_um = float(voxel_size_um)
    # py2rek(switch_order=True) expects (#images, H, W) = (nz, ny, nx).
    py2rek(vol, path, input_header=header, switch_order=True)
    print(f"  saved {path}  ({vol.nbytes / 1e9:.2f} GB, {volume_np.shape} float32)")


def save_slice_pngs(volume_np: np.ndarray, stem: Path):
    """Save central axial/coronal/sagittal slices for a quick visual check."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  (skipping PNG preview: {exc})")
        return
    nz, ny, nx = volume_np.shape
    lo, hi = np.percentile(volume_np, [1.0, 99.5])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, sl) in zip(
        axes,
        [("axial z", volume_np[nz // 2]),
         ("coronal y", volume_np[:, ny // 2]),
         ("sagittal x", volume_np[:, :, nx // 2])],
    ):
        ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    out = stem.with_suffix(".png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  preview {out}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print(f"diffct backend: {dct.backend}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- load ------------------------------------------------------------- #
    sino_raw, geom_raw, meta = load_measured_dataset(DATA_DIR, DETECTOR_BIN, VIEW_STRIDE)
    n_views, det_u_count, det_v_count = sino_raw.shape
    du = meta["du"]
    dv = meta["dv"]
    print(f"detector: {det_u_count} x {det_v_count} (bin {DETECTOR_BIN}), "
          f"pitch du=dv={du:.4f} mm, {n_views} views")

    # ---- geometry (corrections + recenter) -------------------------------- #
    src, det_c, det_u_vec, det_v_vec, iso = build_geometry(geom_raw)

    diag = dct.diagnose_cone_geometry(src, det_c, det_u_vec, det_v_vec)
    fod = float(np.mean(np.linalg.norm(src, axis=1)))                 # source -> isocentre
    fdd = float(np.mean(np.linalg.norm(det_c - src, axis=1)))         # source -> detector
    mag = fdd / fod
    voxel_mm = du / mag
    print(f"isocentre (mm): {iso}")
    print(f"FOD (SID) = {fod:.2f} mm, FDD (SDD) = {fdd:.2f} mm, magnification = {mag:.4f}")
    print(f"voxel size = {voxel_mm:.4f} mm")
    print(f"geometry diagnostics: sid[{diag['sid_min_mm']:.1f},{diag['sid_max_mm']:.1f}] "
          f"sdd[{diag['sdd_min_mm']:.1f},{diag['sdd_max_mm']:.1f}] "
          f"u.v={diag['det_u_dot_det_v_max_abs']:.2e} "
          f"u.ray={diag['det_u_dot_ray_max_abs']:.2e} v.ray={diag['det_v_dot_ray_max_abs']:.2e} "
          f"ray.-src={diag['ray_vs_minus_source_mean']:.4f} "
          f"ray_to_iso_max={diag['ray_to_isocenter_max_mm']:.3f} mm")

    if VOLUME_SHAPE is None:
        n = det_u_count
        volume_shape = (n, n, n)
    else:
        volume_shape = VOLUME_SHAPE
    nz, ny, nx = volume_shape
    print(f"volume: {volume_shape}, voxel_spacing = {voxel_mm:.4f} mm "
          f"-> FOV {nx * voxel_mm:.1f} x {ny * voxel_mm:.1f} x {nz * voxel_mm:.1f} mm")

    # ---- -log normalisation ---------------------------------------------- #
    print("log-normalising projections ...")
    sino = log_normalize(sino_raw)
    del sino_raw
    print(f"sinogram range after -log: [{sino.min():.4f}, {sino.max():.4f}]")

    # Move geometry + sinogram to the active backend (GPU).
    src_d = xp.array(src)
    det_c_d = xp.array(det_c)
    det_u_d = xp.array(det_u_vec)
    det_v_d = xp.array(det_v_vec)
    sino_d = xp.array(sino, dtype=_b.float32)

    detector_shape = (det_u_count, det_v_count)

    # ===================================================================== #
    # (A) FDK  --  cone-beam filtered back-projection (analytic)
    # ===================================================================== #
    print("\n=== (A) FDK (cone-beam FBP, quantitative diffct >= 2.1.0 path) ===")
    t0 = time.time()

    # Quantitative FDK trio (diffct-mlx >= 2.1.0): cosine pre-weights,
    # per-view trapezoidal angular weights, physical zero-padded ramp
    # (|f|/du), voxel-driven (sid/U)^2 gather backprojection with the
    # analytical sdd/(2*pi*sid) constant. Amplitude-true mu in 1/mm, no
    # hand-rolled normalization constant (the previous local constant was
    # radiometrically wrong by a geometry-dependent factor -- see
    # fdk_scale_check.py and the diffct CHANGELOG "Notes").
    # sinogram_scale=1.0: our -log data is already a physical line integral.
    from diffct_mlx.reconstruction_algorithms.cases import _quantitative_fdk_operators

    q_weight, q_filter, q_back = _quantitative_fdk_operators(
        src_d, det_c_d, det_u_d, det_v_d,
        volume_shape=volume_shape, detector_shape=detector_shape,
        du=du, dv=dv, voxel_spacing=voxel_mm, sinogram_scale=1.0,
    )
    if q_back is None:
        raise RuntimeError("quantitative FDK path unavailable on this backend")
    fdk_params = dct.FDKParameters(
        normalization_scale=1.0,
        enforce_positivity=True,
    )
    fdk_volume = dct.reconstruct_fdk(
        sino_d, q_back, fdk_params,
        weight_projections=q_weight, filter_projections=q_filter,
    )
    fdk_np = np.asarray(_b.to_numpy(fdk_volume), dtype=np.float32)
    print(f"FDK done in {time.time() - t0:.1f}s, "
          f"range [{fdk_np.min():.5f}, {fdk_np.max():.5f}] mm^-1")
    save_rek(fdk_np, OUT_DIR / "reconstruction_FDK.rek", voxel_mm * 1000.0)
    save_slice_pngs(fdk_np, OUT_DIR / "reconstruction_FDK")
    del fdk_volume

    # ===================================================================== #
    # (B) SIRT  --  iterative (separable-footprint forward/adjoint)
    # ===================================================================== #
    if SIRT_ITERS <= 0:
        print("\n=== (B) SIRT skipped (SCAN_SIRT_ITERS=0) ===")
        print(f"\nDone. Reconstructions written to {OUT_DIR}")
        return

    print(f"\n=== (B) SIRT ({SIRT_ITERS} iterations) ===")
    t0 = time.time()

    forward_single, back_single, _ = dct.make_cone_3d_operators(
        src_d, det_c_d, det_u_d, det_v_d,
        volume_shape=volume_shape, detector_shape=detector_shape,
        du=du, dv=dv, voxel_spacing=voxel_mm,
        projector_mode="footprint",
    )

    sirt_params = dct.SIRTParameters(
        volume_shape=volume_shape,
        iteration_count=SIRT_ITERS,
        enforce_positivity=True,
        voxel_extreme_values=(0.0, float("inf")),
    )
    # SIRT expects a sequence of per-view projections (each (det_u, det_v)).
    measured_views = [sino_d[i] for i in range(n_views)]
    sirt_volume = dct.reconstruct_sirt(
        measured_views, forward_single, back_single, sirt_params, show_progress=True
    )
    sirt_np = np.asarray(_b.to_numpy(sirt_volume), dtype=np.float32)
    print(f"SIRT done in {time.time() - t0:.1f}s, "
          f"range [{sirt_np.min():.5f}, {sirt_np.max():.5f}] mm^-1")
    save_rek(sirt_np, OUT_DIR / "reconstruction_SIRT.rek", voxel_mm * 1000.0)
    save_slice_pngs(sirt_np, OUT_DIR / "reconstruction_SIRT")

    print(f"\nDone. Reconstructions written to {OUT_DIR}")


if __name__ == "__main__":
    main()
