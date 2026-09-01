"""Render Figures 1 and 2 of the paper (MILP slice grid + difference maps).

Visual style adapted from MILP_Opt_Project/build_reconstruction_comparison_figures.py:
  - PowerNorm(gamma=0.55) for perceptually balanced display
  - Percentile-based contrast windowing (98th/55th), per-row calibrated to reference
  - Center-crop ROI inset (38 % x 38 %, lower-right, white border)
  - Row labels via fig.text, tight_layout(rect=[0.07, ...])
  - Saved as .pdf + .png at dpi=240

Usage::

    python experiments/render_milp_slices.py \
        --config experiments/configs/paper1/main/milp_slices.yaml
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffct")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from differentiable_coverage.eval.geometry import geometry_from_sources
from differentiable_coverage.eval.reco import (
    reconstruct_sart_volume, simulate_sinogram,
)
from differentiable_coverage.eval.trajectories import build_baseline_sources
from differentiable_coverage.eval.vcl import compute_R_gamma
from differentiable_coverage.figures.paper_style import render_reconstruction_plate
from differentiable_coverage.score import sample_unit_sphere

from experiments.run import (  # type: ignore
    _load_phantom, _resolve_geometry, _load_mlx_stack,
    _resolve_roi_context, _method_kwargs_with_roi, _method_roi_center,
    _method_optimizes_roi,
)

# ---------------------------------------------------------------------------
# Display constants (mirrors MILP_opt_project style)
# ---------------------------------------------------------------------------

DISPLAY_GAMMA = 0.55          # PowerNorm exponent: brightens low-intensity structures
DISPLAY_PERCENTILE = 98.0     # vmax = this percentile of positive reference pixels
DISPLAY_BLACK_PERCENTILE = 55.0  # vmin = this percentile of positive reference pixels (soft black point)
INSET_CROP_FRACTION = 0.40    # center crop: keep middle 40 % of each axis for inset


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _positive_values(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32).ravel()
    a = a[np.isfinite(a) & (a > 0.0)]
    return a


def _best_center_slice_index(volume: np.ndarray) -> int:
    """Use the true centre slice for the MILP paper figure.

    The previous ``argmax(sum(volume**2))`` selection was dominated by bright
    absorber blocks away from the object body and therefore hid the actual
    part geometry. The paper comparison in MILP_Opt_Project uses the true
    full-reconstruction centre view for this kind of display, which is also
    the correct slice here.
    """
    return int(np.asarray(volume).shape[0] // 2)


def _contrast_window(
    reference_slice: np.ndarray,
    *,
    percentile: float = DISPLAY_PERCENTILE,
    black_percentile: float = DISPLAY_BLACK_PERCENTILE,
) -> tuple[float, float]:
    """Percentile-based (vmin, vmax) anchored to the reference slice.

    Mirrors MILP_opt_project._contrast_window / _window logic:
    vmin is the soft black point, vmax is the display white point.
    """
    pos = _positive_values(reference_slice)
    if pos.size == 0:
        return 0.0, 0.17
    vmin = float(np.percentile(pos, black_percentile)) if black_percentile > 0.0 else 0.0
    vmax = float(np.percentile(pos, percentile))
    # Degenerate near-binary references (sparse metal phantoms: background 0,
    # object at a single high value) collapse the soft black point onto the
    # white point, which clips every reconstruction below it to black.  Detect
    # that case and drop the black point to 0 so the reconstructions show.
    if vmin >= 0.5 * vmax:
        vmin = 0.0
    if not math.isfinite(vmin) or vmin < 0.0:
        vmin = 0.0
    if not math.isfinite(vmax) or vmax <= vmin:
        vmax = max(vmin + 1e-6, 0.17)
    return vmin, vmax


def _center_crop(
    image: np.ndarray,
    fraction: float = INSET_CROP_FRACTION,
    center: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """Extract a ``fraction``-sized window of a 2-D image for the ROI inset.

    ``center`` is the (row, col) crop centre as fractions in [0, 1];
    the default (0.5, 0.5) reproduces the original centred crop.  A
    non-central centre is clamped so the window stays inside the image,
    which lets a config aim the zoom at an off-centre feature (e.g. the
    bolt-hole ring of the fuel nozzle).
    """
    h, w = image.shape[:2]
    half = fraction / 2.0
    cr = min(max(center[0], half), 1.0 - half)
    cc = min(max(center[1], half), 1.0 - half)
    r0, r1 = int(h * (cr - half)), int(h * (cr + half))
    c0, c1 = int(w * (cc - half)), int(w * (cc + half))
    return image[r0:r1, c0:c1]


def _crop_bounds(
    image: np.ndarray,
    fraction: float,
    center: tuple[float, float],
) -> tuple[int, int, int, int]:
    """Return crop bounds as ``(r0, r1, c0, c1)`` for a fractional window."""
    h, w = image.shape[:2]
    half = fraction / 2.0
    cr = min(max(center[0], half), 1.0 - half)
    cc = min(max(center[1], half), 1.0 - half)
    r0, r1 = int(h * (cr - half)), int(h * (cr + half))
    c0, c1 = int(w * (cc - half)), int(w * (cc + half))
    return r0, r1, c0, c1


def _draw_panel(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    title: str = "",
    inset_image: np.ndarray | None = None,
    gamma: float = DISPLAY_GAMMA,
) -> None:
    """Render a single panel with optional center-crop inset.

    Mirrors _draw_panel / _draw_full_with_inset from MILP_opt_project:
      - PowerNorm gamma correction for the main image
      - 38 % x 38 % lower-right inset with white-border spines
    """
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    ax.imshow(image, cmap="gray", norm=norm, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    if inset_image is not None:
        ins = inset_axes(ax, width="38%", height="38%", loc="lower right", borderpad=0.18)
        ins.imshow(inset_image, cmap="gray", norm=norm, interpolation="nearest")
        ins.set_xticks([])
        ins.set_yticks([])
        for spine in ins.spines.values():
            spine.set_color("white")
            spine.set_linewidth(0.8)


def _save(fig: plt.Figure, path: Path, *, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path.with_suffix('.pdf')} + .png")


def _truncate_colormap(name: str, lo: float, hi: float = 1.0, n: int = 256):
    base = plt.get_cmap(name)
    xs = np.linspace(lo, hi, n)
    return LinearSegmentedColormap.from_list(f"{name}_trunc", base(xs))


# ---------------------------------------------------------------------------
# Reconstruction helper
# ---------------------------------------------------------------------------

def _reconstruct(vol, src, geometry, sart_iter,
                 photon_count=None, noise_seed=0):
    sp, dc, du, dv = geometry_from_sources(
        src, sid=geometry["sid"], sdd=geometry["sdd"]
    )
    sino = simulate_sinogram(
        vol, sp, dc, du, dv,
        det_u=geometry["det_voxels"], det_v=geometry["det_voxels"],
        du=geometry["det_pitch"], dv=geometry["det_pitch"],
        voxel_spacing=geometry["voxel_pitch"],
        photon_count=photon_count, noise_seed=noise_seed,
    )
    mx.eval(sino)
    res = reconstruct_sart_volume(
        vol.shape, sino, sp, dc, du, dv,
        du=geometry["det_pitch"], dv=geometry["det_pitch"],
        voxel_spacing=geometry["voxel_pitch"],
        iteration_count=sart_iter, show_progress=False,
    )
    mx.eval(res.reconstruction)
    return np.asarray(res.reconstruction)


# ---------------------------------------------------------------------------
# PSNR helper
# ---------------------------------------------------------------------------

def _psnr(reco: np.ndarray, ref: np.ndarray) -> float:
    mse = float(np.mean((reco - ref) ** 2))
    if mse <= 0.0:
        return float("inf")
    peak = float(ref.max())
    if peak <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(peak * peak / mse))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("experiment_type") != "figure_milp_slices":
        raise SystemExit("config experiment_type must be figure_milp_slices")

    stack = _load_mlx_stack()

    # Geometry and resolution may be set per phantom, so one plate can hold
    # rows for objects that are scanned on different benches.
    def _geometry_for(ph):
        return _resolve_geometry(ph.get("geometry", cfg["geometry"]),
                                 int(ph["resolution"]))

    geometry = _geometry_for(cfg["phantoms"][0])

    # -----------------------------------------------------------------------
    # Compute reconstructions for every (phantom, method) pair
    # -----------------------------------------------------------------------
    views: dict[str, object] = {}          # tag -> display orientation
    refs: dict[str, np.ndarray] = {}      # tag -> ground-truth volume
    z_idx: dict[str, int] = {}             # tag -> best axial slice index
    recos: dict[tuple[str, str], np.ndarray] = {}  # (tag, method) -> reco volume
    psnrs: dict[tuple[str, str], float] = {}
    labels: dict[str, str] = {}            # method name -> display label

    for ph in cfg["phantoms"]:
        print(f"\n=== {ph['tag']} ===")
        geometry = _geometry_for(ph)
        vol = _load_phantom(ph, stack)
        mx.eval(vol)
        roi_ctx = _resolve_roi_context(
            cfg, vol, geometry, stack, want_mask=False,
        )
        roi_center = roi_ctx["center"]
        vol_np = np.asarray(vol, dtype=np.float32)
        # ``axis: x`` shows the sagittal plane instead of the axial one, which
        # is the plane in which a Defrise-type part actually reveals its
        # laminae. Only the display orientation changes; the reconstruction is
        # untouched.
        view = (lambda a: np.asarray(a, dtype=np.float32).transpose(2, 0, 1)) \
            if ph.get("axis", "z") == "x" else (lambda a: np.asarray(a, dtype=np.float32))
        views[ph["tag"]] = view
        vol_np = view(vol_np)
        refs[ph["tag"]] = vol_np
        z_idx[ph["tag"]] = _best_center_slice_index(vol_np)

        candidates = sample_unit_sphere(cfg["k_max"]) * geometry["sid"]
        vcl_pre = compute_R_gamma(
            vol, candidates,
            sid=geometry["sid"], sdd=geometry["sdd"],
            det_shape=(geometry["det_voxels"], geometry["det_voxels"]),
            du=geometry["det_pitch"], dv=geometry["det_pitch"],
            voxel_spacing=geometry["voxel_pitch"],
            r1=cfg.get("voxel_subsample_r1", 1e-3), seed=0,
        )
        for m in cfg["methods"]:
            name = m["name"]
            label = m.get("label", name)
            labels[name] = label
            print(f"  [{name}] ...", end="", flush=True)
            optimize_roi = _method_optimizes_roi(name)
            method_kwargs = _method_kwargs_with_roi(
                name, m.get("kwargs"), roi_ctx, geometry,
                optimize_roi=optimize_roi,
            )
            selector_roi_center = _method_roi_center(name, roi_ctx, stack)
            src = build_baseline_sources(
                name, cfg["k"], geometry["sid"],
                roi_center=selector_roi_center, vcl_precompute=vcl_pre, volume=vol,
                sdd=geometry["sdd"],
                detector_shape=(geometry["det_voxels"], geometry["det_voxels"]),
                du=geometry["det_pitch"], dv=geometry["det_pitch"],
                voxel_spacing=geometry["voxel_pitch"],
                n_candidates=cfg["k_max"], seed=cfg.get("seed", 0),
                method_kwargs=method_kwargs,
            )
            mx.eval(src)
            reco = _reconstruct(
                vol, src, geometry, cfg["sart_iterations"],
                photon_count=cfg.get("photon_count"),
                noise_seed=int(cfg.get("noise_seed", 0)),
            )
            recos[(ph["tag"], name)] = views[ph["tag"]](reco)
            psnrs[(ph["tag"], name)] = _psnr(reco, vol_np)
            print(f" PSNR={psnrs[(ph['tag'], name)]:.2f} dB")

    # -----------------------------------------------------------------------
    # Figure 1: Slice comparison grid
    #
    # rows = phantoms   |  cols = reference + methods
    # Visual style: PowerNorm(gamma=0.55), percentile windowing anchored to
    # reference, center-crop ROI inset, row labels outside tight_layout.
    # -----------------------------------------------------------------------
    methods = cfg["methods"]
    phs = cfg["phantoms"]
    n_rows = len(phs)
    n_cols = 1 + len(methods)   # col 0 = reference
    panel_size = 2.2
    panel_w = panel_size
    panel_h = 2.35

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows),
        squeeze=False,
    )

    # Optional config-driven inset placement: aim the zoom at an
    # off-centre feature (default = centred crop).
    inset_center = tuple(cfg.get("inset_center", (0.5, 0.5)))
    inset_fraction = float(cfg.get("inset_fraction", INSET_CROP_FRACTION))

    for row_i, ph in enumerate(phs):
        tag = ph["tag"]
        vol_np = refs[tag]
        z = z_idx[tag]
        ref_slice = vol_np[z]
        vmin, vmax = _contrast_window(ref_slice)
        inset_center = tuple(ph.get("inset_center", cfg.get("inset_center", (0.5, 0.5))))
        inset_fraction = float(ph.get("inset_fraction",
                                      cfg.get("inset_fraction", INSET_CROP_FRACTION)))
        inset_ref = _center_crop(ref_slice, inset_fraction, inset_center)

        # Column 0: ground-truth reference
        ax = axes[row_i, 0]
        _draw_panel(
            ax, ref_slice,
            vmin=vmin, vmax=vmax,
            title="Reference" if row_i == 0 else "",
            inset_image=inset_ref,
        )

        # Columns 1..N: methods
        for col_i, m in enumerate(methods, start=1):
            name = m["name"]
            reco = recos[(tag, name)]
            reco_slice = reco[z]
            inset_reco = _center_crop(reco_slice, inset_fraction, inset_center)
            psnr_val = psnrs[(tag, name)]
            col_title = f"{labels[name]}\nPSNR {psnr_val:.2f} dB" if row_i == 0 else ""
            ax = axes[row_i, col_i]
            _draw_panel(
                ax, reco_slice,
                vmin=vmin, vmax=vmax,
                title=col_title,
                inset_image=inset_reco,
            )
            if row_i > 0:
                ax.set_title(f"PSNR {psnr_val:.2f} dB", fontsize=7, pad=4)

        # Row label on the left (outside the grid, like MILP_opt_project)
        y_pos = 1.0 - (row_i + 0.5) / n_rows
        fig.text(0.055, y_pos, tag.capitalize(), ha="right", va="center", fontsize=8)

    fig.subplots_adjust(left=0.07, right=0.995, top=0.965, bottom=0.035, wspace=0.01, hspace=0.045)
    _save(fig, Path(cfg["output_slices"]))

    # -----------------------------------------------------------------------
    # Figure 2: Absolute difference maps |reco - ref|
    #
    # Shared inferno colormap; per-row vmax = max over all method differences.
    # Row labels on the left, shared zoom crop, one colorbar per row.
    # -----------------------------------------------------------------------
    diff_n_cols = len(methods)
    diff_zoom_center = tuple(cfg.get("diff_zoom_center", (0.30, 0.50)))
    diff_zoom_fraction = float(cfg.get("diff_zoom_fraction", 0.20))
    diff_inset_size = str(cfg.get("diff_inset_size", "28%"))
    diff_gamma = float(cfg.get("diff_gamma", 0.6))
    diff_cmap_floor = float(cfg.get("diff_cmap_floor", 0.10))
    diff_cmap = _truncate_colormap("inferno", diff_cmap_floor)
    fig2, axes2 = plt.subplots(
        n_rows, diff_n_cols,
        figsize=(panel_size * diff_n_cols, panel_size * n_rows),
        squeeze=False,
    )
    all_pos_vals: list[np.ndarray] = []
    all_diffs: list[np.ndarray] = []
    for ph in phs:
        tag = ph["tag"]
        vol_np = refs[tag]
        z = z_idx[tag]
        ph_diffs = [np.abs(recos[(tag, m["name"])][z] - vol_np[z]) for m in methods]
        all_diffs.extend(ph_diffs)
        all_pos_vals.extend([d[d > 0.0] for d in ph_diffs if np.any(d > 0.0)])
    global_dmax = float(max(d.max() for d in all_diffs if d.max() > 0.0) or 1.0)
    global_pos = np.concatenate(all_pos_vals)
    global_dmin = 0.0
    global_dmax = float(max(np.percentile(global_pos, 99.7), global_dmin + 1e-6))
    diff_norm = PowerNorm(gamma=diff_gamma, vmin=global_dmin, vmax=global_dmax)
    last_im = None

    for row_i, ph in enumerate(phs):
        tag = ph["tag"]
        vol_np = refs[tag]
        z = z_idx[tag]
        diffs = [np.abs(recos[(tag, m["name"])][z] - vol_np[z]) for m in methods]

        for col_i, (m, diff) in enumerate(zip(methods, diffs)):
            ax = axes2[row_i, col_i]
            last_im = ax.imshow(
                np.clip(diff, global_dmin, global_dmax), cmap=diff_cmap, norm=diff_norm,
                interpolation="nearest",
            )
            ax.axis("off")
            name = m["name"]
            r0, r1, c0, c1 = _crop_bounds(diff, diff_zoom_fraction, diff_zoom_center)
            ax.add_patch(
                Rectangle(
                    (c0, r0), c1 - c0, r1 - r0,
                    fill=False, edgecolor="white", linewidth=0.7,
                )
            )
            ins = inset_axes(ax, width=diff_inset_size, height=diff_inset_size, loc="lower right", borderpad=0.18)
            ins.imshow(
                np.clip(diff[r0:r1, c0:c1], global_dmin, global_dmax),
                cmap=diff_cmap, norm=diff_norm,
                interpolation="nearest",
            )
            ins.set_xticks([])
            ins.set_yticks([])
            for spine in ins.spines.values():
                spine.set_color("white")
                spine.set_linewidth(0.8)
            if row_i == 0:
                ax.set_title(
                    labels[name],
                    fontsize=7, pad=2,
                )

        y_pos = 1.0 - (row_i + 0.5) / n_rows
        fig2.text(0.055, y_pos, tag.capitalize(), ha="right", va="center", fontsize=8)
    fig2.subplots_adjust(left=0.085, right=0.925, top=0.95, bottom=0.05, wspace=0.01, hspace=0.05)
    if last_im is not None:
        cax = fig2.add_axes([0.935, 0.18, 0.016, 0.66])
        cbar = fig2.colorbar(last_im, cax=cax, extend="max")
        cbar.ax.tick_params(labelsize=9, length=3)

    _save(fig2, Path(cfg["output_diff"]))

    # -----------------------------------------------------------------------
    # Unified compact paper plate. Volume metrics remain in the tables and are
    # intentionally not repeated as slice-local annotations.
    # -----------------------------------------------------------------------
    output_plate = cfg.get("output_plate")
    if output_plate:
        cases = []
        for ph in phs:
            tag = ph["tag"]
            z = z_idx[tag]
            cases.append(
                {
                    "label": tag.capitalize(),
                    "reference": refs[tag][z],
                    "reconstructions": {
                        m["name"]: recos[(tag, m["name"])][z] for m in methods
                    },
                    "voxel_spacing_mm": geometry["voxel_pitch"],
                    "inset_center": tuple(
                        ph.get("inset_center", cfg.get("inset_center", (0.5, 0.5)))
                    ),
                    "inset_fraction": float(
                        ph.get("inset_fraction",
                               cfg.get("inset_fraction", INSET_CROP_FRACTION))
                    ),
                    # Display grammar is per phantom: a dense metal part and a
                    # low-contrast aluminium plate need different windows.
                    "display_gamma": float(
                        ph.get("display_gamma", cfg.get("display_gamma", 0.55))
                    ),
                    "display_black_percentile": float(
                        ph.get("display_black_percentile",
                               cfg.get("display_black_percentile", 55.0))
                    ),
                    "display_white_percentile": float(
                        ph.get("display_white_percentile",
                               cfg.get("display_white_percentile", 98.0))
                    ),
                    "display_highlight_headroom": float(
                        ph.get("display_highlight_headroom",
                               cfg.get("display_highlight_headroom", 1.0))
                    ),
                }
            )
        render_reconstruction_plate(
            cases,
            [(m["name"], labels[m["name"]]) for m in methods],
            Path(output_plate),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
