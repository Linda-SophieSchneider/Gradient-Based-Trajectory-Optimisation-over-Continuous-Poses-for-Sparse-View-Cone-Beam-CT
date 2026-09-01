"""Render the real-world camera comparison figures (sec:real_results):

  real_slices_k100_k400.pdf   cropped central axial slices, k=100 (top block)
                              and k=400 (bottom block), reference + four arms,
                              with the profile line drawn through EVERY panel,
                              a scale bar, per-panel SSIM/PSNR vs reference, and
                              the corresponding line profile plotted directly
                              beneath each image row (reference + all arms).
  real_diff_k100_k400.pdf     |arm - reference| absolute-difference maps on a
                              shared scale (brighter = larger error); no line.

All arms are already in the reference frame (geometry-frame registration for
the planned/uniform arms; the circular subset shares the reference scan's own
frame), so slices are compared directly with NO further alignment, and each arm
carries its per-arm least-squares intensity-match factor to the reference.

All eight (arm, k) cells use their residual-selected canonical volumes
(alpha=0.15 everywhere; circular-k100 and all3-k100 completed their sweeps on
2026-08-11) with the LS scales of final_metrics_20260811.csv.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as _ssim, peak_signal_noise_ratio as _psnr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scanner_io.rek2py import rek2py  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER_FIG = HERE.parent / "Differentiable-Coverage" / "paper" / "figures"

VOXEL_MM = 0.278          # 4x-binned reconstruction voxel pitch
SCALE_BAR_MM = 10.0

REF = ("Reference", HERE / "reference_reconstructions" / "output_circular1200_fdk_quant" / "reconstruction_FDK.rek", 1.0)
ARMS = [  # (label, k100 path, k100 scale, k400 path, k400 scale, colour)
    ("Circular subset",
     HERE / "results_final" / "circular_final_k0100" / "reconstruction.rek", 1.972,
     HERE / "results_final" / "circular_final_k0400" / "reconstruction.rek", 1.918, "#1f77b4"),
    ("Uniform on band",
     HERE / "results_final" / "uniform_final_k0100" / "reconstruction.rek", 1.856,
     HERE / "results_final" / "uniform_final_k0400" / "reconstruction.rek", 1.751, "#ff7f0e"),
    ("Planned bundle",
     HERE / "results_final" / "bundle_final_k0100" / "reconstruction.rek", 1.760,
     HERE / "results_final" / "bundle_final_k0400" / "reconstruction.rek", 1.700, "#2ca02c"),
    ("Planned all3",
     HERE / "results_final" / "all3_final_k0100" / "reconstruction.rek", 1.770,
     HERE / "results_final" / "all3_final_k0400" / "reconstruction.rek", 1.737, "#d62728"),
]


def load_slice(path, scale):
    _, vol = rek2py(str(path), switch_order=True)
    vol = np.asarray(vol, np.float32) * scale
    return vol[vol.shape[0] // 2]


def main() -> None:
    ref_full = load_slice(*REF[1:])
    lo, hi = np.percentile(ref_full, [1.0, 99.7])
    dr = float(hi - lo)

    # crop to the object bounding box (+ margin) so the camera fills each panel
    thr = 0.2 * hi
    ys, xs = np.where(ref_full > thr)
    m = 20
    r0, r1 = max(int(ys.min()) - m, 0), min(int(ys.max()) + m, ref_full.shape[0])
    c0, c1 = max(int(xs.min()) - m, 0), min(int(xs.max()) + m, ref_full.shape[1])
    crop = lambda a: a[r0:r1, c0:c1]
    refc = crop(ref_full)
    H, W = refc.shape

    # profile row: horizontal line through the lens (brightest compact blob)
    from scipy import ndimage
    prow_full = int(np.unravel_index(np.argmax(ndimage.gaussian_filter(ref_full, 4)),
                                     ref_full.shape)[0])
    prow = prow_full - r0

    k100 = {"Reference": refc}
    k400 = {"Reference": refc}
    for label, p1, s1, p4, s4, _ in ARMS:
        k100[label] = crop(load_slice(p1, s1))
        k400[label] = crop(load_slice(p4, s4))
    order = ["Reference"] + [a[0] for a in ARMS]
    x_mm = np.arange(W) * VOXEL_MM

    def _scalebar(ax):
        n = SCALE_BAR_MM / VOXEL_MM
        x0, y0 = 0.10 * W, H - 0.07 * H
        ax.plot([x0, x0 + n], [y0, y0], color="w", lw=2.5, solid_capstyle="butt")
        ax.text(x0, y0 - 0.03 * H, f"{SCALE_BAR_MM:.0f} mm",
                color="w", fontsize=6, ha="left", va="bottom")

    def _img_row(gs_row, row, klabel, fig, gs):
        for ci, label in enumerate(order):
            ax = fig.add_subplot(gs[gs_row, ci])
            sl = row[label]
            ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi)
            ax.axhline(prow, color="#ff3b3b", lw=0.9)         # line through EVERY panel
            if label == "Reference":
                _scalebar(ax)
            else:
                ss = _ssim(row["Reference"], sl, data_range=dr)
                pp = _psnr(row["Reference"], sl, data_range=dr)
                ax.text(0.03, 0.97, f"SSIM {ss:.3f}\n{pp:.1f} dB",
                        transform=ax.transAxes, fontsize=6, color="w", va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15", fc="k", ec="none", alpha=0.45))
            if gs_row == 0:
                ax.set_title(label, fontsize=8)
            if ci == 0:
                ax.text(-0.09, 0.5, klabel, transform=ax.transAxes, fontsize=9,
                        rotation=90, va="center", ha="right", fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

    def _prof_row(gs_row, row, fig, gs, show_legend, show_xlabel):
        ax = fig.add_subplot(gs[gs_row, :])
        ax.plot(x_mm, row["Reference"][prow], color="k", lw=1.6, label="reference", zorder=5)
        for label, *_r, colour in ARMS:
            ax.plot(x_mm, row[label][prow], color=colour, lw=1.0, label=label, alpha=0.9)
        ax.set_xlim(x_mm[0], x_mm[-1])
        ax.set_ylabel(r"$\mu$ (1/mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.margins(x=0)
        if show_legend:
            ax.legend(fontsize=6, ncol=1, loc="upper left", framealpha=0.85,
                      handlelength=1.3, borderpad=0.4)
        if show_xlabel:
            ax.set_xlabel("position along the marked line (mm)", fontsize=8)
        return ax

    # combined figure: images with the line through every panel, profile beneath each block
    fig = plt.figure(figsize=(7.16, 5.25))
    gs = fig.add_gridspec(4, 5, height_ratios=[1.0, 0.58, 1.0, 0.58],
                          hspace=0.28, wspace=0.05)
    _img_row(0, k100, "$k=100$", fig, gs)
    _prof_row(1, k100, fig, gs, show_legend=True, show_xlabel=False)
    _img_row(2, k400, "$k=400$", fig, gs)
    _prof_row(3, k400, fig, gs, show_legend=False, show_xlabel=True)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.955, bottom=0.06)
    # bbox_inches="tight" keeps the profile x-label from being clipped at the
    # bottom edge (it was cut off in the fixed-bbox export).
    fig.savefig(PAPER_FIG / "real_slices_k100_k400.pdf",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PAPER_FIG / "real_slices_k100_k400.png", dpi=220,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote real_slices_k100_k400.{pdf,png}")

    # difference maps -- NO profile line on any panel (incl. reference)
    diffs = [np.abs(d[l] - d["Reference"])
             for d in (k100, k400) for l in order if l != "Reference"]
    dmax = float(np.percentile(np.concatenate([x.ravel() for x in diffs]), 99.0))
    fig, axes = plt.subplots(2, 5, figsize=(7.16, 3.35))
    im = None
    for ri, (row, klabel) in enumerate(zip((k100, k400), ("$k=100$", "$k=400$"))):
        for ci, label in enumerate(order):
            ax = axes[ri][ci]
            if label == "Reference":
                ax.imshow(row[label], cmap="gray", vmin=lo, vmax=hi)
                _scalebar(ax)
            else:
                im = ax.imshow(np.abs(row[label] - row["Reference"]), cmap="inferno",
                               vmin=0, vmax=dmax)
            if ri == 0:
                ax.set_title(label, fontsize=8)
            if ci == 0:
                ax.text(-0.09, 0.5, klabel, transform=ax.transAxes, fontsize=9,
                        rotation=90, va="center", ha="right", fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.01)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(r"$|\mu_{\mathrm{arm}}-\mu_{\mathrm{ref}}|$  (1/mm)", fontsize=7)
    fig.subplots_adjust(left=0.03, right=0.9, top=0.9, bottom=0.02, wspace=0.06, hspace=0.08)
    fig.savefig(PAPER_FIG / "real_diff_k100_k400.pdf")
    fig.savefig(PAPER_FIG / "real_diff_k100_k400.png", dpi=220)
    plt.close(fig)
    print("wrote real_diff_k100_k400.{pdf,png}")


if __name__ == "__main__":
    main()
