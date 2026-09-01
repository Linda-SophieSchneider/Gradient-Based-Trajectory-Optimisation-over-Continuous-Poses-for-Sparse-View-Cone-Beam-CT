"""Figure 2: the Defrise flange phantom (replaces industrial_phantoms.pdf).

Panel (a): 3-D view (aluminium shell translucent blue, steel inserts red).
Panel (b): central sagittal slice with the two Defrise LoF stacks, the tilted
           control cracks, and the steel inserts annotated.
Panel (c): axial slice through the upper flange disk (LoF layer, insert,
           shadow-side pore cluster).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
OUT_PNG = OUT_DIR / "defrise_flange.png"
OUT_PDF = OUT_DIR / "defrise_flange.pdf"

VOL = np.load(ROOT / "data" / "lof_flange_v3.npy")
N = VOL.shape[0]
VS = 0.3

POOL_FACTOR = 6
MATRIX_THRESHOLD = 0.035   # aluminium mu = 0.07
DENSE_THRESHOLD = 0.15     # steel mu = 0.24


def pool_max(volume, factor):
    n0, n1, n2 = volume.shape
    return volume.reshape(n0 // factor, factor, n1 // factor, factor,
                          n2 // factor, factor).max(axis=(1, 3, 5))


def surface(mask):
    padded = np.pad(mask, 1, constant_values=False)
    core = padded[1:-1, 1:-1, 1:-1]
    interior = (padded[:-2, 1:-1, 1:-1] & padded[2:, 1:-1, 1:-1]
                & padded[1:-1, :-2, 1:-1] & padded[1:-1, 2:, 1:-1]
                & padded[1:-1, 1:-1, :-2] & padded[1:-1, 1:-1, 2:])
    return core & ~interior


def draw_3d(ax):
    pooled = pool_max(VOL, POOL_FACTOR)
    matrix = pooled > MATRIX_THRESHOLD
    dense = pooled > DENSE_THRESHOLD
    matrix_shell = surface(matrix & ~dense)
    dense_shell = surface(dense)
    colors = np.zeros(matrix_shell.shape + (4,), dtype=float)
    colors[matrix_shell] = [0.42, 0.63, 0.86, 0.16]
    colors[dense_shell] = [0.80, 0.15, 0.11, 0.98]
    voxels = matrix_shell | dense_shell
    ax.voxels(voxels, facecolors=colors, edgecolor="none")
    coords = np.argwhere(matrix | dense)
    lower = np.maximum(coords.min(axis=0) - 3, 0)
    upper = np.minimum(coords.max(axis=0) + 4, np.array(voxels.shape))
    ax.set_xlim(lower[0], upper[0]); ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect((upper - lower).tolist())
    ax.view_init(elev=18, azim=140)
    ax.set_title("(a) Defrise flange", fontsize=10, pad=4)
    ax.set_axis_off()


def mm_to_idx(mm):
    return int(round(mm / VS + (N - 1) / 2))


def main():
    fig = plt.figure(figsize=(7.4, 2.9), constrained_layout=True)
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    draw_3d(ax3d)

    vmax = 0.16
    ax = fig.add_subplot(1, 3, 2)
    sag = VOL[:, :, mm_to_idx(0.0)]
    ax.imshow(sag, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    ax.set_title("(b) sagittal $x=0$", fontsize=10)
    ann = dict(color="#d95f02", fontsize=7.5, ha="center",
               arrowprops=dict(arrowstyle="-", color="#d95f02", lw=0.9))
    ax.annotate("LoF stack (top)", xy=(mm_to_idx(0), mm_to_idx(23.8)),
                xytext=(mm_to_idx(-34), mm_to_idx(38)), **ann)
    ax.annotate("LoF stack (bottom)", xy=(mm_to_idx(0), mm_to_idx(-23.8)),
                xytext=(mm_to_idx(-34), mm_to_idx(-42)), **ann)
    ax.annotate("tilted control cracks", xy=(mm_to_idx(1), mm_to_idx(8.2)),
                xytext=(mm_to_idx(30), mm_to_idx(-2)), **ann)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(1, 3, 3)
    axial = VOL[mm_to_idx(23.8)]
    ax.imshow(axial, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    ax.set_title("(c) axial $z=+24$ mm", fontsize=10)
    ax.annotate("LoF layer", xy=(mm_to_idx(6), mm_to_idx(1)),
                xytext=(mm_to_idx(30), mm_to_idx(30)), **ann)
    ax.annotate("steel insert", xy=(mm_to_idx(0), mm_to_idx(13)),
                xytext=(mm_to_idx(-30), mm_to_idx(34)), **ann)
    ax.annotate("pores", xy=(mm_to_idx(-10.5), mm_to_idx(8)),
                xytext=(mm_to_idx(-32), mm_to_idx(-30)), **ann)
    ax.set_xticks([]); ax.set_yticks([])

    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT_PNG); print(OUT_PDF)

    # fig1 panel (a): sagittal mu slice of the flange (replaces
    # moderate_mu_slice.png in method_overview.tex)
    fig2, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.imshow(sag, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig2.tight_layout(pad=0)
    fig2.savefig(OUT_DIR / "flange_mu_slice.png", dpi=240,
                 bbox_inches="tight", pad_inches=0.01)
    print(OUT_DIR / "flange_mu_slice.png")


if __name__ == "__main__":
    main()
