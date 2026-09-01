"""Shared publication style for result figures.

The result section uses one visual grammar throughout:

* columns are methods in a stable order;
* every qualitative case occupies one compact reconstruction row;
* image windows are fixed by the reference of that case;
* numerical volume metrics remain in the tables rather than being repeated as
  slice-local annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


METHOD_LABELS = {
    "uniform": "Uniform",
    "vcls": "VCLS",
    "coverage": "Coverage",
    "bundle": "Coverage + bundle",
    "composite": "Full composite",
    "fd": "Finite difference",
}

METHOD_COLORS = {
    "uniform": "#777777",
    "vcls": "#0077BB",
    "coverage": "#009988",
    "bundle": "#EE7733",
    "composite": "#AA3377",
    "fd": "#CC3311",
}


def configure_matplotlib() -> None:
    """Apply the paper-wide, colourblind-safe plotting style."""

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _positive(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32).ravel()
    return values[np.isfinite(values) & (values > 0)]


def reference_window(
    reference: np.ndarray,
    *,
    black_percentile: float = 55.0,
    white_percentile: float = 98.0,
    highlight_headroom: float = 1.0,
) -> tuple[float, float]:
    """Return a robust display window anchored to one reference slice.

    ``highlight_headroom`` reserves space above the bright-reference
    percentile.  Together with a smaller display gamma this compresses dense
    blockers without sacrificing low-attenuation object structure.  The same
    monotone mapping is used for the reference and every reconstruction in a
    case, so it changes presentation only, never the comparison.
    """

    values = _positive(reference)
    if values.size == 0:
        return 0.0, 1.0
    vmin = (
        float(np.percentile(values, black_percentile))
        if black_percentile > 0.0
        else 0.0
    )
    vmax = float(np.percentile(values, white_percentile))
    if vmin >= 0.5 * vmax:
        vmin = 0.0
    vmax = vmin + max(float(highlight_headroom), 1.0) * (vmax - vmin)
    return vmin, max(vmax, vmin + 1e-6)


def _crop_bounds(
    image: np.ndarray,
    center: tuple[float, float],
    fraction: float,
) -> tuple[int, int, int, int]:
    height, width = image.shape
    half = fraction / 2.0
    row = min(max(float(center[0]), half), 1.0 - half)
    col = min(max(float(center[1]), half), 1.0 - half)
    return (
        int(height * (row - half)),
        int(height * (row + half)),
        int(width * (col - half)),
        int(width * (col + half)),
    )


def _draw_scale_bar(ax: plt.Axes, voxel_spacing_mm: float, length_mm: float = 10.0) -> None:
    pixels = length_mm / max(float(voxel_spacing_mm), 1e-9)
    x0, y0 = 0.07, 0.08
    transform = ax.transAxes
    image = ax.images[0].get_array()
    width_fraction = min(0.35, pixels / max(image.shape[1], 1))
    ax.plot(
        [x0, x0 + width_fraction],
        [y0, y0],
        color="white",
        linewidth=2.0,
        solid_capstyle="butt",
        transform=transform,
    )
    ax.text(
        x0,
        y0 + 0.025,
        f"{length_mm:g} mm",
        color="white",
        fontsize=6.5,
        ha="left",
        va="bottom",
        transform=transform,
    )


def _draw_reconstruction(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    inset_center: tuple[float, float] | None,
    inset_fraction: float,
    scale_bar_spacing_mm: float | None,
    gamma: float,
) -> None:
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    ax.imshow(image, cmap="gray", norm=norm, interpolation="nearest")
    ax.set_axis_off()
    if inset_center is not None:
        r0, r1, c0, c1 = _crop_bounds(image, inset_center, inset_fraction)
        ax.add_patch(
            Rectangle(
                (c0, r0),
                c1 - c0,
                r1 - r0,
                fill=False,
                edgecolor="white",
                linewidth=0.6,
            )
        )
        inset = inset_axes(
            ax, width="31%", height="31%", loc="lower right", borderpad=0.18
        )
        inset.imshow(
            image[r0:r1, c0:c1],
            cmap="gray",
            norm=norm,
            interpolation="nearest",
        )
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color("white")
            spine.set_linewidth(0.7)
    if scale_bar_spacing_mm is not None:
        _draw_scale_bar(ax, scale_bar_spacing_mm)


def render_reconstruction_plate(
    cases: Sequence[Mapping[str, object]],
    methods: Sequence[tuple[str, str]],
    output: str | Path,
) -> None:
    """Render compact reconstruction comparisons for experimental cases.

    Each case mapping contains:

    ``label``
        Row label.
    ``reference``
        Two-dimensional reference slice.
    ``reconstructions``
        Mapping from method key to two-dimensional reconstruction.
    ``voxel_spacing_mm``
        Pixel spacing used for the reference scale bar.
    ``inset_center`` and ``inset_fraction`` (optional)
        Fractional ROI crop used consistently across all method panels.
    ``display_gamma``, ``display_black_percentile``,
    ``display_white_percentile``, and ``display_highlight_headroom`` (optional)
        Reference-anchored grayscale mapping.  Occlusion phantoms use a lower
        gamma plus highlight headroom to reveal weak structures without letting
        dense blockers saturate.
    """

    configure_matplotlib()
    n_cases = len(cases)
    n_cols = 1 + len(methods)
    fig, axes = plt.subplots(
        n_cases,
        n_cols,
        figsize=(1.52 * n_cols, 1.62 * n_cases),
        squeeze=False,
    )

    for case_index, case in enumerate(cases):
        reference = np.asarray(case["reference"], dtype=np.float32)
        reconstructions = case["reconstructions"]
        gamma = float(case.get("display_gamma", 0.55))
        vmin, vmax = reference_window(
            reference,
            black_percentile=float(
                case.get("display_black_percentile", 55.0)
            ),
            white_percentile=float(
                case.get("display_white_percentile", 98.0)
            ),
            highlight_headroom=float(
                case.get("display_highlight_headroom", 1.0)
            ),
        )
        inset_center = case.get("inset_center")
        inset_fraction = float(case.get("inset_fraction", 0.25))
        spacing = float(case.get("voxel_spacing_mm", 1.0))

        _draw_reconstruction(
            axes[case_index, 0],
            reference,
            vmin=vmin,
            vmax=vmax,
            inset_center=inset_center,
            inset_fraction=inset_fraction,
            scale_bar_spacing_mm=spacing,
            gamma=gamma,
        )

        for column, (key, _) in enumerate(methods, start=1):
            reconstruction = np.asarray(reconstructions[key], dtype=np.float32)
            _draw_reconstruction(
                axes[case_index, column],
                reconstruction,
                vmin=vmin,
                vmax=vmax,
                inset_center=inset_center,
                inset_fraction=inset_fraction,
                scale_bar_spacing_mm=None,
                gamma=gamma,
            )

        label = str(case["label"])
        axes[case_index, 0].text(
            -0.08,
            0.5,
            label,
            rotation=90,
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
            transform=axes[case_index, 0].transAxes,
        )

    headers = ["Reference"] + [label for _, label in methods]
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, fontsize=8, pad=3)

    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        top=0.96,
        bottom=0.04,
        wspace=0.018,
        hspace=0.08,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)
    print(f"Wrote {output_path.with_suffix('.png')}")
