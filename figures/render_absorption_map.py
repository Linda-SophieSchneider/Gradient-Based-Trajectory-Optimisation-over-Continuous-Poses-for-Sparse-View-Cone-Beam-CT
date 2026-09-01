"""Directional absorption map: tau(azimuth, elevation) from a fixed volume.

For a fixed source radius the ray from a source pose to a target point is
determined by the direction alone, so the absorption seen by the bundle term is
a function on the sphere: the X-ray transform of the attenuation map evaluated
towards the target. This renders that map for the Defrise flange, once per
target, so one can read off where absorption is low -- and compare it with the
directions the defects actually need.

    python figures/render_absorption_map.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffct")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx

mx.set_default_device(mx.cpu)  # mlx-cuda 0.30 correctness bug; see project notes
import numpy as np

from differentiable_coverage.absorption_bundle import (
    BundleAbsorptionConfig, bundle_path_integral,
)

ROOT = Path(__file__).resolve().parents[1]
# Env overrides follow the repo convention (driver scripts take env vars instead
# of edits).  Defaults reproduce the v3 figure unchanged.
PHANTOM_NPY = os.environ.get("ABSMAP_PHANTOM", "data/lof_flange_v3.npy")
OUT = ROOT / os.environ.get("ABSMAP_OUT", "figures/absorption_map")
SID, VS = 500.0, 0.3
D_AZ, D_EL = 3.0, 3.0

TARGETS = [
    ("isocentre", (0.0, 0.0, 0.0)),
    ("upper LoF stack", (0.0, 0.0, 24.0)),
    ("pores beside insert", (-10.5, 8.0, 22.5)),
]


def tau_map(vol, target, n_rays=1):
    cfg = BundleAbsorptionConfig(roi_radius=0.0, n_rays_u=n_rays, n_rays_v=n_rays,
                                 n_samples=256, voxel_spacing=VS,
                                 clip_to_volume=True)
    azs = np.arange(-180.0, 180.0, D_AZ)
    els = np.arange(-90.0, 90.0 + D_EL, D_EL)
    A, E = np.meshgrid(azs, els)
    a, e = np.radians(A.ravel()), np.radians(E.ravel())
    pts = np.stack([SID * np.cos(e) * np.cos(a),
                    SID * np.cos(e) * np.sin(a),
                    SID * np.sin(e)], axis=-1).astype(np.float32)
    tau = np.asarray(bundle_path_integral(
        mx.array(pts), mx.array(np.array(target, np.float32)),
        mx.array(vol), cfg, volume_center=mx.zeros(3)))
    return azs, els, tau.reshape(E.shape)


def main():
    vol = np.load(ROOT / PHANTOM_NPY)
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(11.0, 3.1),
                             constrained_layout=True)
    for ax, (label, tgt) in zip(np.atleast_1d(axes), TARGETS):
        azs, els, T = tau_map(vol, tgt)
        im = ax.pcolormesh(azs, els, T, cmap="magma", shading="auto")
        cs = ax.contour(azs, els, T, levels=6, colors="white",
                        linewidths=0.4, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=5.5, fmt="%.1f")
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.set_xlabel(r"azimuth $\theta$ [deg]")
        ax.set_title(rf"{label}: $\tau$ min {T.min():.2f}, max {T.max():.2f}",
                     fontsize=8.5)
        fig.colorbar(im, ax=ax, pad=0.02, label=r"$\tau$")
        print(f"{label:22s} tau {T.min():.3f}-{T.max():.3f}  "
              f"argmin at az={azs[np.unravel_index(T.argmin(), T.shape)[1]]:.0f} "
              f"el={els[np.unravel_index(T.argmin(), T.shape)[0]]:.0f} deg; "
              f"|el| of the 10% lowest-tau directions: "
              f"{np.median(np.abs(np.meshgrid(azs, els)[1].ravel()[np.argsort(T.ravel())[:max(1,T.size//10)]])):.0f} deg")
    np.atleast_1d(axes)[0].set_ylabel(r"elevation $\varphi$ [deg]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(OUT.with_suffix(ext), dpi=220)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
