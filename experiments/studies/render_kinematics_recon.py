"""Render fig:kinematics_recon --- reconstruction slices across the kinematic ladder.

Replaces the earlier bar-chart figure (which only re-plotted tab:kinematics) with
the actual reconstructions: one representative selector per kinematic regime,
reconstructed noise-free on the moderate industrial phantom at k=80, shown as a
single row (reference + four regimes).  The in-plane single-axis circle is
visibly the worst (out-of-plane structure it cannot sample); quality climbs
through the limited C-arm and two-axis gantry to the free sphere --- the ladder
of tab:kinematics, made visual, with the active terms reported in that table.

Run from the repo root:
    python -m experiments.studies.render_kinematics_recon

Writes paper/figures/kinematics_result_plate.png and, from the same slices,
the dedicated two-panel asset paper/figures/overview_recon_pair.png used by
Fig. 1(d) — a purpose-built render, not a pixel crop of the full plate
(hard-coded trims cut panels in half once the plate was regenerated).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffct")

import matplotlib
matplotlib.use("Agg")
import mlx.core as mx
import numpy as np

from experiments.run import _load_mlx_stack, _load_phantom_pair, _resolve_geometry
from experiments.studies.kinematics import (
    PHANTOMS, _build_cache, _reconstruct, _select,
)
from experiments.render_milp_slices import (
    _best_center_slice_index, _psnr,
)
from differentiable_coverage.figures.paper_style import render_reconstruction_plate

# One representative selector per regime (all our continuous selector except the
# conventional single-axis circle baseline), worst -> best kinematic freedom.
REGIMES = [
    ("Single-axis circle", "greedy_adam_circle"),
    ("Bench band",         "__band__"),
    ("Limited C-arm",      "greedy_adam_bundle_carm"),
    ("Full sphere",        "greedy_adam_bundle_two_axis"),
]

# The bench band is the C-arm arm on a rebound envelope, exactly as
# ``experiments.studies.kinematics_band`` does it for the numbers, so the panel
# and the table row come from the same code path.
BAND_IMPL = "greedy_adam_bundle_carm"


def _with_band_envelope():
    import differentiable_coverage.eval.trajectories as _eval_traj
    from experiments.studies.kinematics_band import BAND, BAND_SMOOTH
    saved = (_eval_traj.CArmTwoAxisGantry, _eval_traj.SmoothTwoAxisGantry)
    _eval_traj.CArmTwoAxisGantry, _eval_traj.SmoothTwoAxisGantry = BAND, BAND_SMOOTH
    return _eval_traj, saved


def main():
    # Env overrides so the same script serves the moderate legacy figure and
    # the Defrise-flange replacement (KINRECON_AXIS=x renders the sagittal
    # plane that shows both LoF stacks).
    phantom = os.environ.get("KINRECON_PHANTOM", "moderate")
    # KINRECON_REGIMES selects and orders the panels by implementation name,
    # so the same script can render the full ladder or a reduced plate.
    wanted = [x.strip() for x in os.environ.get("KINRECON_REGIMES", "").split(",") if x.strip()]
    if wanted:
        by_impl = {impl: (label, impl) for label, impl in REGIMES}
        missing = [w for w in wanted if w not in by_impl]
        if missing:
            raise SystemExit(f"unknown regimes {missing}; choose from {list(by_impl)}")
        REGIMES[:] = [by_impl[w] for w in wanted]
    resolution = int(os.environ.get("KINRECON_RESOLUTION", "192"))
    axis = os.environ.get("KINRECON_AXIS", "z")
    k, seed, kmax, sart_iter = 80, 0, 360, 15
    # Inset target: the upper LoF stack for the flange (image-fraction
    # (row, col) in the rendered sagittal plane), object centre otherwise.
    inset = (0.72, 0.5) if phantom == "lof_flange_v3" else (0.5, 0.5)

    stack = _load_mlx_stack()
    info = PHANTOMS[phantom]
    spec = {**info["spec"], "resolution": resolution}
    geom = _resolve_geometry(info["geometry"], resolution)
    vol, _ = _load_phantom_pair(spec, stack)
    mx.eval(vol)
    vol_np = np.asarray(vol, dtype=np.float32)
    if axis == "x":
        def _slice(a):
            return np.asarray(a, dtype=np.float32).transpose(2, 0, 1)[z]
        z = _best_center_slice_index(vol_np.transpose(2, 0, 1))
    else:
        def _slice(a):
            return np.asarray(a, dtype=np.float32)[z]
        z = _best_center_slice_index(vol_np)

    # Sphere (R, gamma) cache: needed by the discrete circle baseline.
    cand = stack["sample_unit_sphere"](kmax, seed=seed) * geom["sid"]
    sphere_pre = _build_cache(stack, vol, geom, cand, 1e-3, seed)

    print(f"Kinematics recon figure: {phantom} {tuple(vol.shape)} k={k}", flush=True)
    recos, psnrs = [], []
    for label, impl in REGIMES:
        band = impl == "__band__"
        if band:
            impl = BAND_IMPL
            mod, saved = _with_band_envelope()
        src = _select(stack, impl, k, vol, geom, sphere_pre, None, kmax, seed,
                      bundle_kwargs={"bundle_n_samples": 256,
                                     "bundle_clip_to_volume": True})
        if band:
            mod.CArmTwoAxisGantry, mod.SmoothTwoAxisGantry = saved
        recon = np.asarray(_reconstruct(vol, src, geom, sart_iter, None, 0))
        p = _psnr(recon, vol_np)
        recos.append(recon)
        psnrs.append(p)
        print(f"  {label:20s} ({impl:28s}) PSNR={p:.2f} dB", flush=True)

    # Use the same compact reconstruction grammar as the noise and ablation
    # studies. The PSNR values printed above are checked against the table but
    # deliberately not repeated as slice-local annotations.
    keys = [label for label, _ in REGIMES]
    render_reconstruction_plate(
        [
            {
                "label": ("Defrise flange" if phantom == "lof_flange_v3"
                          else phantom.capitalize()),
                "reference": _slice(vol_np),
                "reconstructions": {
                    key: _slice(reco) for key, reco in zip(keys, recos)
                },
                "voxel_spacing_mm": geom["voxel_pitch"],
                "inset_center": inset,
                "inset_fraction": 0.28,
                "display_gamma": 0.32,
                "display_black_percentile": 0.0,
                "display_white_percentile": 99.5,
                "display_highlight_headroom": 2.0,
            }
        ],
        [(label, label) for label, _ in REGIMES],
        Path("paper/figures/kinematics_result_plate"),
    )

    # Dedicated Fig. 1(d) asset: the two highest-freedom regimes as complete,
    # uncropped panels with the same reference-anchored display grammar.
    import matplotlib.pyplot as plt

    from differentiable_coverage.figures.paper_style import (
        _draw_reconstruction, configure_matplotlib, reference_window,
    )

    configure_matplotlib()
    # Keys are the panel labels now, so name the two highest-freedom regimes
    # the way REGIMES does.
    pair = [("Limited C-arm", "Limited C-arm"),
            ("Full sphere", "Full sphere")]
    by_key = dict(zip(keys, recos))
    vmin, vmax = reference_window(
        _slice(vol_np), black_percentile=0.0, white_percentile=99.5,
        highlight_headroom=2.0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(3.04, 1.66))
    for ax, (key, label) in zip(axes, pair):
        _draw_reconstruction(
            ax, _slice(by_key[key]),
            vmin=vmin, vmax=vmax,
            inset_center=inset, inset_fraction=0.28,
            scale_bar_spacing_mm=None, gamma=0.32,
        )
        ax.set_title(label, fontsize=8, pad=3)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02,
                        wspace=0.018)
    out = Path("paper/figures/overview_recon_pair")
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
