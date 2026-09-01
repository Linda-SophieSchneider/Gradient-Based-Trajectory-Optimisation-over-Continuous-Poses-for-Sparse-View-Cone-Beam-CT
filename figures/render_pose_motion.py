"""Paper figure: how far the continuous operator moves the poses.

Two azimuth--elevation panels built from the pose dumps of
``experiments.studies.render_trajectory_evolution``: a cold Tuy-greedy start on
a constrained manifold (large, systematic motion) beside a VCLS warm start on
the free sphere (local refinement only). Open circles are the initial poses,
filled dots the optimised poses, and one connector per pose shows its
displacement, so both panels answer the same question at very different scales.

Override the two examples with POSE_MOTION_LEFT / POSE_MOTION_RIGHT (npz stems).

    python figures/render_pose_motion.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POSES = ROOT / "experiments/results/lof_plate_20260825/trajectory_poses"
OUT = ROOT / "figures/pose_motion"

LEFT = os.environ.get("POSE_MOTION_LEFT",
                      "flange_greedy_adam_bundle_two_axis_k40")
RIGHT = os.environ.get("POSE_MOTION_RIGHT",
                       "flange_greedy_adam_bundle_carm_k40")
LADDER = os.environ.get("POSE_MOTION_LADDER", "").strip()
LADDER_STEMS = [
    "flange_greedy_adam_circle_k80",
    "flange_band_bundle_k80",
    "flange_greedy_adam_bundle_carm_k80",
    "flange_greedy_adam_bundle_two_axis_k80",
]
LABELS = {
    "flange_greedy_adam_circle_k80": r"(a) single-axis circle",
    "flange_band_bundle_k80": r"(b) bench band, $\pm30^\circ$",
    "flange_greedy_adam_bundle_carm_k80":
        r"(c) limited C-arm, $\pm110^\circ/\pm45^\circ$",
    "flange_greedy_adam_bundle_two_axis_k80": r"(d) full sphere",
    "flange_greedy_adam_bundle_two_axis_k40":
        r"(a) full sphere, gantry parametrisation",
    "flange_greedy_adam_bundle_carm_k40":
        r"(b) limited C-arm, $\pm110^\circ/\pm45^\circ$",
    "flange_sphere_warm_bundle_k40":
        r"(b) full sphere, direct parametrisation (VCLS start)",
    "flange_greedy_adam_composite_k40":
        r"(b) full sphere, direct parametrisation",
}

START_KW = dict(facecolors="none", edgecolors="#4477aa", s=26, linewidths=0.9,
                zorder=3)
FINAL_KW = dict(color="#cc3311", s=18, zorder=4)


# Envelope half-widths (azimuth, elevation) in degrees, drawn as guide lines.
LIMITS = {"flange_greedy_adam_bundle_carm_k40": (110.0, 45.0),
          "flange_greedy_adam_bundle_carm_k80": (110.0, 45.0),
          "flange_band_bundle_k80": (None, 30.0)}


def _az_el(src):
    """Gantry chart coordinates of a source position.

    Inverse of ``TwoAxisGantry``: x = -sid cos(phi) sin(theta),
    y = sid cos(phi) cos(theta), z = sid sin(phi). Using the same convention
    as the optimiser means the envelope limits appear where the text says.
    """
    r = np.linalg.norm(src, axis=1)
    az = np.degrees(np.arctan2(-src[:, 0], src[:, 1]))
    el = np.degrees(np.arcsin(np.clip(src[:, 2] / r, -1, 1)))
    return az, el


def panel(ax, stem, title, stat_in_title=False, envelope_label=True,
          stat_short=False):
    d = np.load(POSES / f"{stem}.npz", allow_pickle=True)
    start, final, ang = d["start"], d["final"], d["angles_deg"]
    az0, el0 = _az_el(start)
    az1, el1 = _az_el(final)
    for a0, e0, a1, e1 in zip(az0, el0, az1, el1):
        # unwrap the shorter way around the azimuth circle
        a1w = a1 - 360 if a1 - a0 > 180 else (a1 + 360 if a1 - a0 < -180 else a1)
        ax.plot([a0, a1w], [e0, e1], color="0.5", linewidth=0.6, alpha=0.9,
                zorder=2)
    ax.scatter(az0, el0, label="initial", **START_KW)
    ax.scatter(az1, el1, label="optimised", **FINAL_KW)
    ax.set_xlim(-185, 185)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel(r"azimuth $\theta$ [deg]")
    ax.grid(alpha=0.25, linewidth=0.4)
    lim = LIMITS.get(stem)
    if lim is not None:
        if lim[0] is not None:
            for x in (-lim[0], lim[0]):
                ax.axvline(x, color="#333333", linestyle=(0, (4, 2)),
                           linewidth=0.7, zorder=1)
        if lim[1] is not None:
            for y in (-lim[1], lim[1]):
                ax.axhline(y, color="#333333", linestyle=(0, (4, 2)),
                           linewidth=0.7, zorder=1)
        if envelope_label:
            ax.text(-175, lim[1] - 4, "envelope", fontsize=7, color="#333333",
                    ha="left", va="top")
    fmt = lambda x: f"{x:.2f}" if x < 10 else f"{x:.1f}"
    stat = (rf"median $\Delta={fmt(float(np.median(ang)))}^\circ$, "
            rf"max ${fmt(float(ang.max()))}^\circ$")
    if stat_short:
        stat = rf"median $\Delta={float(np.median(ang)):.1f}^\circ$"
    if stat_in_title:
        ax.set_title(f"{title}\n{stat}", fontsize=8.5, linespacing=1.4)
    else:
        ax.set_title(title, fontsize=9)
        ax.text(0.98, 0.03, stat, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7",
                          lw=0.5))
    return float(np.median(ang)), float(ang.max())


def main():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
    if LADDER:
        # One panel per reachable set of tab:kinematics, same order and the
        # same budget as the reconstruction plate, on one shared elevation
        # axis so the panels show how much of the sphere each set uses.
        fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.3), sharey=True)
        for ax, stem in zip(axes, LADDER_STEMS):
            panel(ax, stem, LABELS.get(stem, stem), stat_in_title=True,
                  envelope_label=False, stat_short=True)
            ax.set_ylim(-95, 95)
            ax.set_yticks([-90, -45, 0, 45, 90])
            ax.set_xticks([-180, 0, 180])
        axes[0].set_ylabel(r"elevation $\varphi$ [deg]")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
                   handletextpad=0.3, columnspacing=1.6,
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(pad=0.4, rect=(0, 0.06, 1, 1))
        out = OUT.parent / "pose_motion_ladder"
        for ext in (".pdf", ".png"):
            fig.savefig(out.with_suffix(ext), dpi=300)
            print(out.with_suffix(ext))
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    panel(axes[0], LEFT, LABELS.get(LEFT, LEFT))
    panel(axes[1], RIGHT, LABELS.get(RIGHT, RIGHT))
    axes[0].set_ylabel(r"elevation $\varphi$ [deg]")
    axes[1].set_ylabel(r"elevation $\varphi$ [deg]")
    axes[0].legend(loc="lower left", frameon=False, handletextpad=0.3,
                   borderpad=0.2)
    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(OUT.with_suffix(ext), dpi=300)
    print(OUT.with_suffix(".pdf"))
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
