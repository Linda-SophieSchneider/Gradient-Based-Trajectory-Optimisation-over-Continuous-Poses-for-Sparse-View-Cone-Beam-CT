"""Start-vs-final pose figures for every continuous selection in the main paper.

For each example the figure has two panels: a 3-D view of the source sphere and
a 2-D azimuth/elevation projection. Both show the initial poses (open blue
circles), the optimised poses (filled red dots), and one grey connector per
pose, so the per-pose displacement is visible in a single panel.

Start poses are obtained without touching library code: the dispatched
selectors accept an ``n_steps`` override through ``method_kwargs`` (and
``_constrained_select`` takes ``n_steps`` directly), so ``n_steps=0`` returns
the deterministic initialisation for the same seed, and the full-protocol call
returns the final poses.

Run from the repo root:
    python -m experiments.studies.render_trajectory_evolution

Writes paper/figures/trajectory_evolution/<example>.png (+ .pdf).
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-diffct")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from differentiable_coverage.absorption_bundle import (
    BundleAbsorptionConfig, bundle_path_integral, calibrate_bundle_weight,
)
from differentiable_coverage.score import sample_unit_sphere

from experiments.run import (  # type: ignore
    _load_mlx_stack, _load_phantom_pair, _resolve_geometry, _resolve_roi_context,
)
from experiments.studies.kinematics import _build_cache
from experiments.studies.limited_angle import (
    CONSTRAINTS, VOLUME_CENTER_MM, _constrained_select, _gantry,
)

OUT = Path("paper/figures/trajectory_evolution")
POSES = Path("experiments/results/lof_plate_20260825/trajectory_poses")
STATS_CSV = Path("experiments/results/lof_plate_20260825/trajectory_motion.csv")
SEED = 0
BUNDLE_KW = {"bundle_n_samples": 256, "bundle_clip_to_volume": True}
# Filled by render(); written as a CSV so the motion table and the paper
# figure can be rebuilt without re-running any selection.
STATS: list[dict] = []


def _angles_deg(start, final):
    """Great-circle angle between corresponding start/final view directions."""
    u = start / np.linalg.norm(start, axis=1, keepdims=True)
    v = final / np.linalg.norm(final, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((u * v).sum(1), -1.0, 1.0)))


def _sphere_pair(stack, impl, k, vol, geom, pre, kmax, extra_kw=None):
    """(start, final) sources for a build_baseline_sources selector."""
    def run(n_steps_override):
        kw = dict(BUNDLE_KW) if ("bundle" in impl or "composite" in impl) else {}
        if extra_kw:
            kw.update(extra_kw)
        if n_steps_override is not None:
            kw["n_steps"] = n_steps_override
        src = stack["build_baseline_sources"](
            impl, int(k), geom["sid"], roi_center=mx.zeros(3),
            vcl_precompute=pre, volume=vol, sdd=geom["sdd"],
            detector_shape=(geom["det_voxels"], geom["det_voxels"]),
            du=geom["det_pitch"], dv=geom["det_pitch"],
            voxel_spacing=geom["voxel_pitch"], n_candidates=kmax,
            seed=SEED, method_kwargs=kw)
        mx.eval(src)
        return np.asarray(src, dtype=np.float64)
    return run(0), run(None)


def _limited_pair(vol, geom, cname, roi_directed, roi_ctx, lam, bcfg):
    gantry = _gantry(geom["sid"], CONSTRAINTS[cname])
    common = dict(
        roi_center=(roi_ctx["center"] if roi_directed else mx.zeros(3)),
        roi_points=(roi_ctx["_pw"]["roi_points"] if roi_directed else None),
        roi_weights=(roi_ctx["_pw"]["roi_weights"] if roi_directed else None),
        lambda_bundle=(lam if roi_directed else 0.0),
        bcfg=bcfg, seed=SEED, lr=0.05, n_normals=2000,
        volume_center=mx.array(VOLUME_CENTER_MM),
    )
    start = _constrained_select(gantry, 60, vol, geom, CONSTRAINTS[cname],
                                n_steps=0, **common)
    final = _constrained_select(gantry, 60, vol, geom, CONSTRAINTS[cname],
                                n_steps=100, **common)
    return (np.asarray(start, dtype=np.float64),
            np.asarray(final, dtype=np.float64))


def _az_el(src):
    r = np.linalg.norm(src, axis=1)
    az = np.degrees(np.arctan2(src[:, 1], src[:, 0]))
    el = np.degrees(np.arcsin(np.clip(src[:, 2] / r, -1, 1)))
    return az, el


def render(name, title, start, final, sid):
    fig = plt.figure(figsize=(9.6, 4.2))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 19)
    xs = sid * np.outer(np.cos(u), np.sin(v))
    ys = sid * np.outer(np.sin(u), np.sin(v))
    zs = sid * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="0.85", linewidth=0.3, alpha=0.6)
    for a, b in zip(start, final):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="0.45", linewidth=0.7, alpha=0.8)
    ax.scatter(*start.T, facecolors="none", edgecolors="#4477aa", s=22,
               linewidths=1.0, label="start", depthshade=False)
    ax.scatter(*final.T, color="#cc3311", s=16, label="final",
               depthshade=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title("3-D", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax = fig.add_subplot(1, 2, 2)
    az0, el0 = _az_el(start)
    az1, el1 = _az_el(final)
    for a0, e0, a1, e1 in zip(az0, el0, az1, el1):
        d = a1 - a0
        if d > 180:
            a1w = a1 - 360
        elif d < -180:
            a1w = a1 + 360
        else:
            a1w = a1
        ax.plot([a0, a1w], [e0, e1], color="0.45", linewidth=0.7, alpha=0.8)
    ax.scatter(az0, el0, facecolors="none", edgecolors="#4477aa", s=26,
               linewidths=1.0, label="start")
    ax.scatter(az1, el1, color="#cc3311", s=18, label="final")
    ax.set_xlim(-185, 185)
    ax.set_xlabel("azimuth [deg]")
    ax.set_ylabel("elevation [deg]")
    ax.grid(alpha=0.25)
    ax.set_title("azimuth--elevation", fontsize=10)
    ax.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in (".png", ".pdf"):
        fig.savefig(OUT / f"{name}{ext}", dpi=220)
    plt.close(fig)
    disp = np.linalg.norm(final - start, axis=1)
    ang = _angles_deg(start, final)
    POSES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(POSES / f"{name}.npz", start=start, final=final,
                        angles_deg=ang, sid=sid, title=title)
    STATS.append({
        "example": name, "k": len(start),
        "median_deg": round(float(np.median(ang)), 3),
        "mean_deg": round(float(ang.mean()), 3),
        "p95_deg": round(float(np.percentile(ang, 95)), 3),
        "max_deg": round(float(ang.max()), 3),
        "frac_gt_1deg": round(float((ang > 1.0).mean()), 4),
        "frac_gt_10deg": round(float((ang > 10.0).mean()), 4),
        "mean_mm": round(float(disp.mean()), 2),
        "max_mm": round(float(disp.max()), 2),
    })
    print(f"  {name}: median|mean|max move {np.median(ang):.2f}|{ang.mean():.2f}"
          f"|{ang.max():.2f} deg  (>1deg: {100*(ang>1).mean():.0f}%)", flush=True)


def main():
    groups = os.environ.get("TRAJEVO_GROUPS", "all").split(",")
    want = lambda g: "all" in groups or g in groups
    OUT.mkdir(parents=True, exist_ok=True)
    stack = _load_mlx_stack()

    # ---------------- Defrise flange ----------------
    spec = {"type": "milp_npy", "path": "data/lof_flange_v3.npy",
            "resolution": 384}
    geom = _resolve_geometry("milp", 384)
    vol, _ = _load_phantom_pair(spec, stack)
    mx.eval(vol)
    sid = geom["sid"]

    # (R, gamma) caches: kmax=720 for the warm method-grid arm, kmax=360 for
    # the kinematics protocol.
    t = time.time()
    cand720 = stack["sample_unit_sphere"](720, seed=SEED) * sid
    pre720 = _build_cache(stack, vol, geom, cand720, 1e-3, SEED)
    print(f"flange cache kmax=720: {time.time()-t:.0f}s", flush=True)
    t = time.time()
    cand360 = stack["sample_unit_sphere"](360, seed=SEED) * sid
    pre360 = _build_cache(stack, vol, geom, cand360, 1e-3, SEED)
    print(f"flange cache kmax=360: {time.time()-t:.0f}s", flush=True)

    for k in (20, 40, 80):
        s, f = _sphere_pair(stack, "vcls_adam_bundle", k, vol, geom, pre720, 720)
        render(f"flange_sphere_warm_bundle_k{k}",
               f"Flange, free sphere, VCLS-warm analytic bundle, $k={k}$",
               s, f, sid)

    KIN = [("greedy_adam_composite", "cold composite, free sphere"),
           ("greedy_adam_bundle_two_axis", "cold bundle, two-axis gantry"),
           ("greedy_adam_bundle_carm", "cold bundle, limited C-arm")]
    for impl, label in KIN:
        for k in (20, 40, 80):
            s, f = _sphere_pair(stack, impl, k, vol, geom, pre360, 360)
            render(f"flange_{impl}_k{k}", f"Flange, {label}, $k={k}$",
                   s, f, sid)

    # Limited-angle arms (study protocol: k=60, I0-independent selection).
    vs = geom["voxel_pitch"]
    bcfg = BundleAbsorptionConfig(roi_radius=5.0, n_rays_u=5, n_rays_v=9,
                                  n_samples=256, voxel_spacing=vs,
                                  clip_to_volume=True)
    roi = _resolve_roi_context(
        {"roi": {"type": "sphere", "center_mm": [0.0, 0.0, 24.0],
                 "radius_mm": 5.0, "point_grid": 5}},
        vol, geom, stack, want_mask=False)
    w = np.asarray(roi["weights"], np.float64)
    w = w / max(w.sum(), 1e-12)
    roi["_pw"] = {"roi_points": roi["points"],
                  "roi_weights": mx.array(w.astype(np.float32))}
    probes = sample_unit_sphere(256) * sid
    med = float(mx.median(bundle_path_integral(
        probes, roi["center"], vol, bcfg,
        volume_center=mx.array(VOLUME_CENTER_MM))))
    lam = calibrate_bundle_weight(med, 0.2)
    for cname in ("wedge120", "lamino"):
        for roi_directed, tag, label in (
                (False, "cov", "coverage"),
                (True, "roi_abs", "ROI+absorption")):
            s, f = _limited_pair(vol, geom, cname, roi_directed, roi, lam, bcfg)
            render(f"flange_limited_{cname}_{tag}",
                   f"Flange, {cname}, {label}, $k=60$", s, f, sid)

    # ---------------- ORNL nozzle (cold sphere arms of tab:ornl_runs) -------
    if not want("ornl"):
        _write_stats()
        print("Done.", flush=True)
        return
    try:
        spec = {"type": "ornl_nozzle", "path": "data/ornl_nozzle.h5",
                "section": "L", "resolution": 512}
        geom = _resolve_geometry("ornl", 512)
        vol, _ = _load_phantom_pair(spec, stack)
        mx.eval(vol)
        sid = geom["sid"]
        for impl, label in (("greedy_adam_vcl", "cold cov+VCL"),
                            ("greedy_adam_bundle", "cold cov+bundle")):
            for k in (40, 80):
                s, f = _sphere_pair(stack, impl, k, vol, geom, None, 720)
                render(f"ornl_{impl}_k{k}",
                       f"Nozzle, {label}, Tuy-greedy start, $k={k}$",
                       s, f, sid)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] nozzle examples skipped: {e!r}", flush=True)

    _write_stats()
    print("Done.", flush=True)


def _write_stats():
    if not STATS:
        return
    import csv
    STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with STATS_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(STATS[0].keys()))
        w.writeheader()
        w.writerows(STATS)
    print(f"Wrote {STATS_CSV} ({len(STATS)} rows)", flush=True)


if __name__ == "__main__":
    main()
