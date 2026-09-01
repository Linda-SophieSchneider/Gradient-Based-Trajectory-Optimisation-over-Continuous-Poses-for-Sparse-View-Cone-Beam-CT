"""Kinematic flexibility as a measured advantage of the differentiable selector.

The discrete swap search selects from a fixed candidate pool on whatever orbit
that pool was drawn from.  Changing the kinematics --- single-axis circle,
two-axis gantry, limited C-arm --- means re-discretising the reachable set and
re-running the search, and on a tightly constrained envelope no candidate grid
places poses where a continuous optimiser would.  Our framework instead absorbs
the kinematics into the trajectory parametrisation and optimises a stated
differentiable composite on the constraint manifold directly.

This experiment measures what that buys.  On the off-centre, asymmetric
industrial phantom --- where out-of-plane views carry real information --- it
compares, at matched view budgets and under photon noise:

  single-axis circle   vcls_circle              (conventional CT / Lin et al.)
  free sphere          vcls (discrete)          our greedy + VCL + bundle (Adam)
  two-axis gantry      our greedy + VCL + bundle (Adam, hard 2-axis reparam)
  limited C-arm        vcls_carm_grid (discrete, feasible poses)   <-- fair
                       our greedy + VCL + bundle (Adam on the clamped manifold)

The single-axis circle is the strongest baseline on the symmetric nozzle; here
it should collapse, and the multi-axis continuous selectors should recover the
loss.  The limited-C-arm block is the structural head-to-head: continuous
optimisation on the clamped manifold against the best discrete selection over a
feasible candidate grid.

Usage::

    python run_kinematics_headtohead.py --resolution 192 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from differentiable_coverage.eval.geometry import geometry_from_sources
from differentiable_coverage.eval.reco import simulate_sinogram, reconstruct_sart_volume
from differentiable_coverage.eval.vcl import vcls_select
from differentiable_coverage.trajectory import CArmTwoAxisGantry

from experiments.run import (  # type: ignore
    _compute_metrics,
    _load_mlx_stack,
    _load_phantom_pair,
    _resolve_geometry,
    _resolve_roi_context,
)

OUT_ROOT = Path("experiments/kinematics_headtohead")
QUALITY_METRICS = ["psnr", "ssim", "nrmse", "hfen"]

PHANTOMS = {
    "moderate": {"spec": {"type": "milp_npy",
                          "path": "data/moderate_asd_pocs_384.npy"},
                 "geometry": "milp"},
    "mild": {"spec": {"type": "milp_npy",
                      "path": "data/mild_asd_pocs_384.npy"},
             "geometry": "milp"},
    "lof_plate": {"spec": {"type": "milp_npy",
                           "path": "data/lof_plate_v1.npy"},
                  "geometry": "milp"},
    "lof_plate_v2": {"spec": {"type": "milp_npy",
                              "path": "data/lof_plate_v2.npy"},
                     "geometry": "milp"},
    "lof_flange_v3": {"spec": {"type": "milp_npy",
                               "path": "data/lof_flange_v3.npy"},
                      "geometry": "milp"},
    "lof_flange_v4a": {"spec": {"type": "milp_npy",
                                "path": "data/lof_flange_v4a.npy"},
                       "geometry": "milp"},
    "lof_flange_v4b": {"spec": {"type": "milp_npy",
                                "path": "data/lof_flange_v4b.npy"},
                       "geometry": "milp"},
}

# (label, regime, family, impl, needs_sphere_cache)
# impl == "__carm_grid__" is the custom discrete-feasible C-arm baseline.
METHODS = [
    ("vcls_circle",                "single-axis",   "discrete",  "vcls_circle",                False),
    ("vcls",                       "free-sphere",   "discrete",  "vcls",                       True),
    ("greedy_adam_composite",      "free-sphere",   "ours",      "greedy_adam_composite",      False),
    ("greedy_adam_bundle_two_axis","two-axis",      "ours",      "greedy_adam_bundle_two_axis", False),
    ("vcls_carm_grid",             "limited-carm",  "discrete",  "__carm_grid__",              False),
    ("greedy_adam_bundle_carm",    "limited-carm",  "ours",      "greedy_adam_bundle_carm",    False),
    # Our selector restricted to the single-axis circle. On a one-dimensional
    # manifold the objective reduces to coverage, since neither attenuation nor
    # information can be traded against elevation there, so this row shows what
    # the reachable set costs rather than what the selector adds.
    ("greedy_adam_circle",         "single-axis",   "ours",      "greedy_adam_circle",         False),
]


def _reconstruct(vol_gt, src, geom, sart_iter, photon, noise_seed):
    sp, dc, du, dv = geometry_from_sources(src, sid=geom["sid"], sdd=geom["sdd"])
    sino = simulate_sinogram(
        vol_gt, sp, dc, du, dv,
        det_u=geom["det_voxels"], det_v=geom["det_voxels"],
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"], photon_count=photon, noise_seed=noise_seed)
    mx.eval(sino)
    res = reconstruct_sart_volume(
        vol_gt.shape, sino, sp, dc, du, dv,
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"], iteration_count=sart_iter,
        show_progress=False)
    mx.eval(res.reconstruction)
    return res.reconstruction


def _carm_feasible_candidates(n, sid, seed):
    """Random feasible (theta, phi) poses inside the C-arm envelope, mapped to
    source positions through the gantry.  This is the candidate set a discrete
    swap search would have to use under the same kinematic constraint."""
    g = CArmTwoAxisGantry(sid=sid)
    rng = np.random.default_rng(seed)
    th = rng.uniform(float(g.theta_min), float(g.theta_max), size=n)
    ph = rng.uniform(float(g.phi_min), float(g.phi_max), size=n)
    params = mx.array(np.stack([th, ph], axis=-1).astype(np.float32))
    return g(params)


def _build_cache(stack, vol, geom, candidates, r1, seed):
    pre = stack["compute_R_gamma"](
        vol, candidates, sid=geom["sid"], sdd=geom["sdd"],
        det_shape=(geom["det_voxels"], geom["det_voxels"]),
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"], r1=r1, seed=seed)
    mx.eval(pre.R)
    return pre


def _select(stack, impl, k, vol, geom, sphere_pre, carm_pre, n_candidates, seed,
            bundle_kwargs=None):
    if impl == "__carm_grid__":
        idx, _ = vcls_select(carm_pre, k, seed=seed)
        src = carm_pre.candidate_sources[mx.array(idx)]
        mx.eval(src)
        return src
    src = stack["build_baseline_sources"](
        impl, int(k), geom["sid"], roi_center=mx.zeros(3),
        vcl_precompute=sphere_pre, volume=vol, sdd=geom["sdd"],
        detector_shape=(geom["det_voxels"], geom["det_voxels"]),
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"], n_candidates=n_candidates,
        seed=seed,
        # Quadrature knobs are only understood by the bundle/composite
        # dispatch branches (frozen production rule clip@256, REV-P1-01).
        method_kwargs=dict(bundle_kwargs or {})
        if ("bundle" in impl or "composite" in impl) else {})
    mx.eval(src)
    return src


def _ms(xs):
    return statistics.fmean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phantom", default="moderate", choices=list(PHANTOMS))
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--kmax", type=int, default=360)
    ap.add_argument("--n-candidates", type=int, default=360)
    ap.add_argument("--sart-iter", type=int, default=15)
    ap.add_argument("--photon", type=float, default=1.0e5)
    ap.add_argument("--noisefree", action="store_true",
                    help="reconstruct without photon noise, to isolate the "
                         "kinematic-geometry effect")
    ap.add_argument("--target-views", default="40,80")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--r1", type=float, default=1.0e-3)
    # Frozen production quadrature (bundle_quadrature_convergence study).
    ap.add_argument("--bundle-n-samples", type=int, default=256)
    ap.add_argument("--bundle-clip", type=int, default=1)
    ap.add_argument("--smooth-constraint", action="store_true",
                    help="Build the kinematic envelope into the chart via tanh "
                         "instead of clipping after each step.")
    ap.add_argument("--bundle-target-mm", default=None,
                    help="Absorption target as 'x,y,z' in world mm; default is "
                         "the coverage target (isocentre). Aiming the term at a "
                         "defect changes which directions it calls cheap.")
    ap.add_argument("--bundle-schedule", default=None,
                    choices=["constant", "ramp"],
                    help="Absorption weight schedule; 'ramp' is the documented "
                         "anti-collapse curriculum (coverage first).")
    ap.add_argument("--cov-survival-weight", action="store_true",
                    help="Weight each view's coverage contribution by its "
                         "photon survival exp(-tau) instead of using the "
                         "additive absorption penalty.")
    ap.add_argument("--bundle-lambda", type=float, default=None,
                    help="Override the calibrated absorption weight; 0 "
                         "switches the bundle term off (coverage+VCL only).")
    ap.add_argument(
        "--methods",
        default=None,
        help="Optional comma-separated subset of method labels/implementations.",
    )
    ap.add_argument(
        "--roi-centers",
        default=None,
        help="Optional semicolon-separated 'x,y,z[,r_mm]' world-mm defect-ROI "
             "centres (e.g. the lof_plate's designed crack/pore ROIs); adds "
             "per-ROI roi{j}_{psnr,ssim,hfen} columns. Evaluation-only — "
             "selection objectives are unchanged.",
    )
    ap.add_argument("--roi-radius-mm", type=float, default=5.0,
                    help="Default ROI radius when a centre omits its r_mm.")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT,
        help=f"Output root (default: {OUT_ROOT}).",
    )
    args = ap.parse_args(argv)

    k_values = [int(x) for x in args.target_views.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    photon = None if args.noisefree else args.photon
    methods = METHODS
    if args.methods:
        requested = {x.strip() for x in args.methods.split(",") if x.strip()}
        methods = [
            method
            for method in METHODS
            if method[0] in requested or method[3] in requested
        ]
        matched = {method[0] for method in methods} | {method[3] for method in methods}
        missing = requested - matched
        if missing:
            ap.error(f"unknown method(s): {', '.join(sorted(missing))}")
        if not methods:
            ap.error("--methods selected no methods")
    out_root = args.out_root
    (out_root / "results").mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)

    stack = _load_mlx_stack()
    info = PHANTOMS[args.phantom]
    spec = {**info["spec"], "resolution": args.resolution}
    geom = _resolve_geometry(info["geometry"], args.resolution)
    vol, _ = _load_phantom_pair(spec, stack)
    mx.eval(vol)
    peak = float(vol.max())

    ROI_METRICS = ["roi_psnr", "roi_ssim", "roi_hfen"]
    roi_masks, roi_specs = [], []
    if args.roi_centers:
        for cs in args.roi_centers.split(";"):
            vals = [float(v) for v in cs.split(",") if v.strip()]
            r = vals[3] if len(vals) > 3 else args.roi_radius_mm
            ctx = _resolve_roi_context(
                {"roi": {"type": "sphere", "center_mm": vals[:3],
                         "radius_mm": r}},
                vol, geom, stack, want_mask=True)
            roi_masks.append(ctx["mask"])
            roi_specs.append({"center_mm": vals[:3], "radius_mm": r})
    metric_names = list(QUALITY_METRICS) + \
        [f"roi{j}_{q[4:]}" for j in range(len(roi_masks)) for q in ROI_METRICS]

    print("=" * 72, flush=True)
    print("Kinematic flexibility head-to-head", flush=True)
    print(f"  phantom {args.phantom} {tuple(vol.shape)}  k={k_values}  "
          f"seeds={seeds}  photon={photon}", flush=True)
    print("=" * 72, flush=True)

    need_sphere = any(m[4] for m in methods)
    need_carm = any(m[3] == "__carm_grid__" for m in methods)

    agg = {(lbl, k): {q: [] for q in metric_names}
           for lbl, *_ in methods for k in k_values}
    meta = {lbl: (regime, fam) for lbl, regime, fam, *_ in methods}

    for seed in seeds:
        sphere_pre = carm_pre = None
        if need_sphere:
            t = time.time()
            cand = stack["sample_unit_sphere"](args.kmax, seed=seed) * geom["sid"]
            sphere_pre = _build_cache(stack, vol, geom, cand, args.r1, seed)
            print(f"  [seed {seed}] sphere (R,gamma): {time.time()-t:.1f}s", flush=True)
        if need_carm:
            t = time.time()
            cc = _carm_feasible_candidates(args.kmax, geom["sid"], seed)
            carm_pre = _build_cache(stack, vol, geom, cc, args.r1, seed)
            print(f"  [seed {seed}] C-arm feasible (R,gamma): {time.time()-t:.1f}s",
                  flush=True)
        for lbl, regime, fam, impl, _nc in methods:
            for k in k_values:
                t = time.time()
                try:
                    src = _select(stack, impl, k, vol, geom, sphere_pre,
                                  carm_pre, args.n_candidates, seed,
                                  bundle_kwargs={
                                      **({"smooth_constraint": True}
                                         if args.smooth_constraint else {}),
                                      **({} if args.bundle_target_mm is None else
                                         {"bundle_target": mx.array(
                                             [float(v) for v in
                                              args.bundle_target_mm.split(",")],
                                             dtype=mx.float32)}),
                                      **({} if args.bundle_schedule is None
                                         else {"bundle_schedule": args.bundle_schedule}),
                                      **({"cov_survival_weight": True}
                                         if args.cov_survival_weight else {}),
                                      **({} if args.bundle_lambda is None
                                         else {"lambda_bundle": args.bundle_lambda}),
                                      "bundle_n_samples": args.bundle_n_samples,
                                      "bundle_clip_to_volume":
                                          bool(args.bundle_clip)})
                    recon = _reconstruct(vol, src, geom, args.sart_iter,
                                         photon, seed)
                    m = _compute_metrics(vol, recon, peak, QUALITY_METRICS, stack)
                    for q in QUALITY_METRICS:
                        agg[(lbl, k)][q].append(float(m[q]))
                    for j, rmask in enumerate(roi_masks):
                        rm = _compute_metrics(vol, recon, peak, ROI_METRICS,
                                              stack, roi_mask=rmask)
                        for q in ROI_METRICS:
                            agg[(lbl, k)][f"roi{j}_{q[4:]}"].append(
                                float(rm[q]))
                    print(f"  [seed {seed}] {lbl:26s} ({regime:12s}) k={k}  "
                          f"PSNR={m['psnr']:.3f}  HFEN={m['hfen']:.3f}  "
                          f"({time.time()-t:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  [seed {seed}] {lbl:26s} k={k}  FAIL {repr(e)[:80]}",
                          flush=True)

    rows = []
    for (lbl, k), d in agg.items():
        if not d["psnr"]:
            continue
        regime, fam = meta[lbl]
        row = {"phantom": args.phantom, "method": lbl, "regime": regime,
               "family": fam, "k": k, "n_seeds": len(d["psnr"])}
        for q in metric_names:
            mean, std = _ms(d[q])
            row[f"{q}_mean"] = round(mean, 4)
            row[f"{q}_std"] = round(std, 4)
        rows.append(row)
    rows.sort(key=lambda r: (r["k"], r["regime"], r["method"]))
    _write(rows, args, k_values, seeds, methods, out_root,
           metric_names=metric_names, roi_specs=roi_specs)
    try:
        _plot(rows, args, k_values, out_root)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] figure failed: {e}", flush=True)
    print("\nDone.", flush=True)
    return 0


def _write(rows, args, k_values, seeds, methods, out_root,
           metric_names=None, roi_specs=None):
    metric_names = metric_names or list(QUALITY_METRICS)
    csv_path = out_root / "results" / f"kinematics_{args.phantom}.csv"
    json_path = out_root / "results" / f"kinematics_{args.phantom}.json"
    fields = ["phantom", "method", "regime", "family", "k", "n_seeds"] + \
             [f"{q}_{s}" for q in metric_names for s in ("mean", "std")]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    json_path.write_text(json.dumps(
        {"config": {"phantom": args.phantom, "resolution": args.resolution,
                    "kmax": args.kmax,
                    "photon": (None if args.noisefree else args.photon),
                    "noisefree": bool(args.noisefree),
                    "k_values": k_values, "seeds": seeds,
                    "methods": [method[0] for method in methods],
                    "roi_specs": roi_specs or []},
         "results": rows},
        indent=2))
    print(f"\nWrote {csv_path}  ({len(rows)} rows)", flush=True)
    print(f"Wrote {json_path}", flush=True)


def _plot(rows, args, k_values, out_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = ["single-axis", "limited-carm", "two-axis", "free-sphere"]
    fam_color = {"discrete": "#7f7f7f", "ours": "#d62728"}
    fig, axes = plt.subplots(1, len(k_values),
                             figsize=(6.2 * len(k_values), 4.4), squeeze=False)
    for j, k in enumerate(k_values):
        ax = axes[0][j]
        sub = [r for r in rows if r["k"] == k]
        sub.sort(key=lambda r: (regimes.index(r["regime"])
                                if r["regime"] in regimes else 99, r["family"]))
        labels = [r["method"] for r in sub]
        vals = [r["psnr_mean"] for r in sub]
        errs = [r["psnr_std"] for r in sub]
        colors = [fam_color.get(r["family"], "#1f77b4") for r in sub]
        xpos = list(range(len(sub)))
        ax.bar(xpos, vals, yerr=errs, capsize=3, color=colors)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"{args.phantom}  k={k}")
        ax.grid(True, axis="y", alpha=0.25)
        lo = min(vals) - 1.0
        ax.set_ylim(lo, max(vals) + 0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_color.values()]
    fig.legend(handles, list(fam_color), loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Reconstruction PSNR by kinematic regime "
                 "(discrete vs differentiable)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    png = out_root / "figures" / f"kinematics_{args.phantom}.png"
    pdf = out_root / "figures" / f"kinematics_{args.phantom}.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}", flush=True)
    print(f"Wrote {pdf}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
