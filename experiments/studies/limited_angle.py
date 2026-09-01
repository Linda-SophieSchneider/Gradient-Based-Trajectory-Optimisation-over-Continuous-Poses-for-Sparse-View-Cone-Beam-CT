r"""Limited-angle view selection: continuous optimisation on a constrained
feasible manifold, with absorption-aware ROI targeting.

Limited-angle acquisition (a restricted angular range) is a realistic, common
setting: interventional C-arm, in-situ industrial inspection, laminography of
flat parts.  The Tuy condition is violated, so *which* feasible views are taken
matters a great deal, and for a deeply occluded ROI the few low-attenuation
windows may sit anywhere inside the allowed wedge.  The discrete swap search must
re-discretise the feasible set and is noise-blind; our objective optimises Tuy
coverage (optionally at an ROI) plus a bundle absorption penalty *continuously on
the constraint manifold*, expressed through the gantry parametrisation.

Constraints (selected): the 120-degree azimuthal wedge (``wedge120``) and
laminography at a fixed ~30-degree tilt (``lamino``).

Conditions, all reconstructed under Poisson photon noise:
  uniform_wedge   k evenly-spaced feasible sources (naive)
  vcls_wedge      discrete swap search on a feasible candidate pool
  ours_cov        continuous coverage on the manifold (global)
  ours_roi_abs    continuous coverage at the ROI + bundle absorption avoidance

Metrics: global PSNR and ROI-PSNR/SSIM on a strongly-occluded off-centre ROI.

Usage::

    python run_limited_angle.py --phantom moderate --constraints wedge120,lamino \
        --resolution 192 --seeds 0,1,2
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

from experiments.run import (  # type: ignore
    _compute_metrics,
    _load_mlx_stack,
    _load_phantom_pair,
    _resolve_geometry,
    _resolve_roi_context,
)
from differentiable_coverage.eval.geometry import geometry_from_sources
from differentiable_coverage.eval.reco import simulate_sinogram, reconstruct_sart_volume
from differentiable_coverage.eval.vcl import vcls_select
from differentiable_coverage.absorption_bundle import (
    DEFAULT_BUNDLE_PENALTY_TARGET,
    BundleAbsorptionConfig,
    bundle_path_integral,
    calibrate_bundle_weight,
)
from differentiable_coverage.optimize import adam_ascent
from differentiable_coverage.score import (
    ScoreConfig, saturated_coverage, sample_unit_sphere, greedy_source_init,
)
from differentiable_coverage.trajectory import (CArmTwoAxisGantry,
                                               SmoothTwoAxisGantry)

from experiments.studies.roi_occluded import _find_occluded_center, _topn_separated  # type: ignore

OUT_ROOT = Path("experiments/limited_angle")
METRICS = ["psnr", "roi_psnr", "roi_ssim", "roi_hfen"]
D2R = math.pi / 180.0

# Phantoms in this study are centred on the world origin; the bundle
# world→index mapping is anchored there, independent of each ROI centre
# (REV-P0-02).  Single source of truth for the bundle calls and the JSON
# provenance record.
VOLUME_CENTER_MM = (0.0, 0.0, 0.0)

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
}

# (theta_min, theta_max, phi_min, phi_max) in degrees
CONSTRAINTS = {
    "wedge120": (-60.0, 60.0, -12.0, 12.0),   # 120-deg azimuthal missing-wedge
    "lamino":  (-180.0, 180.0, 25.0, 35.0),   # laminography, fixed ~30-deg tilt
}


def _gantry(sid, spec):
    tmn, tmx, pmn, pmx = spec
    return CArmTwoAxisGantry(sid=sid, theta_min=tmn * D2R, theta_max=tmx * D2R,
                             phi_min=pmn * D2R, phi_max=pmx * D2R)


def _feasible_random(gantry, n, spec, seed):
    tmn, tmx, pmn, pmx = spec
    rng = np.random.default_rng(seed)
    th = rng.uniform(tmn * D2R, tmx * D2R, size=n)
    ph = rng.uniform(pmn * D2R, pmx * D2R, size=n)
    return gantry(mx.array(np.stack([th, ph], -1).astype(np.float32)))


def _feasible_uniform(gantry, k, spec):
    tmn, tmx, pmn, pmx = spec
    th = np.linspace(tmn * D2R, tmx * D2R, k, endpoint=(tmx - tmn < 359))
    ph = np.full(k, 0.5 * (pmn + pmx) * D2R, dtype=np.float32)
    return gantry(mx.array(np.stack([th, ph], -1).astype(np.float32)))


def _to_params(sources, sid):
    s = np.asarray(sources, np.float32)
    theta = np.arctan2(-s[:, 0], s[:, 1])
    phi = np.arcsin(np.clip(s[:, 2] / max(sid, 1e-6), -1, 1))
    return mx.array(np.stack([theta, phi], -1).astype(np.float32))


def _constrained_select(gantry, k, vol, geom, spec, *, roi_center, roi_points,
                        roi_weights, lambda_bundle, bcfg, seed, n_steps=100,
                        lr=0.05, n_normals=2000, volume_center=None,
                        smooth_constraint=True):
    cfg = ScoreConfig(tau=math.sin(15.0 * D2R))
    normals = sample_unit_sphere(n_normals)
    # init: Tuy-greedy on a feasible candidate pool, mapped to (theta, phi).
    # The pool and the uniform baseline are sampled in ANGLES, so they keep the
    # clipped chart passed in; only the optimisation uses the smooth chart,
    # whose parameters are unconstrained raw coordinates (2026-08-27).
    cand = _feasible_random(gantry, 200, spec, seed)
    init_src = greedy_source_init(cand, roi_center, normals, k, cfg,
                                  roi_points=roi_points, roi_weights=roi_weights)
    if smooth_constraint:
        tmn, tmx, pmn, pmx = spec
        gantry = SmoothTwoAxisGantry(
            sid=geom["sid"], theta_min=tmn * D2R, theta_max=tmx * D2R,
            phi_min=pmn * D2R, phi_max=pmx * D2R)
        init = gantry.inverse(_to_params(init_src, geom["sid"]))
    else:
        init = gantry.clamp(_to_params(init_src, geom["sid"]))

    def loss(params, _step):
        src = gantry(params)
        cov = saturated_coverage(src, roi_center, normals,
                                 mx.ones(src.shape[0], dtype=mx.float32), cfg,
                                 roi_points=roi_points, roi_weights=roi_weights)
        l = cov
        if lambda_bundle > 0.0:
            # The phantom volume is centred on the world origin; roi_center is
            # only the bundle target (REV-P0-02: never recentre the volume).
            tau = bundle_path_integral(src, roi_center, vol, bcfg,
                                       volume_center=volume_center)
            l = l - lambda_bundle * mx.mean(tau)
        return l

    refined, _ = adam_ascent(loss, init, n_steps=n_steps, lr=lr,
                             project_fn=gantry.clamp, lr_schedule="cosine",
                             lr_min=lr * 0.05, patience=15, return_best=True)
    return gantry(refined)


def _reconstruct(vol, src, geom, sart_iter, photon, ns):
    sp, dc, du, dv = geometry_from_sources(src, sid=geom["sid"], sdd=geom["sdd"])
    sino = simulate_sinogram(
        vol, sp, dc, du, dv, det_u=geom["det_voxels"], det_v=geom["det_voxels"],
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"], photon_count=photon, noise_seed=ns)
    mx.eval(sino)
    res = reconstruct_sart_volume(
        vol.shape, sino, sp, dc, du, dv, du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"], iteration_count=sart_iter,
        show_progress=False)
    mx.eval(res.reconstruction)
    return res.reconstruction


def _ms(xs):
    return statistics.fmean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phantom", default="moderate", choices=list(PHANTOMS))
    ap.add_argument("--constraints", default="wedge120,lamino")
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--kmax", type=int, default=360)
    ap.add_argument("--sart-iter", type=int, default=15)
    ap.add_argument("--photon", type=float, default=1.0e4)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--roi-radius-mm", type=float, default=5.0)
    ap.add_argument("--n-roi", type=int, default=4)
    ap.add_argument("--roi-centers", default=None,
                    help="semicolon-separated 'x,y,z' world-mm ROI centres; "
                         "bypasses the photon-starvation screening, e.g. for "
                         "phantoms that ship designed defect ROIs (the "
                         "lof_plate keeps tau <= 6.7 by construction, so the "
                         "pre-registered starvation criterion never fires)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--noise-seeds", default="0,1")
    ap.add_argument("--r1", type=float, default=1.0e-3)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bundle-target", type=float,
                    default=DEFAULT_BUNDLE_PENALTY_TARGET)
    # Frozen production quadrature (bundle_quadrature_convergence study):
    # in-volume clipped midpoint rule with 256 samples per ray.
    ap.add_argument("--bundle-n-samples", type=int, default=256)
    ap.add_argument("--bundle-clip", type=int, default=1)
    ap.add_argument("--paired-baseline", default="uniform_wedge",
                    choices=["uniform_wedge", "vcls_wedge", "ours_cov"])
    ap.add_argument(
        "--out-root",
        default=str(OUT_ROOT),
        help="output directory root (default: experiments/limited_angle)",
    )
    args = ap.parse_args(argv)

    constraints = [c.strip() for c in args.constraints.split(",") if c.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    nseeds = [int(x) for x in args.noise_seeds.split(",") if x.strip()]
    out_root = Path(args.out_root)
    (out_root / "results").mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)

    stack = _load_mlx_stack()
    info = PHANTOMS[args.phantom]
    spec = {**info["spec"], "resolution": args.resolution}
    geom = _resolve_geometry(info["geometry"], args.resolution)
    vol, _ = _load_phantom_pair(spec, stack)
    mx.eval(vol)
    peak = float(vol.max())
    vs = geom["voxel_pitch"]
    sid = geom["sid"]

    if args.roi_centers:
        centers = [(0.0, np.array([float(v) for v in c.split(",")], np.float64),
                    0.0, 0.0)
                   for c in args.roi_centers.split(";") if c.strip()]
    else:
        cands = _find_occluded_center(stack, vol, geom, args.photon,
                                      args.roi_radius_mm)
        centers = _topn_separated(cands, args.n_roi, 18.0)
    if not centers:
        raise SystemExit("no occluded ROI found")

    print("=" * 74, flush=True)
    print(f"Limited-angle study  phantom={args.phantom} {tuple(vol.shape)} "
          f"k={args.k} I0={args.photon:.0e}", flush=True)
    print(f"  constraints={constraints}  ROIs={len(centers)}  seeds={seeds} "
          f"noise={nseeds}", flush=True)
    print("=" * 74, flush=True)

    conditions = ["uniform_wedge", "vcls_wedge", "ours_cov", "ours_roi_abs"]
    # ROI metrics aggregate over (ROI, selection seed); the ROI-independent
    # global PSNR of the three global conditions is tracked once per
    # selection seed only — never duplicated per ROI (REV-P1-03 estimand).
    ROI_METRICS = [m for m in METRICS if m != "psnr"]
    agg_roi = {(con, c): {m: [] for m in ROI_METRICS} for con in conditions
               for c in constraints}
    agg_glob = {(con, c): [] for con in conditions for c in constraints}
    rows_detail = []

    # Per-ROI context and bundle-weight calibration are loop-invariant
    # across constraints and selection seeds — computed once per ROI.
    bcfg = BundleAbsorptionConfig(
        roi_radius=args.roi_radius_mm, n_rays_u=5, n_rays_v=9,
        n_samples=args.bundle_n_samples, voxel_spacing=vs,
        clip_to_volume=bool(args.bundle_clip))
    probes = sample_unit_sphere(256) * sid
    roi_ctxs, lam_by_roi = [], []
    for (_sf0, center, _mt, _win) in centers:
        roi = _resolve_roi_context(
            {"roi": {"type": "sphere",
                     "center_mm": [float(c) for c in center],
                     "radius_mm": args.roi_radius_mm, "point_grid": 5}},
            vol, geom, stack, want_mask=True)
        w = np.asarray(roi["weights"], np.float64)
        w = w / max(w.sum(), 1e-12)
        roi["_pw"] = dict(roi_points=roi["points"],
                          roi_weights=mx.array(w.astype(np.float32)))
        med = float(mx.median(bundle_path_integral(
            probes, roi["center"], vol, bcfg,
            volume_center=mx.array(VOLUME_CENTER_MM))))
        lam_by_roi.append(calibrate_bundle_weight(med, args.bundle_target))
        roi_ctxs.append(roi)

    for cname in constraints:
        cspec = CONSTRAINTS[cname]
        gantry = _gantry(sid, cspec)
        for seed in seeds:
            cand_feas = _feasible_random(gantry, args.kmax, cspec, seed)
            pre = stack["compute_R_gamma"](
                vol, cand_feas, sid=sid, sdd=geom["sdd"],
                det_shape=(geom["det_voxels"], geom["det_voxels"]),
                du=geom["det_pitch"], dv=geom["det_pitch"],
                voxel_spacing=vs, r1=args.r1, seed=seed)
            mx.eval(pre.R)
            # ROI-independent selections (once per seed)
            uni_src = _feasible_uniform(gantry, args.k, cspec)
            vidx, _ = vcls_select(pre, args.k, seed=seed)
            vcls_src = pre.candidate_sources[mx.array(vidx)]
            cov_src = _constrained_select(
                gantry, args.k, vol, geom, cspec, roi_center=mx.zeros(3),
                roi_points=None, roi_weights=None, lambda_bundle=0.0,
                bcfg=None, seed=seed, n_steps=args.n_steps, lr=args.lr)
            mx.eval(uni_src); mx.eval(vcls_src); mx.eval(cov_src)
            global_srcs = {"uniform_wedge": uni_src, "vcls_wedge": vcls_src,
                           "ours_cov": cov_src}
            # global reconstructions (ROI-independent), per noise seed
            grecon = {(con, ns): _reconstruct(vol, s, geom, args.sart_iter,
                                              args.photon, ns)
                      for con, s in global_srcs.items() for ns in nseeds}
            for ri, roi in enumerate(roi_ctxs):
                roi_abs_src = _constrained_select(
                    gantry, args.k, vol, geom, cspec, roi_center=roi["center"],
                    lambda_bundle=lam_by_roi[ri], bcfg=bcfg, seed=seed,
                    n_steps=args.n_steps, lr=args.lr,
                    volume_center=mx.array(VOLUME_CENTER_MM), **roi["_pw"])
                mx.eval(roi_abs_src)
                # evaluate all conditions on this ROI
                for con in conditions:
                    per = {m: [] for m in METRICS}
                    for ns in nseeds:
                        if con == "ours_roi_abs":
                            recon = _reconstruct(vol, roi_abs_src, geom,
                                                 args.sart_iter, args.photon, ns)
                        else:
                            recon = grecon[(con, ns)]
                        m = _compute_metrics(vol, recon, peak, METRICS, stack,
                                             roi_mask=roi["mask"])
                        for q in METRICS:
                            per[q].append(float(m[q]))
                        # Raw artifact: ROI, selection seed, and noise seed
                        # stay separate factors (REV-P1-03).
                        rows_detail.append({
                            "constraint": cname, "roi": ri,
                            "selection_seed": seed, "noise_seed": ns,
                            "condition": con,
                            **{q: round(float(m[q]), 6) for q in METRICS}})
                    means = {q: statistics.fmean(per[q]) for q in METRICS}
                    for q in ROI_METRICS:
                        agg_roi[(con, cname)][q].append(means[q])
                    # Global PSNR is ROI-independent for the three global
                    # conditions: record it once per selection seed instead
                    # of duplicating it across the n_roi ROIs.  ours_roi_abs
                    # reconstructs per ROI, so it contributes per (seed, ROI).
                    if con == "ours_roi_abs" or ri == 0:
                        agg_glob[(con, cname)].append(means["psnr"])
                print(f"  [{cname} seed {seed} ROI {ri}]  "
                      f"uni={agg_roi[('uniform_wedge',cname)]['roi_psnr'][-1]:.2f} "
                      f"vcls={agg_roi[('vcls_wedge',cname)]['roi_psnr'][-1]:.2f} "
                      f"cov={agg_roi[('ours_cov',cname)]['roi_psnr'][-1]:.2f} "
                      f"roi_abs={agg_roi[('ours_roi_abs',cname)]['roi_psnr'][-1]:.2f}"
                      f"  (glob ours_cov={agg_glob[('ours_cov',cname)][-1]:.2f})",
                      flush=True)

    rows = []
    for (con, cname), d in agg_roi.items():
        if not d["roi_psnr"]:
            continue
        g = agg_glob[(con, cname)]
        row = {"phantom": args.phantom, "constraint": cname, "condition": con,
               "k": args.k, "photon": args.photon,
               "n": len(d["roi_psnr"]), "psnr_n": len(g)}
        mean, std = _ms(g)
        row["psnr_mean"] = round(mean, 4)
        row["psnr_std"] = round(std, 4)
        for q in ROI_METRICS:
            mean, std = _ms(d[q])
            row[f"{q}_mean"] = round(mean, 4)
            row[f"{q}_std"] = round(std, 4)
        rows.append(row)
    rows.sort(key=lambda r: (r["constraint"], r["condition"]))
    _write(
        rows, rows_detail, args, constraints, seeds, nseeds, centers, out_root
    )
    try:
        _plot(rows, args, constraints, out_root)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] figure failed: {e}", flush=True)
    # console summary
    print("\n--- summary (ROI-PSNR / global-PSNR mean) ---", flush=True)
    for cname in constraints:
        print(f"  {cname}:", flush=True)
        for con in conditions:
            r = next((x for x in rows if x["constraint"] == cname
                      and x["condition"] == con), None)
            if r:
                print(f"    {con:14s} ROI={r['roi_psnr_mean']:.2f}±{r['roi_psnr_std']:.2f}"
                      f"  glob={r['psnr_mean']:.2f}  ROI_SSIM={r['roi_ssim_mean']:.3f}",
                      flush=True)
    print("\nDone.", flush=True)
    return 0


def _paired_rows(rows_detail, baseline):
    """Paired condition-vs-baseline differences (REV-P1-03).

    Pairs on (constraint, ROI, selection_seed, noise_seed); per selection
    seed the paired differences are averaged over the shared noise seeds,
    then cross-seed mean/SD/min/max are reported.  ROI metrics per ROI;
    global PSNR once per condition (``roi = "global"``) for the
    ROI-independent conditions and per ROI for ``ours_roi_abs``.
    """
    by = {(r["constraint"], r["condition"], r["roi"],
           r["selection_seed"], r["noise_seed"]): r for r in rows_detail}
    cnames = sorted({r["constraint"] for r in rows_detail})
    conds = sorted({r["condition"] for r in rows_detail} - {baseline})
    rois = sorted({r["roi"] for r in rows_detail})
    seeds = sorted({r["selection_seed"] for r in rows_detail})
    nseeds = sorted({r["noise_seed"] for r in rows_detail})

    def stats_over_seeds(cname, con, ri, metric):
        per_seed = []
        for s in seeds:
            diffs = []
            for ns in nseeds:
                a = by.get((cname, con, ri, s, ns))
                b = by.get((cname, baseline, ri, s, ns))
                if a is not None and b is not None:
                    diffs.append(a[metric] - b[metric])
            if diffs:
                per_seed.append(statistics.fmean(diffs))
        if not per_seed:
            return None
        return {
            "mean": round(statistics.fmean(per_seed), 4),
            "sd": round(statistics.pstdev(per_seed)
                        if len(per_seed) > 1 else 0.0, 4),
            "min": round(min(per_seed), 4),
            "max": round(max(per_seed), 4),
        }

    out = []
    for cname in cnames:
        for con in conds:
            for ri in rois:
                metrics = [q for q in METRICS
                           if q != "psnr" or con == "ours_roi_abs"]
                row = {"constraint": cname, "condition": con,
                       "baseline": baseline, "roi": ri,
                       "n_selection": len(seeds)}
                got = False
                for q in metrics:
                    st = stats_over_seeds(cname, con, ri, q)
                    if st:
                        got = True
                        for key, val in st.items():
                            row[f"d{q}_{key}"] = val
                if got:
                    out.append(row)
            if con != "ours_roi_abs":
                st = stats_over_seeds(cname, con, rois[0], "psnr")
                if st:
                    out.append({"constraint": cname, "condition": con,
                                "baseline": baseline, "roi": "global",
                                "n_selection": len(seeds),
                                **{f"dpsnr_{k}": v for k, v in st.items()}})
    return out


def _write(
    rows, rows_detail, args, constraints, seeds, nseeds, centers, out_root
):
    csv_path = out_root / "results" / f"limited_angle_{args.phantom}.csv"
    json_path = out_root / "results" / f"limited_angle_{args.phantom}.json"
    paired_path = out_root / "results" / \
        f"limited_angle_{args.phantom}.paired.csv"
    fields = ["phantom", "constraint", "condition", "k", "photon",
              "n", "psnr_n"] + \
             [f"{q}_{s}" for q in METRICS for s in ("mean", "std")]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    paired = _paired_rows(rows_detail, args.paired_baseline)
    if paired:
        pf = ["constraint", "condition", "baseline", "roi", "n_selection"] + \
             [f"d{q}_{s}" for q in METRICS
              for s in ("mean", "sd", "min", "max")]
        with paired_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pf)
            w.writeheader()
            for r in paired:
                w.writerow({k: r.get(k, "") for k in pf})
    json_path.write_text(json.dumps(
        {"config": {"phantom": args.phantom, "resolution": args.resolution,
                    "constraints": {c: CONSTRAINTS[c] for c in constraints},
                    "k": args.k, "photon": args.photon, "kmax": args.kmax,
                    "n_steps": args.n_steps, "lr": args.lr,
                    "bundle_penalty_target": args.bundle_target,
                    "bundle_n_samples": args.bundle_n_samples,
                    "bundle_clip_to_volume": bool(args.bundle_clip),
                    "paired_baseline": args.paired_baseline,
                    "selection_seeds": seeds, "noise_seeds": nseeds,
                    "roi_centers_mm": [c[1].round(2).tolist() for c in centers],
                    # REV-P0-02 provenance: the bundle world→index mapping is
                    # anchored at the volume centre, independent of roi_center.
                    "bundle_volume_center_mm": list(VOLUME_CENTER_MM)},
         "aggregate": rows, "detail": rows_detail,
         "paired": paired}, indent=2))
    print(f"\nWrote {csv_path}  ({len(rows)} rows)", flush=True)
    if paired:
        print(f"Wrote {paired_path}  ({len(paired)} rows)", flush=True)
    print(f"Wrote {json_path}", flush=True)


def _plot(rows, args, constraints, out_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["uniform_wedge", "vcls_wedge", "ours_cov", "ours_roi_abs"]
    colors = {"uniform_wedge": "#7f7f7f", "vcls_wedge": "#1f77b4",
              "ours_cov": "#ff7f0e", "ours_roi_abs": "#d62728"}
    fig, axes = plt.subplots(1, len(constraints),
                             figsize=(6.0 * len(constraints), 4.4), squeeze=False)
    for j, cname in enumerate(constraints):
        ax = axes[0][j]
        sub = {r["condition"]: r for r in rows if r["constraint"] == cname}
        labels = [c for c in order if c in sub]
        roi = [sub[c]["roi_psnr_mean"] for c in labels]
        err = [sub[c]["roi_psnr_std"] for c in labels]
        ax.bar(range(len(labels)), roi, yerr=err, capsize=3,
               color=[colors[c] for c in labels])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("ROI PSNR (dB)")
        ax.set_title(f"{args.phantom}  {cname}")
        ax.grid(True, axis="y", alpha=0.25)
        if roi:
            ax.set_ylim(min(roi) - 1.0, max(roi) + 0.6)
    fig.suptitle("Limited-angle: absorption-aware ROI targeting on the "
                 "constrained manifold (ours) vs discrete / uniform", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = out_root / "figures" / f"limited_angle_{args.phantom}.png"
    pdf = out_root / "figures" / f"limited_angle_{args.phantom}.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}", flush=True)
    print(f"Wrote {pdf}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
