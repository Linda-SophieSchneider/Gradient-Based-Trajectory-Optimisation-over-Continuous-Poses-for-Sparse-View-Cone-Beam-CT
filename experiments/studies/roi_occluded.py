r"""ROI-weighted, absorption-aware selection for a deeply occluded ROI.

This is the regime where a task-specific differentiable objective should beat a
noise-blind global selector.  We place the ROI on an interior region of the
off-centre industrial phantom that is *strongly absorbed from most viewing
directions* --- many rays to it are photon-starved --- but that still has a few
low-attenuation windows.  Reconstructing such an ROI under photon noise rewards
finding those windows.

The discrete swap search optimises a global, noise-blind information surrogate:
it ranks views by clean information content and is blind to which rays to the
ROI survive.  Our continuous objective composes two terms the swap search cannot
express: Tuy coverage evaluated *at the ROI* (geometric targeting) and the
analytic bundle absorption penalty *toward the ROI* (steer away from starved
windows).  Both are differentiable in the source position and optimised jointly.

Conditions (all reconstructed under Poisson photon noise):
  vcls_global   discrete swap search, global noise-blind surrogate
  vcls_roi      discrete swap search, (R, gamma) re-estimated on ROI voxels
  ours_global   continuous coverage, isocentre (no ROI, no absorption)
  ours_roi_geo  continuous coverage at the ROI (geometry only)
  ours_roi_abs  continuous coverage at the ROI + bundle absorption avoidance

The ROI is found automatically as the support voxel whose line integral to a
probe sphere is large for most directions (occluded) yet small for some (a
window).

Usage::

    python run_roi_occluded.py --phantom moderate --resolution 192 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import csv
import json
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
from differentiable_coverage.eval.trajectories import greedy_adam_vcl_continuous
from differentiable_coverage.eval.vcl import vcls_select
from differentiable_coverage.absorption_bundle import (
    BundleAbsorptionConfig,
    bundle_path_integral,
    calibrate_bundle_weight,
)

OUT_ROOT = Path("experiments/roi_occluded")
METRICS = ["roi_psnr", "roi_ssim", "roi_hfen", "psnr"]
PHANTOMS = {
    "moderate": {"spec": {"type": "milp_npy",
                          "path": "data/moderate_asd_pocs_384.npy"},
                 "geometry": "milp"},
    "mild": {"spec": {"type": "milp_npy",
                      "path": "data/mild_asd_pocs_384.npy"},
             "geometry": "milp"},
}


def _world_of_voxel(vox, N, vs):
    return np.array([(vox[2] - (N - 1) / 2) * vs,
                     (vox[1] - (N - 1) / 2) * vs,
                     (vox[0] - (N - 1) / 2) * vs], dtype=np.float32)


def _local_std(vol_np, vox, half, vs, radius_mm):
    """Std of mu inside the ROI sphere around a voxel (structure proxy)."""
    z, y, x = vox
    r = int(round(radius_mm / vs))
    sl = vol_np[max(z - r, 0):z + r + 1, max(y - r, 0):y + r + 1,
                max(x - r, 0):x + r + 1]
    return float(sl.std())


def _find_occluded_center(stack, vol, geom, i0, radius_mm, n_probe=180,
                          n_cand=600, seed=0):
    """Support voxel that is occluded (high photon-starved fraction of probe
    views) yet has a clear window *and* local structure, so the ROI metrics are
    meaningful and view selection has something to recover."""
    vs = geom["voxel_pitch"]
    N = vol.shape[0]
    vol_np = np.asarray(vol)
    vmax = float(vol_np.max())
    sup = np.argwhere(vol_np > 0.15 * vmax)
    rng = np.random.default_rng(seed)
    pick = sup[rng.choice(len(sup), size=min(n_cand, len(sup)), replace=False)]
    probes = stack["sample_unit_sphere"](n_probe, seed=0) * geom["sid"]
    cfg0 = BundleAbsorptionConfig(roi_radius=0.0, n_rays_u=1, n_rays_v=1,
                                  n_samples=80, voxel_spacing=vs)
    # structure threshold: median local std over the support
    struct_thr = 0.20 * vmax
    cands = []
    for vox in pick:
        std = _local_std(vol_np, vox, None, vs, radius_mm)
        if std < struct_thr:
            continue
        c = _world_of_voxel(vox, N, vs)
        tau = np.asarray(bundle_path_integral(probes, mx.array(c), vol, cfg0))
        starved = float(np.mean(i0 * np.exp(-tau) < 10))
        window = float(np.percentile(tau, 10))
        if window < 4.0 and starved > 0.3:
            cands.append((starved, c, float(tau.mean()), window))
    cands.sort(key=lambda r: -r[0])
    return cands  # list of (starved_frac, center_world, mean_tau, p10_tau)


def _topn_separated(cands, n, min_sep_mm):
    """Greedily take the n highest-scoring centres that are at least
    ``min_sep_mm`` apart, so the regime study covers distinct regions."""
    chosen = []
    for c in cands:
        if all(np.linalg.norm(c[1] - o[1]) >= min_sep_mm for o in chosen):
            chosen.append(c)
        if len(chosen) >= n:
            break
    return chosen


def _reconstruct(vol, src, geom, sart_iter, photon, noise_seed):
    sp, dc, du, dv = geometry_from_sources(src, sid=geom["sid"], sdd=geom["sdd"])
    sino = simulate_sinogram(
        vol, sp, dc, du, dv,
        det_u=geom["det_voxels"], det_v=geom["det_voxels"],
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"], photon_count=photon, noise_seed=noise_seed)
    mx.eval(sino)
    res = reconstruct_sart_volume(
        vol.shape, sino, sp, dc, du, dv,
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"], iteration_count=sart_iter,
        show_progress=False)
    mx.eval(res.reconstruction)
    return res.reconstruction


def _cache(stack, vol, geom, cand, r1, seed, roi_mask=None):
    pre = stack["compute_R_gamma"](
        vol, cand, sid=geom["sid"], sdd=geom["sdd"],
        det_shape=(geom["det_voxels"], geom["det_voxels"]),
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"], r1=r1, seed=seed, roi_mask=roi_mask)
    mx.eval(pre.R)
    return pre


def _ms(xs):
    return statistics.fmean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phantom", default="moderate", choices=list(PHANTOMS))
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--kmax", type=int, default=360)
    ap.add_argument("--n-candidates", type=int, default=360)
    ap.add_argument("--sart-iter", type=int, default=15)
    ap.add_argument("--roi-radius-mm", type=float, default=5.0)
    ap.add_argument("--roi-center-mm", default="",
                    help="override ROI centre (x,y,z mm); else auto-find")
    ap.add_argument("--n-roi", type=int, default=6,
                    help="number of distinct occluded ROIs in the regime study")
    ap.add_argument("--min-sep-mm", type=float, default=18.0,
                    help="minimum separation between regime ROIs")
    ap.add_argument("--photon-levels", default="1e3,1e4")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--seeds", default="0,1,2", help="selection seeds")
    ap.add_argument("--noise-seeds", default="0,1,2")
    ap.add_argument("--r1", type=float, default=1.0e-3)
    args = ap.parse_args(argv)

    photon_levels = [float(x) for x in args.photon_levels.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    nseeds = [int(x) for x in args.noise_seeds.split(",") if x.strip()]
    (OUT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)

    stack = _load_mlx_stack()
    info = PHANTOMS[args.phantom]
    spec = {**info["spec"], "resolution": args.resolution}
    geom = _resolve_geometry(info["geometry"], args.resolution)
    vol, _ = _load_phantom_pair(spec, stack)
    mx.eval(vol)
    peak = float(vol.max())
    vs = geom["voxel_pitch"]

    if args.roi_center_mm:
        center = np.array([float(x) for x in args.roi_center_mm.split(",")],
                          dtype=np.float32)
        centers_info = [(0.0, center, 0.0, 0.0)]
    else:
        cands = _find_occluded_center(stack, vol, geom, min(photon_levels),
                                      args.roi_radius_mm)
        centers_info = _topn_separated(cands, args.n_roi, args.min_sep_mm)
        if not centers_info:
            raise SystemExit("no occluded-with-window structured ROI found")

    print("=" * 72, flush=True)
    print(f"ROI-occluded regime study  phantom={args.phantom} "
          f"{tuple(vol.shape)}", flush=True)
    print(f"  {len(centers_info)} occluded ROIs (radius {args.roi_radius_mm} mm), "
          f"k={args.k} photon={photon_levels} seeds={seeds} noise={nseeds}",
          flush=True)
    for ri, (sf0, c, mt, win) in enumerate(centers_info):
        print(f"   ROI {ri}: centre(mm)={c.round(1).tolist()} "
              f"starved={sf0:.2f} mean_tau={mt:.2f} window={win:.2f}", flush=True)
    print("=" * 72, flush=True)

    cfg0 = BundleAbsorptionConfig(roi_radius=0.0, n_rays_u=1, n_rays_v=1,
                                  n_samples=80, voxel_spacing=vs)
    conditions = ["vcls_global", "ours_roi_geo", "ours_roi_abs"]
    agg = {(c, i0): {m: [] for m in METRICS + ["starved"]}
           for c in conditions for i0 in photon_levels}
    # per (ROI, seed) win of absorption-aware ROI targeting over global VCLS
    delta = {i0: [] for i0 in photon_levels}
    rows_detail = []

    for seed in seeds:
        cand = stack["sample_unit_sphere"](args.kmax, seed=seed) * geom["sid"]
        pre_g = _cache(stack, vol, geom, cand, args.r1, seed)
        vidx, _ = vcls_select(pre_g, args.k, seed=seed)
        vcls_src = pre_g.candidate_sources[mx.array(vidx)]
        mx.eval(vcls_src)
        # hoist the global VCLS reconstruction (ROI-independent volume)
        vcls_recon = {(i0, ns): _reconstruct(vol, vcls_src, geom, args.sart_iter,
                                             i0, ns)
                      for i0 in photon_levels for ns in nseeds}
        for ri, (sf0, center, mt, win) in enumerate(centers_info):
            roi = _resolve_roi_context(
                {"roi": {"type": "sphere",
                         "center_mm": [float(c) for c in center],
                         "radius_mm": args.roi_radius_mm, "point_grid": 5}},
                vol, geom, stack, want_mask=True)
            w = np.asarray(roi["weights"], np.float64)
            w = w / max(w.sum(), 1e-12)
            roi_pw = {"roi_points": roi["points"],
                      "roi_weights": mx.array(w.astype(np.float32))}
            bcfg = BundleAbsorptionConfig(roi_radius=args.roi_radius_mm,
                                          n_rays_u=5, n_rays_v=9, n_samples=32,
                                          voxel_spacing=vs)
            probes = stack["sample_unit_sphere"](256, seed=0) * geom["sid"]
            med_tau = float(mx.median(
                bundle_path_integral(probes, roi["center"], vol, bcfg)))
            lam_abs = calibrate_bundle_weight(med_tau)
            common = dict(volume=vol, sdd=geom["sdd"], lambda_cov=1.0,
                          lambda_vcl=0.0, lambda_path=0.0,
                          init_method="greedy_tuy",
                          n_candidates=args.n_candidates, voxel_spacing=vs,
                          seed=seed)
            geo_src = greedy_adam_vcl_continuous(
                args.k, geom["sid"], roi_center=roi["center"],
                lambda_bundle=0.0, **roi_pw, **common)
            abs_src = greedy_adam_vcl_continuous(
                args.k, geom["sid"], roi_center=roi["center"],
                lambda_bundle=lam_abs, bundle_cfg=bcfg, **roi_pw, **common)
            mx.eval(geo_src); mx.eval(abs_src)
            srcs = {"vcls_global": vcls_src, "ours_roi_geo": geo_src,
                    "ours_roi_abs": abs_src}

            def _starved(src, i0):
                tau = np.asarray(bundle_path_integral(src, roi["center"], vol, cfg0))
                return float(np.mean(i0 * np.exp(-tau) < 10))

            for i0 in photon_levels:
                roi_psnr_by_cond = {}
                for cond, src in srcs.items():
                    per = {m: [] for m in METRICS}
                    for ns in nseeds:
                        recon = (vcls_recon[(i0, ns)] if cond == "vcls_global"
                                 else _reconstruct(vol, src, geom,
                                                   args.sart_iter, i0, ns))
                        m = _compute_metrics(vol, recon, peak, METRICS, stack,
                                             roi_mask=roi["mask"])
                        for q in METRICS:
                            per[q].append(float(m[q]))
                    means = {q: statistics.fmean(per[q]) for q in METRICS}
                    roi_psnr_by_cond[cond] = means["roi_psnr"]
                    agg[(cond, i0)]["starved"].append(_starved(src, i0))
                    for q in METRICS:
                        agg[(cond, i0)][q].append(means[q])
                    rows_detail.append({"roi": ri, "seed": seed, "condition": cond,
                                        "photon": i0, **means})
                d = roi_psnr_by_cond["ours_roi_abs"] - roi_psnr_by_cond["vcls_global"]
                delta[i0].append(d)
                print(f"  [seed {seed} ROI {ri}] I0={i0:.0e}  "
                      f"vcls={roi_psnr_by_cond['vcls_global']:.2f} "
                      f"roi_geo={roi_psnr_by_cond['ours_roi_geo']:.2f} "
                      f"roi_abs={roi_psnr_by_cond['ours_roi_abs']:.2f}  "
                      f"Δabs={d:+.2f}", flush=True)

    rows = []
    for (cond, i0), d in agg.items():
        if not d["roi_psnr"]:
            continue
        row = {"phantom": args.phantom, "condition": cond, "photon": i0,
               "k": args.k, "n": len(d["roi_psnr"])}
        sm, _ = _ms(d["starved"])
        row["sel_starved_mean"] = round(sm, 3)
        for q in METRICS:
            mean, std = _ms(d[q])
            row[f"{q}_mean"] = round(mean, 4)
            row[f"{q}_std"] = round(std, 4)
        rows.append(row)
    rows.sort(key=lambda r: (r["photon"], r["condition"]))

    delta_summary = {}
    print("\n--- absorption-aware ROI win over global VCLS (Δ ROI-PSNR) ---",
          flush=True)
    for i0 in photon_levels:
        ds = delta[i0]
        m, s = _ms(ds)
        fpos = float(np.mean([1.0 if x > 0 else 0.0 for x in ds]))
        delta_summary[f"{i0:.0e}"] = {"mean": round(m, 3), "std": round(s, 3),
                                      "n": len(ds), "frac_positive": round(fpos, 2)}
        print(f"  I0={i0:.0e}: mean Δ={m:+.3f} dB  std={s:.3f}  "
              f"frac_win={fpos:.2f}  (n={len(ds)})", flush=True)

    _write(rows, rows_detail, delta_summary, args, photon_levels, seeds, nseeds,
           centers_info)
    try:
        _plot(rows, args, photon_levels)
    except Exception as e:  # pragma: no cover
        print(f"[WARN] figure failed: {e}", flush=True)
    print("\nDone.", flush=True)
    return 0


def _write(rows, rows_detail, delta_summary, args, photon_levels, seeds, nseeds,
           centers_info):
    csv_path = OUT_ROOT / "results" / f"roi_occluded_{args.phantom}.csv"
    json_path = OUT_ROOT / "results" / f"roi_occluded_{args.phantom}.json"
    fields = ["phantom", "condition", "photon", "k", "n", "sel_starved_mean"] + \
             [f"{q}_{s}" for q in METRICS for s in ("mean", "std")]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    json_path.write_text(json.dumps(
        {"config": {"phantom": args.phantom, "resolution": args.resolution,
                    "kmax": args.kmax, "k": args.k,
                    "roi_radius_mm": args.roi_radius_mm,
                    "n_roi": len(centers_info),
                    "roi_centers_mm": [c[1].round(2).tolist() for c in centers_info],
                    "roi_starved": [round(c[0], 3) for c in centers_info],
                    "photon_levels": photon_levels, "seeds": seeds,
                    "noise_seeds": nseeds},
         "delta_abs_vs_global": delta_summary,
         "aggregate": rows, "detail": rows_detail}, indent=2))
    print(f"\nWrote {csv_path}  ({len(rows)} rows)", flush=True)
    print(f"Wrote {json_path}", flush=True)


def _plot(rows, args, photon_levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["vcls_global", "vcls_roi", "ours_global", "ours_roi_geo",
             "ours_roi_abs"]
    colors = {"vcls_global": "#7f7f7f", "vcls_roi": "#1f77b4",
              "ours_global": "#ff7f0e", "ours_roi_geo": "#9467bd",
              "ours_roi_abs": "#d62728"}
    fig, axes = plt.subplots(1, len(photon_levels),
                             figsize=(6.0 * len(photon_levels), 4.4), squeeze=False)
    for j, i0 in enumerate(photon_levels):
        ax = axes[0][j]
        sub = {r["condition"]: r for r in rows if r["photon"] == i0}
        labels = [c for c in order if c in sub]
        vals = [sub[c]["roi_psnr_mean"] for c in labels]
        errs = [sub[c]["roi_psnr_std"] for c in labels]
        ax.bar(range(len(labels)), vals, yerr=errs, capsize=3,
               color=[colors[c] for c in labels])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("ROI PSNR (dB)")
        ax.set_title(f"{args.phantom}  I0={i0:.0e}")
        ax.grid(True, axis="y", alpha=0.25)
        if vals:
            ax.set_ylim(min(vals) - 1.0, max(vals) + 0.6)
    fig.suptitle("Deeply-occluded ROI under photon noise: absorption-aware "
                 "ROI targeting (ours) vs noise-blind discrete", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = OUT_ROOT / "figures" / f"roi_occluded_{args.phantom}.png"
    pdf = OUT_ROOT / "figures" / f"roi_occluded_{args.phantom}.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}", flush=True)
    print(f"Wrote {pdf}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
