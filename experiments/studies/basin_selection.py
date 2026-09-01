"""Diagnostic probe: is the exploration bottleneck the SELECTION criterion?

The repulsive-ensemble explorer
(`differentiable_coverage.eval.trajectories.greedy_ensemble_vcl_continuous`)
settles ``N`` continuous view configurations in structurally different basins
of the VCL surrogate (VCLS, Tuy-greedy, and random-sphere starts), then keeps
the member with the best *clean* VCL surrogate value.  The ablation section
shows that the clean surrogate is the wrong objective once photon noise is
present.  This probe asks two questions on the *same* explored basins:

  1. Do the explored basins already contain a higher-image-quality
     configuration that the surrogate picker discards? (oracle upper bound)
  2. Does the noise-aware photon-weighted OED score recover it, where the
     clean VCL surrogate does not? (is OED the right selector)

For one cell (default MILP-mild 192^3, k=40, 1e4 photons) it
  * runs the ensemble and keeps ALL members (``return_stack=True``),
  * scores each member by (a) the VCL surrogate and (b) the OED A+D score,
  * reconstructs each member under photon noise (SART) -> oracle PSNR,
  * compares four selections: the VCLS baseline, pick-by-VCL (current
    behaviour), pick-by-OED, and pick-by-oracle-PSNR (the exploration ceiling).

Run from the repo root:
    python -m experiments.studies.basin_selection --phantom mild --seeds 0,1,2

Writes ``experiments/basin_selection/results/basin_selection_<phantom>.csv``.
This is a diagnostic study, not bound to a paper table yet; see
``experiments/studies/README.md``.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import mlx.core as mx
import numpy as np

from experiments.run import (  # type: ignore
    _compute_metrics,
    _load_mlx_stack,
    _load_phantom_pair,
    _resolve_geometry,
)
from differentiable_coverage.eval.geometry import geometry_from_sources
from differentiable_coverage.eval.reco import simulate_sinogram, reconstruct_sart_volume
from differentiable_coverage.eval.vcl import vcls_select
from differentiable_coverage.eval.trajectories import greedy_ensemble_vcl_continuous
from differentiable_coverage.vcl_diff import build_vcl_context, vcl_loss_continuous
from differentiable_coverage.oed import oed_loss_continuous
from differentiable_coverage.absorption_bundle import (
    BundleAbsorptionConfig, bundle_path_integral,
)

OUT_ROOT = Path("experiments/basin_selection")
METRICS = ["psnr", "ssim"]

PHANTOMS = {
    "mild": {"spec": {"type": "milp_npy",
                      "path": "data/mild_asd_pocs_384.npy"}, "geometry": "milp"},
    "moderate": {"spec": {"type": "milp_npy",
                          "path": "data/moderate_asd_pocs_384.npy"}, "geometry": "milp"},
    "lof_flange_v3": {"spec": {"type": "milp_npy",
                               "path": "data/lof_flange_v3.npy"}, "geometry": "milp"},
    "lof_flange_v4a": {"spec": {"type": "milp_npy",
                                "path": "data/lof_flange_v4a.npy"}, "geometry": "milp"},
    "lof_flange_v4b": {"spec": {"type": "milp_npy",
                                "path": "data/lof_flange_v4b.npy"}, "geometry": "milp"},
}


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


def _mean_psnr(vol, src, geom, peak, stack, sart_iter, photon, nseeds):
    """Mean reconstruction PSNR of a source set over the noise seeds."""
    vals = []
    for ns in nseeds:
        recon = _reconstruct(vol, src, geom, sart_iter, photon, ns)
        vals.append(_compute_metrics(vol, recon, peak, ["psnr"], stack)["psnr"])
    return statistics.fmean(vals)


def _ms(xs):
    return statistics.fmean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phantom", default="mild", choices=list(PHANTOMS))
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--kmax", type=int, default=200)
    ap.add_argument("--photons", default="1e4,1e3",
                    help="comma-separated photon levels; explored once, scored per level")
    ap.add_argument("--n-ensemble", type=int, default=8)
    ap.add_argument("--init", default="diverse", choices=["diverse", "jitter"])
    ap.add_argument("--explore-objective", default="vcl", choices=["vcl", "oed"],
                    help="objective driving the ensemble exploration gradient "
                         "(vcl = noise-blind, current; oed = pure photon-weighted)")
    ap.add_argument("--repulsion", type=float, default=0.01)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--sart-iter", type=int, default=15)
    ap.add_argument("--r1", type=float, default=1.0e-3)
    ap.add_argument("--score-r1", type=float, default=None,
                    help="voxel subsampling rate for the VCL/OED scoring context; "
                         "defaults to --r1")
    ap.add_argument("--score-shape", type=int, default=128,
                    help="internal cube size for the VCL/OED scoring context")
    ap.add_argument("--score-bundle", default="center", choices=["center", "full"],
                    help="bundle model used for OED reranking")
    ap.add_argument("--score-bundle-samples", type=int, default=80)
    ap.add_argument("--score-alpha", type=float, default=1.0)
    ap.add_argument("--score-beta", type=float, default=1.0)
    ap.add_argument("--dose-aware-oed", action="store_true",
                    help="if set, reranking uses the physically derived "
                         "precision I0 exp(-tau) for each photon level")
    ap.add_argument("--tag", default="",
                    help="optional suffix for output filenames")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--noise-seeds", default="0,1")
    args = ap.parse_args(argv)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    nseeds = [int(x) for x in args.noise_seeds.split(",") if x.strip()]
    photons = [float(x) for x in args.photons.split(",") if x.strip()]
    (OUT_ROOT / "results").mkdir(parents=True, exist_ok=True)

    stack = _load_mlx_stack()
    info = PHANTOMS[args.phantom]
    spec = {**info["spec"], "resolution": args.resolution}
    geom = _resolve_geometry(info["geometry"], args.resolution)
    vol, _ = _load_phantom_pair(spec, stack)
    mx.eval(vol)
    peak = float(vol.max())
    vs = geom["voxel_pitch"]
    sid, sdd = geom["sid"], geom["sdd"]
    roi0 = mx.zeros(3)
    score_r1 = args.score_r1 if args.score_r1 is not None else args.r1

    # Scoring context: one fixed VCL context (same voxel subsample for every
    # member) so VCL/OED rankings are consistent.
    det_du = geom["det_pitch"] * geom["det_voxels"] / float(args.score_shape)
    if args.score_bundle == "center":
        bcfg = BundleAbsorptionConfig(
            roi_radius=0.0, n_rays_u=1, n_rays_v=1,
            n_samples=args.score_bundle_samples, voxel_spacing=vs,
        )
    else:
        bcfg = BundleAbsorptionConfig(
            roi_radius=5.0, n_rays_u=5, n_rays_v=9,
            n_samples=args.score_bundle_samples, voxel_spacing=vs,
        )

    print("=" * 78, flush=True)
    print(f"Basin-selection probe  phantom={args.phantom} {tuple(vol.shape)} "
          f"k={args.k}  N={args.n_ensemble} init={args.init} "
          f"explore={args.explore_objective}", flush=True)
    print(f"  seeds={seeds}  noise={nseeds}  "
          f"photons={[f'{p:.0e}' for p in photons]}", flush=True)
    print(f"  score_ctx={args.score_shape}^3 r1={score_r1:.1e}  "
          f"bundle={args.score_bundle}  "
          f"alpha={args.score_alpha:g} beta={args.score_beta:g}  "
          f"dose_aware={int(args.dose_aware_oed)}", flush=True)
    print("=" * 78, flush=True)

    tag = f"_{args.tag}" if args.tag else ""
    out = (OUT_ROOT / "results"
           / f"basin_selection_{args.phantom}_{args.explore_objective}{tag}.csv")
    mout = (OUT_ROOT / "results"
            / f"basin_selection_{args.phantom}_{args.explore_objective}{tag}_members.csv")

    def _flush_rows():
        if not rows:
            return
        fields = ["phantom", "k", "n_ensemble", "init", "explore"] + list(rows[0].keys())
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({"phantom": args.phantom, "k": args.k,
                            "n_ensemble": args.n_ensemble, "init": args.init,
                            "explore": args.explore_objective, **r})
        with open(mout, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(member_rows[0].keys()))
            w.writeheader()
            w.writerows(member_rows)

    rows = []          # per (seed, photon) records
    member_rows = []   # per (seed, photon, member) dump for offline selector sweeps
    for seed in seeds:
        # --- VCLS baseline source set (reused as the ensemble's "diverse" member)
        pre = stack["compute_R_gamma"](
            vol, _random_sphere(args.kmax, sid, seed),
            sid=sid, sdd=sdd,
            det_shape=(geom["det_voxels"], geom["det_voxels"]),
            du=geom["det_pitch"], dv=geom["det_pitch"],
            voxel_spacing=vs, r1=args.r1, seed=seed)
        mx.eval(pre.R)
        vidx, _ = vcls_select(pre, args.k, seed=seed)
        vcls_src = pre.candidate_sources[mx.array(vidx)]
        mx.eval(vcls_src)

        # --- explore once. We keep the explored basins fixed across photon
        #     levels and only change the reranking criterion afterwards.
        lam_vcl = 1.0 if args.explore_objective == "vcl" else 0.0
        lam_oed = 1.0 if args.explore_objective == "oed" else 0.0
        _, members, base_vals = greedy_ensemble_vcl_continuous(
            args.k, sid, roi_center=roi0, volume=vol, sdd=sdd,
            n_ensemble=args.n_ensemble, repulsion_weight=args.repulsion,
            init_strategy=args.init, ensemble_seed=seed,
            lambda_vcl=lam_vcl, lambda_cov=0.0, lambda_path=0.0,
            lambda_oed=lam_oed, voxel_spacing=vs,
            n_steps=args.n_steps, lr=args.lr, n_candidates=args.kmax,
            vcl_r1=args.r1, vcls_precompute=pre,
            detector_shape=(geom["det_voxels"], geom["det_voxels"]),
            du=geom["det_pitch"], dv=geom["det_pitch"],
            return_stack=True)
        mx.eval(members)

        scoring_ctx = build_vcl_context(
            vol, sid=sid, sdd=sdd, det_shape=(args.score_shape, args.score_shape),
            du=det_du, dv=det_du, voxel_spacing=vs,
            target_shape=(args.score_shape, args.score_shape, args.score_shape),
            r1=score_r1, seed=seed)
        vcl_info = []
        tau_members = []
        for n in range(args.n_ensemble):
            s = members[n]
            vcl_info.append(1.0 - float(vcl_loss_continuous(s, scoring_ctx)))
            tau_members.append(bundle_path_integral(s, roi0, vol, bcfg))
        i_vcl = int(np.argmax(vcl_info))   # current picker (== argmax base_vals)

        # --- photon-dependent step: reconstruct and rank by oracle PSNR
        for photon in photons:
            oed_score, oed_A, oed_D = [], [], []
            for n in range(args.n_ensemble):
                score_photon = photon if args.dose_aware_oed else 1.0
                sc, a, d = oed_loss_continuous(
                    members[n], scoring_ctx, tau_members[n],
                    photon_count=score_photon,
                    alpha=args.score_alpha, beta=args.score_beta,
                    return_components=True,
                )
                oed_score.append(float(sc))
                oed_A.append(float(a))
                oed_D.append(float(d))
            i_oed = int(np.argmax(oed_score))
            i_a = int(np.argmax(oed_A))
            i_d = int(np.argmax(oed_D))
            psnr_vcls = _mean_psnr(vol, vcls_src, geom, peak, stack,
                                   args.sart_iter, photon, nseeds)
            oracle_psnr = [
                _mean_psnr(vol, members[n], geom, peak, stack,
                           args.sart_iter, photon, nseeds)
                for n in range(args.n_ensemble)]
            i_orc = int(np.argmax(oracle_psnr))
            rec = {
                "seed": seed, "photon": photon,
                "psnr_vcls": psnr_vcls,
                "psnr_sel_vcl": oracle_psnr[i_vcl],
                "psnr_sel_oed": oracle_psnr[i_oed],
                "psnr_sel_A": oracle_psnr[i_a],
                "psnr_sel_D": oracle_psnr[i_d],
                "psnr_sel_oracle": oracle_psnr[i_orc],
                "member_spread": max(oracle_psnr) - min(oracle_psnr),
                "oed_hits_oracle": float(i_oed == i_orc),
                "A_hits_oracle": float(i_a == i_orc),
                "D_hits_oracle": float(i_d == i_orc),
                "vcl_hits_oracle": float(i_vcl == i_orc),
            }
            rows.append(rec)
            for n in range(args.n_ensemble):
                member_rows.append({
                    "phantom": args.phantom, "explore": args.explore_objective,
                    "k": args.k, "seed": seed, "photon": photon, "member": n,
                    "vcl_info": vcl_info[n], "oed_A": oed_A[n], "oed_D": oed_D[n],
                    "oed_score": oed_score[n], "oracle_psnr": oracle_psnr[n],
                    "psnr_vcls": psnr_vcls,
                    "score_bundle": args.score_bundle,
                    "score_shape": args.score_shape,
                    "score_r1": score_r1,
                    "dose_aware_oed": int(args.dose_aware_oed),
                })
            print(f"seed {seed} I0={photon:.0e}:  VCLS={psnr_vcls:.3f}  "
                  f"sel_VCL={rec['psnr_sel_vcl']:.3f}  "
                  f"sel_OED={rec['psnr_sel_oed']:.3f}  "
                  f"sel_D={rec['psnr_sel_D']:.3f}  "
                  f"sel_ORACLE={rec['psnr_sel_oracle']:.3f}  "
                  f"(spread={rec['member_spread']:.2f}, "
                  f"OED=oracle:{int(rec['oed_hits_oracle'])})", flush=True)
        _flush_rows()

    # --- aggregate per photon level
    print("-" * 78, flush=True)
    for photon in photons:
        sub = [r for r in rows if r["photon"] == photon]
        mv, sv = _ms([r["psnr_vcls"] for r in sub])
        mvc, svc = _ms([r["psnr_sel_vcl"] for r in sub])
        mo, so = _ms([r["psnr_sel_oed"] for r in sub])
        md, sd = _ms([r["psnr_sel_D"] for r in sub])
        mor, sor = _ms([r["psnr_sel_oracle"] for r in sub])
        print(f"[I0={photon:.0e}]  (n={len(sub)} seeds)", flush=True)
        print(f"  VCLS baseline        {mv:.3f} ± {sv:.3f}", flush=True)
        print(f"  pick-by-VCL  (now)   {mvc:.3f} ± {svc:.3f}   "
              f"Δ vs VCLS {mvc - mv:+.3f}", flush=True)
        print(f"  pick-by-OED          {mo:.3f} ± {so:.3f}   "
              f"Δ vs VCLS {mo - mv:+.3f}   Δ vs pick-by-VCL {mo - mvc:+.3f}", flush=True)
        print(f"  pick-by-D            {md:.3f} ± {sd:.3f}   "
              f"Δ vs VCLS {md - mv:+.3f}   Δ vs pick-by-VCL {md - mvc:+.3f}", flush=True)
        print(f"  pick-by-ORACLE (ceil){mor:.3f} ± {sor:.3f}   "
              f"Δ vs VCLS {mor - mv:+.3f}  <- exploration headroom", flush=True)
        print(f"  OED hit-rate {statistics.fmean([r['oed_hits_oracle'] for r in sub]):.2f}"
              f"   mean member spread "
              f"{statistics.fmean([r['member_spread'] for r in sub]):.2f} dB", flush=True)

    # --- write CSV ("photon" is already a per-row key)
    fields = ["phantom", "k", "n_ensemble", "init", "explore"] + list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({"phantom": args.phantom, "k": args.k,
                        "n_ensemble": args.n_ensemble, "init": args.init,
                        "explore": args.explore_objective, **r})
    print(f"wrote {out}", flush=True)

    # --- per-member dump: lets any selection rule be evaluated OFFLINE with no
    #     new reconstructions (top-m shortlist, alpha/beta OED sweeps, ...).
    with open(mout, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(member_rows[0].keys()))
        w.writeheader()
        w.writerows(member_rows)
    print(f"wrote {mout}", flush=True)


def _random_sphere(n, sid, seed):
    """n quasi-uniform points on the sphere of radius sid (numpy RNG)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3)).astype(np.float32)
    v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
    return mx.array(v * sid)


if __name__ == "__main__":
    main()
