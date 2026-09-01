"""Multi-point ROI coverage evaluation (REV-P1-04).

Answers the two reviewer questions about the centre-point soft-Tuy
surrogate on the simulated phantoms:

  A. Representativeness — for a trajectory selected with the single-centre
     objective, how does the plane-coverage surrogate vary ACROSS the ROI?
     We evaluate C̃(x_p) on a point grid inside the ROI sphere and report
     min/mean/max/SD against the centre value.  Additionally, a trajectory
     selected with the MULTI-POINT objective (the weighted point-cloud form
     already used by the real experiment) is evaluated the same way.

  B. Selection sensitivity — how much does the selected pose set move when
     x_roi is perturbed by a few millimetres?  We re-select with perturbed
     centres and report (i) the optimal-assignment mean/max angular distance
     between the pose sets and (ii) the coverage at the TRUE centre achieved
     by the perturbed-centre selections.

Run from the repo root:
    python -m experiments.studies.coverage_multipoint_eval --phantom moderate
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from experiments.run import (  # type: ignore
    _load_mlx_stack,
    _load_phantom_pair,
    _resolve_geometry,
    _resolve_roi_context,
)
from experiments.studies.bundle_quadrature_convergence import (  # type: ignore
    _offcentre_target,
)
from differentiable_coverage.eval.trajectories import greedy_adam
from differentiable_coverage.score import (
    ScoreConfig, sample_unit_sphere, saturated_coverage,
)

OUT_ROOT = Path("experiments/coverage_multipoint")

PHANTOMS = {
    "moderate": {"spec": {"type": "milp_npy",
                          "path": "data/moderate_asd_pocs_384.npy"},
                 "geometry": "milp"},
    "mild": {"spec": {"type": "milp_npy",
                      "path": "data/mild_asd_pocs_384.npy"},
             "geometry": "milp"},
    "lof_flange_v3": {"spec": {"type": "milp_npy",
                               "path": "data/lof_flange_v3.npy"},
                      "geometry": "milp"},
}


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except Exception:
        return "unknown"


def _per_point_coverage(src, points, normals, cfg):
    """C̃ evaluated at each ROI point individually (plane coverage at x_p)."""
    w = mx.ones(src.shape[0], dtype=mx.float32)
    vals = []
    for i in range(points.shape[0]):
        p = points[i]
        vals.append(float(saturated_coverage(src, p, normals, w, cfg)))
    return np.array(vals)


def _assignment_angles_deg(a, b):
    """Optimal one-to-one angular matching between two pose sets on the
    same sphere (mean/max great-circle distance in degrees)."""
    from scipy.optimize import linear_sum_assignment

    an = np.asarray(a, np.float64)
    bn = np.asarray(b, np.float64)
    an /= np.linalg.norm(an, axis=-1, keepdims=True)
    bn /= np.linalg.norm(bn, axis=-1, keepdims=True)
    cost = np.degrees(np.arccos(np.clip(an @ bn.T, -1.0, 1.0)))
    r, c = linear_sum_assignment(cost)
    d = cost[r, c]
    return float(d.mean()), float(d.max())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phantom", default="moderate", choices=list(PHANTOMS))
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--k-values", default="40,80")
    ap.add_argument("--roi-radius-mm", type=float, default=5.0)
    ap.add_argument("--point-grid", type=int, default=5)
    ap.add_argument("--perturb-mm", default="2,5,10")
    ap.add_argument("--n-normals", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args(argv)

    stack = _load_mlx_stack()
    ph = PHANTOMS[args.phantom]
    spec = dict(ph["spec"], resolution=args.resolution)
    vol, _ = _load_phantom_pair(spec, stack)
    geom = _resolve_geometry(ph["geometry"], args.resolution)
    sid, vs = geom["sid"], geom["voxel_pitch"]

    centre = _offcentre_target(np.asarray(vol, np.float64), vs)
    roi = _resolve_roi_context(
        {"roi": {"type": "sphere",
                 "center_mm": [float(c) for c in centre],
                 "radius_mm": args.roi_radius_mm,
                 "point_grid": args.point_grid}},
        vol, geom, stack, want_mask=False)
    points = roi["points"]
    w = np.asarray(roi["weights"], np.float64)
    w = w / max(w.sum(), 1e-12)
    weights = mx.array(w.astype(np.float32))
    n_points = int(points.shape[0])

    # Paper simulated-study smoothing (Δγ = 15°).
    cfg = ScoreConfig(tau=math.sin(15.0 * math.pi / 180.0))
    normals = sample_unit_sphere(args.n_normals)
    centre_mx = mx.array(centre)

    k_values = [int(x) for x in args.k_values.split(",")]
    perturbs = [float(x) for x in args.perturb_mm.split(",")]
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    results = {}
    for k in k_values:
        # A. single-centre selection (the paper protocol) ...
        src_c = greedy_adam(k, sid, roi_center=centre_mx, score_cfg=cfg,
                            n_normals=args.n_normals, sphere_seed=args.seed)
        mx.eval(src_c)
        pp_c = _per_point_coverage(src_c, points, normals, cfg)
        c_at_centre = float(saturated_coverage(
            src_c, centre_mx, normals,
            mx.ones(k, dtype=mx.float32), cfg))
        # ... and multi-point (weighted point-cloud) selection, the
        # real-experiment objective form.
        src_m = greedy_adam(k, sid, roi_center=centre_mx, score_cfg=cfg,
                            n_normals=args.n_normals, sphere_seed=args.seed,
                            roi_points=points, roi_weights=weights)
        mx.eval(src_m)
        pp_m = _per_point_coverage(src_m, points, normals, cfg)
        mean_ang, max_ang = _assignment_angles_deg(src_c, src_m)

        def _pp_stats(pp):
            return {"min": round(float(pp.min()), 4),
                    "mean": round(float(pp.mean()), 4),
                    "max": round(float(pp.max()), 4),
                    "sd": round(float(pp.std()), 4)}

        # B. selection sensitivity to the centre choice.
        sens = []
        for d in perturbs:
            for _ in range(2):
                u = rng.normal(size=3)
                u /= np.linalg.norm(u)
                c_p = centre + (d * u).astype(np.float32)
                src_p = greedy_adam(k, sid, roi_center=mx.array(c_p),
                                    score_cfg=cfg, n_normals=args.n_normals,
                                    sphere_seed=args.seed)
                mx.eval(src_p)
                a_mean, a_max = _assignment_angles_deg(src_c, src_p)
                cov_true = float(saturated_coverage(
                    src_p, centre_mx, normals,
                    mx.ones(k, dtype=mx.float32), cfg))
                sens.append({"perturb_mm": d,
                             "assign_mean_deg": round(a_mean, 3),
                             "assign_max_deg": round(a_max, 3),
                             "coverage_at_true_centre": round(cov_true, 4),
                             "coverage_drop_vs_centre_sel":
                                 round(c_at_centre - cov_true, 4)})

        results[f"k{k}"] = {
            "centre_selection": {
                "coverage_at_centre": round(c_at_centre, 4),
                "per_point": _pp_stats(pp_c),
            },
            "multipoint_selection": {
                "per_point": _pp_stats(pp_m),
                "assign_vs_centre_sel_deg": {
                    "mean": round(mean_ang, 3), "max": round(max_ang, 3)},
            },
            "centre_sensitivity": sens,
        }
        print(f"[k={k}] centre C̃={c_at_centre:.4f}  per-point "
              f"min/mean/max = {results[f'k{k}']['centre_selection']['per_point']['min']}/"
              f"{results[f'k{k}']['centre_selection']['per_point']['mean']}/"
              f"{results[f'k{k}']['centre_selection']['per_point']['max']}  "
              f"multi-vs-centre assignment {mean_ang:.2f}°/{max_ang:.2f}°",
              flush=True)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    artifact = {
        "study": "coverage_multipoint_eval",
        "phantom": args.phantom,
        "resolution": args.resolution,
        "roi_centre_mm": [float(c) for c in centre],
        "roi_radius_mm": args.roi_radius_mm,
        "n_roi_points": n_points,
        "smoothing": {"tau": cfg.tau, "sigma": cfg.gaussian_sigma()},
        "n_normals": args.n_normals,
        "seed": args.seed,
        "git_head": _git_head(),
        "runtime_s": round(time.time() - t0, 1),
        "results": results,
    }
    jpath = out_root / f"multipoint_{args.phantom}_{args.resolution}.json"
    jpath.write_text(json.dumps(artifact, indent=2))
    print(f"\nWrote {jpath}", flush=True)


if __name__ == "__main__":
    main()
