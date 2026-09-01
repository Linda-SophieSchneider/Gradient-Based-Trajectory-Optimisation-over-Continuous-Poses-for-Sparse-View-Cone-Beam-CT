"""Central-difference step-size sweep for the matched FD arm (REV-P0-01).

Sweeps the finite-difference step ε for the coverage + bundle objective that
``vcls_adam_bundle`` optimises (frozen quadrature rule clip@256), comparing

  1. directional derivatives by central FD at each ε against the analytic
     autograd gradient, and
  2. the Richardson extrapolation FD(ε), FD(ε/2) → (4·FD(ε/2) − FD(ε))/3 as
     an analytic-independent high-accuracy reference.

The classic U-curve (truncation error at large ε, float32 round-off at small
ε) selects ε* for ``derivative_fd_step`` in the matched Table-I rerun; the
agreement of the extrapolated reference with the analytic value at ε* is the
directional-derivative reference check the master plan asks for.

Run from the repo root:
    python -m experiments.studies.fd_epsilon_sweep --phantom moderate
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
)
from differentiable_coverage.absorption_bundle import (
    BundleAbsorptionConfig,
    bundle_path_integral,
    calibrate_bundle_weight,
)
from differentiable_coverage.score import (
    ScoreConfig, sample_unit_sphere, saturated_coverage,
)

OUT_ROOT = Path("experiments/bundle_quadrature")

PHANTOMS = {
    "moderate": {"spec": {"type": "milp_npy",
                          "path": "data/moderate_asd_pocs_384.npy"},
                 "geometry": "milp"},
    "mild": {"spec": {"type": "milp_npy",
                      "path": "data/mild_asd_pocs_384.npy"},
             "geometry": "milp"},
}

# Frozen production quadrature (bundle_quadrature_convergence study).
FROZEN_N_SAMPLES = 256
FROZEN_CLIP = True


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phantom", default="moderate", choices=list(PHANTOMS))
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--eps-list",
                    default="0.001,0.005,0.01,0.05,0.1,0.25,0.5,1.0,2.0")
    ap.add_argument("--n-dirs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args(argv)

    _load_mlx_stack()
    ph = PHANTOMS[args.phantom]
    spec = dict(ph["spec"], resolution=args.resolution)
    vol, _ = _load_phantom_pair(spec, _load_mlx_stack())
    geom = _resolve_geometry(ph["geometry"], args.resolution)
    sid, vs = geom["sid"], geom["voxel_pitch"]

    roi = mx.zeros(3)
    # Paper simulated-study smoothing: Δγ = 15° (σ ≈ 0.085).
    cfg = ScoreConfig(tau=math.sin(15.0 * math.pi / 180.0))
    normals = sample_unit_sphere(300)
    bcfg = BundleAbsorptionConfig(
        roi_radius=5.0, n_rays_u=5, n_rays_v=9,
        n_samples=FROZEN_N_SAMPLES, voxel_spacing=vs,
        clip_to_volume=FROZEN_CLIP,
    )
    probes = sample_unit_sphere(256) * sid
    med_tau = float(mx.median(bundle_path_integral(probes, roi, vol, bcfg)))
    lam = calibrate_bundle_weight(med_tau)

    # Representative selector state: Fibonacci lattice on the SID sphere
    # (the manifold every selector's iterates live on), seed-rotated.
    sources = sample_unit_sphere(args.k, seed=args.seed) * sid
    w = mx.ones(args.k, dtype=mx.float32)

    def objective(src):
        cov = saturated_coverage(src, roi, normals, w, cfg)
        tau = bundle_path_integral(src, roi, vol, bcfg)
        return cov - lam * mx.mean(tau)

    g = np.asarray(mx.grad(objective)(sources), dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    dirs = rng.normal(size=(args.n_dirs,) + tuple(sources.shape))
    dirs /= np.linalg.norm(dirs.reshape(args.n_dirs, -1),
                           axis=1)[:, None, None]
    dd_analytic = np.array([float(np.sum(g * u)) for u in dirs])

    def dd_fd(u, eps):
        um = mx.array(u.astype(np.float32))
        f_p = float(objective(sources + eps * um))
        f_m = float(objective(sources - eps * um))
        return (f_p - f_m) / (2.0 * eps)

    eps_list = [float(x) for x in args.eps_list.split(",")]
    t0 = time.time()
    curve = []
    fd_cache = {}
    for eps in eps_list:
        fd_cache[eps] = np.array([dd_fd(u, eps) for u in dirs])
        rel = np.abs(fd_cache[eps] - dd_analytic) / np.maximum(
            np.abs(dd_analytic), 1e-12)
        curve.append({"eps_mm": eps,
                      "rel_err_median": float(np.median(rel)),
                      "rel_err_p95": float(np.percentile(rel, 95)),
                      "rel_err_max": float(np.max(rel))})
        print(f"  eps={eps:<7g} rel err med/p95/max = "
              f"{curve[-1]['rel_err_median']:.2e}/"
              f"{curve[-1]['rel_err_p95']:.2e}/"
              f"{curve[-1]['rel_err_max']:.2e}", flush=True)

    eps_star = min(curve, key=lambda c: c["rel_err_median"])["eps_mm"]

    # Analytic-independent reference: Richardson extrapolation at ε*.
    fd_e = fd_cache[eps_star]
    fd_h = np.array([dd_fd(u, eps_star / 2.0) for u in dirs])
    richardson = (4.0 * fd_h - fd_e) / 3.0
    rel_ref = np.abs(richardson - dd_analytic) / np.maximum(
        np.abs(dd_analytic), 1e-12)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    artifact = {
        "study": "fd_epsilon_sweep",
        "phantom": args.phantom,
        "resolution": args.resolution,
        "k": args.k,
        "objective": "saturated_coverage - lambda*mean(bundle_tau)",
        "quadrature": {"rule": "clip" if FROZEN_CLIP else "full",
                       "n_samples": FROZEN_N_SAMPLES},
        "lambda_bundle": lam,
        "median_tau_probe": med_tau,
        "n_dirs": args.n_dirs,
        "seed": args.seed,
        "git_head": _git_head(),
        "runtime_s": round(time.time() - t0, 1),
        "curve": curve,
        "eps_star_mm": eps_star,
        "richardson_reference_at_eps_star": {
            "rel_err_median": float(np.median(rel_ref)),
            "rel_err_p95": float(np.percentile(rel_ref, 95)),
            "rel_err_max": float(np.max(rel_ref)),
        },
    }
    jpath = out_root / f"fd_epsilon_{args.phantom}_{args.resolution}.json"
    jpath.write_text(json.dumps(artifact, indent=2))
    print(f"\neps* = {eps_star} mm; Richardson-vs-analytic rel err median "
          f"{artifact['richardson_reference_at_eps_star']['rel_err_median']:.2e}",
          flush=True)
    print(f"Wrote {jpath}", flush=True)


if __name__ == "__main__":
    main()
