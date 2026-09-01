"""Bundle quadrature convergence study (REV-P1-01).

Quantifies the accuracy of the bundle path-integral quadrature — values,
gradients, and induced view rankings — as a function of the per-ray sample
count N and the sampling rule:

  full  legacy rule: N midpoint samples on the whole source→target segment
        (the configuration used by the published runs, n_samples = 32);
  clip  N midpoint samples on the ray ∩ volume-bounding-box sub-segment
        only (``BundleAbsorptionConfig.clip_to_volume = True``).

Everything is compared against a fine clipped reference (N = 4096 by
default).  Tolerances are DECLARED BELOW, before any result is computed;
the frozen production rule for the paper reruns is the cheapest (rule, N)
cell that passes all of them.  Run from the repo root:

    python -m experiments.studies.bundle_quadrature_convergence \
        --phantom moderate --resolution 192
"""
from __future__ import annotations

import argparse
import csv
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
)
from differentiable_coverage.score import sample_unit_sphere

OUT_ROOT = Path("experiments/bundle_quadrature")

PHANTOMS = {
    "moderate": {"spec": {"type": "milp_npy",
                          "path": "data/moderate_asd_pocs_384.npy"},
                 "geometry": "milp"},
    "mild": {"spec": {"type": "milp_npy",
                      "path": "data/mild_asd_pocs_384.npy"},
             "geometry": "milp"},
}

# Pre-declared acceptance tolerances (REV-P1-01: fixed before results).
# A (rule, N) cell PASSES when, against the clipped fine reference:
TOLERANCES = {
    "value_rel_median": 0.01,       # median relative τ̄ error ≤ 1 %
    "value_rel_p95": 0.03,          # 95th-percentile relative τ̄ error ≤ 3 %
    "grad_angle_median_deg": 5.0,   # median gradient angle ≤ 5°
    "grad_angle_p95_deg": 15.0,     # p95 gradient angle ≤ 15°
    "grad_mag_median_lo": 0.95,     # median |g|/|g_ref| within ±5 %
    "grad_mag_median_hi": 1.05,
    "kendall_tau_min": 0.99,        # probe-ranking stability
}


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except Exception:
        return "unknown"


def _offcentre_target(vol_np: np.ndarray, voxel_spacing: float) -> np.ndarray:
    """Deterministic off-centre aim point: the strongest-attenuation voxel
    in a 25–60 %-of-half-extent radial shell, in world mm (x, y, z)."""
    Z, Y, X = vol_np.shape
    ci = (np.array([Z, Y, X], dtype=np.float64) - 1.0) / 2.0
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X),
                             indexing="ij")
    r = np.sqrt(((zz - ci[0]) / ci[0]) ** 2 + ((yy - ci[1]) / ci[1]) ** 2
                + ((xx - ci[2]) / ci[2]) ** 2)
    shell = (r >= 0.25) & (r <= 0.60)
    masked = np.where(shell, vol_np, -np.inf)
    iz, iy, ix = np.unravel_index(int(np.argmax(masked)), vol_np.shape)
    world = np.array([(ix - ci[2]), (iy - ci[1]), (iz - ci[0])],
                     dtype=np.float64) * voxel_spacing
    return world.astype(np.float32)


def _tau_and_grad(sources: mx.array, target: mx.array, vol: mx.array,
                  cfg: BundleAbsorptionConfig, chunk: int = 8):
    """τ̄ per probe and its per-probe source gradient, chunked over probes.

    Probes are independent, so the gradient of sum(τ̄) w.r.t. the chunk's
    sources is exactly the per-probe gradient.
    """
    taus, grads = [], []
    for lo in range(0, sources.shape[0], chunk):
        s = sources[lo:lo + chunk]

        def f(src):
            return mx.sum(bundle_path_integral(src, target, vol, cfg))

        tau = bundle_path_integral(s, target, vol, cfg)
        g = mx.grad(f)(s)
        mx.eval(tau, g)
        taus.append(np.asarray(tau, dtype=np.float64))
        grads.append(np.asarray(g, dtype=np.float64))
    return np.concatenate(taus), np.concatenate(grads)


def _cell_stats(tau, grad, tau_ref, grad_ref):
    from scipy.stats import kendalltau, spearmanr

    floor = max(1e-9, 1e-3 * float(np.median(tau_ref)))
    rel = np.abs(tau - tau_ref) / np.maximum(np.abs(tau_ref), floor)

    mag_ref = np.linalg.norm(grad_ref, axis=-1)
    mag = np.linalg.norm(grad, axis=-1)
    keep = mag_ref > 1e-3 * max(float(np.median(mag_ref)), 1e-12)
    cosang = np.sum(grad * grad_ref, axis=-1) / np.maximum(mag * mag_ref, 1e-30)
    ang = np.degrees(np.arccos(np.clip(cosang[keep], -1.0, 1.0)))
    ratio = mag[keep] / np.maximum(mag_ref[keep], 1e-30)

    kt = float(kendalltau(tau, tau_ref).statistic)
    sr = float(spearmanr(tau, tau_ref).statistic)
    n_top = max(1, len(tau) // 10)
    top = set(np.argsort(-tau)[:n_top])
    top_ref = set(np.argsort(-tau_ref)[:n_top])

    stats = {
        "value_rel_median": float(np.median(rel)),
        "value_rel_p95": float(np.percentile(rel, 95)),
        "value_rel_max": float(np.max(rel)),
        "grad_angle_median_deg": float(np.median(ang)),
        "grad_angle_p95_deg": float(np.percentile(ang, 95)),
        "grad_angle_max_deg": float(np.max(ang)),
        "grad_mag_median": float(np.median(ratio)),
        "grad_probes_kept": int(keep.sum()),
        "kendall_tau": kt,
        "spearman_rho": sr,
        "top_decile_overlap": len(top & top_ref) / n_top,
    }
    stats["pass"] = bool(
        stats["value_rel_median"] <= TOLERANCES["value_rel_median"]
        and stats["value_rel_p95"] <= TOLERANCES["value_rel_p95"]
        and stats["grad_angle_median_deg"] <= TOLERANCES["grad_angle_median_deg"]
        and stats["grad_angle_p95_deg"] <= TOLERANCES["grad_angle_p95_deg"]
        and TOLERANCES["grad_mag_median_lo"] <= stats["grad_mag_median"]
        <= TOLERANCES["grad_mag_median_hi"]
        and stats["kendall_tau"] >= TOLERANCES["kendall_tau_min"]
    )
    return stats


def _fd_selfcheck(sources, target, vol, cfg, eps, n_probes=8, n_dirs=4,
                  seed=0):
    """Analytic directional derivative vs central FD of the SAME objective —
    autograd sanity, and the ε anchor reused by the REV-P0-01 study."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(sources.shape[0], size=min(n_probes, sources.shape[0]),
                     replace=False)
    s = sources[mx.array(idx)]

    def f(src):
        return mx.sum(bundle_path_integral(src, target, vol, cfg))

    g = np.asarray(mx.grad(f)(s), dtype=np.float64)
    errs = []
    for _ in range(n_dirs):
        u = rng.normal(size=s.shape).astype(np.float32)
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        um = mx.array(u)
        f_p = float(f(s + eps * um))
        f_m = float(f(s - eps * um))
        fd = (f_p - f_m) / (2.0 * eps)
        an = float(np.sum(g * u))
        errs.append(abs(an - fd) / max(abs(fd), 1e-9))
    return {"eps_mm": eps, "dd_rel_err_median": float(np.median(errs)),
            "dd_rel_err_max": float(np.max(errs))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phantom", default="moderate", choices=list(PHANTOMS))
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--n-probes", type=int, default=128)
    ap.add_argument("--n-list", default="8,16,32,64,128,256,512")
    ap.add_argument("--ref-n", type=int, default=4096)
    ap.add_argument("--eps-fd", type=float, default=0.5,
                    help="central-FD step (mm) for the autograd self-check; "
                         "matches the paper geometry_fd_step")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args(argv)

    stack = _load_mlx_stack()
    ph = PHANTOMS[args.phantom]
    spec = dict(ph["spec"], resolution=args.resolution)
    vol, _ = _load_phantom_pair(spec, stack)
    geom = _resolve_geometry(ph["geometry"], args.resolution)
    vs = geom["voxel_pitch"]
    sid = geom["sid"]
    vol_np = np.asarray(vol, dtype=np.float64)

    targets = {
        "centre": np.zeros(3, dtype=np.float32),
        "offcentre": _offcentre_target(vol_np, vs),
    }
    n_list = [int(x) for x in args.n_list.split(",")]
    probes = sample_unit_sphere(args.n_probes) * sid    # deterministic

    def cfg_for(n, clip):
        return BundleAbsorptionConfig(
            roi_radius=5.0, n_rays_u=5, n_rays_v=9, n_samples=n,
            voxel_spacing=vs, clip_to_volume=clip,
        )

    results, csv_rows = {}, []
    t_start = time.time()
    for tname, tgt in targets.items():
        target = mx.array(tgt)
        print(f"[{tname}] target = {np.round(tgt, 2).tolist()} mm", flush=True)
        tau_ref, grad_ref = _tau_and_grad(probes, target, vol,
                                          cfg_for(args.ref_n, True))
        # Border-clamp artifact probe: the full-segment rule at ref-N should
        # match the clipped rule at ref-N for air-margined phantoms.
        tau_full_ref, _ = _tau_and_grad(probes, target, vol,
                                        cfg_for(args.ref_n, False))
        floor = max(1e-9, 1e-3 * float(np.median(tau_ref)))
        clamp_bias = float(np.median(
            np.abs(tau_full_ref - tau_ref) / np.maximum(tau_ref, floor)))
        results[tname] = {
            "target_mm": tgt.tolist(),
            "tau_ref_median": float(np.median(tau_ref)),
            "full_vs_clip_ref_rel_median": clamp_bias,
            "cells": {},
        }
        for clip in (False, True):
            rule = "clip" if clip else "full"
            for n in n_list:
                tau, grad = _tau_and_grad(probes, target, vol,
                                          cfg_for(n, clip))
                st = _cell_stats(tau, grad, tau_ref, grad_ref)
                results[tname]["cells"][f"{rule}@{n}"] = st
                csv_rows.append({"target": tname, "rule": rule, "n": n, **{
                    k: st[k] for k in (
                        "value_rel_median", "value_rel_p95",
                        "grad_angle_median_deg", "grad_angle_p95_deg",
                        "grad_mag_median", "kendall_tau",
                        "top_decile_overlap", "pass")}})
                print(f"  {rule}@{n:<5d} value(med/p95) "
                      f"{st['value_rel_median']:.4f}/{st['value_rel_p95']:.4f}"
                      f"  angle(med/p95) {st['grad_angle_median_deg']:.2f}/"
                      f"{st['grad_angle_p95_deg']:.2f} deg"
                      f"  kendall {st['kendall_tau']:.4f}"
                      f"  {'PASS' if st['pass'] else 'fail'}", flush=True)
        results[tname]["fd_selfcheck_full@32"] = _fd_selfcheck(
            probes, target, vol, cfg_for(32, False), args.eps_fd)
        results[tname]["fd_selfcheck_clip@32"] = _fd_selfcheck(
            probes, target, vol, cfg_for(32, True), args.eps_fd)

    # Frozen-rule recommendation: cheapest cell passing at BOTH targets.
    frozen = None
    for n in n_list:
        for rule in ("clip", "full"):
            if all(results[t]["cells"][f"{rule}@{n}"]["pass"]
                   for t in targets):
                frozen = {"rule": rule, "n_samples": n}
                break
        if frozen:
            break

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    artifact = {
        "study": "bundle_quadrature_convergence",
        "phantom": args.phantom,
        "resolution": args.resolution,
        "geometry": geom,
        "n_probes": args.n_probes,
        "probe_convention": "isocentre-centred SID sphere, aimed at target",
        "volume_center_mm": [0.0, 0.0, 0.0],
        "reference": {"rule": "clip", "n_samples": args.ref_n},
        "tolerances": TOLERANCES,
        "git_head": _git_head(),
        "runtime_s": round(time.time() - t_start, 1),
        "frozen_rule_recommendation": frozen,
        "results": results,
    }
    jpath = out_root / f"convergence_{args.phantom}_{args.resolution}.json"
    jpath.write_text(json.dumps(artifact, indent=2))
    cpath = out_root / f"convergence_{args.phantom}_{args.resolution}.csv"
    with cpath.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nFrozen-rule recommendation: {frozen}", flush=True)
    print(f"Wrote {jpath}\nWrote {cpath}", flush=True)


if __name__ == "__main__":
    main()
