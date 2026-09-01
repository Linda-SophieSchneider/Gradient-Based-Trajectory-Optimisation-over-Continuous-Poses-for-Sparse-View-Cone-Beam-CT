"""Quadrature-bias quantification on the REAL camera planning prior (REV-P1-01).

The acquired robot arms were planned with the legacy full-segment 32-sample
bundle rule on the camera FDK prior.  Those acquisitions are historical and
cannot be redone; this study quantifies, on that exact volume and geometry,
how far the legacy estimator's values/gradients/rankings deviate from the
converged clipped reference — the supplement number backing the paper's
one-sentence limitation.

Run from the Differentiable-Coverage repo root (CPU-only):
    python -m experiments.studies.bundle_quadrature_camera
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
mx.set_default_device(mx.cpu)   # mlx-cuda 0.30 correctness bugs; arrays tiny

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "real_experiment"))
from scanner_io.rek2py import rek2py                          # noqa: E402

from differentiable_coverage.absorption_bundle import (      # noqa: E402
    BundleAbsorptionConfig,
)
from differentiable_coverage.score import sample_unit_sphere  # noqa: E402
from experiments.studies.bundle_quadrature_convergence import (  # noqa: E402
    TOLERANCES, _cell_stats, _tau_and_grad,
)

# Not redistributed (measured robot-CT reconstruction, see README "Data");
# point this at your own copy via CAMERA_FDK_PATH.
FDK_PATH = os.environ.get(
    "CAMERA_FDK_PATH",
    "/home/schneider/DiffCT_CUDA_Development/TestReconstructions/output/reconstruction_FDK.rek",
)
FOD_MM = 997.1          # sec:real geometry (re-derived from circular_1200)
OUT = Path("experiments/bundle_quadrature/convergence_camera_prior.json")


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=HERE).stdout.strip()
    except Exception:
        return "unknown"


def main():
    header, vol_np = rek2py(FDK_PATH, switch_order=True)
    voxel_mm = float(header.voxel_size_x_in_um) / 1000.0
    vol = mx.array(np.asarray(vol_np, np.float32))
    print(f"camera prior: shape={vol.shape}, voxel={voxel_mm:.4f} mm", flush=True)

    # Planning convention: probes on the FOD sphere, bundle target = the
    # isocentre/world origin (plan_trajectory.py passes mx.zeros(3)).
    n_probes = 96
    probes = sample_unit_sphere(n_probes) * FOD_MM
    target = mx.zeros(3)

    def cfg_for(n, clip):
        return BundleAbsorptionConfig(
            roi_radius=5.0, n_rays_u=5, n_rays_v=9, n_samples=n,
            voxel_spacing=voxel_mm, clip_to_volume=clip)

    t0 = time.time()
    tau_ref, grad_ref = _tau_and_grad(probes, target, vol, cfg_for(4096, True),
                                      chunk=4)
    cells = {}
    for rule, clip in (("full", False), ("clip", True)):
        for n in (32, 128, 256):
            tau, grad = _tau_and_grad(probes, target, vol, cfg_for(n, clip),
                                      chunk=4)
            st = _cell_stats(tau, grad, tau_ref, grad_ref)
            cells[f"{rule}@{n}"] = st
            print(f"  {rule}@{n:<5d} value(med/p95) {st['value_rel_median']:.4f}/"
                  f"{st['value_rel_p95']:.4f}  angle(med) "
                  f"{st['grad_angle_median_deg']:.2f} deg  kendall "
                  f"{st['kendall_tau']:.4f}  {'PASS' if st['pass'] else 'fail'}",
                  flush=True)

    artifact = {
        "study": "bundle_quadrature_camera_prior",
        "volume": FDK_PATH,
        "volume_shape": [int(s) for s in vol.shape],
        "voxel_mm": voxel_mm,
        "fod_mm": FOD_MM,
        "n_probes": n_probes,
        "probe_convention": "isocentre-centred FOD sphere, target = isocentre "
                            "(the acquired-arm planning convention)",
        "reference": {"rule": "clip", "n_samples": 4096},
        "tolerances": TOLERANCES,
        "tau_ref_median": float(np.median(tau_ref)),
        "git_head": _git_head(),
        "runtime_s": round(time.time() - t0, 1),
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
