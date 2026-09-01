"""Zero-count / pre-clamp fractions for the photon-noise studies (REV-P1-06).

The photon-noise model draws Poisson counts from I = I0 * exp(-p) and clamps
the drawn count to >= 1 before the log (Eq. photon_noise in the paper).  The
reviewers asked how often that clamp region is actually reached.  This probe
reproduces the noise studies' selected view sets (deterministic per selection
seed; the printed studies do not archive the pose sets themselves), computes
one CLEAN forward projection per selection -- no noise draw, no reconstruction
-- and exports, per phantom x method x k x I0:

  frac_expected_lt1   fraction of detector pixels with expected intensity
                      I0 * exp(-p) < 1 photon (the pre-clamp criterion of the
                      revision plan),
  expected_zero_frac  mean Poisson zero probability exp(-I), i.e. the expected
                      fraction of pixels whose drawn count is clamped,
  p_max / p_p99       line-integral tail statistics for context.

Selection reproduction matches experiments/run_noise_eval.py exactly (same
config, same dispatch call, same prior volume, same candidate pool); by
default only the first pre-declared selection seed is probed, which is
declared alongside the reported numbers.

Usage::

    python experiments/studies/noise_clamp_stats.py \
        --config experiments/configs/paper1/ablation/noise_robustness_milp_moderate.yaml
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import yaml

from experiments.run import (  # type: ignore
    _build_vcl_cache_if_needed,
    _load_mlx_stack,
    _load_phantom_pair,
    _resolve_geometry,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="a noise_robustness YAML (the printed-study config)")
    ap.add_argument("--selection-seed", type=int, default=None,
                    help="selection seed to probe (default: first pre-declared)")
    ap.add_argument("--out", default=None,
                    help="output CSV (default experiments/results/"
                         "noise_clamp_stats_<name>.csv)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("experiment_type") != "noise_robustness":
        raise SystemExit("config experiment_type must be noise_robustness")

    import mlx.core as mx
    from differentiable_coverage.eval.geometry import geometry_from_sources
    from differentiable_coverage.eval.reco import simulate_sinogram

    stack = _load_mlx_stack()
    geom = _resolve_geometry(cfg["geometry"], int(cfg["phantom"]["resolution"]))
    vol_gt, vol_prior = _load_phantom_pair(cfg["phantom"], stack)
    mx.eval(vol_gt); mx.eval(vol_prior)

    sel_seeds = [int(s) for s in cfg.get("selection_seeds", [cfg.get("seed", 0)])]
    sel_seed = int(args.selection_seed) if args.selection_seed is not None else sel_seeds[0]
    photon_levels = [float(x) for x in cfg["photon_levels"]]
    k_values = [int(k) for k in cfg["k_values"]]
    roi_center = mx.array([0.0, 0.0, 0.0])

    out_path = Path(args.out) if args.out else (
        Path("experiments/results") / f"noise_clamp_stats_{cfg['name']}.csv")

    print(f"[{cfg['name']}] {cfg['phantom']['path']} @ {tuple(vol_gt.shape)}  "
          f"selection seed {sel_seed}", flush=True)

    t0 = time.time()
    vcl_pre = _build_vcl_cache_if_needed(cfg, vol_prior, geom, stack, seed=sel_seed)
    print(f"  (R,gamma) cache: {time.time() - t0:.1f}s", flush=True)

    rows: list[dict] = []
    for method in cfg["methods"]:
        name = method["name"]
        label = method.get("label", name)
        mkw = dict(method.get("kwargs") or {})
        n_cand = int(mkw.pop("n_candidates", cfg["k_max"]))
        for k in k_values:
            t0 = time.time()
            src = stack["build_baseline_sources"](
                name, k, geom["sid"], roi_center=roi_center,
                vcl_precompute=vcl_pre, volume=vol_prior, sdd=geom["sdd"],
                detector_shape=(geom["det_voxels"], geom["det_voxels"]),
                du=geom["det_pitch"], dv=geom["det_pitch"],
                voxel_spacing=geom["voxel_pitch"], n_candidates=n_cand,
                seed=sel_seed, method_kwargs=dict(mkw),
            )
            mx.eval(src)
            sel_s = time.time() - t0

            # One clean (noise-free) forward projection of the true mu-map.
            sp, dc, du, dv = geometry_from_sources(src, sid=geom["sid"],
                                                   sdd=geom["sdd"])
            sino = simulate_sinogram(
                vol_gt, sp, dc, du, dv,
                det_u=geom["det_voxels"], det_v=geom["det_voxels"],
                du=geom["det_pitch"], dv=geom["det_pitch"],
                voxel_spacing=geom["voxel_pitch"],
                photon_count=None,
            )
            mx.eval(sino)
            p = np.asarray(sino, dtype=np.float64).reshape(-1)
            p = np.maximum(p, 0.0)

            for I0 in photon_levels:
                intensity = I0 * np.exp(-p)
                row = {
                    "phantom": cfg["name"], "method": label, "impl": name,
                    "k": k, "photon_count": I0, "selection_seed": sel_seed,
                    "frac_expected_lt1": round(float(np.mean(intensity < 1.0)), 8),
                    "expected_zero_frac": round(float(np.mean(np.exp(-intensity))), 8),
                    "p_max": round(float(p.max()), 4),
                    "p_p99": round(float(np.percentile(p, 99)), 4),
                    "sel_s": round(sel_s, 1),
                }
                rows.append(row)
                print(f"  {label:>20} k={k:>3} I0={I0:.0e}: "
                      f"lt1={row['frac_expected_lt1']:.6f}  "
                      f"E[zero]={row['expected_zero_frac']:.6f}  "
                      f"p_max={row['p_max']:.2f}  ({sel_s:.0f}s)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}  ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
