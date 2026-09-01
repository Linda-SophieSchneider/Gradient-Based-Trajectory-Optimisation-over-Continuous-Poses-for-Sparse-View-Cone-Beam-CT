"""Controlled photon-noise robustness benchmark.

Question for the paper: once realistic photon noise enters the measurement,
does our precompute-free continuous selector beat the noise-blind discrete
VCLS -- on image quality, on noise, *and* on wall-clock?

Design (controlled):
  * ``selection_seeds`` (default: the single legacy ``seed``) replicates the
    ENTIRE selection per method — candidate-lattice rotation, init, and
    optimisation — so method effects are separated from selection luck
    (REV-P1-02).  Three pre-declared seeds are the paper protocol.
  * For each photon level I0 we draw ``noise_seeds`` independent Poisson
    transmission realisations, reconstruct each with SART, and report the
    mean +/- std of every metric over the noise draws.  Noise seeds are
    paired ACROSS methods at the RNG-stream level (same seed integer =
    same stream); count-level pairing is impossible across different
    trajectories because the Poisson rates differ per geometry.
  * Selection cost is recorded honestly: ``cache_s`` is the object-specific
    O(K_max^2) (R, gamma) build (only VCLS-init methods pay it; cold continuous
    methods get 0), ``sel_s`` is the per-method selection time.

Outputs (REV-P1-02/03 provenance):
  * ``output``            aggregate CSV, one row per (method, k, I0,
                          selection_seed), means/stds over noise draws;
  * ``<output>.raw.csv``  one row per (method, k, I0, selection_seed,
                          noise_seed) with the raw per-draw metrics;
  * ``<output>.paired.csv`` paired method-vs-baseline differences per
                          (k, I0): per-selection-seed paired means (over
                          shared noise seeds) plus their cross-seed mean,
                          SD, min, and max.  Baseline = ``paired_baseline``
                          (default: the first method).

This isolates three contrasts:
  vcls              -> discrete, noise-blind, pays the precompute
  greedy_adam       -> cold continuous coverage, no precompute, no OED
  greedy_adam_oed   -> cold continuous + photon-noise-weighted A+D OED term

Usage::

    python experiments/run_noise_eval.py \
        --config experiments/configs/paper1/ablation/noise_robustness_milp_moderate.yaml
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import yaml

from differentiable_coverage.eval.geometry import geometry_from_sources
from differentiable_coverage.eval.reco import simulate_sinogram, reconstruct_sart_volume

from experiments.run import (  # type: ignore
    _build_vcl_cache_if_needed,
    _compute_metrics,
    _load_mlx_stack,
    _load_phantom_pair,
    _method_needs_vcl_cache,
    _resolve_geometry,
)

METRICS = ["psnr", "ssim", "hfen"]


def _reconstruct(stack, vol_gt, src, geom, sart_iter, photon_count, noise_seed,
                 spectrum=None):
    sp, dc, du, dv = geometry_from_sources(src, sid=geom["sid"], sdd=geom["sdd"])
    sino = simulate_sinogram(
        vol_gt, sp, dc, du, dv,
        det_u=geom["det_voxels"], det_v=geom["det_voxels"],
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"],
        photon_count=photon_count, noise_seed=noise_seed,
        spectrum=spectrum,
    )
    mx.eval(sino)
    res = reconstruct_sart_volume(
        vol_gt.shape, sino, sp, dc, du, dv,
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"],
        iteration_count=sart_iter, show_progress=False,
    )
    mx.eval(res.reconstruction)
    return _compute_metrics(vol_gt, res.reconstruction, float(vol_gt.max()),
                            METRICS, stack)


def _mean_std(values):
    m = statistics.fmean(values)
    s = statistics.pstdev(values) if len(values) > 1 else 0.0
    return m, s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("experiment_type") != "noise_robustness":
        raise SystemExit("config experiment_type must be noise_robustness")

    stack = _load_mlx_stack()
    geom = _resolve_geometry(cfg["geometry"], int(cfg["phantom"]["resolution"]))
    vol_gt, vol_prior = _load_phantom_pair(cfg["phantom"], stack)
    mx.eval(vol_gt); mx.eval(vol_prior)

    sel_seeds = [int(s) for s in
                 cfg.get("selection_seeds", [cfg.get("seed", 0)])]
    photon_levels = [float(x) for x in cfg["photon_levels"]]
    noise_seeds = [int(s) for s in cfg.get("noise_seeds", [0, 1, 2, 3, 4])]
    k_values = [int(k) for k in cfg["k_values"]]
    spectrum = cfg.get("spectrum")
    roi_center = mx.array([0.0, 0.0, 0.0])

    print(f"[{cfg['name']}] {cfg['phantom']['path']} @ {vol_gt.shape}", flush=True)
    print(f"  photon levels  : {photon_levels}", flush=True)
    print(f"  noise seeds    : {noise_seeds}", flush=True)
    print(f"  selection seeds: {sel_seeds}", flush=True)
    if spectrum is not None:
        print(f"  spectrum     : weights={spectrum['weights']} "
              f"mu_scales={spectrum['mu_scales']}  (polychromatic surrogate)",
              flush=True)

    rows: list[dict] = []
    raw_rows: list[dict] = []
    for sel_seed in sel_seeds:
        # The (R, gamma) cache depends on the seeded candidate rotation, so
        # it is rebuilt (or disk-loaded) per selection seed.
        t0 = time.time()
        vcl_pre = _build_vcl_cache_if_needed(cfg, vol_prior, geom, stack,
                                             seed=sel_seed)
        cache_s_shared = time.time() - t0
        print(f"\n=== selection seed {sel_seed}  "
              f"(cache build {cache_s_shared:.1f}s) ===", flush=True)

        for method in cfg["methods"]:
            name = method["name"]
            label = method.get("label", name)
            reselect_per_photon_level = bool(
                method.get("reselect_per_photon_level", False))
            cache_s = cache_s_shared if _method_needs_vcl_cache(method) else 0.0
            # Per-method candidate-pool override: the dense VCLS baseline
            # keeps the full K_max pool, while our continuous methods may
            # select from a coarser pool since they refine off-grid.
            mkw = dict(method.get("kwargs") or {})
            n_cand = int(mkw.pop("n_candidates", cfg["k_max"]))
            for k in k_values:
                src = None
                sel_s = None

                for I0 in photon_levels:
                    if src is None or reselect_per_photon_level:
                        select_kwargs = dict(mkw)
                        if reselect_per_photon_level and \
                                "oed_photon_count" not in select_kwargs:
                            select_kwargs["oed_photon_count"] = float(I0)
                        t0 = time.time()
                        src = stack["build_baseline_sources"](
                            name, k, geom["sid"], roi_center=roi_center,
                            vcl_precompute=vcl_pre,
                            volume=vol_prior, sdd=geom["sdd"],
                            detector_shape=(geom["det_voxels"],
                                            geom["det_voxels"]),
                            du=geom["det_pitch"], dv=geom["det_pitch"],
                            voxel_spacing=geom["voxel_pitch"],
                            n_candidates=n_cand,
                            seed=sel_seed, method_kwargs=select_kwargs,
                        )
                        mx.eval(src)
                        sel_s = time.time() - t0

                    per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
                    for ns in noise_seeds:
                        m = _reconstruct(stack, vol_gt, src, geom,
                                         int(cfg["sart_iterations"]), I0, ns,
                                         spectrum=spectrum)
                        raw = {"method": label, "impl": name, "k": k,
                               "photon_count": I0,
                               "selection_seed": sel_seed, "noise_seed": ns}
                        for key in METRICS:
                            per_metric[key].append(float(m[key]))
                            raw[key] = round(float(m[key]), 6)
                        raw_rows.append(raw)
                    row = {
                        "method": label, "impl": name, "k": k,
                        "photon_count": I0,
                        "selection_seed": sel_seed,
                        "cache_s": round(cache_s, 2), "sel_s": round(sel_s, 2),
                        "total_s": round(cache_s + sel_s, 2),
                        "n_noise": len(noise_seeds),
                    }
                    for key in METRICS:
                        mean, std = _mean_std(per_metric[key])
                        row[f"{key}_mean"] = round(mean, 4)
                        row[f"{key}_std"] = round(std, 4)
                    rows.append(row)
                    print(f"  {label:>18}  k={k}  I0={I0:.0e}  sel={sel_seed}  "
                          f"PSNR={row['psnr_mean']:.3f}±{row['psnr_std']:.3f}  "
                          f"SSIM={row['ssim_mean']:.4f}±{row['ssim_std']:.4f}  "
                          f"total_s={row['total_s']:.1f}", flush=True)

    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "impl", "k", "photon_count", "selection_seed",
              "n_noise",
              "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
              "hfen_mean", "hfen_std", "cache_s", "sel_s", "total_s"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}", flush=True)

    raw_out = out.with_suffix(".raw.csv")
    raw_fields = ["method", "impl", "k", "photon_count", "selection_seed",
                  "noise_seed"] + METRICS
    with raw_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        w.writerows(raw_rows)
    print(f"Wrote {raw_out}", flush=True)

    paired_out = out.with_suffix(".paired.csv")
    paired_rows = _paired_differences(
        raw_rows, cfg.get("paired_baseline", cfg["methods"][0]["name"]))
    if paired_rows:
        with paired_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
            w.writeheader()
            w.writerows(paired_rows)
        print(f"Wrote {paired_out}", flush=True)
    return 0


def _paired_differences(raw_rows: list[dict], baseline_impl: str) -> list[dict]:
    """Paired method-vs-baseline differences (REV-P1-02).

    Pairs on (k, I0, selection_seed, noise_seed); the experimental unit for
    the interval is the SELECTION seed: per seed, the paired differences are
    averaged over the shared noise seeds, then the cross-seed mean, SD, min,
    and max are reported (small-n honesty: every per-seed value is recoverable
    from the raw CSV).
    """
    base = {(r["k"], r["photon_count"], r["selection_seed"], r["noise_seed"]): r
            for r in raw_rows if r["impl"] == baseline_impl}
    if not base:
        return []
    out: list[dict] = []
    impls = sorted({r["impl"] for r in raw_rows} - {baseline_impl})
    cells = sorted({(r["k"], r["photon_count"]) for r in raw_rows})
    sel_seeds = sorted({r["selection_seed"] for r in raw_rows})
    for impl in impls:
        rows_i = [r for r in raw_rows if r["impl"] == impl]
        for (k, I0) in cells:
            row: dict = {"impl": impl, "baseline": baseline_impl,
                         "k": k, "photon_count": I0,
                         "n_selection": len(sel_seeds)}
            for metric in METRICS:
                per_seed: list[float] = []
                for s in sel_seeds:
                    diffs = [r[metric] - base[(k, I0, s, r["noise_seed"])][metric]
                             for r in rows_i
                             if r["k"] == k and r["photon_count"] == I0
                             and r["selection_seed"] == s
                             and (k, I0, s, r["noise_seed"]) in base]
                    if diffs:
                        per_seed.append(statistics.fmean(diffs))
                if not per_seed:
                    continue
                row[f"d{metric}_mean"] = round(statistics.fmean(per_seed), 4)
                row[f"d{metric}_sd"] = round(
                    statistics.pstdev(per_seed) if len(per_seed) > 1 else 0.0, 4)
                row[f"d{metric}_min"] = round(min(per_seed), 4)
                row[f"d{metric}_max"] = round(max(per_seed), 4)
            out.append(row)
    return out


if __name__ == "__main__":
    sys.exit(main())
