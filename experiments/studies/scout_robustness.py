"""Minimal scout-view robustness experiment.

Question
--------
In practice the true attenuation map is *not* available at planning time.
A realistic pipeline first acquires a handful of cheap *scout* projections,
reconstructs a coarse scout volume, and plans the sparse-view scan on that
imperfect prior.  How much does each selector degrade when it must plan on a
scout reconstruction instead of the (oracle) true mu-map -- and how many scout
views are needed before the gap closes?

Workflow (per phantom x scout-count x target-view-count)
--------------------------------------------------------
1. Load the true mu-map.
2. Generate scout projections from the true mu-map using uniform circular
   scout views.
3. Reconstruct a scout volume (SART).
4. Plan the sparse-view scan using the *scout* volume.
5. Simulate the final acquisition from the *true* mu-map at ``I0_final``
   (Beer-Lambert / Poisson photon noise).
6. Reconstruct the final sparse-view volume (SART).
7. Evaluate the final reconstruction against the *true* mu-map.

The final reconstruction is **never** evaluated against the scout
reconstruction.

Reference (oracle / full-prior) results are planned on the true mu-map.  They
are cached to ``results/scout_reference.csv`` and reused on subsequent runs --
existing reference results are *not* recomputed.

Method name mapping (task label -> implementation)
--------------------------------------------------
    uniform_bundle        -> uniform_adam_bundle   (scout-independent baseline)
    vcls                  -> vcls
    vcls_bundle           -> vcls_adam_bundle
    greedy_bundle         -> greedy_adam_bundle
    greedy_bundle_ocr     -> greedy_adam_oed        (photon-weighted A+D design)

Usage::

    python run_minimal_scout_experiment.py            # defaults (128^3, K_max=360)
    python run_minimal_scout_experiment.py --resolution 192 --kmax 720
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from experiments.run import (  # type: ignore
    _compute_metrics,
    _load_mlx_stack,
    _load_phantom_pair,
    _resolve_geometry,
)


def _load_runtime_dependencies() -> None:
    """Load MLX-bound helpers only after command-line parsing.

    Importing this module, compiling it, and asking for ``--help`` must not
    touch a Metal device.  The globals are used exclusively by the experiment
    routines, which run after this function is called by :func:`main`.
    """
    global mx, geometry_from_sources, simulate_sinogram
    global reconstruct_sart_volume, uniform_arc

    import mlx.core as mlx
    from differentiable_coverage.eval.geometry import geometry_from_sources as geometry
    from differentiable_coverage.eval.reco import (
        reconstruct_sart_volume as reconstruct,
        simulate_sinogram as simulate,
    )
    from differentiable_coverage.eval.trajectories import uniform_arc as arc

    mx = mlx
    geometry_from_sources = geometry
    simulate_sinogram = simulate
    reconstruct_sart_volume = reconstruct
    uniform_arc = arc

# --------------------------------------------------------------------------
# Experiment definition
# --------------------------------------------------------------------------

OUT_ROOT = Path("experiments/scout_minimal")

# Phantom registry.  "symmetric" is the (approximately axisymmetric) ORNL
# hexagonal fuel nozzle; "moderate" is the off-centre industrial moderate
# phantom.  These are the two test-object classes that survive in the paper.
PHANTOMS = {
    "symmetric": {
        "spec": {"type": "ornl_nozzle", "path": "data/ornl_nozzle.h5", "section": "L"},
        "geometry": "ornl",
    },
    "moderate": {
        "spec": {"type": "milp_npy", "path": "data/moderate_asd_pocs_384.npy"},
        "geometry": "milp",
    },
    "lof_flange_v3": {
        "spec": {"type": "milp_npy", "path": "data/lof_flange_v3.npy"},
        "geometry": "milp",
    },
    "lof_flange_v4a": {
        "spec": {"type": "milp_npy", "path": "data/lof_flange_v4a.npy"},
        "geometry": "milp",
    },
    "lof_flange_v4b": {
        "spec": {"type": "milp_npy", "path": "data/lof_flange_v4b.npy"},
        "geometry": "milp",
    },
}

QUALITY_METRICS = ["psnr", "ssim", "nrmse", "hfen"]

# task label -> (implementation name, needs (R, gamma) precompute)
METHOD_IMPL = {
    "uniform_bundle": ("uniform_adam_bundle", False),
    "vcls": ("vcls", True),
    "vcls_bundle": ("vcls_adam_bundle", True),
    "greedy_bundle": ("greedy_adam_bundle", False),
    "greedy_bundle_ocr": ("greedy_adam_oed", False),
    # The paper's headline cold arm: coverage + VCL information + bundle.
    # Needs the (R, gamma) precompute, which the scout branch builds from the
    # scout volume, so this arm reads the prior through both of its
    # object-dependent terms.
    "greedy_composite": ("greedy_adam_composite", True),
}

# Reference (oracle) lines: planned on the true mu-map.
REFERENCE_METHODS = [
    "uniform_bundle",
    "vcls",
    "vcls_bundle",
    "greedy_bundle",
    "greedy_bundle_ocr",
    "greedy_composite",
]

# Scout-based selectors (planned on the scout reconstruction).  uniform_bundle
# is scout-independent, so it is only a reference line, not a scout curve.
SCOUT_METHODS = [
    "vcls_scout",
    "vcls_bundle_scout",
    "greedy_bundle_scout",
    "greedy_bundle_ocr_scout",
    "greedy_composite_scout",
]

# scout label -> reference label it derives from
SCOUT_BASE = {
    "vcls_scout": "vcls",
    "vcls_bundle_scout": "vcls_bundle",
    "greedy_bundle_scout": "greedy_bundle",
    "greedy_bundle_ocr_scout": "greedy_bundle_ocr",
    "greedy_composite_scout": "greedy_composite",
}

# --------------------------------------------------------------------------
# Core building blocks
# --------------------------------------------------------------------------

def _geom_kwargs(geom: dict) -> dict:
    return {
        "det_u": geom["det_voxels"], "det_v": geom["det_voxels"],
        "du": geom["det_pitch"], "dv": geom["det_pitch"],
        "voxel_spacing": geom["voxel_pitch"],
    }


def _forward(vol_gt, src, geom, *, photon_count=None, noise_seed=0):
    """Cone-beam forward projection of ``vol_gt`` from ``src`` positions."""
    sp, dc, du, dv = geometry_from_sources(src, sid=geom["sid"], sdd=geom["sdd"])
    sino = simulate_sinogram(
        vol_gt, sp, dc, du, dv,
        det_u=geom["det_voxels"], det_v=geom["det_voxels"],
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"],
        photon_count=photon_count, noise_seed=noise_seed,
    )
    mx.eval(sino)
    return sino, (sp, dc, du, dv)


def _reconstruct(vol_shape, sino, geomvecs, geom, sart_iter):
    sp, dc, du, dv = geomvecs
    res = reconstruct_sart_volume(
        vol_shape, sino, sp, dc, du, dv,
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"],
        iteration_count=sart_iter, show_progress=False,
    )
    mx.eval(res.reconstruction)
    return res.reconstruction


def _scout_sources(n_scout, sid, geometry="circle", elevation=30.0):
    """Source positions of the scout acquisition.

    ``circle`` is the equatorial prescan used by the published sweep. ``band``
    spreads the same number of views over the +-``elevation`` band that the
    measured bench can reach, as a single sinusoidal sweep in elevation, so a
    scout of a Defrise-type part is not acquired from the one orbit that is
    structurally blind to it.
    """
    if geometry == "circle":
        return uniform_arc(int(n_scout), sid)
    if geometry != "band":
        raise ValueError(f"unknown scout geometry {geometry!r}")
    n = int(n_scout)
    th = np.arange(n) * (2.0 * np.pi / n)
    phi = np.radians(elevation) * np.sin(2.0 * np.pi * np.arange(n) / max(n - 1, 1))
    src = np.stack([-sid * np.cos(phi) * np.sin(th),
                    sid * np.cos(phi) * np.cos(th),
                    sid * np.sin(phi)], axis=-1).astype("float32")
    return mx.array(src)


def _reconstruct_scout(vol_gt, geom, n_scout, sart_iter, scout_photon=None, seed=0,
                       scout_geometry="circle"):
    """Scout acquisition -> SART scout volume."""
    src = _scout_sources(int(n_scout), geom["sid"], scout_geometry)
    mx.eval(src)
    sino, geomvecs = _forward(vol_gt, src, geom,
                              photon_count=scout_photon, noise_seed=seed)
    return _reconstruct(vol_gt.shape, sino, geomvecs, geom, sart_iter)


def _build_vcl_cache(stack, plan_volume, geom, k_max, r1, seed):
    candidates = stack["sample_unit_sphere"](k_max, seed=seed) * geom["sid"]
    pre = stack["compute_R_gamma"](
        plan_volume, candidates,
        sid=geom["sid"], sdd=geom["sdd"],
        det_shape=(geom["det_voxels"], geom["det_voxels"]),
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"], r1=r1, seed=seed,
    )
    mx.eval(pre.R)
    return pre


def _plan(stack, impl_name, k, plan_volume, geom, vcl_pre, n_candidates, seed,
          roi_center):
    """Select ``k`` source positions with ``impl_name`` on ``plan_volume``."""
    src = stack["build_baseline_sources"](
        impl_name, int(k), geom["sid"], roi_center=roi_center,
        vcl_precompute=vcl_pre, volume=plan_volume, sdd=geom["sdd"],
        detector_shape=(geom["det_voxels"], geom["det_voxels"]),
        du=geom["det_pitch"], dv=geom["det_pitch"],
        voxel_spacing=geom["voxel_pitch"], n_candidates=n_candidates,
        seed=seed, method_kwargs={},
    )
    mx.eval(src)
    return src


def _plan_line_integral_stats(vol_gt, src, geom, i0_final):
    """Plan-quality stats from clean line integrals of the true mu-map."""
    sino, _ = _forward(vol_gt, src, geom, photon_count=None)
    p = np.asarray(sino, dtype=np.float64).reshape(-1)
    p = np.maximum(p, 0.0)
    intensity = float(i0_final) * np.exp(-p)
    return {
        "li_mean": round(float(p.mean()), 5),
        "li_max": round(float(p.max()), 5),
        "li_p95": round(float(np.percentile(p, 95)), 5),
        "photon_starved_frac": round(float(np.mean(intensity < 10.0)), 6),
    }


def _eval_plan(stack, vol_gt, src, geom, sart_iter, i0_final, seed):
    """Final noisy acquisition + SART + quality metrics vs the true mu-map."""
    sino, geomvecs = _forward(vol_gt, src, geom,
                              photon_count=i0_final, noise_seed=seed)
    recon = _reconstruct(vol_gt.shape, sino, geomvecs, geom, sart_iter)
    metrics = _compute_metrics(vol_gt, recon, float(vol_gt.max()),
                               QUALITY_METRICS, stack)
    return metrics, recon


def _save_volume(path: Path, vol):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(vol, dtype=np.float16))


def _save_plan(path: Path, src):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(src, dtype=np.float32))


# --------------------------------------------------------------------------
# Reference (oracle) computation with disk reuse
# --------------------------------------------------------------------------

REFERENCE_FIELDS = [
    "phantom", "method", "k", "resolution", "kmax", "seed",
    "psnr", "ssim", "nrmse", "hfen",
    "li_mean", "li_max", "li_p95", "photon_starved_frac",
]


def _ref_key(phantom, method, k, resolution, kmax, seed):
    return (phantom, method, int(k), int(resolution), int(kmax), int(seed))


def _load_reference(path: Path):
    """Load cached oracle rows.  Rows are keyed by (phantom, method, k,
    resolution, kmax, seed) so reference results computed under different
    settings or seeds are never silently reused.  A legacy file without the
    settings columns is ignored (warned)."""
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        if (reader.fieldnames is None or "resolution" not in reader.fieldnames
                or "seed" not in reader.fieldnames):
            print(f"[WARN] ignoring legacy reference cache {path} "
                  "(missing resolution/kmax/seed columns)", flush=True)
            return {}
        for row in reader:
            key = _ref_key(row["phantom"], row["method"], int(float(row["k"])),
                           int(float(row["resolution"])), int(float(row["kmax"])),
                           int(float(row["seed"])))
            out[key] = row
    return out


def _compute_reference(stack, args, geom, vol_gt, phantom, k_values,
                       existing, ref_rows, save_dir, roi_center):
    """Plan each reference method on the true mu-map (oracle).  Reuse rows
    already present in ``existing`` (do not recompute)."""
    needs_cache = any(METHOD_IMPL[m][1] for m in REFERENCE_METHODS)
    vcl_pre = None
    if needs_cache:
        miss = any(
            _ref_key(phantom, m, k, args.resolution, args.kmax, args.seed)
            not in existing
            for m in REFERENCE_METHODS if METHOD_IMPL[m][1]
            for k in k_values)
        if miss:
            t = time.time()
            vcl_pre = _build_vcl_cache(stack, vol_gt, geom, args.kmax,
                                       args.r1, args.seed)
            print(f"    [ref] (R,gamma) cache on true mu-map: "
                  f"{time.time() - t:.1f}s", flush=True)

    for method in REFERENCE_METHODS:
        impl, _ = METHOD_IMPL[method]
        for k in k_values:
            key = _ref_key(phantom, method, k, args.resolution, args.kmax,
                           args.seed)
            if key in existing:
                ref_rows[key] = existing[key]
                print(f"    [ref] reuse {method:>18}  k={k}", flush=True)
                continue
            t = time.time()
            src = _plan(stack, impl, k, vol_gt, geom, vcl_pre,
                        args.n_candidates, args.seed, roi_center)
            _save_plan(save_dir / "plans" /
                       f"{phantom}_ref_{method}_k{k}_seed{args.seed}.npy", src)
            plan_stats = _plan_line_integral_stats(vol_gt, src, geom, args.i0_final)
            metrics, recon = _eval_plan(stack, vol_gt, src, geom,
                                        args.sart_iter, args.i0_final, args.seed)
            if not args.no_save_recons:
                _save_volume(save_dir / "recons" /
                             f"{phantom}_ref_{method}_k{k}_seed{args.seed}.npy",
                             recon)
            row = {"phantom": phantom, "method": method, "k": k,
                   "resolution": args.resolution, "kmax": args.kmax,
                   "seed": args.seed,
                   **{m: round(metrics[m], 5) for m in QUALITY_METRICS},
                   **plan_stats}
            ref_rows[key] = row
            print(f"    [ref] {method:>18}  k={k}  "
                  f"PSNR={metrics['psnr']:.3f}  HFEN={metrics['hfen']:.4f}  "
                  f"starved={plan_stats['photon_starved_frac']:.4f}  "
                  f"({time.time() - t:.1f}s)", flush=True)


# --------------------------------------------------------------------------
# Main sweep
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--kmax", type=int, default=360,
                    help="VCLS candidate-pool size for the (R, gamma) precompute")
    ap.add_argument("--n-candidates", type=int, default=200,
                    help="greedy candidate pool for the continuous selectors")
    ap.add_argument("--sart-iter", type=int, default=15)
    ap.add_argument("--i0-final", type=float, default=1.0e5)
    ap.add_argument("--scout-photon", type=float, default=None,
                    help="photon count for the scout acquisition "
                         "(default: noise-free scout)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r1", type=float, default=1.0e-3)
    ap.add_argument("--scout-views", default="8,16,32,64")
    ap.add_argument("--scout-geometry", default="circle", choices=["circle", "band"],
                    help="prescan geometry: equatorial circle (published sweep) "
                         "or a +-30 deg band sweep on the bench manifold")
    ap.add_argument("--target-views", default="40,80")
    ap.add_argument("--phantoms", default="symmetric,moderate")
    ap.add_argument("--plot-k", type=int, default=80,
                    help="target view count rendered in the figure")
    ap.add_argument("--no-save-recons", action="store_true",
                    help="skip writing the final reconstruction volumes")
    ap.add_argument("--skip-plot", action="store_true")
    ap.add_argument("--skip-methods", default="",
                    help="comma-separated base method labels excluded from "
                         "both the reference and scout sweeps, e.g. "
                         "'greedy_bundle_ocr'")
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="output directory root (default experiments/scout_minimal)")
    args = ap.parse_args(argv)

    skip = {x.strip() for x in args.skip_methods.split(",") if x.strip()}
    unknown = skip - set(METHOD_IMPL)
    if unknown:
        raise SystemExit(
            f"unknown --skip-methods {sorted(unknown)}; "
            f"choose from {list(METHOD_IMPL)}"
        )
    if skip:
        REFERENCE_METHODS[:] = [m for m in REFERENCE_METHODS if m not in skip]
        SCOUT_METHODS[:] = [m for m in SCOUT_METHODS if SCOUT_BASE[m] not in skip]
        print(f"skipping methods: {sorted(skip)}", flush=True)

    scout_views = [int(x) for x in args.scout_views.split(",") if x.strip()]
    target_views = [int(x) for x in args.target_views.split(",") if x.strip()]
    phantom_names = [x.strip() for x in args.phantoms.split(",") if x.strip()]
    for ph in phantom_names:
        if ph not in PHANTOMS:
            raise SystemExit(f"unknown phantom '{ph}'; choose from {list(PHANTOMS)}")

    save_dir = Path(args.out_root)
    (save_dir / "scouts").mkdir(parents=True, exist_ok=True)
    (save_dir / "plans").mkdir(parents=True, exist_ok=True)
    (save_dir / "recons").mkdir(parents=True, exist_ok=True)
    (save_dir / "results").mkdir(parents=True, exist_ok=True)
    (save_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Defer all MLX imports and arrays until after parsing so ``--help`` and
    # manifest validation are safe on headless hosts without a Metal device.
    _load_runtime_dependencies()
    stack = _load_mlx_stack()
    roi_center = mx.array([0.0, 0.0, 0.0])

    print("=" * 70, flush=True)
    print("Minimal scout-view robustness experiment", flush=True)
    print(f"  resolution   : {args.resolution}^3", flush=True)
    print(f"  scout views  : {scout_views}", flush=True)
    print(f"  target views : {target_views}", flush=True)
    print(f"  I0_final     : {args.i0_final:.1e}   seed: {args.seed}", flush=True)
    print(f"  K_max (VCLS) : {args.kmax}", flush=True)
    print("=" * 70, flush=True)

    reference_path = save_dir / "results" / "scout_reference.csv"
    existing_ref = _load_reference(reference_path)
    ref_rows: dict = {}
    scout_rows: list[dict] = []

    for phantom in phantom_names:
        info = PHANTOMS[phantom]
        spec = {**info["spec"], "resolution": args.resolution}
        geom = _resolve_geometry(info["geometry"], args.resolution)
        try:
            vol_gt, _ = _load_phantom_pair(spec, stack)
        except FileNotFoundError as e:
            print(f"[WARN] phantom '{phantom}' unavailable ({e}); skipping",
                  flush=True)
            continue
        mx.eval(vol_gt)
        print(f"\n### phantom '{phantom}'  shape={tuple(vol_gt.shape)}  "
              f"max_mu={float(vol_gt.max()):.3f}  geom={info['geometry']}",
              flush=True)

        # ---- reference / oracle lines (planned on the true mu-map) ----------
        _compute_reference(stack, args, geom, vol_gt, phantom, target_views,
                           existing_ref, ref_rows, save_dir, roi_center)

        # ---- scout sweep ----------------------------------------------------
        for n_scout in scout_views:
            t_scout = time.time()
            scout_vol = _reconstruct_scout(
                vol_gt, geom, n_scout, args.sart_iter,
                scout_photon=args.scout_photon, seed=args.seed,
                scout_geometry=args.scout_geometry,
            )
            _save_volume(save_dir / "scouts" /
                         f"{phantom}_scout{n_scout}_seed{args.seed}.npy",
                         scout_vol)
            print(f"  -- scout n={n_scout:>2}  reco {time.time() - t_scout:.1f}s",
                  flush=True)

            # Shared (R, gamma) cache from the scout volume (vcls* methods).
            scout_needs_cache = any(
                METHOD_IMPL[SCOUT_BASE[m]][1] for m in SCOUT_METHODS
            )
            scout_pre = None
            if scout_needs_cache:
                t = time.time()
                scout_pre = _build_vcl_cache(stack, scout_vol, geom, args.kmax,
                                             args.r1, args.seed)
                print(f"     [scout] (R,gamma) on scout vol: "
                      f"{time.time() - t:.1f}s", flush=True)

            for method in SCOUT_METHODS:
                base = SCOUT_BASE[method]
                impl, _ = METHOD_IMPL[base]
                for k in target_views:
                    t = time.time()
                    src = _plan(stack, impl, k, scout_vol, geom, scout_pre,
                                args.n_candidates, args.seed, roi_center)
                    _save_plan(save_dir / "plans" /
                               f"{phantom}_{method}_s{n_scout}_k{k}"
                               f"_seed{args.seed}.npy", src)
                    plan_stats = _plan_line_integral_stats(
                        vol_gt, src, geom, args.i0_final)
                    metrics, recon = _eval_plan(
                        stack, vol_gt, src, geom,
                        args.sart_iter, args.i0_final, args.seed)
                    if not args.no_save_recons:
                        _save_volume(save_dir / "recons" /
                                     f"{phantom}_{method}_s{n_scout}_k{k}"
                                     f"_seed{args.seed}.npy", recon)
                    row = {"phantom": phantom, "method": method,
                           "scout_views": n_scout, "k": k,
                           "resolution": args.resolution, "kmax": args.kmax,
                           "seed": args.seed,
                           **{m: round(metrics[m], 5) for m in QUALITY_METRICS},
                           **plan_stats}
                    scout_rows.append(row)
                    print(f"     {method:>22}  s={n_scout:>2} k={k}  "
                          f"PSNR={metrics['psnr']:.3f}  HFEN={metrics['hfen']:.4f}"
                          f"  starved={plan_stats['photon_starved_frac']:.4f}"
                          f"  ({time.time() - t:.1f}s)", flush=True)

    # ---- persist reference cache (existing rows are preserved) --------------
    _write_reference(reference_path, ref_rows)

    # ---- write combined metrics CSV + JSON ----------------------------------
    _write_outputs(save_dir, scout_rows, ref_rows, scout_views, target_views,
                   args)

    # ---- figures ------------------------------------------------------------
    if not args.skip_plot:
        try:
            _make_figures(save_dir, scout_rows, ref_rows, scout_views,
                          phantom_names, args.plot_k)
        except Exception as e:  # pragma: no cover - plotting is best-effort
            print(f"[WARN] figure generation failed: {e}", flush=True)

    print("\nDone.", flush=True)
    return 0


def _write_reference(path: Path, ref_rows: dict):
    # Merge with whatever is already on disk so a partial run (a subset of
    # phantoms / settings) never destroys cached oracle rows for other cells.
    merged = dict(_load_reference(path))
    merged.update(ref_rows)
    rows = [merged[k] for k in sorted(merged)]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REFERENCE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in REFERENCE_FIELDS})
    print(f"\nWrote {path}  ({len(rows)} reference rows)", flush=True)


def _write_outputs(save_dir, scout_rows, ref_rows, scout_views, target_views,
                   args):
    csv_path = save_dir / "results" / "scout_metrics.csv"
    json_path = save_dir / "results" / "scout_metrics.json"

    scout_fields = ["phantom", "method", "scout_views", "k",
                    "resolution", "kmax", "seed",
                    "psnr", "ssim", "nrmse", "hfen",
                    "li_mean", "li_max", "li_p95", "photon_starved_frac"]

    # Merge with rows already on disk (keyed including seed) so multi-seed and
    # partial runs accumulate rather than clobber the results.
    def _skey(r):
        return (r["phantom"], r["method"], int(float(r["scout_views"])),
                int(float(r["k"])), int(float(r.get("resolution", 0))),
                int(float(r.get("kmax", 0))), int(float(r.get("seed", 0))))

    merged: dict = {}
    if csv_path.exists():
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            if (reader.fieldnames and "resolution" in reader.fieldnames
                    and "seed" in reader.fieldnames):
                for r in reader:
                    merged[_skey(r)] = r
    for r in scout_rows:
        merged[_skey(r)] = r
    out_rows = [merged[k] for k in sorted(merged)]

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scout_fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in scout_fields})
    print(f"Wrote {csv_path}  ({len(out_rows)} scout rows)", flush=True)

    merged_ref = _load_reference(save_dir / "results" / "scout_reference.csv")
    payload = {
        "config": {
            "resolution": args.resolution,
            "kmax": args.kmax,
            "n_candidates": args.n_candidates,
            "sart_iter": args.sart_iter,
            "i0_final": args.i0_final,
            "scout_photon": args.scout_photon,
            "seed": args.seed,
            "scout_views": scout_views,
            "target_views": target_views,
        },
        "method_mapping": {k: v[0] for k, v in METHOD_IMPL.items()},
        "scout_results": out_rows,
        "reference_results": [merged_ref[k] for k in sorted(merged_ref)],
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {json_path}", flush=True)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

_METRIC_LABEL = {"psnr": "PSNR (dB)", "hfen": "HFEN", "ssim": "SSIM",
                 "nrmse": "NRMSE"}
_SCOUT_STYLE = {
    "vcls_scout": ("#1f77b4", "o"),
    "vcls_bundle_scout": ("#2ca02c", "s"),
    "greedy_bundle_scout": ("#ff7f0e", "^"),
    "greedy_bundle_ocr_scout": ("#d62728", "D"),
}
_REF_STYLE = {
    "uniform_bundle": "#7f7f7f",
    "vcls": "#1f77b4",
    "vcls_bundle": "#2ca02c",
    "greedy_bundle": "#ff7f0e",
    "greedy_bundle_ocr": "#d62728",
}


def _make_figures(save_dir, scout_rows, ref_rows, scout_views, phantom_names,
                  plot_k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [p for p in phantom_names
               if any(r["phantom"] == p for r in scout_rows)]
    if not present:
        print("[WARN] no scout rows to plot", flush=True)
        return
    metrics = ["psnr", "hfen"]
    resolution = int({int(float(r.get("resolution", 0)))
                      for r in ref_rows.values()}.pop()) if ref_rows else 0
    kmax = int({int(float(r.get("kmax", 0)))
                for r in ref_rows.values()}.pop()) if ref_rows else 0

    import statistics

    def scout_curve(phantom, method, metric):
        """Mean +/- std over seeds of the per-scout-budget metric."""
        xs, ys, es = [], [], []
        for s in scout_views:
            vals = [float(r[metric]) for r in scout_rows
                    if r["phantom"] == phantom and r["method"] == method
                    and int(float(r["scout_views"])) == s
                    and int(float(r["k"])) == plot_k]
            if vals:
                xs.append(s)
                ys.append(statistics.fmean(vals))
                es.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        return xs, ys, es

    def ref_value(phantom, method, metric):
        vals = [float(r[metric]) for r in ref_rows.values()
                if r["phantom"] == phantom and r["method"] == method
                and int(float(r["k"])) == plot_k
                and int(float(r["resolution"])) == resolution
                and int(float(r["kmax"])) == kmax]
        return statistics.fmean(vals) if vals else None

    nrow, ncol = len(present), len(metrics)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.7 * nrow),
                             squeeze=False)
    handles, labels = [], []
    for i, phantom in enumerate(present):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            # scout curves (mean +/- std over seeds)
            for method, (color, marker) in _SCOUT_STYLE.items():
                xs, ys, es = scout_curve(phantom, method, metric)
                if xs:
                    ax.plot(xs, ys, color=color, marker=marker, lw=1.8,
                            ms=6, label=method)
                    if any(e > 0 for e in es):
                        lo = [y - e for y, e in zip(ys, es)]
                        hi = [y + e for y, e in zip(ys, es)]
                        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
            # reference horizontal lines
            for rmethod, color in _REF_STYLE.items():
                v = ref_value(phantom, rmethod, metric)
                if v is not None:
                    ax.axhline(v, color=color, ls="--", lw=1.1, alpha=0.7,
                               label=f"{rmethod} (oracle)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(scout_views)
            ax.set_xticklabels([str(s) for s in scout_views])
            ax.set_xlabel("scout views")
            ax.set_ylabel(_METRIC_LABEL.get(metric, metric))
            ax.set_title(f"{phantom}  -  {_METRIC_LABEL.get(metric, metric)}"
                         f"  (k={plot_k})")
            ax.grid(True, which="both", alpha=0.25)
            if not handles:  # capture legend from the first populated axis
                handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=7, ncol=3,
                   loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Scout-based planning robustness (target k={plot_k})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    png = save_dir / "figures" / "scout_quality_vs_views.png"
    pdf = save_dir / "figures" / "scout_quality_vs_views.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}", flush=True)
    print(f"Wrote {pdf}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
