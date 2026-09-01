"""Precompute the rotated Fibonacci lattices and the VCL (R, gamma)
caches for the validation-experiment seeds, and save them under
``data/cache/`` so subsequent runs do not pay the 5--22 minute cache
construction cost per (phantom, seed) cell.

The cache files contain the candidate source positions, the sample
indices used for the r1 voxel subsampling, and the full (R, gamma)
arrays.  Loading is then a millisecond ``np.load`` instead of a
SART-projector loop.

Layout::

    data/cache/
        sphere_<N>_seed<S>.npy             (N x 3 unit-sphere lattice)
        vcl_<phantom>_<res>_K<KMAX>_seed<S>.npz

Run::

    python experiments/precompute_caches.py --seeds 0,1,2,3,4 \\
        --phantoms milp_mild,milp_moderate,synthetic_metal_hard,ornl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "data" / "cache"

PHANTOM_REGISTRY = {
    "milp_mild": {
        "type": "milp_npy",
        "path": "data/mild_asd_pocs_384.npy",
        "resolution": 384,
        "geometry": "milp",
    },
    "milp_moderate": {
        "type": "milp_npy",
        "path": "data/moderate_asd_pocs_384.npy",
        "resolution": 384,
        "geometry": "milp",
    },
    "milp_mild_192": {     # small-scale companion for smoke tests
        "type": "milp_npy",
        "path": "data/mild_asd_pocs_384.npy",
        "resolution": 192,
        "geometry": "milp",
    },
    "synthetic_metal_mild": {
        "type": "milp_npy",
        "path": "data/synthetic_metal_dataset/mu_maps/mild.npy",
        "resolution": 256,
        "geometry": "milp",
    },
    "synthetic_metal_moderate": {
        "type": "milp_npy",
        "path": "data/synthetic_metal_dataset/mu_maps/moderate.npy",
        "resolution": 256,
        "geometry": "milp",
    },
    "synthetic_metal_hard": {
        "type": "milp_npy",
        "path": "data/synthetic_metal_dataset/mu_maps/hard.npy",
        "resolution": 256,
        "geometry": "milp",
    },
    "ornl": {
        "type": "ornl_nozzle",
        "path": "data/ornl_nozzle.h5",
        "section": "L",
        "resolution": 512,
        "geometry": "ornl",
    },
}


def _import_runner():
    """Late import to keep --help fast and to avoid pulling MLX on dry-run."""
    sys.path.insert(0, str(REPO))
    from experiments.run import _load_phantom, _resolve_geometry, _load_mlx_stack
    from differentiable_coverage.score import sample_unit_sphere
    from differentiable_coverage.eval.vcl import compute_R_gamma
    return {
        "_load_phantom": _load_phantom,
        "_resolve_geometry": _resolve_geometry,
        "_load_mlx_stack": _load_mlx_stack,
        "sample_unit_sphere": sample_unit_sphere,
        "compute_R_gamma": compute_R_gamma,
    }


def _save_lattice(n: int, seed: int) -> Path:
    """Save the unit-sphere lattice for (n, seed) and return its path."""
    from differentiable_coverage.score import sample_unit_sphere
    out = CACHE_DIR / f"sphere_{n}_seed{seed}.npy"
    if out.exists():
        return out
    pts = np.asarray(sample_unit_sphere(n, seed=seed))
    np.save(out, pts)
    return out


def _save_vcl_cache(phantom_tag: str, k_max: int, seed: int,
                    helpers: dict) -> Path:
    spec = PHANTOM_REGISTRY[phantom_tag]
    out = CACHE_DIR / (
        f"vcl_{phantom_tag}_res{spec['resolution']}_K{k_max}_seed{seed}.npz"
    )
    if out.exists():
        print(f"    cache exists, skipping: {out.name}")
        return out

    print(f"    building {out.name} ...", flush=True)
    geometry = helpers["_resolve_geometry"](
        spec["geometry"], int(spec["resolution"])
    )
    stack = helpers["_load_mlx_stack"]()
    vol = helpers["_load_phantom"](spec, stack)
    mx.eval(vol)

    candidates = helpers["sample_unit_sphere"](k_max, seed=seed) * geometry["sid"]
    t0 = time.time()
    pre = helpers["compute_R_gamma"](
        vol, candidates,
        sid=geometry["sid"], sdd=geometry["sdd"],
        det_shape=(geometry["det_voxels"], geometry["det_voxels"]),
        du=geometry["det_pitch"], dv=geometry["det_pitch"],
        voxel_spacing=geometry["voxel_pitch"],
        r1=1e-3, seed=seed,
    )
    dt = time.time() - t0
    print(f"      done in {dt:.1f}s", flush=True)

    np.savez_compressed(
        out,
        R=pre.R,
        gamma=pre.gamma,
        sample_indices=pre.sample_indices,
        candidate_sources=np.asarray(pre.candidate_sources),
        volume_shape=np.array(pre.volume_shape),
        r1=pre.r1,
        sid=geometry["sid"], sdd=geometry["sdd"],
        det_voxels=geometry["det_voxels"], det_pitch=geometry["det_pitch"],
        voxel_pitch=geometry["voxel_pitch"],
        seed=seed,
    )
    return out


def load_vcl_cache(phantom_tag: str, k_max: int, seed: int):
    """Load a previously-built (R, gamma) cache.  Returns a
    ``VCLPrecompute`` instance compatible with the rest of the pipeline."""
    from differentiable_coverage.eval.vcl import VCLPrecompute

    spec = PHANTOM_REGISTRY[phantom_tag]
    path = CACHE_DIR / (
        f"vcl_{phantom_tag}_res{spec['resolution']}_K{k_max}_seed{seed}.npz"
    )
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    return VCLPrecompute(
        R=d["R"],
        gamma=d["gamma"],
        sample_indices=d["sample_indices"],
        candidate_sources=mx.array(d["candidate_sources"]),
        volume_shape=tuple(int(x) for x in d["volume_shape"]),
        r1=float(d["r1"]),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4",
                   help="Comma-separated seeds.")
    p.add_argument("--phantoms", default="milp_mild,milp_moderate",
                   help=f"Comma-separated; choose from "
                        f"{list(PHANTOM_REGISTRY.keys())}")
    p.add_argument("--k-max", type=int, default=720)
    p.add_argument("--lattice-sizes", default="720,360,256,128",
                   help="Sphere-lattice sizes to precompute as .npy.")
    args = p.parse_args(argv)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    phantoms = [s.strip() for s in args.phantoms.split(",") if s.strip()]
    lattice_sizes = [int(s) for s in args.lattice_sizes.split(",") if s.strip()]

    # 1. Pure-sphere lattices (fast, useful for the gradient_field figure
    #    and any non-cached sphere probe call site).
    print(f"=== Lattice files ({len(lattice_sizes)} sizes x {len(seeds)} seeds) ===")
    for n in lattice_sizes:
        for seed in seeds:
            p = _save_lattice(n, seed)
            print(f"  {p.name}")

    # 2. VCL (R, gamma) caches per (phantom, seed).
    print(f"\n=== VCL caches ({len(phantoms)} phantoms x {len(seeds)} seeds) ===")
    helpers = _import_runner()
    for ph in phantoms:
        if ph not in PHANTOM_REGISTRY:
            print(f"  ! unknown phantom {ph!r}, skipping")
            continue
        print(f"\n  -- {ph} --")
        for seed in seeds:
            try:
                _save_vcl_cache(ph, args.k_max, seed, helpers)
            except FileNotFoundError as e:
                print(f"      phantom file missing: {e}")
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
