"""View Covariance Loss (VCL) baseline after Lin et al., TPAMI 2025.

Reference
---------
J. Lin, A. Ziabari, S. Venkatakrishnan, O. Rahman, G. Buzzard, C. Bouman.
"Tomographic Sparse View Selection using the View Covariance Loss."
IEEE Trans. Pattern Anal. Mach. Intell., 2025.  DOI 10.1109/TPAMI.2025.3600072.

Definitions (Eq. 8–11 in the paper):

* Per-view basis  $T_\\theta = A_\\theta^\\top H A_\\theta x$
  (forward-project → ramp-filter → back-project of the reference object).
* Subsample matrix $S$ keeps a random fraction $r_1$ of voxels.
* $T_\\theta \\gets S T_\\theta / \\|S T_\\theta\\|$            (column-normalised)
* $\\gamma[i] = T_{\\theta_i}^\\top (Sx) / \\|Sx\\|$
* $R[i][j] = T_{\\theta_i}^\\top T_{\\theta_j}$  →  $K_{\\max} \\times K_{\\max}$.

For a chosen subset $\\Omega \\subseteq \\{1, \\dots, K_{\\max}\\}$ of size $K$:

$$L(R_\\Omega, \\gamma_\\Omega) = 1 - \\gamma_\\Omega^\\top R_\\Omega^{-1} \\gamma_\\Omega .$$

VCLS (Algorithm 1) computes $(R, \\gamma)$ once for the full candidate set,
then performs a random-swap search (Algorithm 2) to minimise $L$.

In our setting the candidates are arbitrary points on the source sphere;
diffct_mlx's matched footprint forward/back projectors do all of the heavy
lifting.  Default $r_1=10^{-3}$, $r_2=0.1$ match the paper's defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import time

import mlx.core as mx
import numpy as np

from diffct_mlx import (
    cone_forward_footprint,
)
from diffct_mlx.reconstruction_algorithms._analytic import ramp_filter_3d

from .._torch_bridge import to_backend as _to_backend, from_backend as _from_backend
from .geometry import geometry_from_sources
from ..vcl_backprojection import (
    backproject_single_view,
    extract_sampled_backprojection_np,
    sparse_backprojection_mode,
)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class VCLPrecompute:
    """Precomputed (R, γ) for one (volume, candidate set) pair."""

    R: np.ndarray  # (K_max, K_max), float32
    gamma: np.ndarray  # (K_max,), float32
    sample_indices: np.ndarray  # (NS,) flat voxel indices
    candidate_sources: mx.array  # (K_max, 3)
    volume_shape: tuple[int, int, int]
    r1: float
    used_sparse_backprojection: bool = False
    sparse_backprojection_mode: str | None = None

    @property
    def k_max(self) -> int:
        return int(self.R.shape[0])


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

def _sample_indices(volume_shape: tuple[int, int, int], r1: float,
                    rng: np.random.Generator,
                    roi_mask: np.ndarray | None = None) -> np.ndarray:
    """Random flat indices into ``volume_shape`` keeping fraction ``r1``.
    
    Parameters
    ----------
    volume_shape
        Shape of volume to sample from.
    r1
        Fraction of voxels to keep.
    rng
        Random number generator.
    roi_mask
        Optional boolean mask of shape ``volume_shape``. If provided, samples only
        from voxels where ``roi_mask`` is True. Reduces sampling to object support.
    
    Returns
    -------
    Flat indices into the volume.
    """
    if roi_mask is None:
        # Sample from full volume
        n_total = int(np.prod(volume_shape))
        n_keep = max(1, int(r1 * n_total))
        return rng.choice(n_total, size=n_keep, replace=False).astype(np.int64)
    else:
        # Sample only from ROI
        roi_indices = np.flatnonzero(roi_mask.ravel())
        if roi_indices.size == 0:
            raise ValueError("roi_mask does not contain any selectable voxels.")
        n_keep = max(1, int(r1 * len(roi_indices)))
        return rng.choice(roi_indices, size=n_keep, replace=False).astype(np.int64)


def _subsample_volume_flat(volume_np: np.ndarray, indices: np.ndarray) -> np.ndarray:
    flat = volume_np.reshape(-1)
    return flat[indices]


def _single_view_T(
    volume: mx.array,
    src_pos_single: mx.array,
    det_center_single: mx.array,
    det_u_single: mx.array,
    det_v_single: mx.array,
    *,
    det_shape: tuple[int, int],
    du: float,
    dv: float,
    voxel_spacing: float,
) -> mx.array:
    """Per-view filtered-backprojection T_θ = A^t H A x for a single source.

    Returns a 3-D volume the size of ``volume``.
    """
    # 1) forward-project: shape (1, du, dv)
    src = src_pos_single[None, :]
    dc = det_center_single[None, :]
    duv = det_u_single[None, :]
    dvv = det_v_single[None, :]
    sino = cone_forward_footprint(
        _to_backend(volume), _to_backend(src), _to_backend(dc),
        _to_backend(duv), _to_backend(dvv),
        det_u=det_shape[0], det_v=det_shape[1],
        du=du, dv=dv, voxel_spacing=voxel_spacing,
    )

    # 2) ramp-filter along detector-u axis (the helper expects a 3D sino);
    #    bring the filtered sinogram back to MLX so the (MLX-in/out) bridged
    #    backprojector below can consume it on the torch backend.
    filt = _from_backend(ramp_filter_3d(sino))

    # 3) back-project to (D, H, W) = volume.shape
    back, _used_sparse = backproject_single_view(
        filt, src, dc, duv, dvv,
        output_shape=tuple(volume.shape),
        du=du, dv=dv, voxel_spacing=voxel_spacing,
        sample_indices=None,
        prefer_sparse=False,
    )
    return back


# ---------------------------------------------------------------------------
# (R, γ) computation — Algorithm 1, Steps 1–2
# ---------------------------------------------------------------------------

def compute_R_gamma(
    volume: mx.array,
    candidate_sources: mx.array,
    *,
    sid: float,
    sdd: float,
    det_shape: tuple[int, int],
    du: float = 1.0,
    dv: float = 1.0,
    voxel_spacing: float = 1.0,
    r1: float = 1e-3,
    seed: int = 0,
    show_progress: bool = False,
    roi_mask: np.ndarray | None = None,
    prefer_sparse_backprojection: bool = True,
) -> VCLPrecompute:
    """Precompute the candidate-set autocorrelation matrix ``R`` and
    correlation vector ``γ`` for a single reference volume.

    The candidates can be any points on the source sphere — VCL does not
    require a particular orbit.
    
    Parameters
    ----------
    volume
        Reference volume ``(D, H, W)``.
    candidate_sources
        Candidate source positions ``(K_max, 3)``.
    sid, sdd
        Source-to-isocenter and source-to-detector distance.
    det_shape
        Detector shape ``(det_u, det_v)``.
    du, dv, voxel_spacing
        Detector and voxel spacings.
    r1
        Voxel subsampling rate (default 1e-3).
    seed
        Random seed for reproducibility.
    show_progress
        If True, print timing and diagnostic information.
    roi_mask
        Optional boolean mask of shape matching ``volume``. If provided, samples
        only from ROI voxels (e.g., object support or bounding box). Reduces
        sampling to meaningful voxels.
    prefer_sparse_backprojection
        If True, request sparse/sample-only backprojection from ``diffct_mlx``
        when the installed version exposes a compatible API. Falls back to the
        dense full-volume path automatically otherwise.
    """
    # Start timing
    t_start = time.time()
    
    volume_np = np.asarray(volume, dtype=np.float32)
    rng = np.random.default_rng(seed)
    
    # ROI diagnostic
    if roi_mask is not None:
        n_roi = np.count_nonzero(roi_mask)
    else:
        n_roi = int(np.prod(volume.shape))
    
    t0_sample = time.time()
    indices = _sample_indices(volume.shape, r1, rng, roi_mask=roi_mask)
    t_sample = time.time() - t0_sample
    
    t0_extract = time.time()
    x_sub = _subsample_volume_flat(volume_np, indices)
    t_extract = time.time() - t0_extract
    
    x_norm = float(np.linalg.norm(x_sub))
    if x_norm <= 0.0:
        raise ValueError("Subsampled reference volume has zero norm.")

    k_max = int(candidate_sources.shape[0])
    src_pos, det_center, det_u_vec, det_v_vec = geometry_from_sources(
        candidate_sources, sid=sid, sdd=sdd
    )

    T_sub = np.empty((k_max, indices.size), dtype=np.float32)
    t0_bases = time.time()
    times_per_view = []
    used_sparse_any = False
    sparse_mode = sparse_backprojection_mode() if prefer_sparse_backprojection else None
    
    for i in range(k_max):
        t0_view = time.time()
        src = src_pos[i:i + 1]
        dc = det_center[i:i + 1]
        duv = det_u_vec[i:i + 1]
        dvv = det_v_vec[i:i + 1]
        sino = cone_forward_footprint(
            _to_backend(volume), _to_backend(src), _to_backend(dc),
            _to_backend(duv), _to_backend(dvv),
            det_u=det_shape[0], det_v=det_shape[1],
            du=du, dv=dv, voxel_spacing=voxel_spacing,
        )
        filt = _from_backend(ramp_filter_3d(sino))
        back, used_sparse = backproject_single_view(
            filt, src, dc, duv, dvv,
            output_shape=tuple(volume.shape),
            du=du, dv=dv, voxel_spacing=voxel_spacing,
            sample_indices=indices,
            prefer_sparse=prefer_sparse_backprojection,
        )
        sub = extract_sampled_backprojection_np(back, indices, tuple(volume.shape))
        # Normalise so that ‖T_i‖ = 1; protects R from scale drift.
        nrm = float(np.linalg.norm(sub))
        if nrm > 0:
            sub = sub / nrm
        T_sub[i] = sub
        used_sparse_any = used_sparse_any or used_sparse
        times_per_view.append(time.time() - t0_view)
        
        if show_progress and (i + 1) % 20 == 0:
            print(f"  VCL: {i + 1}/{k_max} candidate bases computed "
                  f"({times_per_view[-1]:.2f}s)")
    
    t_bases = time.time() - t0_bases
    
    # γ[i] = T_i^T (x_sub / ‖x_sub‖)
    t0_gamma = time.time()
    x_sub_normalized = x_sub / x_norm
    gamma = T_sub @ x_sub_normalized
    t_gamma = time.time() - t0_gamma
    
    # R[i, j] = T_i^T T_j (already-normalised so diag(R) = 1)
    t0_R = time.time()
    R = T_sub @ T_sub.T
    t_R = time.time() - t0_R
    
    t_total = time.time() - t_start
    
    # Convert to float32
    R_final = R.astype(np.float32)
    gamma_final = gamma.astype(np.float32)
    
    # === DIAGNOSTICS ===
    if show_progress:
        def mb(arr):
            return arr.nbytes / (1024 ** 2)
        
        print("\n[VCL CACHE CONSTRUCTION DIAGNOSTICS]")
        print(f"Volume shape: {volume.shape}")
        print(f"Total voxels: {int(np.prod(volume.shape)):,}")
        print(f"ROI voxels: {n_roi:,}")
        print(f"Sampling rate r1: {r1}")
        print(f"Sampled voxels Ns: {indices.size:,}")
        print(f"Candidate views K_max: {k_max}")
        print(f"Data type: {R_final.dtype}")
        print(f"Sparse backprojection requested: {prefer_sparse_backprojection}")
        print(f"Sparse backprojection mode: {sparse_mode or 'dense fallback only'}")
        print(f"Sparse backprojection used: {used_sparse_any}")
        
        print(f"\nMemory:")
        print(f"  sample_indices: {mb(indices):.2f} MB")
        print(f"  T_sub (buffer): {mb(T_sub):.2f} MB")
        print(f"  R: {mb(R_final):.2f} MB")
        print(f"  gamma: {mb(gamma_final):.2f} MB")
        
        print(f"\nTiming:")
        print(f"  Sample indices: {t_sample*1000:.1f}ms")
        print(f"  Extract x_sub: {t_extract*1000:.1f}ms")
        print(f"  Basis computation (total): {t_bases:.2f}s")
        print(f"  Basis computation (per-view mean): {np.mean(times_per_view)*1000:.1f}ms")
        print(f"  R computation: {t_R*1000:.1f}ms")
        print(f"  gamma computation: {t_gamma*1000:.1f}ms")
        print(f"  Total: {t_total:.2f}s")
        
        # Numerical validation
        sym_err = np.linalg.norm(R_final - R_final.T) / max(np.linalg.norm(R_final), 1e-12)
        diag = np.diag(R_final)
        
        print(f"\nNumerical checks:")
        print(f"  R finite: {np.all(np.isfinite(R_final))}")
        print(f"  gamma finite: {np.all(np.isfinite(gamma_final))}")
        print(f"  R symmetry error (relative): {sym_err:.2e}")
        print(f"  R diagonal [min, mean, max]: [{np.min(diag):.6f}, {np.mean(diag):.6f}, {np.max(diag):.6f}]")
        print(f"  gamma [min, max]: [{np.min(gamma_final):.6f}, {np.max(gamma_final):.6f}]")
        print()

    return VCLPrecompute(
        R=R_final,
        gamma=gamma_final,
        sample_indices=indices,
        candidate_sources=candidate_sources,
        volume_shape=volume.shape,
        r1=r1,
        used_sparse_backprojection=used_sparse_any,
        sparse_backprojection_mode=sparse_mode if used_sparse_any else None,
    )


# ---------------------------------------------------------------------------
# VCL loss — Algorithm 3
# ---------------------------------------------------------------------------

def vcl_loss(
    R: np.ndarray,
    gamma: np.ndarray,
    indices: Sequence[int] | np.ndarray,
    *,
    ridge: float = 1e-6,
) -> float:
    """Algorithm 3: L = 1 - γ_Ω^T (R_Ω + ridge·I)^{-1} γ_Ω.

    ``ridge`` is a small Tikhonov term that stabilises the inverse on
    near-singular sub-matrices; the paper does not specify a regulariser
    but the Fisher-information derivation assumes invertibility.
    """
    idx = np.asarray(list(indices), dtype=np.int64)
    g = gamma[idx]
    sub = R[np.ix_(idx, idx)]
    sub = sub + ridge * np.eye(sub.shape[0], dtype=sub.dtype)
    try:
        sol = np.linalg.solve(sub, g)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(sub, g, rcond=None)[0]
    return float(1.0 - g @ sol)


# ---------------------------------------------------------------------------
# View subset selection — Algorithm 2 (random-swap search)
# ---------------------------------------------------------------------------

def vcls_select(
    precompute: VCLPrecompute,
    K: int,
    *,
    r2: float = 0.1,
    max_outer_passes: int = 20,
    seed: int = 0,
    ridge: float = 1e-6,
    verbose: bool = False,
) -> tuple[list[int], float]:
    """Greedy random-swap search minimising the VCL.

    Implements Algorithm 2 of Lin et al. 2025: initialise with uniformly-spaced
    candidates, then repeatedly attempt swaps with random subsets of the
    complement (size ``r2 · |Ω - Ω*|``) until a full sweep produces no
    improvement or ``max_outer_passes`` is reached.

    Returns the index list ``Ω*`` and the achieved loss.
    """
    k_max = precompute.k_max
    if K > k_max:
        raise ValueError(f"K={K} exceeds candidate count {k_max}.")

    rng = np.random.default_rng(seed)

    # Uniform initialisation
    selected = sorted(np.linspace(0, k_max - 1, K, dtype=np.int64).tolist())
    best_loss = vcl_loss(precompute.R, precompute.gamma, selected, ridge=ridge)
    if verbose:
        print(f"  VCLS init loss: {best_loss:.6f}")

    for outer in range(max_outer_passes):
        improved_this_pass = False
        # Iterate over a randomised order so we don't bias to small indices.
        order = list(selected)
        rng.shuffle(order)
        for theta_i in order:
            if theta_i not in selected:
                # Was swapped out earlier in this pass; skip.
                continue
            outside = [c for c in range(k_max) if c not in selected]
            if not outside:
                break
            n_try = max(1, int(round(r2 * len(outside))))
            candidates_to_try = rng.choice(outside, size=n_try, replace=False)
            for theta_j in candidates_to_try:
                trial = [c for c in selected if c != theta_i] + [int(theta_j)]
                loss_trial = vcl_loss(precompute.R, precompute.gamma, trial,
                                      ridge=ridge)
                if loss_trial < best_loss - 1e-9:
                    selected = sorted(trial)
                    best_loss = loss_trial
                    improved_this_pass = True
                    break
        if verbose:
            print(f"  VCLS pass {outer+1}: loss={best_loss:.6f}  "
                  f"improved={improved_this_pass}")
        if not improved_this_pass:
            break

    return sorted(selected), best_loss


# ---------------------------------------------------------------------------
# Top-level baseline
# ---------------------------------------------------------------------------

def vcls_sources(
    K: int,
    sid: float,
    *,
    candidate_sources: mx.array,
    volume: mx.array,
    sdd: float,
    det_shape: tuple[int, int],
    du: float = 1.0,
    dv: float = 1.0,
    voxel_spacing: float = 1.0,
    r1: float = 1e-3,
    r2: float = 0.1,
    seed: int = 0,
    ridge: float = 1e-6,
    show_progress: bool = False,
    roi_mask: np.ndarray | None = None,
    prefer_sparse_backprojection: bool = True,
) -> mx.array:
    """End-to-end VCLS baseline: precompute (R, γ), run swap search, return
    the ``(K, 3)`` source positions of the selected views.
    
    Parameters
    ----------
    roi_mask
        Optional boolean mask. If provided, samples only from ROI during
        precomputation (see compute_R_gamma).
    prefer_sparse_backprojection
        If True, prefer sparse/sample-only backprojection when supported by
        the installed ``diffct_mlx`` version.
    """
    pre = compute_R_gamma(
        volume, candidate_sources,
        sid=sid, sdd=sdd, det_shape=det_shape,
        du=du, dv=dv, voxel_spacing=voxel_spacing,
        r1=r1, seed=seed, show_progress=show_progress,
        roi_mask=roi_mask,
        prefer_sparse_backprojection=prefer_sparse_backprojection,
    )
    indices, _final_loss = vcls_select(
        pre, K, r2=r2, seed=seed, ridge=ridge, verbose=show_progress,
    )
    return candidate_sources[mx.array(indices)]
