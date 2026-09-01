"""Simulate sinograms and reconstruct from arbitrary source positions.

Two-step pipeline:

1. ``simulate_sinogram(volume, src_pos, det_*, ...)`` runs the diffct_mlx
   cone-beam *footprint* forward projector to produce a noise-free sinogram.
2. ``reconstruct_sart_volume(volume_shape, sinogram, src_pos, det_*, ...)``
   reconstructs via SART using the matching footprint forward/back-project
   operators.

We expose only SART because it is the chosen algorithm for Paper 1's reco
quality evaluation.  Switching to FDK/SIRT/TV-POCS is a one-line change in
the runner once needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from diffct_mlx import (
    SARTParameters,
    cone_forward_footprint,
    make_cone_3d_operators,
    reconstruct_sart,
)

from .._torch_bridge import to_backend as _to_backend, from_backend as _from_backend


def apply_transmission_noise(
    sinogram: mx.array,
    photon_count: float,
    *,
    seed: int = 0,
) -> mx.array:
    """Add Beer-Lambert / Poisson transmission noise to a line-integral sinogram.

    The forward projector returns optical depth ``p = ∫ μ dℓ`` (dimensionless).
    A monochromatic source emitting ``I0 = photon_count`` photons per detector
    pixel measures ``I = I0 · exp(-p)`` expected photons, corrupted by Poisson
    counting statistics.  The log-transform back to projection domain gives the
    noisy line integral

        p̃ = -log( max(Poisson(I0 · e^{-p}), 1) / I0 ).

    The projection-domain variance is ``Var(p̃) ≈ e^{p} / I0``, i.e. rays through
    dense material (large ``p``, photon starvation) are far noisier -- exactly
    the effect a noise-aware view selector should account for.

    This is an evaluation-only measurement model (no autograd path), so the
    Poisson draw is done in NumPy for an exact (not Gaussian-approximated) sample.
    """
    p = np.asarray(sinogram, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    expected = float(photon_count) * np.exp(-p)
    measured = rng.poisson(expected).astype(np.float64)
    measured = np.maximum(measured, 1.0)            # avoid log(0) at starved rays
    p_noisy = -np.log(measured / float(photon_count))
    return mx.array(p_noisy.astype(np.float32))


def apply_polychromatic_transmission_noise(
    sinogram: mx.array,
    photon_count: float,
    *,
    seed: int = 0,
    weights: tuple[float, ...] = (0.35, 0.45, 0.20),
    mu_scales: tuple[float, ...] = (1.6, 1.0, 0.6),
) -> mx.array:
    """Beer-Lambert / Poisson noise under a discrete surrogate spectrum.

    Generalises :func:`apply_transmission_noise` from a monochromatic
    source to a ``B``-bin surrogate spectrum.  Bin ``b`` carries a
    fraction ``w_b`` of the ``I0 = photon_count`` photons and sees the
    line integral scaled by ``s_b`` (energy dependence of ``mu``), so the
    expected count is

        I = I0 · Σ_b w_b · exp(-s_b · p),

    corrupted by a single Poisson draw and log-transformed back with the
    monochromatic normalisation ``p̃ = -log(I / I0)``.  Because low-energy
    bins (``s_b > 1``) are preferentially absorbed on long paths, the
    effective attenuation saturates with path length — the beam-hardening
    behaviour absent from the monochromatic model.  The defaults are a
    generic three-bin surrogate for a tungsten spectrum, soft/reference/
    hard, and are not calibrated to a specific tube voltage.

    Evaluation-only measurement model, as in
    :func:`apply_transmission_noise`; selection always plans on the
    monochromatic reference volume.
    """
    p = np.asarray(sinogram, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(mu_scales, dtype=np.float64)
    if w.shape != s.shape or w.ndim != 1 or w.size == 0:
        raise ValueError("weights and mu_scales must be equal-length 1-D sequences")
    w = w / w.sum()
    rng = np.random.default_rng(int(seed))
    expected = np.zeros_like(p)
    for wb, sb in zip(w, s):
        expected += wb * np.exp(-sb * p)
    expected *= float(photon_count)
    measured = rng.poisson(expected).astype(np.float64)
    measured = np.maximum(measured, 1.0)            # avoid log(0) at starved rays
    p_noisy = -np.log(measured / float(photon_count))
    return mx.array(p_noisy.astype(np.float32))


@dataclass(frozen=True)
class ReconResult:
    """Container for a single reconstruction outcome."""

    reconstruction: mx.array
    sinogram_shape: tuple[int, int, int]
    n_views: int
    iteration_count: int


def simulate_sinogram(
    volume: mx.array,
    src_pos: mx.array,
    det_center: mx.array,
    det_u_vec: mx.array,
    det_v_vec: mx.array,
    *,
    det_u: int,
    det_v: int,
    du: float = 1.0,
    dv: float = 1.0,
    voxel_spacing: float = 1.0,
    photon_count: float | None = None,
    noise_seed: int = 0,
    spectrum: dict | None = None,
) -> mx.array:
    """Run the footprint cone-beam forward projector.

    Returns a sinogram of shape ``(n_views, det_u, det_v)`` matching the
    diffct_mlx convention.  When ``photon_count`` is given, Beer-Lambert /
    Poisson transmission noise is added (see :func:`apply_transmission_noise`);
    the default ``None`` keeps the historical noise-free behaviour.  When
    additionally ``spectrum`` is given as ``{"weights": [...], "mu_scales":
    [...]}``, the polychromatic surrogate model of
    :func:`apply_polychromatic_transmission_noise` is used instead.
    """
    sino = cone_forward_footprint(
        _to_backend(volume),
        _to_backend(src_pos),
        _to_backend(det_center),
        _to_backend(det_u_vec),
        _to_backend(det_v_vec),
        det_u=det_u,
        det_v=det_v,
        du=du,
        dv=dv,
        voxel_spacing=voxel_spacing,
    )
    sino = _from_backend(sino)
    if photon_count is not None:
        mx.eval(sino)
        if spectrum is not None:
            sino = apply_polychromatic_transmission_noise(
                sino, photon_count, seed=noise_seed,
                weights=tuple(spectrum["weights"]),
                mu_scales=tuple(spectrum["mu_scales"]),
            )
        else:
            sino = apply_transmission_noise(sino, photon_count, seed=noise_seed)
    return sino


def reconstruct_sart_volume(
    volume_shape: tuple[int, int, int],
    sinogram: mx.array,
    src_pos: mx.array,
    det_center: mx.array,
    det_u_vec: mx.array,
    det_v_vec: mx.array,
    *,
    du: float = 1.0,
    dv: float = 1.0,
    voxel_spacing: float = 1.0,
    iteration_count: int = 10,
    relaxation: float = 0.9,
    projector_mode: str = "footprint",
    show_progress: bool = False,
) -> ReconResult:
    """Reconstruct ``volume_shape`` from ``sinogram`` via SART.

    The forward / back operators are built from the explicit source and
    detector arrays so that *arbitrary* (non-circular, optimized) source
    trajectories work without modification.
    """
    n_views = int(src_pos.shape[0])
    det_u_count = int(sinogram.shape[1])
    det_v_count = int(sinogram.shape[2])

    # On the torch backend the operators and SART solver need backend-native
    # tensors; convert once here (identity on the mlx/Apple path).
    sinogram_b = _to_backend(sinogram)
    src_pos_b = _to_backend(src_pos)
    det_center_b = _to_backend(det_center)
    det_u_vec_b = _to_backend(det_u_vec)
    det_v_vec_b = _to_backend(det_v_vec)

    # SART expects a Sequence of per-view projections, not a stacked array.
    measured_projections = [sinogram_b[i] for i in range(n_views)]

    forward_single, back_single, _ = make_cone_3d_operators(
        src_pos_b,
        det_center_b,
        det_u_vec_b,
        det_v_vec_b,
        volume_shape=volume_shape,
        detector_shape=(det_u_count, det_v_count),
        du=du,
        dv=dv,
        voxel_spacing=voxel_spacing,
        projector_mode=projector_mode,
    )

    params = SARTParameters(
        volume_shape=volume_shape,
        iteration_count=iteration_count,
        sart_iteration_count=1,
        normalized_sart_relaxation=relaxation,
        enforce_positivity=True,
        shuffle_projection_order=True,
        projection_order_seed=0,
    )

    recon = reconstruct_sart(
        measured_projections,
        forward_single,
        back_single,
        params,
        show_progress=show_progress,
    )
    recon = _from_backend(recon)

    return ReconResult(
        reconstruction=recon,
        sinogram_shape=tuple(sinogram.shape),
        n_views=n_views,
        iteration_count=iteration_count,
    )
