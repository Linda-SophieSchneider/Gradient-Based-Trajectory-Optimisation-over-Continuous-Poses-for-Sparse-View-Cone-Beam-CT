"""Baseline trajectories for reconstruction-quality evaluation.

Every builder returns a ``(k, 3)`` ``mx.array`` of source positions on (or
near) a sphere of radius ``sid`` around the origin.  Detector geometry is
recovered downstream by :func:`differentiable_coverage.eval.geometry_from_sources`.

Baselines
---------
* ``uniform_arc`` — k sources evenly spaced on a full-circle orbit in the xy plane.
* ``random_sphere`` — k Fibonacci-distributed sources on the full sphere.
* ``greedy_discrete`` — discrete greedy selection of k sources from a dense
  candidate grid (200 Fibonacci samples by default), no gradient refinement.
* ``greedy_adam`` — ``greedy_discrete`` warm-start refined by ``T`` Adam steps
  on the differentiable-coverage objective (this is the paper's "Method").

The last two share the same warm-start RNG so that "discrete vs. optimized"
isolates the value of continuous refinement.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..absorption import AbsorptionConfig, compute_absorption_gate
from ..absorption_bundle import calibrate_bundle_weight
from .._torch_bridge import to_backend as _to_backend, from_backend as _from_backend
from ..landscape import random_sphere_sources
from ..optimize import adam_ascent
from ..score import (
    ScoreConfig,
    coverage_covariance_information,
    greedy_source_init,
    saturated_coverage,
    sample_unit_sphere,
)
from ..trajectory import (CArmTwoAxisGantry, CircularArc, Free3D,
                          SmoothTwoAxisGantry, TwoAxisGantry)


BASELINE_NAMES = (
    "uniform_arc",
    "random_sphere",
    "greedy_discrete",
    "vcls",                       # VCLS on sphere (our generalisation)
    "vcls_circle",                # VCLS on 200 circle candidates (Lin et al. setup)
    "greedy_adam",                # geometry-only sphere Adam refinement
    "greedy_adam_circle",         # geometry-only circle Adam refinement
    "greedy_adam_sg",             # + absorption (stop-gradient, fraction gate)
    "greedy_adam_fd",             # + absorption (finite-difference VJP, fraction gate)
    "greedy_adam_sg_icov",        # + absorption + uniform-w Icov
    "greedy_adam_sg_icov_obj",    # + absorption + object-aware Icov (mean gate)
    "greedy_adam_path",           # + path-length gate (multiplicative, kills long paths)
    "greedy_adam_path_icov",      # + path-length gate + Icov (multiplicative)
    "greedy_adam_path_div",       # diversity-first additive: cov + λ_path · mean(ν_path)
    "greedy_adam_path_div_icov",  # same + Icov
    "greedy_adam_path_div_circle",  # circle-restricted diversity-first path-div
    "greedy_adam_vcl",            # cov + λ_vcl · I_vcl  (continuous gradient VCL)
    "greedy_adam_vcl_pure",       # pure I_vcl, no coverage prior
    "greedy_adam_all",            # cov + λ_path · ν + λ_vcl · I_vcl  (kitchen sink)
    "vcls_adam_vcl",              # VCLS warm-start + Adam on cov+VCL
    "vcls_adam_geo",              # VCLS warm-start + Adam on coverage only
    "multistart_adam",            # 5-start (greedy+vcls+3·random) → best surrogate
    "vcls_adam_anneal",           # VCLS + Adam with σ-annealing on Tuy kernel
    "vcls_adam_langevin",         # VCLS + Riemannian SGLD refinement on S²
    "vcls_adam_ensemble",         # VCLS + repulsive ensemble (SVGD-style) refinement
    "greedy_adam_vcl_anneal",     # cold cov+VCL + σ-annealing
    "greedy_adam_vcl_langevin",   # cold cov+VCL + Riemannian SGLD
    "greedy_adam_vcl_ensemble",   # cold cov+VCL + repulsive ensemble
    "vcls_adam_bundle_center",    # VCLS + analytic 1-ray bundle absorption penalty
    "vcls_adam_bundle",           # VCLS + analytic 5x9-ray bundle absorption penalty
    "greedy_adam_bundle_center",  # greedy_tuy + analytic 1-ray bundle absorption
    "greedy_adam_bundle",         # greedy_tuy + analytic 5x9-ray bundle absorption
    "greedy_adam_composite",      # cold cov + VCL + analytic 5x9-ray bundle
    "uniform_adam_bundle",        # uniform sphere + analytic 5x9-ray bundle absorption
    "greedy_adam_vcl_two_axis",   # free-sphere VCL objective on a 2-axis gantry
    "vcls_adam_vcl_two_axis",     # VCLS warm start + 2-axis gantry
    "greedy_adam_bundle_two_axis",  # historical ID: cov+VCL+bundle on 2 axes
    "vcls_adam_bundle_two_axis",    # historical ID: warm cov+VCL+bundle on 2 axes
    "greedy_adam_vcl_carm",       # VCL objective on a realistic limited C-arm
    "vcls_adam_vcl_carm",         # VCLS warm start on a realistic limited C-arm
    "greedy_adam_bundle_carm",    # historical ID: cov+VCL+bundle on limited C-arm
    "vcls_adam_bundle_carm",      # historical ID: warm cov+VCL+bundle on limited C-arm
    "greedy_adam_icov_fft",       # greedy_tuy + soft Icov with Fourier-slice γ
                                  # (volume-aware; no SART precomputation)
    "vcls_adam_icov_fft",         # VCLS init + soft Icov with Fourier-slice γ + bundle
    "greedy_adam_oed",            # greedy_tuy + photon-noise-weighted A+D OED score
    "vcls_adam_oed",              # VCLS init + photon-noise-weighted A+D OED score
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Per-step angular displacement of the continuous selectors, in radians.
# See the chart-consistency note in build_baseline_sources: this is the
# learning rate of the angular (gantry) charts and, scaled by sid, of the
# Cartesian free-sphere chart.
ANGULAR_LR = 0.05


def _default_score_config(half_angle_deg: float = 15.0) -> ScoreConfig:
    tau = math.sin(math.radians(half_angle_deg))
    return ScoreConfig(tau=tau)


def _matched_vcl_context(
    volume: mx.array,
    *,
    sid: float,
    sdd: float,
    target_shape: tuple[int, int, int],
    r1: float,
    detector_shape: tuple[int, int],
    working_detector_shape: tuple[int, int],
    du: float,
    dv: float,
    voxel_spacing: float,
    sample_seed: int,
    ridge: float,
    geometry_vjp_mode: str,
    geometry_fd_step: float,
    prefer_sparse_backprojection: bool,
):
    """Build a VCL context that preserves the acquisition's physical extents."""
    from ..vcl_diff import build_vcl_context
    from ..vcl_geometry import resolve_vcl_working_geometry

    working = resolve_vcl_working_geometry(
        volume_shape=tuple(int(v) for v in volume.shape),
        target_shape=target_shape,
        detector_shape=detector_shape,
        working_detector_shape=working_detector_shape,
        du=du,
        dv=dv,
        voxel_spacing=voxel_spacing,
    )
    return build_vcl_context(
        volume,
        sid=sid,
        sdd=sdd,
        det_shape=working.detector_shape,
        du=working.du,
        dv=working.dv,
        voxel_spacing=working.voxel_spacing,
        target_shape=working.volume_shape,
        r1=r1,
        seed=sample_seed,
        ridge=ridge,
        prefer_sparse_backprojection=prefer_sparse_backprojection,
        geometry_vjp_mode=geometry_vjp_mode,
        geometry_fd_step=geometry_fd_step,
    )


def _fibonacci_candidates(n: int, sid: float, *, seed: int | None = None) -> mx.array:
    """Fibonacci-lattice candidate sources on the sphere of radius ``sid``.

    If ``seed`` is given, a seeded SO(3) rotation is applied to the
    lattice (see :func:`sample_unit_sphere`) so that different seeds
    produce different but still quasi-uniform candidate sets.
    """
    return sample_unit_sphere(n, seed=seed) * sid


def _circle_candidates(n: int, sid: float) -> mx.array:
    """``n`` uniformly-spaced source positions on the equatorial circle.

    This is the candidate set used by Lin et al.~\\cite{LinVCL2025} —
    single-axis rotation in the xy-plane.  Use for circle-restricted
    baselines (``vcls_circle`` / ``greedy_adam_circle``).
    """
    angles = mx.arange(n, dtype=mx.float32) * (2.0 * math.pi / n)
    sx = -sid * mx.sin(angles)
    sy = sid * mx.cos(angles)
    sz = mx.zeros_like(sx)
    return mx.stack([sx, sy, sz], axis=-1)


def _sanitize_trace_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tag)


def _write_optim_trace(
    trace_dir: str,
    trace_tag: str,
    rows: list[dict],
    sources_hist: list[np.ndarray],
) -> tuple[Path, Path]:
    """Write per-step optimisation traces for offline debugging."""
    out_dir = Path(trace_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_trace_tag(trace_tag)
    metrics_path = out_dir / f"{stem}_metrics.csv"
    sources_path = out_dir / f"{stem}_sources.npz"

    fieldnames = list(rows[0].keys())
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        sources_path,
        sources=np.stack(sources_hist, axis=0),
        step=np.array([row["step"] for row in rows], dtype=np.int32),
        phase=np.array([row["phase"] for row in rows], dtype=object),
    )
    return metrics_path, sources_path


# ---------------------------------------------------------------------------
# Surrogate term recording
# ---------------------------------------------------------------------------
# Every result CSV in this repository records psnr/ssim/nrmse/hfen but not the
# objective value that produced the pose set, so a surrogate-against-image-
# quality pair could never be built from an archived run.  The two continuous
# drivers therefore publish the terms of the iterate they return here, and
# ``experiments/run.py`` appends them to its result rows.
#
# The recording reuses the term arrays the optimiser already evaluated (see
# :class:`_TermRecorder`), so it adds no objective evaluation to the
# optimisation loop, draws no random number and touches no parameter.

_LAST_OPTIM_TERMS: dict = {}


def reset_optim_terms() -> None:
    """Drop the terms of the previous selection run.

    Call before a selection so a selector that evaluates no continuous
    objective (``vcls``, ``greedy_discrete``, ...) reports no terms rather
    than the previous cell's.
    """
    _LAST_OPTIM_TERMS.clear()


def last_optim_terms() -> dict:
    """Surrogate terms of the most recent continuous selection run.

    Keys are ``objective`` and the individual terms (``cov``, ``tau_mean``,
    ``vcl_info``, ...) plus the ``step`` they were evaluated at, all taken
    from the evaluation the optimiser itself performed on the iterate it
    returned.  Empty when the last selector ran no continuous objective.
    """
    return dict(_LAST_OPTIM_TERMS)


def _publish_optim_terms(row: dict | None) -> None:
    reset_optim_terms()
    if not row:
        return
    # ``objective`` first, bookkeeping last: this order becomes the column
    # order of the result CSV's appended block.
    _LAST_OPTIM_TERMS.update(
        {name: value for name, value in row.items()
         if name not in ("phase", "step")}
    )
    _LAST_OPTIM_TERMS["step"] = row.get("step")


def _terms_recording_enabled() -> bool:
    """Whether the drivers record their own term values (default: yes).

    ``DIFFCT_RECORD_TERMS=0`` restores the pre-E0 code path exactly: no
    ``adam_ascent`` callback, no term registry, no extra CSV columns.
    """
    return os.environ.get("DIFFCT_RECORD_TERMS", "1").lower() not in (
        "0", "false", "no",
    )


class _TermRecorder:
    """Collects the surrogate terms the optimiser itself evaluated.

    ``evaluate_terms`` hands its result to :meth:`offer`, which keeps it only
    while the recorder is armed.  The recorder is armed once before each Adam
    step, so the evaluation it keeps is always the base-iterate one; the
    perturbed calls of ``derivative_mode="fd_central"`` all arrive disarmed.
    :meth:`on_step` then reads that already-computed evaluation instead of
    calling the objective a second time, which is what makes the per-step
    trace free and keeps it provably unable to change the optimisation.

    ``keep_sources`` is only needed for the ``_sources.npz`` trace companion;
    with terms recording alone the per-step device-to-host copy is skipped.
    """

    def __init__(self, evaluate_terms, *, keep_sources: bool) -> None:
        self._evaluate = evaluate_terms
        self._keep_sources = keep_sources
        self._armed = False
        self._held = None
        self.rows: list[dict] = []
        self.sources: list[np.ndarray] = []
        self._params: list = []

    def offer(self, evaluated) -> None:
        if self._armed:
            self._armed = False
            self._held = evaluated

    def arm(self) -> None:
        self._armed = True
        self._held = None

    def record(self, phase: str, step_idx: int, params: mx.array,
               schedule_step: int, evaluated=None) -> dict:
        if evaluated is None:
            # Only reached off the optimisation loop (the ``init`` row and the
            # CMA-ES fallback), never per Adam step.
            evaluated = self._evaluate(params, schedule_step)
        sources, terms = evaluated
        mx.eval(sources, *terms.values())
        row = {"phase": phase, "step": step_idx}
        row.update({name: float(value) for name, value in terms.items()})
        self.rows.append(row)
        self._params.append(params)
        if self._keep_sources:
            self.sources.append(np.asarray(sources))
        return row

    def on_step(self, step: int, params: mx.array, _value: float) -> None:
        """``adam_ascent`` callback: record this step's own evaluation."""
        held, self._held = self._held, None
        self.record("iter", step, params, step, evaluated=held)
        self.arm()

    def final(self, params: mx.array, schedule_step: int) -> dict:
        """Row of the iterate the optimiser returned.

        ``adam_ascent(return_best=True)`` returns one of the arrays it passed
        to the callback, so the row is found by identity and no further
        objective evaluation is needed.  The fallback covers the CMA-ES arm,
        which takes no callback.
        """
        for idx, evaluated_params in enumerate(self._params):
            if evaluated_params is params:
                row = dict(self.rows[idx])
                row["phase"] = "best"
                self.rows.append(row)
                self._params.append(params)
                if self._keep_sources:
                    self.sources.append(self.sources[idx])
                return row
        return self.record("best", schedule_step, params, schedule_step)


def _sources_to_two_axis_params(
    sources: mx.array, sid: float,
) -> mx.array:
    """Convert source positions into ``(theta, phi)`` gantry parameters."""
    theta = mx.arctan2(-sources[:, 0], sources[:, 1])
    phi = mx.arcsin(mx.clip(sources[:, 2] / max(float(sid), 1e-6), -1.0, 1.0))
    return mx.stack([theta, phi], axis=-1)


def _sources_to_carm_params(
    sources: mx.array, sid: float,
) -> mx.array:
    gantry = CArmTwoAxisGantry(sid=sid)
    return gantry.clamp(_sources_to_two_axis_params(sources, sid))


def greedy_adam_path_div_circle(
    k: int,
    sid: float,
    *,
    roi_center: mx.array,
    volume: mx.array,
    sdd: float = 900.0,
    n_candidates: int = 200,
    n_normals: int = 2000,   # protocol value (2026-08-27): the 15 deg
                             # coverage bandwidth tiles the sphere with
                             # ~58 caps, so z must give many samples per
                             # cap; z=300 left ~5 and its sampling noise
                             # cost up to 3 dB on the flange.
    n_steps: int = 100,
    lr: float = 0.05,
    lambda_path: float = 0.3,
    score_cfg: ScoreConfig | None = None,
    roi_points: mx.array | None = None,
    roi_weights: mx.array | None = None,
) -> mx.array:
    """Diversity-first path-bonus refinement **restricted to a circular orbit**.

    Circle-restricted analogue of :func:`greedy_adam_path_div`.  Greedy
    selects ``k`` angles from 200 candidates on the xy-plane circle, then
    Adam refines the angles through :class:`CircularArc` on the objective

        L = C̃_geo(θ) + λ_path · mean(ν_path(θ))

    where ν is the differentiable absorption gate via fd_src.  Comparable to
    Lin et al.'s VCLS-circle setup but using our continuous-on-orbit
    framework instead of discrete swap search.
    """
    cfg = score_cfg or _default_score_config()
    radon_normals = sample_unit_sphere(n_normals)
    candidates = _circle_candidates(n_candidates, sid)
    selected = greedy_source_init(
        candidates, roi_center, radon_normals, k, cfg,
        roi_points=roi_points, roi_weights=roi_weights,
    )

    sel_np = np.asarray(selected, dtype=np.float32)
    theta0 = np.arctan2(-sel_np[:, 0], sel_np[:, 1])
    theta = mx.array(theta0, dtype=mx.float32)

    sigma = cfg.gaussian_sigma()
    arc = CircularArc(sid=sid)
    abs_cfg = _build_absorption_config(sid, sdd, volume=volume, gate_type="path")

    def cov_fn(params, _step):
        sources = arc(params)
        cov = saturated_coverage(
            sources, roi_center, radon_normals,
            mx.ones(sources.shape[0], dtype=mx.float32),
            cfg,
            roi_points=roi_points,
            roi_weights=roi_weights,
        )
        # Additive short-path bonus
        nu = compute_absorption_gate(
            sources, roi_center, volume, abs_cfg,
            grad_mode="fd_src", gate_type="path",
        )
        return cov + lambda_path * mx.mean(nu)

    refined_theta, _ = adam_ascent(
        cov_fn, theta, n_steps=n_steps, lr=lr,
        lr_schedule="cosine", lr_min=lr * 0.05,
        patience=15, return_best=True,
    )
    return arc(refined_theta)


def greedy_adam_circle(
    k: int,
    sid: float,
    *,
    roi_center: mx.array,
    n_candidates: int = 200,
    n_normals: int = 2000,   # protocol value (2026-08-27): the 15 deg
                             # coverage bandwidth tiles the sphere with
                             # ~58 caps, so z must give many samples per
                             # cap; z=300 left ~5 and its sampling noise
                             # cost up to 3 dB on the flange.
    n_steps: int = 100,
    lr: float = 0.05,
    score_cfg: ScoreConfig | None = None,
    roi_points: mx.array | None = None,
    roi_weights: mx.array | None = None,
) -> mx.array:
    """Greedy warm-start + Adam refinement **restricted to a circular orbit**.

    Builds 200 candidates on the equatorial circle, runs the discrete
    greedy selector on them, converts the chosen positions to scan angles,
    then refines the angles with Adam through the
    :class:`differentiable_coverage.trajectory.CircularArc` parametrisation.
    """
    cfg = score_cfg or _default_score_config()
    radon_normals = sample_unit_sphere(n_normals)
    candidates = _circle_candidates(n_candidates, sid)
    selected = greedy_source_init(
        candidates, roi_center, radon_normals, k, cfg,
        roi_points=roi_points, roi_weights=roi_weights,
    )

    # Recover the per-source angle from the (x, y) projection.
    sel_np = np.asarray(selected, dtype=np.float32)
    theta0 = np.arctan2(-sel_np[:, 0], sel_np[:, 1])  # matches CircularArc convention
    theta = mx.array(theta0, dtype=mx.float32)

    sigma = cfg.gaussian_sigma()
    arc = CircularArc(sid=sid)

    def cov_fn(params, _step):
        sources = arc(params)
        return saturated_coverage(
            sources, roi_center, radon_normals,
            mx.ones(sources.shape[0], dtype=mx.float32),
            cfg,
            roi_points=roi_points,
            roi_weights=roi_weights,
        )

    refined_theta, _ = adam_ascent(cov_fn, theta, n_steps=n_steps, lr=lr)
    return arc(refined_theta)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def uniform_arc(k: int, sid: float) -> mx.array:
    """k evenly-spaced sources on a full-circle xy-plane orbit."""
    angles = mx.arange(k, dtype=mx.float32) * (2.0 * math.pi / k)
    sx = -sid * mx.sin(angles)
    sy = sid * mx.cos(angles)
    sz = mx.zeros_like(sx)
    return mx.stack([sx, sy, sz], axis=-1)


def random_sphere(k: int, sid: float, *, seed: int = 0) -> mx.array:
    """k random sources on the sphere of radius ``sid`` (jittered Fibonacci)."""
    return random_sphere_sources(k, sid, seed=seed)


def greedy_discrete(
    k: int,
    sid: float,
    *,
    roi_center: mx.array,
    n_candidates: int = 200,
    score_cfg: ScoreConfig | None = None,
    n_normals: int = 2000,   # protocol value (2026-08-27): the 15 deg
                             # coverage bandwidth tiles the sphere with
                             # ~58 caps, so z must give many samples per
                             # cap; z=300 left ~5 and its sampling noise
                             # cost up to 3 dB on the flange.
    sphere_seed: int | None = None,
    roi_points: mx.array | None = None,
    roi_weights: mx.array | None = None,
) -> mx.array:
    """Greedy selection of k sources from ``n_candidates`` Fibonacci samples.

    ``sphere_seed`` seeds the SO(3) rotation of the candidate lattice
    (see :func:`_fibonacci_candidates`); ``None`` reproduces the
    deterministic original behaviour.
    """
    cfg = score_cfg or _default_score_config()
    candidates = _fibonacci_candidates(n_candidates, sid, seed=sphere_seed)
    radon_normals = sample_unit_sphere(n_normals)
    selected = greedy_source_init(
        candidates, roi_center, radon_normals, k, cfg,
        roi_points=roi_points, roi_weights=roi_weights,
    )
    return selected


def _roi_footprint_distribution(
    quantity: mx.array,
    sid: float,
    sdd: float,
    det_shape: tuple[int, int] = (128, 128),
    du: float = 2.0,
    dv: float = 2.0,
    voxel_spacing: float = 1.0,
    n_probe: int = 16,
) -> np.ndarray:
    """Per-view ROI-footprint-weighted means for a probe set of sources.

    Used by the path-length / mean-attenuation calibrations so that ``α``
    and ``β`` see exactly the same statistic the gate is going to threshold.
    """
    from diffct_mlx import cone_forward_footprint
    from ..absorption import _soft_footprint
    from .geometry import geometry_from_sources as _geom

    sources = random_sphere_sources(n_probe, sid, seed=0)
    src_pos, det_c, det_u, det_v = _geom(sources, sid=sid, sdd=sdd)
    sino = _from_backend(cone_forward_footprint(
        _to_backend(quantity), _to_backend(src_pos), _to_backend(det_c),
        _to_backend(det_u), _to_backend(det_v),
        det_u=det_shape[0], det_v=det_shape[1],
        du=du, dv=dv, voxel_spacing=voxel_spacing,
    ))
    # Re-use the gate's own ROI footprint so the statistic matches exactly.
    probe_cfg = AbsorptionConfig(
        alpha=0.0, eta=0.0, sid=sid, sdd=sdd,
        det_u=det_shape[0], det_v=det_shape[1],
        du=du, dv=dv, voxel_spacing=voxel_spacing,
        roi_radius=48.0, beta_pixel=0.0, beta_frac=0.0,
    )
    weights = _soft_footprint(probe_cfg, sources, mx.array([0.0, 0.0, 0.0]))
    w_sum = mx.sum(weights, axis=(1, 2)) + 1e-9
    mean_per_view = mx.sum(weights * sino, axis=(1, 2)) / w_sum
    return np.asarray(mean_per_view, dtype=np.float32)


def _calibrate_path_gate(
    volume: mx.array,
    sid: float,
    sdd: float,
    *,
    det_shape: tuple[int, int] = (128, 128),
    du: float = 2.0,
    dv: float = 2.0,
    voxel_spacing: float = 1.0,
    mask_threshold_frac: float = 0.05,
    target_grad_scale: float = 1.5,
) -> tuple[float, float, float]:
    """Joint (α, β) calibration for the path-length gate.

    ``α`` is the median of per-view ROI-footprint-weighted mean path lengths
    so that half the probe views fall on each side of the sigmoid.  ``β`` is
    chosen so that ``β · std(L) ≈ target_grad_scale``: large enough that the
    gate moves meaningfully across views, small enough that it stays smooth
    rather than binary (target_grad_scale ≈ 1–2 gives a useful gradient).
    Returns ``(alpha, beta, mask_threshold)``.
    """
    threshold = mask_threshold_frac * float(mx.max(volume))
    mask = (volume > threshold).astype(mx.float32)
    distribution = _roi_footprint_distribution(
        mask, sid, sdd,
        det_shape=det_shape, du=du, dv=dv,
        voxel_spacing=voxel_spacing,
    )
    alpha = float(np.median(distribution))
    std = float(distribution.std())
    if std < 1e-6:
        beta = 1.0
    else:
        beta = target_grad_scale / std
    return alpha, beta, threshold


def _calibrate_alpha_mean(
    volume: mx.array,
    sid: float,
    sdd: float,
    det_shape: tuple[int, int] = (128, 128),
    du: float = 2.0,
    dv: float = 2.0,
    voxel_spacing: float = 1.0,
    n_probe: int = 8,
) -> float:
    """Calibration for the *mean*-attenuation gate: α = median of per-view
    ROI-footprint-weighted mean attenuations.  Puts the sigmoid right at the
    50/50 transition so neighbouring views fall on opposite sides."""
    import numpy as np
    from diffct_mlx import cone_forward_footprint
    from .geometry import geometry_from_sources

    sources = random_sphere_sources(n_probe, sid, seed=0)
    src_pos, det_c, det_u, det_v = geometry_from_sources(
        sources, sid=sid, sdd=sdd
    )
    sino = _from_backend(cone_forward_footprint(
        _to_backend(volume), _to_backend(src_pos), _to_backend(det_c),
        _to_backend(det_u), _to_backend(det_v),
        det_u=det_shape[0], det_v=det_shape[1],
        du=du, dv=dv, voxel_spacing=voxel_spacing,
    ))
    sino_np = np.asarray(sino, dtype=np.float32)
    per_view_mean = sino_np.reshape(n_probe, -1).mean(axis=1)
    return float(np.median(per_view_mean))


def _calibrate_alpha(
    volume: mx.array,
    sid: float,
    sdd: float,
    det_shape: tuple[int, int] = (128, 128),
    du: float = 2.0,
    dv: float = 2.0,
    voxel_spacing: float = 1.0,
    quantile: float = 0.75,
    n_probe: int = 8,
) -> float:
    """Per-phantom alpha calibration following the paper's §V-C heuristic.

    Forward-projects the volume from a handful of random source positions on
    the sphere and returns the requested quantile of the non-background
    pixel line-integral values.  This puts alpha into the soft-transition
    regime of the sigmoid so the gate is graded rather than binary, which
    matters a lot for high-Z phantoms whose line integrals can reach 30–50.
    """
    import numpy as np
    from diffct_mlx import cone_forward_footprint
    from .geometry import geometry_from_sources

    sources = random_sphere_sources(n_probe, sid, seed=0)
    src_pos, det_c, det_u, det_v = geometry_from_sources(
        sources, sid=sid, sdd=sdd
    )
    sino = _from_backend(cone_forward_footprint(
        _to_backend(volume), _to_backend(src_pos), _to_backend(det_c),
        _to_backend(det_u), _to_backend(det_v),
        det_u=det_shape[0], det_v=det_shape[1],
        du=du, dv=dv, voxel_spacing=voxel_spacing,
    ))
    vals = np.asarray(sino, dtype=np.float32).reshape(-1)
    vals = vals[vals > 1e-6]
    if vals.size == 0:
        return 2.0
    return float(np.quantile(vals, quantile))


def _build_absorption_config(
    sid: float,
    sdd: float,
    volume: mx.array | None = None,
    gate_type: str = "fraction",
) -> AbsorptionConfig:
    """Absorption-gate geometry calibrated to the phantom.

    Gate-type-specific calibration:
    * ``fraction``: α = 75-percentile of pixel line integrals.
    * ``mean``:     α = median of per-view ROI-footprint-weighted mean L.
    * ``path``:     α = median of per-view ROI-footprint-weighted path length,
                    β = 1.5 / std(L)  so the sigmoid is graded, not binary.
    """
    det_shape = (128, 128)
    du = dv = 2.0
    beta_frac = 12.0
    if volume is not None:
        if gate_type == "path":
            alpha, beta_frac, _ = _calibrate_path_gate(
                volume, sid, sdd,
                det_shape=det_shape, du=du, dv=dv, voxel_spacing=1.0,
            )
        elif gate_type == "mean":
            alpha = _calibrate_alpha_mean(
                volume, sid, sdd,
                det_shape=det_shape, du=du, dv=dv, voxel_spacing=1.0,
            )
        else:
            alpha = _calibrate_alpha(
                volume, sid, sdd,
                det_shape=det_shape, du=du, dv=dv, voxel_spacing=1.0,
            )
    else:
        alpha = 2.0
    return AbsorptionConfig(
        alpha=alpha,
        eta=0.30,
        sid=sid,
        sdd=sdd,
        det_u=det_shape[0],
        det_v=det_shape[1],
        du=du,
        dv=dv,
        voxel_spacing=1.0,
        roi_radius=48.0,
        beta_pixel=8.0,
        beta_frac=beta_frac,
    )


def multistart_adam_sources(
    k: int,
    sid: float,
    *,
    roi_center: mx.array,
    volume: mx.array,
    sdd: float = 900.0,
    n_random_restarts: int = 3,
    n_candidates: int = 200,
    n_normals: int = 2000,   # protocol value (2026-08-27): the 15 deg
                             # coverage bandwidth tiles the sphere with
                             # ~58 caps, so z must give many samples per
                             # cap; z=300 left ~5 and its sampling noise
                             # cost up to 3 dB on the flange.
    n_steps: int = 100,
    lr: float = 0.5,
    lambda_cov: float = 1.0,
    lambda_vcl: float = 0.2,
    lambda_path: float = 0.2,
    vcl_target_shape: tuple[int, int, int] = (128, 128, 128),
    vcl_r1: float = 1e-3,
    detector_shape: tuple[int, int] = (320, 320),
    vcl_detector_shape: tuple[int, int] = (128, 128),
    du: float = 1.0,
    dv: float = 1.0,
    voxel_spacing: float = 1.0,
    vcl_sample_seed: int = 0,
    vcl_ridge: float = 1e-3,
    vcl_geometry_vjp_mode: str = "full_finite_difference",
    vcl_geometry_fd_step: float = 0.5,
    score_cfg: ScoreConfig | None = None,
    roi_points: mx.array | None = None,
    roi_weights: mx.array | None = None,
    prefer_sparse_backprojection: bool = True,
) -> mx.array:
    """Best-of-N Adam refinement from ``n_random_restarts + 2`` diverse inits.

    Runs Adam on the combined cov + path + VCL objective starting from:

      1. Discrete greedy-Tuy selection (Tuy-coverage optimum on candidates).
      2. Discrete VCLS swap-search (VCL optimum on candidates).
      3. ``n_random_restarts`` random Fibonacci-jittered sphere positions.

    Each candidate trajectory is refined with the same Adam loop (cosine
    LR schedule, patience-15 early stopping, return-best-iterate).  The
    trajectory with the *highest combined surrogate value* at its best
    iterate is returned.  Selection by surrogate (rather than SART PSNR)
    keeps the routine free of any reconstruction cost during selection.
    """
    from ..vcl_diff import vcl_loss_continuous
    from ..score import coverage_covariance_information  # noqa: F401
    from .vcl import compute_R_gamma, vcls_select

    cfg = score_cfg or _default_score_config()
    radon_normals = sample_unit_sphere(n_normals)
    sigma = cfg.gaussian_sigma()
    model = Free3D()

    vcl_ctx = _matched_vcl_context(
        volume, sid=sid, sdd=sdd,
        target_shape=vcl_target_shape, r1=vcl_r1,
        detector_shape=detector_shape,
        working_detector_shape=vcl_detector_shape,
        du=du, dv=dv, voxel_spacing=voxel_spacing,
        sample_seed=vcl_sample_seed, ridge=vcl_ridge,
        geometry_vjp_mode=vcl_geometry_vjp_mode,
        geometry_fd_step=vcl_geometry_fd_step,
        prefer_sparse_backprojection=prefer_sparse_backprojection,
    )
    abs_cfg = _build_absorption_config(sid, sdd, volume=volume, gate_type="path")

    # Build inits
    inits: list[tuple[str, mx.array]] = []
    inits.append((
        "greedy",
        greedy_discrete(k, sid, roi_center=roi_center, score_cfg=cfg,
                        n_candidates=n_candidates, n_normals=n_normals,
                        roi_points=roi_points, roi_weights=roi_weights),
    ))
    candidates = sample_unit_sphere(n_candidates) * sid
    pre = compute_R_gamma(
        volume, candidates, sid=sid, sdd=sdd, det_shape=detector_shape,
        du=du, dv=dv, voxel_spacing=voxel_spacing,
        r1=vcl_r1, seed=0,
        prefer_sparse_backprojection=prefer_sparse_backprojection,
    )
    idx, _ = vcls_select(pre, k, seed=0)
    inits.append(("vcls", candidates[mx.array(idx)]))
    for r in range(n_random_restarts):
        inits.append((f"rand{r}", random_sphere_sources(k, sid, seed=100 + r)))

    def project_to_sphere(p: mx.array) -> mx.array:
        n = mx.linalg.norm(p, axis=-1, keepdims=True)
        return p * (sid / mx.maximum(n, 1e-6))

    def combined_loss(params, _step):
        sources = project_to_sphere(params)
        cov = saturated_coverage(
            sources, roi_center, radon_normals,
            mx.ones(sources.shape[0], dtype=mx.float32),
            cfg,
            roi_points=roi_points,
            roi_weights=roi_weights,
        )
        nu = compute_absorption_gate(
            sources, roi_center, volume, abs_cfg,
            grad_mode="fd_src", gate_type="path",
        )
        path_bonus = mx.mean(nu)
        vcl = vcl_loss_continuous(sources, vcl_ctx)
        return (lambda_cov * cov
                + lambda_path * path_bonus
                + lambda_vcl * (1.0 - vcl))

    # Selection uses the VCL information score directly — it is the
    # closest closed-form NMSE surrogate and the best predictor of
    # reconstruction quality among our cheap objectives.
    best_i_vcl = float("-inf")
    best_sources = None
    for tag, init in inits:
        refined, _ = adam_ascent(
            combined_loss, init, n_steps=n_steps, lr=lr,
            project_fn=project_to_sphere,
            lr_schedule="cosine", lr_min=lr * 0.05,
            patience=15, return_best=True,
        )
        i_vcl = 1.0 - float(vcl_loss_continuous(refined, vcl_ctx))
        if i_vcl > best_i_vcl:
            best_i_vcl = i_vcl
            best_sources = refined

    return best_sources


def greedy_adam_vcl_continuous(
    k: int,
    sid: float,
    *,
    roi_center: mx.array,
    volume: mx.array,
    sdd: float = 900.0,
    lambda_vcl: float = 1.0,
    lambda_cov: float = 0.0,         # weight of geometric coverage (0 = pure VCL)
    lambda_path: float = 0.0,        # additional path-bonus weight
    vcl_target_shape: tuple[int, int, int] = (128, 128, 128),
    vcl_r1: float = 1e-3,
    n_candidates: int = 200,
    n_normals: int = 2000,   # protocol value (2026-08-27): the 15 deg
                             # coverage bandwidth tiles the sphere with
                             # ~58 caps, so z must give many samples per
                             # cap; z=300 left ~5 and its sampling noise
                             # cost up to 3 dB on the flange.
    n_steps: int = 100,
    lr: float = 0.05,   # validated by LR sweep (#48): lr=0.5 overshoots, 0.05 ≈ no-op-safe
    score_cfg: ScoreConfig | None = None,
    init_method: str = "greedy_tuy",   # "greedy_tuy", "vcls", or "uniform_sphere"
    voxel_spacing: float = 1.0,
        # mm/voxel of the volume.  Threaded into the internal
        # compute_R_gamma call used by the ``vcls`` init when no
        # caller-supplied ``vcls_precompute`` is available — the
        # cone-forward needs it for correct ray paths.
    detector_shape: tuple[int, int] = (320, 320),
    du: float = 1.0,
    dv: float = 1.0,
    vcl_detector_shape: tuple[int, int] = (128, 128),
    vcl_sample_seed: int = 0,
    vcl_ridge: float = 1e-3,
    vcl_geometry_vjp_mode: str = "full_finite_difference",
    vcl_geometry_fd_step: float = 0.5,
    vcls_precompute=None,              # pre-built (R, γ) cache to reuse
    cov_decay: str = "constant",       # "constant" or "cosine"
    bundle_schedule: str = "constant",  # "constant" or "ramp" (sin²: 0 -> lambda_bundle)
    cov_survival_weight: bool = False,   # nu_i = exp(-tau_i) in the
                                        # coverage sum instead of nu=1
    bundle_target: mx.array | None = None,  # absorption target; None =
                                        # roi_center. Separating it from
                                        # the coverage target is what
                                        # makes 'which direction has less
                                        # absorption' a question about the
                                        # defect rather than the isocentre.
    cov_sigma_schedule: str = "constant",  # "constant" or "cosine"
    cov_sigma_mult: float = 4.0,        # starting σ multiplier when annealing
    noise_schedule: str = "none",       # "none", "langevin_cosine", "langevin_const"
    noise_temp: float = 0.0,            # starting temperature (k_B T) of Langevin noise
    noise_seed: int = 0,
    derivative_mode: str = "analytic",  # gradient estimator: "analytic" or
                                        # "fd_central" (matched central FD of
                                        # the SAME objective; REV-P0-01)
    derivative_fd_step: float = 1e-3,   # step (parameter units) for fd_central
    optimizer: str = "adam",            # "adam" or "cmaes" — derivative-free
                                        # CMA-ES on the SAME objective/init/
                                        # projection (REV-P1-08 comparator)
    optimizer_budget_evals: int | None = None,
                                        # cmaes evaluation budget; default =
                                        # the FD arm's cost 2·3k·n_steps
    cmaes_sigma0: float | None = None,  # initial CMA-ES step (mm);
                                        # default 0.02·sid ≈ 1.1° arc
    lambda_bundle: float = 0.0,         # additive bundle-absorption penalty (MLX-autograd)
    lambda_bundle_auto_scale: bool = False,
                                        # if True, divide λ_bundle by the median
                                        # bundle attenuation on uniform probe
                                        # views so the raw λ value is the target
                                        # contribution scale at initialisation
    bundle_cfg=None,                    # BundleAbsorptionConfig; auto-calibrated if None
    lambda_oed: float = 0.0,            # weight of the photon-noise-weighted A+D OED score
    lambda_oed_auto_scale: bool = True,  # divide λ_oed by OED(init) so λ = init contribution
    oed_photon_count: float = 1.0,      # emitted photons / detector pixel for PW-OED
    oed_ridge: float = 1e-2,            # prior precision ρ in F = Σ w_i T_iT_i^T + ρI
    oed_alpha: float = 1.0,             # A-optimality (noise) weight inside the OED score
    oed_beta: float = 1.0,              # D-optimality (information) weight inside the OED score
    lambda_icov_fft: float = 0.0,       # weight of soft I_cov with Fourier-slice γ
    lambda_icov_fft_auto_scale: bool = True,
                                        # auto-divide λ_icov_fft by I_cov(init)
                                        # so the term's contribution at init
                                        # matches the raw λ value regardless
                                        # of phantom amplitude or resolution
    seed: int = 0,                      # seeds candidate-sphere rotation and VCLS swap
    trace_dir: str | None = None,
    trace_tag: str | None = None,
    roi_points: mx.array | None = None,
    roi_weights: mx.array | None = None,
    prefer_sparse_backprojection: bool = True,
) -> mx.array:
    """Greedy warm-start + Adam refinement with a *differentiable VCL term*.

    Combines the geometric coverage objective with the closed-form
    reconstruction-error surrogate VCL = 1 − γᵀR⁻¹γ (Lin et al. 2025):

        L = − [λ_cov(t) · C_geo + λ_path · mean(ν_path) + λ_vcl · I_VCL]

    where ``I_VCL = γᵀR⁻¹γ ∈ [0, 1]`` and is maximised.  Setting
    ``λ_cov = 0`` recovers the pure-VCL variant; ``λ_path = 0`` drops the
    path-length bonus.

    ``cov_decay="cosine"`` makes λ_cov step-adaptive: it starts at the
    given ``lambda_cov`` value at step 0 and decays as
    ``λ_cov(t) = lambda_cov · cos²(π t / 2T)`` to zero at the last step.
    This drives initial diversity through coverage and then lets VCL take
    over for refinement once sphere coverage is saturated.  The path and
    VCL weights stay constant.

    ``cov_sigma_schedule="cosine"`` (**σ-annealing / homotopy**) starts the
    Tuy Gaussian bandwidth at ``cov_sigma_mult · σ_target`` (broad kernel →
    near-convex landscape, large coverage gradients far from any Tuy
    plane) and cools to ``σ_target`` by ``σ_t = σ · (1 + (m-1) cos²(πt/2T))``.
    This is a standard continuation strategy and gives Adam genuine
    exploration head-room before the landscape sharpens.

    ``noise_schedule="langevin_cosine"`` (**Riemannian SGLD on S²**) adds
    Gaussian noise of variance ``2 lr_t T_t`` projected onto the source
    tangent plane after each Adam step, with the temperature annealed as
    ``T_t = noise_temp · cos²(πt/2T)``.  In the hot phase the iterates can
    hop between surrogate basins; in the cold phase the dynamics reduce to
    plain Adam.  Set ``noise_temp = 0`` (default) to disable.

    The VCL term operates on a downsampled volume (default 128³) so each
    Adam step stays around 1 second; the geometric coverage term continues
    to use the full Radon-normal set.
    """
    import math as _math_cov
    from ..vcl_diff import vcl_loss_continuous

    cfg = score_cfg or _default_score_config()
    radon_normals = sample_unit_sphere(n_normals)

    if init_method == "vcls":
        # Discrete VCLS swap-search picks a globally-good (in the VCL sense)
        # starting subset; Adam then refines off the grid.  We prefer the
        # caller-supplied (R, γ) cache when available (same K_max and
        # detector geometry as the matching VCLS baseline); otherwise we
        # build one locally with the n_candidates parameter.
        from .vcl import compute_R_gamma, vcls_select

        if vcls_precompute is not None:
            vcls_pre = vcls_precompute
        else:
            candidates = sample_unit_sphere(n_candidates, seed=seed) * sid
            vcls_pre = compute_R_gamma(
                volume, candidates,
                sid=sid, sdd=sdd, det_shape=detector_shape,
                du=du, dv=dv,
                voxel_spacing=voxel_spacing,
                r1=vcl_r1, seed=seed,
                prefer_sparse_backprojection=prefer_sparse_backprojection,
            )
        indices, _ = vcls_select(vcls_pre, k, seed=seed)
        init = vcls_pre.candidate_sources[mx.array(indices)]
    elif init_method == "greedy_tuy":
        init = greedy_discrete(
            k, sid,
            roi_center=roi_center,
            n_candidates=n_candidates,
            sphere_seed=seed,
            score_cfg=cfg,
            n_normals=n_normals,
            roi_points=roi_points,
            roi_weights=roi_weights,
        )
    elif init_method == "uniform_sphere":
        init = sample_unit_sphere(k, seed=seed) * sid
    else:
        raise ValueError(f"Unknown init_method '{init_method}'")

    # The VCL context (downsampled volume + subsampled voxels) backs both the
    # VCL information term and the OED information score (which reuses the same
    # view bases), so it is built whenever either is active.  With both off the
    # selector is genuinely VCL-free: no volume downsample and no per-step k x k
    # solve, which is what makes the cold-start bundle path independent of the
    # object-specific R precompute.
    vcl_ctx = (
        _matched_vcl_context(
            volume, sid=sid, sdd=sdd,
            target_shape=vcl_target_shape, r1=vcl_r1,
            detector_shape=detector_shape,
            working_detector_shape=vcl_detector_shape,
            du=du, dv=dv, voxel_spacing=voxel_spacing,
            sample_seed=vcl_sample_seed, ridge=vcl_ridge,
            geometry_vjp_mode=vcl_geometry_vjp_mode,
            geometry_fd_step=vcl_geometry_fd_step,
            prefer_sparse_backprojection=prefer_sparse_backprojection,
        )
        if (lambda_vcl > 0.0 or lambda_oed > 0.0)
        else None
    )

    def _sigma_at(step: int) -> float:
        """σ-annealing schedule for the Tuy Gaussian kernel."""
        if cov_sigma_schedule == "cosine" and n_steps > 1:
            t_frac = step / (n_steps - 1)
            mult_t = 1.0 + (cov_sigma_mult - 1.0) * (
                _math_cov.cos(_math_cov.pi * t_frac / 2.0) ** 2
            )
            return sigma_target * mult_t
        return sigma_target

    if lambda_path > 0.0:
        abs_cfg = _build_absorption_config(sid, sdd, volume=volume, gate_type="path")
    else:
        abs_cfg = None

    # Optional MLX-native bundle-integral absorption penalty.  Unlike the
    # fd_src cone_forward gate, the bundle integral is fully autograd-
    # differentiable through trilinear sampling — no FD passes.  Used as
    # an additive *un-bounded* penalty `-λ · mean(τ̄)` to keep gradients
    # alive everywhere (a bounded gate ν = exp(-α·τ̄) would saturate when
    # a source looks through the densest part of the phantom).
    # The bundle integral backs both the additive absorption penalty and the
    # OED photon weights w_i = exp(-τ̄_i), so it is set up whenever either needs it.
    if lambda_bundle > 0.0 or lambda_oed > 0.0 or cov_survival_weight:
        from ..absorption_bundle import (
            BundleAbsorptionConfig, bundle_path_integral,
        )
        bcfg = bundle_cfg or BundleAbsorptionConfig(
            voxel_spacing=1.0,  # caller can override via bundle_cfg
        )
    else:
        bcfg = None
        bundle_path_integral = None  # silence linter

    # Optional volume-aware Fourier-slice I_cov.  A single 3D FFT computes
    # the phantom-specific Radon-direction weights (cost: seconds on
    # 384³).  Per Adam step we then evaluate γ_F^T R_F^{-1} γ_F on the
    # k × k coverage covariance matrix — no SART forward, no per-phantom
    # cache, fully autograd-native through the trilinear FFT sampling.
    if lambda_icov_fft > 0.0:
        from ..fourier_radon import fourier_radon_weights
        from ..score import coverage_covariance_information

        w_fft_icov = fourier_radon_weights(volume, radon_normals)
        mx.eval(w_fft_icov)

        # Auto-calibrate λ_icov_fft against the init-source magnitude of
        # the score, so the relative weight against the unit-magnitude
        # coverage term is volume-amplitude- and resolution-independent.
        # Without this, a hard-coded λ either dominates or vanishes
        # depending on phantom anisotropy and FFT resolution.
        if lambda_icov_fft_auto_scale:
            init_icov = float(coverage_covariance_information(
                init, roi_center, radon_normals,
                mx.ones(k, dtype=mx.float32), cfg,
                direction_weights=w_fft_icov,
            ))
            lambda_icov_fft = lambda_icov_fft / max(init_icov, 1e-12)
    else:
        w_fft_icov = None
        coverage_covariance_information = None

    # Optionally calibrate λ_bundle against the probe-view attenuation scale,
    # which keeps combined bundle+OED experiments comparable to the named
    # bundle baselines without hard-coding a phantom-specific λ.
    if lambda_bundle > 0.0 and lambda_bundle_auto_scale:
        # Probes live on the isocentre-centred SID sphere — the actual source
        # manifold — aiming at roi_center (unified 2026-08-11; the previous
        # ROI-centred probe sphere placed probes off-manifold for off-centre
        # ROIs and disagreed with the limited-angle/real-experiment recipe).
        probes = sample_unit_sphere(256) * sid
        _cal_target = bundle_target
        tau_probe = bundle_path_integral(
            probes,
            roi_center if _cal_target is None else _cal_target,
            volume, bcfg)
        med_tau = float(mx.median(tau_probe))
        lambda_bundle = calibrate_bundle_weight(med_tau, lambda_bundle)

    # Auto-calibrate λ_oed against its init-source magnitude, so the raw
    # λ value is the OED term's contribution at initialisation regardless of
    # phantom amplitude / k / resolution (mirrors the I_cov_fft calibration).
    if lambda_oed > 0.0 and lambda_oed_auto_scale:
        from ..oed import oed_loss_continuous
        init_tau = bundle_path_integral(init, roi_center, volume, bcfg)
        init_oed = float(oed_loss_continuous(
            init, vcl_ctx, init_tau,
            photon_count=oed_photon_count,
            ridge=oed_ridge, alpha=oed_alpha, beta=oed_beta,
        ))
        lambda_oed = lambda_oed / max(abs(init_oed), 1e-12)

    sigma_target = cfg.gaussian_sigma()
    model = Free3D()

    def evaluate_terms(params, step):
        sources = model(params)
        norms = mx.linalg.norm(sources, axis=-1, keepdims=True)
        sources = sources * (sid / mx.maximum(norms, 1e-6))
        loss = mx.array(0.0, dtype=mx.float32)
        # Step-adaptive λ_cov: starts at lambda_cov, decays to 0 over n_steps
        # by cos²(πt/2T), so coverage drives early diversity and VCL takes
        # over refinement once the sphere is reasonably covered.
        if cov_decay == "cosine" and n_steps > 1:
            t_frac = step / (n_steps - 1)
            lam_cov_t = lambda_cov * (_math_cov.cos(_math_cov.pi * t_frac / 2.0) ** 2)
        else:
            lam_cov_t = lambda_cov
        sigma_t = _sigma_at(step)
        cov = mx.array(0.0, dtype=mx.float32)
        path_bonus = mx.array(0.0, dtype=mx.float32)
        tau_mean = mx.array(0.0, dtype=mx.float32)
        vcl_info = mx.array(0.0, dtype=mx.float32)
        icov_fft = mx.array(0.0, dtype=mx.float32)
        oed_info = mx.array(0.0, dtype=mx.float32)
        if lam_cov_t > 0.0:
            cfg_now = ScoreConfig(tau=cfg.tau, sigma=sigma_t)
            if cov_survival_weight:
                # Absorption as informativeness rather than as a cost: a view
                # contributes to the covered Radon directions only in
                # proportion to its photon survival nu_i = exp(-tau_bar_i).
                # Unlike the additive -lambda*mean(tau) penalty this cannot
                # reward moving away from a direction that no other view
                # covers, because the lost coverage is not compensated by any
                # reduction in absorption.
                nu_cov = mx.exp(-bundle_path_integral(
                    sources,
                    roi_center if bundle_target is None else bundle_target,
                    volume, bcfg))
            else:
                nu_cov = mx.ones(sources.shape[0], dtype=mx.float32)
            cov = saturated_coverage(
                sources, roi_center, radon_normals,
                nu_cov,
                cfg_now,
                roi_points=roi_points,
                roi_weights=roi_weights,
            )
            loss = loss + lam_cov_t * cov
        if lambda_path > 0.0:
            nu = compute_absorption_gate(
                sources, roi_center, volume, abs_cfg,
                grad_mode="fd_src", gate_type="path",
            )
            path_bonus = mx.mean(nu)
            loss = loss + lambda_path * path_bonus
        if lambda_bundle > 0.0:
            # Additive bundle penalty: subtract λ · mean(τ̄) so the maximiser
            # *reduces* absorption.  No bounded gate → no saturation; the
            # gradient flows MLX-native through trilinear sampling.
            #
            # bundle_schedule="ramp" grows the weight from 0 to lambda_bundle
            # as sin²(πt/2T) — the mirror of the cosine λ_cov decay.  Coverage
            # establishes Tuy diversity first; the bundle then refines toward
            # low-absorption views among the already well-covered configs.
            # This is the anti-collapse curriculum: a strong absorption term
            # applied before coverage is established pulls every source toward
            # the same low-density cone and destroys angular diversity.
            if bundle_schedule == "ramp" and n_steps > 1:
                t_frac_b = step / (n_steps - 1)
                lam_bundle_t = lambda_bundle * (
                    _math_cov.sin(_math_cov.pi * t_frac_b / 2.0) ** 2
                )
            else:
                lam_bundle_t = lambda_bundle
            tau_bar = bundle_path_integral(
                sources,
                roi_center if bundle_target is None else bundle_target,
                volume, bcfg)
            tau_mean = mx.mean(tau_bar)
            loss = loss - lam_bundle_t * tau_mean
        if lambda_vcl > 0.0:
            vcl = vcl_loss_continuous(sources, vcl_ctx)
            vcl_info = 1.0 - vcl
            loss = loss + lambda_vcl * vcl_info
        if lambda_oed > 0.0:
            # Photon-noise-weighted optimal-experimental-design score.  Each
            # VCL view basis is weighted by photon survival w_i = exp(-τ̄_i);
            # the A-optimality part targets reconstruction noise, the
            # D-optimality part targets information.  This is the term that can
            # beat VCLS under photon noise, because VCLS is the unweighted,
            # noise-blind special case.
            from ..oed import oed_loss_continuous
            tau_oed = bundle_path_integral(sources, roi_center, volume, bcfg)
            oed_info = oed_loss_continuous(
                sources, vcl_ctx, tau_oed,
                photon_count=oed_photon_count,
                ridge=oed_ridge, alpha=oed_alpha, beta=oed_beta,
            )
            loss = loss + lambda_oed * oed_info
        if lambda_icov_fft > 0.0:
            # Soft I_cov with phantom-aware Fourier weights.  Same sphere
            # of Radon normals as the coverage term; nu = 1 for every
            # source (the coverage covariance score's own gating already
            # lives inside ψ).  We add the score with a positive sign so
            # Adam ascent maximises it.
            cfg_now = ScoreConfig(tau=cfg.tau, sigma=sigma_t)
            nu_icov = mx.ones(sources.shape[0])
            icov_fft = coverage_covariance_information(
                sources, roi_center, radon_normals, nu_icov, cfg_now,
                direction_weights=w_fft_icov,
            )
            loss = loss + lambda_icov_fft * icov_fft
        evaluated = sources, {
            "objective": loss,
            "cov": cov,
            "path_bonus": path_bonus,
            "tau_mean": tau_mean,
            "vcl_info": vcl_info,
            "icov_fft": icov_fft,
            "oed_info": oed_info,
            "lambda_cov_t": mx.array(lam_cov_t, dtype=mx.float32),
            "sigma_t": mx.array(sigma_t, dtype=mx.float32),
        }
        # Hand the evaluation to the recorder, which keeps it only when armed
        # (once per Adam step, before the gradient is formed).  Nothing here
        # touches ``loss`` or the parameters.
        recorder.offer(evaluated)
        return evaluated

    def coverage_fn(params, step):
        _sources, terms = evaluate_terms(params, step)
        return terms["objective"]

    record_terms = trace_dir is not None or _terms_recording_enabled()
    recorder = _TermRecorder(evaluate_terms, keep_sources=trace_dir is not None)

    noise_fn = None
    if noise_schedule != "none" and noise_temp > 0.0:
        # Riemannian SGLD on S²(sid).  Draw isotropic Gaussian noise, project
        # onto the tangent plane of each source (so the step stays on the
        # sphere after retraction), and scale by sqrt(2 lr_t T_t) — the
        # Langevin discretisation of dX = -∇L dt + √(2T) dW with T_t
        # following the chosen schedule.
        _rng = mx.random.key(noise_seed)
        _rng_state = {"key": _rng}

        def _temp_at(step: int) -> float:
            if noise_schedule == "langevin_cosine" and n_steps > 1:
                t_frac = step / (n_steps - 1)
                return noise_temp * (_math_cov.cos(_math_cov.pi * t_frac / 2.0) ** 2)
            return noise_temp  # "langevin_const"

        def noise_fn(params, step, lr_t):
            T_t = _temp_at(step)
            if T_t <= 0.0:
                return mx.zeros_like(params)
            _rng_state["key"], sub = mx.random.split(_rng_state["key"])
            xi = mx.random.normal(shape=params.shape, key=sub)
            # Tangent-plane projection: ξ − (ξ·ŝ) ŝ
            s_hat = params / mx.maximum(
                mx.linalg.norm(params, axis=-1, keepdims=True), 1e-9
            )
            xi_tan = xi - mx.sum(xi * s_hat, axis=-1, keepdims=True) * s_hat
            scale = float(_math_cov.sqrt(2.0 * lr_t * T_t))
            return scale * xi_tan

    def project_to_sphere(p: mx.array) -> mx.array:
        n = mx.linalg.norm(p, axis=-1, keepdims=True)
        return p * (sid / mx.maximum(n, 1e-6))

    callback = None
    if record_terms:
        if trace_dir is not None:
            # The only extra objective evaluation, and it happens before the
            # optimiser starts.  It exists purely so the trace file carries the
            # initialisation; terms-only recording skips it.
            recorder.record("init", -1, init, 0)
        callback = recorder.on_step
        recorder.arm()

    if optimizer == "cmaes":
        from ..optimize import cmaes_ascent
        budget = optimizer_budget_evals
        if budget is None:
            budget = 2 * 3 * k * n_steps      # = the matched FD arm's cost
        refined, _ = cmaes_ascent(
            coverage_fn, init,
            sigma0=cmaes_sigma0 if cmaes_sigma0 is not None else 0.02 * sid,
            budget_evals=budget,
            seed=seed,
            project_fn=project_to_sphere,
        )
    elif optimizer == "adam":
        refined, _ = adam_ascent(
            coverage_fn, init, n_steps=n_steps, lr=lr,
            project_fn=project_to_sphere,
            lr_schedule="cosine", lr_min=lr * 0.05,
            patience=15, return_best=True,
            noise_fn=noise_fn,
            callback=callback,
            grad_mode=derivative_mode,
            fd_step=derivative_fd_step,
        )
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    if record_terms:
        _publish_optim_terms(recorder.final(refined, max(n_steps - 1, 0)))
        if trace_dir is not None:
            _write_optim_trace(
                trace_dir,
                trace_tag or "greedy_adam_vcl_continuous",
                recorder.rows,
                recorder.sources,
            )
    return refined


def two_axis_gantry_vcl_continuous(
    k: int,
    sid: float,
    *,
    roi_center: mx.array,
    volume: mx.array,
    sdd: float = 900.0,
    lambda_vcl: float = 1.0,
    lambda_cov: float = 0.0,
    lambda_path: float = 0.0,
    vcl_target_shape: tuple[int, int, int] = (128, 128, 128),
    vcl_r1: float = 1e-3,
    n_candidates: int = 200,
    n_normals: int = 2000,   # protocol value (2026-08-27): the 15 deg
                             # coverage bandwidth tiles the sphere with
                             # ~58 caps, so z must give many samples per
                             # cap; z=300 left ~5 and its sampling noise
                             # cost up to 3 dB on the flange.
    n_steps: int = 100,
    lr: float = 0.05,
    score_cfg: ScoreConfig | None = None,
    init_method: str = "greedy_tuy",
    voxel_spacing: float = 1.0,
    detector_shape: tuple[int, int] = (320, 320),
    du: float = 1.0,
    dv: float = 1.0,
    vcl_detector_shape: tuple[int, int] = (128, 128),
    vcl_sample_seed: int = 0,
    vcl_ridge: float = 1e-3,
    vcl_geometry_vjp_mode: str = "full_finite_difference",
    vcl_geometry_fd_step: float = 0.5,
    vcls_precompute=None,
    cov_decay: str = "constant",
    bundle_schedule: str = "constant",  # "constant" or "ramp" (sin²: 0 -> lambda_bundle)
    cov_survival_weight: bool = False,   # nu_i = exp(-tau_i) in the
                                        # coverage sum instead of nu=1
    bundle_target: mx.array | None = None,  # absorption target; None =
                                        # roi_center. Separating it from
                                        # the coverage target is what
                                        # makes 'which direction has less
                                        # absorption' a question about the
                                        # defect rather than the isocentre.
    cov_sigma_schedule: str = "constant",
    cov_sigma_mult: float = 4.0,
    noise_schedule: str = "none",
    noise_temp: float = 0.0,
    noise_seed: int = 0,
    lambda_bundle: float = 0.0,
    bundle_cfg=None,
    lambda_icov_fft: float = 0.0,
    lambda_icov_fft_auto_scale: bool = True,
    seed: int = 0,
    limited_carm: bool = False,
    smooth_constraint: bool = True,   # DEFAULT (2026-08-27): build the envelope
                                     # into the chart via tanh instead of
                                     # clipping after the step. Simpler (no
                                     # projection, no pinning special case) and
                                     # identical to the measured-study recipe.
                                     # Set False for the legacy clipped chart.
    trace_dir: str | None = None,
    trace_tag: str | None = None,
    roi_points: mx.array | None = None,
    roi_weights: mx.array | None = None,
    prefer_sparse_backprojection: bool = True,
) -> mx.array:
    """Continuous optimisation with a hard 2-axis gantry reparameterisation."""
    import math as _math_cov
    from ..vcl_diff import vcl_loss_continuous

    cfg = score_cfg or _default_score_config()
    radon_normals = sample_unit_sphere(n_normals)
    _carm_cls = SmoothTwoAxisGantry if smooth_constraint else CArmTwoAxisGantry
    gantry = _carm_cls(sid=sid) if limited_carm else TwoAxisGantry(sid=sid)

    if init_method == "vcls":
        from .vcl import compute_R_gamma, vcls_select

        if vcls_precompute is not None:
            vcls_pre = vcls_precompute
        else:
            candidates = sample_unit_sphere(n_candidates, seed=seed) * sid
            vcls_pre = compute_R_gamma(
                volume, candidates,
                sid=sid, sdd=sdd, det_shape=detector_shape,
                du=du, dv=dv,
                voxel_spacing=voxel_spacing,
                r1=vcl_r1, seed=seed,
                prefer_sparse_backprojection=prefer_sparse_backprojection,
            )
        indices, _ = vcls_select(vcls_pre, k, seed=seed)
        init_sources = vcls_pre.candidate_sources[mx.array(indices)]
    elif init_method == "greedy_tuy":
        init_sources = greedy_discrete(
            k, sid,
            roi_center=roi_center,
            n_candidates=n_candidates,
            sphere_seed=seed,
            score_cfg=cfg,
            n_normals=n_normals,
            roi_points=roi_points,
            roi_weights=roi_weights,
        )
    else:
        raise ValueError(f"Unknown init_method '{init_method}'")

    init = (
        _sources_to_carm_params(init_sources, sid)
        if limited_carm else _sources_to_two_axis_params(init_sources, sid)
    )
    if limited_carm and smooth_constraint:
        # The smooth chart optimises unconstrained raw coordinates, so the
        # feasible initial angles have to be pulled back through the tanh.
        init = gantry.inverse(init)
    sigma_target = cfg.gaussian_sigma()
    # VCL context only when the VCL term is active (see free-sphere note).
    vcl_ctx = (
        _matched_vcl_context(
            volume, sid=sid, sdd=sdd,
            target_shape=vcl_target_shape, r1=vcl_r1,
            detector_shape=detector_shape,
            working_detector_shape=vcl_detector_shape,
            du=du, dv=dv, voxel_spacing=voxel_spacing,
            sample_seed=vcl_sample_seed, ridge=vcl_ridge,
            geometry_vjp_mode=vcl_geometry_vjp_mode,
            geometry_fd_step=vcl_geometry_fd_step,
            prefer_sparse_backprojection=prefer_sparse_backprojection,
        )
        if lambda_vcl > 0.0
        else None
    )

    def _sigma_at(step: int) -> float:
        if cov_sigma_schedule == "cosine" and n_steps > 1:
            t_frac = step / (n_steps - 1)
            mult_t = 1.0 + (cov_sigma_mult - 1.0) * (
                _math_cov.cos(_math_cov.pi * t_frac / 2.0) ** 2
            )
            return sigma_target * mult_t
        return sigma_target

    if lambda_path > 0.0:
        abs_cfg = _build_absorption_config(sid, sdd, volume=volume, gate_type="path")
    else:
        abs_cfg = None

    if lambda_bundle > 0.0 or cov_survival_weight:
        from ..absorption_bundle import (
            BundleAbsorptionConfig, bundle_path_integral,
        )
        bcfg = bundle_cfg or BundleAbsorptionConfig(voxel_spacing=1.0)
    else:
        bcfg = None
        bundle_path_integral = None

    if lambda_icov_fft > 0.0:
        from ..fourier_radon import fourier_radon_weights
        from ..score import coverage_covariance_information

        w_fft_icov = fourier_radon_weights(volume, radon_normals)
        mx.eval(w_fft_icov)
        if lambda_icov_fft_auto_scale:
            init_icov = float(coverage_covariance_information(
                init_sources, roi_center, radon_normals,
                mx.ones(k, dtype=mx.float32), cfg,
                direction_weights=w_fft_icov,
            ))
            lambda_icov_fft = lambda_icov_fft / max(init_icov, 1e-12)
    else:
        w_fft_icov = None
        coverage_covariance_information = None

    def evaluate_terms(params, step):
        sources = gantry(params)
        loss = mx.array(0.0, dtype=mx.float32)
        if cov_decay == "cosine" and n_steps > 1:
            t_frac = step / (n_steps - 1)
            lam_cov_t = lambda_cov * (_math_cov.cos(_math_cov.pi * t_frac / 2.0) ** 2)
        else:
            lam_cov_t = lambda_cov
        sigma_t = _sigma_at(step)
        cov = mx.array(0.0, dtype=mx.float32)
        path_bonus = mx.array(0.0, dtype=mx.float32)
        tau_mean = mx.array(0.0, dtype=mx.float32)
        vcl_info = mx.array(0.0, dtype=mx.float32)
        icov_fft = mx.array(0.0, dtype=mx.float32)
        oed_info = mx.array(0.0, dtype=mx.float32)
        if lam_cov_t > 0.0:
            cfg_now = ScoreConfig(tau=cfg.tau, sigma=sigma_t)
            if cov_survival_weight:
                # Absorption as informativeness rather than as a cost: a view
                # contributes to the covered Radon directions only in
                # proportion to its photon survival nu_i = exp(-tau_bar_i).
                # Unlike the additive -lambda*mean(tau) penalty this cannot
                # reward moving away from a direction that no other view
                # covers, because the lost coverage is not compensated by any
                # reduction in absorption.
                nu_cov = mx.exp(-bundle_path_integral(
                    sources,
                    roi_center if bundle_target is None else bundle_target,
                    volume, bcfg))
            else:
                nu_cov = mx.ones(sources.shape[0], dtype=mx.float32)
            cov = saturated_coverage(
                sources, roi_center, radon_normals,
                nu_cov,
                cfg_now,
                roi_points=roi_points,
                roi_weights=roi_weights,
            )
            loss = loss + lam_cov_t * cov
        if lambda_path > 0.0:
            nu = compute_absorption_gate(
                sources, roi_center, volume, abs_cfg,
                grad_mode="fd_src", gate_type="path",
            )
            path_bonus = mx.mean(nu)
            loss = loss + lambda_path * path_bonus
        if lambda_bundle > 0.0:
            # "ramp" grows the bundle weight from 0 to lambda_bundle as
            # sin²(πt/2T) — see the free-sphere selector for the rationale.
            if bundle_schedule == "ramp" and n_steps > 1:
                t_frac_b = step / (n_steps - 1)
                lam_bundle_t = lambda_bundle * (
                    _math_cov.sin(_math_cov.pi * t_frac_b / 2.0) ** 2
                )
            else:
                lam_bundle_t = lambda_bundle
            tau_bar = bundle_path_integral(
                sources,
                roi_center if bundle_target is None else bundle_target,
                volume, bcfg)
            tau_mean = mx.mean(tau_bar)
            loss = loss - lam_bundle_t * tau_mean
        if lambda_vcl > 0.0:
            vcl = vcl_loss_continuous(sources, vcl_ctx)
            vcl_info = 1.0 - vcl
            loss = loss + lambda_vcl * vcl_info
        if lambda_icov_fft > 0.0:
            cfg_now = ScoreConfig(tau=cfg.tau, sigma=sigma_t)
            nu_icov = mx.ones(sources.shape[0])
            icov_fft = coverage_covariance_information(
                sources, roi_center, radon_normals, nu_icov, cfg_now,
                direction_weights=w_fft_icov,
            )
            loss = loss + lambda_icov_fft * icov_fft
        evaluated = sources, {
            "objective": loss,
            "cov": cov,
            "path_bonus": path_bonus,
            "tau_mean": tau_mean,
            "vcl_info": vcl_info,
            "icov_fft": icov_fft,
            "oed_info": oed_info,
            "lambda_cov_t": mx.array(lam_cov_t, dtype=mx.float32),
            "sigma_t": mx.array(sigma_t, dtype=mx.float32),
        }
        # Hand the evaluation to the recorder, which keeps it only when armed
        # (once per Adam step, before the gradient is formed).  Nothing here
        # touches ``loss`` or the parameters.
        recorder.offer(evaluated)
        return evaluated

    def coverage_fn(params, step):
        _sources, terms = evaluate_terms(params, step)
        return terms["objective"]

    record_terms = trace_dir is not None or _terms_recording_enabled()
    recorder = _TermRecorder(evaluate_terms, keep_sources=trace_dir is not None)

    callback = None
    if record_terms:
        if trace_dir is not None:
            recorder.record("init", -1, init, 0)
        callback = recorder.on_step
        recorder.arm()

    refined, _ = adam_ascent(
        coverage_fn, init, n_steps=n_steps, lr=lr,
        project_fn=gantry.clamp if limited_carm else None,
        lr_schedule="cosine", lr_min=lr * 0.05,
        patience=15, return_best=True,
        callback=callback,
    )
    if record_terms:
        _publish_optim_terms(recorder.final(refined, max(n_steps - 1, 0)))
        if trace_dir is not None:
            _write_optim_trace(
                trace_dir,
                trace_tag or "two_axis_gantry_vcl_continuous",
                recorder.rows,
                recorder.sources,
            )
    return gantry(refined)


def greedy_ensemble_vcl_continuous(
    k: int,
    sid: float,
    *,
    roi_center: mx.array,
    volume: mx.array,
    sdd: float = 900.0,
    n_ensemble: int = 4,
    repulsion_weight: float = 0.1,
    repulsion_schedule: str = "constant",   # "constant" or "cosine"
    init_strategy: str = "jitter",           # "jitter" or "diverse"
    init_jitter: float = 0.15,
    ensemble_seed: int = 0,
    final_refine_steps: int = 0,             # extra Adam steps on best member, no rep
    lambda_vcl: float = 1.0,
    lambda_cov: float = 0.0,
    lambda_path: float = 0.0,
    vcl_target_shape: tuple[int, int, int] = (128, 128, 128),
    vcl_r1: float = 1e-3,
    n_candidates: int = 200,
    n_normals: int = 2000,   # protocol value (2026-08-27): the 15 deg
                             # coverage bandwidth tiles the sphere with
                             # ~58 caps, so z must give many samples per
                             # cap; z=300 left ~5 and its sampling noise
                             # cost up to 3 dB on the flange.
    n_steps: int = 100,
    lr: float = 0.05,   # see LR-sweep diagnostic #48
    score_cfg: ScoreConfig | None = None,
    init_method: str = "greedy_tuy",
    detector_shape: tuple[int, int] = (320, 320),
    du: float = 1.0,
    dv: float = 1.0,
    vcl_detector_shape: tuple[int, int] = (128, 128),
    vcl_sample_seed: int = 0,
    vcl_ridge: float = 1e-3,
    vcl_geometry_vjp_mode: str = "full_finite_difference",
    vcl_geometry_fd_step: float = 0.5,
    vcls_precompute=None,
    roi_points: mx.array | None = None,
    roi_weights: mx.array | None = None,
    prefer_sparse_backprojection: bool = True,
    lambda_oed: float = 0.0,
    oed_ridge: float = 1e-2,
    oed_alpha: float = 1.0,
    oed_beta: float = 1.0,
    voxel_spacing: float = 1.0,
    return_stack: bool = False,
) -> mx.array:
    """Repulsive ensemble of $N$ continuous configurations on the sphere.

    $N$ copies of the $k$-view configuration are optimised jointly under

        $$\\mathcal L_\\text{tot} = \\frac{1}{N}\\sum_n \\mathcal L_\\text{base}(S^{(n)})
            \\;-\\; \\lambda_\\text{rep}\\,\\frac{1}{N(N-1)}
            \\sum_{n\\ne m} \\log d_\\text{ch}(S^{(n)}, S^{(m)})$$

    where $d_\\text{ch}$ is the symmetric chamfer distance between two
    $k$-view point sets on the source sphere and $\\mathcal L_\\text{base}$
    is the base surrogate (VCL + optional coverage / path).  The
    log-repulsion pushes ensemble members apart in configuration space,
    so the $N$ trajectories settle in *different* basins of the surrogate
    landscape — an explicit, differentiable form of multistart.

    The single configuration with the best base objective (no repulsion)
    is returned.  $N=4$–$8$ keeps MLX runtime manageable.
    """
    import math as _math_ens
    from ..vcl_diff import vcl_loss_continuous

    cfg = score_cfg or _default_score_config()
    radon_normals = sample_unit_sphere(n_normals)

    def _init_vcls() -> mx.array:
        from .vcl import compute_R_gamma, vcls_select
        if vcls_precompute is not None:
            vcls_pre = vcls_precompute
        else:
            candidates = sample_unit_sphere(n_candidates, seed=ensemble_seed) * sid
            vcls_pre = compute_R_gamma(
                volume, candidates,
                sid=sid, sdd=sdd, det_shape=detector_shape,
                du=du, dv=dv,
                voxel_spacing=voxel_spacing,
                r1=vcl_r1, seed=ensemble_seed,
                prefer_sparse_backprojection=prefer_sparse_backprojection,
            )
        indices, _ = vcls_select(vcls_pre, k, seed=ensemble_seed)
        return vcls_pre.candidate_sources[mx.array(indices)]

    def _init_greedy() -> mx.array:
        return greedy_discrete(
            k, sid,
            roi_center=roi_center,
            n_candidates=n_candidates,
            score_cfg=cfg,
            n_normals=n_normals,
            roi_points=roi_points,
            roi_weights=roi_weights,
        )

    if init_strategy == "diverse":
        # Genuinely different basins: VCLS, greedy_tuy, and (N−2) random_sphere.
        # No tangent jitter — the diversity is already structural.
        inits = [_init_vcls(), _init_greedy()]
        for n in range(n_ensemble - 2):
            inits.append(random_sphere(k, sid, seed=ensemble_seed + 11 * n + 1))
        init_stack = mx.stack(inits[:n_ensemble], axis=0)
    else:
        # Legacy: jitter a single warm-start tangentially.
        if init_method == "vcls":
            base_init = _init_vcls()
        elif init_method == "greedy_tuy":
            base_init = _init_greedy()
        else:
            raise ValueError(f"Unknown init_method '{init_method}'")

        rng = mx.random.key(ensemble_seed)
        inits = []
        for n in range(n_ensemble):
            rng, sub = mx.random.split(rng)
            if n == 0 and init_jitter > 0.0:
                inits.append(base_init)  # first replica = exact warm-start
                continue
            xi = mx.random.normal(shape=base_init.shape, key=sub)
            s_hat = base_init / mx.maximum(
                mx.linalg.norm(base_init, axis=-1, keepdims=True), 1e-9
            )
            xi_tan = xi - mx.sum(xi * s_hat, axis=-1, keepdims=True) * s_hat
            jittered = base_init + init_jitter * sid * xi_tan
            norms = mx.linalg.norm(jittered, axis=-1, keepdims=True)
            inits.append(jittered * (sid / mx.maximum(norms, 1e-6)))
        init_stack = mx.stack(inits, axis=0)  # (N, k, 3)

    sigma = cfg.gaussian_sigma()
    model = Free3D()
    # VCL context only when the VCL term is active (see free-sphere note).
    vcl_ctx = (
        _matched_vcl_context(
            volume, sid=sid, sdd=sdd,
            target_shape=vcl_target_shape, r1=vcl_r1,
            detector_shape=detector_shape,
            working_detector_shape=vcl_detector_shape,
            du=du, dv=dv, voxel_spacing=voxel_spacing,
            sample_seed=vcl_sample_seed, ridge=vcl_ridge,
            geometry_vjp_mode=vcl_geometry_vjp_mode,
            geometry_fd_step=vcl_geometry_fd_step,
            prefer_sparse_backprojection=prefer_sparse_backprojection,
        )
        if (lambda_vcl > 0.0 or lambda_oed > 0.0)
        else None
    )
    if lambda_oed > 0.0:
        # Photon-weighted OED exploration: members optimise the noise-aware
        # A+D score directly (differentiable in the sources through both the
        # view-basis correlation and the bundle line integral tau_bar).
        from ..oed import oed_loss_continuous
        from ..absorption_bundle import BundleAbsorptionConfig, bundle_path_integral
        _oed_bcfg = BundleAbsorptionConfig(
            roi_radius=0.0, n_rays_u=1, n_rays_v=1,
            n_samples=80, voxel_spacing=voxel_spacing,
        )
    if lambda_path > 0.0:
        abs_cfg = _build_absorption_config(sid, sdd, volume=volume, gate_type="path")
    else:
        abs_cfg = None

    def _base_objective(sources_n: mx.array) -> mx.array:
        """Per-member base objective (to maximise), no repulsion."""
        norms = mx.linalg.norm(sources_n, axis=-1, keepdims=True)
        s = sources_n * (sid / mx.maximum(norms, 1e-6))
        obj = mx.array(0.0, dtype=mx.float32)
        if lambda_cov > 0.0:
            obj = obj + lambda_cov * saturated_coverage(
                s, roi_center, radon_normals,
                mx.ones(s.shape[0], dtype=mx.float32),
                cfg,
                roi_points=roi_points,
                roi_weights=roi_weights,
            )
        if lambda_path > 0.0:
            nu = compute_absorption_gate(
                s, roi_center, volume, abs_cfg,
                grad_mode="fd_src", gate_type="path",
            )
            obj = obj + lambda_path * mx.mean(nu)
        if lambda_vcl > 0.0:
            vcl = vcl_loss_continuous(s, vcl_ctx)
            obj = obj + lambda_vcl * (1.0 - vcl)
        if lambda_oed > 0.0:
            tau_bar = bundle_path_integral(s, roi_center, volume, _oed_bcfg)
            obj = obj + lambda_oed * oed_loss_continuous(
                s, vcl_ctx, tau_bar,
                ridge=oed_ridge, alpha=oed_alpha, beta=oed_beta,
            )
        return obj

    def _chamfer_log(a: mx.array, b: mx.array) -> mx.array:
        """log of symmetric squared chamfer distance between two (k,3) sets."""
        # Pairwise squared distance (k, k).
        diff = a[:, None, :] - b[None, :, :]
        d2 = mx.sum(diff * diff, axis=-1)
        # Chamfer = mean(min over the other set), symmetric average.
        ch = 0.5 * (mx.mean(mx.min(d2, axis=1)) + mx.mean(mx.min(d2, axis=0)))
        return mx.log(ch + 1e-6)

    def _rep_weight_at(step: int) -> float:
        """Annealing schedule for repulsion strength."""
        if repulsion_schedule == "cosine" and n_steps > 1:
            t_frac = step / (n_steps - 1)
            return repulsion_weight * (_math_ens.cos(_math_ens.pi * t_frac / 2.0) ** 2)
        return repulsion_weight

    def ensemble_fn(params_stack, step):
        # params_stack: (N, k, 3).  Sum of base objectives + (annealed) repulsion.
        N = params_stack.shape[0]
        base_total = mx.array(0.0, dtype=mx.float32)
        for n in range(N):
            base_total = base_total + _base_objective(params_stack[n])
        base_mean = base_total / N
        lam_rep_t = _rep_weight_at(step)
        if lam_rep_t <= 0.0 or N < 2:
            return base_mean
        rep_total = mx.array(0.0, dtype=mx.float32)
        pair_count = 0
        for n in range(N):
            for m in range(n + 1, N):
                rep_total = rep_total + _chamfer_log(
                    params_stack[n], params_stack[m]
                )
                pair_count += 1
        rep_mean = rep_total / max(pair_count, 1)
        # adam_ascent MAXIMISES → bigger log d = better separation = +contribution.
        return base_mean + lam_rep_t * rep_mean

    def project_stack(p: mx.array) -> mx.array:
        n = mx.linalg.norm(p, axis=-1, keepdims=True)
        return p * (sid / mx.maximum(n, 1e-6))

    refined_stack, _ = adam_ascent(
        ensemble_fn, init_stack, n_steps=n_steps, lr=lr,
        project_fn=project_stack,
        lr_schedule="cosine", lr_min=lr * 0.05,
        patience=15, return_best=True,
    )
    mx.eval(refined_stack)

    # Pick the ensemble member with the best base objective (no repulsion).
    best_n, best_val = 0, float("-inf")
    base_vals = []
    for n in range(n_ensemble):
        v = float(_base_objective(refined_stack[n]))
        base_vals.append(v)
        if v > best_val:
            best_val = v
            best_n = n
    best = refined_stack[best_n]

    # Diagnostic hook: expose the full explored stack and its per-member base
    # (surrogate) objective values, so a caller can compare selection criteria
    # (surrogate vs. noise-aware vs. oracle reconstruction quality) over the
    # *same* explored basins.  Returns before the optional final-refine pass so
    # the members are the raw exploration output.
    if return_stack:
        return best, refined_stack, base_vals

    # Optional final-refine pass: pure Adam on the best member, no repulsion,
    # no noise.  Lets the chosen basin polish itself once exploration is done.
    if final_refine_steps > 0:
        def single_fn(p, _step):
            return _base_objective(p)
        def project_single(p):
            n = mx.linalg.norm(p, axis=-1, keepdims=True)
            return p * (sid / mx.maximum(n, 1e-6))
        best, _ = adam_ascent(
            single_fn, best, n_steps=final_refine_steps, lr=lr,
            project_fn=project_single,
            lr_schedule="cosine", lr_min=lr * 0.05,
            patience=15, return_best=True,
        )
        mx.eval(best)
    return best


def greedy_adam_absorption(
    k: int,
    sid: float,
    *,
    roi_center: mx.array,
    volume: mx.array,
    sdd: float = 900.0,
    grad_mode: str = "none",
    lambda_cov: float = 0.0,
    lambda_path: float = 0.0,
    icov_object_aware: bool = False,
    gate_type: str = "fraction",
    diversity_aware: bool = False,
    n_candidates: int = 200,
    n_normals: int = 2000,   # protocol value (2026-08-27): the 15 deg
                             # coverage bandwidth tiles the sphere with
                             # ~58 caps, so z must give many samples per
                             # cap; z=300 left ~5 and its sampling noise
                             # cost up to 3 dB on the flange.
    n_steps: int = 100,
    lr: float = 5.0,
    score_cfg: ScoreConfig | None = None,
    roi_points: mx.array | None = None,
    roi_weights: mx.array | None = None,
) -> mx.array:
    """Greedy warm-start + Adam refinement on the **absorption-aware** objective.

    Parameters
    ----------
    grad_mode
        ``"none"``  → stop-gradient absorption (ν computed but detached),
        ``"fd_src"`` → finite-difference VJP through cone_forward,
        ``"tangential"`` → FD VJP projected onto the viewing-sphere tangent.
    lambda_cov
        Weight of the optional VCL-inspired $I_{\\mathrm{cov}}$ regulariser
        (§VII).  ``0`` disables it.
    volume
        ``(nz, ny, nx)`` attenuation volume used to compute ν.
    """
    cfg = score_cfg or _default_score_config()
    radon_normals = sample_unit_sphere(n_normals)
    init = greedy_discrete(
        k, sid,
        roi_center=roi_center,
        n_candidates=n_candidates,
        score_cfg=cfg,
        n_normals=n_normals,
        roi_points=roi_points,
        roi_weights=roi_weights,
    )

    model = Free3D()
    abs_cfg = _build_absorption_config(sid, sdd, volume=volume, gate_type=gate_type)

    # Pre-compute object-aware Radon importance weights if requested.
    direction_weights = None
    if icov_object_aware and lambda_cov > 0.0:
        from .radon_importance import radon_importance_weights

        direction_weights = radon_importance_weights(volume, radon_normals)

    def coverage_fn(params, _step):
        sources = model(params)
        norms = mx.linalg.norm(sources, axis=-1, keepdims=True)
        sources = sources * (sid / mx.maximum(norms, 1e-6))
        nu = compute_absorption_gate(
            sources, roi_center, volume, abs_cfg,
            grad_mode=grad_mode, gate_type=gate_type,
        )
        if diversity_aware:
            # Diversity = full geometric coverage with no ν weighting.  ν enters
            # only as an additive short-path bonus so every view still
            # contributes to the saturation accumulator and the optimiser
            # cannot abandon long-path views to chase a small ν-bonus.
            cov = saturated_coverage(
                sources, roi_center, radon_normals,
                mx.ones(sources.shape[0], dtype=mx.float32),
                cfg,
                roi_points=roi_points,
                roi_weights=roi_weights,
            )
            loss = cov
            if lambda_path > 0.0:
                loss = loss + lambda_path * mx.mean(nu)
            if lambda_cov > 0.0:
                # Icov also uses uniform ones — pure diversity penalty.
                ones = mx.ones(sources.shape[0], dtype=mx.float32)
                icov = coverage_covariance_information(
                    sources, roi_center, radon_normals, ones, cfg,
                    direction_weights=direction_weights,
                )
                loss = loss + lambda_cov * icov
            return loss
        else:
            # Legacy multiplicative form (ν-weighted accumulator).  Kept for
            # backwards-compat with the existing ablation rows.
            cov = saturated_coverage(
                sources, roi_center, radon_normals, nu, cfg,
                roi_points=roi_points,
                roi_weights=roi_weights,
            )
            if lambda_cov > 0.0:
                icov = coverage_covariance_information(
                    sources, roi_center, radon_normals, nu, cfg,
                    direction_weights=direction_weights,
                )
                return cov + lambda_cov * icov
            return cov

    refined, _ = adam_ascent(coverage_fn, init, n_steps=n_steps, lr=lr)
    norms = mx.linalg.norm(refined, axis=-1, keepdims=True)
    refined = refined * (sid / mx.maximum(norms, 1e-6))
    return refined


def greedy_adam(
    k: int,
    sid: float,
    *,
    roi_center: mx.array,
    n_candidates: int = 200,
    n_normals: int = 2000,   # protocol value (2026-08-27): the 15 deg
                             # coverage bandwidth tiles the sphere with
                             # ~58 caps, so z must give many samples per
                             # cap; z=300 left ~5 and its sampling noise
                             # cost up to 3 dB on the flange.
    n_steps: int = 100,
    lr: float = 5.0,
    score_cfg: ScoreConfig | None = None,
    roi_points: mx.array | None = None,
    roi_weights: mx.array | None = None,
    sphere_seed: int | None = None,     # SO(3) lattice rotation; None = legacy
) -> mx.array:
    """Greedy warm-start + Adam refinement on the differentiable-coverage objective.

    This is the *method under test*.  We refine in Free3D parametrization with
    a soft sphere-radius re-projection at each step so the source stays on the
    same sphere as the baselines.
    """
    cfg = score_cfg or _default_score_config()
    radon_normals = sample_unit_sphere(n_normals)
    init = greedy_discrete(
        k,
        sid,
        roi_center=roi_center,
        n_candidates=n_candidates,
        score_cfg=cfg,
        n_normals=n_normals,
        sphere_seed=sphere_seed,
        roi_points=roi_points,
        roi_weights=roi_weights,
    )
    model = Free3D()

    def coverage_fn(params, _step):
        sources = model(params)
        # Re-project onto sphere of radius sid (soft, differentiable)
        norms = mx.linalg.norm(sources, axis=-1, keepdims=True)
        sources = sources * (sid / mx.maximum(norms, 1e-6))
        return saturated_coverage(
            sources, roi_center, radon_normals,
            mx.ones(sources.shape[0], dtype=mx.float32),
            cfg,
            roi_points=roi_points,
            roi_weights=roi_weights,
        )

    refined, _history = adam_ascent(
        coverage_fn,
        init,
        n_steps=n_steps,
        lr=lr,
    )
    # Final hard re-projection so the optimised set really sits on the sphere
    norms = mx.linalg.norm(refined, axis=-1, keepdims=True)
    refined = refined * (sid / mx.maximum(norms, 1e-6))
    return refined


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def build_baseline_sources(
    name: str,
    k: int,
    sid: float,
    *,
    roi_center: mx.array,
    seed: int = 0,
    score_cfg: ScoreConfig | None = None,
    vcl_precompute=None,
    volume: mx.array | None = None,
    sdd: float = 900.0,
    detector_shape: tuple[int, int] = (320, 320),
    du: float = 1.0,
    dv: float = 1.0,
    voxel_spacing: float = 1.0,
    n_candidates: int = 200,
    prefer_sparse_backprojection: bool = True,
    trace_dir: str | None = None,
    trace_tag: str | None = None,
    method_kwargs: dict | None = None,
) -> mx.array:
    """Return ``(k, 3)`` source positions for the named baseline.

    Parameters
    ----------
    name
        One of :data:`BASELINE_NAMES`.
    k
        Number of sources / views.
    sid
        Source-to-isocenter distance (sphere radius).
    roi_center
        ``(3,)`` ROI centre passed to the coverage objective for the greedy
        and Adam baselines.
    seed
        RNG seed for ``random_sphere`` and the VCLS swap search.
    vcl_precompute
        Optional :class:`differentiable_coverage.eval.vcl.VCLPrecompute`
        object containing ``(R, γ)`` for the candidate set.  Only required
        when ``name == "vcls"`` — pass ``None`` for the other baselines.
    """
    def _kw(defaults: dict) -> dict:
        return {**defaults, **method_kwargs}

    if name == "uniform_arc":
        return uniform_arc(k, sid)
    method_kwargs = method_kwargs or {}
    # ---- chart-consistent step size (locked 2026-08-27) -------------------
    # Adam normalises the gradient, so its learning rate IS the per-step
    # displacement measured in the units of the optimisation chart. The
    # gantry charts optimise (theta, phi) in radians, whereas the free-sphere
    # chart optimises Cartesian source positions in millimetres. The same
    # numerical learning rate is therefore only geometrically equivalent
    # after scaling the Cartesian chart by the sphere radius: the pullback
    # metric of the angular chart is sid^2, i.e. lr_cartesian = sid * lr_ang.
    # Equivalently, the Cartesian chart is optimised in units of sid. Without
    # this scaling lr=0.05 moves a pose by 0.0057 deg per step at sid=500 mm
    # (0.57 deg over the 100-step budget), far below both the 7 deg candidate
    # spacing and the 15 deg coverage bandwidth, and the selector never
    # leaves its initialisation. Explicit method_kwargs["lr"] still wins.
    if ("adam" in name and not name.endswith(("_two_axis", "_carm"))
            and "lr" not in method_kwargs):
        method_kwargs = {**method_kwargs, "lr": ANGULAR_LR * sid}
    if name == "random_sphere":
        return random_sphere(k, sid, seed=seed)
    if name == "greedy_discrete":
        return greedy_discrete(**_kw({
            "k": k, "sid": sid, "roi_center": roi_center,
            "score_cfg": score_cfg,
            # Selection-seed replication rotates the candidate lattice
            # (REV-P1-02); YAML kwargs can still override sphere_seed.
            "sphere_seed": seed,
        }))
    if name == "greedy_adam":
        return greedy_adam(**_kw({
            "k": k, "sid": sid, "roi_center": roi_center,
            "score_cfg": score_cfg, "n_candidates": n_candidates,
            "sphere_seed": seed,
        }))
    if name == "vcls":
        if vcl_precompute is None:
            raise ValueError(
                "VCLS baseline requires a precomputed (R, γ); "
                "pass vcl_precompute= or run via the eval runner."
            )
        from .vcl import vcls_select

        indices, _loss = vcls_select(vcl_precompute, k, seed=seed)
        return vcl_precompute.candidate_sources[mx.array(indices)]
    if name in ("greedy_adam_sg", "greedy_adam_fd",
                "greedy_adam_sg_icov", "greedy_adam_sg_icov_obj",
                "greedy_adam_path", "greedy_adam_path_icov",
                "greedy_adam_path_div", "greedy_adam_path_div_icov"):
        if volume is None:
            raise ValueError(
                f"Baseline '{name}' is absorption-aware and needs the "
                "phantom volume; pass volume= or run via the eval runner."
            )
        if name == "greedy_adam_sg":
            return greedy_adam_absorption(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center, "volume": volume, "sdd": sdd,
                "grad_mode": "none", "lambda_cov": 0.0, "score_cfg": score_cfg,
            }))
        if name == "greedy_adam_fd":
            return greedy_adam_absorption(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center, "volume": volume, "sdd": sdd,
                "grad_mode": "fd_src", "lambda_cov": 0.0, "score_cfg": score_cfg,
            }))
        if name == "greedy_adam_sg_icov":
            return greedy_adam_absorption(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center, "volume": volume, "sdd": sdd,
                "grad_mode": "none", "lambda_cov": 0.05, "score_cfg": score_cfg,
            }))
        if name == "greedy_adam_sg_icov_obj":
            return greedy_adam_absorption(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center, "volume": volume, "sdd": sdd,
                "grad_mode": "none", "lambda_cov": 0.05,
                "icov_object_aware": True, "gate_type": "mean",
                "score_cfg": score_cfg,
            }))
        if name == "greedy_adam_path":
            return greedy_adam_absorption(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center, "volume": volume, "sdd": sdd,
                "grad_mode": "none", "lambda_cov": 0.0,
                "gate_type": "path", "score_cfg": score_cfg,
            }))
        if name == "greedy_adam_path_icov":
            return greedy_adam_absorption(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center, "volume": volume, "sdd": sdd,
                "grad_mode": "none", "lambda_cov": 0.05,
                "gate_type": "path", "score_cfg": score_cfg,
            }))
        if name == "greedy_adam_path_div":
            return greedy_adam_absorption(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center, "volume": volume, "sdd": sdd,
                "grad_mode": "fd_src", "gate_type": "path",
                "diversity_aware": True,
                "lambda_path": 0.3, "lambda_cov": 0.0,
                "score_cfg": score_cfg,
            }))
        if name == "greedy_adam_path_div_icov":
            return greedy_adam_absorption(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center, "volume": volume, "sdd": sdd,
                "grad_mode": "fd_src", "gate_type": "path",
                "diversity_aware": True,
                "lambda_path": 0.3, "lambda_cov": 0.05,
                "score_cfg": score_cfg,
            }))
    if name in ("greedy_adam_vcl", "greedy_adam_vcl_pure", "greedy_adam_all"):
        if volume is None:
            raise ValueError(
                f"Baseline '{name}' needs the phantom volume for the VCL term."
            )
        if name == "greedy_adam_vcl":
            return greedy_adam_vcl_continuous(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center,
                "volume": volume, "sdd": sdd,
                "lambda_cov": 1.0, "lambda_vcl": 0.2, "lambda_path": 0.0,
                "score_cfg": score_cfg, "seed": seed,
                "detector_shape": detector_shape,
                "du": du, "dv": dv, "voxel_spacing": voxel_spacing,
                "prefer_sparse_backprojection": prefer_sparse_backprojection,
                "trace_dir": trace_dir, "trace_tag": trace_tag,
            }))
        if name == "greedy_adam_vcl_pure":
            return greedy_adam_vcl_continuous(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center,
                "volume": volume, "sdd": sdd,
                "lambda_cov": 0.0, "lambda_vcl": 1.0, "lambda_path": 0.0,
                "score_cfg": score_cfg, "seed": seed,
                "detector_shape": detector_shape,
                "du": du, "dv": dv, "voxel_spacing": voxel_spacing,
                "prefer_sparse_backprojection": prefer_sparse_backprojection,
                "trace_dir": trace_dir, "trace_tag": trace_tag,
            }))
        if name == "greedy_adam_all":
            return greedy_adam_vcl_continuous(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center,
                "volume": volume, "sdd": sdd,
                "lambda_cov": 1.0, "lambda_vcl": 0.2, "lambda_path": 0.2,
                "score_cfg": score_cfg, "seed": seed,
                "detector_shape": detector_shape,
                "du": du, "dv": dv, "voxel_spacing": voxel_spacing,
                "prefer_sparse_backprojection": prefer_sparse_backprojection,
                "trace_dir": trace_dir, "trace_tag": trace_tag,
            }))
    if name == "greedy_adam_vcl_two_axis":
        if volume is None:
            raise ValueError(
                f"Baseline '{name}' needs the phantom volume for the VCL term."
            )
        return two_axis_gantry_vcl_continuous(
            k, sid, roi_center=roi_center, volume=volume, sdd=sdd,
            lambda_cov=1.0, lambda_vcl=0.2, lambda_path=0.0,
            score_cfg=score_cfg, seed=seed,
            detector_shape=detector_shape,
            du=du, dv=dv, voxel_spacing=voxel_spacing,
            prefer_sparse_backprojection=prefer_sparse_backprojection,
            trace_dir=trace_dir, trace_tag=trace_tag,
            **method_kwargs,
        )
    if name == "greedy_adam_vcl_carm":
        if volume is None:
            raise ValueError(
                f"Baseline '{name}' needs the phantom volume for the VCL term."
            )
        return two_axis_gantry_vcl_continuous(
            k, sid, roi_center=roi_center, volume=volume, sdd=sdd,
            lambda_cov=1.0, lambda_vcl=0.2, lambda_path=0.0,
            score_cfg=score_cfg, seed=seed, limited_carm=True,
            detector_shape=detector_shape,
            du=du, dv=dv, voxel_spacing=voxel_spacing,
            prefer_sparse_backprojection=prefer_sparse_backprojection,
            trace_dir=trace_dir, trace_tag=trace_tag,
            **method_kwargs,
        )
    if name in ("vcls_adam_vcl", "vcls_adam_geo"):
        if volume is None:
            raise ValueError(
                f"Baseline '{name}' needs the phantom volume for the VCLS "
                "warm-start and the differentiable VCL term."
            )
        if name == "vcls_adam_vcl":
            return greedy_adam_vcl_continuous(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center,
                "volume": volume, "sdd": sdd,
                "lambda_cov": 1.0, "lambda_vcl": 0.2, "lambda_path": 0.0,
                "init_method": "vcls",
                "score_cfg": score_cfg,
                "detector_shape": detector_shape,
                "du": du, "dv": dv, "voxel_spacing": voxel_spacing,
                "n_candidates": n_candidates,
                "vcls_precompute": vcl_precompute,
                "prefer_sparse_backprojection": prefer_sparse_backprojection,
                "seed": seed,
                "trace_dir": trace_dir, "trace_tag": trace_tag,
            }))
        if name == "vcls_adam_geo":
            return greedy_adam_vcl_continuous(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center,
                "volume": volume, "sdd": sdd,
                "lambda_cov": 1.0, "lambda_vcl": 0.0, "lambda_path": 0.0,
                "init_method": "vcls",
                "score_cfg": score_cfg,
                "lr": 5.0,
                "detector_shape": detector_shape,
                "du": du, "dv": dv, "voxel_spacing": voxel_spacing,
                "n_candidates": n_candidates,
                "vcls_precompute": vcl_precompute,
                "prefer_sparse_backprojection": prefer_sparse_backprojection,
                "seed": seed,
                "trace_dir": trace_dir, "trace_tag": trace_tag,
            }))
    if name == "vcls_adam_vcl_two_axis":
        if volume is None:
            raise ValueError(
                f"Baseline '{name}' needs the phantom volume for the VCLS "
                "warm-start and the differentiable VCL term."
            )
        return two_axis_gantry_vcl_continuous(
            k, sid, roi_center=roi_center, volume=volume, sdd=sdd,
            lambda_cov=1.0, lambda_vcl=0.2, lambda_path=0.0,
            init_method="vcls",
            score_cfg=score_cfg,
            detector_shape=detector_shape,
            du=du, dv=dv, voxel_spacing=voxel_spacing,
            n_candidates=n_candidates,
            vcls_precompute=vcl_precompute,
            prefer_sparse_backprojection=prefer_sparse_backprojection,
            seed=seed,
            trace_dir=trace_dir, trace_tag=trace_tag,
            **method_kwargs,
        )
    if name == "vcls_adam_vcl_carm":
        if volume is None:
            raise ValueError(
                f"Baseline '{name}' needs the phantom volume for the VCLS "
                "warm-start and the differentiable VCL term."
            )
        return two_axis_gantry_vcl_continuous(
            k, sid, roi_center=roi_center, volume=volume, sdd=sdd,
            lambda_cov=1.0, lambda_vcl=0.2, lambda_path=0.0,
            init_method="vcls",
            score_cfg=score_cfg,
            detector_shape=detector_shape,
            du=du, dv=dv, voxel_spacing=voxel_spacing,
            n_candidates=n_candidates,
            vcls_precompute=vcl_precompute,
            prefer_sparse_backprojection=prefer_sparse_backprojection,
            seed=seed, limited_carm=True,
            trace_dir=trace_dir, trace_tag=trace_tag,
            **method_kwargs,
        )
    if name in ("vcls_adam_bundle_center", "vcls_adam_bundle",
                "vcls_adam_bundle_fd", "vcls_cmaes_bundle",
                # Cold (Tuy-greedy) twins of the matched derivative and
                # optimiser ablation, added 2026-08-27 when the paper moved
                # to cold-start-only continuous selection.
                "greedy_adam_bundle_fd", "greedy_cmaes_bundle",
                "greedy_adam_bundle_center", "greedy_adam_bundle",
                "greedy_adam_composite",
                "uniform_adam_bundle",
                "greedy_adam_bundle_two_axis", "vcls_adam_bundle_two_axis",
                "greedy_adam_bundle_carm", "vcls_adam_bundle_carm"):
        if volume is None:
            raise ValueError(
                f"Baseline '{name}' needs the phantom volume for the bundle gate."
            )
        from ..absorption_bundle import (
            BundleAbsorptionConfig, bundle_path_integral,
        )
        if name.startswith("vcls_"):
            init_for = "vcls"
        elif name.startswith("uniform_"):
            init_for = "uniform_sphere"
        else:
            init_for = "greedy_tuy"
        is_center = name.endswith("_center")
        is_two_axis = name.endswith("_two_axis")
        is_carm = name.endswith("_carm")
        # ``vcls_adam_bundle_fd`` is the matched derivative-only twin of
        # ``vcls_adam_bundle`` (REV-P0-01): identical objective, rays,
        # calibration, init, optimizer, schedule, and stopping — only the
        # gradient estimator switches to central finite differences.
        # ``vcls_cmaes_bundle`` is the derivative-free twin (REV-P1-08):
        # identical objective/init/projection, CMA-ES instead of Adam.
        is_fd = name.endswith("_bundle_fd")
        is_cmaes = "cmaes" in name
        # Quadrature knobs (YAML-expressible; the frozen production rule is
        # clip@256 per the bundle_quadrature_convergence study).
        method_kwargs = dict(method_kwargs)
        bundle_n_samples = int(method_kwargs.pop("bundle_n_samples", 32))
        bundle_clip = bool(method_kwargs.pop("bundle_clip_to_volume", False))
        bcfg = BundleAbsorptionConfig(
            roi_radius=0.0 if is_center else 5.0,
            n_rays_u=1 if is_center else 5,
            n_rays_v=1 if is_center else 9,
            n_samples=bundle_n_samples,
            voxel_spacing=float(voxel_spacing),
            clip_to_volume=bundle_clip,
        )
        # Calibrate λ_bundle so the penalty has comparable scale to the
        # coverage term (≈ 1) at the median direction.  Compute median τ̄
        # on a uniform sphere probe and set λ = 0.2 / median(τ̄).
        from ..score import sample_unit_sphere
        # Isocentre-centred probe sphere (the source manifold) — see the
        # calibration comment in greedy_adam_vcl_continuous.
        probes = sample_unit_sphere(256) * sid
        _cal_target = method_kwargs.get("bundle_target")
        tau_probe = bundle_path_integral(
            probes,
            roi_center if _cal_target is None else _cal_target,
            volume, bcfg)
        med_tau = float(mx.median(tau_probe))
        lam_bundle = calibrate_bundle_weight(med_tau)
        if is_two_axis or is_carm:
            return two_axis_gantry_vcl_continuous(
                **_kw({
                    "k": k, "sid": sid, "roi_center": roi_center,
                    "volume": volume, "sdd": sdd,
                    "lambda_cov": 1.0, "lambda_vcl": 0.2, "lambda_path": 0.0,
                    "lambda_bundle": lam_bundle, "bundle_cfg": bcfg,
                    "init_method": init_for,
                    "score_cfg": score_cfg,
                    "detector_shape": detector_shape,
                    "du": du, "dv": dv,
                    "n_candidates": n_candidates,
                    "vcls_precompute": vcl_precompute,
                    "voxel_spacing": voxel_spacing,
                    "prefer_sparse_backprojection": prefer_sparse_backprojection,
                    "seed": seed, "limited_carm": is_carm,
                    "trace_dir": trace_dir, "trace_tag": trace_tag,
                })
            )
        return greedy_adam_vcl_continuous(**_kw({
            "k": k, "sid": sid, "roi_center": roi_center,
            "volume": volume, "sdd": sdd,
            "lambda_cov": 1.0,
            "lambda_vcl": 0.2 if name == "greedy_adam_composite" else 0.0,
            "lambda_path": 0.0,
            "derivative_mode": "fd_central" if is_fd else "analytic",
            "optimizer": "cmaes" if is_cmaes else "adam",
            "lambda_bundle": lam_bundle, "bundle_cfg": bcfg,
            "init_method": init_for,
            "score_cfg": score_cfg,
            "detector_shape": detector_shape,
            "du": du, "dv": dv, "voxel_spacing": voxel_spacing,
            "n_candidates": n_candidates,
            "vcls_precompute": vcl_precompute,
            "prefer_sparse_backprojection": prefer_sparse_backprojection,
            "seed": seed,
            "trace_dir": trace_dir, "trace_tag": trace_tag,
        }))
    if name in (
        "vcls_adam_anneal", "vcls_adam_langevin", "vcls_adam_ensemble",
        "greedy_adam_vcl_anneal", "greedy_adam_vcl_langevin",
        "greedy_adam_vcl_ensemble",
    ):
        if volume is None:
            raise ValueError(
                f"Baseline '{name}' needs the phantom volume for the VCL term."
            )
        init_for = "vcls" if name.startswith("vcls_") else "greedy_tuy"
        path_weight = 0.2 if name.startswith("vcls_") else 0.0
        if name.endswith("_anneal"):
            return greedy_adam_vcl_continuous(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center,
                "volume": volume, "sdd": sdd,
                "lambda_cov": 1.0, "lambda_vcl": 0.2,
                "lambda_path": path_weight,
                "init_method": init_for,
                "score_cfg": score_cfg,
                "cov_decay": "cosine",
                "cov_sigma_schedule": "cosine",
                "cov_sigma_mult": 4.0,
                "detector_shape": detector_shape,
                "du": du, "dv": dv, "voxel_spacing": voxel_spacing,
                "n_candidates": n_candidates,
                "vcls_precompute": vcl_precompute,
                "prefer_sparse_backprojection": prefer_sparse_backprojection,
                "seed": seed,
                "trace_dir": trace_dir, "trace_tag": trace_tag,
            }))
        if name.endswith("_langevin"):
            return greedy_adam_vcl_continuous(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center,
                "volume": volume, "sdd": sdd,
                "lambda_cov": 1.0, "lambda_vcl": 0.2,
                "lambda_path": path_weight,
                "init_method": init_for,
                "score_cfg": score_cfg,
                "noise_schedule": "langevin_cosine",
                "noise_temp": 0.02,
                "noise_seed": seed,
                "detector_shape": detector_shape,
                "du": du, "dv": dv, "voxel_spacing": voxel_spacing,
                "n_candidates": n_candidates,
                "vcls_precompute": vcl_precompute,
                "prefer_sparse_backprojection": prefer_sparse_backprojection,
                "seed": seed,
                "trace_dir": trace_dir, "trace_tag": trace_tag,
            }))
        if name.endswith("_ensemble"):
            return greedy_ensemble_vcl_continuous(**_kw({
                "k": k, "sid": sid, "roi_center": roi_center,
                "volume": volume, "sdd": sdd,
                "lambda_cov": 1.0, "lambda_vcl": 0.2,
                "lambda_path": path_weight,
                "init_method": init_for,
                "score_cfg": score_cfg,
                "n_ensemble": 4,
                "repulsion_weight": 0.1,
                "init_jitter": 0.15,
                "ensemble_seed": seed,
                "detector_shape": detector_shape,
                "du": du, "dv": dv, "voxel_spacing": voxel_spacing,
                "n_candidates": n_candidates,
                "vcls_precompute": vcl_precompute,
                "prefer_sparse_backprojection": prefer_sparse_backprojection,
            }))
    if name in ("greedy_adam_icov_fft", "vcls_adam_icov_fft"):
        if volume is None:
            raise ValueError(
                f"Baseline '{name}' needs the phantom volume for the "
                "Fourier-slice Radon importance and the bundle term."
            )
        from ..absorption_bundle import (
            BundleAbsorptionConfig, bundle_path_integral,
        )
        from ..score import sample_unit_sphere
        bcfg = BundleAbsorptionConfig(
            roi_radius=5.0, n_rays_u=5, n_rays_v=9, n_samples=32,
            voxel_spacing=float(voxel_spacing),
        )
        # Isocentre-centred probe sphere (the source manifold) — see the
        # calibration comment in greedy_adam_vcl_continuous.
        probes = sample_unit_sphere(256) * sid
        _cal_target = method_kwargs.get("bundle_target")
        tau_probe = bundle_path_integral(
            probes,
            roi_center if _cal_target is None else _cal_target,
            volume, bcfg)
        med_tau = float(mx.median(tau_probe))
        lam_bundle = calibrate_bundle_weight(med_tau)
        init_for = "vcls" if name.startswith("vcls_") else "greedy_tuy"
        return greedy_adam_vcl_continuous(
            k, sid, roi_center=roi_center, volume=volume, sdd=sdd,
            lambda_cov=1.0, lambda_vcl=0.0, lambda_path=0.0,
            lambda_icov_fft=0.2,
            lambda_bundle=lam_bundle, bundle_cfg=bcfg,
            init_method=init_for,
            score_cfg=score_cfg,
            detector_shape=detector_shape,
            du=du, dv=dv, voxel_spacing=voxel_spacing,
            n_candidates=n_candidates,
            vcls_precompute=vcl_precompute,
            prefer_sparse_backprojection=prefer_sparse_backprojection,
            seed=seed,
            trace_dir=trace_dir, trace_tag=trace_tag,
            **method_kwargs,
        )
    if name in ("greedy_adam_oed", "vcls_adam_oed"):
        # Photon-noise-weighted optimal-experimental-design selector.
        # Objective = coverage + λ_oed · (A + D optimality on the photon-
        # weighted view Fisher information).  No separate VCL or bundle
        # penalty: the photon survival weight already carries the absorption.
        if volume is None:
            raise ValueError(
                f"Baseline '{name}' needs the phantom volume for the OED term."
            )
        from ..absorption_bundle import BundleAbsorptionConfig
        # Quadrature knobs as in the bundle branch (frozen production rule:
        # clip@256).  The OED photon weights exp(-τ̄) are especially
        # sensitive to the legacy full-segment border-clamp bias.
        method_kwargs = dict(method_kwargs)
        bundle_n_samples = int(method_kwargs.pop("bundle_n_samples", 32))
        bundle_clip = bool(method_kwargs.pop("bundle_clip_to_volume", False))
        bcfg = BundleAbsorptionConfig(
            roi_radius=5.0, n_rays_u=5, n_rays_v=9,
            n_samples=bundle_n_samples,
            voxel_spacing=float(voxel_spacing),
            clip_to_volume=bundle_clip,
        )
        init_for = "vcls" if name.startswith("vcls_") else "greedy_tuy"
        return greedy_adam_vcl_continuous(**_kw({
            "k": k, "sid": sid, "roi_center": roi_center,
            "volume": volume, "sdd": sdd,
            "lambda_cov": 1.0, "lambda_vcl": 0.0, "lambda_path": 0.0,
            "lambda_oed": 0.2, "bundle_cfg": bcfg,
            "init_method": init_for,
            "score_cfg": score_cfg,
            "detector_shape": detector_shape,
            "du": du, "dv": dv,
            "n_candidates": n_candidates,
            "vcls_precompute": vcl_precompute,
            "voxel_spacing": voxel_spacing,
            "prefer_sparse_backprojection": prefer_sparse_backprojection,
            "seed": seed,
            "trace_dir": trace_dir, "trace_tag": trace_tag,
        }))
    if name == "multistart_adam":
        if volume is None:
            raise ValueError("multistart_adam needs the phantom volume.")
        return multistart_adam_sources(**_kw({
            "k": k, "sid": sid, "roi_center": roi_center, "volume": volume, "sdd": sdd,
            "score_cfg": score_cfg,
            "detector_shape": detector_shape,
            "du": du, "dv": dv, "voxel_spacing": voxel_spacing,
        }))
    if name == "vcls_circle":
        # vcls_circle restricts the candidate pool to the equatorial
        # circle (Lin et al.\ matched-orbit baseline).  The (R, γ) cache
        # that the runner builds is sphere-based, so we always rebuild a
        # circle-restricted cache here even if `vcl_precompute` was
        # supplied (its sphere candidates would silently bypass the
        # circle restriction and make vcls_circle indistinguishable
        # from vcls).
        if volume is None:
            raise ValueError(
                "vcls_circle needs the phantom volume to build a "
                "circle-restricted (R, γ) cache."
            )
        from .vcl import compute_R_gamma, vcls_select

        circle_pre = compute_R_gamma(
            volume, _circle_candidates(n_candidates, sid),
            sid=sid, sdd=sdd, det_shape=detector_shape,
            du=du, dv=dv, voxel_spacing=voxel_spacing,
            r1=1e-3, seed=seed,
            prefer_sparse_backprojection=prefer_sparse_backprojection,
        )
        indices, _loss = vcls_select(circle_pre, k, seed=seed)
        return circle_pre.candidate_sources[mx.array(indices)]
    if name == "greedy_adam_circle":
        return greedy_adam_circle(**_kw({
            "k": k, "sid": sid, "roi_center": roi_center, "score_cfg": score_cfg,
        }))
    if name == "greedy_adam_path_div_circle":
        if volume is None:
            raise ValueError("greedy_adam_path_div_circle needs the phantom volume.")
        return greedy_adam_path_div_circle(**_kw({
            "k": k, "sid": sid, "roi_center": roi_center, "volume": volume, "sdd": sdd,
            "score_cfg": score_cfg,
        }))
    raise ValueError(f"Unknown baseline '{name}'. Choose from {BASELINE_NAMES}.")
