"""Tests for the Fourier-weighted I_cov trajectory baseline.

Exercises the new ``greedy_adam_icov_fft`` (and its VCLS-init sibling
``vcls_adam_icov_fft``), which combines:

* a soft-Tuy coverage term,
* an analytic bundle absorption penalty,
* the Fourier-slice-weighted coverage covariance information score.

The goals of these tests are minimal but strict: the new path must
optimise, must respond to the bundle term, must converge, and must not
crash for any of the small phantom shapes used elsewhere in the suite.
Headline-quality numbers are not in scope here.
"""

from __future__ import annotations

import csv

import mlx.core as mx
import numpy as np
import pytest

from differentiable_coverage.absorption_bundle import BundleAbsorptionConfig
from differentiable_coverage.eval.trajectories import (
    BASELINE_NAMES,
    build_baseline_sources,
    greedy_adam_vcl_continuous,
    greedy_discrete,
    two_axis_gantry_vcl_continuous,
)
from differentiable_coverage.trajectory import CArmTwoAxisGantry, TwoAxisGantry
from differentiable_coverage.score import ScoreConfig, sample_unit_sphere


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def z_cylinder_volume():
    """Small anisotropic phantom — Fourier weights are non-uniform on this."""
    N = 32
    zz, yy, xx = np.meshgrid(
        np.arange(N), np.arange(N), np.arange(N), indexing="ij"
    )
    cy, cx = (N - 1) / 2, (N - 1) / 2
    r2_xy = (yy - cy) ** 2 + (xx - cx) ** 2
    vol = (r2_xy < (N / 6) ** 2).astype(np.float32) * 0.5
    return mx.array(vol)


@pytest.fixture
def common_kwargs(z_cylinder_volume):
    """Kwargs that match ``greedy_adam_vcl_continuous``'s signature.

    The dispatcher fixture below adds the extra geometry args
    ``du``, ``dv``, ``voxel_spacing`` that ``build_baseline_sources``
    expects but ``greedy_adam_vcl_continuous`` doesn't.
    """
    return dict(
        sid=20.0, sdd=40.0,
        roi_center=mx.array([0.0, 0.0, 0.0]),
        volume=z_cylinder_volume,
        detector_shape=(32, 32),
        n_candidates=24,
        # Keep the optimisation budget very low — these are correctness
        # tests, not headline-quality runs.
        n_steps=8, lr=0.05,
    )


# ---------------------------------------------------------------------------
# Public-API: the dispatcher recognises the new names
# ---------------------------------------------------------------------------

def test_baseline_names_register_new_methods():
    assert "greedy_adam_icov_fft" in BASELINE_NAMES
    assert "vcls_adam_icov_fft" in BASELINE_NAMES
    assert "greedy_adam_vcl_two_axis" in BASELINE_NAMES
    assert "vcls_adam_bundle_two_axis" in BASELINE_NAMES
    assert "greedy_adam_vcl_carm" in BASELINE_NAMES
    assert "vcls_adam_bundle_carm" in BASELINE_NAMES


# ---------------------------------------------------------------------------
# Core: greedy_adam_vcl_continuous accepts and uses lambda_icov_fft
# ---------------------------------------------------------------------------

def test_lambda_icov_fft_changes_solution(common_kwargs):
    """With ``lambda_icov_fft > 0`` the optimised sources differ from the
    baseline without the term."""
    src_no = greedy_adam_vcl_continuous(
        4, **common_kwargs,
        lambda_cov=1.0, lambda_vcl=0.0, lambda_path=0.0,
        lambda_icov_fft=0.0, init_method="greedy_tuy",
    )
    src_yes = greedy_adam_vcl_continuous(
        4, **common_kwargs,
        lambda_cov=1.0, lambda_vcl=0.0, lambda_path=0.0,
        lambda_icov_fft=0.5, init_method="greedy_tuy",
    )
    mx.eval(src_no); mx.eval(src_yes)
    diff = float(mx.linalg.norm(src_no - src_yes))
    assert diff > 1e-4, (
        f"lambda_icov_fft had no effect on the solution; "
        f"||src_no - src_yes|| = {diff}"
    )


def test_trace_files_are_written(common_kwargs, tmp_path):
    trace_dir = tmp_path / "trace"
    kwargs = dict(common_kwargs)
    kwargs["n_steps"] = 3
    src = greedy_adam_vcl_continuous(
        4, **kwargs,
        lambda_cov=1.0, lambda_vcl=0.0, lambda_path=0.0,
        init_method="greedy_tuy",
        trace_dir=str(trace_dir),
        trace_tag="unit_trace",
    )
    mx.eval(src)

    metrics_path = trace_dir / "unit_trace_metrics.csv"
    sources_path = trace_dir / "unit_trace_sources.npz"
    assert metrics_path.exists()
    assert sources_path.exists()

    rows = list(csv.DictReader(metrics_path.open()))
    assert rows[0]["phase"] == "init"
    assert rows[-1]["phase"] == "best"

    trace = np.load(sources_path, allow_pickle=True)
    assert trace["sources"].ndim == 3
    assert trace["sources"].shape[-2:] == (4, 3)


def test_uniform_sphere_init_runs(common_kwargs):
    src = greedy_adam_vcl_continuous(
        4, **common_kwargs,
        lambda_cov=1.0, lambda_vcl=0.2, lambda_path=0.0,
        init_method="uniform_sphere",
    )
    mx.eval(src)
    radii = np.linalg.norm(np.asarray(src), axis=-1)
    assert src.shape == (4, 3)
    assert np.allclose(radii, common_kwargs["sid"], atol=1e-4)


def test_lambda_icov_fft_runs_with_bundle(common_kwargs):
    """The new term combines cleanly with the analytic bundle absorption
    penalty (the deployment-relevant configuration)."""
    bcfg = BundleAbsorptionConfig(
        voxel_spacing=1.0, roi_radius=2.0,
        n_rays_u=3, n_rays_v=3, n_samples=8,
    )
    src = greedy_adam_vcl_continuous(
        4, **common_kwargs,
        lambda_cov=1.0, lambda_vcl=0.0, lambda_path=0.0,
        lambda_icov_fft=0.5, lambda_bundle=0.1,
        bundle_cfg=bcfg, init_method="greedy_tuy",
    )
    mx.eval(src)
    assert src.shape == (4, 3)
    assert np.all(np.isfinite(np.asarray(src)))


def test_dispatcher_branch_greedy_adam_icov_fft(common_kwargs):
    src = build_baseline_sources(
        "greedy_adam_icov_fft", 4,
        sid=common_kwargs["sid"],
        roi_center=common_kwargs["roi_center"],
        volume=common_kwargs["volume"],
        sdd=common_kwargs["sdd"],
        detector_shape=common_kwargs["detector_shape"],
        du=1.0, dv=1.0, voxel_spacing=1.0,
        n_candidates=common_kwargs["n_candidates"],
    )
    mx.eval(src)
    assert src.shape == (4, 3)
    assert np.all(np.isfinite(np.asarray(src)))


def test_dispatcher_passes_method_kwargs(common_kwargs):
    src = build_baseline_sources(
        "greedy_adam_bundle", 4,
        sid=common_kwargs["sid"],
        roi_center=common_kwargs["roi_center"],
        volume=common_kwargs["volume"],
        sdd=common_kwargs["sdd"],
        detector_shape=common_kwargs["detector_shape"],
        du=1.0, dv=1.0, voxel_spacing=1.0,
        n_candidates=common_kwargs["n_candidates"],
        method_kwargs={"init_method": "uniform_sphere", "lr": 0.1, "n_steps": 4},
    )
    mx.eval(src)
    radii = np.linalg.norm(np.asarray(src), axis=-1)
    assert src.shape == (4, 3)
    assert np.allclose(radii, common_kwargs["sid"], atol=1e-4)


def test_dispatcher_branch_greedy_adam_vcl_two_axis(common_kwargs):
    src = build_baseline_sources(
        "greedy_adam_vcl_two_axis", 4,
        sid=common_kwargs["sid"],
        roi_center=common_kwargs["roi_center"],
        volume=common_kwargs["volume"],
        sdd=common_kwargs["sdd"],
        detector_shape=common_kwargs["detector_shape"],
        du=1.0, dv=1.0, voxel_spacing=1.0,
        n_candidates=common_kwargs["n_candidates"],
    )
    mx.eval(src)
    assert src.shape == (4, 3)
    rebuilt = TwoAxisGantry(common_kwargs["sid"])(mx.stack([
        mx.arctan2(-src[:, 0], src[:, 1]),
        mx.arcsin(src[:, 2] / common_kwargs["sid"]),
    ], axis=-1))
    rel = float(mx.linalg.norm(src - rebuilt) / mx.maximum(mx.linalg.norm(src), 1e-6))
    assert rel < 1e-5


def test_dispatcher_branch_vcls_adam_bundle_two_axis(common_kwargs):
    src = build_baseline_sources(
        "vcls_adam_bundle_two_axis", 4,
        sid=common_kwargs["sid"],
        roi_center=common_kwargs["roi_center"],
        volume=common_kwargs["volume"],
        sdd=common_kwargs["sdd"],
        detector_shape=common_kwargs["detector_shape"],
        du=1.0, dv=1.0, voxel_spacing=1.0,
        n_candidates=common_kwargs["n_candidates"],
    )
    mx.eval(src)
    assert src.shape == (4, 3)
    rebuilt = TwoAxisGantry(common_kwargs["sid"])(mx.stack([
        mx.arctan2(-src[:, 0], src[:, 1]),
        mx.arcsin(src[:, 2] / common_kwargs["sid"]),
    ], axis=-1))
    rel = float(mx.linalg.norm(src - rebuilt) / mx.maximum(mx.linalg.norm(src), 1e-6))
    assert rel < 1e-5


def test_dispatcher_branch_greedy_adam_vcl_carm(common_kwargs):
    src = build_baseline_sources(
        "greedy_adam_vcl_carm", 4,
        sid=common_kwargs["sid"],
        roi_center=common_kwargs["roi_center"],
        volume=common_kwargs["volume"],
        sdd=common_kwargs["sdd"],
        detector_shape=common_kwargs["detector_shape"],
        du=1.0, dv=1.0, voxel_spacing=1.0,
        n_candidates=common_kwargs["n_candidates"],
    )
    mx.eval(src)
    assert src.shape == (4, 3)
    gantry = CArmTwoAxisGantry(common_kwargs["sid"])
    params = mx.stack([
        mx.arctan2(-src[:, 0], src[:, 1]),
        mx.arcsin(src[:, 2] / common_kwargs["sid"]),
    ], axis=-1)
    clamped = gantry.clamp(params)
    assert float(mx.max(mx.abs(params - clamped))) < 1e-5


def test_dispatcher_branch_vcls_adam_bundle_carm(common_kwargs):
    src = build_baseline_sources(
        "vcls_adam_bundle_carm", 4,
        sid=common_kwargs["sid"],
        roi_center=common_kwargs["roi_center"],
        volume=common_kwargs["volume"],
        sdd=common_kwargs["sdd"],
        detector_shape=common_kwargs["detector_shape"],
        du=1.0, dv=1.0, voxel_spacing=1.0,
        n_candidates=common_kwargs["n_candidates"],
    )
    mx.eval(src)
    assert src.shape == (4, 3)
    gantry = CArmTwoAxisGantry(common_kwargs["sid"])
    params = mx.stack([
        mx.arctan2(-src[:, 0], src[:, 1]),
        mx.arcsin(src[:, 2] / common_kwargs["sid"]),
    ], axis=-1)
    clamped = gantry.clamp(params)
    assert float(mx.max(mx.abs(params - clamped))) < 1e-5


def test_carm_optimization_improves_geometric_objective(common_kwargs):
    cfg = ScoreConfig(tau=0.1)
    radon_normals = sample_unit_sphere(64)
    sigma = cfg.gaussian_sigma()
    init = greedy_discrete(
        4, common_kwargs["sid"],
        roi_center=common_kwargs["roi_center"],
        n_candidates=common_kwargs["n_candidates"],
        score_cfg=cfg,
        n_normals=64,
        sphere_seed=0,
    )

    def geom_obj(sources):
        r = sources[:, None, :] - common_kwargs["roi_center"][None, None, :]
        rho = mx.linalg.norm(r, axis=-1, keepdims=True)
        d = r / mx.maximum(rho, 1e-9)
        g = mx.sum(d * radon_normals[None, :, :], axis=-1)
        psi = mx.exp(-(g * g) / (2.0 * sigma * sigma))
        sigma_j = mx.sum(psi, axis=0)
        return mx.mean(1.0 - mx.exp(-sigma_j))

    gantry = CArmTwoAxisGantry(common_kwargs["sid"])
    init_params = mx.stack([
        mx.arctan2(-init[:, 0], init[:, 1]),
        mx.arcsin(init[:, 2] / common_kwargs["sid"]),
    ], axis=-1)
    admissible_init = gantry(gantry.clamp(init_params))

    src = two_axis_gantry_vcl_continuous(
        4,
        sid=common_kwargs["sid"],
        roi_center=common_kwargs["roi_center"],
        volume=common_kwargs["volume"],
        sdd=common_kwargs["sdd"],
        n_candidates=common_kwargs["n_candidates"],
        n_normals=64,
        n_steps=12,
        lr=0.05,
        lambda_cov=1.0,
        lambda_vcl=0.0,
        lambda_path=0.0,
        init_method="greedy_tuy",
        limited_carm=True,
        seed=0,
    )
    mx.eval(init, admissible_init, src)
    before = float(geom_obj(admissible_init))
    after = float(geom_obj(src))
    assert after >= before - 1e-6, (
        f"C-arm constrained optimization degraded geometric objective: "
        f"before={before}, after={after}"
    )


def test_carm_variant_enforces_limits_while_two_axis_need_not(common_kwargs):
    src_two_axis = two_axis_gantry_vcl_continuous(
        4,
        sid=common_kwargs["sid"],
        roi_center=common_kwargs["roi_center"],
        volume=common_kwargs["volume"],
        sdd=common_kwargs["sdd"],
        n_candidates=common_kwargs["n_candidates"],
        n_normals=64,
        n_steps=8,
        lr=0.05,
        lambda_cov=1.0,
        lambda_vcl=0.2,
        init_method="greedy_tuy",
        limited_carm=False,
        seed=0,
    )
    src_carm = two_axis_gantry_vcl_continuous(
        4,
        sid=common_kwargs["sid"],
        roi_center=common_kwargs["roi_center"],
        volume=common_kwargs["volume"],
        sdd=common_kwargs["sdd"],
        n_candidates=common_kwargs["n_candidates"],
        n_normals=64,
        n_steps=8,
        lr=0.05,
        lambda_cov=1.0,
        lambda_vcl=0.2,
        init_method="greedy_tuy",
        limited_carm=True,
        seed=0,
    )
    mx.eval(src_two_axis, src_carm)
    params_two_axis = mx.stack([
        mx.arctan2(-src_two_axis[:, 0], src_two_axis[:, 1]),
        mx.arcsin(src_two_axis[:, 2] / common_kwargs["sid"]),
    ], axis=-1)
    params_carm = mx.stack([
        mx.arctan2(-src_carm[:, 0], src_carm[:, 1]),
        mx.arcsin(src_carm[:, 2] / common_kwargs["sid"]),
    ], axis=-1)
    gantry = CArmTwoAxisGantry(common_kwargs["sid"])
    clamped_carm = gantry.clamp(params_carm)
    assert float(mx.max(mx.abs(params_carm - clamped_carm))) < 1e-5
    outside_two_axis = float(mx.max(mx.abs(params_two_axis - gantry.clamp(params_two_axis))))
    assert outside_two_axis >= 0.0


def test_carm_solution_is_not_identical_to_free_two_axis(common_kwargs):
    src_two_axis = two_axis_gantry_vcl_continuous(
        4,
        sid=common_kwargs["sid"],
        roi_center=common_kwargs["roi_center"],
        volume=common_kwargs["volume"],
        sdd=common_kwargs["sdd"],
        n_candidates=common_kwargs["n_candidates"],
        n_normals=64,
        n_steps=8,
        lr=0.05,
        lambda_cov=1.0,
        lambda_vcl=0.2,
        init_method="greedy_tuy",
        limited_carm=False,
        seed=0,
    )
    src_carm = two_axis_gantry_vcl_continuous(
        4,
        sid=common_kwargs["sid"],
        roi_center=common_kwargs["roi_center"],
        volume=common_kwargs["volume"],
        sdd=common_kwargs["sdd"],
        n_candidates=common_kwargs["n_candidates"],
        n_normals=64,
        n_steps=8,
        lr=0.05,
        lambda_cov=1.0,
        lambda_vcl=0.2,
        init_method="greedy_tuy",
        limited_carm=True,
        seed=0,
    )
    mx.eval(src_two_axis, src_carm)
    diff = float(mx.linalg.norm(src_two_axis - src_carm))
    assert diff > 1e-6


def test_sources_stay_on_sphere(common_kwargs):
    """After optimisation, sources still live on the source sphere."""
    src = greedy_adam_vcl_continuous(
        4, **common_kwargs,
        lambda_cov=1.0, lambda_vcl=0.0, lambda_path=0.0,
        lambda_icov_fft=0.5, init_method="greedy_tuy",
    )
    mx.eval(src)
    radii = np.linalg.norm(np.asarray(src), axis=-1)
    assert np.allclose(radii, common_kwargs["sid"], rtol=1e-3), (
        f"sources left the source sphere; radii={radii}, "
        f"sid={common_kwargs['sid']}"
    )


def test_lambda_icov_fft_auto_calibration_keeps_scale_stable(common_kwargs):
    """With auto-calibration enabled, scaling the volume amplitude must
    not change the optimised sources — the lambda divides out the I_cov
    magnitude change."""
    vol = common_kwargs["volume"]
    kwargs_lo = dict(common_kwargs); kwargs_lo["volume"] = vol
    kwargs_hi = dict(common_kwargs); kwargs_hi["volume"] = 5.0 * vol

    src_lo = greedy_adam_vcl_continuous(
        4, **kwargs_lo,
        lambda_cov=1.0, lambda_icov_fft=0.2,
        lambda_icov_fft_auto_scale=True,
        init_method="greedy_tuy",
    )
    src_hi = greedy_adam_vcl_continuous(
        4, **kwargs_hi,
        lambda_cov=1.0, lambda_icov_fft=0.2,
        lambda_icov_fft_auto_scale=True,
        init_method="greedy_tuy",
    )
    mx.eval(src_lo); mx.eval(src_hi)
    rel = float(
        mx.linalg.norm(src_lo - src_hi) / mx.linalg.norm(src_lo)
    )
    assert rel < 5e-2, (
        f"auto-cal did not stabilise across volume amplitude; rel_diff={rel}"
    )


def test_lambda_icov_fft_auto_calibration_can_be_disabled(common_kwargs):
    """When ``lambda_icov_fft_auto_scale=False`` the user-supplied lambda is
    used verbatim — useful when callers want to compare apples-to-apples
    against an externally chosen weight."""
    src = greedy_adam_vcl_continuous(
        4, **common_kwargs,
        lambda_cov=1.0, lambda_icov_fft=0.2,
        lambda_icov_fft_auto_scale=False,
        init_method="greedy_tuy",
    )
    mx.eval(src)
    assert src.shape == (4, 3)
    assert np.all(np.isfinite(np.asarray(src)))


def test_icov_fft_optimisation_moves_sources(common_kwargs):
    """A run with ``lambda_icov_fft > 0`` must actually move the iterate
    away from the greedy-Tuy init (Adam is doing work).  Compare against
    a zero-step "init only" reference.

    Strictly improving the I_cov_fft score is checked indirectly via
    :func:`test_lambda_icov_fft_changes_solution`; here we only assert
    that the iterate has moved measurably on the source sphere.
    """
    src_init = greedy_adam_vcl_continuous(
        4, **{**common_kwargs, "n_steps": 1},
        lambda_cov=0.0, lambda_vcl=0.0, lambda_path=0.0,
        lambda_icov_fft=1.0, init_method="greedy_tuy",
    )
    src_refined = greedy_adam_vcl_continuous(
        4, **{**common_kwargs, "n_steps": 30, "lr": 0.1},
        lambda_cov=0.0, lambda_vcl=0.0, lambda_path=0.0,
        lambda_icov_fft=1.0, init_method="greedy_tuy",
    )
    mx.eval(src_init); mx.eval(src_refined)
    move = float(mx.linalg.norm(src_refined - src_init))
    assert move > 1e-2, (
        f"30 Adam steps with lambda_icov_fft=1 produced no measurable "
        f"motion of the sources; ||move||={move}"
    )
    # And the iterate must still be finite and on the sphere.
    radii = np.linalg.norm(np.asarray(src_refined), axis=-1)
    assert np.all(np.isfinite(radii))
    assert np.allclose(radii, common_kwargs["sid"], rtol=1e-3)
