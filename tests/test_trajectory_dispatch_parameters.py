from __future__ import annotations

import mlx.core as mx

import differentiable_coverage.eval.trajectories as trajectories


def _base_call_kwargs():
    return {
        "sid": 500.0,
        "roi_center": mx.zeros(3),
        "volume": mx.ones((8, 8, 8)),
        "sdd": 900.0,
        "detector_shape": (256, 256),
        "du": 0.5,
        "dv": 0.6,
        "voxel_spacing": 0.3,
        "n_candidates": 16,
    }


def test_vcls_vcl_warm_start_matches_cold_objective_terms(monkeypatch):
    captured = {}

    def fake_selector(**kwargs):
        captured.update(kwargs)
        return mx.zeros((kwargs["k"], 3))

    monkeypatch.setattr(
        trajectories, "greedy_adam_vcl_continuous", fake_selector
    )
    trajectories.build_baseline_sources(
        "vcls_adam_vcl", 4, **_base_call_kwargs()
    )

    assert captured["lambda_cov"] == 1.0
    assert captured["lambda_vcl"] == 0.2
    assert captured["lambda_path"] == 0.0
    assert captured["init_method"] == "vcls"
    assert captured["du"] == 0.5
    assert captured["dv"] == 0.6
    assert captured["voxel_spacing"] == 0.3


def test_constrained_historical_bundle_id_is_composite(monkeypatch):
    captured = {}

    def fake_bundle(*_args, **_kwargs):
        return mx.ones(256)

    def fake_selector(**kwargs):
        captured.update(kwargs)
        return mx.zeros((kwargs["k"], 3))

    monkeypatch.setattr(
        "differentiable_coverage.absorption_bundle.bundle_path_integral",
        fake_bundle,
    )
    monkeypatch.setattr(trajectories, "calibrate_bundle_weight", lambda _x: 0.1)
    monkeypatch.setattr(
        trajectories, "two_axis_gantry_vcl_continuous", fake_selector
    )
    trajectories.build_baseline_sources(
        "greedy_adam_bundle_carm", 4, **_base_call_kwargs()
    )

    assert captured["lambda_cov"] == 1.0
    assert captured["lambda_vcl"] == 0.2
    assert captured["lambda_bundle"] == 0.1
    assert captured["lambda_path"] == 0.0
    assert captured["limited_carm"] is True


def test_free_sphere_bundle_id_is_bundle_only(monkeypatch):
    captured = {}

    def fake_bundle(*_args, **_kwargs):
        return mx.ones(256)

    def fake_selector(**kwargs):
        captured.update(kwargs)
        return mx.zeros((kwargs["k"], 3))

    monkeypatch.setattr(
        "differentiable_coverage.absorption_bundle.bundle_path_integral",
        fake_bundle,
    )
    monkeypatch.setattr(trajectories, "calibrate_bundle_weight", lambda _x: 0.1)
    monkeypatch.setattr(
        trajectories, "greedy_adam_vcl_continuous", fake_selector
    )
    trajectories.build_baseline_sources(
        "greedy_adam_bundle", 4, **_base_call_kwargs()
    )

    assert captured["lambda_cov"] == 1.0
    assert captured["lambda_vcl"] == 0.0
    assert captured["lambda_bundle"] == 0.1
    assert captured["lambda_path"] == 0.0


def test_free_sphere_composite_matches_constrained_objective(monkeypatch):
    captured = {}

    def fake_bundle(*_args, **_kwargs):
        return mx.ones(256)

    def fake_selector(**kwargs):
        captured.update(kwargs)
        return mx.zeros((kwargs["k"], 3))

    monkeypatch.setattr(
        "differentiable_coverage.absorption_bundle.bundle_path_integral",
        fake_bundle,
    )
    monkeypatch.setattr(trajectories, "calibrate_bundle_weight", lambda _x: 0.1)
    monkeypatch.setattr(
        trajectories, "greedy_adam_vcl_continuous", fake_selector
    )
    trajectories.build_baseline_sources(
        "greedy_adam_composite", 4, **_base_call_kwargs()
    )

    assert captured["init_method"] == "greedy_tuy"
    assert captured["lambda_cov"] == 1.0
    assert captured["lambda_vcl"] == 0.2
    assert captured["lambda_bundle"] == 0.1
    assert captured["lambda_path"] == 0.0


def test_cold_explorer_uses_greedy_start_and_no_path_term(monkeypatch):
    captured = {}

    def fake_selector(**kwargs):
        captured.update(kwargs)
        return mx.zeros((kwargs["k"], 3))

    monkeypatch.setattr(
        trajectories, "greedy_adam_vcl_continuous", fake_selector
    )
    trajectories.build_baseline_sources(
        "greedy_adam_vcl_langevin", 4, **_base_call_kwargs()
    )

    assert captured["init_method"] == "greedy_tuy"
    assert captured["lambda_vcl"] == 0.2
    assert captured["lambda_path"] == 0.0
    assert captured["noise_schedule"] == "langevin_cosine"
