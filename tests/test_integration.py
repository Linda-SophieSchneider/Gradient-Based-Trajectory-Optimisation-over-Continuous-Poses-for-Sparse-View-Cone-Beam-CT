"""Integration test: runs the full demo pipeline and checks expected results."""

from dataclasses import replace

import pytest


@pytest.mark.integration
def test_demo_coverage_improves():
    """The demo must show measurable coverage improvement."""
    import math
    import mlx.core as mx
    from diffct_mlx.phantoms import shepp_logan_3d
    from differentiable_coverage import (
        AbsorptionConfig,
        CircularArc,
        ScoreConfig,
        absorption_gate,
        anneal,
        gradient_ascent,
        sample_unit_sphere,
        saturated_coverage,
    )

    mx.random.seed(0)

    volume = shepp_logan_3d(64, 64, 64) * 0.02
    roi_center = mx.zeros(3)
    radon_normals = sample_unit_sphere(500)
    score_cfg = ScoreConfig(tau=0.07)
    absorption_cfg = AbsorptionConfig(
        alpha=4.0, eta=0.05, roi_radius=12.0,
        sid=200.0, sdd=400.0,
        det_u=64, det_v=64, du=1.0, dv=1.0,
        voxel_spacing=1.0,
    )

    k = 24
    n_steps = 150
    lr = 0.05
    sigma_start = score_cfg.tau
    sigma_end = score_cfg.gaussian_sigma()

    arc = CircularArc(sid=200.0)
    theta_init = mx.random.uniform(shape=(k,)) * (math.pi / 2.0)

    def coverage_fn(theta, step):
        sigma = anneal(step, n_steps, sigma_start, sigma_end)
        cfg = ScoreConfig(tau=score_cfg.tau, sigma=sigma)
        abs_step_cfg = replace(
            absorption_cfg,
            beta_pixel=anneal(step, n_steps, start=1.0, end=absorption_cfg.beta_pixel),
            beta_frac=anneal(step, n_steps, start=2.0, end=absorption_cfg.beta_frac),
        )
        sources = arc(theta)
        nu = absorption_gate(sources, roi_center, volume, abs_step_cfg)
        return saturated_coverage(sources, roi_center, radon_normals, nu, cfg)

    # measure initial coverage at sigma_end
    src_init = arc(theta_init)
    nu_init = absorption_gate(src_init, roi_center, volume, absorption_cfg)
    cov_init = float(saturated_coverage(src_init, roi_center, radon_normals, nu_init,
                                        ScoreConfig(tau=score_cfg.tau, sigma=sigma_end)))

    final_theta, _ = gradient_ascent(coverage_fn, theta_init, lr=lr, n_steps=n_steps)

    src_final = arc(final_theta)
    nu_final = absorption_gate(src_final, roi_center, volume, absorption_cfg)
    cov_final = float(saturated_coverage(src_final, roi_center, radon_normals, nu_final,
                                         ScoreConfig(tau=score_cfg.tau, sigma=sigma_end)))

    improvement = cov_final - cov_init
    assert improvement > 0.0, f"Coverage did not improve: {cov_init:.4f} -> {cov_final:.4f}"
    assert cov_init == pytest.approx(0.2504, abs=0.02)
    assert improvement == pytest.approx(0.0691, abs=0.02)
