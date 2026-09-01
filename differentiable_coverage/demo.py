"""End-to-end demo of the differentiable_coverage package.

Runs several small demonstrations covering:

1. circular-arc absorption-aware optimization,
2. greedy warm-start plus refinement,
3. multi-restart landscape analysis,
4. kinematic-model comparison,
5. ROI helper utilities,
6. ROI-point aggregated optimization.

A former Demo 4 ("absorption gradient mode ablation") was removed: in the
low-attenuation regime of the Shepp-Logan phantom the gate value ``nu`` is
essentially constant across views, so its gradient is numerically zero and
the three gradient modes (``none`` / ``fd_src`` / ``tangential``) produced
identical optimisation traces.  The modes remain available in
:mod:`differentiable_coverage.absorption` for future work but no longer
appear in the headline experiments.
"""

from __future__ import annotations

import math
from dataclasses import replace

import mlx.core as mx
import diffct_mlx

from differentiable_coverage import (
    AbsorptionConfig,
    CircularArc,
    Free3D,
    Helix,
    ROISelection,
    TwoAxisGantry,
    ScoreConfig,
    absorption_gate,
    compute_absorption_gate,
    adam_ascent,
    anneal,
    coverage_stats,
    gradient_ascent,
    greedy_source_init,
    multi_restart,
    random_sphere_sources,
    roi_from_bbox,
    roi_from_mask,
    sample_unit_sphere,
    saturated_coverage,
)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def _build_phantom(size: int = 64) -> mx.array:
    return mx.array(diffct_mlx.shepp_logan_3d(size, size, size) * 0.02, dtype=mx.float32)


# ---------------------------------------------------------------------------
# Demo 1 — circular-arc gradient ascent with absorption gate
# ---------------------------------------------------------------------------

def demo_circular_arc() -> None:
    print("=" * 60)
    print("Demo 1: circular-arc gradient ascent (absorption-aware)")
    print("=" * 60)

    mu_volume = _build_phantom(64)
    roi_center = mx.array([0.0, 0.0, 0.0])
    radon = sample_unit_sphere(n=500)

    score_cfg = ScoreConfig(tau=0.07)
    abs_cfg = AbsorptionConfig(
        alpha=4.0, eta=0.05,
        sid=200.0, sdd=400.0,
        det_u=64, det_v=64,
        roi_radius=12.0,
    )

    k, n_steps = 24, 150
    parametrize = CircularArc(sid=abs_cfg.sid)
    mx.random.seed(0)
    init_theta = mx.random.uniform(shape=(k,)) * (math.pi / 2.0)
    sigma_end = score_cfg.gaussian_sigma()
    converged_cfg = ScoreConfig(tau=score_cfg.tau, sigma=sigma_end)

    def evaluate(theta: mx.array) -> float:
        sources = parametrize(theta)
        nu = absorption_gate(sources, roi_center, mu_volume, abs_cfg)
        return float(saturated_coverage(sources, roi_center, radon, nu, converged_cfg))

    def coverage(theta: mx.array, step: int) -> mx.array:
        sigma = anneal(step, n_steps, start=score_cfg.tau, end=sigma_end)
        cfg = ScoreConfig(tau=score_cfg.tau, sigma=sigma)
        abs_step_cfg = replace(
            abs_cfg,
            beta_pixel=anneal(step, n_steps, start=1.0, end=abs_cfg.beta_pixel),
            beta_frac=anneal(step, n_steps, start=2.0, end=abs_cfg.beta_frac),
        )
        sources = parametrize(theta)
        nu = absorption_gate(sources, roi_center, mu_volume, abs_step_cfg)
        return saturated_coverage(sources, roi_center, radon, nu, cfg)

    initial = evaluate(init_theta)
    final_theta, _ = gradient_ascent(coverage, init_theta, lr=0.05, n_steps=n_steps)
    final = evaluate(final_theta)

    print(f"  initial coverage (sigma_end): {initial:.4f}")
    print(f"  final   coverage (sigma_end): {final:.4f}")
    print(f"  improvement                 : {final - initial:+.4f}")
    print()


# ---------------------------------------------------------------------------
# Demo 2 — greedy warm-start + Adam refinement
# ---------------------------------------------------------------------------

def demo_greedy_warmstart() -> None:
    print("=" * 60)
    print("Demo 2: greedy warm-start + Adam refinement")
    print("=" * 60)

    roi_center = mx.zeros(3)
    radon = sample_unit_sphere(n=300)
    score_cfg = ScoreConfig(tau=0.07)
    k = 12

    # 200-candidate pool on a full circle
    angles = mx.linspace(0.0, 2.0 * math.pi, 200)
    candidates = CircularArc(sid=200.0)(angles)

    def evaluate(sources: mx.array) -> float:
        nu = mx.ones(sources.shape[0])
        return float(saturated_coverage(sources, roi_center, radon, nu, score_cfg))

    # Greedy init from the candidate pool
    sources_greedy = greedy_source_init(candidates, roi_center, radon, k=k, cfg=score_cfg)
    cov_greedy = evaluate(sources_greedy)

    # Adam refinement starting from the greedy solution
    def coverage_fn(srcs: mx.array, step: int) -> mx.array:
        nu = mx.ones(srcs.shape[0])
        return saturated_coverage(srcs, roi_center, radon, nu, score_cfg)

    sources_refined, history = adam_ascent(coverage_fn, sources_greedy, lr=5.0, n_steps=100)
    cov_refined = evaluate(sources_refined)

    print(f"  after greedy init  ({k} sources from 200 candidates): {cov_greedy:.4f}")
    print(f"  after Adam refinement (100 steps, lr=5):              {cov_refined:.4f}")
    print(f"  improvement from refinement:                          {cov_refined - cov_greedy:+.4f}")
    print()


# ---------------------------------------------------------------------------
# Demo 3 — multi-restart landscape analysis
# ---------------------------------------------------------------------------

def demo_landscape() -> None:
    print("=" * 60)
    print("Demo 3: multi-restart landscape analysis (8 restarts)")
    print("=" * 60)

    roi_center = mx.zeros(3)
    radon = sample_unit_sphere(n=200)
    score_cfg = ScoreConfig(tau=0.07)
    k = 8
    n_restarts = 8

    def cov_fn(params: mx.array, step: int) -> mx.array:
        nu = mx.ones(params.shape[0])
        return saturated_coverage(params, roi_center, radon, nu, score_cfg)

    results, best = multi_restart(
        coverage_fn=cov_fn,
        init_fn=lambda seed: random_sphere_sources(k=k, sid=200.0, seed=seed),
        optimizer_fn=lambda fn, p: adam_ascent(fn, p, lr=5.0, n_steps=80),
        n_restarts=n_restarts,
    )
    stats = coverage_stats(results)

    print(f"  restarts : {n_restarts}  |  k = {k} sources")
    print(f"  best     : {stats['best']:.4f}")
    print(f"  mean±std : {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  min      : {stats['min']:.4f}")
    per_restart = "  ".join(f"{r.coverage:.4f}" for r in results)
    print(f"  all      : {per_restart}")
    print()


# ---------------------------------------------------------------------------
# Demo 4 — kinematic model comparison
# ---------------------------------------------------------------------------

def demo_kinematic_comparison() -> None:
    print("=" * 60)
    print("Demo 4: kinematic model comparison")
    print("=" * 60)

    roi_center = mx.zeros(3)
    radon = sample_unit_sphere(n=300)
    score_cfg = ScoreConfig(tau=0.07)
    k, n_steps = 8, 60
    sid = 200.0

    def geo_coverage(sources: mx.array) -> float:
        nu = mx.ones(sources.shape[0])
        return float(saturated_coverage(sources, roi_center, radon, nu, score_cfg))

    models = {
        "CircularArc": (
            CircularArc(sid=sid),
            mx.linspace(0.0, 2.0 * math.pi, k),
            0.05,
        ),
        "Helix       ": (
            Helix(sid=sid, pitch=20.0),
            mx.linspace(0.0, 2.0 * math.pi, k),
            0.05,
        ),
        "TwoAxisGantry": (
            TwoAxisGantry(sid=sid),
            mx.stack([
                mx.linspace(0.0, 2.0 * math.pi, k),
                mx.zeros(k),
            ], axis=-1),
            0.05,
        ),
        "Free3D      ": (
            Free3D(),
            random_sphere_sources(k=k, sid=sid, seed=42),
            5.0,
        ),
    }

    for name, (model, init_params, lr) in models.items():
        def make_fn(m):
            def coverage_fn(params: mx.array, step: int) -> mx.array:
                sources = m(params)
                nu = mx.ones(sources.shape[0])
                return saturated_coverage(sources, roi_center, radon, nu, score_cfg)
            return coverage_fn

        initial = geo_coverage(model(init_params))
        final_params, _ = adam_ascent(make_fn(model), init_params, lr=lr, n_steps=n_steps)
        final = geo_coverage(model(final_params))
        print(f"  {name}  initial={initial:.4f}  final={final:.4f}  Δ={final-initial:+.4f}")
    print()


# ---------------------------------------------------------------------------
# Demo 5 — ROI helper utilities
# ---------------------------------------------------------------------------

def demo_roi_helpers() -> None:
    print("=" * 60)
    print("Demo 5: ROI helper utilities")
    print("=" * 60)

    roi_bbox = roi_from_bbox(
        min_corner=(24, 24, 24),
        max_corner=(40, 40, 40),
        voxel_spacing=1.0,
        origin=(0.0, 0.0, 0.0),
        points_per_axis=3,
    )

    mask = mx.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=mx.float32,
    )
    roi_mask = roi_from_mask(mask, voxel_spacing=2.0, origin=(10.0, 20.0, 30.0), max_points=8)

    def report(name: str, roi: ROISelection) -> None:
        center = roi.center.tolist()
        print(f"  {name:8s} center={center}  radius={roi.radius:.3f}  points={roi.points.shape[0]}")

    report("bbox", roi_bbox)
    report("mask", roi_mask)
    print()


# ---------------------------------------------------------------------------
# Demo 6 — ROI-point aggregated optimization
# ---------------------------------------------------------------------------

def demo_roi_point_aggregation() -> None:
    print("=" * 60)
    print("Demo 6: ROI-point aggregated optimization")
    print("=" * 60)

    roi = roi_from_bbox(
        min_corner=(24, 24, 24),
        max_corner=(40, 40, 40),
        voxel_spacing=1.0,
        origin=(0.0, 0.0, 0.0),
        points_per_axis=3,
    )
    radon = sample_unit_sphere(n=200)
    score_cfg = ScoreConfig(tau=0.07)
    k, n_steps = 12, 60
    arc = CircularArc(sid=200.0)
    mx.random.seed(21)
    init_theta = mx.random.uniform(shape=(k,)) * math.pi

    def cov_center(theta: mx.array) -> float:
        sources = arc(theta)
        nu = mx.ones(sources.shape[0])
        return float(saturated_coverage(sources, roi.center, radon, nu, score_cfg))

    def cov_roi(theta: mx.array) -> float:
        sources = arc(theta)
        nu = mx.ones(sources.shape[0])
        return float(
            saturated_coverage(
                sources,
                roi.center,
                radon,
                nu,
                score_cfg,
                roi_points=roi.points,
                roi_weights=roi.weights,
            )
        )

    def coverage_fn(theta: mx.array, step: int) -> mx.array:
        sources = arc(theta)
        nu = mx.ones(sources.shape[0])
        return saturated_coverage(
            sources,
            roi.center,
            radon,
            nu,
            score_cfg,
            roi_points=roi.points,
            roi_weights=roi.weights,
        )

    init_center = cov_center(init_theta)
    init_roi = cov_roi(init_theta)
    final_theta, _ = adam_ascent(coverage_fn, init_theta, lr=0.05, n_steps=n_steps)
    final_center = cov_center(final_theta)
    final_roi = cov_roi(final_theta)

    print(f"  roi center  = {roi.center.tolist()}")
    print(f"  roi radius  = {roi.radius:.3f}")
    print(f"  roi points  = {roi.points.shape[0]}")
    print(f"  center-only initial={init_center:.4f}  final={final_center:.4f}  Δ={final_center-init_center:+.4f}")
    print(f"  roi-aware   initial={init_roi:.4f}  final={final_roi:.4f}  Δ={final_roi-init_roi:+.4f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    demo_circular_arc()
    demo_greedy_warmstart()
    demo_landscape()
    demo_kinematic_comparison()
    demo_roi_helpers()
    demo_roi_point_aggregation()


if __name__ == "__main__":
    main()
