"""Differentiable soft coverage for continuous trajectory optimization.

Implementation of `future_work_differentiable_coverage.md`:
turns the saturated-coverage objective of the MILP paper into a
differentiable function of continuous source positions, enabling
gradient-based trajectory refinement on top of (or instead of)
discrete view selection.
"""

from .absorption import AbsorptionConfig, absorption_gate, compute_absorption_gate
from .absorption import _detector_containment_gate as detector_containment_gate
from .joint import JointLoopConfig, JointLoopResult, joint_loop
from .landscape import RestartResult, coverage_stats, multi_restart, random_sphere_sources
from .optimize import adam_ascent, anneal, gradient_ascent
from .roi import ROISelection, roi_from_bbox, roi_from_mask, roi_from_points
from .score import (
    ScoreConfig,
    accumulated_coverage,
    greedy_source_init,
    orthogonality_kernel,
    ray_directions,
    sample_unit_sphere,
    saturated_coverage,
)
from .trajectory import CArmTwoAxisGantry, CircularArc, Free3D, Helix, TwoAxisGantry

__version__ = "0.1.0.dev0"

__all__ = [
    "AbsorptionConfig",
    "CArmTwoAxisGantry",
    "CircularArc",
    "Free3D",
    "Helix",
    "TwoAxisGantry",
    "ScoreConfig",
    "JointLoopConfig",
    "JointLoopResult",
    "RestartResult",
    "absorption_gate",
    "compute_absorption_gate",
    "joint_loop",
    "accumulated_coverage",
    "adam_ascent",
    "anneal",
    "coverage_stats",
    "detector_containment_gate",
    "gradient_ascent",
    "greedy_source_init",
    "multi_restart",
    "orthogonality_kernel",
    "random_sphere_sources",
    "ray_directions",
    "ROISelection",
    "roi_from_bbox",
    "roi_from_mask",
    "roi_from_points",
    "sample_unit_sphere",
    "saturated_coverage",
]
