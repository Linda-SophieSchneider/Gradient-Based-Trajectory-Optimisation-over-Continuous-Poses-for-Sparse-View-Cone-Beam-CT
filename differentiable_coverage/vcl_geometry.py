"""Physical geometry for the downsampled continuous-VCL working grid."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose


@dataclass(frozen=True)
class VCLWorkingGeometry:
    """Resolved physical sampling used by the internal VCL operator."""

    volume_shape: tuple[int, int, int]
    detector_shape: tuple[int, int]
    du: float
    dv: float
    voxel_spacing: float


def resolve_vcl_working_geometry(
    *,
    volume_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    detector_shape: tuple[int, int],
    working_detector_shape: tuple[int, int] = (128, 128),
    du: float,
    dv: float,
    voxel_spacing: float,
) -> VCLWorkingGeometry:
    """Preserve physical volume and detector extents after downsampling.

    ``build_vcl_context`` accepts a scalar voxel spacing.  Consequently the
    source and target volume shapes must imply the same scale factor on all
    three axes.  All paper experiments use cubic inputs and cubic working
    grids, so rejecting anisotropic resampling is safer than silently assigning
    an incorrect physical pitch.
    """
    source = tuple(int(v) for v in volume_shape)
    target = tuple(min(int(s), int(t)) for s, t in zip(source, target_shape))
    detector = tuple(int(v) for v in detector_shape)
    working_detector = tuple(int(v) for v in working_detector_shape)

    if len(source) != 3 or len(target) != 3 or any(v <= 0 for v in source + target):
        raise ValueError("volume_shape and target_shape must contain three positive sizes")
    if (
        len(detector) != 2
        or len(working_detector) != 2
        or any(v <= 0 for v in detector + working_detector)
    ):
        raise ValueError("detector shapes must contain two positive sizes")
    if du <= 0.0 or dv <= 0.0 or voxel_spacing <= 0.0:
        raise ValueError("du, dv, and voxel_spacing must be positive")

    scales = tuple(s / t for s, t in zip(source, target))
    if not all(isclose(scales[0], scale, rel_tol=1e-6, abs_tol=1e-9) for scale in scales[1:]):
        raise ValueError(
            "continuous VCL currently requires isotropic volume downsampling; "
            f"got source={source}, target={target}"
        )

    return VCLWorkingGeometry(
        volume_shape=target,
        detector_shape=working_detector,
        du=float(du) * detector[0] / working_detector[0],
        dv=float(dv) * detector[1] / working_detector[1],
        voxel_spacing=float(voxel_spacing) * scales[0],
    )
