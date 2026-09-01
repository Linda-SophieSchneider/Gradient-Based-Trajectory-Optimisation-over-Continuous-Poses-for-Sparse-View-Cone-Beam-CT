from __future__ import annotations

import pytest

from differentiable_coverage.vcl_geometry import resolve_vcl_working_geometry


def test_working_geometry_preserves_ornl_physical_extents():
    geom = resolve_vcl_working_geometry(
        volume_shape=(512, 512, 512),
        target_shape=(128, 128, 128),
        detector_shape=(1024, 1024),
        working_detector_shape=(128, 128),
        du=0.2,
        dv=0.2,
        voxel_spacing=0.118125,
    )

    assert geom.volume_shape == (128, 128, 128)
    assert geom.detector_shape == (128, 128)
    assert geom.voxel_spacing == pytest.approx(0.4725)
    assert geom.du == pytest.approx(1.6)
    assert geom.dv == pytest.approx(1.6)
    assert geom.volume_shape[0] * geom.voxel_spacing == pytest.approx(
        512 * 0.118125
    )
    assert geom.detector_shape[0] * geom.du == pytest.approx(1024 * 0.2)


def test_working_geometry_never_upsamples_volume():
    geom = resolve_vcl_working_geometry(
        volume_shape=(96, 96, 96),
        target_shape=(128, 128, 128),
        detector_shape=(256, 256),
        du=0.5,
        dv=0.5,
        voxel_spacing=1.2,
    )
    assert geom.volume_shape == (96, 96, 96)
    assert geom.voxel_spacing == pytest.approx(1.2)


def test_anisotropic_downsampling_is_rejected_for_scalar_voxel_pitch():
    with pytest.raises(ValueError, match="isotropic"):
        resolve_vcl_working_geometry(
            volume_shape=(256, 192, 128),
            target_shape=(128, 128, 128),
            detector_shape=(256, 256),
            du=0.5,
            dv=0.5,
            voxel_spacing=1.0,
        )
