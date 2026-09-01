"""Shared fixtures for the differentiable_coverage test suite."""

import math
import os
import platform

import mlx.core as mx
import pytest

# mlx-cuda 0.30 has GPU correctness/abort bugs; all mx-side arrays in this
# project are tiny, so on Linux the suite runs mx on CPU — the same pin
# configure_runtime() applies on the torch backend.  No effect on macOS.
# Set DIFFCOV_TEST_ALLOW_GPU=1 to exercise the GPU backend deliberately.
if (platform.system() == "Linux"
        and not os.environ.get("DIFFCOV_TEST_ALLOW_GPU")
        and hasattr(mx, "set_default_device") and hasattr(mx, "cpu")):
    mx.set_default_device(mx.cpu)


@pytest.fixture
def roi_center():
    return mx.zeros(3)


@pytest.fixture
def small_sources():
    """4 sources arranged in a circle at radius 200, in the xy-plane."""
    angles = mx.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    sid = 200.0
    x = -sid * mx.sin(angles)
    y = sid * mx.cos(angles)
    z = mx.zeros_like(angles)
    return mx.stack([x, y, z], axis=-1)


@pytest.fixture
def radon_normals():
    from differentiable_coverage import sample_unit_sphere
    return sample_unit_sphere(50)


@pytest.fixture
def score_cfg():
    from differentiable_coverage import ScoreConfig
    return ScoreConfig(tau=0.07)


@pytest.fixture
def ones_nu(small_sources):
    return mx.ones(small_sources.shape[0])
