from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from differentiable_coverage.runtime import configure_runtime


class _CudaProbe:
    def item(self):
        return 1.0


def test_cuda_runtime_pins_mlx_to_cpu_and_smoke_tests_device(monkeypatch):
    chosen = []
    synchronized = []

    fake_diffct = ModuleType("diffct_mlx")
    fake_diffct.backend = "torch"
    fake_diffct.cone_forward_footprint = object()
    fake_diffct.cone_backward_footprint = object()

    fake_mlx_core = ModuleType("mlx.core")
    fake_mlx_core.cpu = object()
    fake_mlx_core.set_default_device = chosen.append
    fake_mlx = ModuleType("mlx")
    fake_mlx.core = fake_mlx_core

    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "test"
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_name=lambda _idx: "Test CUDA GPU",
        synchronize=lambda: synchronized.append(True),
    )
    fake_torch.ones = lambda *_args, **_kwargs: _CudaProbe()

    monkeypatch.setitem(sys.modules, "diffct_mlx", fake_diffct)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    status = configure_runtime("cuda", cuda_smoke_test=True)

    assert status.resolved == "cuda"
    assert status.diffct_backend == "torch"
    assert status.cuda_device == "Test CUDA GPU"
    assert chosen == [fake_mlx_core.cpu]
    assert synchronized == [True]


def test_cuda_runtime_rejects_native_mlx_backend(monkeypatch):
    fake_diffct = ModuleType("diffct_mlx")
    fake_diffct.backend = "mlx"
    fake_diffct.cone_forward_footprint = object()
    fake_diffct.cone_backward_footprint = object()
    fake_mlx_core = ModuleType("mlx.core")
    fake_mlx = ModuleType("mlx")
    fake_mlx.core = fake_mlx_core
    monkeypatch.setitem(sys.modules, "diffct_mlx", fake_diffct)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

    with pytest.raises(RuntimeError, match="backend == 'torch'"):
        configure_runtime("cuda")


def test_invalid_runtime_is_rejected_before_import():
    with pytest.raises(RuntimeError, match="Unsupported"):
        configure_runtime("not-a-runtime")
