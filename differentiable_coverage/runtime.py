"""Runtime selection and preflight checks for MLX and torch/CUDA backends."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version


@dataclass(frozen=True)
class RuntimeStatus:
    requested: str
    resolved: str
    diffct_backend: str
    diffct_version: str
    torch_version: str | None
    cuda_available: bool
    cuda_device: str | None
    mlx_default_device: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def configure_runtime(
    requested: str | None = None,
    *,
    cuda_smoke_test: bool = False,
) -> RuntimeStatus:
    """Configure the lightweight MLX side and validate the heavy backend.

    ``requested`` defaults to ``DIFFCOV_RUNTIME`` and accepts ``auto``,
    ``metal``, or ``cuda``.  On the CUDA path MLX is pinned to CPU because
    paper-scale projector and reconstruction work is delegated to
    ``diffct_mlx``'s torch/CUDA backend.
    """
    mode = (requested or os.environ.get("DIFFCOV_RUNTIME", "auto")).strip().lower()
    if mode not in {"auto", "metal", "cuda"}:
        raise RuntimeError(
            f"Unsupported DIFFCOV_RUNTIME={mode!r}; expected auto, metal, or cuda"
        )

    import diffct_mlx
    import mlx.core as mx

    backend = str(getattr(diffct_mlx, "backend", "mlx")).lower()
    required = ("cone_forward_footprint", "cone_backward_footprint")
    missing = [name for name in required if not hasattr(diffct_mlx, name)]
    if missing:
        raise RuntimeError(
            "diffct_mlx is missing the matched footprint API required by VCL: "
            + ", ".join(missing)
        )

    torch_version = None
    cuda_available = False
    cuda_device = None
    if backend == "torch":
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device = str(torch.cuda.get_device_name(torch.cuda.current_device()))

        if not hasattr(mx, "set_default_device") or not hasattr(mx, "cpu"):
            raise RuntimeError(
                "The torch/CUDA backend requires an MLX build with "
                "mx.set_default_device(mx.cpu)"
            )
        mx.set_default_device(mx.cpu)
        mlx_device = "cpu"

        if cuda_smoke_test and cuda_available:
            probe = torch.ones(1, device="cuda")
            if float(probe.item()) != 1.0:
                raise RuntimeError("CUDA tensor smoke test returned an invalid result")
            torch.cuda.synchronize()
    else:
        mlx_device = "native"

    if mode == "cuda":
        if backend != "torch":
            raise RuntimeError(
                "CUDA execution requires diffct_mlx.backend == 'torch'. "
                "Set DIFFCT_BACKEND=torch before Python starts."
            )
        if not cuda_available:
            raise RuntimeError(
                "CUDA execution requested, but torch.cuda.is_available() is false"
            )
        resolved = "cuda"
    elif mode == "metal":
        if backend != "mlx":
            raise RuntimeError(
                "Metal execution requires the native diffct_mlx MLX backend"
            )
        resolved = "metal"
    elif backend == "torch":
        resolved = "cuda" if cuda_available else "torch-cpu"
    else:
        resolved = "metal"

    return RuntimeStatus(
        requested=mode,
        resolved=resolved,
        diffct_backend=backend,
        diffct_version=_package_version("diffct-mlx"),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_device=cuda_device,
        mlx_default_device=mlx_device,
    )
