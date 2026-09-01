"""Reconstruction-quality evaluation for differentiable-coverage trajectories.

Pipeline
--------
1. ``phantoms.build_phantom(name, shape)`` returns an attenuation volume.
2. ``trajectories.build_baseline_sources(name, k, sid, ...)`` returns a
   ``(k, 3)`` array of source positions for a baseline.
3. ``geometry.geometry_from_sources(src_pos, sid, sdd, ...)`` builds the
   matching detector geometry following the diffct_mlx convention.
4. ``reco.reconstruct_sart(volume, src_pos, det_*, ...)`` simulates the
   sinogram and reconstructs.
5. ``metrics.rmse / psnr / roi_rmse`` quantify the reconstruction.

Used by ``experiments/run.py`` and the CLI studies under
``experiments/studies/`` to score every (phantom, baseline, k) cell.
"""

from .geometry import geometry_from_sources
from .metrics import psnr, rmse, roi_rmse, ssim
from .phantoms import PHANTOM_NAMES, build_phantom
from .reco import reconstruct_sart_volume, simulate_sinogram
from .trajectories import BASELINE_NAMES, build_baseline_sources

__all__ = [
    "PHANTOM_NAMES",
    "BASELINE_NAMES",
    "build_phantom",
    "build_baseline_sources",
    "geometry_from_sources",
    "reconstruct_sart_volume",
    "simulate_sinogram",
    "rmse",
    "psnr",
    "roi_rmse",
    "ssim",
]
