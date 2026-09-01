"""Phantom catalogue for reconstruction-quality evaluation.

Three phantoms covering complementary failure modes for sparse-view CT:

* ``shepp_logan``     — Standard isotropic medical phantom.  Low contrast,
                         many small ellipsoids.  Tests overall geometric
                         coverage; benefits from angular diversity.
* ``anisotropic``     — Bundle of two highly elongated ellipsoids oriented
                         along orthogonal axes.  Tests direction-specific
                         coverage: a trajectory that under-samples one axis
                         will blur that ellipsoid first.
* ``industrial``      — Aluminium-density bulk with a high-Z (steel-like)
                         dense insert and a low-density void.  Tests the
                         absorption gate: rays through the dense insert are
                         physically unusable and should be down-weighted.

All phantoms return attenuation values in arbitrary units (peak ≈ 1.0 for
soft phantoms, peak ≈ 4.0 for the industrial phantom to mimic a strong
attenuator) on an ``(nz, ny, nx)`` voxel grid.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

try:
    from diffct_mlx import shepp_logan_3d as _diffct_shepp_logan_3d
except ImportError:  # pragma: no cover - keeps module importable without diffct
    _diffct_shepp_logan_3d = None


PHANTOM_NAMES = ("shepp_logan", "anisotropic", "industrial_ornl")
PHANTOM_NAMES_ALL = (
    "shepp_logan",
    "anisotropic",
    "industrial_ornl",
    "industrial_synth",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _voxel_grid(nz: int, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return zz, yy, xx coordinate arrays in [-1, 1] centered on the volume."""
    z = np.linspace(-1.0, 1.0, nz, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return zz, yy, xx


def _ellipsoid(zz, yy, xx, center, radii, value):
    cz, cy, cx = center
    rz, ry, rx = radii
    mask = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    return mask.astype(np.float32) * value


# ---------------------------------------------------------------------------
# Phantoms
# ---------------------------------------------------------------------------

def _shepp_logan(shape: tuple[int, int, int]) -> np.ndarray:
    """Use diffct_mlx's Shepp-Logan if available, else a coarse manual fallback."""
    if _diffct_shepp_logan_3d is not None:
        vol = np.asarray(_diffct_shepp_logan_3d(shape), dtype=np.float32)
        return vol

    # Minimal fallback: head outline + a couple of inserts.
    nz, ny, nx = shape
    zz, yy, xx = _voxel_grid(nz, ny, nx)
    vol = np.zeros_like(zz)
    vol += _ellipsoid(zz, yy, xx, (0, 0, 0), (0.92, 0.69, 0.9), 1.0)
    vol -= _ellipsoid(zz, yy, xx, (0, 0, 0), (0.874, 0.6624, 0.88), 0.98)
    vol += _ellipsoid(zz, yy, xx, (0, -0.0184, -0.22), (0.41, 0.21, 0.16), 0.2)
    vol += _ellipsoid(zz, yy, xx, (0, -0.0184, 0.22), (0.31, 0.22, 0.11), 0.2)
    vol = np.clip(vol, 0.0, None).astype(np.float32)
    return vol


def _anisotropic(shape: tuple[int, int, int]) -> np.ndarray:
    """Two elongated ellipsoids oriented along orthogonal axes."""
    nz, ny, nx = shape
    zz, yy, xx = _voxel_grid(nz, ny, nx)
    vol = np.zeros_like(zz)
    # Background sphere
    vol += _ellipsoid(zz, yy, xx, (0, 0, 0), (0.85, 0.85, 0.85), 0.20)
    # Long thin rod along x
    vol += _ellipsoid(zz, yy, xx, (0, -0.15, 0.0), (0.10, 0.10, 0.70), 0.60)
    # Long thin rod along y (orthogonal)
    vol += _ellipsoid(zz, yy, xx, (0, 0.15, 0.0), (0.10, 0.70, 0.10), 0.60)
    # Small sphere off-center as a high-frequency feature
    vol += _ellipsoid(zz, yy, xx, (0.4, 0, 0), (0.08, 0.08, 0.08), 0.80)
    vol = np.clip(vol, 0.0, 1.5).astype(np.float32)
    return vol


def _industrial(shape: tuple[int, int, int]) -> np.ndarray:
    """Aluminium-like bulk with a steel-like dense insert and an air void.

    Attenuation values are in arbitrary linearized units; the dense insert is
    ~5x the matrix, which is well past the absorption-gate threshold for a
    moderately bright X-ray spectrum.
    """
    nz, ny, nx = shape
    zz, yy, xx = _voxel_grid(nz, ny, nx)
    vol = np.zeros_like(zz)
    # Aluminium block (oblate)
    vol += _ellipsoid(zz, yy, xx, (0, 0, 0), (0.75, 0.85, 0.85), 0.8)
    # Dense steel-like insert, off-center
    vol += _ellipsoid(zz, yy, xx, (0.1, 0.25, -0.20), (0.18, 0.18, 0.18), 3.2)
    # Second smaller dense insert
    vol += _ellipsoid(zz, yy, xx, (-0.2, -0.30, 0.15), (0.12, 0.12, 0.12), 3.2)
    # Air void (subtract)
    vol -= _ellipsoid(zz, yy, xx, (0.0, 0.0, 0.40), (0.12, 0.22, 0.12), 0.8)
    vol = np.clip(vol, 0.0, 5.0).astype(np.float32)
    return vol


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_phantom(name: str, shape: tuple[int, int, int]) -> mx.array:
    """Return an ``(nz, ny, nx)`` attenuation volume as an ``mx.array``.

    Parameters
    ----------
    name
        One of :data:`PHANTOM_NAMES_ALL`:
        ``shepp_logan``, ``anisotropic``, ``industrial_ornl``, ``industrial_synth``.
    shape
        ``(nz, ny, nx)``.

    Notes
    -----
    ``industrial_ornl`` requires that the ORNL hexagonal-nozzle HDF5 files are
    available at the path pointed to by ``DIFFCT_ORNL_PATH`` (or
    ``~/Documents/Data/ORNL_Nozzle`` by default).  If the dataset is not found,
    a :class:`FileNotFoundError` is raised with instructions.
    """
    if name == "shepp_logan":
        vol = _shepp_logan(shape)
    elif name == "anisotropic":
        vol = _anisotropic(shape)
    elif name == "industrial_synth":
        vol = _industrial(shape)
    elif name == "industrial_ornl":
        # Deferred import so the module stays importable without h5py.
        from .datasets.ornl_nozzle import load_as_phantom

        return load_as_phantom("any", shape)
    elif name == "industrial":  # backward-compat alias for the smoke test
        vol = _industrial(shape)
    else:
        raise ValueError(
            f"Unknown phantom '{name}'. Choose from {PHANTOM_NAMES_ALL}."
        )
    return mx.array(vol)
