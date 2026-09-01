"""Loader for the ORNL hexagonal fuel-nozzle CT dataset.

Source
------
„X-ray Computed Tomography Data of Dense Metallic Components"
ORNL Manufacturing Demonstration Facility, Metrotom 800 cone-beam scanner
(Rahman & Ziabari, ICCP 2025).

DOI: https://doi.ccs.ornl.gov/dataset/57bb2cf5-32fb-553a-85e6-087a45d3f600

Actual HDF5 schema (verified 2026-05-23)
---------------------------------------
Each file (``L``/``M``/``T`` sections = Lower/Middle/Top) contains:

* ``/reconstruction/FDK``           — ``(nz, ny, nx)`` float16, baseline FDK.
* ``/projection/NegativeLogNorm_Proj`` — ``(n_views, row, col)`` float16.
* ``/projection/RawCounts``         — raw counts uint16 (unused here).

All geometry parameters live as **root-level attributes**:

* ``angles``               (n_views,)  radians
* ``src_iso_dist``         scalar      — SID in mm
* ``iso_det_dist``         scalar      — IDD in mm  (SDD = SID + IDD)
* ``voxel_size``           scalar      — isotropic voxel pitch in mm
* ``det_pixel_size``       scalar      — isotropic detector pixel in mm
* ``det_column_offset``    scalar      — centre-of-rotation u offset (mm)
* ``det_row_offset``       scalar      — centre-of-rotation v offset (mm)
* ``BHC_params``           (4,)        — Van-de-Casteele BHC parameters

The reconstruction is large (~992 × 1036² ≈ 4 GB as float32), so we read it
with HDF5 strided indexing and finish with a small in-memory resample.

Usage
-----
::

    export DIFFCT_ORNL_PATH=/path/to/DenseMetallicComponentData
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ORNL_DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "DIFFCT_ORNL_PATH",
        str(Path(__file__).resolve().parents[3] / "DenseMetallicComponentData"),
    )
)


# Section-tag → filename substring heuristics.  The vendor uses "SRC L/M/T"
# in their filenames (Lower / Middle / Top of the hexagonal nozzle).
_SECTION_PATTERNS = {
    "L": ("src l ", " l ", "_l_", "lower"),
    "M": ("src m ", " m ", "_m_", "middle", "medium"),
    "T": ("src t ", " t ", "_t_", "top", "upper"),
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ORNLNozzleData:
    """In-memory view of one ORNL nozzle section."""

    section: str
    source_path: Path
    reconstruction: np.ndarray  # (nz, ny, nx) float32
    sid: float
    idd: float
    voxel_size_mm: float
    det_pixel_mm: float
    n_views: int
    det_column_offset_mm: float
    det_row_offset_mm: float

    @property
    def sdd(self) -> float:
        return self.sid + self.idd

    def summary(self) -> str:
        rec = self.reconstruction
        return (
            f"ORNL Nozzle '{self.section}' @ {self.source_path.name}\n"
            f"  recon: shape={rec.shape} dtype={rec.dtype} "
            f"min/max={rec.min():.4g}/{rec.max():.4g} "
            f"mean={rec.mean():.4g}\n"
            f"  geometry: SID={self.sid:.2f}mm  IDD={self.idd:.2f}mm  "
            f"SDD={self.sdd:.2f}mm  voxel={self.voxel_size_mm:.4f}mm  "
            f"det_pixel={self.det_pixel_mm:.3f}mm  n_views={self.n_views}"
        )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_ornl_files(root: Path | None = None) -> dict[str, Path]:
    """Map section tag (``L`` / ``M`` / ``T``) to file path under ``root``."""
    root = Path(root) if root is not None else ORNL_DEFAULT_DATA_DIR
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    if root.is_file() and root.suffix.lower() in {".h5", ".hdf5"}:
        out["L"] = root
        return out

    candidates = sorted(list(root.rglob("*.hdf5")) + list(root.rglob("*.h5")))
    for path in candidates:
        name = path.name.lower()
        matched = False
        for tag, patterns in _SECTION_PATTERNS.items():
            if any(p in name for p in patterns):
                # Prefer the *first* match per tag (sorted deterministically)
                out.setdefault(tag, path)
                matched = True
                break
        if not matched:
            out.setdefault("unknown", path)
    return out


# ---------------------------------------------------------------------------
# Strided HDF5 read
# ---------------------------------------------------------------------------

def _attr_scalar(h5file, key: str) -> float:
    arr = np.asarray(h5file.attrs[key])
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[0])


def _read_recon_strided(
    h5file,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Stride-read ``/reconstruction/FDK`` to roughly ``target_shape``.

    We pick per-axis integer strides so the strided shape is no smaller than
    the target, then do a final lightweight trilinear interpolation onto the
    exact target grid.  This avoids loading the full ~4 GB array.
    """
    rec_ds = h5file["/reconstruction/FDK"]
    src_shape = rec_ds.shape
    if any(t <= 0 for t in target_shape):
        raise ValueError(f"target_shape must be positive: {target_shape}")
    strides = tuple(
        max(1, src_shape[i] // max(target_shape[i], 1))
        for i in range(3)
    )
    # h5py slicing supports striding directly (and avoids loading everything)
    sub = rec_ds[::strides[0], ::strides[1], ::strides[2]]
    # Convert float16 -> float32
    return np.asarray(sub, dtype=np.float32)


def _trilinear_to_shape(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    if volume.shape == target_shape:
        return volume.astype(np.float32, copy=False)
    from scipy.ndimage import map_coordinates  # type: ignore

    src_shape = volume.shape
    zi = np.linspace(0.0, src_shape[0] - 1.0, target_shape[0], dtype=np.float32)
    yi = np.linspace(0.0, src_shape[1] - 1.0, target_shape[1], dtype=np.float32)
    xi = np.linspace(0.0, src_shape[2] - 1.0, target_shape[2], dtype=np.float32)
    gz, gy, gx = np.meshgrid(zi, yi, xi, indexing="ij")
    coords = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], axis=0)
    out = map_coordinates(volume, coords, order=1, mode="nearest")
    return out.reshape(target_shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ornl_nozzle_volume(
    path: Path | str,
    section: str | None = None,
    target_shape: tuple[int, int, int] | None = None,
) -> ORNLNozzleData:
    """Open one ORNL HDF5 file and pull out the baseline reconstruction + geometry.

    Parameters
    ----------
    path
        Path to one of the ``TCR- Single Channeled SRC {L,M,T} ...hdf5`` files.
    section
        Optional ``L``/``M``/``T`` label.  Guessed from filename if omitted.
    target_shape
        If given, the reconstruction is strided-read and resampled to this
        ``(nz, ny, nx)`` shape.  Recommended: anything ≤ source resolution
        keeps the loader fast.  If ``None``, the *full* volume is loaded —
        beware ~4 GB float32.
    """
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ORNL HDF5 file not found: {path}")
    if section is None:
        lname = path.name.lower()
        for tag, patterns in _SECTION_PATTERNS.items():
            if any(p in lname for p in patterns):
                section = tag
                break
        section = section or "unknown"

    with h5py.File(path, "r") as h:
        if target_shape is not None:
            sub = _read_recon_strided(h, target_shape)
            rec = _trilinear_to_shape(sub, target_shape)
        else:
            rec = np.asarray(h["/reconstruction/FDK"][()], dtype=np.float32)

        # Clip negatives — FDK can produce small negative artefacts
        rec = np.clip(rec, 0.0, None)

        sid = _attr_scalar(h, "src_iso_dist")
        idd = _attr_scalar(h, "iso_det_dist")
        voxel_size = _attr_scalar(h, "voxel_size")
        det_pixel = _attr_scalar(h, "det_pixel_size")
        det_col_off = _attr_scalar(h, "det_column_offset")
        det_row_off = _attr_scalar(h, "det_row_offset")
        n_views = int(np.asarray(h.attrs["angles"]).size)

    return ORNLNozzleData(
        section=section,
        source_path=path,
        reconstruction=rec,
        sid=sid,
        idd=idd,
        voxel_size_mm=voxel_size,
        det_pixel_mm=det_pixel,
        n_views=n_views,
        det_column_offset_mm=det_col_off,
        det_row_offset_mm=det_row_off,
    )


def load_as_phantom(
    section: str,
    shape: tuple[int, int, int],
    *,
    data_dir: Path | None = None,
) -> mx.array:
    """High-level entry point: locate the right file and return an mx.array.

    Parameters
    ----------
    section
        ``L``, ``M``, ``T``, or ``any``.  ``any`` picks the first available
        section (deterministic by section ordering).
    shape
        Target voxel-grid shape ``(nz, ny, nx)``.
    """
    files = find_ornl_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No ORNL HDF5 files found under "
            f"{data_dir or ORNL_DEFAULT_DATA_DIR}.  "
            "Set DIFFCT_ORNL_PATH or pass data_dir explicitly."
        )
    if section == "any":
        # Prefer L, then M, then T, then any other
        for tag in ("L", "M", "T"):
            if tag in files:
                chosen_section = tag
                chosen_path = files[tag]
                break
        else:
            chosen_section, chosen_path = next(iter(files.items()))
    elif section in files:
        chosen_section = section
        chosen_path = files[section]
    else:
        raise KeyError(
            f"Section '{section}' not found.  Available: {sorted(files)}."
        )

    data = load_ornl_nozzle_volume(chosen_path, section=chosen_section,
                                   target_shape=shape)
    return mx.array(data.reconstruction)
