"""Reconstruction-quality metrics.

All metrics consume reconstructions and references as ``mx.array`` or
``np.ndarray`` and return Python floats.  ``mx.eval`` is called explicitly
where needed so that timings reflect actual compute, not pending graphs.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _validate_pair(recon, reference) -> tuple[np.ndarray, np.ndarray]:
    a = _to_numpy(recon)
    b = _to_numpy(reference)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: recon {a.shape}, reference {b.shape}")
    return a, b


def _mask_array(mask, shape) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.shape != shape:
        raise ValueError(f"mask shape mismatch: mask {m.shape}, volume {shape}")
    if not m.any():
        raise ValueError("ROI mask is empty")
    return m


def _masked_pair(a: np.ndarray, b: np.ndarray, mask=None) -> tuple[np.ndarray, np.ndarray]:
    if mask is None:
        return a, b
    m = _mask_array(mask, a.shape)
    return a[m], b[m]


def _crop_to_mask(a: np.ndarray, b: np.ndarray, mask) -> tuple[np.ndarray, np.ndarray]:
    m = _mask_array(mask, a.shape)
    coords = np.argwhere(m)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    sl = tuple(slice(int(lo_i), int(hi_i)) for lo_i, hi_i in zip(lo, hi))
    return a[sl], b[sl]


def rmse(recon, reference, *, mask=None) -> float:
    """Root-mean-squared error over the whole volume or a masked ROI."""
    a, b = _validate_pair(recon, reference)
    a, b = _masked_pair(a, b, mask)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def psnr(recon, reference, peak: float | None = None, *, mask=None) -> float:
    """Peak-signal-to-noise ratio in dB.

    ``peak`` defaults to ``max(reference) - min(reference)``; pass an explicit
    value to make PSNR comparable across phantoms.
    """
    a, b = _validate_pair(recon, reference)
    a, b = _masked_pair(a, b, mask)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    if peak is None:
        peak = float(b.max() - b.min())
        if peak <= 0.0:
            peak = float(b.max()) if float(b.max()) > 0 else 1.0
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def ssim(recon, reference, *, data_range: float | None = None,
         sigma: float = 1.5, mask=None) -> float:
    """3-D structural similarity index between ``recon`` and ``reference``.

    Canonical Wang et al. (2004) definition extended to 3-D: an isotropic
    Gaussian window with ``sigma`` (default 1.5 ⇒ effective win_size 11) is
    applied via skimage's ``structural_similarity`` with ``channel_axis=None``
    so the full 3-D volume is treated as one scalar field (no per-slice
    averaging, no per-channel splitting).

    ``data_range`` defaults to ``max(reference) - min(reference)``.
    """
    from skimage.metrics import structural_similarity

    a, b = _validate_pair(recon, reference)
    if mask is not None:
        a, b = _crop_to_mask(a, b, mask)
    if a.ndim != 3:
        raise ValueError(f"ssim expects a 3-D volume; got shape {a.shape}")
    if data_range is None:
        data_range = float(b.max() - b.min())
        if data_range <= 0:
            data_range = float(b.max()) if float(b.max()) > 0 else 1.0
    kwargs = {
        "data_range": data_range,
        "channel_axis": None,
        "gaussian_weights": True,
        "sigma": sigma,
        "use_sample_covariance": False,
    }
    min_dim = min(a.shape)
    if min_dim < 11:
        kwargs["gaussian_weights"] = False
        kwargs["win_size"] = max(3, min_dim if (min_dim % 2 == 1) else (min_dim - 1))
    return float(structural_similarity(
        b.astype("float32"), a.astype("float32"), **kwargs
    ))


def nrmse(recon, reference, *, mask=None) -> float:
    """Normalised RMSE in the convention of Lin et al. (TPAMI 2025).

    $$\\mathrm{NRMSE}(\\hat{x}, x) = \\| \\hat{x} - x \\| / \\| x \\| .$$

    Unitless; lower is better.  This is the metric used in Lin et al.'s
    reported tables, so adding it makes direct comparison straightforward.
    """
    a, b = _validate_pair(recon, reference)
    a, b = _masked_pair(a, b, mask)
    num = float(np.sqrt(np.sum((a - b) ** 2)))
    den = float(np.sqrt(np.sum(b ** 2)))
    if den <= 0.0:
        return float("nan")
    return num / den


def hfen(recon, reference, *, sigma: float = 1.5, mask=None) -> float:
    """High-Frequency Error Norm: L2 norm of the Laplacian-of-Gaussian
    of the residual.

    HFEN is sensitive to streak artefacts and edge degradation that PSNR
    averages away.  We use the standard isotropic LoG with a Gaussian
    of bandwidth ``sigma`` voxels, applied independently along all three
    spatial axes.  Lower is better.
    """
    from scipy.ndimage import gaussian_laplace

    a, b = _validate_pair(recon, reference)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    log_recon = gaussian_laplace(a, sigma=sigma)
    log_ref = gaussian_laplace(b, sigma=sigma)
    diff = log_recon - log_ref
    if mask is not None:
        diff = diff[_mask_array(mask, diff.shape)]
    return float(np.sqrt(np.sum(diff ** 2)))


def frc_resolution(recon, reference, *, threshold: float = 0.5,
                   n_bins: int | None = None) -> float:
    """Spatial resolution from the Fourier Ring Correlation (volume FRC).

    Computes the 3-D FRC

      $$\\mathrm{FRC}(q) = \\frac{\\sum_{|k|=q} F(k) \\overline{G(k)}}
               {\\sqrt{\\sum |F(k)|^2 \\sum |G(k)|^2}}$$

    where $F, G$ are the Fourier transforms of the recon and reference and
    the sums are over a thin shell at radius $|k| = q$.  The reported
    resolution is the cutoff frequency at which FRC drops below
    ``threshold = 0.5`` (the standard one-bit criterion), expressed as the
    cycles-per-voxel value.  Lower frequency = coarser resolution = worse.

    Returns the cutoff frequency in [0, 0.5] cycles per voxel.  If FRC
    never crosses the threshold the Nyquist frequency 0.5 is returned.
    """
    a = _to_numpy(recon).astype(np.float32)
    b = _to_numpy(reference).astype(np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: recon {a.shape}, reference {b.shape}")
    F = np.fft.fftshift(np.fft.fftn(a))
    G = np.fft.fftshift(np.fft.fftn(b))
    nz, ny, nx = a.shape
    z = np.arange(nz) - nz // 2
    y = np.arange(ny) - ny // 2
    x = np.arange(nx) - nx // 2
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    # Frequency magnitude in cycles per voxel.
    r = np.sqrt((zz / nz) ** 2 + (yy / ny) ** 2 + (xx / nx) ** 2)
    n_bins = n_bins or min(nz, ny, nx) // 2
    bin_edges = np.linspace(0, 0.5, n_bins + 1)
    bin_idx = np.minimum(np.digitize(r, bin_edges) - 1, n_bins - 1)

    FG = F * np.conj(G)
    F_mag2 = (np.abs(F) ** 2).astype(np.float64)
    G_mag2 = (np.abs(G) ** 2).astype(np.float64)

    frc = np.zeros(n_bins, dtype=np.float64)
    for j in range(n_bins):
        sel = (bin_idx == j)
        if not sel.any():
            frc[j] = 0.0
            continue
        num = float(np.real(FG[sel].sum()))
        den = float(np.sqrt(F_mag2[sel].sum() * G_mag2[sel].sum()) + 1e-12)
        frc[j] = num / den

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    below = np.where(frc < threshold)[0]
    if below.size == 0:
        return float(bin_centers[-1])
    first_below = int(below[0])
    if first_below == 0:
        return float(bin_centers[0])
    # Linear interpolation for sub-bin precision
    y1, y2 = frc[first_below - 1], frc[first_below]
    x1, x2 = bin_centers[first_below - 1], bin_centers[first_below]
    if y1 == y2:
        return float(x2)
    return float(x1 + (threshold - y1) * (x2 - x1) / (y2 - y1))


def slice_psnr(recon, reference, *, axis: int = 0,
               peak: float | None = None) -> np.ndarray:
    """PSNR per 2-D slice along the chosen axis.

    Returns a 1-D array of slice PSNRs in dB, length equal to the size of
    ``axis``.  Useful for diagnosing axial-position-dependent
    reconstruction quality (e.g. cone-beam artefacts at the volume edges).
    """
    a = _to_numpy(recon).astype(np.float32)
    b = _to_numpy(reference).astype(np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: recon {a.shape}, reference {b.shape}")
    if peak is None:
        peak = float(b.max() - b.min())
        if peak <= 0.0:
            peak = float(b.max()) if float(b.max()) > 0 else 1.0
    a = np.moveaxis(a, axis, 0)
    b = np.moveaxis(b, axis, 0)
    n = a.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mse = float(np.mean((a[i] - b[i]) ** 2))
        out[i] = float("inf") if mse <= 0 else (
            20.0 * math.log10(peak) - 10.0 * math.log10(mse)
        )
    return out


def _centered_sphere_mask(shape: tuple[int, int, int],
                          roi_radius_frac: float) -> np.ndarray:
    nz, ny, nx = shape
    z = np.linspace(-1.0, 1.0, nz)
    y = np.linspace(-1.0, 1.0, ny)
    x = np.linspace(-1.0, 1.0, nx)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return (zz**2 + yy**2 + xx**2) <= roi_radius_frac**2


def roi_rmse(recon, reference, *, roi_radius_frac: float = 0.5,
             mask=None) -> float:
    """RMSE restricted to a centred spherical ROI or an explicit mask.

    ``roi_radius_frac`` is the ROI radius as a fraction of half the smallest
    spatial extent.  ``0.5`` carves out the central half of the volume — the
    region where the phantom is interesting and where edge artefacts are
    least dominant.
    """
    a, b = _validate_pair(recon, reference)
    roi_mask = mask
    if roi_mask is None:
        roi_mask = _centered_sphere_mask(a.shape, roi_radius_frac)
    return rmse(a, b, mask=roi_mask)


def roi_psnr(recon, reference, peak: float | None = None, *,
             roi_radius_frac: float = 0.5, mask=None) -> float:
    a, b = _validate_pair(recon, reference)
    roi_mask = mask
    if roi_mask is None:
        roi_mask = _centered_sphere_mask(a.shape, roi_radius_frac)
    return psnr(a, b, peak=peak, mask=roi_mask)


def roi_nrmse(recon, reference, *, roi_radius_frac: float = 0.5,
              mask=None) -> float:
    a, b = _validate_pair(recon, reference)
    roi_mask = mask
    if roi_mask is None:
        roi_mask = _centered_sphere_mask(a.shape, roi_radius_frac)
    return nrmse(a, b, mask=roi_mask)


def roi_hfen(recon, reference, *, sigma: float = 1.5,
             roi_radius_frac: float = 0.5, mask=None) -> float:
    a, b = _validate_pair(recon, reference)
    roi_mask = mask
    if roi_mask is None:
        roi_mask = _centered_sphere_mask(a.shape, roi_radius_frac)
    return hfen(a, b, sigma=sigma, mask=roi_mask)


def roi_ssim(recon, reference, *, data_range: float | None = None,
             sigma: float = 1.5, roi_radius_frac: float = 0.5,
             mask=None) -> float:
    a, b = _validate_pair(recon, reference)
    roi_mask = mask
    if roi_mask is None:
        roi_mask = _centered_sphere_mask(a.shape, roi_radius_frac)
    return ssim(a, b, data_range=data_range, sigma=sigma, mask=roi_mask)
