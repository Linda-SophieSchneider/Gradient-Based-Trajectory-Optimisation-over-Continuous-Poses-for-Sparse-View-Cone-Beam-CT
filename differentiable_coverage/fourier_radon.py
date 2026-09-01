"""Volume-aware Radon-plane importance weights via the 3D Fourier slice.

The 3D Radon plane transform of a volume $V$ obeys the Fourier slice
identity

.. math::
    \\mathcal{F}_1\\{R(V)(\\mu, \\cdot)\\}(\\nu) = \\hat V(\\nu\\,\\mu),

so the squared energy associated with a Radon plane normal $\\mu$ is the
line integral of the 3D power spectrum along the line $t \\mapsto t\\,\\mu$
through the origin (Natterer 1986, §II.1).  This module computes a
discrete approximation by trilinearly sampling $|\\hat V|^2$ along
those lines.

These weights are used as ``direction_weights`` in
:func:`differentiable_coverage.score.coverage_covariance_information`
to give the additive soft-Tuy / $I_\\mathrm{cov}$ objective access to a
phantom-aware view-importance signal *without* the SART forward operator
that VCLS' $R = T^\\top T$ requires.  The cost is a single 3D FFT (seconds
on Apple Silicon for $384^3$) plus an $O(z)$ slice sampling per phantom.
"""

from __future__ import annotations

import mlx.core as mx

from .absorption_bundle import _trilinear_sample


def fourier_radon_weights(
    volume: mx.array,
    radon_normals: mx.array,
    *,
    n_samples: int = 65,
    normalize: bool = True,
) -> mx.array:
    """Fourier-slice importance weights for Radon plane normals.

    Parameters
    ----------
    volume
        Reference volume of shape ``(Z, Y, X)``.  Cast to ``float32``
        internally.  Larger volumes give finer frequency resolution but
        only logarithmic-time FFT cost.
    radon_normals
        Unit vectors of shape ``(z, 3)`` in world ``(x, y, z)`` order,
        matching the convention used throughout
        :mod:`differentiable_coverage.score`.
    n_samples
        Number of points sampled along each Fourier line.  An odd value
        guarantees that the DC bin is hit exactly.
    normalize
        If ``True`` (the default), the returned weights sum to one
        (so they can be used as a probability-like importance) and
        become scale-invariant in $V$.  If ``False``, raw energies are
        returned.

    Returns
    -------
    weights
        Non-negative array of shape ``(z,)``.

    Notes
    -----
    For a real volume $V$, $|\\hat V|^2$ is even, so the weights are
    automatically antipodally symmetric: $w(\\mu) = w(-\\mu)$.
    """
    if radon_normals.ndim != 2 or radon_normals.shape[-1] != 3:
        raise ValueError(
            f"radon_normals must have shape (z, 3); got {radon_normals.shape}"
        )

    V = mx.array(volume, dtype=mx.float32)
    Z, Y, X = V.shape

    # Compute the centred 3D power spectrum.  We work on the CPU stream
    # because the GPU FFT path is the default but its memory footprint
    # is large for big volumes; this keeps the API uniform across sizes.
    V_hat = mx.fft.fftn(V)
    power = (mx.abs(V_hat)) ** 2
    power_centered = mx.fft.fftshift(power)

    # The volume is stored in (Z, Y, X) layout, while ``radon_normals``
    # lives in world (x, y, z) order.  Permute axes to align them with
    # the array's index convention.
    mu_idx = mx.stack(
        [radon_normals[:, 2], radon_normals[:, 1], radon_normals[:, 0]],
        axis=-1,
    )

    # Sample positions stay within the array.
    L = float(min(Z, Y, X)) / 2.0 - 1.0
    t = mx.linspace(-1.0, 1.0, n_samples) * L
    centre = mx.array(
        [float(Z // 2), float(Y // 2), float(X // 2)], dtype=mx.float32,
    )

    # positions shape: (n_normals, n_samples, 3) in (z, y, x) index order.
    positions = (
        centre[None, None, :]
        + t[None, :, None] * mu_idx[:, None, :]
    )

    samples = _trilinear_sample(power_centered, positions)  # (n_norm, n_samp)
    weights = mx.sum(samples, axis=-1)                       # (n_norm,)

    if normalize:
        total = mx.sum(weights)
        weights = weights / mx.maximum(total, 1e-12)
    return weights
