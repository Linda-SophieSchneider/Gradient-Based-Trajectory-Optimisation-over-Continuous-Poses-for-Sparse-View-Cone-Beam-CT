"""MLX-native ramp filter — differentiable, no NumPy detour.

diffct_mlx's stock ``ramp_filter_3d`` converts to NumPy for the FFT, which
breaks the autograd chain.  We use ``mx.fft`` directly so the filtered
sinogram remains a differentiable function of the (volume, source) inputs
of the forward projection.

The filter is the standard discrete Ram-Lak ramp, ``2 |f|`` for frequencies
``f`` in cycles per sample, applied along the detector-u axis of a 3-D
sinogram of shape ``(n_views, n_u, n_v)``.
"""

from __future__ import annotations

import mlx.core as mx


def ramp_filter_3d_mlx(sinogram: mx.array) -> mx.array:
    """Apply the Ram-Lak ramp filter along axis 1 (detector-u) in MLX.

    Returns a sinogram of the same shape with the same dtype.  Suitable for
    use inside an MLX autograd graph; the only non-differentiable component
    is the construction of the ramp weights, which depend on the *shape*
    (a static quantity) rather than any traced array.
    """
    if sinogram.ndim != 3:
        raise ValueError(f"Expected 3D sinogram (n_views, n_u, n_v); got {sinogram.shape}")
    n_views, n_u, n_v = sinogram.shape

    # Discrete Ram-Lak ramp along axis=1.
    # mx.fft.fftfreq does not exist on this mlx build (0.30.0, pinned for
    # mlx-cuda) -- reproduce numpy.fft.fftfreq(n, d=1.0)'s formula directly:
    # [0, 1, ..., ceil(n/2)-1, -floor(n/2), ..., -1] / n.
    k = mx.arange(n_u, dtype=mx.float32)
    freqs = mx.where(k < (n_u + 1) // 2, k, k - n_u) / n_u   # (n_u,)
    ramp = (2.0 * mx.abs(freqs)).astype(mx.float32)
    ramp = ramp.reshape(1, n_u, 1)               # broadcast over views and v-axis

    sino_c = sinogram.astype(mx.complex64)
    sino_fft = mx.fft.fft(sino_c, axis=1)
    sino_fft = sino_fft * ramp.astype(mx.complex64)
    filtered = mx.fft.ifft(sino_fft, axis=1)
    return mx.real(filtered).astype(sinogram.dtype)
