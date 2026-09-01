"""Noise-aware Bayesian optimal-experimental-design (OED) view objective.

Lin et al.'s VCL surrogate maximises the clean information score
``γ^T R^{-1} γ`` with ``R = T T^T`` built from the per-view filtered
back-projection bases ``T_i``.  It is noise-blind: every view enters ``R``
with unit weight, so two view sets with equal information but very different
photon statistics receive the same score.

This module recasts view selection as optimal experimental design with a
photon-noise-weighted Fisher information.  Under a Beer-Lambert / Poisson
transmission model the projection-domain precision of view ``i`` scales with
the photon survival ``w_i = e^{-τ̄_i}`` (``τ̄_i`` = bundle line integral, the
same quantity the absorption term already computes).  The image-space Fisher
information is

    F = Σ_i w_i T_i T_i^T + ρ I        (ρ = ridge prior precision).

Two classic design criteria follow, both functions of the *k×k* weighted
correlation matrix ``R_w = diag(√w) R diag(√w)`` via the matrix-determinant
lemma / Woodbury identity (so we never form the huge image-space matrix):

  * D-optimality  ``log det(I + R_w/ρ)``      — total information (resolution).
  * A-optimality  ``k − ρ·tr((ρI + R_w)^{-1})`` — the effective number of
    well-determined modes (a reward that is *maximised*).  It is the
    source-dependent complement of the posterior variance ``tr(F^{-1})``;
    maximising A is what minimises that reconstruction-noise term.

The combined objective ``α·A + β·D`` is *maximised* (consistent with the
Adam-ascent convention of the selectors).  VCL is the special case of an
unweighted, information-only score; the photon weighting is what lets a
continuous selector pick lower-noise views that the noise-blind surrogate
cannot distinguish.
"""

from __future__ import annotations

import mlx.core as mx


# ---------------------------------------------------------------------------
# Custom-VJP scalar functions of a small SPD matrix M = ρI + R_w (k×k).
# The k×k solves run on the CPU stream because the GPU solver is not enabled.
# ---------------------------------------------------------------------------

@mx.custom_function
def _logdet_spd(M: mx.array) -> mx.array:
    """``log det M`` for an SPD matrix, via Cholesky."""
    with mx.stream(mx.cpu):
        L = mx.linalg.cholesky(M)
        return 2.0 * mx.sum(mx.log(mx.diagonal(L)))


@_logdet_spd.vjp
def _logdet_spd_vjp(primals, cotangent, _out):
    # d log det(M) = tr(M^{-1} dM)  ->  grad_M = cotangent · M^{-1}.
    # A single-input custom_function passes ``primals`` as the array itself.
    M = primals
    k = M.shape[0]
    with mx.stream(mx.cpu):
        m_inv = mx.linalg.solve(M, mx.eye(k, dtype=M.dtype))
    return cotangent * m_inv


@mx.custom_function
def _trace_inv_spd(M: mx.array) -> mx.array:
    """``tr(M^{-1})`` for an SPD matrix."""
    k = M.shape[0]
    with mx.stream(mx.cpu):
        m_inv = mx.linalg.solve(M, mx.eye(k, dtype=M.dtype))
    return mx.sum(mx.diagonal(m_inv))


@_trace_inv_spd.vjp
def _trace_inv_spd_vjp(primals, cotangent, _out):
    # d tr(M^{-1}) = -tr(M^{-1} dM M^{-1})  ->  grad_M = -cotangent · M^{-1} M^{-1}.
    # A single-input custom_function passes ``primals`` as the array itself.
    M = primals
    k = M.shape[0]
    with mx.stream(mx.cpu):
        m_inv = mx.linalg.solve(M, mx.eye(k, dtype=M.dtype))
    return -cotangent * (m_inv @ m_inv)


# ---------------------------------------------------------------------------
# OED information score
# ---------------------------------------------------------------------------

def oed_information(
    R: mx.array,
    weights: mx.array,
    *,
    ridge: float = 1e-2,
    alpha: float = 1.0,
    beta: float = 1.0,
    return_components: bool = False,
) -> mx.array:
    """Combined A + D optimal-design score (higher is better).

    Parameters
    ----------
    R : ``(k, k)``
        View-basis correlation matrix ``T T^T`` (as in the VCL context).
    weights : ``(k,)``
        Per-view measurement precision. In the simplest transmission model this
        is proportional to the surviving photon count ``I_0 e^{-τ̄_i}``.
    ridge : float
        Prior precision ρ; regularises the inverse and sets the
        information-vs-prior scale.
    alpha, beta : float
        Weights of the A-optimality (noise) and D-optimality (information)
        terms.  Both terms are O(k), so they are directly comparable.
    return_components : bool
        If True, also return the bare A and D terms (before α/β weighting),
        so a caller can sweep α/β offline without recomputing the solves.

    Returns
    -------
    Scalar mx.array ``α·A + β·D`` to be maximised, or ``(score, A, D)`` when
    ``return_components`` is set.
    """
    k = R.shape[0]
    sqrt_w = mx.sqrt(mx.maximum(weights, 0.0))
    R_w = (sqrt_w[:, None] * sqrt_w[None, :]) * R          # diag(√w) R diag(√w)
    M = ridge * mx.eye(k, dtype=R.dtype) + R_w

    # D-optimality: log det(I + R_w/ρ) = log det(M) − k log ρ.
    import math
    d_opt = _logdet_spd(M) - k * math.log(ridge)
    # A-optimality (S-dependent part): k − ρ·tr(M^{-1}) = Σ_j λ_j/(ρ+λ_j) ∈ [0, k].
    a_opt = k - ridge * _trace_inv_spd(M)

    score = alpha * a_opt + beta * d_opt
    if return_components:
        return score, a_opt, d_opt
    return score


def oed_loss_continuous(
    sources: mx.array,
    ctx,
    tau_bar: mx.array,
    *,
    photon_count: float = 1.0,
    ridge: float = 1e-2,
    alpha: float = 1.0,
    beta: float = 1.0,
    return_components: bool = False,
) -> mx.array:
    """Photon-noise-weighted OED score for continuous source positions.

    Builds the VCL view-basis correlation ``R = T T^T`` from *ctx* and weights
    each view by its expected precision ``w_i = I_0 e^{-τ̄_i}``, where
    ``tau_bar`` is the differentiable bundle line integral per source and
    ``photon_count = I_0`` is the emitted photon count per detector pixel.
    Returns the combined A + D optimal-design score (higher is better), to be
    added to the Adam-ascent objective. With ``return_components`` it also
    returns the bare A and D terms for offline α/β sweeps.
    """
    from .vcl_diff import view_basis_matrix

    # T_mat is B^T in the paper's notation: row i is t_i = T_{θ_i} x_{r1}
    # (the per-view projected reference direction), NOT the N×N operator
    # stack T(L).  Hence R = B^T B = T_mat T_mat^T is the k×k correlation.
    T_mat = view_basis_matrix(sources, ctx)        # (k, NS) == B^T
    R = T_mat @ mx.transpose(T_mat)                # (k, k)
    w = float(photon_count) * mx.exp(-tau_bar)
    return oed_information(R, w, ridge=ridge, alpha=alpha, beta=beta,
                          return_components=return_components)
