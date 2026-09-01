"""Gradient ascent drivers with optional temperature annealing.

The objective is supplied as ``coverage_fn(params, step)`` so the caller can
anneal smoothing temperatures (``sigma``, ``beta_pixel``, ``beta_frac``)
along the schedule recommended in §2 of the future-work note.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx


def anneal(step: int, n_steps: int, start: float, end: float,
           kind: str = "geometric") -> float:
    """Anneal a scalar from ``start`` to ``end`` over ``n_steps`` iterations."""
    if n_steps <= 1:
        return end
    t = step / (n_steps - 1)
    if kind == "linear":
        return start + (end - start) * t
    return start * (end / start) ** t


def gradient_ascent(
    coverage_fn: Callable[[mx.array, int], mx.array],
    params: mx.array,
    *,
    lr: float = 1e-3,
    n_steps: int = 200,
    tol_grad: float = 0.0,
    tol_rel: float = 0.0,
    callback: Callable[[int, mx.array, float], None] | None = None,
) -> tuple[mx.array, list[float]]:
    """Plain gradient ascent on ``coverage_fn(params, step) -> scalar``.

    Parameters
    ----------
    coverage_fn :
        Objective; called as ``coverage_fn(params, step)`` -> scalar.
    params :
        Initial parameter array.
    lr :
        Learning rate.
    n_steps :
        Maximum number of steps.
    tol_grad :
        Stop when the L∞ norm of the gradient falls below this threshold.
        Set to 0 (default) to disable.
    tol_rel :
        Stop when the relative improvement over the last step falls below
        this threshold.  Set to 0 (default) to disable.
    callback :
        Optional ``callback(step, params, value)`` called for the evaluated
        iterate before the corresponding gradient update.

    Returns
    -------
    params_final :
        Optimised parameter array.
    history :
        Per-step objective values.
    """
    history: list[float] = []
    prev_value: float | None = None

    for step in range(n_steps):
        evaluated_params = params
        value, grad = mx.value_and_grad(
            lambda p: coverage_fn(p, step)
        )(evaluated_params)
        mx.eval(value, grad)
        current = float(value)
        history.append(current)

        if callback is not None:
            callback(step, evaluated_params, current)

        if tol_grad > 0.0:
            if float(mx.max(mx.abs(grad))) < tol_grad:
                break

        if tol_rel > 0.0 and prev_value is not None:
            rel = abs(current - prev_value) / (abs(prev_value) + 1e-12)
            if rel < tol_rel:
                break

        prev_value = current
        params = evaluated_params + lr * grad
        mx.eval(params)

    return params, history


@dataclass
class _AdamState:
    m: mx.array
    v: mx.array
    t: int = 0


def central_fd_grad(
    fn: Callable[[mx.array], mx.array],
    params: mx.array,
    fd_step: float,
) -> mx.array:
    """Coordinate-wise central-difference gradient of a scalar objective.

    Evaluates ``fn`` exactly ``2 * params.size`` times.  This is the matched
    finite-difference estimator for derivative-only ablations: the objective
    is EXACTLY the one the analytic mode differentiates — only the gradient
    estimator changes (REV-P0-01).
    """
    import numpy as np

    flat = np.asarray(params, dtype=np.float64).reshape(-1)
    g = np.zeros(flat.size, dtype=np.float64)
    for i in range(flat.size):
        p_plus = flat.copy()
        p_plus[i] += fd_step
        p_minus = flat.copy()
        p_minus[i] -= fd_step
        f_p = float(fn(mx.array(
            p_plus.astype(np.float32).reshape(params.shape))))
        f_m = float(fn(mx.array(
            p_minus.astype(np.float32).reshape(params.shape))))
        g[i] = (f_p - f_m) / (2.0 * fd_step)
    return mx.array(g.astype(np.float32).reshape(params.shape))


def cmaes_ascent(
    coverage_fn: Callable[[mx.array, int], mx.array],
    params: mx.array,
    *,
    sigma0: float,
    budget_evals: int,
    seed: int = 0,
    project_fn: Callable[[mx.array], mx.array] | None = None,
    popsize: int | None = None,
) -> tuple[mx.array, list[float]]:
    """Derivative-free CMA-ES ascent on the ``adam_ascent`` objective
    interface (the REV-P1-08 budget-matched comparator).

    Maximises ``coverage_fn(params, 0)`` with pycma — the objective closure,
    initial iterate, and manifold projection are EXACTLY the ones the
    gradient path uses; only the optimiser differs.  Step-dependent
    annealing schedules are frozen at step 0, so matched comparisons must
    use constant schedules (the bundle-selector configs do).

    Returns ``(best_params, history)`` where ``history`` holds the
    best-so-far objective per generation, so any smaller evaluation budget
    can be read off the curve post hoc.
    """
    import numpy as np
    import cma

    shape = params.shape
    x0 = np.asarray(params, dtype=np.float64).reshape(-1)

    def neg_obj(x):
        p = mx.array(x.astype(np.float32).reshape(shape))
        if project_fn is not None:
            p = project_fn(p)
        return -float(coverage_fn(p, 0))

    opts = {
        # pycma treats seed 0/None as "random"; shift to stay deterministic.
        "seed": int(seed) + 1,
        "maxfevals": int(budget_evals),
        "verbose": -9,
    }
    if popsize is not None:
        opts["popsize"] = int(popsize)
    es = cma.CMAEvolutionStrategy(x0, float(sigma0), opts)
    history: list[float] = []
    best = float("-inf")
    while not es.stop():
        xs = es.ask()
        vals = [neg_obj(x) for x in xs]
        es.tell(xs, vals)
        best = max(best, -min(vals))
        history.append(best)
    xbest = es.result.xbest if es.result.xbest is not None else x0
    out = mx.array(np.asarray(xbest, dtype=np.float32).reshape(shape))
    if project_fn is not None:
        out = project_fn(out)
    return out, history


def adam_ascent(
    coverage_fn: Callable[[mx.array, int], mx.array],
    params: mx.array,
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    n_steps: int = 200,
    tol_grad: float = 0.0,
    tol_rel: float = 0.0,
    callback: Callable[[int, mx.array, float], None] | None = None,
    project_fn: Callable[[mx.array], mx.array] | None = None,
    lr_schedule: str = "constant",         # "constant" or "cosine"
    lr_min: float = 0.0,
    patience: int | None = None,           # early stopping after N steps w/o improvement
    return_best: bool = True,              # return best-seen iterate, not last
    noise_fn: Callable[[mx.array, int, float], mx.array] | None = None,
    grad_mode: str = "analytic",           # "analytic" or "fd_central"
    fd_step: float = 1e-3,                 # step for grad_mode="fd_central"
) -> tuple[mx.array, list[float]]:
    """Adam gradient ascent on ``coverage_fn(params, step) -> scalar``.

    Parameters
    ----------
    coverage_fn :
        Objective; called as ``coverage_fn(params, step)`` -> scalar.
    params :
        Initial parameter array.
    lr :
        Learning rate (step size).
    beta1, beta2 :
        Exponential decay rates for first and second moment estimates.
    eps :
        Numerical stability constant.
    n_steps :
        Maximum number of steps.
    tol_grad :
        Stop when the L∞ gradient norm falls below this value (0 = disabled).
    tol_rel :
        Stop when relative improvement per step falls below this value
        (0 = disabled).
    callback :
        Optional ``callback(step, params, value)`` called for the evaluated
        iterate before the corresponding Adam update.

    Returns
    -------
    params_final :
        Optimised parameter array.
    history :
        Per-step objective values.
    """
    import math as _math
    state = _AdamState(m=mx.zeros_like(params), v=mx.zeros_like(params))
    history: list[float] = []
    prev_value: float | None = None

    best_params = params
    best_value = float("-inf")
    steps_since_improvement = 0

    for step in range(n_steps):
        # Optional cosine LR schedule from lr down to lr_min over n_steps.
        if lr_schedule == "cosine":
            t_frac = step / max(n_steps - 1, 1)
            lr_t = lr_min + 0.5 * (lr - lr_min) * (1.0 + _math.cos(_math.pi * t_frac))
        else:
            lr_t = lr

        # ``value`` and ``grad`` both belong to this pre-update iterate.  Keep
        # the array explicitly so best-iterate tracking and callbacks cannot
        # accidentally pair that value with the parameters produced below.
        evaluated_params = params
        if grad_mode == "analytic":
            value, grad = mx.value_and_grad(
                lambda p: coverage_fn(p, step)
            )(evaluated_params)
        elif grad_mode == "fd_central":
            value = coverage_fn(evaluated_params, step)
            grad = central_fd_grad(
                lambda p: coverage_fn(p, step), evaluated_params, fd_step)
        else:
            raise ValueError(f"unknown grad_mode {grad_mode!r}")
        mx.eval(value, grad)

        current = float(value)
        history.append(current)

        # Track the iterate that was actually evaluated.  Patience uses the
        # same comparison even when the caller requests the final iterate.
        if current > best_value:
            best_value = current
            if return_best:
                best_params = evaluated_params
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        if callback is not None:
            callback(step, evaluated_params, current)

        # Stop before forming an update that will never be evaluated or
        # returned as the best iterate.
        if patience is not None and steps_since_improvement >= patience:
            break

        if tol_grad > 0.0 and float(mx.max(mx.abs(grad))) < tol_grad:
            break

        if tol_rel > 0.0 and prev_value is not None:
            rel = abs(current - prev_value) / (abs(prev_value) + 1e-12)
            if rel < tol_rel:
                break

        prev_value = current

        state.t += 1
        state.m = beta1 * state.m + (1.0 - beta1) * grad
        state.v = beta2 * state.v + (1.0 - beta2) * grad * grad
        mx.eval(state.m, state.v)

        m_hat = state.m / (1.0 - beta1 ** state.t)
        v_hat = state.v / (1.0 - beta2 ** state.t)
        params = params + lr_t * m_hat / (mx.sqrt(v_hat) + eps)

        # Optional Langevin / SGLD-style noise injection.  noise_fn is
        # responsible for the full magnitude (including any sqrt(2*lr_t*T_t)
        # factor and tangent-plane projection) — the optimizer just adds it.
        if noise_fn is not None:
            noise = noise_fn(params, step, lr_t)
            params = params + noise

        if project_fn is not None:
            params = project_fn(params)

        mx.eval(params)

    out = best_params if return_best else params
    return out, history
