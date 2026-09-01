"""Kinematic reparametrisations of the source positions (§3.4).

Each callable maps low-dimensional parameters to ``(k, 3)`` source
positions; the chain rule composes automatically with MLX autograd, so
``mx.grad(objective_via_params)`` returns gradients w.r.t. the parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class CircularArc:
    """Single-axis rotation around the world z-axis at fixed radius and height."""

    sid: float
    z: float = 0.0

    def __call__(self, theta: mx.array) -> mx.array:
        z = mx.full(theta.shape, self.z, dtype=theta.dtype)
        return mx.stack([-self.sid * mx.sin(theta), self.sid * mx.cos(theta), z], axis=-1)


@dataclass(frozen=True)
class Helix:
    """Helical path: circular in xy with z linear in the rotation angle."""

    sid: float
    pitch: float

    def __call__(self, theta: mx.array) -> mx.array:
        z = self.pitch * theta / (2.0 * mx.pi)
        return mx.stack([-self.sid * mx.sin(theta), self.sid * mx.cos(theta), z], axis=-1)


@dataclass(frozen=True)
class TwoAxisGantry:
    """Two-axis Cartesian gantry: azimuth θ around z, then elevation φ from xy-plane.

    Source position:
        x = -sid · cos(φ) · sin(θ)
        y =  sid · cos(φ) · cos(θ)
        z =  sid · sin(φ)

    Parameters
    ----------
    sid : float
        Source-to-isocenter distance.

    Call
    ----
    params : ``(k, 2)`` array — ``params[:, 0]`` is θ (azimuth), ``params[:, 1]`` is φ (elevation).

    Returns
    -------
    sources : ``(k, 3)``
    """

    sid: float

    def __call__(self, params: mx.array) -> mx.array:
        theta = params[:, 0]
        phi = params[:, 1]
        cos_phi = mx.cos(phi)
        x = -self.sid * cos_phi * mx.sin(theta)
        y = self.sid * cos_phi * mx.cos(theta)
        z = self.sid * mx.sin(phi)
        return mx.stack([x, y, z], axis=-1)


@dataclass(frozen=True)
class SmoothTwoAxisGantry:
    """Two-axis gantry whose box constraints are built into the chart.

    ``CArmTwoAxisGantry`` enforces its envelope by clipping, i.e. by projecting
    after the update. The gradient there still points out of the feasible set,
    so a pose that overshoots a boundary lands on it and stays: outside the box
    the clipped forward map is constant, its derivative vanishes, and Adam's
    momentum keeps pressing outward. This class removes the failure mode
    instead of repairing it, by making every parameter value feasible by
    construction:

        angle = centre + half_range * tanh(raw)

    so the reachable set is the image of the whole parameter space, the
    gradient is always tangent to it, and no projection is needed. A pose can
    approach a boundary arbitrarily closely but never has to be pushed back.
    An axis whose range covers the full circle is left periodic and untouched,
    because a tanh would break its wrap-around.

    ``inverse`` maps feasible angles back to raw parameters for the
    initialisation, with a small margin so the greedy start does not begin at
    an infinite raw coordinate.
    """

    sid: float
    theta_min: float = -110.0 * mx.pi / 180.0
    theta_max: float = 110.0 * mx.pi / 180.0
    phi_min: float = -45.0 * mx.pi / 180.0
    phi_max: float = 45.0 * mx.pi / 180.0
    inverse_margin: float = 0.999

    @property
    def theta_periodic(self) -> bool:
        return (self.theta_max - self.theta_min) >= 2.0 * mx.pi - 1e-6

    def _squash(self, raw, lo, hi):
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        return centre + half * mx.tanh(raw)

    def clamp(self, params: mx.array) -> mx.array:
        # Nothing to project: the chart cannot leave the feasible set.
        return params

    def __call__(self, params: mx.array) -> mx.array:
        raw_theta = params[:, 0]
        theta = (raw_theta if self.theta_periodic
                 else self._squash(raw_theta, self.theta_min, self.theta_max))
        phi = self._squash(params[:, 1], self.phi_min, self.phi_max)
        cos_phi = mx.cos(phi)
        x = -self.sid * cos_phi * mx.sin(theta)
        y = self.sid * cos_phi * mx.cos(theta)
        z = self.sid * mx.sin(phi)
        return mx.stack([x, y, z], axis=-1)

    def inverse(self, angles: mx.array) -> mx.array:
        """Raw parameters for given (theta, phi) angles inside the envelope."""
        def unsquash(a, lo, hi):
            centre = 0.5 * (lo + hi)
            half = max(0.5 * (hi - lo), 1e-9)
            t = mx.clip((a - centre) / half,
                        -self.inverse_margin, self.inverse_margin)
            return mx.arctanh(t)
        theta = (angles[:, 0] if self.theta_periodic
                 else unsquash(angles[:, 0], self.theta_min, self.theta_max))
        phi = unsquash(angles[:, 1], self.phi_min, self.phi_max)
        return mx.stack([theta, phi], axis=-1)


@dataclass(frozen=True)
class CArmTwoAxisGantry:
    """Two-axis gantry with realistic box constraints on orbit and angulation.

    Defaults model a typical medical C-arm envelope:
    - orbital rotation ``theta`` limited to about ±110 degrees
    - cranial/caudal angulation ``phi`` limited to about ±45 degrees
    """

    sid: float
    theta_min: float = -110.0 * mx.pi / 180.0
    theta_max: float = 110.0 * mx.pi / 180.0
    phi_min: float = -45.0 * mx.pi / 180.0
    phi_max: float = 45.0 * mx.pi / 180.0

    def clamp(self, params: mx.array) -> mx.array:
        theta = mx.clip(params[:, 0], self.theta_min, self.theta_max)
        phi = mx.clip(params[:, 1], self.phi_min, self.phi_max)
        return mx.stack([theta, phi], axis=-1)

    def __call__(self, params: mx.array) -> mx.array:
        params = self.clamp(params)
        theta = params[:, 0]
        phi = params[:, 1]
        cos_phi = mx.cos(phi)
        x = -self.sid * cos_phi * mx.sin(theta)
        y = self.sid * cos_phi * mx.cos(theta)
        z = self.sid * mx.sin(phi)
        return mx.stack([x, y, z], axis=-1)


class Free3D:
    """Identity reparametrisation: parameters are the 3-D source positions."""

    def __call__(self, sources: mx.array) -> mx.array:
        return sources
