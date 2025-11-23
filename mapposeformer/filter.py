"""The filter the corrections are fed back through.

An SE(2) Kalman filter over the pose: odometry moves the estimate, one model
output corrects it, and the caller hands the result back as the next frame's
prior. ``engine/sequence.py`` drives it and says why that loop is the
experiment.

**The correction is the innovation.** The model is handed the filter's own
estimate as its prior, and it returns ``delta`` such that
``compose(prior, delta)`` is where it believes the vehicle is. The filter's
predicted correction is zero by construction -- it already believes its own
estimate -- so ``delta`` *is* the innovation, with no change of frame in
between. That is why the covariance here is carried in the body frame: it is
the frame the model reports in, and keeping it there costs one adjoint at
prediction time and saves one at every update.

**Two gates, both of them the model's own.** ``trust`` asks whether a frame
looks like one the model gets right and ``mass`` asks whether there was any
evidence at all. Neither compares the answer against the prior, so neither
can catch a frame that is confidently wrong -- what absorbs those is the
averaging itself, and ``docs/RESULTS.md`` shows the tail going to zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from mapposeformer import geometry as G


@dataclass(frozen=True)
class FilterParams:
    """Gates and noise. The defaults are the evaluator's, so open loop and
    closed loop refuse the same frames for the same reasons."""

    trust_threshold: float = 0.5
    """Minimum ``sigmoid(trust_logit)``. Matches ``engine/evaluator.py``."""

    min_mass: float = 4.0
    """Minimum assignment mass, roughly an effective correspondence count.
    Matches ``engine/evaluator.py``."""

    drift_frac: float = 0.01
    yaw_drift_deg_per_m: float = 0.02
    """Odometry error per metre travelled, as process noise. These mirror
    ``data/sample.py``'s ``EgoParams``, because the filter should model the
    odometry it is actually given rather than a tuned approximation of it."""

    init_sigma_long_m: float = 1.5
    init_sigma_lat_m: float = 0.5
    init_sigma_yaw_deg: float = 1.0
    """The first frame's prior, which has no predecessor to inherit from.
    These mirror ``PriorParams``, so the sequence starts where an open-loop
    frame would."""

    floor_m: float = 0.01
    floor_rad: float = 1e-4
    """A diagonal floor. A filter that has accepted a run of confident updates
    can otherwise drive its covariance low enough that every later correction
    earns a negligible gain -- an estimate too sure of itself to be moved,
    which looks like divergence and is really the arithmetic of the update."""


def adjoint(t: Tensor) -> Tensor:
    """@brief The SE(2) adjoint of ``t``, which moves a body-frame
    perturbation through a rigid motion.

    A body-frame error means ``true = compose(estimate, eps)``. Composing the
    estimate with ``u`` on the right leaves the error in the *old* body frame,
    and this is what carries it into the new one.

    @param t ``(3,)`` pose.
    @return ``(3, 3)``.
    """
    c, s = torch.cos(t[2]), torch.sin(t[2])
    out = torch.zeros(3, 3, dtype=t.dtype)
    out[0, 0], out[0, 1], out[0, 2] = c, -s, t[1]
    out[1, 0], out[1, 1], out[1, 2] = s, c, -t[0]
    out[2, 2] = 1.0
    return out


@dataclass
class Step:
    """What one frame did, for the sequence metrics to read afterwards."""

    accepted: bool
    nis: float
    reason: str
    """Empty when accepted, else which gate refused it: ``trust`` or ``mass``.
    The first gate that fires wins, so this is a cause and not a set."""


class LocalizationKF:
    """An SE(2) Kalman filter over the pose, corrected by the model.

    The state is an absolute pose and a body-frame covariance. ``predict``
    moves it by odometry, ``update`` folds in one model output, and the caller
    hands ``pose`` back to the next frame as its prior -- which is what closes
    the loop.
    """

    def __init__(self, pose: Tensor, p: FilterParams = FilterParams()):
        """@param pose ``(3,)`` the starting estimate, in the world frame."""
        self.p = p
        self.pose = pose.detach().clone().double()
        self.cov = torch.diag(
            torch.tensor(
                [
                    p.init_sigma_long_m**2,
                    p.init_sigma_lat_m**2,
                    math.radians(p.init_sigma_yaw_deg) ** 2,
                ],
                dtype=torch.float64,
            )
        )

    def predict(self, ego: Tensor) -> None:
        """@brief Move the estimate by measured odometry.

        @param ego ``(3,)`` relative pose, in the current body frame, as
            odometry reports it -- drift included, since that is the input a
            deployment has.
        """
        ego = ego.detach().double()
        ad = adjoint(G.inverse(ego))
        dist = float(ego[:2].norm())
        q = torch.diag(
            torch.tensor(
                [
                    (self.p.drift_frac * dist) ** 2,
                    (self.p.drift_frac * dist) ** 2,
                    (math.radians(self.p.yaw_drift_deg_per_m) * dist) ** 2,
                ],
                dtype=torch.float64,
            )
        )
        self.cov = ad @ self.cov @ ad.T + q
        self.pose = G.compose(self.pose, ego)

    def update(
        self, delta: Tensor, cov: Tensor, trust: float, mass: float
    ) -> Step:
        """@brief Fold in one model output, or refuse it and say why.

        @param delta ``(3,)`` the correction, in the frame of the prior this
            filter supplied -- so it is already the innovation.
        @param cov ``(3, 3)`` the model's covariance for it.
        @param trust ``sigmoid(trust_logit)``.
        @param mass Total assignment mass.

        @return The :class:`Step` describing what happened.
        """
        delta = delta.detach().double()
        r = cov.detach().double()
        s = self.cov + r
        # Solved rather than inverted: the innovation covariance is small and
        # symmetric positive definite, and a Cholesky solve of it is both
        # cheaper and better conditioned than forming an inverse to multiply.
        chol = torch.linalg.cholesky(s)
        whitened = torch.linalg.solve_triangular(
            chol, delta.unsqueeze(-1), upper=False
        ).squeeze(-1)
        nis = float(whitened.square().sum())

        for bad, why in (
            (trust < self.p.trust_threshold, "trust"),
            (mass < self.p.min_mass, "mass"),
        ):
            if bad:
                return Step(False, nis, why)

        gain = torch.cholesky_solve(self.cov.T, chol).T
        self.pose = G.compose(self.pose, gain @ delta)
        # Joseph form. The short form stays symmetric only while the gain is
        # exactly optimal, and a gated filter's is not: rejected frames leave
        # the covariance where prediction put it.
        eye = torch.eye(3, dtype=torch.float64)
        keep = eye - gain
        self.cov = keep @ self.cov @ keep.T + gain @ r @ gain.T
        self._floor()
        return Step(True, nis, "")

    def _floor(self) -> None:
        """Keep the diagonal above the floor, symmetrically."""
        f = torch.tensor(
            [self.p.floor_m**2, self.p.floor_m**2, self.p.floor_rad**2],
            dtype=torch.float64,
        )
        d = torch.clamp(torch.diagonal(self.cov), min=f)
        self.cov = (self.cov + self.cov.T) / 2
        self.cov[range(3), range(3)] = d
