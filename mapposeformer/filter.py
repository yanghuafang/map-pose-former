"""The filter the corrections are fed back through.

An SE(2) Kalman filter over the pose: odometry moves the estimate, one model
output corrects it, and the caller hands the result back as the next frame's
prior. Feeding the estimate back is what makes the loop an experiment rather
than a replay: a filter averages independent error away over a run of frames
but cannot touch a bias, so closing the loop is the only test that tells the
two apart.
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
averaging itself.

``mass`` any assignment-based model reports. ``trust`` presumes a head that
scores its own frames, and the design in ``docs/ROADMAP.md`` does not have
one -- it widens the covariance on an ambiguous frame instead of refusing it.
Set ``trust_threshold`` to zero and pass 1.0 to run without that gate, which
is what a model with no such head has to do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from mapposeformer import geometry as G


@dataclass(frozen=True)
class FilterParams:
    """Gates and noise.

    The thresholds below match ``engine/evaluator.py``, so that open loop and
    closed loop refuse the same frames for the same reasons. That evaluator
    currently refuses nothing -- a refusal threshold needs a covariance to
    stand on, which arrives with M3 -- so the correspondence is *not* yet true
    and has to be established when M4 closes the loop.
    """

    measurement: str = "posterior"
    """What the model hands over. **Neither option here is correct, and the
    measurement says so.**

    ``posterior`` takes the model's ``cov``, which has the prior fused in. So
    ``S = P + R`` adds this filter's own belief to a matrix that already
    contains a copy of it: the prior is counted twice.

    ``information`` takes ``H / 2 s^2``, what this frame's landmarks alone
    say, and counts the prior once.

    The second is the principled one and it measures **worse** -- 60 test
    scenes, trans 0.171 against 0.165, yaw 0.210 deg against 0.189.
    That result is not noise and it is not an argument for double counting.
    It is a sign the diagnosis was incomplete.

    It is tempting to read ``delta`` as a MAP estimate, because
    ``prior_information`` is passed into ``solve_pose_directional``. That
    reading is wrong, and only measuring it settles the question. The prior
    enters only as ``(H + P) d = -g`` with ``g`` the *measurement* gradient,
    so it damps the step and does not appear in the fixed point: at
    convergence ``g = 0``, which is the maximum-likelihood condition. Measured
    on a well-posed road, tightening the prior from 5 m to 0.05 m -- a
    hundredfold -- moves a recovered 0.80 m correction by 4.8e-7 m.

    That is the *converged* statement, and ``refine_iters`` is 3. Where three
    iterations have not reached the fixed point the damping does move the
    answer: median 0.013 m over 256 val frames, past 0.05 m on 33% of them,
    with a distance-kernel stand-in for the matcher. It moves it back towards
    the point-to-point initialisation rather than towards the prior mean --
    multiply the prior by 1e4 and the pose stops at the initialisation (norm
    1.5705 against the initialisation's 1.5713) instead of shrinking towards
    zero. A shortened step, not a prior term: there is no reading under which
    ``delta`` is a MAP estimate.

    So ``delta`` has no prior term, and ``information`` is the matching choice.
    It still measures worse, and the honest position is that the reason is
    **not known**. The most likely candidate is that the two errors partially
    cancel: the covariance is independently measured as pessimistic by roughly
    2 to 4x, so fusing the prior into it tightens an over-wide matrix and the
    filter gains from trusting the frame more than the stated covariance
    would allow. That is a coincidence, not a design, and it should not
    survive the covariance being calibrated.

    ``posterior`` therefore stays the default on the measurement alone, with
    no theory behind it, until the calibration is fixed and the comparison is
    re-run."""

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
    init_sigma_lat_m: float = 0.6
    init_sigma_yaw_deg: float = 1.0
    """The first frame's prior, which has no previous frame to inherit from.
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
    """What one frame did. ``accepted`` and ``reason`` are what the sequence
    metrics read afterwards; ``nis`` is reported and read by nobody."""

    accepted: bool
    nis: float
    """Normalised innovation squared, ``delta^T S^-1 delta``. Formed before
    the gates, so it is computed on every frame including refused ones --
    and **nothing in the pipeline reads it**. ``engine/sequence.py`` takes
    ``accepted`` and ``reason`` only, ``SequenceResult`` has no field for it
    and no metrics key carries it; the sole reader in the tree is
    ``tests/test_filter.py``, where it is what separates the two
    ``measurement`` settings. It is kept for offline analysis. There is no
    innovation gate here: the two gates are the model's own, as the module
    docstring says."""
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
        self, delta: Tensor, measurement: Tensor, trust: float, mass: float
    ) -> Step:
        """@brief Fold in one model output, or refuse it and say why.

        @param delta ``(3,)`` the correction, in the frame of the prior this
            filter supplied -- so it is already the innovation.
        @param measurement ``(3, 3)``. Under ``measurement="information"``
            the measurement information ``H / 2 s^2``; under ``"posterior"``
            the model's fused covariance. See :class:`FilterParams`.
        @param trust ``sigmoid(trust_logit)``.
        @param mass Total assignment mass.

        @return The :class:`Step` describing what happened.
        """
        delta = delta.detach().double()
        eye = torch.eye(3, dtype=torch.float64)

        if self.p.measurement == "posterior":
            # The double-counting option, kept to be measured against.
            r = measurement.detach().double()
            s = self.cov + r
            # Solved rather than inverted: the innovation covariance is small
            # and symmetric positive definite, and a Cholesky solve of it is
            # both cheaper and better conditioned than forming an inverse.
            chol = torch.linalg.cholesky(s)
            whitened = torch.linalg.solve_triangular(
                chol, delta.unsqueeze(-1), upper=False
            ).squeeze(-1)
            nis = float(whitened.square().sum())
            gain = torch.cholesky_solve(self.cov.T, chol).T
            posterior = None
        elif self.p.measurement == "information":
            y = measurement.detach().double()
            y = 0.5 * (y + y.T)
            prior_information = torch.cholesky_solve(
                eye, torch.linalg.cholesky(self.cov)
            )
            total = prior_information + y
            total = 0.5 * (total + total.T)
            chol_total = torch.linalg.cholesky(total)
            posterior = torch.cholesky_solve(eye, chol_total)
            posterior = 0.5 * (posterior + posterior.T)

            # The gain that takes the innovation to the correction. In
            # information form the posterior estimate is `P_post Y z`, so this
            # is the gain without ever inverting `Y`.
            gain = posterior @ y

            # NIS needs `S^-1 = (P + Y^-1)^-1`, and `Y^-1` may not exist. By
            # Woodbury, `S^-1 = Y - Y P_post Y`, which is defined for a
            # singular `Y` and correctly contributes nothing along its null
            # direction -- an unconstrained innovation is not a surprising one.
            s_inv = y - y @ posterior @ y
            s_inv = 0.5 * (s_inv + s_inv.T)
            nis = float(delta @ s_inv @ delta)
        else:
            raise ValueError(f"unknown measurement {self.p.measurement!r}")

        for bad, why in (
            (trust < self.p.trust_threshold, "trust"),
            (mass < self.p.min_mass, "mass"),
        ):
            if bad:
                return Step(False, nis, why)

        self.pose = G.compose(self.pose, gain @ delta)
        if posterior is not None:
            self.cov = posterior
        else:
            # Joseph form. The short form stays symmetric only while the gain
            # is exactly optimal, and a gated filter's is not: rejected frames
            # leave the covariance where prediction put it.
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
