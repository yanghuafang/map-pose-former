"""The filter's algebra, and the gates it refuses frames on."""

from __future__ import annotations

import math

import pytest
import torch

from mapposeformer import geometry as G
from mapposeformer.filter import FilterParams, LocalizationKF, adjoint


def _spd(m: torch.Tensor) -> bool:
    """Symmetric and positive definite, which every covariance must stay."""
    return bool(
        torch.allclose(m, m.T, atol=1e-9)
        and (torch.linalg.eigvalsh(m) > 0).all()
    )


def test_the_adjoint_transports_a_body_frame_error():
    """``eps`` in the old body frame is ``Ad(u^-1) eps`` in the new one.

    Checked against the composition itself rather than against an algebraic
    restatement of it, so an error in the matrix cannot be echoed by the test.
    """
    pose = torch.tensor([12.0, -3.0, 0.7], dtype=torch.float64)
    ego = torch.tensor([8.0, 1.5, 0.25], dtype=torch.float64)
    direction = torch.tensor([1.0, -2.0, 0.3], dtype=torch.float64)
    ad = adjoint(G.inverse(ego))

    residual = []
    for scale in (1e-3, 5e-4):
        eps = direction * scale
        truth = G.relative(
            G.compose(pose, ego), G.compose(G.compose(pose, eps), ego)
        )
        residual.append(float((truth - ad @ eps).norm()))

    # The adjoint is a first derivative, so it cannot match a finite
    # perturbation exactly -- what identifies it as the *right* derivative is
    # that halving the perturbation quarters what is left over. A wrong matrix
    # leaves a residual linear in the perturbation, which only halves.
    assert residual[0] < 1e-5
    assert residual[1] < residual[0] / 3.5


def test_prediction_grows_the_covariance_with_distance():
    kf = LocalizationKF(torch.zeros(3))
    before = torch.diagonal(kf.cov).clone()
    kf.predict(torch.tensor([8.0, 0.0, 0.0]))
    after = torch.diagonal(kf.cov)
    assert (after >= before).all()
    assert _spd(kf.cov)


def _information(cov: torch.Tensor) -> torch.Tensor:
    """A measurement noise covariance, as information.

    These tests state a noise level directly and run the ``information`` arm,
    which is the one whose semantics are unambiguous: what this frame alone
    measured. The default arm takes a prior-fused covariance instead, for
    reasons `FilterParams` sets out and does not defend as correct.
    """
    return torch.linalg.inv(cov.double())


def test_a_confident_correct_update_moves_the_estimate_and_tightens_it():
    kf = LocalizationKF(torch.zeros(3), FilterParams(measurement="information"))
    wide = torch.diagonal(kf.cov).clone()
    delta = torch.tensor([0.4, 0.1, 0.01])
    cov = torch.diag(torch.tensor([0.01, 0.01, 1e-4]))

    step = kf.update(delta, _information(cov), trust=0.9, mass=40.0)
    assert step.accepted and step.reason == ""
    # The model is far more confident than the prior, so the estimate should
    # land near the correction rather than between the two.
    assert torch.allclose(kf.pose, delta.double(), atol=0.05)
    assert (torch.diagonal(kf.cov) < wide).all()
    assert _spd(kf.cov)


def test_a_frame_the_model_distrusts_is_refused():
    for trust, mass, reason in ((0.1, 60.0, "trust"), (0.9, 1.0, "mass")):
        kf = LocalizationKF(
            torch.zeros(3), FilterParams(measurement="information")
        )
        step = kf.update(
            torch.tensor([0.1, 0.0, 0.0]),
            _information(torch.diag(torch.tensor([0.01, 0.01, 1e-4]))),
            trust=trust,
            mass=mass,
        )
        assert not step.accepted and step.reason == reason


def test_the_filter_tracks_a_driven_sequence():
    """Twenty frames of odometry with drift, corrected by a noisy but honest
    model. The estimate should stay near truth rather than walk away."""
    torch.manual_seed(0)
    p = FilterParams(measurement="information")
    truth = torch.zeros(3, dtype=torch.float64)
    kf = LocalizationKF(truth.clone().float(), p)
    cov = torch.diag(torch.tensor([0.04, 0.01, 4e-4], dtype=torch.float64))

    for _ in range(20):
        step = torch.tensor([8.0, 0.0, 0.02], dtype=torch.float64)
        truth = G.compose(truth, step)
        # Odometry the filter is given: the true step, drifted.
        drift = torch.randn(3, dtype=torch.float64) * torch.tensor(
            [0.08, 0.08, math.radians(0.16)], dtype=torch.float64
        )
        kf.predict(G.compose(step, drift))
        # The model measures the remaining error, with noise matching cov.
        err = G.relative(kf.pose, truth)
        noise = torch.randn(3, dtype=torch.float64) * torch.sqrt(
            torch.diagonal(cov)
        )
        kf.update(err + noise, _information(cov), trust=0.9, mass=40.0)
        assert _spd(kf.cov)

    remaining = G.relative(kf.pose, truth)
    assert float(remaining[:2].norm()) < 1.0
    assert abs(float(remaining[2])) < math.radians(2.0)


def test_odometry_alone_tracks_the_truth():
    """The direction the filter is driven in, which is easy to get backwards.

    ``measured_egomotion`` answers two different questions depending on
    argument order: the motion just made, and where the past sits as seen from
    now. The filter wants the first; a sample's ``hist_rel`` wants the second.
    Taking the wrong one walks the estimate backwards a frame at a time and
    the only symptom is a closed loop that diverges -- which reads as a bad
    model.

    So: drive the filter with odometry and no corrections at all, and it must
    stay near the truth, drifting slowly rather than running away.
    """
    from mapposeformer.data import build_world
    from mapposeformer.data.sample import EgoParams, measured_egomotion

    world = build_world(0)
    frames = list(range(0, 60, 4))
    gen = torch.Generator().manual_seed(0)
    kf = LocalizationKF(world.trajectory[frames[0]].clone())

    for i in range(1, len(frames)):
        prev = world.trajectory[frames[i - 1]]
        truth = world.trajectory[frames[i]]
        kf.predict(measured_egomotion(prev, truth, EgoParams(), gen))
        drift = float((kf.pose.float()[:2] - truth[:2]).norm())
        travelled = float((truth[:2] - world.trajectory[frames[0]][:2]).norm())
        assert drift < 0.05 * travelled + 0.5, (
            f"frame {i}: drifted {drift:.2f} m over {travelled:.1f} m travelled"
        )


def test_the_posterior_arm_counts_the_prior_twice():
    """Why the filter takes information and not the model's covariance.

    The model fuses the prior into `cov`, because a NEES against the truth
    should be measured on the estimate the model actually produced -- and that
    estimate used the prior. The filter's own state *is* that prior. Handing it
    the fused matrix therefore adds a copy of the filter's belief to something
    that already contains one, and the innovation covariance comes out too
    tight: the filter trusts each frame more than the evidence in it supports.

    Here the measurement on its own is worth `sigma = 1 m` and the prior a
    further 1 m, so a correctly-counted innovation covariance is
    `P + R = 1 + 1 = 2`, while the posterior arm sees `P + (1/1 + 1/1)^-1`
    = `1 + 0.5` = 1.5. Under-stating `S` is what inflates NIS and makes a gate
    built on it fire on frames that were never surprising.
    """

    y = torch.eye(3, dtype=torch.float64)  # measurement worth sigma = 1
    fused = torch.linalg.inv(y + y)  # what the model reports as `cov`
    delta = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

    nis = {}
    for mode, m in (("information", y), ("posterior", fused)):
        kf = LocalizationKF(torch.zeros(3), FilterParams(measurement=mode))
        kf.cov = torch.eye(3, dtype=torch.float64)  # prior worth sigma = 1
        nis[mode] = kf.update(delta, m, trust=0.9, mass=40.0).nis

    assert nis["information"] == pytest.approx(1 / 2, rel=1e-6), nis
    assert nis["posterior"] == pytest.approx(1 / 1.5, rel=1e-6), nis
    assert nis["posterior"] > nis["information"]


def test_a_measurement_with_no_along_track_information_is_survivable():
    """Three lane dividers and no pole: the road direction is unobservable.

    The measurement information is then singular, and there is no covariance
    to invert at all. In information form that needs no special case -- the
    null direction simply contributes nothing -- and the filter must come out
    with its along-track uncertainty *unchanged* rather than improved.
    """

    y = torch.diag(torch.tensor([0.0, 4.0, 100.0], dtype=torch.float64))
    kf = LocalizationKF(torch.zeros(3), FilterParams(measurement="information"))
    kf.cov = torch.diag(torch.tensor([2.0, 2.0, 0.01], dtype=torch.float64))
    before = float(kf.cov[0, 0])

    step = kf.update(
        torch.tensor([0.5, 0.3, 0.01], dtype=torch.float64),
        y,
        trust=0.9,
        mass=40.0,
    )
    assert step.accepted
    assert _spd(kf.cov)
    # Nothing was learned along track, and nothing was claimed.
    assert float(kf.cov[0, 0]) == pytest.approx(before, rel=1e-9)
    # Across track, where the measurement did speak, it tightened.
    assert float(kf.cov[1, 1]) < 2.0
