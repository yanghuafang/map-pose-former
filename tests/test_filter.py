"""The filter's algebra, and the gates it refuses frames on."""

from __future__ import annotations

import math

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


def test_a_confident_correct_update_moves_the_estimate_and_tightens_it():
    kf = LocalizationKF(torch.zeros(3))
    wide = torch.diagonal(kf.cov).clone()
    delta = torch.tensor([0.4, 0.1, 0.01])
    cov = torch.diag(torch.tensor([0.01, 0.01, 1e-4]))

    step = kf.update(delta, cov, trust=0.9, mass=40.0)
    assert step.accepted and step.reason == ""
    # The model is far more confident than the prior, so the estimate should
    # land near the correction rather than between the two.
    assert torch.allclose(kf.pose, delta.double(), atol=0.05)
    assert (torch.diagonal(kf.cov) < wide).all()
    assert _spd(kf.cov)


def test_a_frame_the_model_distrusts_is_refused():
    for trust, mass, reason in ((0.1, 60.0, "trust"), (0.9, 1.0, "mass")):
        kf = LocalizationKF(torch.zeros(3))
        step = kf.update(
            torch.tensor([0.1, 0.0, 0.0]),
            torch.diag(torch.tensor([0.01, 0.01, 1e-4])),
            trust=trust,
            mass=mass,
        )
        assert not step.accepted and step.reason == reason


def test_the_filter_tracks_a_driven_sequence():
    """Twenty frames of odometry with drift, corrected by a noisy but honest
    model. The estimate should stay near truth rather than walk away."""
    torch.manual_seed(0)
    p = FilterParams()
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
        kf.update(err + noise, cov, trust=0.9, mass=40.0)
        assert _spd(kf.cov)

    remaining = G.relative(kf.pose, truth)
    assert float(remaining[:2].norm()) < 1.0
    assert abs(float(remaining[2])) < math.radians(2.0)
