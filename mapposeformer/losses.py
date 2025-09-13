"""Training objectives.

Four terms, and the interesting one is not the pose term.

``pose``      Huber on the predicted correction. The objective everyone expects.
``volume``    Cross-entropy over the hypothesis grid, against a soft target.
``match``     Direct supervision of the assignment matrix, from geometry.
``cov``       Gaussian NLL, so the reported uncertainty means something.
``trust``     Can the model tell when it has failed?

The ``match`` term is what makes the rest work. The pose term alone gives a
single 3-vector of gradient to share among a hundred thousand assignment
entries, and a model trained that way discovers that predicting the mean of the
prior is a decent local minimum long before it discovers correspondence. The
match term gives every detected point its own target, from the one source that
cannot be wrong: apply the true correction and see which map point it lands on.

That target needs no correspondence bookkeeping in the dataset -- only the
ground-truth pose, which is the one thing every localization dataset has. The
same code will work unchanged on nuScenes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from mapposeformer import geometry as G

_EPS = 1e-8


@dataclass(frozen=True)
class LossParams:
    """Term weights and the two thresholds that define the auxiliary targets."""

    w_pose: float = 1.0
    w_volume: float = 1.0
    w_match: float = 1.0
    w_cov: float = 0.1
    w_trust: float = 0.2
    huber_delta_m: float = 1.0
    yaw_lever_m: float = 20.0
    """Converts a heading error into a comparable distance, so one Huber can
    cover all three axes. At 20 m, one degree of yaw costs the same as 35 cm of
    translation -- roughly the ratio at which the two matter to a vehicle."""
    match_radius_m: float = 1.0
    """A detected point is deemed to correspond to the nearest map point within
    this radius of where the true correction puts it. Comfortably above the
    detection noise (about 0.35 m including correlated bias) and comfortably
    below the 3.5 m lane spacing, so the target is neither missing nor
    ambiguous."""
    volume_sigma_cells: float = 1.0
    """Width of the soft target, in grid cells. A one-hot target would tell the
    surface nothing about how wrong a neighbouring cell is, and the covariance
    read off it would be a delta function."""
    trust_tol_m: float = 0.5
    trust_tol_deg: float = 0.5


def _pose_residual(pred: Tensor, gt: Tensor, lever_m: float) -> Tensor:
    """``(B, 3)`` error in the ground-truth frame, yaw scaled to metres.

    Resolved in the *ground-truth* frame rather than the prediction's, so a
    heading mistake does not rotate the axes its own translation error is
    reported on.
    """
    e = G.relative(gt, pred)
    return torch.stack([e[:, 0], e[:, 1], e[:, 2] * lever_m], dim=-1)


def pose_loss(pred: Tensor, gt: Tensor, p: LossParams) -> Tensor:
    r = _pose_residual(pred, gt, p.yaw_lever_m)
    return F.huber_loss(r, torch.zeros_like(r), delta=p.huber_delta_m)


def volume_loss(
    logits: Tensor, cells: Tensor, pitch: Tensor, gt: Tensor, p: LossParams
) -> Tensor:
    """Cross-entropy against a Gaussian centred on the true correction.

    Args:
        logits: ``(B, G)``.
        cells: ``(G, 3)`` the hypothesis coordinates, from the volume head.
        pitch: ``(3,)`` cell pitch per axis, in that axis's own units. One sigma
            per axis and not a scalar: metres and radians are not comparable,
            and a single sigma makes the yaw axis either one-hot or uniform
            depending on which unit was chosen.
        gt: ``(B, 3)`` the true correction.
    """
    sigma = p.volume_sigma_cells * pitch.clamp_min(_EPS)
    d = (cells.unsqueeze(0) - gt.unsqueeze(1)) / sigma
    target = F.softmax(-0.5 * d.square().sum(-1), dim=-1)
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def match_loss(
    assign: Tensor, batch: dict[str, Tensor], p: LossParams
) -> tuple[Tensor, Tensor]:
    """Supervise the assignment from geometry alone.

    Returns ``(loss, matched_fraction)``. The fraction is not used by the
    optimizer; it is reported because it says how much supervision the term
    actually carried on a given batch, and a silent collapse to zero positives
    is otherwise invisible.
    """
    det = batch["det_pts"].flatten(1, 2)
    mp = batch["map_pts"].flatten(1, 2)
    dvalid = batch["det_pmask"].flatten(1)
    mvalid = batch["map_pmask"].flatten(1)

    aligned = G.transform_points(batch["delta"], det)
    dist = torch.cdist(aligned, mp)
    dist = dist.masked_fill(~mvalid.unsqueeze(1), float("inf"))
    best, idx = dist.min(dim=2)

    positive = dvalid & (best <= p.match_radius_m)
    prob_pos = assign.gather(2, idx.unsqueeze(-1)).squeeze(-1)
    # Positives: push mass onto the right map point. Negatives: push total mass
    # off every map point, which is how a false-positive detection is taught to
    # abstain rather than to drag the pose towards whatever it happens to be
    # nearest.
    row_mass = assign.sum(-1).clamp(0.0, 1.0 - 1e-6)
    loss_pos = -torch.log(prob_pos.clamp_min(_EPS))
    loss_neg = -torch.log1p(-row_mass)

    negative = dvalid & ~positive
    loss = (positive.float() * loss_pos + negative.float() * loss_neg).sum()
    loss = loss / dvalid.sum().clamp_min(1)
    return loss, positive.sum() / dvalid.sum().clamp_min(1)


def covariance_loss(cov: Tensor, pred: Tensor, gt: Tensor) -> Tensor:
    """Gaussian NLL of the pose residual under the predicted covariance.

    Without this term nothing in the objective mentions ``cov`` at all, and the
    model happily emits a number the downstream filter would weight by. An
    uncalibrated covariance is worse than none: a filter told a bad frame is
    certain will follow it.

    The residual is **detached**. Attached, the term becomes a learned
    per-sample weight on the pose loss, and the cheapest way to reduce it is to
    declare hard frames uncertain rather than to localize them. Detached, it
    can only calibrate.
    """
    r = G.relative(gt, pred.detach()).unsqueeze(-1)
    chol = torch.linalg.cholesky(cov)
    whitened = torch.linalg.solve_triangular(chol, r, upper=False)
    logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (whitened.squeeze(-1).square().sum(-1) + logdet).mean()


def trust_loss(trust_logit: Tensor, pred: Tensor, gt: Tensor, p: LossParams) -> Tensor:
    """Teach the model to predict its own success.

    The target is derived from the model's own detached error, which makes this
    the learned replacement for the classical pipeline's two hand-tuned gates
    (surface too flat, best cost too high). Detached deliberately: the gradient
    must improve the *prediction of* failure, never make failure look smaller.
    """
    with torch.no_grad():
        e = G.relative(gt, pred.detach())
        ok = (e[:, :2].norm(dim=-1) <= p.trust_tol_m) & (
            e[:, 2].abs() <= torch.deg2rad(torch.tensor(p.trust_tol_deg, device=e.device))
        )
    return F.binary_cross_entropy_with_logits(trust_logit, ok.float())


def compute_losses(
    out: dict[str, Tensor],
    batch: dict[str, Tensor],
    cells: Tensor,
    pitch: Tensor,
    p: LossParams,
) -> tuple[Tensor, dict[str, float]]:
    """Total loss and a dict of scalars for logging."""
    gt = batch["delta"]
    lp = pose_loss(out["delta"], gt, p)
    lv = volume_loss(out["logits"], cells, pitch, gt, p)
    lm, matched = match_loss(out["assign"], batch, p)
    lc = covariance_loss(out["cov"], out["delta"], gt)
    lt = trust_loss(out["trust_logit"], out["delta"], gt, p)
    total = p.w_pose * lp + p.w_volume * lv + p.w_match * lm + p.w_cov * lc + p.w_trust * lt
    return total, {
        "loss": float(total.detach()),
        "pose": float(lp.detach()),
        "volume": float(lv.detach()),
        "match": float(lm.detach()),
        "cov": float(lc.detach()),
        "trust": float(lt.detach()),
        "matched_frac": float(matched),
    }
