"""What the matcher is trained on, and why the pose loss is not enough.

The pose comes out of a closed-form solve, so a loss on the pose does reach
the assignment -- gradient flows back through the Procrustes formula. It is
just a very thin signal for a 96 x 72 decision: one three-vector of error to
explain seven thousand weights, and any number of wrong assignments produce a
pose that is right on average. Supervising the correspondences directly is
what makes the matcher learn association rather than a way to cancel its own
mistakes.

The labels are not in the data, because a real map has no record of which
detection came from which element. They are *derived* from the one label there
is: apply the true correction and a detection lands on the element it came
from. That makes the supervision exactly as good as the correction is, which
is the right dependence -- if the labels were wrong the invariant tests in
``tests/test_data.py`` would already be failing.

Three terms, and each answers a different failure:

  match      the assignment puts its mass on the right element
  matchable  a detection with no counterpart is allowed to say so, instead of
             being pushed onto whichever map element is least implausible
  pose       what is actually being asked for, and the only term that knows
             the difference between a near miss and a far one
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from mapposeformer import geometry as G
from mapposeformer.metrics import pose_error

#: Anything below this is a log of zero.
EPS = 1e-9


@dataclass
class LossParams:
    """Term weights, and what counts as a correspondence."""

    match_w: float = 1.0
    matchable_w: float = 0.2
    pose_w: float = 1.0
    #: A detected point is on a map element if it lands this close to one of
    #: its points once the true correction is applied. The generator spaces
    #: points 1.5 to 1.7 m apart, so this is under half a spacing.
    match_radius_m: float = 0.75
    #: Fraction of a detection's points that must land on the map *somewhere*
    #: before it counts as having a counterpart. Deliberately not per element:
    #: a long detection legitimately spans several map chunks.
    min_overlap: float = 0.5
    #: Radius for *point* labels, which is a different question from element
    #: overlap and wants a different number. Map points sit 1.68 m apart, so
    #: the nearest one can be 0.84 m away with no noise at all. Measured on
    #: the test split, the true-corrected distance is 0.52 m at the median and
    #: 1.23 m at p90, and labelling captures 74.5% of real points at 0.75 m,
    #: 88.4% at 1.0 m and only 90.1% at 1.25 m. 1.0 is the knee.
    point_radius_m: float = 1.0
    #: ``covering`` labels a detection with every map element it lands on, in
    #: proportion; ``best`` labels it with the single element it covers most,
    #: which is what a one-hot cross-entropy wants and what called 95.6% of
    #: road boundaries unmatched. A flag, so the two differ by one line.
    match_target: str = "covering"
    #: Radians are small numbers next to metres: 1 degree of heading is worth
    #: about 0.9 m at the far edge of a 50 m crop, and this is that ratio.
    yaw_w: float = 50.0
    #: Huber knee for the pose term, in metres. Beyond this the gradient is
    #: constant instead of growing, which is what stops an untrained model's
    #: metres-wide pose from drowning the match term that would fix it.
    #: L1 has a constant gradient too, but its *magnitude* is the full
    #: weighted error -- at a 22 m error and a 50x yaw weight that is a
    #: gradient two orders above the match term's, and it diverged: the match
    #: loss went 6.1 -> 1.8e7 -> 4.8e12 while the pose error rose from 20 m to
    #: 95 m.
    huber_delta_m: float = 1.0


def element_truth(
    det_pts: Tensor,
    det_pmask: Tensor,
    map_pts: Tensor,
    map_pmask: Tensor,
    delta: Tensor,
    p: LossParams,
) -> tuple[Tensor, Tensor]:
    """Which map element each detection came from, where there is one.

    Labels, not predictions: nothing here needs a gradient, and the distance
    tensor is (B, D, P, M, P), which is over a hundred megabytes at a full
    batch. Building it inside the graph would be paying to differentiate a
    constant.

    **A detection usually covers several map elements.** The map is chunked at
    a fixed 12 m; a camera frustum sees whatever range it has. Measured on the
    test split, a detected road boundary lands on **four** map elements and a
    lane divider on two, with 88% of their points accounted for. Asking which
    single element a detection belongs to is the wrong question, and answering
    it with a threshold labelled 95.6% of road boundaries as matching nothing
    -- the class that constrains lateral position best.

    So the label is a *distribution* over map elements rather than an index.
    The assignment is already dense and ``Matcher.points`` already spreads a
    detection over every element it has mass on, so this only asks the
    supervision for what the architecture could always represent.

    @return ``(target, has_match)``. ``target`` is ``(B, D, M)``, summing to
        one over the map wherever ``has_match`` is true and meaningless
        elsewhere -- clutter, a false positive, or a landmark whose element
        fell outside the crop.
    """
    B, D, P, _ = det_pts.shape
    with torch.no_grad():
        moved = G.transform_points(delta, det_pts.reshape(B, -1, 2)).reshape(
            B, D, P, 2
        )
        # (B, D, P, M, P): nearest point of each map element, per detection
        # point.
        gap = (
            (moved[:, :, :, None, None, :] - map_pts[:, None, None, :, :, :])
            .square()
            .sum(-1)
        )
        gap = gap.masked_fill(~map_pmask[:, None, None, :, :], float("inf"))
        near = gap.min(-1).values < p.match_radius_m**2

        # Fraction of the detection's own points that landed on that element.
        counted = (near & det_pmask[..., None]).sum(2)
        overlap = counted / det_pmask.sum(-1, keepdim=True).clamp_min(1)

        # Explained by the map as a whole, rather than by any one element
        # of it.
        total = overlap.sum(-1)
        has_match = (total >= p.min_overlap) & det_pmask.any(-1)
        if p.match_target == "best":
            best, index = overlap.max(-1)
            one_hot = torch.zeros_like(overlap)
            one_hot.scatter_(-1, index.unsqueeze(-1), 1.0)
            return one_hot, has_match & (best >= p.min_overlap)
        if p.match_target != "covering":
            raise ValueError(f"unknown match_target {p.match_target!r}")
        return overlap / total.clamp_min(EPS).unsqueeze(-1), has_match


def point_truth(
    det_pts: Tensor,
    det_pmask: Tensor,
    map_pts: Tensor,
    map_pmask: Tensor,
    delta: Tensor,
    p: LossParams,
) -> tuple[Tensor, Tensor]:
    """The same labels at point resolution, for point tokens.

    Matching points directly needs no notion of an element at all, and with it
    goes the question that had no good answer -- *which* map element does a
    detection spanning four of them belong to. A detected point either lands
    on a map point when the true correction is applied, or it does not.

    @return ``(target, has_match)``. ``target`` is ``(B, D*P, M*P)``, one-hot
        on the nearest map point; ``has_match`` is ``(B, D*P)``.
    """
    B, D, P, _ = det_pts.shape
    M = map_pts.shape[1]
    with torch.no_grad():
        moved = G.transform_points(delta, det_pts.reshape(B, -1, 2))
        flat_map = map_pts.reshape(B, -1, 2)
        valid = map_pmask.reshape(B, -1)

        gap = torch.cdist(moved, flat_map)
        gap = gap.masked_fill(~valid[:, None, :], float("inf"))
        near, index = gap.min(-1)

        has_match = (near < p.point_radius_m) & det_pmask.reshape(B, -1)
        target = torch.zeros(B, D * P, M * map_pts.shape[2], device=gap.device)
        target.scatter_(2, index.unsqueeze(-1), 1.0)
        return target * has_match.unsqueeze(-1), has_match


def compute_losses(
    out: dict[str, Tensor], batch: dict[str, Tensor], p: LossParams
) -> tuple[Tensor, dict[str, float]]:
    """@return ``(total, parts)``; ``parts`` is for logging, detached."""
    # Point tokens are matched point to point, so the labels are too. The
    # element scheme has to answer "which element is this detection", which a
    # detection spanning four of them cannot; the point scheme never asks.
    truth = point_truth if out["granularity"] == "point" else element_truth
    target, has_match = truth(
        out["det_pts"],
        out["det_pmask"],
        batch["map_pts"],
        batch["map_pmask"],
        batch["delta"],
        p,
    )

    # Losses are computed in fp32 whatever the forward pass ran in. Two
    # reasons, and the second is the one that bites: a log and a
    # cross-entropy over probabilities near zero lose most of bf16's eight
    # mantissa bits, and `binary_cross_entropy` is on autocast's promote-to-
    # fp32 list, so leaving the inputs in bf16 fails outright on the dtype.
    # The cast is a few thousand elements against the model's millions.
    #
    # The match term is cross-entropy on the raw scores, not on the assignment
    # the pose is solved from. The assignment carries matchability as a
    # factor, so training on it would let the model lower this term by
    # declaring everything matchable -- an answer to a question nobody asked.
    scores = out["scores"].float()
    if has_match.any():
        # Cross-entropy against a distribution, not an index: the target holds
        # mass on every map element the detection covers, in proportion.
        logp = F.log_softmax(scores[has_match], dim=-1)
        match = -(target[has_match] * logp).sum(-1).mean()
    else:
        match = scores.sum() * 0.0

    matchable = F.binary_cross_entropy(
        out["det_matchable"].float().clamp(EPS, 1 - EPS),
        has_match.float(),
        reduction="none",
    )
    valid = (
        out["det_pmask"].reshape(has_match.shape)
        if out["granularity"] == "point"
        else out["det_pmask"].any(-1)
    )
    matchable = (matchable * valid).sum() / valid.sum().clamp_min(1)

    # Huber, not L1. A model that has not learned to match yet produces a
    # pose metres wide, and an unbounded pose gradient at that scale wrecks
    # the matcher that would have fixed it -- measured, the match loss
    # diverged to 4.8e12 while the pose error grew. Beyond the knee the
    # gradient is bounded and the match term is left to do the early work.
    err = pose_error(out["delta"].float(), batch["delta"].float())
    scaled = torch.cat([err[:, :2], p.yaw_w * err[:, 2:]], dim=-1)
    pose = (
        F.huber_loss(
            scaled,
            torch.zeros_like(scaled),
            delta=p.huber_delta_m,
            reduction="none",
        )
        .sum(-1)
        .mean()
    )

    total = p.match_w * match + p.matchable_w * matchable + p.pose_w * pose
    return total, {
        "match": match.item(),
        "matchable": matchable.item(),
        "pose": pose.item(),
        "matched_frac": has_match.float().mean().item(),
    }
