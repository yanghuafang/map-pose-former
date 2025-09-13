"""From soft correspondences to a pose, in closed form.

Weighted Procrustes in SE(2). Given detection points, map points and the soft
assignment between them, this solves for the rigid transform that best aligns
one to the other -- exactly, in one expression, differentiably.

Why closed form rather than an MLP. The transform is *determined* by the
correspondences; there is nothing left to learn once they are known. Solving it
analytically means the model's entire capacity goes into the question that is
genuinely hard -- which point is which -- and the answer inherits the
equivariance of the solution rather than having to approximate it from data. It
also means the head has no parameters, so it cannot be the thing that overfits,
cannot be pruned away, and quantizes to whatever precision the arithmetic is
done in.

The regression head below was kept for contrast, not for use. That was the
wrong way round, and the experiment says so: in distribution the two are within
2% on translation, and *off* distribution the regression head wins by a factor
of seven, because ``tanh`` bounds it and a rigid fit over bad correspondences is
bounded by nothing. See ``docs/RESULTS.md``.

What survives here is the parameter count, the inability to overfit, and the
assignment matrix -- which is computed either way, so it is not what is being
traded. What does not survive is accuracy. The repair is robustness, not
retreat: weighted Procrustes has standard answers to outliers and this
implementation uses none of them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mapposeformer import geometry as G

_EPS = 1e-6


def weighted_procrustes_se2(
    src: Tensor, dst: Tensor, weight: Tensor
) -> tuple[Tensor, Tensor]:
    """Rigid transform taking ``src`` onto ``dst``, weighted per point.

    Args:
        src: ``(B, K, 2)`` source points -- detections, in the ego frame.
        dst: ``(B, K, 2)`` their targets -- map points, in the anchor frame.
        weight: ``(B, K)`` non-negative confidence per correspondence.

    Returns:
        ``(pose (B, 3), mass (B,))``. ``pose`` is ``(x, y, yaw)`` such that
        ``transform_points(pose, src) ≈ dst``. ``mass`` is the total weight,
        returned because a caller must know when the answer rests on nothing:
        with no correspondences the pose is identity, which is a *default*, not
        an estimate.
    """
    w = weight.clamp_min(0.0).unsqueeze(-1)
    mass = w.sum(dim=1).clamp_min(_EPS)

    src_bar = (w * src).sum(dim=1) / mass
    dst_bar = (w * dst).sum(dim=1) / mass
    a = src - src_bar.unsqueeze(1)
    b = dst - dst_bar.unsqueeze(1)

    # The 2-D case of the SVD in general Kabsch collapses to one atan2: the
    # optimal rotation angle is the argument of the weighted cross/dot pair.
    cross = (w.squeeze(-1) * (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])).sum(1)
    dot = (w.squeeze(-1) * (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1])).sum(1)
    # atan2 is undefined at the origin and its gradient is unbounded near it,
    # which is exactly the degenerate case -- all weight on one point, so no
    # rotation is observable. Fall back to zero rotation there rather than
    # propagating an arbitrary angle with an enormous gradient.
    degenerate = (cross.square() + dot.square()) < _EPS
    yaw = torch.atan2(cross, torch.where(degenerate, torch.ones_like(dot), dot))
    yaw = torch.where(degenerate, torch.zeros_like(yaw), yaw)

    c, s = torch.cos(yaw), torch.sin(yaw)
    rot_src = torch.stack(
        [c * src_bar[:, 0] - s * src_bar[:, 1], s * src_bar[:, 0] + c * src_bar[:, 1]],
        dim=-1,
    )
    t = dst_bar - rot_src
    return torch.cat([t, yaw.unsqueeze(-1)], dim=-1), mass.squeeze(-1)


class ProcrustesPoseHead(nn.Module):
    """Turn an assignment matrix into a pose. Parameter-free."""

    def forward(self, assign: Tensor, det_pts: Tensor, map_pts: Tensor):
        """Args: ``(B, K, L)``, ``(B, K, 2)``, ``(B, L, 2)``.

        Each detected point is matched not to one map point but to the
        assignment-weighted average of all of them. That average is the
        minimum-variance target under the model's own uncertainty, and it keeps
        the head differentiable in the assignment rather than only in the
        argmax -- a hard nearest-neighbour would give zero gradient to every
        map point that did not win.
        """
        # fp32 throughout, whatever autocast is doing outside: ``assign @
        # map_pts`` multiplies a weight by a coordinate, and bf16 would put a
        # lattice under the answer. See ``geometry.exact_arithmetic``.
        with G.exact_arithmetic(assign.device.type):
            assign, det_pts = assign.float(), det_pts.float()
            w = assign.sum(dim=2)
            target = assign @ map_pts.float() / w.clamp_min(_EPS).unsqueeze(-1)
            return weighted_procrustes_se2(det_pts, target, w)


class RegressionPoseHead(nn.Module):
    """Baseline: predict the pose straight from a pooled feature.

    Present so the closed-form head has something to be compared against, and
    bounded by ``tanh`` so it cannot emit a correction the prior distribution
    never contains.
    """

    def __init__(self, dim: int, extent: tuple[float, float, float]):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        self.register_buffer("extent", torch.tensor(extent), persistent=False)

    def forward(self, global_feat: Tensor) -> Tensor:
        return torch.tanh(self.mlp(global_feat)) * self.extent
