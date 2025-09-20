"""From soft correspondences to a pose, in closed form and robustly.

Weighted Procrustes in SE(2): given detection points, map points and the soft
assignment between them, solve for the rigid transform that best aligns one to
the other. Exactly, differentiably, and with no parameters -- the transform is
*determined* by the correspondences, so all the model's capacity goes to the
question that is actually hard.

**The first version of this head lost to its own baseline.** A plain weighted
least-squares fit is unbounded: a handful of confident, wrong correspondences
drag it arbitrarily far, and one ablation reached 30.9 degrees of heading error
where a ``tanh``-bounded regressor could only be vague. ``docs/RESULTS.md`` has
the table; ``docs/ARCHITECTURE.md`` has why it is structural rather than bad
luck.

The repair is the standard one, and it is two lines each. Abstentions are
**gated out** rather than scaled down, at a threshold relative to the frame's
strongest match. Then the residuals are **reweighted** -- Geman-McClure, twice,
unrolled so each pass is a real term in the gradient.
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

    @param src ``(B, K, 2)`` source points -- detections, in the ego frame.
    @param dst ``(B, K, 2)`` their targets -- map points, in the anchor frame.
    @param weight ``(B, K)`` non-negative confidence per correspondence.

    @return ``(pose (B, 3), mass (B,))``. ``pose`` is ``(x, y, yaw)`` such that
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
        [
            c * src_bar[:, 0] - s * src_bar[:, 1],
            s * src_bar[:, 0] + c * src_bar[:, 1],
        ],
        dim=-1,
    )
    t = dst_bar - rot_src
    return torch.cat([t, yaw.unsqueeze(-1)], dim=-1), mass.squeeze(-1)


class ProcrustesPoseHead(nn.Module):
    """Turn an assignment matrix into a pose, robustly. Parameter-free."""

    def __init__(
        self,
        irls_iters: int = 2,
        irls_scale_m: float = 1.0,
        min_row_mass: float = 0.05,
    ):
        """Args:
        irls_iters: Reweight-and-resolve passes after the initial fit. Two is
            where the measured benefit stops on this problem; the estimator is
            unrolled, so each one is a real term in the gradient.
        irls_scale_m: The residual at which a correspondence stops looking like
            noise and starts looking like a mistake. Set at the matching
            radius the loss uses, because that is already the distance at
            which this project calls two points the same point.
        min_row_mass: Abstention threshold, **as a fraction of the strongest
            match in the same frame**. Zero disables the gate. The relative
            form is deliberate -- see the module docstring.
        """
        super().__init__()
        self.irls_iters = irls_iters
        self.irls_scale_m = irls_scale_m
        self.min_row_mass = min_row_mass

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
        # map_pts`` multiplies a weight by a *coordinate*, and bf16 would put a
        # lattice under the answer. See ``geometry.exact_arithmetic``.
        with G.exact_arithmetic(assign.device.type):
            assign, det_pts, map_pts = (
                assign.float(),
                det_pts.float(),
                map_pts.float(),
            )
            w = assign.sum(dim=2)
            target = assign @ map_pts / w.clamp_min(_EPS).unsqueeze(-1)
            # An abstention contributes nothing rather than a little. The
            # comparison is not differentiable and does not need to be: it
            # selects, and ``w`` still carries the gradient where it passes.
            if self.min_row_mass > 0.0:
                floor = self.min_row_mass * w.max(dim=1, keepdim=True).values
                w = w * (w >= floor)

            pose, mass = weighted_procrustes_se2(det_pts, target, w)
            scale_sq = self.irls_scale_m**2
            for _ in range(self.irls_iters):
                # Geman-McClure: weight falls off as the residual grows and
                # reaches zero only in the limit, so no correspondence is ever
                # discarded discontinuously and the gradient stays smooth.
                r = (G.transform_points(pose, det_pts) - target).square().sum(-1)
                pose, mass = weighted_procrustes_se2(
                    det_pts, target, w * scale_sq / (scale_sq + r)
                )
            return pose, mass


class RegressionPoseHead(nn.Module):
    """Baseline: predict the pose straight from a pooled feature.

    Present so the closed-form head has something to be compared against, and
    bounded by ``tanh`` so it cannot emit a correction the prior distribution
    never contains. That bound is why it won the first comparison, and it is
    the property the robust solve above was written to match honestly.
    """

    def __init__(self, dim: int, extent: tuple[float, float, float]):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        self.register_buffer("extent", torch.tensor(extent), persistent=False)

    def forward(self, global_feat: Tensor) -> Tensor:
        return torch.tanh(self.mlp(global_feat)) * self.extent
