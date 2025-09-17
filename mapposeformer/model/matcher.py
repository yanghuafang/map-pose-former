"""Soft correspondence between detected points and map points.

The output is a soft assignment matrix, and it is the model's only path to a
pose: the head downstream solves a rigid transform in closed form from it. That
is a deliberate constraint. A network that regresses a pose directly from a
pooled feature can be right for reasons that have nothing to do with geometry,
and it fails silently off-distribution. A network that must first say *this
detected point is that map point* can only be right for the right reason -- and
when it is wrong, the assignment shows you where.

The formulation is LightGlue's: dual softmax scaled by a per-point
*matchability*. There is no Sinkhorn loop, which keeps it one fused expression,
exportable, and free of an iteration count to tune. A point with no counterpart
-- a false positive, or a map element outside the camera's view -- gets low
matchability and contributes nothing, which is the role a dustbin plays
elsewhere without needing an extra row and column.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

#: Finite rather than ``-inf``. A row that is entirely padding then softmaxes
#: to something uniform and harmless instead of NaN, and the masks below zero
#: it out anyway.
_MASK_SCORE = -1e4


class SoftMatcher(nn.Module):
    """Dual-softmax assignment with learned matchability."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.matchability = nn.Linear(dim, 1)
        self.scale = dim**-0.25

    def forward(
        self, fa: Tensor, fb: Tensor, pad_a: Tensor, pad_b: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Match set ``a`` (detections) against set ``b`` (map).

        @param fa ``(B, K, D)`` detection point features.
        @param fb ``(B, L, D)`` map point features.
        @param pad_a ``(B, K)`` True where the token is padding.
        @param pad_b ``(B, L)`` likewise.

        @return ``(assign (B, K, L), scores (B, K, L))``. ``assign`` is in
            ``[0, 1]`` and its row sums are at most one: mass short of one is
            the model declining to match. ``scores`` is the raw compatibility,
            returned for the matching loss, which needs the logits and not the
            product.
        """
        a = self.proj(fa) * self.scale
        b = self.proj(fb) * self.scale
        scores = a @ b.transpose(1, 2)
        invalid = pad_a.unsqueeze(2) | pad_b.unsqueeze(1)
        scores = scores.masked_fill(invalid, _MASK_SCORE)

        sa = torch.sigmoid(self.matchability(fa)).masked_fill(pad_a.unsqueeze(-1), 0.0)
        sb = torch.sigmoid(self.matchability(fb)).masked_fill(pad_b.unsqueeze(-1), 0.0)

        assign = F.softmax(scores, dim=2) * F.softmax(scores, dim=1)
        assign = assign * sa * sb.transpose(1, 2)
        return assign.masked_fill(invalid, 0.0), scores
