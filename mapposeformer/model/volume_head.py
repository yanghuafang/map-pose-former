"""The learned cost volume: a score for every pose hypothesis on a grid.

camera-map-localization scores an explicit grid of ``(forward, left, yaw)``
hypotheses against a distance transform and takes the argmin. This head predicts
that same surface instead of computing it, over the same three axes, and it is
kept for the same three reasons the classical version needed it:

* **Ambiguity is visible.** A stretch of parallel lane lines should produce a
  ridge along the road, not a peak. A single regressed pose cannot say that; a
  surface can, and looking at it is how you find out whether a model has
  actually localized or has merely guessed the mean of the prior.
* **Covariance falls out of it.** The spread of the surface about its peak is a
  measurement covariance, which is what the downstream filter needs. It is
  computed here the same way the classical repo computes it -- a softmax-weighted
  second moment -- rather than regressed by a separate head, so the uncertainty
  and the evidence cannot disagree.
* **It survives quantization.** Classification over a grid degrades gracefully
  as precision drops; direct coordinate regression does not.

The grid extent must match the prior's truncation bounds. A target outside the
grid has no correct cell, and the loss would be asking for something the head
cannot represent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mapposeformer import geometry as G
from mapposeformer.model.attention import AttentionPool


@dataclass(frozen=True)
class GridParams:
    """Resolution and extent of the hypothesis grid, in the anchor frame."""

    num_x: int = 19
    num_y: int = 17
    num_yaw: int = 13
    extent_x_m: float = 4.5
    extent_y_m: float = 2.0
    extent_yaw_deg: float = 3.0

    @property
    def size(self) -> int:
        return self.num_x * self.num_y * self.num_yaw


class VolumeHead(nn.Module):
    """Pooled features -> hypothesis logits, covariance, and a trust score."""

    def __init__(self, dim: int, heads: int, grid: GridParams, min_std=(0.05, 0.05, 0.002)):
        super().__init__()
        self.grid = grid
        self.pool = AttentionPool(dim, heads)
        self.logits = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, grid.size)
        )
        # One scalar, not a full 3x3. The *shape* of the uncertainty is already
        # in the surface; what a learned term adds is calibration -- the surface
        # is sharper or flatter than the true error by a roughly constant factor,
        # and that is all this is allowed to fix.
        self.log_scale = nn.Linear(dim, 1)
        self.trust = nn.Linear(dim, 1)

        deg = torch.pi / 180.0
        axes = (
            torch.linspace(-grid.extent_x_m, grid.extent_x_m, grid.num_x),
            torch.linspace(-grid.extent_y_m, grid.extent_y_m, grid.num_y),
            torch.linspace(
                -grid.extent_yaw_deg * deg, grid.extent_yaw_deg * deg, grid.num_yaw
            ),
        )
        cells = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        self.register_buffer("cells", cells.reshape(-1, 3), persistent=False)
        # Cell pitch per axis, in that axis's own units. The volume loss needs
        # it to size its soft target, and deriving it there would mean an
        # ``unique()`` over the whole grid on every step.
        self.register_buffer(
            "pitch",
            torch.tensor([float(a[1] - a[0]) if len(a) > 1 else 1.0 for a in axes]),
            persistent=False,
        )
        self.register_buffer("min_var", torch.tensor(min_std).square(), persistent=False)

    def forward(self, tokens: Tensor, pad: Tensor) -> dict[str, Tensor]:
        """Args: ``(B, N, D)`` tokens from both sets, ``(B, N)`` padding mask.

        Returns a dict with ``logits (B, G)``, ``delta (B, 3)`` (the soft
        argmax), ``cov (B, 3, 3)``, ``trust_logit (B,)`` and ``feat (B, D)``.
        """
        g = self.pool(tokens, pad)
        logits = self.logits(g)

        # The statistics below weight *coordinates* by probabilities, so they
        # run in fp32 for the reason ``geometry.exact_arithmetic`` gives. The
        # logits themselves are returned untouched: their loss is a cross
        # entropy, which autocast already keeps in fp32 by policy.
        with G.exact_arithmetic(tokens.device.type):
            prob = F.softmax(logits.float(), dim=-1)
            mean = prob @ self.cells
            residual = self.cells.unsqueeze(0) - mean.unsqueeze(1)
            cov = torch.einsum("bg,bgi,bgj->bij", prob, residual, residual)
            # ``g`` is cast rather than the result: with autocast off inside
            # this block, a bf16 activation against fp32 weights is an
            # error rather than a silent promotion.
            cov = cov * torch.exp(self.log_scale(g.float())).view(-1, 1, 1)
            # A floor on the diagonal, for the same reason the classical
            # filter gates on a flat cost surface: a peak one cell wide would
            # otherwise report near-zero variance and let a single frame
            # dominate the filter.
            cov = cov + torch.diag_embed(self.min_var.expand_as(mean))
        return {
            "logits": logits,
            "delta": mean,
            "cov": cov,
            "trust_logit": self.trust(g).squeeze(-1),
            "feat": g,
        }
