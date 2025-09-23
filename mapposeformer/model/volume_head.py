"""The cost volume: the alignment error at every pose hypothesis on a grid.

camera-map-localization scores a grid of ``(forward, left, yaw)`` hypotheses
against a distance transform and takes the argmin. This head scores the same
grid on the same three axes -- and, like the classical version, it **computes**
that surface rather than predicting it.

That is the decision this file exists for. An earlier version regressed the
4199 logits from a pooled vector with an MLP: a third of the model's parameters
spent drawing a picture of a cost surface, whose ridges are what the *prior
over ridges* looks like and whose covariance is a prediction of uncertainty
rather than a measurement of one. ``docs/ARCHITECTURE.md`` has the full
argument and the two papers it follows.

Computing it is free, which is the pleasant part: the assignment-weighted
squared error is a quadratic in the hypothesis, so the whole grid follows in
closed form from the same statistics :mod:`~mapposeformer.model.pose_head`
already forms to solve for the pose. See :func:`grid_cost`.

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

_EPS = 1e-6


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


def grid_cost(
    assign: Tensor, det: Tensor, mp: Tensor, t: Tensor, rot: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Assignment-weighted squared alignment error at every hypothesis.

    For a hypothesis :math:`(R, t)` the cost is

    .. math:: C(R,t) = \sum_{ij} a_{ij} \, \lVert R d_i + t - m_j \rVert^2

    which looks like a sum over ``G × K × L`` and is not. Expanding the square
    separates the hypothesis from the data completely: :math:`\lVert R d_i
    \rVert^2 = \lVert d_i \rVert^2` because rotations preserve length, and
    every remaining term factors through one of eleven numbers --- the total
    mass, two weighted second moments, two weighted centroids, and the
    :math:`2 \times 2` cross-covariance :math:`M = \sum_{ij} a_{ij} d_i
    m_j^\top`. Those are exactly the statistics weighted Procrustes forms to
    find the *minimum* of this surface, so evaluating all 4199 cells of it
    costs a handful of small matmuls on top.

    @param assign ``(B, K, L)`` soft correspondence.
    @param det ``(B, K, 2)`` detection points, ego frame.
    @param mp ``(B, L, 2)`` map points, anchor frame.
    @param t ``(G, 2)`` hypothesis translations.
    @param rot ``(G, 2, 2)`` hypothesis rotations.

    @return ``(cost (B, G), mass (B,))``. Exact -- ``tests/test_model.py``
        checks it against the brute-force sum over every cell.
    """
    w = assign.sum(dim=2)  # (B, K) mass on each detection point
    v = assign.sum(dim=1)  # (B, L) mass on each map point
    mass = w.sum(dim=1)
    sq_d = (w * det.square().sum(-1)).sum(1)
    sq_m = (v * mp.square().sum(-1)).sum(1)
    cen_d = torch.einsum("bk,bkc->bc", w, det)
    cen_m = torch.einsum("bl,blc->bc", v, mp)
    cross = det.transpose(1, 2) @ (assign @ mp)  # (B, 2, 2)

    cost = (
        (sq_d + sq_m).unsqueeze(1)
        + mass.unsqueeze(1) * t.square().sum(-1)
        + 2 * torch.einsum("gc,gcj,bj->bg", t, rot, cen_d)
        # tr(R M), not the Frobenius product: the transpose is the whole
        # difference between this surface and a differently shaped one.
        - 2 * torch.einsum("gij,bji->bg", rot, cross)
        - 2 * cen_m @ t.transpose(0, 1)
    )
    return cost, mass


class VolumeHead(nn.Module):
    """The measured cost surface, its covariance, and a trust score."""

    def __init__(
        self,
        dim: int,
        heads: int,
        grid: GridParams,
        min_std=(0.05, 0.05, 0.002),
    ):
        super().__init__()
        self.grid = grid
        self.pool = AttentionPool(dim, heads)
        # One learned scalar turns mean squared metres into logits, plus a
        # per-sample correction: a frame resting on two correspondences and one
        # resting on two hundred have surfaces worth trusting differently, and
        # the shape alone cannot say which is which.
        self.log_sharpness = nn.Parameter(torch.zeros(1))
        self.sharpness = nn.Linear(dim, 1)
        nn.init.zeros_(self.sharpness.weight)
        nn.init.zeros_(self.sharpness.bias)
        # One scalar, not a full 3x3. The *shape* of the uncertainty is already
        # in the surface; what a learned term adds is calibration -- the
        # surface is sharper or flatter than the true error by a roughly
        # constant factor, and that is all this is allowed to fix.
        self.log_scale = nn.Linear(dim, 1)
        # Zero, so calibration starts at "no correction". Random weights here
        # put ``exp(log_scale)`` anywhere over two orders of magnitude, and the
        # samples that drew a small one report near-zero variance against a
        # 1.5 m residual -- an NLL in the hundreds, for the first few hundred
        # steps, from nothing but the initializer.
        nn.init.zeros_(self.log_scale.weight)
        nn.init.zeros_(self.log_scale.bias)
        self.trust = nn.Linear(dim, 1)

        deg = torch.pi / 180.0
        axes = (
            torch.linspace(-grid.extent_x_m, grid.extent_x_m, grid.num_x),
            torch.linspace(-grid.extent_y_m, grid.extent_y_m, grid.num_y),
            torch.linspace(
                -grid.extent_yaw_deg * deg,
                grid.extent_yaw_deg * deg,
                grid.num_yaw,
            ),
        )
        cells = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        self.register_buffer("cells", cells, persistent=False)
        # The same hypotheses as rotation matrices and translations, because
        # that is the form the closed form wants and rebuilding them per step
        # would be trigonometry in the inner loop.
        c, s = torch.cos(cells[:, 2]), torch.sin(cells[:, 2])
        rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
        self.register_buffer("cell_t", cells[:, :2].contiguous(), persistent=False)
        self.register_buffer("cell_rot", rot, persistent=False)
        # Cell pitch per axis, in that axis's own units. The volume loss needs
        # it to size its soft target, and deriving it there would mean an
        # ``unique()`` over the whole grid on every step.
        self.register_buffer(
            "pitch",
            torch.tensor([float(a[1] - a[0]) if len(a) > 1 else 1.0 for a in axes]),
            persistent=False,
        )
        self.register_buffer("min_var", torch.tensor(min_std).square(), persistent=False)

    def forward(
        self,
        assign: Tensor,
        det: Tensor,
        mp: Tensor,
        tokens: Tensor,
        pad: Tensor,
    ) -> dict[str, Tensor]:
        """Args: ``(B, K, L)`` assignment, its two point sets, and the tokens.

        Returns a dict with ``logits (B, G)``, ``delta (B, 3)`` (the soft
        argmin), ``cov (B, 3, 3)``, ``trust_logit (B,)`` and ``feat (B, D)``.
        """
        g = self.pool(tokens, pad)

        # Coordinates multiplied by weights, throughout -- which is the case
        # ``geometry.exact_arithmetic`` exists for. bf16 would put a 0.25 m
        # lattice under a 40 m map point and the surface would inherit it.
        with G.exact_arithmetic(tokens.device.type):
            cost, mass = grid_cost(
                assign.float(),
                det.float(),
                mp.float(),
                self.cell_t,
                self.cell_rot,
            )
            # Per correspondence, so the scale is a mean squared residual in
            # metres and does not move with how much the matcher matched. The
            # shift is free -- softmax ignores it -- and keeps the exponent
            # away from the large common offset every cell shares.
            cost = cost / mass.clamp_min(_EPS).unsqueeze(-1)
            cost = cost - cost.min(dim=-1, keepdim=True).values
            sharp = F.softplus(self.log_sharpness + self.sharpness(g.float()))
            logits = -cost * sharp

            prob = F.softmax(logits, dim=-1)
            mean = prob @ self.cells
            residual = self.cells.unsqueeze(0) - mean.unsqueeze(1)
            cov = torch.einsum("bg,bgi,bgj->bij", prob, residual, residual)
            # Clamped before the exponential. Training pushes this weight
            # positive by design, and fp32 ``exp`` overflows at 88 -- so
            # without a bound the scale reaches inf eventually rather than
            # by accident, and every weight in the model is NaN one step
            # later. Eight is three orders of magnitude of correction
            # either way, far more than a calibration factor needs.
            scale = self.log_scale(g.float()).clamp(-8.0, 8.0)
            cov = cov * torch.exp(scale).view(-1, 1, 1)
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
