"""Which detection is which map element, and which point is which point.

Matching runs at point resolution: every detected point against every map
point, 768 x 576 scores, because that product *is* the assignment the solve
takes. Pooling each polyline into one element token scores 96 x 72 -- 64x
fewer -- and buys nothing the clock can see, because the model is loader-bound
and pooled tokens ran 10% *slower*. It also needs a second stage to expand an
element match back down to points, and one number per detection cannot say
which of four map chunks each of its points lies on. Both are built, so they
run as arms differing by exactly one thing.

**The matching is learned.** Rounds of self-attention (which of the four
parallel lines is this) and cross-attention (which map element does it look
like), then LightGlue's partial assignment: a dual softmax, so a pair has to be
each other's best match rather than merely a good one, gated by a matchability
score per token. Matchability is what lets a detection match *nothing* --
clutter, a false positive, a landmark outside the crop -- without forcing its
mass onto whichever map element is least implausible.

**The pooled path's expansion is not learned.** Both elements already carry
their points in their own frame, so once the elements correspond their points
correspond by shape: a soft nearest neighbour in local coordinates, with no
parameters and nothing to overfit. What shape alone cannot recover is which
chunk each point is on, so `per_point` adds a learned query that lets a point
choose its own map element rather than its element choosing for it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mapposeformer.model.attention import GeometricAttention
from mapposeformer.model.encoder import local_points


class MatcherLayer(nn.Module):
    """One round: context from one's own set, then a look at the other's."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mode: str,
        head_dim: int = 0,
        rope_bands: int = 0,
    ) -> None:
        super().__init__()
        self.self_attn = GeometricAttention(
            dim, heads, mode, head_dim, rope_bands
        )
        self.cross_attn = GeometricAttention(
            dim, heads, mode, head_dim, rope_bands
        )
        self.norm1, self.norm2, self.norm3 = (
            nn.LayerNorm(dim) for _ in range(3)
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )

    def forward(self, x: Tensor, xf, xv, xo, y: Tensor, yf, yv, yo) -> Tensor:
        """Update ``x`` given its own set and the other set ``y``."""
        x = x + self.self_attn(self.norm1(x), xf, self.norm1(x), xf, xv, xo)
        x = x + self.cross_attn(self.norm2(x), xf, self.norm2(y), yf, yv, yo)
        return x + self.ffn(self.norm3(x))


class Matcher(nn.Module):
    """Tokens in -- point or pooled element -- a point-level assignment out.

    @param temperature_m Width of the point-level soft nearest neighbour, in
        metres. Points within an element are 1.5 to 1.7 m apart here, so this
        is the scale at which "the same place along the element" stops being
        one point and becomes two.
    """

    def __init__(
        self,
        dim: int,
        layers: int = 2,
        heads: int = 2,
        mode: str = "relative",
        temperature_m: float = 1.0,
        per_point: bool = False,
        head_dim: int = 0,
        rope_bands: int = 0,
    ) -> None:
        super().__init__()
        self.map_layers = nn.ModuleList(
            MatcherLayer(dim, heads, mode, head_dim, rope_bands)
            for _ in range(layers)
        )
        self.det_layers = nn.ModuleList(
            MatcherLayer(dim, heads, mode, head_dim, rope_bands)
            for _ in range(layers)
        )
        self.final = nn.Linear(dim, dim)
        self.matchable = nn.Linear(dim, 1)
        # Lets a single detected *point* choose its map element, rather than
        # its whole element choosing for it. Two numbers in -- where the point
        # sits in its own element's frame -- so the query is the element's
        # token nudged by which end of it the point is at.
        #
        # Built only when it is used. A parameter with no path to the loss is
        # dead weight that still lands in every checkpoint, and the test that
        # asserts every parameter has a gradient is worth more than the
        # convenience of always having this one.
        self.point_query = (
            nn.Sequential(
                nn.Linear(dim + 2, dim), nn.GELU(), nn.Linear(dim, dim)
            )
            if per_point
            else None
        )
        self.temperature_m = temperature_m
        self.dim = dim

    def elements(
        self, map_e: tuple, det_e: tuple
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """The element-level assignment.

        @param map_e, det_e Each ``(token, frame, oriented, valid)``.

        @return ``(assign, scores, det_matchable, map_matchable, map_tok,
            det_tok)``. The two token sets are the attention rounds' output,
            handed back so ``point_elements`` can score a point against an
            element without running them twice. ``assign``
            is ``(B, D, M)``, non-negative, and is what the pose is solved
            from: a row summing to well under one is a detection the matcher
            declined to place. ``scores`` are the raw logits behind it, kept
            separate because the two questions -- *which* element, and whether
            there is one at all -- are supervised apart. Training on the
            product lets the model lower a "which" loss by raising
            matchability everywhere, which is not an answer to the question.
        """
        m, mf, mo, mv = map_e
        d, df, do, dv = det_e
        for m_layer, d_layer in zip(
            self.map_layers, self.det_layers, strict=True
        ):
            m, d = (
                m_layer(m, mf, mv, mo, d, df, dv, do),
                d_layer(d, df, dv, do, m, mf, mv, mo),
            )

        fm, fd = self.final(m), self.final(d)
        # The projections stay under autocast -- they are ordinary matmuls and
        # the weights are held in the autocast dtype. Only what happens *to*
        # their outputs needs the wider type.
        raw_d, raw_m = (
            self.matchable(d).squeeze(-1),
            self.matchable(m).squeeze(-1),
        )

        # The scores and the softmaxes over them are computed in fp32 even
        # under bf16 autocast, and this is not hygiene -- it is the difference
        # between a run that trains and one that does not.
        #
        # Measured on point tokens, 4 layers, identical seeds and schedule:
        # under bf16 the largest logit went 48 -> 1 144 -> 28 032 -> 4.9e5 by
        # step 250 and the match loss followed it, 6.4 -> 1 044, with the
        # assignment mass pinned at zero. In fp32 the same run held its logits
        # under 35 and brought the pose from 110 m to 20 m while the mass grew
        # 0.01 -> 0.25. bf16 carries eight mantissa bits, and a score matrix
        # this size is a sum over `dim` of products that must then survive two
        # softmaxes; element tokens have 96 x 72 of them and tolerate it,
        # point tokens have 768 x 576 and do not.
        with torch.autocast(device_type=fd.device.type, enabled=False):
            fd32, fm32 = fd.float(), fm.float()
            scores = torch.einsum("bdk,bmk->bdm", fd32, fm32) / self.dim**0.5
            scores = scores.masked_fill(~dv[:, :, None], -1e4)
            scores = scores.masked_fill(~mv[:, None, :], -1e4)

            # Dual softmax: a pair must be each other's best, not merely a
            # good one. A single row softmax always hands out all of its mass,
            # so a detection with no counterpart would still fully match
            # something.
            both = F.softmax(scores, dim=2) * F.softmax(scores, dim=1)

            sig_d = torch.sigmoid(raw_d.float()) * dv
            sig_m = torch.sigmoid(raw_m.float()) * mv
            assign = both * sig_d[:, :, None] * sig_m[:, None, :]
        return assign, scores, sig_d, sig_m, fm, fd

    def point_elements(
        self,
        map_tok: Tensor,
        det_tok: Tensor,
        det_local: Tensor,
        det_pmask: Tensor,
        map_valid: Tensor,
        sig_d: Tensor,
        sig_m: Tensor,
    ) -> Tensor:
        """Which map element each detected *point* belongs to.

        The element-level assignment cannot answer this. A detection spanning
        four map chunks has points in all four, and one number per detection
        can only say *which chunk the detection is*, not which chunk each
        point is on -- so every point ends up spread across all four, dragged
        toward their common centroid.

        The query is the detection's own token together with where the point
        sits inside it, so two points at opposite ends of the same element ask
        different questions. Everything else is the element stage's: the same
        keys, the same matchability.

        @param det_local ``(B, D, P, 2)`` from ``local_points``.

        @return ``(B, D, P, M)``, non-negative.
        """
        if self.point_query is None:
            raise RuntimeError("Matcher was built without the per-point head")
        B, D, P, _ = det_local.shape
        q = self.point_query(
            torch.cat(
                [det_tok[:, :, None].expand(B, D, P, self.dim), det_local], -1
            )
        )
        scores = torch.einsum("bdpk,bmk->bdpm", q, map_tok) / self.dim**0.5
        scores = scores.masked_fill(~map_valid[:, None, None, :], -1e4)
        # Both axes of the dual softmax have to be masked. Masking only the
        # map axis leaves the `dim=1` pass normalising over padded detection
        # slots, which at this data's real-slot rate quietly discards most of
        # the mass a detection should receive.
        scores = scores.masked_fill(~det_pmask.any(-1)[:, :, None, None], -1e4)
        both = F.softmax(scores, dim=3) * F.softmax(scores, dim=1)
        gate = sig_d[:, :, None, None] * sig_m[:, None, None, :]
        return both * gate * det_pmask[..., None]

    def points(
        self,
        assign: Tensor,
        det_pts: Tensor,
        det_pmask: Tensor,
        det_frame: Tensor,
        map_pts: Tensor,
        map_pmask: Tensor,
        map_frame: Tensor,
    ) -> Tensor:
        """Spread each element match over its points, by shape.

        Both elements carry their points in their own frame, so the points
        line up when the shapes do -- no parameters, and nothing that can
        learn a correspondence the geometry does not support.

        @param assign ``(B, D, M)`` per element, or ``(B, D, P, M)`` per
            detected point. The second is what a detection spanning several
            map chunks needs, and the first is the special case where every
            point of a detection is told the same thing.

        @return ``(B, D*P, M*P)``, ready for the pose solve.
        """
        dl = local_points(det_pts, det_pmask, det_frame)[..., :2]
        ml = local_points(map_pts, map_pmask, map_frame)[..., :2]

        # (B, D, P, M, P): every detected point against every map point, but
        # only within a pair of elements, and only as a distance.
        d2 = (
            (dl[:, :, :, None, None, :] - ml[:, None, None, :, :, :])
            .square()
            .sum(-1)
        )
        logit = -d2 / (2 * self.temperature_m**2)
        logit = logit.masked_fill(~map_pmask[:, None, None, :, :], -1e4)
        # Normalised *within* the pair, over the map element's own points.
        # Softmaxing over every map point at once would re-decide which
        # element matched -- a decision the element stage has already taken,
        # and taking it twice divides the evidence by the candidate count.
        within = F.softmax(logit, dim=-1)
        within = within * det_pmask[:, :, :, None, None]
        within = within * map_pmask.any(-1)[:, None, None, :, None]

        B, D, P, M, _ = within.shape
        if assign.dim() == 3:
            assign = assign[:, :, None].expand(B, D, P, M)
        full = within * assign[..., None]
        return full.reshape(B, D * P, M * P)
