"""One token per point, with the pooled element encoder beside it.

A lane divider is eight points, and so is the next lane divider over. Giving
attention one token per *point* means 1 344 of them and a score matrix
quadratic in that; pooling each element into one token means 168, and
64x less attention. That saving is real and it is not the constraint --
what pooling costs is resolution, and the pose is solved on the points at
ten centimetres. So ``PointEncoder`` is what the model runs, and
``ElementEncoder`` is the arm it is measured against; the two differ in one
thing only.

``ElementEncoder`` is VectorNet's subgraph encoder: embed each point, pool the
element, tell every point what its element looks like, embed again, pool
again. Two rounds is enough for a shape as simple as a 12 m polyline chunk,
and each round is where a point learns its place in the whole rather than
only its own coordinates.

**No absolute coordinate enters a token.** Each element carries its own frame
-- centroid, and heading from its first valid point to its last -- and its
points are encoded in that frame. Rigidly move the whole scene and every token
is unchanged; only the frames move. That is what lets the matcher be
SE(2)-equivariant by construction rather than by learning an invariance from
data, and it is why the frame comes out of this module beside the token
instead of being baked into it.

A point landmark has no shape and no direction: a pole is one point, so its
local coordinates are (0, 0) whatever the scene does, and its token is its
class. ``oriented`` says so, because a heading invented for it would be a
heading the matcher would then trust.

**One assumption worth naming.** A polyline's heading runs first point to
last, so a map that stores a lane divider one way and a detector that reports
it the other produce frames 180 degrees apart and tokens that do not match.
Here both sides are cut from the same world, so the order agrees by
construction. A real detector would not promise that, and the fix is to treat
the direction as an axis rather than a ray -- which costs the encoder the
ability to tell one end of an element from the other, so it is worth doing
only once something needs it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mapposeformer.data.classes import NUM_ATTRS, NUM_CLASSES

# : Feature per point, in its element's own frame: the local coordinate, its
# : distance from the centroid, and where the point falls along the element.
POINT_FEATURES = 4


def element_frames(pts: Tensor, pmask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Each element's own frame, and whether it has a direction at all.

    The centroid is the masked mean. The heading runs from the first valid
    point to the last, which is well defined because polyline points arrive
    ordered along the line -- and being a property of the points, it rotates
    with them, which is the whole point.

    @param pts ``(B, N, P, 2)``.
    @param pmask ``(B, N, P)`` true where the point exists.

    @return ``(frame, oriented, valid)``. ``frame`` is ``(B, N, 3)`` as
        ``(x, y, yaw)``; ``oriented`` is ``(B, N)``, false for elements with
        fewer than two points, whose yaw is set to zero and means nothing;
        ``valid`` is ``(B, N)``, false for padding.
    """
    count = pmask.sum(-1)
    valid = count > 0
    centroid = (pts * pmask.unsqueeze(-1)).sum(-2) / count.clamp_min(
        1
    ).unsqueeze(-1)

    # First and last valid point, found by pushing padding to the ends.
    idx = torch.arange(pts.shape[-2], device=pts.device)
    big = torch.iinfo(torch.int32).max
    first = (
        torch.where(pmask, idx, big).min(-1).values.clamp_max(pts.shape[-2] - 1)
    )
    last = torch.where(pmask, idx, -1).max(-1).values.clamp_min(0)
    take = lambda i: torch.gather(  # noqa: E731
        pts, 2, i[..., None, None].expand(-1, -1, 1, 2)
    ).squeeze(2)
    span = take(last) - take(first)

    # A directionless element gets +x substituted before the atan2, not a
    # zero yaw patched in after it: atan2(0, 0) is 0 going forward but NaN
    # coming back, and `where` multiplies that NaN by zero, which is still NaN.
    oriented = count > 1
    ray = torch.where(oriented.unsqueeze(-1), span, span.new_tensor([1.0, 0.0]))
    yaw = torch.atan2(ray[..., 1], ray[..., 0])
    return torch.cat([centroid, yaw.unsqueeze(-1)], -1), oriented, valid


def local_points(pts: Tensor, pmask: Tensor, frame: Tensor) -> Tensor:
    """The points as their own element sees them: centred, then derotated.

    @return ``(B, N, P, POINT_FEATURES)``, zero where the point is padding.
    """
    rel = pts - frame[..., None, :2]
    c, s = torch.cos(-frame[..., 2]), torch.sin(-frame[..., 2])
    x = c[..., None] * rel[..., 0] - s[..., None] * rel[..., 1]
    y = s[..., None] * rel[..., 0] + c[..., None] * rel[..., 1]

    # Where the point sits along the element, so a polyline's endpoints are
    # distinguishable from its middle without an absolute coordinate.
    order = torch.arange(pts.shape[-2], device=pts.device, dtype=pts.dtype)
    span = pmask.sum(-1, keepdim=True).clamp_min(2) - 1
    along = (order / span).clamp(0.0, 1.0)

    # The same trap as the heading: a point *at* its own centroid has
    # d|q|/dq = q/|q| = 0/0. The epsilon under the root makes that gradient
    # zero, which is the answer the limit does not have.
    radius = torch.sqrt(x * x + y * y + 1e-12)
    feat = torch.stack([x, y, radius, along], dim=-1)
    return feat * pmask.unsqueeze(-1)


def _masked_max(h: Tensor, pmask: Tensor) -> Tensor:
    """Pool over points, ignoring padding.

    Max rather than mean: an element's identity is its distinctive parts, and
    a mean over eight points dilutes them by however many points the element
    happened to be cut into.

    An element with *no* points -- a padded slot -- maxes over nothing and
    comes out -inf, which the next LayerNorm turns into NaN. Zeroing the token
    afterwards does not help, because ``0 * NaN`` is NaN. So the -inf is
    replaced here, before anything arithmetic can see it.
    """
    filled = h.masked_fill(~pmask.unsqueeze(-1), float("-inf")).max(-2).values
    return torch.where(
        pmask.any(-1).unsqueeze(-1), filled, torch.zeros_like(filled)
    )


class ElementEncoder(nn.Module):
    """Points, class and attribute in; one token and one frame out.

    Map and detections get their own instance. The semantics are shared -- a
    lane divider means the same thing on both sides, so the class and
    attribute embeddings are passed in and held in common -- but the
    *statistics* are not: the map is surveyed and the detections are a noisy,
    cluttered, occasionally mislabelled view of it. One set of weights would
    have to be both.
    """

    def __init__(
        self,
        dim: int,
        cls_embed: nn.Embedding,
        attr_embed: nn.Embedding,
        extra_dim: int = 0,
    ) -> None:
        """@param extra_dim Width of the per-element side channel: confidence
        and reported noise for detections, nothing for the map.
        """
        super().__init__()
        self.cls_embed, self.attr_embed = cls_embed, attr_embed
        self.point = nn.Sequential(
            nn.Linear(POINT_FEATURES, dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.subgraph = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(
                dim
                + cls_embed.embedding_dim
                + attr_embed.embedding_dim
                + extra_dim,
                dim,
            ),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        pts: Tensor,
        pmask: Tensor,
        cls: Tensor,
        attr: Tensor,
        extra: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """@return ``(token, frame, oriented, valid)``.

        The last three are ``element_frames``' output, passed through: the
        matcher needs the frames, and only this module knows the points.
        """
        frame, oriented, valid = element_frames(pts, pmask)

        h = self.point(local_points(pts, pmask, frame))
        h = self.subgraph(
            torch.cat([h, _masked_max(h, pmask).unsqueeze(-2).expand_as(h)], -1)
        )
        pooled = _masked_max(h, pmask)

        parts = [pooled, self.cls_embed(cls), self.attr_embed(attr)]
        if extra is not None:
            parts.append(extra)
        token = self.fuse(torch.cat(parts, -1))

        # Padding contributes nothing. The -inf that would have made this a
        # NaN is already gone -- see _masked_max -- so this is only tidiness.
        return token * valid.unsqueeze(-1), frame, oriented, valid


def embeddings(dim: int) -> tuple[nn.Embedding, nn.Embedding]:
    """The class and attribute tables the two encoders share."""
    return nn.Embedding(NUM_CLASSES, dim), nn.Embedding(NUM_ATTRS, dim)


class PointEncoder(nn.Module):
    """One token per *point*, for matching at point resolution.

    Element tokens exist to make attention 64x cheaper, and measurement says
    that saving is not the constraint: this model is loader-bound, the
    assignment tensor is the same 768 x 576 either way, and element tokens
    ran 10% *slower* than point tokens. What pooling costs is resolution --
    eight points collapsed into one max-pooled vector cannot say where each
    point was, and the pose needs that at ten centimetres.

    So this encodes each point on its own. The features are the same ones the
    element encoder builds, taken per point rather than pooled: where the
    point sits in its own element's frame, what class the element is, and for
    a detection what the detector claimed about it.

    Each point also carries a frame, which is its own position together with
    its element's heading. Nothing absolute reaches the token -- the position
    is used by the rotary encoding, which only ever sees differences.
    """

    def __init__(
        self,
        dim: int,
        cls_embed: nn.Embedding,
        attr_embed: nn.Embedding,
        extra_dim: int = 0,
    ) -> None:
        super().__init__()
        self.cls_embed, self.attr_embed = cls_embed, attr_embed
        width = (
            POINT_FEATURES
            + cls_embed.embedding_dim
            + attr_embed.embedding_dim
            + extra_dim
        )
        self.fuse = nn.Sequential(
            nn.Linear(width, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        pts: Tensor,
        pmask: Tensor,
        cls: Tensor,
        attr: Tensor,
        extra: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """@return ``(token, frame, oriented, valid)``, flattened to one token
        per point so the matcher sees ``(B, N*P, ...)`` and needs no
        change at all.
        """
        B, N, P, _ = pts.shape
        frame, oriented, _ = element_frames(pts, pmask)
        local = local_points(pts, pmask, frame)

        parts = [
            local,
            self.cls_embed(cls)[:, :, None].expand(B, N, P, -1),
            self.attr_embed(attr)[:, :, None].expand(B, N, P, -1),
        ]
        if extra is not None:
            parts.append(extra[:, :, None].expand(B, N, P, extra.shape[-1]))
        token = self.fuse(torch.cat(parts, -1)) * pmask.unsqueeze(-1)

        # The point's own position, with its element's heading. Absolute here
        # and relative by the time it reaches a score.
        yaw = frame[..., None, 2:3].expand(B, N, P, 1)
        point_frame = torch.cat([pts, yaw], -1)
        return (
            token.reshape(B, N * P, -1),
            point_frame.reshape(B, N * P, 3),
            oriented[:, :, None].expand(B, N, P).reshape(B, N * P),
            pmask.reshape(B, N * P),
        )
