"""Attention whose scores carry the geometry between elements.

Matching is not a similarity problem. Four lane dividers in a row look alike by
construction, and the thing that tells them apart is where they sit relative to
everything else in the frame -- the boundary to the left, the stop line ahead.
So the score between two elements has to see their relative geometry, not only
their appearance.

**Where the geometry goes decides what the network is invariant to**, and this
module offers the three choices because what equivariance is worth is a
question to be measured, not assumed.

``relative`` -- the offset and bearing from element *i* to element *j*, measured
in *i*'s own frame, added to the score through a small MLP. Rigidly move the
whole scene and every one of these is unchanged, so the network is
SE(2)-equivariant by construction. It costs an N x N intermediate, which over
1 344 point tokens was prohibitive and over 168 element tokens is not -- the
same idea that was measured at 2.4x the step time is 64x smaller here.

``rope`` -- LightGlue's rotary encoding: rotate queries and keys by their own
position, so that q_i . k_j depends on p_j - p_i for free, with no N x N
intermediate at all. The offsets are in the *shared* frame rather than each
element's own, which makes this translation-equivariant but not rotation-
equivariant. That is not a defect to be fixed. The prior fixes heading to
within 3 degrees, so the two sets arrive nearly aligned and "this detection
points the same way as that map element" is real evidence -- which full
equivariance throws away and this keeps.

``absolute`` -- positions projected and added to the tokens, the baseline that
learns any invariance it wants from data, or does not.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mapposeformer import geometry as G

#: Relative geometry between a pair: offset in the query's frame, relative
#: bearing as a unit vector, and whether that bearing means anything.
PAIR_FEATURES = 6


def pair_geometry(
    q_frame: Tensor, k_frame: Tensor, k_oriented: Tensor
) -> Tensor:
    """Where each key sits, as each query sees it.

    @param q_frame ``(B, Nq, 3)``, @param k_frame ``(B, Nk, 3)``.
    @param k_oriented ``(B, Nk)`` false where the key has no heading.

    @return ``(B, Nq, Nk, PAIR_FEATURES)``. Unchanged by a rigid motion of the
        whole scene wherever the query has a heading: the offset is derotated
        into the query's frame and the bearing is a difference of headings. A
        query with no heading -- a pole is one point and has no direction to
        derotate by -- carries yaw zero in every scene, so its offsets stay in
        the shared frame and turn with it.
    """
    delta = k_frame[:, None, :, :2] - q_frame[:, :, None, :2]
    yaw = q_frame[..., 2]
    c, s = torch.cos(-yaw)[..., None], torch.sin(-yaw)[..., None]
    x = c * delta[..., 0] - s * delta[..., 1]
    y = s * delta[..., 0] + c * delta[..., 1]

    rel = G.wrap_angle(k_frame[:, None, :, 2] - q_frame[:, :, None, 2])
    oriented = k_oriented[:, None, :].expand_as(rel).to(rel.dtype)
    # Heading as (cos, sin) rather than an angle: an MLP fed a wrapped angle
    # has to learn that -pi and +pi are the same input, and it does not.
    return torch.stack(
        [
            x,
            y,
            torch.sqrt(x * x + y * y + 1e-12),
            torch.cos(rel) * oriented,
            torch.sin(rel) * oriented,
            oriented,
        ],
        dim=-1,
    )


class RelativeBias(nn.Module):
    """One additive score bias per head, from the geometry of the pair."""

    def __init__(self, heads: int, hidden: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(PAIR_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, heads),
        )

    def forward(
        self, q_frame: Tensor, k_frame: Tensor, k_oriented: Tensor
    ) -> Tensor:
        """@return ``(B, heads, Nq, Nk)``."""
        geo = pair_geometry(q_frame, k_frame, k_oriented)
        return self.mlp(geo).permute(0, 3, 1, 2)


#: Bands the default derivation stops at. Five reaches 1.875 m, just coarser
#: than the 1.714 m map point pitch; a sixth would encode below what the map
#: can express while spending six more channels of content capacity to do it.
MAP_PITCH_BANDS = 5


class RotaryFrames(nn.Module):
    """LightGlue's rotary encoding, extended from (x, y) to (x, y, yaw).

    The head dimension is split three ways, one part per coordinate, and each
    part carries ``bands`` frequencies. Each band rotates its pair of channels
    by a frequency times its coordinate, so a dot product between a rotated
    query and a rotated key depends on the *difference* of those coordinates
    and nothing else. Position never enters the values, only the scores.
    """

    def __init__(
        self, head_dim: int, scale_m: float = 30.0, bands: int = 0
    ) -> None:
        """@param scale_m Longest wavelength, in metres. The map crop reaches
        50 m out, so there is no structure to encode at a coarser scale than
        roughly this.
        @param bands Frequency bands; 0 derives `min(head_dim // 6, 5)`.

            The cap is the point of it. Wavelengths are `scale_m / 2^k`, so
            five bands bottom out at 1.875 m -- just coarser than the 1.714 m
            map point pitch that point tokens exist to resolve. Bands beyond
            that encode detail the map cannot express: at head_dim 64 an
            uncapped derivation gives ten, reaching 5.9 cm, and measures 3.05
            points of recall worse than five.

            Capping also decouples the two questions. An uncapped derivation
            makes positional *bandwidth* a function of head *width*, so a
            head-count sweep at fixed `dim` silently sweeps the encoding too.
            Capped, every head width at or above 30 derives the same five, and
            "does head count matter" is answerable without also asking "does
            positional resolution matter".

            Below head_dim 30 there is no room for five and the derivation
            takes what fits, which is what makes small models constructible at
            all. Stating `bands` explicitly overrides both.
        """
        super().__init__()
        # Three coordinates, two channels each: six per band. A head_dim that
        # is not a multiple of six leaves a remainder, and the remainder rides
        # through unrotated rather than being an error -- this project's own
        # default is head_dim 64, and refusing it would make the mode
        # unreachable exactly where it is wanted.
        bands = bands or min(head_dim // 6, MAP_PITCH_BANDS)
        if bands == 0:
            raise ValueError(f"head_dim must be at least 6, got {head_dim}")
        if 6 * bands > head_dim:
            raise ValueError(
                f"{bands} bands need {6 * bands} channels, "
                f"head_dim is {head_dim}"
            )
        self.rotated = 6 * bands
        k = torch.arange(bands, dtype=torch.float32)
        freq = (2 * math.pi / scale_m) * (2.0**k)
        # Yaw is an angle already: one cycle per turn is its natural unit.
        self.register_buffer("freq_xy", freq, persistent=False)
        self.register_buffer("freq_yaw", 2.0**k, persistent=False)

    def forward(self, h: Tensor, frame: Tensor) -> Tensor:
        """Rotate ``(B, heads, N, head_dim)`` by each element's own frame."""
        ang = torch.cat(
            [
                frame[..., 0, None] * self.freq_xy,
                frame[..., 1, None] * self.freq_xy,
                frame[..., 2, None] * self.freq_yaw,
            ],
            dim=-1,
        )[:, None]  # (B, 1, N, 3 * bands)
        cos, sin = torch.cos(ang), torch.sin(ang)
        spun, rest = h[..., : self.rotated], h[..., self.rotated :]
        even, odd = spun[..., 0::2], spun[..., 1::2]
        spun = torch.stack(
            [even * cos - odd * sin, even * sin + odd * cos], dim=-1
        ).flatten(-2)
        return torch.cat([spun, rest], dim=-1)


class GeometricAttention(nn.Module):
    """Multi-head attention, with the geometry injected the chosen way.

    Heads are not free of consequence here even though they are free of cost:
    softmax normalises, so one head holds exactly one attention distribution,
    and an element plausibly needs two at once -- its lateral neighbours, to
    settle which of four parallel lines it is, and along-track structure, to
    settle where along the road it is. The limit in the other direction is
    rank: a head's score matrix is rank at most ``head_dim``.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mode: str,
        head_dim: int = 0,
        rope_bands: int = 0,
    ) -> None:
        """@param head_dim Width of each head; 0 derives `dim // heads`.

        Deriving it is why "how many heads?" cannot be asked directly. With
        `head_dim = dim // heads`, changing the head count also changes the
        per-head score-matrix rank cap -- 2 heads at dim 128 is head_dim 64,
        4 heads is 32 -- so the sweep moves the very quantity the rank
        argument is about. Stating it separately costs parameters instead:
        the inner width becomes `heads * head_dim` and the projections widen
        with it, which is the right trade here because parameters are close to
        free and confounded comparisons are not.
        @param rope_bands Passed to :class:`RotaryFrames`; 0 derives it.
        """
        super().__init__()
        if head_dim == 0 and dim % heads:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        if mode not in ("relative", "rope", "absolute"):
            raise ValueError(f"unknown geometry mode {mode!r}")
        self.heads = heads
        self.head_dim = head_dim or dim // heads
        self.mode = mode
        inner = self.heads * self.head_dim
        self.q = nn.Linear(dim, inner)
        self.k = nn.Linear(dim, inner)
        self.v = nn.Linear(dim, inner)
        self.out = nn.Linear(inner, dim)
        self.bias = RelativeBias(heads) if mode == "relative" else None
        self.rotary = (
            RotaryFrames(self.head_dim, bands=rope_bands)
            if mode == "rope"
            else None
        )
        self.absolute = nn.Linear(4, dim) if mode == "absolute" else None

    def _split(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.heads, self.head_dim).transpose(1, 2)

    def _place(self, tok: Tensor, frame: Tensor) -> Tensor:
        """The absolute mode's only difference: position added to the token."""
        if self.absolute is None:
            return tok
        xy, yaw = frame[..., :2], frame[..., 2]
        pos = torch.cat(
            [xy, torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]], -1
        )
        return tok + self.absolute(pos)

    def forward(
        self,
        q_tok: Tensor,
        q_frame: Tensor,
        kv_tok: Tensor,
        kv_frame: Tensor,
        kv_valid: Tensor,
        kv_oriented: Tensor,
    ) -> Tensor:
        """@return ``(B, Nq, dim)``, the attended update to the queries."""
        q = self._split(self.q(self._place(q_tok, q_frame)))
        k = self._split(self.k(self._place(kv_tok, kv_frame)))
        v = self._split(self.v(kv_tok))

        if self.rotary is not None:
            q, k = self.rotary(q, q_frame), self.rotary(k, kv_frame)

        # Padding is suppressed with a large finite penalty rather than -inf.
        # A frame whose map crop came back empty would mask a whole softmax
        # row, and softmax over all -inf is NaN -- which then survives every
        # downstream `where` and poisons the gradient. exp(-1e4) is zero in
        # every dtype here, so this costs nothing and cannot produce a NaN.
        mask = torch.zeros_like(kv_valid, dtype=q.dtype)
        mask = mask.masked_fill(~kv_valid, -1e4)[:, None, None]
        if self.bias is not None:
            mask = mask + self.bias(q_frame, kv_frame, kv_oriented)

        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(h.transpose(1, 2).flatten(2))
