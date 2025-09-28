"""Pre-norm transformer blocks, and the null token that keeps them finite.

Nothing exotic here; the file exists so the rest of the model reads as
architecture rather than as plumbing. One decision is load-bearing and easy to
miss: every token set is prefixed with a **null token** that is never masked.

Attention over a set where every key is padding produces a softmax over
nothing, which is NaN, and a NaN survives every mask applied afterwards --
``0 * NaN`` is still NaN. A frame at the edge of a scene with no visible map
element is rare and completely legal, so the model must be finite there rather
than merely unlikely to hit it. One always-valid key makes the softmax
well-posed, and gives attention somewhere to send probability mass when a token
has no good match.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# : Additive score for a masked key. Finite, for the reason ``matcher.py``
# gives: : a row of ``-inf`` softmaxes to NaN, and NaN survives every mask
# after it.
_MASK_SCORE = -1e4


class FeedForward(nn.Sequential):
    """Standard two-layer MLP with GELU."""

    def __init__(self, dim: int, mult: int = 2):
        super().__init__(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim)
        )


class SelfBlock(nn.Module):
    """Self-attention within one set: context inside map, or in detections."""

    def __init__(self, dim: int, heads: int, ffn_mult: int = 2):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_mult)

    def forward(self, x: Tensor, pad: Tensor) -> Tensor:
        """``x`` is ``(B, N, D)``; ``pad`` is ``(B, N)``, True where invalid."""
        h = self.norm_attn(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + a
        return x + self.ffn(self.norm_ffn(x))


class CrossBlock(nn.Module):
    """Cross-attention: ``x`` reads ``y``. This is where matching happens."""

    def __init__(self, dim: int, heads: int, ffn_mult: int = 2):
        super().__init__()
        self.heads = heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_mult)

    def forward(self, x: Tensor, y: Tensor, y_pad: Tensor) -> Tensor:
        h = self.norm_kv(y)
        a, _ = self.attn(
            self.norm_q(x), h, h, key_padding_mask=y_pad, need_weights=False
        )
        x = x + a
        return x + self.ffn(self.norm_ffn(x))


class AttentionPool(nn.Module):
    """Pool a masked token set into one vector with a learned query.

    Mean pooling would work, but the volume head wants a summary weighted by
    *informativeness* -- a frame whose evidence is one pole and forty metres of
    parallel lane line should not be summarised as mostly lane line.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: Tensor, pad: Tensor) -> Tensor:
        """Returns ``(B, D)``, zero for a row with no unmasked token.

        This is the one attention with no null token in front of it, because
        the volume head is handed the token sets with theirs already stripped,
        and a frame at the edge of a scene can have neither a visible map
        element nor a detection.

        ``nn.MultiheadAttention`` currently returns zeros for such a row rather
        than the NaN the null token exists to prevent elsewhere, which is the
        answer we want -- no evidence, no summary. Written out rather than
        relied upon, because it is a convention of one implementation and this
        graph is meant to leave PyTorch for TensorRT, where a softmax over
        nothing is free to do something else.
        """
        empty = pad.all(dim=1, keepdim=True)
        h = self.norm(x)
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, h, h, key_padding_mask=pad, need_weights=False)
        return out.squeeze(1).masked_fill(empty, 0.0)


def prepend_null(
    x: Tensor, pad: Tensor, token: Tensor
) -> tuple[Tensor, Tensor]:
    """Prefix a never-masked token, so no attention softmax is ever empty.

    Returns the extended ``(x, pad)``; strip with ``x[:, 1:]`` afterwards.
    """
    b = x.shape[0]
    x = torch.cat([token.expand(b, 1, -1), x], dim=1)
    pad = torch.cat(
        [torch.zeros(b, 1, dtype=torch.bool, device=pad.device), pad], dim=1
    )
    return x, pad
