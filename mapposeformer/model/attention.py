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
import torch.nn.functional as F
from torch import Tensor

# : Additive score for a masked key. Finite, for the reason ``matcher.py``
# gives: : a row of ``-inf`` softmaxes to NaN, and NaN survives every mask
# after it.
_MASK_SCORE = -1e4


class MultiheadAttention(nn.Module):
    """Multi-head attention with its projections exposed as ``nn.Linear``.

    The arithmetic is torch's. What differs is the layout. ``nn.Multihead-
    Attention`` packs the input projection into one ``in_proj_weight``
    *Parameter*, and both ``quantize.py`` and ``prune.py`` find their work by
    walking the module tree for ``nn.Linear`` -- so attention was invisible to
    both. Four Linears make it visible without changing the answer, and
    :func:`unpack_attention` maps an existing checkpoint onto this layout, so
    nothing retrains.

    The projections are also independently *sized*, which ``prune.py`` needs:
    ``nn.MultiheadAttention`` requires its internal width to equal
    ``embed_dim``, and dropping a head makes those differ.

    Masking is additive and finite, using this file's ``_MASK_SCORE``, so a row
    whose keys are all padding attends uniformly instead of producing NaN.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        self.heads, self.head_dim = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _heads(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, None]:
        """@return ``(output, None)``, shaped like torch's so call sites are
        unchanged. Weights are never returned; nothing here asks for them."""
        b, n, _ = q.shape
        qh, kh, vh = (
            self._heads(self.q_proj(q)),
            self._heads(self.k_proj(k)),
            self._heads(self.v_proj(v)),
        )
        bias = None
        if attn_mask is not None:
            bias = attn_mask.view(b, self.heads, n, -1)
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, :]
            if bias is None:
                # (B, 1, 1, M), which broadcasts. Materialising the full
                # (B, heads, N, M) here would allocate the very tensor this
                # class exists to stop allocating: at batch 64 that is 7.25 M
                # scores a layer, and there are four of them per pass.
                bias = torch.zeros_like(pad, dtype=qh.dtype).masked_fill(
                    pad, _MASK_SCORE
                )
            else:
                bias = bias.masked_fill(pad, _MASK_SCORE)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias)
        # ``heads * head_dim``, not the input width: pruning drops heads and
        # leaves ``head_dim`` alone, so the two stop being equal.
        width = self.heads * self.head_dim
        return self.out_proj(out.transpose(1, 2).reshape(b, n, width)), None


def unpack_attention(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """@brief Rewrite a checkpoint's packed attention onto the split layout.

    @param state A ``state_dict`` saved when attention was
        ``nn.MultiheadAttention``.
    @return The same weights under ``q_proj``/``k_proj``/``v_proj``. Values are
        copied, not recomputed, so the converted model is numerically the one
        that was trained.
    """
    out: dict[str, Tensor] = {}
    for key, value in state.items():
        if key.endswith("in_proj_weight") or key.endswith("in_proj_bias"):
            stem = key.rsplit("in_proj_", 1)[0]
            kind = "weight" if key.endswith("weight") else "bias"
            third = value.shape[0] // 3
            for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
                out[f"{stem}{name}.{kind}"] = value[i * third : (i + 1) * third]
        else:
            out[key] = value
    return out


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
        self.attn = MultiheadAttention(dim, heads)
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
        self.attn = MultiheadAttention(dim, heads)
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
        self.attn = MultiheadAttention(dim, heads)

    def forward(self, x: Tensor, pad: Tensor) -> Tensor:
        """Returns ``(B, D)``, zero for a row with no unmasked token.

        This is the one attention with no null token in front of it, because
        the volume head is handed the token sets with theirs already stripped,
        and a frame at the edge of a scene can have neither a visible map
        element nor a detection.

        Such a row attends to nothing in particular, and the ``masked_fill``
        below turns that into an explicit zero -- no evidence, no summary.
        Stated in the code rather than left to the softmax, because this graph
        is meant to leave PyTorch for TensorRT.
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
