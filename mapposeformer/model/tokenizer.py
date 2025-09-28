"""Point sets to tokens.

One token per **point**, not per element. An element-level token would have to
compress a whole polyline into one vector and then ask a pose head to recover
geometry from it; point tokens keep the coordinate that the pose head actually
needs, and let attention discover the element structure it is told about
through the element embedding.

Positions enter as Fourier features rather than raw metres. A linear layer on
raw coordinates has to learn to be sensitive at the 10 cm scale while
accepting inputs at the 50 m scale, and it resolves that conflict by being bad
at the first. Log-spaced sinusoids make both scales available at unit
magnitude, which is the same trick NeRF and every modern positional encoding
uses, for the same reason.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class FourierFeatures(nn.Module):
    """Log-spaced sin/cos of each coordinate.

    @param num_bands Wavelengths per axis.
    @param min_wavelength_m Finest resolvable detail. Below the detection noise
        floor is wasted capacity and invites aliasing.
    @param max_wavelength_m Coarsest band; should exceed the map crop diameter
        so the longest band is monotone across the whole input.
    """

    def __init__(
        self,
        num_bands: int = 16,
        min_wavelength_m: float = 0.4,
        max_wavelength_m: float = 256.0,
    ):
        super().__init__()
        w = torch.logspace(
            math.log10(min_wavelength_m),
            math.log10(max_wavelength_m),
            num_bands,
        )
        self.register_buffer("freq", 2 * math.pi / w, persistent=False)
        self.out_dim = 4 * num_bands

    def forward(self, xy: Tensor) -> Tensor:
        """``(..., 2)`` metres -> ``(..., 4 * num_bands)``."""
        p = xy.unsqueeze(-1) * self.freq  # (..., 2, bands)
        return torch.cat([p.sin(), p.cos()], dim=-1).flatten(-2)


class PointTokenizer(nn.Module):
    """``(B, N, P, 2)`` points -> ``(B, N*P, D)`` tokens.

    Each token carries where the point is, what class and paint style its
    element has, which element it belongs to and where along it the point sits,
    whether that element is a polyline or a lone point, and how much the
    detector trusts it. The element embedding is the load-bearing one: it turns
    an unordered bag of points back into a set of polylines, and without it
    attention cannot tell two lane lines from one.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        num_attrs: int,
        max_elements: int,
        points_per_element: int,
        num_bands: int = 16,
        num_quality: int = 3,
    ):
        super().__init__()
        self.fourier = FourierFeatures(num_bands)
        self.proj = nn.Sequential(
            nn.Linear(self.fourier.out_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.cls_emb = nn.Embedding(num_classes, dim)
        # Paint style. One embedding row, and the only thing that tells the
        # model a detected stripe end is where the paint stops rather than
        # where the view does -- see ``data/classes.py::MarkType``.
        self.attr_emb = nn.Embedding(num_attrs, dim)
        self.elem_emb = nn.Embedding(max_elements, dim)
        self.pos_emb = nn.Embedding(points_per_element, dim)
        # What the sensor says about its own element: a confidence and a
        # reported positional uncertainty. Together these are the one thing
        # about a detection the geometry does not show, and a map element
        # arrives with the survey's version of the same numbers.
        self.quality_proj = nn.Linear(num_quality, dim)
        # A pole is one point repeated into its padding; a lane chunk is eight
        # distinct ones. Same tensor shape, entirely different evidence, so the
        # distinction is given explicitly rather than left to be inferred from
        # coincident coordinates.
        self.shape_emb = nn.Embedding(2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        pts: Tensor,
        pmask: Tensor,
        cls: Tensor,
        attr: Tensor,
        quality: Tensor,
        extra: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Args are ``(B, N, P, 2)``, ``(B, N, P)`` bool, two ``(B, N)`` long.

        @param quality ``(B, N, Q)`` confidence and log-uncertainty per
            element, as ``model.py::_quality`` packs it.
        @param extra ``(B, N, D)`` added per element before the norm, or None.
            The temporal path uses it to say how old a detection is; there is
            nothing else it is for.

        @return ``(tokens (B, N*P, D), pad (B, N*P))`` where ``pad`` is True
            for positions attention must ignore.
        """
        _, n, p, _ = pts.shape
        dev = pts.device
        x = self.proj(self.fourier(pts))
        x = x + self.cls_emb(cls).unsqueeze(2)
        x = x + self.attr_emb(attr).unsqueeze(2)
        x = x + self.elem_emb(torch.arange(n, device=dev)).view(1, n, 1, -1)
        x = x + self.pos_emb(torch.arange(p, device=dev)).view(1, 1, p, -1)
        x = x + self.shape_emb((pmask.sum(-1) > 1).long()).unsqueeze(2)
        x = x + self.quality_proj(quality).unsqueeze(2)
        if extra is not None:
            x = x + extra.unsqueeze(2)
        return self.norm(x).flatten(1, 2), ~pmask.flatten(1, 2)
