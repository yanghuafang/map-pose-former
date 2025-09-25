"""MapPoseFormer: the assembled model.

    detections ─┐                          ┌─ soft assignment ─ Procrustes ─ delta
                ├─ tokenize ─ L x [self | cross] ─┤
    local map ──┘                          └─ attention pool ─ volume ─ logits, cov, trust

Two paths out, on purpose. The **assignment path** produces the pose: it is
geometric, parameter-free at the end, and it says which detected point it used.
The **volume path** produces the uncertainty and the trust score: it is a
distribution over hypotheses, so it can express "somewhere along this stretch of
road" in a way a single pose cannot.

They are trained together but they are not the same estimate, and the code does
not pretend otherwise -- ``delta`` and ``delta_volume`` are both returned, and
the gap between them is a genuinely useful diagnostic. When the assignment is
confident and the volume is a ridge, the disagreement is the ridge's doing, and
that is exactly the frame the downstream filter should be told to distrust.

Everything is in the **anchor frame**: the map is cropped around the prior pose
and expressed there, detections are expressed in the true ego frame, and the
output is the transform between the two. No world coordinate enters the model,
so it is SE(2)-equivariant by construction rather than by training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from mapposeformer import geometry as G
from mapposeformer.data.classes import NUM_CLASSES
from mapposeformer.model.attention import CrossBlock, SelfBlock, prepend_null
from mapposeformer.model.matcher import SoftMatcher
from mapposeformer.model.pose_head import ProcrustesPoseHead, RegressionPoseHead
from mapposeformer.model.tokenizer import PointTokenizer
from mapposeformer.model.volume_head import GridParams, VolumeHead


@dataclass
class ModelParams:
    """Model size and shape. The defaults are the student, not the teacher.

    Small on purpose: the second half of this project is pruning, distillation
    and INT8 deployment, and none of those teach anything on a model that was
    already too big to matter. ``docs/ROADMAP.md`` has the teacher config.
    """

    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_mult: int = 2
    num_bands: int = 16
    max_map_elements: int = 72
    max_det_elements: int = 32
    points_per_element: int = 8
    refine_iters: int = 2
    """Matching passes. One is the single-shot model. Each extra pass costs a
    full trunk forward and no parameters at all."""
    map_sigma_m: float = 0.05
    """Survey tolerance on a map element. Not zero: a map is accurate, not
    exact, and telling the model otherwise would make every map point infinitely
    more trustworthy than every detection."""
    irls_iters: int = 2
    """Robust reweighting passes inside the pose head. Zero is the plain
    weighted least-squares fit that lost to the regression baseline."""
    irls_scale_m: float = 1.0
    min_row_mass: float = 0.05
    """Abstention gate, relative to the frame's strongest match. See
    ``pose_head.py``; zero disables it."""
    grid: GridParams = field(default_factory=GridParams)
    pose_head: str = "procrustes"
    """``procrustes`` or ``regression``; see ``pose_head.py`` for why the
    default is the one with no parameters."""


class MapPoseFormer(nn.Module):
    """Predict the correction from a prior pose to the true pose."""

    def __init__(self, p: ModelParams | None = None):
        super().__init__()
        self.p = p = p or ModelParams()
        self.det_tokens = PointTokenizer(
            p.dim, NUM_CLASSES, p.max_det_elements, p.points_per_element, p.num_bands
        )
        self.map_tokens = PointTokenizer(
            p.dim, NUM_CLASSES, p.max_map_elements, p.points_per_element, p.num_bands
        )
        self.det_null = nn.Parameter(torch.randn(1, 1, p.dim) * 0.02)
        self.map_null = nn.Parameter(torch.randn(1, 1, p.dim) * 0.02)

        mk = lambda cls: nn.ModuleList(  # noqa: E731 - a table reads better than four loops
            [cls(p.dim, p.num_heads, p.ffn_mult) for _ in range(p.num_layers)]
        )
        self.det_self, self.map_self = mk(SelfBlock), mk(SelfBlock)
        self.det_cross, self.map_cross = mk(CrossBlock), mk(CrossBlock)

        self.matcher = SoftMatcher(p.dim)
        self.procrustes = ProcrustesPoseHead()
        self.volume = VolumeHead(p.dim, p.num_heads, p.grid)
        if p.pose_head == "regression":
            deg = torch.pi / 180.0
            self.regression = RegressionPoseHead(
                p.dim,
                (p.grid.extent_x_m, p.grid.extent_y_m, p.grid.extent_yaw_deg * deg),
            )
        elif p.pose_head != "procrustes":
            raise ValueError(f"unknown pose_head {p.pose_head!r}")

    def _trunk(self, batch: dict[str, Tensor], moved: Tensor):
        """Tokenize both sides, attend, and match. One refinement pass.

        @param batch The sample.
        @param moved Detection points in the frame the current estimate
            implies -- the only thing that differs between passes.
        @return ``(assign, scores, tokens, pad)``.
        """
        d, dpad = self.det_tokens(moved, batch["det_pmask"], batch["det_cls"])
        m, mpad = self.map_tokens(batch["map_pts"], batch["map_pmask"], batch["map_cls"])
        d, dpad = prepend_null(d, dpad, self.det_null)
        m, mpad = prepend_null(m, mpad, self.map_null)

        for i in range(self.p.num_layers):
            d = self.det_self[i](d, dpad)
            m = self.map_self[i](m, mpad)
            # Cross-attention reads the *pre-update* other side, so the two
            # directions see the same state.
            d_new = self.det_cross[i](d, m, mpad)
            m = self.map_cross[i](m, d, dpad)
            d = d_new

        d, dpad = d[:, 1:], dpad[:, 1:]
        m, mpad = m[:, 1:], mpad[:, 1:]
        assign, scores = self.matcher(d, m, dpad, mpad)
        return assign, scores, torch.cat([d, m], 1), torch.cat([dpad, mpad], 1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Args: a batch from the dataset. ``prior`` and ``gt`` are for the
        caller and are never read here.

        @return ``delta (B, 3)``, ``deltas (B, I, 3)`` one per refinement pass
            for deep supervision, plus ``delta_volume``, ``logits``, ``cov``,
            ``trust_logit``, ``assign``, ``scores`` and ``mass``.
        """
        det_xy = batch["det_pts"].flatten(1, 2)
        map_xy = batch["map_pts"].flatten(1, 2)
        delta = torch.zeros(det_xy.shape[0], 3, device=det_xy.device, dtype=det_xy.dtype)
        deltas = []
        for _ in range(self.p.refine_iters):
            # The warp is **detached**. It re-anchors the tokenizer's view so
            # matching gets easier; gradient through a chain of warps would
            # make each pass responsible for the ones after it.
            moved = G.transform_points(delta.detach(), det_xy).view_as(batch["det_pts"])
            assign, scores, tokens, pad = self._trunk(batch, moved)
            # Solved on the *original* coordinates, so this is the total
            # correction and not an increment.
            delta, mass = self.procrustes(assign, det_xy, map_xy)
            deltas.append(delta)

        out = self.volume(assign, det_xy, map_xy, tokens, pad)
        out["delta_volume"] = out.pop("delta")
        out["delta_match"] = delta
        if self.p.pose_head == "regression":
            deltas.append(self.regression(out["feat"]))
        out["assign"] = assign
        out["scores"] = scores
        out["mass"] = mass
        out["deltas"] = torch.stack(deltas, dim=1)
        # What the model actually matched, so the losses need not re-derive it
        # from the batch. Identical to the batch's detections today; the
        # temporal path makes them differ.
        out["det_xy"] = det_xy
        out["det_valid"] = batch["det_pmask"].flatten(1)
        out["delta"] = deltas[-1]
        return out
