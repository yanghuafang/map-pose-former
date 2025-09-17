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

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Args: a batch from the dataset. Only the six point-set tensors are
        read -- ``prior`` and ``gt`` are for the caller, never the model.

        Returns:
            ``delta (B, 3)`` the correction, plus ``delta_volume``, ``logits``,
            ``cov``, ``trust_logit``, ``assign`` and ``mass``.
        """
        d, dpad = self.det_tokens(batch["det_pts"], batch["det_pmask"], batch["det_cls"])
        m, mpad = self.map_tokens(batch["map_pts"], batch["map_pmask"], batch["map_cls"])
        d, dpad = prepend_null(d, dpad, self.det_null)
        m, mpad = prepend_null(m, mpad, self.map_null)

        for i in range(self.p.num_layers):
            d = self.det_self[i](d, dpad)
            m = self.map_self[i](m, mpad)
            # Cross-attention reads the *pre-update* other side, so the two
            # directions see the same state. Updating in place would make the
            # map's view of the detections one layer newer than the reverse,
            # which is an asymmetry with no justification behind it.
            d_new = self.det_cross[i](d, m, mpad)
            m = self.map_cross[i](m, d, dpad)
            d = d_new

        d, dpad = d[:, 1:], dpad[:, 1:]
        m, mpad = m[:, 1:], mpad[:, 1:]

        assign, scores = self.matcher(d, m, dpad, mpad)
        det_xy = batch["det_pts"].flatten(1, 2)
        map_xy = batch["map_pts"].flatten(1, 2)
        delta_match, mass = self.procrustes(assign, det_xy, map_xy)

        out = self.volume(
            assign, det_xy, map_xy, torch.cat([d, m], 1), torch.cat([dpad, mpad], 1)
        )
        out["delta_volume"] = out.pop("delta")
        out["assign"] = assign
        out["scores"] = scores
        out["mass"] = mass
        out["delta_match"] = delta_match
        out["delta"] = (
            self.regression(out["feat"])
            if self.p.pose_head == "regression"
            else delta_match
        )
        return out
