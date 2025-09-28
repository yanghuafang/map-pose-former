"""MapPoseFormer: the assembled model. Start here.

    map ──────────┐
                  ├─ tokenize ─ attend ─ assignment ─┬─ Procrustes ── delta
    detections ───┘                                  │
      this frame, and the last two                   └─ the same cost,
      warped here by egomotion                       on a grid ─── cov, trust

**One claim, read twice.** The assignment -- which detected point is which map
point -- is the only thing the model asserts. The pose is the minimum of the
alignment error it implies, and the cost volume is that same error evaluated
across a grid instead of at its minimum, so with the plain solve the two agree
to within half a cell and a test says so.

Robustness breaks the tie on purpose: reweighting and the abstention gate make
the pose the minimum of a *different*, downweighted objective, while the
surface is still built from the raw assignment. The gap between them is
therefore a signal rather than an inconsistency -- it is the best available
predictor of a frame whose covariance is about to lie, which is what
``disagree`` reports and ``docs/RESULTS.md`` measures.

Three mechanisms sit around that core. Each has its own file or its own section
in ``docs/ARCHITECTURE.md``; the one-line versions:

* **The past arrives by egomotion**, as more detection tokens rather than as a
  recurrence, because what accumulates is evidence about the pose *error*.
* **Matching runs twice**, the second time on detections moved into the frame
  the first estimate implies. No parameters, much easier correspondence.
* **Nothing is solved on the moved coordinates.** Every pass reads the original
  points, so each yields a total correction and the volume stays anchored.

Everything is in the **anchor frame**, and no world coordinate enters the
model."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from mapposeformer import geometry as G
from mapposeformer.data.classes import NUM_ATTRS, NUM_CLASSES
from mapposeformer.model.attention import CrossBlock, SelfBlock, prepend_null
from mapposeformer.model.matcher import SoftMatcher
from mapposeformer.model.pose_head import (
    ProcrustesPoseHead,
    RegressionPoseHead,
)
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
    """Robust reweighting passes inside the pose head. Zero leaves the
    abstention gate on; the plain weighted least-squares fit that lost to the
    regression baseline needs ``min_row_mass = 0`` as well."""
    irls_scale_m: float = 1.0
    min_row_mass: float = 0.05
    """Abstention gate, relative to the frame's strongest match. See
    ``pose_head.py``; zero disables it."""
    grid: GridParams = field(default_factory=GridParams)
    pose_head: str = "procrustes"
    """``procrustes`` or ``regression``; see ``pose_head.py`` for why the
    default is the one with no parameters, and for the experiment that nearly
    made it the other one."""


def _quality(conf: Tensor, sigma: Tensor) -> Tensor:
    """``(B, N)`` confidence and ``(B, N, 2)`` metres -> ``(B, N, 3)`` features.

    Log for the sigmas: they span an order of magnitude between a near pole and
    a distant lane chunk, and a linear layer fed metres is bad at the small end
    -- which is the end that matters, because those are the elements worth
    weighting up.
    """
    return torch.cat([conf.unsqueeze(-1), sigma.clamp_min(1e-3).log()], dim=-1)


class MapPoseFormer(nn.Module):
    """Predict the correction from a prior pose to the true pose."""

    def __init__(self, p: ModelParams | None = None):
        super().__init__()
        self.p = p = p or ModelParams()
        self.det_tokens = PointTokenizer(
            p.dim,
            NUM_CLASSES,
            NUM_ATTRS,
            p.max_det_elements,
            p.points_per_element,
            p.num_bands,
        )
        self.map_tokens = PointTokenizer(
            p.dim,
            NUM_CLASSES,
            NUM_ATTRS,
            p.max_map_elements,
            p.points_per_element,
            p.num_bands,
        )
        # Confidence 1.0 and the survey tolerance, logged, to match _quality.
        survey = torch.tensor([1.0, p.map_sigma_m, p.map_sigma_m])
        survey[1:] = survey[1:].log()
        self.register_buffer("map_quality", survey, persistent=False)
        self.det_null = nn.Parameter(torch.randn(1, 1, p.dim) * 0.02)
        self.map_null = nn.Parameter(torch.randn(1, 1, p.dim) * 0.02)

        def stack(block):
            """@brief One block per layer. @param block Block class."""
            return nn.ModuleList(
                [
                    block(p.dim, p.num_heads, p.ffn_mult)
                    for _ in range(p.num_layers)
                ]
            )

        self.det_self, self.map_self = stack(SelfBlock), stack(SelfBlock)
        self.det_cross, self.map_cross = stack(CrossBlock), stack(CrossBlock)

        self.matcher = SoftMatcher(p.dim)
        self.procrustes = ProcrustesPoseHead(
            p.irls_iters, p.irls_scale_m, p.min_row_mass
        )
        self.volume = VolumeHead(p.dim, p.num_heads, p.grid)
        if p.pose_head == "regression":
            deg = torch.pi / 180.0
            self.regression = RegressionPoseHead(
                p.dim,
                (
                    p.grid.extent_x_m,
                    p.grid.extent_y_m,
                    p.grid.extent_yaw_deg * deg,
                ),
            )
        elif p.pose_head != "procrustes":
            raise ValueError(f"unknown pose_head {p.pose_head!r}")

    def _detections(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """This frame's detections, packed with their quality features."""
        conf, sigma = batch["det_conf"], batch["det_sigma"]
        return {
            "pts": batch["det_pts"],
            "pmask": batch["det_pmask"],
            "cls": batch["det_cls"],
            "attr": batch["det_attr"],
            "quality": _quality(conf, sigma),
        }

    def _map_quality(self, cls: Tensor, dtype: torch.dtype) -> Tensor:
        """The survey's version of the same three numbers.

        Constant -- see ``map_sigma_m`` for why it is not zero -- so the layer
        learns one offset for the map side and reads a real signal on the
        detection side.
        """
        b, n = cls.shape
        return self.map_quality.to(dtype).view(1, 1, -1).expand(b, n, -1)

    def _trunk(
        self, det: dict[str, Tensor], batch: dict[str, Tensor], moved: Tensor
    ):
        """Tokenize both sides, attend, and match. One refinement pass."""
        d, dpad = self.det_tokens(
            moved,
            det["pmask"],
            det["cls"],
            det["attr"],
            det["quality"],
        )
        m, mpad = self.map_tokens(
            batch["map_pts"],
            batch["map_pmask"],
            batch["map_cls"],
            batch["map_attr"],
            self._map_quality(batch["map_cls"], moved.dtype),
        )
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
        return assign, scores, torch.cat([d, m], 1), torch.cat([dpad, mpad], 1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Args: a batch from the dataset. ``prior`` and ``gt`` are for the
        caller and are never read here.

        @return ``delta (B, 3)`` the correction, ``deltas (B, I, 3)`` one per
            refinement pass for deep supervision, plus ``delta_volume``,
            ``logits``, ``cov``, ``trust_logit``, ``assign``, ``mass``, and the
            ``det_xy``/``det_valid`` the losses need in order to score the
            detections the model actually used rather than the ones the batch
            happened to carry.
        """
        det = self._detections(batch)
        det_xy = det["pts"].flatten(1, 2)
        map_xy = batch["map_pts"].flatten(1, 2)

        delta = torch.zeros(
            det_xy.shape[0], 3, device=det_xy.device, dtype=det_xy.dtype
        )
        deltas = []
        for _ in range(self.p.refine_iters):
            # The warp is **detached**. It re-anchors the tokenizer's view so
            # matching gets easier; letting the gradient run back through a
            # chain of warps would make each pass responsible for the ones
            # after it, which is a much longer path for no extra signal.
            moved = G.transform_points(delta.detach(), det_xy).view_as(
                det["pts"]
            )
            assign, scores, tokens, pad = self._trunk(det, batch, moved)
            # Solved on the *original* coordinates, so this is the total
            # correction and not an increment -- no composition, and the next
            # pass starts from a pose in the same frame as the last.
            delta, mass = self.procrustes(assign, det_xy, map_xy)
            deltas.append(delta)

        out = self.volume(assign, det_xy, map_xy, tokens, pad)
        out["delta_volume"] = out.pop("delta")
        out["delta_match"] = delta
        if self.p.pose_head == "regression":
            # Appended rather than substituted, so ``deltas[:, -1]`` is always
            # the model's answer and the pose loss supervises it whichever head
            # is selected -- with the match passes still supervised behind it,
            # which is what keeps the baseline a *matched* trunk read by a
            # regressor rather than a regressor on its own.
            deltas.append(self.regression(out["feat"]))
        out["assign"] = assign
        out["scores"] = scores
        out["mass"] = mass
        out["deltas"] = torch.stack(deltas, dim=1)
        out["det_xy"] = det_xy
        out["det_valid"] = det["pmask"].flatten(1)
        out["delta"] = deltas[-1]
        return out
