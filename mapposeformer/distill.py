"""Distillation: supervise the student from the teacher's distributions.

Kept apart from ``losses.py`` because the source of truth is different. The
losses there come from geometry -- apply the true correction and see which map
point a detection lands on -- and are as right as the labels. These come from a
larger model that is also wrong, sometimes confidently, and the weight on them
says how much of that to inherit.

**What is distilled is the assignment, not the pose.** The pose is three
numbers the ground truth already supplies exactly, so a teacher adds nothing to
it. The assignment is 768 x 576 soft correspondences, and the teacher's opinion
about which map point a detection *might* be -- including the mass it withholds
-- is the part that carries information the labels do not: a label says one map
point is correct, the teacher says which of the wrong ones were plausible.

The cost surface follows for the same reason. It is one distribution over 4199
hypotheses, computed from the assignment, so distilling it is a second, coarser
view of the same claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

_EPS = 1e-9


@dataclass(frozen=True)
class DistillParams:
    """Weights on the teacher's two distributions, and the softening."""

    teacher: str = ""
    """Checkpoint to distil from. Empty is the ordinary supervised run, which
    is what every config here does unless it says otherwise."""
    w_match: float = 1.0
    """On the assignment. The term that carries, for the reason the match loss
    carries in supervised training: it is the only one with an opinion per
    correspondence rather than per frame."""
    w_volume: float = 0.5
    """On the cost surface. Lower, because the surface is computed *from* the
    assignment -- the two are one claim read twice, and weighting them equally
    counts it twice."""
    temperature: float = 2.0
    """Softens the surface only. The assignment is already a product of two
    softmaxes rather than a logit, so there is nothing there to soften without
    changing what the number means."""

    # These weights are first choices, and the magnitudes may not respect them.
    # On an untrained teacher the assignment KL is ~1e-3 against a volume KL of
    # ~10, because both models withhold most of their mass and agree about
    # doing so, and a divergence between two near-empty rows is near zero. A
    # trained teacher should peak, but "should" is not a measurement -- the
    # gradient split reaching the trunk is what settles it, the same way
    # docs/TRAINING.md settles it for the supervised terms.


def assignment_kl(student: Tensor, teacher: Tensor, valid: Tensor) -> Tensor:
    """@brief KL from the teacher's soft correspondence to the student's.

    Each row is a distribution over the map points *and one more outcome*: the
    mass an assignment row withholds is the model declining to match, which is
    a real answer and the one a detection with no counterpart should give. So
    the row is completed with ``1 - rowsum`` and the divergence taken over
    ``L + 1``. Dropping that column would score a student that matches
    everything the same as one that abstains correctly.

    @param student ``(B, K, L)`` the student's assignment, rows summing to at
        most one.
    @param teacher ``(B, K, L)`` the teacher's, same shape.
    @param valid ``(B, K)`` bool, True where the detection point is real.
    @return Scalar, the mean over valid rows.
    """

    def _complete(a: Tensor) -> Tensor:
        abstain = (1.0 - a.sum(-1, keepdim=True)).clamp_min(0.0)
        return torch.cat([a, abstain], dim=-1).clamp_min(_EPS)

    p, q = _complete(teacher), _complete(student)
    kl = (p * (p.log() - q.log())).sum(-1)
    kl = kl * valid
    return kl.sum() / valid.sum().clamp_min(1.0)


def volume_kl(student: Tensor, teacher: Tensor, temperature: float) -> Tensor:
    """@brief KL between the two cost surfaces, softened.

    Scaled by ``T**2`` so the gradient magnitude does not change with the
    temperature, which is Hinton's correction and the reason a temperature can
    be tuned without retuning the weight beside it.

    @param student ``(B, G)`` logits over the hypothesis grid.
    @param teacher ``(B, G)`` likewise.
    @param temperature Softening; 1.0 leaves the surfaces as they are.
    @return Scalar.
    """
    t = max(temperature, _EPS)
    p = F.softmax(teacher / t, dim=-1)
    q = F.log_softmax(student / t, dim=-1)
    return (p * (p.clamp_min(_EPS).log() - q)).sum(-1).mean() * t * t


def distill_losses(
    student: dict[str, Tensor],
    teacher: dict[str, Tensor],
    p: DistillParams,
) -> tuple[Tensor, dict[str, float]]:
    """@brief The teacher's contribution to the total loss.

    @param student One forward pass of the model being trained.
    @param teacher The same batch through the frozen teacher. Its tensors must
        already be detached; :func:`load_teacher` returns a model that cannot
        produce anything else.
    @param p Weights and temperature.
    @return The weighted total, and scalars for logging.
    """
    km = assignment_kl(
        student["assign"], teacher["assign"], student["det_valid"]
    )
    kv = volume_kl(student["logits"], teacher["logits"], p.temperature)
    total = p.w_match * km + p.w_volume * kv
    return total, {
        "kd": float(total.detach()),
        "kd_match": float(km.detach()),
        "kd_volume": float(kv.detach()),
    }


def load_teacher(path: str, device: str) -> torch.nn.Module:
    """@brief The trained teacher, frozen, in eval mode.

    Every checkpoint stores the config that produced it, so the teacher's
    architecture is read back rather than guessed at -- and the shapes that
    have to agree are checked here rather than surfacing as a broadcast a
    hundred lines later. Distillation only requires the two models to *agree on
    their outputs*; width, depth and head count are free, and being free is the
    whole point.

    @param path Checkpoint written by the trainer.
    @param device Where to put it.
    @return The model, in eval mode with gradients off.
    @throws ValueError If teacher and student disagree on any output shape.
    """
    from mapposeformer.config import upgrade
    from mapposeformer.model.model import MapPoseFormer

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MapPoseFormer(upgrade(ckpt["config"]).model)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval().requires_grad_(False)


def check_shapes_agree(student_cfg, teacher_cfg) -> None:
    """@brief Refuse a teacher whose outputs the student cannot be compared to.

    @param student_cfg The student's ``ModelParams``.
    @param teacher_cfg The teacher's.
    @throws ValueError On the first field that differs.
    """
    fields = (
        "max_map_elements",
        "max_det_elements",
        "points_per_element",
        "history",
    )
    for f in fields:
        a, b = getattr(student_cfg, f), getattr(teacher_cfg, f)
        if a != b:
            raise ValueError(
                f"teacher and student disagree on {f}: {b} against {a}. "
                "Distillation compares their assignment matrices, which this "
                "changes the shape of."
            )
    if student_cfg.grid != teacher_cfg.grid:
        raise ValueError(
            "teacher and student disagree on the hypothesis grid, so their "
            "cost surfaces are not the same distribution"
        )
