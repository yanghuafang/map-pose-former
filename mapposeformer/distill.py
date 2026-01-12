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

A cost volume would be a second, coarser view of the same claim; there is no
volume head here, so the assignment is the whole of it.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mapposeformer.config import DistillParams

_EPS = 1e-9


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
    # The assignment is `elements` and validity is `det_pmask`, at point
    # resolution. There is no volume head, so there is no `volume_kl` term:
    # a KL is fed the tensor it names, never something that resembles it.
    valid = student["det_pmask"].reshape(student["det_pmask"].shape[0], -1)
    km = assignment_kl(student["elements"], teacher["elements"], valid)
    total = p.w_match * km
    return total, {"kd": float(total.detach()), "kd_match": float(km.detach())}


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
    from mapposeformer.checkpoint import build_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt)
    return model.to(device).eval().requires_grad_(False)


def check_shapes_agree(student_cfg, teacher_cfg) -> None:
    """@brief Refuse a teacher the student's assignment cannot be compared to.

    Distillation is a KL between two assignment matrices, so they must be the
    same shape. That shape is ``(B, D*P, M*P)`` and every term comes from the
    **sample** geometry, not from the model: ``max_map_elements`` and its
    companions live on :class:`SampleParams`. Reading them off the *model*
    config instead raises ``AttributeError``, and not until the first
    distillation run -- which is why this check is explicit.

    Deliberately *not* checked: `dim`, `layers`, `heads`. A teacher the same
    shape as its student would teach it nothing -- the whole point is that the
    student is smaller. Only the things that make their outputs comparable
    have to match.

    @param student_cfg The student's :class:`SampleParams`.
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
                "Distillation compares their assignment matrices, and this "
                "changes the shape of them."
            )
