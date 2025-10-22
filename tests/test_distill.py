"""The distillation losses, and what they refuse.

Checked without a teacher checkpoint: these are functions of two output dicts,
so a second forward pass of the same model stands in for one. What matters is
that the divergences are zero when the two agree, positive when they do not,
blind to padded rows, and that a mismatched teacher is rejected rather than
broadcast against.
"""

import pytest
import torch

from mapposeformer.distill import (
    DistillParams,
    assignment_kl,
    check_shapes_agree,
    distill_losses,
    volume_kl,
)
from mapposeformer.model.model import ModelParams


def _assign(b=2, k=6, ll=5, seed=0):
    """A row-stochastic-or-less assignment, shaped like the matcher's."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(b, k, ll, generator=g).softmax(-1)
    return a * torch.rand(b, k, 1, generator=g)  # rows sum to at most one


def test_a_teacher_that_agrees_costs_nothing():
    a = _assign()
    valid = torch.ones(2, 6)
    assert float(assignment_kl(a, a, valid)) == pytest.approx(0.0, abs=1e-6)
    v = torch.randn(2, 40)
    assert float(volume_kl(v, v, 2.0)) == pytest.approx(0.0, abs=1e-6)


def test_disagreement_is_positive_and_finite():
    s, t = _assign(seed=1), _assign(seed=2)
    valid = torch.ones(2, 6)
    kl = assignment_kl(s, t, valid)
    assert torch.isfinite(kl) and float(kl) > 0.0
    kv = volume_kl(torch.randn(2, 40), torch.randn(2, 40), 2.0)
    assert torch.isfinite(kv) and float(kv) > 0.0


def test_the_withheld_mass_is_part_of_the_answer():
    """A row that abstains must not look like one that matched everywhere.

    Both rows below put the same *relative* weight on the same map point; they
    differ only in how much mass they keep back, which is the model declining
    to match. Without the abstain column the divergence is zero and the student
    is free to match everything.
    """
    shy = torch.zeros(1, 1, 3)
    shy[0, 0, 0] = 0.1
    eager = torch.zeros(1, 1, 3)
    eager[0, 0, 0] = 0.9
    valid = torch.ones(1, 1)
    assert float(assignment_kl(shy, eager, valid)) > 0.1


def test_padded_rows_do_not_contribute():
    s, t = _assign(seed=3), _assign(seed=4)
    valid = torch.ones(2, 6)
    valid[:, 3:] = 0.0
    both = assignment_kl(s, t, torch.ones(2, 6))
    kept = assignment_kl(s, t, valid)
    # Blanking the rows the mask drops must leave the masked answer alone.
    s2, t2 = s.clone(), t.clone()
    s2[:, 3:] = 0.0
    t2[:, 3:] = 0.5
    assert float(assignment_kl(s2, t2, valid)) == pytest.approx(
        float(kept), abs=1e-6
    )
    assert float(both) != pytest.approx(float(kept), abs=1e-6)


def test_gradient_reaches_the_student_only():
    s = _assign(seed=5).requires_grad_(True)
    t = _assign(seed=6)
    out = {
        "assign": s,
        "logits": torch.randn(2, 40, requires_grad=True),
        "det_valid": torch.ones(2, 6),
    }
    ref = {"assign": t, "logits": torch.randn(2, 40)}
    total, logs = distill_losses(out, ref, DistillParams())
    total.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert t.grad is None
    assert set(logs) == {"kd", "kd_match", "kd_volume"}


def test_a_teacher_of_the_wrong_shape_is_refused():
    student = ModelParams()
    check_shapes_agree(student, ModelParams(dim=256, num_layers=8))
    with pytest.raises(ValueError, match="max_det_elements"):
        check_shapes_agree(student, ModelParams(max_det_elements=16))


def test_a_config_from_before_a_field_existed_still_loads():
    """A checkpoint outlives the code that wrote it.

    Every checkpoint stores the Config that produced it, which is what makes a
    run reproducible and what makes each stored config a hostage to the next
    field added. Adding `distill` broke every checkpoint saved before it: the
    unpickled object has no such attribute, and `dataclasses.replace` raises on
    a field nobody asked about. Missing fields fill from their defaults, which
    is right by construction -- the run predates the field, so it cannot have
    depended on it.
    """
    import dataclasses

    from mapposeformer.config import Config, upgrade, with_overrides

    old = Config()
    del old.__dict__["distill"]  # a config written before the field existed
    with pytest.raises(AttributeError):
        dataclasses.replace(old)

    upgraded = upgrade(old)
    assert upgraded.distill == DistillParams()
    assert with_overrides(upgraded, {"train": {"lr": 1e-4}}).train.lr == 1e-4
