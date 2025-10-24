"""What each geometry mode is invariant to, asserted rather than described.

The three modes differ in exactly one way -- what a rigid motion of the scene
does to their scores -- and that difference is the thing experiment 4 measures.
A test that only checked shapes would pass for all three while the design
question went unanswered.
"""

import math

import pytest
import torch

from mapposeformer.model.attention import (
    GeometricAttention,
    RotaryFrames,
    pair_geometry,
)


def _frames(n=6, seed=0):
    gen = torch.Generator().manual_seed(seed)
    xy = torch.rand(1, n, 2, generator=gen) * 40 - 20
    yaw = torch.rand(1, n, 1, generator=gen) * 2 * math.pi - math.pi
    return torch.cat([xy, yaw], -1)


def _move(frame, pose):
    """Rigidly move a set of element frames."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    x = pose[0] + c * frame[..., 0] - s * frame[..., 1]
    y = pose[1] + s * frame[..., 0] + c * frame[..., 1]
    return torch.stack([x, y, frame[..., 2] + pose[2]], -1)


def _attend(mode, tok, frame, moved=None, dim=12, heads=2):
    torch.manual_seed(0)
    att = GeometricAttention(dim, heads, mode).eval()
    f = frame if moved is None else moved
    valid = torch.ones(f.shape[:2], dtype=torch.bool)
    oriented = valid.clone()
    return att(tok, f, tok, f, valid, oriented)


@pytest.fixture
def scene():
    frame = _frames()
    torch.manual_seed(1)
    return torch.randn(1, 6, 12), frame


def test_relative_geometry_is_unchanged_by_a_rigid_motion(scene):
    """The input to the score bias, before any weights touch it."""
    _, frame = scene
    before = pair_geometry(frame, frame, torch.ones(1, 6, dtype=torch.bool))
    moved = _move(frame, (13.0, -7.0, 0.9))
    after = pair_geometry(moved, moved, torch.ones(1, 6, dtype=torch.bool))
    assert torch.allclose(before, after, atol=1e-4)


def test_relative_mode_is_fully_se2_equivariant(scene):
    tok, frame = scene
    for pose in ((11.0, -4.0, 0.0), (0.0, 0.0, 1.3), (11.0, -4.0, 1.3)):
        a = _attend("relative", tok, frame)
        b = _attend("relative", tok, frame, _move(frame, pose))
        assert torch.allclose(a, b, atol=1e-4), f"moved by {pose}"


def test_rope_mode_is_translation_equivariant_and_not_rotation_equivariant(
    scene,
):
    """Not a defect.

    The prior pins heading to 3 degrees, so the two sets arrive nearly
    aligned and their shared orientation is evidence. This mode keeps that;
    the fully equivariant one gives it away.
    """
    tok, frame = scene
    slid = _attend("rope", tok, frame, _move(frame, (11.0, -4.0, 0.0)))
    turned = _attend("rope", tok, frame, _move(frame, (0.0, 0.0, 1.3)))
    plain = _attend("rope", tok, frame)
    assert torch.allclose(plain, slid, atol=1e-4)
    assert not torch.allclose(plain, turned, atol=1e-2)


def test_absolute_mode_is_equivariant_to_nothing(scene):
    tok, frame = scene
    plain = _attend("absolute", tok, frame)
    slid = _attend("absolute", tok, frame, _move(frame, (11.0, -4.0, 0.0)))
    assert not torch.allclose(plain, slid, atol=1e-2)


def test_rotary_scores_depend_only_on_the_offset():
    """The identity the rotary encoding exists for, checked directly.

    q . k after rotation must equal what it would be if both elements were
    slid to put the query at the origin.
    """
    rot = RotaryFrames(head_dim=12)
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, 12), torch.randn(1, 1, 1, 12)
    fa, fb = (
        torch.tensor([[[3.0, -1.0, 0.2]]]),
        torch.tensor([[[9.0, 4.0, 1.1]]]),
    )
    here = (rot(q, fa) * rot(k, fb)).sum()
    there = (rot(q, fa - fa) * rot(k, fb - fa)).sum()
    assert torch.allclose(here, there, atol=1e-4)


def test_padding_wins_no_attention(scene):
    """A padded key must not change the answer for any query."""
    tok, frame = scene
    torch.manual_seed(0)
    att = GeometricAttention(12, 2, "relative").eval()
    valid = torch.ones(1, 6, dtype=torch.bool)
    ref = att(tok, frame, tok, frame, valid, valid)

    junk_tok = torch.cat([tok, torch.randn(1, 2, 12) * 50], 1)
    junk_frame = torch.cat([frame, _frames(2, seed=9)], 1)
    padded = torch.cat([valid, torch.zeros(1, 2, dtype=torch.bool)], 1)
    got = att(tok, frame, junk_tok, junk_frame, padded, padded)
    assert torch.allclose(ref, got, atol=1e-5)


def test_a_query_with_no_valid_keys_is_finite(scene):
    """An empty map crop masks a whole softmax row; that must not be a NaN."""
    tok, frame = scene
    att = GeometricAttention(12, 2, "relative").eval()
    none = torch.zeros(1, 6, dtype=torch.bool)
    out = att(tok, frame, tok, frame, none, none)
    assert torch.isfinite(out).all()


def test_gradients_are_finite_in_every_mode(scene):
    tok, frame = scene
    for mode in ("relative", "rope", "absolute"):
        t = tok.clone().requires_grad_(True)
        torch.manual_seed(0)
        att = GeometricAttention(12, 2, mode)
        valid = torch.ones(1, 6, dtype=torch.bool)
        att(t, frame, t, frame, valid, valid).sum().backward()
        assert torch.isfinite(t.grad).all(), mode
        assert t.grad.abs().sum() > 0, mode


def test_head_count_does_not_change_the_parameter_count(scene):
    """Heads are free at fixed width, which is why the count is a real choice.

    q, k and v contract over ``dim`` whether it is split or not, so the only
    thing more heads buy is more attention distributions.
    """
    sizes = {
        h: sum(
            p.numel() for p in GeometricAttention(12, h, "rope").parameters()
        )
        for h in (1, 2)
    }
    assert sizes[1] == sizes[2]


def test_rotary_works_at_the_configuration_actually_used():
    """``head_dim`` 64 is this project's default and is not a multiple of six.

    Three coordinates at two channels each means six per band, so most useful
    widths leave a remainder. It rides through unrotated: demanding a multiple
    of six would make the mode unreachable at exactly the width the project
    runs, and a test at a width that happens to divide would not notice.
    """
    torch.manual_seed(0)
    att = GeometricAttention(128, 2, "rope").eval()
    frame = _frames(5)
    tok = torch.randn(1, 5, 128)
    valid = torch.ones(1, 5, dtype=torch.bool)
    out = att(tok, frame, tok, frame, valid, valid)
    assert out.shape == (1, 5, 128)
    assert torch.isfinite(out).all()
    # Still translation-equivariant with a partial rotation.
    slid = _move(frame, (9.0, -3.0, 0.0))
    assert torch.allclose(
        out, att(tok, slid, tok, slid, valid, valid), atol=1e-4
    )
