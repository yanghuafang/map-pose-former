"""The element encoder, and the invariance it is supposed to have by design.

The tokens claim to be unchanged by a rigid motion of the whole scene. That is
not a soft property to be encouraged by augmentation -- it is either true of
the arithmetic or it is not, so it is tested as an identity.
"""

import math

import torch

from mapposeformer import geometry as G
from mapposeformer.data.classes import LandmarkClass, MarkType
from mapposeformer.model.encoder import (
    ElementEncoder,
    element_frames,
    embeddings,
)


def _scene():
    """Two polylines and one pole, with the pole padded out to width 4."""
    pts = torch.zeros(1, 3, 4, 2)
    pmask = torch.zeros(1, 3, 4, dtype=torch.bool)
    # A lane divider running along +x at y = 0.
    pts[0, 0] = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    pmask[0, 0] = True
    # A road boundary running along +y at x = 5.
    pts[0, 1] = torch.tensor([[5.0, 0.0], [5.0, 2.0], [5.0, 4.0], [5.0, 6.0]])
    pmask[0, 1] = True
    # A pole: one point, no shape, no direction.
    pts[0, 2, 0] = torch.tensor([3.0, -8.0])
    pmask[0, 2, 0] = True

    cls = torch.tensor(
        [
            [
                LandmarkClass.LANE_DIVIDER,
                LandmarkClass.ROAD_BOUNDARY,
                LandmarkClass.POLE,
            ]
        ]
    )
    attr = torch.tensor([[MarkType.SOLID, MarkType.SOLID, MarkType.NONE]])
    return pts, pmask, cls, attr


def _encoder(dim=16):
    torch.manual_seed(0)
    enc = ElementEncoder(dim, *embeddings(dim))
    enc.eval()
    return enc


def test_the_frame_is_the_centroid_and_the_direction_of_travel():
    pts, pmask, _, _ = _scene()
    frame, oriented, valid = element_frames(pts, pmask)
    assert torch.allclose(frame[0, 0], torch.tensor([3.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(
        frame[0, 1], torch.tensor([5.0, 3.0, math.pi / 2]), atol=1e-6
    )
    assert torch.allclose(frame[0, 2, :2], torch.tensor([3.0, -8.0]), atol=1e-6)
    assert oriented.tolist() == [[True, True, False]]
    assert valid.tolist() == [[True, True, True]]


def test_a_rigid_motion_of_the_scene_leaves_every_token_unchanged():
    """The equivariance claim, as an identity rather than a tendency.

    If this fails the matcher is not SE(2)-equivariant by construction, and
    every argument in the roadmap that rests on that is void.
    """
    pts, pmask, cls, attr = _scene()
    enc = _encoder()
    before, _, _, _ = enc(pts, pmask, cls, attr)

    move = torch.tensor([[17.0, -23.0, 1.1]])
    moved = G.transform_points(move, pts.reshape(1, -1, 2)).reshape(pts.shape)
    after, _, _, _ = enc(moved, pmask, cls, attr)

    assert torch.allclose(before, after, atol=1e-5), (
        (before - after).abs().max()
    )


def test_the_frames_move_with_the_scene():
    """Tokens stay put, frames do not -- that is the split the design needs."""
    pts, pmask, _, _ = _scene()
    move = torch.tensor([[17.0, -23.0, 1.1]])
    moved = G.transform_points(move, pts.reshape(1, -1, 2)).reshape(pts.shape)

    frame, _, _ = element_frames(pts, pmask)
    moved_frame, _, _ = element_frames(moved, pmask)

    want = G.compose(move.expand(3, -1), frame[0])
    assert torch.allclose(moved_frame[0, :, :2], want[:, :2], atol=1e-5)
    # Yaw only where it means something; the pole's is not a heading.
    assert torch.allclose(
        G.wrap_angle(moved_frame[0, :2, 2] - want[:2, 2]),
        torch.zeros(2),
        atol=1e-5,
    )


def test_a_padded_slot_produces_no_token_and_no_nan():
    """An empty element pools over nothing, and -inf is a NaN waiting."""
    pts, pmask, cls, attr = _scene()
    pmask[0, 2] = False  # the pole's only point goes away
    token, _, _, valid = _encoder()(pts, pmask, cls, attr)
    assert torch.isfinite(token).all()
    assert torch.equal(token[0, 2], torch.zeros(16))
    assert valid.tolist() == [[True, True, False]]


def test_the_token_sees_shape_and_not_position():
    """Two identical polylines in different places must encode identically."""
    pts, pmask, cls, attr = _scene()
    pts[0, 1] = pts[0, 0] + torch.tensor([40.0, 11.0])  # same shape, moved
    cls[0, 1] = cls[0, 0]
    attr[0, 1] = attr[0, 0]
    token, _, _, _ = _encoder()(pts, pmask, cls, attr)
    assert torch.allclose(token[0, 0], token[0, 1], atol=1e-5)


def test_a_shorter_element_encodes_differently():
    """Invariance must not have gone so far as to discard the shape too."""
    pts, pmask, cls, attr = _scene()
    enc = _encoder()
    long_token, _, _, _ = enc(pts, pmask, cls, attr)
    pmask[0, 0, 3] = False  # a 6 m chunk rather than a 12 m one
    short_token, _, _, _ = enc(pts, pmask, cls, attr)
    assert not torch.allclose(long_token[0, 0], short_token[0, 0], atol=1e-3)


def test_gradients_reach_the_points():
    """Three ways this module can produce a NaN only in the backward pass.

    A pole has no direction, so its ``atan2`` argument is (0, 0); its one
    point sits on its own centroid, so ``d|q|/dq`` is 0/0; and an empty slot
    maxes over nothing. All three are finite going forward and NaN coming
    back, which is the kind of bug that surfaces as a loss that goes to NaN on
    step 400 with nothing to point at.
    """
    pts, pmask, cls, attr = _scene()
    pmask[0, 1] = False  # and one element with nothing in it at all
    pts.requires_grad_(True)
    token, _, _, _ = _encoder()(pts, pmask, cls, attr)
    token.sum().backward()
    assert pts.grad is not None and torch.isfinite(pts.grad).all()
    assert pts.grad.abs().sum() > 0
