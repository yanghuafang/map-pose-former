"""The matcher, and the geometric chain it sits in the middle of.

The element stage is learned, so an untrained one matches nothing in
particular and there is little to assert about it beyond shape and masking.
The point stage is not learned, and the chain encoder -> points -> solve has a
right answer: hand it the true element correspondence and it must return the
pose that produced the scene.
"""

import math

import torch

from mapposeformer import geometry as G
from mapposeformer.data.classes import LandmarkClass, MarkType
from mapposeformer.model.encoder import (
    ElementEncoder,
    element_frames,
    embeddings,
    local_points,
)
from mapposeformer.model.matcher import Matcher
from mapposeformer.solve import solve_pose

DIM, ELEMS, PTS = 12, 4, 4


def _map(seed=0):
    """Four polylines, spread out and pointing different ways."""
    gen = torch.Generator().manual_seed(seed)
    base = torch.linspace(0, 6, PTS)[None, :, None] * torch.tensor([[1.0, 0.0]])
    pts, centres = [], torch.rand(ELEMS, 2, generator=gen) * 40 - 20
    for i in range(ELEMS):
        a = i * math.pi / ELEMS
        rot = torch.tensor(
            [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
        )
        pts.append(base[0] @ rot.T + centres[i])
    pts = torch.stack(pts).unsqueeze(0)
    pmask = torch.ones(1, ELEMS, PTS, dtype=torch.bool)
    cls = torch.full((1, ELEMS), int(LandmarkClass.LANE_DIVIDER))
    attr = torch.full((1, ELEMS), int(MarkType.SOLID))
    return pts, pmask, cls, attr


def _encoders():
    torch.manual_seed(0)
    cls_e, attr_e = embeddings(DIM)
    enc = ElementEncoder(DIM, cls_e, attr_e).eval()
    return enc


def test_the_true_element_match_recovers_the_pose_that_made_the_scene():
    """Encoder, point matching and solve, end to end, with a right answer.

    Nothing here is trained. If this fails, the geometry is wrong somewhere in
    the chain and no amount of learning upstream would rescue it.
    """
    enc, matcher = _encoders(), Matcher(DIM, layers=1, heads=2)
    map_pts, map_pmask, cls, attr = _map()

    for xy_yaw in ([0.0, 0.0, 0.0], [1.5, -0.6, 0.03], [-3.0, 1.2, -0.04]):
        delta = torch.tensor([xy_yaw])
        # Detections sit where applying `delta` lands them on the map.
        det_pts = G.transform_points(
            G.inverse(delta), map_pts.reshape(1, -1, 2)
        ).reshape(map_pts.shape)

        _, map_frame, _, _ = enc(map_pts, map_pmask, cls, attr)
        _, det_frame, _, _ = enc(det_pts, map_pmask, cls, attr)

        truth = torch.eye(ELEMS).unsqueeze(0)
        assign = matcher.points(
            truth, det_pts, map_pmask, det_frame, map_pts, map_pmask, map_frame
        )
        got, mass = solve_pose(
            det_pts.reshape(1, -1, 2), map_pts.reshape(1, -1, 2), assign
        )
        assert mass.item() > 0
        assert torch.allclose(got, delta, atol=1e-4), f"{got} != {delta}"


def test_the_assignment_is_non_negative_and_bounded_by_matchability():
    map_pts, map_pmask, cls, attr = _map()
    enc = _encoders()
    m = enc(map_pts, map_pmask, cls, attr)
    d = enc(map_pts, map_pmask, cls, attr)
    matcher = Matcher(DIM, layers=1, heads=2)
    assign, _, sig_d, *_ = matcher.elements(m, d)
    assert (assign >= 0).all()
    # A dual softmax is at most 1 per entry, so matchability is the ceiling.
    assert (assign.sum(2) <= sig_d + 1e-5).all()


def test_padding_is_assigned_nothing():
    map_pts, map_pmask, cls, attr = _map()
    map_pmask[0, 3] = False  # one element becomes an empty slot
    enc = _encoders()
    e = enc(map_pts, map_pmask, cls, attr)
    assign, *_ = Matcher(DIM, layers=1, heads=2).elements(e, e)
    assert torch.allclose(assign[0, 3], torch.zeros(ELEMS), atol=1e-6)
    assert torch.allclose(assign[0, :, 3], torch.zeros(ELEMS), atol=1e-6)


def test_the_element_assignment_is_equivariant_under_a_rigid_motion():
    """What the relative mode buys: the match ignores where the scene is."""
    map_pts, map_pmask, cls, attr = _map()
    enc, matcher = _encoders(), Matcher(DIM, layers=1, heads=2, mode="relative")
    matcher.eval()

    def run(pts):
        e = enc(pts, map_pmask, cls, attr)
        return matcher.elements(e, e)[0]

    move = torch.tensor([[25.0, -14.0, 0.7]])
    moved = G.transform_points(move, map_pts.reshape(1, -1, 2)).reshape(
        map_pts.shape
    )
    assert torch.allclose(run(map_pts), run(moved), atol=1e-5)


def test_gradients_reach_the_tokens_through_both_stages():
    map_pts, map_pmask, cls, attr = _map()
    enc, matcher = _encoders(), Matcher(DIM, layers=1, heads=2)
    pts = map_pts.clone().requires_grad_(True)
    e = enc(pts, map_pmask, cls, attr)
    assign, *_ = matcher.elements(e, e)
    full = matcher.points(
        assign, pts, map_pmask, e[1], map_pts, map_pmask, e[1]
    )
    full.sum().backward()
    assert torch.isfinite(pts.grad).all()
    assert pts.grad.abs().sum() > 0


def test_each_point_can_pick_its_own_map_element():
    """The composition defect, and whether the per-point arm removes it.

    A detection spanning two map chunks has points on both. With one
    assignment per detection every point is told the same thing, so a point
    eight metres past the end of a chunk still sends mass there -- measured at
    an even 50/50 split, which drags the pose toward the chunks' common
    centroid. Per point, the answer can differ along the detection.

    The head is untrained here, so what is asserted is that the *shape* of the
    answer is now expressible, not that it is correct.
    """
    torch.manual_seed(0)
    matcher = Matcher(DIM, layers=1, heads=2, per_point=True).eval()

    map_pts = torch.zeros(1, 2, 4, 2)
    map_pts[0, 0] = torch.tensor([[0.0, 0], [2, 0], [4, 0], [6, 0]])
    map_pts[0, 1] = torch.tensor([[8.0, 0], [10, 0], [12, 0], [14, 0]])
    det = torch.tensor([[[[0.0, 0], [4, 0], [10, 0], [14, 0]]]])
    det_pmask = torch.ones(1, 1, 4, dtype=torch.bool)

    det_frame, _, _ = element_frames(det, det_pmask)
    local = local_points(det, det_pmask, det_frame)[..., :2]
    torch.manual_seed(0)
    map_tok = torch.randn(1, 2, DIM)
    det_tok = torch.randn(1, 1, DIM)
    ones = torch.ones(1, 1)

    per_point = matcher.point_elements(
        map_tok,
        det_tok,
        local,
        det_pmask,
        torch.ones(1, 2, dtype=torch.bool),
        ones,
        torch.ones(1, 2),
    )
    assert per_point.shape == (1, 1, 4, 2)
    assert (per_point >= 0).all()
    # The four points are no longer forced to agree: two points at opposite
    # ends of the element ask different questions of the map.
    spread = per_point[0, 0]
    assert not torch.allclose(spread[0], spread[-1], atol=1e-6), (
        "every point still gets the same answer; nothing was gained"
    )


def test_a_per_point_assignment_reaches_the_solve():
    """The (B, D, P, M) shape has to survive into the point-level assignment."""
    map_pts, map_pmask, cls, attr = _map()
    enc = _encoders()
    e = enc(map_pts, map_pmask, cls, attr)
    matcher = Matcher(DIM, layers=1, heads=2, per_point=True)
    per_point = torch.rand(1, ELEMS, PTS, ELEMS)
    full = matcher.points(
        per_point, map_pts, map_pmask, e[1], map_pts, map_pmask, e[1]
    )
    assert full.shape == (1, ELEMS * PTS, ELEMS * PTS)
    assert torch.isfinite(full).all()


def test_padding_does_not_dilute_the_per_point_dual_softmax():
    """How much padding sits beside a detection must not change its mass.

    The dual softmax normalises once over map elements and once over
    detections. Masking only the map axis leaves the detection pass
    normalising over padded slots, so a real detection receives a fraction of
    what it should -- and the fraction depends on how full the frame happens
    to be, which is invisible in any average.

    Real detections competing with each other *should* change each other's
    share; that is what the column softmax is for. Padding should not.
    """
    torch.manual_seed(0)
    matcher = Matcher(DIM, layers=1, heads=2, per_point=True).eval()
    P, M, N = 3, 5, 2  # points, map elements, real detections

    torch.manual_seed(1)
    real_tok = torch.randn(1, N, DIM)
    real_local = torch.randn(1, N, P, 2)
    map_tok = torch.randn(1, M, DIM)
    map_valid = torch.ones(1, M, dtype=torch.bool)

    def run(pad):
        d = N + pad
        tok = torch.cat([real_tok, torch.randn(1, pad, DIM)], 1)
        local = torch.cat([real_local, torch.randn(1, pad, P, 2)], 1)
        pmask = torch.zeros(1, d, P, dtype=torch.bool)
        pmask[0, :N] = True
        return matcher.point_elements(
            map_tok,
            tok,
            local,
            pmask,
            map_valid,
            torch.ones(1, d),
            torch.ones(1, M),
        )

    few, many = run(2), run(8)
    assert torch.allclose(few[0, :N], many[0, :N], atol=1e-5), (
        "the amount of padding is changing real detections' mass"
    )
    assert many[0, N:].abs().max() < 1e-6, "padding received mass"
