"""The dataset's contract: shapes, masks, splits, and the one invariant."""

import math

import torch
from torch import Tensor

from mapposeformer import geometry as G
from mapposeformer.data import DataParams, SyntheticDataset, build_world
from mapposeformer.data.classes import NUM_CLASSES, LandmarkClass

#: Frames the invariant is checked on. Spread across scenes rather than
#: consecutive, so a single malformed world cannot pass by being outvoted.
FRAMES = range(0, 2000, 97)


def test_world_contains_every_class():
    """A world missing a class would make the ablation experiment meaningless."""
    world = build_world(0)
    present = {int(c) for c in world.cls}
    assert present == set(range(NUM_CLASSES))


def test_padding_is_never_a_phantom_landmark():
    """Points past ``npts`` repeat the last real point, never sit at the origin."""
    world = build_world(3)
    for i in range(len(world)):
        n = int(world.npts[i])
        if n < world.pts.shape[1]:
            assert torch.allclose(world.pts[i, n:], world.pts[i, n - 1])


def test_sample_shapes_and_masks():
    p = DataParams()
    s = SyntheticDataset(p, "train")[17]
    sp = p.sample
    assert s["map_pts"].shape == (sp.max_map_elements, sp.points_per_element, 2)
    assert s["det_pts"].shape == (sp.max_det_elements, sp.points_per_element, 2)
    # Every element is either fully valid (a polyline) or valid at one point
    # (a pole) or entirely padding. Anything else means _pack has drifted.
    for mask in (s["map_pmask"], s["det_pmask"]):
        counts = {int(c) for c in mask.sum(-1)}
        assert counts <= {0, 1, sp.points_per_element}
    assert bool(s["det_pmask"].any()), "a frame with no detections at all"


def _distance_to_map(pts: Tensor, map_pts: Tensor, map_pmask: Tensor) -> Tensor:
    """Shortest distance from each point to the map's polyline *segments*.

    Not to its stored points. A map element carries eight samples over a 12 m
    chunk, so consecutive points sit 1.7 m apart and a detection lying exactly
    along a lane line is still up to 0.85 m from the nearest stored sample.
    Point-to-point distance therefore reports the map's sampling pitch -- about
    0.5 m here -- with any real label error buried underneath it. Worse, the
    samples it would measure against include chunk endpoints, the one artefact
    :func:`~mapposeformer.data.world.chunk_for_map` exists to keep out of this
    problem.

    A one-point element (a pole) becomes a zero-length segment, which is the
    point itself, so both geometries go through the same expression.
    """
    a, b = [], []
    for e in range(map_pts.shape[0]):
        q = map_pts[e][map_pmask[e]]
        if q.numel() == 0:
            continue
        a.append(q[:-1] if q.shape[0] > 1 else q)
        b.append(q[1:] if q.shape[0] > 1 else q)
    a, b = torch.cat(a), torch.cat(b)
    d = b - a
    t = (((pts[:, None] - a) * d).sum(-1) / d.square().sum(-1).clamp_min(1e-9)).clamp(0, 1)
    return (pts[:, None] - (a + t[..., None] * d)).norm(dim=-1).min(-1).values


def _inlier_frac(sample: dict[str, Tensor], delta: Tensor, radius: float = 1.0) -> Tensor:
    """Fraction of detected points landing within ``radius`` of the map.

    The statistic and the radius the matching loss uses to decide which
    detections have a counterpart at all, so a label that drifts here drags
    ``matched_frac`` down with it in training -- which is the symptom
    ``docs/TRAINING.md`` tells the reader to look for.
    """
    pts = G.transform_points(delta, sample["det_pts"][sample["det_pmask"]])
    dist = _distance_to_map(pts, sample["map_pts"], sample["map_pmask"])
    return (dist < radius).float().mean()


def test_true_correction_aligns_detections_onto_the_map():
    """The invariant the whole project rests on.

    Applying the ground-truth correction to the detections must land them on
    the map. If it does not, the labels are wrong and every metric downstream
    keeps looking healthy while measuring the wrong thing.

    The bound is loose because the detection model makes it so: a quarter of
    elements are dropped and a Poisson number of clutter elements are added,
    and neither has a map counterpart by construction. Measured mean over these
    frames is 0.86, so 0.75 fails on a broken label rather than on noise.
    """
    ds = SyntheticDataset(DataParams(), "train")
    frac = torch.stack([_inlier_frac(s, s["delta"]) for s in (ds[i] for i in FRAMES)])
    assert frac.mean() > 0.75, f"mean inlier fraction {float(frac.mean()):.3f}"


def test_true_correction_beats_a_wrong_one():
    """...and it must beat nearby wrong corrections, or it is only accidentally
    right. An absolute residual cannot catch a systematic label offset; a
    comparison can, because clutter and dropout penalise every candidate alike.

    The offsets are a full match radius, not half of one: a 0.5 m shift leaves
    most points still inside the 1 m radius, so the fraction barely moves and
    the test would be measuring its own tolerance.

    **Lateral and heading only.** Measured on these frames, a 1 m offset costs
    0.38 of inlier fraction laterally and 3 deg of yaw costs 0.40 -- but 1 m
    *along track* costs between 0.04 and 0.09, because lane geometry runs
    parallel to travel and sliding a hypothesis down the road aligns it with
    the same lines it started on. That is the aliasing ``data/classes.py``
    describes, arriving as a number. Asserting a longitudinal win here would
    encode a claim this evidence does not support; quantifying how weak it is
    belongs to the M1 ablations.
    """
    ds = SyntheticDataset(DataParams(), "train")
    samples = [ds[i] for i in FRAMES]
    true = torch.stack([_inlier_frac(s, s["delta"]) for s in samples]).mean()
    for name, offset in (
        ("lateral +1 m", (0.0, 1.0, 0.0)),
        ("lateral -1 m", (0.0, -1.0, 0.0)),
        ("yaw +3 deg", (0.0, 0.0, math.radians(3.0))),
        ("yaw -3 deg", (0.0, 0.0, -math.radians(3.0))),
    ):
        wrong = torch.stack(
            [_inlier_frac(s, s["delta"] + torch.tensor(offset)) for s in samples]
        ).mean()
        assert true > wrong + 0.15, f"{name}: true {true:.3f} vs wrong {wrong:.3f}"


def test_splits_never_share_geometry():
    p = DataParams()
    a = build_world(SyntheticDataset(p, "train").base + 0)
    b = build_world(SyntheticDataset(p, "val").base + 0)
    assert not torch.allclose(a.trajectory[:10], b.trajectory[:10])


def test_class_ablation_removes_the_class():
    from mapposeformer.data.sample import SampleParams

    keep = (int(LandmarkClass.LANE_DIVIDER), int(LandmarkClass.ROAD_BOUNDARY))
    p = DataParams(sample=SampleParams(keep_classes=keep))
    s = SyntheticDataset(p, "train")[30]
    present = {int(c) for c in s["map_cls"][s["map_pmask"].any(-1)]}
    assert present <= set(keep)
