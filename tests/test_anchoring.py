"""Where the vehicle is in the world must not reach the model.

Both inputs are expressed relative to their own anchor -- the map around the
prior pose, the detections around the true pose -- so the same local geometry
produces byte-identical tensors wherever in the world it sits. That is the
property that stops the model memorising a city instead of learning to match,
and it is a property of the *input construction*, so it can be tested exactly
rather than approximately.

(The network itself is not equivariant: its Fourier features read anchor-frame
coordinates directly. It does not need to be. The anchoring already removes the
only thing there was to memorise.)
"""

import math

import torch

from mapposeformer.data.sample import SampleParams, build_sample
from mapposeformer.data.world import (
    World,
    WorldParams,
    build_world,
    chunk_for_map,
)


def _move_world(world: World, dx: float, dy: float, dyaw: float) -> World:
    c, s = math.cos(dyaw), math.sin(dyaw)
    rot = torch.tensor([[c, -s], [s, c]])
    pts = world.pts @ rot.T + torch.tensor([dx, dy])
    traj = world.trajectory.clone()
    traj[:, :2] = traj[:, :2] @ rot.T + torch.tensor([dx, dy])
    traj[:, 2] = traj[:, 2] + dyaw
    return World(
        pts=pts,
        npts=world.npts,
        cls=world.cls,
        attr=world.attr,
        trajectory=traj,
        params=world.params,
    )


def test_sample_is_unchanged_by_moving_the_whole_world():
    wp = WorldParams()
    world = build_world(11, wp)
    # Chunk first, then move both, rather than chunking each world separately.
    # ``chunk_for_map`` cuts by arclength and so is rigid-invariant in exact
    # arithmetic, but at a 5 km offset float32 resolves to about half a
    # millimetre and the cut points would disagree in the last digits. Moving
    # an already-chunked map keeps the element boundaries identical by
    # construction, which is what leaves this test measuring ``build_sample``.
    chunked = chunk_for_map(world, wp.map_chunk_m, wp.step_m)
    moved, moved_chunked = (
        _move_world(w, dx=5_000.0, dy=-1_200.0, dyaw=1.1)
        for w in (world, chunked)
    )

    sp = SampleParams()
    for frame in (10, 60):
        a = build_sample(world, chunked, frame, 5, sp)
        b = build_sample(moved, moved_chunked, frame, 5, sp)
        for key in ("map_pts", "det_pts", "delta"):
            assert torch.allclose(a[key], b[key], atol=1e-3), key
        for key in (
            "map_cls",
            "det_cls",
            "map_attr",
            "det_attr",
            "map_pmask",
            "det_pmask",
        ):
            assert torch.equal(a[key], b[key]), key
        # The pose itself did move -- otherwise the test would be checking that
        # ``_move_world`` does nothing.
        assert not torch.allclose(a["gt"], b["gt"])
