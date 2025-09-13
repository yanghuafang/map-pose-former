"""A procedural road world: the stage-0 dataset.

Real data comes later (see ``docs/ROADMAP.md``). This module exists because a
learned localizer needs one thing before it needs realism: a setting where the
answer is *known* and the evidence available to it can be *controlled*. Turning
poles off and watching along-track error explode is a two-line experiment here
and an impossible one on a public dataset.

The world is deliberately simple and deliberately not a toy. It has curvature,
so heading matters; it has intersections, so along-track evidence is sparse
rather than absent; and every landmark class from ``classes.py`` is present, so
the ablation table can be reproduced end to end.

Everything in this module is in the **world frame**: X east, Y north, yaw
counter-clockwise, metres and radians. Nothing here knows about a vehicle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from mapposeformer.data.classes import LandmarkClass


@dataclass(frozen=True)
class WorldParams:
    """Shape of the generated road. Defaults are a suburban arterial."""

    length_m: float = 600.0
    """Total centreline length. One world is one scene."""
    step_m: float = 2.0
    """Centreline sampling pitch. Also the pitch of every along-road element."""
    num_lanes: int = 4
    """Total lanes, both directions. Dividers sit between them."""
    lane_width_m: float = 3.5
    map_chunk_m: float = 12.0
    """Length of a stored *map* element. Real vector maps are chunked, and one
    600 m polyline would be a single token whose points are mostly out of view.

    Only the map is chunked. The world keeps its along-road geometry as
    continuous lines, and detections are cut out of those by the camera
    frustum -- see :func:`chunk_for_map` for why that distinction is the
    difference between an experiment and a leak."""
    min_radius_m: float = 120.0
    """Tightest curve. Below this the fixed detection range starts to see the
    road leave the field of view entirely, which is a different experiment."""
    straight_frac: float = 0.45
    """Fraction of segments that are straight."""
    pole_spacing_m: float = 22.0
    sign_spacing_m: float = 70.0
    intersection_spacing_m: float = 110.0
    """Mean gap between intersections. This is the single most consequential
    number in the file: stop lines and crossings are the only along-track
    evidence a nuScenes-style map carries, so this sets how often the
    longitudinal degree of freedom is observable at all."""
    lateral_jitter_m: float = 0.35
    """Placement noise on roadside furniture, so poles are not on a perfect
    line the model could regress from lateral offset alone."""


@dataclass
class World:
    """One generated scene, as continuous ground-truth geometry.

    Along-road elements run the full length of the road. This is the *world*,
    not a map: a map is a chunked, stored artifact derived from it by
    :func:`chunk_for_map`, and detections are cut from the world directly by
    the camera frustum.

    Attributes:
        pts: ``(E, P, 2)`` element points, world frame, metres. Padded.
        npts: ``(E,)`` valid point count per element; the rest of ``pts`` is
            repeated tail padding, never garbage, so a bug that ignores this
            mask degrades quietly rather than exploding -- which is why tests
            check it explicitly.
        cls: ``(E,)`` :class:`LandmarkClass` value per element.
        trajectory: ``(T, 3)`` ground-truth vehicle poses along the road.
    """

    pts: Tensor
    npts: Tensor
    cls: Tensor
    trajectory: Tensor

    def __len__(self) -> int:
        return int(self.cls.shape[0])


def _centreline(p: WorldParams, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    """Integrate a piecewise-constant-curvature path.

    Returns ``(points (N, 2), headings (N,))``. Arcs rather than a curvature
    random walk: a random walk produces wobble at the sampling pitch, and a
    road that wobbles every two metres gives the model a texture to latch onto
    that no real road has.
    """
    n = int(p.length_m / p.step_m) + 1
    xy = torch.zeros(n, 2)
    head = torch.zeros(n)
    i, yaw = 0, 0.0
    while i < n - 1:
        seg = int(torch.randint(15, 60, (1,), generator=gen).item())
        if torch.rand(1, generator=gen).item() < p.straight_frac:
            kappa = 0.0
        else:
            sign = 1.0 if torch.rand(1, generator=gen).item() < 0.5 else -1.0
            radius = p.min_radius_m * (1.0 + 3.0 * torch.rand(1, generator=gen).item())
            kappa = sign / radius
        for _ in range(min(seg, n - 1 - i)):
            xy[i + 1, 0] = xy[i, 0] + p.step_m * math.cos(yaw)
            xy[i + 1, 1] = xy[i, 1] + p.step_m * math.sin(yaw)
            yaw += kappa * p.step_m
            head[i + 1] = yaw
            i += 1
    head[0] = head[1] if n > 1 else 0.0
    return xy, head


def _offset(xy: Tensor, head: Tensor, lateral_m: float) -> Tensor:
    """Shift a path sideways by ``lateral_m`` (positive = left of travel)."""
    normal = torch.stack([-torch.sin(head), torch.cos(head)], dim=-1)
    return xy + lateral_m * normal


def _milestones(length_m: float, mean_gap_m: float, gen: torch.Generator) -> list[float]:
    """Arclength positions of Poisson-spaced roadside features."""
    out, s = [], float(torch.rand(1, generator=gen).item()) * mean_gap_m
    while s < length_m:
        out.append(s)
        gap = -mean_gap_m * math.log(max(1e-6, float(torch.rand(1, generator=gen).item())))
        s += max(0.25 * mean_gap_m, gap)
    return out


def build_world(seed: int, p: WorldParams | None = None) -> World:
    """Generate one scene, reproducibly.

    Args:
        seed: The scene identity. Train/val/test are disjoint seed ranges, so
            no geometry is ever shared across splits -- the synthetic analogue
            of the geographic split the real-data milestone needs.
        p: Road shape; defaults to :class:`WorldParams`.
    """
    p = p or WorldParams()
    gen = torch.Generator().manual_seed(seed)
    xy, head = _centreline(p, gen)
    half_road = 0.5 * p.num_lanes * p.lane_width_m

    elements: list[tuple[int, Tensor]] = []

    # --- Along-road geometry: dividers between lanes, boundaries at the edges.
    # Continuous, full length. Chunking is a property of the stored map, not of
    # the road.
    for k in range(1, p.num_lanes):
        lat = half_road - k * p.lane_width_m
        elements.append((LandmarkClass.LANE_DIVIDER, _offset(xy, head, lat)))
    for lat in (half_road, -half_road):
        elements.append((LandmarkClass.ROAD_BOUNDARY, _offset(xy, head, lat)))

    # --- Roadside furniture: single points, the strongest along-track evidence.
    def _at(s: float) -> tuple[Tensor, Tensor]:
        i = min(int(s / p.step_m), xy.shape[0] - 1)
        return xy[i], head[i]

    for cls, spacing, extra in (
        (LandmarkClass.POLE, p.pole_spacing_m, 2.0),
        (LandmarkClass.TRAFFIC_SIGN, p.sign_spacing_m, 1.2),
    ):
        for s in _milestones(p.length_m, spacing, gen):
            c, h = _at(s)
            side = 1.0 if torch.rand(1, generator=gen).item() < 0.5 else -1.0
            jit = p.lateral_jitter_m * float(torch.randn(1, generator=gen).item())
            n = torch.stack([-torch.sin(h), torch.cos(h)])
            elements.append((cls, (c + side * (half_road + extra + jit) * n).view(1, 2)))

    # --- Intersections: the only features perpendicular to travel.
    for s in _milestones(p.length_m, p.intersection_spacing_m, gen):
        c, h = _at(s)
        n = torch.stack([-torch.sin(h), torch.cos(h)])
        fwd = torch.stack([torch.cos(h), torch.sin(h)])
        # Stop line spans the near half of the road; crossing spans all of it,
        # a few metres further on. That asymmetry is real and it is also useful:
        # the two give slightly different lateral evidence at the same station.
        span = torch.linspace(-half_road, 0.0, 5).unsqueeze(-1)
        elements.append((LandmarkClass.STOP_LINE, c + span * n))
        span = torch.linspace(-half_road, half_road, 9).unsqueeze(-1)
        elements.append((LandmarkClass.PED_CROSSING, c + 4.0 * fwd + span * n))

    max_pts = max(e.shape[0] for _, e in elements)
    pts = torch.zeros(len(elements), max_pts, 2)
    npts = torch.zeros(len(elements), dtype=torch.long)
    cls = torch.zeros(len(elements), dtype=torch.long)
    for i, (c, e) in enumerate(elements):
        pts[i, : e.shape[0]] = e
        pts[i, e.shape[0] :] = e[-1]  # tail padding repeats the last point
        npts[i] = e.shape[0]
        cls[i] = int(c)

    # --- Ground truth trajectory: drive the middle of the leftmost forward lane.
    lane_centre = 0.5 * p.lane_width_m
    path = _offset(xy, head, lane_centre)
    trajectory = torch.cat([path, head.unsqueeze(-1)], dim=-1)
    return World(pts=pts, npts=npts, cls=cls, trajectory=trajectory)


def chunk_for_map(world: World, chunk_m: float, step_m: float) -> World:
    """Cut the world's continuous elements into stored map elements.

    **This is the one asymmetry the synthetic data depends on**, and getting it
    wrong silently destroys the ablation this project exists to run.

    A map is chunked at fixed positions decided when it was surveyed. A detector
    sees whatever the frustum contains, cut at a range, not at a map boundary.
    If both sides were chunked the same way, the two point sets would share
    element *endpoints* at fixed world positions -- and a chunk endpoint is a
    perfect along-track landmark. A model given only lane geometry would then
    localize along the road beautifully, from an artefact of how the polylines
    happened to be cut, and the ablation would report that lane lines constrain
    along-track position. They do not.

    So: the map is chunked here, once per scene; detections are cut out of the
    continuous world by the frustum, whose ends sit at a fixed *range* and
    therefore carry no information about position along the road.
    """
    chunk_pts = max(2, int(chunk_m / step_m) + 1)
    elements: list[tuple[int, Tensor]] = []
    for i in range(len(world)):
        n = int(world.npts[i])
        line, c = world.pts[i, :n], int(world.cls[i])
        if n < chunk_pts:
            elements.append((c, line))
            continue
        for s in range(0, n - 1, chunk_pts - 1):
            piece = line[s : s + chunk_pts]
            if piece.shape[0] >= 2:
                elements.append((c, piece))

    max_pts = max(e.shape[0] for _, e in elements)
    pts = torch.zeros(len(elements), max_pts, 2)
    npts = torch.zeros(len(elements), dtype=torch.long)
    cls = torch.zeros(len(elements), dtype=torch.long)
    for i, (c, e) in enumerate(elements):
        pts[i, : e.shape[0]] = e
        pts[i, e.shape[0] :] = e[-1]
        npts[i] = e.shape[0]
        cls[i] = c
    return World(pts=pts, npts=npts, cls=cls, trajectory=world.trajectory)
