"""Turn a world and a frame index into one training sample.

This is where the input contract of the whole project is defined, so it is
worth stating plainly. Two point sets go in:

* the **map**, cropped around the *prior* pose and expressed in the prior's
  frame -- what the vehicle believes it should be seeing;
* the **detections**, cropped to the camera's field of view from the *true*
  pose and expressed in the true ego frame -- what it actually sees.

They come from two different representations of the same geometry, and that is
deliberate: the map is chunked at fixed world positions, detections are cut by
the frustum at a fixed range. See :func:`~mapposeformer.data.world.chunk_for_map`
for why sharing the chunking would quietly invalidate the whole experiment.

The answer is the rigid transform between them. Because both sets are
anchored -- neither carries a world coordinate -- the network is SE(2)
equivariant by construction: it cannot memorise where in the map it is, only
how two point sets align. ``tests/test_anchoring.py`` holds that property.

The prior pose leaves this module in the sample only so evaluation can compose
the prediction back into world coordinates. It is never fed to the network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from mapposeformer import geometry as G
from mapposeformer.data.classes import NUM_CLASSES
from mapposeformer.data.world import World


@dataclass(frozen=True)
class PriorParams:
    """How wrong the incoming pose estimate is.

    Anisotropic on purpose. Dead reckoning between map updates drifts fastest
    along the direction of travel, and along-track is also the direction lane
    geometry cannot see -- so the hard axis and the weak evidence are the same
    axis. A symmetric prior would quietly hide that.
    """

    sigma_long_m: float = 1.5
    sigma_lat_m: float = 0.6
    sigma_yaw_deg: float = 1.0
    max_long_m: float = 4.5
    max_lat_m: float = 2.0
    max_yaw_deg: float = 3.0
    """Truncation bounds. They must match the cost-volume grid extent in the
    model config: a target outside the grid has no correct cell, and the
    volume loss would be asking for something unrepresentable."""


@dataclass(frozen=True)
class PerceptionParams:
    """The detection model this project does not train, described statistically.

    Every number here is a knob for an experiment. Realism is not the goal --
    controllability is. See ``docs/DATASET.md``.
    """

    fov_deg: float = 100.0
    max_range_m: float = 50.0
    """Matched to the map query radius. A detection beyond the crop has no map
    counterpart and can only be noise to the matcher."""
    min_range_m: float = 2.0
    element_dropout: float = 0.25
    """Probability a visible element is missed entirely."""
    clutter_mean: float = 1.5
    """Mean number of false-positive elements per frame (Poisson)."""
    point_sigma_m: float = 0.12
    range_sigma_frac: float = 0.004
    """Extra point noise proportional to range: distant detections are worse,
    which is what gives the model a reason to weight near evidence higher."""
    element_bias_sigma_m: float = 0.20
    """Lateral bias applied to a whole element. Independent per-point noise
    averages out over eight points; correlated bias does not, and it is the
    error that actually limits a real detector."""
    class_flip_prob: float = 0.03


@dataclass(frozen=True)
class SampleParams:
    """Cropping and tokenisation. These fix the network's input shape."""

    map_radius_m: float = 50.0
    points_per_element: int = 8
    max_map_elements: int = 72
    """Larger than it looks necessary because map elements are short: at a 12 m
    chunk and a 50 m radius the five along-road lines alone account for about
    forty."""
    max_det_elements: int = 32
    prior: PriorParams = field(default_factory=PriorParams)
    perception: PerceptionParams = field(default_factory=PerceptionParams)
    keep_classes: tuple[int, ...] = tuple(range(NUM_CLASSES))
    """Class ablation. Restricting this is the experiment described in
    ``classes.py``: drop the along-track anchors and watch longitudinal error
    grow while lateral error does not."""


def _resample(pts: Tensor, n: int) -> Tensor:
    """Resample a polyline to exactly ``n`` points, evenly by arclength.

    A fixed point count per element is what makes the input shape static, which
    the ONNX/TensorRT milestone needs. Sampling by arclength rather than by
    index keeps the spacing uniform after an element has been clipped to the
    field of view.
    """
    if pts.shape[0] == 1:
        return pts.expand(n, 2).clone()
    seg = (pts[1:] - pts[:-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])
    if float(cum[-1]) < 1e-6:
        return pts[:1].expand(n, 2).clone()
    t = torch.linspace(0.0, float(cum[-1]), n)
    hi = torch.searchsorted(cum, t).clamp(1, pts.shape[0] - 1)
    lo = hi - 1
    w = ((t - cum[lo]) / (cum[hi] - cum[lo]).clamp_min(1e-6)).unsqueeze(-1)
    return pts[lo] + w * (pts[hi] - pts[lo])


def _pack(
    kept: list[tuple[int, Tensor]], max_elements: int, points_per_element: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a list of ``(class, points)`` into fixed-shape tensors.

    Returns ``(pts (E, P, 2), pmask (E, P), cls (E,))``. A single-point element
    -- a pole -- keeps one valid point and repeats it into the padding, so a
    consumer that forgets the mask sees a degenerate point rather than a zero
    at the origin pretending to be a detection.

    Overflow drops the *farthest* elements rather than an arbitrary tail: the
    cap is a budget, and spending it on what is nearest is the only ordering
    that does not make the truncation a random ablation.
    """
    e, p = max_elements, points_per_element
    kept = sorted(kept, key=lambda cq: float(cq[1].norm(dim=-1).min()))
    pts = torch.zeros(e, p, 2)
    pmask = torch.zeros(e, p, dtype=torch.bool)
    cls = torch.zeros(e, dtype=torch.long)
    for i, (c, q) in enumerate(kept[:e]):
        pts[i] = _resample(q, p)
        pmask[i, :] = q.shape[0] > 1
        pmask[i, 0] = True
        cls[i] = c
    return pts, pmask, cls


def _crop(world: World, pose: Tensor, keep: set[int]) -> list[tuple[int, Tensor]]:
    """World elements in ``pose``'s frame, with their padding stripped."""
    inv = G.inverse(pose)
    local = G.transform_points(inv, world.pts.reshape(-1, 2)).reshape(world.pts.shape)
    valid = torch.arange(world.pts.shape[1]).unsqueeze(0) < world.npts.unsqueeze(1)
    out = []
    for i in range(len(world)):
        c = int(world.cls[i])
        if c not in keep:
            continue
        out.append((c, local[i][valid[i]]))
    return out


def _visible_map(el: list[tuple[int, Tensor]], radius: float) -> list[tuple[int, Tensor]]:
    """Keep the part of each element inside the query radius."""
    out = []
    for c, q in el:
        m = q.norm(dim=-1) <= radius
        if bool(m.any()):
            out.append((c, q[m]))
    return out


def _visible_camera(
    el: list[tuple[int, Tensor]], p: PerceptionParams
) -> list[tuple[int, Tensor]]:
    """Keep the part of each element inside a forward camera's frustum."""
    half = math.radians(p.fov_deg) * 0.5
    out = []
    for c, q in el:
        rng = q.norm(dim=-1)
        ang = torch.atan2(q[:, 1], q[:, 0]).abs()
        m = (rng >= p.min_range_m) & (rng <= p.max_range_m) & (ang <= half)
        if bool(m.any()):
            out.append((c, q[m]))
    return out


def _corrupt(
    el: list[tuple[int, Tensor]], p: PerceptionParams, gen: torch.Generator
) -> list[tuple[int, Tensor]]:
    """Apply the detector's error model: dropout, bias, noise, clutter, mislabel."""
    out = []
    for c, q in el:
        if torch.rand(1, generator=gen).item() < p.element_dropout:
            continue
        rng = q.norm(dim=-1, keepdim=True)
        sigma = p.point_sigma_m + p.range_sigma_frac * rng
        noisy = q + sigma * torch.randn(q.shape, generator=gen)
        # Correlated lateral bias: perpendicular to the element's own direction
        # for a polyline, arbitrary for a single point.
        if q.shape[0] >= 2:
            d = q[-1] - q[0]
            n = torch.stack([-d[1], d[0]])
            n = n / n.norm().clamp_min(1e-6)
        else:
            a = float(torch.rand(1, generator=gen).item()) * 2 * math.pi
            n = torch.tensor([math.cos(a), math.sin(a)])
        noisy = (
            noisy + p.element_bias_sigma_m * float(torch.randn(1, generator=gen).item()) * n
        )
        if torch.rand(1, generator=gen).item() < p.class_flip_prob:
            c = int(torch.randint(0, NUM_CLASSES, (1,), generator=gen).item())
        out.append((c, noisy))

    n_clutter = int(torch.poisson(torch.tensor(p.clutter_mean), generator=gen).item())
    half = math.radians(p.fov_deg) * 0.5
    for _ in range(n_clutter):
        rng = p.min_range_m + (p.max_range_m - p.min_range_m) * float(
            torch.rand(1, generator=gen).item()
        )
        ang = (2 * float(torch.rand(1, generator=gen).item()) - 1) * half
        base = torch.tensor([rng * math.cos(ang), rng * math.sin(ang)])
        c = int(torch.randint(0, NUM_CLASSES, (1,), generator=gen).item())
        if torch.rand(1, generator=gen).item() < 0.5:
            out.append((c, base.view(1, 2)))
        else:
            d = torch.randn(2, generator=gen)
            d = 6.0 * d / d.norm().clamp_min(1e-6)
            out.append((c, torch.stack([base, base + d])))
    return out


def build_sample(
    world: World, map_world: World, frame: int, seed: int, sp: SampleParams
) -> dict[str, Tensor]:
    """One frame: anchored map, anchored detections, and the transform between.

    Args:
        world: The scene as continuous geometry, from
            :func:`~mapposeformer.data.world.build_world`. Detections are cut
            from this.
        map_world: The same scene chunked into stored map elements, from
            :func:`~mapposeformer.data.world.chunk_for_map`. Passed in rather
            than derived here because it is a per-scene artifact, not a
            per-frame one.
        frame: Index into ``world.trajectory``.
        seed: Draws the prior error and the detection noise. Distinct from the
            world seed so the same geometry can be re-run under new noise.
        sp: Cropping, tokenisation, and both noise models.

    Returns:
        A dict of fixed-shape tensors; see ``docs/ARCHITECTURE.md`` for the
        table of shapes, frames and units.
    """
    gen = torch.Generator().manual_seed(seed)
    keep = set(int(c) for c in sp.keep_classes)
    gt = world.trajectory[frame]

    pp = sp.prior
    err = torch.stack(
        [
            (pp.sigma_long_m * torch.randn(1, generator=gen)).clamp(
                -pp.max_long_m, pp.max_long_m
            )[0],
            (pp.sigma_lat_m * torch.randn(1, generator=gen)).clamp(
                -pp.max_lat_m, pp.max_lat_m
            )[0],
            (math.radians(pp.sigma_yaw_deg) * torch.randn(1, generator=gen)).clamp(
                -math.radians(pp.max_yaw_deg), math.radians(pp.max_yaw_deg)
            )[0],
        ]
    )
    prior = G.compose(gt, err)
    delta = G.relative(prior, gt)

    map_el = _visible_map(_crop(map_world, prior, keep), sp.map_radius_m)
    det_el = _corrupt(
        _visible_camera(_crop(world, gt, keep), sp.perception), sp.perception, gen
    )

    map_pts, map_pmask, map_cls = _pack(map_el, sp.max_map_elements, sp.points_per_element)
    det_pts, det_pmask, det_cls = _pack(det_el, sp.max_det_elements, sp.points_per_element)
    return {
        "map_pts": map_pts,
        "map_pmask": map_pmask,
        "map_cls": map_cls,
        "det_pts": det_pts,
        "det_pmask": det_pmask,
        "det_cls": det_cls,
        "delta": delta,
        "prior": prior,
        "gt": gt,
    }
