"""Turn a world and a frame index into one training sample.

The input contract of the project. Two point sets go in:

* the **map**, cropped around the *prior* pose and expressed there -- what the
  vehicle believes it should be seeing;
* the **detections**, cut by the camera frustum from the *true* pose and
  expressed in the true ego frame -- what it actually sees.

They come from different representations of the same geometry: the map is
chunked at fixed world positions, detections are cut at a fixed range, and a
dashed line reaches the map as an attribute and the detector as paint. See
:func:`~mapposeformer.data.world.chunk_for_map` for why sharing the chunking
would invalidate the experiment.

Each sample also carries the previous ``history`` frames' detections in *their*
ego frames, plus the measured egomotion that brings them here. What accumulates
is evidence about the pose *error*, so the past folds in through a known
transform and the single unknown correction aligns the whole set.

Neither point set carries a world coordinate, so the same local geometry gives
identical tensors wherever it sits; ``tests/test_anchoring.py`` holds that. The
prior leaves here only so evaluation can compose the prediction back.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from mapposeformer import geometry as G
from mapposeformer.data.classes import NUM_CLASSES, MarkType
from mapposeformer.data.world import World, WorldParams


@dataclass
class Element:
    """One map element or one detection, in some anchor frame.

    A record rather than a tuple: ``el[2]`` stopped being readable around the
    third field.
    """

    cls: int
    attr: int
    pts: Tensor
    conf: float = 1.0
    """Detector confidence in ``[0, 1]``. A surveyed map element is 1.0; a
    detection carries whatever the error model gave it."""
    sigma: tuple[float, float] = (0.0, 0.0)
    """Reported positional uncertainty in metres, as ``(point, bias)``.

    Two numbers because the detector has two error modes and they behave
    completely differently under averaging: independent per-point noise shrinks
    over an eight-point element, and a whole-element lateral offset does not.
    A matcher told only "0.3 m" cannot tell the eight-point lane chunk it can
    trust from the one it cannot."""


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
    model config; ``config._validate`` refuses a pair that does not."""


@dataclass(frozen=True)
class EgoParams:
    """Relative egomotion, as odometry delivers it: accurate, not exact.

    Noiseless egomotion would be an oracle -- the model could fuse an
    arbitrarily long history for free and the temporal experiment would measure
    nothing. Error grows with distance travelled, so that is how it is modelled.
    """

    drift_frac: float = 0.01
    """Translation error per metre travelled: 4 cm over the 4 m between history
    frames, against a 1.5 m prior. That gap is why accumulating is worth it."""
    yaw_drift_deg_per_m: float = 0.02


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
    conf_near: float = 0.92
    """Confidence of a true detection at zero range, before noise."""
    conf_range_penalty: float = 0.30
    """How much of that is lost at maximum range."""
    conf_sigma: float = 0.08
    sigma_report_frac: float = 0.25
    """Log-normal spread on the *reported* uncertainty. A detector's covariance
    is an estimate too, and one that were exact would hand the model the noise
    realisation it is supposed to be robust to."""
    clutter_conf_mean: float = 0.45
    clutter_conf_sigma: float = 0.15
    """Clutter scores lower than truth on average, and the two distributions
    **overlap** -- so confidence is evidence, never a label. A score that
    separated them would be solving the problem the model is being asked to."""


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
    history: int = 2
    """Past frames folded into each sample. Zero is exactly the single-frame
    model, which is what the ablation against it needs."""
    history_stride: int = 2
    """Trajectory steps back per history frame. At the world's 2 m pitch that
    reaches 4 m and 8 m behind: far enough to see different landmarks, near
    enough that the egomotion between them is still accurate."""
    prior: PriorParams = field(default_factory=PriorParams)
    perception: PerceptionParams = field(default_factory=PerceptionParams)
    ego: EgoParams = field(default_factory=EgoParams)
    keep_classes: tuple[int, ...] = tuple(range(NUM_CLASSES))
    """Class ablation. Restricting this is the experiment described in
    ``classes.py``: drop the along-track anchors and watch longitudinal error
    grow while lateral error does not."""
    stripe_dashed: bool = True
    """Whether the detector sees dashed lines as paint. Off is the control for
    "does the stripe pattern carry along-track information?"; the map is
    unchanged either way, because a map never stored the stripes."""


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
    kept: list[Element], max_elements: int, points_per_element: int
) -> dict[str, Tensor]:
    """Pad a list of elements into fixed-shape tensors.

    Returns ``pts (E, P, 2)``, ``pmask (E, P)``, ``cls (E,)``, ``attr (E,)``,
    ``conf (E,)`` and ``sigma (E, 2)``. A single-point element -- a pole --
    keeps one valid
    point and repeats it into the padding, so a consumer that forgets the mask
    sees a degenerate point rather than a zero at the origin pretending to be a
    detection.

    Overflow drops the *farthest* elements rather than an arbitrary tail: the
    cap is a budget, and spending it on what is nearest is the only ordering
    that does not make the truncation a random ablation.
    """
    e, p = max_elements, points_per_element
    kept = sorted(kept, key=lambda el: float(el.pts.norm(dim=-1).min()))
    out = {
        "pts": torch.zeros(e, p, 2),
        "pmask": torch.zeros(e, p, dtype=torch.bool),
        "cls": torch.zeros(e, dtype=torch.long),
        "attr": torch.zeros(e, dtype=torch.long),
        "conf": torch.zeros(e),
        "sigma": torch.zeros(e, 2),
    }
    for i, el in enumerate(kept[:e]):
        out["pts"][i] = _resample(el.pts, p)
        out["pmask"][i, :] = el.pts.shape[0] > 1
        out["pmask"][i, 0] = True
        out["cls"][i] = el.cls
        out["attr"][i] = el.attr
        out["conf"][i] = el.conf
        out["sigma"][i] = torch.tensor(el.sigma)
    return out


def _crop(
    world: World, pose: Tensor, keep: set[int], radius_m: float
) -> list[Element]:
    """World elements in ``pose``'s frame, with their padding stripped.

    Whole elements, not clipped ones. Callers that need a clip do it
    afterwards, which matters for :func:`_painted`: arclength is measured from
    an element's first point, and that point has to be the same one in every
    frame or the stripe pattern would slide with the vehicle and become an
    along-track cue that says where the *crop* is.

    ``radius_m`` is a **pre-filter**, not the crop: it discards elements no
    caller could keep, before anything is materialised per element. The loop
    below is Python and its cost tracks the element count, which is 46 for a
    generated scene and over a thousand for a nuScenes one -- enough that
    without this the dataloader starves the GPU.

    @param world The scene.
    @param pose ``(3,)`` the frame to express elements in.
    @param keep Class ids to retain.
    @param radius_m Discard elements with no point within this range.
    @return The surviving elements, in ``pose``'s frame.
    """
    inv = G.inverse(pose)
    local = G.transform_points(inv, world.pts.reshape(-1, 2)).reshape(
        world.pts.shape
    )
    valid = torch.arange(world.pts.shape[1]).unsqueeze(
        0
    ) < world.npts.unsqueeze(1)
    # Range and class, both vectorised, both before the loop. Padding is
    # pushed to infinity so it cannot make an element look near.
    rng = local.norm(dim=-1).masked_fill(~valid, float("inf"))
    near = rng.min(dim=1).values <= radius_m
    wanted = torch.zeros(len(world), dtype=torch.bool)
    for c in keep:
        wanted |= world.cls == c
    chosen = (near & wanted).nonzero(as_tuple=False).flatten().tolist()

    # One tensor-to-list conversion instead of two per element.
    cls_of, attr_of = world.cls.tolist(), world.attr.tolist()
    return [
        Element(cls=cls_of[i], attr=attr_of[i], pts=local[i][valid[i]])
        for i in chosen
    ]


def _visible_map(el: list[Element], radius: float) -> list[Element]:
    """Keep the part of each element inside the query radius.

    The map is never striped and never split: a stored vector map is a
    polyline plus an attribute, and reproducing that faithfully is the whole
    reason the attribute is interesting.
    """
    out = []
    for e in el:
        m = e.pts.norm(dim=-1) <= radius
        if bool(m.any()):
            out.append(replace(e, pts=e.pts[m]))
    return out


def _runs(mask: Tensor) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` spans of True in a 1-D mask."""
    idx = mask.nonzero(as_tuple=False).flatten().tolist()
    if not idx:
        return []
    spans, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            spans.append((start, prev + 1))
            start = i
        prev = i
    spans.append((start, prev + 1))
    return spans


def _painted(pts: Tensor, wp: WorldParams, phase: float) -> Tensor:
    """Which points of a dashed line sit on paint rather than in a gap.

    Arclength runs from the element's first point, which is a fixed world
    position, so a given stripe occupies the same stretch of road in every
    frame. That is what makes a stripe end a landmark instead of an artefact.
    """
    seg = (pts[1:] - pts[:-1]).norm(dim=-1)
    s = torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])
    pitch = wp.stripe_m + wp.gap_m
    return ((s + phase) % pitch) < wp.stripe_m


def _visible_camera(
    el: list[Element], p: PerceptionParams, wp: WorldParams, stripe: bool
) -> list[Element]:
    """Elements as a forward camera delivers them: clipped, split, and striped.

    Two things happen here that ``_visible_map`` does not do, and both are
    properties of *seeing* rather than of *storing*:

    * a line that leaves the frustum and comes back is two detections, not one
      element with a jump across the gap;
    * a dashed line is paint, so it arrives as stripes -- and the ends of those
      stripes are the only along-track evidence lane geometry ever has.
    """
    half = math.radians(p.fov_deg) * 0.5
    out = []
    for i, e in enumerate(el):
        q = e.pts
        rng = q.norm(dim=-1)
        ang = torch.atan2(q[:, 1], q[:, 0]).abs()
        m = (rng >= p.min_range_m) & (rng <= p.max_range_m) & (ang <= half)
        polyline = q.shape[0] > 1
        if stripe and e.attr == MarkType.DASHED and polyline:
            # A phase per element, deterministic in the element's index so it
            # is a property of the scene and not of the frame.
            m = m & _painted(
                q, wp, phase=(i * 0.37) % 1.0 * (wp.stripe_m + wp.gap_m)
            )
        for a, b in _runs(m):
            if polyline and b - a < 2:
                continue  # a one-point fragment of a line is not a detection
            out.append(replace(e, pts=q[a:b]))
    return out


def _draw_class(pool: Sequence[int], gen: torch.Generator) -> int:
    """A class for a false positive, or for a mislabelled true one.

    Drawn from **what this frame actually saw, with multiplicity**, so a class
    is hallucinated about as often as it is detected. Two simpler rules were
    tried first, and each distorted the observability ablation:

    * *Uniform over the enum.* A third of clutter took classes nuScenes has
      none of, so it could never match anything, and restricting
      ``keep_classes`` removed that garbage along with the evidence.
    * *Uniform over the classes the scene contains.* Equal clutter on unequal
      populations contaminates rare classes hardest. Measured on the nuScenes
      test split: 10.0 road boundaries per frame against 0.7 traffic signs,
      0.5 clutter elements each, so 4.7% noise on one class and 42.9% on the
      other -- and the rare classes are the along-track anchors the ablation
      exists to weigh.

    Multiplicity equalises the ratio instead of the count, leaving every class
    at ``clutter_mean / total``, so the ablation compares classes rather than
    their contamination. All three rules agree on generated scenes, which hold
    every class in comparable numbers; a real map is not balanced.

    @param pool The frame's own detected classes, or the world's when nothing
        has been detected yet.
    @param gen The frame's generator.
    @return One of ``pool``.
    """
    i = int(torch.randint(0, len(pool), (1,), generator=gen).item())
    return pool[i]


def _confidence(
    rng_m: float, p: PerceptionParams, gen: torch.Generator, clutter: bool
) -> float:
    """A detector score: informative about truth, never decisive."""
    if clutter:
        mean, sigma = p.clutter_conf_mean, p.clutter_conf_sigma
    else:
        mean = p.conf_near - p.conf_range_penalty * min(
            rng_m / p.max_range_m, 1.0
        )
        sigma = p.conf_sigma
    return float(
        torch.clamp(
            mean + sigma * torch.randn(1, generator=gen), 0.02, 0.99
        ).item()
    )


def _reported_sigma(
    rng_m: float, p: PerceptionParams, gen: torch.Generator
) -> tuple[float, float]:
    """The uncertainty the detector claims, near the one it actually has.

    Truthful in expectation and wrong on any given element, which is the point:
    an exact covariance would hand the model the noise realisation rather than
    the noise model.
    """
    true = (
        p.point_sigma_m + p.range_sigma_frac * rng_m,
        p.element_bias_sigma_m,
    )
    jitter = torch.exp(p.sigma_report_frac * torch.randn(2, generator=gen))
    return (float(true[0] * jitter[0]), float(true[1] * jitter[1]))


def _corrupt(
    el: list[Element],
    p: PerceptionParams,
    gen: torch.Generator,
    classes: tuple[int, ...],
) -> list[Element]:
    """The detector's error model: dropout, bias, noise, clutter, mislabel."""
    out = []
    # What this frame has detected so far, with repeats: the pool clutter and
    # mislabels draw from. See :func:`_draw_class`.
    seen: list[int] = []
    for e in el:
        q = e.pts
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
            noisy
            + p.element_bias_sigma_m
            * float(torch.randn(1, generator=gen).item())
            * n
        )
        cls = e.cls
        if torch.rand(1, generator=gen).item() < p.class_flip_prob:
            cls = _draw_class(seen or classes, gen)
        seen.append(cls)
        mean_rng = float(rng.mean())
        out.append(
            Element(
                cls=cls,
                attr=e.attr,
                pts=noisy,
                conf=_confidence(mean_rng, p, gen, clutter=False),
                sigma=_reported_sigma(mean_rng, p, gen),
            )
        )

    n_clutter = int(
        torch.poisson(torch.tensor(p.clutter_mean), generator=gen).item()
    )
    half = math.radians(p.fov_deg) * 0.5
    for _ in range(n_clutter):
        rng = p.min_range_m + (p.max_range_m - p.min_range_m) * float(
            torch.rand(1, generator=gen).item()
        )
        ang = (2 * float(torch.rand(1, generator=gen).item()) - 1) * half
        base = torch.tensor([rng * math.cos(ang), rng * math.sin(ang)])
        c = _draw_class(seen or classes, gen)
        conf = _confidence(rng, p, gen, clutter=True)
        if torch.rand(1, generator=gen).item() < 0.5:
            pts = base.view(1, 2)
        else:
            d = torch.randn(2, generator=gen)
            pts = torch.stack([base, base + 6.0 * d / d.norm().clamp_min(1e-6)])
        out.append(
            Element(
                cls=c,
                attr=int(MarkType.NONE),
                pts=pts,
                conf=conf,
                sigma=_reported_sigma(rng, p, gen),
            )
        )
    return out


def _from_detector(
    rows: dict[str, Tensor], keep: set[int], sp: SampleParams
) -> dict[str, Tensor]:
    """Pack what a real detector reported at one keyframe.

    No error model runs: these are already wrong in a detector's own way, and
    corrupting them further would model the same failure twice. The sigma a
    synthetic detection carries is replaced by a constant, because a real
    mapper reports a score and not a covariance -- if one ever does, this is
    where it arrives.

    @param rows One keyframe's elements, from
        :meth:`~mapposeformer.data.detections.SceneDetections.frame`.
    @param keep Class ablation.
    @param sp Cropping and tokenisation.
    @return The same packed tensors ``_detect`` returns.
    """
    kept = [
        Element(
            cls=int(rows["cls"][i]),
            attr=int(MarkType.NONE),
            pts=rows["pts"][i, : int(rows["npts"][i])],
            conf=float(rows["conf"][i]),
            sigma=(
                sp.perception.point_sigma_m,
                sp.perception.element_bias_sigma_m,
            ),
        )
        for i in range(len(rows["cls"]))
        if int(rows["cls"][i]) in keep
    ]
    return _pack(kept, sp.max_det_elements, sp.points_per_element)


def _detect(
    world: World,
    pose: Tensor,
    keep: set[int],
    sp: SampleParams,
    gen: torch.Generator,
) -> dict[str, Tensor]:
    """One frame of detections, in the true ego frame at ``pose``."""
    wp = world.params
    el = _visible_camera(
        _crop(world, pose, keep, sp.perception.max_range_m),
        sp.perception,
        wp,
        sp.stripe_dashed,
    )
    present = tuple(
        sorted({int(c) for c in world.cls.unique().tolist()} & keep)
    )
    return _pack(
        _corrupt(el, sp.perception, gen, present or tuple(sorted(keep))),
        sp.max_det_elements,
        sp.points_per_element,
    )


def _measured_egomotion(
    curr: Tensor, past: Tensor, p: EgoParams, gen: torch.Generator
) -> Tensor:
    """``curr⁻¹ ∘ past`` as odometry reports it, drift growing with travel.

    Composed on the right, in the past frame, because that is where the error
    accumulated: the vehicle drove from there to here and the integration is
    what went wrong.
    """
    rel = G.relative(curr, past)
    dist = float(rel[:2].norm())
    err = torch.tensor(
        [
            p.drift_frac * dist * float(torch.randn(1, generator=gen).item()),
            p.drift_frac * dist * float(torch.randn(1, generator=gen).item()),
            math.radians(p.yaw_drift_deg_per_m)
            * dist
            * float(torch.randn(1, generator=gen).item()),
        ]
    )
    return G.compose(rel, err)


def build_sample(
    world: World,
    map_world: World,
    frame: int,
    seed: int,
    sp: SampleParams,
    detector=None,
) -> dict[str, Tensor]:
    """One frame: anchored map, anchored detections, history, and the transform.

    @param world The scene as continuous geometry, from
        :func:`~mapposeformer.data.world.build_world`. Detections are cut from
        this.
    @param map_world The same scene chunked into stored map elements, from
        :func:`~mapposeformer.data.world.chunk_for_map`. Passed in rather than
        derived here because it is a per-scene artifact, not a per-frame one.
    @param frame Index into ``world.trajectory``.
    @param seed Draws the prior error, the detection noise and the egomotion
        drift. Distinct from the world seed so the same geometry can be re-run
        under new noise.
    @param sp Cropping, tokenisation, and all three noise models.

    @return A dict of fixed-shape tensors; see ``docs/ARCHITECTURE.md`` for the
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
            (
                math.radians(pp.sigma_yaw_deg) * torch.randn(1, generator=gen)
            ).clamp(
                -math.radians(pp.max_yaw_deg), math.radians(pp.max_yaw_deg)
            )[0],
        ]
    )
    prior = G.compose(gt, err)
    delta = G.relative(prior, gt)

    map_el = _visible_map(
        _crop(map_world, prior, keep, sp.map_radius_m), sp.map_radius_m
    )
    m = _pack(map_el, sp.max_map_elements, sp.points_per_element)
    if detector is None:
        d = _detect(world, gt, keep, sp, gen)
    else:
        d = _from_detector(detector(frame), keep, sp)

    # --- The past, in its own frames, plus the egomotion that moves it here.
    hist, rel = [], []
    for k in range(1, sp.history + 1):
        f = max(0, frame - k * sp.history_stride)
        past = world.trajectory[f]
        if detector is None:
            hist.append(_detect(world, past, keep, sp, gen))
        else:
            hist.append(_from_detector(detector(f), keep, sp))
        rel.append(_measured_egomotion(gt, past, sp.ego, gen))
    e, p = sp.max_det_elements, sp.points_per_element

    def stack(key: str, shape: tuple[int, ...]) -> Tensor:
        """History frames as one tensor, or an empty one when there are none.

        @param key Field name shared by every history frame's pack.
        @param shape Per-frame shape, so the empty case still types correctly.
        @return ``(K, *shape)``, with ``K = 0`` when ``history`` is zero.
        """
        if not hist:
            return torch.zeros(0, *shape, dtype=d[key].dtype)
        return torch.stack([h[key] for h in hist])

    return {
        "map_pts": m["pts"],
        "map_pmask": m["pmask"],
        "map_cls": m["cls"],
        "map_attr": m["attr"],
        "det_pts": d["pts"],
        "det_pmask": d["pmask"],
        "det_cls": d["cls"],
        "det_attr": d["attr"],
        "det_conf": d["conf"],
        "det_sigma": d["sigma"],
        "hist_pts": stack("pts", (e, p, 2)),
        "hist_pmask": stack("pmask", (e, p)),
        "hist_cls": stack("cls", (e,)),
        "hist_attr": stack("attr", (e,)),
        "hist_conf": stack("conf", (e,)),
        "hist_sigma": stack("sigma", (e, 2)),
        "hist_rel": torch.stack(rel) if rel else torch.zeros(0, 3),
        "delta": delta,
        "prior": prior,
        "gt": gt,
    }
