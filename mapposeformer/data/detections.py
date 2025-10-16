"""Real detections, read from a file a detector wrote.

Perception is an input to this project, and until now that input was
synthesised: detections were cut from the map and corrupted, so both point sets
were the same polylines with noise on top. A real detector is wrong in ways an
error model does not reproduce -- different chunking, geometry that bends the
wrong way at range, missing pieces, hallucinated topology -- and closing that
gap is what separates "a point-set registration network" from "a localizer".

**The detector does not run here, and cannot.** MapTR and StreamMapNet are
written against ``mmdet3d 1.0.0rcX`` and ``mmcv 1.x``, which cap at Python 3.10
and torch 2.0; this project runs 3.12 and torch 2.11. That is not a problem to
solve but a boundary to draw, and it is the same one already drawn around the
nuScenes devkit: **the detector runs offline, in its own environment, and
writes a file.** Nothing in ``mapposeformer`` imports it, so the two cannot
constrain each other's dependencies.

What that file must contain is below. It is deliberately a *format* rather than
an API, so any mapper can satisfy it -- including a future one this project has
never heard of, and including a stub used to test the path itself.

## The contract

One ``.npz`` per scene, named for the scene, holding four arrays:

| array | shape | meaning |
|---|---|---|
| ``frame`` | ``(N,)`` int | keyframe index within the scene |
| ``cls`` | ``(N,)`` int | :class:`~mapposeformer.data.classes.LandmarkClass` |
| ``conf`` | ``(N,)`` float | detector score in ``[0, 1]`` |
| ``pts`` | ``(N, P, 2)`` float | the element, **ego frame, metres** |
| ``npts`` | ``(N,)`` int | valid points in each row of ``pts`` |

Two conventions matter, and both are easy to get silently wrong:

* **The ego frame is X forward, Y left**, the vehicle convention this project
  uses everywhere. A detector emitting the camera convention will produce a
  model that trains happily to a rotated answer.
* **Coordinates are that keyframe's own ego frame**, not the world's. The whole
  design rests on no world coordinate reaching the network.

:func:`validate` checks what can be checked cheaply. It cannot check the frame
convention, which is why :func:`plausibility` exists: applying the ground-truth
correction should land real detections on the map, and if it does not, the
convention is wrong. That is the same invariant the synthetic data is held to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from mapposeformer.data.classes import NUM_CLASSES

REQUIRED = ("frame", "cls", "conf", "pts", "npts")


@dataclass
class SceneDetections:
    """One scene's detections, indexed by keyframe.

    Held as a dict of frame to row indices rather than searched per access: a
    dataloader asks for one frame at a time and a linear scan over a scene's
    rows would show up in the step time.
    """

    cls: Tensor
    conf: Tensor
    pts: Tensor
    npts: Tensor
    rows: dict[int, Tensor]

    def frame(self, index: int) -> dict[str, Tensor]:
        """@brief Every element detected at one keyframe.

        @param index Keyframe index within the scene.
        @return Dict of ``cls``, ``conf``, ``pts``, ``npts``; empty tensors
            when the detector reported nothing, which is a legitimate frame.
        """
        rows = self.rows.get(index)
        if rows is None:
            return {
                "cls": self.cls[:0],
                "conf": self.conf[:0],
                "pts": self.pts[:0],
                "npts": self.npts[:0],
            }
        return {
            "cls": self.cls[rows],
            "conf": self.conf[rows],
            "pts": self.pts[rows],
            "npts": self.npts[rows],
        }


def validate(data: dict[str, np.ndarray]) -> None:
    """@brief Reject a detection file that cannot be right.

    @param data The arrays loaded from one scene's ``.npz``.
    @throws ValueError If a required array is missing, the lengths disagree,
        a class is out of range, or a confidence is outside ``[0, 1]``.
    """
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise ValueError(f"detection file is missing {missing}")
    n = len(data["frame"])
    wrong = {k: len(data[k]) for k in REQUIRED if len(data[k]) != n}
    if wrong:
        raise ValueError(f"arrays disagree on length: {n} against {wrong}")
    if data["pts"].ndim != 3 or data["pts"].shape[2] != 2:
        raise ValueError(f"pts must be (N, P, 2), got {data['pts'].shape}")
    if n and (data["cls"].min() < 0 or data["cls"].max() >= NUM_CLASSES):
        raise ValueError(f"class outside 0..{NUM_CLASSES - 1}")
    if n and (data["conf"].min() < 0.0 or data["conf"].max() > 1.0):
        raise ValueError("confidence outside [0, 1]")
    if n and (
        data["npts"].min() < 1 or data["npts"].max() > data["pts"].shape[1]
    ):
        raise ValueError("npts outside 1..P")


def load_scene(path: Path) -> SceneDetections:
    """@brief Read one scene's detections.

    @param path The scene's ``.npz``.
    @return Its detections, indexed by keyframe.
    @throws ValueError If the file violates the contract.
    """
    with np.load(path) as raw:
        data = {k: raw[k] for k in raw.files}
    validate(data)
    frame = torch.from_numpy(data["frame"]).long()
    rows: dict[int, Tensor] = {}
    for f in frame.unique().tolist():
        rows[int(f)] = (frame == f).nonzero(as_tuple=False).flatten()
    return SceneDetections(
        cls=torch.from_numpy(data["cls"]).long(),
        conf=torch.from_numpy(data["conf"]).float(),
        pts=torch.from_numpy(data["pts"]).float(),
        npts=torch.from_numpy(data["npts"]).long(),
        rows=rows,
    )


def plausibility(sample: dict[str, Tensor], radius_m: float = 1.0) -> float:
    """@brief Fraction of detections that land on the map under the true
    correction.

    The one check that catches a wrong frame convention, and the reason it
    exists separately from :func:`validate`: a detector emitting camera axes,
    or world coordinates, or a mirrored Y, produces a file that validates
    perfectly and trains to a confidently wrong answer. Applying the
    ground-truth correction has to put its output on the map, or the geometry
    disagrees with the labels and every metric downstream measures nothing.

    Around 0.9 is what the synthetic path scores; a detector that is genuinely
    worse will score lower, so this is a floor to reason about rather than a
    threshold to pass. Near zero means a convention error, not a bad detector.

    @param sample One output of
        :func:`~mapposeformer.data.sample.build_sample`.
    @param radius_m Distance within which a point counts as landing on the map.
    @return The fraction, or 0.0 when the frame has no detections.
    """
    from mapposeformer import geometry as G

    pts = sample["det_pts"][sample["det_pmask"]]
    if pts.numel() == 0:
        return 0.0
    aligned = G.transform_points(sample["delta"], pts)
    starts, ends = [], []
    for e in range(sample["map_pts"].shape[0]):
        q = sample["map_pts"][e][sample["map_pmask"][e]]
        if q.shape[0] >= 2:
            starts.append(q[:-1])
            ends.append(q[1:])
    if not starts:
        return 0.0
    a, b = torch.cat(starts), torch.cat(ends)
    d = b - a
    t = (
        ((aligned[:, None] - a) * d).sum(-1)
        / d.square().sum(-1).clamp_min(1e-9)
    ).clamp(0, 1)
    dist = (aligned[:, None] - (a + t[..., None] * d)).norm(dim=-1).min(-1)
    return float((dist.values < radius_m).float().mean())
