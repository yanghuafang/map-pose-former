"""nuScenes as a :class:`~mapposeformer.data.world.World`, one per scene.

The point of this file is how little of it there is. Everything downstream --
the crop, the prior, the detector's error model, the history, the sample
contract -- already works on a ``World``, so a real map needs a reader and
nothing else. ``build_sample`` does not know which of the two it was handed.

Three things about nuScenes shape the reader:

* **The map is a whole city and a scene is 200 metres of it.** Cropping 38 000
  nodes per frame would dominate the step time, so a ``World`` is built per
  scene from the elements near that scene's trajectory. This mirrors the
  synthetic side, where a world *is* one scene.
* **The devkit is not imported.** The expansion is nodes, lines and polygons in
  one JSON, which is a dictionary lookup and a list comprehension. Adding a
  heavy dependency to read it would cost more than it saves.
* **Traffic lights exist, and the project's own documentation said they would
  not.** 307 of them in Boston alone, each with a pose. They are the only
  point landmark nuScenes carries, and along-track observability is exactly
  what point landmarks provide -- see :data:`NUSCENES_CLASSES`.

The lane/divider/boundary mapping follows the online-mapping convention
(MapTR, StreamMapNet) rather than inventing one, so a detector trained on that
convention drops in at M2b without a translation layer.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from mapposeformer.data.classes import LandmarkClass, MarkType
from mapposeformer.data.dataset import DataParams
from mapposeformer.data.sample import build_sample
from mapposeformer.data.world import (
    World,
    WorldParams,
    _pack_elements,
    chunk_for_map,
)

#: The four cities, and the map each log was recorded in.
LOCATIONS = (
    "boston-seaport",
    "singapore-onenorth",
    "singapore-queenstown",
    "singapore-hollandvillage",
)

#: What the map expansion actually carries, as our classes.
#:
#: No poles: nuScenes has none, and the M1 ablation put the cost of that at
#: 2.3x worse longitudinal error. Traffic lights are the only point landmark
#: it does carry, and they are **sparse** -- 307 across Boston, which puts one
#: within 120 m of only about a fifth of scenes. Whether that is enough to
#: matter is measurable rather than arguable, and
#: `NuScenesParams.traffic_lights` switches them off for the comparison.
NUSCENES_CLASSES = (
    LandmarkClass.LANE_DIVIDER,
    LandmarkClass.ROAD_BOUNDARY,
    LandmarkClass.PED_CROSSING,
    LandmarkClass.TRAFFIC_SIGN,
)


@dataclass(frozen=True)
class NuScenesParams:
    """Where the data is, and how much of it to put in a scene."""

    root: str = ""
    """Dataset root: the directory holding ``maps/`` and the metadata."""
    version: str = "v1.0-trainval"
    map_radius_m: float = 120.0
    """Elements this far from any trajectory point join the scene's world.
    Comfortably past the 50 m query radius, so a crop near the end of a scene
    still sees a full neighbourhood."""
    step_m: float = 2.0
    """Polylines are resampled to this pitch, matching the synthetic world so
    the two produce elements of comparable density."""
    map_chunk_m: float = 12.0
    """Length of a stored map element. The map is chunked at these boundaries
    and detections are not, which is the asymmetry the ablation depends on --
    see :func:`~mapposeformer.data.world.chunk_for_map`."""
    source_chunk_m: float = 600.0
    """Length of an element in the world detections are cut from. Long, so a
    frustum-sized cut almost never lands on one of its boundaries; 600 m is the
    synthetic world's element length, so the two paths carry the same geometry.

    It is not ``map_chunk_m``, and that mattered: chunking once at ingest and
    using the result as both the map and the detection source gave every
    element the same endpoints and the same arclength samples on both sides.
    Half of all lane geometry then arrived with its correspondence already
    solved, along-track included, and the observability ablation measured
    nothing. The bound exists only to cap padding in ``_pack_elements``."""
    traffic_lights: bool = True
    """Include traffic lights as ``TRAFFIC_SIGN``. Off reproduces the class set
    the project assumed nuScenes had."""
    min_frames: int = 12
    """Scenes shorter than this are dropped rather than padded."""


@dataclass
class _MapData:
    """One city's vector map, indexed for lookup."""

    node: dict[str, tuple[float, float]] = field(default_factory=dict)
    line: dict[str, list[str]] = field(default_factory=dict)
    polygon: dict[str, list[str]] = field(default_factory=dict)
    elements: list[tuple[int, Tensor]] = field(default_factory=list)


def _resample_polyline(pts: Tensor, step_m: float) -> Tensor:
    """Even arclength resampling, so element density does not follow the
    surveyor's node spacing.

    @param pts ``(N, 2)`` polyline vertices.
    @param step_m Target spacing.
    @return ``(M, 2)`` with M at least 2.
    """
    if pts.shape[0] < 2:
        return pts
    seg = (pts[1:] - pts[:-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])
    total = float(cum[-1])
    if total < step_m:
        return pts[[0, -1]]
    t = torch.linspace(0.0, total, max(2, int(total / step_m) + 1))
    hi = torch.searchsorted(cum, t).clamp(1, pts.shape[0] - 1)
    lo = hi - 1
    w = ((t - cum[lo]) / (cum[hi] - cum[lo]).clamp_min(1e-9)).unsqueeze(-1)
    return pts[lo] + w * (pts[hi] - pts[lo])


def _principal_bar(pts: Tensor) -> Tensor:
    """An area feature as the bar across the road, not as its outline.

    nuScenes stores crossings and stop lines as polygons. Closing one into a
    loop gives a rectangle whose *long* edges run along the road, so a third of
    its segments end up carrying lane-like geometry under a class label that
    says perpendicular -- which is worse than omitting the feature, because the
    model is told the wrong thing rather than nothing.

    What a localizer can use is the extent across the road: the polygon's
    principal axis, which for a stop line or a crossing is the direction it
    spans. That is also what the synthetic world generates for these classes,
    so the two datasets mean the same thing by "stop line" and the class
    ablation compares like with like.

    @param pts ``(N, 2)`` polygon vertices.
    @return ``(2, 2)`` the segment spanning the polygon's long axis.
    """
    centre = pts.mean(0)
    centred = pts - centre
    # Principal direction of a 2-D point set: the eigenvector of its
    # covariance with the larger eigenvalue.
    _, vecs = torch.linalg.eigh(centred.T @ centred / max(len(pts), 1))
    axis = vecs[:, -1]
    extent = centred @ axis
    return torch.stack(
        [centre + extent.min() * axis, centre + extent.max() * axis]
    )


def load_map(root: str, location: str, p: NuScenesParams) -> _MapData:
    """Read one city's expansion JSON into polylines, in world metres.

    @param root Dataset root.
    @param location One of :data:`LOCATIONS`.
    @param p Reader settings.
    @return The city's elements as ``(class, points)`` pairs.
    """
    raw = json.loads(
        (Path(root) / "maps" / "expansion" / f"{location}.json").read_text()
    )
    data = _MapData(
        node={n["token"]: (n["x"], n["y"]) for n in raw["node"]},
        line={ln["token"]: ln["node_tokens"] for ln in raw["line"]},
        polygon={
            pg["token"]: pg["exterior_node_tokens"] for pg in raw["polygon"]
        },
    )

    def pts_of(tokens: list[str]) -> Tensor | None:
        if not tokens:
            return None
        xy = torch.tensor([data.node[t] for t in tokens if t in data.node])
        return xy if xy.ndim == 2 and xy.shape[0] >= 2 else None

    def add(
        cls: LandmarkClass, pts: Tensor | None, closed: bool = False
    ) -> None:
        if pts is None:
            return
        if closed:
            pts = torch.cat([pts, pts[:1]])
        resampled = _resample_polyline(pts, p.step_m)
        if resampled.shape[0] >= 2:
            data.elements.append((int(cls), resampled))

    # Dividers: both the lane-to-lane markings and the centre line between
    # opposing traffic. The online-mapping convention merges them, and so does
    # the geometry -- both are paint running along the road.
    for layer in ("lane_divider", "road_divider"):
        for el in raw.get(layer, []):
            add(
                LandmarkClass.LANE_DIVIDER,
                pts_of(data.line.get(el["line_token"], [])),
            )

    # Boundary: the drivable-area outline, which is what a detector segments.
    for el in raw.get("drivable_area", []):
        for token in el.get("polygon_tokens", []):
            add(
                LandmarkClass.ROAD_BOUNDARY,
                pts_of(data.polygon.get(token, [])),
                closed=True,
            )

    # Crossings, reduced to the bar across the road. Road boundary keeps its
    # outline above, because a boundary *is* a line.
    #
    # **Stop lines are not read.** nuScenes annotates a stop *zone* rather than
    # a stop *bar*: 83% of the polygons are rounder than 2:1, median aspect
    # 1.5, so there is no direction to extract and a principal axis points
    # essentially at random. Under the STOP_LINE label that supplies
    # perpendicular evidence that is not perpendicular, which is worse than
    # supplying nothing: an omitted class costs the model what it knew, a
    # mislabelled one teaches it something false about every other member of
    # that class. A stop zone's centroid is a real point landmark and reading
    # it as one is the obvious alternative, but it would
    # mean the class denotes different geometry on each dataset, so it is left
    # for a decision rather than taken quietly here.
    for el in raw.get("ped_crossing", []):
        poly = pts_of(data.polygon.get(el["polygon_token"], []))
        if poly is not None:
            add(LandmarkClass.PED_CROSSING, _principal_bar(poly))

    # Traffic lights: a single point each, from the pose the map stores. The
    # only along-track anchor nuScenes has that is not perpendicular paint.
    if p.traffic_lights:
        for el in raw.get("traffic_light", []):
            pose = el.get("pose") or {}
            if "tx" in pose and "ty" in pose:
                data.elements.append(
                    (
                        int(LandmarkClass.TRAFFIC_SIGN),
                        torch.tensor([[pose["tx"], pose["ty"]]]),
                    )
                )
    return data


def scene_world(city: _MapData, trajectory: Tensor, p: NuScenesParams) -> World:
    """The map near one scene's path, as a :class:`World`.

    @param city Output of :func:`load_map`.
    @param trajectory ``(T, 3)`` ego poses, world frame.
    @param p Reader settings.
    @return The world detections are cut from. The stored map is
        :func:`~mapposeformer.data.world.chunk_for_map` of this, derived per
        scene by the dataset, exactly as the synthetic path does it.
    """
    path = trajectory[:, :2]
    kept: list[tuple[int, int, Tensor]] = []
    chunk_pts = max(2, int(p.source_chunk_m / p.step_m) + 1)
    for cls, pts in city.elements:
        # Nearest trajectory point to any vertex; cheap and exact enough at a
        # 120 m radius, where being a metre wrong about the bound is harmless.
        d = torch.cdist(pts, path).min()
        if float(d) > p.map_radius_m:
            continue
        # nuScenes has no mark_type, so every painted line is SOLID. Argoverse
        # 2 does carry it, and that is where the dash experiment moves.
        attr = int(
            MarkType.SOLID
            if cls in (LandmarkClass.LANE_DIVIDER, LandmarkClass.ROAD_BOUNDARY)
            else MarkType.NONE
        )
        if pts.shape[0] <= chunk_pts:
            kept.append((cls, attr, pts))
            continue
        for s in range(0, pts.shape[0] - 1, chunk_pts - 1):
            piece = pts[s : s + chunk_pts]
            if piece.shape[0] >= 2:
                kept.append((cls, attr, piece))
    if not kept:
        kept = [
            (int(LandmarkClass.LANE_DIVIDER), int(MarkType.SOLID), path[:2])
        ]
    return _pack_elements(
        kept,
        trajectory,
        WorldParams(step_m=p.step_m, map_chunk_m=p.map_chunk_m),
    )


def _yaw_from_quaternion(q: list[float]) -> float:
    """Heading from a ``(w, x, y, z)`` rotation, about the world Z axis."""
    w, x, y, z = q
    return float(
        torch.atan2(
            torch.tensor(2.0 * (w * z + x * y)),
            torch.tensor(1.0 - 2.0 * (y * y + z * z)),
        )
    )


def load_scenes(p: NuScenesParams) -> dict[str, dict]:
    """Every scene's keyframe trajectory and the city it was driven in.

    Keyframe poses are matched to ``ego_pose`` by timestamp: a sample's
    timestamp is its LIDAR_TOP reading's, and that reading's ego pose carries
    the same value. Doing it this way reads 650 MB instead of the 2 GB that
    going through ``sample_data.json`` would cost, and
    ``tests/test_nuscenes.py`` checks the match is exact.

    @param p Reader settings, including the dataset root.
    @return ``{scene_name: {"location": str, "trajectory": (T, 3) tensor}}``.
    """
    meta = Path(p.root) / p.version
    scenes = json.loads((meta / "scene.json").read_text())
    logs = {
        lg["token"]: lg for lg in json.loads((meta / "log.json").read_text())
    }
    samples = json.loads((meta / "sample.json").read_text())
    poses = json.loads((meta / "ego_pose.json").read_text())

    by_time: dict[int, dict] = {}
    for e in poses:
        by_time.setdefault(e["timestamp"], e)
    by_scene: dict[str, list] = {}
    for s in samples:
        by_scene.setdefault(s["scene_token"], []).append(s)

    out: dict[str, dict] = {}
    for sc in scenes:
        rows = sorted(
            by_scene.get(sc["token"], []), key=lambda s: s["timestamp"]
        )
        pose_rows = [
            by_time[s["timestamp"]] for s in rows if s["timestamp"] in by_time
        ]
        if len(pose_rows) < p.min_frames:
            continue
        traj = torch.tensor(
            [
                [
                    e["translation"][0],
                    e["translation"][1],
                    _yaw_from_quaternion(e["rotation"]),
                ]
                for e in pose_rows
            ]
        )
        out[sc["name"]] = {
            "location": logs[sc["log_token"]]["location"],
            "trajectory": traj,
        }
    return out


#: Fraction of each city's extent given to train, val and test, along the axis
#: the city is longest on. They sum to less than one: the gaps are buffers.
SPLIT_FRACTIONS = {
    "train": (0.00, 0.62),
    "val": (0.66, 0.79),
    "test": (0.83, 1.00),
}


def geographic_split(
    scenes: dict[str, dict], buffer_m: float = 0.0
) -> dict[str, list[str]]:
    """Partition scenes by *where they were driven*, not by scene id.

    The official nuScenes split shares ground between train and val, so a
    localizer evaluated on it has seen the roads it is being tested on and is
    partly reciting. StreamMapNet made this point and re-split geographically;
    this does the same thing from the ego poses alone, so it needs no external
    file and no devkit.

    Each city is cut along whichever of x or y it is longest on, and the gaps
    between the three bands are dead ground that no split uses. A scene is
    assigned by the extent of its whole trajectory, not its centroid: a scene
    that starts in train's band and ends in val's belongs to neither.

    @param scenes Output of :func:`load_scenes`.
    @param buffer_m Extra margin added to each scene's extent before testing
        which band contains it. Guards against a scene seeing across the cut.
    @return ``{"train": [name, ...], "val": [...], "test": [...]}``.
    """
    out: dict[str, list[str]] = {k: [] for k in SPLIT_FRACTIONS}
    by_city: dict[str, list[str]] = {}
    for name, s in scenes.items():
        by_city.setdefault(s["location"], []).append(name)

    for names in by_city.values():
        xy = torch.cat([scenes[n]["trajectory"][:, :2] for n in names])
        lo, hi = xy.min(0).values, xy.max(0).values
        axis = int(torch.argmax(hi - lo))
        span = float(hi[axis] - lo[axis])
        for name in names:
            t = scenes[name]["trajectory"][:, axis]
            a = (float(t.min()) - buffer_m - float(lo[axis])) / max(span, 1e-6)
            b = (float(t.max()) + buffer_m - float(lo[axis])) / max(span, 1e-6)
            for split, (f0, f1) in SPLIT_FRACTIONS.items():
                if a >= f0 and b <= f1:
                    out[split].append(name)
                    break
    return out


class NuScenesDataset(Dataset):
    """Frames of real nuScenes map, as anchored point sets.

    Reads what :mod:`tools.prepare_nuscenes` cached, so nothing here parses
    JSON and no worker imports a devkit. The sample it returns is byte-for-byte
    the shape :class:`~mapposeformer.data.synthetic.SyntheticDataset` returns,
    which is the whole point: the model, the losses and the evaluator do not
    learn that the data changed.

    **Detections are still synthetic.** They are cut from this map and corrupted
    by the same error model, so what changes here is the *map*, not the
    perception. M2b swaps in a pretrained mapper's output, and until it does,
    the sim-to-real gap is untouched -- see ``docs/OPEN_ITEMS.md``.
    """

    def __init__(self, params: DataParams, split: str):
        cache = Path(params.nuscenes_cache)
        manifest = json.loads((cache / "manifest.json").read_text())
        if split not in manifest["splits"]:
            raise ValueError(f"unknown split {split!r}")
        self.p = params
        self.cache = cache
        self.names = manifest["splits"][split]
        self.stride = params.frame_stride
        self.margin = params.edge_margin
        self._cache: dict[str, tuple[World, World]] = {}
        self._dets: dict[str, object] = {}
        self.detections_dir = params.detections_dir
        self._index: list[tuple[str, int]] = []
        sp = params.sample
        lo = max(self.margin, sp.history * sp.history_stride)
        for name in self.names:
            n = manifest["frames"][name]
            self._index += [
                (name, f) for f in range(lo, n - self.margin, self.stride)
            ]
        if not self._index:
            raise ValueError(f"split {split!r} has no usable frames")

    def _world(self, name: str) -> tuple[World, World]:
        """The scene, and the chunked map derived from it.

        Two worlds, for the reason
        :func:`~mapposeformer.data.world.chunk_for_map` gives: the map is cut
        at fixed survey boundaries and a detection is cut at a range, so the
        two must not share element endpoints.
        """
        if name not in self._cache:
            world = torch.load(
                self.cache / "scenes" / f"{name}.pt", weights_only=False
            )
            self._cache[name] = (
                world,
                chunk_for_map(
                    world,
                    world.params.map_chunk_m,
                    world.params.step_m,
                ),
            )
        return self._cache[name]

    def set_epoch(self, epoch: int) -> None:
        """Reseed the noise for a new epoch, matching the synthetic dataset."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        name, frame = self._index[idx]
        world, chunked = self._world(name)
        epoch = getattr(self, "_epoch", 0) if self.p.augment else 0
        # crc32, not hash(): Python salts string hashing per process, so
        # `hash(name)` gives a different answer in every interpreter and in
        # every dataloader worker. Two evaluations of one checkpoint then draw
        # different detector noise and disagree by more than the effect being
        # measured -- which is how a class ablation came to report that less
        # evidence was better. The synthetic dataset seeds from an integer and
        # never had the problem.
        stable = zlib.crc32(name.encode()) & 0xFFFFF
        seed = stable * 100_003 + frame * 97 + epoch
        # Detections are cut from `world` and the map is `chunked`, so the
        # two sides never share an element boundary. They are still the same
        # polylines with noise on top -- that is M2a's scope, and what M2b
        # replaces with a detector's own geometry.
        return build_sample(
            world, chunked, frame, seed, self.p.sample, self._detector(name)
        )

    def _detector(self, name: str):
        """@brief A real detector's output for one scene, if configured.

        @param name Scene name.
        @return A ``frame -> rows`` callable, or None for the synthetic path.
        """
        if not self.detections_dir:
            return None
        if name not in self._dets:
            from mapposeformer.data.detections import load_scene

            self._dets[name] = load_scene(
                Path(self.detections_dir) / f"{name}.npz"
            )
        return self._dets[name].frame
