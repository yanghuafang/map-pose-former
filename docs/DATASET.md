# Data

## Why the first dataset is generated

KITTI Odometry, which the classical project uses, **ships no HD map**. That
repository synthesizes one as a corridor offset from the ground-truth path. For
a search-based backend that is a mild oracle and it says so. For a *learned*
backend it is fatal: the map becomes a deterministic function of the pose the
network is being asked to predict, and a transformer will learn to invert it.
The numbers would be excellent and would mean nothing.

So stage 0 is procedural. That is not a compromise — it buys something no public
dataset offers:

- **The answer is exactly known**, with no localization pipeline of its own
  standing between the label and the truth.
- **The evidence is controllable.** Turning poles off is one config line.
- **Splits cannot leak**, because they are disjoint seed ranges and no two
  splits ever generate the same road.

Stage 1 is nuScenes, and it is a real map. See [ROADMAP.md](ROADMAP.md).

## Getting nuScenes

Accept the Terms of Use at <https://www.nuscenes.org/nuscenes#download> first —
`scripts/download_nuscenes.sh` deliberately does not, and cannot.

```bash
scripts/download_nuscenes.sh --dry-run   # check the URLs, download nothing
scripts/download_nuscenes.sh             # maps + metadata, ~0.5 GB
```

**That half a gigabyte is the whole of what M3 needs.** Perception is an input
here: detections are cut from the map and corrupted by an error model, and no
detector is trained or run. So the real-data milestone wants the HD map and the
ego poses, and has no use for a camera image. The ~350 GB of sensor archives is
`--blobs`, and it earns nothing until the real-detector path.

The script checks every URL before transferring a byte, resumes an interrupted
download, and names the files it expects afterwards — an archive that unpacked
into the wrong directory otherwise looks exactly like success.

## The world

`mapposeformer/data/world.py` builds one scene per seed: a 600 m road of
piecewise-constant-curvature arcs, four lanes, and every landmark class.

| Class | Geometry | Placement |
|---|---|---|
| `LANE_DIVIDER` | polyline, 24 m chunks | between lanes |
| `ROAD_BOUNDARY` | polyline, 24 m chunks | road edges |
| `POLE` | single point | roadside, mean 22 m spacing |
| `TRAFFIC_SIGN` | single point | roadside, mean 70 m spacing |
| `STOP_LINE` | polyline, perpendicular | at intersections, near half of the road |
| `PED_CROSSING` | polyline, perpendicular | at intersections, full width |

Arcs rather than a curvature random walk, because a road that wobbles at the
2 m sampling pitch gives the model a texture to latch onto that no real road
has. Chunked polylines because real vector maps are chunked, and because a
single 600 m element would be one token whose points are mostly out of view.

The single most consequential number in the file is
`intersection_spacing_m = 110`. Stop lines and crossings are the only
along-track evidence a nuScenes-style map carries, so it sets how often the
longitudinal degree of freedom is observable at all.

## One frame

`mapposeformer/data/sample.py`. Two point sets and the transform between them:

```
prior = gt ∘ error          error ~ truncated Gaussian, anisotropic
delta = prior⁻¹ ∘ gt        the training target
map   = world ∩ radius(prior),  expressed in the prior's frame
det   = world ∩ frustum(gt),    expressed in the true ego frame, corrupted
```

**The prior error is anisotropic on purpose**: 1.5 m along track, 0.6 m across,
1.0° of heading. Dead reckoning drifts fastest along the direction of travel,
and along-track is also the direction lane geometry cannot see — so the hard
axis and the weak evidence are the same axis. A symmetric prior would hide that.

**The detector is a statistical model, not a network.** Element dropout, a
correlated lateral bias per element, range-dependent point noise, false-positive
clutter, and occasional class flips. The correlated bias matters most: eight
points of independent noise average out, and a whole-element offset does not —
which is the error that actually limits a real detector.

Every one of those numbers is a knob for an experiment. Realism is not the goal
here; controllability is.

## The invariant

Apply the true correction to the detections and they must land on the map.
`tests/test_data.py::test_true_correction_aligns_detections_onto_the_map` checks
it on five frames and fails above a 0.6 m median residual. If the labels drift,
every downstream metric keeps looking healthy while measuring the wrong thing.

## The ablation that matters

`mapposeformer/data/classes.py` states a claim:

| Class | Lateral | Longitudinal | Heading |
|---|---|---|---|
| lane divider, road boundary | strong | **~none** | strong |
| ped crossing, stop line | weak | **strong** | strong |
| pole, traffic sign | strong | **strong** | moderate |

Lane geometry runs *parallel* to travel, so sliding a hypothesis down the road
costs almost nothing. A model given only lane detections has an unobservable
degree of freedom, and no amount of training fixes it.

That is a claim, and it is testable:

```bash
tools/eval.py runs/base/best.pt --split test                          # baseline
tools/eval.py runs/base/best.pt --split test data.sample.keep_classes=[0,1]
```

Same checkpoint, less evidence. **Longitudinal RMSE should grow sharply while
lateral and heading barely move.** If it does not, either the model is not using
the landmarks it claims to, or something is leaking. Training a model on the
restricted classes as well (`configs/ablate_lane_only.yaml`) separates "cannot
see it at inference" from "never learned to".

## Cost

About 4.5 ms per sample on one CPU core, so eight dataloader workers keep an
A6000 fed. Nothing is cached to disk: a scene is 600 m of arcs and rebuilding it
is cheaper than reading it back.

## What is not here yet

No temporal dimension. Frames are drawn independently, so the model cannot yet
accumulate evidence across a sequence the way the classical repo's cost
aggregation does, and there is no closed-loop evaluation. Both are on the
roadmap and both are listed in [OPEN_ITEMS.md](OPEN_ITEMS.md).
