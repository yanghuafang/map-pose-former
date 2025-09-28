# Data

## Why the first dataset is generated

KITTI Odometry, which the classical project uses, ships no HD map; that repo
synthesizes one by offsetting the ground-truth path. For a search backend that
is a mild oracle. For a learned one it is fatal — the map becomes a function of
the pose being predicted, and a transformer will invert it.

So stage 0 is procedural, which buys three things no public dataset offers:

- **The answer is exact**, with no localization pipeline between label and truth.
- **The evidence is controllable.** Turning poles off is one config line, and so
  are the dashes and the detector's confidence.
- **Splits cannot leak**: disjoint seed ranges, so no two splits share a road.

Stage 1 is nuScenes, with real detections — see [ROADMAP.md](ROADMAP.md).

## Getting nuScenes

Accept the Terms of Use at <https://www.nuscenes.org/nuscenes#download> first;
`scripts/download_nuscenes.sh` cannot.

```bash
scripts/download_nuscenes.sh --dry-run   # check URLs, transfer nothing
scripts/download_nuscenes.sh             # maps + metadata, ~0.5 GB
scripts/download_nuscenes.sh --blobs     # camera archives, ~350 GB
```

Both halves are needed. The map and metadata give the real map; the camera
archives are what a pretrained mapper reads to produce real detections. The
script checks every URL before transferring a byte, resumes, and names the files
it expects — an archive unpacked into the wrong directory otherwise looks like
success.

## The world

`data/world.py` builds one scene per seed: 600 m of piecewise-constant-curvature
arcs, four lanes, every landmark class.

| Class | Geometry | Attribute | Placement |
|---|---|---|---|
| `LANE_DIVIDER` | polyline, 12 m chunks | solid or dashed | between lanes |
| `ROAD_BOUNDARY` | polyline, 12 m chunks | solid | road edges |
| `POLE` | single point | — | roadside, mean 22 m spacing |
| `TRAFFIC_SIGN` | single point | — | roadside, mean 70 m spacing |
| `STOP_LINE` | polyline, perpendicular | — | intersections, near half of the road |
| `PED_CROSSING` | polyline, perpendicular | — | intersections, full width |

Arcs rather than a curvature random walk: a road that wobbles at the 2 m
sampling pitch gives the model a texture no real road has. Chunked polylines
because real vector maps are chunked, and because one 600 m element would be a
token whose points are mostly out of view.

`intersection_spacing_m = 110` is the most consequential number in the file:
stop lines and crossings are the only along-track evidence a nuScenes-style map
carries, so it sets how often the longitudinal degree of freedom is observable.

## The dashes

A dashed lane line is paint with ends, and a stripe end is a point feature in a
class otherwise blind to along-track position. Real vector maps discard those
ends: nuScenes and Argoverse 2 both store a continuous polyline plus a
`mark_type` attribute.

This dataset reproduces that asymmetry. The map stores the polyline and the
attribute; the detector sees the stripes.

```
world:   ────────────────────────────────   continuous, attribute DASHED
map:     ────────────  ────────────         chunked at 12 m, still continuous
det:     ───   ───   ───   ───              3 m of paint, 6 m of gap
```

The stripe phase is keyed to the element's first world point, not to the crop.
Keyed to the crop, the pattern would slide with the vehicle and become a cue
about where the *frustum* is — which would look like along-track skill and be
the opposite. `_crop` returns whole elements for this reason.

`data.sample.stripe_dashed=false` removes the stripes without touching the map:
same attribute, no paint pattern.

## One frame

`data/sample.py`. Two point sets and the transform between them:

```
prior = gt ∘ error          error ~ truncated Gaussian, anisotropic
delta = prior⁻¹ ∘ gt        the training target
map   = world ∩ radius(prior),  in the prior's frame
det   = world ∩ frustum(gt),    in the true ego frame, corrupted
```

**The prior error is anisotropic**: 1.5 m along track, 0.6 m across, 1.0° of
heading. Dead reckoning drifts fastest along travel, and along-track is also the
direction lane geometry cannot see — the hard axis and the weak evidence are the
same axis. A symmetric prior would hide that.

**The detector is statistical, not a network.** Element dropout, correlated
lateral bias per element, range-dependent point noise, clutter, occasional class
flips, a confidence, and a two-part uncertainty. The correlated bias matters
most: eight points of independent noise average out, a whole-element offset does
not, and that is the error that limits real detectors.

**Confidence and uncertainty overlap between true and false.** True detections
score ~0.92 at zero range falling to ~0.62 at 50 m; clutter scores ~0.45. A
score that separated them cleanly would be answering the question the model is
being asked.

Every number here is a knob. Realism is not the goal; controllability is.

## The invariants

Apply the true correction to the detections and they land on the map:
`test_true_correction_aligns_detections_onto_the_map` fails below a 0.75 inlier
fraction. If the labels drift, every downstream metric looks healthy while
measuring the wrong thing.

## Ablations

`data/classes.py` states a claim:

| Class | Lateral | Longitudinal | Heading |
|---|---|---|---|
| lane divider, road boundary | strong | **~none** | strong |
| ped crossing, stop line | weak | **strong** | strong |
| pole, traffic sign | strong | **strong** | moderate |

Lane geometry runs parallel to travel, so sliding a hypothesis down the road
costs almost nothing. A model given only lane detections has an unobservable
degree of freedom that no amount of training fixes.

Four experiments; only the last needs its own training run, because the rest
share weights with the base checkpoint:

```bash
tools/eval.py runs/base/best.pt --split test 'data.sample.keep_classes=[0,1]'
tools/eval.py runs/base/best.pt --split test data.sample.stripe_dashed=false
tools/eval.py runs/base/best.pt --split test model.refine_iters=1
```

Longitudinal RMSE should grow sharply under the first while lateral and heading
barely move. If it does not, the model is not using the landmarks it claims to,
or something is leaking. Training on the restricted classes
(`configs/ablate_lane_only.yaml`) separates "cannot see it at inference" from
"never learned to".

## Cost

A sample cuts detections from three frames, so it costs roughly three times what
it did. Nothing is cached to disk: rebuilding 600 m of arcs is cheaper than
reading it back. Run `tools/bench.py` before assuming where the time goes.

## Not here yet

No closed-loop evaluation, and no real perception. Both are on the roadmap and
in [OPEN_ITEMS.md](OPEN_ITEMS.md).
