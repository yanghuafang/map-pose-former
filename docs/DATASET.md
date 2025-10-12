# Data

## Why the first dataset is generated

KITTI Odometry, which the classical project uses, ships no HD map; that repo
synthesizes one by offsetting the ground-truth path. For a search backend that
is a mild oracle. For a learned one it is fatal — the map becomes a function of
the pose being predicted, and a transformer will invert it.

So stage 0 is procedural, which buys three things no public dataset offers:

- **The answer is exact**, with no localization pipeline between label and truth.
- **The evidence is controllable.** Turning poles off is one config line, and so
  are the dashes, the history, and the detector's confidence.
- **Splits cannot leak**: disjoint seed ranges, so no two splits share a road.

Stage 1 is nuScenes: M2a replaces the generated map with a surveyed one and
leaves the error model in place, M2b replaces the error model with a real
detector — see [ROADMAP.md](ROADMAP.md).

## Getting nuScenes

Accept the Terms of Use at <https://www.nuscenes.org/nuscenes#download> first;
`scripts/download_nuscenes.sh` cannot.

```bash
scripts/download_nuscenes.sh --dry-run   # check URLs, transfer nothing
scripts/download_nuscenes.sh             # map, poses, CAN bus: 1.6 GB
scripts/download_nuscenes.sh --blobs     # ...and 316 GB of camera and lidar
```

**The default is all M2a needs**, because perception is an input here:
detections are cut from the map and corrupted, and no detector is ever run. The
blobs matter at M2b, where a pretrained mapper reads the 53 GB of keyframe
images inside them; the other 263 GB are sweeps and lidar that nothing here
reads.

The script checks every URL before transferring a byte, resumes, and names the
files it expects — an archive unpacked into the wrong directory otherwise looks
like success.

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
crossings and traffic lights are the only along-track evidence a nuScenes-style
map carries, so it sets how often the longitudinal degree of freedom is observable.

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

## The split

Geographic, not the official one: nuScenes' own train and val share roads, so a
localizer evaluated on them has seen the ground it is tested on. Each city is
cut along its longest axis with dead ground between the bands, and a scene joins
a split only if its **whole trajectory** fits inside one — a scene that starts
in train's band and ends in val's belongs to neither.

Measured separation between test and train is **307 m**, six times the 50 m map
query radius, so no map element is shared. That comparison is made **per city**:
every nuScenes map has its own origin, so scenes on different continents sit at
the same coordinates and comparing them together reports a collision that is not
there. `tests/test_nuscenes.py` holds both properties.

720 of 850 scenes survive; the rest straddle a boundary and are dropped.

## What nuScenes actually carries

The map expansion stores areas as polygons, and a polygon is not what a
localizer can use. Each class needs the geometry its label denotes, or the model
is told the wrong thing rather than nothing:

| nuScenes layer | read as | why |
|---|---|---|
| `lane_divider`, `road_divider` | polyline | already a line |
| `drivable_area` | closed outline | a boundary *is* a line |
| `ped_crossing` | the polygon's long axis | an elongated area; its axis is the bar across the road |
| `traffic_light` | a point | the only point landmark nuScenes has, and sparse -- one within 120 m of about a fifth of scenes |
| `stop_line` | **not read** | see below |

**Stop lines are dropped, and the reason is the useful part.** nuScenes
annotates a stop *zone*, not a stop *bar*: 83% of the polygons are rounder than
2:1, median aspect 1.5, so there is no direction to extract. Closing them into
outlines put a third of their segments *along* the road under a label asserting
they run across it. Evidence that is mislabelled is worse than evidence that is
absent: an omitted class costs the model what it knew, a mislabelled one teaches
it something false about every other member of that class.

The general lesson outlives nuScenes: **a class is a claim about what geometry
constrains**, and an ingest that satisfies the label while violating the claim
is harder to find than one that simply omits the class, because everything
downstream keeps working.

## The map is not the detector

The stored map is chunked at survey boundaries; detections are cut out of the
continuous world by the frustum, whose ends sit at a fixed *range* and so carry
no information about position along the road. If both sides were cut the same
way they would share element endpoints at fixed world positions, and a shared
endpoint is a perfect along-track landmark — a model given only lane geometry
would localize along the road from an artefact of how the polylines were cut.

The nuScenes reader did exactly that for its first two runs: it chunked once at
ingest and handed the result to `build_sample` as both arguments, 1088 source
elements against 1088 map elements. Both datasets now derive the map with
`chunk_for_map` and cut detections from the unchunked world;
`test_the_map_and_the_detection_source_are_chunked_apart` fails if that stops
being true. [RESULTS.md](RESULTS.md) has what it cost.

## One frame

`data/sample.py`. Two point sets, a history, and the transform between them:

```
prior = gt ∘ error          error ~ truncated Gaussian, anisotropic
delta = prior⁻¹ ∘ gt        the training target
map   = world ∩ radius(prior),  in the prior's frame
det   = world ∩ frustum(gt),    in the true ego frame, corrupted
hist  = the same, from gt[frame − k·stride], in *its* ego frame
rel   = gt⁻¹ ∘ gt_past,     as odometry measures it, with drift
```

**The prior error is anisotropic**: 1.5 m along track, 0.6 m across, 1.0° of
heading. Dead reckoning drifts fastest along travel, and along-track is also the
direction lane geometry cannot see — the hard axis and the weak evidence are the
same axis. A symmetric prior would hide that.

**Clutter draws its class from what the frame actually detected, with
multiplicity.** This looked like a detail and was not, and it took two attempts:

- *Uniform over the enum.* On nuScenes, which has no poles and whose stop-line
  annotation is unusable, a third of clutter carried classes the map cannot
  supply, so it never matched anything — and restricting `keep_classes` then
  removed that garbage along with the evidence.
- *Uniform over the classes the scene contains.* Equal clutter on unequal
  populations contaminates rare classes hardest. Measured on the nuScenes test
  split: 10.0 road boundaries per frame against 0.7 traffic signs, 0.5 clutter
  elements each, so **4.7% noise on one class and 42.9% on the other** — and the
  rare classes are the along-track anchors the ablation exists to weigh.

Drawing with multiplicity equalises the *ratio* instead of the count, leaving
every class near `clutter_mean / total`: 7.5%, 13.2%, 8.6% and 6.2% on the four
classes above. All three rules agree on generated scenes, which hold every class
in comparable numbers. A real map is not balanced, and the observability
ablation is the instrument sensitive enough to notice.

**The detector is statistical, not a network.** Element dropout, correlated
lateral bias per element, range-dependent point noise, clutter, occasional class
flips, a confidence, and a two-part uncertainty. The correlated bias matters
most: eight points of independent noise average out, a whole-element offset does
not, and that is the error that limits real detectors.

**Confidence and uncertainty overlap between true and false.** True detections
score ~0.92 at zero range falling to ~0.62 at 50 m; clutter scores ~0.45. A
score that separated them cleanly would be answering the question the model is
being asked.

**Egomotion drifts** at 1% of distance travelled and 0.02°/m — about 4 cm and
0.08° over the 4 m between history frames. Small against the 1.5 m prior, which
is what makes accumulating worthwhile. Noiseless egomotion would be an oracle.

Every number here is a knob. Realism is not the goal; controllability is.

## The invariants

Apply the true correction to the detections and they land on the map:
`test_true_correction_aligns_detections_onto_the_map` fails below a 0.75 inlier
fraction. If the labels drift, every downstream metric looks healthy while
measuring the wrong thing.

The history has the same invariant, conjugated:
`test_history_lands_on_the_map_through_egomotion` warps past detections through
`hist_rel` then `delta` and requires them to align as well as the current frame
— 0.904 against 0.904. A sign error there would train happily and silently turn
the past into noise.

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
tools/train.py --config configs/ablate_no_history.yaml
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
