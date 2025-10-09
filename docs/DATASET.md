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

Stage 1 is nuScenes, which M5 reaches in two halves: first the generated map
is replaced with a surveyed one and the error model stays, then the error
model is replaced with a real detector — see [ROADMAP.md](ROADMAP.md).

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

### The tensors

Both sources emit this dict, at these shapes, so nothing downstream branches on
which one produced it. The shapes are fixed, which means padded: `*_pmask` says
which points are real, and a model that ignores it is fitting the padding.
Metres and radians throughout, in the vehicle convention `geometry.py` states.

| | shape | frame |
|---|---|---|
| `map_pts` | `(72, 8, 2)` | prior — the map is cropped where the vehicle *believes* it is |
| `map_pmask` | `(72, 8)` bool | |
| `map_cls`, `map_attr` | `(72,)` | class, and solid or dashed |
| `det_pts` | `(32, 8, 2)` | true ego — this frame's detections |
| `det_pmask` | `(32, 8)` bool | |
| `det_cls`, `det_attr` | `(32,)` | as *reported*, so occasionally flipped |
| `det_conf` | `(32,)` | the detector's score |
| `det_sigma` | `(32, 2)` | its claimed noise: per-point, and the whole-element bias |
| `hist_*` | `(2, 32, …)` | each past ego — the same six fields, 2 frames back |
| `hist_rel` | `(2, 3)` | odometry `gt⁻¹ ∘ gt_past`, drift included |
| `delta` | `(3,)` | **the target**, `prior⁻¹ ∘ gt` |
| `prior`, `gt` | `(3,)` | world; for evaluation, never an input |

72 map elements covers a 50 m crop with margin, and 8 points is what a 12 m
polyline chunk needs. Both are budgets rather than measurements: an element past
the cap is dropped, so raising the crop without raising the cap silently loses
map.

## The invariants

Apply the true correction to the detections and they land on the map:
`test_true_correction_aligns_detections_onto_the_map` fails below a 0.75 inlier
fraction. If the labels drift, every downstream metric looks healthy while
measuring the wrong thing.

The history has the same invariant, conjugated:
`test_history_lands_on_the_map_through_egomotion` warps past detections through
`hist_rel` then `delta` and requires them to align as well as the current frame
— 0.927 for the history against 0.921 for this one, meaned over 21 `train`
samples at the 1 m match radius. A sign error there would train happily and
silently turn the past into noise.

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

Two of these are pure data knobs: they change what the model is shown without
changing the shape of it, so one checkpoint answers both.

| override | asks |
|---|---|
| `data.sample.keep_classes=[0,1]` | lanes only — is along-track error unobservable without point landmarks? |
| `data.sample.stripe_dashed=false` | do dashes carry information, or only repeat? |

Longitudinal RMSE should grow sharply under the first while lateral and heading
barely move. If it does not, the model is not using the landmarks it claims to,
or something is leaking.

Two more need a run of their own. Dropping the history changes the input shape,
so a model trained with it cannot be evaluated without it. And *training* on the
restricted classes, rather than only evaluating that way, separates "cannot see
it at inference" from "never learned to".

