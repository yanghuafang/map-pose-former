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

