# map-pose-former

**Where is the car, given what the camera sees and what the map says?**

Match detected landmarks against an HD map, solve the rigid transform in closed
form, and report a pose correction with a covariance honest enough for a filter
to trust. The learned counterpart to
[camera-map-localization](https://github.com/yanghuafang/camera-map-localization),
which answers the same question by searching a grid of pose hypotheses. Same
input contract, same frame conventions, same error metrics, so the two backends
are comparable rather than merely similar.

Written to be **read**. Every non-obvious decision carries its reason, every
number carries the protocol that produced it, and where a measurement overturned
the argument that motivated it, the measurement wins and says so.

![one frame of the task](docs/img/frame.svg)

Grey is the map, cropped around the drifted prior — where the vehicle *believes*
it is. Red is what the camera reports this frame, in the frame it is actually
in. Orange is the previous two frames warped here by odometry: that it lands on
top of the red is the temporal path checking its own arithmetic. Green is every
detection after the true correction, sitting on the map — that correction is
what the model has to produce. Short red stripes are dashed lane paint, which
the map stores as an attribute and the detector sees as geometry.

The gap between red and green is the whole problem. `tools/viz_sample.py` draws
it for any frame.

## What goes in, what comes out

One frame at a time. Every tensor is fixed-shape and padded, and no absolute
world coordinate is anywhere in it — the model cannot memorise a city instead
of learning to match.

| in | | |
|---|---|---|
| map elements | `(72, 8, 2)` | polylines and points, **in the prior's frame** — the map is cropped where the vehicle *believes* it is |
| detections | `(32, 8, 2)` | what perception reports this frame, in the true ego frame, with class, confidence and its own claimed noise |
| history | `(2, 32, 8, 2)` | the previous two frames' detections, plus the odometry that relates them |

| out | | |
|---|---|---|
| `delta` | `(3,)` | the correction `(x, y, yaw)` in the prior's frame, metres and radians: `compose(prior, delta)` is where the vehicle is |
| covariance | `(3, 3)` | how much to trust it — what a filter needs to fuse rather than follow |
| mass | scalar | how much evidence the answer rests on |

`delta` is the whole task. The prior is wrong by 1.5 m along track, 0.6 m
across and 1° of heading; recovering that is what is being learned.

## How it localizes

Association is the hard part, not geometry. Given correct correspondences the
pose is a closed-form least-squares solve; what is difficult is that lane
dashes repeat every few metres and parallel lines are locally identical, so the
nearest map element to a detection is routinely the wrong one — and wrong most
often *along* the road, the axis the landmarks constrain worst.

So the network spends its parameters on matching and none on the pose:

```
  map elements ──── encode ──┐
                             ├── match ── assign ── Procrustes ── delta
  detections    ──── encode ─┘                          │
  this frame + 2 warped here                            └── curvature ── cov
```

One token per *point* rather than per element, a rotary encoding that carries
relative geometry into the attention scores with no N × N bias tensor, a
partial assignment that lets a detection match nothing at all, and then a
**zero-parameter** weighted Procrustes solve — which cannot overfit, quantizes
exactly, and makes a wrong pose a *visible* wrong assignment.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) walks the whole network — the
shapes at each stage, the Procrustes derivation, and why the robust weight has
to be annealed rather than switched on. [docs/ROADMAP.md](docs/ROADMAP.md)
argues for each choice and says what it still has to prove.

## Getting started

```bash
./scripts/setup.sh                              # conda environment
./scripts/ci.sh                                 # lint and tests
tools/viz_sample.py                             # draw one frame of the task
tools/train.py --config configs/synth_base.yaml # the reference run
tools/eval.py runs/base/best.pt --split test    # score it
```

`docs/DATASET.md` describes what a sample contains and the ablations it exists
for. `docs/ROADMAP.md` is the plan and the reasoning behind it.
