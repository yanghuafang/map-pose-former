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

## What it achieves

Test split, 8 880 generated frames, seeds disjoint from training. The baseline
is the point: 0.271 m means nothing until you know it started at 1.611 m. The
last column scores the covariance rather than the pose — the error measured in
units of the uncertainty claimed for it, where 0.789 is calibrated, above it
over-confident and below it cautious.

| | translation | recall @ 25 cm, 0.5° | NEES |
|---|---|---|---|
| do nothing — the prior's own error | 1.611 m | 1.3% | — |
| **this model, one frame at a time** | **0.271 m** | **96.3%** | 0.909 |
| the same, through the filter | **0.091 m** | — | — |

Closing the loop is worth **2.98x**, with no scene diverged and 99.9% of frames
accepted — and it is worth that *because* the covariance is honest. A model with
the same accuracy and a covariance 2.3x too wide scores 0.188 m on the same
scenes, and looks identical in every open-loop table.

And what it costs to make it small. Same test split, one seed throughout — the
student rows are seed 0 of the three in [docs/RESULTS.md](docs/RESULTS.md):

| | params | recall | closed loop | weights |
|---|---|---|---|---|
| teacher | 1.709 M | 96.3% | 0.091 m | 6.52 MiB |
| distilled student | 0.914 M | 96.7% | **0.091 m** | 3.5 MiB |
| the same student, no teacher | 0.914 M | 95.1% | 0.106 m | 3.5 MiB |
| teacher, pruned 15%, fine-tuned | 1.45 M | 97.4% | not measured | 5.5 MiB |
| teacher, INT8 | 1.709 M | 95.2% | 0.093 m | **1.69 MiB** |

**A distilled student matches the teacher at 53% of the parameters.** Everything
else lands within 2 mm of it, which is the real finding: compiled to TensorRT
fp16 the *solve* is 85% of the frame (RTX 2070, batch 1), so compressing the
network is capped at 1.17x however well it is done. Compress for memory, not
for speed.

The pruned arm's +1.1 points is the learning-rate restart, not the pruning — a
fine-tuned control that prunes nothing gains +0.91 pp on the same paired test
split, against a half-width of 0.33. Pruning here is free, not profitable.

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

## The covariance is half the output

A pose without an uncertainty can only be *followed*. A pose with an honest one
can be **fused** — weighted against odometry, against the last frame, against
anything else the vehicle knows. That is the difference between a number and a
measurement, and it is why the classical counterpart searches a grid: the cost
surface it builds *is* the uncertainty. A regressor that emits three numbers
throws that away and asks the filter to trust it blindly.

**Honest is not the same as small.** Under-confidence is wasteful — the filter
converges slower than it could. Over-confidence is dangerous: a filter told a
bad frame is certain will follow it off the road, and no downstream gate can
undo that. So the covariance is scored, not assumed, and `metrics.py` scores it
four ways:

| | |
|---|---|
| NEES median | **0.789 is calibrated**, not 1.0 — a χ² with 3 degrees of freedom has mean 3 and median 2.366, and one threshold cannot serve both |
| ANEES | the mean, which one catastrophic frame moves a long way |
| coverage | the *count* inside the 95% ellipsoid, which it does not |
| tail | the fraction confidently wrong — the frames that actually hurt |

And a warning worth stating at the front door: **a calibrated scalar can hide a
tail.** A NEES median of 0.786 against the 0.789 target — measured on a
2.28 M-parameter sibling of this network, same 8 880-frame test split, but with
a *fitted* covariance scale rather than the curvature below — is honest by every
test above, and the covariance that scored it was 20.6× overconfident on the
worst 1% of frames — 2.249 m of error against a claimed 0.308 m — while the
other 99% sat at 0.80×, slightly conservative
([the tail](docs/RESULTS.md#the-covariance-can-be-calibrated-and-still-be-wrong)).
Aggregate like for like, too: a median of per-frame ratios on one side against a
ratio of RMS on the other manufactures an anisotropy inversion that is not
there. So the tail is reported beside the median, per axis.

This design derives the covariance from the **curvature of the cost** it just
minimised — `2 s² H⁻¹`, with `s²` the weighted mean squared residual
`cost / (dof − 3)` — rather than fitting a head to predict one. There is no
scale to tune and nothing that can be calibrated to look right on the split it
was tuned on. `metrics.py` judges it and `filter.py` consumes it.

**Where the data comes from.** Stage 0 is generated, because a learned
localizer needs a setting where the answer is *known* and the evidence can be
*controlled* before it needs realism — turning poles off and watching
along-track error explode is a two-line experiment here and an impossible one
on a public dataset. nuScenes arrives at `M5`. Both sources emit identical
tensors, so nothing downstream can tell a generated road from a surveyed one.
[docs/DATASET.md](docs/DATASET.md) has the generator, the detector's error
model, and the one asymmetry the whole thing depends on.

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

## Three ideas worth taking

**Compute the pose, do not regress it.** Correspondences determine the
transform, so the pose head is a zero-parameter weighted Procrustes with a
damped Gauss-Newton refinement, and all capacity goes to matching.

**Take the covariance from the curvature of the cost you just minimised.**
Nothing is fitted, so the measured question is whether a free covariance is
good enough for a filter — and it is: it collects the full 2.98x.

**Recall is a threshold metric and a filter is not.** INT8 costs a
statistically separated −1.06 points of frame recall and 2 mm of closed-loop
error — a factor of fifty between what the frame-level metric charges and what
the vehicle pays. A frame that misses a 250 mm gate by 2 mm scores as a total
loss while contributing 2 mm. The corollary bites the other way too:
across six seeds a side, distillation gains 1.58 points of recall and does not
separate — t = 1.46 — while its NEES moves 0.497 to 0.691 against an honest
0.789 at t = 2.98, six seeds of six. The teacher transfers calibration, not
accuracy, and closed loop that is worth 15 mm where the frame-level gap is
noise. Measure a perception module by what consumes it.

## Where this is

Open loop and closed loop are both measured, on generated data; real data is
next. A result lands here when its effect is larger than the seed variance
measured for this configuration, on the same protocol, with the two runs
differing by exactly one thing.

**And a single number is one draw, not a constant** — six trainings of one
configuration, differing only by seed, span 0.226 m to 0.336 m of open-loop
translation error around a mean of 0.258 m. Any difference smaller than that
band is unresolvable, so comparisons here run three seeds a side.

[docs/RESULTS.md](docs/RESULTS.md) has that measurement and what it costs every
other comparison. `docs/ROADMAP.md` argues each design choice and names the
measurement that settled it.

## What is here

| | |
|---|---|
| `mapposeformer/data/` | synthetic worlds, nuScenes, the detector contract |
| `mapposeformer/model/` | the encoders, the attention, the two-stage assignment |
| `mapposeformer/solve.py` | the pose and its covariance — no parameters |
| `mapposeformer/losses.py` | what the assignment is trained on, and why the pose loss is not enough |
| `mapposeformer/engine/` | train, evaluate, and run a sequence closed loop |
| `mapposeformer/geometry.py` | SE(2) compose, inverse, relative, transform |
| `mapposeformer/metrics.py` | pose error, and whether the covariance is honest |
| `mapposeformer/filter.py` | the SE(2) Kalman filter corrections are fed back through |
| `mapposeformer/config.py` | YAML plus `section.field=value` overrides |
| `scripts/` | environment, CI, and running things on a remote GPU box |
| `tools/bench.py` | how fast the data arrives, which bounds the step |

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
