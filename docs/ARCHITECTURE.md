# Architecture

## The problem, in one line

A vehicle has a pose estimate that has drifted. It sees landmarks. It has a map.
**What correction turns the estimate into the truth?**

That is the same question [camera-map-localization](https://github.com/yanghuafang/camera-map-localization)
answers by searching a grid of pose hypotheses and scoring each against a
distance transform. This project answers it with a network, and deliberately
keeps the *interface* identical so the two backends are comparable frame for
frame.

## Per-frame flow

```
detections ─┐                                  ┌─ soft assignment ─ Procrustes ─ delta
            ├─ tokenize ─ L x [self | cross] ──┤
  local map ─┘                                 └─ attention pool ─ volume ─ logits, cov, trust
```

1. **Tokenize.** Each point of each polyline becomes one token, carrying a
   Fourier encoding of its position, its class, which element it belongs to, and
   where along that element it sits.
2. **Attend.** `L` layers, each doing self-attention within the detections,
   self-attention within the map, and cross-attention in both directions.
   Self-attention gives an element its context ("I am the third of four parallel
   lines"); cross-attention does the matching.
3. **Match.** A dual-softmax assignment matrix scaled by a learned
   per-point matchability. A point with no counterpart gets low matchability and
   contributes nothing.
4. **Solve.** Weighted Procrustes in SE(2), in closed form, from the assignment.
   The pose head has no parameters.
5. **Score.** In parallel, pooled features predict a cost volume over the same
   `(forward, left, yaw)` grid the classical search evaluates — which yields the
   measurement covariance and a trust flag.

## Frames, and why there is only one

Everything is in the **anchor frame**: X forward, Y left, yaw counter-clockwise,
metres and radians. This is `core::Frames`' vehicle convention, chosen for the
same reason — so that a translation of `(1, 0)` means *one metre forward*.

The map is cropped around the **prior pose** and expressed there. The detections
are expressed in the **true ego frame**, because that is where a sensor produces
them. The output is the transform between the two.

No world coordinate enters the network. That is not a convenience; it is what
makes the problem learnable. A model given absolute coordinates on a dataset
with a handful of map regions will memorise the regions, score well, and
localize nothing. `tests/test_anchoring.py` moves an entire world five kilometres
and rotates it, and asserts the input tensors are unchanged.

## Inputs

Per frame. Shapes are for the default config; all are static, because the
TensorRT milestone needs them to be.

| Input | Shape | Frame · units | Origin |
|---|---|---|---|
| `map_pts` | `(72, 8, 2)` | anchor (prior pose) · m | Map cropped to a 50 m radius, each element resampled to 8 points by arclength |
| `map_pmask` | `(72, 8)` bool | — | Which points are real. A pole is one valid point; a lane chunk is eight |
| `map_cls` | `(72,)` | — | `LandmarkClass` |
| `det_pts` | `(32, 8, 2)` | true ego · m | Landmarks inside a 100° × 60 m frustum, with the detector's error model applied |
| `det_pmask`, `det_cls` | as above | — | |

Carried in the sample but **never read by the model**:

| Field | Shape | Purpose |
|---|---|---|
| `delta` | `(3,)` | The training target: the correction, in the anchor frame |
| `prior` | `(3,)` | World-frame anchor, so evaluation can compose the prediction back |
| `gt` | `(3,)` | World-frame truth, for the same reason |

Element counts are budgets, and overflow drops the **farthest** elements rather
than an arbitrary tail — otherwise the cap would act as a random ablation.

## Outputs

| Output | Shape | Meaning |
|---|---|---|
| `delta` | `(B, 3)` | The correction `(x, y, yaw)` in the anchor frame. Compose onto the prior to get the estimate |
| `delta_match` | `(B, 3)` | The Procrustes solution. Identical to `delta` unless the regression head is selected |
| `delta_volume` | `(B, 3)` | The soft argmax of the cost volume — a second, independent estimate |
| `logits` | `(B, 4199)` | The learned cost volume over a 19 × 17 × 13 grid |
| `cov` | `(B, 3, 3)` | Measurement covariance, from the volume's softmax-weighted spread |
| `trust_logit` | `(B,)` | Learned replacement for the classical flat-surface and high-cost gates |
| `assign` | `(B, 256, 576)` | Soft correspondence, detection points × map points |
| `mass` | `(B,)` | Total assignment weight. Near zero means the pose is a *default*, not an estimate |

`(delta, cov, trust)` is exactly what `LocalizationKF::Update` consumes in the
classical repo. The filter, its gates and its evaluation all stay; only the
measurement source changes. That is what makes this a study rather than a
rewrite.

## Why a closed-form pose head

The transform is *determined* by the correspondences. Once the model has said
which detected point is which map point, there is nothing left to learn, and
solving it analytically has four consequences:

- All capacity goes to the question that is actually hard — correspondence.
- The head cannot overfit, because it has no parameters.
- The model can only be right for the right reason, and when it is wrong the
  assignment matrix says where.
- It quantizes to whatever precision the arithmetic runs at, and it cannot be
  pruned away.

`RegressionPoseHead` is kept as a baseline, not as an option. Training both and
comparing them in and out of distribution is one of the more instructive
experiments here — and it has now been run, with a result that contradicts the
paragraph above.

**Measured, the closed-form head loses.** In distribution the two are within 2%
on translation; out of distribution the regression head is seven times better,
because a rigid fit over a handful of wrong correspondences is unbounded and a
`tanh` is not. The first two bullets survive, and so does the third — `assign`
is computed either way, so interpretability is not what the closed form buys.
The accuracy argument is retracted; see [RESULTS.md](RESULTS.md). The open
repair is a robust solve, not the baseline.

## Why a cost volume as well

A single predicted pose cannot express "somewhere along this stretch of road",
and a stretch of parallel lane lines *should* produce a ridge. Three things fall
out of predicting the surface instead of only its peak:

- **Ambiguity is visible.** Looking at the surface is how you find out whether a
  model has localized or has learned the mean of the prior.
- **Covariance falls out.** Its softmax-weighted spread about the peak is the
  measurement covariance the filter needs — computed the same way the classical
  repo computes it, so the uncertainty and the evidence cannot disagree.
- **It survives quantization.** Classification over a grid degrades gracefully
  as precision falls; coordinate regression does not.

The grid extent must match the prior's truncation bounds, or a target lands
outside the grid and has no correct cell. `config._validate` refuses that
configuration rather than training it.

## Numerical care

Two things in this model produce NaN if left alone, and both are handled where
they arise rather than papered over downstream:

- **Empty attention.** A frame at the edge of a scene can have no visible map
  element. A softmax over an entirely masked key set is NaN, and `0 * NaN` is
  still NaN, so every mask applied afterwards is useless. Every token set is
  prefixed with a never-masked **null token**. `tests/test_model.py::test_empty_input_is_finite`
  is that token's reason to exist.
- **Degenerate rotation.** `atan2` at the origin is undefined and its gradient
  is unbounded near it — which is exactly the case where no rotation is
  observable. The Procrustes head returns zero rotation there, and reports
  `mass` so the caller can tell a default from an estimate.

## Module map

```
mapposeformer/
  geometry.py        SE(2). Everything else assumes it is right.
  data/
    classes.py       The six landmark classes and what each constrains.
    world.py         Procedural road generation.
    sample.py        World + frame -> the anchored input contract above.
    synthetic.py     Dataset, with splits as disjoint seed ranges.
  model/
    tokenizer.py     Points -> tokens. Fourier position, class, element, index.
    attention.py     Pre-norm blocks, and the null token.
    matcher.py       Dual-softmax assignment with learned matchability.
    pose_head.py     Weighted SE(2) Procrustes, closed form.
    volume_head.py   The learned cost volume, its covariance and trust score.
    model.py         The assembly. Start here.
  losses.py          Five terms; the interesting one is `match`.
  metrics.py         Error resolved onto ground-truth axes, with signed bias.
  engine/            The training loop and the evaluator, in full.
```
