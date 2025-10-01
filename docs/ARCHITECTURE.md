# Architecture

A vehicle has a pose estimate that has drifted. It sees landmarks. It has a map.
**What correction turns the estimate into the truth?**

[camera-map-localization](https://github.com/yanghuafang/camera-map-localization)
answers by searching a grid of pose hypotheses against a distance transform.
This answers with a network, keeping the same interface so the two are
interchangeable.

## Per-frame flow

```
    map ──────────┐
                  ├─ tokenize ─ attend ─ assignment ─┬─ Procrustes ── delta
    detections ───┘                                  │
      this frame, and the last two                   └─ the same cost,
      warped here by egomotion                          on a grid ─── cov, trust
```

1. **Accumulate.** Past detections are warped into the current ego frame by
   measured egomotion and become more detection tokens.
2. **Tokenize.** One token per point, carrying a Fourier encoding of position,
   class, paint style, detector confidence and uncertainty, age, element index,
   and index within the element.
3. **Attend.** `L` layers of self-attention within each set and cross-attention
   both ways. Self-attention gives context ("third of four parallel lines");
   cross-attention matches.
4. **Match.** Dual-softmax assignment scaled by learned per-point matchability.
   A point with no counterpart gets low matchability and contributes nothing.
5. **Solve.** Weighted Procrustes in SE(2), reweighted twice against its own
   residuals.
6. **Refine.** Detections move into the frame the estimate implies and are
   matched again, at a tenth of the residual.
7. **Score.** The same assignment, evaluated across the hypothesis grid instead
   of at its minimum, gives the covariance and the trust flag.

## Frames

Everything is in the **anchor frame**: X forward, Y left, yaw
counter-clockwise, metres and radians — `core::Frames`' vehicle convention, so
`(1, 0)` means one metre forward.

The map is cropped around the **prior pose** and expressed there. Detections sit
in the **true ego frame**, where a sensor produces them. Past detections sit in
the ego frame of their own moment and reach this one through `hist_rel`.

No world coordinate enters the network. A model given absolute coordinates over
a handful of map regions memorises the regions, scores well, and localizes
nothing. `tests/test_anchoring.py` moves a world 5 km, rotates it, and asserts
the input tensors are unchanged — history and egomotion included.

## Inputs

Static shapes, because the TensorRT milestone needs them.

| Input | Shape | Frame · units | Origin |
|---|---|---|---|
| `map_pts` | `(72, 8, 2)` | anchor · m | Map within a 50 m radius, each element resampled to 8 points by arclength |
| `map_pmask` | `(72, 8)` bool | — | Real points. A pole is one; a lane chunk is eight |
| `map_cls` | `(72,)` | — | `LandmarkClass` |
| `map_attr` | `(72,)` | — | `MarkType`: solid, dashed, or unpainted |
| `det_pts` | `(32, 8, 2)` | true ego · m | Landmarks in a 100° × 50 m frustum, corrupted by the error model |
| `det_pmask`, `det_cls`, `det_attr` | as above | — | |
| `det_conf` | `(32,)` | — | Detector score. Overlaps between true and false |
| `det_sigma` | `(32, 2)` | m | Reported uncertainty: independent point noise, and whole-element offset |
| `hist_pts` | `(2, 32, 8, 2)` | past ego · m | The previous two frames |
| `hist_pmask`, `hist_cls`, `hist_attr`, `hist_conf`, `hist_sigma` | as above | — | |
| `hist_rel` | `(2, 3)` | anchor · m, rad | Measured `curr⁻¹ ∘ past`, with drift proportional to distance |

Carried in the sample and never read by the model: `delta` (the target),
`prior` and `gt` (world-frame, so evaluation can compose the prediction back).

Element counts are budgets. Overflow drops the farthest elements; dropping an
arbitrary tail would make the cap a random ablation.

**Global egomotion is not an input.** The classical engine uses it as a fallback
measurement when map matching fails — a filter's decision, not a network's.

## Outputs

| Output | Shape | Meaning |
|---|---|---|
| `delta` | `(B, 3)` | The correction `(x, y, yaw)` in the anchor frame |
| `deltas` | `(B, 2, 3)` | One estimate per refinement pass, for deep supervision. `deltas[:, -1]` is `delta` |
| `delta_match` | `(B, 3)` | The Procrustes solution; equals `delta` unless the regression head is selected |
| `delta_volume` | `(B, 3)` | Soft argmin of the cost surface — the same objective at grid resolution |
| `logits` | `(B, 4199)` | The cost surface over a 19 × 17 × 13 grid |
| `cov` | `(B, 3, 3)` | Measurement covariance, from the surface's softmax-weighted spread |
| `trust_logit` | `(B,)` | Learned replacement for the classical flat-surface and high-cost gates |
| `assign` | `(B, 768, 576)` | Soft correspondence, detection points × map points |
| `mass` | `(B,)` | Assignment weight the answer rests on. Gated on, not merely reported |
| `det_xy`, `det_valid` | `(B, 768, 2)`, `(B, 768)` | The detections actually matched, so the losses need not re-derive the assembly |

`(delta, cov, trust)` is what `LocalizationKF::Update` consumes. The filter, its
gates and its evaluation stay; only the measurement source changes.

## The cost surface is computed, not regressed

It was an MLP from a pooled 128-d vector to 4199 logits — 1.1 M parameters, a
third of the model, predicting a surface. A regressed ridge is what the prior
over ridges looks like, so it can appear without evidence, and a covariance read
off it predicts uncertainty instead of measuring it. It also generalizes like a
regressor: the ablated finding in
[SegLocNet](https://arxiv.org/abs/2502.20077), and why
[OrienterNet](https://arxiv.org/abs/2304.02009) matches exhaustively.

Computing it is free. The assignment-weighted squared error

```
C(R, t) = Σ_ij a_ij ‖R d_i + t − m_j‖²
```

looks like a sum over `4199 × 768 × 576` and is not: `‖R d‖² = ‖d‖²` under
rotation, and every remaining term factors through eleven numbers — the total
mass, two weighted second moments, two weighted centroids, and the 2 × 2
cross-covariance `M = Σ a_ij d_i m_jᵀ`. Those are the statistics weighted
Procrustes already forms to find the minimum, so the whole grid costs a few
2 × 2 matmuls. `tests/test_model.py` checks the identity against brute force.

Two consequences hold: the covariance is the measured curvature of the fit, and
the two output paths cannot disagree — a test asserts the pose lands within half
a cell of the grid's argmin.

### What it does not do

A third claim — that the surface would show aliasing as a ridge along the road —
is false. With correspondences held fixed the cost rises in every direction, and
faster along track than across, which is backwards from the physics. Only
re-association shows the asymmetry. [RESULTS.md](RESULTS.md) measures all three
surfaces and explains why.

Two different quantities, both useful:

- **Fit curvature**, computed here: how well the pose is determined *given* that
  these correspondences are right. The standard least-squares covariance.
- **Correspondence ambiguity**, which produces the ridge: whether the pose could
  be elsewhere and look as good. Needs each detection to re-choose its map point
  per hypothesis — what a distance transform does, and what breaks the
  factorisation the closed form depends on.

So the reported covariance is optimistic where the map aliases, which is where
it most needs to be right. `tools/viz_volume.py --mode fit` and `--mode reassoc`
draw both; [ROADMAP.md](ROADMAP.md) has the affordable repair.

The grid extent must match the prior's truncation bounds, or a target has no
correct cell. `config._validate` refuses that configuration.

## The pose head, and why it needed repairing

The correspondences determine the transform, so the head solves it analytically
and has no parameters: capacity goes to matching, the head cannot overfit or be
pruned away, and a wrong answer is legible in the assignment matrix.

**The first version lost to its own baseline.** In distribution it and
`RegressionPoseHead` were within 2% on translation; out of distribution the
regressor won seven to one, and one ablation produced 30.9° of heading error. A
plain weighted least-squares fit is unbounded, so a few confident wrong
correspondences move it arbitrarily far, where a `tanh` against the grid extent
cannot. [RESULTS.md](RESULTS.md) has the table.

The repair:

- **Abstention is a gate, not a discount.** A detection that matched far worse
  than the rest of its frame contributes nothing. The threshold is relative to
  the frame's strongest match — the matcher's absolute calibration moves two
  orders of magnitude across training, and an absolute threshold gates every row
  at step one.
- **Reweighting.** Geman-McClure, twice, unrolled so each pass carries gradient.
- **`mass` is gated on.** A pose resting on three confident wrong matches looks
  like one resting on three right ones to the features that produced it, so the
  trust head cannot see the difference. `mass` is arithmetic and answers a
  different question.

Whether the robust solve now beats the baseline is unrun — see
[OPEN_ITEMS.md](OPEN_ITEMS.md).

## Refinement

One pass must solve correspondence at the prior's error — 1.5 m along track,
nearly half a lane spacing. A second pass, on detections moved into the frame
the first estimate implies, works at an order of magnitude less residual.
Shared weights, so no parameters; 2.0× the step time.

The warp is **detached**: it re-anchors the tokenizer's view, and gradient
through a chain of warps would make each pass responsible for the ones after it.
Nothing is *solved* on the moved coordinates — every pass reads the original
points, so each yields a total correction, and the cost surface stays anchored
to the prior. Every pass is supervised on RAFT's schedule.

## Temporal fusion

What accumulates is evidence about the pose **error**, not about a pose. Every
anchor is the same drifting estimate at a different time, and

```
A_t⁻¹ ∘ G_{t−k}  =  (A_t⁻¹ ∘ G_t) ∘ (G_t⁻¹ ∘ G_{t−k})  =  delta ∘ rel_ego
```

so a past detection reaches the current ego frame through `rel_ego` alone, which
odometry measures. The unknown `delta` then aligns the accumulated set at once.

Fusion is therefore more detection tokens, warped, with an age embedding — not a
recurrence and not a state, so static shapes survive. `hist_rel` carries drift
at 1% of distance travelled and 0.02°/m; noiseless egomotion would let the model
fuse an arbitrarily long history for free.

## Numerical care

- **Empty attention.** A frame at the edge of a scene can have no visible map
  element, and a softmax over an entirely masked key set is NaN that survives
  every later mask. Every token set is prefixed with a never-masked null token.
- **Degenerate rotation.** `atan2` at the origin is undefined with unbounded
  gradient — exactly the case where no rotation is observable. The head returns
  zero rotation there and reports `mass`.
- **The surface's common offset.** Every cell of `C(R, t)` shares a large
  constant and varies by a few percent of it. The cost is divided by mass,
  making it a mean squared residual, and shifted by its own minimum.

## Module map

```
mapposeformer/
  geometry.py        SE(2). Everything else assumes it is right.
  data/              Procedural worlds, the sample contract, the dataset
  model/             Tokenizer, attention, matcher, pose head, volume head
  engine/            The training loop and the evaluator
  losses.py          Five terms; `match` is the one that carries
  metrics.py         Pose error, and whether the covariance is honest
```
