# Roadmap

## The problem

A vehicle has a prior pose that is wrong by a metre or two, a map of surveyed
landmarks, and a perception system reporting landmarks it can see. Recover the
correction, and a covariance honest enough for a filter to trust.

The hard part is not the geometry. Given correct correspondences the pose is a
closed-form least-squares solve. The hard part is **association**: lane dashes
repeat every few metres and parallel lines are locally identical, so the nearest
map element to a detection is routinely the wrong one — and it is most often
wrong along the road, which is the axis the landmarks constrain worst.

## The architecture

    map points ───┐
                  ├ point encode ─ rope attention ─ assign ─ least squares ─ delta
    detections ───┘        4 layers, self + cross   │
      this frame + history                          └ curvature ─ cov
                                                      └ spread ─ ambiguity

**Point tokens.** One token per point, 1 344 of them. Pooling each element's
points into one token leaves 168 tokens and makes the attention 64× smaller —
and that saving is not the constraint: the assignment is 768 × 576 either way,
the model is loader-bound, and pooled tokens ran 10% *slower*. What pooling
costs is resolution. A detected road boundary spanning four 12 m map chunks
cannot say which chunk each of its points is on, and the pose needs that
answered at ten centimetres. Matching every point against every point scores
**95.9% recall against 83.1%** — 12.7 points, 6.8 pooled seed standard
deviations, three seeds a side. The saving was real; the resolution was worth
more.

**Relative geometric attention.** No absolute coordinate enters a token. The
score carries relative offset and bearing, as LightGlue's rotary encoding
carries relative keypoint position. Translate both sets and every score is
unchanged, so the invariance the task wants is there by construction rather
than learned from augmentation.

**Translation, and not rotation.** Full SE(2) equivariance is the tidier claim
and the wrong one. Translation is the case for it: the prior is wrong by up to
4.5 m, so a matcher that behaves identically everywhere on the map is exactly
what the task wants. Rotation is not. The prior fixes heading to within 3°, so
detections arrive very nearly aligned with the map, and a rotation-equivariant
score cannot distinguish a detected lane lying along a map lane from one
crossing it at 30° — the relative geometry is identical once each element is
read in its own frame. That is usable evidence traded away for an invariance
this task never exercises. Measured, the rotary encoding beats absolute
positions by **9.2 pp at 7.6 sd**, so the equivariance is load-bearing rather
than decorative; the fully equivariant bias needs an N × N tensor and costs
2.4× the step time for 2.5% of accuracy.

**Four layers, two heads of 64, `dim` 128.** Depth is the one structural axis
that separated without argument: four layers beat two by **9.8 pp at 18 sd**.
Width saturates — 128 beats 64 by 5.5 pp at 6.3 sd, and 256 buys −0.70 pp. Head
count does not separate at all, in either direction: 4×64 against 2×64 is 1.03
sd and 2×32 against 2×64 is 0.38 sd, against a pooled within-cell spread of
4.33 pp. 2×64 stays because the registration fixed the rule before the arms ran
— a null means neither knob is worth the parameters — even though 2×32 is 31%
cheaper.

**Point-to-point residuals**, and a zero-parameter weighted least-squares solve
with IRLS reweighting, three refinement passes. The head has no parameters, so
it cannot overfit, quantizes exactly, and makes the pose the minimum of a
stated objective rather than the output of a regressor.

The verdict is the calibration, not the accuracy: NEES 0.68–0.76 against the
line residual's 0.04–0.20, three seeds an arm, no overlap, where the recall gap
is +1.6 pp at only 2.0 sd. The price is a confidently-wrong tail ten to fifteen
times larger — 0.30–0.45% of frames against 0.01–0.06% — and M4 is where that
gets charged.

**A recorded negative result: the anisotropy argument.** It is the strongest
geometric argument in this design, and the measurement refused it. Matching a
detected point to a *map point* constrains both axes equally, and the objective
`Σ aᵢⱼ‖R dᵢ + t − mⱼ‖²` has translation Hessian `2·mass·I` — **isotropic,
whatever the landmarks look like**. Matching a point to the *line* it lies on
constrains only the perpendicular offset, so each correspondence contributes a
rank-1 Hessian and a set of mostly-parallel lane lines produces exactly the
ill-conditioning along the road that the errors show. Poles and signs stay
point-to-point; they are the landmarks that pin longitudinal position, and that
asymmetry is the whole observability story.

That much is checkable on geometry alone, with no network and no checkpoint —
build a straight road, fix the assignment, and invert the Hessian. Measured
that way, **point-to-point answers `σ_long/σ_lat` = 1.01 whatever the landmarks
are**, while point-to-line ranges from 16.6 down to 2.91 as along-track
evidence arrives. Lane lines alone make `H` singular rather than merely
ill-conditioned, with the null direction along the road — which is the honest
answer, and why the covariance path fuses the prior rather than inverting the
measurement on its own.

The learned assignment does not cooperate. At four layers the line residual
reports a covariance **7.7 to 17.7× too wide with its major axis 65 to 79°
out**, on all three seeds, where point-to-point holds 1.07–1.14× and 11–14°. A
covariance argument that is correct on a fixed assignment can still lose to one
that survives a learned assignment, and depth is what separates them.
[RESULTS.md](RESULTS.md) has both tables.

**A covariance from curvature.** Near the minimum the cost is `c_min + dᵀHd/2`,
so the pose is uncertain by how far `d` moves before the cost rises by the noise
in the cost itself: `2 s² H⁻¹`, with `s²` the weighted mean squared residual
`cost / (dof − 3)`. The divisor is what makes the covariance shrink as evidence
arrives — a raw sum cancels against `H` exactly. Derived, with nothing fitted —
so there is no scale to tune, and nothing that can be calibrated to look right
on the split it was tuned on.

**Ambiguity as a shape, not a gate.** A competing pose metres down the road is
the failure a single-frame confidence score cannot see, and it widens the
covariance along the direction the two candidates disagree on. It never drops
the frame.

Where that signal comes from matters. It is *not* the local minima of a
fixed-assignment cost surface: that surface is the conditional paraboloid of one
set of correspondences, and its extra minima are float noise on a flat floor —
measured at one grid cell of separation and 0.02% of the surface's range. Real
multimodality lives in the **assignment**, where a detection carries mass on two
map elements metres apart, or in a surface that re-associates at every
hypothesis, which is what the classical pipeline does and what costs `G` times
the matching.

## The experiments, in order

M3's two come first because they cost nothing: both run against checkpoints
that already exist, with no training.

| # | question | arms | decides |
|---|---|---|---|
| 0 | ~~how large is seed variance?~~ | **answered at six seeds, one config**: longitudinal CV 17.7% against lateral 0.67% and recall 0.13% — the variance is almost entirely along-track | the error bar every row below is read against |
| 1 | ~~does a point-to-line residual give the right covariance *shape*?~~ | **answered, and against the residual it was named for**: point-to-*point* is the calibrated one at point resolution — NEES 0.68–0.76 against line's 0.04–0.20, 3 seeds an arm, no overlap. `tools/refcov.py` is the instrument | whether the covariance can be honest at all — it can |
| 2 | does assignment spread catch the tail? | detector AUC on the 39 known-bad frames | ambiguity |
| 3 | ~~do pooled element tokens cost closed loop?~~ | **answered both ways**: point beats element by 12.7 points of recall open loop, 3 seeds each, and the point-token teacher closes the loop at 0.091 m | whether 64× cheaper attention is worth buying — it is not |
| 4 | ~~how much equivariance is worth having?~~ | **answered: 9.2 pp at 7.6 sd.** `rope` 0.9280 against `absolute` 0.8357, 3 seeds each, one shared configuration | the equivariance claim — **load-bearing, not decorative** |
| 5 | ~~how shallow can it go?~~ | **not below 4: two layers costs 9.8 pp at 18 sd** | depth — and capacity, however bought, wrecks the line residual's covariance tenfold |
| 6 | ~~what does point-to-point's tail cost in closed loop?~~ | **nothing it cannot afford.** 99.9% of frames accepted, longest refusal run 2 frames, no scene diverged; the 0.1% refused were refused on `mass` — too little map in view — not on a confidently-wrong pose | the calibration win **survives M4**, and is what produces the 2.98x |
| 7 | ~~does INT8 survive the loop?~~ | **it costs 2 mm.** 0.091 m fp32 against 0.093 m INT8, no divergence, 99.9% accepted, refusal runs *shorter* than fp32's | a separated −1.06 pp open loop is 2 mm on 91 mm closed. **Recall is a threshold metric and a filter is not** — quantize |
| 8 | what does the history buy? | with and without history, 3 seeds a side — a run of its own, since dropping it changes the input shape | whether temporal fusion earns its 3× sample cost |

## The bar for `main`

A change lands when its effect exceeds the seed variance M1 measures, on the
same protocol, with the arms differing by exactly one thing. Anything smaller
is unresolved, not negative.
