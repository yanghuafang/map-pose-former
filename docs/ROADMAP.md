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

## The bar for `main`

A change lands when its effect exceeds the seed variance M1 measures, on the
same protocol, with the arms differing by exactly one thing. Anything smaller
is unresolved, not negative.
