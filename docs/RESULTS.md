# Results

Every number here is measured, and every table says what it was measured on.

**The result as it stands.** Point tokens beat element tokens by **12.7 points
of recall** — 95.9% against 83.1%, three seeds each, 6.8 seed standard
deviations. That is by far the largest architectural effect this project has
measured, and it was invisible until the matcher forced fp32 for the score
einsum: bf16 over a 768×576 score matrix made the point-token arm untrainable.
Details below under
[Tokens](#tokens-point-beats-element-by-127-points-of-recall).

**`recall @25 cm+0.5°` is a joint gate**, not a translation threshold:
`ErrorSummary.THRESHOLDS` in `metrics.py` sets `((0.25, 0.5), (0.5, 1.0),
(1.0, 2.0))` and `ErrorSummary.update` counts a frame only when
`(trans <= 0.25) & (yaw_deg <= 0.5)`. This matters for the headline comparison —
a recall deficit can be a *heading* deficit rather than purely an association
one, and the tables below separate the two.

**Unless a table says otherwise, it is the synthetic source** —
`configs/synth_base.yaml`, whose 800/60/120 scenes are generated rather than
recorded. Leaving that implicit costs whole runs: an arm trained against
`configs/nuscenes.yaml` cannot be compared with a single row in this file. A
checkpoint's own `config.yaml` records its source; `scripts/run_arms.sh` fixes
it for a sweep.

Where a figure comes from a different network than the row it is read against,
it says so — that distinction matters, because a result about a *different
network* sizes a problem rather than predicting an answer.

## How to read a row

Unless a row says otherwise it is **one seed**, and rows marked *(mean of n)*
are means. A careful reader mistook a two-seed mean here for a second replicate
and concluded the same configuration had reported two different numbers. Every
multi-seed figure in this file is labelled, because a band is the difference
between a result and an anecdote.

## The number to beat

The prior is drawn, so the baseline is exact rather than estimated: predict a
zero correction and the error **is** the prior's own.

| | translation | longitudinal | lateral | yaw | recall @25 cm+0.5° |
|---|---|---|---|---|---|
| do nothing | 1.611 m | 1.497 m | 0.596 m | 0.993° | 1.3% |

8 880 test frames, 120 scenes. Everything below is read against this row.

## M1 — how much can one run settle?

Six trainings of one configuration, differing **only** in `train.seed`. The
scene seeds come from a separate base, so the data is identical across runs and
this measures training variance alone.

*Measured on the reference network — point-level assignment, point-to-point
residual, a fitted covariance scale, 2.28 M parameters at four layers. The
conclusion is about the protocol, not the architecture, so it carries over.*

### Open loop, one frame at a time

| seed | longitudinal | lateral | yaw | translation | recall @25 cm+0.5° | NEES median |
|---|---|---|---|---|---|---|
| 0 | 0.322 | 0.095 | 0.209° | 0.336 | 96.0% | 0.786 |
| 1 | 0.205 | 0.096 | 0.204° | 0.226 | 96.3% | 0.734 |
| 2 | 0.234 | 0.095 | 0.199° | 0.252 | 96.1% | 0.772 |
| 3 | 0.240 | 0.095 | 0.201° | 0.258 | 96.3% | 0.726 |
| 4 | 0.224 | 0.094 | 0.199° | 0.243 | 96.2% | 0.771 |
| 5 | 0.213 | 0.095 | 0.197° | 0.234 | 96.3% | 0.776 |
| **mean** | **0.240** | **0.095** | **0.202°** | **0.258** | **96.20%** | 0.761 |
| **CV** | **17.7%** | **0.67%** | 2.2% | 15.4% | **0.13%** | 3.2% |

**The variance is almost entirely along-track.** Lateral error reproduces to
0.67% across six independent trainings — effectively deterministic — and recall
to 0.13%. Longitudinal moves 17.7%, and because it dominates the translation
norm it drags `trans` to 15.4% with it.

That is exactly where the variance *should* be. Lane geometry runs parallel to
travel, so sliding a hypothesis down the road costs almost nothing and training
noise has room to move the answer. Lateral is pinned hard by the same lane
lines, so every seed lands in the same place. **The axis that is structurally
weak is the axis that is empirically unstable** — and it is the same axis whose
uncertainty a point-to-point residual structurally cannot express.

### Closed loop, through the filter

| seed | longitudinal | lateral | translation |
|---|---|---|---|
| 0 | 0.056 | 0.067 | 0.087 |
| 1 | 0.058 | 0.068 | 0.089 |
| 2 | 0.055 | 0.067 | 0.087 |
| 3 | 0.069 | 0.067 | 0.096 |
| 4 | 0.056 | 0.066 | 0.087 |
| 5 | 0.056 | 0.067 | 0.087 |
| **mean** | **0.058** | **0.067** | **0.089** |
| **CV** | 9.1% | 0.94% | **4.1%** |

Closing the loop is worth **2.9×** on the mean (0.258 → 0.089), and almost all
of that is along-track: 0.240 → 0.058, a factor of **4.1**, against 1.4× on
lateral. Intermittent along-track evidence accumulates over a run of frames, so
the filter recovers precisely the axis a single frame cannot see. Closed loop,
longitudinal ends up *better* than lateral.

Closed loop is also the more reproducible metric — 4.1% against open loop's
15.4% — because averaging over 74 frames suppresses the frame-to-frame
variance that dominates a single-frame RMSE.

### What this changes

**Compare arms on `long` and on recall, not on `trans`.** `trans` is the
noisiest of the five numbers and it averages away the one asymmetry the whole
design turns on. Recall @25 cm is the most reproducible number in the file.

**A single run settles almost nothing about a 10% effect.** Any difference
smaller than roughly 18% of the longitudinal error is inside the seed band, and
a sweep run one seed per arm falls for exactly that:

| | spread |
|---|---|
| seed alone, architecture fixed | 0.226 → 0.336 |
| a head-count sweep, one seed per arm | 0.247 → 0.366 |

The two are the same width. That sweep measured seed noise and reported it as
an effect of attention heads.

**And a single-seed headline can be an unlucky draw.** Seed 0's 0.336 m sits
1.9 standard deviations above the six-run mean of 0.258, so any comparison made
against 0.336 is made against the worst draw of six.

## M2 — the matcher, measured

First full run of the element-token matcher: 800 scenes, 40 epochs, the same
data and schedule as the reference network. One seed.

| | `long` | `lat` | `yaw` | `trans` | recall @25 cm+0.5° |
|---|---|---|---|---|---|
| do nothing | 1.497 | 0.596 | 0.993° | 1.611 | 1.3% |
| reference network, six-run mean | **0.240** | **0.095** | **0.202°** | **0.258** | **96.20%** |
| element tokens *(one seed)* | 0.292 | 0.130 | 0.308° | 0.320 | 81.1% |
| distance, in reference seed σ | 1.2 | **55.3** | **24.4** | 1.6 | **119.4** |

**It is behind, and M1 is what makes that statement possible.** On `trans` the
gap is 1.6 σ — inconclusive, and a reasonable person would have called it a
draw. On recall it is 119 σ. The two metrics disagree because `trans` carries
the seed noise of the longitudinal axis while recall carries almost none, which
is exactly what M1 measured and exactly why it ran first.

**One caveat before reading this as an architecture result.** The element-token
matcher has 1.02 M parameters against the reference's 2.28 M, two attention
layers against four, and two heads against four. The comparison therefore
conflates the design with its capacity, and a fair test needs the same
parameter budget.

**Where it loses is the useful part.** Longitudinal — the hard, weakly
observable axis — is only 19% worse. Lateral is 37% worse and heading 52%, and
those are the axes that lane lines pin down tightly and that the reference
reproduced to under 1% across seeds. Losing precision *there* points at spatial
resolution in the assignment rather than at the matching being confused: this
design pools eight points into one element token and then resolves points
inside a matched pair with a **zero-parameter** soft nearest neighbour, where
the reference matched points directly.

`matcher.py` already names that as the open question — a learned point stage
would earn its parameters by handling a detection that sees only part of a map
element — and this is the measurement that says it is worth trying, rather than
an assumption built in ahead of evidence.

## The deficit is resolution, not capacity

M2's comparison conflated the design with its size — 1.02 M parameters against
the reference's 2.28 M, two attention layers against four. So the depth was
doubled and nothing else changed: 1.81 M parameters, `model.layers=4`.

| | 2 layers | 4 layers | change | still behind |
|---|---|---|---|---|
| `long` | 0.292 | **0.274** | −6.2% | 14.3% |
| `lat` | 0.130 | **0.130** | **0.0%** | 36.8% |
| `yaw` | 0.308° | 0.303° | −1.6% | 50.4% |
| `trans` | 0.320 | 0.304 | −5.0% | 17.8% |
| recall @25 cm+0.5° | 81.1% | 81.7% | +0.7% | −14.5 pp |

**77% more parameters moved the axis that was already closest and left the
others where they were.** Lateral is identical to three decimals. Recall gained
0.6 points against a 14.5-point deficit.

That is the opposite of what a capacity limit looks like — a model short of
parameters improves everywhere when given more. This one improved only
longitudinally, which is the axis that is *hard for everyone* and where extra
depth buys better association. Where it is actually losing, depth is irrelevant.

So the gap is architectural, and it is a **resolution** limit rather than a
reasoning one. The suspect is the part of this design that has no parameters at
all: eight points are pooled into one element token, and points inside a
matched pair are then resolved by a soft nearest neighbour in each element's
own local frame. That frame comes from the element's own centroid and its
first-to-last heading — and a detection cut by a camera frustum is a *partial,
noisy* view of a map element chunked at fixed 12 m boundaries, so the two
frames do not agree even when the elements correspond correctly.

`matcher.py` names this as its open question in as many words. The next
measurement is to isolate it: hand the solve the *true* element correspondence
on real samples and see what the zero-parameter point stage alone recovers.

## Correct labels, and why they were not enough

The map is chunked at 12 m and a frustum is not, so a detection covers several
map elements — four for a road boundary, two for a lane divider, with 88% of
their points accounted for. Labelling that with a single index and a 50%
threshold called 95.6% of road boundaries unmatched. Making the label a
distribution over covered elements took real detections carrying a label from
56.7% to 84.1%, and road boundaries from 4.4% to 80.8%.

It did not help, and the way it failed is the finding.

| | `long` | `lat` | `yaw` | `trans` | recall |
|---|---|---|---|---|---|
| 2 layers, one label *(mean of 2 seeds)* | 0.291 | 0.138 | 0.312° | 0.323 | 80.2% |
| 2 layers, distribution *(one seed)* | 0.341 | 0.153 | 0.369° | 0.374 | 75.4% |
| 4 layers, one label | 0.274 | 0.130 | 0.303° | 0.304 | 81.7% |
| 4 layers, distribution | **0.264** | 0.130 | **0.422°** | **0.294** | **83.1%** |

The deep arm gives the best translation and recall the element-token design
produced and, as printed, its **worst heading by far** — 0.422° with a single
frame 28.3° out, against a 2.97° worst case before. That one frame is 50.6% of this
arm's squared yaw error over 8 880 test frames: leave it out and the row reads
**0.296°**, the *best* heading in this table rather than the worst. The
heading column here is one frame, not a result. The shallow arm is worse
everywhere.

**The element stage and the point stage do not compose.** `Matcher.points`
normalises the within-pair distribution over each map element's own points, so
it sums to one for *every* element regardless of distance, and multiplying by
the element assignment then hands each detected point mass in proportion to
that assignment and nothing else. Demonstrated on two chunks laid end to end
with a detection spanning both and the true 50/50 label:

| detected point | mass to chunk 0 | mass to chunk 1 |
|---|---|---|
| x = 0 | 0.500 | 0.500 |
| x = 14 | **0.500** | 0.500 |

The point eight metres past the end of chunk 0 still sends half its mass there.
A one-hot assignment hid this completely — with one element holding all the
mass there is nothing to misallocate. Correct labels spread the assignment, and
the defect became the dominant error.

The factorisation `P(m, q | d, p) = P(m | d) · P(q | d, p, m)` assumes the
whole detection belongs to element `m`. It cannot express *point p belongs to
chunk m(p)*, which is exactly what a detection spanning several chunks needs.

**The obvious repair does not work, and why is the useful part.** Bringing each
map element into the detecting element's frame before comparing points does
separate the chunks — every point then lands on the one it lies on. But the
transform that does it is the relative pose between the two frames, and the
detections are in the ego frame while the map is in the prior's, so that
relative pose *contains the correction being solved for*. Asked to recover a
known `(1.5, −0.6, 0.03)` from exact correspondences, the solve returns
`(1.147, −0.260, 0.043)`.

| | invariant to the unknown correction? | tells one chunk from the next? |
|---|---|---|
| each element's own frame | **yes**, exact | **no** — two chunks of a lane are byte-identical there |
| a shared frame | no — biased by up to 4.5 m | yes |

Local frames are exact *within* an element and blind *between* elements; a
shared frame is the reverse. Iterating the two — correspondence, pose,
correspondence — is the classical answer, but the first correspondence has to
be drawn under a 4.5 m prior with map points 1.68 m apart, and that ambiguity
is precisely why a *learned* matcher beats a geometric one by 4.4× here. So
this is a design question for the matcher, not a bug with a patch.

**And lateral error has not moved.** It is 0.130 at 1.02 M parameters, 0.130 at
1.81 M, and 0.130 under both labelling schemes — against the reference's
0.095, reproduced to 0.67% across six seeds. Capacity does not touch it and
supervision does not touch it, which points at the resolution the design can
represent rather than at what it has learned.

## M3 — the covariance, measured

Point-to-line residuals, and a covariance read off the curvature of the cost
the solve just minimised. Nothing fitted: no scale head, no temperature.

| | `long` | `lat` | `yaw` | `trans` | recall @25 cm+0.5° | NEES median |
|---|---|---|---|---|---|---|
| point-to-point, 4 layers | 0.264 | 0.130 | 0.422° | 0.294 | 83.1% | — |
| **point-to-line, 4 layers** | 0.265 | **0.114** | **0.252°** | **0.289** | **84.2%** | 0.153 |
| point-to-line, 2 layers | 0.303 | 0.126 | 0.265° | 0.328 | 80.5% | 0.283 |
| reference network, six-run mean | 0.240 | 0.095 | 0.202° | 0.258 | 96.20% | 0.761 |

Those first two rows do **not** differ in the residual alone — the 4-layer
point-to-point row above used the earlier closed-form solve with its own IRLS,
so it differs in the residual *and* in where the robust weight is applied. A
proper single-variable control is run further down, and it says something
different from what this comparison suggests. Read that one.

**The lateral floor broke.** It had been 0.130 at 1.02 M parameters, 0.130 at
1.81 M, and 0.130 under both labelling schemes — untouched by capacity and
untouched by supervision. The residual moved it, which says the limit was
always what the cost could *express* rather than what the model could learn.

### Is the size right?

*(`tools/eval.py`, test split, 8 880 frames, the 4-layer point-to-line arm.
`reported` is the model's own sigma and `actual` the marginal RMSE over the
whole split; the `tools/refcov.py` numbers above are per-frame over 96 frames,
and they put the width at 2.8–3.7× rather than 2.3×.)*

No. It is **pessimistic by roughly a factor of 2.3 in sigma**:

| | reported | actual |
|---|---|---|
| `σ_long` | 0.600 m | 0.265 m |
| `σ_lat` | 0.321 m | 0.114 m |
| NEES median | 0.153 | 0.789 is honest |
| coverage | 0.991 | 0.95 expected |

A filter told this will under-weight good frames and converge more slowly than
it could. That is the *safe* direction — the dangerous one is overconfidence,
which no downstream gate can undo — but it is not honest, and `2 s² H⁻¹` says
the scale comes from `s² = cost / (dof − 3)`, the mean squared residual. Imperfect
correspondences inflate that cost, and a matcher at 84% recall has plenty of
imperfect ones. So the size is downstream of the matcher, and reporting it as
a covariance defect would be blaming the wrong stage.

## The residual, controlled properly

The M3 comparison above was not single-variable: its point-to-point row used
the earlier closed-form solve with its own IRLS, so it differed in the residual
*and* in where the robust weight applies. This one changes `model.residual`
and nothing else — same tokens, same solve, same reweighting, same schedule.

| | `long` | `lat` | `yaw` | `trans` | recall @25 cm+0.5° |
|---|---|---|---|---|---|
| point-to-point | 0.308 | **0.105** | **0.229°** | 0.326 | **86.5%** |
| point-to-line | **0.265** | 0.114 | 0.252° | **0.289** | 84.2% |

**Point-to-line is not uniformly better.** Uncontrolled, it appears to buy 12%
of lateral error and 40% of heading. Controlled, it buys 14% of *longitudinal*
error and 11% of translation, and it costs 9% of lateral, 10% of heading and
2.3 points of recall.

That is a coherent result rather than a disappointing one. A rank-1
perpendicular residual removes a lane line's constraint *along* itself, which
is exactly the constraint that was never real — so the along-track axis
improves. It also removes information the isotropic residual was getting for
free: a lane line held laterally by both of its axes is over-constrained but
not wrongly constrained, and giving that up costs lateral precision.

So the choice is not "which is better" but which axis to buy. Point-to-line is
still the only residual that can produce an anisotropic covariance at all —
point-to-point's translation Hessian is `2·mass·I` and reports
`σ_long/σ_lat` = 1.01 whatever the landmarks are — and an honest covariance is
the thing this project is for. But the accuracy it costs is real and is now on
the record.

**Their calibrations are not comparable under a `mass` divisor.** A line
residual constrains one degree of freedom and a point residual two, while both
add exactly one to `mass` — which was the divisor. So point-to-point's variance
estimate was inflated by about two, and the NEES it reported, 0.020 against
point-to-line's 0.184, measured that rather than the arm.

The divisor counts degrees of freedom instead: `trace(n nᵀ)` is 1 and
`trace(I)` is 2, and `proj_d` is already the assignment-weighted sum of them,
so its trace is what each detected point contributes. Read under
`s² = cost / (dof − 3)`, the controlled residual comparison is
[three seeds an arm](#the-residual-at-point-resolution--three-seeds-and-it-reverses),
and it goes the other way.

