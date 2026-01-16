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

## How the model is sized

Sizing usually trades parameters against latency. **At batch 1 that trade
barely exists**: `dim` 64 to `dim` 256 is 15.7× the parameters for +1.6% of
batch-1 latency (`tools/cost.py`, RTX 2070, synthetic tensors, p50 of 30). At
batch 64 it is 2.69×, so parameters are free for the milliseconds and not for
the days — which is why the teacher is sized against training cost. Attention
over many tokens and kernel-launch overhead dominate, not arithmetic over
weights.

So **tokens are what cost**.

This table is what each knob is decided by. The right-hand column is the rule;
the status is what measuring it returned.

| knob | decided by | status |
|---|---|---|
| tokens | as few as carry the answer | **measured** — one per *point*, not per element: 95.9% against 83.1% recall, 12.7 pp at 6.8 sd, `RESULTS.md` |
| residual | whichever axis is worth buying | **measured** — point-to-point, 3 seeds an arm: NEES 0.68-0.76 against line's 0.04-0.20, no overlap; +1.6 pp recall at 2.0 sd; costs a 10-15x larger confidently-wrong tail |
| `dim` | the smallest that does not lose accuracy; pose precision comes from the point residual path, not from token width | **128 beats 64 by 5.5 pp at 6.3 sd**, and 256 buys −0.70 pp — saturated |
| layers | swept upward from 1 until accuracy stops improving | **measured — 4, by +9.8 pp at 18 sd over 2.** Also the largest calibration swing found: NEES 1.016 at two layers against 0.081 at four |
| heads | see below — rank, against what one softmax can hold | **not separated** at 2 seeds a cell; 2×64 kept, `RESULTS.md` |
| `head_dim` | stated, not derived from head count, so the two sweep apart | **not separated** — 2×32 against 2×64 is 0.38 sd |
| refinement passes | as few as the anneal needs | **3 IRLS passes.** Swept at eval: no value clearly wins, so the default is the one the anneal was tuned against |
| **student** | the largest that fits the latency budget on the target device | export works, and the budget is not where it looks: **the solve is 85% of a compiled frame**, so depth is the only lever that moves the clock |
| **teacher** | grown until accuracy saturates, since its cost is training time only | **`dim` 128, 4 layers, 2 heads of 64, 1.709 M** — every axis measured, and width is the one that saturated |

**Why more than one head, and why not many.** Multi-head attention costs
nothing in *parameters* at fixed `dim`: `q·k` contracts over `dim` whether it is
split or not, the concat is a view, and the output projection exists either way.
It is not free in activations — the score tensor is linear in head count, and at
point-token resolution that is 1.8 M pairs per layer against 28 k at element
resolution.

What extra heads buy is extra *attention distributions*. Softmax normalises, so
one head holds exactly one — and an element here plausibly needs two at once:
its lateral neighbours, to answer which of four parallel lines it is, and
along-track structure like stop lines and poles, to answer where along the road
it is. One softmax must split its mass between them.

The limit in the other direction is rank. A head's score matrix is rank at most
`head_dim`, so heads can be too narrow rather than too few. At `dim` 128,
deriving `head_dim` as `dim // heads` would give 64 at 2 heads and 32 at 4 —
which is why it is stated instead.

**A narrow head is not a slow one.** Swept over 1 344 point tokens, 4 heads at
`head_dim` 32 is the *fastest* of the three — p50 7.632 ms, against 7.972 at
2×64 and 8.444 at 1×128. At batch 1 this model is launch-bound, so a matmul
saving never reaches the wall clock, and "32 is too small to keep a tensor core
fed" is an argument about arithmetic that this regime does not run on.

The accuracy half is no better supported: the same sweep gave 0.336, 0.366 and
0.247 for 4, 2 and 1 heads — non-monotonic, spanning 48% of the smallest value,
one seed each. That measured seed noise, not head count.

So **the argument for 2 heads is weak, and the measurement is what settles it**
— but only because the two questions are stated apart. Deriving `head_dim` as
`dim // heads` makes a head-count sweep move the per-head rank cap with it, and
deriving `rope_bands` as `head_dim // 6` moves positional bandwidth too; both
are stated independently so a sweep asks one question. Measured at two seeds a
cell, 4×64 is not separated from 2×64 (1.03 sd) and 2×32 is not separated
either (0.38 sd). 2×64 is kept because the registration said in advance that a
null means neither knob is worth the parameters — even though 2×32 is 31%
cheaper.

Language models use 32 or more heads because they track many relations over
long contexts. Point tokens carry more relation classes than pooled tokens do —
within-element point ordering, and temporal-copy identity among the 512 warped
history points — so the comparison is less unfair at 1 344 tokens than at 168.
It still did not separate.

**The rank argument does not bind.** Decomposing the ideal score logit gives
planar proximity rank 3, heading agreement rank 2, and at most 10 realisable
class pairs — an intrinsic rank of **at most 15**, against a per-head cap of
64. Extra *heads* add no rank at all, since the cap is per head; what they buy
is softmax multiplicity, which this data does not measure. 4×64 costs +62%
parameters and 1.73× the training step for a gap nothing can read, so the
memory is better spent on `dim`, `layers` and `tokens` — the axes that have
actually separated.

Sizes: teacher `dim` 128, 4 layers, 2 heads of 64, 1.709 M parameters; student
the same at 2 layers, 0.914 M. A teacher is worth training only if the gap it
opens is worth distilling, and distillation is the one compression stage with a
closed-loop effect: 3 of 3 seeds beat their no-teacher controls and two match
the teacher's 0.091 m, at three seeds a side and so not yet certified — and the
student, not the teacher, is what ships.

## What is already measured

Grouped by what each result is *about*, because that decides how far it
generalises: a fact about the task holds whatever network reads it; a fact
about one architecture holds only until someone measures another.

**About the task and the data — these hold whatever the network.**

| | |
|---|---|
| association must be learned | the first correspondence is drawn under a 4.5 m prior with map points 1.68 m apart; a learned matcher beats a geometric one 4.4× open loop — an aside in `RESULTS.md`, with no arm behind it |
| along-track is the weak axis | and the evidence that fixes it arrives intermittently, which is why accumulating frames helps |
| a synthetic observability result need not survive | it did not reproduce on nuScenes |

**About the filter and the system — arithmetic and statistics, no network.**

| | |
|---|---|
| closing the loop helps | 2.98× — 0.271 m to 0.091 m, 120 test scenes, 8 880 frames — and it is the only test that separates bias from noise |
| a gate cannot catch a confidently-wrong frame | neither gate compares the answer against the prior, so widening is the only lever left. `trust` has no head to read and refused nothing; M4's refusals are all on `mass` — 0.1% of frames, longest run 2 frames. Whether widening converts the tail is not yet measured |
| overlapping history correlates consecutive measurements | a filter treating them as independent is over-confident |
| NEES/dof is calibrated at 0.789 at three degrees of freedom | not 1.0 — that is the *median* target, where the conventional one is the mean. Both are reported; the median leads because a mean over a heavy tail is not a calibration, and the gap between them is the tail |
| a calibrated NEES median does not mean a correct covariance | a median of 0.786 against a 0.789 target sat on top of a covariance 20.6× overconfident on the worst 1% of frames — 2.249 m of error against a claimed 0.308 m — while the other 99% ran 0.80×. Report the per-frame ratio's p99 beside the median |
| a point-to-point Hessian is isotropic in translation | `2·mass·I`, reporting `σ_long/σ_lat` 1.01 on lane lines alone and 1.01 again with eight poles added — the shape it reports is fixed by the weights, not by the landmarks, which is why the covariance has to be scored rather than argued for |
| distillation is the one compression stage with a closed-loop effect | the distilled student beats the same student trained with no teacher on 3 of 3 seeds — 1.2×, 2.1×, 6.2× — and 2 of 3 match the teacher's 0.091 m at 53% of the parameters (0.914 M against 1.709 M). Open loop the gap is +2.70 pp at t = 0.86 and the paired t is 1.90 against a critical 4.303: large, consistent in sign, and not certified at three seeds a side |

**About one architecture — measured at 1 344 point tokens, four layers, `dim`
128.** Each is a measurement on one configuration, not a law. They say why the
design does not take the obvious branch at each point; any is re-openable by a
measurement on another.

| | |
|---|---|
| pooling points into elements | lost open loop — 83.1% recall against point tokens' 95.9%, 12.7 pp at 6.8 pooled seed sd, three seeds a side, `RESULTS.md` — and 64× cheaper attention did not buy it back; the closed loop was only ever run with point tokens, where it reaches 0.091 m |
| biasing attention by inter-point distance | 2.5% accuracy for 2.4× the step time, from a 454 MB score-sized tensor per pass |
| a second refinement pass | trained with one, one wins outright: 0.221 against 0.336, ANEES 4.783 to 1.820 |
| conditioning the trunk on the prior's width | reversed sign between two training ranges — a correlation, not a cue |
| feeding pose disagreement to the calibration heads | no accuracy change, and training on the gap destroyed the gap's diagnostic value |
| collapsing to a single head | accuracy did not separate 1, 2 and 4; latency did, over 1 344 tokens, and favoured 4 |
| pruning channels or heads | 35% of weights for 0% of latency; heads 3.5%, the same as the weight buys |

**The deployment findings are measured on the compiled frame**, because this
model is *activation-bound* — 14.4 M attention scores a sample (1.8 M pairs per
layer, 4 layers, 2 heads) against 1.709 M parameters — and a parameter count
predicts nothing about it. TensorRT fp16 is 13.6× the PyTorch trunk, which
leaves **the solve at 85% of a compiled frame**: a network made infinitely fast
would buy 1.17× end to end, and that is INT8's ceiling. INT8 costs 1.06 pp of
recall open loop and 2 mm closed. Pruning reaches only the feed-forward hidden
width — keep 0.75 removes 7.7% of the model, not 25%.

## Milestones

**M0 — the harness.** *Done.* Data, geometry, metrics, the filter, the scripts.
Everything true whatever network you build.

**M1 — how much can one run settle?** *Done.* Six trainings of one
configuration differing only in `train.seed`, reported as a spread: longitudinal
CV 17.7% against lateral 0.67% and recall 0.13%. That is the error bar every row
below is read against, and it ran first because three runs of one model came out
48% apart, non-monotonically, so a 10% effect was unresolvable and nobody knew
it.

**M2 — the matcher.** *Done.* Point tokens, rotary relative attention,
per-point assignment, the least-squares solve. Ablated against pooled element
tokens and absolute encoding, 3 seeds an arm.

**M3 — the covariance.** *Done.* Curvature in place of a fitted scale, and the
residual chosen by which one scores calibrated rather than by which one looks
anisotropic. Ambiguity is read from the assignment, not from the local minima
of a fixed-assignment surface — those are numerical, and measured to be so.

**M4 — closed loop.** The filter, with ambiguity widening the covariance rather
than gating the frame. **Done: 0.271 m open loop to 0.091 m closed, 2.98x, no
scene diverged** — 120 generated test scenes, 8 880 frames,
`configs/synth_base.yaml`. The gain arrives *because* the covariance is
honest — an accurate pose with a dishonest one measures 0.188 m here and looks
fine in every open-loop table.

**M5 — real data, in two halves.** *Open.* The map is already real and cached:
the nuScenes expansion gives surveyed lane dividers, road boundaries, crossings
and 307 traffic-light poses in Boston alone, and `data/nuscenes.py` reads it
without importing the devkit. What is still generated is *perception*.

*First half* — the shipped configuration on that map, split geographically so no
location is shared between training and test, with detections still cut from the
map and corrupted. No new code: the cache is built and `configs/nuscenes.yaml`
exists. What it buys is the first result on real road geometry.

*Second half* — detections from a real mapper. The class mapping already follows
the online-mapping convention (MapTR, StreamMapNet) so a detector drops in
without a translation layer, and `data/detections.py` specifies the file it must
write. The detector runs offline in its own environment, because its
dependencies cap at Python 3.10 and torch 2.0 where this project runs 3.12.
That boundary is deliberate.

**M6 — distillation.** *Done, and not where it was expected.* Six seeds a side:
recall gains 1.58 points and does not separate (paired t = 1.46), while NEES
moves 0.497 to 0.691 against an honest 0.789 and does (t = 2.98, six of six
seeds). The teacher transfers **calibration**, not accuracy — which is what the
filter charges for, and why the two arms differ by 15 mm closed loop and by
nothing measurable frame by frame. The failure rate is still unresolved: 5 of 6
distilled runs reach the good basin against 3 of 6 untaught, which needs about
twenty seeds a side rather than six.

**M7 — deployment.** *Measured in part.* TensorRT on the desktop; Core ML on
Apple silicon, where int8 compute is Neural Engine only and 4-bit is weight-only
compression. Pruning and quantization are measured on the compiled frame, where
the solve is most of it.

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
