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

## Which of these numbers can be trusted

**Accuracy survives and calibration is the fragile half.** Numerics reach the
pose only through the assignment: `eval.py` runs in fp32 and the solve is
deterministic once the correspondences are fixed. A precision change costs
whatever it costs the assignment — fp16 export moves 0.65% of points onto a
different correspondence, and the downstream price of that is left open below
rather than assumed to be zero. The covariance is where the care goes — its
divisor, its floor and the filter's use of it each carry a test that states the
invariant rather than the number.

## How calibration is reported, and why 0.789

Every calibration row here leads with a **median** NEES per degree of freedom —
normalised estimation error squared, the pose error measured in units of the
covariance the model reported — against a target of **0.789**, and that is not
the conventional choice. The convention — Bar-Shalom, Li and Kirubarajan, and
the practice every filtering toolkit inherits from it — averages NEES and tests
it against the degrees of freedom, so the normalised target is **1.0**. Both
numbers appear in every report this project produces: `anees_median` against
0.789 and `anees_mean` against 1.0.

The arithmetic first. A calibrated NEES is chi-square distributed and the
chi-square is right-skewed: at three degrees of freedom its mean is 3 and its
median 2.366. Per dof that is 1.0 and 0.789. Judging a median against 1.0 calls
a textbook-calibrated model pessimistic by 21%, which is a mistake worth
naming because it is easy to make.

The median leads because **a mean over this error distribution is not a
calibration**. Association failures put a small fraction of frames arbitrarily
far out and the mean follows them. This file contains the worked example: a
model whose mean ANEES reads 1.7–3.0 against a target of 1.0 while its median
frame is honest, because 0.30–0.45% of frames carry the entire statistic. The
two diagnoses have opposite repairs, and rescaling to fix the mean would wreck
the frames that were already right. That an average NEES is dominated by its
large errors is a known objection to the convention rather than a discovery
here.

So read the pair, not either alone — **the gap between them is the tail**, and
a reader wanting the conventional statistic should take `anees_mean` against
1.0 and treat the median as its robust companion.

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

## Tokens: point beats element by 12.7 points of recall

*(2025-12-26. Synthetic, 4 layers, `dim` 128, `rope` on both sides, 8 880 test
frames. Three seeds each on the treatment and the `rope` control, one on the
`relative` control.)*

| | `long` | `lat` | `yaw` | recall @25 cm+0.5° | NEES median |
|---|---|---|---|---|---|
| **point tokens** *(mean of 3)* | **0.243** | **0.095** | **0.178** | **95.9%** ± 1.3 | 0.040–0.198 |
| element tokens, `rope` *(mean of 3)* | 0.508 | 0.184 | 1.195 | 83.1% ± 3.0 | 0.018–0.282 |
| element tokens, `relative` *(1 seed)* | 0.270 | 0.128 | 0.262 | 79.9% | 0.196 |

**12.7 points of recall, 6.8 pooled seed standard deviations.** Every point
seed beats every element seed with no overlap — worst point 94.4% against best
element 85.6% — which is a firmer statement than any σ count.

**A single control seed and a borrowed band both inflate the margin.** Element
tokens are **2.4× noisier** than point tokens — sd 3.0 points against 1.3 — so
scoring the gap with one element seed for the control, against the point arm's
spread, reads 16.1 points at 12.6 σ. Three seeds a side against a pooled band
reads 12.7 at 6.8. The effect survives; the confidence in it does not.

**Element tokens are also less reproducible**, which is a result in its own
right: on `trans` their spread is CV 15.3% against point tokens' 4.2%.

**The choice of element control does not change the verdict.** That was a real
worry: `ModelParams` carries two geometry fields — `geometry`, read for element
tokens, and `point_geometry`, read for point tokens — so the first sweep
compared point+`rope` against element+`relative` and was two variables. Run
properly, the two element arms land at 83.1% (mean of 3) and 79.9% (1 seed) —
inside each other's seed spread — so the tokens result stands whichever is
called the control. `rope` costs element tokens accuracy — lateral 0.184
against 0.128, longitudinal 0.508 against 0.270 — without touching their
recall, which is its own open question.

**And the two metrics rank the seeds differently.** One element-token `rope`
seed has the *worst* validation `trans` of the three (0.4601) and the *best*
test recall (85.6%), because its errors are a heavy tail rather than a wide
bulk — yaw RMSE 2.626° against another seed's 0.305°. This is the concrete
reason not to compare arms on `trans`, and the reason `best.pt` selecting on it
is a defect rather than a detail.

**Why the point-token matcher needs fp32.** Under bf16 autocast a 768×576
score matrix drives the largest logit to 1.9e7 where the element arm's 96×72
tolerates the same code, and the point-token arm cannot train at all.
`matcher.py` forces fp32 for that einsum for exactly this reason.

Two points of the 12.7 are not association: recall is a joint 0.25 m **and** 0.5°
gate, and element+`rope`'s heading RMSE is 6.7× worse — 1.195 against 0.178.
The rest is.

## The residual at point resolution — three seeds, and it reverses

*(Same run. Three seeds an arm, measured 2026-01-05, 0 failures.)*

| | recall @25 cm+0.5° | NEES median (0.789 target) | tail >10 |
|---|---|---|---|
| point-to-line *(s0/s1/s2)* | 94.4 / 96.6 / 96.6 — **95.9%** | 0.198 / 0.040 / 0.066 | 0.06 / 0.01 / 0.02% |
| **point-to-point** *(s0/s1/s2)* | 98.0 / 96.8 / 97.7 — **97.5%** | **0.681 / 0.761 / 0.701** | 0.33 / 0.45 / 0.30% |

At point resolution the residual arm **reverses**, and the third seed holds it.

**Calibration is the decisive half, and it does not overlap.** Every
point-to-point seed lands at 0.68–0.76 against the 0.789 target; every
point-to-line seed lands at 0.04–0.20. The worst point-to-point arm is 3.4×
better calibrated than the best point-to-line one, and no seed of either arm
comes near the other's range. Excluding the tail, point-to-point's ANEES is
0.97–1.07 — the first covariance this project has produced that is simply
right, rather than right on average by cancelling two errors.

**Recall is the weaker half.** +1.6 points (97.5% against 95.9%), SE of the
difference 0.82 pp, so **2.0 sd**. Real, but it is the calibration that carries
this verdict, not the recall — and the heads arm is a standing reminder of what
a 2 sd gap is worth when the spread is only estimated from three seeds.

**The cost is the tail.** Point-to-point is confidently wrong on 0.30–0.45% of
frames against point-to-line's 0.01–0.06% — a factor of ten to fifteen, and the
reason its *mean* ANEES reads 1.7–3.0 while its median is honest. A localiser
feeding a filter is judged partly on how rarely it lies confidently, so this is
not a rounding detail: it is the price of the calibration, and M4 is where it
gets charged. Widening on assignment ambiguity is the mechanism that should
convert those frames from confidently wrong to honestly uncertain, and whether
it does is a measurement nobody has made yet.

That contradicts [the residual control](#the-residual-controlled-properly),
which chose point-to-line at element resolution. Two things differ. The earlier
comparison ran under the `mass` divisor, which penalised point-to-point by
about two because a pole constrains two directions and a line point one while
both counted as one correspondence. Counting degrees of freedom instead removes
that penalty. And it ran at element resolution, where a pooled token cannot
express which point matched.

### The over-wide covariance is **depth**, and at four layers it inverts

*(`tools/refcov.py`, 64 frames x 32 redrawn priors, 2026-01-05. One arm of each
depth, both `line` residual, both under the same training recipe — so depth is
the only difference.)*

The section below credits point-to-point with an honest covariance and charges
point-to-line with one 4.16x too wide. Every arm in that comparison had **four
layers**, and the depth sweep then showed NEES swinging from ~1.0 at two layers
to 0.048 at four on the *same* line residual. So the obvious question is whether
"the residual reports an over-wide covariance" was ever about the residual.

| | 2 layers | 4 layers |
|---|---|---|
| reference `σ_long/σ_lat` | 1.115 | 0.704 |
| reported `σ_long/σ_lat` | 1.794 | **3.984** |
| size, long | **1.44x** | **17.72x** |
| size, lat | **0.89x** | 3.13x |
| principal axis off by | 13.4° | **78.5°** |
| verdict | reference near-isotropic, no call | **INVERTED** |

**At two layers the line residual is very nearly right.** 1.44x along track and
0.89x across it, axis 13.4° out — the same neighbourhood as point-to-point's
1.10x/1.32x and 12.7°. At four layers the same residual reports **σ_long of
1.0966 m**, seventeen times the 0.0619 m the errors actually show.

**And it is systematic, at three seeds.** Measured on the 2070 after the fact:

| | σ_long | σ_lat | axis off by | orientation verdict |
|---|---|---|---|---|
| 4 layers, line, s0 | 17.15× | 3.48× | 72.2° | reference too round to rule |
| 4 layers, line, s1 | 17.72× | 3.13× | 78.5° | **INVERTED** |
| 4 layers, line, s2 | 7.67× | 2.04× | 65.2° | reference too round to rule |
| 4 layers, point-to-point, s0/s1/s2 | 1.10 / 1.07 / 1.14× | 1.32 / 1.27 / 1.37× | 12.7 / 13.7 / 11.1° | **RIGHT WAY UP** ×3 |
| 2 layers, line, s0/s1/s2 | 1.44 / 1.29 / 1.13× | 0.89 / 0.84 / 0.92× | 13.4 / 7.3 / 8.5° | right way up where callable |

**The formal `INVERTED` label fired on one of the three**, because the other
two references were too near-circular for `refcov` to rule on orientation at
all — which is the tool declining to invent a finding, and correct. The part
that *is* consistent across all three is the substance: **7.7 to 17.7× too wide
with the major axis 65 to 79° out.** Point-to-point at the same depth holds
1.07–1.14× and 11–14° on every seed.

So the accounting has to change in two places:

**The absolute claim needs a depth qualifier.** "Point-to-line is 4.16x too
wide" is true of a 4-layer model. A 2-layer one is 1.44x. The width was
attributed to the residual and belongs at least as much to the depth.

**The comparative claim survives, and is strengthened.** The point-token line
and point-token point-to-point arms were both 4-layer, so that contrast was
controlled and point-to-point still wins. What changes is the reading: depth is
what *breaks* the line residual's covariance, and point-to-point is what holds
it together under depth — 1.10x and right way up where line goes 17.72x and
inverts. The residual choice matters **more** as the model gets deeper, not
less.

**And there is now a live tension in the depth sweep.** Four layers buys +10.4
points of recall and costs a covariance that is seventeen times too wide and
pointing the wrong way. On a localiser judged by what a filter can do with its
output, those are not obviously trading in the right direction — and the arm
that has both, 4 layers with point-to-point, is the one configuration this
sweep has not isolated.

### And the shape agrees with the scalar, which was not guaranteed

*(`tools/refcov.py`, 64 frames x 32 redrawn priors each, 2026-01-05. One arm of
each residual.)*

NEES settles the residual on a *scalar*, and a scalar cannot see the shape of
an ellipse. A NEES median of 0.786 against a 0.789 target is exactly the kind
of number that passes while the covariance is 20.6× overconfident on the 1% of
frames that matter — see
[the tail](#the-covariance-can-be-calibrated-and-still-be-wrong). So the
verdict above is not safe until the matrix agrees with it.

| | sigma_long | sigma_lat | reported/reference |
|---|---|---|---|
| **point-to-point** reference | 0.0587 | 0.0805 | — |
| **point-to-point** reported | 0.0649 | 0.1062 | **1.10x / 1.32x** |
| point-to-line reference | 0.0792 | 0.0834 | — |
| point-to-line reported | 0.3294 | 0.1522 | **4.16x / 1.82x** |

**Point-to-point is the right size and points the right way.** 1.10x and 1.32x
against a reference computed by redrawing the prior 32 times per frame, verdict
`RIGHT WAY UP`, principal axis off by **12.7 degrees** over the 58 of 64 frames
elongated enough to have one.

**Point-to-line fails on shape worse than it fails on scale.** It is 4.16x too
wide along track against 1.82x across it, so the error is not a uniform
inflation that a single scale factor could repair — it is differential. And
the direction is not merely wrong but invented: the reference is
near-isotropic at 0.950, while the model reports an anisotropy of 2.165 and
places its major axis **54.3 degrees** away. A model claiming a strong
long/lat asymmetry where the truth is very nearly a circle is asserting
structure that is not in the data.

This is the half of the residual verdict that could have overturned it and did
not. The scalar and the matrix now say the same thing.

**0.681 was real.** The two replicates came back at 0.761 and 0.701, so the
single seed that re-opened this was not a fluke — which was not the safe bet,
given that the three point-to-line seeds span a factor of five on exactly this
statistic. The open question moves from *is it calibrated* to *what does the
tail cost in closed loop*.

## Heads — registered before the arms report

*Written 2025-12-31 09:10, with the six arms 40 minutes in and no epoch
reported. Registered in advance because the predicted outcome is a null, and a
null is the easiest result in the world to talk yourself out of afterwards.*

Three cells, two seeds each, point tokens, line residual, `grad_clip` 10.0,
`rope_bands` pinned at 5 in all three:

| cell | heads | head_dim | params | asks |
|---|---|---|---|---|
| A | 2 | 64 | 1.709 M | control |
| B | 4 | 64 | 2.764 M | does the *number* of attention distributions matter? |
| C | 2 | 32 | 1.181 M | does head *width* — the rank cap — matter? |

**Decided on** recall @25 cm+0.5°, with lateral RMSE co-primary. Not on
`trans`: longitudinal error varies 17.7% between seeds where lateral varies
0.67%.

**What is readable.** The seed band is this network's own — sd 0.0127 on recall
over the three point-token seeds. At two seeds a cell, SE(diff) is 1.27 pp, so
only a gap above **2.5 pp** can be called. On `trans` the threshold is 0.018 m.

**Predicted: both contrasts null.** The rank argument says a head's score
matrix has rank at most `head_dim`; at point tokens the matrix is 768 × 576, so
a cap of 32 or 64 against a maximum of 576 sounds binding. It should not be.
What the score has to express is proximity in an SE(2)-transformed plane plus a
class match, and that is low-rank by construction — 32 is likely already more
than it needs. For the count, `ROADMAP.md` argues an element needs two
attention distributions at once, lateral neighbours and along-track structure,
which predicts 2 is enough and 4 adds nothing.

**So the informative outcomes are the refutations.** If C is worse by more than
2.5 pp, rank is binding and `head_dim` is a real knob. If B is better, the
count argument understated how many relation classes point tokens carry — there
are at least two that element tokens never had, within-element point ordering
and temporal-copy identity among the 512 warped-history points.

**A null here does not mean "no effect".** It means not separated at 2.5 pp with
two seeds. The honest reading of a null is that neither knob is worth the
parameters — B costs +62% and C saves −31% — not that the rank argument was
proved.

## TensorRT, and why the solve is now the whole problem

*(RTX 2070, batch 1. `tools/export.py --trunk-only` writes the ONNX,
`mapposeformer/tensorrt.py` builds the engine, and `tools/trt_latency.py`
times it — p50 of 300 after 50 warmup, the same protocol `tools/latency.py`
uses for PyTorch, so the speedup is a subtraction rather than a claim.)*

**The model cannot be exported whole, and not for want of trying.**
`torch.onnx.export` fails outright inside the solve:

> `No ONNX function found for aten._linalg_solve_ex`

There is no operator to lower a linear solve to, in any opset. So deployment is
split by necessity rather than by preference: **trunk and matcher on the
accelerator, solve on the host.** What crosses the boundary is the assignment,
768 x 576 at point resolution — 1.7 MB a frame, and the real price of the split.

| | ms | note |
|---|---|---|
| PyTorch, whole model | 20.62 | `tools/pareto.py` |
| — the solve within it | 6.17 | `tools/solve_cost.py`, synchronised |
| — so the trunk in PyTorch | ~14.45 | by difference |
| **TensorRT, trunk only** | **2.69** | 372 frames/s, engine 8.0 MB |
| **trunk + host solve** | **8.86** | **2.3x end to end** |

**FP16 takes it further, and settles INT8 without building it.**

| | trunk | + host solve | end to end |
|---|---|---|---|
| PyTorch | ~14.45 ms | 20.62 ms | 1.0x |
| TensorRT fp32 | 2.69 ms | 8.86 ms | 2.3x |
| **TensorRT fp16** | **1.06 ms** (942 fps) | **7.23 ms** | **2.85x** |
| *a trunk that cost nothing* | 0 | 6.17 ms | 3.34x |

FP16 is 2.5x the fp32 engine and **13.6x the PyTorch trunk**, on a Turing card
with real half-precision tensor cores. **So the solve is now 85% of the frame**,
and a network made infinitely fast would buy only 1.17x more. INT8 can at best
shave part of 1.06 ms, so it is not worth the `modelopt` Q/DQ work — the
simulated-INT8 accuracy result stands on its own and the speed question is
answered by arithmetic rather than by another engine.

**Verified against PyTorch, and it is not exact.** (Engine and eager model run
on the same real inputs, in the export's own key order.)

| output | max abs err | mean abs err | **argmax agreement** |
|---|---|---|---|
| assignment, 768 x 576 | 1.1e-01 (11.8% of range) | 4.8e-06 | **99.35%** |
| scores | 3.77 (0.04% of range) | 2.9e-02 | **99.61%** |

**0.65% of detection points choose a different map correspondence** — about 5
of 768 per frame. The mean error is negligible and the tail is not, which is
what half precision does to a saturated softmax, and precisely why
`matcher.py` already forces fp32 for that einsum under bf16 autocast.

Whether five wrong correspondences a frame matter is **not yet measured**. The
solve is robust — iteratively reweighted least squares (IRLS) is built to
discard bad correspondences — so they may be absorbed entirely. But this
project has been caught twice by changes that held recall and moved the
covariance, so the claim stops here:
**fp16 is 13.6x faster at 99.35% assignment agreement, and the downstream cost
of that 0.65% is unknown.**

The latency numbers were re-taken on real inputs, because the harness had been
feeding zeros: it filled a tensor from the sample only when the tensor's *name*
matched a sample key, and the ONNX exporter names every input `args_N`, so it
matched nothing and padded everything. It now maps positionally, in the
export's own key order, and refuses outright rather than padding if the counts
disagree.

| engine | on zeros | on real inputs |
|---|---|---|
| fp32 | 2.689 ms | **2.673 ms** |
| fp16 | 1.061 ms | **1.049 ms** (953 fps) |

Within 1%, so the conclusions above stand unchanged — which is the answer that
had to be *measured* rather than assumed, since "dense kernels do not care
about values" is a plausible argument and this project has been wrong with
plausible arguments before.

**And it inverts the problem.** The solve was 28% of a PyTorch frame; compiled
to fp32 it is **70%** of a much shorter one, and at fp16 **85%** — the share
climbs precisely because the solve is the part that does not compile. Every
remaining optimisation on this path is worth at most the 15% that is left at
fp16, against a ceiling of about 3.4x, and fp16 already collects 2.85x of it.
Compressing the network further is close to pointless; the work that remains is
the solve.

That reframes the Raspberry Pi question. It is not *"can the network be made
small enough"* — TensorRT already answers that, and INT8 might shave the 2.69 ms
further. It is **"can six milliseconds of damped Gauss-Newton with a Cholesky
factorisation run on the target, per frame, with no GPU to fall back to."**

## The frontier, with one column that must not be read as deployment

*(`tools/pareto.py`, test split, batch-1 latency on the RTX 2070, p50/p99 over
200 iterations, every row measured in one process on one machine.)*

| | params | trans | NEES | recall @25cm | p50 | p99 |
|---|---|---|---|---|---|---|
| **p2p** | 1.71 M | 0.274 | **0.681** | **98.0%** | 20.62 | 37.93 |
| p2p+int8 | 1.71 M | 0.282 | 0.682 | 98.0% | *34.27* | *84.76* |
| keep 0.75 | 1.58 M | 0.275 | 0.651 | 97.5% | 20.69 | 45.26 |
| keep 0.75 + int8 | 1.58 M | 0.276 | 0.640 | 97.5% | *34.98* | *64.85* |
| keep 0.50 | 1.45 M | 0.319 | **0.473** | 95.7% | 20.34 | 48.50 |
| keep 0.50 + int8 | 1.45 M | 0.321 | 0.471 | 95.7% | *34.13* | *77.28* |
| line | 1.71 M | **0.251** | **0.198** | 94.4% | 20.74 | 37.71 |
| **L2** | **0.91 M** | 0.390 | 0.989 | 86.6% | **14.25** | **27.13** |

**The `+int8` latencies are italicised because they are not deployment
numbers.** `quantize.py` implements *simulated* quantization: it rounds a
tensor and converts it straight back, which adds arithmetic and removes none.
That the rows come out 66% slower measures the simulation, not INT8.
`tensorrt.py` exists precisely for this and says so in its own docstring — only
real integer kernels can answer it. **So the INT8 claim this project can make
today is about accuracy, where it is genuinely free — NEES 0.681 to 0.682,
recall unchanged — and nothing at all about speed.**

The columns that *are* deployment numbers say three things.

**Pruning buys nothing measurable.** keeping 0.50 removes 15% of the parameters
and lands at 20.34 ms against the baseline's 20.62 — inside the noise — while
costing 2.3 points of recall and a third of the calibration. It does not reduce
peak memory either (7.04 GiB against 7.09). On a launch-bound model there is
nothing for it to save.

**Depth is the only row that moves the clock.** `L2` is 14.25 ms against 20.62,
a **31% reduction**, and the only configuration in the table that is genuinely
cheaper. It costs 11.4 points of recall, which is a real price — but it is a
price paid for something, which pruning is not.

**And the residual verdict is visible in one row.** `line` has the *best*
translation RMSE in the table at 0.251 and the *worst* calibration at NEES
0.198, with 3.6 points less recall than `p2p`. A frontier ranked on RMSE alone
would have chosen it.

## What the solve costs, and the ceiling it puts on compression

*(`tools/solve_cost.py`, the point-token point-to-point arm, batch 1 on the
RTX 2070, p50 of 20, CUDA-synchronised. Batch 1 because that is the deployment
case.)*

| | ms | share |
|---|---|---|
| full forward | **21.92** | 100% |
| `solve_pose_directional` | 5.42 | 24.7% |
| `curvature_covariance` | 0.44 | 2.0% |
| `measurement_information` | 0.30 | 1.4% |
| **the solve, all of it** | **6.17** | **28.1%** |
| network — trunk and matcher | 15.75 | 71.9% |

**The solve is 28% of an eager frame and none of it compresses.** It is a damped
Gauss-Newton iteration with a Cholesky factorisation: no weights to quantize,
nothing an NPU accelerates, and unchanged when the network shrinks. So the
ceiling on every compression result above is **about 3.5x** — that is what
"compressing the trunk and matcher to nothing" buys, and INT8 plus pruning
together get nowhere near it.

The ceiling is quoted loosely because it comes out differently under two
protocols: 3.55x against this tool's 21.92 ms eager frame, 3.34x against the
20.62 ms one the latency table reports. They measure the same thing 6% apart,
which is the honest precision of a batch-1 wall-clock number and smaller than
any decision resting on it.

This is the number that decides whether a Raspberry Pi is plausible, and it
reframes the target: the work is not making the network smaller, it is either
making the *solve* cheaper or accepting 6 ms of irreducible host-side
arithmetic per frame. On a Pi both sides get slower, and the solve does not
have a GPU to fall back to.

## Compression — quantize, do not prune

*(`tools/paired.py` on the point-token point-to-point arm, test split, 1600
frames, 2026-01-14. Paired: the same batch through both models, McNemar on the
discordant frames. No seeds, because these are deterministic transforms of one
model.)*

| | weights | Δ recall | pose moved (median) | NEES |
|---|---|---|---|---|
| **INT8, per channel** | **6.52 → 1.69 MiB (−74%)** | **−0.19 pp** — *not separated* | 7.7 mm | 0.678 → **0.685** |
| prune, keep 0.75 | −7.7% params | −0.81 pp — **separated** | 21.2 mm | 0.678 → 0.645 |
| prune, keep 0.50 | −15.4% params | −1.94 pp — **separated** | 45.2 mm | 0.678 → **0.452** |

**Quantization is close to free and pruning is not**, which is the opposite of
how the two are usually ranked by effort. INT8 removes three quarters of the
weight bytes with a recall difference inside the noise, a median pose shift of
7.7 mm against a 250 mm gate, and a **NEES ratio of 0.996** — the covariance is
untouched. That last number is also the evidence that keeping the solve in fp32
worked: `quantize()` wraps `nn.Linear` and `solve.py` contains none, so the
Cholesky and the Gauss-Newton iteration never saw an integer.

### On the teacher, INT8 is not free

*(`tools/paired.py` on the teacher checkpoint — the configuration this file's
sweeps arrive at: point tokens, point-to-point, `dim` 128, 4 layers, 2 heads of
64, `rope_bands` pinned at 5, 1.709 M parameters — test split, **all 8 880
frames**, 2026-01-14.)*

| | teacher fp32 | teacher INT8 |
|---|---|---|
| recall @25 cm | 0.9626 | 0.9520 |
| weights | 6.52 MiB | 1.69 MiB (−74.1%) |
| NEES median | 0.908 | 0.962 |

**−1.06 pp ± 0.32 — separated**, on 206 discordant frames: 150 lost against 56
gained. That is five times the difference the table above reports, and the two
are not directly comparable. The row above measured a *different model* — the
point-token point-to-point arm, `geometry=relative`, no `rope_bands` pin — over
1 600 frames, where the paired half-width is 2.4× wider. Both readings are
honest about their own model; the one that governs deployment is this one,
because the teacher is what would ship.

**The median hides the tail.** NEES moves 0.908 → 0.962, which reads as
untouched — but the per-frame **ratio's p99 is 8.35**. On the worst 1% of frames
the covariance inflates eightfold. Reading "NEES ratio 0.996, the covariance is
untouched" off the median alone is a median-only reading of a median-only
statistic. The median is still right; it is just not the whole claim.

**Where the pose goes.** Median 32.2 mm, p99 161.4 mm, against a 250 mm gate —
so the 150 frames INT8 loses are not frames it ruins, they are frames that were
already sitting near the threshold. Nothing is catastrophically wrong; the gate
is simply close enough that a 32 mm jitter crosses it a hundred and fifty times.

**Priced against the speed it buys**, INT8 costs 1.06 pp of recall for at most
1.17× end-to-end, on a frame that `tools/solve_cost.py` measures as 85% solve.
So the reason to quantize is the 74% of weight bytes, not the latency.

### And through the filter it costs 2 mm

*(`tools/run_sequence.py --quantize`, same 120 scenes, same protocol as M4.)*

| | open-loop recall | closed loop `trans` | `long` | `lat` | `yaw` |
|---|---|---|---|---|---|
| teacher fp32 | 0.9626 | **0.091** | 0.060 | 0.069 | 0.128° |
| teacher INT8 | 0.9520 | **0.093** | 0.062 | 0.069 | 0.129° |

No scene diverged, 99.9% of frames accepted, and the longest refusal run was
*shorter* than fp32's — 1 frame against 2. **A cost that is separated and real
open loop is 2 mm on 91 mm once a filter integrates it**, and that gap is the
most useful thing on this page about how to read a compression table.

**Recall is a threshold metric and a filter is not.** INT8 moves the pose by a
median of 32.2 mm against a 250 mm gate. A frame sitting at 249 mm that moves to
251 mm counts as a whole frame lost, and 150 frames did exactly that — but it is
2 mm worse, and 2 mm is what the filter sees. The open-loop number is not wrong;
it answers "how many frames cross a line", which is not the question a vehicle
asks.

**What the widened tail did, and what is not measured.** The worry this table
raised was the per-frame NEES ratio's p99 of 8.35: a covariance eightfold wider
on the worst 1% of frames. Integrated over a scene it cost nothing, which is
consistent with the inflation landing on the frames INT8 actually damaged — a
covariance that grows where the error grows is a filter being told to trust
those frames less, which is correct. **That correlation is the plausible reading
and it is not measured here.** The alternative, that the inflation is harmless
because the damage is small either way, fits the same evidence. What is measured
is the 2 mm.

**So INT8 ships.** 74% of the weight bytes for 2 mm of closed-loop error is the
best trade in this section, and it only looks that way because the loop was
measured. On the open-loop table alone the honest reading was "a separated
1.06 pp regression", and that would have been a defensible reason to reject it.

**Before fine-tuning, pruning damages calibration faster than accuracy.** At
keep 0.50 recall falls 2% while NEES falls from 0.678 to 0.452 — a third of the
way to useless, and *away* from the 0.789 target rather than toward it. A table
reporting recall alone would have called that a cheap 15% saving.

**After fine-tuning, both levels recover completely.** Ten epochs through
`train.init_from`, measured on the test split:

| | params | recall @25 cm | NEES |
|---|---|---|---|
| `p2p` baseline | 1.71 M | 0.980 | 0.681 |
| keep 0.75 — no tune | 1.58 M | −0.81 pp | 0.651 |
| **`prune_keep75_finetuned`** | 1.58 M | **0.980** | **0.799** |
| keep 0.50 — no tune | 1.45 M | −1.94 pp | 0.452 |
| **`prune_keep50_finetuned`** | 1.45 M | **0.980** | **0.627** |

Recall returns to baseline exactly at both levels, including the heavy prune
that had lost two points, and most of the calibration damage reverses with it.

**So the intermediate state is not the result.** `tools/prune.py`'s own
docstring says why: *"the point of the cycle is what the fine-tune recovers,
and a table that only shows the end state cannot say whether the pruning
hurt."* The reverse error is just as easy — reporting the state *before* the
recovery and calling it the cost.

The honest verdict is narrower than either. **Pruning is free in accuracy once
fine-tuned, and still buys nothing measurable**: no latency, since the model is
launch-bound; no peak memory, since the attention scores dominate and the
feed-forward does not; and 15% of parameters on a model whose weights are
6.5 MB against roughly 78 MB of activations. It is not harmful. There is simply
no reason to want it here, which leaves **TensorRT as the only compression that
pays.**

**And pruning does not buy what it is usually bought for.** Halving the
feed-forward hidden width left peak memory at 7.04 GiB against roughly 7.09
unpruned: the dominant tensors are the attention scores, not the feed-forward.
It does not reduce latency either, since `tools/cost.py` puts this model at
launch-bound at batch 1 — 16x the parameters moved latency 2%, while halving
*depth* moved it 29%.

> **For deployment: quantize the trunk and matcher to INT8, leave the solve in
> fp32, and reach for depth rather than pruning if more is needed.**


### Distillation, three seeds a side — and the variance is an optimisation failure

*(`tools/run_sequence.py` on the test split, 120 scenes, 8 880 frames;
open-loop recall from `tools/eval.py`. Three seeds a side, 40 epochs each,
`mapposeformer/distill.py` — `scripts/run_arms.sh` arms `student_s{0,1,2}`
against `student_no_teacher_s{0,1,2}`. Measured 2026-01-17.)*

The student is the teacher's configuration at two layers: 0.914 M parameters
against 1.709 M. `student_no_teacher_*` is the same arm with no teacher, which
is the only thing that makes the distilled arms readable.

| seed | distilled | no teacher | distilled | no teacher |
|---|---|---|---|---|
| | recall | recall | closed loop | closed loop |
| s0 | 0.967 | 0.951 | **0.091** | 0.106 |
| s1 | 0.890 | 0.893 | 0.252 | 0.523 |
| s2 | 0.957 | 0.889 | **0.092** | 0.571 |
| mean | 0.938 | 0.911 | 0.145 | 0.400 |

**Open loop the gap is +2.70 pp at t = 0.86 — not separated.** Closed loop the
distilled arm wins on three of three seeds, by 1.2x, 2.1x and 6.2x. The paired
t is 1.90 against a critical 4.303 at two degrees of freedom, so the effect is
large, consistent in sign, and **not certified at three seeds**.

**Two of three distilled seeds match the teacher exactly** — 0.091 and 0.092
against its 0.091 — at 53% of the parameters. The third does not, and the
reason is visible long before the filter:

| epoch | 9 | 10 | 15 | 20 | 27 |
|---|---|---|---|---|---|
| s0 | 0.472 | 0.406 | 0.348 | 0.291 | 0.285 |
| **s1** | **0.695** | **0.675** | **0.561** | **0.531** | **0.496** |
| s2 | 0.440 | 0.355 | 0.300 | 0.324 | 0.283 |

**Seed 1 is in a worse basin by epoch 9 and never leaves it.** It is not
unstable — it descends monotonically — it is simply on a worse trajectory from
the start, and its no-teacher counterpart does the same thing. So this is the
initialisation and the data order, not the teacher.

**That is the finding, and it is about depth rather than distillation.** The
four-layer teacher trained once and converged. The two-layer student reaches the
good basin on two runs of three, and its untaught control on one of three. Depth
was bought for accuracy in the depth sweep; it also buys *optimisation
reliability*, which no open-loop accuracy table shows.

It also changes what the comparison should measure. A mean over three runs, one
of which failed to converge, is not a description of either arm — the quantity
that matters is **how often a run reaches the good basin, and whether the
teacher changes that rate**. Three seeds estimate a rate near one-in-three
appallingly, so six a side were run; the next section is what they said.

### Six seeds a side: distillation separates on calibration, not accuracy

*(Same protocol as above, six seeds each. `student_s{0..5}` against
`student_no_teacher_s{0..5}`, 40 epochs, test split, 8 880 frames. Paired by
seed, because both sides use the same six initialisations.)*

| seed | recall | | NEES | |
|---|---|---|---|---|
| | distilled | no teacher | distilled | no teacher |
| s0 | 0.967 | 0.951 | 0.760 | 0.507 |
| s1 | 0.890 | 0.893 | 0.455 | 0.452 |
| s2 | 0.957 | 0.889 | 0.767 | 0.355 |
| s3 | 0.969 | 0.966 | 0.790 | 0.752 |
| s4 | 0.902 | 0.891 | 0.599 | 0.285 |
| s5 | 0.960 | 0.960 | 0.777 | 0.628 |
| **mean** | **0.941** | **0.925** | **0.691** | **0.497** |

**Recall does not separate, at six seeds as at three.** +1.58 pp, paired
t = 1.46 against a critical 2.571. The prediction registered before any of this
was measured said distillation would not separate on recall, and it does not.

**Calibration separates.** +0.195 of NEES, paired t = 2.98, p < 0.05, and six
of six seeds improve. The mean moves from 0.497 to 0.691 against the honest
0.789 — the untaught student is badly over-wide, the distilled one is close to
right.

So the teacher transfers *calibration*, not accuracy. That is the finding, and
it is not the one a compression table would look for: a student distilled to
match its teacher's assignment ends up with a covariance that means what it
says, while the same architecture trained on labels alone does not.

**It also explains the closed loop.** A filter weights each correction by the
covariance it is handed, so an over-wide one under-weights every correction and
collects less. That is why the two arms differ by 15 mm through the filter and
by nothing measurable frame by frame — 0.091 m against 0.106 m at seed 0, where
open-loop recall differed by 1.6 pp and did not separate.

**What still does not separate is the failure rate.** Three of six untaught runs
reach the good basin and five of six distilled ones do, which is 5/6 against 3/6
— a Fisher exact p of about 0.55, nowhere near a result. Seed 1 fails in both
arms, so that failure is the two-layer optimisation problem rather than anything
the teacher could fix. Distinguishing a rate near one-half needs about twenty
seeds a side, not six, and this project does not have a claim about it.

### The compression comparison, measured

*(Everything below is the test split, 8 880 frames. Open-loop deltas are paired
McNemar against the row's own baseline; closed loop is `tools/run_sequence.py`
over 120 sequences. Pruned rows are fine-tuned 10 epochs, and each side carries
the control that fine-tunes without pruning.)*

| | params | weights | Δ recall, paired | closed loop |
|---|---|---|---|---|
| **teacher** | 1.709 M | 6.52 MiB | — (96.3%) | **0.091 m** |
| teacher, fine-tuned | 1.709 M | 6.52 MiB | +0.91 pp — separated | 0.090 m |
| teacher, pruned 7.7% | 1.58 M | 6.0 MiB | +0.72 pp — separated | 0.092 m |
| teacher, pruned 15.4% | 1.45 M | 5.5 MiB | +1.10 pp — separated | 0.092 m |
| teacher, INT8 | 1.709 M | **1.69 MiB** | −1.06 pp — separated | 0.093 m |
| **distilled student** | **0.914 M** | 3.49 MiB | — (96.7%) | **0.091 m** |
| student, fine-tuned | 0.914 M | 3.49 MiB | +0.35 pp — separated | — |
| student, pruned 7.2% | 0.85 M | 3.2 MiB | +0.25 pp — not separated | — |
| student, pruned 14.4% | 0.78 M | 3.0 MiB | +0.15 pp — not separated | — |
| student, INT8 | 0.914 M | **0.90 MiB** | −0.23 pp — separated | 0.092 m |
| student, **no teacher** | 0.914 M | 3.49 MiB | −1.54 pp | **0.106 m** |

**Every row lands within 2 mm of 0.091 m except the one that had no teacher.**
That is the whole table in a sentence. The registered prediction — no variant
moving the closed loop by more than about 5 mm — holds for every compression
step and fails only where the comparison was never about compression.

**Pruning contributes nothing, at either size.** On both models the arm that
restarts the schedule and prunes nothing gains as much or more than the arms
that prune: +0.91 against +0.72 and +1.10 on the teacher, +0.35 against +0.25
and +0.15 on the student, and on the student only the control separates. What a
prune-and-fine-tune buys is the learning-rate restart. Take pruning for a memory
budget — 15% of the parameters leave at no measurable cost — and never for
accuracy.

**INT8 is the only step that pays, and it pays in bytes.** 74% of the weights on
both models, for 2 mm on the teacher and 1 mm on the student. The student
quantizes better than the teacher by every measure — −0.23 pp against −1.06,
9.8 mm of pose movement against 32.2, a NEES-ratio tail of 2.10 against 8.35 —
and the untaught student behaves the same way (−0.21 pp), so that robustness is
**depth, not distillation**: two layers accumulate less quantization error than
four.

**And none of it buys speed.** Compiled to fp16 the solve is 85% of the frame,
so the ceiling on compressing the network is 1.17x end to end. The student's
half-size trunk is about 4% of a frame. These rows are a memory table, not a
latency table, and reading them as a Pareto frontier would be reading them
wrong.

### Registered before the compression comparison is measured

The distilled students and their no-teacher controls are still training, and the
full comparison — teacher, distilled student, student trained alone, pruned,
quantized — is not yet takeable. Writing the expectation down first is what
makes it a result rather than a story told afterwards, and this file has one
prediction that was confirmed and one that had to be withdrawn.

**The prediction.** Every compression variant lands within about 5 mm of the
teacher's 0.091 m closed loop, and none moves end-to-end latency by more than
about 10%. The distilled student beats its no-teacher control by less than that
control's own seed band, which two seeds put at 5.8 pp — so distillation does
not separate at three seeds a side.

**Why.** The solve is 85% of the compiled frame, so the network's whole share is
15% and the largest structural change available — two layers instead of four,
-29% on the network — is about 4% end-to-end. Pruning reaches only the
feed-forward width and moves latency by roughly nothing; INT8 is capped at
1.17x by the same arithmetic. And the filter has already been shown to absorb a
separated open-loop regression: INT8's -1.06 pp arrived as 2 mm.

**What would falsify it.** A distilled student clearing 5.8 pp over its control,
or any variant moving the closed loop by more than 5 mm. Either would mean the
network matters more than the latency breakdown says, and the breakdown would be
the thing to re-examine.

**What it is worth if it holds.** Not a ranking. The useful statement would be
that compressing this network is free and unnecessary in equal measure, because
the network is not the bottleneck — which is a deployment conclusion, and a
more transferable one than knowing which variant won.

### Pruning the teacher: the gain was the schedule, not the pruning

Pruned and fine-tuned for ten epochs, the teacher comes back **better** — and the
more aggressive prune comes back better still, which pruning does not do. What
the two arms share is not the pruning. It is ten further epochs from a
checkpoint whose best epoch was 19, under a **fresh warmup-and-cosine
schedule**. A learning-rate restart is a known way off a plateau, and the run
had early-stopped at 31 after twelve epochs of no improvement.

So the comparison needs an arm that restarts the schedule and prunes nothing:

| arm | params removed | Δ recall vs the teacher, paired over 8 880 frames |
|---|---|---|
| **fine-tuned, not pruned** | **0%** | **+0.91 pp ± 0.33 — separated** |
| pruned to keep 0.75, fine-tuned | 7.7% | +0.72 pp ± 0.34 — separated |
| pruned to keep 0.50, fine-tuned | 15.4% | +1.10 pp ± 0.32 — separated |

**All three gains are the restart.** Both pruned arms land within 0.2 pp of the
arm that pruned nothing, against a half-width of 0.33 — so pruning's own
contribution is not separable from zero in either direction. Read without the
control, this table says "pruning improves the model by up to 1.1 pp, and more
pruning is better", which is false and would have been easy to publish.

What survives is worth having, and it is a different claim: **structural
pruning here is free rather than profitable.** Fifteen percent of the
parameters can go at no measurable cost to recall, and the calibration moves
with the fine-tune rather than with the pruning — NEES 0.908 to 0.861 for the
control against 0.883 at keep 0.50. If the reason to prune is a memory budget,
take it. If the reason is accuracy, fine-tune and keep the weights.

The second lesson is about this project's own method. `aug_on_*` cost six
40-epoch arms for want of a matched control, and that was written down as a
cautionary case hours before this table produced the identical trap. Writing the
lesson down is not the same as applying it.

## `rope_bands` — an interior optimum at 5, and the wavelength argument rescued

*(Three cells, `head_dim` 64, one recipe — same epochs, same `grad_clip`, same
residual — differing only in the band count: 10 derived (3 seeds), 5 pinned
(2 seeds), 3 pinned (1 seed).)*

`RotaryFrames` rotates `6 * bands` channels and passes the rest through, so
positional resolution is bought out of content capacity:

| bands | rotated | content | shortest wavelength | recall @25 cm | sd |
|---|---|---|---|---|---|
| 3 | 18 | 46 | 7.500 m | 0.940 | — *(1 seed)* |
| **5** | 30 | 34 | 1.875 m | **0.9585** | 0.64 pp |
| 10 | 60 | 4 | 0.059 m | 0.9280 | 1.97 pp |

**Five is an interior maximum**: 3 loses 1.85 pp on one seed and 10 loses
3.05 pp at 2.5 sd. So it is not "content always wins" — it is a genuine trade
with an optimum, and the value the width sweep happened to pin is the right one.

**This also rescues the wavelength argument, which I had written off.** At 3
bands the shortest wavelength is 7.5 m — four times the 1.714 m map point pitch
— and it costs accuracy exactly as that reasoning predicted. The error was never
that wavelength did not matter. It was assuming *only* wavelength mattered, and
never printing what the resolution cost in content channels. One line of
arithmetic, `6 * bands` against `head_dim`, governs both halves.

**And the arithmetic points past the cells that were run.** At `head_dim` 64,
`rope_bands=8` rotates 48 channels and leaves 16 for content, with a half-width
of 0.268 m against the 0.25 m recall gate, at zero parameter cost. The sweep
measured 3, 5 and 10, so 8 is a prediction from the channel budget rather than
a result.

### And at `head_dim` 128 the band count stops mattering

The 256-wide cell rerun with 10 bands instead of 5 is the same single-variable
change, one width up:

| `head_dim` | bands | rotated / content | recall |
|---|---|---|---|
| 64 | 5 | 30 / 34 | **0.9585** |
| 64 | 10 | 60 / **4** | 0.9280 |
| 128 | 5 | 30 / 98 | 0.9515 |
| 128 | **10** | 60 / 68 | **0.9520** |

**At `head_dim` 128 the two settings are indistinguishable** — 0.9515 against
0.9520 — where at 64 they differed by 3.05 pp. That is exactly what the content
budget predicts: at 128 channels both settings leave plenty for content (98 and
68), so the trade does not bite. It bites only when one side is starved, and 10
bands at `head_dim` 64 leaves four.

> **So the band count is not a global knob to tune. It matters only when
> `6 * bands` approaches `head_dim`.** Deriving it is what fails —
> `head_dim // 6` gives 10 at 64 and 21 at 128, and the first starves while the
> second is numerically dead at a 2.9e-5 m wavelength. **State `rope_bands`
> wherever `head_dim` changes, and check `6 * bands` against it.**

This also settles the width verdict against its one remaining objection. The
width sweep pinned 5 bands everywhere, so `dim256`'s loss to `dim128` could have
been the encoding rather than the width. At 10 bands `dim256` scores 0.952 —
still below `dim128`'s 0.9585. **Saturation at 128 is real at both band
settings.**

## Geometry — the equivariance claim is load-bearing

*(Six arms, three seeds a cell, one recipe: 25 epochs, cached, `grad_clip`
10.0. One cell is `point_geometry=rope`, the other `absolute`, and nothing else
differs. Measured 2025-12-28.)*

| cell | recall @25 cm | mean | sd | NEES median |
|---|---|---|---|---|
| **rope** | 0.934 / 0.944 / 0.906 | **0.9280** | 1.97 pp | 0.637 / 0.259 / 0.937 |
| absolute | 0.830 / 0.833 / 0.844 | 0.8357 | 0.74 pp | 0.393 / 0.512 / 0.555 |

**+9.2 pp, SE 1.21 pp — 7.6 sd.**

This sweep tests whether the equivariance is decorative, and states the reason
it might not be: at point resolution, two points on two parallel lane lines at
the same station have **identical content features**, so the only thing
separating them is their frames. `absolute` adds position to the token and has
to learn the invariance; `rope` carries relative position into the score and is
translation-invariant by construction.

**Learning it costs 9.2 points of recall.** The claim is load-bearing.

Note the asymmetry in the spreads: `rope` is the *noisier* cell at 1.97 pp
against `absolute`'s 0.74. Both derive `rope_bands` — but the `absolute` cell
does not use `RotaryFrames` at all, so band count is simply irrelevant to it.
The two cells are not noisy for comparable reasons and the spreads should not
be read against each other.

## Width, and a pattern that now holds on three axes

*(`rope_bands` pinned at 5 in every width cell, 25 epochs, cached, measured
2025-12-28.)*

| width | recall @25 cm | sd | NEES median, mean of cell |
|---|---|---|---|
| `dim64` (3 seeds) | 0.893 / 0.918 / 0.900 → **0.9037** | 1.29 pp | **1.000** |
| `dim128` (2 seeds) | 0.963 / 0.954 → **0.9585** | 0.64 pp | **0.108** |

**+5.5 pp, SE 0.87 pp — 6.3 sd.** Width separates — but **not cleanly, and the
gap is not all width.**

Every width cell pins `rope_bands=5`, and `RotaryFrames` rotates `6 * bands`
channels out of `head_dim`, leaving the rest for content. `head_dim` is
`dim // heads`, so it moves with the width:

| cell | `head_dim` | rotated | **content** | recall |
|---|---|---|---|---|
| `dim64` | 32 | 30 | **2** | 0.9037 |
| `dim128` | 64 | 30 | 34 | 0.9585 |
| `dim256` | 128 | 30 | 98 | 0.945 |

**`dim64` has two content channels.** That is the identical starvation the
heads analysis used to reject its `head_dim` 32 cell, and it means "`dim128`
beats `dim64` by 5.5 pp" conflates *width* with *content capacity*. The sweep
cannot separate them.

**The comparison that matters is unaffected.** Saturation between 128 and 256
is 34 content channels against 98 — neither starved — so `dim256_s0` at 0.945
against `dim128`'s 0.9585 remains a fair reading. What is no longer safe is the
claim that narrowness *per se* costs 5.5 pp.

Checked against the other sweeps: tokens, residual and depth all compare cells
with identical `head_dim` and identical band counts, so they are unaffected.
Geometry compares `rope` at four content channels against `absolute`, which
uses no rotation at all — so its 9.2 pp is if anything a **lower bound** on what
rope is worth.

**And the capacity-versus-calibration trade now holds on three independent
axes**, which is what turns it from a coincidence into a property of this
model:

| axis | smaller | larger | NEES, smaller → larger |
|---|---|---|---|
| depth | `L2` | `L4` | 1.016 → 0.081 |
| **width** | **`dim64`** | **`dim128`** | **1.000 → 0.108** |
| residual *(the repair)* | line @ 4 layers | point-to-point @ 4 layers | 0.04–0.20 → 0.68–0.76 |

**Capacity buys recall and destroys the line residual's covariance, by about
tenfold, and it does not matter whether the capacity is bought with layers or
with width.** The small models are the honest ones and the large models are the
accurate ones — except with point-to-point, which is the only configuration
measured that is both. That is why the residual verdict matters *more* as the
teacher grows, and it is the clearest argument this project has produced for
the design it chose.

**What `dim256` has to answer is therefore not "is wider better".** That is
settled twice over. It is whether accuracy has **saturated** by 128, because
the roadmap sizes a teacher by growing it until it does, and that single number
sets the teacher's width.

### `dim256`, both seeds: accuracy saturates at 128

| width | recall @25 cm | NEES median |
|---|---|---|
| `dim64` (3 seeds) | 0.9037 | ~1.000 |
| `dim128` (2 seeds) | **0.9585** | 0.108 |
| `dim256` (2 seeds) | 0.945 / 0.958 → **0.9515** | 0.290 / 0.106 |

Read against the rule registered before the arm reported — a gain needs 0.990,
because one seed against two at this regime's 1.29 pp noise gives SE 1.58 pp —
**0.945 is not separated from `dim128`**, and the point estimate is 1.35 pp
*lower* (−0.85 sd). Doubling width again buys nothing measurable.

**And `dim256` is not handicapped in this comparison.** At `head_dim` 128 with
5 bands it carries 30 rotated channels and **98** for content, against
`dim128`'s 30 and 34. Strictly more capacity of both kinds, and no better
result. `dim64 → dim128` bought 5.5 pp at 6.3 sd; `dim128 → dim256` buys
nothing.

**The second seed agreed: 0.958, mean 0.9515, −0.70 pp against `dim128` at
−0.89 sd.** Both `dim256` seeds land inside `dim128`'s range.

| step | gain |
|---|---|
| `dim64` → `dim128` | **+5.5 pp** |
| `dim128` → `dim256` | **−0.70 pp** |

> **So the teacher is `dim` 128.** `ROADMAP` sizes a teacher by growing it
> *"until accuracy saturates"*, and saturation is at 128 — a 256-wide teacher
> would be four times the parameters for nothing. Sizing a teacher from the
> roadmap rather than from a measurement is exactly what a saturation sweep
> exists to prevent.

The cascade is favourable. At `dim` 128 with `inner = dim`, `head_dim` derives
to **64** — the configuration the depth, geometry, tokens and residual sweeps
all ran. The teacher stops being a novel architecture and becomes one this
project has already measured from four directions.

> Held in view: `dim128` is two seeds. At 6.3 sd the direction is not in doubt;
> the magnitude would be firmer with a third seed.

## Depth — the first sweep that separated cleanly

*(Six arms, three seeds a cell, launched together under the same recipe: 40
epochs, cached, `grad_clip` 10.0, `rope_bands` derived to 10. Measured
2025-12-28.)*

| cell | recall @25 cm+0.5° | mean | sd | NEES median |
|---|---|---|---|---|
| `L2` 2 layers | 0.866 / 0.867 / 0.864 | 0.8657 | **0.15 pp** | 0.989 / 1.057 / 1.001 |
| `L4` 4 layers | 0.968 / 0.970 / 0.953 | **0.9637** | 0.93 pp | 0.040 / 0.048 / 0.156 |

**+9.8 pp, SE 0.54 pp — 18 sd.** After a tokenizer result at 6.8 sd and a heads
result that could not reach 1.1, this is the first structural contrast that
separated without argument, and it did so because both cells let `rope_bands`
derive rather than pinning it. **Two layers is not enough.**

**The calibration goes the other way, and by more.** Mean NEES is 1.016 at two
layers against 0.081 at four — **12.5× apart**, and on opposite sides of the
0.789 target: the shallow model is mildly overconfident, the deep one wildly
pessimistic. Depth buys accuracy and sells calibration, on the *same* residual.

> **This is a property of depth *with the line residual*, not of depth.** The
> point-to-point arms are also four layers, and they score NEES 0.681–0.761
> with a covariance 1.10× the reference and the right way up. So the repair for
> the deep model's covariance is the residual, which is already the verdict —
> and [the refcov comparison](#the-over-wide-covariance-is-depth-and-at-four-layers-it-inverts)
> shows what the line residual does at this depth: 17.7× too wide, inverted,
> principal axis 78.5° out.

The practical reading for the teacher: **4 layers, point-to-point**. That pair
is the only configuration measured here with both the recall of depth and an
honest covariance, and it has three seeds behind it.

## Why the heads arms were so noisy — registered before the width arms report

*(Written 2025-12-31 11:55, with the 64-wide cell mid-training and no epoch
measured. Registered in advance because it is a post-hoc explanation of a null,
which is the easiest kind of story to tell and the hardest to trust.)*

The heads cells could not be separated because their seeds disagreed by 5.2 to
9.2 points of recall. The depth cells, same codebase and same hardware,
disagree by **0.1 pp** — two 2-layer seeds at 0.866 and 0.867, with NEES 0.989
against 1.057 where the heads cells varied tenfold. Whatever made the heads
arms noisy is therefore **not** intrinsic to this project's seed variance.

**The candidate is `rope_bands=5`.** It is the one structural difference: the
heads arms pinned it, the depth arms let it derive to 10. At 5 bands the
shortest wavelength is 1.875 m, *coarser than the 1.714 m map point pitch the
point-token arm exists to resolve* — so the encoding cannot cleanly separate
adjacent map points, and the assignment has to settle a genuinely ambiguous
problem. A model resolving an ambiguity it has no information to resolve is
exactly a model whose answer depends on its initialisation.

> **The prediction does not hold, and the way it first looked confirmed is the
> lesson.** The 64-wide cell came back at sd 1.29 pp against the 2-layer cell's
> 0.15, F = 71.3 — two cells chosen out of eleven. Across all eleven the
> grouping does not hold: the derived-band control has **sd 1.97 pp**, noisier
> than the pinned-5 128-wide cell's 0.64. Excluding the heads cells the two
> groups average 0.97 and 0.99 pp — no difference at all. The heads cells were
> noisy for a reason of their own, and generalising from them was the mistake.
>
> Registering a prediction in advance is worth nothing if it is then checked
> against a subset. That is the error, and it is worse than being wrong about
> bands.

## Heads — the arms reported, and the threshold was wrong

*(Measured 2025-12-31 on the test split, 0 failures. Read this against the
registration above, which was written before any arm had reported an epoch.)*

| cell | seeds | recall @25 cm | mean | within-cell spread |
|---|---|---|---|---|
| A 2×64 *(control)* | s0 / s1 | 0.915 / 0.967 | 0.9410 | **5.2 pp** |
| B 4×64 | s0 / s1 | 0.964 / 0.972 | 0.9680 | 0.8 pp |
| C 2×32 | s0 / s1 | 0.967 / 0.875 | 0.9210 | **9.2 pp** |

**The registered rule fired, and it has to be rejected anyway.** B beats A by
2.70 pp against a 2.5 pp threshold, which reads as a refutation of the count
argument. It is not one. That threshold was built on sd 0.0127 — the
seed band of the three point-token seeds, a *different configuration* — and
these cells do not have that spread. Their pooled sd is **4.33 pp**, 3.4×
larger. Against the spread actually observed:

| contrast | gap | SE(diff) | separation |
|---|---|---|---|
| B vs A | +2.70 pp | 2.63 pp | **1.03 sd** |
| C vs A | −2.00 pp | 5.28 pp | **0.38 sd** |

Neither is readable. Importing an error bar from another configuration is the
mistake, and here it very nearly converted noise into a reported architectural
finding.

**The variance is inside the cells, not between them.** Cell C's two seeds
differ by 9.2 points of recall at *identical* configuration, which is larger
than any gap between cells. Calibration is worse still: NEES varies about 10×
within a single cell — A reads 0.675 and 0.070, C reads 0.996 and 0.085. Any
single-seed calibration claim anywhere in this repository should be read with
that number in mind.

**More seeds do not rescue it at any sane price.** Holding the observed
spreads, a third seed moves B vs A from 1.03 to 1.26 sd — still nothing. The
first readable count is about **eight seeds a cell**, and if the pooled 4.33 pp
is the honest figure it is twenty-one. That is 18 to 57 further arms, one to
two days of the whole machine, to resolve a knob the rank argument predicts is
not binding.

> **Verdict: not separated, and not worth separating.** `2 heads × 64` stays,
> on the registration's own terms — *"neither knob is worth the parameters"*,
> B costing +62% and C saving −31%. This is **not** evidence that the rank
> argument is right; it is a measurement with too little power to test it, and
> the registration said so in advance.

**Narrow heads are not slow ones, so the rank cap is not a latency argument.**
`ROADMAP.md` sizes `head_dim` by rank, not by tensor-core occupancy: measured
over 1 344 point tokens, 4 heads at `head_dim` 32 is the *fastest* of the
three — p50 7.632 ms, against 7.972 at 2×64 and 8.444 at 1×128.

## What a configuration costs

*(2025-12-26, `tools/cost.py` on the RTX 2070, synthetic tensors, no
dataloader. Milliseconds, p50 of 30.)*

| cell | params | fwd b1 | fwd b64 | fwd+bwd b64 | peak MiB |
|---|---|---|---|---|---|
| heads A 2×64 | 1.709 M | 34.00 | 160.44 | 575 | 4843 |
| heads B 4×64 | 2.764 M | 37.22 | 260.27 | 994 | 6199 |
| heads C 2×32 | 1.181 M | 33.80 | 142.80 | 462 | 4181 |
| tokens point | 1.709 M | 36.48 | 176.92 | 612 | 5006 |
| tokens element | 1.813 M | 43.80 | **78.78** | 149 | 3214 |
| layers 2 | 0.914 M | 24.33 | 103.47 | 341 | 3149 |
| layers 4 | 1.709 M | 34.21 | 177.57 | 613 | 5006 |
| `dim` 64 | 0.433 M | 34.16 | 131.89 | 418 | 3028 |
| `dim` 256 | 6.793 M | 34.71 | 354.14 | OOM | 6565 |

**"Parameters are close to free" is true at batch 1 and false at batch 64.**
`dim` 64 → 256 is **15.7× the parameters** for **+1.6%** of batch-1 latency —
the roadmap's claim, confirmed — but **2.69×** at batch 64. Training runs at
batch 64, so parameters are not free for the thing that costs days; they are
free for the thing that costs milliseconds. The roadmap quotes the batch-1
figure while sizing a model by training cost.

**"Tokens are what cost" holds, and it is the stronger claim.** Point tokens
have *fewer* parameters than element tokens — 1.709 M against 1.813 M — and
cost **2.25×** as much at batch 64. That is the price of the 12.7-point recall
gain, and it is worth stating next to it.

**At batch 1 the head cells are within 10% of each other** (33.80, 34.00,
37.22), which is this model being launch-bound at batch 1: whatever the heads
arms say about accuracy, head count is nearly free at inference and expensive
in training — B costs **1.73×** the step time (994/575 on forward *and
backward*; the forward-only ratio is 1.62, and a training step is not
forward-only) and 1 356 MiB more.

The practical consequence: `dim` 256 could not complete a backward pass in
7.6 GiB, so a teacher at that width needs the A6000 or gradient checkpointing.

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

### Is the shape right?

A covariance can pass every scalar test and still have the wrong shape. This is
the number that catches that:

| | reported `σ_long/σ_lat` | actual ratio | |
|---|---|---|---|
| reference network | 0.809 | 3.030 | **mixed aggregations**, not comparable; see [the aggregation trap](#the-covariance-can-be-calibrated-and-still-be-wrong) |
| point-to-point, on geometry alone | 1.01 | — | blind to the landmarks |
| **point-to-line, 4 layers** | **1.866** | 2.334 | right-side-up, within 20% |

**That "actual ratio" column is a marginal, and the covariance is a
conditional — they are not the same quantity.** 2.334 is the ratio of
longitudinal to lateral RMSE *aggregated over all 8 880 frames*, which
includes how much the error varies **between** frames. A covariance is a claim
about one frame: given this geometry and these landmarks, here is the spread.

`tools/refcov.py` measures the conditional directly, which is possible here
only because the prior is *drawn* — fix a scene and a frame, redraw the sample
seed 64 times, and the spread of the 64 resulting errors is what the model
ought to be reporting. On the 4-layer point-to-line arm, 96 frames × 64 draws:

| | σ_long | σ_lat | ratio |
|---|---|---|---|
| measured per-frame reference | 0.1008 | 0.0930 | **1.083** |
| reported by the model | 0.3702 | 0.2627 | **1.409** |

Two corrections follow. The covariance is **2.8× to 3.7× too wide**, not the
2.27× that NEES 0.153 implies — the scalar understates it. And the per-frame
error is **very nearly isotropic**, so the model over-states its anisotropy by
about 30% rather than being "right-side-up within 20%".

The ellipse points along the road, which is where the error is. But "the
ellipse points along the road" is a statement about the *marginal* error, and
at the per-frame level the road direction is much less distinguished than that
table suggests. What is conditionally anisotropic and what is marginally
anisotropic are different claims, and a table of marginals cannot tell them
apart.

That the ellipse points the right way at all is the whole of M3 in one line,
and it is not something a better-tuned scale head could have achieved —
point-to-point's translation Hessian is `2·mass·I` whatever the landmarks are.

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

## Augmentation — not separable, and it was never what made those arms look good

The augmented arms reached 0.977 mean recall and two of their three seeds
trained the full 40 epochs without ever tripping patience. That is the
signature of augmentation removing an overfit, and it could not be read that
way, because those arms had no matched control: their nominal one — the
point-token arms — ran `geometry=relative` where they ran `rope`, and worse,
those arms have `augment: true` in their stored config while `set_epoch` was
never called when they trained, so **the config field does not distinguish the
two conditions**. Nothing on disk said which arms actually saw fresh noise.

The no-augmentation arms are that control: same recipe, same `grad_clip`, same
40 epochs, `data.augment=false` stated explicitly so the checkpoint records the
condition.

| (all `residual=line`, 40 epochs) | `trans` m | recall @25 cm | NEES |
|---|---|---|---|
| augmented, s0/s1/s2 | 0.230 / 0.240 / 0.209 | 0.982 / 0.968 / 0.982 | 0.022 / 0.079 / 0.036 |
| no augmentation, s0/s1/s2 | 0.245 / 0.246 / 0.266 | 0.971 / 0.968 / 0.974 | 0.037 / 0.043 / 0.035 |

**Augmentation is worth +0.63 pp of recall at t = 1.27 — not separated.**
Translation is 0.026 m better with it, at t = 2.28, which is under the 2.78 this
project's three-seed arms need at df = 4. Suggestive on translation, absent on
recall, and not enough to act on either way.

**So the 0.982 was not augmentation.** Turning it off keeps 0.971 of it. What
produced it is the variable nobody was testing: these arms run
`model.residual=line`, and their NEES of 0.022–0.079 against the honest 0.789 is
the same catastrophic over-wide covariance the depth sweep found. The line
residual buys roughly 1.5 pp of recall over the teacher's 0.963 and pays for it
with the entire covariance — which is the residual verdict already on the
record, arrived at a second time from a different direction.

The cost of finding this out was six 40-epoch arms, three of them uncached and
therefore slow. The cheaper version existed: state the control's condition in
the arm definition rather than trusting a config field, and the first three arms
would have been readable on their own.

## M4 — closed loop

*(`tools/run_sequence.py` on the teacher checkpoint, 120 test scenes, 8 880
frames, `measurement=information`. Measured 2026-01-08.)*

| | open loop | closed loop | gain |
|---|---|---|---|
| reference network, six-run mean | 0.258 | **0.089** | **2.9x** |
| 4 layers, covariance 2.3x too wide | 0.289 | 0.188 | 1.54x |
| **teacher, NEES 0.909** | 0.271 | **0.091** | **2.98x** |

**An over-wide covariance costs half the filter gain.** The 1.54x row is the
point-to-line arm, whose covariance is [2.3x too wide](#is-the-size-right):
imperfect correspondences from a matcher at 84.2% recall inflate
`s² = cost / (dof − 3)`, so the filter under-weights every correction it is
handed. The teacher matches at 96.3% and reports NEES 0.909 out of the same
solve, and the gain is 2.98x, against the reference network's 2.9x.

Per axis it is not merely close, it is the same model:

| closed loop | `long` | `lat` | `yaw` |
|---|---|---|---|
| reference network | 0.058 | 0.067 | — |
| inflated covariance | 0.150 | 0.115 | — |
| **teacher** | **0.060** | **0.069** | **0.128°** |

The inflated run was 2.6x worse along track and 1.7x worse across it. The
teacher is within 3 mm of the reference on both. An over-wide covariance does
not damage a filter diffusely — it under-weights every correction, and the axis
carrying the most correction loses the most.

**The loop is healthy by every margin it has.** No scene diverged, 99.9% of
frames were accepted, and the longest run of consecutive refusals was 2 frames
— short enough that the filter coasts on its own propagation and recovers. The
0.1% refused were refused on `mass`, meaning too little map was in view to
constrain a pose, which is the gate doing its job rather than failing.

`trust` refused nothing, and it never could: this design has no trust head, so
`engine/sequence.py` passes `trust=1.0` and the counter is vestigial.

**This is the number the project exists to produce.** A single frame lands at
0.271 m; the same model through the filter lands at **0.091 m**, three times
better, because the covariance is honest enough for the filter to know how much
to believe it. Every structural verdict above — point tokens, point-to-point,
four layers, rope — was chosen against a calibration target rather than an
accuracy one, and this is what that choice buys. An accurate pose with a
dishonest covariance would have measured 0.188 m here and looked fine in every
open-loop table on this page.

## The covariance can be calibrated and still be wrong

The reference network's NEES median of **0.786** against a 0.789 target is
honest by every scalar test the harness applies, and its covariance is still
wrong. Where it is wrong is not where a median of per-frame ratios says it is.

> **The instrument was verified before it was believed**, against M1's seed-0
> row — longitudinal 0.3222 against 0.322, lateral 0.0952 against 0.095, yaw
> 0.2091° against 0.209°, translation 0.3360 against 0.336, recall 0.9600
> against 0.960, NEES median 0.7860 against 0.786. Six of six to four decimals,
> so what follows is the same model.

**A reported 0.809 against a measured 3.030 compares two different
populations.** The 0.809 is a *median of per-frame ratios* and the 3.030 a
*ratio of RMS*. With a heavy-tailed error those are not the same statistic, and
this error is violently heavy-tailed. Measured like for like:

| aggregation | reported | measured | same side of 1.0? |
|---|---|---|---|
| medians both sides | 0.793 | 0.711 | **yes** |
| RMS both sides | 1.054 | 3.385 | **yes** |
| *median against RMS* | *0.809* | *3.030* | *no* |

The sign of the discrepancy flips with the aggregation, which is the signature
of comparing incomparable things. **There is no inversion.** 99.1% of frames
report `σ_long < σ_lat`, and 59.9% of frames genuinely have
`|e_long| < |e_lat|`. The ellipse points the right way for the typical frame.

**The real failure is a tail, and it is worse than an inversion would be.**

| | median | 99th | 99.9th | max |
|---|---|---|---|---|
| `\|err_long\|` | 0.045 | 0.211 | **4.049** | 7.999 |
| `\|err_lat\|` | 0.063 | 0.259 | 0.332 | 0.461 |
| `\|e_long\|/σ_long` | 0.671 | 2.946 | **62.1** | 120.6 |

Along-track error jumps 19× between the 99th percentile and the 99.9th, and
the reported σ tracks the first and not the second. On the **worst 1% of
frames the error averaged 2.249 m against a claimed 0.308 m — 20.6×
overconfident** — while the other 99% sat at 0.80×, slightly conservative.

So the model is honest on 99% of frames and catastrophically wrong on the 1%
where along-track geometry aliases. Every median statistic calls that
calibrated, and it *is* calibrated, on 99% of frames. The RMS "3.030" was never
the shape of the ellipse; it was that 1%, arriving in a statistic that had no
way to say so.

**This makes M4 more important, not less.** A filter is not harmed by the 99%.
It is harmed by a frame it is told is certain and is not, because that is the
one it cannot recover from — and "ambiguity widens the covariance rather than
gating the frame" is precisely the repair this diagnosis calls for.

`Calibration` reports `tail_z` per axis — `|z|` at the 99.9th percentile over
the calibrated 3.29 — which reads **19.0×** on the reference network's
longitudinal axis and 1.3× and 1.4× on the two that were fine. That is the
number this section turns on.

The cause is structural rather than a tuning failure. With point-to-point
residuals the objective is `Σ aᵢⱼ‖R dᵢ + t − mⱼ‖²`, whose translation Hessian
is `2·mass·I` — isotropic whatever the landmarks are. Measured on geometry
alone, with the assignment held fixed:

| landmarks | residual | `cond(H)` | reported `σ_long/σ_lat` |
|---|---|---|---|
| three lane lines | point-to-point | 340 | 1.01 |
| three lane lines | point-to-line | **∞** | 16.6 |
| + 2 poles | point-to-point | 344 | 1.01 |
| + 2 poles | point-to-line | 10 424 | 5.34 |
| + 8 poles | point-to-point | 365 | 1.01 |
| + 8 poles | point-to-line | 3 042 | 2.91 |

Point-to-point answers **1.01 to every one of them** — adding eight poles to a
bare road does not move it, because the Hessian does not depend on where the
landmarks are. Point-to-line varies from 16.6 to 2.91 as along-track evidence
arrives and straddles the 3.030 the errors actually show. *Straddles, not
predicts*: the pole count is chosen here, so what this establishes is that one
residual can reach the observed anisotropy and the other cannot reach it at any
landmark density.

Lane lines alone make the Hessian **singular** rather than merely
ill-conditioned, with the null direction along the road. That is the honest
answer — with no along-track landmark there is no along-track measurement — and
it is why the covariance has to fuse the prior rather than invert the
measurement on its own.
