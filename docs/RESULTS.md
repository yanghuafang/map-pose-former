# Results

Every table here is reproducible from a checkpoint and a command. All of it is
open loop, which flatters any localizer: M1's numbers are on generated scenes,
M2a's on nuScenes' surveyed map with the same synthetic detector in front of it.

## M1 — the converged baseline

`configs/synth_base.yaml`, 37 000 steps on one RTX A6000. Test split: 120
scenes, 8 880 frames, from a seed range disjoint from training.

```bash
tools/train.py --config configs/synth_base.yaml
tools/eval.py runs/base/best.pt --split test
```

| | trans | long | lat | yaw | recall @ 0.25 m, 0.5° |
|---|---|---|---|---|---|
| do nothing (the prior) | 1.611 | 1.497 | 0.596 | 0.993° | 1.3% |
| all frames | **0.336** | 0.322 | 0.095 | 0.209° | **96.0%** |
| trusted (99.2% kept) | 0.297 | 0.282 | 0.095 | 0.196° | 96.4% |
| *M0, the architecture this replaced* | *0.521* | *0.498* | *0.154* | *0.590°* | *86.0%* |

**A 4.8× reduction on translation, and 35% better than M0** on the same split
under the same protocol. Recall at 25 cm and half a degree goes from 86.0% to
96.0%. Bias is 14 mm along track and 3 mm lateral, so there is no systematic
lag — the failure this problem is prone to.

### These numbers replace the ones this section published

Every M1 table was measured before the detector's clutter rule was corrected
during M2a, and the correction made the synthetic test set harder. Nothing about
the model changed; the measuring stick did.

| clutter rule | trans | long | lat |
|---|---|---|---|
| uniform over the classes present — what M1 published | 0.308 | 0.293 | 0.095 |
| drawn with multiplicity — current | **0.336** | **0.322** | 0.095 |

Only the longitudinal axis moves. Multiplicity concentrates clutter on the
classes that are detected most, which on a generated scene is lane geometry, and
along-track is the weakest axis and so the first to degrade. Lateral is
unchanged to three decimals because lane geometry constrains it far too strongly
for extra clutter to dent.

Two conclusions did not survive the re-measurement, and are corrected below
rather than quietly restated: the dashed-stripe experiment, which was worth 16%
of longitudinal error and is worth 2%, and the second refinement pass, which
was a wash and is now a small loss.

The rebuild is vindicated on accuracy. It is not vindicated on uncertainty.

### The covariance is honest almost everywhere, and catastrophic on 0.4%

The mean ANEES is 4.53, which reads as a covariance four times too small
everywhere. **That reading is wrong**, and it is worth recording because it was
the working diagnosis for a while and it has the opposite repair.

| statistic | value |
|---|---|
| NEES/dof, median | **0.78** |
| NEES/dof, p90 / p95 / p99 | 2.22 / 2.88 / 5.01 |
| NEES/dof, worst frame | **3 707** |
| frames above 10 | **0.44%** (39 of 8 880) |
| ANEES excluding those | **1.03** |
| coverage at the 95% ellipsoid | 0.936 |

**The median's target is 0.789, not 1.** A calibrated NEES is chi-square
distributed and the chi-square is right-skewed: at three degrees of freedom the
mean is 3 and the median 2.366. So a median of 0.78 is calibrated to within 1%,
not pessimistic — and `Calibration` used to report it as pessimistic, because
its band was centred on one.

What the mean is reporting is 39 frames where the model is
confidently, wildly wrong — the same tail as the 7.85 m worst-case longitudinal
error. Rescaling the covariance to bring the mean to one would wreck the
99.6% that are already honest.

`Calibration` now reports the median, the tail fraction and the tail-excluded
mean beside ANEES, and takes its verdict from the median. A single mean over a
heavy-tailed distribution is not a calibration statistic.

**What identifies the bad frames.** Nothing available does, reliably. Sorted
into quintiles by assignment mass the *middle* quintiles are worst (ANEES
5.6–6.2) and only the top quintile is clean (0.93), so mass is not even monotone
in the thing it would have to detect. Five candidate statistics, comparing good
frames against bad:

| statistic | good | bad | separation |
|---|---|---|---|
| trust score | 0.981 | 0.843 | **1.72** |
| match/volume disagreement | 0.052 | 0.173 | **1.67** |
| assignment mass | 61.7 | 48.3 | 0.87 |
| fraction of rows matched | 0.205 | 0.171 | 0.67 |
| surface entropy | 5.178 | 5.295 | 0.65 |

A separation under two standard deviations, on 39 frames out of 8 880, is not
something to gate on: one frame moves the recall of any threshold by several
points.

**What not to do about it.** Do not globally rescale, and do not raise `w_cov`
to chase the mean — both damage the 99.6% of frames that are already honest. The
re-associated surface remains the principled fix: it measures whether the pose
could be elsewhere and look as good, which is exactly what these frames are.
Expensive, bounded by a top-k restriction, and unwritten.

## M2a — the first real map

`configs/nuscenes.yaml`, 12 000 steps. Test split: 59 scenes on ground the
training split never touches, 957 frames.

| | trans | long | lat | yaw |
|---|---|---|---|---|
| do nothing (the prior) | 1.591 | 1.478 | 0.589 | 0.978° |
| all frames | **1.049** | 0.907 | 0.527 | 5.811° |
| trusted (70.6% kept) | 0.713 | 0.661 | 0.268 | 0.540° |

**A 1.5× reduction on surveyed geometry, 2.2× on trusted frames.** The backend
works on a real map, and works far less well than on generated scenes — where
the same architecture reaches 4.8×.

That gap is the result. A generated scene carries every class in comparable
numbers; nuScenes carries 13.8 lane-geometry detections a frame against 1.1
crossings and 0.6 traffic signs, and the along-track anchors the synthetic
ablation showed to be decisive are the ones a real map is poorest in.

The yaw column is a tail, not a level: 5.811° RMSE against a 0.540° trusted
figure, with one frame at 178.952°. A ped crossing is a bar with no direction
and a traffic light is a bare point, so a mismatch on either can flip heading.
Under lane geometry alone, where nothing is flippable, yaw RMSE is 0.581°.

### The headline was 0.280 m until the map stopped being the detector

`scene_world()` chunked the nuScenes map at ingest, and `NuScenesDataset` then
passed that chunked world to `build_sample(world, world, ...)` as **both** the
stored map and the source of detections — 1088 source elements against 1088 map
elements, one to one. The synthetic path has always passed
`build_sample(world, chunked, ...)`, and `chunk_for_map` says why in its own
docstring, written long before nuScenes was read:

> If both sides were chunked the same way, the two point sets would share
> element *endpoints* at fixed world positions — and a chunk endpoint is a
> perfect along-track landmark. A model given only lane geometry would then
> localize along the road beautifully, from an artefact of how the polylines
> happened to be cut, and the ablation would report that lane lines constrain
> along-track position. They do not.

Which is what happened. Under the shared chunking the model reached 0.280 m on
trusted frames, its ablation was flat to three decimals, and **lane dividers
alone scored the best longitudinal error of any class subset**. Giving nuScenes
the two-world structure the synthetic path uses — 87 source elements against
1088 map elements — costs 2.5× on translation and is the honest number.
`test_the_map_and_the_detection_source_are_chunked_apart` holds it.

### The observability claim does not reproduce here

One checkpoint, less evidence, all frames:

| evidence kept | trans | long | lat | yaw |
|---|---|---|---|---|
| everything | 1.049 | 0.907 | 0.527 | 5.811° |
| no crossings | 1.100 | 0.940 | 0.570 | 5.809° |
| no traffic lights | 1.055 | 0.918 | 0.520 | 5.808° |
| lane geometry only | 1.049 | 0.945 | 0.454 | 0.581° |
| lane dividers only `[0]` | 1.578 | **0.893** | 1.300 | 1.692° |
| road boundary only `[1]` | 1.561 | **1.404** | 0.681 | 1.326° |

**Removing the along-track anchors costs nothing measurable** — 0.907 m to
0.945 m longitudinal, against the synthetic world's 0.322 m to 1.367 m for the
same experiment. Reported as it stands.

Two rows do carry signal, and they disagree about lane geometry:

- **Road boundary alone recovers nothing along track**: 1.404 m against a
  1.478 m prior, 5%. That is the predicted physics, on a real map, from the
  class that behaves like the synthetic model of lane geometry — a long
  continuous outline running with the road.
- **Lane dividers alone are the best longitudinal arm**, 0.893 m, while making
  lateral error *worse than the prior* (1.300 m against 0.589 m).

The difference between them is length. nuScenes' lane dividers have a median
extent of 18 m — they start and stop at intersections — so their ends are
along-track features, where a drivable-area outline has none within the crop.
Whether those ends are a property of the *road* or of nuScenes' segmentation of
it is exactly what M2a cannot answer, because its detections are cut from the
map and therefore inherit the map's element boundaries. **That is M2b's
question, and it is now the reason to run M2b** rather than a detail of it.

### Five bugs, and what each was found by

Every one was found by the ablation reporting that removing evidence improved
accuracy, and the first four were real bugs that were not the cause:

| | what it was |
|---|---|
| area polygons read as outlines | a stop line's long edges ran *along* the road under a label saying across |
| clutter drawn over the whole enum | a third of it in classes nuScenes has none of, so it never matched |
| clutter drawn uniformly per class | equal clutter on unequal populations: 4.7% noise on road boundaries against 42.9% on traffic signs |
| `hash()` in the sample seed | Python salts string hashing per process, so every evaluation drew different detector noise |
| **the map was the detection source** | **one element list served as both, handing over the correspondence** |

The seed is why the first three did not fix it: each arm ran as a separate
`tools/eval.py` process, so no two arms were ever compared on the same noise —
one configuration returned 0.421, 0.449 and 0.555 on three runs. With that
fixed the arms agreed with each other and still disagreed with physics, which
is what pointed at the data, not the measurement.

**An ablation is an instrument, and an instrument that disagrees with itself
will report physics running backwards.** Four real bugs were found and fixed
while it was broken, and each fix looked like progress because the numbers
moved.

## M4 — distillation

`configs/synth_teacher.yaml` then `configs/synth_distill.yaml`, both step-matched
to the 37 000 steps M1 needed. Test split, 8 880 frames.

| | params | trans | long | lat | yaw | recall @0.25 m, 0.5° |
|---|---|---|---|---|---|---|
| teacher | 25.83 M | 0.218 | 0.198 | 0.091 | 0.220° | 96.7% |
| student, trained alone | 2.28 M | 0.336 | 0.322 | 0.095 | 0.209° | 96.0% |
| **student, distilled** | 2.28 M | **0.264** | 0.246 | 0.097 | 0.215° | 95.5% |

**Distillation is worth 21% of translation error at zero deployment cost.** The
student is the same size and the same latency; the teacher is discarded after
training. On the validation split, where both were selected, it closes 70% of
the teacher–student gap: 0.3048 to 0.2415 against a teacher at 0.2142.

| | val trans | best at step | of |
|---|---|---|---|
| teacher | 0.2142 | 29 600 | 37 000 |
| student, alone | 0.3048 | 33 300 | 37 000 |
| student, distilled | 0.2415 | 17 575 | 37 000 |

**The distilled student converged in half the budget** and did not improve over
the remaining 19 000 steps. Distillation is a denser signal than the labels --
every one of 768 × 576 correspondences carries an opinion, where the pose loss
carries three numbers -- so it is not surprising that it gets there sooner. It
does mean the 7.9 h run could have been 4 h.

**Recall fell while RMSE improved**, 96.0% to 95.5% at 25 cm. Matching a
teacher's soft assignment pulls in the tail instead of sharpening the median:
the student trades a few near-misses for far fewer large errors. Both numbers
are reported because quoting only the RMSE would hide the trade.

### What the two KD terms did

| step | `kd_match` | `kd_volume` | ratio |
|---|---|---|---|
| 50 | 5.02 | 139.74 | 28× |
| 6 300 | 0.50 | 2.60 | 5.2× |
| 36 400 | 0.035 | 0.076 | 2.2× |

The weights are `w_match = 1.0` against `w_volume = 0.5`, so the surface term
was contributing an order of magnitude more gradient than intended at the start
and roughly the intended share by the end. An earlier smoke run against an
*untrained* teacher gave a ratio of 1:10 000 in the other direction, because two
models that both withhold most of their assignment mass agree about doing so.
Rebalancing the weights on either measurement would have been wrong; the ratio
is a function of how converged the teacher is, not of the weights.

## M4 — the compression Pareto

`scripts/m4.sh`, one `tools/pareto.py` process so every row is the same split,
the same protocol, the same machine. Batch 1, 200 warmup iterations, p50 and p99
over 1000. Test split, 8 880 frames.

| | params | trans | trusted | recall @25cm | p50 | p99 |
|---|---|---|---|---|---|---|
| teacher | 25.83 M | 0.218 | 0.185 | 96.7% | 35.26 ms | 35.65 ms |
| student, alone | 2.28 M | 0.336 | 0.297 | 96.0% | **20.32 ms** | 20.64 ms |
| student, distilled | 2.28 M | 0.264 | 0.224 | 95.5% | 20.95 ms | 21.26 ms |
| pruned to 75% | 2.02 M | **0.254** | 0.248 | 95.7% | 21.01 ms | 25.99 ms |
| pruned to 50% | 1.76 M | 0.267 | 0.248 | 95.7% | 20.42 ms | 20.73 ms |
| pruned to 25% | 1.49 M | 0.261 | 0.240 | 95.7% | 21.11 ms | 21.44 ms |
| pruned to 25%, INT8 | 1.49 M | 0.261 | 0.242 | 95.6% | 34.40 ms | 34.81 ms |

### Removing 35% of the parameters bought nothing

That is the result. From 2.28 M to 1.49 M, latency goes 20.95 ms to 21.11 ms —
within noise, and in the wrong direction. The accuracy is free too: 0.264 to
0.261. Both facts have one cause, and the project predicted it before any of
this ran: **the model is activation-bound, not parameter-bound.** Attention runs
over ~1400 tokens and a 768 × 576 assignment matrix; the feed-forward weights
that pruning removes were never the bottleneck.

| | what it changed | what it cost |
|---|---|---|
| distillation | 0.336 → 0.264, −21% | 8 h to train a teacher, discarded after |
| pruning to 25% | 0.264 → 0.261, −1% | 35% of parameters, 0% of latency |
| INT8 (simulated) | 0.261 → 0.261, 0% | +63% latency, which is the simulation |

The teacher is the same shape: 11× the parameters of the student for 1.7× the
latency. If parameters drove cost, it would be 11×.

### What that means for the milestone

**Pruning is the wrong lever for this architecture.** It is not that pruning failed — it removed a third of the weights for no
accuracy — it is that weights were not what made this model slow. The lever that
would move latency is the one `docs/OPEN_ITEMS.md` now ranks first: replacing
`nn.MultiheadAttention` with `scaled_dot_product_attention`, which would stop
materialising the attention matrix.

**The INT8 row is a simulation and its latency must not be read as INT8's.**
Quantize-dequantize adds rounding and removes no arithmetic, so 34.40 ms is the
cost of *pretending*. What the row does say is that accuracy is untouched at 8
bits — 0.261 either way — which is the number worth carrying into M5. It also
reaches only 49% of the weights, because the rest are inside
`nn.MultiheadAttention`; the same module, a third time.

**Distillation is the only stage that paid.** It is also the only one that
changes what the model *knows* rather than how it is stored.

## M1 — the observability ablation

One checkpoint, evaluated under less evidence. Only what the model may see
differs, so nothing here is a training artefact.

| evidence kept | trans | long | lat | yaw |
|---|---|---|---|---|
| everything | **0.336** | **0.322** | 0.095 | 0.209° |
| no perpendicular `[0,1,4,5]` | 0.402 | 0.389 | 0.100 | 0.222° |
| **nuScenes' classes as then assumed** `[0,1,2,3]` | 1.061 | 1.057 | 0.100 | 0.273° |
| lane geometry only `[0,1]` | 1.371 | 1.367 | 0.104 | 0.315° |
| *(the prior, for scale)* | *1.611* | *1.497* | *0.596* | *0.993°* |
| no lane geometry `[2,3,4,5]` | 11.415 | 9.997 | 5.510 | 27.780° |

**Lane geometry still cannot see along track.** Restricted to dividers and
boundaries, longitudinal RMSE is 1.367 m against a prior of 1.497 — essentially
nothing recovered — while lateral falls from 0.596 to 0.104, an 83% recovery
from the same evidence in the same frames. The asymmetry is the whole table, and
it is the one conclusion that the clutter correction left untouched.

**This row stands in for nuScenes with the wrong classes, and the number is
retained rather than corrected.** It assumes stop lines and no point landmarks;
reading the map showed the opposite — the stop-line annotation is an area with
no bar direction, and there are 307 traffic lights. The real proxy is
`[0,1,2,5]`. What the row still shows is the *mechanism*: removing point
landmarks costs longitudinal error and leaves lateral alone.

Taken at face value it says nuScenes' class set costs 3.3× longitudinal, not
the 2.3× M0 measured.
The gap widened because the full-evidence number improved and the restricted one
did not, which is what it means for poles and signs to be carrying the
along-track information. **M2 should expect this**, and it is a sharper
prediction than the one it replaces.

**The `[2,3,4,5]` catastrophe is smaller and still a catastrophe**: 11.4 m
against M0's 14.2 m. The robust solve helped and did not fix it.

## M1 — do the mechanisms pay?

Same checkpoint, one mechanism disabled at evaluation.

| | trans | long | lat | yaw |
|---|---|---|---|---|
| baseline | 0.336 | 0.322 | 0.095 | 0.209° |
| no dashed stripes | 0.339 | 0.328 | **0.085** | **0.169°** |
| one refinement pass | **0.316** | **0.300** | 0.097 | 0.224° |
| plain solve, no IRLS or gate | 0.358 | 0.301 | 0.194 | 0.495° |

**The dashed-stripe result did not survive re-measurement.** It was 16% of
longitudinal error, 0.293 against 0.340; it is now 1.9%, 0.322 against 0.328.
The arm barely moved — the baseline did, because the corrected clutter rule made
along-track harder for everything, and the gap closed from the other side. What
remains is the *sign*: removing the stripes still costs longitudinal and still
improves lateral and heading, which is the signature of a feature that
constrains one axis. What is gone is the claim that it is worth much.

That the effect shrinks when clutter concentrates on lane geometry is at least
consistent: a stripe end is a lane-class feature, and it competes with lane-class
clutter for the matcher's attention.

**The second refinement pass is now a small loss, not a wash.** One pass scores
0.316 against two at 0.336 — 6% better for half the step time. It was 0.309
against 0.308 before. The caveat stands: this model was *trained* with two
passes, so it measures inference, not whether training with one would have been
as good. But the direction has changed from "buys nothing" to "costs something",
and the training-time ablation is now worth the hour it takes.

**The robust solve earns its place**, and this is unchanged. Turning off
reweighting and the abstention gate doubles lateral error (0.095 → 0.194) and
more than doubles heading (0.209° → 0.495°). Longitudinal is marginally *better*
without it, which fits: robustness trades a little accuracy on easy frames for
much more on hard ones.

## M1 — the step-matched A/B

Two runs stopped at 2 000 steps rather than at equal wall clock: temporal fusion
changes the input width, so equal time would hand the shorter model more
training and the comparison would measure throughput instead of the mechanism.

| arm | val trans RMSE | ms/step |
|---|---|---|
| the model | **1.107** | 307 |
| no history | 1.239 | 305 |
| *(the prior, for scale)* | *1.67* | — |

**History is worth 10.7% of translation at 2 000 steps**, and the step time does
not move — 305 ms against 307. That is not evidence it is free: `tools/bench.py`
reports the loader as the limit on this box, so a model-side cost of two thirds
of the detection tokens is hidden underneath it.

What 2 000 steps cannot say is what history is worth at convergence; M0 needed
37 000. The converged tables above are that measurement.

```bash
tools/train.py --config configs/synth_base.yaml train.max_steps=2000
tools/train.py --config configs/ablate_no_history.yaml train.max_steps=2000
```
## M1 — the head rematch

The robust solve against the `tanh`-bounded regression baseline that beat its
predecessor seven to one. Both trained to convergence, test split.

| | Procrustes | Regression |
|---|---|---|
| baseline trans | 0.336 | **0.248** |
| baseline yaw | **0.209°** | 0.430° |
| lane geometry only, trans | 1.371 | **0.971** |
| no lane geometry, trans | 11.415 | **2.381** |
| no lane geometry, yaw | 27.780° | **1.616°** |

**The closed-form head still loses on translation, and still loses badly off
distribution.** Robustness narrowed the out-of-distribution gap from 7× to 4.6×
and nothing more, and the in-distribution gap widened to 35% under the corrected
clutter rule — the regression head is less disturbed by clutter, which is what a
bounded estimator over a pooled feature would be. What the closed form does win is **heading** — 0.211° against
0.430°, a factor of two, and that holds in distribution where most frames live.

So the split is cleaner than last time rather than resolved: solving the
geometry gives better rotation and worse translation, and a bounded regressor
degrades far more gracefully when the evidence is wrong. The honest reading is
that neither is dominant, and the parameter-free head is justified by rotation
accuracy, interpretability and quantization behaviour rather than by RMSE.

## Identities, each with a test

Properties rather than accuracy — what the rebuild had to establish, and what
would break silently if it regressed.

| Property | Result | Test |
|---|---|---|
| Closed-form grid cost equals the brute-force sum over every cell | 3e-7 relative, argmin agrees | `test_grid_cost_matches_the_brute_force_sum` |
| The pose is the minimum of the surface the volume reports | within half a cell on all three axes | `test_the_volume_minimum_is_the_pose_the_head_solves` |
| The robust solve survives a fifth of its correspondences being wrong | < 0.10 m, < 0.5° | `test_the_robust_solve_survives_wrong_correspondences` |
| Warped history aligns to the map as well as the current frame | 0.904 against 0.904 | `test_history_lands_on_the_map_through_egomotion` |
| Gradient accumulation reproduces one large batch | 1e-4 relative | `test_gradient_accumulation_matches_one_large_batch` |
| Moving the world 5 km changes no input tensor | exact | `test_sample_is_unchanged_by_moving_the_whole_world` |
| The graph exports and ONNX matches eager | 1.2e-5 | `test_onnx_export_matches_eager` |

## The cost surface does not show ambiguity

The computed surface delivered two of its three claims. The third — that it
would show aliasing as a ridge along the road — is **false**, and
`tools/viz_volume.py` is what showed it.

Frame 30, oracle correspondences, cost rise per metre:

| surface | forward | lateral |
|---|---|---|
| fixed correspondences — **what the model reports** | 4.61 | 3.31 |
| re-associated, every class | 0.11 | 0.79 |
| re-associated, lane geometry only | 0.08 | 0.64 |

Holding correspondences fixed, sliding the hypothesis moves every point off its
own target, so the cost rises in every direction — faster along track than
across, backwards from the physics, and unchanged when the along-track landmarks
are removed. Re-association gives a surface 7× steeper laterally, and 31%
flatter along track again without those landmarks.

So the reported covariance is the **conditional fit covariance**: how well the
pose is determined *given* these correspondences, not whether the pose could be
elsewhere and look as good. That is a real quantity and a useful one, but it is
optimistic exactly where the map aliases — which is the same direction the
measured ANEES of 4.53 points, and probably part of the same story.

```bash
tools/viz_volume.py --mode fit     --out fit.svg
tools/viz_volume.py --mode reassoc --out all.svg
```

## Cost of a run

`tools/bench.py`, one A6000, at each config's own batch size.

| | params | batch | allocated | reserved | forward | forward + backward |
|---|---|---|---|---|---|---|
| student (`synth_base.yaml`) | 2.3 M | 64 | 6.93 GiB | 8.01 GiB | 689 f/s | **190 f/s** |
| teacher (`synth_teacher.yaml`) | 25.8 M | 40 | 17.80 GiB | 19.10 GiB | 188 f/s | **59 f/s** |

Both columns, because *reserved* is what has to fit on the card and allocated is
what the tensors need. The gap is the caching allocator's, and it is 15% here.

Memory is linear in batch — 0.11 GiB per sample for the student, 0.44 for the
teacher — because **this model is activation-bound, not parameter-bound**. The
student's weights are 9 MB against 6.9 GiB of activations: attention runs over
~1400 tokens and a 768 × 576 assignment matrix is saved for backward at every
layer of every refinement pass.

Three things follow, and the third is the one that matters for M4. The teacher
is 11× the parameters and only 3.2× the step time, which makes distillation
affordable. Its batch is 40 rather than 48 so that 19.1 GiB reserved fits a
24 GiB card, so neither model needs this one. And **INT8 shrinks weights, which
are not what is large here** —
which is why the compression milestone measures the teacher as well as the
student, rather than reporting a flat table and explaining it afterwards.

At ~6% of the card's dense bf16 throughput the model uses the GPU poorly, and
that is a property of its shape rather than of its batch size: `head_dim` is 32,
and every attention matmul contracts over it. [OPEN_ITEMS.md](OPEN_ITEMS.md)
lists what would change it.

---

*M0's own tables — the baseline, its ablations, the first head comparison and
the prior sweep — described an architecture this one replaced. They are in the
git history at the first commit rather than here, because a reader comparing
two sets of numbers for a model that no longer exists is being given work rather
than knowledge.*
