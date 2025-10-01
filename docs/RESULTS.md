# Results

Every table here is reproducible from a checkpoint and a command. These are
open-loop numbers on generated data, and both flatter a localizer.

## M1 — the converged baseline

`configs/synth_base.yaml`, 37 000 steps on one RTX A6000. Test split: 120
scenes, 8 880 frames, from a seed range disjoint from training.

```bash
tools/train.py --config configs/synth_base.yaml
tools/eval.py runs/m1_base/best.pt --split test
```

| | trans | long | lat | yaw | recall @ 0.25 m, 0.5° |
|---|---|---|---|---|---|
| do nothing (the prior) | 1.611 | 1.497 | 0.596 | 0.993° | 1.3% |
| all frames | **0.308** | 0.293 | 0.095 | 0.211° | **96.0%** |
| trusted (99.1% kept) | 0.285 | 0.269 | 0.095 | 0.195° | 96.5% |
| *M0, the architecture this replaced* | *0.521* | *0.498* | *0.154* | *0.590°* | *86.0%* |

**A 5.2× reduction on translation, and 41% better than M0** on the same split
under the same protocol. Recall at 25 cm and half a degree goes from 86.0% to
96.0%. Bias is 14 mm along track and 3 mm lateral, so there is no systematic
lag — the failure this problem is prone to.

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

So the typical frame is slightly *pessimistic* and 99.6% of them are calibrated
to within 3%. What the mean is reporting is 39 frames where the model is
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

## M1 — the observability ablation

One checkpoint, evaluated under less evidence. Only what the model may see
differs, so nothing here is a training artefact.

| evidence kept | trans | long | lat | yaw |
|---|---|---|---|---|
| everything | **0.308** | **0.293** | 0.095 | 0.211° |
| no perpendicular `[0,1,4,5]` | 0.397 | 0.384 | 0.100 | 0.225° |
| **nuScenes' classes** `[0,1,2,3]` | 1.074 | 1.069 | 0.099 | 0.273° |
| lane geometry only `[0,1]` | 1.408 | 1.405 | 0.104 | 0.319° |
| *(the prior, for scale)* | *1.611* | *1.497* | *0.596* | *0.993°* |
| no lane geometry `[2,3,4,5]` | 11.569 | 10.209 | 5.444 | 27.460° |

**Lane geometry still cannot see along track.** Restricted to dividers and
boundaries, longitudinal RMSE is 1.405 m against a prior of 1.497 — essentially
nothing recovered — while lateral falls from 0.596 to 0.104, an 83% recovery
from the same evidence in the same frames. The asymmetry is the whole table.

**nuScenes' class set now costs 3.6× longitudinal**, not the 2.3× M0 measured.
The gap widened because the full-evidence number improved and the restricted one
did not, which is what it means for poles and signs to be carrying the
along-track information. **M2 should expect this**, and it is a sharper
prediction than the one it replaces.

**The `[2,3,4,5]` catastrophe is smaller and still a catastrophe**: 11.6 m
against M0's 14.2 m. The robust solve helped and did not fix it.

## M1 — do the mechanisms pay?

Same checkpoint, one mechanism disabled at evaluation.

| | trans | long | lat | yaw |
|---|---|---|---|---|
| baseline | **0.308** | 0.293 | 0.095 | 0.211° |
| no dashed stripes | 0.351 | 0.340 | 0.085 | **0.172°** |
| one refinement pass | 0.309 | 0.293 | 0.097 | 0.225° |
| plain solve, no IRLS or gate | 0.339 | **0.278** | 0.194 | 0.500° |

**The dashes carry along-track information, as predicted.** Removing the stripe
geometry costs 16% of longitudinal error (0.293 → 0.340) while *improving*
lateral and heading — exactly the signature of a feature that constrains one
axis. This is the experiment a public dataset cannot run cheaply, and it worked.

**The robust solve earns its place.** Turning off reweighting and the abstention
gate doubles lateral error (0.095 → 0.194) and more than doubles heading
(0.211° → 0.500°). Longitudinal is marginally *better* without it, which fits:
robustness trades a little accuracy on easy frames for much more on hard ones.

**The second refinement pass buys nothing** — 0.309 against 0.308. It costs 2×
the step time. The honest caveat is that this model was *trained* with two
passes, so this measures whether the second pass matters at inference, not
whether training with one would have been as good. But it is a strong hint that
`refine_iters=2` is wasted compute, and the training-time ablation is cheap.

## M1 — the head rematch

The robust solve against the `tanh`-bounded regression baseline that beat its
predecessor seven to one. Both trained to convergence, test split.

| | Procrustes | Regression |
|---|---|---|
| baseline trans | 0.308 | **0.261** |
| baseline yaw | **0.211°** | 0.430° |
| lane geometry only, trans | 1.408 | **1.001** |
| no lane geometry, trans | 11.569 | **2.381** |
| no lane geometry, yaw | 27.460° | **1.611°** |

**The closed-form head still loses on translation, and still loses badly off
distribution.** Robustness narrowed the out-of-distribution gap from 7× to 4.9×
and nothing more. What the closed form does win is **heading** — 0.211° against
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

| | params | batch | VRAM | forward | forward + backward |
|---|---|---|---|---|---|
| student (`synth_base.yaml`) | 2.3 M | 64 | 6.9 GiB | 793 f/s | **211 f/s** |
| teacher (`synth_teacher.yaml`) | 25.8 M | 48 | 21.2 GiB | 216 f/s | **61 f/s** |

Memory is linear in batch — 0.11 GiB per sample for the student, 0.44 for the
teacher — because **this model is activation-bound, not parameter-bound**. The
student's weights are 9 MB against 6.9 GiB of activations: attention runs over
~1400 tokens and a 768 × 576 assignment matrix is saved for backward at every
layer of every refinement pass.

Three things follow, and the third is the one that matters for M4. The teacher
is 11× the parameters and only 3.5× the step time, which makes distillation
affordable. The teacher at batch 48 fits a 24 GiB card, so neither model needs
this one. And **INT8 shrinks weights, which are not what is large here** —
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
