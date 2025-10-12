# Roadmap

Each milestone is a commit boundary and a documentation section.

## What actually blocks what

The numbering is chronological, not causal, and the difference matters because
the second half is the stated purpose of the project:

```
M1 measurement ──┬── M2 nuScenes ── M3 closed loop
                 └── M4 compression ── M5 deployment
```

**M4 and M5 do not depend on real data.** Distillation, pruning, quantization
and the TensorRT export can all be exercised end to end on the synthetic stage
as soon as M1 has a converged checkpoint — about ten hours of GPU. Doing that
first de-risks the export while the model is still cheap to retrain, and turns
"repeat it on nuScenes" into re-running a pipeline that already works. Running
them last, behind two milestones that could each take weeks, risks a project
that never reaches the thing it was for.

**Compression here is pedagogical, and that should be said plainly.** No latency
budget forces it: at batch 1 the student is already far inside anything a
vehicle at 10 Hz would need. The Pareto table is worth producing because the
techniques are worth learning, not because the model is too slow.

## Done — M0: the stage-0 system

Procedural data, the model, five losses, the training loop, the evaluator, the
ablation harness, the tests. Runs end to end on CPU in minutes
(`scripts/run_smoke.sh`) and on the A6000 in a couple of hours.

Measured: translation RMSE from 1.61 m to 0.52 m open loop, 0.39 m on trusted
frames. [RESULTS.md](RESULTS.md).

## Done — M1: the architecture those measurements asked for

M0's ablations pointed at the design rather than the hyperparameters:

- **The cost surface is computed, not regressed.** It was an MLP from a pooled
  vector to 4199 logits — a third of the parameters predicting a surface that
  follows in closed form from statistics the pose head already had.
- **The pose head is robust.** Abstentions gated relative to the frame's
  strongest match, then Geman-McClure reweighting twice. This repairs the head
  losing seven to one out of distribution.
- **`mass` is a gate.** The `[2,3,4,5]` ablation produced 14 m of error with
  9.97 m of it trusted; the evaluator now refuses on evidence separately.
- **Matching runs twice**, the second pass on detections re-anchored by the
  first estimate. Shared weights, no parameters.
- **The past enters through egomotion** — the temporal fusion that was M2, and
  it needed no new geometry, only the conjugation the classical repo already
  uses for cost aggregation.
- **Three new inputs**: detector confidence, a reported per-detection
  uncertainty, and `mark_type`.
- **The evaluator asks whether the covariance is honest**, with ANEES and 95%
  coverage. An RMSE cannot say that, and a filter weights by what it cannot check.

A step-matched A/B ranked them before any of them was run to convergence: the
rebuild is 15.5% better than the pre-rebuild model at equal steps, and history
accounts for 12.9 of those points and earns its 2.4×.

**Converged, it holds.** 0.336 m test translation RMSE against a 1.611 m prior,
35% better than the 0.521 m of the architecture it replaced, with recall at
25 cm rising from 86.0% to 96.0%. The observability ablation reproduces and
sharpens. The dashed-stripe experiment kept its sign and lost its magnitude:
16% of longitudinal error when first measured, 2% under this milestone's
corrected clutter rule. The robust solve earns its place on lateral and
heading. Full tables in [RESULTS.md](RESULTS.md).

Two results argue with the design. **0.44% of frames are confidently, wildly
wrong** — the mean ANEES of 4.53 is that tail, not a covariance four times too
small everywhere, and the repair is to detect those frames rather than to
rescale. It blocks M3 either way. And **the head rematch did not go the closed
form's way**: robustness narrowed the out-of-distribution gap from 7× to 4.9×
and no further, though the closed form wins heading by a factor of two.

### Done: the export smoke test, pulled forward from M5

Every milestone below assumes the graph exports, and that assumption was wrong.
`torch.export` captured the model, but ONNX conversion failed on
`aten.repeat_interleave.self_int` — used to build a *constant* index vector for
the age embedding, and replaced by `expand`/`reshape` at no cost. The default
config now exports and matches eager to 1.2e-5; `tests/test_export.py` holds it.

Found in an hour, at M1, where the fix was one line. Found at M5 it would have
been one line on top of everything built above it.

### Open: a surface that shows ambiguity

The computed surface delivered two of its three claims. The third is measured
false: with correspondences fixed the surface is the paraboloid of the fit, and
steeper along track than across.

Re-association produces a ridge and breaks the factorisation the closed form
depends on. The full version costs `hypotheses × detections × map points` — 1.9
billion per sample — which is why `viz_volume.py` can do one frame and a
training step cannot.

**The affordable version** restricts re-association to the map points the
matcher already finds plausible: top `k` entries of each assignment row,
soft-min over those. At `k = 8` over the top 128 detection points that is ~4 M
distance evaluations per sample, and it degrades to the current surface at
`k = 1`. Whether the resulting covariance is better calibrated is what
`Calibration` is now there to answer.

The other half of the geometric encoding is also open: the bias is on
cross-attention, where the bounded prior makes a cross-set distance meaningful.
GeoTransformer puts its structure embedding in *self*-attention, with distances
and triplet angles. Untried here.

## M2 — nuScenes

**M2a is done and its answer is a qualified yes.** It reduces translation error
from 1.591 m to 1.049 m open loop, 0.713 m on trusted frames — against 5.2× for
the same architecture on generated scenes. What it cannot do is the
observability ablation it was also meant to carry: its detections are cut from
the map, so they inherit the map's element boundaries, and separating the road's
geometry from nuScenes' segmentation of it needs a detector that segments
independently. That is M2b.

nuScenes + map expansion v1.3 on the **geographically disjoint split** (the
StreamMapNet split; the official train/val scenes overlap spatially and a
localizer evaluated on it is partly reciting).

**Two phases, because bundling them makes the result unreadable.** The ingest
and the sim-to-real step are independent, they fail for different reasons, and
run together their effects cannot be separated.

**M2a — real map, synthetic detections.** Reuse the existing error model on the
real map. This proves `prepare_nuscenes.py`, the geographic split and the CAN
egomotion, and it tests M1's prediction that nuScenes' class set costs 2.3×
worse longitudinal error. No mapper and no imagery — 1.6 GB of map and poses —
and on its own it answers "does this backend work on a real map?".

**M2b — real detections.** Then, and only then, swap in a pretrained mapper's
output. Because M2a exists, the sim-to-real gap becomes a *measured delta*
against it rather than a number confounded with everything else that changed.

**The detections must be real, and this is a change of plan.** Cutting them from
the map and corrupting them upgrades the *map* to real and leaves perception an
assumption — and leaves both point sets as the same polylines with noise: same
chunking, same vertex counts, same topology. A real detector gets none of that
right. The correlated-bias model captures the statistical part and none of the
structural part.

So: **run a pretrained online mapper, do not train one.** MapTRv2 or
StreamMapNet inference over the nuScenes cameras, once, offline, cached. The
scope decision is unchanged — perception stays an input — but the inputs stop
being a corrupted copy of the answer.

Four things change:

- an offline `tools/prepare_nuscenes.py` writing the tensors this project
  already consumes, so the training loop never imports the devkit
- a pretrained mapper's output as `det_*`, its scores as `det_conf`
- egomotion from the CAN bus, replacing differenced ground truth
- **no poles and no traffic signs**: the map expansion carries neither, so
  along-track observability is structurally weaker. M1's ablation predicts 2.3×
  worse longitudinal error, and this milestone checks that prediction.

**Map topology becomes an input here**, because this is the first map with any.
A real vector map is a graph: lanes have predecessors, successors and neighbours,
and two boundaries of the *same* lane are more mutually informative than two
that happen to be parallel. Cheapest first — a per-element embedding of its
topological role — then an adjacency bias on map self-attention, the same
mechanism `GeometricBias` provides for cross-attention. The synthetic world has
no topology worth the name, so building this earlier would mean inventing the
thing the experiment is meant to test.

Argoverse 2 afterwards as a generalization test, never trained on. Its lane
boundaries carry `mark_type`, which is the real-data version of the dash
experiment.

`scripts/download_nuscenes.sh` and `scripts/download_argoverse2.sh` fetch both;
run them by hand, since nuScenes' terms are accepted by a person. The camera
archives are needed as well as the map, because the mapper reads images.

## M3 — Closed loop

Everything so far is open loop: one frame, one correction, error against the
label. A deployment feeds the correction into a filter, and the next frame's
prior is the previous frame's output.

This is where `LocalizationKF` returns: the model's `(delta, cov, trust)` is
what it consumes, and `mass` is the second gate it should read. The honest
comparison is a sequence run of both backends through the same filter and the
same `eval_sequence` metrics.

The temporal path already fuses evidence open loop, so this measures something
different: whether the correction survives being fed back, where a small
systematic bias compounds instead of averaging out.

**The prior's covariance becomes an input here**, and only here. Open loop the
prior is drawn from a fixed distribution the model can simply learn. Closed loop
the filter produces a per-frame `3 × 3` — tight after a good update, wide after
a run of rejections — and a model that knows how far to look can match
differently in the two cases. It also closes an inconsistency: the surface's
extent is pinned to a fixed truncation bound, so a wider prior has no correct
cell.

## M4 — Compression

**Neither model is trained yet.** `synth_teacher.yaml` and `synth_base.yaml` are
configurations, and nothing in this milestone is implemented.

| | student | teacher |
|---|---|---|
| config | `synth_base.yaml` | `synth_teacher.yaml` |
| size | 2.3 M params, 128-dim, 4 layers | 26 M params, 256-dim, 8 layers |
| role | **the deployment artifact** | what it is distilled from |
| distillation | the pupil | the source |
| pruning, quantization | **applied here** | measured here too, as a control |
| exported to TensorRT | yes | no |

In this order, because each stage changes what the next works with:

1. **Distillation, teacher → student**, on the assignment matrix and the cost
   surface. Both are distributions, so KL is natural, and the teacher's soft
   assignment carries far more than its pose.
2. **Structured pruning of the student**: attention heads and FFN channels, not
   unstructured masks — `torch.nn.utils.prune` zeros weights without removing
   them, which gives no GPU speedup. Prune, fine-tune, re-measure.
3. **Quantization of the student**: PTQ for the calibration curve, then QAT with
   `nvidia-modelopt`. The Procrustes head is arithmetic rather than weights, and
   the cost surface is a closed form over the same statistics — so the part of
   this model most worth trusting is the part quantization cannot touch.

Report a Pareto table: accuracy against latency, one row per configuration.

### Cost in hours

From the measured throughputs — student 211 frames/s, teacher 61, forward and
backward on one A6000 — against a nuScenes epoch of ~28 000 keyframes. These are
compute times; the engineering to build M2 dominates all of them.

| stage | estimate | basis |
|---|---|---|
| teacher training, 40 epochs | **~5.1 h** | 1.12 M frames ÷ 61 |
| student training, 40 epochs | ~1.5 h | 1.12 M ÷ 211 |
| distillation, teacher outputs cached | **~1.5 h** | one teacher pass (2 min), then the student's own rate |
| distillation, teacher run live | ~2.9 h | `1/(1/211 + 1/216)` = 107 frames/s |
| pruning: fine-tune, three cycles | ~1.1 h | each fine-tune ≈ 25% of a run |
| quantization: PTQ calibration | ~1 min | 512 batches forward at 793 frames/s |
| quantization: QAT fine-tune | ~40 min | ≈30% of a run at ~1.4× step cost |
| TensorRT: export, three engines, benchmark | **<1 h** | INT8 build with calibration is the long pole |

**About ten hours end to end**, roughly half what it was before the geometric
bias was defaulted off. Caching the teacher's outputs saves 1.4 h at 48 GB for
the assignment matrices; it works only on real data, since the synthetic stage
redraws its noise every epoch.

VRAM is the other constraint worth naming, and it is linear in batch size:
0.11 GiB per sample for the student, 0.44 for the teacher. At the batch 64 both
configs now use, that is 7.0 and 28.2 GiB — the teacher would not fit on a 16 GB
card, and its batch would have to drop to about 35.

### Why the teacher is measured too

At 2.3 M parameters over ~1400 tokens the student is bound by kernel launch
overhead and memory traffic, not arithmetic, so INT8 may buy nothing measurable
on an A6000 and 2:4 sparsity on layers this narrow is close to noise.

That is a real result about this model on this card and will be reported as one,
but a compression chapter concluding "nothing moved" teaches the technique
badly. The same three treatments therefore also run on the teacher, where there
is enough arithmetic to bite. The student remains the deliverable; the teacher
rows separate "the technique does not work" from "this model is not where it
pays".

The protocol is fixed in advance, because a latency number without one is
unfalsifiable: batch 1, CUDA graphs, 200 warmup iterations then p50 and p99 over
1000, the same for every row including the fp32 baseline.

## M5 — Deployment

ONNX export via `torch.export` — the input shapes are already static for this,
and the dynamo path produces a far cleaner graph than the legacy tracer for a
model this full of masks and einsums. Then TensorRT: FP16, INT8, and 2:4
structured sparsity, all three supported on Ampere.

The finale: a small C++ TensorRT backend plugging into
camera-map-localization's engine as an alternative to `PoseSampler`, so the same
`run_sequence` and `eval_sequence` evaluate both backends under one metric and
one filter.
