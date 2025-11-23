# Roadmap

Each milestone is a commit boundary and a documentation section.

## What actually blocks what

The numbering is chronological, not causal, and the difference matters because
the second half is the stated purpose of the project:

```
M1 measurement ──┬── M2 nuScenes ── M3 closed loop
                 └── M4 make it fast (runtime first, then compression)
```

**M4 does not depend on real data.** Distillation, pruning, quantization and the
TensorRT export can all be exercised on the synthetic stage as soon as M1 has a
converged checkpoint. Doing that first de-risks the export while the model is
still cheap to retrain, and turns "repeat it on nuScenes" into re-running a
pipeline that already works. Running it last, behind two milestones that could
each take weeks, risks a project that never reaches the thing it was for.

**Compression and deployment were two milestones and are now one, because
separating them produced a wrong answer.** The original plan measured
compression in PyTorch (M4) and built the TensorRT runtime afterwards (M5). Run
that way, pruning bought 0% of latency and the conclusion was "the model is
activation-bound". Then the runtime landed and the same two checkpoints were 4×
faster, with pruning worth 6.6% — the earlier measurement had been dominated by
framework dispatch and was measuring the interpreter.

The lesson generalises past this project: **build the runtime you will deploy on
before you measure what to compress.** A compression number is a statement about
a runtime, and taking it on a different one measures that other runtime instead.
The milestones are merged so the ordering cannot be got wrong again.

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
  losing to its own baseline out of distribution. [RESULTS.md](RESULTS.md) has
  the rematch.
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
sharpens: lane geometry recovers nothing along track and 83% laterally. The
robust solve earns its place on lateral and heading. The dashed-stripe
experiment kept its sign and lost its magnitude — 16% of longitudinal error
when first measured, 2% once the clutter rule was corrected. Full tables in
[RESULTS.md](RESULTS.md).

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

**M2a is done and its answer is a qualified yes; M2b has a contract and no
detector; M2c is unstarted.**

M2a asked whether this backend works on a real map. It does, and far less well
than on generated scenes — [RESULTS.md](RESULTS.md) has both. What M2a cannot
do is the observability ablation it was also meant to carry: its detections are
cut from the map, so they inherit the map's element boundaries, and the one
class whose ends fall inside the crop is the one that recovers along-track
position. Separating the road's geometry from nuScenes' segmentation of it
needs a detector that segments independently, which is M2b.

nuScenes + map expansion v1.3 on the **geographically disjoint split** (the
StreamMapNet split; the official train/val scenes overlap spatially and a
localizer evaluated on it is partly reciting).

**Three phases, because bundling changes makes their effects unreadable.** The
ingest, the sim-to-real step and the model change fail for different reasons,
and run together nothing can be attributed to any of them.

The ingest and the geographic split are *one* phase, not two: a split cannot be
tested without an ingest to produce poses, and an ingest whose split leaks is
worthless. Everything else is separable and therefore separated.

**M2a — real map, synthetic detections.** Reuse the existing error model on the
real map. This proves `prepare_nuscenes.py`, the geographic split and the CAN
egomotion, and it tests M1's prediction that dropping point landmarks costs
longitudinal error — though not the figure M1 attached to it, which stood in
for nuScenes with a class set the map turned out not to have. No mapper and no
imagery — 1.6 GB of map and poses — and on its own it answers "does this backend
work on a real map?".

**M2b — real detections.** Then, and only then, swap in a pretrained mapper's
output. Because M2a exists, the sim-to-real gap becomes a *measured delta*
against it rather than a number confounded with everything else that changed.

The contract is written and tested; the detector is not run.
`mapposeformer/data/detections.py` defines a file format -- one `.npz` per
scene, elements in that keyframe's ego frame -- and `data.detections_dir` makes
the dataset read it instead of cutting from the map. A stub exercises the whole
path in `tests/test_detections.py`.

**The detector runs in its own environment, and this is a boundary rather than
a problem.** MapTR and StreamMapNet are written against `mmdet3d 1.0.0rcX` and
`mmcv 1.x`, which cap at Python 3.10 and torch 2.0; this project runs 3.12 and
torch 2.11. Nothing in `mapposeformer` imports the detector, so neither
constrains the other's dependencies -- the same rule already drawn around the
nuScenes devkit.

Two checks stand between a detector's output and a number worth reporting.
`validate` rejects a malformed file. `plausibility` catches the failure it
cannot see: a mirrored axis or a camera-convention frame produces a file that
validates perfectly and trains to a confidently wrong answer, so applying the
ground-truth correction must land the detections on the map. The synthetic path
scores about 0.9; near zero means a convention error rather than a weak
detector.

What remains is the detector itself: a second conda environment, a checkpoint,
and one inference pass over the 53 GB of keyframe images already on disk.

**M2c — map topology as an input.** A real vector map is a graph, and this is
the first map here with one: lanes have predecessors, successors and neighbours,
and two boundaries of the *same* lane are more mutually informative than two
that merely run parallel. Cheapest first — a per-element embedding of its
topological role — then an additive bias on map self-attention scores, keyed to
whether two elements are adjacent in the graph.

Separate from M2b for the reason M2b is separate from M2a: it is a **model**
change measured on unchanged data, so it ablates cleanly against M2a's
checkpoint. Folded into M2b it would confound "real detections" with "knows the
lane graph" and neither number would mean anything.

Building it earlier was impossible rather than merely unwise — the synthetic
world has no topology worth the name, so it would have meant inventing the thing
the experiment is meant to test.

Argoverse 2 comes after all three, as a generalization test never trained on.
Its lane boundaries carry `mark_type`, which is the real-data version of the
dash experiment.

## Done — M3: closed loop

`LocalizationKF` holds the estimate, `engine/sequence.py` drives one scene at
a time, `tools/run_sequence.py` reports the trajectory. **Translation RMSE
falls from 0.336 m to 0.087 m, no scene diverges, and the confidently-wrong
tail goes to zero.** [RESULTS.md](RESULTS.md) has the tables.

This milestone asked whether a small bias compounds instead of averaging out.
It does not: what compounds is the evidence.

**Still open here.** The prior's covariance is not yet a model input. Open loop
the prior is drawn from a fixed distribution the model can simply learn; closed
loop the filter produces a per-frame `3 × 3` — tight after a good update, wide
after a run of rejections — and a model that knows how far to look could match
differently in the two cases. It also closes an inconsistency: the surface's
extent is pinned to a fixed truncation bound, so a wider prior has no correct
cell. The two pose backends have also not been run through the same filter.

## M4 — Make it fast

Two phases, and **the runtime comes first**. Compression is measured in a
runtime, so the runtime has to exist before any of it means anything — see
"What actually blocks what" for the measurement this ordering was learned from.

### M4a — the runtime

**Started: fp32 engines build and run 4× faster than PyTorch.** 20.95 ms to
5.28 ms on the distilled student, at identical precision and identical weights.
This phase was M5 and ran second; that ordering is why the compression numbers
below had to be taken twice. [RESULTS.md](RESULTS.md) has both.

ONNX export via `torch.export` — the input shapes are already static for this,
and the dynamo path produces a far cleaner graph than the legacy tracer for a
model this full of masks and einsums.

**TensorRT 11 does not take precision as a build flag.** `BuilderFlag.FP16`,
`BuilderFlag.INT8` and `IInt8EntropyCalibrator2` are gone; a strongly typed
network takes each layer's type from the graph. So FP16 means exporting a half
model, and INT8 means exporting one carrying quantize/dequantize nodes, which is
`nvidia-modelopt`'s job. 2:4 sparsity is still a builder flag
(`SPARSE_WEIGHTS`) and is untried.

The finale: a small C++ TensorRT backend plugging into
camera-map-localization's engine as an alternative to `PoseSampler`, so the same
`run_sequence` and `eval_sequence` evaluate both backends under one metric and
one filter.

### M4b — distillation, pruning, quantization

**All three are implemented and run.** Distillation pays, pruning pays a little
once measured on the runtime, and quantization is unmeasurable for speed until
the graph carries Q/DQ nodes. The teacher is trained and kept only to distil
from; the student is the deployment artifact.

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
   assignment carries far more than its pose — a label says which map point is
   correct, the teacher says which of the wrong ones were plausible. The pose
   is not distilled: it is three numbers the ground truth already gives
   exactly. **Run: 0.336 m to 0.264 m on test, at the same size and the same
   latency** — [RESULTS.md](RESULTS.md) has the tables.

   The assignment row is completed with the mass it withholds before the
   divergence is taken, so a student that matches everything cannot score the
   same as one that abstains correctly. The teacher runs live rather than
   cached, because the synthetic dataset redraws its noise every epoch and a
   cached pass would describe a frame the student never sees.
2. **Structured pruning of the student**: feed-forward channels, not
   unstructured masks — `torch.nn.utils.prune` zeros weights without removing
   them, which gives no GPU speedup. Prune, fine-tune, re-measure. **Run, and
   it bought nothing**: 2.28 M parameters to 1.49 M for 0.16 ms, which is
   noise. The model is activation-bound, so weights were never the cost.
   [RESULTS.md](RESULTS.md) has the Pareto table.

   Attention heads were meant to be the other half of this and are not
   reachable — `nn.MultiheadAttention` ties its projection width to
   `embed_dim`. [OPEN_ITEMS.md](OPEN_ITEMS.md) has what would change that.
3. **Quantization of the student**: PTQ for the calibration curve, then QAT with
   `nvidia-modelopt`. Neither the pose head nor the cost surface has weights to
   quantize; `mapposeformer/quantize.py` says why that matters.

   `mapposeformer/quantize.py` simulates INT8 to price it in accuracy, which
   needs no vendor runtime and runs in the test suite. It reaches 49% of the
   student's weights; the rest is inside `nn.MultiheadAttention` and out of
   reach, so this shrinks the weights by a third rather than three quarters.
   **Run: accuracy is untouched at 8 bits**, 0.261 either way. The speed
   question is M5's, because it needs integer kernels.

Report a Pareto table: accuracy against latency, one row per configuration.

### Cost in hours

From the measured throughputs — student 190 frames/s, teacher 59, forward and
backward on one A6000 — against the **37 000 steps M1 needed to converge**, which
is 2.37 M frames at the student's batch of 64 and 1.48 M at the teacher's 40.

Step-matching is the protocol, not epoch-matching. `synth_teacher.yaml` names
2000 scenes against the student's 800, so 60 epochs of it is 8.88 M frames and
41.8 h — an earlier version of this table costed a nuScenes epoch instead and
came out 8× short. The teacher gets the same optimizer budget the student had;
whether it needs more is a question its first run answers.

| stage | estimate | basis |
|---|---|---|
| teacher training, 37 000 steps | **~7.0 h** | 1.48 M frames ÷ 59 |
| student training, 37 000 steps | ~3.5 h | 2.37 M ÷ 190 |
| distillation, teacher outputs cached | **~3.5 h** | one teacher pass (4 min), then the student's own rate |
| distillation, teacher run live | ~7.0 h | `1/(1/190 + 1/188)` = 95 frames/s |
| pruning: fine-tune, three cycles | ~2.6 h | each fine-tune ≈ 25% of a run |
| quantization: PTQ calibration | ~1 min | 512 batches forward at 689 frames/s |
| quantization: QAT fine-tune | ~1.5 h | ≈30% of a run at ~1.4× step cost |
| TensorRT: export, three engines, benchmark | **<1 h** | INT8 build with calibration is the long pole |

**About sixteen hours end to end**, most of it the teacher. Caching the
teacher's outputs saves 3.5 h at 48 GB for the assignment matrices; it works
only on real data, since the synthetic stage redraws its noise every epoch.

VRAM is the other constraint worth naming, and it is linear in batch size:
measured, 0.108 GiB per sample for the student and 0.445 for the teacher. At
their own batch sizes — 64 and 40 — that is 6.9 and 17.8 GiB allocated, 8.0 and
19.1 GiB reserved. Reserved is what must fit, which is why the teacher's batch
is 40 and not 48: 48 would reserve about 23 GiB, 96% of a 24 GiB card.

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

