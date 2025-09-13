# Roadmap

Each milestone is a commit boundary and a documentation section. What is done
is marked; what is not is described precisely enough to start.

## Done — M0: the stage-0 system

Procedural data, the model, the five losses, the training loop, the evaluator,
the ablation harness, and the tests. Runs end to end on CPU in minutes
(`scripts/run_smoke.sh`) and on the A6000 in a couple of hours
(`configs/synth_base.yaml`).

## M1 — Ablations

Reproduce the observability table in `data/classes.py` as measured numbers.
This milestone is the point of the synthetic stage: on real data none of these
experiments is possible, because the evidence cannot be switched off.

**Done — the class ablations.** Longitudinal error grows from 0.50 m to 1.43 m
on lane geometry alone, against a 1.50 m prior, while heading does not move.
Poles and signs turn out to carry most of the along-track information, and
nuScenes has neither. Full table and reading in [RESULTS.md](RESULTS.md).

That last row also produced something nobody asked for: strip the lane geometry
and the closed-form head returns 14 m of error, nine times *worse* than doing
nothing, and the trust head does not catch it. A `mass` gate is the obvious
answer and it is not yet written — M4 cannot ship without one.

**Done — the head comparison, and it went the other way.** The regression
baseline matches the closed-form head in distribution and beats it seven to one
outside it, because a rigid fit over bad correspondences is unbounded. The
accuracy claim in `ARCHITECTURE.md` is retracted there rather than quietly
edited. The follow-up is a *robust* closed-form solve — reweighting, or gating
the assignment on matchability before solving instead of only scaling by it —
which is the experiment that would settle the argument properly.

**Still to do:**

- a robust closed-form head, against both baselines
- prior noise swept past the grid extent. Sweeping it *within* the grid is done
  and shows no cliff — 67% of the error removed at the trained spread, 60% at
  three times it. Going past the extent means moving the grid too, which is a
  retrain, because `config._validate` pins the two together on purpose.

## M2 — Temporal fusion

Frames are currently independent. The classical repo accumulates evidence across
a window, and the insight that makes it work transfers directly: **what
accumulates is evidence about the pose *error*, not about a pose**. Every anchor
is the same drifting estimate seen at a different time, so a past observation
must be conjugated into the current anchor frame by the relative egomotion,
`M · T_offset · M⁻¹`, before it can be fused.

Concretely: a streaming memory of the last K frames' tokens, warped by
`T_curr_prev`, attended to by the current frame. The dataset already generates
frames along a trajectory, so this needs a sequence sampler, not new geometry.

## M3 — nuScenes

`nuScenes` + map expansion v1.3, on the **geographically disjoint split** (the
StreamMapNet split, not the official one — the official train/val scenes overlap
spatially and a localizer evaluated on it is partly reciting).

Three things change and nothing else should:

- an offline `tools/prepare_nuscenes.py` writing the same tensors this project
  already consumes, so the training loop never imports the devkit
- egomotion from the CAN bus, replacing differenced ground truth
- **no poles and no traffic signs.** The map expansion does not carry them, so
  along-track observability is structurally weaker than in the synthetic stage
  and the metrics will say so. This is stated in `data/classes.py` in advance
  rather than discovered in a results table.

Argoverse 2 afterwards as a generalization test, never trained on. Its Motion
Forecasting split carries the vector map and the ego trajectory with no imagery
— 6.4 GB for val against the Sensor split's 900 GB — and its lane boundaries
carry `mark_type`, solid against dashed. A dashed line's stripe *ends* are
along-track evidence a continuous polyline throws away, and this is the only
dataset here that can test whether storing them helps.

`scripts/download_nuscenes.sh` and `scripts/download_argoverse2.sh` fetch both.
Run them by hand: nuScenes' terms are accepted by a person, not a script.

Waymo is **not used**, and the reason is worth keeping because it is a real
constraint rather than a preference. Its map is the only one here carrying
`stop_sign` and `speed_bump` — a point landmark and a perpendicular one, which
is exactly the along-track evidence `data/classes.py` says nuScenes lacks. But
Google's auth endpoints are unreachable from the training box, and the
OpenDataLab mirror needs a second account and terms acceptance of its own for a
dataset this project has not yet earned the right to need.

So the along-track gap stays measured rather than closed: the M1 ablation puts
it at 2.3x worse longitudinal error on nuScenes' class set, and that number
stands as the cost of the map, not as a defect to be papered over with a third
dataset. Revisit if M3 makes it the binding constraint.

## M4 — Closed loop

Everything so far is open loop: one frame, one correction, error measured
against the label. A deployment feeds the correction back into a filter, and the
next frame's prior is the previous frame's output. Open-loop numbers always look
better than the system is.

This is where the classical repo's `LocalizationKF` returns: the model's
`(delta, cov, trust)` is exactly what it consumes. The honest comparison is a
sequence run of both backends through the same filter and the same
`eval_sequence` metrics.

## M5 — Compression

In this order, because each stage changes what the next one has to work with:

1. **Distillation.** Train `configs/synth_teacher.yaml` (256-dim, 8 layers),
   then distil into the 128-dim student on the assignment matrix and the cost
   volume — both are distributions, so KL is the natural objective and the
   teacher's soft assignment carries far more than its pose does.
2. **Structured pruning.** Attention heads and FFN channels, *not* unstructured
   masks: `torch.nn.utils.prune` zeros weights without removing them, which
   gives exactly zero speedup on a GPU. Prune, then fine-tune, then re-measure.
3. **Quantization.** PTQ first for the calibration curve, then QAT with
   `nvidia-modelopt`. The volume head should survive INT8 comfortably —
   classification over a grid degrades gracefully — and the Procrustes head is
   arithmetic, not weights, so it stays in fp32 or fp16 by construction.

Report a Pareto table: accuracy against latency, one row per configuration. A
compression milestone with no table is a claim.

### The target is the A6000, and that constrains what this can show

Decided up front, because it decides what the table means. The model is 3.4 M
parameters over about 830 tokens, and at that size an A6000 is bound by kernel
launch overhead and memory traffic rather than by arithmetic. INT8 may buy
nothing measurable, and 2:4 sparsity on layers this narrow is close to noise.

That is a real result and it will be reported as one. What it is not is
evidence that the *techniques* do not work — it is evidence that this model on
this card is not where they pay. An embedded target is where they would be, and
naming that distinction is the point of fixing the target in advance rather
than discovering a flat table and explaining it afterwards.

So the protocol is fixed before the first measurement, because a latency number
without one is unfalsifiable:

- batch 1, which is what a vehicle runs
- CUDA graphs, so launch overhead is amortized rather than measured
- 200 warmup iterations, then p50 and p99 over 1000
- the same protocol for every row, including the fp32 baseline

## M6 — Deployment

ONNX export (the input shapes are already static for this reason), then
TensorRT: FP16, INT8, and 2:4 structured sparsity — all three supported on
Ampere, none of them FP8, which the A6000 does not have.

The finale worth doing: a small C++ TensorRT backend that plugs into
camera-map-localization's engine as an alternative to `PoseSampler`, so the same
`run_sequence` and `eval_sequence` tools evaluate both backends under one metric
and one filter. That closes the loop on the whole exercise.
