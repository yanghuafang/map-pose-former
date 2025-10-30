# Training

```bash
./scripts/setup.sh --cuda                          # once, on the Ubuntu box
python tools/train.py --config configs/synth_base.yaml
python tools/eval.py runs/base/best.pt --split test
```

Or from the laptop, against the A6000:

```bash
./scripts/remote-ubuntu.sh --sync tools/train.py --config configs/synth_base.yaml
```

## The five loss terms

| Term | Weight | What it teaches |
|---|---|---|
| `pose` | 1.0 | Huber on the correction, at every refinement pass, on ground-truth axes |
| `volume` | 1.0 | Cross-entropy over the hypothesis grid against a Gaussian target |
| `match` | 1.0 | Which detected point is which map point |
| `cov` | 0.1 | Gaussian NLL, so the reported uncertainty is calibrated |
| `trust` | 0.2 | Whether this frame succeeded |

**`match` is the term that makes the rest work.** The pose term alone gives one
3-vector of gradient to share among 768 × 576 assignment entries, and a model
trained that way finds "predict the mean of the prior" long before it finds
correspondence. `match` gives every detected point its own target from the one
source that cannot be wrong: apply the true correction and see which map point
it lands on. Detections with no counterpart within 1 m are pushed to abstain
rather than drag the pose towards whatever they are nearest.

That target needs only the ground-truth pose, which every localization dataset
has, so the same code runs unchanged on nuScenes. It reads detections from the
model's output rather than the batch — the model matches this frame's plus the
previous frames' warped here, and re-deriving that assembly in the loss would be
the same geometry written twice.

**`volume` now trains the matcher.** When the surface was an MLP, this gradient
went into the MLP. The surface is computed from the assignment now, so the only
things it can change are the correspondences and one sharpness scalar — a
second, geometric supervision on matching.

Three details are easy to get wrong:

- The **covariance residual is detached**. Attached, the NLL becomes a learned
  per-sample weight on the pose loss, and the cheapest way to reduce it is to
  declare hard frames uncertain rather than localize them.
- The **trust target is detached** for the reverse reason: the gradient must
  improve the prediction *of* failure, never make failure look smaller.
- **Every refinement pass is supervised**, on RAFT's schedule, pass `i` of `I`
  weighted `0.8 ** (I − 1 − i)`. A refinement starting from a bad first estimate
  never reaches a good second one. When the regression head is selected its
  output is appended to the same list, so the baseline stays a matched trunk
  read by a regressor.

Every log line ends with elapsed time and an ETA computed from the rate so
far, so a run says how long it has left rather than leaving it to be derived
from a step count and a benchmark:

```
epoch 3 step 1200/12000 loss 7.42 ... frames_per_s 182.4 | 7m03s elapsed, 1h03m left
```

## Distillation

`configs/synth_distill.yaml`, or `distill.teacher=<checkpoint>` on any run. Two
extra terms appear in the log — `kd_match` on the assignment and `kd_volume` on
the cost surface — and the teacher's architecture is read back from its own
checkpoint, so only the output shapes have to agree. Width, depth and head
count are free, which is the point.

The teacher runs beside the student rather than being cached, at roughly 2× the
step time. Caching needs the same frame twice, and the synthetic dataset
redraws its noise every epoch.

## Reading the metrics

`tools/eval.py` prints four blocks:

```
do nothing (the prior's own error -- the number to beat)
all frames
trusted frames (NN% of the split; MM% refused for want of evidence)
is the covariance honest?
```

A translation RMSE of 1.2 m sounds like localization until the prior was 1.6 m
out to begin with. **Read the first block first.**

Each block reports RMSE, signed bias and worst case on longitudinal, lateral and
heading separately. The bias column matters: a steady 1 m along-track lag and
1 m of symmetric along-track jitter have identical RMSE, and the first is the
failure this problem is prone to. Only the signed mean tells them apart.

**Two gates, and the second is arithmetic.** A frame is trusted only if the
learned score passes *and* `mass` clears `min_mass`, four effective
correspondences by default. The trust head cannot catch a pose solved from three
confident wrong matches, because the features that would distinguish it are the
ones that already went wrong: the `[2,3,4,5]` ablation returned 14 m of error
with 9.97 m of it still trusted. On an undertrained model the mass gate
correctly refuses everything.

**Calibration** reports ANEES and 95% coverage. One is honest; above one is
overconfident, which is the dangerous direction — a filter told a bad frame is
certain will follow it, and no downstream gate undoes that.

Errors are resolved on the **ground-truth** axes, never the estimate's: the axes
an error is reported on must not move with the error, or a heading mistake
rotates its own yardstick. Same shape as `kitti::PoseError`.

## Two things about the hyperparameters

**The loss weights are balanced, and loss magnitudes say otherwise.** `volume`
looks like two thirds of the total, but most of that is the entropy of its own
soft target — a constant that produces no gradient. By gradient reaching the
shared trunk the split is `match` 42%, `volume` 36%, `pose` 10%, `cov` 9%,
`trust` 3%. `match` dominating is the design working.

**`grad_clip` is doing the learning rate's job.** The median gradient norm is
6.0 against a clip of 1.0, so every step is clipped and the nominal `lr` of
3e-4 is an effective 5e-5 with per-step normalization. It trains, and it is
stable, but the knob does not mean what it says. Raising the clip requires
lowering the rate in proportion — they are one setting, not two.

## What goes wrong, and what it looks like

| Symptom | Likely cause |
|---|---|
| `matched_frac` near zero | `match_radius_m` below the detection noise, or a wrong `delta` label — check `test_true_correction_aligns_detections_onto_the_map` |
| `pose` *rises* over the first few hundred steps | The solve rests on too few correspondences. `min_row_mass` must be relative to the frame's strongest match; an absolute one gates every row at step one |
| `pose` falls, `all/rmse_trans_m` matches the do-nothing block | The model predicts the prior mean; `match` is not carrying |
| `trusted/rejected_low_mass` stuck at 1.0 | Normal early, or `min_mass` set for a scale the matcher never reaches |
| `cov` in the hundreds at step 0, falling fast | Normal: a measured surface over a random assignment. If it stays, lower `loss.w_cov` |
| `volume` far above `ln(4199) = 8.34` and flat | The surface is peaked in the wrong place. Check the grid extent against the prior's truncation bounds |
| Loss NaN in the first steps | `warmup_frac` too small, or `amp: fp16` — use `bf16` |
| `trusted/fraction` is 0.0 | Early training; the trust head cannot exceed 0.5 until frames succeed |

## Which GPU a run lands on

Runs go to GPU 0. `remote-ubuntu.sh` defaults to `auto`, which is **GPU 0 if
it has room and nothing otherwise**; `scripts/pick_gpu.sh` says why busy means
*has no room* rather than *is working*.

Set
`MPF_GPU_FREE_MIB` to what a job needs, or `MPF_GPU` to a name to force a
specific card by name — a training run that accepts the default is only
checked against 4 500 MiB, which is less than it needs.

**Name a card for anything being measured.** A TensorRT engine is built per
compute capability, so a latency number from one card is not comparable with
one from another.

## Hardware

Measured on an RTX A6000 (48 GB, sm_86), and a 24 GB card runs everything: the
student reserves 8.0 GiB at batch 64 and the teacher 19.1 at batch 40. The
teacher's batch is 40 and not 48 precisely so it fits one -- 48 would reserve
about 23 GiB, 96% of a 24 GB card. Reserved is what has to fit, not allocated.

Single GPU, no DDP: the model is 2.3 M parameters and distributed training would
add indirection that teaches nothing.

- **bf16, not fp16.** Same exponent range as fp32, so no loss scaler, and the
  Procrustes `atan2` cannot underflow.
- **TF32 is fine.** The geometry runs in fp32 inside `geometry.exact_arithmetic`,
  which guards the three places that multiply a weight by a *coordinate* and
  records what bf16 costs there. A CPU test suite cannot notice, since autocast
  is off there, so `test_geometry_heads_ignore_autocast` asks for bf16
  explicitly.
- **No FP8** on Ampere; the quantization milestone targets INT8 and 2:4 sparsity.
- **The box has no IPv6 route but DNS hands out AAAA records**, so an unpinned
  download opens on IPv6 and hangs until it times out, which reads as a dead URL.
  `scripts/download_nuscenes.sh` passes `-4`.
- `train.compile=true` is off by default: a minute of warmup, and the first
  error hides behind a graph break.

## Throughput

`tools/bench.py` times the generator, the loader and the model apart, because
one `frames_per_s` cannot say which is the limit. Measured on the A6000, the
student runs at 190 frames/s forward-and-backward, the teacher at 59, and the
data pipeline delivers 247–463. [RESULTS.md](RESULTS.md) has the batch sizes
and the memory each needs. **The model is the constraint now**; the loader was,
before the temporal path and the second matching pass.

Neither model is parameter-bound — the student's weights are 9 MB against
6.9 GiB of activations — so cutting batch size is the lever, not cutting width.
[RESULTS.md](RESULTS.md) has the table and why the activations dominate.

## Checking on a run

```bash
./scripts/status.sh
```

Reports the training runs on the box, their progress and ETA, and warns when
more than one is sharing the GPU.

It counts runs by *parent* process, because a run's dataloader workers are
forks that share its argv and naive matching reports 33 runs where there is
one. It also never matches on a pattern that appears in its own command line:
`pkill -f tools/train.py` matches the shell running it, so pkill kills itself,
the targets survive, and the follow-up check reports success. Three runs once
shared one GPU for five hours that way, all writing into the same directory.

## Reproducing a run

Every checkpoint stores the `Config` that produced it — a checkpoint whose input
shape and grid extent are unknown cannot be loaded, only guessed at.
`tools/eval.py` reads that config back and applies overrides on top, which is
what makes the ablations fair: the model is unchanged and only the evidence
differs. Only `history` changes the input width and needs its own training run.
