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
| `pose` | 1.0 | Huber on the correction, resolved on ground-truth axes |
| `volume` | 1.0 | Cross-entropy over the hypothesis grid, against a Gaussian target |
| `match` | 1.0 | Which detected point is which map point |
| `cov` | 0.1 | Gaussian NLL, so the reported uncertainty is calibrated |
| `trust` | 0.2 | Whether this frame succeeded |

**`match` is the term that makes the rest work.** The pose term alone gives a
single 3-vector of gradient to share among 256 × 576 assignment entries, and a
model trained that way finds "predict the mean of the prior" long before it
finds correspondence. The match term gives every detected point its own target,
from the one source that cannot be wrong: apply the true correction and see
which map point it lands on. Positives push mass onto that point; detections
with no counterpart within 1 m are pushed to abstain instead of dragging the
pose towards whatever they happen to be nearest.

That target needs no correspondence bookkeeping in the dataset — only the
ground-truth pose, which every localization dataset has. The same code will run
unchanged on nuScenes.

Two details are deliberate and easy to get wrong:

- The **covariance residual is detached**. Attached, the NLL becomes a learned
  per-sample weight on the pose loss, and the cheapest way to reduce it is to
  declare hard frames uncertain rather than to localize them.
- The **trust target is detached** for the same reason in reverse: the gradient
  must improve the prediction *of* failure, never make failure look smaller.

## Reading the metrics

`tools/eval.py` prints three blocks, and the first is the one usually missing:

```
do nothing (the prior's own error -- the number to beat)
all frames
trusted frames (NN% of the split)
```

A translation RMSE of 1.2 m sounds like localization until you notice the prior
was 1.6 m out to begin with. **Always read the first block first.**

Each block reports RMSE, **signed bias** and worst case, on longitudinal,
lateral and heading separately. The bias column is not decoration. A steady 1 m
along-track lag and 1 m of symmetric along-track jitter have identical RMSE, and
the first is the failure this problem is prone to — lane geometry aliases along
the road, so a hypothesis that has slid forward costs almost nothing. Only the
signed mean tells them apart.

Errors are resolved onto the **ground-truth** axes, never the estimate's: the
axes an error is reported on must not move with the error being reported, or a
heading mistake rotates its own yardstick. Same reasoning, same code shape, as
`kitti::PoseError` in the classical repo.

## Things that go wrong, and what they look like

| Symptom | Likely cause |
|---|---|
| `matched_frac` near zero | `match_radius_m` is below the detection noise, or the `delta` label is wrong — check `test_true_correction_aligns_detections_onto_the_map` |
| `pose` falls, `all/rmse_trans_m` matches the do-nothing block | The model is predicting the prior mean. The `match` term is not carrying |
| `cov` spikes to double digits early | Normal for the first few hundred steps: the surface is sharp before it is accurate. If it stays there, lower `loss.w_cov` |
| Loss NaN in the first steps | `warmup_frac` too small, or `amp: fp16` — use `bf16`, which needs no loss scaler and cannot underflow the Procrustes `atan2` |
| `trusted/fraction` is 0.0 | Early training. The trust head cannot exceed 0.5 until some frames actually succeed |

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

## Hardware notes

One RTX A6000 (48 GB, Ampere, sm_86). Single GPU, no DDP: the model is 3.4 M
parameters and distributed training would add indirection that teaches nothing
about localization.

- **bf16, not fp16.** Same exponent range as fp32, so no loss scaler, and the
  Procrustes `atan2` cannot underflow.
- **TF32 matmuls** are on by default in recent PyTorch on Ampere and are fine
  here — the geometry is done in fp32 inside an autocast-exempt closed form.
  That exemption is `geometry.exact_arithmetic`, and both heads use it: bf16
  would otherwise reach the two places that multiply a weight by a *coordinate*
  and put a 0.25 m lattice under a 40 m map point. It cost 8 mm of translation
  and a covariance error five times the variance floor when it was missing, and
  a CPU test suite cannot notice its absence — autocast is off there — so
  `tests/test_model.py::test_geometry_heads_ignore_autocast` asks for bf16
  explicitly.
- **No FP8.** Ampere has no FP8 tensor cores; the quantization milestone targets
  INT8 and 2:4 structured sparsity, both of which sm_86 does support.
- **The box has no IPv6 route but DNS hands out AAAA records**, so any download
  that is not pinned to IPv4 opens on IPv6 and hangs until it times out — which
  reads as a dead URL and is not. `scripts/download_nuscenes.sh` passes `-4`
  for this reason; check `ip -6 route show default` before believing a network
  failure.
- `train.compile=true` is off by default. It costs a minute of warmup and hides
  the first error behind a graph break.

## What it actually runs at

`tools/bench.py` runs the generator, the loader and the model apart, because
the single `frames_per_s` in the training log cannot say which of them is the
limit. On the training box — one RTX A6000 and an i9-9820X with 20 logical
cores, at `configs/synth_base.yaml`:

| | frames/s |
|---|---|
| one sample, cached scene | 52 per worker |
| one sample, new scene | 27 per worker |
| loader, cold cache (the first epoch) | 269 |
| loader, warm cache (every epoch after) | 340 and rising with cache coverage |
| model, forward only | 2700 |
| model, forward + backward | **746** |

**The loader is the limit here, not the GPU**, and it is worth knowing which:
the first attempt at this ran at 180 frames/s with the A6000 at 0% utilization,
and no amount of attention to the model would have moved it. The two things
that did were keeping every generated scene rather than one — a shuffled
sampler asks for a different scene at nearly every index, and a rebuild is the
difference between 27 and 52 samples per second per worker — and raising
`num_workers` to 16. Twenty is slower than sixteen, which is contention, not
headroom.

Run it before tuning anything. A change that makes the model faster is worth
nothing while the answer above says `data`.

## Reproducing a run

Every checkpoint stores the `Config` that produced it, because a checkpoint
whose input shape and grid extent are unknown cannot be loaded, only guessed at.
`tools/eval.py` reads that config back and applies overrides on top, which is
what makes the ablation above a fair comparison: the model is unchanged and only
the evidence differs.
