# The published weights

Five checkpoints and one ONNX export, attached to a tagged release because
binaries do not belong in git history. `scripts/release.sh <tag>` assembles
them and prints a manifest; `--publish` creates the release.

**Where the files come from.** Not the working tree, unless something was
trained in it. `release.sh` pulls each checkpoint off the training host named
by `MPF_REMOTE_HOST` and `MPF_REMOTE_DIR` — the same two variables
`scripts/remote-ubuntu.sh` reads — so point them at the checkout that holds
the runs:

```bash
MPF_REMOTE_HOST=you@gpu-host MPF_REMOTE_DIR=path/to/checkout \
  scripts/release.sh v0.1
```

A wrong `MPF_REMOTE_DIR` is worth ruling out first: rsync finds nothing there,
and the run reads as one that was never trained.

Every number below is the test split, 8 880 generated frames. **Closed loop**
is `tools/run_sequence.py` over 120 sequences, which is what a vehicle
experiences; open-loop recall is a per-frame gate at 25 cm and 0.5 degrees.

| file | params | recall | closed loop | what it is |
|---|---|---|---|---|
| `teacher.pt` | 1.709 M | 96.3% | **0.091 m** | four layers, the reference model |
| `student-distilled.pt` | 0.914 M | 96.7% | **0.091 m** | two layers, distilled from it |
| `student-no-teacher.pt` | 0.914 M | 95.1% | 0.106 m | the same arm with no teacher |
| `teacher-pruned.pt` | 1.45 M | 97.4% | 0.092 m | feed-forward width cut 15.4% |
| `student-pruned.pt` | 0.78 M | 96.8% | — | the same, on the student |
| `teacher-trunk.onnx` | — | — | — | the deployable trunk; the solve stays on the host |
| `teacher-trunk-int8.onnx` | — | 95.2%¹ | 0.093 m¹ | the same trunk, calibrated, carrying quantize/dequantize nodes |

¹ Measured on the PyTorch simulation, not on a built engine. No INT8 engine has
been compiled or timed in this project.

**Why the control ships.** `student-no-teacher.pt` is not a model anyone would
deploy. It is the arm that makes the distilled student's result readable: the
two differ only in whether a teacher was present, so without it the 0.091 m is
a number with nothing to read it against. A claim nobody can check is not worth
publishing.

**INT8 ships as a graph, not as a checkpoint.** A quantized `.pt` would be
1.005× the size of `teacher.pt` and slower to run: simulated quantization wraps
every `nn.Linear` rather than narrowing it, so the weights stay fp32 and the
rounding is *added* to each linear instead of replacing it. It prices INT8 in
accuracy and compresses nothing, and a file labelled "quantized" would claim
otherwise.

`teacher-trunk-int8.onnx` is the artefact that does compress. It carries 344
`QuantizeLinear`/`DequantizeLinear` nodes across 86 layers, which is the form
TensorRT and the mobile runtimes fold into integer kernels — 6.52 → 1.69 MiB of
weights, −74%, once the engine is built.

**The `.onnx` itself is not smaller, and that is not a contradiction.** It is
9.8 MB against the fp32 graph's 8.4 MB, because ONNX stores the weights in fp32
and the Q/DQ nodes on top say what to do with them. The reduction happens when a
runtime builds the engine; the file is the recipe, not the result. Nothing here
has measured that engine — the numbers below are the simulation, and a built
INT8 engine is future work.

To reproduce the quantized numbers in process, load a checkpoint and apply it:

```python
from mapposeformer.quantize import QuantParams, calibrate, quantize

quantize(model, QuantParams())
calibrate(model, train_batches, limit=16)  # on TRAIN, never on the test split
```

Calibrating on the split being scored would report a number the model was tuned
on, which is why `tools/export.py --int8` calibrates on train. INT8 costs
−1.06 pp of frame recall on the teacher and 2 mm of closed-loop error; on the
student, −0.23 pp and 1 mm. Export one with:

```bash
tools/export.py runs/teacher/best.pt --trunk-only --int8 --out teacher-int8.onnx
```

## Loading one

```python
from mapposeformer.checkpoint import load_checkpoint

model, cfg = load_checkpoint("teacher.pt")
```

Use `load_checkpoint`, not `MapPoseFormer(cfg.model)`. A pruned checkpoint
carries a `prune_plan` that reshapes the feed-forwards before the weights load,
and constructing from the config alone fails on every pruned module.

Each `.pt` ships with the `config.yaml` its run recorded, so the configuration
that produced it is beside it rather than inferred.

## What these weights are not

They are trained on **generated** data — `configs/synth_base.yaml`, 800/60/120
scenes. The nuScenes loader and its cache are in the repository; a result on
real road geometry is not. Treat these as a reproducible reference for the
method, not as a localizer for a vehicle.
