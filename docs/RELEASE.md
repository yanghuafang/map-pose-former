# Releasing weights

Model weights do not go in git. Every version of a binary stays in history
forever, so committing them would fix the repository at about 600 MB and grow it
with each retrain — and the teacher is 99 MB against GitHub's 100 MB per-file
limit. `runs/` and `build/` are in `.gitignore` for that reason.

They go on a **GitHub release** instead: 2 GB per file, outside history, and a
release tagged against a commit says the thing that matters — *these weights
came from this code*.

## What is published

| file | from | params | test trans | note |
|---|---|---|---|---|
| `student.pt` | `runs/m1_base` | 2.28 M | 0.336 | the M1 baseline |
| `distilled.pt` | `runs/distilled` | 2.28 M | 0.264 | **the deployment artifact** |
| `pruned25.pt` | `runs/p25` | 1.49 M | 0.261 | 35% of the channels removed |
| `teacher.pt` | `runs/teacher` | 25.83 M | 0.218 | only useful for distilling from |
| `*.onnx` | the same four | — | — | portable graph, batch 1, static shapes |

`scripts/release.sh` does it, and is a dry run unless told otherwise:

```bash
./scripts/release.sh v0.1              # fetch, export, print the manifest
./scripts/release.sh v0.1 --publish    # ...and create the release
```

It pulls any missing checkpoint from the training box, exports the ONNX beside
it, and refuses a file over the 2 GB per-asset limit. Publishing is public and
awkward to retract, so it never happens by accident.

## Loading a checkpoint

Every checkpoint carries the `Config` that produced it, so nothing has to be
guessed. `tools/eval.py` reads it back and applies overrides on top.

```bash
tools/eval.py distilled.pt --split test
```

**A pruned checkpoint needs its plan applied first.** Pruning removes channels,
so the saved widths no longer match what the config implies and a bare
`load_state_dict` fails on shape. The plan travels with the weights and
`tools/export.py`'s `load()` replays it:

```python
from tools.export import load

model, cfg = load("pruned25.pt")  # applies ckpt["prune_plan"], then loads
```

## Why the ONNX and not the TensorRT engine

A TensorRT engine is compiled for one compute capability and one TensorRT
version. The ones built here run on sm_86 with TensorRT 11.2 and nowhere else,
so publishing them would mostly generate reports from people whose card cannot
load them. The ONNX is the portable artifact, and `tools/export.py --trt`
rebuilds an engine in minutes on whatever hardware is actually present.

Both exports were checked against eager PyTorch before release: the worst
disagreement across all four outputs is 1.1e-5.

## Reproducing instead

Publishing nothing is a defensible alternative, and for a project whose purpose
is the pipeline rather than the weights it may be the better one. The whole
sequence is in the repository and costs about sixteen hours on one A6000, or
on any 24 GB card -- the teacher is the largest thing here and reserves 19.1 GiB:

```bash
tools/train.py --config configs/synth_base.yaml                 # the student
tools/train.py --config configs/synth_teacher.yaml              # the teacher
tools/train.py --config configs/synth_distill.yaml              # distil it
scripts/m4.sh                                                   # prune, quantize, measure
```

Numbers to expect are in [RESULTS.md](RESULTS.md), each beside the command that
produces it.
