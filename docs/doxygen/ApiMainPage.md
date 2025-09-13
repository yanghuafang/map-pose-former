# map-pose-former API

Generated from the docstrings in `mapposeformer/`. This is the reference for
*what each function takes and returns*; the reasons live in the prose
documentation, and reading that first will make this page much shorter.

| Start here | |
|---|---|
| [ARCHITECTURE.md](https://github.com/yanghuafang/map-pose-former/blob/main/docs/ARCHITECTURE.md) | The model, its frames, and its input/output contract |
| [DATASET.md](https://github.com/yanghuafang/map-pose-former/blob/main/docs/DATASET.md) | What the data contains and the ablations it exists for |
| [RESULTS.md](https://github.com/yanghuafang/map-pose-former/blob/main/docs/RESULTS.md) | Every number that has been measured, and on what |

## The shape of the library

```
mapposeformer/
  geometry.py        SE(2). Everything else assumes it is right.
  data/              Procedural worlds, nuScenes, the sample contract
  model/             Encoder, relative attention, matcher, the assembly
  solve.py           The pose and its covariance -- no parameters
  filter.py          The SE(2) filter the corrections are fused through
  engine/            The training loop, the evaluator, the sequence driver
  losses.py          What the assignment is trained on
  metrics.py         Pose error, and whether the covariance is honest
  config.py          Dataclasses first, YAML second
  checkpoint.py      Rebuilding a model from what a run saved
  distill.py         The assignment a teacher hands a student
  prune.py           Structured pruning, and the plan that reloads it
  quantize.py        Simulated INT8 over `nn.Linear`
  tensorrt.py        The compiled engine, and its input contract
```

`mapposeformer.model.model.MapPoseFormer` is the assembly, and the one class to
read first. Everything above it in the call graph is data; everything below it
is one of the parts named in its own docstring.

**The pose head has no parameters.** `solve.py` is a weighted Procrustes with a
damped Gauss-Newton refinement, and its covariance is the curvature of the cost
it just minimised. So the network's whole job is the assignment, and a wrong
pose is a visible wrong assignment rather than a number with no explanation.

## Conventions in these pages

- A **pose** is always a 3-vector `(x, y, yaw)` in the vehicle convention:
  X forward, Y left, yaw counter-clockwise, metres and radians.
- Tensor shapes are written as `(B, K, 2)`, with `B` batch, `K` detection
  points and `L` map points. Those letters are the einsum indices the code
  uses, so a docstring and its implementation read alike.
- A parameter documented as *borrowed* is not copied and must outlive the call.
