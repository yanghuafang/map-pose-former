# map-pose-former API

Generated from the docstrings in `mapposeformer/`. This is the reference for
*what each function takes and returns*; the reasons live in the prose
documentation, and reading that first will make this page much shorter.

| Start here | |
|---|---|
| [ARCHITECTURE.md](https://github.com/yanghuafang/map-pose-former/blob/main/docs/ARCHITECTURE.md) | The model, its frames, and its input/output contract |
| [DATASET.md](https://github.com/yanghuafang/map-pose-former/blob/main/docs/DATASET.md) | What the data contains and the ablations it exists for |
| [TRAINING.md](https://github.com/yanghuafang/map-pose-former/blob/main/docs/TRAINING.md) | The loss terms and how to read the metrics |

## The shape of the library

```
mapposeformer/
  geometry.py        SE(2). Everything else assumes it is right.
  data/              Procedural worlds, the sample contract, the dataset
  model/             Tokenizer, attention, matcher, pose head, volume head
  engine/            The training loop and the evaluator
  losses.py          Five terms; the interesting one is `match`
  metrics.py         Pose error, and whether the covariance is honest
  config.py          Dataclasses first, YAML second
```

`mapposeformer.model.model.MapPoseFormer` is the assembly, and the one class to
read first. Everything above it in the call graph is data; everything below it
is one of the five parts named in its own docstring.

## Conventions in these pages

- A **pose** is always a 3-vector `(x, y, yaw)` in the vehicle convention:
  X forward, Y left, yaw counter-clockwise, metres and radians.
- Tensor shapes are written as `(B, K, 2)`, with `B` batch, `K` detection
  points, `L` map points and `G` pose hypotheses. Those letters are the einsum
  indices the code uses, so a docstring and its implementation read alike.
- A parameter documented as *borrowed* is not copied and must outlive the call.
