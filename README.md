# map-pose-former

**Where is the car, given what the camera sees and what the map says?** — again,
but learned this time.

A readable PyTorch implementation of transformer map-matching localization:
match detected landmarks against an HD map, solve the rigid transform in closed
form, and report a pose correction with a calibrated covariance. Then prune it,
distil it, quantize it, and put it on TensorRT.

It is the learned counterpart to
[camera-map-localization](https://github.com/yanghuafang/camera-map-localization),
which answers the same question by searching a grid of pose hypotheses. The
input contract, the frame conventions and the error metrics are deliberately
identical, so the two backends are comparable frame for frame.

Written to be **read**. Every non-obvious decision carries the reason it was
made, and where the implementation falls short of its own documentation, it says
so — [OPEN_ITEMS.md](docs/OPEN_ITEMS.md).

## Try it in three commands

```bash
./scripts/setup.sh        # venv + CPU torch
./scripts/ci.sh           # 27 tests
./scripts/run_smoke.sh    # train, evaluate, and draw a frame
```

No dataset download, no GPU. The scenes are generated locally.

## What it does, per frame

```
detections ─┐                                  ┌─ soft assignment ─ Procrustes ─ delta
            ├─ tokenize ─ L x [self | cross] ──┤
  local map ─┘                                 └─ attention pool ─ volume ─ logits, cov, trust
```

The map, cropped around the drifted **prior pose** and expressed in its frame.
The detections, in the **true ego frame**, where a sensor produces them. The
answer is the transform between the two — which means no world coordinate ever
enters the network, and it cannot memorise a city instead of learning to match.

Output is `(delta, covariance, trust)`: exactly what the classical repo's
`LocalizationKF::Update` already consumes.

## The two ideas worth taking away

**Solve the geometry, learn the correspondence.** Once the model has said which
detected point is which map point, the transform is *determined* — so the pose
head solves weighted Procrustes in closed form and has no parameters at all. All
the capacity goes to the question that is genuinely hard. The model can only be
right for the right reason, and when it is wrong the assignment matrix says
where. `RegressionPoseHead` is kept as a baseline, and comparing the two out of
distribution is the more instructive half of the experiment — which is why it is
worth saying that the baseline won it, seven to one, and that the argument above
is currently one and a half ideas rather than two.
[RESULTS.md](docs/RESULTS.md) has the table.

**Predict the surface, not just its peak.** A stretch of parallel lane lines
*should* produce a ridge along the road, not a peak — lane geometry runs parallel
to travel, so sliding a hypothesis forward costs almost nothing. A single
regressed pose cannot say that. A cost volume over the same `(forward, left,
yaw)` grid the classical search evaluates can, its softmax-weighted spread *is*
the measurement covariance, and classification over a grid survives INT8 far
better than coordinate regression does.

## What is real, and what is not

- **The data is generated.** KITTI ships no HD map, and the classical repo's
  workaround — synthesizing one from the ground-truth path — is fatal for a
  learned model: the map becomes a function of the pose it is asked to predict.
  Procedural scenes buy a known answer and *controllable evidence*. nuScenes is
  [M3](docs/ROADMAP.md).
- **Perception is an input.** No detector is trained or run. Detections come
  from the map with a statistical error model applied — dropout, correlated
  per-element bias, range-dependent noise, clutter.
- **Open loop only.** The prior is drawn from a distribution, not produced by
  the previous frame. Open-loop numbers always look better than the system is.
- **The numbers are open loop and generated.** On the test split the model
  takes translation RMSE from 1.61 m to 0.52 m, and to 0.39 m on the 95% of
  frames it says to trust — see [RESULTS.md](docs/RESULTS.md), including the
  ablation where removing lane geometry makes it 9x *worse* than the prior.

Full list: [OPEN_ITEMS.md](docs/OPEN_ITEMS.md).

## The experiment this is built for

Map features are not interchangeable. Lane lines and road boundaries constrain
lateral position and heading and say almost *nothing* about along-track
position; poles, signs, stop lines and crossings are what pin it. That is a
claim, and generated data is what makes it testable:

```bash
tools/eval.py runs/base/best.pt --split test                          # baseline
tools/eval.py runs/base/best.pt --split test data.sample.keep_classes=[0,1]
```

Same checkpoint, less evidence. Longitudinal RMSE should grow sharply while
lateral and heading barely move. On a public dataset this experiment is not
possible at all.

## Tools

| Command | Purpose |
|---|---|
| `tools/train.py` | Train. `--config` plus `section.field=value` overrides |
| `tools/eval.py` | Evaluate a checkpoint, with the do-nothing baseline alongside |
| `tools/viz_sample.py` | Draw one frame as an SVG: map, detections, and both corrections |

| Script | Purpose |
|---|---|
| `scripts/setup.sh` | venv and torch, `--cuda` for the training box |
| `scripts/ci.sh` | Format, lint, tests; `--smoke` adds the end-to-end run |
| `scripts/run_smoke.sh` | Train, evaluate and visualise in minutes, on CPU |
| `scripts/remote-ubuntu.sh` | Mirror this tree to the Ubuntu box and run there |

## Layout

```
mapposeformer/      The library. Start at model/model.py
  data/             Procedural worlds, the sample contract, the dataset
  model/            Tokenizer, attention, matcher, pose head, volume head
  engine/           The training loop and the evaluator, in full, no framework
configs/            YAML overriding the dataclass defaults, nothing more
tools/              Command-line entry points
tests/              pytest: geometry, data invariants, model numerics
docs/               Start at docs/README.md
```

## License

MIT — see [LICENSE](LICENSE).
