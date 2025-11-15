# map-pose-former

**Where is the car, given what the camera sees and what the map says?** — again,
but learned this time.

Transformer map-matching localization in PyTorch: match detected landmarks
against an HD map, solve the rigid transform in closed form, report a pose
correction with a measured covariance. Then distil it, prune it, quantize it and
put it on TensorRT.

The learned counterpart to
[camera-map-localization](https://github.com/yanghuafang/camera-map-localization),
which answers the same question by searching a grid of pose hypotheses. Same
input contract, same frame conventions, same error metrics, so the two backends
are comparable frame for frame.

Written to be **read**. Every non-obvious decision carries its reason, and where
the code falls short of its own documentation it says so —
[OPEN_ITEMS.md](docs/OPEN_ITEMS.md).

## What it achieves

Test split, 8 880 generated frames, seeds disjoint from training. One RTX A6000.

| | trans | long | lat | yaw | recall @ 0.25 m, 0.5° |
|---|---|---|---|---|---|
| do nothing (the prior) | 1.611 | 1.497 | 0.596 | 0.993° | 1.3% |
| **this model** | **0.336** | 0.322 | 0.095 | 0.209° | **96.0%** |
| *the architecture it replaced* | *0.521* | *0.498* | *0.154* | *0.590°* | *86.0%* |

A 4.8× reduction on translation, 35% better than the model it replaced. On
nuScenes, on a geographically disjoint split, 1.591 m to 1.049 m.

Then made small and fast, which is what the project is for:

| | params | trans | p50 latency |
|---|---|---|---|
| teacher | 25.83 M | 0.218 | 35.26 ms |
| student, trained alone | 2.28 M | 0.336 | 20.32 ms |
| student, distilled | 2.28 M | **0.264** | 20.95 ms |
| distilled, pruned 35% | 1.49 M | 0.261 | 21.11 ms |
| **the same, on TensorRT** | 1.49 M | **0.261** | **4.93 ms** |

![accuracy against latency](docs/img/pareto.svg)

**Distillation is worth 21% of the error at no deployment cost** — the teacher
is discarded after training. **Pruning bought nothing until the model left
PyTorch**, and then 6.6%: the compression measurement had been dominated by
framework dispatch, and a third of the parameters removed could not move a
number the interpreter was setting. Compiling to TensorRT is worth 4×, at
identical weights and identical precision.

## The picture that explains the problem

Same frame, same oracle matching, different evidence:

| every landmark class | lane geometry only |
|---|---|
| ![peak](docs/img/volume_all.svg) | ![ridge](docs/img/volume_lanes.svg) |
| a peak: position is determined | a **ridge** along the road: sliding forward costs almost nothing |

Lane lines run parallel to travel, so a hypothesis that has slid a metre down
the road still lies on the same lines. Crossings, poles and signs pin
along-track position; removing them flattens the surface 31% along track while
lateral and heading barely move.

**The model does not report that surface** — it reports the cheaper one, which
is a paraboloid with no ridge. Finding that out is what the tool was built for.
`--mode fit` draws it; [RESULTS.md](docs/RESULTS.md) has the measurement.

## What it does, per frame

```
    map ──────────┐
                  ├─ tokenize ─ attend ─ assignment ─┬─ Procrustes ── delta
    detections ───┘                                  │
      this frame, and the last two                   └─ the same cost,
      warped here by egomotion                          on a grid ──── cov, trust
```

![one frame](docs/img/frame.svg)

Grey is the map, cropped around the drifted prior. Red is what the camera
reports this frame. Orange is the previous two frames warped here by odometry —
that it lands on top of the red is the temporal path checking its own
arithmetic. Green is everything after the true correction, sitting on the map.
Short red stripes are dashed lane paint, which the map stores as an attribute
and the detector sees as geometry.

No world coordinate reaches the network, so it cannot memorise a city instead of
learning to match. Output is `(delta, covariance, trust)` — what the classical
repo's `LocalizationKF::Update` already consumes.

## Three ideas worth taking

**Solve the geometry, learn the correspondence.** Once the model has said which
detected point is which map point, the transform is determined — so the pose
head is closed-form weighted Procrustes with no parameters, and all capacity
goes to matching.

That once carried an accuracy claim it could not support: against a plain
regression baseline the first version **lost seven to one out of distribution**,
because a rigid fit over confident wrong matches is unbounded and a `tanh` is
not. The claim was retracted in place. The head is now a robust estimator, and
the rematch split the difference: it wins heading by a factor of two and still
loses translation.

**Compute the cost surface, do not predict it.** An MLP trained to draw cost
surfaces produces ridges that look right without evidence behind them. The
assignment-weighted error over all 4199 hypotheses is a closed form in eleven
statistics the Procrustes solve already computes — a few 2 × 2 matmuls, and
1.1 M parameters deleted. The two output paths become one objective read twice.

**What accumulates is evidence about the pose *error*, not about a pose.** That
is what lets a past detection reach this frame through measured egomotion alone,
before the correction is known; [ARCHITECTURE.md](docs/ARCHITECTURE.md) has the
algebra. Fusion is more tokens, warped, with an age embedding. Static shapes
survive, which TensorRT needs.

## What is real, and what is not

- **The map is real; the detections are not.** Stage 0 is procedural, because
  KITTI ships no HD map and synthesizing one from ground truth makes the map a
  function of the pose being predicted. nuScenes' surveyed map arrived at M2a on
  a geographically disjoint split.
- **Perception is an input.** No detector is trained, and none has been run:
  detections are still cut from the map and corrupted, on both datasets. The
  file contract for a real one is written and tested.
- **Open loop only.** The prior is drawn from a distribution, not produced by
  the previous frame, which flatters any localizer.
- **0.44% of frames are confidently wrong.** The median frame is calibrated and
  the tail-excluded ANEES is 1.03, but 39 frames of 8 880 have a covariance that
  is catastrophically too small. Closed loop hands these to a Kalman filter, so
  the open problem is detecting them, not rescaling everything.

Full list: [OPEN_ITEMS.md](docs/OPEN_ITEMS.md).

## Running it

```bash
./scripts/setup.sh        # conda env + CPU torch
./scripts/ci.sh           # format, lint, tests
./scripts/run_smoke.sh    # train, evaluate, and draw a frame
```

No dataset, no GPU: the scenes are generated locally. CUDA training and the
TensorRT export need Linux; everything else runs on macOS too, which is why
`scripts/remote-ubuntu.sh` exists — edit on a laptop, run on the box.

**What you need to train it.** The numbers here were measured on an A6000, but
nothing requires one:

| | VRAM reserved | fits |
|---|---|---|
| smoke run, tests | none — CPU | any laptop |
| the student, batch 64 | 8.0 GiB | a 12 GB card |
| the teacher, batch 40 | 19.1 GiB | **a 24 GB card — a 3090 or 4090** |

The teacher's batch is 40 rather than 48 for exactly this reason: 48 would
reserve about 23 GiB, which is 96% of a 24 GB card and too close to the edge.
Halve `train.batch_size` for less, at some cost in throughput.

### Experiments

One checkpoint, less evidence — only the last needs its own training run:

```bash
tools/eval.py runs/base/best.pt --split test                                    # baseline
tools/eval.py runs/base/best.pt --split test 'data.sample.keep_classes=[0,1]'   # observability
tools/eval.py runs/base/best.pt --split test data.sample.stripe_dashed=false    # the dashes
tools/eval.py runs/base/best.pt --split test model.refine_iters=1               # refinement
tools/eval.py runs/base/best.pt --split test model.irls_iters=0                 # robustness
tools/train.py --config configs/ablate_no_history.yaml                          # temporal
```

Restricted to lane geometry the model recovers essentially nothing along track —
1.367 m against a 1.497 m prior — while recovering 83% laterally. That asymmetry
is what this project exists to measure, and it is the one conclusion that has
survived every correction to the instrument.

## Tools

| Command | Purpose |
|---|---|
| `tools/train.py` | Train. `--config` plus `section.field=value` overrides |
| `tools/eval.py` | Evaluate a checkpoint, with the do-nothing baseline alongside |
| `tools/viz_sample.py` | Draw one frame: map, detections, warped history, both corrections |
| `tools/viz_volume.py` | Draw the cost surface, in either of the two senses it has |
| `tools/bench.py` | Where the step time goes — generator, loader and model, apart |
| `tools/prepare_nuscenes.py` | Turn nuScenes into per-scene worlds, once, offline |
| `tools/prune.py` | Remove feed-forward channels structurally, and say what it cost |
| `tools/latency.py` | One forward pass at batch 1, p50 and p99, under a fixed protocol |
| `tools/pareto.py` | Accuracy against latency, every configuration in one table |
| `tools/plot_pareto.py` | Draw that table, so the shape of the result is visible |
| `tools/export.py` | Checkpoint to ONNX, and on to a TensorRT engine |
| `tools/trt_latency.py` | Time an engine, under the protocol the Pareto table uses |

| Script | Purpose |
|---|---|
| `scripts/setup.sh` | conda env and torch, `--cuda` for the training box |
| `scripts/ci.sh` | Format, lint, tests; `--smoke` adds the end-to-end run |
| `scripts/status.sh` | What is training on the box, how far along, and whether runs are competing |
| `scripts/run_smoke.sh` | Train, evaluate and visualise in minutes, on CPU |
| `scripts/docs.sh` | Doxygen API reference from the docstrings |
| `scripts/remote-ubuntu.sh` | Mirror this tree to the Ubuntu box, run there, fetch results back |
| `scripts/release.sh` | Assemble the trained artifacts and attach them to a GitHub release |
| `scripts/download_*.sh` | nuScenes and Argoverse 2, resumable |

## Layout

```
mapposeformer/      The library. Start at model/model.py
  data/             Procedural worlds, the sample contract, the dataset
  model/            Tokenizer, attention, matcher, pose head, volume head
  engine/           The training loop and the evaluator
configs/            YAML overriding the dataclass defaults
tools/              Command-line entry points
tests/              Geometry, data invariants, model numerics
docs/               Start at docs/README.md
```

## License

MIT — see [LICENSE](LICENSE).
