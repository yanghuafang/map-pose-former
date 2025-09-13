# Documentation

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The model, its frames, and its input/output contract |
| [DATASET.md](DATASET.md) | Why the data is generated, what it contains, and the ablation it exists for |
| [TRAINING.md](TRAINING.md) | The five loss terms, how to read the metrics, what goes wrong |
| [RESULTS.md](RESULTS.md) | The measured numbers: the M0 baseline and the observability ablation |
| [ROADMAP.md](ROADMAP.md) | The milestones, in order, with what each one changes |
| [OPEN_ITEMS.md](OPEN_ITEMS.md) | What is unfinished, unverified, or out of scope |

Read them in that order on a first pass. Read `OPEN_ITEMS.md` before trusting a
number.

The companion project, [camera-map-localization](https://github.com/yanghuafang/camera-map-localization),
solves the same problem by search rather than by learning. Its
`docs/ARCHITECTURE.md` is worth reading alongside this one: the observability
table, the frame conventions and the error metrics here are taken from it
directly, on purpose, so the two backends are comparable.
