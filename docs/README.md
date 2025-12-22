# Documentation

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The model, its frames, and its input/output contract |
| [DATASET.md](DATASET.md) | What the data contains and the ablations it exists for |
| [TRAINING.md](TRAINING.md) | The loss terms, how to read the metrics, what goes wrong |
| [RESULTS.md](RESULTS.md) | The measured numbers, and which are superseded |
| [ROADMAP.md](ROADMAP.md) | The milestones, with what each one changes |
| [OPEN_ITEMS.md](OPEN_ITEMS.md) | What is unfinished, unverified, or out of scope |
| [RELEASE.md](RELEASE.md) | Publishing weights, and why they are not in git |

Read them in that order. **Read `OPEN_ITEMS.md` before trusting a number** —
every table in `RESULTS.md` is one run of one configuration, open loop, and
`OPEN_ITEMS.md` is where that is said plainly. `RESULTS.md`'s head-count sweep
puts a number on what that costs: three runs of the same model, one flag apart,
span 48% non-monotonically. Differences smaller than that are not separated.

`scripts/docs.sh` generates the API reference from the docstrings.

The companion project,
[camera-map-localization](https://github.com/yanghuafang/camera-map-localization),
solves the same problem by search. Its `docs/ARCHITECTURE.md` is worth reading
alongside this one: the observability table, the frame conventions and the error
metrics here come from it directly, so the two backends are comparable.

## Where this sits in the literature

Not a leaderboard — these numbers are on generated data. What matters is which
problem this is, because two families of paper share the words and not the task.

**Pose refinement**, which is this one: a prior already within metres, correct
the drift. [BEV-Locator](https://link.springer.com/article/10.1007/s11432-023-4114-6)
cross-attends map queries against BEV features and regresses the pose;
[EgoVM](https://arxiv.org/abs/2307.08991) matches vectorized map elements
against BEV features through a transformer decoder;
[U-ViLAR](https://arxiv.org/abs/2507.04503) splits the problem into
differentiable association and registration with an uncertainty for each, which
is structurally what this project does. That is the comparison set.

**Coarse re-localization**, which is not: tens of metres of uncertainty, from
images, against OpenStreetMap. [OrienterNet](https://arxiv.org/abs/2304.02009)
matches a neural BEV against a neural map for a probability volume over 3-DoF
poses; [SegLocNet](https://arxiv.org/abs/2502.20077) does it by exhaustive
template matching and ablates why — regression-based pose estimation and learned
neural maps both hurt generalization to unseen cities. Putting a 0.5 m RMSE next
to a 59% recall@1 m would compare nothing to nothing.

The idea taken from the second family is the one that transfers: a surface over
poses beats a point estimate, and it should be computed rather than predicted.
`ARCHITECTURE.md` has where that worked here and where it did not.
