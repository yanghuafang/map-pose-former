# Results

Measured numbers, and what they mean. Every table here is reproducible from a
checkpoint and a command; none of them is quoted from a training log.

Read [OPEN_ITEMS.md](OPEN_ITEMS.md) first. These are **open-loop** numbers on
**generated** data, and both of those flatter a localizer.

## M0 — the stage-0 baseline

`configs/synth_base.yaml`, one RTX A6000, 37 000 steps in 92 minutes at 470
frames/s. Evaluated on the test split: 120 scenes, 8 880 frames, generated from
a seed range disjoint from training, so no road is ever shared.

```bash
tools/train.py --config configs/synth_base.yaml
tools/eval.py runs/base/best.pt --split test
```

| | trans | long | lat | yaw | recall @ 0.25 m, 0.5° |
|---|---|---|---|---|---|
| do nothing (the prior) | 1.611 | 1.497 | 0.596 | 0.993° | 1.3% |
| all frames | 0.521 | 0.498 | 0.154 | 0.590° | 86.0% |
| trusted (95.4% kept) | **0.393** | 0.374 | 0.123 | 0.227° | 89.4% |

Metres and degrees, RMSE, resolved on the ground-truth axes.

**Read the first row first.** A 0.52 m translation RMSE means something only
against the 1.61 m it started from — a 3.1× reduction, and 4.1× on the frames
the model says to trust.

**The trust head earns its 4.6%.** Dropping those frames barely moves
translation (0.521 → 0.393) but takes heading from 0.590° to 0.227°, and the
worst-case yaw from 35.8° to 2.0°. It is not rejecting frames that are slightly
wrong; it is rejecting the handful that are catastrophically wrong, which is
exactly what a measurement gate is for.

**Bias is negligible** — 7 mm along track, 9 mm lateral. The failure this
problem is prone to is a steady along-track lag, and there is not one here.

## M1 — the observability ablation

The claim in [`data/classes.py`](../mapposeformer/data/classes.py) is that the
landmark classes are not interchangeable: lane geometry runs parallel to travel
and cannot see along-track position, while perpendicular and point landmarks
can. That is a claim, and it is testable.

**One checkpoint, evaluated under less evidence.** The model is unchanged; only
what it is allowed to see differs, so nothing here is a training artefact.

```bash
tools/eval.py runs/base/best.pt --split test 'data.sample.keep_classes=[0,1]'
```

| evidence kept | long | lat | yaw |
|---|---|---|---|
| everything | **0.498** | **0.154** | 0.590° |
| no perpendicular — lanes, poles, signs `[0,1,4,5]` | 0.606 | 0.165 | 0.599° |
| **nuScenes' classes** — no poles or signs `[0,1,2,3]` | 1.123 | 0.267 | 0.510° |
| lane geometry only `[0,1]` | 1.428 | 0.272 | 0.557° |
| *(the prior, for scale)* | *1.497* | *0.596* | *0.993°* |
| no lane geometry `[2,3,4,5]` | 11.771 | 8.012 | 30.909° |

Four things fall out, and the first three were predicted in advance.

**Lane geometry cannot see along track.** Restricted to lane dividers and road
boundaries, longitudinal RMSE is 1.428 m against a prior of 1.497 m — the model
recovers essentially none of it. Lateral halves and heading improves by nearly
half, from the same evidence, in the same frames. The asymmetry is the whole
table in one row.

**Poles and signs carry most of it.** Removing the perpendicular classes costs
almost nothing (0.498 → 0.606), while removing poles and signs costs more than
twice that (0.498 → 1.123). Density is why: poles sit every 22 m and signs
every 70 m, while stop lines and crossings only appear at intersections, 110 m
apart. Along-track evidence that arrives intermittently constrains
intermittently.

**This predicts the real-data milestone.** `[0,1,2,3]` is exactly
`NUSCENES_AVAILABLE` — the map expansion carries no poles and no traffic signs.
So M3 should expect roughly **2.3× worse longitudinal error** than the
synthetic stage, from the map's contents alone and before any question of
real-world difficulty. `classes.py` said this before the number existed; the
number agrees.

**The closed-form head does not degrade gracefully.** Strip the lane geometry
and the model is not merely worse than the prior, it is 9× worse — 14.2 m of
translation RMSE against 1.6 m. Few correspondences and a rigid solve mean a
handful of bad matches move the answer arbitrarily far, where a regressor
bounded by `tanh` would simply have returned something small and wrong.

That is a real cost of solving the geometry rather than learning it, and it is
the argument for `mass` and for the trust head existing at all. **The trust
head does not catch it**: it still reports 9.97 m on the frames it keeps.
Feeding that to a filter would be worse than feeding it nothing, so `mass` is
gated separately: a frame is kept only if the trust score passes *and* the
assignment mass clears `min_mass`.

## M1 — the two pose heads, and a claim that did not survive

`configs/synth_base.yaml` twice, identical but for `model.pose_head`. Both runs
train all five losses, so the regression head sits on a trunk that correspondence
supervision shaped — this is not "regression instead of matching", it is
regression reading a matched trunk. `delta_match` is still computed in both.

```bash
tools/train.py --config configs/synth_base.yaml model.pose_head=regression   train.out_dir=runs/regression
```

All frames, test split. Bold is the better of the pair.

| | Procrustes | Regression |
|---|---|---|
| **in distribution** | | |
| trans | **0.521** | 0.531 |
| long | **0.498** | 0.530 |
| lat | 0.154 | **0.034** |
| yaw | 0.590° | **0.156°** |
| worst-case yaw | 35.8° | **2.1°** |
| recall @ 0.25 m, 0.5° | 86.0% | **94.8%** |
| **lane geometry only** `[0,1]` | | |
| trans | **1.453** | 1.482 |
| yaw | 0.557° | **0.254°** |
| **no lane geometry** `[2,3,4,5]` | | |
| trans | 14.239 | **2.000** |
| yaw | 30.909° | **1.201°** |
| **prior at `sigma_long_m=3.5`** | | |
| trans | 1.138 | **1.112** |

**The closed-form head does not win, and off distribution it loses badly.**
`ARCHITECTURE.md` said the gap would be small in distribution and not small
outside it. Half of that is right: the two are within 2% on translation in
distribution. The other half points the wrong way — outside distribution the
gap is a factor of seven, in the regression head's favour.

**Why.** RMSE is a tail statistic, and the two heads have completely different
tails. The Procrustes solve is a *rigid fit*: a handful of confident, wrong
correspondences drag it arbitrarily far, which is how one ablation reaches
30.9° of heading error. `RegressionPoseHead` is bounded by `tanh` against the
grid extent and simply cannot emit that answer — its worst case is 2.1° because
its worst case *cannot* be worse. Bounded and vague beats exact and occasionally
absurd, when the loss is squared.

**What survives of the argument for solving the geometry.** The parameter count
(none), the fact that it cannot overfit, and the interpretability — and note
that the interpretability is not actually traded away here, because `assign` is
computed and supervised either way. What does not survive is the accuracy claim.

**What this suggests next, and it is not "use the regression head".** The
Procrustes head's problem is outliers, and weighted Procrustes has a standard
answer to outliers that this implementation does not use: reweight, or gate the
assignment on matchability before solving rather than only scaling by it. The
honest experiment is a robust closed-form head against both of these, not a
retreat to the baseline. Until that is run, the table above stands as measured
and the README's "two ideas worth taking away" is one and a half.

## M1 — how far the prior can drift

Same checkpoint again, evaluated against priors wider than the one it trained
on. `sigma_long_m` rises; the truncation bound stays at its trained 4.5 m,
because that bound has to equal the cost volume's extent — a target outside the
grid has no correct cell, and `config._validate` refuses the configuration
rather than training it.

```bash
tools/eval.py runs/base/best.pt --split test data.sample.prior.sigma_long_m=3.5
```

| `sigma_long_m` | prior long | model long | share of the error removed |
|---|---|---|---|
| **1.5** *(as trained)* | 1.497 | 0.498 | 67% |
| 2.5 | 2.347 | 0.874 | 63% |
| 3.5 | 2.895 | 1.126 | 61% |
| 4.5 | 3.242 | 1.291 | 60% |

**There is no cliff.** Doubling the prior's spread costs six points of the
error it removes, not a collapse: the model is matching landmarks, and a
landmark 3 m away is the same landmark. A model that had learned the prior's
mean instead would come apart here, which is what makes this the cheapest
available check that it did not.

It is also not the experiment the roadmap asked for. "Swept past the grid
extent" needs the extent to move too, and that is a retrain rather than an
eval — the truncation bound and the grid are pinned to each other by
construction. What is measured here is the softer question: how the model
behaves as the prior fills the grid it was given. The harder one is still open.
