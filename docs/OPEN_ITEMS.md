# Open items

What is unfinished, unverified, or out of scope. A reader should not have to
infer any of it from an absence.

## The two that matter most

- **0.44% of frames are confidently, wildly wrong.** The median frame is
  calibrated (NEES/dof 0.78) and excluding the tail ANEES is 1.03, so the
  covariance is honest almost everywhere — but 39 frames of 8 880 sit above
  NEES/dof 10, one at 3 707. M3 feeds this to a Kalman filter, and those frames
  are what would break it. **This blocks closed loop.** The fix is to detect
  them, not to rescale: the match/volume disagreement separates them best
  (25.6% recall at 1% FPR) and is already computed and used for nothing.
- **The second refinement pass buys nothing measurable**: 0.309 against 0.308
  with `refine_iters=1` at evaluation, for 2× the step time. The model was
  trained with two passes, so this is not the same as training with one, but
  that experiment is cheap and has not been run.

- **No converged accuracy number describes the current code.** The pre-rebuild
  tables in [RESULTS.md](RESULTS.md) are marked superseded there. Since then a
  four-arm A/B at 2000 steps has ranked the mechanisms and turned the geometric
  bias off — not enough to say what any of them is worth at the 37 000 steps M0
  needed. One seed, one config, synthetic data.
- **The reported covariance is optimistic where the map aliases.** The surface is
  computed with correspondences fixed, making it the curvature of the fit rather
  than the ambiguity of the match: measured, it is steeper along track than
  across, and identical with and without the along-track landmarks.

## Not implemented

- **No real data.** Everything is procedurally generated. The numbers bound the
  backend — whether the architecture can match point sets and recover a pose —
  and say nothing about localizing a real vehicle.
- **No real perception.** Detections are cut from the map and corrupted, so both
  point sets are the same polylines with noise on top. A real detector produces
  different chunking, geometry that bends the wrong way at range, missing pieces
  and hallucinated topology, none of which is modelled. A real detector produces
  different chunking, geometry that bends the wrong way at range, missing pieces
  and hallucinated topology, none of which is modelled. M2 addresses this by
  running a pretrained mapper, not by training one.
- **No closed-loop evaluation.** The prior is drawn from a distribution rather
  than produced by the previous frame's output. M3.
- **No compression and no deployment.** M4 and M5 — the stated purpose of the
  project, so their absence is the largest gap in it. The static input shapes
  and the parameter-free head anticipate them; neither has been exported.
- **No map topology.** The synthetic world has none worth the name. It becomes
  an input at M2.
- **No prior covariance as an input.** Open loop it adds nothing, since the
  prior's spread is a constant the model can learn. It becomes real at M3.

## Out of scope

- **Perception is an input.** No detector is trained or run in the training
  loop, mirroring the classical repo. The detector's failure modes are assumed
  rather than measured.
- **Three degrees of freedom.** `(forward, left, yaw)`. Nothing constrains z,
  roll or pitch, and no metric reports them.
- **Two dimensions.** Landmark height is dropped at ingest, so an elevated pole
  and a painted mark at the same ground position are the same point. Correct for
  a BEV formulation; wrong for an image-plane one.

## Unverified

- **Hyperparameters are unswept.** Loss weights, learning rate, model width,
  match radius, IRLS scale, abstention threshold, refinement passes and
  history length are first choices with reasons.
- **The head rematch has not been run.** The robust solve exists because the
  plain one lost seven to one out of distribution. Until it is measured against
  both baselines, the argument in `pose_head.py` is a hypothesis with tests.
- **The dashed-stripe experiment has not been run.** The machinery is there and
  `data.sample.stripe_dashed=false` is the control.
- **The refinement pass has not been ablated alone.** `refine_iters=1` shares
  weights with a trained checkpoint and can be swept from `tools/eval.py`.
- **Calibration has never been reported on a trained model.** `Calibration` is
  correct against synthetic errors. Given the point about the surface above, the
  honest prior is "ANEES above one".

## The model uses the GPU poorly, by shape rather than by accident

About 6% of the A6000's dense bf16 throughput. The cause is `head_dim = 32` —
every attention matmul contracts over it, and that is far too small to keep a
tensor core fed. Both models have it: 128-dim over 4 heads, 256 over 8.

Two things are untried, and neither is batch size. **Widening the heads is not
one of them** — it was tried, and at a fixed `dim` of 128 both one and two heads
are *slower* end to end than four. The matmul saving is real and never reaches
the wall clock, which is the clearest evidence that this model is launch-bound
rather than arithmetic-bound. `RESULTS.md` has the sweep.
**`F.scaled_dot_product_attention`** in place of `nn.MultiheadAttention` with
explicit masks would stop materialising the attention matrix, cutting both time
and the memory that dominates this model. **CUDA graphs** are in M4's latency
protocol but not used in training.

## Hyperparameters with a diagnosis but no measurement

- **`grad_clip = 1.0` clips 100% of steps** at a median gradient norm of 6.0,
  so it is acting as the learning rate and the cosine schedule is largely
  cancelled. The repair is `grad_clip` near 10 with `lr` near 1e-4 — coupled,
  because raising the clip alone raises the effective rate sixfold. Unmeasured.
- **`ffn_mult = 2`** where transformers conventionally use 4. Narrow, and the
  FFN matmuls are the *well-shaped* ones on this hardware, so widening is
  cheaper in throughput than it looks. Untried.
- **`loss.w_cov = 0.1`** may be too weak: ANEES wanders 0.73 to 3.36 across a
  run rather than holding at one.
- **`refine_iters = 2` has not been ablated on its own**, unlike history. It
  shares weights with any trained checkpoint, so
  `tools/eval.py model.refine_iters=1` costs minutes.

## Rough edges

- **Element caps truncate.** 72 map elements, 32 detections per frame. Overflow
  drops the farthest — the right ordering, but not reported per frame.
- **Clutter has no class prior.** False positives draw a class uniformly, making
  a spurious stop line as likely as a spurious lane divider. A real detector's
  confusions are far more structured.
- **`float()` on every loss term each step** forces a GPU sync at the logging
  interval. Immaterial at 50-step logging, worth knowing before profiling.
- **The smoke run's mass gate refuses everything.** That is the gate working —
  an undertrained matcher has no evidence — but it leaves the trusted block
  empty, which reads as a failure and is not.
