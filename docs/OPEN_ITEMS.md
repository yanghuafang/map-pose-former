# Open items

What is unfinished, unverified, or out of scope. A reader should not have to
infer any of it from an absence.

## What blocks the next milestone

- **The observability ablation does not reproduce on nuScenes.** Removing both
  along-track classes costs 0.907 m to 0.945 m longitudinal, against 0.322 m to
  1.367 m for the same experiment on generated scenes. The two informative rows
  disagree: a drivable-area outline alone recovers 5% of along-track error, as
  predicted, while lane dividers alone are the best longitudinal arm. nuScenes'
  lane dividers are 18 m and end at intersections, and M2a cannot say whether
  those ends belong to the road or to nuScenes' segmentation of it, because its
  detections are cut from the map. **M2b is the experiment**, not a refinement
  of this one.

- **A quarter of a percent of frames are confidently, wildly wrong.** The median
  frame is calibrated (NEES/dof 0.77) and excluding the tail the mean is 1.02,
  so the covariance is honest almost everywhere — but 21 frames of 8 880 sit
  above NEES/dof 10, the worst at 1 104. **No single-frame detector reaches
  them.** Assignment mass separates them best, at AUC 0.77, which is not enough
  to act on. **It no longer blocks anything**: closed loop the tail over the
  same split is zero. The open-loop tail is real and still undetected; what
  changed is that nothing downstream needs it detected.
- **One refinement pass beats two, and the default still says two.** Trained
  with one, test translation is 0.221 against 0.336 at 1.9× the throughput, and
  calibration and closed loop both improve as well — [RESULTS.md](RESULTS.md)
  has the pair. The default is unmoved because changing it means retraining the
  teacher, the student and the pruned variants, and this is one seed. Recorded
  alongside it is the trap: a checkpoint trained for two passes must be
  evaluated at two, and is 36× worse in the loop at one.
- **The reported covariance is optimistic where the map aliases.** The surface is
  computed with correspondences fixed, making it the curvature of the fit rather
  than the ambiguity of the match: measured, it is steeper along track than
  across, and identical with and without the along-track landmarks.

## Not implemented

- **One real dataset, one split, one seed.** M2a reads nuScenes; Argoverse 2 is
  the untrained-on generalization test and has not been touched. Every number
  here is a single run of a single configuration.
- **Closed loop runs on simulated odometry.** The filter integrates the same
  drift model the samples carry, 1% of distance travelled, and frames sit 8 m
  apart. Both are favourable: real odometry is biased rather than merely noisy,
  and a longer gap between corrections gives the estimate more room to drift.
  The closed-loop numbers should be read as what the loop does when its inputs
  behave, which is the question this milestone asked.
- **No real perception.** Detections are cut from the map and corrupted, so both
  point sets are the same polylines with noise on top. This is as true of the
  nuScenes path as of the synthetic one: M2a made the *map* real and left
  perception an assumption. A real detector produces
  different chunking, geometry that bends the wrong way at range, missing pieces
  and hallucinated topology, none of which is modelled. M2b addresses this by
  running a pretrained mapper, not by training one.
- **Closed loop is synthetic and single-backend.** M3 runs the loop, but only
  on generated scenes and only through the Procrustes head. nuScenes has never
  been driven through a filter, and the regression backend has not been run
  against the same one — which is the comparison that would say whether its
  off-distribution collapse matters when the prior is always good.
- **Deployment stops at the ONNX.** Compression and the runtime are measured
  (M4), but nothing here runs inside the system it was written for: the C++
  TensorRT backend that would plug into camera-map-localization, so that one
  filter and one metric score both, is unwritten.
- **No map topology.** The synthetic world has none worth the name. nuScenes
  is the first map here with one; it becomes an input at M2c.
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
- **The head rematch settled nothing.** The robust solve wins heading by a
  factor of two and still loses translation, in and out of distribution
  ([RESULTS.md](RESULTS.md)). The parameter-free head is kept for rotation
  accuracy, interpretability and quantization behaviour, not for RMSE.
- **`refine_iters=2` has not been ablated in *training*.** Disabling the second
  pass at evaluation *improves* accuracy — 0.316 against 0.336 — but the model
  was trained with two, so that is not the same experiment. One run.

## The model uses the GPU poorly, by shape rather than by accident

About 6% of the A6000's dense bf16 throughput. The cause is `head_dim = 32`
in the student — every attention matmul contracts over it, and that is far too
small to keep a tensor core fed. The teacher already avoids it: 256-dim over the
same 4 heads is `head_dim = 64`.

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

## Rough edges

- **The FP16 engine is unmeasured.** The export works — `exact_arithmetic`'s
  fp32 island needed casts at its edges, not the builder — but no FP16 engine
  has been built or timed, so the row is still empty. Accuracy at fp16 is
  measured and free; only the latency is missing.
- **INT8 through TensorRT is unbuilt.** TensorRT 11 removed
  `BuilderFlag.INT8` and the calibrator classes; precision now comes from
  quantize/dequantize nodes in the graph, which `nvidia-modelopt` inserts. The
  simulated INT8 in `quantize.py` prices the accuracy — unchanged at 8 bits —
  and says nothing about the speed.

- **`nn.MultiheadAttention` blocks two of M4's three stages.** It holds 49% of
  the student's parameters and neither pruning nor quantization can reach them.
  Heads cannot be pruned because the module requires its internal projection
  width to equal `embed_dim`, and dropping a head makes the two differ. Its
  weights cannot be quantized because the input projection is a raw parameter
  rather than a child module, and the output projection is a `Linear` subclass
  whose `.weight` the parent reads directly, so wrapping it breaks the forward.
  So structured pruning reaches the feed-forwards only, and INT8 shrinks the
  weights by a third rather than three quarters. Replacing it with an explicit
  `scaled_dot_product_attention` would unblock both *and* stop materialising
  the attention matrix, which is wanted below for its own sake. This is now the
  single highest-value change in the file.
- **Pruning scores by weight magnitude**, which ignores what the activations
  do. A Taylor or activation-aware criterion is the obvious next thing to try,
  and the prune/fine-tune/re-measure cycle is what would say whether it pays.

- **One token per point, not per element.** 96 detection elements and 72 map
  elements become 1 346 tokens, and attention is quadratic in that: 58 M scores
  per sample, 6.9 GiB at batch 64, against 0.16 GiB for the token features they
  come from. That ratio, not the parameter count, is what this model spends.
- **Element caps truncate.** 72 map elements, 32 detections per frame. Overflow
  drops the farthest — the right ordering, but not reported per frame.
- **Clutter has no *structured* confusion model.** False positives now draw
  from what the frame detected, with multiplicity, so contamination is roughly
  equal across classes instead of falling on the rare ones — but the draw is
  still independent of what the class is. A real detector confuses a road
  boundary for a lane divider far more often than for a traffic light.
- **`float()` on every loss term each step** forces a GPU sync at the logging
  interval. Immaterial at 50-step logging, worth knowing before profiling.
- **The smoke run's mass gate refuses everything.** That is the gate working —
  an undertrained matcher has no evidence — but it leaves the trusted block
  empty, which reads as a failure and is not.
