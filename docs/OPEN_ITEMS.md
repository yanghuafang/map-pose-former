# Open items

What is unfinished, unverified, or deliberately out of scope. Honesty about
scope is part of the point; a reader should not have to infer any of this from
an absence.

## Not implemented

- **No real data.** Everything to date is procedurally generated. The numbers
  bound the *backend* — whether the architecture can match point sets and
  recover a pose — and say nothing about localizing a real vehicle.
- **No temporal fusion.** Frames are independent. The classical repo accumulates
  evidence over a 70-frame window and gains substantially from it. M2.
- **No closed-loop evaluation.** Open loop only: the prior is drawn from a
  distribution rather than produced by the previous frame's output. Open-loop
  numbers always look better than the system is. M4.
- **No compression and no deployment.** M5 and M6. The input shapes are already
  static and the pose head is already parameter-free, both in anticipation.

## Deliberately out of scope

- **Perception is an input.** No detector is trained or run. Detections come
  from the map with a statistical error model applied. This mirrors the
  classical repo, which makes the same choice for the same reason, and it means
  the detector's failure modes are *assumed* rather than measured.
- **Three degrees of freedom.** `(forward, left, yaw)`. Nothing here constrains
  z, roll or pitch, and no metric reports them.
- **Two dimensions.** Landmark height is dropped at ingest. An elevated pole and
  a painted mark at the same ground position are the same point to this model.
  For a BEV formulation that is correct; it would not be for an image-plane one.

## Unverified

- **Hyperparameters are unswept.** Loss weights, learning rate, model width and
  match radius are first choices with reasons, not tuned values.

## Known rough edges

- **Element caps truncate.** 56 map elements and 32 detections. Overflow drops
  the farthest, which is the right ordering, but it is still a cap and it is not
  reported per frame.
- **Clutter has no class prior.** False-positive detections draw a class
  uniformly, which makes a spurious stop line as likely as a spurious lane
  divider. A real detector's confusions are far more structured.
- **`float()` on every loss term each step** forces a GPU sync at the logging
  interval. Immaterial at 50-step logging, worth knowing before profiling.
