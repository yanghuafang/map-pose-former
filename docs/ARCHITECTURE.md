# Architecture

How the network is put together, and which part of it learns. `ROADMAP.md`
argues for these choices; this describes the code that implements them.

```
  map points ─────── encode ─┐
                             ├── match ── assign ── solve ── delta
  detection points ─ encode ─┘   4 x Transformer      │
  this frame + 2 warped here     (self + cross)       └── mass, covariance
```

Three stages, and **the last one has no parameters**. Everything the network
learns, it learns about association — which detection is which map element.
Given correct correspondences the pose was never in question: it is the
minimiser of a stated least-squares objective, and that has a formula.

| | parameters |
|---|---|
| map encoder | 0.051 M |
| detection encoder | 0.052 M |
| matcher | 1.607 M |
| **pose solve** | **0** |
| total at `dim` 128, 4 layers | 1.709 M |

## What flows through it

A sample arrives as fixed-shape tensors (`DATASET.md` has the full table).
History is width, not depth: the previous two frames' detections are warped
into the current ego frame by the odometry that measured the motion and joined
to this frame's as more elements.

| | shape | |
|---|---|---|
| map points in | `(B, 72, 8, 2)` | 72 elements, 8 points each |
| detection points in | `(B, 96, 8, 2)` | 32 this frame + 2 × 32 warped history |
| point tokens | `(B, 1344, 128)` | one per point, both sides |
| assignment | `(B, 768, 576)` | how much each detected point belongs to each map point |
| **output** | `(B, 3)` + `(B,)` | the correction, and the evidence behind it |

## Stage 1 — one token per point

`model/encoder.py`. Each point is embedded on its own: where it sits in its
element's frame, what class the element is, and for a detection what the
detector claimed about it. Each point also carries a frame — its own position,
with its element's heading — which the rotary encoding only ever reads as a
difference.

A lane divider is eight points and so is the next one over, so the obvious
saving is to pool. Attention over *points* is 1 344 tokens and a score matrix
quadratic in that; attention over *elements* is 168, which is 64× less. **That
saving is real and it is not the constraint**: the assignment the solve takes
is the same size either way, the model is loader-bound, and pooled tokens ran
10% *slower*. What pooling costs is resolution — **83.1% recall against
95.9%**, because eight points collapsed into one vector cannot say which of
four 12 m map chunks a given point of a detected road boundary lies on.
`tokens` keeps both, so they run as arms differing by exactly one thing; the
pooled path is described under *Stage 2*.

**No absolute coordinate enters a token.** Each element carries its own frame —
centroid, and heading from its first valid point to its last — and its points
are encoded in *that* frame. Move the whole scene rigidly and every token is
identical; only the frames move. `tests/test_encoder.py` asserts this as an
equality rather than encouraging it with augmentation.

Keeping the frame *beside* the token rather than inside it is what leaves the
equivariance question open for the matcher to answer, which is where it
belongs. A point landmark has no direction — a pole is one point — so
`oriented` says so instead of inventing a heading the matcher would then trust.

