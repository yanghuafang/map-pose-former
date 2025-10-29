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

## Stage 2 — match, and assign

`model/matcher.py`, `model/attention.py`.

### The block is an ordinary Transformer layer

This is where the project's name comes from, and it is worth saying plainly
because the class is called `MatcherLayer` rather than anything with
"transformer" in it. One round is the standard **pre-norm decoder layer**:

```python
x = x + self_attn(norm1(x), norm1(x))  # multi-head attention over its own set
x = x + cross_attn(norm2(x), norm2(y))  # the other set, normed by this layer
x = x + ffn(norm3(x))  # Linear -> GELU -> Linear, 2x expansion
```

Multi-head attention, a feed-forward network, LayerNorm before each, a residual
around each, stacked four deep — two such stacks, one per token set, so eight
blocks in all. About **93% of the parameters are these blocks** — 1.59 M of
1.709 M — so the Transformer is not a component of this model, it very nearly
is the model. Everything else is a thin encoder and a solver with no
parameters at all.

Two departures from `nn.TransformerEncoderLayer`, both forced:

- **The geometry has to get inside the score.** A relative-offset bias or a
  rotary rotation happens between the query and the key, and
  `nn.MultiheadAttention` has no way to express either.
- **Neither set is "the encoder".** Two token sets update *each other* every
  round — the map reads the detections while the detections read the map.
  `nn.Transformer` offers one direction, encoder into decoder.

At `dim` 128 the two heads are 64 wide each, and `head_dim` is stated rather
than derived so head *count* and head *width* can be swept apart. Measured,
4×64 is not separated from 2×64; 2×64 is kept as the cheapest cell nothing
separates from.

### What the two attentions are for

Rounds of **self-attention** for context (which of four parallel lines is this,
given the kerb to the left and the stop line ahead) and **cross-attention** for
matching. The score carries the relative geometry between tokens, because four
lane dividers in a row look alike by construction and what separates them is
where each sits relative to everything else.

Where that geometry enters decides what the network is invariant to, and all
three choices are implemented so the question can be measured:

| mode | the offset is measured in | invariant to |
|---|---|---|
| `relative` | the query element's own frame | any rigid motion of the scene |
| `rope` | the shared frame, LightGlue-style rotary | translation only |
| `absolute` | nothing; positions added to tokens | whatever it learns |

`rope` is what point tokens run. Its weaker invariance is a feature rather
than a compromise: the prior pins heading to within 3°, so map and detections
arrive nearly aligned, and "this detection points the way that map element
does" is real evidence that full equivariance throws away — worth 9.2
percentage points of recall at 7.6 standard deviations over `absolute`.
`relative` needs an N × N bias tensor, which over 1 344 tokens costs 2.4× the
step time for 2.5% of accuracy, so the rotary encoding is how relative
position gets in without one.

Then **LightGlue's partial assignment**: a dual softmax, so a pair has to be
each other's best match rather than merely a good one, gated by a matchability
score per token. Matchability is what lets a detection match *nothing* —
clutter, a false positive, a landmark outside the crop — instead of forcing its
mass onto whichever map element is least implausible.

At point resolution that product **is** the assignment the solve wants —
`(B, 768, 576)`, one weight per detected point and map point — so there is no
second stage, and with it goes the defect that a detection spanning four map
chunks cannot tell its own points apart.

### The scores are computed in fp32, and this decides whether it trains

Under bf16 autocast this model does not train at all, and the failure does not
look like a precision problem — it looks like a diverging model. Identical
seeds, schedule and data, differing only in autocast:

| step | bf16 largest logit / match loss | fp32 |
|---|---|---|
| 150 | 1 144 / 11.5 | 15.4 / 6.5 |
| 250 | 489 472 / 1 044 | 32.4 / 7.7 |
| 500 | 1.9e7 / 16 124 | ~35 / ~6 |

A score is a sum over `dim` of products which must then survive **two**
softmaxes, and bf16 carries eight mantissa bits. The pooled path has 96 × 72
of them and tolerates it; point resolution has 768 × 576 and does not — which
is why the failure is invisible until the tokenizer changes, and why it
presents as divergence rather than as rounding.

So the matmul and both softmaxes run in fp32 regardless of autocast, while the
projections feeding them stay in the autocast dtype: it is what happens *to*
the scores that needs the wider type, not producing them.

### The pooled alternative, and the second stage it needs

`tokens=element` pools each polyline's points into one token with VectorNet's
subgraph encoder — embed each point, pool the element, tell every point what
its element looks like, embed again, pool again — giving `(B, 168, 128)` and an
element assignment of `(B, 96, 72)`.

That assignment is not what the solve takes, so a second stage expands it, and
**that stage has no parameters**. Both elements already carry their points in
their own frame, so once the elements correspond their points correspond by
shape: a soft nearest neighbour in local coordinates, normalised *within* the
pair, because softmaxing over every map point at once would re-decide which
element matched — a decision the element stage already took, and taking it
twice divides the evidence by the number of candidates.

What the expansion cannot recover is the resolution pooling threw away: one
element assignment per detection says which chunk the *detection* is on, not
which chunk each of its points is on, so a road boundary spanning four chunks
has its points spread over all four and dragged toward their common centroid.
`point_assign=point` buys some of that back with a learned per-point query, and
the keys it queries are still pooled.

