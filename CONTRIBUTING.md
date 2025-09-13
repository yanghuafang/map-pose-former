# Contributing

This project is written to be **read**, and that is the acceptance criterion. A
change that makes the code faster and its reason harder to find has to be
argued for.

Three rules follow:

1. **Every non-obvious decision carries its reason** — not what the code does,
   but why it is this and not the obvious alternative, and what that cost.
2. **Claims are measured or they are marked.** `docs/RESULTS.md` carries every
   measured number, the command that reproduces it, and the negative results in
   full.
3. **The prose and the code must agree.** A document that states a shape gets a
   test that fails when the config stops producing it.

## Setup

```bash
git clone git@github.com:yanghuafang/map-pose-former.git
cd map-pose-former
./scripts/setup.sh          # conda env + CPU torch, --cuda on a GPU box
./scripts/ci.sh             # format, lint, tests
```

Everything but CUDA training and the TensorRT export runs on macOS and Linux
alike; that split is why `scripts/remote-ubuntu.sh` exists.

## Style

**Python — the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).**
`ruff` enforces the mechanical half: 80 columns, import grouping, naming, no
lambda bound to a name. `ruff format .` fixes most of what it finds.

**C++ — the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).**
No C++ yet. The deployment path here is Python -- `tools/export.py` writes the
ONNX and `mapposeformer/tensorrt.py` builds the engine -- and if a compiled
backend ever arrives it belongs in camera-map-localization's tree, beside its
sources and under its own configuration.

**API documentation — Doxygen.** `@param`, `@return`, `@throws`, `@warning`
inside the docstring:

```python
def transform_points(pose: Tensor, pts: Tensor) -> Tensor:
    """Move points from the frame ``pose`` describes into ``pose``'s parent.

    @param pose ``(..., 3)``.
    @param pts ``(..., N, 2)``, in the child frame.

    @return ``(..., N, 2)`` in the parent frame.
    """
```

The Doxygen config is not in the tree yet, so `scripts/docs.sh` does not run;
the docstrings are written for it.

## Writing

The code style above is about correctness. This is about whether anyone learns
anything, which is the point of the repository.

**Lead with the question, then the answer.** A reader should know what a
document is for within two sentences, and meet its headline result before its
method. `README.md` opens with "where is the car" and reaches the numbers before
the tensor contract; a document that makes a reader earn the result by reading
the derivation first has the order backwards.

**Every number carries the protocol that produced it.** Split, frame count,
seeds, hardware. "0.091 m" is a claim; "0.091 m over 120 test sequences, 8 880
frames" is a measurement. A number without a protocol cannot be checked and so
cannot be wrong, which is the problem with it.

**State the baseline beside the result.** 0.271 m means nothing until you know
the prior it corrects started at 1.611 m. Every results table here has a "do
nothing" row for that reason.

**Bold the claim, not the noun.** Bold marks the sentence a skimmer should read
in full, so a paragraph gets at most one. Bolding terminology teaches nothing.

**Tables answer "which", prose answers "why".** Put comparisons in a table and
reasoning underneath it. A table with a paragraph in a cell is neither.

**Negative results are content, not apology.** What failed, by how much, and
what it cost to find out. The most useful entries here are the experiments that
said no — an underpowered sweep, an uncontrolled comparison, a design argument
that lost to its own measurement. Write them as findings, not confessions.

**Say what is not known, where the claim is made.** A limit in a closing section
is a limit a reader meets after deciding. Everything here is measured on
generated data, and each headline says so.

**Be brief.** Cut anything that would not change what a reader does. Length is
not thoroughness; it is usually a draft that was never edited.

**One project, stated directly.** No version archaeology — no "the earlier
version", no branch names, no narrating a repair. A decision is written as
though it were always so, and the measurement that settled it is given.

**Punctuation.** Markdown prose uses the Unicode em dash; Python and shell use
`--` and never an em dash, so a diff never turns on how a dash was encoded.
Mathematical notation in docstrings — `Σ`, `⁻¹`, `×` — is deliberate, and
`ruff`'s `RUF001` and `RUF002` are off for it. Backtick every identifier, path
and flag.

## Tensor conventions

Shapes use the einsum indices the code uses, so docstring and implementation
read alike: `B` batch, `D` detections, `M` map elements, `P` points per element,
`N` either set where the code is generic, `Nq`/`Nk` query and key tokens in
attention, `dim` feature width. Once `Matcher.points` spreads an element match
over its points the two axes carry `D*P` and `M*P` points, still written `D`
and `M`.

A pose is always `(x, y, yaw)` in the vehicle convention — X forward, Y left,
yaw counter-clockwise, metres and radians. Never a 4×4.

## Changes

**One commit per independently-verifiable state.** The history is part of what
this project teaches, so `git log` should read as a sequence of claims a reader
can check. Each commit is a self-consistent state — code, configs, docs and
measured numbers agree within it — and its test suite passes on its own tree.

That is *not* one commit per milestone. A milestone splits wherever its parts
land at different times and can be measured apart: M2 splits into the encoder,
the attention, the matcher and the solve, because each is separately testable;
M5 splits into the map ingest and the real detector, which fail for different
reasons.

**The test is the ablation.** If a change can be switched off and measured
independently, it can be committed independently — and should be, because a
reader who can see it alone in the history can see what it cost.

**Nothing arrives whole.** Even the first commit is only the rules and the
scripts the tree is read under; the localizer lands a testable part at a time.

The corollary: a follow-up commit that fixes documentation means the earlier
commit was wrong. Fix it in place. The same goes for a performance fix that
exists only to make a milestone work — it belongs to that milestone, not beside
it.

1. **Branch** from `main`.
2. **Scope** — one argument per change. A refactor and a behaviour change in one
   diff cannot be reviewed, only accepted.
3. **Tests** — new behaviour needs a test that fails without it. The suite runs
   in seconds on CPU; keep it that way.
4. **Docs** — if you change a shape, a default or a claim, change the document
   that states it.
5. **Numbers** — a table needs the command that reproduces it. If a claim is not
   measured yet, say so where a reader will look, not in the commit message.
6. **Gate** — `./scripts/ci.sh` must pass; `--smoke` before anything touching
   the data.

## Where to start reading

[docs/ROADMAP.md](docs/ROADMAP.md) for what is being built and why, then
[docs/README.md](docs/README.md) in the order it lists. The reading order
through the code arrives with the code.
