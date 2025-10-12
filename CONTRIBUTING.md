# Contributing

This project is written to be **read**, and that is the acceptance criterion. A
change that makes the code faster and its reason harder to find has to be
argued for.

Three rules follow:

1. **Every non-obvious decision carries its reason** — not what the code does,
   but why it is this and not the obvious alternative, and what that cost.
2. **Claims are measured or they are marked.** [docs/RESULTS.md](docs/RESULTS.md)
   keeps tables that turned out to be wrong, with what replaced them.
3. **The prose and the code must agree.** `tests/test_docs.py` fails when
   `docs/ARCHITECTURE.md` states a shape the config no longer produces.

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
No C++ yet; it arrives with the TensorRT backend at M5. `.clang-format` and
`.clang-tidy` are already here, because that backend compiles into
camera-map-localization's tree and is read beside its sources.

**API documentation — Doxygen.** `@param`, `@return`, `@throws`, `@warning`
inside the docstring. `JAVADOC_AUTOBRIEF` is on, so the first sentence is the
brief and no `@brief` is written by hand:

```python
def relative(a: Tensor, b: Tensor) -> Tensor:
    """``a⁻¹ ∘ b``: where ``b`` is, as seen from ``a``.

    @param a ``(..., 3)`` the frame to look from.
    @param b ``(..., 3)`` the pose to describe.
    @return ``(..., 3)`` the transform between them.
    """
```

`./scripts/docs.sh` builds the reference into `build/doxygen/html`.

## Tensor conventions

Shapes use the einsum indices the code uses, so docstring and implementation
read alike: `B` batch, `K` detection points, `L` map points, `G` pose
hypotheses, `D` feature dimension.

A pose is always `(x, y, yaw)` in the vehicle convention — X forward, Y left,
yaw counter-clockwise, metres and radians. Never a 4×4.

## Changes

**One commit per independently-verifiable state.** The history is part of what
this project teaches, so `git log` should read as a sequence of claims a reader
can check. Each commit is a self-consistent state — code, configs, docs and
measured numbers agree within it — and its test suite passes on its own tree.

That is *not* one commit per milestone, which was the rule here first and was
wrong. A milestone splits wherever its parts land at different times and can be
measured apart: M1 is one commit per mechanism, because its own A/B turns them
on and off one at a time; M2 splits into the ingest, real detections and map
topology, which fail for different reasons.

**The test is the ablation.** If a change can be switched off and measured
independently, it can be committed independently — and should be, because a
reader who can see it alone in the history can see what it cost.

**The first commit is the exception.** A working system arrives whole: its
parts are scaffolding until the last one lands, and there is no ablation to run
against half a localizer. Everything after it splits.

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
   the model or the data.

## Where to start reading

`mapposeformer/model/model.py`. Its docstring names the five parts and links the
argument for each. Then [docs/README.md](docs/README.md) in the order it lists.
