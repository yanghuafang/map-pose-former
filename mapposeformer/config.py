"""Configuration: dataclasses first, YAML second.

The dataclasses in the other modules *are* the schema -- their defaults are the
documented values and their docstrings are the documentation. A YAML file only
overrides them, and an override that names a field which does not exist is an
error rather than a silently ignored line, because the alternative is spending
an afternoon on a typo that changed nothing.

There is no configuration framework here on purpose. A learner should be able
to follow a number from the command line to the tensor it scales without
leaving the repository.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mapposeformer.data.dataset import DataParams
from mapposeformer.distill import DistillParams
from mapposeformer.losses import LossParams
from mapposeformer.model.model import ModelParams


@dataclass
class TrainParams:
    """Optimizer, schedule, and where the run goes."""

    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.03
    """Fraction of total steps spent warming up. A transformer trained from a
    cold LayerNorm with no warmup diverges within a few dozen steps."""
    grad_clip: float = 1.0
    num_workers: int = 8
    amp: str = "bf16"
    """``bf16``, ``fp16`` or ``off``. bf16 on Ampere and later: it has the same
    exponent range as fp32, so it needs no loss scaler and the Procrustes head's
    atan2 cannot underflow the way it can in fp16."""
    seed: int = 0
    out_dir: str = "runs/default"
    eval_every: int = 1
    log_every: int = 50
    device: str = "cuda"
    compile: bool = False
    """``torch.compile``. Off by default because it costs a minute of warmup
    and hides the first error behind a graph break."""
    max_steps: int = 0
    """Stop after this many optimizer steps; 0 means run the full schedule.
    The smoke test uses it."""
    accum_steps: int = 1
    """Micro-batches accumulated into one optimizer step.

    The effective batch is ``batch_size * accum_steps`` at the memory cost of
    ``batch_size`` alone, so a smaller card costs wall clock rather than
    gradient quality. Neither default config needs it; it exists so a reader
    on a 12 GiB card can halve the batch, set this to 2, and get the same
    optimizer trajectory."""


@dataclass
class Config:
    data: DataParams = field(default_factory=DataParams)
    model: ModelParams = field(default_factory=ModelParams)
    loss: LossParams = field(default_factory=LossParams)
    train: TrainParams = field(default_factory=TrainParams)
    distill: DistillParams = field(default_factory=DistillParams)


def upgrade(obj: Any) -> Any:
    """@brief Give a config unpickled from an old checkpoint its newer fields.

    A checkpoint stores the ``Config`` that produced it, which is what makes a
    run reproducible -- and what makes every stored config a hostage to the
    next field added to the dataclass. Unpickling restores the attributes that
    existed when it was written and no others, so a config saved before
    ``distill`` existed has no ``distill``, and the first
    ``dataclasses.replace`` raises ``AttributeError`` on a field nobody asked
    about.

    So missing fields are filled from their defaults, recursively. A default is
    the right answer here by construction: the run predates the field, so it
    cannot have depended on it, and the default is what "not configured" means
    everywhere else.

    @param obj A dataclass instance, possibly missing fields.
    @return The same object, mutated in place, for convenience.
    """
    if not dataclasses.is_dataclass(obj):
        return obj
    for f in dataclasses.fields(obj):
        if not hasattr(obj, f.name):
            if f.default is not dataclasses.MISSING:
                value = f.default
            elif f.default_factory is not dataclasses.MISSING:
                value = f.default_factory()
            else:
                continue
            # object.__setattr__, because most of these dataclasses are frozen.
            object.__setattr__(obj, f.name, value)
        else:
            upgrade(getattr(obj, f.name))
    return obj


def with_overrides(obj: Any, values: dict[str, Any]) -> Any:
    """Return a copy of a (possibly nested) dataclass with fields replaced.

    A copy rather than a mutation: most of the parameter dataclasses are frozen,
    which is what stops a training loop from quietly editing the configuration
    it is later going to save next to the weights.

    An override naming a field that does not exist raises. Silently ignoring it
    is how an afternoon gets spent on a typo that changed nothing.
    """
    kwargs: dict[str, Any] = {}
    known = {f.name for f in dataclasses.fields(obj)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(
                f"{type(obj).__name__} has no field {key!r}; "
                f"known: {', '.join(sorted(known))}"
            )
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            kwargs[key] = with_overrides(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return dataclasses.replace(obj, **kwargs)


def load_config(
    path: str | Path | None, overrides: dict[str, Any] | None = None
) -> Config:
    """Read a YAML config, apply CLI overrides, and validate the result.

    @param path YAML file with top-level ``data``, ``model``, ``loss``,
        ``train`` sections, any of which may be omitted.
    @param overrides Already-parsed dotted overrides, e.g. ``{"train": {"lr":
        1e-3}}``, applied on top of the file.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        # Deferred: the smoke path and the tests need no config file.
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
    for section, values in (overrides or {}).items():
        raw.setdefault(section, {}).update(values)
    cfg = with_overrides(Config(), raw)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Catch the cross-section mismatches that produce quiet, wrong training."""
    sp, mp, gp = cfg.data.sample, cfg.model, cfg.model.grid
    shape = (
        sp.max_map_elements,
        sp.max_det_elements,
        sp.points_per_element,
        sp.history,
    )
    want = (
        mp.max_map_elements,
        mp.max_det_elements,
        mp.points_per_element,
        mp.history,
    )
    if shape != want:
        raise ValueError(
            "data.sample and model disagree on input shape "
            "(map/det/points/history): "
            f"{'/'.join(map(str, shape))} vs {'/'.join(map(str, want))}"
        )
    if mp.refine_iters < 1:
        raise ValueError(
            f"model.refine_iters must be at least 1, got {mp.refine_iters}"
        )
    pp = sp.prior
    # A prior error outside the grid has no correct cell, so the volume loss
    # would be asking the head for something it cannot represent -- and the
    # resulting soft target quietly piles up on the boundary instead.
    bounds = (
        (pp.max_long_m, gp.extent_x_m),
        (pp.max_lat_m, gp.extent_y_m),
        (pp.max_yaw_deg, gp.extent_yaw_deg),
    )
    for axis, (prior_max, grid_extent) in zip("xy\u03b8", bounds, strict=True):
        if prior_max > grid_extent:
            raise ValueError(
                f"prior truncation on {axis} ({prior_max}) exceeds the grid "
                f"extent ({grid_extent}); that target has no cell"
            )


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``section.field=value`` strings from the command line.

    Values go through YAML scalar rules, so ``3e-4`` is a float, ``true`` is a
    bool, and ``[0,1,4]`` is a list.
    """
    import yaml

    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"override {item!r} is not of the form section.field=value"
            )
        key, _, value = item.partition("=")
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out
