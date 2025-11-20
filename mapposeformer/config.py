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
from mapposeformer.losses import LossParams
from mapposeformer.model.model import ModelParams


@dataclass(frozen=True)
class DistillParams:
    """What the student is asked to copy from the teacher, and how hard."""

    teacher: str = ""
    """Checkpoint to distil from. Empty is the ordinary supervised run, which
    is what every config here does unless it says otherwise."""
    w_match: float = 1.0
    """On the assignment, which is the only distribution there is to copy: it
    has an opinion per correspondence rather than per frame, which is the same
    reason the match loss carries in supervised training.

    A first choice rather than a measured one. On an untrained teacher the
    assignment KL is about 1e-3, because both models withhold most of their
    mass and agree about doing so, and a divergence between two near-empty
    rows is near zero. A trained teacher should peak; "should" is not a
    measurement, and what settles it is the gradient split reaching the
    trunk."""


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
    gnc_frac: float = 0.3
    """Fraction of training over which the robust scale anneals from
    ``model.robust_sigma_start_m`` down to ``model.robust_sigma_m``. Graduated
    non-convexity: a redescending weight rejects everything while the pose is
    still bad, so it has to be widened first and tightened as the pose earns
    it."""
    grad_clip: float = 10.0
    """Global gradient-norm clip, and it is a guard rather than a knob.

    At 1.0 it was not a guard. Measured over every logged training step of ten
    converged runs, it bound on **100%** of steps without fp32 match scores --
    median gradient norm 2.1 to 7.5 -- so the step size was set by the clip
    and not by the schedule. Computing the scores in fp32 cuts gradient
    magnitudes about fivefold and the clip then binds on 26-40%, which is
    better and still too much.

    What makes that a measurement problem rather than a tuning one is the
    *spread*. A model with more parameters has a systematically larger
    gradient norm, so it is clipped on a different fraction of its steps, and
    two runs meant to differ only in structure end up differing in effective
    learning rate. Across five token configurations:

    | clip | binds on | spread across runs |
    |---|---|---|
    | 1.0 | 25.8-40.1% | 14.4 pp |
    | 5.0 | 2.8-6.6% | 3.7 pp |
    | **10.0** | **2.2-4.1%** | **1.9 pp** |
    | 50.0 | 0.1-2.2% | 2.1 pp |

    10.0 is where the spread stops falling: it still catches the genuine
    spikes -- p99 is 26 to 132 and the maximum 729 -- while leaving the bulk of
    steps alone. Removing the clip entirely is not an option at that maximum.

    **This changes training.** A run at 1.0 and a run at 10.0 are not a
    controlled comparison, so every run records its clip in the `config.yaml`
    it writes, and `clip_binding` in the validation line reports what fraction
    of steps it actually bound -- the question never has to be reconstructed
    from logs."""

    patience: int = 0
    """Stop after this many evaluations with no improvement; 0 never stops.

    A fixed epoch count is wrong in both directions when models differ in
    capacity, and the range measured here differs by 16x -- `dim` 64 is
    0.433 M parameters against `dim` 256's 6.793 M. A 4-layer point-token run
    peaks at epoch 18-19 and is up to 29% *worse* by 40, while 2-layer element
    runs peak at 29-31. Pick 25 and the small models are truncated before
    their best; pick 40 and the large ones spend fifteen epochs getting worse.

    **Truncation is the dangerous half**, because it is invisible: a run cut
    off before its peak simply looks worse than one that reached its own, and
    the comparison silently measures the schedule instead of the structure.
    Early stopping gives each run exactly as long as it keeps improving, which
    is the only rule that treats unequal capacities equally.

    `select_on` decides what counts as improvement, so patience follows
    whatever metric the run is being judged by."""

    resume: bool = False
    """Continue from ``last.pt`` in ``out_dir`` if one is there.

    Off by default, because silently resuming is how you train a model whose
    history you cannot account for. On, it restores the optimiser moments, the
    step the schedules are functions of, and the RNG -- and it must not delete
    `metrics.jsonl`, which is otherwise cleared at construction so a re-run
    cannot interleave two histories."""

    select_on: str = "trans_rmse"
    """Which validation metric picks ``best.pt``, and whether lower wins.

    `trans_rmse` is the default and is deliberately **not** what this project
    says runs should be compared on: `RESULTS.md` asks for `long` and recall,
    because longitudinal error varies 17.7% between seeds where lateral varies
    0.67%. Selecting on `trans_rmse` and then publishing lateral, yaw, recall
    and NEES means those quiet metrics inherit trans's selection noise --
    measured across nine converged runs, the free improvement the selector
    hands a run ranges 0.005 to 0.019 m, which is larger than the effect some
    reported rows claim.

    It stays the default anyway, because changing it silently would make new
    checkpoints incomparable with every existing one. Set it deliberately --
    ``train.select_on=all/recall_0.25m_0.5deg`` -- and the run records the
    choice in its own `config.yaml`."""

    select_higher_is_better: bool = False
    """``True`` for a metric like recall, where the maximum is the best."""
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
    init_from: str = ""
    """Checkpoint to start from instead of random initialisation. The weights
    are loaded and the optimizer is not, because this is for fine-tuning a
    model that has been changed -- pruned, most of the time -- and not for
    resuming a run that stopped."""
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
    #: Empty `teacher` means no distillation, which is the usual case.
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

    def merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
        """Deep-merge ``src`` into ``dst``, one level at a time.

        A shallow ``update`` is wrong from three dots on:
        ``data.sample.map_radius_m=60`` arrives as ``{"data": {"sample":
        {"map_radius_m": 60}}}``, and assigning that ``sample`` over the
        file's would drop every other key the file put there -- silently,
        because the defaults that replace them are valid values.
        """
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = value

    merge(raw, overrides or {})
    cfg = with_overrides(Config(), raw)
    _validate(cfg)
    return cfg


def dump_config(cfg: Config) -> str:
    """Serialise a whole config back to the YAML that would reproduce it.

    Round-trips through ``load_config``, which is the only property that
    matters: a run directory carries this file so the run can be re-launched
    exactly, months later, without trusting a shell history.

    @param cfg The configuration to write out.

    @return YAML text with one top-level section per dataclass field.
    """
    import yaml

    return yaml.safe_dump(
        dataclasses.asdict(cfg), sort_keys=True, default_flow_style=False
    )


def _validate(cfg: Config) -> None:
    """Catch the cross-section mismatches that produce quiet, wrong training.

    The covariance fuses the prior's own uncertainty, so the model states what
    it believes that to be. If it disagrees with the prior the data actually
    draws, every covariance is wrong by a constant and the calibration
    statistics report it as a model failure -- which is a long way to chase a
    number that is written down in two places.
    """
    pp, mp = cfg.data.sample.prior, cfg.model
    pairs = (
        ("long", pp.sigma_long_m, mp.prior_sigma_long_m),
        ("lat", pp.sigma_lat_m, mp.prior_sigma_lat_m),
        ("yaw", pp.sigma_yaw_deg, mp.prior_sigma_yaw_deg),
    )
    for axis, drawn, believed in pairs:
        if abs(drawn - believed) > 1e-6:
            raise ValueError(
                f"the data draws a prior with {axis} sigma {drawn} but the "
                f"model is told {believed}; the covariance would be fused "
                f"with the wrong prior"
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
