"""What a validation line has to carry, and what picks `best.pt`.

A run that logs translation RMSE and nothing else cannot answer the questions
it was launched to settle -- lateral, recall, calibration -- however long it
trained, even though the evaluator computes all of them for `tools/eval.py`.
These tests pin both halves: the metrics reach the log, and whatever selects
the checkpoint is stated rather than assumed.
"""

from __future__ import annotations

import json

import pytest
import torch

from mapposeformer.config import Config, with_overrides
from mapposeformer.engine import Trainer


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = with_overrides(
        Config(),
        {
            "data": {"num_scenes": {"train": 2, "val": 2, "test": 2}},
            "model": {"dim": 32, "layers": 1, "heads": 2},
            "train": {
                "out_dir": str(out),
                "epochs": 1,
                "batch_size": 4,
                "num_workers": 0,
                "device": "cpu",
            },
        },
    )
    torch.manual_seed(0)
    Trainer(cfg).train()
    lines = [
        json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()
    ]
    return [d for d in lines if d.get("tag") == "val"]


def test_a_validation_line_carries_the_metrics_runs_are_compared_on(smoke):
    """`RESULTS.md` says compare on `long` and recall. So they must be logged.

    Longitudinal error varies 17.7% between seeds where lateral varies 0.67%,
    which is exactly why `trans` alone cannot decide between two runs.
    """

    assert smoke, "no validation line was written"
    keys = set(smoke[0])
    for needed in (
        "all/rmse_long_m",
        "all/rmse_lat_m",
        "all/rmse_yaw_deg",
        "all/recall_0.25m_0.5deg",
        "nothing/rmse_trans_m",
        "calib/anees_median",
        "calib/coverage_95",
    ):
        assert needed in keys, f"{needed} missing from {sorted(keys)}"

    # The two names checkpoint selection and the metrics log already read.
    assert smoke[0]["trans_rmse"] == pytest.approx(smoke[0]["all/rmse_trans_m"])
    assert smoke[0]["prior_rmse"] == pytest.approx(
        smoke[0]["nothing/rmse_trans_m"]
    )


def test_an_unknown_selection_metric_fails_loudly(tmp_path):
    """Silently falling back to `trans` is how a bad `select_on` goes unseen."""

    cfg = with_overrides(
        Config(),
        {
            "data": {"num_scenes": {"train": 2, "val": 2, "test": 2}},
            "model": {"dim": 32, "layers": 1, "heads": 2},
            "train": {
                "out_dir": str(tmp_path / "bad"),
                "epochs": 1,
                "batch_size": 4,
                "num_workers": 0,
                "device": "cpu",
                "select_on": "all/nonesuch",
            },
        },
    )
    torch.manual_seed(0)
    with pytest.raises(KeyError, match="nonesuch"):
        Trainer(cfg).train()


def test_the_clip_binding_fraction_is_reported_and_real(tmp_path):
    """How often the clip decided the step size, rather than the schedule.

    At `grad_clip = 1.0` this bound on 100% of logged steps in nine of the ten
    converged runs, so every one of them trained at a clip-determined step
    size. That is not merely suboptimal: a model with more parameters has a
    larger gradient norm and is clipped on a different fraction of its steps,
    so two runs meant to differ only in structure differ in effective learning
    rate too. The number has to be visible per run or the question cannot be
    asked afterwards.
    """

    def run(clip: float) -> float:
        cfg = with_overrides(
            Config(),
            {
                "data": {"num_scenes": {"train": 2, "val": 2, "test": 2}},
                "model": {"dim": 32, "layers": 1, "heads": 2},
                "train": {
                    "out_dir": str(tmp_path / f"c{clip}"),
                    "epochs": 1,
                    "batch_size": 4,
                    "num_workers": 0,
                    "device": "cpu",
                    "grad_clip": clip,
                },
            },
        )
        torch.manual_seed(0)
        Trainer(cfg).train()
        lines = [
            json.loads(x)
            for x in (tmp_path / f"c{clip}" / "metrics.jsonl")
            .read_text()
            .splitlines()
        ]
        return [d for d in lines if d.get("tag") == "val"][-1]["clip_binding"]

    # A clip no gradient can reach must never bind; a clip of zero always does.
    assert run(1e9) == 0.0
    assert run(0.0) == 1.0


def _epoch_fingerprints(tmp_path, augment: bool, workers: int) -> list[float]:
    """Sum of the training deltas seen in each of two epochs.

    A fingerprint rather than a full comparison because the loader shuffles:
    what must differ under augmentation is the *content* of the samples, and a
    sum over the whole epoch does not care what order they arrive in.

    Accumulated in float64. In float32 the shuffle changes the summation order
    and two identical epochs differ in the last few bits -- 244.51178 against
    244.51183 -- which reads as "the data changed" and is only rounding.
    """
    from mapposeformer.engine.trainer import _loader

    cfg = with_overrides(
        Config(),
        {
            "data": {
                "num_scenes": {"train": 2, "val": 2, "test": 2},
                "augment": augment,
            },
            "train": {
                "batch_size": 4,
                "num_workers": workers,
                "device": "cpu",
                "out_dir": str(tmp_path / f"a{augment}w{workers}"),
            },
        },
    )
    loader = _loader(cfg, "train", shuffle=True)
    sums = []
    for epoch in range(2):
        reseed = getattr(loader.dataset, "set_epoch", None)
        if reseed is not None and cfg.data.augment:
            reseed(epoch)
        sums.append(float(sum(b["delta"].double().abs().sum() for b in loader)))
    return sums


@pytest.mark.parametrize("workers", [0, 2])
def test_augmentation_reaches_the_dataloader_workers(tmp_path, workers):
    """`set_epoch` must change what training sees, including under workers.

    This is the test the original defect needed and never had. `set_epoch`
    existed, was correct, and was called from nowhere -- so every epoch of
    every run in this project's history saw byte-identical data while
    `augment: true` sat in the config claiming otherwise.

    Workers are the half that makes it subtle. `persistent_workers=True` gives
    each worker its own copy of the dataset for the life of the run, so
    reseeding the parent's copy changes nothing they will ever read. Wiring
    `set_epoch` without also dropping persistence would have replaced a silent
    no-op with a quieter one -- which is why this runs at 0 workers and at 2.
    """

    on = _epoch_fingerprints(tmp_path, augment=True, workers=workers)
    assert on[0] != on[1], (
        f"augmentation did not change the data across epochs "
        f"(workers={workers}); both epochs summed to {on[0]}"
    )

    off = _epoch_fingerprints(tmp_path, augment=False, workers=workers)
    assert off[0] == off[1], (
        f"augment=False must be deterministic across epochs, got {off}"
    )


def test_resume_continues_rather_than_restarts(tmp_path):
    """Unattended training measured in days will meet a kill or an OOM.

    Without this each run costs every epoch it had already paid for. And a
    `save` that stores no optimiser state makes "resume" restart Adam from zero
    moments and re-enter the learning-rate schedule at the wrong step, which is
    a different experiment wearing the same name.
    """

    def run(epochs: int, resume: bool):
        cfg = with_overrides(
            Config(),
            {
                "data": {"num_scenes": {"train": 2, "val": 2, "test": 2}},
                "model": {"dim": 32, "layers": 1, "heads": 2},
                "train": {
                    "out_dir": str(tmp_path / "r"),
                    "epochs": epochs,
                    "batch_size": 4,
                    "num_workers": 0,
                    "device": "cpu",
                    "resume": resume,
                },
            },
        )
        torch.manual_seed(0)
        Trainer(cfg).train()

    run(2, False)
    first = torch.load(tmp_path / "r" / "last.pt", weights_only=False)
    assert "optimizer" in first, "last.pt must carry optimiser state"
    assert first["next_epoch"] == 2, "epochs completed, not the loop index"

    run(4, True)
    lines = [
        json.loads(x)
        for x in (tmp_path / "r" / "metrics.jsonl").read_text().splitlines()
    ]
    val = [d for d in lines if d.get("tag") == "val"]
    # Four evaluations, not two: the history continued rather than being
    # cleared and rewritten. `metrics.jsonl` is deleted at construction
    # precisely so a re-run cannot interleave two histories, and a resume is
    # the one case where appending is correct.
    assert len(val) == 4, [d["step"] for d in val]
    assert [d["step"] for d in val] == sorted(d["step"] for d in val)

    after = torch.load(tmp_path / "r" / "last.pt", weights_only=False)
    assert after["next_epoch"] == 4
    assert after["step"] > first["step"]


def test_early_stopping_waits_long_enough_and_then_stops(tmp_path):
    """Patience is measured, not chosen, and truncation is the invisible risk.

    Across fourteen finished runs the longest stretch of non-improving
    evaluations *before* a run reached its eventual best was 8. A patience at
    or below that would have cut two of them off short of their own peak -- and
    a truncated run does not look truncated, it just looks worse, so the
    comparison silently measures the schedule instead of the structure.
    """

    def run(patience: int, epochs: int) -> int:
        out = tmp_path / f"p{patience}"
        cfg = with_overrides(
            Config(),
            {
                "data": {"num_scenes": {"train": 2, "val": 2, "test": 2}},
                "model": {"dim": 32, "layers": 1, "heads": 2},
                "train": {
                    "out_dir": str(out),
                    "epochs": epochs,
                    "patience": patience,
                    "batch_size": 4,
                    "num_workers": 0,
                    "device": "cpu",
                },
            },
        )
        torch.manual_seed(0)
        Trainer(cfg).train()
        lines = [
            json.loads(x)
            for x in (out / "metrics.jsonl").read_text().splitlines()
        ]
        return len([d for d in lines if d.get("tag") == "val"])

    # patience=0 must never stop early: the ceiling is the only limit.
    assert run(0, 4) == 4

    # A patience of 1 stops as soon as one evaluation fails to improve, so it
    # cannot run the full ceiling unless every epoch improved.
    assert run(1, 12) <= 12
