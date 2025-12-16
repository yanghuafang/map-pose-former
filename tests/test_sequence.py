"""The closed loop, and who is allowed to drive it."""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from mapposeformer.data import DataParams, SyntheticDataset
from mapposeformer.engine import eval_sequence, format_sequence
from mapposeformer.engine.sequence import SequenceResult
from mapposeformer.model import MapPoseFormer, ModelParams
from mapposeformer.model.geometric import GeometricBaseline

ONE_SCENE = {"train": 1, "val": 1, "test": 1}


def _source() -> SyntheticDataset:
    return SyntheticDataset(replace(DataParams(), num_scenes=ONE_SCENE), "test")


def _geometric() -> GeometricBaseline:
    return GeometricBaseline(ModelParams(), sigma=1.5, iters=1).eval()


def test_either_backend_drives_the_same_loop():
    """The filter reads four keys and does not ask what produced them.

    That is what makes the learned/geometric comparison a comparison: the
    scenes, the odometry, the gates and the metrics are one code path, and the
    backend is the only thing that differs.
    """
    learned = MapPoseFormer(
        replace(ModelParams(), dim=32, num_layers=1, num_heads=2)
    ).eval()
    for model in (_geometric(), learned):
        m = eval_sequence(model, _source(), limit=1)
        assert m["seq/frames"] > 0
        assert math.isfinite(m["seq/rmse_trans_m"])
        assert 0.0 <= m["seq/accepted"] <= 1.0
        # Renders without a KeyError, which is the only thing keeping the
        # metric names and the report that prints them in step.
        assert format_sequence(m)


def test_the_longest_refusal_run_is_the_longest_run():
    """Refusals summed would say the same for one long lockout as for many
    isolated declines, and only one of those is a broken filter."""
    r = SequenceResult(torch.empty(0), torch.empty(0))
    r.accepted = [True, False, True, False, False, False, True, False]
    assert r.longest_refusal == 3
