"""The closed loop, which only two consecutive frames can check.

`run_sequences` produces the headline M4 numbers, and its failure mode is
invisible to a suite of single-frame tests however large: a local named `key`
inside `for key in keys` rebinds the scene key to the string `"information"`,
so every frame after the first asks the dataset for a scene called that.
"""

from __future__ import annotations

import pytest
import torch

from mapposeformer.config import Config, with_overrides
from mapposeformer.data import build_dataset
from mapposeformer.engine.sequence import run_sequences
from mapposeformer.filter import FilterParams
from mapposeformer.model import MapPoseFormer, ModelParams


@pytest.fixture(scope="module")
def small():
    cfg = with_overrides(
        Config(),
        {"data": {"num_scenes": {"train": 2, "val": 2, "test": 2}}},
    )
    ds = build_dataset(cfg.data, "test")
    torch.manual_seed(0)
    model = MapPoseFormer(ModelParams(dim=32, layers=1, heads=2)).eval()
    return model, ds


@pytest.mark.parametrize("measurement", ["information", "posterior"])
def test_the_loop_runs_every_frame_of_every_scene(small, measurement):
    """Two scenes end to end, which is what a shadowed `key` breaks.

    The assertion that matters is not the error -- an untrained model has no
    business being accurate -- but that each scene advanced through all of its
    frames rather than dying on the second one.
    """

    model, ds = small
    p = FilterParams(measurement=measurement, trust_threshold=0.0)
    out = run_sequences(model, ds, "cpu", p, limit=2)

    assert out.scenes == 2
    expected = sum(len(ds.frames_of(k)) for k in ds.sequences()[:2])
    assert out.frames == expected, (out.frames, expected)
    assert len(out.finals) == 2
    assert all(torch.isfinite(torch.tensor(f)) for f in out.finals)


def test_the_loop_refuses_to_run_without_what_it_needs(small):
    """A model reporting no covariance must fail loudly, not silently coast.

    A filter handed nothing would otherwise run on odometry alone and report a
    perfectly plausible drift curve that measures the ego simulator rather
    than the model.
    """

    model, ds = small

    class Blind(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, batch, **kw):
            out = self.inner(batch, **kw)
            return {k: v for k, v in out.items() if k != "information"}

    with pytest.raises(ValueError, match="information"):
        run_sequences(
            Blind(model),
            ds,
            "cpu",
            FilterParams(measurement="information"),
            limit=1,
        )
