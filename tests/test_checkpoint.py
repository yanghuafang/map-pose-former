"""Reading a checkpoint back: the config, the plan, and only then the weights.

Skipping the plan is the failure this file exists for. It costs nothing on an
ordinary checkpoint and makes every pruned one unreadable, so it goes unnoticed
until the end of a compression experiment.
"""

from __future__ import annotations

import pytest
import torch

from mapposeformer.checkpoint import build_model, load_checkpoint
from mapposeformer.config import Config
from mapposeformer.model.model import MapPoseFormer
from mapposeformer.prune import prune_model
from tests.test_model import _batch


def _write(tmp_path, model, plan=None):
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": Config(),
            "prune_plan": plan,
        },
        path,
    )
    return str(path)


def test_a_pruned_checkpoint_loads_and_answers_the_same(tmp_path):
    torch.manual_seed(0)
    model = MapPoseFormer(Config().model)
    plan = prune_model(model, keep_frac=0.5)
    model.eval()
    with torch.no_grad():
        want = model(_batch(2))["delta"]

    got, cfg = load_checkpoint(_write(tmp_path, model, plan))
    with torch.no_grad():
        assert torch.allclose(want, got.eval()(_batch(2))["delta"], atol=1e-6)
    assert cfg.model.dim == Config().model.dim, "config came back changed"


def test_an_unpruned_checkpoint_needs_no_plan(tmp_path):
    torch.manual_seed(0)
    model = MapPoseFormer(Config().model).eval()
    got, _ = load_checkpoint(_write(tmp_path, model))
    with torch.no_grad():
        assert torch.allclose(
            model(_batch(2))["delta"], got.eval()(_batch(2))["delta"]
        )


def test_overrides_reach_the_config_and_not_the_weights(tmp_path):
    """A caller overrides the data to change the evidence, never the model."""
    torch.manual_seed(0)
    path = _write(tmp_path, MapPoseFormer(Config().model))

    _, cfg = load_checkpoint(path, ["data.sample.keep_classes=[0,1]"])
    assert tuple(cfg.data.sample.keep_classes) == (0, 1)
    # A shape override is a different model, and must fail rather than load
    # some other network's weights into it.
    with pytest.raises(RuntimeError):
        load_checkpoint(path, ["model.dim=64"])


def test_build_model_defaults_to_the_stored_config(tmp_path):
    torch.manual_seed(0)
    model = MapPoseFormer(Config().model)
    ckpt = torch.load(
        _write(tmp_path, model), map_location="cpu", weights_only=False
    )
    assert build_model(ckpt).p.dim == Config().model.dim
