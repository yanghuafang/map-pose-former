"""Structured pruning: the model gets smaller, still runs, and reloads.

The last is the one that would be missed. A pruned model no longer matches the
width its config implies, so a checkpoint of it is unloadable unless the plan
travels with the weights -- and an unloadable checkpoint makes the whole
compression stage a measurement nobody can reproduce.
"""

import pytest
import torch

from mapposeformer.model.attention import FeedForward
from mapposeformer.model.model import MapPoseFormer, ModelParams
from mapposeformer.prune import (
    apply_plan,
    ffn_importance,
    parameter_count,
    prune_ffn,
    prune_model,
)
from tests.test_model import _batch


def test_pruning_removes_parameters_and_the_model_still_runs():
    torch.manual_seed(0)
    model = MapPoseFormer(ModelParams()).eval()
    before, _ = parameter_count(model)
    plan = prune_model(model, 0.5)
    after, _ = parameter_count(model)
    assert after < before and len(plan) == 16
    with torch.no_grad():
        out = model(_batch(2))
    for k, v in out.items():
        assert torch.isfinite(v).all(), k


def test_the_channels_kept_are_the_ones_that_mattered():
    """A channel with a zeroed output projection contributes nothing, so it
    must be the first to go however large its input projection is."""
    ffn = FeedForward(8, 2)
    with torch.no_grad():
        ffn[0].weight.fill_(1.0)
        ffn[2].weight.fill_(1.0)
        ffn[2].weight[:, 3] = 0.0  # channel 3 can only ever emit zero
    assert float(ffn_importance(ffn).detach()[3]) == 0.0
    kept = prune_ffn(ffn, ffn[0].out_features - 1)
    assert 3 not in kept.tolist()


def test_a_pruned_checkpoint_reloads_from_its_config(tmp_path):
    torch.manual_seed(0)
    params = ModelParams()
    model = MapPoseFormer(params).eval()
    plan = prune_model(model, 0.375)
    with torch.no_grad():
        want = model(_batch(2))["delta"]

    path = tmp_path / "pruned.pt"
    torch.save({"model": model.state_dict(), "prune_plan": plan}, path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    fresh = MapPoseFormer(params)
    with pytest.raises(RuntimeError):
        fresh.load_state_dict(ckpt["model"])  # wrong shape without the plan
    apply_plan(fresh, ckpt["prune_plan"])
    fresh.load_state_dict(ckpt["model"])
    fresh.eval()
    with torch.no_grad():
        got = fresh(_batch(2))["delta"]
    assert torch.allclose(want, got, atol=1e-6)


def test_it_refuses_a_plan_it_cannot_apply():
    model = MapPoseFormer(ModelParams())
    with pytest.raises(KeyError, match="no_such"):
        apply_plan(model, {"no_such.ffn": 4})
    with pytest.raises(ValueError, match="keep_frac"):
        prune_model(model, 0.0)
