"""Structured pruning: the model gets smaller, still runs, and reloads.

The last is the one that would be missed. A pruned model no longer matches the
width its config implies, so a checkpoint of it is unloadable unless the plan
travels with the weights -- and an unloadable checkpoint makes the whole
compression stage a measurement nobody can reproduce.
"""

import pytest
import torch

from mapposeformer.model.attention import FeedForward, MultiheadAttention
from mapposeformer.model.model import MapPoseFormer, ModelParams
from mapposeformer.prune import (
    apply_plan,
    ffn_importance,
    head_importance,
    parameter_count,
    prune_ffn,
    prune_heads,
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


def test_the_heads_kept_are_the_ones_that_mattered():
    """A head whose output projection is zero writes nothing to the residual
    stream, so it goes first however confidently it attends."""
    attn = MultiheadAttention(8, 4)
    with torch.no_grad():
        attn.v_proj.weight.fill_(1.0)
        attn.out_proj.weight.fill_(1.0)
        attn.out_proj.weight[:, 2 * attn.head_dim : 3 * attn.head_dim] = 0.0
    assert float(head_importance(attn).detach()[2]) == 0.0
    assert 2 not in prune_heads(attn, 3).tolist()


def test_query_and_key_do_not_decide_which_head_survives():
    """The softmax normalises their scale away: a head with tiny query weights
    attends nearly uniformly, which is not the same as contributing little."""
    attn = MultiheadAttention(8, 2)
    with torch.no_grad():
        attn.q_proj.weight.fill_(1.0)
        attn.k_proj.weight.fill_(1.0)
        attn.q_proj.weight[: attn.head_dim] = 1e-6
    before = head_importance(attn).clone()
    with torch.no_grad():
        attn.k_proj.weight[: attn.head_dim] = 5.0
    assert torch.equal(before, head_importance(attn))


def test_pruning_heads_narrows_attention_below_the_model_width():
    """Which is the whole reason attention.py exists: ``nn.MultiheadAttention``
    cannot express a projection narrower than ``embed_dim``."""
    torch.manual_seed(0)
    model = MapPoseFormer(ModelParams()).eval()
    before, _ = parameter_count(model)
    plan = prune_model(model, keep_frac=1.0, head_frac=0.5)
    after, _ = parameter_count(model)

    assert after < before and plan, "nothing was pruned"
    for mod in model.modules():
        if isinstance(mod, MultiheadAttention):
            assert mod.heads == 2
            assert mod.q_proj.out_features < mod.q_proj.in_features
    with torch.no_grad():
        out = model(_batch(2))
    for k, v in out.items():
        assert torch.isfinite(v).all(), k


def test_one_plan_carries_both_prunings():
    """Feed-forwards and attention share a dict, so a checkpoint written before
    heads were prunable replays through the same code."""
    torch.manual_seed(0)
    model = MapPoseFormer(ModelParams()).eval()
    plan = prune_model(model, keep_frac=0.5, head_frac=0.5)
    with torch.no_grad():
        want = model(_batch(2))["delta"]

    fresh = MapPoseFormer(ModelParams())
    apply_plan(fresh, plan)
    fresh.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.allclose(want, fresh.eval()(_batch(2))["delta"], atol=1e-6)


def test_it_refuses_a_head_fraction_it_cannot_apply():
    model = MapPoseFormer(ModelParams())
    with pytest.raises(ValueError, match="head_frac"):
        prune_model(model, 0.5, 1.5)
    with pytest.raises(TypeError, match="not prunable"):
        apply_plan(model, {"matcher": 2})


def test_a_pruning_that_removes_nothing_records_nothing():
    """`tools/eval.py` and friends decide whether to replay by whether the plan
    is empty, so a no-op that still filled it would make every unpruned
    checkpoint take the pruned path."""
    model = MapPoseFormer(ModelParams())
    assert prune_model(model, keep_frac=1.0, head_frac=1.0) == {}
    assert set(prune_model(model, keep_frac=1.0, head_frac=0.5)) == {
        name
        for name, mod in model.named_modules()
        if isinstance(mod, MultiheadAttention)
    }
