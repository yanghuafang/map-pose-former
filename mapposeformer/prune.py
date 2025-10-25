"""Structured pruning: remove channels, do not mask them.

``torch.nn.utils.prune`` writes zeros into weights and leaves the tensors the
size they were. That is the right tool for studying sparsity and the wrong one
for making a model faster: a zero costs exactly what a non-zero costs, and the
GPU never learns the difference. Everything here physically slices, so a pruned
model has fewer FLOPs and a smaller state dict, and `tools/bench.py` can see it.

**Only the feed-forward hidden channels are prunable, and that is a property of
the model rather than of pruning.** They are 46% of the student's parameters,
so this is the larger lever anyway -- but attention heads are
the other obvious one and cannot be touched while the blocks use
``nn.MultiheadAttention``, which requires its internal projection width to equal
``embed_dim``. Dropping a head makes those differ, and the module has no way to
express it. Replacing it with an explicit ``scaled_dot_product_attention`` would
allow head pruning and stop materialising the attention matrix, which
``docs/OPEN_ITEMS.md`` already wanted for other reasons.

A pruned model no longer matches the width its config implies, so the plan is
saved beside the weights and replayed before the state dict is loaded. Without
that a pruned checkpoint is unloadable, which would make the whole stage a
measurement nobody can reproduce.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mapposeformer.model.attention import FeedForward


def ffn_importance(ffn: FeedForward) -> Tensor:
    """@brief Per-hidden-channel importance, by weight magnitude.

    A hidden channel is touched by one row of the first matrix and one column
    of the second, and its contribution is bounded by both: a channel with a
    large input projection and a zero output projection contributes nothing.
    So the score is the norm of the pair, not of either alone.

    Magnitude is the honest baseline rather than the best criterion. It ignores
    what the activations actually do, which a Taylor or activation-aware score
    would use, and the protocol here -- prune, fine-tune, re-measure -- is what
    says whether that matters.

    @param ffn The module to score.
    @return ``(hidden,)`` non-negative scores.
    """
    w_in: Tensor = ffn[0].weight  # (hidden, dim)
    w_out: Tensor = ffn[2].weight  # (dim, hidden)
    return w_in.norm(dim=1) * w_out.norm(dim=0)


def prune_ffn(ffn: FeedForward, keep: int) -> Tensor:
    """@brief Slice a feed-forward down to its ``keep`` best channels.

    In place, and the module is a smaller one afterwards -- ``keep`` rows of the
    first matrix, ``keep`` columns of the second, and the bias between them.

    @param ffn The module to shrink.
    @param keep How many hidden channels to retain; at least one.
    @return The indices kept, ascending, so the plan can be replayed.
    @throws ValueError If ``keep`` is not in ``1..hidden``.
    """
    hidden = ffn[0].out_features
    if not 1 <= keep <= hidden:
        raise ValueError(f"keep must be in 1..{hidden}, got {keep}")
    idx = ffn_importance(ffn).argsort(descending=True)[:keep].sort().values

    lin_in, lin_out = ffn[0], ffn[2]
    new_in = nn.Linear(lin_in.in_features, keep, bias=lin_in.bias is not None)
    new_out = nn.Linear(
        keep, lin_out.out_features, bias=lin_out.bias is not None
    )
    with torch.no_grad():
        new_in.weight.copy_(lin_in.weight[idx])
        if lin_in.bias is not None:
            new_in.bias.copy_(lin_in.bias[idx])
        new_out.weight.copy_(lin_out.weight[:, idx])
        if lin_out.bias is not None:
            new_out.bias.copy_(lin_out.bias)
    ffn[0], ffn[2] = (
        new_in.to(lin_in.weight.device),
        new_out.to(lin_out.weight.device),
    )
    return idx


def prune_model(model: nn.Module, keep_frac: float) -> dict[str, int]:
    """@brief Shrink every feed-forward in the model by the same fraction.

    Uniformly, because a per-layer budget is a second experiment: which layers
    tolerate pruning is worth knowing and is not knowable before the first
    measurement says whether any of them do.

    @param model The model to prune, in place.
    @param keep_frac Fraction of hidden channels to keep, in ``(0, 1]``.
    @return The plan: module name to kept width, for
        :func:`apply_plan` to replay.
    @throws ValueError If ``keep_frac`` is outside ``(0, 1]``.
    """
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
    plan: dict[str, int] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, FeedForward):
            keep = max(1, round(mod[0].out_features * keep_frac))
            prune_ffn(mod, keep)
            plan[name] = keep
    return plan


def apply_plan(model: nn.Module, plan: dict[str, int]) -> None:
    """@brief Reshape a fresh model to a saved plan, before loading weights.

    The values copied here are discarded by the ``load_state_dict`` that
    follows; what matters is that the modules end up the right shape. This is
    what makes a pruned checkpoint loadable from its config.

    @param model A model built from the checkpoint's config.
    @param plan The dict :func:`prune_model` returned.
    @throws KeyError If the plan names a module the model does not have.
    """
    named = dict(model.named_modules())
    for name, keep in plan.items():
        if name not in named:
            raise KeyError(f"plan names {name!r}, which this model has not")
        prune_ffn(named[name], keep)


def parameter_count(model: nn.Module) -> tuple[int, int]:
    """@brief Total parameters, and how many are in feed-forwards.

    @param model Any model.
    @return ``(total, in_feed_forwards)``.
    """
    total = sum(p.numel() for p in model.parameters())
    ffn = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, FeedForward)
        for p in m.parameters()
    )
    return total, ffn
