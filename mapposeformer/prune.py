"""Structured pruning: remove channels, do not mask them.

``torch.nn.utils.prune`` writes zeros into weights and leaves the tensors the
size they were. That is the right tool for studying sparsity and the wrong one
for making a model faster: a zero costs exactly what a non-zero costs, and the
GPU never learns the difference. Everything here physically slices, so a pruned
model has fewer FLOPs and a smaller state dict, and `tools/bench.py` can see it.

**Two things are prunable: feed-forward hidden channels and attention heads.**
Feed-forwards are 46% of the student's parameters and attention 49%, so between
them they are all but 5% of it. Both are scored the same way: the product of
what feeds a unit and what it feeds, so either end being small makes the unit
cheap to lose.

A pruned model no longer matches the width its config implies, so the plan
travels with the weights and ``checkpoint.py`` replays it on the way back in.
Without that a pruned checkpoint is unloadable, which would make the whole
stage a measurement nobody can reproduce.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mapposeformer.model.attention import FeedForward, MultiheadAttention


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


def head_importance(attn: MultiheadAttention) -> Tensor:
    """@brief Per-head importance, by weight magnitude.

    A head owns a slice of rows in each input projection and a slice of columns
    in the output. Only two of those bound what it contributes: the value
    projection, which sets the scale of what it reads, and the output
    projection, which sets how much of that reaches the residual stream. Query
    and key decide *where* it looks, and the softmax normalises their scale
    away -- a head with tiny query weights attends almost uniformly rather than
    weakly.

    The same magnitude caveat as :func:`ffn_importance` applies, and so does
    the same protocol for finding out whether it matters.

    @param attn The module to score.
    @return ``(heads,)`` non-negative scores.
    """
    h, d = attn.heads, attn.head_dim
    v = attn.v_proj.weight.view(h, d, -1).flatten(1).norm(dim=1)
    out = attn.out_proj.weight.view(-1, h, d).transpose(0, 1).flatten(1)
    return v * out.norm(dim=1)


def prune_heads(attn: MultiheadAttention, keep: int) -> Tensor:
    """@brief Slice an attention module down to its ``keep`` best heads.

    In place. ``head_dim`` is untouched, so the projections narrow to
    ``keep * head_dim`` and stop matching the model width -- which is the whole
    reason this needs its own attention class.

    @param attn The module to shrink.
    @param keep How many heads to retain; at least one.
    @return The head indices kept, ascending, so the plan can be replayed.
    @throws ValueError If ``keep`` is not in ``1..heads``.
    """
    if not 1 <= keep <= attn.heads:
        raise ValueError(f"keep must be in 1..{attn.heads}, got {keep}")
    idx = head_importance(attn).argsort(descending=True)[:keep].sort().values
    rows = (
        idx.unsqueeze(1) * attn.head_dim
        + torch.arange(attn.head_dim, device=idx.device)
    ).flatten()

    dim, width = attn.q_proj.in_features, keep * attn.head_dim
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj"):
            old = getattr(attn, name)
            new = nn.Linear(dim, width, device=old.weight.device)
            new.weight.copy_(old.weight[rows])
            new.bias.copy_(old.bias[rows])
            setattr(attn, name, new)
        old = attn.out_proj
        new = nn.Linear(width, dim, device=old.weight.device)
        new.weight.copy_(old.weight[:, rows])
        new.bias.copy_(old.bias)
        attn.out_proj = new
    attn.heads = keep
    return idx


def prune_model(
    model: nn.Module, keep_frac: float, head_frac: float = 1.0
) -> dict[str, int]:
    """@brief Shrink every feed-forward and attention in the model.

    Uniformly, because a per-layer budget is a second experiment: which layers
    tolerate pruning is worth knowing and is not knowable before the first
    measurement says whether any of them do.

    @param model The model to prune, in place.
    @param keep_frac Fraction of feed-forward hidden channels to keep, in
        ``(0, 1]``.
    @param head_frac Fraction of attention heads to keep, in ``(0, 1]``. The
        default keeps all of them, so a caller that only wants the
        feed-forwards gets what it always got.
    @return The plan: module name to kept width, for :func:`apply_plan` to
        replay. Both kinds share one dict, because a name identifies its own
        module and therefore what the number means. A module that loses
        nothing is left out, so a fraction of 1.0 produces no entries.
    @throws ValueError If either fraction is outside ``(0, 1]``.
    """
    for label, frac in (("keep_frac", keep_frac), ("head_frac", head_frac)):
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"{label} must be in (0, 1], got {frac}")
    plan: dict[str, int] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, FeedForward):
            width = mod[0].out_features
            keep = max(1, round(width * keep_frac))
            if keep < width:
                prune_ffn(mod, keep)
                plan[name] = keep
        elif isinstance(mod, MultiheadAttention):
            keep = max(1, round(mod.heads * head_frac))
            if keep < mod.heads:
                prune_heads(mod, keep)
                plan[name] = keep
    return plan


def apply_plan(model: nn.Module, plan: dict[str, int]) -> None:
    """@brief Reshape a fresh model to a saved plan, before loading weights.

    The values copied here are discarded by the ``load_state_dict`` that
    follows; what matters is that the modules end up the right shape. This is
    what makes a pruned checkpoint loadable from its config.

    Which of the two prunings to replay is read from the module the plan names,
    so a checkpoint written before heads were prunable replays unchanged.

    @param model A model built from the checkpoint's config.
    @param plan The dict :func:`prune_model` returned.
    @throws KeyError If the plan names a module the model does not have.
    @throws TypeError If it names one that cannot be pruned.
    """
    named = dict(model.named_modules())
    for name, keep in plan.items():
        if name not in named:
            raise KeyError(f"plan names {name!r}, which this model has not")
        mod = named[name]
        if isinstance(mod, FeedForward):
            prune_ffn(mod, keep)
        elif isinstance(mod, MultiheadAttention):
            prune_heads(mod, keep)
        else:
            raise TypeError(f"{name} is a {type(mod).__name__}, not prunable")


def split_plan(
    model: nn.Module, plan: dict[str, int]
) -> tuple[dict[str, int], dict[str, int]]:
    """@brief Partition a plan into its feed-forward and attention halves.

    By module type rather than by name, so a caller reporting what it pruned
    does not have to know what the model calls its submodules.

    @param model The model the plan was made for, pruned or not.
    @param plan The dict :func:`prune_model` returned.
    @return ``(feed_forwards, attentions)``.
    """
    named = dict(model.named_modules())
    ffn = {
        n: w for n, w in plan.items() if isinstance(named.get(n), FeedForward)
    }
    return ffn, {n: w for n, w in plan.items() if n not in ffn}


def parameter_count(model: nn.Module) -> tuple[int, int]:
    """@brief Total parameters, and how many are in prunable modules.

    @param model Any model.
    @return ``(total, in_feed_forwards_and_attention)``.
    """
    total = sum(p.numel() for p in model.parameters())
    prunable = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, (FeedForward, MultiheadAttention))
        for p in m.parameters()
    )
    return total, prunable
