"""What a checkpoint holds, and how to get a model back out of it.

Three things, and all three are needed. The weights, obviously. The config the
model was built from, which older files predate fields of -- ``config.upgrade``
fills those in. And, if the model was pruned, the plan that says how narrow it
really is: a pruned checkpoint is not the width its config implies, and the
weights will not load until the plan has reshaped the model to receive them.

That sequence lives here because nine call sites needed it and four had drifted
from it. Every one of those four could load an ordinary checkpoint and none
could load a pruned one, which is a failure that only appears at the end of a
compression experiment.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from mapposeformer.config import (
    Config,
    ModelParams,
    parse_overrides,
    upgrade,
    with_overrides,
)
from mapposeformer.model.attention import unpack_attention
from mapposeformer.model.model import MapPoseFormer
from mapposeformer.prune import apply_plan


def build_model(
    ckpt: dict[str, Any], params: ModelParams | None = None
) -> MapPoseFormer:
    """@brief Rebuild the model a checkpoint describes, and load its weights.

    @param ckpt An already-loaded checkpoint.
    @param params The model config to build from. Defaults to the checkpoint's
        own; a caller passes one only to apply overrides on top of it.
    @return The model, on the CPU, in training mode as constructed.
    """
    model = MapPoseFormer(
        upgrade(ckpt["config"]).model if params is None else params
    )
    if ckpt.get("prune_plan"):
        apply_plan(model, ckpt["prune_plan"])
    model.load_state_dict(unpack_attention(ckpt["model"]))
    return model


def load_checkpoint(
    path: str, overrides: Iterable[str] = ()
) -> tuple[MapPoseFormer, Config]:
    """@brief Read a checkpoint from disk: the model, and the config it needs.

    @param path The file to read.
    @param overrides ``section.field=value`` strings, applied on top of the
        stored config. One that changes the model's shape will fail the load
        rather than build a different model quietly.
    @return ``(model, config)``. Callers build their dataset from the config,
        so the evidence matches what the checkpoint was trained on.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = with_overrides(
        upgrade(ckpt["config"]), parse_overrides(list(overrides))
    )
    return build_model(ckpt, cfg.model), cfg
