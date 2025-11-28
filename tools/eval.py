#!/usr/bin/env python3
"""Score a checkpoint on a split.

    tools/eval.py runs/base/best.pt --split test
    tools/eval.py runs/base/best.pt --split val data.sample.keep_classes=[0,1]

Overrides apply on top of the configuration stored in the checkpoint, which is
what makes the data ablations in docs/DATASET.md a one-line change rather than
a retrain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.checkpoint import build_model
from mapposeformer.config import parse_overrides, upgrade, with_overrides
from mapposeformer.data import build_dataset
from mapposeformer.engine.evaluator import evaluate, format_report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*", help="section.field=value")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # `upgrade` fills in fields added since the checkpoint was written, so an
    # older run stays loadable instead of failing on an attribute nobody had
    # when it trained.
    cfg = upgrade(ckpt["config"])
    if args.overrides:
        cfg = with_overrides(cfg, parse_overrides(args.overrides))
    device = args.device or cfg.train.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # Through `build_model`, not by constructing and loading here. A pruned
    # checkpoint's tensors do not match the width its config implies, and only
    # the stored `prune_plan` reconciles them -- so a site that builds its own
    # model reads every ordinary checkpoint and no compressed one, and finds
    # that out at the end of a compression experiment, with nothing before
    # that point to warn anyone.
    model = build_model(ckpt, cfg.model).to(device)

    dataset = build_dataset(cfg.data, args.split)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    print(f"{args.checkpoint}  split={args.split}  frames={len(dataset)}")
    print(format_report(evaluate(model, loader, device)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
