#!/usr/bin/env python3
"""Evaluate a checkpoint on a split.

    tools/eval.py runs/default/best.pt --split test
    tools/eval.py runs/default/best.pt data.sample.keep_classes=[0,1]

The second form is the ablation from ``mapposeformer/data/classes.py``: keep
only the classes that run parallel to the road, and watch longitudinal error
grow while lateral error does not. The checkpoint's own config is the baseline;
overrides are applied on top of it, so the model is unchanged and only the
evidence it is given differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.config import parse_overrides, with_overrides
from mapposeformer.data import build_dataset
from mapposeformer.engine import evaluate, format_report
from mapposeformer.model import MapPoseFormer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--trust-threshold", type=float, default=0.5)
    ap.add_argument("overrides", nargs="*", help="section.field=value")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = with_overrides(ckpt["config"], parse_overrides(args.overrides))

    model = MapPoseFormer(cfg.model)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    dataset = build_dataset(cfg.data, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    print(f"{args.checkpoint}  split={args.split}  frames={len(dataset)}")
    print(
        format_report(
            evaluate(model, loader, args.device, args.trust_threshold)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
