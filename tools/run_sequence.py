#!/usr/bin/env python3
"""Drive a checkpoint around the loop its corrections are meant to close.

    tools/run_sequence.py runs/base/best.pt --split test
    tools/run_sequence.py runs/base/best.pt --split test --limit 20

    tools/run_sequence.py runs/teacher/best.pt --split test --quantize

Open loop hands the model a freshly drawn prior every frame. A vehicle hands it
back its own last answer, so a bias compounds where independent noise averages
away -- and that is the difference this measures.

**Compressed models belong here too, and this is where they are hardest on
themselves.** `tools/paired.py` scores a compression step on open-loop recall,
which is the frame-by-frame view; a filter integrates whatever the covariance
says over a whole scene, so a step that leaves the median calibration intact and
inflates its tail is punished here and nowhere else. INT8 on the teacher does
exactly that -- NEES median 0.908 -> 0.962, but the per-frame ratio's p99 is
8.35 -- and an open-loop table cannot say what that costs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from mapposeformer.checkpoint import build_model
from mapposeformer.config import parse_overrides, upgrade, with_overrides
from mapposeformer.data import build_dataset
from mapposeformer.engine.sequence import run_sequences
from mapposeformer.filter import FilterParams


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--limit", type=int, default=0, help="scenes to run; 0 is all of them"
    )
    ap.add_argument(
        "--measurement",
        default="information",
        choices=("information", "posterior"),
        help="what the filter is handed; 'posterior' double-counts the prior",
    )
    # Applied in this process rather than loaded from disk, for the reason
    # `tools/paired.py` gives: `quantize` swaps every Linear for a
    # FakeQuantLinear, so the result no longer matches the state dict a fresh
    # model expects and there is no checkpoint format that would carry it.
    ap.add_argument(
        "--quantize",
        action="store_true",
        help="quantize in-process, calibrated on the train split",
    )
    ap.add_argument("--calib-batches", type=int, default=16)
    ap.add_argument("overrides", nargs="*", help="section.field=value")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = upgrade(ckpt["config"])
    if args.overrides:
        cfg = with_overrides(cfg, parse_overrides(args.overrides))
    device = args.device or cfg.train.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # `build_model` rather than `MapPoseFormer(cfg.model)`: a pruned checkpoint
    # carries a `prune_plan` that reshapes the model before the weights load,
    # and constructing from the config alone fails on every pruned FFN. That is
    # the same fault `tools/eval.py` had, and the reason it is worth fixing in
    # both is that a compression stage nobody can evaluate through the loop is
    # a compression stage nobody can accept.
    model = build_model(ckpt, cfg.model).to(device)
    model.eval().requires_grad_(False)

    if args.quantize:
        # Imported here: quantization is opt-in, and a plain closed-loop run
        # should not need the module at all.
        from mapposeformer.quantize import QuantParams, calibrate, quantize

        n = quantize(model, QuantParams())
        # Calibrated on TRAIN. Activation ranges are learned parameters like
        # any other, and learning them from the split being scored would report
        # a number the model was tuned on.
        cl = DataLoader(
            build_dataset(cfg.data, "train"),
            batch_size=cfg.train.batch_size,
            num_workers=2,
        )
        seen = calibrate(
            model,
            ({k: v.to(device) for k, v in b.items()} for b in cl),
            limit=args.calib_batches,
        )
        print(f"quantized {n} Linear modules, calibrated {seen} on train")

    dataset = build_dataset(cfg.data, args.split, sequential=True)
    fp = FilterParams(measurement=args.measurement)
    print(
        f"{args.checkpoint}  split={args.split}"
        f"  measurement={args.measurement}"
        f"{'  INT8' if args.quantize else ''}"
    )
    print(run_sequences(model, dataset, device, fp, limit=args.limit).format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
