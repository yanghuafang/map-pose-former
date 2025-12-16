#!/usr/bin/env python3
"""Match geometrically instead of with a transformer, and score it the same way.

    tools/baseline_matcher.py --split test
    tools/baseline_matcher.py --split test --sigma 1.5

Every ablation in RESULTS.md varies something *inside* the learned matcher.
None asks whether it needs to be learned at all. This asks; RESULTS.md has the
answer and what it means. The matcher itself is
``mapposeformer/model/geometric.py``, which says what it concedes and why.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.checkpoint import load_checkpoint
from mapposeformer.data import build_dataset
from mapposeformer.engine import evaluate, format_report
from mapposeformer.model.geometric import GeometricBaseline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        default="runs/m1_base/best.pt",
        help="read the data config from here, and compare against it",
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument(
        "--iters",
        type=int,
        default=1,
        help="re-match after moving detections, as the model does",
    )
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    learned, cfg = load_checkpoint(args.checkpoint, args.overrides)
    ds = build_dataset(cfg.data, args.split)

    def run(model, label):
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4)
        print(f"\n=== {label}")
        print(
            format_report(
                evaluate(model.to(args.device).eval(), loader, args.device, 0.5)
            )
        )

    run(
        GeometricBaseline(cfg.model, args.sigma, args.iters),
        f"geometric, sigma={args.sigma}, iters={args.iters}",
    )
    run(learned, f"learned ({args.checkpoint})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
