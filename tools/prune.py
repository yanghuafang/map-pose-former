#!/usr/bin/env python3
"""Prune a trained checkpoint structurally, by channels and by heads.

    tools/prune.py runs/base/best.pt --keep 0.5 --out runs/pruned/init.pt
    tools/prune.py runs/base/best.pt --keep-heads 0.5 --out runs/heads/init.pt

Writes weights plus the plan that reshapes a fresh model to receive them, so
the result loads from its own config. Fine-tune it with

    tools/train.py --config configs/synth_base.yaml \\
        train.init_from=runs/pruned/init.pt train.out_dir=runs/pruned

Pruning without fine-tuning is reported here so the drop is visible: the point
of the cycle is what the fine-tune recovers, and a table that only shows the
end state cannot say whether the pruning hurt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.checkpoint import build_model
from mapposeformer.prune import parameter_count, prune_model, split_plan


def main() -> int:
    """@brief Prune and write. @return Process exit status."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    # Both default to keeping everything. With two knobs, a default that
    # prunes means asking for heads silently halves the channels too.
    ap.add_argument(
        "--keep",
        type=float,
        default=1.0,
        help="fraction of feed-forward channels to retain",
    )
    ap.add_argument(
        "--keep-heads",
        type=float,
        default=1.0,
        help="fraction of attention heads to retain",
    )
    ap.add_argument("--out", required=True, help="where to write the result")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt)

    before, prunable_before = parameter_count(model)
    plan = prune_model(model, args.keep, args.keep_heads)
    if not plan:
        ap.error("nothing to prune: --keep or --keep-heads must be below 1.0")
    after, prunable_after = parameter_count(model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "prune_plan": plan,
            "config": ckpt["config"],
        },
        out,
    )
    print(f"{args.checkpoint} -> {out}")
    print(
        f"  params {before / 1e6:.2f}M -> {after / 1e6:.2f}M "
        f"({100 * (1 - after / before):.1f}% removed)"
    )
    print(
        f"  of which prunable {prunable_before / 1e6:.2f}M -> "
        f"{prunable_after / 1e6:.2f}M across {len(plan)} modules"
    )
    ffn, attn = split_plan(model, plan)
    if ffn:
        print(f"  hidden width now {min(ffn.values())}")
    if attn:
        print(f"  heads now {min(attn.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
