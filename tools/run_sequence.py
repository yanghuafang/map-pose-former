#!/usr/bin/env python3
"""Run a pose backend closed loop and report the trajectory.

    tools/run_sequence.py runs/control/best.pt --split test
    tools/run_sequence.py runs/control/best.pt --backend geometric
    tools/run_sequence.py runs/control/best.pt --limit 10 filter.min_mass=8

The last form overrides a filter setting rather than a model one; `filter.`
prefixes are peeled off before the rest reach the checkpoint's config.

``--backend geometric`` swaps the transformer for nearest-neighbour matching
and changes nothing else. Open loop the learned matcher wins by more than four
times; a filter averages over frames, so whether that gap survives being
filtered is a question about the system rather than about either backend, and
running the two separately cannot answer it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.config import parse_overrides, upgrade, with_overrides
from mapposeformer.data import build_dataset
from mapposeformer.engine import eval_sequence, format_sequence
from mapposeformer.filter import FilterParams
from mapposeformer.model import MapPoseFormer
from mapposeformer.model.attention import unpack_attention
from mapposeformer.model.geometric import GeometricBaseline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "checkpoint",
        help="the learned backend, and the config both backends read",
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument(
        "--backend", default="learned", choices=["learned", "geometric"]
    )
    ap.add_argument(
        "--sigma", type=float, default=1.5, help="geometric kernel width, m"
    )
    ap.add_argument(
        "--iters", type=int, default=1, help="geometric re-match passes"
    )
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--limit", type=int, default=0, help="scenes; 0 is all")
    ap.add_argument("overrides", nargs="*", help="section.field=value")
    args = ap.parse_args()

    # `filter.` overrides are peeled off first: they belong to the loop rather
    # than to the checkpoint's config, and passing them through `with_overrides`
    # would ask a model config for a field it has never had.
    fp, rest = FilterParams(), []
    for o in args.overrides:
        if o.startswith("filter."):
            field, _, value = o[len("filter.") :].partition("=")
            fp = replace(fp, **{field: float(value)})
        else:
            rest.append(o)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = with_overrides(upgrade(ckpt["config"]), parse_overrides(rest))
    if args.backend == "geometric":
        model = GeometricBaseline(cfg.model, args.sigma, args.iters)
        label = f"geometric sigma={args.sigma} iters={args.iters}"
    else:
        model = MapPoseFormer(cfg.model)
        model.load_state_dict(unpack_attention(ckpt["model"]))
        label = args.checkpoint
    model.to(args.device).eval()

    source = build_dataset(cfg.data, args.split)
    print(f"{label}  split={args.split}")
    print(
        format_sequence(
            eval_sequence(model, source, fp, args.device, args.limit)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
