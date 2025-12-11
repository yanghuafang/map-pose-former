#!/usr/bin/env python3
"""Accuracy against latency, one row per configuration, as one table.

    tools/pareto.py teacher=runs/teacher/best.pt student=runs/base/best.pt \\
        distilled=runs/distilled/best.pt 'pruned=runs/pruned/best.pt' \\
        'pruned+int8=runs/pruned/best.pt:quantize'

Each argument is ``label=checkpoint`` with an optional ``:quantize`` suffix.
Every row is measured the same way in the same process -- the same split, the
same protocol, the same machine -- because a Pareto table assembled from
numbers gathered on different days is a table of anecdotes.

The eval is called rather than parsed. A driver that scrapes another tool's
stdout breaks quietly when the format changes, and reports whatever the regex
happened to match.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.config import upgrade
from mapposeformer.data import build_dataset
from mapposeformer.engine import evaluate
from mapposeformer.model.attention import unpack_attention
from mapposeformer.model.model import MapPoseFormer
from mapposeformer.prune import apply_plan, parameter_count
from mapposeformer.quantize import QuantParams, calibrate, quantize


def _load(path: str, device: str, want_quant: bool, batch):
    """@brief One row's model, pruned and quantized as its spec asks.

    @param path Checkpoint. @param device Where to run.
    @param want_quant Whether to simulate INT8.
    @param batch A calibration batch, or None when not quantizing.
    @return ``(model, config)``.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = upgrade(ckpt["config"])
    model = MapPoseFormer(cfg.model)
    if ckpt.get("prune_plan"):
        apply_plan(model, ckpt["prune_plan"])
    model.load_state_dict(unpack_attention(ckpt["model"]))
    model = model.to(device).eval()
    if want_quant:
        quantize(model, QuantParams())
        calibrate(model, [batch] * 4, limit=4)
    return model, cfg


@torch.no_grad()
def _latency(
    model, batch, warmup: int, iters: int, device: str
) -> tuple[float, float]:
    """@brief p50 and p99 milliseconds at batch 1.
    @param model The model. @param batch A batch-1 input. @param warmup Warmup
    iterations. @param iters Measured iterations. @param device Device.
    @return ``(p50_ms, p99_ms)``."""
    for _ in range(warmup):
        model(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    ordered = sorted(samples)
    return (
        statistics.median(ordered) * 1e3,
        ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))] * 1e3,
    )


def main() -> int:
    """@brief Build the table. @return Process exit status."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rows", nargs="+", help="label=checkpoint[:quantize]")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument(
        "--json",
        help="also write the rows here, for tools/plot_pareto.py. A plot that "
        "re-parsed the table below would break the moment it was reformatted.",
    )
    args = ap.parse_args()

    table = []
    for spec in args.rows:
        label, _, rest = spec.partition("=")
        path, _, flag = rest.partition(":")
        want_quant = flag == "quantize"

        # Built once unquantized, only to read its config and shape a
        # calibration batch; the measured model is built below.
        _, cfg = _load(path, args.device, False, None)
        dataset = build_dataset(cfg.data, args.split)
        one = {k: v.unsqueeze(0).to(args.device) for k, v in dataset[0].items()}
        model, cfg = _load(path, args.device, want_quant, one)

        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
        result = evaluate(model, loader, args.device)
        m = result["metrics"]
        p50, p99 = _latency(model, one, args.warmup, args.iters, args.device)
        params, _ = parameter_count(model)
        table.append(
            (
                label,
                params / 1e6,
                m["all/rmse_trans_m"],
                m["trusted/rmse_trans_m"],
                m["all/recall_0.25m_0.5deg"],
                p50,
                p99,
            )
        )
        print(f"  measured {label}", file=sys.stderr)

    print(f"\n{args.split} split, batch 1 latency, {args.iters} iterations\n")
    print("| | params | trans | trusted | recall @25cm | p50 | p99 |")
    print("|---|---|---|---|---|---|---|")
    for label, p, tr, tru, rec, p50, p99 in table:
        print(
            f"| {label} | {p:.2f} M | {tr:.3f} | {tru:.3f} | "
            f"{rec:.1%} | {p50:.2f} ms | {p99:.2f} ms |"
        )

    if args.json:
        keys = (
            "label",
            "params_m",
            "trans_m",
            "trusted_m",
            "recall",
            "p50_ms",
            "p99_ms",
        )
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "iters": args.iters,
                    "runtime": "pytorch",
                    "rows": [dict(zip(keys, r, strict=True)) for r in table],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
