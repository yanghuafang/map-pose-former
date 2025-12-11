#!/usr/bin/env python3
"""Time one forward pass, under the protocol M4's table is reported with.

    tools/latency.py runs/base/best.pt
    tools/latency.py runs/pruned/best.pt --quantize

Batch 1, CUDA graphs where they capture, 200 warmup iterations, then p50 and
p99 over 1000 -- the same for every row including the fp32 baseline. Fixed in
advance because a latency number without a protocol is unfalsifiable, and
because the temptation to report the best of three is strongest exactly when a
technique has not paid.

p99 beside p50, since a localizer runs in a control loop: a median that fits
the budget and a tail that does not is a system that misses frames, and the
median alone cannot say so.

**`--quantize` will be slower, and that is not a bug.** It simulates INT8 by
rounding and coming straight back, so it adds arithmetic and removes none: it
prices quantization in *accuracy*. Speed needs integer kernels, which needs
TensorRT, which is M5. Reporting a fake-quant latency as an INT8 latency would
be the single easiest way to publish a wrong number from this repository.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.config import upgrade
from mapposeformer.data import build_dataset
from mapposeformer.model.attention import unpack_attention
from mapposeformer.model.model import MapPoseFormer
from mapposeformer.prune import apply_plan, parameter_count
from mapposeformer.quantize import QuantParams, calibrate, quantize


def _load(path: str, device: str):
    """@brief A checkpoint's model, pruned as it was saved.
    @param path Checkpoint. @param device Where to put it.
    @return ``(model, config)``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = upgrade(ckpt["config"])
    model = MapPoseFormer(cfg.model)
    if ckpt.get("prune_plan"):
        apply_plan(model, ckpt["prune_plan"])
    model.load_state_dict(unpack_attention(ckpt["model"]))
    return model.to(device).eval(), cfg


def _percentiles(samples: list[float]) -> tuple[float, float]:
    """@brief p50 and p99 in milliseconds.
    @param samples Per-iteration seconds. @return ``(p50_ms, p99_ms)``."""
    ordered = sorted(samples)
    p50 = statistics.median(ordered) * 1e3
    p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))] * 1e3
    return p50, p99


def main() -> int:
    """@brief Time it and print one row. @return Process exit status."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--quantize", action="store_true", help="simulate INT8")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--no-graphs", action="store_true", help="skip CUDA graphs")
    args = ap.parse_args()

    model, cfg = _load(args.checkpoint, args.device)
    params, _ = parameter_count(model)

    dataset = build_dataset(cfg.data, "val")
    sample = dataset[0]
    batch = {
        k: v.unsqueeze(0).to(args.device) for k, v in sample.items()
    }  # batch 1, the deployment shape

    if args.quantize:
        quantize(model, QuantParams())
        calibrate(model, [batch] * 4, limit=4)

    graphed = False
    with torch.no_grad():
        for _ in range(args.warmup):
            model(batch)
        if args.device == "cuda":
            torch.cuda.synchronize()

        run = lambda: model(batch)  # noqa: E731
        if args.device == "cuda" and not args.no_graphs:
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    model(batch)
                run = graph.replay
                graphed = True
            except Exception as exc:  # capture is best-effort, not required
                why = type(exc).__name__
                print(f"  CUDA graph capture failed ({why}); timing eagerly")

        samples = []
        for _ in range(args.iters):
            if args.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            run()
            if args.device == "cuda":
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)

    p50, p99 = _percentiles(samples)
    how = "cuda graphs" if graphed else "eager"
    note = "  (simulated INT8)" if args.quantize else ""
    print(f"{args.checkpoint}  batch 1, {how}, {args.iters} iters")
    print(f"  params  {params / 1e6:.2f}M{note}")
    print(f"  p50     {p50:7.3f} ms   ({1e3 / p50:.0f} frames/s)")
    print(f"  p99     {p99:7.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
