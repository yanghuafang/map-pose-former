#!/usr/bin/env python3
"""Time a TensorRT engine under the protocol M4's table used.

    tools/trt_latency.py build/student.engine

Batch 1, 200 warmup iterations, p50 and p99 over 1000 -- the same numbers
tools/latency.py reports for PyTorch, so the two are comparable and the
speedup is a subtraction rather than a claim.

Inputs are the same validation frame every iteration and stay resident on the
device. Timing a copy from host memory would be timing PCIe, and a deployed
localizer already has its detections on the GPU.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.config import Config
from mapposeformer.data import build_dataset


def main() -> int:
    """@brief Time the engine. @return Exit status."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("engine")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(args.engine).read_bytes())
    context = engine.create_execution_context()

    sample = build_dataset(Config().data, "val")[0]
    stream = torch.cuda.Stream()
    held = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            key = name if name in sample else None
            source = (
                sample[key].unsqueeze(0).numpy().astype(dtype)
                if key
                else np.zeros(shape, dtype=dtype)
            )
            held[name] = torch.from_numpy(
                np.ascontiguousarray(source.reshape(shape))
            ).cuda()
        else:
            held[name] = torch.empty(shape, dtype=torch.float32).cuda()
        context.set_tensor_address(name, held[name].data_ptr())

    for _ in range(args.warmup):
        context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    samples = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        context.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)

    ordered = sorted(samples)
    p50 = statistics.median(ordered) * 1e3
    p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))] * 1e3
    print(f"{args.engine}  batch 1, {args.iters} iters")
    print(f"  p50  {p50:7.3f} ms   ({1e3 / p50:.0f} frames/s)")
    print(f"  p99  {p99:7.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
