#!/usr/bin/env python3
"""Where the time goes: data, forward, backward, end to end.

    tools/bench.py --config configs/synth_base.yaml

The training log prints one ``frames_per_s``, and one number cannot say which
part of the step is the limit. Running the three parts apart can: the data
pipeline without a model, the model without a data pipeline, and then the loop
that has both. The end-to-end rate can only be as high as the lower of the
first two, so whichever of them it sits against is the thing to fix.

Timing a GPU needs ``synchronize`` around the measured region, because a CUDA
launch returns before the work is done -- a forward pass looks like 200 us
without it, which is the queueing and not the arithmetic.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.config import load_config, parse_overrides
from mapposeformer.data.synthetic import SyntheticDataset
from mapposeformer.losses import compute_losses
from mapposeformer.model.model import MapPoseFormer


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _time(fn, device: str, iters: int, warmup: int = 5) -> float:
    """Mean seconds per call, warmup excluded."""
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", help="YAML config; omit for the dataclass defaults"
    )
    ap.add_argument(
        "--batches", type=int, default=30, help="batches per section"
    )
    ap.add_argument("overrides", nargs="*", help="section.field=value")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    device = cfg.train.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    bs = cfg.train.batch_size
    print(f"device {device}  batch {bs}  workers {cfg.train.num_workers}")
    if device.startswith("cuda"):
        print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"torch threads {torch.get_num_threads()}\n")

    dataset = SyntheticDataset(cfg.data, "train")
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )

    # --- the generator alone, in this process, no loader machinery --------
    # A separate question from the one below: whether a sample costs what it
    # costs because generating a scene is expensive, or because the loader is
    # asking for a new one every time. The dataset holds a single scene, so a
    # shuffled sampler misses it on nearly every index.
    probe = SyntheticDataset(cfg.data, "train")
    per_scene = len(probe._frames)
    probe[0]
    hit = _time(lambda: probe[1], "cpu", 20)
    miss = [0]

    def one_miss():
        miss[0] += 1
        probe[(miss[0] * per_scene) % len(probe)]

    miss_s = _time(one_miss, "cpu", 20)
    for name, sec in (("cached scene", hit), ("new scene", miss_s)):
        print(
            f"one sample, {name:14s}{sec * 1e3:6.2f} ms"
            f"   ({1 / sec:5.0f}/s/worker)"
        )
    print(f"{per_scene} frames per scene, {len(probe)} frames total\n")

    # --- data alone: drain the loader, touch nothing else ------------------
    # Twice, because the dataset caches scenes and the two passes are the two
    # regimes a run actually spends time in: the first epoch, which builds
    # every scene it touches, and every epoch after it, which does not.
    # Reporting only the cold number would understate a 40-epoch run by more
    # than a factor of two; reporting only the warm one would promise a first
    # epoch that does not arrive.
    def drain(n_batches: int):
        it = iter(loader)
        last = next(it)  # worker startup is not throughput
        t0 = time.perf_counter()
        for _ in range(n_batches):
            last = next(it)
        return last, n_batches * bs / (time.perf_counter() - t0)

    batch, cold = drain(args.batches)
    _, warm = drain(args.batches)
    print(f"data, cold cache     {cold:8.0f} frames/s   (first epoch)")
    print(f"data, warm cache     {warm:8.0f} frames/s   (every epoch after)")
    data_fps = warm

    # --- model alone: one batch, resident, no loader in the loop -----------
    batch = {k: v.to(device) for k, v in batch.items()}
    model = MapPoseFormer(cfg.model).to(device)
    volume = model.volume
    amp = cfg.train.amp
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if amp == "bf16" and device.startswith("cuda")
        else torch.autocast(device_type="cpu", enabled=False)
    )

    def forward():
        with torch.no_grad(), ctx:
            model(batch)

    def step():
        with ctx:
            out = model(batch)
            total, _ = compute_losses(
                out, batch, volume.cells, volume.pitch, cfg.loss
            )
        total.backward()
        model.zero_grad(set_to_none=True)

    # Peak VRAM over one optimizer step, which is what has to fit. Measured
    # after a warm-up step so the allocator has stopped growing, and reported
    # as *allocated* rather than reserved: reserved is the caching allocator's
    # high-water mark and says more about fragmentation than about the model.
    peak_alloc = peak_reserved = 0.0
    if device.startswith("cuda"):
        step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step()
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30

    fwd = _time(forward, device, args.batches)
    fwd_bwd = _time(step, device, args.batches)
    if device.startswith("cuda"):
        params = sum(p.numel() for p in model.parameters())
        print(
            f"model, {params / 1e6:.1f}M params, batch {bs}: "
            f"{peak_alloc:.2f} GiB allocated, {peak_reserved:.2f} GiB reserved"
        )
    for name, sec in (("forward only", fwd), ("forward + backward", fwd_bwd)):
        print(
            f"{name:20s} {bs / sec:8.0f} frames/s   ({sec * 1e3:7.1f} ms/batch)"
        )

    limit = "data" if data_fps < bs / fwd_bwd else "model"
    print(
        f"\nsteady state is {limit}-bound:"
        f" {min(data_fps, bs / fwd_bwd):.0f} frames/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
