#!/usr/bin/env python3
"""Where the input side's time goes: generating scenes, and the loader.

    tools/bench.py                     # the dataclass defaults
    tools/bench.py --config configs/synth_base.yaml

The training log prints one ``frames_per_s``, and one number cannot say what
sets it. A step can never run faster than its data arrives, so measuring the
data alone puts a ceiling on the whole loop -- and on this project it has been
the ceiling, which is why it is worth measuring before anything else exists.

Two comparisons, because each has been the answer at a different time. A
cached scene against a new one separates the cost of building the synthetic
world from the cost of the loader asking for a fresh scene on nearly every
index. A cold cache against a warm one separates the first epoch from all the
epochs after it -- reporting only the cold number understates a 40-epoch run by
more than half, and only the warm one promises a first epoch that never comes.
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
from mapposeformer.data import build_dataset


def _time(fn, iters: int, warmup: int = 5) -> float:
    """Mean seconds per call, warmup excluded."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
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

    dataset = build_dataset(cfg.data, "train")
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
    probe = build_dataset(cfg.data, "train")
    if not hasattr(probe, "frames_of"):
        raise SystemExit(
            "bench.py measures the generator; clear data.cache_dir to run it"
        )
    per_scene = len(probe.frames_of(probe.sequences()[0]))
    probe[0]
    hit = _time(lambda: probe[1], 20)
    miss = [0]

    def one_miss():
        miss[0] += 1
        probe[(miss[0] * per_scene) % len(probe)]

    miss_s = _time(one_miss, 20)
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

    _, cold = drain(args.batches)
    _, warm = drain(args.batches)
    print(f"data, cold cache     {cold:8.0f} frames/s   (first epoch)")
    print(f"data, warm cache     {warm:8.0f} frames/s   (every epoch after)")


if __name__ == "__main__":
    raise SystemExit(main())
