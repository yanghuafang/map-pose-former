#!/usr/bin/env python3
"""Materialise a split once, so training stops regenerating it every epoch.

    tools/build_cache.py --config configs/synth_base.yaml --split train
    tools/build_cache.py --config configs/synth_base.yaml --split val \
        --split test

Every sample in this project is *generated* -- build the synthetic world, crop
the map, simulate detections, warp two history frames -- at about 5 ms each.
An arm is 59 200 samples over 40 epochs, so it pays that cost 2.37 million
times, and the dataloader caps throughput at roughly 64 frames per second
while the GPU could do 105 to 430 depending on token mode. The model is not
the expensive part; making its input is.

**The cache is exact, not an approximation, and only because of a bug.**
`SyntheticDataset.set_epoch` exists to reseed the sample noise per epoch, and
nothing in the tree ever calls it -- so `_epoch` is 0 for the life of every
run and all 40 epochs see byte-identical data. That makes a single
materialisation equivalent to what training already does. If `set_epoch` is
ever wired up, this cache becomes wrong and `DataParams.cache_dir` must be
cleared: augmentation and caching are opposite choices, and the project has
never measured whether augmentation helps.

The whole training split is 0.88 GiB, so it is stored as one file of stacked
tensors per split and memory-mapped on load. Indexing is then a slice.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.config import load_config, parse_overrides
from mapposeformer.data import build_dataset


#: Written beside the tensors so a stale cache is detectable rather than
#: silently wrong. A cache built for different scenes or a different sampler
#: is not a cache, it is a different dataset.
def signature(cfg, split: str) -> dict:
    d = cfg.data
    return {
        "split": split,
        "source": d.source,
        "num_scenes": dict(d.num_scenes),
        "frame_stride": d.frame_stride,
        "augment": d.augment,
        "sample": repr(d.sample),
        "world": repr(getattr(d, "world", "")),
    }


def build(cfg, split: str, out: Path) -> None:
    ds = build_dataset(cfg.data, split)
    n = len(ds)
    first = ds[0]
    store = {
        k: torch.empty((n, *tuple(v.shape)), dtype=v.dtype)
        for k, v in first.items()
    }
    t0 = time.perf_counter()
    for i in range(n):
        s = ds[i]
        for k, v in s.items():
            store[k][i] = v
        if i % 5000 == 0 and i:
            rate = i / (time.perf_counter() - t0)
            print(
                f"  {i}/{n}  {rate:.0f}/s  eta {(n - i) / rate / 60:.1f} min",
                flush=True,
            )
    took = time.perf_counter() - t0
    size = sum(v.numel() * v.element_size() for v in store.values())

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": signature(cfg, split), "n": n, **store}, out)
    print(
        f"{split}: {n} samples, {size / 2**30:.2f} GiB,"
        f" built in {took / 60:.1f} min -> {out}"
    )

    # A cache nobody checked is a liability. Re-read it and compare a sample
    # of indices against freshly generated ones, including the last, because
    # an off-by-one at the end is the classic way this goes wrong.
    back = torch.load(out, map_location="cpu", weights_only=False)
    checks = sorted({0, 1, n // 3, n // 2, n - 2, n - 1})
    for i in checks:
        want = ds[i]
        for k, v in want.items():
            if not torch.equal(back[k][i], v):
                raise SystemExit(f"VERIFY FAILED at index {i}, key {k}")
    print(f"  verified {len(checks)} indices against freshly generated samples")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", action="append", default=None)
    ap.add_argument("--out", default="", help="cache directory")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    splits = args.split or ["train", "val"]
    root = Path(args.out or cfg.data.cache_dir or "cache")
    for s in splits:
        build(cfg, s, root / f"{s}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
