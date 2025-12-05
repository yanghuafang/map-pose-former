"""A split materialised once, read back as tensor slices.

Generating a sample costs about 5 ms -- world, map crop, detections, two
warped history frames -- and a run asks for 2.37 million of them. The
dataloader is what caps training throughput; the model is not.

**The cache is exact, not an approximation.** A materialised split is drawn
once, so every epoch reads the same samples. That makes `data.cache_dir` and
`data.augment` opposite choices: this class has no `set_epoch`, so the
trainer's per-epoch reseed is a no-op on it, and `_loader` treats a cached
split as un-augmented. A signature stored beside the tensors makes a cache
built for another configuration detectable -- see `tools/build_cache.py`.

Sequence runs do **not** use this. The closed loop calls `sample_at` with a
prior it chooses per frame, which by definition cannot be precomputed, so
`build_dataset` hands those callers the generator.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    """@param path The ``<split>.pt`` written by ``tools/build_cache.py``."""

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"no cache at {p}; build it with tools/build_cache.py"
            )
        # mmap so forty dataloader workers share one copy of 0.88 GiB rather
        # than each paging in their own.
        blob = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        self.signature = blob.pop("signature", {})
        self.n = int(blob.pop("n"))
        self._store = blob

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        # `clone` because a mmap slice handed to a pinning dataloader is a view
        # onto the file, and pinning it would fault the whole thing in.
        return {k: v[i].clone() for k, v in self._store.items()}
