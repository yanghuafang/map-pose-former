"""The stage-0 dataset: procedural scenes, indexed as frames.

Splits are disjoint *seed ranges*, not disjoint frames of shared scenes. A
split that shares geometry measures memorisation; this one cannot, which is the
synthetic stand-in for the geographic split nuScenes needs (its official split
overlaps spatially, and a localizer evaluated on it is partly reciting).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from mapposeformer.data.dataset import DataParams
from mapposeformer.data.sample import build_sample
from mapposeformer.data.world import World, build_world, chunk_for_map

#: Seed offsets that keep the three splits from ever generating the same road.
SPLIT_SEED_BASE = {"train": 0, "val": 10_000_000, "test": 20_000_000}


class SyntheticDataset(Dataset):
    """Frames of procedurally generated road, as anchored point sets."""

    def __init__(self, params: DataParams, split: str):
        if split not in SPLIT_SEED_BASE:
            raise ValueError(f"unknown split {split!r}")
        self.p = params
        self.split = split
        self.base = SPLIT_SEED_BASE[split]
        self.n_scenes = params.num_scenes[split]
        self._epoch = 0
        self._cache: dict[int, tuple[World, World]] = {}
        probe, _ = self._world(0)
        lo = params.edge_margin
        hi = probe.trajectory.shape[0] - params.edge_margin
        self._frames = list(range(lo, hi, params.frame_stride))

    def set_epoch(self, epoch: int) -> None:
        """Reseed the noise for a new epoch (training split only)."""
        self._epoch = epoch

    def _world(self, scene: int) -> tuple[World, World]:
        """The scene, and the chunked map derived from it.

        Every scene is kept, for the length of the run. A ``(world, chunked)``
        pair is 130 kB, so the largest split here is 104 MB in a worker, and
        the rebuild it avoids is half of a sample's cost: 35 ms against 18 ms
        measured on the training box. Training shuffles, so a sampler asks for
        a different scene at almost every index and a one-scene cache misses
        almost every time -- it would only pay under the sequential order that
        nothing here uses.

        Unbounded is deliberate rather than careless. The bound is the split,
        which is a config value a reader can see, and an eviction policy would
        be machinery guarding a hundred megabytes.
        """
        if scene not in self._cache:
            world = build_world(self.base + scene, self.p.world)
            chunked = chunk_for_map(
                world, self.p.world.map_chunk_m, self.p.world.step_m
            )
            self._cache[scene] = (world, chunked)
        return self._cache[scene]

    def __len__(self) -> int:
        return self.n_scenes * len(self._frames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        scene, k = divmod(idx, len(self._frames))
        frame = self._frames[k]
        epoch = self._epoch if (self.p.augment and self.split == "train") else 0
        seed = ((self.base + scene) * 100_003 + frame) * 97 + epoch
        world, chunked = self._world(scene)
        return build_sample(world, chunked, frame, seed, self.p.sample)
