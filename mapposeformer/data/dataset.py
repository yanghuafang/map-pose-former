"""What a dataset is here, and which one a config asks for.

Both sources emit the same fixed-shape tensors -- ``docs/ARCHITECTURE.md`` has
the table -- so nothing downstream branches on which one it holds. That is the
point of the split: the trainer, the losses and the evaluator cannot tell a
generated road from a surveyed one, which is what makes a number measured on
one comparable to a number measured on the other.

``DataParams`` has lived here since there was one source, because it describes
*data* rather than how the geometry was produced. ``build_dataset`` arrives now,
with the second reader: a factory written before there was anything to choose
between would have been a seam invented rather than named.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch.utils.data import Dataset

from mapposeformer.data.sample import SampleParams
from mapposeformer.data.world import WorldParams


@dataclass
class DataParams:
    """How many scenes, how densely sampled, and under what noise."""

    source: str = "synthetic"
    """``synthetic`` or ``nuscenes``. The two produce identical tensors; what
    differs is whether the map is generated or surveyed. See
    ``data/nuscenes.py`` and ``docs/ROADMAP.md`` (M2a)."""
    detections_dir: str = ""
    """Where a detector wrote its output, one ``.npz`` per scene. Empty means
    detections are cut from the map instead -- the synthetic path, which is
    M2a. See ``data/detections.py`` for the format and why the detector cannot
    run in this environment."""
    nuscenes_cache: str = ""
    """Where ``tools/prepare_nuscenes.py`` wrote its output. Required when
    ``source`` is ``nuscenes``, unused otherwise."""
    num_scenes: dict[str, int] = field(
        default_factory=lambda: {"train": 400, "val": 40, "test": 80}
    )
    """Scenes per split. Synthetic only; nuScenes has as many as the
    geographic split leaves it."""
    frame_stride: int = 4
    """Trajectory poses between consecutive frames. At the default 2 m pitch
    this is 8 m, far enough apart that neighbouring frames are not near
    duplicates."""
    edge_margin: int = 4
    """Frames skipped at each end of a scene, where the road runs out."""
    world: WorldParams = field(default_factory=WorldParams)
    sample: SampleParams = field(default_factory=SampleParams)
    augment: bool = True
    """Redraw the prior error and detection noise every epoch on the training
    split. Validation never does, so its numbers compare across epochs."""


def build_dataset(params: DataParams, split: str) -> Dataset:
    """@brief The dataset named by ``params.source``.

    One place decides, so nothing downstream branches on which dataset it is
    holding -- the shapes are identical and the trainer must not care.

    Both readers are imported inside the branch rather than at module scope.
    The nuScenes one pulls in a JSON map expansion the synthetic path never
    touches, and importing either from here at module scope would close a cycle
    back through ``DataParams``.

    @param params Data configuration.
    @param split ``train``, ``val`` or ``test``.
    @return A ``torch.utils.data.Dataset`` of anchored samples.
    @throws ValueError If ``source`` names neither reader.
    """
    if params.source == "synthetic":
        from mapposeformer.data.synthetic import SyntheticDataset

        return SyntheticDataset(params, split)
    if params.source == "nuscenes":
        from mapposeformer.data.nuscenes import NuScenesDataset

        return NuScenesDataset(params, split)
    raise ValueError(f"unknown data.source {params.source!r}")
