"""What a dataset is here: the configuration every source is built from.

``DataParams`` lives beside neither reader on purpose. It describes *data* --
how many scenes, how densely sampled, under what noise -- and none of that is a
property of how the geometry was produced. Putting it in ``synthetic.py`` would
make the second source import the first to ask what a frame stride is.

There is deliberately no factory here yet. One source needs no dispatcher, and
writing one before the second reader exists would be inventing a seam rather
than naming one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mapposeformer.data.sample import SampleParams
from mapposeformer.data.world import WorldParams


@dataclass
class DataParams:
    """How many scenes, how densely sampled, and under what noise."""

    num_scenes: dict[str, int] = field(
        default_factory=lambda: {"train": 400, "val": 40, "test": 80}
    )
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
    split. Validation never does, so its numbers are comparable across epochs."""
