"""The nuScenes reader, where a cache exists to read.

Skipped without one, because building it needs the 3 GB of metadata and this
suite runs on a laptop. What is checked here is what a wrong reader would get
wrong quietly: the split must not leak, and the geometry must satisfy the same
invariant the synthetic data does.

Point `MPF_NUSCENES_CACHE` at `tools/prepare_nuscenes.py`'s output to run them.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from mapposeformer import geometry as G
from mapposeformer.config import Config
from mapposeformer.data import build_dataset

CACHE = os.environ.get("MPF_NUSCENES_CACHE", "")
pytestmark = pytest.mark.skipif(
    not CACHE or not Path(CACHE).is_dir(), reason="no nuScenes cache"
)


def _config():
    cfg = Config()
    cfg.data.source = "nuscenes"
    cfg.data.nuscenes_cache = CACHE
    return cfg


def test_the_splits_are_geographically_disjoint():
    """Train and test must not share ground.

    The official nuScenes split does, which is why this project does not use
    it: a localizer evaluated on roads it trained on is partly reciting. The
    bound is the map query radius -- if the nearest test pose is further from
    every train pose than the model can see, no map element is shared.
    """
    manifest = json.loads((Path(CACHE) / "manifest.json").read_text())
    where = manifest["locations"]

    def poses(split: str, city: str) -> torch.Tensor | None:
        rows = [
            torch.load(
                Path(CACHE) / "scenes" / f"{n}.pt", weights_only=False
            ).trajectory[:, :2]
            for n in manifest["splits"][split]
            if where[n] == city
        ]
        return torch.cat(rows) if rows else None

    # **Per city.** Every map has its own origin, so two scenes in different
    # cities routinely share an (x, y) while being continents apart. Comparing
    # them together reports a false collision -- which it did, at 3.1 m.
    radius = Config().data.sample.map_radius_m
    checked = 0
    for city in sorted(set(where.values())):
        train, test = poses("train", city), poses("test", city)
        if train is None or test is None:
            continue
        closest = float(torch.cdist(test[::5], train[::5]).min())
        assert closest > radius, f"{city}: test within {closest:.1f} m of train"
        checked += 1
    assert checked >= 2, "not enough cities had both splits to be a real check"


def test_the_true_correction_lands_detections_on_the_real_map():
    """The invariant, on surveyed geometry rather than generated geometry.

    Same check as `tests/test_data.py` makes on the synthetic world. If it
    fails here the reader has mixed up a frame, a rotation or a units
    convention, and every metric downstream would keep looking healthy.
    """
    dataset = build_dataset(_config().data, "test")
    fractions = []
    for i in range(0, min(len(dataset), 240), 24):
        sample = dataset[i]
        points = sample["det_pts"][sample["det_pmask"]]
        if points.numel() == 0:
            continue
        aligned = G.transform_points(sample["delta"], points)
        segments = []
        for e in range(sample["map_pts"].shape[0]):
            q = sample["map_pts"][e][sample["map_pmask"][e]]
            if q.shape[0] >= 2:
                segments.append((q[:-1], q[1:]))
        if not segments:
            continue
        a = torch.cat([s[0] for s in segments])
        b = torch.cat([s[1] for s in segments])
        d = b - a
        t = (
            ((aligned[:, None] - a) * d).sum(-1)
            / d.square().sum(-1).clamp_min(1e-9)
        ).clamp(0, 1)
        dist = (aligned[:, None] - (a + t[..., None] * d)).norm(dim=-1).min(-1)
        fractions.append((dist.values < 1.0).float().mean())
    assert fractions, "no frame produced a detection"
    assert float(torch.stack(fractions).mean()) > 0.7


def test_the_sample_shape_is_the_synthetic_one():
    """The model must not be able to tell which dataset it was handed."""
    cfg = _config()
    real = build_dataset(cfg.data, "test")[0]
    cfg.data.source = "synthetic"
    fake = build_dataset(cfg.data, "train")[0]
    assert set(real) == set(fake)
    for key in sorted(real):
        assert real[key].shape == fake[key].shape, key
        assert real[key].dtype == fake[key].dtype, key


def test_a_sample_is_the_same_in_every_process():
    """The dataset must not depend on the interpreter it is built in.

    Python salts string hashing per process, so a scene name run through
    `hash()` gives a different answer in every process and every dataloader
    worker. Seeded that way, two evaluations of one checkpoint draw different
    detector noise and disagree by more than the effect being measured -- which
    is how a class ablation came to report that removing evidence improved
    accuracy, and survived three unrelated bug fixes before the seed was
    suspected.

    Checked in a subprocess, because a fresh interpreter is exactly what the
    bug needed and an in-process check cannot see it.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(f"""
        from mapposeformer.config import Config
        from mapposeformer.data import build_dataset
        cfg = Config()
        cfg.data.source = "nuscenes"
        cfg.data.nuscenes_cache = {CACHE!r}
        s = build_dataset(cfg.data, "test")[7]
        print(float(s["det_pts"].sum()), float(s["delta"].sum()))
    """)
    runs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(runs) == 1, f"sample differs between processes: {runs}"


def test_the_map_and_the_detection_source_are_chunked_apart():
    """The two sides must not share element boundaries.

    :func:`~mapposeformer.data.world.chunk_for_map` explains why: a map is cut
    at survey boundaries and a detection at a range, so a shared boundary is a
    landmark at a fixed world position and a perfect along-track anchor.

    The nuScenes reader chunked once at ingest and handed the result to
    ``build_sample`` as *both* the map and the source of detections. Half of
    all lane geometry then arrived with its correspondence already solved --
    identical arclength samples on both sides, point i against point i -- and
    the observability ablation went flat: lane dividers alone scored the best
    longitudinal error of any class subset, against a claim that they carry
    none of it. Nothing failed, because nothing was checked.
    """
    ds = build_dataset(_config().data, "test")
    world, chunked = ds._world(ds.names[0])

    # The element count is the check, not the element length: nuScenes stores
    # short polylines -- 18 m median -- so a length threshold tuned to the
    # synthetic world's 600 m elements says nothing. What went wrong was
    # one-to-one, and that is what would show up here.
    assert chunked.pts.shape[0] > 3 * world.pts.shape[0], (
        f"map has {chunked.pts.shape[0]} elements against the source's "
        f"{world.pts.shape[0]}: it is not chunked apart"
    )
