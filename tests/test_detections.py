"""The contract a detector must satisfy, and the path that consumes it.

Exercised with a stub rather than a real mapper: MapTR and StreamMapNet need
`mmcv 1.x` and Python 3.10, so they cannot run in this environment by design.
What is tested here is the part that lives in this repository -- the format,
its validation, and that a file of detections actually reaches the model.
"""

import numpy as np
import pytest
import torch

from mapposeformer.data.detections import load_scene, plausibility, validate


def _good(n=5, p=8):
    return {
        "frame": np.zeros(n, dtype=np.int64),
        "cls": np.zeros(n, dtype=np.int64),
        "conf": np.full(n, 0.8, dtype=np.float32),
        "pts": np.zeros((n, p, 2), dtype=np.float32),
        "npts": np.full(n, p, dtype=np.int64),
    }


def test_validate_accepts_a_well_formed_file():
    validate(_good())


@pytest.mark.parametrize(
    "break_it, message",
    [
        (lambda d: d.pop("conf"), "missing"),
        (lambda d: d.update(cls=d["cls"][:2]), "length"),
        (lambda d: d.update(cls=d["cls"] + 99), "class outside"),
        (lambda d: d.update(conf=d["conf"] * 5), "confidence outside"),
        (lambda d: d.update(npts=d["npts"] * 100), "npts outside"),
        (lambda d: d.update(pts=d["pts"][..., 0]), "must be"),
    ],
)
def test_validate_rejects_what_it_should(break_it, message):
    """Each of these produces a file that would otherwise train to nonsense."""
    data = _good()
    break_it(data)
    with pytest.raises(ValueError, match=message):
        validate(data)


def test_a_detection_file_reaches_the_model(tmp_path):
    """The end-to-end path, with detections that are trivially right.

    Built by taking the map itself as the detector's output, so the true
    correction must align them almost perfectly -- which makes this a test of
    the plumbing rather than of any detector's accuracy.
    """
    from mapposeformer.data.sample import SampleParams, build_sample
    from mapposeformer.data.world import build_world, chunk_for_map

    world = build_world(0)
    chunked = chunk_for_map(world, 12.0, 2.0)
    sp = SampleParams(history=0)
    truth = build_sample(world, chunked, 40, 7, sp)

    # The detector "reports" exactly what the synthetic one saw.
    valid = truth["det_pmask"].any(-1)
    n = int(valid.sum())
    np.savez(
        tmp_path / "scene.npz",
        frame=np.full(n, 40, dtype=np.int64),
        cls=truth["det_cls"][valid].numpy().astype(np.int64),
        conf=truth["det_conf"][valid].numpy().astype(np.float32),
        pts=truth["det_pts"][valid].numpy().astype(np.float32),
        npts=truth["det_pmask"][valid].sum(-1).numpy().astype(np.int64),
    )

    scene = load_scene(tmp_path / "scene.npz")
    replayed = build_sample(world, chunked, 40, 7, sp, detector=scene.frame)

    assert int(replayed["det_pmask"].any(-1).sum()) == n
    assert plausibility(replayed) > 0.7


def test_plausibility_catches_a_wrong_frame_convention():
    """A mirrored Y axis validates cleanly and is completely wrong.

    This is the failure `validate` cannot see and the reason `plausibility`
    exists: the file is well formed, the shapes are right, and the model would
    train to a confidently mirrored answer.
    """
    from mapposeformer.data.sample import SampleParams, build_sample
    from mapposeformer.data.world import build_world, chunk_for_map

    world = build_world(0)
    chunked = chunk_for_map(world, 12.0, 2.0)
    sample = build_sample(world, chunked, 40, 7, SampleParams(history=0))
    honest = plausibility(sample)

    mirrored = dict(sample)
    mirrored["det_pts"] = sample["det_pts"] * torch.tensor([1.0, -1.0])
    assert honest > 0.7
    assert plausibility(mirrored) < honest - 0.3
