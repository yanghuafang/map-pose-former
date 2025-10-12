"""The numbers in the docs must not drift from the numbers in the code.

``docs/ARCHITECTURE.md`` publishes the input and output shapes as a table, and
a reader who cannot trust it is worse off than one who was handed nothing --
this project's claim is that its documentation is accurate, so the claim is
tested. Every one of these was already wrong once: the map budget grew from 56
elements to 72 and the crop shrank from 55 m to 50 m, and the table kept saying
otherwise through a green test suite.

Checked by substring rather than by parsing the table. A regex over Markdown
would fail on formatting changes that harm nobody, and the failure message here
is the exact string that should have been present.
"""

from pathlib import Path

from mapposeformer.config import Config
from mapposeformer.losses import LossParams

ROOT = Path(__file__).resolve().parent.parent


def test_architecture_shapes_match_the_config():
    doc = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
    c = Config()
    sp, gp = c.data.sample, c.model.grid
    m, d, p, h = (
        sp.max_map_elements,
        sp.max_det_elements,
        sp.points_per_element,
        sp.history,
    )
    expected = {
        "map_pts": f"`({m}, {p}, 2)`",
        "map_pmask": f"`({m}, {p})` bool",
        "map_cls": f"`({m},)`",
        "det_pts": f"`({d}, {p}, 2)`",
        "hist_pts": f"`({h}, {d}, {p}, 2)`",
        "hist_rel": f"`({h}, 3)`",
        "map crop radius": f"{sp.map_radius_m:g} m radius",
        "volume logits": f"`(B, {gp.num_x * gp.num_y * gp.num_yaw})`",
        "volume grid": f"{gp.num_x} × {gp.num_y} × {gp.num_yaw} grid",
        # Every frame's detections, the history included: that width is the
        # whole visible consequence of temporal fusion, and it was wrong in
        # the table for exactly as long as the table said 256.
        "assignment matrix": f"`(B, {(1 + h) * d * p}, {m * p})`",
        "refinement passes": f"`(B, {c.model.refine_iters}, 3)`",
    }
    stale = {k: v for k, v in expected.items() if v not in doc}
    assert not stale, f"docs/ARCHITECTURE.md no longer states: {stale}"


def test_training_doc_matches_the_loss_defaults():
    doc = (ROOT / "docs" / "TRAINING.md").read_text()
    sp = Config().data.sample
    # The history is part of the width: the model matches this frame's
    # detections *and* the previous frames', warped here.
    det, mp = (
        (1 + sp.history) * sp.max_det_elements * sp.points_per_element,
        sp.max_map_elements * sp.points_per_element,
    )
    entries = f"{det} × {mp} assignment entries"
    radius = f"within {LossParams().match_radius_m:g} m"
    stale = [s for s in (entries, radius) if s not in doc]
    assert not stale, f"docs/TRAINING.md no longer states: {stale}"


def test_the_two_nuscenes_class_lists_agree():
    """`classes.py` states what nuScenes has; `nuscenes.py` reads it.

    They were written months apart and drifted: the first still claimed stop
    lines and no point landmarks after the reader had learned that nuScenes'
    stop-line annotation is unusable and that it carries 307 traffic lights.
    A wrong list here is invisible -- it is a *claim* about a dataset, so
    nothing fails, and the synthetic ablation standing in for nuScenes stands
    in for the wrong thing.
    """
    from mapposeformer.data.classes import NUSCENES_AVAILABLE
    from mapposeformer.data.nuscenes import NUSCENES_CLASSES

    assert sorted(int(c) for c in NUSCENES_AVAILABLE) == sorted(
        int(c) for c in NUSCENES_CLASSES
    )
