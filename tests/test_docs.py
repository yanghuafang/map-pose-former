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
    m, d, p = sp.max_map_elements, sp.max_det_elements, sp.points_per_element
    expected = {
        "map_pts": f"`({m}, {p}, 2)`",
        "map_pmask": f"`({m}, {p})` bool",
        "map_cls": f"`({m},)`",
        "det_pts": f"`({d}, {p}, 2)`",
        "map crop radius": f"{sp.map_radius_m:g} m radius",
        "volume logits": f"`(B, {gp.num_x * gp.num_y * gp.num_yaw})`",
        "volume grid": f"{gp.num_x} × {gp.num_y} × {gp.num_yaw} grid",
        "assignment matrix": f"`(B, {d * p}, {m * p})`",
    }
    stale = {k: v for k, v in expected.items() if v not in doc}
    assert not stale, f"docs/ARCHITECTURE.md no longer states: {stale}"


def test_training_doc_matches_the_loss_defaults():
    doc = (ROOT / "docs" / "TRAINING.md").read_text()
    sp = Config().data.sample
    det, mp = (
        sp.max_det_elements * sp.points_per_element,
        sp.max_map_elements * sp.points_per_element,
    )
    entries = f"{det} × {mp} assignment entries"
    radius = f"within {LossParams().match_radius_m:g} m"
    stale = [s for s in (entries, radius) if s not in doc]
    assert not stale, f"docs/TRAINING.md no longer states: {stale}"


def test_readme_states_the_real_test_count():
    """The README tells a first-time reader what ``ci.sh`` should print. A
    stale count is a small lie in the first thing anyone runs."""
    count = sum(
        line.startswith("def test_")
        for f in sorted((ROOT / "tests").glob("test_*.py"))
        for line in f.read_text().splitlines()
    )
    readme = (ROOT / "README.md").read_text()
    assert f"# {count} tests" in readme, f"README should say '# {count} tests'"
