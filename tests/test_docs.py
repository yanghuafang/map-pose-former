"""The shapes in the docs must not drift from the shapes in the code.

``docs/DATASET.md`` publishes the sample's tensor shapes as a table, and a
reader who cannot trust it is worse off than one who was handed nothing. A
constant repeated in prose drifts from the constant in the code, and a test
suite that never reads the prose stays green while it happens. This project's
claim is that its documentation is accurate, so the claim is tested.

Checked by substring rather than by parsing the table. A regex over Markdown
would fail on formatting changes that harm nobody, and the failure message here
is the exact string that should have been present.
"""

from pathlib import Path

from mapposeformer.config import Config
from mapposeformer.metrics import Calibration

ROOT = Path(__file__).resolve().parent.parent


def test_dataset_doc_states_the_shapes_the_config_produces():
    doc = (ROOT / "docs" / "DATASET.md").read_text()
    sp = Config().data.sample
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
        "det_pmask": f"`({d}, {p})` bool",
        "det_sigma": f"`({d}, 2)`",
        "hist_pts": f"`({h}, {d}, …)`",
        "hist_rel": f"`({h}, 3)`",
        "map element budget": f"{m} map elements",
        "map crop radius": f"{sp.map_radius_m:g} m crop",
    }
    stale = {k: v for k, v in expected.items() if v not in doc}
    assert not stale, f"docs/DATASET.md no longer states: {stale}"


def test_the_readme_states_the_input_and_output_contract():
    """The README is the first thing read, so its shapes drift the worst.

    It publishes the contract as a table -- what goes in, what comes out --
    and a reader who cannot trust that has no reason to trust the rest.
    """
    doc = (ROOT / "README.md").read_text()
    sp = Config().data.sample
    m, d, p, h = (
        sp.max_map_elements,
        sp.max_det_elements,
        sp.points_per_element,
        sp.history,
    )
    expected = {
        "map elements": f"`({m}, {p}, 2)`",
        "detections": f"`({d}, {p}, 2)`",
        "history": f"`({h}, {d}, {p}, 2)`",
        "delta": "`(3,)`",
        "prior error": f"{sp.prior.sigma_long_m:g} m along track",
        # The calibration target is derived, not remembered: a chi-square with
        # three degrees of freedom has median 2.366, so per degree of freedom a
        # perfect estimator scores this and not 1.0.
        "calibration target": f"{Calibration.CHI2_MEDIAN:.3f} is calibrated",
    }
    stale = {k: v for k, v in expected.items() if v not in doc}
    assert not stale, f"README.md no longer states: {stale}"


def test_the_quoted_token_counts_are_the_ones_the_data_produces():
    """The saving element tokens buy is quoted in four places.

    It drifted within a single afternoon -- a single-frame count in two module
    docstrings against the with-history count in the roadmap, which made the
    same ratio read as 63x, 64x and 167x. The numbers are derived here so the
    next edit has to agree with the config or fail.
    """
    sp = Config().data.sample
    elements = (1 + sp.history) * sp.max_det_elements + sp.max_map_elements
    points = elements * sp.points_per_element
    ratio = round((points / elements) ** 2)

    wanted = {
        ROOT / "docs" / "ROADMAP.md": [
            f"{elements} tokens",
            f"{ratio}× smaller",
        ],
        ROOT / "mapposeformer" / "model" / "encoder.py": [
            f"means {elements}",
            f"{ratio}x less",
        ],
        ROOT / "mapposeformer" / "model" / "attention.py": [
            f"{elements} element tokens",
            f"{ratio}x smaller",
        ],
    }
    stale = {
        f.name: [s for s in strings if s not in f.read_text()]
        for f, strings in wanted.items()
    }
    stale = {k: v for k, v in stale.items() if v}
    assert not stale, f"token counts have drifted: {stale}"


def test_the_architecture_doc_states_the_shapes_it_claims():
    """``ARCHITECTURE.md`` publishes the tensor widths the stages pass along.

    Same reason as the table above: a document that repeats a constant will
    drift from it, and a reader who cannot trust the shapes is worse off than
    one who was handed none.
    """
    doc = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
    c = Config()
    sp = c.data.sample
    det = (1 + sp.history) * sp.max_det_elements
    mp, p = sp.max_map_elements, sp.points_per_element
    elements = det + mp
    expected = {
        "map in": f"`(B, {mp}, {p}, 2)`",
        "detections in": f"`(B, {det}, {p}, 2)`",
        "tokens": f"`(B, {elements}, {c.model.dim})`",
        "element assignment": f"`(B, {det}, {mp})`",
        "point assignment": f"`(B, {det * p}, {mp * p})`",
        # The saving element tokens buy, quoted in prose as well as in the
        # table above. It was once written three different ways. The ratio is
        # points_per_element squared, because pooling replaces exactly `p`
        # point tokens with one element token and attention is quadratic.
        "point token count": f"{elements * p:,} tokens".replace(",", " "),
        "element token count": f"elements* is {elements}",
        "attention ratio": f"{p**2}× less",
    }
    stale = {k: v for k, v in expected.items() if v not in doc}
    assert not stale, f"docs/ARCHITECTURE.md no longer states: {stale}"
