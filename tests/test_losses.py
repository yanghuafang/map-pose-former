"""The training signal, and the labels it derives rather than reads.

No dataset records which detection came from which map element, so the match
labels are built from the one label there is -- the true correction. These
tests check that derivation, because a silent error in it would train the
matcher to agree with a mistake.
"""

import torch

from mapposeformer import geometry as G
from mapposeformer.config import Config
from mapposeformer.data import build_dataset
from mapposeformer.losses import LossParams, compute_losses, element_truth
from mapposeformer.model import MapPoseFormer, ModelParams


def _batch(n=2):
    ds = build_dataset(Config().data, "train")
    return {k: torch.stack([ds[i][k] for i in range(n)]) for k in ds[0]}


def test_a_detection_is_labelled_with_the_element_it_came_from():
    """Built by hand, so the right answer is known rather than inferred."""
    map_pts = torch.zeros(1, 3, 4, 2)
    map_pts[0, 0] = torch.tensor([[0.0, 0], [2, 0], [4, 0], [6, 0]])
    map_pts[0, 1] = torch.tensor([[0.0, 9], [2, 9], [4, 9], [6, 9]])
    map_pts[0, 2] = torch.tensor([[0.0, 20], [2, 20], [4, 20], [6, 20]])
    map_pmask = torch.ones(1, 3, 4, dtype=torch.bool)

    delta = torch.tensor([[1.5, -0.5, 0.0]])
    # Two detections cut from elements 1 and 2, placed so `delta` lands them.
    det = torch.stack([map_pts[0, 1], map_pts[0, 2]]).unsqueeze(0)
    det = G.transform_points(G.inverse(delta), det.reshape(1, -1, 2)).reshape(
        1, 2, 4, 2
    )
    det_pmask = torch.ones(1, 2, 4, dtype=torch.bool)

    target, has = element_truth(
        det, det_pmask, map_pts, map_pmask, delta, LossParams()
    )
    assert has.tolist() == [[True, True]]
    # The label is a distribution now, so the right element holds all the mass.
    assert target.argmax(-1).tolist() == [[1, 2]]
    assert torch.allclose(target.sum(-1), torch.ones(1, 2), atol=1e-5)
    assert target[0, 0, 1] > 0.99 and target[0, 1, 2] > 0.99


def test_a_detection_spanning_two_map_chunks_is_labelled_with_both():
    """The case a one-hot label cannot express, and the map guarantees.

    The map is chunked at a fixed 12 m and a camera frustum is not, so a
    detected road boundary lands on four map elements and a lane divider on
    two. Labelling that as one element -- or, once it fails a threshold, as
    *no* element -- discards the landmark class that pins lateral position.
    """
    # Two map chunks laid end to end, and one detection spanning both.
    map_pts = torch.zeros(1, 2, 4, 2)
    map_pts[0, 0] = torch.tensor([[0.0, 0], [2, 0], [4, 0], [6, 0]])
    map_pts[0, 1] = torch.tensor([[8.0, 0], [10, 0], [12, 0], [14, 0]])
    map_pmask = torch.ones(1, 2, 4, dtype=torch.bool)

    det = torch.tensor([[[[0.0, 0], [4, 0], [10, 0], [14, 0]]]])
    det_pmask = torch.ones(1, 1, 4, dtype=torch.bool)

    target, has = element_truth(
        det, det_pmask, map_pts, map_pmask, torch.zeros(1, 3), LossParams()
    )
    assert has.tolist() == [[True]], "a detection covering the map is matched"
    assert torch.allclose(target.sum(-1), torch.ones(1, 1), atol=1e-5)
    # Half its points on each chunk, so half the mass on each.
    assert torch.allclose(target[0, 0], torch.tensor([0.5, 0.5]), atol=1e-5)


def test_clutter_is_labelled_as_matching_nothing():
    """The label that lets matchability mean anything."""
    map_pts = torch.zeros(1, 2, 4, 2)
    map_pts[0, 0] = torch.tensor([[0.0, 0], [2, 0], [4, 0], [6, 0]])
    map_pts[0, 1] = torch.tensor([[0.0, 9], [2, 9], [4, 9], [6, 9]])
    map_pmask = torch.ones(1, 2, 4, dtype=torch.bool)

    det = torch.full((1, 1, 4, 2), 60.0)  # nowhere near the map
    det_pmask = torch.ones(1, 1, 4, dtype=torch.bool)
    _, has = element_truth(
        det, det_pmask, map_pts, map_pmask, torch.zeros(1, 3), LossParams()
    )
    assert has.tolist() == [[False]]


def test_the_pose_term_is_zero_on_an_exact_answer():
    batch = _batch()
    M = batch["map_pts"].shape[1]
    out = {
        "delta": batch["delta"].clone(),
        "granularity": "element",
        "elements": torch.full((2, 1, M), 1.0 / M),
        "scores": torch.zeros(2, 1, M),
        "det_matchable": torch.full((2, 1), 0.5),
        "det_pts": batch["det_pts"][:, :1],
        "det_pmask": batch["det_pmask"][:, :1],
    }
    _, parts = compute_losses(out, batch, LossParams())
    assert parts["pose"] < 1e-6


def test_putting_mass_on_the_right_element_lowers_the_match_term():
    """The term has to prefer the truth, which is worth asserting once."""
    batch = _batch()
    torch.manual_seed(0)
    # `tokens="element"` is stated because this test labels with
    # `element_truth`, and the two have to agree on resolution: the element
    # path scores (B, 96, 72) where the point path scores (B, 768, 576).
    # Leaving it to the default made the test depend on what the default
    # happened to be, which is how it broke when the default became `point`.
    model = MapPoseFormer(
        ModelParams(dim=32, layers=1, heads=2, tokens="element")
    ).eval()
    out = model(batch, sigma_m=12.0)
    p = LossParams()

    target, has = element_truth(
        out["det_pts"],
        out["det_pmask"],
        batch["map_pts"],
        batch["map_pmask"],
        batch["delta"],
        p,
    )
    assert has.any(), "no detection matched anything; the test says nothing"

    _, before = compute_losses(out, batch, p)
    # Logits that say exactly what the label says, and nothing else.
    truth = torch.where(target > 0, 20.0, -20.0)
    _, after = compute_losses(dict(out, scores=truth), batch, p)
    assert after["match"] < before["match"]


def test_the_total_is_finite_and_reaches_every_parameter():
    batch = _batch()
    torch.manual_seed(0)
    model = MapPoseFormer(ModelParams(dim=32, layers=1, heads=2))
    total, parts = compute_losses(
        model(batch, sigma_m=12.0), batch, LossParams()
    )
    assert torch.isfinite(total)
    total.backward()
    dead = [
        n
        for n, q in model.named_parameters()
        if q.grad is None or not q.grad.any()
    ]
    assert not dead, f"no gradient reaches: {dead}"
    assert 0.0 <= parts["matched_frac"] <= 1.0


def test_the_loss_survives_an_autocast_forward():
    """Training runs the model in bf16; the loss must still compute.

    This failed on the first GPU run and could not fail on CPU without asking
    for autocast explicitly, which is what this does. Probabilities near zero
    have little left after bf16's eight mantissa bits, and
    ``binary_cross_entropy`` is promoted to fp32 by autocast anyway -- so the
    loss casts rather than inheriting whatever the forward pass used.
    """
    batch = _batch()
    torch.manual_seed(0)
    model = MapPoseFormer(ModelParams(dim=32, layers=1, heads=2))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(batch, sigma_m=12.0)
    total, _ = compute_losses(out, batch, LossParams())
    assert total.dtype is torch.float32
    assert torch.isfinite(total)
    total.backward()


def test_the_one_hot_label_is_still_reachable_as_an_arm():
    """``match_target='best'`` reproduces the labelling that came before.

    Kept as a flag rather than deleted because it is the control: the change
    from one label to a distribution has to be measurable against something,
    and an arm that needs a code edit is not an arm.

    Four chunks and a detection spanning all of them is the road-boundary case
    exactly -- no single element holds half the points, so the one-hot scheme
    calls it unmatched while the distribution keeps every one of them.
    """
    map_pts = torch.zeros(1, 4, 4, 2)
    for i in range(4):
        base = i * 8.0
        map_pts[0, i] = torch.tensor(
            [[base, 0], [base + 2, 0], [base + 4, 0], [base + 6, 0]]
        )
    map_pmask = torch.ones(1, 4, 4, dtype=torch.bool)
    # One point on each chunk -- four, which is what a real road boundary hits.
    det = torch.tensor([[[[0.0, 0], [8, 0], [16, 0], [24, 0]]]])
    det_pmask = torch.ones(1, 1, 4, dtype=torch.bool)
    zero = torch.zeros(1, 3)

    spread, has_spread = element_truth(
        det, det_pmask, map_pts, map_pmask, zero, LossParams()
    )
    _, has_best = element_truth(
        det,
        det_pmask,
        map_pts,
        map_pmask,
        zero,
        LossParams(match_target="best"),
    )
    assert has_spread.item(), "the map explains every point of it"
    assert not has_best.item(), "no single chunk holds half of them"
    assert torch.allclose(spread[0, 0], torch.full((4,), 0.25), atol=1e-5)


def test_point_labels_use_a_radius_the_map_spacing_allows():
    """Map points are 1.68 m apart, so a point label needs room for that.

    A detected point can be 0.84 m from the nearest map point with no noise at
    all, purely because that is half the spacing. Labelling at the element
    radius of 0.75 m throws away a quarter of the real points; 1.0 m is the
    measured knee, capturing 88.4% against 90.1% at 1.25 m.
    """
    from mapposeformer.losses import point_truth

    batch = _batch(4)
    tight = LossParams(point_radius_m=0.75)
    knee = LossParams(point_radius_m=1.0)
    _, few = point_truth(
        batch["det_pts"],
        batch["det_pmask"],
        batch["map_pts"],
        batch["map_pmask"],
        batch["delta"],
        tight,
    )
    target, many = point_truth(
        batch["det_pts"],
        batch["det_pmask"],
        batch["map_pts"],
        batch["map_pmask"],
        batch["delta"],
        knee,
    )
    assert many.sum() > few.sum(), "a wider radius must label more points"
    # One-hot on the nearest map point, and nothing where there is no match.
    assert target.sum(-1).max() <= 1.0 + 1e-6
    assert torch.allclose(target.sum(-1) > 0, many)


def test_a_wildly_wrong_pose_does_not_drown_the_match_term():
    """The failure that killed the first point-token run, as an invariant.

    An untrained model produces a pose metres wide. With an L1 pose term the
    gradient at that scale is the full weighted error -- two orders above the
    match term's -- so the matcher that would have fixed the pose is destroyed
    instead. Measured: the match loss went 6.1 -> 1.8e7 -> 4.8e12 over 37k
    steps while the pose error rose from 20 m to 95 m.

    Huber bounds it. The test is that the pose gradient stops growing once the
    error is far past the knee, which is the property L1 does not have.
    """
    batch = _batch(2)
    p = LossParams()
    grads = []
    for scale in (1.0, 50.0):
        delta = torch.zeros_like(batch["delta"], requires_grad=True)
        out = {
            "delta": delta + scale * torch.tensor([10.0, 10.0, 0.1]),
            "granularity": "element",
            "elements": torch.full((2, 1, 72), 1.0 / 72),
            "scores": torch.zeros(2, 1, 72),
            "det_matchable": torch.full((2, 1), 0.5),
            "det_pts": batch["det_pts"][:, :1],
            "det_pmask": batch["det_pmask"][:, :1],
        }
        total, _ = compute_losses(out, batch, p)
        total.backward()
        grads.append(float(delta.grad.abs().max()))

    # Fifty times the error must not mean fifty times the gradient.
    assert grads[1] < 1.5 * grads[0], (
        f"pose gradient still grows with the error: {grads}"
    )
