"""The whole network on a real sample: shapes, history, and the anneal.

The parts are tested where they live. What is left for here is the wiring --
that history becomes width, that every parameter is on a path to the loss, and
that the robust scale does what the trainer will schedule it to do.
"""

import math
from dataclasses import replace

import pytest
import torch

from mapposeformer import geometry as G
from mapposeformer.config import Config
from mapposeformer.data import build_dataset
from mapposeformer.losses import LossParams, element_truth
from mapposeformer.model import MapPoseFormer, ModelParams
from mapposeformer.model.encoder import element_frames
from mapposeformer.model.matcher import Matcher
from mapposeformer.solve import (
    point_normals,
    projections,
    solve_pose_directional,
)


@pytest.fixture(scope="module")
def batch():
    ds = build_dataset(Config().data, "train")
    return {k: torch.stack([ds[i][k] for i in range(2)]) for k in ds[0]}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return MapPoseFormer(ModelParams(dim=32, layers=1, heads=2)).eval()


def test_history_becomes_width(batch, model):
    """Three frames of detections against one map, as one set of elements."""
    sp = Config().data.sample
    det_pts, det_pmask, det_cls, _, extra = model.detections(batch)
    want = (1 + sp.history) * sp.max_det_elements
    assert det_pts.shape[1] == want
    assert det_pmask.shape[1] == want
    assert det_cls.shape[1] == want
    assert extra.shape == (2, want, 3)  # confidence and two noise axes


def test_the_warped_history_lands_on_the_map(batch, model):
    """``hist_rel`` brings a past detection into the current ego frame.

    The data tests assert this of the labels; this asserts the model applies
    it in the same direction. A sign error trains happily and quietly turns
    two thirds of the evidence into noise, so the check is the same one the
    labels get: apply the true correction and the points must land on the map.

    The warped history does *not* sit on top of the current frame -- it is 4
    and 8 m behind, and warping moves it back, not on. Landing on the map is
    the invariant; overlapping this frame is not.
    """
    det_pts, det_pmask, *_ = model.detections(batch)
    sp = Config().data.sample
    moved = G.transform_points(
        batch["delta"], det_pts.reshape(2, -1, 2)
    ).reshape(det_pts.shape)

    map_pts = batch["map_pts"][batch["map_pmask"]].reshape(-1, 2)
    past = moved[:, sp.max_det_elements :][det_pmask[:, sp.max_det_elements :]]
    now = moved[:, : sp.max_det_elements][det_pmask[:, : sp.max_det_elements]]

    def inliers(p):
        return (torch.cdist(p, map_pts).min(1).values < 1.0).float().mean()

    assert inliers(past) > 0.75, inliers(past)
    # And no worse than this frame's, which is what a sign error would break.
    assert inliers(past) > inliers(now) - 0.15


def test_the_output_is_a_pose_and_the_evidence_behind_it(batch, model):
    out = model(batch)
    assert out["delta"].shape == (2, 3)
    assert out["mass"].shape == (2,)
    assert torch.isfinite(out["delta"]).all()
    assert (out["mass"] >= 0).all()


def test_a_wider_robust_scale_keeps_more_evidence(batch, model):
    """What the trainer anneals, and the direction it has to anneal in."""
    masses = [
        model(batch, sigma_m=s)["mass"].mean().item() for s in (1.0, 12.0)
    ]
    assert masses[1] > 10 * masses[0], masses


def test_every_parameter_is_on_a_path_to_the_pose(batch):
    """A parameter with no gradient is a part of the design that does nothing.

    This is the test that catches a branch wired up but never reached -- the
    kind of mistake that costs a training run and leaves no trace in the loss.
    """
    torch.manual_seed(0)
    m = MapPoseFormer(ModelParams(dim=32, layers=1, heads=2))
    out = m(batch, sigma_m=12.0)
    (out["delta"].abs().sum() + out["elements"].sum()).backward()
    dead = [
        n for n, p in m.named_parameters() if p.grad is None or not p.grad.any()
    ]
    assert not dead, f"no gradient reaches: {dead}"


def test_the_covariance_comes_out_and_is_usable(batch, model):
    """A pose without an uncertainty can only be followed, not fused."""
    out = model(batch)
    cov = out["cov"]
    assert cov.shape == (2, 3, 3)
    assert torch.isfinite(cov).all()
    # Symmetric, and positive definite -- a filter inverts this.
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)
    assert (torch.linalg.eigvalsh(cov) > 0).all()


def test_the_measurement_is_anisotropic_where_point_to_point_cannot_be(batch):
    """An anisotropic covariance, asserted on the measurement not the prior.

    An untrained model contributes almost no information, so the fused
    covariance is just the prior: assert on it and the answer is 1.5/0.6
    whatever the landmarks are. So the assignment here is the *true*
    correspondence, and the assertion is on the Hessian, which carries no
    prior at all.

    Point-to-point residuals give a translation Hessian of ``2·mass·I`` and
    report 1.01 whatever the landmarks are. These projections must not.
    """
    tgt, has = element_truth(
        batch["det_pts"],
        batch["det_pmask"],
        batch["map_pts"],
        batch["map_pmask"],
        batch["delta"],
        LossParams(),
    )
    assign = tgt * has.unsqueeze(-1)
    det_frame, _, _ = element_frames(batch["det_pts"], batch["det_pmask"])
    map_frame, oriented, _ = element_frames(
        batch["map_pts"], batch["map_pmask"]
    )
    points = Matcher(16, layers=1, heads=2).points(
        assign,
        batch["det_pts"],
        batch["det_pmask"],
        det_frame,
        batch["map_pts"],
        batch["map_pmask"],
        map_frame,
    )
    proj = projections(
        point_normals(batch["map_pts"], batch["map_pmask"]), oriented
    )
    B = batch["delta"].shape[0]
    _, mass, hessian, _, _ = solve_pose_directional(
        batch["det_pts"].reshape(B, -1, 2),
        batch["map_pts"].reshape(B, -1, 2),
        points,
        proj,
        iters=5,
        sigma_m=2.0,
    )
    assert (mass > 1.0).all(), "the true correspondence should carry evidence"
    cov = torch.linalg.pinv(hessian)
    ratio = (cov[:, 0, 0] / cov[:, 1, 1]).sqrt()
    assert (ratio > 1.2).all(), (
        f"isotropic, as point-to-point would be: {ratio}"
    )


def test_the_prior_bounds_the_covariance(batch, model):
    """Fusing cannot make the answer *less* certain than the prior alone."""
    out = model(batch)
    sigma = out["cov"].diagonal(dim1=-2, dim2=-1).sqrt()
    p = model.p
    bound = torch.tensor(
        [
            p.prior_sigma_long_m,
            p.prior_sigma_lat_m,
            math.radians(p.prior_sigma_yaw_deg),
        ]
    )
    assert (sigma <= bound + 1e-4).all(), f"{sigma} exceeds the prior {bound}"


def test_point_to_point_is_reachable_as_a_flag(batch):
    """``residual='point'`` reproduces the isotropic Hessian, and it ships.

    A point-to-point residual makes the covariance structurally isotropic, and
    the claim of the line residual is that it need not be. That claim is only
    measurable against the isotropic baseline, so neither side is an edit:
    `point` is the default and `line` is one flag away.
    """
    torch.manual_seed(0)
    p = ModelParams(dim=32, layers=1, heads=2, residual="point")
    out = MapPoseFormer(p).eval()(batch, sigma_m=12.0)
    assert torch.isfinite(out["cov"]).all()


def test_point_tokens_produce_the_assignment_the_solve_wants(batch):
    """Matching at point resolution has no second stage, and needs none.

    Element tokens assign a detection to a map element and then spread that
    over points, which cannot say *this* point belongs to *that* chunk. Point
    tokens skip the question: what the matcher produces is already the
    (detection point, map point) assignment the solve consumes.
    """
    torch.manual_seed(0)
    p = ModelParams(dim=48, layers=1, heads=2, tokens="point")
    out = MapPoseFormer(p).eval()(batch, sigma_m=12.0)

    sp = Config().data.sample
    det = (1 + sp.history) * sp.max_det_elements * sp.points_per_element
    mp = sp.max_map_elements * sp.points_per_element
    assert out["elements"].shape == (2, det, mp)
    assert torch.isfinite(out["cov"]).all()
    assert torch.isfinite(out["delta"]).all()


def test_both_token_paths_reach_every_parameter(batch):
    """Neither path should carry weights the loss never sees."""
    for tokens in ("element", "point"):
        torch.manual_seed(0)
        m = MapPoseFormer(ModelParams(dim=48, layers=1, heads=2, tokens=tokens))
        out = m(batch, sigma_m=12.0)
        (out["delta"].abs().sum() + out["elements"].sum()).backward()
        dead = [
            n
            for n, q in m.named_parameters()
            if q.grad is None or not q.grad.any()
        ]
        assert not dead, f"{tokens}: no gradient reaches {dead}"


def test_the_residual_flag_is_honoured_on_both_token_paths(batch):
    """`residual=point` must mean point-to-point whichever tokens are used.

    It is easy not to: a point-token path that builds its projections without
    consulting the flag runs rank-1 line residuals under
    `tokens=point, residual=point` while reporting itself as point-to-point.
    That silently makes the single most important ablation in the design
    unmeasurable -- a configuration that does not differ by the one thing it
    claims to.
    """
    for tokens in ("element", "point"):
        torch.manual_seed(0)
        line = MapPoseFormer(
            ModelParams(dim=48, layers=1, tokens=tokens, residual="line")
        ).eval()
        torch.manual_seed(0)
        point = MapPoseFormer(
            ModelParams(dim=48, layers=1, tokens=tokens, residual="point")
        ).eval()
        h_line = line(batch, sigma_m=12.0)["hessian"]
        h_point = point(batch, sigma_m=12.0)["hessian"]
        assert not torch.allclose(h_line, h_point, atol=1e-4), (
            f"{tokens}: the residual flag changed nothing"
        )
        # Point-to-point is isotropic in translation by construction.
        tr = h_point[:, :2, :2]
        off = tr[:, 0, 1].abs().max()
        assert torch.allclose(tr[:, 0, 0], tr[:, 1, 1], rtol=1e-3), tr
        assert off < 1e-3 * tr[:, 0, 0].max(), f"{tokens}: not isotropic {off}"


def test_the_match_scores_are_fp32_under_autocast(batch):
    """bf16 logits are what kill the point-token path.

    Under bf16 autocast, with point tokens and everything else held equal,
    the largest logit went 48 -> 1 144 -> 28 032 -> 4.9e5 by step 250 and the
    match loss followed it from 6.4 to 1 044, while the assignment mass stayed
    at zero. The same run in fp32 held its logits under 35 and took the pose
    from 110 m to 20 m. The scores are a sum over `dim` that has to survive
    two softmaxes, and eight mantissa bits is not enough at 768 x 576.

    So this is a dtype assertion rather than a numerical one: the thing that
    broke was invisible in the loss until it was six orders wide.
    """

    if not torch.cuda.is_available():
        pytest.skip("autocast dtype promotion is a CUDA path")

    for tokens in ("element", "point"):
        p = ModelParams(dim=48, layers=1, heads=2, tokens=tokens)
        model = MapPoseFormer(p).cuda().eval()
        cuda_batch = {
            k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(cuda_batch)
        assert out["scores"].dtype is torch.float32, (
            tokens,
            out["scores"].dtype,
        )
        assert out["delta"].dtype is torch.float32, (tokens, out["delta"].dtype)


def test_the_width_flags_are_bit_exact_on_their_defaults(batch):
    """`head_dim=0` and `rope_bands=0` must reproduce the derived widths.

    Zero means "derive from `dim`", so a configuration that asks for the
    derivation must be bit-identical to one that spells the derived values out
    -- same weight shapes, same output bits. Anything else makes two runs
    incomparable purely because one of them wrote the fields down.

    Both are asked for explicitly here. The dataclass defaults are the shipped
    widths (`head_dim` 64, `rope_bands` 5) rather than the derivation, so that
    the network anyone gets by default is the measured one and reads off the
    fields without arithmetic.
    """

    for tokens in ("element", "point"):
        p = ModelParams(
            dim=48, layers=1, heads=2, tokens=tokens, head_dim=0, rope_bands=0
        )
        assert p.head_dim == 0 and p.rope_bands == 0

        torch.manual_seed(0)
        model = MapPoseFormer(p).eval()
        # Explicit values that restate the derived ones must build the same
        # module, down to parameter shapes.
        torch.manual_seed(0)
        spelled = MapPoseFormer(
            replace(p, head_dim=48 // 2, rope_bands=(48 // 2) // 6)
        ).eval()

        # The invariant that makes them bit-exact: on the default the inner
        # attention width collapses to `dim`, so every projection is dim -> dim.
        for layer in model.matcher.map_layers:
            for att in (layer.self_attn, layer.cross_attn):
                assert att.heads * att.head_dim == p.dim, (tokens, att.head_dim)
                assert att.q.out_features == p.dim
                assert att.out.in_features == p.dim

        a = {k: v.shape for k, v in model.state_dict().items()}
        b = {k: v.shape for k, v in spelled.state_dict().items()}
        assert a == b, tokens

        with torch.no_grad():
            got, want = model(batch), spelled(batch)
        assert torch.equal(got["delta"], want["delta"]), tokens
        assert torch.equal(got["scores"], want["scores"]), tokens


def test_head_count_and_head_width_move_independently():
    """The whole point of the two fields, as an arithmetic check.

    If `dim = heads * head_dim` were an identity, no configuration could hold
    two of the three fixed. Here `dim` is the token width and
    `heads * head_dim` is the attention width, and they are free of each other.
    """

    from mapposeformer.model.attention import GeometricAttention

    # Count at fixed width: 2 -> 4 heads, head_dim pinned at 64.
    two = GeometricAttention(128, 2, "rope", head_dim=64)
    four = GeometricAttention(128, 4, "rope", head_dim=64)
    assert two.head_dim == four.head_dim == 64
    assert four.q.out_features == 2 * two.q.out_features

    # Width at fixed count: head_dim 64 -> 32, heads pinned at 2.
    narrow = GeometricAttention(128, 2, "rope", head_dim=32)
    assert narrow.heads == two.heads == 2
    assert narrow.head_dim == 32

    # And the rank cap really is head_dim: a head's scores are q @ k^T.
    for att in (two, four, narrow):
        assert att.q.out_features // att.heads == att.head_dim

    # Bands need not follow width, which is what would otherwise confound a
    # head-count sweep. Derived, and capped: head_dim 32 has room for 5 bands
    # and head_dim 64 for 10, but the derivation stops at 5 either way. That
    # cap is what makes this test's own claim true -- without it, changing the
    # head count at fixed `dim` would silently change the positional encoding
    # too.
    assert GeometricAttention(128, 4, "rope").rotary.freq_xy.numel() == 5
    assert GeometricAttention(128, 2, "rope").rotary.freq_xy.numel() == 5
    # Pinned, both give the same positional bandwidth.
    assert (
        GeometricAttention(
            128, 4, "rope", head_dim=32, rope_bands=5
        ).rotary.freq_xy.numel()
        == GeometricAttention(
            128, 2, "rope", head_dim=64, rope_bands=5
        ).rotary.freq_xy.numel()
    )
