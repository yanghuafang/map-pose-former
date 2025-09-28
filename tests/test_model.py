"""The model's shape contract, its numerics, and the geometry it must be exact
on."""

import math

import torch

from mapposeformer import geometry as G
from mapposeformer.config import Config
from mapposeformer.data import DataParams, SyntheticDataset
from mapposeformer.losses import LossParams, compute_losses
from mapposeformer.model import (
    MapPoseFormer,
    ModelParams,
    weighted_procrustes_se2,
)
from mapposeformer.model.pose_head import ProcrustesPoseHead
from mapposeformer.model.volume_head import GridParams, VolumeHead, grid_cost


def _batch(n: int = 4) -> dict[str, torch.Tensor]:
    d = SyntheticDataset(DataParams(), "train")
    items = [d[i * 37] for i in range(n)]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def test_procrustes_is_exact_on_a_known_transform():
    """With correct correspondences the head recovers the pose exactly.

    This is the whole argument for solving the transform in closed form: there
    is no residual error to train away.
    """
    g = torch.Generator().manual_seed(1)
    src = 10 * torch.randn(3, 40, 2, generator=g)
    yaw = torch.tensor([0.3, -1.2, 2.9])
    c, s = torch.cos(yaw), torch.sin(yaw)
    t = torch.tensor([[1.0, -2.0], [0.5, 0.25], [-4.0, 3.0]])
    dst = (
        torch.stack(
            [
                c[:, None] * src[..., 0] - s[:, None] * src[..., 1],
                s[:, None] * src[..., 0] + c[:, None] * src[..., 1],
            ],
            dim=-1,
        )
        + t[:, None]
    )

    pose, mass = weighted_procrustes_se2(src, dst, torch.ones(3, 40))
    assert torch.allclose(pose[:, :2], t, atol=1e-4)
    assert torch.allclose(pose[:, 2], yaw, atol=1e-5)
    assert torch.allclose(mass, torch.full((3,), 40.0))


def test_procrustes_returns_identity_when_it_has_nothing():
    """No correspondences must give identity and zero mass, not NaN.

    Identity is a *default*, and ``mass`` is how a caller tells the difference
    between that and an estimate. The trust head exists to make the same
    distinction in a form the filter can act on.
    """
    pose, mass = weighted_procrustes_se2(
        torch.randn(2, 10, 2), torch.randn(2, 10, 2), torch.zeros(2, 10)
    )
    assert torch.isfinite(pose).all()
    assert torch.allclose(pose, torch.zeros_like(pose))
    assert float(mass.max()) < 1e-5


def test_forward_shapes_and_finiteness():
    b = _batch()
    out = MapPoseFormer(ModelParams())(b)
    grid = ModelParams().grid
    assert out["delta"].shape == (4, 3)
    assert out["logits"].shape == (4, grid.size)
    assert out["cov"].shape == (4, 3, 3)
    assert out["trust_logit"].shape == (4,)
    for k, v in out.items():
        assert torch.isfinite(v).all(), k


def test_empty_input_is_finite():
    """A frame with no detections and no map is legal at the edge of a scene.

    Without the never-masked null token, attention softmaxes over an empty key
    set and produces NaN -- which then survives every mask applied afterwards,
    because ``0 * NaN`` is NaN. This test is that token's reason to exist.
    """
    b = _batch(2)
    b["det_pmask"] = torch.zeros_like(b["det_pmask"])
    b["map_pmask"] = torch.zeros_like(b["map_pmask"])
    model = MapPoseFormer(ModelParams())
    out = model(b)
    for k, v in out.items():
        assert torch.isfinite(v).all(), k

    # The volume head's pool is the one attention with no null token in front
    # of it -- model.py strips both before packing. Its summary of nothing must
    # be zero, and by construction rather than by an attention implementation's
    # choice of what an empty softmax returns.
    tokens = torch.randn(2, 16, model.volume.pool.query.shape[-1])
    pad = torch.ones(2, 16, dtype=torch.bool)
    pad[0, 3] = False
    pooled = model.volume.pool(tokens, pad).detach()
    assert torch.isfinite(pooled).all()
    assert float(pooled[1].abs().max()) == 0.0
    assert float(pooled[0].abs().max()) > 0.0


def test_covariance_is_symmetric_positive_definite():
    out = MapPoseFormer(ModelParams())(_batch())
    cov = out["cov"]
    assert torch.allclose(cov, cov.transpose(1, 2), atol=1e-5)
    assert bool((torch.linalg.eigvalsh(cov) > 0).all())


def test_losses_are_finite_and_backpropagate():
    model = MapPoseFormer(ModelParams())
    b = _batch()
    out = model(b)
    total, scalars = compute_losses(
        out, b, model.volume.cells, model.volume.pitch, LossParams()
    )
    assert math.isfinite(float(total.detach()))
    assert scalars["matched_frac"] > 0.0, (
        "no positive correspondences to learn from"
    )
    total.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    # Everything must receive gradient. A head that never does is a head that
    # is not actually wired into the loss, which is invisible in a loss curve.
    assert not missing, missing


def test_regression_head_is_bounded_by_the_grid():
    p = ModelParams(pose_head="regression")
    out = MapPoseFormer(p)(_batch())
    limit = torch.tensor(
        [
            p.grid.extent_x_m,
            p.grid.extent_y_m,
            math.radians(p.grid.extent_yaw_deg),
        ]
    )
    assert bool((out["delta"].abs() <= limit + 1e-5).all())


def test_geometry_heads_ignore_autocast():
    """bf16 must not reach arithmetic that multiplies weights by coordinates.

    The one property in this model a CPU test suite cannot stumble into: on CPU
    autocast is off, so the model has to be asked for bf16 explicitly. It hid
    once already -- ``docs/TRAINING.md`` claimed the geometry was
    autocast-exempt while nothing in the code made it so, and the cost was 8 mm
    of translation and a covariance error five times the variance floor.

    The tolerance is 1e-4, not a bf16-sized 1e-2: the point is that the guarded
    blocks are running in fp32, so the two paths should differ only by whatever
    bf16 did to the *features* upstream, which these inputs bypass entirely.
    """
    torch.manual_seed(0)
    head = ProcrustesPoseHead()
    # Coordinates spread over the 50 m crop, where bf16's lattice is coarsest,
    # and few enough correspondences that averaging cannot hide the error --
    # which is the sparse frame the model finds hard anyway.
    map_pts = (torch.rand(2, 64, 2) - 0.5) * 100.0
    pose = torch.tensor([1.2, -0.4, 0.02])
    det_pts = G.transform_points(G.inverse(pose), map_pts[:, :24])
    assign = torch.zeros(2, 24, 64)
    assign[:, torch.arange(24), torch.arange(24)] = 1.0

    ref = head(assign, det_pts, map_pts)[0]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        got = head(assign, det_pts, map_pts)[0]
    assert torch.allclose(ref, got, atol=1e-4), (ref - got).abs().max()
    # ...and it is the right answer, not merely a stable one.
    assert torch.allclose(ref, pose.expand(2, 3), atol=1e-4)

    # The volume head cannot be checked the same way, because its logits come
    # from layers that autocast is *meant* to run in bf16. What the guard owns
    # is everything downstream of them, so what it can be held to is the dtype:
    # the statistics must come back fp32 no matter what ran above. Numerically
    # the guard takes the covariance difference from 0.0131 to 0.0018 -- the
    # remainder is the bf16 logits, and it belongs there.
    volume = VolumeHead(64, 4, GridParams())
    tokens, pad = torch.randn(2, 32, 64), torch.zeros(2, 32, dtype=torch.bool)
    with (
        torch.no_grad(),
        torch.autocast(device_type="cpu", dtype=torch.bfloat16),
    ):
        out = volume(assign, det_pts, map_pts, tokens, pad)
    for key in ("delta", "cov", "logits"):
        assert out[key].dtype is torch.float32, (
            f"{key} came back {out[key].dtype}"
        )


def test_grid_cost_matches_the_brute_force_sum():
    """The closed form is an identity, not an approximation.

    ``grid_cost`` claims that the assignment-weighted squared alignment error
    at all 4199 hypotheses follows from eleven numbers. That is either exactly
    true or the cost volume is measuring something else, and the difference is
    invisible in a loss curve -- a wrong-but-smooth surface trains fine and
    localizes badly. So it is checked against the sum it replaces.
    """
    torch.manual_seed(0)
    head = VolumeHead(32, 4, GridParams())
    det = torch.randn(3, 48, 2) * 25
    mp = torch.randn(3, 64, 2) * 25
    assign = torch.rand(3, 48, 64).pow(6)

    cost, mass = grid_cost(assign, det, mp, head.cell_t, head.cell_rot)
    moved = (
        torch.einsum("gij,bkj->bgki", head.cell_rot, det)
        + head.cell_t[None, :, None]
    )
    sq = (moved[:, :, :, None, :] - mp[:, None, None]).square().sum(-1)
    brute = torch.einsum("bkl,bgkl->bg", assign, sq)

    assert torch.allclose(cost, brute, rtol=1e-4, atol=1e-3), (
        (cost - brute).abs().max()
    )
    assert bool((cost.argmin(-1) == brute.argmin(-1)).all())
    assert torch.allclose(mass, assign.sum((1, 2)))


def test_the_volume_minimum_is_the_pose_the_head_solves():
    """The two output paths are one objective, so they cannot disagree.

    Weighted Procrustes finds the minimum of exactly the surface ``grid_cost``
    evaluates -- the per-point target form and the full double sum differ by a
    constant in the pose, so they share a minimiser. That is the claim the
    architecture rests on, and it is checkable to within half a cell.

    Robustness is switched off here on purpose: IRLS deliberately minimises a
    *different*, reweighted objective, and the agreement being tested is the
    one between the plain solve and the surface.
    """
    torch.manual_seed(0)
    head = VolumeHead(32, 4, GridParams())
    plain = ProcrustesPoseHead(irls_iters=0, min_row_mass=0.0)
    map_pts = (torch.rand(4, 64, 2) - 0.5) * 80.0
    truth = torch.tensor(
        [
            [1.5, 0.7, 0.02],
            [-2.0, -0.5, -0.03],
            [0.5, 0.0, 0.0],
            [3.0, 1.0, 0.04],
        ]
    )
    det_pts = G.transform_points(G.inverse(truth), map_pts[:, :40])
    assign = torch.zeros(4, 40, 64)
    assign[:, torch.arange(40), torch.arange(40)] = 1.0

    pose, _ = plain(assign, det_pts, map_pts)
    cost, _ = grid_cost(assign, det_pts, map_pts, head.cell_t, head.cell_rot)
    best = head.cells[cost.argmin(-1)]

    # Half a cell on each axis: the grid cannot do better, and neither can any
    # argmin over it.
    assert torch.allclose(pose, truth, atol=1e-4), pose - truth
    assert bool(((pose - best).abs() <= 0.5 * head.pitch + 1e-6).all()), (
        pose - best
    )


def test_the_robust_solve_survives_wrong_correspondences():
    """The failure that cost the closed-form head its first comparison.

    A rigid fit over a handful of confident, wrong matches is unbounded: it
    moves the answer arbitrarily far, which is how an ablation reached 30.9
    degrees of heading error where a bounded regressor could only be vague.
    Reweighting is the standard repair, and this is the case it has to handle.
    """
    torch.manual_seed(0)
    map_pts = (torch.rand(8, 60, 2) - 0.5) * 80.0
    truth = torch.zeros(8, 3)
    truth[:, 0], truth[:, 1], truth[:, 2] = 1.2, -0.4, 0.02
    det_pts = G.transform_points(G.inverse(truth), map_pts[:, :40])

    assign = torch.zeros(8, 40, 60)
    assign[:, torch.arange(40), torch.arange(40)] = 1.0
    # A fifth of the correspondences point somewhere else entirely, and say so
    # just as confidently as the rest.
    assign[:, :8] = 0.0
    assign[:, torch.arange(8), torch.arange(50, 58)] = 1.0

    naive = ProcrustesPoseHead(irls_iters=0, min_row_mass=0.0)(
        assign, det_pts, map_pts
    )[0]
    robust = ProcrustesPoseHead()(assign, det_pts, map_pts)[0]
    err = lambda p: (p - truth).abs()  # noqa: E731

    assert float(err(robust)[:, :2].norm(dim=-1).mean()) < 0.10
    assert float(err(robust)[:, 2].abs().max()) < math.radians(0.5)
    # ...and it is the reweighting doing it, not the problem being easy.
    assert float(err(naive)[:, :2].norm(dim=-1).mean()) > 5 * float(
        err(robust)[:, :2].norm(dim=-1).mean()
    )


def test_refinement_runs_the_trunk_once_per_pass():
    """Deep supervision needs one estimate per pass; the last is the answer."""
    for iters in (1, 3):
        out = MapPoseFormer(ModelParams(refine_iters=iters))(_batch(2))
        assert out["deltas"].shape == (2, iters, 3)
        assert torch.equal(out["delta"], out["deltas"][:, -1])


def test_calibration_separates_honest_from_overconfident():
    """ANEES near one when the covariance matches the errors, and above it when
    the covariance is too small.

    The statistic that says whether a filter can believe this model. An accurate
    pose with a dishonest covariance is worse than an inaccurate one with a
    truthful covariance, because the filter weights by the second number and
    has no way to check it.
    """
    from mapposeformer.metrics import Calibration

    torch.manual_seed(0)
    n = 20000
    std = torch.tensor([0.4, 0.15, 0.004])
    gt = torch.zeros(n, 3)
    pred = torch.randn(n, 3) * std

    honest = Calibration()
    honest.update(pred, gt, torch.diag_embed(std.square().expand(n, 3)))
    d = honest.as_dict()
    assert 0.9 < d["anees"] < 1.1, d
    assert 0.93 < d["coverage_95"] < 0.97, d
    # The median of a calibrated NEES is the chi-square median, 0.789 per dof,
    # not 1.0. Judged against one it reads "pessimistic" for an estimator that
    # is exactly right, which is a verdict worth asserting and not just a
    # statistic worth printing.
    assert abs(d["anees_median"] - Calibration.CHI2_MEDIAN) < 0.03, d
    assert "calibrated" in honest.format(), honest.format()

    liar = Calibration()
    liar.update(pred, gt, torch.diag_embed((std / 3).square().expand(n, 3)))
    d = liar.as_dict()
    assert d["anees"] > 5.0, d
    assert d["coverage_95"] < 0.5, d


def test_gradient_accumulation_matches_one_large_batch():
    """Two micro-batches must give what one batch of twice the size gives.

    This is the property that makes `train.accum_steps` a memory trade rather
    than a different optimizer, and it is not exactly true: `match_loss`
    divides by the *batch's* count of valid detections, so splitting a batch
    changes that normalizer. Measured, the error is ~1e-4 relative -- four
    orders of magnitude below the gradient noise between two independent
    batches of this size, which is what accumulation exists to reduce.
    """
    torch.manual_seed(0)
    cfg = Config()
    model = MapPoseFormer(cfg.model)
    params = list(model.parameters())
    dataset = SyntheticDataset(cfg.data, "train")
    items = [dataset[i] for i in range(8)]
    full = {k: torch.stack([it[k] for it in items]) for k in items[0]}

    def grads(batch, scale):
        out = model(batch)
        total, _ = compute_losses(
            out, batch, model.volume.cells, model.volume.pitch, cfg.loss
        )
        raw = torch.autograd.grad(total * scale, params, allow_unused=True)
        return [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(raw, params, strict=True)
        ]

    single = grads(full, 1.0)
    accumulated = [torch.zeros_like(p) for p in params]
    for lo in (0, 4):
        micro = {k: v[lo : lo + 4] for k, v in full.items()}
        for into, g in zip(accumulated, grads(micro, 0.5), strict=True):
            into += g

    a = torch.cat([g.flatten() for g in single])
    b = torch.cat([g.flatten() for g in accumulated])
    assert float((a - b).norm() / a.norm()) < 1e-3
