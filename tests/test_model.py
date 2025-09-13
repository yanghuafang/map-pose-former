"""The model's shape contract, its numerics, and the geometry it must be exact on."""

import math

import torch

from mapposeformer import geometry as G
from mapposeformer.data import DataParams, SyntheticDataset
from mapposeformer.losses import LossParams, compute_losses
from mapposeformer.model import MapPoseFormer, ModelParams, weighted_procrustes_se2
from mapposeformer.model.pose_head import ProcrustesPoseHead
from mapposeformer.model.volume_head import GridParams, VolumeHead


def _batch(n: int = 4) -> dict[str, torch.Tensor]:
    d = SyntheticDataset(DataParams(), "train")
    items = [d[i * 37] for i in range(n)]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def test_procrustes_is_exact_on_a_known_transform():
    """With correct correspondences the head must recover the pose, not approach it.

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
    assert scalars["matched_frac"] > 0.0, "no positive correspondences to learn from"
    total.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    # Everything must receive gradient. A head that never does is a head that is
    # not actually wired into the loss, which is invisible in a loss curve.
    assert not missing, missing


def test_regression_head_is_bounded_by_the_grid():
    p = ModelParams(pose_head="regression")
    out = MapPoseFormer(p)(_batch())
    limit = torch.tensor(
        [p.grid.extent_x_m, p.grid.extent_y_m, math.radians(p.grid.extent_yaw_deg)]
    )
    assert bool((out["delta"].abs() <= limit + 1e-5).all())


def test_geometry_heads_ignore_autocast():
    """bf16 must not reach the arithmetic that multiplies weights by coordinates.

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
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = volume(tokens, pad)
    for key in ("delta", "cov"):
        assert out[key].dtype is torch.float32, f"{key} came back {out[key].dtype}"
