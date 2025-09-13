"""SE(2) algebra. Cheap to test, and everything downstream is wrong without it."""

import math

import torch

from mapposeformer import geometry as G


def _random_poses(n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    t = 20 * torch.randn(n, 2, generator=g)
    yaw = math.pi * (2 * torch.rand(n, 1, generator=g) - 1)
    return torch.cat([t, yaw], dim=-1)


def test_inverse_undoes_compose():
    a = _random_poses(64)
    identity = G.compose(a, G.inverse(a))
    assert torch.allclose(identity, torch.zeros_like(identity), atol=1e-5)


def test_relative_is_the_transform_between():
    a, b = _random_poses(64), _random_poses(64)
    assert torch.allclose(G.compose(a, G.relative(a, b)), b, atol=1e-5)


def test_transform_points_agrees_with_compose():
    """Moving a frame and moving the points in it must be the same operation."""
    a, b = _random_poses(32), _random_poses(32)
    origin = torch.zeros(32, 1, 2)
    moved = G.transform_points(a, G.transform_points(b, origin))
    assert torch.allclose(moved[:, 0], G.compose(a, b)[:, :2], atol=1e-5)


def test_wrap_angle_is_continuous_across_the_branch_cut():
    """+179 deg and -181 deg are the same error and must wrap to the same value."""
    a = G.wrap_angle(torch.tensor(math.radians(179.0)))
    b = G.wrap_angle(torch.tensor(math.radians(-181.0)))
    assert abs(float(a - b)) < 1e-6
