"""SE(2) pose algebra.

Every pose in this repository is a 3-vector ``(x, y, yaw)`` in the **vehicle**
convention: X forward, Y left, yaw counter-clockwise about the up axis, in
metres and radians. That is the same convention as ``core::Frames`` in
camera-map-localization, and for the same reason: a translation of ``(1, 0)``
should mean *one metre forward*, so that reading a number tells you what moved.

A pose is never a matrix here. Localization corrects three degrees of freedom
(forward, left, heading), so a 3-vector is the whole state; carrying a 4x4
around would only invite the question of what the other thirteen numbers mean.

All functions broadcast over leading batch dimensions and are differentiable.
"""

from __future__ import annotations

import contextlib
import math

import torch
from torch import Tensor


@contextlib.contextmanager
def exact_arithmetic(device_type: str):
    """Run a block of coordinate arithmetic outside autocast, in fp32.

    Three places multiply a weight by a *coordinate*: the Procrustes target, the
    volume's soft argmin, and the cost surface. bf16 keeps 8 mantissa bits, so a
    40 m coordinate lands on a 0.25 m lattice and the pose inherits the error --
    measured at 8 mm of translation, and five times the variance floor on the
    covariance. The guarded arithmetic is a few hundred thousand FLOPs against
    the model's tens of millions: precision given away for no speed.

    CPU cannot catch this by accident, because autocast is off there;
    ``tests/test_model.py`` asks for bf16 explicitly.

    @param device_type Autocast device string, e.g. ``"cuda"``.
    """
    with torch.autocast(device_type=device_type, enabled=False):
        yield


def wrap_angle(a: Tensor) -> Tensor:
    """Wrap radians into ``[-pi, pi)``.

    Used on every yaw difference. Without it a heading error of ``+179°`` and
    one of ``-181°`` -- the same error -- get different losses, and the one
    crossing the branch cut gets a gradient pointing the wrong way.
    """
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(a: Tensor, b: Tensor) -> Tensor:
    """``a ∘ b``: apply ``b`` in the frame ``a`` defines.

    @param a ``(..., 3)`` pose.
    @param b ``(..., 3)`` pose, expressed in ``a``'s frame.

    @return ``(..., 3)`` the composed pose, in ``a``'s parent frame.
    """
    ca, sa = torch.cos(a[..., 2]), torch.sin(a[..., 2])
    x = a[..., 0] + ca * b[..., 0] - sa * b[..., 1]
    y = a[..., 1] + sa * b[..., 0] + ca * b[..., 1]
    return torch.stack([x, y, wrap_angle(a[..., 2] + b[..., 2])], dim=-1)


def inverse(a: Tensor) -> Tensor:
    """The pose that undoes ``a``: ``compose(a, inverse(a)) == identity``."""
    ca, sa = torch.cos(a[..., 2]), torch.sin(a[..., 2])
    x = -(ca * a[..., 0] + sa * a[..., 1])
    y = -(-sa * a[..., 0] + ca * a[..., 1])
    return torch.stack([x, y, wrap_angle(-a[..., 2])], dim=-1)


def relative(a: Tensor, b: Tensor) -> Tensor:
    """``a⁻¹ ∘ b``: where ``b`` is, as seen from ``a``.

    This is the shape of the quantity this whole project predicts: the anchor
    pose is ``a``, the true pose is ``b``, and the network's output is what
    turns one into the other.
    """
    return compose(inverse(a), b)


def transform_points(pose: Tensor, pts: Tensor) -> Tensor:
    """Move points from the frame ``pose`` describes into ``pose``'s parent.

    @param pose ``(..., 3)``.
    @param pts ``(..., N, 2)``, in the child frame.

    @return ``(..., N, 2)`` in the parent frame.
    """
    c, s = torch.cos(pose[..., 2]), torch.sin(pose[..., 2])
    x, y = pts[..., 0], pts[..., 1]
    out_x = pose[..., 0].unsqueeze(-1) + c.unsqueeze(-1) * x - s.unsqueeze(-1) * y
    out_y = pose[..., 1].unsqueeze(-1) + s.unsqueeze(-1) * x + c.unsqueeze(-1) * y
    return torch.stack([out_x, out_y], dim=-1)

