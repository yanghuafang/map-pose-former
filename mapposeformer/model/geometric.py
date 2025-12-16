"""Nearest-neighbour matching, wearing the model's interface.

The transformer's justification is that association is ambiguous enough to be
worth learning. That is a claim about a comparison, so something has to be on
the other side of it: this matches by distance and class alone, then hands the
assignment to the same closed-form solver and the same covariance head.

It lives in the package rather than beside the script that first needed it
because two callers now drive it -- ``tools/baseline_matcher.py`` open loop and
``engine/sequence.py`` closed loop -- and neither can tell it from the real
model, which is the point.

Two choices are worth knowing before reading the code. The covariance is the
raw cost surface with the learned scale zeroed, so it reports what the geometry
alone supports. And it sees the current frame only, where the model folds in
two past frames -- handicapped deliberately, because a classical matcher given
temporal fusion is a different experiment.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mapposeformer import geometry as G
from mapposeformer.model.model import ModelParams
from mapposeformer.model.pose_head import ProcrustesPoseHead
from mapposeformer.model.volume_head import VolumeHead


def geometric_assign(
    det_xy: Tensor,
    det_ok: Tensor,
    det_cls: Tensor,
    map_xy: Tensor,
    map_ok: Tensor,
    map_cls: Tensor,
    sigma: float,
) -> Tensor:
    """@brief Mutual nearest neighbour, softened, class-constrained.

    @param det_xy ``(B, K, 2)`` detection points, in the prior's frame.
    @param det_ok ``(B, K)`` which of them are real rather than padding.
    @param det_cls ``(B, K)`` class per point.
    @param map_xy ``(B, L, 2)``, @param map_ok ``(B, L)``,
        @param map_cls ``(B, L)``.
    @param sigma Kernel width in metres. The prior is wrong by about a metre
        along track, so this is the scale at which a correspondence is
        plausible rather than a tuned constant.

    @return ``(B, K, L)`` weights. Zero where a pair is padded, of different
        classes, or not each other's best.
    """
    d2 = torch.cdist(det_xy, map_xy).square()
    w = torch.exp(-d2 / (2.0 * sigma * sigma))

    ok = det_ok.unsqueeze(-1) & map_ok.unsqueeze(1)
    same = det_cls.unsqueeze(-1) == map_cls.unsqueeze(1)
    w = w * (ok & same).to(w.dtype)

    # Mutual best. A one-sided nearest neighbour lets every detection claim the
    # same map point, which the closed form then averages into a pose pulled
    # towards whatever is densest rather than towards what matches.
    best_row = w == w.max(dim=2, keepdim=True).values
    best_col = w == w.max(dim=1, keepdim=True).values
    return w * (best_row & best_col & (w > 0)).to(w.dtype)


class GeometricBaseline(torch.nn.Module):
    """A model-shaped object that never learned anything.

    It answers the four keys ``engine/evaluator.py`` and ``engine/sequence.py``
    read, so both score it without knowing the difference.
    """

    def __init__(self, p: ModelParams, sigma: float, iters: int = 1):
        """@param p The model config, for the solver and grid shapes only.
        @param sigma Kernel width, in metres.
        @param iters Re-match this many times, moving detections between.
        """
        super().__init__()
        self.sigma, self.iters = sigma, iters
        self.procrustes = ProcrustesPoseHead(
            p.irls_iters, p.irls_scale_m, p.min_row_mass
        )
        # Zero-initialised, which is how the real one starts: exp(0) is 1, so
        # the covariance that comes out is the surface itself with no learned
        # correction, and the trust logit is 0 -- half, which passes the gate.
        self.volume = VolumeHead(p.dim, p.num_heads, p.grid)
        with torch.no_grad():
            for layer in (self.volume.log_scale, self.volume.trust):
                layer.weight.zero_()
                layer.bias.zero_()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        det_xy = batch["det_pts"].flatten(1, 2)
        map_xy = batch["map_pts"].flatten(1, 2)
        det_ok = batch["det_pmask"].flatten(1)
        map_ok = batch["map_pmask"].flatten(1)
        pts = batch["det_pts"].shape[2]
        det_cls = batch["det_cls"].repeat_interleave(pts, dim=1)
        map_cls = batch["map_cls"].repeat_interleave(pts, dim=1)

        # Iterated, because the model re-matches after moving detections and a
        # one-shot baseline would be losing to ICP rather than to a transformer.
        moved = det_xy
        for _ in range(self.iters):
            assign = geometric_assign(
                moved, det_ok, det_cls, map_xy, map_ok, map_cls, self.sigma
            )
            delta, mass = self.procrustes(assign, det_xy, map_xy)
            moved = G.transform_points(delta, det_xy)

        # The pool never sees a real token, but log_scale and trust are zeroed,
        # so nothing it produces reaches the answer.
        b = det_xy.shape[0]
        tokens = det_xy.new_zeros(b, 1, self.volume.log_scale.in_features)
        pad = det_xy.new_zeros(b, 1, dtype=torch.bool)
        out = self.volume(assign, det_xy, map_xy, tokens, pad)
        out["delta"] = delta
        out["mass"] = mass
        return out
