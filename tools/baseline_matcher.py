#!/usr/bin/env python3
"""Match geometrically instead of with a transformer, and score it the same way.

    tools/baseline_matcher.py --split test
    tools/baseline_matcher.py --split test --sigma 1.5

Every ablation in RESULTS.md varies something *inside* the learned matcher.
None asks whether it needs to be learned at all. This asks; RESULTS.md has the
answer and what it means.

Two choices here are worth knowing before reading the code. The covariance is
the raw surface with the learned scale zeroed, so it reports what the geometry
alone supports. And the baseline sees the current frame only, where the model
folds in two past frames -- handicapped deliberately, because a classical
matcher given temporal fusion is a different experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer import geometry as G
from mapposeformer.config import parse_overrides, upgrade, with_overrides
from mapposeformer.data import build_dataset
from mapposeformer.engine import evaluate, format_report
from mapposeformer.model import MapPoseFormer
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

    It answers the four keys ``engine/evaluator.py`` reads, so the same
    evaluator scores it without knowing the difference.
    """

    def __init__(self, cfg, sigma: float, iters: int = 1):
        super().__init__()
        self.sigma, self.iters = sigma, iters
        self.procrustes = ProcrustesPoseHead(
            cfg.model.irls_iters, cfg.model.irls_scale_m, cfg.model.min_row_mass
        )
        # Zero-initialised, which is how the real one starts: exp(0) is 1, so
        # the covariance that comes out is the surface itself with no learned
        # correction, and the trust logit is 0 -- half, which passes the gate.
        self.volume = VolumeHead(
            cfg.model.dim, cfg.model.num_heads, cfg.model.grid
        )
        with torch.no_grad():
            self.volume.log_scale.weight.zero_()
            self.volume.log_scale.bias.zero_()
            self.volume.trust.weight.zero_()
            self.volume.trust.bias.zero_()

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
        out = self.volume(assign, det_xy, map_xy, tokens, pad, delta)
        out["delta"] = delta
        out["mass"] = mass
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        default="runs/m1_base/best.pt",
        help="read the data config from here, and compare against it",
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument(
        "--iters",
        type=int,
        default=1,
        help="re-match after moving detections, as the model does",
    )
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = with_overrides(upgrade(ck["config"]), parse_overrides(args.overrides))
    ds = build_dataset(cfg.data, args.split)

    def run(model, label):
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4)
        print(f"\n=== {label}")
        print(
            format_report(
                evaluate(model.to(args.device).eval(), loader, args.device, 0.5)
            )
        )

    run(
        GeometricBaseline(cfg, args.sigma, args.iters),
        f"geometric, sigma={args.sigma}, iters={args.iters}",
    )
    learned = MapPoseFormer(cfg.model)
    learned.load_state_dict(ck["model"])
    run(learned, f"learned ({args.checkpoint})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
