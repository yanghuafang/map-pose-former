#!/usr/bin/env python3
"""Compare two models on the same frames, frame by frame.

    tools/paired.py runs/teacher/best.pt runs/teacher_int8/best.pt
    tools/paired.py runs/a/best.pt runs/b/best.pt --split test --device cuda

**Why this exists rather than another seed sweep.** Every structural question
in this project was decided by training three seeds a configuration and asking
whether the gap cleared the seed band. That is the right tool when the two
things differ by *training* -- different weights, different initialisation,
different noise -- and it is the wrong one here. Pruning, quantization and
distillation produce a model that is a deterministic function of another model,
so the two can be run over the *same* frames and compared per frame. Nothing is
redrawn, so nothing needs averaging over.

The difference in resolution is not marginal. The seed band of the point-token,
point-to-point configuration puts the half-width on a recall difference at
about 0.62 pp, so a compression step costing half a point is invisible. Paired
on 8 880 test frames, the half-width is ``1.96*sqrt(b + c)/8880`` over the
*discordant* frames alone -- about **0.07 pp** when ten frames disagree. Ten to
twenty times sharper, at no training cost, because a frame both models get
right carries no information about which is better and is correctly given no
weight.

**So the test is McNemar's**, not a difference of proportions. `b` is frames
the baseline gets right and the candidate gets wrong; `c` the reverse. Frames
they agree on -- the overwhelming majority -- are ignored, which is the whole
point.

What it reports beyond recall: how far the poses actually moved, against the
250 mm gate that defines recall, and what happened to the covariance. A
compression step that holds recall while moving NEES has not succeeded: a
number that looks fine on accuracy can still be wrong about uncertainty.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.checkpoint import load_checkpoint
from mapposeformer.data import build_dataset
from mapposeformer.metrics import pose_error

#: The gate that defines this project's headline recall, from
#: ``ErrorSummary.THRESHOLDS``.
GATE_M, GATE_DEG = 0.25, 0.5


def _hit(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-frame recall indicator, computed exactly as `ErrorSummary` does."""
    e = pose_error(pred.detach(), gt.detach()).double().cpu()
    trans = e[:, :2].norm(dim=-1)
    yaw_deg = e[:, 2].abs().rad2deg()
    return (trans <= GATE_M) & (yaw_deg <= GATE_DEG)


def _nees(
    pred: torch.Tensor, gt: torch.Tensor, cov: torch.Tensor
) -> torch.Tensor:
    """Per-frame NEES per degree of freedom.

    Factorised on the CPU for the reason `metrics.py` gives: these are 3x3
    matrices and cuSOLVER's workspace allocation fails on a card that is busy.
    """
    e = pose_error(pred.detach(), gt.detach()).double().unsqueeze(-1).cpu()
    chol = torch.linalg.cholesky(cov.detach().double().cpu())
    w = torch.linalg.solve_triangular(chol, e, upper=False).squeeze(-1)
    return w.square().sum(-1) / 3.0


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """@return ``(delta, half_width)`` on the recall difference, as fractions
        of the *discordant* count, before dividing by the frame total.

    The normal approximation is used rather than the exact binomial: at the
    counts this will see it agrees to well under the precision anyone reads,
    and it degrades to a stated infinity at ``b + c == 0`` rather than to a
    misleading zero. No p-value is reported -- these frames are strongly
    correlated within a scene, so a p-value would assume an independence that
    is not there, the same reason `metrics.py` reports the KS statistic alone.
    """
    n = b + c
    if n == 0:
        return 0.0, float("inf")
    return float(c - b), 1.96 * math.sqrt(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", help="the model being compared against")
    ap.add_argument("candidate", help="the compressed / distilled model")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="frames, 0 = all")
    # Quantization is applied in this process rather than saved and reloaded.
    # `quantize` swaps every Linear for a FakeQuantLinear, so the result no
    # longer matches the state dict a fresh model expects -- a quantized
    # checkpoint is not loadable without a config field that says to rebuild it
    # that way. Doing it here measures the same thing and adds no format.
    ap.add_argument(
        "--quantize",
        action="store_true",
        help="quantize the candidate in-process, calibrated on the train split",
    )
    ap.add_argument("--calib-batches", type=int, default=16)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # `load_checkpoint` returns the model on the CPU and as constructed, which
    # is right for a trainer resuming its own run and wrong here. Eval mode and
    # no-grad are stated rather than assumed: dropout and running statistics
    # would otherwise make the two models differ for a reason that has nothing
    # to do with the compression being measured.
    base, cfg = load_checkpoint(args.baseline)
    cand, _ = load_checkpoint(args.candidate)
    base = base.to(device).eval().requires_grad_(False)
    cand = cand.to(device).eval().requires_grad_(False)

    if args.quantize:
        # Imported here: quantization is opt-in, and an ordinary paired
        # comparison should not need the module at all.
        from mapposeformer.quantize import (
            QuantParams,
            calibrate,
            quantize,
            weight_bytes,
        )

        n = quantize(cand, QuantParams())
        # Calibrated on TRAIN, never on the split being measured. The
        # activation ranges are learned parameters like any other, and learning
        # them from test frames would be reporting a number the model was
        # tuned on.
        calib = build_dataset(cfg.data, "train")
        cl = DataLoader(calib, batch_size=cfg.train.batch_size, num_workers=2)
        seen = calibrate(
            cand,
            ({k: v.to(device) for k, v in b.items()} for b in cl),
            limit=args.calib_batches,
        )
        fp32_b, quant_b = weight_bytes(cand, QuantParams())
        print(f"quantized {n} Linear modules, calibrated {seen} on train")
        print(
            f"  weights {fp32_b / 2**20:.2f} MiB fp32"
            f" -> {quant_b / 2**20:.2f} MiB int8"
            f"  ({100 * (1 - quant_b / max(fp32_b, 1)):.1f}% smaller)"
        )
        # Said out loud because it is the number INT8 is usually sold on and
        # the one that matters least here: `quantize.py` records that the
        # weights are single-digit MB against roughly 7 GiB of activations.
        print(
            "  (weights are the small half of the cost; activations dominate)\n"
        )

    ds = build_dataset(cfg.data, args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size or cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        pin_memory=device.startswith("cuda"),
    )

    hit_b, hit_c, moved, nees_b, nees_c = [], [], [], [], []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # The SAME batch object into both, which is what makes this paired
            # rather than two runs that happen to use one seed. No reliance on
            # loader determinism, no second pass over the data.
            ob, oc = base(batch), cand(batch)
            gt = batch["delta"]
            hit_b.append(_hit(ob["delta"], gt))
            hit_c.append(_hit(oc["delta"], gt))
            d = (
                pose_error(oc["delta"].detach(), ob["delta"].detach())
                .double()
                .cpu()
            )
            moved.append(d[:, :2].norm(dim=-1))
            nees_b.append(_nees(ob["delta"], gt, ob["cov"]))
            nees_c.append(_nees(oc["delta"], gt, oc["cov"]))
            seen += gt.shape[0]
            if args.limit and seen >= args.limit:
                break

    hb, hc = torch.cat(hit_b), torch.cat(hit_c)
    mv = torch.cat(moved)
    nb, nc = torch.cat(nees_b), torch.cat(nees_c)
    n = hb.numel()

    b = int((hb & ~hc).sum())  # baseline right, candidate wrong
    c = int((~hb & hc).sum())  # candidate right, baseline wrong
    delta, half = mcnemar(b, c)

    def q(t, p):
        return float(t.quantile(torch.tensor(p, dtype=t.dtype)))

    print(f"baseline  {args.baseline}")
    print(f"candidate {args.candidate}")
    print(f"split={args.split}  frames={n}\n")
    print(f"  recall baseline   {float(hb.double().mean()):.4f}")
    print(f"  recall candidate  {float(hc.double().mean()):.4f}")
    print(
        f"  discordant        {b + c}  (baseline-only {b}, candidate-only {c})"
    )
    if half == float("inf"):
        print("  delta recall      0.0000 pp -- the two agree on every frame")
    else:
        print(
            f"  delta recall      {delta / n * 100:+.4f} pp"
            f"  +/- {half / n * 100:.4f}"
            f"   {'SEPARATED' if abs(delta) > half else 'not separated'}"
        )
    print()
    print(
        f"  pose moved        median {q(mv, 0.5) * 1000:.1f} mm"
        f"   p99 {q(mv, 0.99) * 1000:.1f} mm"
        f"   (gate is {GATE_M * 1000:.0f} mm)"
    )
    ratio = nc / nb.clamp_min(1e-12)
    print(
        f"  NEES median       {q(nb, 0.5):.3f} -> {q(nc, 0.5):.3f}"
        f"   (0.789 is honest)"
    )
    print(
        f"  NEES ratio        median {q(ratio, 0.5):.3f}"
        f"   p99 {q(ratio, 0.99):.3f}"
    )
    # Stated separately because recall and calibration fail independently, and
    # a compression step that holds one while moving the other is the case this
    # tool exists to make visible.
    if abs(q(ratio, 0.5) - 1.0) > 0.25:
        print(
            "  ^ the covariance moved by more than a quarter; recall alone"
            " does not describe this change"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
