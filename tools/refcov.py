#!/usr/bin/env python3
"""The covariance the model *should* report, computed rather than assumed.

    tools/refcov.py runs/base/best.pt --frames 128 --draws 32

Every calibration number in this project is a scalar -- NEES, ANEES, coverage
-- and a scalar cannot see the orientation of an ellipse. A covariance can pass
every one of them while being 20x overconfident on the 1% of frames where
along-track geometry aliases: measured error there averages 2.249 m against a
claimed 0.308 m, with a NEES median of 0.786 calling the whole thing honest.
That failure is invisible to any statistic that reduces a 3x3 matrix to one
number.

(How to read the ratios below. A median of per-frame ratios and a ratio of RMS
are not the same statistic, and on a heavy-tailed error they describe different
populations: 0.809 measured one way against 3.030 measured the other shows an
inversion that is not there. Compared like for like, both sit on the same side
of 1.0 and the ellipse points the right way. So the machinery here guards
against an inversion no measurement has yet found -- which is what an
instrument is for.)

This computes the reference the scalars are hiding. For a fixed scene and
frame -- fixed geometry, fixed landmarks -- the sample seed is redrawn K
times, which redraws the prior error and the detection noise: exactly the
randomness the reported covariance claims to describe. The empirical
covariance of the K resulting pose errors is what the model *ought* to be
reporting, as a matrix, with an orientation.

**The ICP literature approximates this object because it cannot draw from the
prior.** Here the prior is drawn by the data generator, so it is directly
computable, and there is no excuse for approximating it.

What it reports per frame: the reported and reference covariances, their size
ratio, and the angle between their principal axes. A model whose ellipse is
the right size and the wrong way round is the failure this exists to catch.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.checkpoint import build_model
from mapposeformer.config import parse_overrides, upgrade, with_overrides
from mapposeformer.data.sample import build_sample
from mapposeformer.data.synthetic import SyntheticDataset
from mapposeformer.metrics import pose_error

#: An ellipse this close to circular has no meaningful major axis, so its
#: bearing is noise and reporting it would invent a finding. Measured on a
#: four-layer model at K=16 the reference came back at 0.942 -- essentially
#: isotropic -- and the angle against it read 39.5 degrees, which says nothing.
ROUND_ENOUGH = 1.2


def principal_angle(cov: torch.Tensor) -> float | None:
    """@return Bearing of the translation block's major axis in degrees, or
        ``None`` when the ellipse is too round for that to mean anything.

    Only the 2x2 translation block: yaw is a different unit, and putting it in
    an eigendecomposition compares metres with radians.
    """
    vals, vecs = torch.linalg.eigh(cov[:2, :2].double())
    lo, hi = float(vals.min()), float(vals.max())
    if lo <= 0 or (hi / lo) ** 0.5 < ROUND_ENOUGH:
        return None
    major = vecs[:, int(torch.argmax(vals))]
    return math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="test")
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--draws", type=int, default=32, help="K priors per frame")
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = upgrade(ckpt["config"])
    if args.overrides:
        cfg = with_overrides(cfg, parse_overrides(args.overrides))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Via `build_model` so a pruned or otherwise reshaped checkpoint loads;
    # see `checkpoint.py` on why five sites must not each do this themselves.
    model = build_model(ckpt, cfg.model).to(device).eval()

    # The generator, not the cache: this needs to redraw samples, which is the
    # one thing a materialised split cannot do.
    ds = SyntheticDataset(cfg.data, args.split)
    keys = ds.sequences()
    frames = ds.frames_of(keys[0])

    rows = []
    with torch.no_grad():
        for i in range(args.frames):
            key = keys[i % len(keys)]
            frame = frames[(i // len(keys)) % len(frames)]
            world, chunked = ds._world(key)
            base = ((ds.base + key) * 100_003 + frame) * 97

            batch = []
            for k in range(args.draws):
                s = build_sample(
                    world, chunked, frame, base + 7919 * k, cfg.data.sample
                )
                batch.append(s)
            stacked = {
                k: torch.stack([b[k] for b in batch]).to(device)
                for k in batch[0]
            }
            out = model(stacked)

            # The error each draw actually made, and the covariance the model
            # reported for it. The reference is the spread of the first; the
            # claim is the mean of the second.
            err = pose_error(
                out["delta"].float(), stacked["delta"].float()
            ).cpu()
            reference = torch.cov(err.T.double())
            reported = out["cov"].double().mean(0).cpu()

            rows.append(
                {
                    "ref_long": float(reference[0, 0].clamp_min(0).sqrt()),
                    "ref_lat": float(reference[1, 1].clamp_min(0).sqrt()),
                    "rep_long": float(reported[0, 0].clamp_min(0).sqrt()),
                    "rep_lat": float(reported[1, 1].clamp_min(0).sqrt()),
                    "ref_angle": principal_angle(reference),
                    "rep_angle": principal_angle(reported),
                }
            )
            if i % 32 == 0:
                print(f"  frame {i}/{args.frames}", flush=True)

    def med(k):
        return float(torch.tensor([r[k] for r in rows]).median())

    ref_ratio = med("ref_long") / max(med("ref_lat"), 1e-9)
    rep_ratio = med("rep_long") / max(med("rep_lat"), 1e-9)
    # Angles live on a half-circle, so the difference wraps at 90 degrees.
    # Only frames where *both* ellipses are elongated enough to have an axis.
    pairs = [
        (r["ref_angle"], r["rep_angle"])
        for r in rows
        if r["ref_angle"] is not None and r["rep_angle"] is not None
    ]
    dth = (
        torch.tensor([abs(a - b) for a, b in pairs])
        if pairs
        else torch.tensor([])
    )
    if dth.numel():
        dth = torch.minimum(dth, 180.0 - dth)

    print(f"\n{args.checkpoint}  {len(rows)} frames x {args.draws} draws")
    print(f"{'':<22}{'sigma_long':>12}{'sigma_lat':>11}{'ratio':>9}")
    print(
        f"{'reference (measured)':<22}{med('ref_long'):>12.4f}"
        f"{med('ref_lat'):>11.4f}{ref_ratio:>9.3f}"
    )
    print(
        f"{'reported by the model':<22}{med('rep_long'):>12.4f}"
        f"{med('rep_lat'):>11.4f}{rep_ratio:>9.3f}"
    )
    print(
        f"\nsize   reported/reference:"
        f" long {med('rep_long') / max(med('ref_long'), 1e-9):.2f}x"
        f"   lat {med('rep_lat') / max(med('ref_lat'), 1e-9):.2f}x"
    )
    print(
        f"shape  anisotropy reported {rep_ratio:.3f}"
        f" against measured {ref_ratio:.3f}"
    )
    # "Inverted" only means something when the reference is actually elongated.
    # Calling a near-circular reference inverted is a coin flip dressed up as a
    # finding, which is the mistake this whole tool exists to avoid.
    if abs(ref_ratio - 1.0) < (ROUND_ENOUGH - 1.0):
        print(
            "       reference is near-isotropic; orientation is not"
            " determined and no verdict is given"
        )
    else:
        verdict = (
            "RIGHT WAY UP"
            if (rep_ratio - 1) * (ref_ratio - 1) > 0
            else "INVERTED"
        )
        print(f"       {verdict}")
    if dth.numel():
        print(
            f"       principal axis off by {float(dth.median()):.1f} deg"
            f" (median over {dth.numel()} of {len(rows)} frames"
            f" elongated enough to have one)"
        )
    else:
        print(
            "       no frame had both ellipses elongated enough to compare axes"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
