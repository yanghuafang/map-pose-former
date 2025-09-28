#!/usr/bin/env python3
"""Draw one sample as an SVG, to see what the model is actually given.

    tools/viz_sample.py --index 30 --out /tmp/sample.svg
    tools/viz_sample.py --index 30 --checkpoint runs/base/best.pt --out p.svg

Layers, in the anchor frame:

    grey    the local map, as the prior pose believes it lies
    orange  detections from previous frames, warped here by measured egomotion
    red     this frame's detections, where they arrive -- offset by the prior
    green   all of them after the true correction, i.e. aligned
    blue    all of them after the *model's* correction, if a checkpoint is given

The distance between red and green is the problem. The distance between blue
and green is the error. Nothing in a scalar metric shows a ridge along the
road; this does, and looking at it before trusting a number is a habit worth
forming.

Orange is worth a second look on its own. It is the accumulated evidence, and
whether it lands *on top of* the red is the question the temporal path lives or
dies by: if the egomotion warp is right the two agree, and the model has three
frames of landmarks to match instead of one.

SVG by hand rather than matplotlib: it is fifty lines, it has no dependency,
and the output opens in any browser -- including over ssh from the training
box."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer import geometry as G
from mapposeformer.config import Config
from mapposeformer.data.classes import LandmarkClass
from mapposeformer.data.synthetic import SyntheticDataset

_W, _H, _PAD = 900, 900, 30


def _project(pts: torch.Tensor, span: float) -> torch.Tensor:
    """Vehicle frame (X forward, Y left) to SVG pixels (X right, Y down).

    Rotated so that forward is *up* on the page, which is how anyone looking at
    a road scene expects to read it.
    """
    scale = (min(_W, _H) - 2 * _PAD) / (2 * span)
    x = _W / 2 - pts[..., 1] * scale
    y = _H / 2 - pts[..., 0] * scale
    return torch.stack([x, y], dim=-1)


def _draw(pts, pmask, cls, span, colour, width, dot) -> list[str]:
    out = []
    for i in range(pts.shape[0]):
        keep = pmask[i]
        if not bool(keep.any()):
            continue
        p = _project(pts[i][keep], span)
        name = LandmarkClass(int(cls[i])).name.lower()
        if p.shape[0] == 1:
            out.append(
                f'<circle cx="{p[0, 0]:.1f}" cy="{p[0, 1]:.1f}" r="{dot}" '
                f'fill="{colour}"><title>{name}</title></circle>'
            )
        else:
            d = " ".join(f"{q[0]:.1f},{q[1]:.1f}" for q in p)
            out.append(
                f'<polyline points="{d}" fill="none" stroke="{colour}" '
                f'stroke-width="{width}"><title>{name}</title></polyline>'
            )
    return out


def _history(
    sample: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Past detections in the current ego frame -- what the model sees.

    Exactly the composition ``model._detections`` performs, done here so the
    picture cannot drift from the tensor: if this warp were wrong the orange
    would miss the red, which is the failure the layer exists to make visible.
    """
    pts, pmask, cls = [], [], []
    for k in range(sample["hist_rel"].shape[0]):
        e, p, _ = sample["hist_pts"][k].shape
        warped = G.transform_points(
            sample["hist_rel"][k], sample["hist_pts"][k].reshape(e * p, 2)
        ).view(e, p, 2)
        pts.append(warped)
        pmask.append(sample["hist_pmask"][k])
        cls.append(sample["hist_cls"][k])
    if not pts:
        return (
            torch.zeros(0, 1, 2),
            torch.zeros(0, 1, dtype=torch.bool),
            torch.zeros(0, dtype=torch.long),
        )
    return torch.cat(pts), torch.cat(pmask), torch.cat(cls)


def render(
    sample: dict[str, torch.Tensor], span: float, pred: torch.Tensor | None
) -> str:
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
        f'viewBox="0 0 {_W} {_H}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
    ]
    hp, hm, hc = _history(sample)
    # Everything the model matches against the map: this frame, then the past.
    all_pts = torch.cat([sample["det_pts"], hp])
    all_pmask = torch.cat([sample["det_pmask"], hm])
    all_cls = torch.cat([sample["det_cls"], hc])

    body += _draw(
        sample["map_pts"],
        sample["map_pmask"],
        sample["map_cls"],
        span,
        "#9aa0a6",
        3,
        5,
    )
    body += _draw(hp, hm, hc, span, "#f29900", 2, 3)
    body += _draw(
        sample["det_pts"],
        sample["det_pmask"],
        sample["det_cls"],
        span,
        "#d93025",
        2,
        4,
    )
    for pose, colour in ((sample["delta"], "#188038"), (pred, "#1a73e8")):
        if pose is None:
            continue
        moved = G.transform_points(pose, all_pts.flatten(0, 1)).view_as(all_pts)
        body += _draw(moved, all_pmask, all_cls, span, colour, 2, 4)
    origin = _project(torch.zeros(1, 2), span)[0]
    body.append(
        f'<circle cx="{origin[0]:.1f}" cy="{origin[1]:.1f}" r="6" fill="#000"/>'
    )
    legend = [
        ("#9aa0a6", "map (anchor frame)"),
        ("#f29900", "past frames, warped here by egomotion"),
        ("#d93025", "detections, as received"),
        ("#188038", "all detections + true correction"),
    ]
    if pred is not None:
        legend.append(("#1a73e8", "all detections + predicted correction"))
    for i, (colour, text) in enumerate(legend):
        y = 24 + 20 * i
        body.append(
            f'<rect x="16" y="{y - 10}" width="14" height="4" fill="{colour}"/>'
        )
        body.append(
            f'<text x="38" y="{y}" font-family="monospace" '
            f'font-size="13">{text}</text>'
        )
    body.append("</svg>")
    return "\n".join(body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="sample.svg")
    ap.add_argument("--checkpoint")
    args = ap.parse_args()

    cfg = Config()
    pred = None
    if args.checkpoint:
        from mapposeformer.model import MapPoseFormer

        ckpt = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        cfg = ckpt["config"]
        model = MapPoseFormer(cfg.model)
        model.load_state_dict(ckpt["model"])
        model.eval()

    sample = SyntheticDataset(cfg.data, args.split)[args.index]
    if args.checkpoint:
        with torch.no_grad():
            pred = model({k: v.unsqueeze(0) for k, v in sample.items()})[
                "delta"
            ][0]

    span = cfg.data.sample.map_radius_m
    Path(args.out).write_text(render(sample, span, pred))
    print(f"wrote {args.out}  (true correction {sample['delta'].tolist()})")
    if pred is not None:
        print(f"                 predicted     {pred.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
