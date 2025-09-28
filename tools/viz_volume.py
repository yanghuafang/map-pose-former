"""Draw a cost surface over the pose grid, in either of its two senses.

    tools/viz_volume.py --mode fit               # the fit's own curvature
    tools/viz_volume.py --mode reassoc           # nearest-map-point cost
    tools/viz_volume.py --mode reassoc --keep-classes 0,1   # lanes alone

``fit`` is what the model reports: the assignment-weighted squared error at each
hypothesis, correspondences **held fixed**. It answers how well the pose is
determined *given* that these matches are right.

``reassoc`` lets every detection re-choose its nearest map point at each
hypothesis, which is what a distance transform does. It answers whether the pose
could be somewhere else and look as good.

Only the second produces a **ridge**, and that is the limitation worth seeing:
with the assignment fixed, sliding a hypothesis moves every point off its own
target, so the cost rises in every direction. ``docs/RESULTS.md`` has the
numbers; ``docs/ROADMAP.md`` has the affordable repair.

The assignment is an **oracle** by default -- apply the true correction and take
the map point each detection lands on -- which separates the geometry from the
model. ``--checkpoint`` draws the model's own assignment instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer import geometry as G
from mapposeformer.config import (
    Config,
    parse_overrides,
    with_overrides,
)
from mapposeformer.data.synthetic import SyntheticDataset
from mapposeformer.model.volume_head import GridParams, grid_cost

_W, _H, _PAD, _TOP = 720, 540, 60, 46

# : Distance past which a detection is simply unmatched, so a hypothesis cannot
# : be rewarded for dragging a point towards something implausibly far away.
_TRUNCATE_M = 2.0


def oracle_assignment(
    sample: dict[str, torch.Tensor], radius_m: float
) -> torch.Tensor:
    """The correspondence the labels imply: correct, and available to no model.

    Apply the true correction to the detections and take the nearest map point
    within ``radius_m`` -- exactly the target ``losses.match_loss`` trains
    towards, so a surface drawn through it is the one a perfect matcher gives.
    """
    det = sample["det_pts"].flatten(0, 1)
    mp = sample["map_pts"].flatten(0, 1)
    dist = torch.cdist(G.transform_points(sample["delta"], det), mp)
    dist = dist.masked_fill(
        ~sample["map_pmask"].flatten(0).unsqueeze(0), float("inf")
    )
    best, idx = dist.min(dim=1)

    assign = torch.zeros(det.shape[0], mp.shape[0])
    hit = sample["det_pmask"].flatten(0) & (best <= radius_m)
    assign[hit.nonzero().flatten(), idx[hit]] = 1.0
    return assign.unsqueeze(0)


def fit_cost(assign, det, mp, cell_t, cell_rot) -> torch.Tensor:
    """The model's surface: correspondences fixed, mean squared residual."""
    cost, mass = grid_cost(assign, det, mp, cell_t, cell_rot)
    return (cost / mass.clamp_min(1e-6).unsqueeze(-1)).squeeze(0)


def reassociated_cost(
    det, mp, valid_det, valid_map, cell_t, cell_rot
) -> torch.Tensor:
    """The classical surface: nearest map point, chosen afresh per hypothesis.

    Brute force over ``hypotheses x detections x map points``. Affordable here
    because this is one frame in a drawing tool, and *not* affordable in a
    training step -- which is the whole reason the model reports the other one.
    """
    d = det[valid_det]
    m = mp[valid_map]
    moved = G.transform_points(
        torch.cat([cell_t, torch.zeros(len(cell_t), 1)], -1),
        d.unsqueeze(0).expand(len(cell_t), -1, -1),
    )
    near = (
        torch.cdist(moved, m.unsqueeze(0).expand(len(cell_t), -1, -1))
        .min(-1)
        .values
    )
    return near.clamp_max(_TRUNCATE_M).square().mean(-1)


def _colour(t: float) -> str:
    """Dark blue through magenta to yellow: ordered, and legible in print."""
    stops = (
        (13, 8, 66),
        (60, 20, 120),
        (140, 40, 110),
        (220, 90, 60),
        (250, 220, 60),
    )
    x = max(0.0, min(1.0, t)) * (len(stops) - 1)
    i = min(int(x), len(stops) - 2)
    f = x - i
    lo, hi = stops[i], stops[i + 1]
    r, g, b = (round(a + f * (c - a)) for a, c in zip(lo, hi, strict=True))
    return f"#{r:02x}{g:02x}{b:02x}"


def render(
    prob: torch.Tensor, grid: GridParams, truth, title: str, note: str
) -> str:
    """``prob`` is ``(num_x, num_y)`` over (forward, left); bright is likely."""
    # Forward up the page and left to the left, which is how anyone looking at
    # a road scene expects to read it.
    cw = (_W - 2 * _PAD) / grid.num_y
    ch = (_H - _PAD - _TOP) / grid.num_x
    p = prob / prob.max().clamp_min(1e-12)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
        f'viewBox="0 0 {_W} {_H}">'
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{_PAD}" y="22" font-family="monospace" '
        f'font-size="14">{title}</text>',
        f'<text x="{_PAD}" y="39" font-family="monospace" font-size="12" '
        f'fill="#5f6368">{note}</text>',
    ]
    for i in range(grid.num_x):
        for j in range(grid.num_y):
            out.append(
                f'<rect x="{_PAD + j * cw:.1f}" y="{_TOP + i * ch:.1f}" '
                f'width="{cw + 1:.1f}" height="{ch + 1:.1f}" '
                f'fill="{_colour(float(p[i, j]))}"/>'
            )
    cx = _PAD + (0.5 - float(truth[1]) / (2 * grid.extent_y_m)) * (
        _W - 2 * _PAD
    )
    cy = _TOP + (0.5 - float(truth[0]) / (2 * grid.extent_x_m)) * (
        _H - _PAD - _TOP
    )
    out.append(
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="none" '
        f'stroke="#3ddc84" stroke-width="3"/>'
        f'<text x="{cx + 12:.1f}" y="{cy + 4:.1f}" fill="#3ddc84" '
        f'font-family="monospace" font-size="12">truth</text>'
    )
    out.append(
        f'<text x="{_PAD}" y="{_H - 14}" font-family="monospace" '
        f'font-size="11" '
        f'fill="#5f6368">horizontal: left +{grid.extent_y_m:g} m to '
        f"-{grid.extent_y_m:g} m &#183; vertical: forward "
        f"+{grid.extent_x_m:g} m to "
        f"-{grid.extent_x_m:g} m</text>"
    )
    return "\n".join(out) + "\n</svg>"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("fit", "reassoc"), default="reassoc")
    ap.add_argument("--index", type=int, default=30)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="volume.svg")
    ap.add_argument(
        "--checkpoint", help="use the model's assignment, not the oracle"
    )
    ap.add_argument(
        "--keep-classes", help="comma-separated LandmarkClass values"
    )
    ap.add_argument("--title")
    ap.add_argument("overrides", nargs="*", help="section.field=value")
    args = ap.parse_args()

    cfg, model = Config(), None
    if args.checkpoint:
        from mapposeformer.model import MapPoseFormer

        ckpt = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        cfg = ckpt["config"]
        model = MapPoseFormer(cfg.model)
        model.load_state_dict(ckpt["model"])
        model.eval()

    over = parse_overrides(args.overrides)
    if args.keep_classes:
        over.setdefault("data", {}).setdefault("sample", {})["keep_classes"] = [
            int(c) for c in args.keep_classes.split(",")
        ]
    cfg = with_overrides(cfg, over)

    sample = SyntheticDataset(cfg.data, args.split)[args.index]
    grid = cfg.model.grid
    # The (forward, left) plane at zero yaw. Marginalising over yaw instead
    # would blur the two surfaces towards each other and hide what differs.
    ax = torch.linspace(-grid.extent_x_m, grid.extent_x_m, grid.num_x)
    ay = torch.linspace(-grid.extent_y_m, grid.extent_y_m, grid.num_y)
    cell_t = torch.stack(torch.meshgrid(ax, ay, indexing="ij"), -1).reshape(
        -1, 2
    )
    cell_rot = torch.eye(2).expand(len(cell_t), 2, 2)

    map_xy = sample["map_pts"].flatten(0, 1)
    if model is not None:
        with torch.no_grad():
            out = model({k: v.unsqueeze(0) for k, v in sample.items()})
        assign, det_xy, valid = (
            out["assign"],
            out["det_xy"][0],
            out["det_valid"][0],
        )
        source = "learned assignment"
    else:
        assign = oracle_assignment(sample, cfg.loss.match_radius_m)
        det_xy, valid = (
            sample["det_pts"].flatten(0, 1),
            sample["det_pmask"].flatten(0),
        )
        source = "oracle correspondences"

    if args.mode == "fit":
        cost = fit_cost(
            assign, det_xy.unsqueeze(0), map_xy.unsqueeze(0), cell_t, cell_rot
        )
        note = "correspondences held fixed -- the curvature of the fit itself"
    else:
        cost = reassociated_cost(
            det_xy,
            map_xy,
            valid,
            sample["map_pmask"].flatten(0),
            cell_t,
            cell_rot,
        )
        note = (
            "nearest map point re-chosen at every hypothesis,"
            " as a distance transform does"
        )

    cost = (cost - cost.min()).view(grid.num_x, grid.num_y)
    prob = torch.softmax(-cost.flatten() / cost.max().clamp_min(1e-9) * 6.0, 0)
    prob = prob.view(grid.num_x, grid.num_y)

    kept = list(cfg.data.sample.keep_classes)
    title = (
        args.title
        or f"{args.mode}  |  {source}  |  classes {kept}  |  frame {args.index}"
    )
    Path(args.out).write_text(render(prob, grid, sample["delta"], title, note))

    # The number the picture is a picture of: how much the cost rises when a
    # hypothesis slides, per metre, along each axis.
    i, j = (int(v) for v in (cost == 0).nonzero()[0])
    print(f"wrote {args.out}")
    print(
        f"  cost rise: {float(cost[:, j].max()) / grid.extent_x_m:.3f}"
        f" per m forward, "
        f"{float(cost[i, :].max()) / grid.extent_y_m:.3f} per m lateral"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
