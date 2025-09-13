"""Pose error, decomposed the way it needs to be decomposed.

A single translation RMSE hides the failure this problem is prone to. Lane
geometry aliases along the road, so a model that has slid a metre forward pays
almost nothing for it -- and a steady one-metre along-track lag and a metre of
symmetric along-track jitter have *identical* RMSE. Only the signed mean tells
them apart, and only the along-track/lateral split says which axis it is on.

So every number here comes in three forms: an RMSE, a signed bias, and a worst
case, on each of the three axes the model actually predicts. This mirrors
``kitti::PoseError`` and ``ErrorSummary`` in camera-map-localization, down to
resolving the error onto the **ground-truth** axes rather than the estimate's --
the axes an error is reported on must not move with the error being reported,
or a heading mistake rotates its own yardstick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from mapposeformer import geometry as G


def pose_error(pred: Tensor, gt: Tensor) -> Tensor:
    """``(B, 3)`` signed error as ``(longitudinal_m, lateral_m, yaw_rad)``.

    Both arguments are corrections in the anchor frame. Positive longitudinal
    means the estimate is *ahead* of truth; positive lateral means *left*.
    """
    return G.relative(gt, pred)


@dataclass
class ErrorSummary:
    """Accumulates errors over a split and reports them.

    Streaming rather than storing every frame, so a long evaluation costs a
    fixed amount of memory.
    """

    n: int = 0
    _sum: Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float64))
    _sumsq: Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float64))
    _maxabs: Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float64))
    _trans_sq: float = 0.0
    _recall: Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float64))

    #: Success thresholds, as ``(translation_m, yaw_deg)``. A frame counts only
    #: if it meets both: a pose that is 10 cm out but 5 degrees off is not a
    #: localization.
    THRESHOLDS = ((0.25, 0.5), (0.5, 1.0), (1.0, 2.0))

    def update(self, pred: Tensor, gt: Tensor) -> None:
        e = pose_error(pred.detach(), gt.detach()).double().cpu()
        self.n += e.shape[0]
        self._sum += e.sum(0)
        self._sumsq += e.square().sum(0)
        self._maxabs = torch.maximum(self._maxabs, e.abs().max(0).values)
        trans = e[:, :2].norm(dim=-1)
        self._trans_sq += float(trans.square().sum())
        yaw_deg = e[:, 2].abs().rad2deg()
        for i, (tm, yd) in enumerate(self.THRESHOLDS):
            self._recall[i] += float(((trans <= tm) & (yaw_deg <= yd)).sum())

    def as_dict(self) -> dict[str, float]:
        n = max(self.n, 1)
        rmse = (self._sumsq / n).sqrt()
        bias = self._sum / n
        deg = 180.0 / math.pi
        out = {
            "rmse_trans_m": math.sqrt(self._trans_sq / n),
            "rmse_long_m": float(rmse[0]),
            "rmse_lat_m": float(rmse[1]),
            "rmse_yaw_deg": float(rmse[2]) * deg,
            "bias_long_m": float(bias[0]),
            "bias_lat_m": float(bias[1]),
            "bias_yaw_deg": float(bias[2]) * deg,
            "max_long_m": float(self._maxabs[0]),
            "max_lat_m": float(self._maxabs[1]),
            "max_yaw_deg": float(self._maxabs[2]) * deg,
        }
        for i, (tm, yd) in enumerate(self.THRESHOLDS):
            out[f"recall_{tm}m_{yd}deg"] = float(self._recall[i]) / n
        return out

    def format(self) -> str:
        d = self.as_dict()
        lines = [f"frames {self.n}", "               rmse      bias      max"]
        for label, key in (
            ("long  m", "long_m"),
            ("lat   m", "lat_m"),
            ("yaw deg", "yaw_deg"),
        ):
            lines.append(
                f"  {label}   {d['rmse_' + key]:8.3f}  "
                f"{d['bias_' + key]:8.3f}  {d['max_' + key]:8.3f}"
            )
        lines.append(f"  trans m   {d['rmse_trans_m']:8.3f}")
        lines += [
            f"  recall <= {tm} m, {yd} deg : {d[f'recall_{tm}m_{yd}deg']:.3f}"
            for tm, yd in self.THRESHOLDS
        ]
        return "\n".join(lines)
