"""Pose error, decomposed the way this problem needs.

A single translation RMSE hides the failure this problem is prone to. Lane
geometry aliases along the road, so a model that has slid a metre forward pays
almost nothing -- and a steady 1 m lag and 1 m of symmetric jitter have
*identical* RMSE. Only the signed mean separates them, and only the
along-track/lateral split says which axis it is on.

So every number comes in three forms: RMSE, signed bias, worst case, on each of
the three axes. This mirrors ``kitti::PoseError`` down to resolving the error on
the **ground-truth** axes -- the axes an error is reported on must not move with
the error, or a heading mistake rotates its own yardstick.

:class:`Calibration` adds what the classical repo does not need: is the
covariance honest? An accurate pose with a dishonest covariance is worse for a
filter than the reverse, because the filter weights by the second number and
cannot check it.
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
class Calibration:
    """Is the reported covariance honest?

    Two statistics, and they fail in different directions:

    **ANEES** -- average normalized estimation error squared, the mean of
    ``rᵀ Σ⁻¹ r`` over three degrees of freedom. One is perfect. Below one the
    model is *pessimistic*: it under-weights its own good frames and the filter
    converges slower than it could. Above one it is **overconfident**, which is
    the dangerous direction -- a filter told a bad frame is certain will follow
    it, and there is no downstream gate that can undo that.

    **Coverage** -- the fraction of frames whose error actually lands inside the
    95% ellipsoid. For three degrees of freedom that is a chi-square threshold
    of 7.815. ANEES is a mean and one catastrophic frame moves it a long way;
    coverage is a count and does not.

    **Median NEES and the tail fraction**, because a mean over a heavy-tailed
    distribution is not a calibration statistic. This model scores ANEES 4.53,
    which reads as a covariance four times too small everywhere; the median
    frame is at 0.78 and 0.4% sit above ten. The two diagnoses have opposite
    repairs, and rescaling to fix the mean would wreck the frames that are
    already honest.

    **The median's target is 0.789, not 1.** A calibrated NEES is chi-square
    distributed, and the chi-square is right-skewed: at three degrees of
    freedom its mean is 3 and its median 2.366. So a perfect estimator scores
    1.0 on the mean and 0.789 on the median, and the same threshold cannot
    serve both -- see :attr:`Calibration.CHI2_MEDIAN`.
    """

    n: int = 0
    _nees_sum: float = 0.0
    _covered: int = 0
    _tail: int = 0
    _nees: list[float] = field(default_factory=list)

    #: NEES per degree of freedom above which a frame is not merely
    #: overconfident but wrong about being confident at all.
    TAIL = 10.0

    #: Chi-square 95th percentile at three degrees of freedom.
    CHI2_95 = 7.815

    #: Chi-square *median* at three degrees of freedom, per dof: what a
    #: perfectly calibrated estimator scores on ``anees_median``. It is 0.789,
    #: not 1.0 -- the chi-square is right-skewed, so its median sits below its
    #: mean. Judging the median against 1.0 calls a textbook-calibrated model
    #: pessimistic by 21%, which is what this verdict used to do.
    CHI2_MEDIAN = 2.3660 / 3.0

    def update(self, pred: Tensor, gt: Tensor, cov: Tensor) -> None:
        e = pose_error(pred.detach(), gt.detach()).double().unsqueeze(-1)
        chol = torch.linalg.cholesky(cov.detach().double())
        whitened = torch.linalg.solve_triangular(chol, e, upper=False).squeeze(
            -1
        )
        nees = whitened.square().sum(-1)
        self.n += int(nees.shape[0])
        self._nees_sum += float(nees.sum())
        self._covered += int((nees <= self.CHI2_95).sum())
        self._tail += int((nees / 3.0 > self.TAIL).sum())
        # Kept rather than streamed: a median needs the values, and a split is
        # tens of thousands of floats.
        self._nees.extend((nees / 3.0).tolist())

    def as_dict(self) -> dict[str, float]:
        n = max(self.n, 1)
        values = sorted(self._nees)
        median = values[len(values) // 2] if values else 0.0
        kept = [v for v in values if v <= self.TAIL]
        return {
            "anees": self._nees_sum / n / 3.0,
            "anees_median": median,
            "anees_no_tail": sum(kept) / max(len(kept), 1),
            "tail_fraction": self._tail / n,
            "coverage_95": self._covered / n,
        }

    def format(self) -> str:
        d = self.as_dict()
        # The median, not the mean, decides the verdict: the mean is a tail
        # statistic here and says "overconfident everywhere" when the truth is
        # "honest almost everywhere and badly wrong on a few frames".
        #
        # Banded around CHI2_MEDIAN rather than around one, within a factor of
        # 1.25 either way.
        ratio = d["anees_median"] / self.CHI2_MEDIAN
        verdict = (
            "overconfident"
            if ratio > 1.25
            else ("pessimistic" if ratio < 0.8 else "calibrated")
        )
        return (
            f"  NEES median {d['anees_median']:7.3f}   "
            f"({self.CHI2_MEDIAN:.3f} is honest -- {verdict})\n"
            f"  ANEES mean  {d['anees']:7.3f}   "
            f"(1.0 is honest; {d['anees_no_tail']:.2f} excluding the tail)\n"
            f"  tail        {d['tail_fraction']:7.2%}   "
            f"(frames above NEES/dof {self.TAIL:.0f} -- confidently wrong)\n"
            f"  coverage    {d['coverage_95']:7.3f}   "
            "(0.95 expected at the 95% ellipsoid)"
        )


@dataclass
class ErrorSummary:
    """Accumulates errors over a split and reports them.

    Streaming rather than storing every frame, so a long evaluation costs a
    fixed amount of memory.
    """

    n: int = 0
    _sum: Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float64)
    )
    _sumsq: Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float64)
    )
    _maxabs: Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float64)
    )
    _trans_sq: float = 0.0
    _recall: Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float64)
    )

    #: Success thresholds, as ``(translation_m, yaw_deg)``. A frame counts
    # only : if it meets both: a pose that is 10 cm out but 5 degrees off is
    # not a : localization.
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
