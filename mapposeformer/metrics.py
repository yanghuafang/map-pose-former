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

    **The conventional target is the mean, and that is deliberately not the
    one led with here.** Standard consistency testing -- Bar-Shalom, Li and
    Kirubarajan, and the practice every filtering toolkit inherits from it --
    averages NEES over runs and tests it against the degrees of freedom, so
    the normalised target is 1.0. Both are reported: ``anees_median`` against
    0.789 and ``anees_mean`` against 1.0.

    The median leads because a mean over this error distribution is not a
    calibration. Association failures put a small fraction of frames
    arbitrarily far out, and the mean follows them: the worked example above
    is a model whose mean says "four times too small everywhere" while its
    median frame is honest and 0.4% of frames carry the whole statistic. That
    an average NEES is dominated by its large errors is a known objection to
    the convention, not a novelty of this project.

    Reporting the mean anyway is what keeps the numbers comparable with
    published work -- and the pair is more informative than either alone,
    because the gap between them *is* the tail. A reader who wants the
    conventional statistic should read ``anees_mean`` against 1.0 and take
    the median as the robust companion, not as a replacement.
    """

    n: int = 0
    _nees_sum: float = 0.0
    _covered: int = 0
    _tail: int = 0
    _nees: list[float] = field(default_factory=list)
    #: Per-axis marginal z-scores, `e_i / sqrt(Sigma_ii)`, one list per axis.
    _z: list[list[float]] = field(default_factory=lambda: [[], [], []])
    #: Per-frame `sigma_long/sigma_lat` and `|e_long|/|e_lat|`. Kept per frame
    #: and reduced by median, so neither side is decided by the worst 1%.
    _sigma_ratio: list[float] = field(default_factory=list)
    _abs_ratio: list[float] = field(default_factory=list)
    #: `(reported generalised variance, nees/dof)` per frame, for coverage
    #: conditioned on the model's own confidence.
    _by_var: list[tuple[float, float]] = field(default_factory=list)

    #: NEES per degree of freedom above which a frame is not merely
    #: overconfident but wrong about being confident at all.
    TAIL = 10.0

    #: Chi-square 95th percentile at three degrees of freedom.
    CHI2_95 = 7.815

    #: Chi-square *median* at three degrees of freedom, per dof: what a
    #: perfectly calibrated estimator scores on ``anees_median``. It is 0.789,
    #: not 1.0 -- the chi-square is right-skewed, so its median sits below its
    #: mean. Judging the median against 1.0 calls a textbook-calibrated model
    #: pessimistic by 21%, which is the mistake this constant exists to
    #: prevent.
    CHI2_MEDIAN = 2.3660 / 3.0

    #: Median of ``|N(0, 1)|``. A calibrated axis scores this on
    #: ``median|z|``; above it the reported sigma is too small.
    HALF_NORMAL_MEDIAN = 0.6744897501960817

    #: Bins for variance-conditioned coverage. Five is enough to see a trend
    #: and few enough that each bin still holds a usable count.
    VAR_BINS = 5

    #: The 99.9th percentile of ``|N(0, 1)|``. A calibrated axis scores this on
    #: ``tail_z``; a model whose body is honest can still score 62.1 here on
    #: its longitudinal axis.
    TAIL_Q = 0.999
    TAIL_Z = 3.2905267314919255

    def update(self, pred: Tensor, gt: Tensor, cov: Tensor) -> None:
        # On the CPU, and for a reason that is not about speed. These are 3x3
        # matrices, but `torch.linalg.cholesky` on CUDA opens a cuSOLVER
        # handle, and creating one needs workspace the card does not have when
        # other jobs are training on it -- CUSOLVER_STATUS_INTERNAL_ERROR, in
        # `cusolverDnCreate`, which took down the measurement of an entire
        # experiment twice. Everything here is accumulated as Python floats
        # anyway, so nothing is lost by leaving the device.
        e = pose_error(pred.detach(), gt.detach()).double().unsqueeze(-1).cpu()
        chol = torch.linalg.cholesky(cov.detach().double().cpu())
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

        # Per-axis, whitened by the **marginal** sigma rather than by the
        # Cholesky factor above. That is the whole point: Cholesky whitening
        # mixes the axes, so an error that is too tight along track and too
        # loose across it -- the exact failure this class exists to catch --
        # partly cancels and the per-axis statistic goes quiet. Dividing by
        # `sqrt(Sigma_ii)` keeps each axis answerable on its own.
        e_flat = e.squeeze(-1)
        var = torch.diagonal(cov.detach().double().cpu(), dim1=-2, dim2=-1)
        z = e_flat / var.clamp_min(1e-18).sqrt()
        for i in range(3):
            self._z[i].extend(z[:, i].tolist())
        sd = var.clamp_min(1e-18).sqrt()
        self._sigma_ratio.extend((sd[:, 0] / sd[:, 1]).tolist())
        self._abs_ratio.extend(
            (e_flat[:, 0].abs() / e_flat[:, 1].abs().clamp_min(1e-9)).tolist()
        )
        # Generalised variance, as a scalar stand-in for "how confident did the
        # model say it was on this frame".
        gvar = var.prod(-1).clamp_min(1e-30).log()
        self._by_var.extend(
            zip(gvar.tolist(), (nees / 3.0).tolist(), strict=True)
        )

    @staticmethod
    def _ks_uniform(u: list[float]) -> float:
        """@return Kolmogorov-Smirnov statistic of ``u`` against ``U(0, 1)``.

        **The statistic, never a p-value.** These frames come from a handful of
        scenes and are strongly correlated within each, so the independence a
        p-value assumes is absent -- at the 8 880 frames of the full split, KS
        rejects on about 2% miscalibration, which would make every model this
        project will ever train "significantly miscalibrated". The effect size
        is the useful part, so only the effect size is reported.
        """
        if not u:
            return 0.0
        x = sorted(u)
        m = len(x)
        d = 0.0
        for i, v in enumerate(x):
            d = max(d, (i + 1) / m - v, v - i / m)
        return d

    def _axis_pit(self, axis: int) -> tuple[float, float]:
        """@return ``(ks, tightness)`` for one axis.

        The folded probability integral transform: for a calibrated axis,
        ``z`` is standard normal, so ``|z|`` has CDF ``2*Phi(|z|) - 1`` and that
        transform is uniform on ``(0, 1)``. Folding discards the sign on
        purpose -- the question here is whether the reported sigma has the right
        *size*, and a bias is already reported by :class:`ErrorSummary`.

        ``tightness`` is ``median|z|`` over the calibrated 0.674, so above one
        means the errors are larger than the covariance claims -- the reported
        sigma is too **tight**. It carries the sign that the KS statistic
        throws away, and it is the sign that makes an inversion visible: an
        inverted ellipse reads too tight on one axis and too loose on another.
        """
        z = self._z[axis]
        if not z:
            return 0.0, 0.0
        folded = [math.erf(abs(v) / math.sqrt(2.0)) for v in z]
        med = sorted(abs(v) for v in z)[len(z) // 2]
        return self._ks_uniform(folded), med / self.HALF_NORMAL_MEDIAN

    def _tail_z(self, axis: int) -> float:
        """@return ``|z|`` at the 99.9th percentile, over the calibrated 3.29.

        **This is the statistic the median-based ones cannot reach.** PIT-KS and
        ``tightness`` both describe the body of the distribution, and a model
        can be flawless there while being catastrophically wrong on a thin
        tail. That model is not hypothetical: its along-track error is 0.211 m
        at the 99th percentile and 4.049 m at the 99.9th, and the sigma it
        reports tracks the first and not the second. On the worst 1% of frames
        its error averaged 2.249 m against a claimed 0.308 m -- 20x
        overconfident -- while the other 99% sat at 0.80x, slightly
        conservative. Every median statistic in this class calls that model
        honest, and it is honest, on 99% of frames.

        A filter is not harmed by the 99%. It is harmed by the 1%, because a
        frame it is told is certain and is not is one it cannot recover from.
        """
        z = self._z[axis]
        if not z:
            return 0.0
        ordered = sorted(abs(v) for v in z)
        idx = min(int(self.TAIL_Q * len(ordered)), len(ordered) - 1)
        return ordered[idx] / self.TAIL_Z

    def _conditioned_coverage(self) -> list[float]:
        """Coverage within each fifth of the split, ordered by how confident
        the model said it was.

        A single coverage number is an average, and an average hides the
        failure that matters: a model can cover 95% overall while covering 70%
        of the frames it called certain and 99% of the ones it called
        uncertain. A filter is hurt precisely by the first group, because those
        are the frames it weights most.
        """
        if not self._by_var:
            return []
        ordered = sorted(self._by_var)
        out = []
        step = max(len(ordered) // self.VAR_BINS, 1)
        for b in range(self.VAR_BINS):
            lo = b * step
            hi = len(ordered) if b == self.VAR_BINS - 1 else (b + 1) * step
            chunk = ordered[lo:hi]
            if not chunk:
                continue
            hit = sum(1 for _, nees in chunk if nees * 3.0 <= self.CHI2_95)
            out.append(hit / len(chunk))
        return out

    def anisotropy(self) -> tuple[float, float]:
        """@return ``(reported, measured)`` ratio of longitudinal to lateral
            sigma over the split.

        This is the statistic that catches an inverted covariance: a model can
        be too tight along track and too loose across it while its NEES median
        -- 0.786, against the 0.789 an honest model scores -- calls the whole
        thing honest. No scalar built on ``r^T Sigma^-1 r`` can see that,
        because the two errors cancel inside the quadratic form.

        **Medians on both sides, and the "both" is the whole point.** A
        headline of 0.809 reported against 3.030 measured takes a *median of
        per-frame ratios* on the reported side and a *ratio of RMS* on the
        measured side. Those are two different populations when the errors are
        heavy-tailed, and these errors are violently heavy-tailed -- along-track
        error runs 0.211 m at the 99th percentile and 4.049 m at the 99.9th.
        The median describes the body and the RMS is almost entirely the worst
        1%, so putting one against the other manufactures an inversion that is
        not there. Measured like for like on the same checkpoint, both sides
        sit **below** 1.0 (0.793 reported against 0.711 measured); on RMS both
        sides sit above it (1.054 against 3.385). The sign flips with the
        aggregation, which is the signature of comparing incomparable things.

        **Both sides are also marginal**, which remains a real limitation: the
        spread over a split includes variation between scenes, while the
        reported covariance is conditional on the frame. ``tools/refcov.py``
        redraws the prior per frame to get the conditional reference properly.
        """
        ordered_rep = sorted(self._sigma_ratio)
        ordered_mea = sorted(self._abs_ratio)
        if not ordered_rep:
            return 0.0, 0.0
        return (
            ordered_rep[len(ordered_rep) // 2],
            ordered_mea[len(ordered_mea) // 2],
        )

    def as_dict(self) -> dict[str, float]:
        n = max(self.n, 1)
        values = sorted(self._nees)
        median = values[len(values) // 2] if values else 0.0
        kept = [v for v in values if v <= self.TAIL]
        rep_aniso, mea_aniso = self.anisotropy()
        out = {
            "anees": self._nees_sum / n / 3.0,
            "anees_median": median,
            "anees_no_tail": sum(kept) / max(len(kept), 1),
            "tail_fraction": self._tail / n,
            "coverage_95": self._covered / n,
            "aniso_reported": rep_aniso,
            "aniso_measured": mea_aniso,
        }
        for axis, name in enumerate(("long", "lat", "yaw")):
            ks, tight = self._axis_pit(axis)
            out[f"pit_ks_{name}"] = ks
            out[f"tightness_{name}"] = tight
            out[f"tail_z_{name}"] = self._tail_z(axis)
        for b, cov in enumerate(self._conditioned_coverage()):
            out[f"coverage_q{b + 1}"] = cov
        return out

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
            "(0.95 expected at the 95% ellipsoid)\n" + self._shape_report(d)
        )

    def _shape_report(self, d: dict[str, float]) -> str:
        """The half of calibration that ``r^T Sigma^-1 r`` cannot see."""
        lines = []
        for name in ("long", "lat", "yaw"):
            ks, tight = d[f"pit_ks_{name}"], d[f"tightness_{name}"]
            # Banded like the NEES verdict above, and for the same reason:
            # the useful question is not "is it exactly right" but "which way
            # is it wrong, and by enough to act on".
            how = (
                "too tight"
                if tight > 1.25
                else ("too loose" if tight < 0.8 else "honest")
            )
            tail = d[f"tail_z_{name}"]
            # The body and the tail get separate verdicts because they fail
            # separately: a model can be honest in the body and catastrophic
            # in the tail.
            lines.append(
                f"  PIT-KS {name:<5}{ks:7.3f}   "
                f"(body: sigma {how}, median|z| {tight:.2f}x;"
                f" tail: |z| at 99.9th is {tail:.1f}x calibrated"
                f"{' -- OVERCONFIDENT' if tail > 2.0 else ''})"
            )
        rep, mea = d["aniso_reported"], d["aniso_measured"]
        # Inversion is the failure a scalar cannot reach: it needs the two
        # ratios to sit on opposite sides of 1.0, not merely to differ.
        inverted = (rep - 1.0) * (mea - 1.0) < 0
        lines.append(
            f"  aniso       {rep:7.3f}   "
            f"(long/lat reported against {mea:.3f} measured"
            f"{' -- INVERTED' if inverted else ''})"
        )
        qs = [f"{d[k]:.2f}" for k in sorted(d) if k.startswith("coverage_q")]
        if qs:
            lines.append(
                "  coverage by reported variance, confident first: "
                + " ".join(qs)
            )
        return "\n".join(lines)


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
