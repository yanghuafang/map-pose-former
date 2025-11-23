"""One scene at a time, with the model's answer fed back as the next prior.

Open loop measures the model. It cannot measure the system, because what
breaks a deployed localizer is not a bad frame but a bad frame *believed*: the
estimate moves, the next map crop is taken from the wrong place, and the
following frame is asked a harder question than it would otherwise have faced.

``LocalizationKF`` holds the estimate, the model is handed it as the prior, and
whatever comes back is gated and folded in. What comes out is a trajectory
rather than another RMSE, so the questions are different -- does the error stay
bounded, how often is a frame refused, and is the filter honest about how well
it is doing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from mapposeformer import geometry as G
from mapposeformer.data.sample import measured_egomotion
from mapposeformer.filter import FilterParams, LocalizationKF

#: Final translation error above which a sequence is called diverged. The
#: prior's own error is 1.6 m and the open-loop model reaches 0.34 m, so a
#: sequence ending five metres out did not drift -- it left.
DIVERGED_M = 5.0


@dataclass
class SequenceResult:
    """One scene's trajectory, and what the filter did along it."""

    error: Tensor
    """``(T, 3)`` posterior error per frame, as ``(long, lat, yaw)`` in the
    true body frame."""
    nees: Tensor
    """``(T,)`` the filter's own NEES per degree of freedom -- whether its
    covariance describes the error it actually has."""
    accepted: list[bool] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    nis: list[float] = field(default_factory=list)

    @property
    def diverged(self) -> bool:
        return float(self.error[-1, :2].norm()) > DIVERGED_M

    @property
    def longest_refusal(self) -> int:
        """The longest run of consecutive refusals.

        This is the number that separates a gate doing its job from a gate
        locked out. Isolated refusals are a filter declining a bad frame. A
        long run is the loop feeding on itself: the estimate has drifted far
        enough that the map crop is taken from the wrong place, so the model
        reports low trust, so the correction that would recover the estimate is
        the one refused.
        """
        best = run = 0
        for ok in self.accepted:
            run = 0 if ok else run + 1
            best = max(best, run)
        return best


def run_sequence(
    model,
    source,
    key,
    fp: FilterParams = FilterParams(),
    device: str = "cpu",
    seed: int = 0,
) -> SequenceResult:
    """@brief Drive one scene through the filter, front to back.

    @param model An evaluated :class:`~mapposeformer.model.MapPoseFormer`.
    @param source A dataset answering ``frames_of``, ``sample_at`` and
        ``truth_at``. Both sources do, which is why this cannot tell a
        generated road from a surveyed one.
    @param key The scene, as ``source.sequences()`` names it.
    @param fp Gates and process noise.
    @param seed Draws the odometry drift. The scene's own noise is seeded by
        the source, so this only controls what integrating it costs.

    @return The :class:`SequenceResult`.
    """
    frames = source.frames_of(key)
    gen = torch.Generator().manual_seed(seed)
    ego_params = source.p.sample.ego

    # The first frame has no predecessor, so its prior is drawn the way an
    # open-loop frame's is -- the sequence starts where the old measurement
    # started, and everything after it is the loop's doing.
    first = source.sample_at(key, frames[0])
    kf = LocalizationKF(first["prior"], fp)

    err, nees, res = [], [], SequenceResult(torch.empty(0), torch.empty(0))
    for i, frame in enumerate(frames):
        truth = source.truth_at(key, frame)
        if i:
            # Drift is composed in the arriving frame, which is where an
            # integrated error is realised.
            prev = source.truth_at(key, frames[i - 1])
            kf.predict(measured_egomotion(prev, truth, ego_params, gen))
            sample = source.sample_at(key, frame, kf.pose.float())
        else:
            sample = first

        batch = {
            k: v.unsqueeze(0).to(device)
            for k, v in sample.items()
            if torch.is_tensor(v)
        }
        with torch.no_grad():
            out = model(batch)

        step = kf.update(
            out["delta"][0].cpu(),
            out["cov"][0].cpu(),
            float(torch.sigmoid(out["trust_logit"][0])),
            float(out["mass"][0]),
        )
        res.accepted.append(step.accepted)
        res.reasons.append(step.reason)
        res.nis.append(step.nis)

        # Error in the *true* body frame, so long and lat mean along and
        # across the road rather than along and across the estimate.
        e = G.relative(truth.double(), kf.pose)
        err.append(e)
        whitened = torch.linalg.solve_triangular(
            torch.linalg.cholesky(kf.cov), e.unsqueeze(-1), upper=False
        ).squeeze(-1)
        nees.append(float(whitened.square().sum()) / 3.0)

    res.error = torch.stack(err)
    res.nees = torch.tensor(nees)
    return res


def eval_sequence(
    model,
    source,
    fp: FilterParams = FilterParams(),
    device: str = "cpu",
    limit: int = 0,
) -> dict[str, float]:
    """@brief Every scene in the source, summarised.

    @param limit Stop after this many scenes; 0 means all of them.
    @return A flat dict, shaped like the open-loop evaluator's so the two can
        sit in one table.
    """
    keys = source.sequences()
    if limit:
        keys = keys[:limit]

    err, nees, acc, reasons, diverged = [], [], [], [], 0
    finals, streaks = [], []
    for i, key in enumerate(keys):
        r = run_sequence(model, source, key, fp, device, seed=i)
        err.append(r.error)
        nees.append(r.nees)
        acc += r.accepted
        reasons += [x for x in r.reasons if x]
        finals.append(float(r.error[-1, :2].norm()))
        streaks.append(r.longest_refusal)
        diverged += int(r.diverged)

    e = torch.cat(err)
    n = torch.cat(nees)
    total = max(len(acc), 1)
    # A median and a 95th beside the RMSE, for the reason ``metrics.py`` gives
    # for reporting them beside ANEES: a few diverged scenes own a mean, and a
    # summary that only reports the mean cannot say whether the typical frame
    # is fine or nothing is.
    dist = e[:, :2].norm(dim=-1)
    out = {
        "seq/rmse_trans_m": float(e[:, :2].square().sum(-1).mean().sqrt()),
        "seq/median_trans_m": float(dist.median()),
        "seq/p95_trans_m": float(dist.quantile(0.95)),
        "seq/rmse_long_m": float(e[:, 0].square().mean().sqrt()),
        "seq/rmse_lat_m": float(e[:, 1].square().mean().sqrt()),
        "seq/rmse_yaw_deg": float(e[:, 2].square().mean().sqrt())
        * 180
        / math.pi,
        "seq/final_trans_m": float(torch.tensor(finals).mean()),
        "seq/diverged": diverged / max(len(keys), 1),
        "seq/accepted": sum(acc) / total,
        "seq/longest_refusal": float(max(streaks)) if streaks else 0.0,
        "seq/mean_refusal_run": float(
            torch.tensor(streaks, dtype=torch.float).mean()
        )
        if streaks
        else 0.0,
        "seq/nees_median": float(n.median()),
        "seq/nees_tail": float((n > 10.0).float().mean()),
        "seq/scenes": float(len(keys)),
        "seq/frames": float(total),
    }
    for why in ("trust", "mass"):
        out[f"seq/rejected_{why}"] = reasons.count(why) / total
    return out


def format_sequence(m: dict[str, float]) -> str:
    """A block shaped like ``engine/evaluator.py``'s, for the same reason."""
    lines = [
        f"closed loop over {int(m['seq/scenes'])} scenes, "
        f"{int(m['seq/frames'])} frames",
        f"  trans {m['seq/rmse_trans_m']:.3f}  long {m['seq/rmse_long_m']:.3f}"
        f"  lat {m['seq/rmse_lat_m']:.3f}  yaw {m['seq/rmse_yaw_deg']:.3f} deg",
        f"  median {m['seq/median_trans_m']:.3f}  "
        f"p95 {m['seq/p95_trans_m']:.3f}",
        f"  final {m['seq/final_trans_m']:.3f} m   "
        f"diverged {m['seq/diverged']:.1%}",
        f"  longest refusal run {int(m['seq/longest_refusal'])} frames, "
        f"mean per scene {m['seq/mean_refusal_run']:.1f}",
        f"  accepted {m['seq/accepted']:.1%}   refused: "
        f"trust {m['seq/rejected_trust']:.1%}, "
        f"mass {m['seq/rejected_mass']:.1%}",
        f"  filter NEES/dof median {m['seq/nees_median']:.3f}   "
        f"tail {m['seq/nees_tail']:.4f}",
    ]
    return "\n".join(lines)
