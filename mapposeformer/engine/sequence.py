"""Closing the loop: the model's own estimate becomes the next frame's prior.

Open loop asks the model to correct a prior that was *drawn* -- independent
every frame, never wrong in the same direction twice. That is not the question
a vehicle asks. A vehicle carries its last answer forward, so a bias repeats
and compounds while independent noise averages away, and **only a closed loop
tells the two apart**. Closing the loop is worth 2.9x on translation and 4.1x
along track, and it turns the noisiest metric in the project into its most
reproducible one -- 4.1% spread across seeds against open loop's 15.4%.

The loop itself is three steps per frame. Odometry moves the estimate forward,
the map is cropped where that estimate believes it is, and the model's
correction is folded back in. Nothing here re-draws a prior, which is the whole
difference: after the first frame the prior's error is the *filter's* error,
and the model has to live with what it produced a frame ago.

Errors are measured against the truth the scene was generated from, so a
sequence run needs a source that can hand back frames in trajectory order and
accept a supplied prior. ``truth_at``, ``sequences`` and ``frames_of`` are
that contract, and both data sources implement it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from mapposeformer.data.sample import measured_egomotion
from mapposeformer.filter import FilterParams, LocalizationKF
from mapposeformer.metrics import ErrorSummary


@dataclass
class SequenceResult:
    """What a set of sequence runs did, summarised the way the metrics want."""

    errors: ErrorSummary = field(default_factory=ErrorSummary)
    scenes: int = 0
    frames: int = 0
    accepted: int = 0
    refused_trust: int = 0
    refused_mass: int = 0
    diverged: int = 0
    #: Final error per scene, which is what a driver would notice. A run can
    #: look healthy on the mean and still end somewhere else entirely.
    finals: list[float] = field(default_factory=list)
    #: Longest unbroken run of refused frames in any scene. A filter coasting
    #: on odometry for fifty frames is a different failure from one that
    #: refused fifty scattered frames, and the mean cannot tell them apart.
    longest_refusal: int = 0

    def as_dict(self) -> dict[str, float]:
        n = max(self.frames, 1)
        d = {f"seq/{k}": v for k, v in self.errors.as_dict().items()}
        d.update(
            {
                "seq/scenes": float(self.scenes),
                "seq/frames": float(self.frames),
                "seq/accepted": self.accepted / n,
                "seq/refused_trust": self.refused_trust / n,
                "seq/refused_mass": self.refused_mass / n,
                "seq/diverged": self.diverged / max(self.scenes, 1),
                "seq/final_m": (
                    sum(self.finals) / len(self.finals) if self.finals else 0.0
                ),
                "seq/longest_refusal": float(self.longest_refusal),
            }
        )
        return d

    def format(self) -> str:
        d = self.as_dict()
        return (
            f"closed loop over {self.scenes} scenes, {self.frames} frames\n"
            f"  trans {d['seq/rmse_trans_m']:.3f}"
            f"  long {d['seq/rmse_long_m']:.3f}"
            f"  lat {d['seq/rmse_lat_m']:.3f}"
            f"  yaw {d['seq/rmse_yaw_deg']:.3f} deg\n"
            f"  final {d['seq/final_m']:.3f} m"
            f"   diverged {d['seq/diverged']:.1%}\n"
            f"  accepted {d['seq/accepted']:.1%}"
            f"   refused: trust {d['seq/refused_trust']:.1%},"
            f" mass {d['seq/refused_mass']:.1%}\n"
            f"  longest refusal run {self.longest_refusal} frames"
        )


#: A scene whose estimate ends this far from the truth has not drifted, it has
#: lost the road. Counted separately because averaging a diverged scene into a
#: translation RMSE tells you nothing about either.
DIVERGED_M = 5.0


@torch.no_grad()
def run_sequences(
    model: nn.Module,
    dataset,
    device: str,
    p: FilterParams = FilterParams(),
    limit: int = 0,
) -> SequenceResult:
    """Drive the filter over whole scenes, feeding each estimate forward.

    @param dataset Any source implementing ``sequences``, ``frames_of``,
        ``truth_at`` and ``sample_at``.
    @param limit Stop after this many scenes; 0 runs all of them.

    @return Errors measured against the generating truth, plus what the gates
        did.
    """
    model.eval()
    out = SequenceResult()
    gen = torch.Generator().manual_seed(0)
    keys = dataset.sequences()
    if limit:
        keys = keys[:limit]

    for key in keys:
        frames = dataset.frames_of(key)
        if not frames:
            continue
        # The first frame has nothing to inherit from, so it starts where an
        # open-loop frame would: the truth, displaced by a draw from the same
        # prior the training data uses.
        truth = dataset.truth_at(key, frames[0])
        sample = dataset.sample_at(key, frames[0])
        kf = LocalizationKF(sample["prior"].clone(), p)

        run = 0
        for i, frame in enumerate(frames):
            truth = dataset.truth_at(key, frame)
            if i:
                # Odometry, as the vehicle measures it -- drift included,
                # because a noiseless ego would make the loop an oracle.
                #
                # `(prev, truth)`, not `(truth, prev)`: the filter needs the
                # motion it *made*, forward from the last frame. The sample's
                # `hist_rel` wants the opposite order -- where the past is,
                # seen from now -- because it warps old detections into this
                # frame. Same function, opposite question, and taking the
                # wrong one walks the estimate backwards a frame at a time.
                prev = dataset.truth_at(key, frames[i - 1])
                ego = measured_egomotion(prev, truth, dataset.p.sample.ego, gen)
                kf.predict(ego)

            sample = dataset.sample_at(key, frame, prior_pose=kf.pose.float())
            batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
            pred = model(batch)
            # The filter takes what *this frame* measured, not the model's
            # fused covariance: its own state is already the prior the model
            # was given, and adding a matrix that contains a copy of it counts
            # the same information twice. `posterior` is the setting that does.
            wanted = "cov" if p.measurement == "posterior" else "information"
            measurement = pred.get(wanted)
            if measurement is None:
                raise ValueError(
                    f"the closed loop needs {wanted!r}; this model reports none"
                )

            step = kf.update(
                pred["delta"][0].cpu(),
                measurement[0].cpu(),
                trust=1.0,
                mass=float(pred["mass"][0]),
            )
            out.frames += 1
            if step.accepted:
                out.accepted += 1
                run = 0
            else:
                run += 1
                out.longest_refusal = max(out.longest_refusal, run)
                if step.reason == "trust":
                    out.refused_trust += 1
                else:
                    out.refused_mass += 1

            # The error of the *estimate*, not of one correction: this is what
            # the vehicle believes after everything it has seen so far.
            out.errors.update(kf.pose.float().unsqueeze(0), truth.unsqueeze(0))

        final = float((kf.pose.float()[:2] - truth[:2]).norm())
        out.finals.append(final)
        out.scenes += 1
        if final > DIVERGED_M or math.isnan(final):
            out.diverged += 1

    return out
