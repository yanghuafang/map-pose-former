"""Evaluation: run a split, summarise the error, report what was refused.

Four blocks come back, and the first is the one usually missing from a paper.
**do-nothing** is the prior's own error -- the number the model must beat before
any other number means anything. **all** is every frame. **trusted** is what a
downstream filter would actually see. **calibration** asks whether the reported
covariance is honest, which no RMSE can say.

**Two gates, not one.** The learned trust score answers "does this look like a
frame I get right?". It is structurally incapable of answering "did I have any
evidence at all?", because a pose from three confident wrong correspondences
looks exactly like one from three right ones to the features that produced it.
So ``mass`` is gated separately and arithmetically: the ablation that removed
lane geometry produced 14 m of error with the trust head keeping 9.97 m of it.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from mapposeformer.metrics import Calibration, ErrorSummary


def _to(batch: dict[str, Tensor], device: str) -> dict[str, Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    trust_threshold: float = 0.5,
    min_mass: float = 4.0,
) -> dict[str, object]:
    """Run one pass and return metrics plus the three summaries.

    @param trust_threshold Minimum ``sigmoid(trust_logit)`` to accept a frame.
    @param min_mass Minimum total assignment mass. Roughly "effective number of
        correspondences the answer rests on", so the default asks for four.
        Below that a rigid solve is not over-determined enough for its residual
        to mean anything, and the pose it returns is closer to a default than
        to an estimate.

    @return ``{"nothing", "all", "trusted", "calibration", "metrics"}`` --
        three :class:`ErrorSummary`, one :class:`Calibration`, and a flat
        metrics dict with everything under a prefix, ready for TensorBoard.
    """
    was_training = model.training
    model.eval()
    nothing, every, trusted = ErrorSummary(), ErrorSummary(), ErrorSummary()
    calib = Calibration()
    n_trusted = n_total = n_low_mass = 0

    for batch in loader:
        batch = _to(batch, device)
        out = model(batch)
        every.update(out["delta"], batch["delta"])
        nothing.update(torch.zeros_like(batch["delta"]), batch["delta"])
        calib.update(out["delta"], batch["delta"], out["cov"])
        enough = out["mass"] >= min_mass
        keep = (torch.sigmoid(out["trust_logit"]) >= trust_threshold) & enough
        n_total += keep.numel()
        n_trusted += int(keep.sum())
        n_low_mass += int((~enough).sum())
        if bool(keep.any()):
            trusted.update(out["delta"][keep], batch["delta"][keep])

    metrics = {f"nothing/{k}": v for k, v in nothing.as_dict().items()}
    metrics.update({f"all/{k}": v for k, v in every.as_dict().items()})
    metrics.update({f"trusted/{k}": v for k, v in trusted.as_dict().items()})
    metrics.update({f"calib/{k}": v for k, v in calib.as_dict().items()})
    metrics["trusted/fraction"] = n_trusted / max(n_total, 1)
    metrics["trusted/rejected_low_mass"] = n_low_mass / max(n_total, 1)
    model.train(was_training)
    return {
        "nothing": nothing,
        "all": every,
        "trusted": trusted,
        "calibration": calib,
        "metrics": metrics,
    }


def format_report(result: dict[str, object]) -> str:
    """A human-readable block, shaped like the classical repo's eval output."""
    m = result["metrics"]  # type: ignore[index]
    frac, low = m["trusted/fraction"], m["trusted/rejected_low_mass"]
    blocks = [
        ("do nothing (the prior's own error -- the number to beat)", "nothing"),
        ("all frames", "all"),
        (
            f"trusted frames ({frac:.1%} of the split; "
            f"{low:.1%} refused for want of evidence)",
            "trusted",
        ),
    ]
    summaries: list[str] = []
    for title, key in blocks:
        summary: ErrorSummary = result[key]  # type: ignore[assignment]
        summaries.append(f"{title}\n{summary.format()}")
    calib: Calibration = result["calibration"]  # type: ignore[assignment]
    summaries.append(f"is the covariance honest?\n{calib.format()}")
    return "\n\n".join(summaries)
