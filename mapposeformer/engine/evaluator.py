"""Evaluation on a split: what the model scores, and what doing nothing scores.

The second number is the one that gives the first any meaning. A localizer
reporting 0.3 m is doing well or badly depending entirely on how wrong the
prior was, and the prior here is drawn, so the baseline is knowable exactly:
predict a zero correction and the error *is* the prior's own.

The calibration block arrives with the covariance. There is still no trust
gate: refusing a frame needs a threshold with something behind it, and this
design widens the covariance on an ambiguous frame rather than dropping it.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from mapposeformer.metrics import Calibration, ErrorSummary


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: str
) -> dict[str, object]:
    """@return Summaries keyed ``nothing`` and ``all``, plus ``calibration``.

    ``calibration`` is present only when the model reports a covariance; a
    model that does not is still evaluable on accuracy alone.
    """
    model.eval()
    nothing, everything = ErrorSummary(), ErrorSummary()
    calibration = Calibration()
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch)
        everything.update(out["delta"], batch["delta"])
        nothing.update(torch.zeros_like(batch["delta"]), batch["delta"])
        if "cov" in out:
            calibration.update(out["delta"], batch["delta"], out["cov"])
    result: dict[str, object] = {"nothing": nothing, "all": everything}
    if calibration.n:
        result["calibration"] = calibration
    return result


def format_report(result: dict[str, object]) -> str:
    """A human-readable block, shaped like the classical repo's eval output."""
    titles = {
        "nothing": "do nothing (the prior's own error -- the number to beat)",
        "all": "all frames",
    }
    blocks = [f"{titles[k]}\n{result[k].format()}" for k in ("nothing", "all")]
    if "calibration" in result:
        blocks.append(
            f"is the covariance honest?\n{result['calibration'].format()}"
        )
    return "\n\n".join(blocks)
