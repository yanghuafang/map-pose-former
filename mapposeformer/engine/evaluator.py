"""Evaluation: run a split, summarise the error, report what was refused.

Three summaries come back, and the first one is the one usually missing from a
paper. **do-nothing** is the error of predicting no correction at all -- that is,
the prior's own error, and the number the model has to beat before any of its
other numbers mean anything. A translation RMSE of 1.2 m sounds like
localization until you notice the prior was 1.6 m out to begin with.

**all** is every frame. **trusted** is the frames the model says it succeeded on,
which is the error a downstream filter would actually see, and is the only
accuracy number a deployment cares about -- a confidently wrong frame costs far
more than an honestly rejected one.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from mapposeformer.metrics import ErrorSummary


def _to(batch: dict[str, Tensor], device: str) -> dict[str, Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    trust_threshold: float = 0.5,
) -> dict[str, object]:
    """Run one pass and return metrics plus the two summaries.

    Returns:
        ``{"nothing": ..., "all": ..., "trusted": ..., "metrics": dict}`` where
        the first three are :class:`ErrorSummary`. The metrics dict flattens all
        three under a prefix, ready for TensorBoard.
    """
    was_training = model.training
    model.eval()
    nothing, every, trusted = ErrorSummary(), ErrorSummary(), ErrorSummary()
    n_trusted = n_total = 0

    for batch in loader:
        batch = _to(batch, device)
        out = model(batch)
        every.update(out["delta"], batch["delta"])
        nothing.update(torch.zeros_like(batch["delta"]), batch["delta"])
        keep = torch.sigmoid(out["trust_logit"]) >= trust_threshold
        n_total += keep.numel()
        n_trusted += int(keep.sum())
        if bool(keep.any()):
            trusted.update(out["delta"][keep], batch["delta"][keep])

    metrics = {f"nothing/{k}": v for k, v in nothing.as_dict().items()}
    metrics.update({f"all/{k}": v for k, v in every.as_dict().items()})
    metrics.update({f"trusted/{k}": v for k, v in trusted.as_dict().items()})
    metrics["trusted/fraction"] = n_trusted / max(n_total, 1)
    model.train(was_training)
    return {"nothing": nothing, "all": every, "trusted": trusted, "metrics": metrics}


def format_report(result: dict[str, object]) -> str:
    """A human-readable block, in the shape of the classical repo's eval output."""
    frac = result["metrics"]["trusted/fraction"]  # type: ignore[index]
    blocks = [
        ("do nothing (the prior's own error -- the number to beat)", "nothing"),
        ("all frames", "all"),
        (f"trusted frames ({frac:.1%} of the split)", "trusted"),
    ]
    summaries: list[str] = []
    for title, key in blocks:
        summary: ErrorSummary = result[key]  # type: ignore[assignment]
        summaries.append(f"{title}\n{summary.format()}")
    return "\n\n".join(summaries)
