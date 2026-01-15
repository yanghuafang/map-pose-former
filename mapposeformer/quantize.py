"""Post-training quantization, simulated, to price INT8 in accuracy.

Quantize-dequantize rather than integer kernels: this answers "what does INT8
cost?" and not "how fast is INT8?". The second question needs TensorRT and
belongs to M7, and separating them means the accuracy curve can be measured on
a laptop, without a GPU or a vendor runtime.

**Most of this model is not quantizable, and that is the interesting part.**
The pose head is weighted Procrustes -- arithmetic over coordinates with no
parameters -- and the cost surface is a closed form over the same statistics.
Neither has weights to quantize, and ``geometry.exact_arithmetic`` already
holds both in fp32 because a 0.25 m lattice under a 40 m map point is a
measurable error. So quantization reaches the tokenizer, the attention
projections and the feed-forwards, and stops exactly where the geometry starts.
The part of this model most worth trusting is the part INT8 cannot touch.

That is also why the milestone measures the teacher. The student's weights are
3.5 MiB against ~7 GiB of activations, so shrinking weights is not what makes
this model cheaper, and a compression chapter that measured only the student
would report "nothing moved" and teach the technique badly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class QuantParams:
    """How coarsely to simulate, and how much of the tail to clip."""

    bits: int = 8
    per_channel: bool = True
    """Weights get one scale per output channel, activations one per tensor.
    A channel whose weights are ten times smaller than its neighbour's loses a
    quarter of its range to that neighbour under a shared scale, and output
    channels of a Linear are exactly that heterogeneous."""
    percentile: float = 99.99
    """Activation range, as a percentile of absolute value rather than the max.
    One outlier sets the scale for every other number in the tensor, and this
    model has outliers by construction: a map point 50 m away is 500 times a
    lane width."""


def _qmax(bits: int) -> float:
    """@brief Largest magnitude a symmetric signed code can represent.
    @param bits Bit width. @return ``2**(bits-1) - 1`` as a float."""
    return float(2 ** (bits - 1) - 1)


def fake_quantize(x: Tensor, scale: Tensor, bits: int) -> Tensor:
    """@brief Round to the grid ``scale`` defines, then come straight back.

    Symmetric and zero-preserving: zero maps to code zero exactly, which
    matters because padding is zero everywhere in this model and a zero-point
    offset would give every padded token a small non-zero value.

    **Torch's own operator rather than the arithmetic it stands for.** Written
    out as ``(x / scale).round().clamp(...) * scale`` this computes the same
    numbers to the last bit, and exports to ONNX as Div, Round, Clip and Mul --
    four float operations that no runtime can recognise as a quantization. The
    ATen op lowers to a QuantizeLinear/DequantizeLinear pair instead, which is
    the form TensorRT and the mobile runtimes fold into integer kernels. The
    simulation is unchanged; what changes is whether the exported graph can be
    deployed as INT8 at all.

    @param x The tensor to quantize.
    @param scale Step size: a scalar, or one per output channel as ``(out, 1)``.
    @param bits Bit width.
    @return ``x`` after a round trip through the grid, same dtype and shape.
    """
    q = int(_qmax(bits))
    # Detached, because the step size is a constant of the quantization rather
    # than something being fitted: this is post-training, so nothing here is
    # learning a scale. The forward values are identical either way, and the
    # per-channel operator refuses a scale that carries a gradient -- weight
    # scales are derived from the weights, so without this an export dies on a
    # graph nobody intended to build.
    scale = scale.detach().clamp_min(torch.finfo(x.dtype).tiny).float()
    if scale.ndim == 0:
        zero = torch.zeros((), dtype=torch.int32, device=x.device)
        return torch.fake_quantize_per_tensor_affine(x, scale, zero, -q - 1, q)
    # Axis 0, because weight_scale reduces a `(out, in)` weight over its input
    # dimension and keeps one step size per output channel.
    zero = torch.zeros(scale.numel(), dtype=torch.int32, device=x.device)
    return torch.fake_quantize_per_channel_affine(
        x, scale.reshape(-1), zero, 0, -q - 1, q
    )


def weight_scale(w: Tensor, p: QuantParams) -> Tensor:
    """@brief The step size for a weight matrix.

    @param w ``(out, in)`` weight.
    @param p Bit width and whether to go per channel.
    @return A scalar, or ``(out, 1)`` when per-channel.
    """
    if p.per_channel:
        amax = w.abs().amax(dim=1, keepdim=True)
    else:
        amax = w.abs().amax()
    return amax / _qmax(p.bits)


class FakeQuantLinear(nn.Module):
    """A ``Linear`` that rounds its input and its weights before multiplying.

    Wraps rather than replaces, so the original module keeps its parameters and
    a run can be un-quantized by unwrapping. The activation scale is filled by
    :func:`calibrate` and is ``None`` until then, which means weight-only
    quantization -- a useful row in its own right, since it is what shrinks a
    checkpoint without touching the arithmetic.
    """

    def __init__(self, inner: nn.Linear, p: QuantParams):
        super().__init__()
        self.inner = inner
        self.p = p
        self.register_buffer("act_scale", torch.zeros(()))
        self.calibrated = False

    def forward(self, x: Tensor) -> Tensor:
        """@brief Quantized-simulated linear. @param x Input. @return Output."""
        if self.calibrated:
            x = fake_quantize(x, self.act_scale.to(x.dtype), self.p.bits)
        w = fake_quantize(
            self.inner.weight,
            weight_scale(self.inner.weight, self.p),
            self.p.bits,
        )
        return nn.functional.linear(x, w, self.inner.bias)


def _linears(model: nn.Module) -> list[tuple[nn.Module, str, nn.Linear]]:
    """@brief Every ``Linear`` that quantization should touch, with its parent.

    ``GeometricAttention`` holds its four projections as ordinary ``Linear``
    children, so every quantizable weight in this model is reachable this way
    and nothing needs a special case.

    @param model Any model.
    @return Triples of ``(parent, attribute name, module)``.
    """
    found = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                found.append((parent, name, child))
    return found


def quantize(model: nn.Module, p: QuantParams | None = None) -> int:
    """@brief Wrap every reachable ``Linear`` in simulated quantization.

    In place. Weight-only until :func:`calibrate` runs.

    @param model The model to quantize.
    @param p Settings; the defaults are INT8, per channel.
    @return How many modules were wrapped.
    """
    p = p or QuantParams()
    wrapped = _linears(model)
    for parent, name, child in wrapped:
        setattr(parent, name, FakeQuantLinear(child, p))
    return len(wrapped)


@torch.no_grad()
def calibrate(model: nn.Module, batches, limit: int = 16) -> int:
    """@brief Learn each wrapped layer's activation range from real data.

    A percentile of absolute value, not the maximum: one outlier would set the
    scale for the whole tensor, and this model produces outliers by
    construction -- a map point at 50 m is hundreds of times a lane width.

    @param model A model already through :func:`quantize`.
    @param batches An iterable of batches the model can be called on.
    @param limit How many batches to observe.
    @return The number of layers calibrated.
    @throws ValueError If the model has no quantized layers.
    """
    layers = [m for m in model.modules() if isinstance(m, FakeQuantLinear)]
    if not layers:
        raise ValueError("nothing to calibrate: call quantize() first")

    seen: dict[FakeQuantLinear, list[Tensor]] = {m: [] for m in layers}
    handles = []
    for m in layers:

        def hook(mod, args, _seen=seen):
            x = args[0].detach().float().abs().flatten()
            # A sample, not the tensor: an exact percentile over every
            # activation of every batch would hold the whole calibration set
            # in memory for no extra precision.
            if x.numel() > 8192:
                idx = torch.randint(0, x.numel(), (8192,), device=x.device)
                x = x[idx]
            _seen[mod].append(x.cpu())

        handles.append(m.register_forward_pre_hook(hook))

    for i, batch in enumerate(batches):
        if i >= limit:
            break
        model(batch)
    for h in handles:
        h.remove()

    q = _qmax(layers[0].p.bits)
    for m in layers:
        pooled = torch.cat(seen[m]) if seen[m] else torch.zeros(1)
        amax = torch.quantile(pooled, m.p.percentile / 100.0)
        m.act_scale.copy_(amax / q)
        m.calibrated = True
    return len(layers)


def weight_bytes(
    model: nn.Module, p: QuantParams | None = None
) -> tuple[int, int]:
    """@brief What the weights cost in fp32, and what they would cost quantized.

    Reported because it is the number INT8 is usually sold on, and the number
    that matters least here: the student's weights are 3.5 MiB against roughly
    7 GiB of activations at batch 64.

    @param model Any model, quantized or not.
    @param p Settings, for the bit width.
    @return ``(fp32_bytes, quantized_bytes)``.
    """
    p = p or QuantParams()
    total = sum(x.numel() for x in model.parameters())
    quantizable = sum(
        m.inner.weight.numel()
        for m in model.modules()
        if isinstance(m, FakeQuantLinear)
    ) or sum(m.weight.numel() for _, _, m in _linears(model))
    rest = total - quantizable
    return total * 4, rest * 4 + quantizable * p.bits // 8
