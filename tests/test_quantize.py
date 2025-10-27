"""Simulated INT8: the round trip is lossy but bounded, and zero stays zero.

Accuracy only. Whether INT8 is *faster* needs integer kernels and a vendor
runtime, which is M5; keeping the two apart is what lets the mechanics be
checked here, on a laptop, in a second.
"""

import pytest
import torch

from mapposeformer.model.model import MapPoseFormer, ModelParams
from mapposeformer.quantize import (
    FakeQuantLinear,
    QuantParams,
    _linears,
    calibrate,
    fake_quantize,
    quantize,
    weight_bytes,
    weight_scale,
)
from tests.test_model import _batch


def test_zero_survives_the_round_trip():
    """Padding is zero everywhere in this model, and a zero-point offset would
    give every padded token a small non-zero value to attend to."""
    x = torch.tensor([[0.0, 0.5, -0.5, 2.0]])
    q = fake_quantize(x, weight_scale(x, QuantParams()), 8)
    assert float(q[0, 0]) == 0.0


def test_the_error_is_bounded_by_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(16, 32)
    p = QuantParams()
    scale = weight_scale(w, p)
    err = (fake_quantize(w, scale, p.bits) - w).abs()
    assert torch.all(err <= scale / 2 + 1e-6)


def test_per_channel_beats_per_tensor_on_a_lopsided_matrix():
    """One channel a thousand times larger than the rest is what a shared scale
    is worst at, and output channels of a Linear are exactly that uneven.

    Measured on the small rows: averaging over the whole matrix hides the
    damage behind the large row, which quantizes well under either scheme.
    """
    torch.manual_seed(0)
    w = torch.randn(8, 32) * 0.001
    w[0] *= 1000.0
    err = {}
    for per_channel in (True, False):
        p = QuantParams(per_channel=per_channel)
        q = fake_quantize(w, weight_scale(w, p), p.bits)
        err[per_channel] = float((q - w)[1:].abs().mean())
    assert err[True] < err[False] / 100, err


def test_quantizing_the_model_changes_it_and_stays_finite():
    """How far the pose moves is deliberately not asserted.

    On an untrained model the assignment is noise, so the pose is noise whether
    or not it is quantized, and any threshold here would be measuring the seed.
    What INT8 costs in accuracy is a run against a trained checkpoint.
    """
    torch.manual_seed(0)
    model = MapPoseFormer(ModelParams()).eval()
    batch = _batch(2)
    with torch.no_grad():
        before = model(batch)["delta"]
    assert quantize(model) > 0
    with torch.no_grad():
        after = model(batch)["delta"]
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after), "quantization changed nothing"


def test_calibration_fills_the_activation_scales():
    torch.manual_seed(0)
    model = MapPoseFormer(ModelParams()).eval()
    quantize(model)
    layers = [m for m in model.modules() if isinstance(m, FakeQuantLinear)]
    assert not any(m.calibrated for m in layers)
    assert calibrate(model, [_batch(2), _batch(2)], limit=2) == len(layers)
    assert all(m.calibrated and float(m.act_scale) > 0 for m in layers)
    with torch.no_grad():
        assert torch.isfinite(model(_batch(2))["delta"]).all()


def test_it_refuses_to_calibrate_an_unquantized_model():
    with pytest.raises(ValueError, match="quantize"):
        calibrate(MapPoseFormer(ModelParams()), [_batch(2)])


def test_half_the_weights_are_out_of_reach():
    """INT8 shrinks the student by a third, not by three quarters.

    Half its parameters sit inside `nn.MultiheadAttention`, which the wrapper
    has to skip because the parent reads `out_proj.weight` itself. The same
    module blocks structured head pruning, for the same reason -- two of M4's
    three stages stopped by one module choice, which is what
    docs/OPEN_ITEMS.md wants replaced.
    """
    model = MapPoseFormer(ModelParams())
    total = sum(p.numel() for p in model.parameters())
    reachable = sum(c.weight.numel() for _, _, c in _linears(model))
    assert 0.45 < reachable / total < 0.55, reachable / total

    fp32, int8 = weight_bytes(model)
    assert 0.6 < int8 / fp32 < 0.7, (fp32, int8)
