"""The model must survive the trip to ONNX, checked now rather than at M5.

The deployment milestone is last, and every decision before it assumes the graph
exports. That assumption was wrong once already: ``repeat_interleave`` built a
constant index vector the ONNX exporter has no conversion for, and it would have
surfaced only after everything else was built on top of it.

Skipped when the export stack is absent -- it is not needed to train, and
``scripts/setup.sh --deploy`` installs it.
"""

import importlib.util
import subprocess
import sys
import warnings

import numpy
import pytest
import torch

from mapposeformer.config import Config
from mapposeformer.data import SyntheticDataset
from mapposeformer.model import MapPoseFormer

INPUTS = ("map_", "det_", "hist_")


class _Positional(torch.nn.Module):
    """ONNX takes positional tensors; the model takes a dict."""

    def __init__(self, model: torch.nn.Module, keys: list[str]):
        super().__init__()
        self.model, self.keys = model, keys

    def forward(self, *args):
        out = self.model(dict(zip(self.keys, args, strict=True)))
        return out["delta"], out["cov"], out["trust_logit"], out["mass"]


def _have(module: str) -> bool:
    """Is `module` installed, without importing it.

    ``pytest.importorskip`` would import onnxruntime **into this process**,
    which is the thing the subprocess below exists to avoid.
    """
    return importlib.util.find_spec(module) is not None


def _wrapped():
    # Seeded: the tolerance below is absolute, and unseeded weights change the
    # scale of the outputs from run to run.
    torch.manual_seed(0)
    cfg = Config()
    sample = SyntheticDataset(cfg.data, "test")[11]
    batch = {k: v.unsqueeze(0) for k, v in sample.items()}
    keys = [k for k in batch if k.startswith(INPUTS)]
    model = MapPoseFormer(cfg.model).eval()
    return _Positional(model, keys), tuple(batch[k] for k in keys)


def test_the_graph_is_capturable():
    """``torch.export`` needs no extra dependency, so this always runs."""
    wrapped, args = _wrapped()
    assert torch.export.export(wrapped, args) is not None


def test_onnx_export_matches_eager(tmp_path):
    """And the exported graph must compute the same answer, not merely exist.

    onnxruntime runs in a **subprocess**. Its thread pool outlives the session
    and races Python's finalisation on teardown, aborting the process with
    "recursive_mutex lock failed" roughly one run in ten -- after every test has
    passed, which reads as a broken suite and is not one. Single-threaded
    session options reduce it and do not remove it. A library that aborts at
    exit does not belong in the test process, so it gets its own.
    """
    if not (_have("onnxruntime") and _have("onnxscript")):
        pytest.skip("export stack not installed; scripts/setup.sh --deploy")

    wrapped, args = _wrapped()
    with torch.no_grad():
        reference = wrapped(*args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = torch.onnx.export(wrapped, args, dynamo=True, verbose=False)

    model_path = tmp_path / "model.onnx"
    program.save(str(model_path))
    inputs = tmp_path / "inputs.npz"
    outputs = tmp_path / "outputs.npz"
    numpy.savez(inputs, **{f"a{i}": a.numpy() for i, a in enumerate(args)})

    script = f"""
import numpy, onnxruntime
data = numpy.load({str(inputs)!r})
session = onnxruntime.InferenceSession(
    {str(model_path)!r}, providers=["CPUExecutionProvider"])
feed = {{spec.name: data[f"a{{i}}"]
        for i, spec in enumerate(session.get_inputs())}}
numpy.savez({str(outputs)!r},
            **{{str(i): v for i, v in enumerate(session.run(None, feed))}})
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=300)
    assert outputs.exists(), "the onnxruntime subprocess produced no outputs"

    got = numpy.load(outputs)
    for i, (name, want) in enumerate(
        zip(("delta", "cov", "trust", "mass"), reference, strict=True)
    ):
        # Relative, because `delta` is metres of correction and `mass` is a
        # sum of assignment weights -- three orders of magnitude apart, and one
        # absolute tolerance cannot be right for both. The two graphs differ
        # only in operator order, so anything above 1e-4 relative is a genuine
        # conversion fault rather than fp32 reassociation.
        assert numpy.allclose(
            want.numpy(), got[str(i)], rtol=1e-4, atol=1e-6
        ), name
