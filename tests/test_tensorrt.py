"""The TensorRT path, as far as a machine without TensorRT can check it.

Skipped without the runtime, which is a Linux/CUDA wheel: the point of keeping
this module import-free is that training and evaluation still run on a laptop.
What is checked here is the part that has already been wrong once -- the API
this project assumed and the one TensorRT 11 actually has.
"""

import importlib.util

import pytest

HAVE_TRT = importlib.util.find_spec("tensorrt") is not None


@pytest.mark.skipif(not HAVE_TRT, reason="tensorrt not installed")
def test_the_build_uses_an_api_that_exists():
    """TensorRT 11 removed the workflow every tutorial still describes.

    `BuilderFlag.FP16`, `BuilderFlag.INT8` and `IInt8EntropyCalibrator2` are
    gone; precision is a property of the graph and a strongly typed network
    takes each layer's type from the ONNX it was handed. This project wrote the
    calibrator version first and had to find out. If a future version restores
    the flags, this test says so rather than leaving the module's explanation
    quietly wrong.
    """
    import tensorrt as trt

    assert hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")
    assert not hasattr(trt, "IInt8EntropyCalibrator2")
    assert not hasattr(trt.BuilderFlag, "FP16")
    assert not hasattr(trt.BuilderFlag, "INT8")


@pytest.mark.skipif(not HAVE_TRT, reason="tensorrt not installed")
def test_a_graph_that_does_not_parse_is_refused():
    from mapposeformer.tensorrt import build_engine

    with pytest.raises(RuntimeError, match="ONNX did not parse"):
        build_engine(__file__, "/tmp/never-written.engine")


def test_the_export_wrapper_returns_what_a_filter_consumes():
    """Four tensors cross the boundary, not the model's whole output dict.

    The assignment is 768 x 576 and diagnostic; putting it through an engine
    would cost more than everything the filter actually reads.
    """
    import torch

    from tools.export import Positional

    class _Stub(torch.nn.Module):
        def forward(self, batch):
            n = batch["a"].shape[0]
            return {
                "delta": torch.zeros(n, 3),
                "cov": torch.zeros(n, 3, 3),
                "trust_logit": torch.zeros(n),
                "mass": torch.zeros(n),
                "assign": torch.zeros(n, 768, 576),
            }

    out = Positional(_Stub(), ["a"])(torch.zeros(2, 1))
    assert len(out) == 4
    assert [tuple(t.shape) for t in out] == [(2, 3), (2, 3, 3), (2,), (2,)]
