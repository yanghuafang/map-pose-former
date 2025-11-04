"""Build a TensorRT engine, and answer the question INT8 left open.

M4 measured what INT8 costs in accuracy and could not measure what it saves:
simulated quantization rounds and comes straight back, adding arithmetic and
removing none. Only real integer kernels can say, and only TensorRT has them
here. So this module turns a claim into a number.

**TensorRT 11 does not build engines the way this project's roadmap assumed.**
The plan was written against the implicit-quantization workflow every tutorial
still describes -- ``BuilderFlag.FP16``, ``BuilderFlag.INT8``, and an
``IInt8EntropyCalibrator2`` fed with representative data. None of those exist in
11.2: the builder flags are gone, the calibrator classes are gone, and the only
quantization symbols left are ``IQuantizeLayer`` and ``IDynamicQuantizeLayer``.

Precision is now a property of the *graph*, not of the build. A strongly typed
network takes each layer's type from the ONNX it was given, so an FP16 engine
comes from an FP16 export and an INT8 engine comes from a graph carrying
quantize/dequantize nodes -- which is what ``nvidia-modelopt`` inserts, and why
the roadmap named it for the quantization stage even while describing the
calibrator for the build.

That is a better design than the one it replaced: the precision a reader sees in
the graph is the precision that runs, rather than a flag that silently permits a
kernel the builder may or may not choose.

Nothing here is imported unless used. TensorRT is a Linux/CUDA wheel with no
macOS build, and training and evaluation must keep running on a laptop -- the
boundary already drawn around the nuScenes devkit and the detector.
"""

from __future__ import annotations

from pathlib import Path


def build_engine(onnx_path: str, engine_path: str) -> dict[str, object]:
    """@brief Compile an ONNX graph into a strongly typed TensorRT engine.

    The engine's precision is whatever the graph says. To get FP16, export a
    half-precision model; to get INT8, export one carrying Q/DQ nodes. There is
    deliberately no ``precision`` argument, because in TensorRT 11 there is no
    such switch and offering one would be describing a build that cannot
    happen.

    @param onnx_path The graph to compile.
    @param engine_path Where to write the serialised engine.
    @return A summary: layer count, engine bytes, and the input names.
    @throws RuntimeError If the graph does not parse or the build fails.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as fh:
        if not parser.parse(fh.read()):
            errors = "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
            raise RuntimeError(f"ONNX did not parse:\n{errors}")

    config = builder.create_builder_config()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT could not build the engine")

    # IHostMemory is a view over TensorRT's own buffer, not a bytes object:
    # it has no len() and must be copied out before the builder goes away.
    blob = bytes(serialized)
    out = Path(engine_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    return {
        "layers": network.num_layers,
        "bytes": len(blob),
        "inputs": [
            network.get_input(i).name for i in range(network.num_inputs)
        ],
    }
