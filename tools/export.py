#!/usr/bin/env python3
"""Export a checkpoint to ONNX, and optionally build a TensorRT engine.

    tools/export.py runs/distilled/best.pt --out build/student.onnx
    tools/export.py runs/distilled/best.pt --out build/student.onnx --trt fp16

The model takes a dict and ONNX takes positional tensors, so it is wrapped. The
wrapper lives here rather than at each call site, so the path that is exported
and the path that is deployed are one path.

Only four outputs cross the boundary: delta, cov, information and mass. That is
what LocalizationKF::Update consumes, and everything else the model returns --
the assignment, the surface, the per-pass poses -- is diagnostic. Exporting them
would put a 768 x 576 tensor through the engine for nothing.

**Precision comes from the graph, not from a build flag.** TensorRT 11 removed
``BuilderFlag.FP16``, ``BuilderFlag.INT8`` and the calibrator API; a strongly
typed engine takes each layer's type from the ONNX it was handed. So ``--half``
exports a half-precision model, and INT8 means exporting one that already
carries quantize/dequantize nodes. See ``mapposeformer/tensorrt.py``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.checkpoint import build_model
from mapposeformer.config import upgrade
from mapposeformer.data import build_dataset
from mapposeformer.data.dataset import DataParams

#: The inputs the graph takes, in the order it takes them.
INPUTS = ("map_", "det_", "hist_")


class Positional(torch.nn.Module):
    """ONNX takes positional tensors; the model takes a dict."""

    def __init__(self, model: torch.nn.Module, keys: list[str]):
        super().__init__()
        self.model, self.keys = model, keys

    def forward(self, *args):
        """@brief The tensors a filter consumes.

        Four tensors, and which four is the design decision: `information` is
        the measurement in information form, which is what a Kalman update
        actually wants. `cov` is the posterior -- the prior already fused in --
        so a filter handed that would count its own prior twice. Both cross the
        boundary because the two readers need different matrices, and
        `solve.py` says so at the point it builds them.

        @param args Input tensors, in ``self.keys`` order.
        @return ``(delta, cov, information, mass)``."""
        out = self.model(dict(zip(self.keys, args, strict=True)))
        return out["delta"], out["cov"], out["information"], out["mass"]


class Trunk(torch.nn.Module):
    """Everything up to the assignment, and nothing after it.

    **The whole model cannot be exported, and this is not a limitation of the
    exporter.** `solve_pose_directional` runs a damped Gauss-Newton iteration
    whose inner step is a linear solve, and `torch.onnx.export` fails on it
    outright: *"No ONNX function found for aten._linalg_solve_ex"*. There is no
    operator to lower it to, in any opset.

    So deployment is split, necessarily rather than by preference: the trunk and
    matcher run on the accelerator, and the solve runs on the host. That is also
    what `tools/solve_cost.py` already implied -- the solve is 28% of a frame
    and compresses by nothing, so the best any engine can do on this path is
    3.6x.

    What crosses the boundary is the assignment, which at point resolution is
    768 x 576. That is 1.7 MB a frame, and it is the real cost of the split.
    """

    def __init__(self, model: torch.nn.Module, keys: list[str]):
        super().__init__()
        self.model, self.keys = model, keys

    def forward(self, *args):
        """@return ``(elements, scores)`` -- the assignment and its logits."""
        m = self.model
        b = dict(zip(self.keys, args, strict=True))
        det_pts, det_pmask, det_cls, det_attr, extra = m.detections(b)
        map_e = m.map_encoder(
            b["map_pts"], b["map_pmask"], b["map_cls"], b["map_attr"]
        )
        det_e = m.det_encoder(det_pts, det_pmask, det_cls, det_attr, extra)
        elements, scores, _, _, _, _ = m.matcher.elements(map_e, det_e)
        return elements, scores


def split_of(params: DataParams, split: str):
    """@brief A split, from the cache if it is there and the generator if not.

    A missing cache is not a reason to refuse to export. The cache is a
    materialised copy of a split the reader underneath can still produce, and
    clearing ``cache_dir`` asks that reader directly -- a few generated scenes
    instead of the 971 MiB the cache would have to be built to.

    @param params The checkpoint's data configuration.
    @param split ``train``, ``val`` or ``test``.
    @return A dataset.
    """
    try:
        return build_dataset(params, split)
    except FileNotFoundError:
        return build_dataset(replace(params, cache_dir=""), split)


def example_sample(params: DataParams) -> dict:
    """@brief One sample, for its shapes. The values never reach the graph.

    Tracing needs a tensor of the right shape and dtype per input and nothing
    else, which is why any split will do and the values are never read.

    @param params The checkpoint's data configuration.
    @return The first validation sample.
    """
    return split_of(params, "val")[0]


def load(path: str):
    """@brief A checkpoint's model, pruned as saved, on CPU in eval mode.
    @param path Checkpoint. @return ``(model, config)``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = upgrade(ckpt["config"])
    model = build_model(ckpt, cfg.model)
    return model.eval(), cfg


def main() -> int:
    """@brief Export, and build an engine if asked. @return Exit status."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument(
        "--trunk-only",
        action="store_true",
        help="export the network without the solve, which cannot be exported",
    )
    ap.add_argument("--out", required=True, help="where to write the .onnx")
    ap.add_argument(
        "--half", action="store_true", help="export the model in fp16"
    )
    ap.add_argument(
        "--trt", action="store_true", help="also build a TensorRT engine"
    )
    ap.add_argument(
        "--int8",
        action="store_true",
        help="calibrate and export a graph carrying quantize/dequantize nodes",
    )
    ap.add_argument(
        "--calib-batches",
        type=int,
        default=16,
        help="batches to calibrate on, from the train split",
    )
    args = ap.parse_args()

    model, cfg = load(args.checkpoint)
    sample = example_sample(cfg.data)
    keys = [k for k in sample if k.startswith(INPUTS)]
    example = tuple(sample[k].unsqueeze(0) for k in keys)

    if args.int8:
        # Calibrated on **train**. Activation ranges read off the split being
        # scored would tune the model on its own test set, and the resulting
        # number would be the one thing this project is careful never to
        # report.
        from torch.utils.data import DataLoader

        from mapposeformer.quantize import QuantParams, calibrate, quantize

        wrapped = quantize(model, QuantParams())
        loader = DataLoader(split_of(cfg.data, "train"), batch_size=2)
        calibrate(model, loader, limit=args.calib_batches)
        print(f"{wrapped} layers quantized, calibrated on train")

    if args.half:
        # The engine's precision is the graph's precision: TensorRT 11 has no
        # FP16 builder flag, so half means exporting a half model.
        model = model.half()
        example = tuple(
            t.half() if t.is_floating_point() else t for t in example
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # `load` returns an eval model, but wrapping it makes a fresh Module whose
    # own flag defaults to training, and the exporter warns on that flag rather
    # than on the weights underneath. Nothing here reads it -- there is no
    # dropout and no batch norm in this model -- so the warning is noise, which
    # is exactly why it should not be printed on every export.
    wrapper = Trunk(model, keys) if args.trunk_only else Positional(model, keys)

    # **The names are the deployment contract.** Left to itself the exporter
    # calls the inputs args_0 to args_16 and the outputs whatever aten op
    # produced them, and then the only way to learn that args_9 is the
    # detector's reported noise is to re-derive it from this file. The graph is
    # published and the engine built from it is bound by position, so a reader
    # who guesses wrong gets a plausible pose from transposed inputs rather
    # than an error. `keys` is already the order the wrapper unpacks.
    outputs = (
        ["elements", "scores"]
        if args.trunk_only
        else ["delta", "cov", "information", "mass"]
    )
    program = torch.onnx.export(
        wrapper.eval(),
        example,
        input_names=list(keys),
        output_names=outputs,
        dynamo=True,
        verbose=False,
    )
    program.save(str(out))
    # Built before the f-string rather than inside it: a conditional spanning
    # lines inside a replacement field is Python 3.12 syntax, and this project
    # supports 3.10.
    tail = ", solve left on the host" if args.trunk_only else ""
    named = f"{len(outputs)} outputs ({', '.join(outputs)}){tail}"
    print(f"{args.checkpoint} -> {out}")
    print(f"  {len(keys)} inputs, {named}, batch 1, static shapes")

    if args.trt:
        from mapposeformer.tensorrt import build_engine

        engine = out.with_suffix(".engine")
        info = build_engine(str(out), str(engine))
        print(f"  engine -> {engine}")
        print(f"  {info['layers']} layers, {info['bytes'] / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
