#!/usr/bin/env python3
"""Export a checkpoint to ONNX, and optionally build a TensorRT engine.

    tools/export.py runs/distilled/best.pt --out build/student.onnx
    tools/export.py runs/distilled/best.pt --out build/student.onnx --trt fp16

The model takes a dict and ONNX takes positional tensors, so it is wrapped --
the same wrapper tests/test_export.py has used since M1, kept here so the tested
path and the deployed path are one path.

Only four outputs cross the boundary: delta, cov, trust_logit and mass. That is
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
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.config import upgrade
from mapposeformer.data import build_dataset
from mapposeformer.model.model import MapPoseFormer
from mapposeformer.prune import apply_plan

#: The inputs the graph takes, in the order it takes them.
INPUTS = ("map_", "det_", "hist_")


class Positional(torch.nn.Module):
    """ONNX takes positional tensors; the model takes a dict."""

    def __init__(self, model: torch.nn.Module, keys: list[str]):
        super().__init__()
        self.model, self.keys = model, keys

    def forward(self, *args):
        """@brief The four tensors a filter consumes.
        @param args Input tensors, in ``self.keys`` order.
        @return ``(delta, cov, trust_logit, mass)``."""
        out = self.model(dict(zip(self.keys, args, strict=True)))
        return out["delta"], out["cov"], out["trust_logit"], out["mass"]


def load(path: str):
    """@brief A checkpoint's model, pruned as saved, on CPU in eval mode.
    @param path Checkpoint. @return ``(model, config)``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = upgrade(ckpt["config"])
    model = MapPoseFormer(cfg.model)
    if ckpt.get("prune_plan"):
        apply_plan(model, ckpt["prune_plan"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg


def main() -> int:
    """@brief Export, and build an engine if asked. @return Exit status."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True, help="where to write the .onnx")
    ap.add_argument(
        "--half", action="store_true", help="export the model in fp16"
    )
    ap.add_argument(
        "--trt", action="store_true", help="also build a TensorRT engine"
    )
    args = ap.parse_args()

    model, cfg = load(args.checkpoint)
    dataset = build_dataset(cfg.data, "val")
    sample = dataset[0]
    keys = [k for k in sample if k.startswith(INPUTS)]
    example = tuple(sample[k].unsqueeze(0) for k in keys)
    if args.half:
        # The engine's precision is the graph's precision: TensorRT 11 has no
        # FP16 builder flag, so half means exporting a half model.
        model = model.half()
        example = tuple(
            t.half() if t.is_floating_point() else t for t in example
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    program = torch.onnx.export(
        Positional(model, keys), example, dynamo=True, verbose=False
    )
    program.save(str(out))
    print(f"{args.checkpoint} -> {out}")
    print(f"  {len(keys)} inputs, 4 outputs, batch 1, static shapes")

    if args.trt:
        from mapposeformer.tensorrt import build_engine

        engine = out.with_suffix(".engine")
        info = build_engine(str(out), str(engine))
        print(f"  engine -> {engine}")
        print(f"  {info['layers']} layers, {info['bytes'] / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
