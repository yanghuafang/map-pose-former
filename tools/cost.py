#!/usr/bin/env python3
"""What a configuration costs: parameters, latency, and peak memory.

    tools/cost.py  # the shapes this project is choosing between
    tools/cost.py --batch 1 --batch 64
    tools/cost.py --device cuda --repeats 50

`ROADMAP.md` sizes this model on two claims -- that **parameters are close to
free** and that **tokens are what cost** -- and neither has been measured here.
The evidence behind them was gathered on a different network, in eager PyTorch:
2.28 M parameters down to 1.49 M moved latency by 0%, and an 11x larger model
cost 1.7x the time. That is exactly the kind of figure this project has learned
not to carry across architectures unchecked.

It matters now rather than in principle. The heads experiment runs cells at
+62% and -31% parameters, and "did the bigger cell win?" cannot be traded
against cost without knowing the cost. The tokens result moved the model from
168 tokens to 1 344, which is the axis the roadmap says is the expensive one.

**This is deliberately not `bench.py`.** That one measures the data side, which
has been the real ceiling -- these runs are loader-bound, and a step never goes
faster than its data arrives. This measures the model alone, on synthetic
tensors of the right shapes, so it can run on a busy machine without competing
for the dataloader. What it reports is therefore a *floor* on step time, not a
prediction of `frames_per_s`.

Batch 1 and batch 64 answer different questions. At 64 the arithmetic dominates
and the numbers scale with work; at 1 the model is launch-bound, and launch
bound is where fewer, wider heads measure *slower* -- the measurement that
retires the tensor-core argument.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.model import MapPoseFormer, ModelParams

#: The shapes a sample actually has; see `DATASET.md`. Synthesised rather than
#: loaded, because loading is the thing this tool exists to stay out of.
SHAPES = {
    "map_pts": ((72, 8, 2), torch.float32),
    "map_pmask": ((72, 8), torch.bool),
    "map_cls": ((72,), torch.int64),
    "map_attr": ((72,), torch.int64),
    "det_pts": ((32, 8, 2), torch.float32),
    "det_pmask": ((32, 8), torch.bool),
    "det_cls": ((32,), torch.int64),
    "det_attr": ((32,), torch.int64),
    "det_conf": ((32,), torch.float32),
    "det_sigma": ((32, 2), torch.float32),
    "hist_pts": ((2, 32, 8, 2), torch.float32),
    "hist_pmask": ((2, 32, 8), torch.bool),
    "hist_cls": ((2, 32), torch.int64),
    "hist_attr": ((2, 32), torch.int64),
    "hist_conf": ((2, 32), torch.float32),
    "hist_sigma": ((2, 32, 2), torch.float32),
    "hist_rel": ((2, 3), torch.float32),
    "delta": ((3,), torch.float32),
    "prior": ((3,), torch.float32),
    "gt": ((3,), torch.float32),
}

#: The configurations worth pricing: the heads cells, the token modes, and the
#: depth and width the sizing table says were never measured.
CELLS = [
    ("heads A  2x64", dict(tokens="point", heads=2, head_dim=64, rope_bands=5)),
    ("heads B  4x64", dict(tokens="point", heads=4, head_dim=64, rope_bands=5)),
    ("heads C  2x32", dict(tokens="point", heads=2, head_dim=32, rope_bands=5)),
    ("tokens  point", dict(tokens="point")),
    ("tokens element", dict(tokens="element")),
    ("layers      2", dict(tokens="point", layers=2)),
    ("layers      4", dict(tokens="point", layers=4)),
    ("dim        64", dict(tokens="point", dim=64)),
    ("dim       128", dict(tokens="point", dim=128)),
    ("dim       256", dict(tokens="point", dim=256)),
]


def _free(device: str) -> None:
    """Drop whatever the failed attempt left behind before the next cell."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def batch(n: int, device: str) -> dict[str, torch.Tensor]:
    """A batch of `n` samples with the right shapes and plausible values.

    The map and detection points are spread over a 50 m crop so the geometry
    encodings see realistic magnitudes; masks are all-true, which is the
    *expensive* case and therefore the honest one to quote.
    """
    out = {}
    for k, (shape, dtype) in SHAPES.items():
        full = (n, *shape)
        if dtype is torch.bool:
            out[k] = torch.ones(full, dtype=dtype, device=device)
        elif dtype is torch.int64:
            out[k] = torch.zeros(full, dtype=dtype, device=device)
        elif k.endswith("_pts"):
            out[k] = (torch.rand(full, device=device) - 0.5) * 50.0
        else:
            out[k] = torch.rand(full, device=device) * 0.1
    return out


def time_it(fn, repeats: int, device: str) -> tuple[float, float]:
    """@return ``(p50, p90)`` milliseconds, after five warm-up calls.

    Warm-up matters more than usual here: the first CUDA call pays kernel
    autotuning and allocator growth, and quoting it would make every small
    model look slow.
    """
    for _ in range(5):
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2], times[int(0.9 * len(times))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--batch", type=int, action="append", default=None)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument(
        "--backward", action="store_true", help="time forward+backward too"
    )
    args = ap.parse_args()
    batches = args.batch or [1, 64]
    device = args.device

    print(f"device {device}   repeats {args.repeats}")
    head = f"{'cell':<16}{'params':>10}"
    for n in batches:
        head += f"{f'fwd b{n} p50':>13}{'p90':>8}"
        if args.backward:
            head += f"{f'fwd+bwd b{n}':>13}"
    head += f"{'peak MiB':>10}"
    print(head)

    for name, over in CELLS:
        p = ModelParams(**{**dict(dim=128, layers=4, residual="line"), **over})
        try:
            model = MapPoseFormer(p).to(device)
        except Exception as exc:  # a cell the code refuses is worth printing
            print(f"{name:<16} {type(exc).__name__}: {exc}")
            continue
        model.eval()
        n_par = sum(x.numel() for x in model.parameters())
        row = f"{name:<16}{n_par / 1e6:>9.3f}M"
        peak = 0.0
        for n in batches:
            b = batch(n, device)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()

            # `model` and `b` are bound as defaults, not captured: a closure
            # over a loop variable reads its latest value, not the one from
            # the iteration that created the function.
            def fwd(model=model, b=b):
                with torch.no_grad():
                    model(b)

            try:
                p50, p90 = time_it(fwd, args.repeats, device)
                row += f"{p50:>13.2f}{p90:>8.2f}"
            except torch.OutOfMemoryError:
                # The biggest cell running out of memory is a *result* -- it is
                # the cost being measured -- so it must not take the rest of
                # the table with it.
                row += f"{'OOM':>13}{'':>8}"
                _free(device)
            if args.backward:
                model.train()

                def fwdbwd(model=model, b=b):
                    out = model(b)
                    out["delta"].square().sum().backward()
                    model.zero_grad(set_to_none=True)

                try:
                    bp50, _ = time_it(fwdbwd, max(args.repeats // 3, 5), device)
                    row += f"{bp50:>13.2f}"
                except torch.OutOfMemoryError:
                    row += f"{'OOM':>13}"
                    _free(device)
                model.eval()
            if device.startswith("cuda"):
                peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
        row += f"{peak:>10.0f}"
        del b
        print(row, flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
