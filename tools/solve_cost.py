#!/usr/bin/env python3
"""How much of a frame's latency is the solve, and not the network.

    tools/solve_cost.py runs/base/best.pt --batch 1

`tools/cost.py` measures the model as a whole, which is the right unit for
choosing between architectures and the wrong one for deciding where deployment
effort goes. This model is not only a transformer: after the matcher produces
an assignment it runs a damped Gauss-Newton iteration with a Cholesky
factorisation, and that part **does not quantize, does not run on an NPU, and
does not shrink when the weights do**. Every compression result this project
has is about the other part.

So the question a Raspberry Pi or a phone actually poses is: if the trunk and
matcher were made free, what would remain? That is what this measures.

**Method.** The solve functions are wrapped in place with a timer and the whole
forward is run, so the split comes from one execution rather than from two runs
that might differ. The wrapper adds a `perf_counter` pair per call, which is
tens of nanoseconds against milliseconds of work.

**On CUDA the timing is synchronised**, because kernel launches are
asynchronous and an unsynchronised timer would attribute the network's work to
whatever happened to be measured next -- the classic way to conclude that a
matmul is free.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer import solve as solve_mod
from mapposeformer.checkpoint import load_checkpoint
from mapposeformer.data import build_dataset

#: The three pieces that are not the network: the pose iteration, the
#: covariance read off its curvature, and the information form the filter gets.
WRAPPED = (
    "solve_pose_directional",
    "curvature_covariance",
    "measurement_information",
)


def instrument(device: str) -> dict[str, list[float]]:
    """Wrap the solve functions with a synchronised timer, in place.

    @return A dict the wrappers append durations to, one list per function.
    """
    times: dict[str, list[float]] = {n: [] for n in WRAPPED}
    cuda = device.startswith("cuda")

    def wrap(name):
        original = getattr(solve_mod, name)

        def timed(*a, **kw):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = original(*a, **kw)
            if cuda:
                torch.cuda.synchronize()
            times[name].append(time.perf_counter() - t0)
            return out

        return original, timed

    for n in WRAPPED:
        _, timed = wrap(n)
        setattr(solve_mod, n, timed)
    # The model imported these by value at module load, so patching the module
    # alone would leave the model calling the originals. Patch there too.
    from mapposeformer.model import model as model_mod

    for n in WRAPPED:
        if hasattr(model_mod, n):
            setattr(model_mod, n, getattr(solve_mod, n))
    return times


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint)
    model = model.to(device).eval().requires_grad_(False)

    ds = build_dataset(cfg.data, args.split)
    sample = {
        k: torch.stack([ds[i][k] for i in range(args.batch)]).to(device)
        for k in ds[0]
    }

    times = instrument(device)
    cuda = device.startswith("cuda")
    totals: list[float] = []
    with torch.no_grad():
        for i in range(args.warmup + args.repeats):
            if i == args.warmup:  # discard warmup, including its solve times
                totals.clear()
                for v in times.values():
                    v.clear()
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(sample)
            if cuda:
                torch.cuda.synchronize()
            totals.append(time.perf_counter() - t0)

    def med(xs):
        return sorted(xs)[len(xs) // 2] * 1000.0

    total = med(totals)
    print(
        f"{args.checkpoint}  batch={args.batch}  device={device}"
        f"  p50 of {args.repeats}\n"
    )
    print(f"  {'full forward':<28}{total:8.2f} ms")
    solve_total = 0.0
    for n in WRAPPED:
        if not times[n]:
            continue
        # Summed per forward, not per call: the pose iteration runs once but
        # the refinement inside it does not, and what a deployment pays is the
        # per-frame total.
        per_forward = sum(times[n]) / len(totals) * 1000.0
        solve_total += per_forward
        print(
            f"  {n:<28}{per_forward:8.2f} ms"
            f"   {100 * per_forward / total:5.1f}%"
        )
    print(
        f"  {'--- solve, all of it':<28}{solve_total:8.2f} ms"
        f"   {100 * solve_total / total:5.1f}%"
    )
    print(
        f"  {'--- network (the rest)':<28}{total - solve_total:8.2f} ms"
        f"   {100 * (total - solve_total) / total:5.1f}%"
    )
    print()
    # The deployment reading, stated rather than left to the reader: this is
    # the floor that compressing the network cannot go below.
    print(
        f"  compressing the trunk and matcher to nothing would leave"
        f" {solve_total:.2f} ms,"
    )
    print(
        f"  so the best achievable speedup on this path is"
        f" {total / max(solve_total, 1e-9):.1f}x."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
