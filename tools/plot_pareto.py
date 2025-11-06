#!/usr/bin/env python3
"""Draw accuracy against latency, so the shape of the result is visible.

    tools/pareto.py ... --json runs/pareto.json
    tools/plot_pareto.py runs/pareto.json --trt runs/trt.json \\
        --out docs/img/pareto.svg

The table this draws from says the same things, and says them worse. The M4
finding is that four PyTorch configurations sit in a vertical line -- a third of
the parameters removed and the latency does not move -- and that the TensorRT
points are somewhere else entirely. A reader has to reconstruct that from
columns of numbers; a scatter shows it before they have read the axes.

Matplotlib is imported here and nowhere in ``mapposeformer``. Training and
evaluation must keep running on a machine that has not installed a plotting
stack, which is the boundary already drawn around the nuScenes devkit and the
detector. ``pip install matplotlib`` when you want the figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Down and to the left is better: less error, less time.
XLABEL = "latency at batch 1, p50 (ms)"
YLABEL = "translation RMSE (m)"


def _load(path: str) -> list[dict]:
    """@brief Rows from a ``tools/pareto.py --json`` file.

    @param path The file.
    @return Its rows, each carrying the runtime it was measured on.
    """
    data = json.loads(Path(path).read_text())
    runtime = data.get("runtime", "pytorch")
    return [dict(r, runtime=runtime) for r in data["rows"]]


def main() -> int:
    """@brief Draw the figure. @return Exit status."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", help="output of tools/pareto.py --json")
    ap.add_argument(
        "--trt",
        action="append",
        default=[],
        metavar="LABEL=P50",
        help="a TensorRT p50 in ms for a row already in the json, e.g. "
        "distilled=5.27. Accuracy is not repeated: an fp32 engine runs the "
        "same weights through the same arithmetic, so only the time differs.",
    )
    ap.add_argument("--out", default="docs/img/pareto.svg")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = _load(args.json)
    by_label = {r["label"]: r for r in rows}
    for spec in args.trt:
        label, _, p50 = spec.partition("=")
        if label not in by_label:
            print(
                f"--trt names {label!r}, which is not in the json",
                file=sys.stderr,
            )
            return 1
        rows.append(
            dict(
                by_label[label],
                p50_ms=float(p50),
                label=f"{label}, TRT",
                runtime="tensorrt",
            )
        )
    if not rows:
        print("no rows to draw", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styles = {
        "pytorch": dict(marker="o", color="#4c72b0", label="PyTorch eager"),
        "tensorrt": dict(marker="^", color="#c44e52", label="TensorRT fp32"),
    }
    seen = set()
    for r in rows:
        st = dict(styles.get(r["runtime"], styles["pytorch"]))
        # One legend entry per runtime, not one per point.
        if r["runtime"] in seen:
            st.pop("label", None)
        seen.add(r["runtime"])
        # Area with parameter count, so size carries the third dimension the
        # axes cannot: a small marker that is fast and accurate is the goal.
        ax.scatter(
            r["p50_ms"],
            r["trans_m"],
            s=30 + 340 * (r["params_m"] / 26.0),
            alpha=0.75,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            **st,
        )
        ax.annotate(
            r["label"],
            (r["p50_ms"], r["trans_m"]),
            textcoords="offset points",
            xytext=(9, 5),
            fontsize=8,
            zorder=4,
        )

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title("Accuracy against latency — marker area is parameter count")
    ax.grid(alpha=0.25, zorder=0)
    ax.set_xlim(left=0)
    # Proxy handles, because a legend built from the scatter would inherit one
    # point's area and show the parameter count as though it were the key.
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                linestyle="none",
                markersize=7,
                marker=styles[k]["marker"],
                color=styles[k]["color"],
                label=styles[k]["label"],
            )
            for k in seen
        ],
        frameon=False,
        loc="best",
    )
    ax.annotate(
        "lower left is better",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        fontsize=8,
        style="italic",
        alpha=0.6,
    )
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"{len(rows)} points -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
