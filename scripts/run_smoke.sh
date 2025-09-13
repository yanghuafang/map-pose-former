#!/bin/bash

# run_smoke.sh -- train, evaluate and visualise, on generated data, in minutes.
#
# No dataset download and no GPU. This is the "does it work at all" gate, and
# it is the first thing to run on a fresh checkout. It does not produce a good
# localizer -- a hundred steps on twenty-four scenes cannot -- it produces a
# loss that fell, a checkpoint that loads, and an SVG worth looking at.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
. scripts/env.sh && mpf_activate

out="${1:-runs/smoke}"
python tools/train.py --config configs/synth_smoke.yaml "train.out_dir=${out}"
python tools/eval.py "${out}/best.pt" --split test --device cpu
python tools/viz_sample.py --index 30 --checkpoint "${out}/best.pt" --out "${out}/sample.svg"
echo "open ${out}/sample.svg"
