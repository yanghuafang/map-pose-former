#!/bin/bash

# m4.sh -- the compression milestone, end to end, on the training box.
#
#   ./scripts/m4.sh                    # wait for the GPU, then every stage
#   ./scripts/m4.sh --keep 0.5         # one channel fraction instead of three
#   ./scripts/m4.sh --keep-heads 0.5   # prune heads instead of channels
#
# Run after a teacher and a distilled student exist. It waits for any training
# already on the box to finish rather than competing with it -- three runs
# sharing one GPU is a mistake this project has made once already.
#
# The output is docs/RESULTS.md's M4 table, produced by tools/pareto.py in a
# single process so every row is measured the same way.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
. scripts/env.sh && mpf_activate

# This milestone trains, and the default only checks for room enough to run a
# verification job. Say what training actually needs rather than accepting it.
export MPF_GPU_FREE_MIB="${MPF_GPU_FREE_MIB:-8200}"

# What to prune, as `flag:fraction` pairs. Channels by default; heads take the
# same fine-tune budget, so the two prunings can be read against each other.
PLAN=(keep:0.75 keep:0.5 keep:0.25)
case "${1:-}" in
  --keep)       PLAN=("keep:$2") ;;
  --keep-heads) PLAN=("keep-heads:$2") ;;
  "")           ;;
  *) echo "unknown option $1" >&2; exit 2 ;;
esac

# A quarter of the 37 000 steps the student needed, which is the fine-tune
# budget docs/ROADMAP.md costs the milestone at. At a third of the learning
# rate: this is recovering from an injury, not learning the task again.
FT_STEPS=9250
FT_LR=1.0e-4

say() { printf '\n=== %s\n' "$1"; }

say "waiting for the GPU"
while pgrep -f "train[.]py" >/dev/null; do sleep 120; done
echo "clear"

say "the distilled student, on test"
python tools/eval.py runs/distilled/best.pt --split test | sed -n '/^all frames/,/recall <= 0.25/p'

# Prune whichever student is actually better. If distillation did not help,
# pruning its output would carry that loss into every row below it.
base=$(python - <<'PY'
import torch
from mapposeformer.config import upgrade  # noqa: F401
best, pick = 1e9, "runs/m1_base/best.pt"
for path in ("runs/m1_base/best.pt", "runs/distilled/best.pt"):
    try:
        m = torch.load(path, map_location="cpu", weights_only=False)["metrics"]
        v = m["all/rmse_trans_m"]
        if v < best:
            best, pick = v, path
    except Exception:
        pass
print(pick)
PY
)
say "pruning from ${base}"

rows=(teacher=runs/teacher/best.pt student=runs/m1_base/best.pt)
[[ -f runs/distilled/best.pt ]] && rows+=(distilled=runs/distilled/best.pt)

for entry in "${PLAN[@]}"; do
  flag="${entry%%:*}"; frac="${entry#*:}"
  if [ "${flag}" = keep ]; then tag="p${frac/0./}"; label="pruned"
  else tag="h${frac/0./}"; label="heads"; fi
  say "prune --${flag} ${frac}, then fine-tune ${FT_STEPS} steps"
  python tools/prune.py "${base}" "--${flag}" "${frac}" \
    --out "runs/${tag}/init.pt"
  python tools/train.py --config configs/synth_base.yaml \
    "train.init_from=runs/${tag}/init.pt" "train.max_steps=${FT_STEPS}" \
    "train.lr=${FT_LR}" "train.out_dir=runs/${tag}" 2>&1 | tail -2
  rows+=("${label}@${frac}=runs/${tag}/best.pt")
  smallest="${tag}"
done

# INT8 on the smallest survivor: the row that says whether quantization costs
# anything once the model has already been made small.
rows+=("${rows[-1]%%=*}+int8=runs/${smallest}/best.pt:quantize")

say "the table"
python tools/pareto.py "${rows[@]}" --split test | tee runs/m4_table.md
