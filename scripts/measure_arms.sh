#!/bin/bash
#
# measure_arms.sh -- score finished checkpoints on a held-out split.
#
#   scripts/measure_arms.sh <split> <arm ...>
#   scripts/measure_arms.sh test tokens_point_s0 tokens_element_s0
#
# Training logs validation metrics every epoch; `docs/RESULTS.md` reports test.
# `tools/eval.py` computes the full set from a checkpoint, so an arm is scored
# after it lands rather than retrained in order to be scored.
#
# Both element arms are reported: the one under `relative` geometry and the one
# under `rope`. Which of them is called the control changes the sign of the
# tokens verdict, so neither is quietly dropped and the table says which is
# which.
#
# Threads are capped because evaluation is loader-bound, not compute-bound: an
# unconstrained eval measured 372% CPU against 142% capped, for the same work.
# Uncapped, a measurement run starves whatever is still training beside it.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
. scripts/env.sh && mpf_activate 2>/dev/null || true

if [ "$#" -lt 2 ]; then
  echo "usage: measure_arms.sh <split> <arm ...>" >&2
  echo "  no default set: a measurement script that guesses which arms you" >&2
  echo "  meant is how one arm's evaluation overwrites another's." >&2
  exit 2
fi
SPLIT="$1"
shift
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
# Measurement must not compete with training for VRAM, and on a host kept
# deliberately saturated there is none to compete for: with eight arms running,
# the freest card had 101 MiB and evaluation died trying to allocate 108. So
# pick a card only if one genuinely has room, and otherwise run on the CPU --
# slower, but it finishes, which a GPU run that OOMs does not.
NEED_EVAL_MIB="${MPF_EVAL_MIB:-2500}"
best_gpu=""
best_free=0
while IFS=, read -r idx free; do
  idx=$(echo "$idx" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
  if [ "$free" -gt "$best_free" ]; then best_free=$free; best_gpu=$idx; fi
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)

if [ -n "$best_gpu" ] && [ "$best_free" -ge "$NEED_EVAL_MIB" ]; then
  export CUDA_VISIBLE_DEVICES="${MPF_GPU_INDEX:-$best_gpu}"
  DEVICE_ARG=""
  echo "measuring on GPU ${CUDA_VISIBLE_DEVICES} (${best_free} MiB free)"
else
  export CUDA_VISIBLE_DEVICES=""
  DEVICE_ARG="--device cpu"
  echo "no GPU has ${NEED_EVAL_MIB} MiB free (best ${best_free}); measuring on CPU"
fi

ARMS=("$@")

# Named for the arm as well as the clock. A timestamp alone collides -- the
# scheduler detaches each measurement, so two can start within the same second
# -- and `tee` truncates, so one arm's evaluation overwrites another's with no
# error anywhere.
out="runs/measured_$(printf '%s' "${ARMS[0]:-set}" | tr -c 'A-Za-z0-9_' '_')_$(date +%Y%m%d-%H%M%S).log"

failures=0
echo "########## arm measurement, split=$SPLIT" | tee "$out"

for a in "${ARMS[@]}"; do
  ckpt="runs/$a/best.pt"
  if [ ! -f "$ckpt" ]; then
    echo "MISSING $a" | tee -a "$out"; continue
  fi
  # An arm that has not finished is worth measuring but must be labelled, or
  # a partial number is indistinguishable from a final one. Mid-training
  # values move: across three seeds the spread at epoch 11 was 17.8% while two
  # converged replicates sit 1.0% apart.
  state="FINAL"
  log=$(ls -t "runs/${a}"_*.log 2>/dev/null | head -1)
  [ -n "$log" ] && ! grep -q '^########## DONE' "$log" && state="PARTIAL"

  echo "" | tee -a "$out"
  echo "===== $a [$state]" | tee -a "$out"
  [ -f "runs/$a/config.yaml" ] && grep -E "^  (tokens|residual|geometry|point_geometry|layers|dim|head_dim|rope_bands):" \
    "runs/$a/config.yaml" | tr -d ' ' | tr '\n' ' ' | tee -a "$out" && echo "" | tee -a "$out"
  # `| tee` makes the pipeline's status tee's, so the eval's own failure was
  # invisible and the script exited 0 while writing a traceback. A scheduler
  # reading that exit code learned nothing.
  python3 tools/eval.py "$ckpt" --split "$SPLIT" $DEVICE_ARG 2>&1 | tee -a "$out"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "FAILED $a" | tee -a "$out"
    failures=$((failures + 1))
  fi
done

echo "" | tee -a "$out"
echo "########## DONE arm measurement, $failures failed" | tee -a "$out"
echo "$out"
exit $(( failures > 0 ? 1 : 0 ))
