#!/bin/bash
#
# sweep_refine.sh -- how many refinement passes the solve actually needs.
#
#   scripts/sweep_refine.sh <split> <checkpoint ...>
#
# `refine_iters` is the number of damped Gauss-Newton passes in
# `solve_pose_directional`. It costs no training to sweep, because the solve
# has **no parameters**: the same weights produce the same assignment, and only
# what the solver does with it changes. So this is an eval-time override on
# checkpoints that already exist.
#
# It is worth asking for three reasons. The live default is 3 IRLS passes
# inside the solve, and nobody has checked whether 3 is right. Each pass is
# pure inference cost on the deployment path. And the passes change `cost`,
# which sets `s^2 = cost / (dof - 3)`, so they move the **covariance** as well
# as the pose. The covariance is 2 to 5x too wide, which makes that the
# interesting half.
#
# **What this can and cannot say.** The model was trained at 3, so its
# assignment is adapted to 3. A sweep here measures how the solve behaves under
# a fixed assignment, not what the optimum would be if the model were retrained
# at each value. Read it as robustness and as inference cost, and if some value
# clearly wins, retrain there before believing it.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
. scripts/env.sh && mpf_activate 2>/dev/null || true

if [ "$#" -lt 2 ]; then
  echo "usage: sweep_refine.sh <split> <checkpoint ...>" >&2
  echo "  one checkpoint of each residual kind is the useful set: the passes" >&2
  echo "  interact with the residual, since a rank-1 line residual leaves a" >&2
  echo "  direction the solve cannot move in." >&2
  exit 2
fi
SPLIT="$1"
shift
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export CUDA_VISIBLE_DEVICES="${MPF_GPU_INDEX:-0}"

CKPTS=("$@")

# 0 is excluded deliberately: it returns an all-zero Hessian, so the reported
# covariance would be exactly the prior, and `solve.py` now raises rather than
# letting that pass silently.
ITERS=(1 2 3 5 8)

out="runs/refine_$(date +%Y%m%d-%H%M%S).log"
echo "########## refine_iters sweep, split=$SPLIT" | tee "$out"

for c in "${CKPTS[@]}"; do
  [ -f "runs/$c/best.pt" ] || { echo "MISSING $c" | tee -a "$out"; continue; }
  for n in "${ITERS[@]}"; do
    echo "" | tee -a "$out"
    echo "===== $c refine_iters=$n" | tee -a "$out"
    python3 tools/eval.py "runs/$c/best.pt" --split "$SPLIT" \
      "model.refine_iters=$n" 2>&1 | tee -a "$out"
  done
done

echo "" | tee -a "$out"
echo "########## DONE refine sweep" | tee -a "$out"
echo "$out"
