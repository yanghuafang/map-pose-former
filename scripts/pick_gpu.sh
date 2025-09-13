#!/bin/bash

# pick_gpu.sh -- print the UUID of GPU 0, if it has room.
#
#   scripts/pick_gpu.sh        # GPU 0's UUID, or nothing
#   scripts/pick_gpu.sh 8200   # ...but only if 8200 MiB are free
#
# A UUID rather than the index, because CUDA's enumeration is not nvidia-smi's:
# pinning "0" can select a different card than the one inspected. The UUID
# survives both a reorder and a reboot.
#
# **Busy means "has no room", not "is working".** Utilisation is instantaneous
# and says nothing about whether a second process would fit -- a card at 100%
# with 40 GiB free is a fine place to put an eval, and a card at 0% with 200 MiB
# free is not. The failure this avoids is an out-of-memory an hour into someone
# else's training run, which costs both.
#
# Prints nothing when the card has no room. What the caller does with that is
# remote-ubuntu.sh's business, and it says so there.

set -uo pipefail

# The default is a verification task -- an eval or a test run at batch 32, about
# 4.5 GiB -- because that is what the check exists for. A training job must say
# what it needs: the student reserves 8.0 GiB at its own batch size and the
# teacher 19.1.
need="${1:-4500}"

nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits \
  2>/dev/null | awk -F', *' -v need="${need}" '
  NR == 1 && ($2 + 0) >= (need + 0) { print $1 }   # GPU 0, or nothing
'
