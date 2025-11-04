#!/bin/bash

# remote-ubuntu.sh -- run any of the scripts here on the Ubuntu box.
#
# The RTX A6000 is on the Linux host and this repository is edited on a laptop,
# so every experiment crosses a network. This mirrors the working tree --
# uncommitted edits included -- and runs a command there.
#
#   ./remote-ubuntu.sh --sync scripts/ci.sh              # copy the tree, then run CI
#   ./remote-ubuntu.sh --sync tools/train.py --config configs/synth_base.yaml
#   ./remote-ubuntu.sh --sync                            # copy and stop
#   ./remote-ubuntu.sh --shell 'nvidia-smi; df -h ~'     # one-off probe
#   ./remote-ubuntu.sh --detach tools/train.py --config configs/synth_base.yaml
#
# Copying is opt-in because it is the only destructive step: it is rsync
# --delete against the remote checkout, so whatever is there is made to match
# this machine. A command that only reads or evaluates should not have to think
# about that -- and datasets and run directories live outside the tree for
# exactly this reason (see MPF_REMOTE_DATA below).
#
# Commands run from the remote repository root, not from scripts/, because the
# tools here are invoked as `tools/train.py` and `scripts/ci.sh` from the root.

set -euo pipefail

REMOTE_HOST="${MPF_REMOTE_HOST:-yanghuafang@192.168.10.13}"
REMOTE_DIR="${MPF_REMOTE_DIR:-study-projects/map-pose-former}"
REMOTE_DATA="${MPF_REMOTE_DATA:-/DATA/map-pose-former}"
REMOTE_GPU="${MPF_GPU:-auto}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Kill training on the host. By PID and with a bracketed pattern, because
# `pkill -f tools/train.py` matches the shell running it -- the string is in
# its own argument list -- so pkill kills itself, the targets survive, and a
# following pgrep reports success. That mistake left three runs competing for
# one GPU for five hours.
mpf_stop_runs() {
  ssh "${REMOTE_HOST}" 'for p in $(pgrep -f "train[.]py"); do kill -9 "$p" 2>/dev/null; done
    sleep 3; echo "still running: $(pgrep -cf "train[.]py")"'
}

usage() {
  cat <<'EOF'
Usage: remote-ubuntu.sh [--sync] [--detach] [--shell] [command ...]

Run a command on the Ubuntu host, optionally mirroring this tree there first.

  remote-ubuntu.sh scripts/ci.sh           Run; do not copy anything.
  remote-ubuntu.sh --sync scripts/ci.sh    Copy this tree over, then run.
  remote-ubuntu.sh --sync                  Copy this tree over and stop.
  remote-ubuntu.sh --shell 'nvidia-smi'    Run shell text rather than argv.

Options:
  --sync      rsync --delete this working tree to the host. Anything edited
              only on the host, inside the checkout, is lost. Runs, datasets
              and checkpoints are kept outside it and are never touched.
  --detach    Run the command under nohup, detached from this ssh session,
              logging to runs/ on the host, and return immediately. A training
              run outlives the laptop that started it; without this it dies
              with the connection.
  --shell     Treat the arguments as shell text, so pipes and semicolons work.
  -h, --help  Show this help.

Environment:
  MPF_REMOTE_HOST  user@host        (default yanghuafang@192.168.10.13)
  MPF_REMOTE_DIR   checkout path    (default study-projects/map-pose-former)
  MPF_REMOTE_DATA  dataset path     (default /DATA/map-pose-former, a 3.6 TB
                   drive on the host). Absolute paths are used as given; a
                   relative one is taken from the remote home directory.
                   Exported to the command as MPF_DATA_ROOT.
  MPF_GPU          GPU to pin       (default "auto": GPU 0 if it has room).
                   A name forces a specific card.
  MPF_GPU_FREE_MIB free VRAM "auto" requires (default 4500, a verification
                   run).
EOF
}

do_sync=false
as_shell=false
do_detach=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync)    do_sync=true; shift ;;
    --shell)   as_shell=true; shift ;;
    --detach)  do_detach=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --)        shift; break ;;
    *)         break ;;
  esac
done

if [[ $# -eq 0 ]] && [[ "$do_sync" == false ]]; then
  echo "Nothing to do: give a command, or --sync to place the tree." >&2
  usage >&2
  exit 1
fi

if [[ "$do_sync" == true ]]; then
  echo "Syncing ${repo_root}/ -> ${REMOTE_HOST}:${REMOTE_DIR}/"
  # .git stays on the laptop, which is the source of truth for history. runs/
  # and build/ are excluded in both directions: --delete would otherwise wipe
  # the checkpoints of whatever is training on the host right now, and the
  # TensorRT engines, which take minutes to compile and exist only there.
  ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR} ${REMOTE_DATA}"
  rsync -az --delete \
    --exclude '.git/' \
    --exclude '.claude/' \
    --exclude 'env/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude 'runs/' \
    --exclude 'build/' \
    --exclude '.DS_Store' \
    "${repo_root}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
fi

if [[ $# -eq 0 ]]; then
  exit 0
fi

if [[ "$as_shell" == true ]]; then
  remote_cmd="$*"
else
  remote_cmd="$(printf '%q ' "$@")"
fi

# Detached runs log into runs/, which is the one directory --sync never
# deletes -- so re-syncing the tree mid-experiment cannot destroy the record of
# what is currently running.
if [[ "$do_detach" == true ]]; then
  remote_cmd_body="${remote_cmd}"
  log="runs/$(date +%Y%m%d-%H%M%S).log"
  # PYTHONUNBUFFERED, because stdout to a file is block buffered: without it the
  # log arrives in 4 kB instalments and `tail -f` on a training run shows
  # nothing for the first several hundred steps, which is exactly the window in
  # which a run is worth watching.
  #
  # Braced, because the command may be compound. `nohup cmd1 && cmd2 > log &`
  # nohups only the first, redirects only the second, and backgrounds the pair
  # -- so the first writes to this ssh session's pipe and dies of SIGPIPE when
  # it closes, which is a training run lost some hours later for no visible
  # reason. The group takes the redirect and the background as a unit.
  remote_cmd="mkdir -p runs; export PYTHONUNBUFFERED=1;"
  remote_cmd+=" { ${remote_cmd_body} ; } > ${log} 2>&1 < /dev/null &"
  remote_cmd+=" pid=\$!; disown 2>/dev/null;"
  remote_cmd+=" echo detached pid \$pid logging to ${log}"
  echo "follow it with: $0 --shell 'tail -f ${log}'"
fi

tty_flag=()
if [[ -t 0 ]]; then tty_flag=(-t); fi

# `cd || exit` rather than `cd &&`: without it a missing checkout runs the
# command in the home directory instead, which surfaces as a strange failure
# somewhere later rather than as the missing tree it is.
#
# CUDA's own enumeration is not nvidia-smi's, so `cuda:0` is not a promise
# about which card. scripts/pick_gpu.sh resolves GPU 0 to its UUID, which survives a reorder and a reboot, and the run is pinned to
# it.
#
# The default is `auto`, which asks scripts/pick_gpu.sh for GPU 0's UUID and
# gets nothing back when the card has no room; that script carries the reason
# for both halves of it. When nothing comes back the variable is left unset
# rather than set empty: an empty CUDA_VISIBLE_DEVICES hides every GPU, and a run that
# quietly fell back to CPU is worse than one that never started.
# Datasets live off the checkout, on their own drive, so that --sync's
# rsync --delete can never reach them. An absolute MPF_REMOTE_DATA is used as
# given; a relative one is read from the remote home directory.
case "${REMOTE_DATA}" in
  /*) data_root="${REMOTE_DATA}" ;;
  *)  data_root="\$HOME/${REMOTE_DATA}" ;;
esac

prologue="cd ${REMOTE_DIR} || exit 1"
prologue+="; export MPF_DATA_ROOT=${data_root}"
if [ "${REMOTE_GPU}" = "auto" ]; then
  prologue+="; gpu=\$(bash scripts/pick_gpu.sh \"\${MPF_GPU_FREE_MIB:-4500}\")"
else
  prologue+="; gpu=\$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader"
  prologue+=" 2>/dev/null | grep -m1 -F \"${REMOTE_GPU}\" | cut -d, -f1)"
fi
prologue+="; [ -n \"\$gpu\" ] && export CUDA_VISIBLE_DEVICES=\$gpu"
prologue+="; . scripts/env.sh && mpf_activate"

# A login shell, so the command starts from the PATH the host's profile builds
# -- CUDA and any pyenv/conda shims live there and `ssh host cmd` reads no
# profile at all. The conda environment is activated if setup.sh made one;
# scripts/env.sh is a no-op when it did not.
exec ssh ${tty_flag[@]+"${tty_flag[@]}"} "${REMOTE_HOST}" \
  "bash -lc '${prologue}; ${remote_cmd}'"
