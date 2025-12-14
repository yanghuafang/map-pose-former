#!/bin/bash

# remote-ubuntu.sh -- run any of the scripts here on the training host.
#
# The GPU is on a Linux host and this repository is edited somewhere else, so
# every experiment crosses a network. This mirrors the working tree --
# uncommitted edits included -- and runs a command there.
#
#   ./remote-ubuntu.sh --sync scripts/ci.sh              # copy the tree, then run CI
#   ./remote-ubuntu.sh --sync tools/bench.py --config configs/nuscenes.yaml
#   ./remote-ubuntu.sh --sync                            # copy and stop
#   ./remote-ubuntu.sh --shell 'nvidia-smi; df -h ~'     # one-off probe
#   ./remote-ubuntu.sh --detach scripts/run_smoke.sh
#   ./remote-ubuntu.sh --fetch runs/                     # everything trained
#   ./remote-ubuntu.sh --fetch runs/smoke                # one run
#   ./remote-ubuntu.sh --fetch runs/*/metrics.jsonl      # or just the numbers
#
# Copying is opt-in because it is the only destructive step: it is rsync
# --delete against the remote checkout, so whatever is there is made to match
# this machine. A command that only reads or evaluates should not have to think
# about that -- and datasets and run directories live outside the tree for
# exactly this reason (see MPF_REMOTE_DATA below).
#
# Commands run from the remote repository root, not from scripts/, because the
# tools here are invoked as `tools/bench.py` and `scripts/ci.sh` from the root.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Which machine has the GPU, and where its checkouts sit, is a property of the
# machine rather than of this project -- so it is read from scripts/remote.env,
# which is never committed. Typing four variables on every invocation is how
# one of them ends up wrong, and a wrong MPF_REMOTE_DIR points every command at
# a checkout that is merely *similar*, which fails much later than it should.
#
# Assignments there are written `MPF_REMOTE_DIR="${MPF_REMOTE_DIR:-...}"`, so a
# variable given on the command line still beats the file.
if [ -f "${repo_root}/scripts/remote.env" ]; then
  # shellcheck disable=SC1091
  . "${repo_root}/scripts/remote.env"
fi

# The host has no default, deliberately. A wrong one fails slowly, somewhere
# inside a command that looked like it should work; an unset one fails here,
# with a sentence that says what to set.
REMOTE_HOST="${MPF_REMOTE_HOST:-}"
REMOTE_DIR="${MPF_REMOTE_DIR:-map-pose-former}"
REMOTE_DATA="${MPF_REMOTE_DATA:-map-pose-former-data}"
REMOTE_GPU="${MPF_GPU:-auto}"

usage() {
  cat <<'EOF'
Usage: remote-ubuntu.sh [--sync] [--detach] [--shell] [command ...]

Run a command on the training host, optionally mirroring this tree there
first.

  remote-ubuntu.sh scripts/ci.sh           Run; do not copy anything.
  remote-ubuntu.sh --sync scripts/ci.sh    Copy this tree over, then run.
  remote-ubuntu.sh --sync                  Copy this tree over and stop.
  remote-ubuntu.sh --shell 'nvidia-smi'    Run shell text rather than argv.

Options:
  --fetch P.. Copy paths back from the host, merging rather than mirroring.
              --fetch runs/ brings everything; naming a run or a glob brings
              less, which is usually what is wanted -- runs/ is hundreds of
              megabytes and most of it is optimizer state nobody reads. Quote a
              glob to have the host expand it.
  --sync      rsync --delete this working tree to the host. Anything edited
              only on the host, inside the checkout, is lost. Runs, datasets
              and checkpoints are kept outside it and are never touched.
  --detach    Run the command under nohup, detached from this ssh session,
              logging to runs/ on the host, and return immediately. A training
              run outlives the session that started it; without this it dies
              with the connection.
  --shell     Treat the arguments as shell text, so pipes and semicolons work.
  -h, --help  Show this help.

Environment:
  Set these once per machine in scripts/remote.env, which is not committed:

      MPF_REMOTE_HOST="${MPF_REMOTE_HOST:-you@gpu-host}"
      MPF_REMOTE_DIR="${MPF_REMOTE_DIR:-path/to/checkout}"

  MPF_REMOTE_HOST  user@host        Required -- the machine with the GPU.
  MPF_REMOTE_DIR   checkout path    (default map-pose-former, under the remote
                   home directory)
  MPF_REMOTE_DATA  dataset path     (default map-pose-former-data). Absolute
                   paths are used as given; a relative one is taken from the
                   remote home directory. Datasets live outside the checkout so
                   that --sync's rsync --delete can never reach them. Exported
                   to the command as MPF_DATA_ROOT.
  MPF_GPU          GPU to pin       (default "auto": GPU 0 if it has room).
                   A name forces a specific card.
  MPF_GPU_FREE_MIB free VRAM "auto" requires (default 4500, a verification
                   run).
EOF
}

do_sync=false
as_shell=false
do_detach=false
fetch=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync)    do_sync=true; shift ;;
    --fetch)   shift; while [[ $# -gt 0 && "$1" != --* ]]; do fetch+=("$1"); shift; done ;;
    --shell)   as_shell=true; shift ;;
    --detach)  do_detach=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --)        shift; break ;;
    *)         break ;;
  esac
done

if [[ $# -eq 0 ]] && [[ "$do_sync" == false ]] && [[ ${#fetch[@]} -eq 0 ]]; then
  echo "Nothing to do: give a command, --sync to place the tree, or --fetch." >&2
  usage >&2
  exit 1
fi

# Checked after --help, so the help text is readable without a host set.
if [ -z "${REMOTE_HOST}" ]; then
  echo "set MPF_REMOTE_HOST to the machine with the GPU, then re-run:" >&2
  echo "  export MPF_REMOTE_HOST=you@gpu-host" >&2
  exit 2
fi

# Pulling back is the opposite direction and deliberately not the opposite
# flag. --sync mirrors, so it deletes; --fetch merges, because this machine
# has runs of its own and a mirror would remove them.
#
# It takes paths rather than assuming runs/. That directory holds every
# checkpoint of every experiment, most of it last.pt and optimizer state that
# nobody reads twice, so pulling all of it is usually a slow way to get one
# file. --fetch runs/ still works when that is what you want.
if [[ ${#fetch[@]} -gt 0 ]]; then
  for want in "${fetch[@]}"; do
    echo "Fetching ${REMOTE_HOST}:${REMOTE_DIR}/${want} -> ${repo_root}/"
    # --relative keeps the remote path, so runs/distilled lands in
    # runs/distilled rather than in the repository root.
    # No --info=stats1: macOS still ships rsync 2.6.9, which does not have it.
    rsync -az --relative \
      "${REMOTE_HOST}:${REMOTE_DIR}/./${want}" "${repo_root}/" \
      || echo "  nothing matched ${want}" >&2
  done
fi

if [[ "$do_sync" == true ]]; then
  echo "Syncing ${repo_root}/ -> ${REMOTE_HOST}:${REMOTE_DIR}/"
  # .git stays here, which is the source of truth for history. runs/
  # and build/ are excluded in both directions: --delete would otherwise wipe
  # the checkpoints of whatever is training on the host right now, and the
  # TensorRT engines, which take minutes to compile and exist only there.
  #
  # cache/ is on that list for the same reason. It is 971 MiB, takes 45
  # minutes to build and exists nowhere else, so a --sync of an unrelated
  # one-line edit would delete it and every run configured with
  # data.cache_dir=cache would die at startup. Anything generated on the host
  # belongs in this list, or outside the checkout.
  ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR} ${REMOTE_DATA}"
  rsync -az --delete \
    --exclude '.git/' \
    --exclude 'env/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude 'runs/' \
    --exclude 'build/' \
    --exclude 'cache/' \
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
# about which card. scripts/pick_gpu.sh resolves GPU 0 to its UUID, which
# survives a reorder and a reboot, and the run is pinned to it.
#
# The default is `auto`, which asks scripts/pick_gpu.sh for GPU 0's UUID and
# gets nothing back when the card has no room; that script carries the reason
# for both halves of it. When nothing comes back the variable is left unset
# rather than set empty: an empty CUDA_VISIBLE_DEVICES hides every GPU, and a
# run that quietly fell back to CPU is worse than one that never started.
#
# Datasets live outside the checkout so that --sync's rsync --delete can never
# reach them. An absolute MPF_REMOTE_DATA is used as given; a relative one is
# read from the remote home directory.
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
