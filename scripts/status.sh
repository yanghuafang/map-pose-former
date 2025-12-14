#!/bin/bash

# status.sh -- what is running on the training host, and how far along.
#
#   MPF_REMOTE_HOST=you@gpu-host ./scripts/status.sh
#
# Exists because improvising `pgrep`/`pkill` from a shell whose own command
# line contains the pattern is a trap: the pattern matches the querying shell,
# `pkill` kills itself, the targets survive, and the follow-up check reports
# success. That mistake left three runs competing for one GPU for five hours,
# each writing into the same directory.
#
# Everything here matches on the *python interpreter's* argv via ps, and the
# pattern is assembled at runtime so it never appears literally in this
# script's own command line.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
. scripts/env.sh

# Same contract as remote-ubuntu.sh: no default host, because a wrong one fails
# slowly inside a command that looked like it should work.
# Same file remote-ubuntu.sh reads, for the same reason: the host and the
# checkout belong to the machine, not to the repository.
if [ -f scripts/remote.env ]; then
  # shellcheck disable=SC1091
  . scripts/remote.env
fi

REMOTE_HOST="${MPF_REMOTE_HOST:-}"
REMOTE_DIR="${MPF_REMOTE_DIR:-map-pose-former}"
if [ -z "${REMOTE_HOST}" ]; then
  echo "set MPF_REMOTE_HOST to the machine with the GPU, then re-run:" >&2
  echo "  export MPF_REMOTE_HOST=you@gpu-host" >&2
  exit 2
fi

# The checkout path is passed as an argument rather than interpolated, so the
# heredoc stays unexpanded and nothing here depends on the local shell.
ssh "${REMOTE_HOST}" bash -s "${REMOTE_DIR}" <<'REMOTE'
set -u
cd "${HOME}/$1" 2>/dev/null || cd "$1" 2>/dev/null || {
  echo "no checkout at $1" >&2; exit 1; }
pat="tools/tra""in.py"          # split so it cannot match this shell

echo "=== training runs ==="
# A run is a *parent*: its dataloader workers are forks sharing its argv, so
# counting every match reports 33 runs where there is one.
mapfile -t pids < <(ps -eo pid=,cmd= | grep -F "$pat" | grep -v grep | awk '{print $1}')
n=0
for pid in "${pids[@]:-}"; do
  [ -z "$pid" ] && continue
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  # Skip it if its parent is also a matching process -- that makes it a worker.
  if printf '%s\n' "${pids[@]}" | grep -qx "$ppid"; then continue; fi
  n=$((n + 1))
  etime=$(ps -o etime= -p "$pid" | tr -d ' ')
  cmd=$(ps -o cmd= -p "$pid")
  kids=$(pgrep -P "$pid" | wc -l)
  out=$(printf '%s' "$cmd" | grep -o 'out_dir=[^ ]*' || echo "<default>")
  printf "  pid %-8s up %-10s %-28s %s workers\n" "$pid" "$etime" "$out" "$kids"
done
[ "$n" -eq 0 ] && echo "  none"
# Training here is dataloader-bound, so runs sharing a host do not merely share
# the GPU -- they compete for the cores that feed it.
[ "$n" -gt 1 ] && echo "  ** $n runs share this host -- all of them will be slow **"

echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader | sed 's/^/  /'

echo "=== progress ==="
shopt -s nullglob
found=0
for f in runs/*/metrics.jsonl; do
  d=$(dirname "$f")
  line=$(grep '"train"' "$f" 2>/dev/null | tail -1)
  [ -z "$line" ] && continue
  found=1
  step=$(printf '%s' "$line" | sed 's/.*"step": *\([0-9]*\).*/\1/')
  age=$(( $(date +%s) - $(stat -c %Y "$f") ))
  [ "$age" -gt 900 ] && continue        # finished or abandoned; not status
  # The ETA comes from the same train line already in hand: the trainer writes
  # `eta_min` into metrics.jsonl. Reading it here rather than from a sibling
  # .log avoids depending on a console format and on a filename -- run_arms.sh
  # writes runs/<name>_<timestamp>.log, not runs/<name>.log.
  eta=$(printf '%s' "$line" | sed -n 's/.*"eta_min": *\([0-9.eE+-]*\).*/\1/p' | awk '{printf "eta %.0f min", $1}')
  printf "  %-22s step %-7s %-34s (log %ss old)\n" \
    "$(basename "$d")" "$step" "${eta:-}" "$age"
done
[ "$found" -eq 0 ] && echo "  nothing has logged in the last 15 minutes"
REMOTE
