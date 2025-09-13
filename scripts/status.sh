#!/bin/bash

# status.sh -- what is running on the training box, and how far along.
#
#   ./scripts/status.sh
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

REMOTE_HOST="${MPF_REMOTE_HOST:-yanghuafang@192.168.10.13}"
REMOTE_DIR="${MPF_REMOTE_DIR:-study-projects/map-pose-former}"

ssh "${REMOTE_HOST}" bash -s <<'REMOTE'
set -u
cd "${HOME}/study-projects/map-pose-former" 2>/dev/null || exit 0
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
[ "$n" -gt 1 ] && echo "  ** $n runs share one GPU -- all of them will be slow **"

echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader | grep -i a6000 | sed 's/^/  /'

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
  # The trainer prints "... | 30m48s elapsed, 58m00s left"; take the last log
  # line that has one, since an eval line in between has none.
  eta=$(grep -o '| [0-9hms]* elapsed, [0-9hms]* left' "${d}.log" 2>/dev/null | tail -1)
  printf "  %-22s step %-7s %-34s (log %ss old)\n" \
    "$(basename "$d")" "$step" "${eta:-}" "$age"
done
[ "$found" -eq 0 ] && echo "  nothing has logged in the last 15 minutes"
REMOTE
