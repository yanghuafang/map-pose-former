#!/bin/bash

# download_argoverse2.sh -- Argoverse 2, for the generalization test.
#
#   ./scripts/download_argoverse2.sh          # 6.4 GB: the val split, never trained on
#   ./scripts/download_argoverse2.sh --full   # ...and train and test, 62 GB
#   ./scripts/download_argoverse2.sh --force  # discard what is there and refetch
#
# Re-running is how a download finishes: without --force every completed file is
# skipped and every partial one continues from where it stopped. Over this path
# that matters -- see lib.sh on why a single connection to S3 is unusable.
#
# **Motion Forecasting, not Sensor.** Each scenario in it carries a vector map
# -- lane segments with their boundaries, and crosswalks -- alongside the ego
# trajectory, and no imagery at all. That is exactly what this project consumes,
# and it is 62 GB against the Sensor split's 900.
#
# The map also carries something nuScenes does not: `mark_type` on each lane
# boundary, solid against dashed. A dashed line's stripe *ends* are along-track
# evidence that a single continuous polyline throws away -- see the observability
# table in mapposeformer/data/classes.py. Whether storing them helps is a real
# experiment, and this is the only dataset here that can run it.
#
# The default is the val split alone, because Argoverse 2's role in the roadmap
# is to be the set that is never trained on. Downloading train invites using it.
#
# Public S3, no account and no credentials, unlike nuScenes. The
# licence is CC BY-NC-SA 4.0: non-commercial, which this is.

set -euo pipefail
# Remember the name before the cd: $0 is relative to where the caller
# stood, and --help reads this file back with sed after moving.
self="$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "${BASH_SOURCE[0]}")"
. ./lib.sh

BASE="https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting"
DEST="$(data_root)/argoverse2"

splits=(val)
while [ $# -gt 0 ]; do
  case "$1" in
    --force)   FORCE=1 ;;
    --full)    splits=(val train test) ;;
    -h|--help) usage "${self}"; exit 0 ;;
    *)         echo "unknown option $1 (try --help)" >&2; exit 1 ;;
  esac
  shift
done

urls=()
for s in "${splits[@]}"; do urls+=("${BASE}/${s}.tar"); done
download_set "${DEST}/archives" "${urls[@]}"

echo
for s in "${splits[@]}"; do
  echo "=== extracting ${s}.tar"
  tar -xf "${DEST}/archives/${s}.tar" -C "${DEST}"
done
echo
echo "done -> ${DEST}"
echo "archives kept in ${DEST}/archives; delete them if the space is wanted."
