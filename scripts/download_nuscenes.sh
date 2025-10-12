#!/bin/bash

# download_nuscenes.sh -- nuScenes, for the real-data milestone.
#
#   ./scripts/download_nuscenes.sh           # 1.6 GB: everything M2a needs
#   ./scripts/download_nuscenes.sh --mini    # ...and the 4 GB mini sensor split
#   ./scripts/download_nuscenes.sh --blobs   # ...and 316 GB of camera and lidar
#   ./scripts/download_nuscenes.sh --force   # discard what is there and refetch
#
# Re-running is how a download finishes: without --force every completed file is
# skipped and every partial one continues from where it stopped. --force is for
# the case where the bytes on disk are suspect, not for the case where the last
# run was interrupted.
#
# **Run this yourself.** The files are public, but nuScenes' Terms of Use are
# accepted by a person with an account -- not by a script, and not by an agent
# on that person's behalf. Downloading them is that acceptance.
#
# The default is small on purpose. This project treats perception as an input:
# detections come from the map with an error model applied and no detector is
# ever run, so the milestone needs the map and the poses and none of the imagery.
#
#   v1.0-trainval_meta.tgz            ego poses, calibration, sample tokens
#   nuScenes-map-expansion-v1.3.zip   the HD map, which is the whole point
#   can_bus.zip                       egomotion, replacing differenced ground truth
#
# The blobs matter at M2b, where a pretrained mapper -- MapTR or StreamMapNet --
# reads the 53 GB of keyframe images inside them and its output is fed in as
# detections. Until then they are 316 GB that nothing reads.

set -euo pipefail
# Remember the name before the cd: $0 is relative to where the caller
# stood, and --help reads this file back with sed after moving.
self="$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "${BASH_SOURCE[0]}")"
. ./lib.sh

BASE="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"
DEST="$(data_root)/nuscenes"

files=(v1.0-trainval_meta.tgz nuScenes-map-expansion-v1.3.zip can_bus.zip)
while [ $# -gt 0 ]; do
  case "$1" in
    --force)   FORCE=1 ;;
    --mini)    files+=(v1.0-mini.tgz) ;;
    --blobs)   for i in $(seq -w 1 10); do files+=("v1.0-trainval${i}_blobs.tgz"); done ;;
    -h|--help) usage "${self}"; exit 0 ;;
    *)         echo "unknown option $1 (try --help)" >&2; exit 1 ;;
  esac
  shift
done

urls=()
for f in "${files[@]}"; do urls+=("${BASE}/${f}"); done
download_set "${DEST}/archives" "${urls[@]}"

echo
for f in "${files[@]}"; do
  echo "=== extracting ${f}"
  case "${f}" in
    # The map expansion unpacks bare .json files that belong under maps/, which
    # is where the devkit looks for them; the tarballs carry their own paths.
    *.zip) unzip -q -o "${DEST}/archives/${f}" -d "${DEST}/maps" ;;
    *.tgz) tar -xzf "${DEST}/archives/${f}" -C "${DEST}" ;;
  esac
done
echo
echo "done -> ${DEST}"
echo "archives kept in ${DEST}/archives; delete them if the space is wanted."
