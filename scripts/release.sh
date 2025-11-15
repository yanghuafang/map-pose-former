#!/bin/bash

# release.sh -- attach the trained artifacts to a GitHub release.
#
#   ./scripts/release.sh v0.1              # assemble and show what would go up
#   ./scripts/release.sh v0.1 --publish    # ...and actually create it
#
# Weights are not in git and should not be: every version of a binary stays in
# history forever, and the teacher alone is 99 MB against GitHub's 100 MB
# per-file limit. A release carries them instead, tagged against a commit, which
# says the thing that matters -- these weights came from this code.
#
# **A dry run by default.** Publishing is public, and a release is awkward to
# retract once anyone has fetched it. This prints the manifest and stops unless
# --publish is given.
#
# Artifacts are fetched from the training box if they are not already here, so
# this works from the laptop where nothing was trained. See docs/RELEASE.md for
# what each file is and how to load it.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The ONNX export needs torch, so this needs the project environment for the
# same reason scripts/ci.sh does.
. scripts/env.sh && mpf_activate

TAG="${1:-}"
PUBLISH=false
[[ "${2:-}" == "--publish" ]] && PUBLISH=true

if [[ -z "${TAG}" ]]; then
  echo "Usage: scripts/release.sh <tag> [--publish]" >&2
  exit 1
fi

# label:run -- the four worth publishing. The teacher is here because
# distillation cannot be reproduced without it, not because anyone would deploy
# 25.8M parameters.
RUNS=(student:m1_base distilled:distilled pruned25:p25 teacher:teacher)

say() { printf '\n=== %s\n' "$1"; }

say "collecting checkpoints"
paths=()
for spec in "${RUNS[@]}"; do paths+=("runs/${spec##*:}/best.pt"); done

# Every time, not only when a file is absent. rsync moves what differs and
# nothing when the copies agree, so this is free in the common case -- and the
# case it is not free in is the one worth catching: a checkpoint retrained on
# the box after the laptop's copy arrived, which a missing-files check cannot
# see and which would publish the older weights without saying so.
./scripts/remote-ubuntu.sh --fetch "${paths[@]}" ||
  echo "  box unreachable; publishing the local copies" >&2

missing=()
for path in "${paths[@]}"; do
  [[ -f "${path}" ]] || missing+=("${path}")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "refusing: no checkpoint at ${missing[*]}" >&2
  exit 1
fi

say "exporting ONNX"
mkdir -p build/release
for spec in "${RUNS[@]}"; do
  label="${spec%%:*}" run="${spec##*:}"
  if [[ build/release/${label}.onnx -nt runs/${run}/best.pt ]]; then
    echo "  ${label}.onnx is current"
  else
    # No `|| true`: an export that fails must not leave a stale file in the
    # manifest and a release carrying last week's graph.
    python tools/export.py "runs/${run}/best.pt" \
      --out "build/release/${label}.onnx" | grep -E "^runs"
  fi
done

# Every checkpoint on disk is called best.pt, and a release asset's name comes
# from the filename -- `file#label` sets only the label GitHub displays. Four
# files called best.pt collide on upload, so they are staged under the name they
# should carry.
say "staging"
files=()
for spec in "${RUNS[@]}"; do
  label="${spec%%:*}" run="${spec##*:}"
  cp -f "runs/${run}/best.pt" "build/release/${label}.pt"
  files+=("build/release/${label}.pt" "build/release/${label}.onnx")
done

say "manifest"
total=0
for path in "${files[@]}"; do
  size=$(du -m "${path}" | cut -f1)
  total=$((total + size))
  printf "  %-40s %5s MB\n" "$(basename "${path}")" "${size}"
done
printf "  %-40s %5s MB\n" "total" "${total}"

# 2 GB per file is the release limit, and nothing here is close -- but a check
# that never fires is still the one that catches the day someone publishes a
# teacher ten times this size.
for path in "${files[@]}"; do
  size=$(du -m "${path}" | cut -f1)
  if [[ "${size}" -gt 2000 ]]; then
    echo "refusing: ${path} is ${size} MB, over the 2 GB per-file limit" >&2
    exit 1
  fi
done

if [[ "${PUBLISH}" != true ]]; then
  say "dry run"
  echo "Nothing published. Re-run with --publish to create the release:"
  echo "  ./scripts/release.sh ${TAG} --publish"
  exit 0
fi

# A release points at a commit, and the point of tagging one is to say that
# these weights came from this code. Publishing while HEAD is unpushed would
# tag whatever the remote happens to have instead.
head=$(git rev-parse HEAD)
branch=$(git rev-parse --abbrev-ref HEAD)
if ! git merge-base --is-ancestor "${head}" "origin/${branch}" 2>/dev/null; then
  git fetch -q origin "${branch}" 2>/dev/null || true
  if ! git merge-base --is-ancestor "${head}" "origin/${branch}" 2>/dev/null; then
    echo "refusing: HEAD is not on origin/${branch}. Push first, or the" >&2
    echo "release would tag code that did not produce these weights." >&2
    exit 1
  fi
fi

say "publishing ${TAG} at ${head:0:7}"
# `gh release view` looks a release up by tag, and a draft has no tag until it
# is published -- so a half-made draft from a failed run is invisible to it and
# `create` collides all over again. The API lists drafts; that is what to ask.
existing=$(gh api "repos/{owner}/{repo}/releases" --jq \
  ".[] | select(.tag_name==\"${TAG}\") | .id" 2>/dev/null | head -1)
if [[ -n "${existing}" ]]; then
  echo "${TAG} already exists (id ${existing}); uploading into it"
  gh release upload "${TAG}" --clobber "${files[@]}"
  exit 0
fi
gh release create "${TAG}" \
  --target "${head}" \
  --title "map-pose-former ${TAG}" \
  --notes-file docs/RELEASE.md \
  "${files[@]}"
