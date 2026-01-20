#!/bin/bash

# release.sh -- attach the trained weights to a GitHub release.
#
#   ./scripts/release.sh v0.1              # assemble and print what would go up
#   ./scripts/release.sh v0.1 --publish    # ...and actually create it
#
# Weights are not in git and should not be: every version of a binary stays in
# history forever, and nothing here needs to be diffed. A release carries them
# instead, tagged against a commit, which says the thing that matters -- these
# weights came from this code.
#
# **A dry run by default.** Publishing is public and awkward to retract once
# anyone has fetched it, so this prints the manifest and stops unless
# --publish is given.
#
# Checkpoints are fetched from the training host when they are not already
# here, so this works from a machine where nothing was trained. Each one is
# then loaded before it is copied, so a checkpoint from a checkout this code
# has outgrown fails here rather than in someone else's import. See
# docs/RELEASE.md for what each file is and how to load it.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
. scripts/env.sh && mpf_activate

TAG="${1:-}"
PUBLISH=false
[[ "${2:-}" == "--publish" ]] && PUBLISH=true
if [[ -z "${TAG}" ]]; then
  echo "Usage: scripts/release.sh <tag> [--publish]" >&2
  exit 1
fi

# Asked for now rather than at the end. Collecting the checkpoints moves
# several hundred megabytes off the training host, and finding out after that
# there is no `gh` to publish them with is a transfer nobody needed.
if ${PUBLISH}; then
  command -v gh >/dev/null || { echo "gh is not installed" >&2; exit 1; }
fi

# label:run -- what is worth publishing, and nothing else.
#
# The teacher is here because distillation cannot be reproduced without it, not
# because anyone would deploy four layers. `student_no_teacher` is here because
# it is the control: without it the distilled student's closed-loop result is a
# number with nothing to read it against, and a claim nobody can check is not
# worth publishing.
#
# INT8 ships as a graph and not as a checkpoint, and that distinction is the
# whole of it. A quantized .pt would be 1.005x the size of the one beside it
# and slower: simulated quantization wraps each `nn.Linear` rather than
# narrowing it, so the weights stay fp32 and the rounding is added to the
# arithmetic rather than replacing it. The ONNX below carries
# QuantizeLinear/DequantizeLinear pairs, which is the form a runtime folds into
# integer kernels -- so that is the artefact that is actually smaller and
# faster once built, and the one worth publishing.
RUNS=(
  teacher:teacher
  student-distilled:student_s0
  student-no-teacher:student_no_teacher_s0
  teacher-pruned:teacher_prune_keep50_finetuned
  student-pruned:student_prune_keep50_finetuned
)

say() { printf '\n=== %s\n' "$1"; }

say "collecting checkpoints"
mkdir -p build/release
paths=()
for spec in "${RUNS[@]}"; do
  label="${spec%%:*}"; run="${spec##*:}"
  src="runs/${run}/best.pt"
  if [[ ! -f "${src}" ]]; then
    echo "  ${src} not here; fetching from the training host"
    # config.yaml comes with it. The checkpoint carries its own config, so
    # nothing here needs the file -- but docs/RELEASE.md promises it beside
    # each weight, and fetching only best.pt leaves the copy below with
    # nothing to copy and no way to say so.
    ./scripts/remote-ubuntu.sh --fetch \
      "runs/${run}/best.pt" "runs/${run}/config.yaml" >/dev/null
  fi
  # A wrong checkout path on the host looks exactly like a run that was never
  # trained: rsync reports it and --fetch keeps going, so by the time the
  # absence is noticed here the reason has scrolled away. Name where it looked.
  [[ -f "${src}" ]] || {
    echo "  MISSING ${src}" >&2
    echo "  Looked on ${MPF_REMOTE_HOST:-<MPF_REMOTE_HOST unset>}" \
         "in ${MPF_REMOTE_DIR:-map-pose-former}/ -- check those before" \
         "concluding the run is gone." >&2
    exit 1
  }

  # A path that resolves is not an artefact that works. A checkout left on an
  # older architecture still has a runs/teacher, and its pickle names modules
  # this code no longer has: it copies cleanly, hashes cleanly, and fails at
  # the first load anyone runs. So open it before copying it, and print what
  # it built -- the parameter counts are in docs/RELEASE.md, so a checkpoint
  # from the wrong directory is a number that disagrees rather than a bad
  # release.
  n=$(python - "${src}" <<'PY'
import sys

from mapposeformer.checkpoint import load_checkpoint

model, _ = load_checkpoint(sys.argv[1])
print(f"{sum(p.numel() for p in model.parameters()):,}")
PY
  ) || {
    echo "  ${src} does not load with this code; nothing published" >&2
    exit 1
  }

  out="build/release/${label}.pt"
  cp "${src}" "${out}"
  # Not silently skipped: the checkpoint is self-contained, so a missing
  # config.yaml is not fatal, but it is a promise in docs/RELEASE.md that
  # this release will not be keeping.
  paths+=("${out}")
  if [[ -f "runs/${run}/config.yaml" ]]; then
    cp "runs/${run}/config.yaml" "build/release/${label}.config.yaml"
    # Uploaded, not merely staged. Copying it into build/release/ puts it in
    # the manifest, because that is a hash of the directory -- so leaving it
    # out of `paths` publishes a SHA256SUMS listing files the release does not
    # have, which is worse than not promising them at all.
    paths+=("build/release/${label}.config.yaml")
  else
    echo "  no config.yaml beside ${run}; the checkpoint carries its own" >&2
  fi
  printf '  %-22s %7s %11s params  %s\n' \
    "${label}" "$(du -h "${out}" | cut -f1)" "${n}" "${src}"
done

say "exporting ONNX for the deployable path"
# The log is kept rather than discarded. The exporter is chatty on a success --
# it warns about torchvision on every machine that has none -- so the output is
# held back and shown only when it is the explanation for something. Sending it
# to /dev/null instead reduced a missing cache to "export failed", which says
# what happened and nothing about why.
#
# Under build/ rather than build/release/, so a stray log cannot end up beside
# the artefacts and be mistaken for one.
log=build/export.log

# onnx NAME [flags...] -- export one graph, or say why it could not be made.
# A failed export is not fatal: the weights are still worth releasing on their
# own, and the reason belongs on the terminal rather than in a file nobody will
# think to open.
onnx() {
  local name="$1"; shift
  if python tools/export.py "runs/teacher/best.pt" --trunk-only "$@" \
       --out "build/release/${name}.onnx" >"${log}" 2>&1; then
    paths+=("build/release/${name}.onnx")
    printf '  %-22s %8s\n' "${name}.onnx" \
      "$(du -h "build/release/${name}.onnx" | cut -f1)"
  else
    echo "  ${name} export failed. Why:" >&2
    tail -15 "${log}" | sed 's/^/    /' >&2
  fi
}

onnx teacher-trunk
# Calibrated on train inside tools/export.py, never on the split these numbers
# are reported against.
onnx teacher-trunk-int8 --int8

say "manifest"
# Hashed from the upload list, not from a glob over the directory. The two are
# not the same set and the difference is not academic: a glob lists whatever
# happens to be staged, which is how a SHA256SUMS went out naming five
# config.yaml files that were copied there and never uploaded. A manifest for
# files a reader cannot fetch is worse than promising them nothing.
names=()
for p in "${paths[@]}"; do names+=("${p#build/release/}"); done
( cd build/release && sha256sum "${names[@]}" > SHA256SUMS && cat SHA256SUMS )
paths+=(build/release/SHA256SUMS)

if ! ${PUBLISH}; then
  say "dry run"
  echo "${#paths[@]} files would go up. Re-run with --publish to create ${TAG}."
  exit 0
fi

say "publishing ${TAG}"
gh release create "${TAG}" "${paths[@]}" \
  --title "${TAG}" --notes-file docs/RELEASE.md
