#!/bin/bash

# docs.sh -- generate the API reference from the docstrings.
#
#   ./scripts/docs.sh            # build into build/doxygen/html
#   ./scripts/docs.sh --publish  # ...and push it to the gh-pages branch
#   ./scripts/docs.sh --open   # ...and open it
#
# The prose documentation under docs/ is written by hand and is the part worth
# reading. This generates the other half: the per-function reference, which is
# useful to search and pointless to write twice.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v doxygen >/dev/null 2>&1; then
  echo "doxygen not found. brew install doxygen, or apt install doxygen." >&2
  exit 1
fi

# Doxygen will not create a nested OUTPUT_DIRECTORY itself.
mkdir -p build/doxygen
doxygen docs/doxygen/Doxyfile
out="build/doxygen/html/index.html"
echo "wrote ${out}"

if [[ "${1:-}" == "--open" ]]; then
  case "$(uname -s)" in
    Darwin) open "${out}" ;;
    *) xdg-open "${out}" >/dev/null 2>&1 || echo "open ${out} by hand" ;;
  esac
fi

# Publishing is opt-in for the same reason releasing weights is: it is public,
# and a wrong reference page is worse than none. The build above has to succeed
# first, so a broken run cannot overwrite a good site.
if [[ "${1:-}" == "--publish" ]]; then
  command -v git >/dev/null || { echo "git is not installed" >&2; exit 1; }
  root=$(git rev-parse --show-toplevel)
  sha=$(git rev-parse --short HEAD)
  work=$(mktemp -d)
  # A worktree rather than a branch switch: the source tree stays where it is,
  # so an interrupted publish cannot leave the checkout on gh-pages.
  if git show-ref --quiet refs/heads/gh-pages; then
    git worktree add -q "${work}" gh-pages
  else
    git worktree add -q --detach "${work}"
    git -C "${work}" checkout -q --orphan gh-pages
    git -C "${work}" rm -rq --cached . 2>/dev/null || true
    rm -rf "${work:?}"/* 2>/dev/null || true
  fi
  rm -rf "${work:?}"/*
  cp -R build/doxygen/html/. "${work}/"
  # Jekyll would otherwise swallow every directory Doxygen names with a leading
  # underscore, and the search index is one of them.
  touch "${work}/.nojekyll"
  git -C "${work}" add -A
  if git -C "${work}" diff --cached --quiet; then
    echo "reference unchanged; nothing to publish"
  else
    git -C "${work}" commit -q -m "docs: API reference at ${sha}"
    git -C "${work}" push -q origin gh-pages
    echo "published gh-pages at ${sha}"
  fi
  git worktree remove --force "${work}"
fi
