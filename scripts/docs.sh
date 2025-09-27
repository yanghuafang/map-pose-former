#!/bin/bash

# docs.sh -- generate the API reference from the docstrings.
#
#   ./scripts/docs.sh          # build into build/doxygen/html
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
