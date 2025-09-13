#!/bin/bash

# ci.sh -- every gate, in the order that fails fastest.
#
#   ./scripts/ci.sh            # format, lint, tests
#   ./scripts/ci.sh --smoke    # ...and the end-to-end smoke run (a few minutes)
#
# ruff is optional: if it is not installed the style gates are skipped with a
# notice rather than a failure, so a fresh checkout is never blocked on a tool
# that has nothing to do with localization.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
. scripts/env.sh && mpf_activate

step() { printf '\n=== %s\n' "$1"; }

if command -v ruff >/dev/null 2>&1; then
  step "format"; ruff format --check .
  step "lint";   ruff check .
else
  echo "ruff not installed; skipping format and lint (pip install ruff)"
fi

step "tests"
python -m pytest tests -q

if [[ "${1:-}" == "--smoke" ]]; then
  step "smoke"
  ./scripts/run_smoke.sh runs/ci-smoke
fi

printf '\nall gates passed\n'
