#!/bin/bash

# env.sh -- activate the project's conda environment.
#
# Sourced, never run. `conda activate` is a shell *function*, and it only exists
# after conda's profile script has been sourced -- which a non-interactive shell
# has not done. Calling it straight from a script fails with "run 'conda init'",
# which is advice for a human at a prompt and not for this.
#
# The search below exists for the same reason. `ssh host command` gets a
# non-login shell that reads no profile, so conda is absent from PATH on a
# machine where it is perfectly well installed -- which is exactly how
# scripts/remote-ubuntu.sh runs everything.
#
# Activation is best-effort. A checkout with no environment yet still runs its
# tests against whatever `python` is on PATH, and being told to run setup.sh is
# better delivered by an ImportError than by a script that refuses to start.

MPF_ENV="${MPF_ENV:-map-pose-former}"

# mpf_conda_base -- print the conda installation prefix, or nothing.
mpf_conda_base() {
  if command -v conda >/dev/null 2>&1; then
    conda info --base 2>/dev/null && return 0
  fi
  local dir
  for dir in "${CONDA_ROOT:-}" ~/miniconda3 ~/miniforge3 ~/mambaforge ~/anaconda3 \
             /opt/homebrew/Caskroom/miniconda/base /opt/conda; do
    [ -n "${dir}" ] && [ -x "${dir}/bin/conda" ] && echo "${dir}" && return 0
  done
  return 1
}

# mpf_activate -- put the project environment on PATH, if it exists.
mpf_activate() {
  local base
  base="$(mpf_conda_base)" || return 0
  [ -f "${base}/etc/profile.d/conda.sh" ] || return 0
  # shellcheck disable=SC1091
  . "${base}/etc/profile.d/conda.sh"
  conda activate "${MPF_ENV}" 2>/dev/null || return 0
}
