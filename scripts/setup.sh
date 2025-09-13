#!/bin/bash

# setup.sh -- create the conda environment this project runs in.
#
#   ./scripts/setup.sh                   # CPU torch; enough for tests and the smoke run
#   ./scripts/setup.sh --cuda            # CUDA 12.8 torch, for the Ubuntu box
#   ./scripts/setup.sh --cuda --deploy   # ...and the export/compression stack
#
# conda for the interpreter, pip for the packages. conda-forge supplies a Python
# that does not depend on what the OS shipped -- which the training box needs,
# since Ubuntu splits `ensurepip` into a package whose install wants a root
# password this script does not have. torch's own index is the only place the
# CPU and CUDA builds are both first-class.
#
# `--override-channels -c conda-forge` because Anaconda's default channels carry
# terms of service, and a script must not accept those on anyone's behalf.
#
# Doxygen is not installed here: it is a system package, not a Python one.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
. scripts/env.sh

cuda=0
deploy=0
for arg in "$@"; do
  case "${arg}" in
    --cuda)   cuda=1 ;;
    --deploy) deploy=1 ;;
    -h|--help) sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option ${arg}" >&2; exit 2 ;;
  esac
done

index="https://download.pytorch.org/whl/cpu"
if (( cuda )); then
  # cu128 covers Ampere (the A6000 is sm_86) through Blackwell. Ampere has no
  # FP8, so the quantization milestone targets INT8 and 2:4 sparsity.
  index="https://download.pytorch.org/whl/cu128"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install miniconda, then re-run this script." >&2
  exit 1
fi

# Created only if absent, so re-running updates the packages rather than
# losing the environment.
if conda env list | grep -qE "^${MPF_ENV}[[:space:]]"; then
  echo "environment ${MPF_ENV} exists; updating packages"
else
  conda create -y -n "${MPF_ENV}" --override-channels -c conda-forge \
    "python=${MPF_PYTHON:-3.12}"
fi
mpf_activate

python -m pip install --upgrade pip --quiet
python -m pip install --index-url "${index}" torch
python -m pip install -r requirements.txt

if (( deploy )); then
  # The ONNX path. onnxruntime is what checks the exported graph still
  # computes the same answer.
  python -m pip install onnx onnxscript onnxruntime
  if (( cuda )); then
    # M5 and M6. Both are Linux/CUDA wheels and have no macOS build, which is
    # why they are behind --cuda rather than behind --deploy alone.
    # nvidia-modelopt replaces pytorch-quantization, which is deprecated.
    python -m pip install tensorrt nvidia-modelopt
  else
    echo "skipping tensorrt and nvidia-modelopt: no CPU-only build exists"
  fi
fi

python - <<'PY'
import importlib.util as u

import torch

print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
    print(f"device {torch.cuda.get_device_name(0)}  capability {cap}")
have = [m for m in ("onnx", "onnxruntime", "tensorrt", "modelopt") if u.find_spec(m)]
print(f"deploy stack: {', '.join(have) if have else 'not installed (--deploy)'}")
PY
echo "environment ready: conda activate ${MPF_ENV}"
