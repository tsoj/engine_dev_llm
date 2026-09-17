#!/usr/bin/env bash
# `uv run` with the PyTorch backend extra (cuda/rocm/cpu) for this machine.
#
#   ./run.sh train.py --run_name x      # same as: uv run --extra <backend> python train.py --run_name x
#   ./run.sh ruff check .               # any other command works too
#
# uv has no per-machine default for extras, and a plain `uv run` would replace
# the installed GPU build of torch with the default one from PyPI. The backend is
# detected automatically; override it with TORCH_BACKEND=cuda|rocm|cpu.
set -euo pipefail

backend=${TORCH_BACKEND:-}
if [[ -z "$backend" ]]; then
  if command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1; then
    backend=cuda
  elif [[ -e /dev/kfd ]]; then  # AMD ROCm kernel driver
    backend=rocm
  else
    backend=cpu
  fi
fi

if [[ $# -gt 0 && "$1" == *.py ]]; then
  set -- python "$@"
fi

exec uv run --project "$(dirname "$0")" --extra "$backend" "$@"
