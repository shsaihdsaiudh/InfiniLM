#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
infini_root=${INFINI_ROOT:-"$HOME/.infini-dsv4"}
python_bin=${PYTHON_BIN:-"$repo_root/.venv/bin/python"}
infini_core_python=${INFINICORE_PYTHON:-"$repo_root/../InfiniCore/python"}
torch_lib=$(
    "$python_bin" -c \
        'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'
)

export LD_LIBRARY_PATH="$infini_root/lib:$torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo_root/python:$infini_core_python${PYTHONPATH:+:$PYTHONPATH}"

exec "$python_bin" "$repo_root/dev_dsv4/model_native_smoke.py" "$@"
