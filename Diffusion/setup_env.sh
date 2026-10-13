#!/usr/bin/env bash
# Validate the centrally managed MPS/CPU environment.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${SCRIPT_DIR}/../RL_env/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  printf '%s\n' "Shared environment missing. Follow ../RL_env/README.md." >&2
  exit 1
fi
"${PYTHON}" -m pip check
"${PYTHON}" "${SCRIPT_DIR}/runtime.py" --check
