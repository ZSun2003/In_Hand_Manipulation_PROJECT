#!/usr/bin/env bash
# Run an hpc Python entry point with the shared IRP core environment.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ $# -eq 0 ]]; then
  printf '%s\n' \
    "Usage: ./run_hpc.sh PYTHON_ARGS..." \
    "Examples:" \
    "  ./run_hpc.sh preflight.py --mode static --config configs/smoke.json" \
    "  ./run_hpc.sh status_hpc.py --run-dir runs/cx3_l40s_full" >&2
  exit 2
fi

exec "${SCRIPT_DIR}/hpc_python.sh" "$@"

