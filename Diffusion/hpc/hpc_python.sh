#!/usr/bin/env bash
# Execute the shared IRP core Python without installing into it.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=hpc_env.sh
source "${SCRIPT_DIR}/hpc_env.sh"
screw_use_core_env

cd "${SCRIPT_DIR}"
exec "${SCREW_PYTHON}" "$@"
