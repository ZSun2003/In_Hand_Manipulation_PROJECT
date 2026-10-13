#!/usr/bin/env bash
# Stable generic name for the existing audited hpc AutoDL source package.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export SCREW_PACKAGE_INCLUDE_MODELS="${SCREW_PACKAGE_INCLUDE_MODELS:-0}"
exec "${SCRIPT_DIR}/build_autodl_package.sh" "$@"
