#!/usr/bin/env bash
# Build the isolated source-only HX1 A100 14,000-world Diffusion package.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
STAMP="${SCREW_PACKAGE_STAMP:-$(date -u '+%Y%m%d')}"

export SCREW_PACKAGE_INCLUDE_MODELS=0
export SCREW_PACKAGE_REQUIRE_MODELS=0
export SCREW_PACKAGE_ARCHIVE_NAME="hpc_HX1_A100_W14000_DIFFUSION_ONLY_EP4000_BLEND100_${STAMP}.tar.gz"

exec "${SCRIPT_DIR}/build_autodl_package.sh" "$@"
