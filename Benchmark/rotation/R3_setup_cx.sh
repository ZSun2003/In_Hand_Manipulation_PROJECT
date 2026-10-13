#!/usr/bin/env bash
# Install the shared HPC runtime; creates no environment inside this gripper.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
exec bash "$REPO_ROOT/RL_env/build.sh" --profile local "$@"
