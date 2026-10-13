#!/usr/bin/env bash
# Validate and register the shared IRP core environment for SCREW.
#
# This compatibility entry point intentionally never creates a venv and never
# runs pip install.  The central environment is built once by IRP/RL_env and
# is read-only from the perspective of every project.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HPC_ROOT="${SCRIPT_DIR}"
REQUIREMENTS="${HPC_ROOT}/requirements-hpc.txt"

# shellcheck source=hpc_env.sh
source "${SCRIPT_DIR}/hpc_env.sh"

if [[ "$(uname -s)" != "Linux" ]]; then
  printf '%s\n' \
    "ERROR: the shared CUDA environment can only be used on Linux." \
    "Do not create or copy it from macOS."
  exit 64
fi
if [[ ! -f "${REQUIREMENTS}" ]]; then
  printf 'ERROR: missing %s\n' "${REQUIREMENTS}" >&2
  exit 66
fi

screw_use_core_env
PYTHON="${SCREW_PYTHON}"

mkdir -p \
  "${HPC_ROOT}/.cache/warp" \
  "${HPC_ROOT}/.cache/xdg" \
  "${HPC_ROOT}/checkpoints" \
  "${HPC_ROOT}/datasets" \
  "${HPC_ROOT}/logs" \
  "${HPC_ROOT}/outputs" \
  "${HPC_ROOT}/runs"

export XDG_CACHE_HOME="${HPC_ROOT}/.cache/xdg"
export WARP_CACHE_PATH="${HPC_ROOT}/.cache/warp"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

"${PYTHON}" - <<'PY'
from importlib import metadata
from pathlib import Path
import os
import platform
import sys

expected = Path(os.environ["SCREW_HPC_ENV"]).resolve()
prefix = Path(sys.prefix).resolve()
if prefix != expected:
    raise SystemExit(
        f"shared environment prefix mismatch: {prefix} != {expected}"
    )
if platform.system() != "Linux":
    raise SystemExit("shared HPC environment is not running on Linux")
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"Python {sys.version_info.major}.{sys.version_info.minor} is "
        "unsupported; the core profile requires Python 3.10-3.12"
    )

import torch

if torch.version.cuda is None:
    raise SystemExit("the shared PyTorch build has no CUDA support")
if str(torch.version.cuda) != "12.8":
    raise SystemExit(
        f"the shared core profile must use PyTorch CUDA 12.8, "
        f"got {torch.version.cuda}"
    )
required_versions = {
    "torch": "2.7.1",
    "mujoco": "3.10.0",
    "mujoco-warp": "3.10.0.3",
    "warp-lang": "1.14.0",
}
for package, required in required_versions.items():
    installed = metadata.version(package)
    normalized = installed.split("+", 1)[0]
    if normalized != required:
        raise SystemExit(
            f"{package} must be exactly {required}, got {installed}"
        )

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda_build", torch.version.cuda)
print("cuda_visible_now", torch.cuda.is_available())
print("environment", prefix)
print("environment_mode", "shared-read-only")
PY

# Dependency validation is read-only.  In particular, this script contains no
# pip install command and cannot update the shared environment.
"${PYTHON}" -m pip check

{
  printf 'validated_utc='
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  printf 'host='
  hostname
  printf 'kernel='
  uname -srmo
  printf 'shared_environment=%s\n' "${SCREW_HPC_ENV}"
  printf 'environment_mode=shared-read-only\n'
  "${PYTHON}" -m pip freeze
} >"${HPC_ROOT}/environment-manifest.txt"

if [[ -f "${HPC_ROOT}/preflight.py" && -f "${HPC_ROOT}/configs/smoke.json" ]]; then
  "${PYTHON}" "${HPC_ROOT}/preflight.py" \
    --mode static \
    --config "${HPC_ROOT}/configs/smoke.json"
else
  printf '%s\n' \
    "NOTE: core preflight.py/configs are not present; static preflight skipped."
fi

printf '%s\n' \
  "Shared HPC environment validated without modification." \
  "Environment: ${SCREW_HPC_ENV}" \
  "Next: cd \"${HPC_ROOT}\" and start the required scheduler/AutoDL launcher."
