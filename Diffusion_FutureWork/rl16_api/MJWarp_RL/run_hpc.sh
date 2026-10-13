#!/usr/bin/env bash
# Run G6-only r22 Diffusion + native-MJWarp jobs.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
DEFAULT_CORE_ENV="${PROJECT_ROOT}/../../RL_env/profiles/core/RL_env"
RL_env="${IRP_HPC_CORE_ENV:-${DEFAULT_CORE_ENV}}"

if [[ $# -eq 0 ]]; then
  cat >&2 <<'EOF'
Usage:
  ./MJWarp_RL/run_hpc.sh PYTHON_ARGS...

Examples:
  # Direct diagnostics: one complete episode, never formal-training eligible.
  ./MJWarp_RL/run_hpc.sh -m MJWarp_RL.capacity_scan \
    --task translate --diagnostic-worlds 200 \
    --output MJWarp_RL/artifacts/translate_200_diagnostic.json
  ./MJWarp_RL/run_hpc.sh -m MJWarp_RL.capacity_scan \
    --task rotate --diagnostic-worlds 1440 \
    --output MJWarp_RL/artifacts/rotate_1440_diagnostic.json

  # Formal capacity certification is G6/task-specific (20k worlds).
  ./MJWarp_RL/run_hpc.sh -m MJWarp_RL.capacity_scan \
    --task translate --output /path/to/capacity_g6_translate.json
  ./MJWarp_RL/run_hpc.sh -m MJWarp_RL.capacity_scan \
    --task rotate --output /path/to/capacity_g6_rotate.json

  # A single-task debug launch. Formal two-task ordering is owned by
  # sequential_train_hpc.py: G6-Translation, then G6-Rotation.
  ./MJWarp_RL/run_hpc.sh -m MJWarp_RL.train \
    --task translate \
    --capacity-report /path/to/capacity_g6_translate.json \
    --output MJWarp_RL/artifacts/g6_translation_90m_seed7
  ./MJWarp_RL/run_hpc.sh -m MJWarp_RL.status_hpc \
    --run-dir MJWarp_RL/artifacts/g6_translation_90m_seed7 \
    --total-steps 90000000

r22 G6-only formal contract per task:
  root_z=0.560, initial_object_com_z=0.340
  phase=all, worlds=20000, episode_ticks=4500, total_steps=90000000
EOF
  exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  printf '%s\n' "ERROR: native MJWarp HPC training requires Linux." >&2
  exit 64
fi
case "${RL_env}" in
  /*) ;;
  *)
    printf 'ERROR: IRP_HPC_CORE_ENV must be absolute: %s\n' "${RL_env}" >&2
    exit 64
    ;;
esac
if [[ ! -d "${RL_env}" ]]; then
  printf '%s\n' \
    "ERROR: shared IRP core environment is missing: ${RL_env}" \
    "Build the sibling RL_env core profile, or set IRP_HPC_CORE_ENV." >&2
  exit 65
fi
RL_env="$(cd -- "${RL_env}" && pwd -P)"
case "${RL_env}" in
  "${PROJECT_ROOT}"|"${PROJECT_ROOT}"/*)
    printf '%s\n' \
      "ERROR: refusing a GRAB_G6-local virtual environment: ${RL_env}" \
      "HPC training must reuse the external IRP core environment." >&2
    exit 65
    ;;
esac
PYTHON="${RL_env}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  printf 'ERROR: shared environment has no Python: %s\n' "${PYTHON}" >&2
  exit 65
fi

mkdir -p \
  "${PROJECT_ROOT}/.cache/hpc/xdg" \
  "${PROJECT_ROOT}/.cache/hpc/warp" \
  "${PROJECT_ROOT}/.cache/hpc/tmp"
export IRP_HPC_CORE_ENV="${RL_env}"
export VIRTUAL_ENV="${RL_env}"
export PATH="${RL_env}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache/hpc/xdg"
export WARP_CACHE_PATH="${PROJECT_ROOT}/.cache/hpc/warp"
export TMPDIR="${PROJECT_ROOT}/.cache/hpc/tmp"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

"${PYTHON}" - "${RL_env}" <<'PY'
from importlib import metadata
from pathlib import Path
import platform
import sys

expected = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != expected:
    raise SystemExit(
        f"shared environment prefix mismatch: {sys.prefix} != {expected}"
    )
if platform.system() != "Linux":
    raise SystemExit("native MJWarp HPC training requires Linux")
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit("the core profile requires Python 3.10-3.12")
required = {
    "torch": "2.7.1",
    "mujoco": "3.10.0",
    "mujoco-warp": "3.10.0.3",
    "warp-lang": "1.14.0",
}
for package, wanted in required.items():
    actual = metadata.version(package)
    if actual.split("+", 1)[0] != wanted:
        raise SystemExit(f"{package}: expected {wanted}, found {actual}")

import torch

if str(torch.version.cuda) != "12.8":
    raise SystemExit(
        f"the shared core profile must use PyTorch CUDA 12.8, "
        f"found {torch.version.cuda}"
    )
PY

cd "${PROJECT_ROOT}"
exec "${PYTHON}" "$@"
