#!/usr/bin/env bash
# Run Gripper3 GRAB r21 control-height/contact-timing jobs.
set -Eeuo pipefail

HPC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$HPC_ROOT/.." && pwd -P)"
IRP_RL_ENV_ROOT="${IRP_RL_ENV_ROOT:-$(cd -- "$PROJECT_ROOT/../.." && pwd -P)/RL_env}"
PROFILE_ENV="${IRP_HPC_CORE_ENV:-$IRP_RL_ENV_ROOT/profiles/core/RL_env}"
PYTHON="${IRP_HPC_CORE_PYTHON:-$PROFILE_ENV/bin/python}"

usage() {
  cat <<'EOF'
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

  # Formal capacity certification is task-specific (20k worlds).
  ./MJWarp_RL/run_hpc.sh -m MJWarp_RL.capacity_scan \
    --task translate --output /path/to/capacity_g3_translate.json
  # Run this in the other gripper to create the required common report:
  ./MJWarp_RL/run_hpc.sh -m MJWarp_RL.capacity_scan \
    --task translate \
    --common-with /path/to/capacity_g3_translate.json \
    --output /path/to/capacity_common_translate.json

  # A single-task debug launch. Formal four-task ordering is owned by
  # sequential_train_hpc.py: G3-T, G6-T, G3-R, G6-R.
  ./MJWarp_RL/run_hpc.sh -m MJWarp_RL.train \
    --task translate \
    --capacity-report /path/to/capacity_common_translate.json \
    --output MJWarp_RL/artifacts/g3_translation_60m_seed7
  ./MJWarp_RL/run_hpc.sh -m MJWarp_RL.status_hpc \
    --run-dir MJWarp_RL/artifacts/g3_translation_60m_seed7 \
    --total-steps 60000000

r21 formal contract per task:
  phase=all, worlds=20000, episode_ticks=3000, total_steps=60000000

The default runtime is:
  ../../RL_env/profiles/core/RL_env

Override only when intentionally testing another central core profile:
  IRP_HPC_CORE_ENV=/absolute/profile/RL_env
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi
case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  printf 'ERROR: central GRAB profile is missing: %s\n' "$PROFILE_ENV" >&2
  printf 'Build it once with: bash %s/build.sh --profile core\n' \
    "$IRP_RL_ENV_ROOT" >&2
  exit 65
fi

export IRP_RL_ENV_ROOT
export IRP_HPC_PROFILE="core"
export VENV_DIR="$PROFILE_ENV"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
exec "$PYTHON" "$@"
