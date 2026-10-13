#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_ENV="${IRP_HPC_CORE_ENV:-${PACKAGE_ROOT}/../RL_env/profiles/core/RL_env}"
PYTHON_BIN="${CORE_ENV}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python environment is missing: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p \
  "${PACKAGE_ROOT}/results" \
  "${PACKAGE_ROOT}/.cache/tmp" \
  "${PACKAGE_ROOT}/.cache/matplotlib"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${PACKAGE_ROOT}/.cache/tmp"
export MPLCONFIGDIR="${PACKAGE_ROOT}/.cache/matplotlib"
export XDG_CACHE_HOME="${PACKAGE_ROOT}/.cache"
export WARP_CACHE_PATH="${PACKAGE_ROOT}/.cache/warp"
export MUJOCO_GL=egl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ACTION="${1:-run}"
CONTROLLER_MODE="${CONTROLLER_MODE:-diffusion-ik}"
TRIALS_PER_CELL="${TRIALS_PER_CELL:-2}"
ANGULAR_BINS="${ANGULAR_BINS:-360}"
RADIAL_BINS="${RADIAL_BINS:-50}"
SPAWN_RADIUS_MM="${SPAWN_RADIUS_MM:-100}"
MAX_LIVE_WORLDS="${MAX_LIVE_WORLDS:-7000}"
EXPORT_SUCCESS_DEMOS="${EXPORT_SUCCESS_DEMOS:-auto}"
SUCCESS_DEMO_MIN_FREE_GIB=40

if [[ ! "${MAX_LIVE_WORLDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_LIVE_WORLDS must be a positive integer." >&2
  exit 2
fi

ARGS=(
  run
  --diffusion "${SCREW_SCAN_DIFFUSION:-${PACKAGE_ROOT}/hpc/runs/fresh/checkpoints/diffusion.pt}"
  --config "${SCREW_SCAN_CONFIG:-${PACKAGE_ROOT}/hpc/configs/autodl_rtx_pro_6000_r100_success_selfdistill_w18000_tqc16.json}"
  --output-dir "${PACKAGE_ROOT}/results"
  --trials-per-cell "${TRIALS_PER_CELL}"
  --angular-bins "${ANGULAR_BINS}"
  --radial-bins "${RADIAL_BINS}"
  --spawn-radius-mm "${SPAWN_RADIUS_MM}"
  --max-live-worlds "${MAX_LIVE_WORLDS}"
)

case "${EXPORT_SUCCESS_DEMOS}" in
  auto|1|true|yes)
    FREE_KIB="$(df -Pk "${PACKAGE_ROOT}" | awk 'NR == 2 {print $4}')"
    REQUIRED_KIB="$((SUCCESS_DEMO_MIN_FREE_GIB * 1024 * 1024))"
    if [[ "${FREE_KIB}" =~ ^[0-9]+$ ]] && (( FREE_KIB >= REQUIRED_KIB )); then
      ARGS+=(--export-success-demos)
      echo "Successful policy-rollout export enabled (${FREE_KIB} KiB free)." >&2
    elif [[ "${EXPORT_SUCCESS_DEMOS}" == "auto" ]]; then
      echo "Successful policy-rollout export disabled: less than 40 GiB free." >&2
    else
      echo "ERROR: successful rollout export requires at least 40 GiB free." >&2
      exit 2
    fi
    ;;
  0|false|no)
    ;;
  *)
    echo "ERROR: EXPORT_SUCCESS_DEMOS must be auto, 1, or 0." >&2
    exit 2
    ;;
esac

case "${CONTROLLER_MODE}" in
  diffusion-ik)
    ;;
  diffusion-ik-tqc)
    ARGS+=(--with-tqc --tqc "${SCREW_SCAN_TQC:-${PACKAGE_ROOT}/hpc/runs/fresh/checkpoints/residual_tqc_policy.pt}")
    ;;
  *)
    echo "ERROR: CONTROLLER_MODE must be diffusion-ik or diffusion-ik-tqc" >&2
    exit 2
    ;;
esac

if [[ -n "${EVALUATION_SEED:-}" ]]; then
  ARGS+=(--seed "${EVALUATION_SEED}")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  ARGS+=(--max-steps "${MAX_STEPS}")
fi
case "${ACTION}" in
  validate)
    ARGS+=(--validate-only)
    ;;
  run)
    ;;
  *)
    echo "Usage: $0 [validate|run]" >&2
    exit 2
    ;;
esac

cd "${PACKAGE_ROOT}"
exec "${PYTHON_BIN}" \
  "${PACKAGE_ROOT}/evaluation/workspace_policy_rollouts.py" \
  "${ARGS[@]}"
