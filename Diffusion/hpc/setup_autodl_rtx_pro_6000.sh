#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
AUTODL_DATA_ROOT="${SCREW_AUTODL_DATA_ROOT:-/root/autodl-tmp}"
MINIMUM_FREE_GIB="${SCREW_AUTODL_MIN_FREE_GIB:-110}"
MINIMUM_DRIVER="575.51.03"

# shellcheck source=hpc_env.sh
source "${SCRIPT_DIR}/hpc_env.sh"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ "$(uname -s)" != "Linux" ]]; then
  fail "this installer must run inside the Linux AutoDL instance"
fi

case "${SCRIPT_DIR}" in
  "${AUTODL_DATA_ROOT}"|"${AUTODL_DATA_ROOT}"/*)
    ;;
  *)
    fail "extract the complete hpc folder under ${AUTODL_DATA_ROOT}; current path is ${SCRIPT_DIR}"
    ;;
esac

if [[ ! "${MINIMUM_FREE_GIB}" =~ ^[0-9]+$ ]] || ((MINIMUM_FREE_GIB < 50)); then
  fail "SCREW_AUTODL_MIN_FREE_GIB must be an integer >= 50 (110 is recommended for Extreme V7 Diffusion-only)"
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi is unavailable; create an AutoDL GPU instance first"
fi

# Minimal AutoDL images can omit the GLVND EGL loader even though the NVIDIA
# driver is present.  MuJoCo imports its EGL backend before the GPU preflight,
# so install the small system runtime when this root-owned deployment can do
# so, and otherwise fail with an actionable message.
if ! ldconfig -p 2>/dev/null | grep -F "libEGL.so.1" >/dev/null; then
  if [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y libegl1
    ldconfig
  else
    fail "libEGL.so.1 is unavailable; install the Ubuntu libegl1 package"
  fi
fi

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
GPU_NAME="${GPU_NAMES%%$'\n'*}"
NORMALIZED_NAME="$(printf '%s' "${GPU_NAME}" | tr '[:lower:]' '[:upper:]' | tr -cd '[:alnum:]')"
if [[
  "${NORMALIZED_NAME}" != *RTXPRO6000* ||
  "${NORMALIZED_NAME}" != *BLACKWELL* ||
  "${NORMALIZED_NAME}" == *MAXQ*
]]; then
  fail "expected a full-power RTX PRO 6000 Blackwell (not Max-Q), got ${GPU_NAME}"
fi

DRIVER_VERSIONS="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
DRIVER_VERSION="${DRIVER_VERSIONS%%$'\n'*}"
LOWEST_VERSION="$(
  printf '%s\n%s\n' "${MINIMUM_DRIVER}" "${DRIVER_VERSION}" \
    | sort -V \
    | head -n 1
)"
if [[ "${LOWEST_VERSION}" != "${MINIMUM_DRIVER}" ]]; then
  fail "driver ${DRIVER_VERSION} is below this package's conservative deployment gate (>= ${MINIMUM_DRIVER})"
fi

AVAILABLE_KIB="$(df -Pk "${SCRIPT_DIR}" | awk 'NR==2 {print $4}')"
REQUIRED_KIB=$((MINIMUM_FREE_GIB * 1024 * 1024))
if [[ ! "${AVAILABLE_KIB}" =~ ^[0-9]+$ ]]; then
  fail "could not determine free space on the AutoDL data disk"
fi
if ((AVAILABLE_KIB < REQUIRED_KIB)); then
  AVAILABLE_GIB=$((AVAILABLE_KIB / 1024 / 1024))
  fail "only ${AVAILABLE_GIB} GiB is free; expand the AutoDL data disk and provide at least ${MINIMUM_FREE_GIB} GiB"
fi

mkdir -p \
  "${SCRIPT_DIR}/.cache/tmp" \
  "${SCRIPT_DIR}/.cache/xdg" \
  "${SCRIPT_DIR}/.cache/warp_rtx_pro_6000"

export TMPDIR="${SCRIPT_DIR}/.cache/tmp"
export XDG_CACHE_HOME="${SCRIPT_DIR}/.cache/xdg"
export WARP_CACHE_PATH="${SCRIPT_DIR}/.cache/warp_rtx_pro_6000"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
screw_use_core_env

printf '%s\n' \
  "AutoDL target: ${GPU_NAME}" \
  "Driver: ${DRIVER_VERSION}" \
  "Package: ${SCRIPT_DIR}" \
  "Free data-disk space: $((AVAILABLE_KIB / 1024 / 1024)) GiB" \
  "Shared environment: ${SCREW_HPC_ENV}" \
  "Validating the pinned CUDA 12.8/12.9 training stack without modifying it..."

bash "${SCRIPT_DIR}/setup_hpc.sh"

PYTHON="${SCREW_PYTHON}"
"${PYTHON}" "${SCRIPT_DIR}/autodl_rtx_pro_6000_preflight.py"
"${PYTHON}" "${SCRIPT_DIR}/preflight.py" \
  --mode static \
  --config "${SCRIPT_DIR}/configs/autodl_rtx_pro_6000_extreme.json"
"${PYTHON}" "${SCRIPT_DIR}/preflight.py" \
  --mode gpu \
  --config "${SCRIPT_DIR}/configs/smoke.json"

printf '%s\n' \
  "AutoDL RTX PRO 6000 Extreme V7 Diffusion-only environment is ready." \
  "Start in the background:" \
  "  cd \"${SCRIPT_DIR}\" && ./autodl_rtx_pro_6000.sh start" \
  "Follow status:" \
  "  ./autodl_rtx_pro_6000.sh status"
