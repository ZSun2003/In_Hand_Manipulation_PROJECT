#!/usr/bin/env bash

# Shared by PBS and Slurm scripts. This file intentionally never searches a
# parent directory: jobs must be submitted while the current directory is the
# self-contained hpc folder.
SCREW_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=../hpc_env.sh
source "${SCREW_COMMON_DIR}/../hpc_env.sh"

screw_load_python_module() {
  # A venv created from an EasyBuild Python still needs that same module's
  # runtime libraries (notably libffi) inside every PBS allocation.
  if ! type module >/dev/null 2>&1; then
    if [[ -r /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
    elif [[ -r /usr/share/Modules/init/bash ]]; then
      # shellcheck disable=SC1091
      source /usr/share/Modules/init/bash
    fi
  fi
  if ! type module >/dev/null 2>&1; then
    printf '%s\n' \
      "ERROR: the CX3 environment-modules command is unavailable." >&2
    return 69
  fi

  local python_module
  python_module="${SCREW_PYTHON_MODULE:-Python/3.11.5-GCCcore-13.2.0}"
  module purge
  # CX3 exposes its production module tree through tools/prod.  HX1 already
  # exposes production modules directly and does not provide that module.
  if module is-avail tools/prod >/dev/null 2>&1; then
    module load tools/prod
  fi
  module load "${python_module}"
  export SCREW_PYTHON_MODULE="${python_module}"
  printf 'Loaded Python module: %s\n' "${python_module}"
}

screw_start_allocation_log() {
  if [[ "${SCREW_ALLOCATION_LOG_ACTIVE:-0}" -eq 1 ]]; then
    return 0
  fi
  if ! command -v tee >/dev/null 2>&1; then
    printf '%s\n' \
      "WARNING: tee is unavailable; allocation output will only use scheduler logs." \
      >&2
    return 0
  fi

  local job_tag
  job_tag="$(screw_job_tag)"
  SCREW_ALLOCATION_LOG="${SCREW_HPC_ROOT}/logs/allocation-${job_tag}.log"
  export SCREW_ALLOCATION_LOG
  if ! : >>"${SCREW_ALLOCATION_LOG}"; then
    printf 'WARNING: cannot append allocation log: %s\n' \
      "${SCREW_ALLOCATION_LOG}" >&2
    return 0
  fi

  SCREW_ALLOCATION_LOG_ACTIVE=1
  export SCREW_ALLOCATION_LOG_ACTIVE
  # tee writes one copy to the persistent append-only log and one copy back to
  # the wrapper's original stdout, which PBS/Slurm continue to capture.
  exec > >(tee -a -- "${SCREW_ALLOCATION_LOG}") 2>&1
  printf 'Persistent allocation log: %s\n' "${SCREW_ALLOCATION_LOG}"
}

screw_hpc_init() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    printf '%s\n' "ERROR: HPC jobs require Linux." >&2
    return 64
  fi
  screw_load_python_module

  local submit_dir
  submit_dir="${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-${PWD}}}"
  SCREW_HPC_ROOT="$(cd -- "${submit_dir}" && pwd -P)"
  export SCREW_HPC_ROOT

  if [[ ! -f "${SCREW_HPC_ROOT}/requirements-hpc.txt" ]]; then
    printf '%s\n' \
      "ERROR: submit the job from inside the hpc directory." \
      "Expected: ${SCREW_HPC_ROOT}/requirements-hpc.txt" >&2
    return 66
  fi

  cd "${SCREW_HPC_ROOT}"
  mkdir -p \
    "${SCREW_HPC_ROOT}/.cache/warp" \
    "${SCREW_HPC_ROOT}/.cache/xdg" \
    "${SCREW_HPC_ROOT}/logs" \
    "${SCREW_HPC_ROOT}/runs"
  screw_start_allocation_log

  screw_use_core_env

  export PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE=1
  export XDG_CACHE_HOME="${SCREW_HPC_ROOT}/.cache/xdg"
  export WARP_CACHE_PATH="${SCREW_HPC_ROOT}/.cache/warp"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  local allocated_cpus
  allocated_cpus="${PBS_NCPUS:-${SLURM_CPUS_PER_TASK:-4}}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${allocated_cpus}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${allocated_cpus}}"

  "${SCREW_PYTHON}" - <<'PY'
from pathlib import Path
import os
import platform
import sys

import torch

root = Path(os.environ["SCREW_HPC_ROOT"]).resolve()
expected = Path(os.environ["SCREW_HPC_ENV"]).resolve()
prefix = Path(sys.prefix).resolve()
if prefix != expected:
    raise SystemExit(
        f"unexpected shared environment: {prefix}; expected {expected}"
    )
try:
    prefix.relative_to(root)
except ValueError:
    pass
else:
    raise SystemExit(f"shared environment must be outside hpc: {prefix}")
if platform.system() != "Linux":
    raise SystemExit("HPC virtual environment is not running on Linux")
if torch.version.cuda is None:
    raise SystemExit("PyTorch is not a CUDA build")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible inside this scheduled job")
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda_build", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
PY

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    printf '%s\n' "ERROR: nvidia-smi is unavailable in this GPU job." >&2
    return 69
  fi
}

screw_require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    printf 'ERROR: required file is missing: %s\n' "${path}" >&2
    return 66
  fi
}

screw_job_tag() {
  local raw="${PBS_JOBID:-${SLURM_JOB_ID:-manual}}"
  printf '%s' "${raw}" | tr -c 'A-Za-z0-9_.-' '_'
}

screw_record_job() {
  local run_dir="$1"
  local state="$2"
  local raw_exit_code="${3:-}"
  local message="${4:-}"
  local args=(
    "${SCREW_HPC_ROOT}/job_status.py"
    --run-dir "${run_dir}"
    --state "${state}"
    --message "${message}"
  )
  if [[ -n "${raw_exit_code}" ]]; then
    args+=(--raw-exit-code "${raw_exit_code}")
  fi
  if ! "${SCREW_PYTHON}" "${args[@]}"; then
    printf 'WARNING: could not write scheduler job status for %s\n' \
      "${run_dir}" >&2
    return 0
  fi
  case "${state}" in
    completed|interrupted_resumable|failed)
      SCREW_JOB_TERMINAL_RECORDED=1
      ;;
  esac
}

screw_stop_live_monitor() {
  if [[ -n "${SCREW_SYSTEM_MONITOR_PID:-}" ]]; then
    kill "${SCREW_SYSTEM_MONITOR_PID}" 2>/dev/null || true
    wait "${SCREW_SYSTEM_MONITOR_PID}" 2>/dev/null || true
    SCREW_SYSTEM_MONITOR_PID=""
  fi
}

screw_collect_diagnostics() {
  local exit_code="$1"
  if [[ -z "${SCREW_MONITOR_RUN_DIR:-}" ]]; then
    return 0
  fi
  local collector="${SCREW_HPC_ROOT}/collect_diagnostics.py"
  if [[ ! -f "${collector}" ]]; then
    printf 'WARNING: diagnostic collector is missing: %s\n' \
      "${collector}" >&2
    return 0
  fi
  local args=(
    "${collector}"
    --run-dir "${SCREW_MONITOR_RUN_DIR}"
    --exit-code "${exit_code}"
    --job-id "${PBS_JOBID:-${SLURM_JOB_ID:-manual}}"
  )
  if [[ -n "${SCREW_ALLOCATION_LOG:-}" ]]; then
    args+=(--allocation-log "${SCREW_ALLOCATION_LOG}")
  fi
  if ! "${SCREW_PYTHON}" "${args[@]}"; then
    printf 'WARNING: could not create diagnostics for %s\n' \
      "${SCREW_MONITOR_RUN_DIR}" >&2
  fi
  return 0
}

screw_hpc_exit_trap() {
  local exit_code="$1"
  if [[ "${exit_code}" -ne 0 ]] \
    && [[ "${SCREW_JOB_TERMINAL_RECORDED:-0}" -ne 1 ]] \
    && [[ -n "${SCREW_MONITOR_RUN_DIR:-}" ]]; then
    screw_record_job \
      "${SCREW_MONITOR_RUN_DIR}" \
      failed \
      "${exit_code}" \
      "Scheduler wrapper exited before normal completion." || true
  fi
  screw_stop_live_monitor
  screw_collect_diagnostics "${exit_code}"
}

screw_start_live_monitor() {
  local run_dir="$1"
  screw_require_file "${SCREW_HPC_ROOT}/system_monitor.py"
  screw_require_file "${SCREW_HPC_ROOT}/status_hpc.py"
  screw_require_file "${SCREW_HPC_ROOT}/job_status.py"
  mkdir -p "${run_dir}/status"
  SCREW_MONITOR_RUN_DIR="${run_dir}"
  SCREW_JOB_TERMINAL_RECORDED=0
  export SCREW_MONITOR_RUN_DIR
  screw_record_job \
    "${run_dir}" \
    starting \
    "" \
    "Scheduled allocation started."
  "${SCREW_PYTHON}" "${SCREW_HPC_ROOT}/system_monitor.py" \
    --run-dir "${run_dir}" \
    --interval "${SCREW_SYSTEM_INTERVAL:-15}" \
    --history-interval "${SCREW_SYSTEM_HISTORY_INTERVAL:-15}" &
  SCREW_SYSTEM_MONITOR_PID=$!
  export SCREW_SYSTEM_MONITOR_PID
  trap 'screw_hpc_exit_trap "$?"' EXIT
  printf '%s\n' \
    "Live status:" \
    "  ./hpc_python.sh status_hpc.py --run-dir ${run_dir}" \
    "Watch:" \
    "  ./hpc_python.sh status_hpc.py --run-dir ${run_dir} --watch 10" \
    "Graceful stop:" \
    "  ./hpc_python.sh status_hpc.py --run-dir ${run_dir} --request-stop"
}
