#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=hpc_env.sh
source "${SCRIPT_DIR}/hpc_env.sh"

CONFIG="${SCRIPT_DIR}/configs/autodl_rtx_pro_6000_w14000_diffusion_ik_residual.json"
CONTROL_DIR="${SCRIPT_DIR}/runs/autodl_rtx_pro_6000_w14000_controller"
FULL_RUN_DIR="${SCRIPT_DIR}/runs/autodl_rtx_pro_6000_w14000_diffusion_ik_residual_full"
DIFFUSION_RUN_DIR="${SCRIPT_DIR}/runs/autodl_rtx_pro_6000_w14000_diffusion_only_full"
LOG_DIR="${SCRIPT_DIR}/logs"
FULL_LOG="${LOG_DIR}/autodl_rtx_pro_6000_w14000_diffusion_ik_residual.log"
DIFFUSION_LOG="${LOG_DIR}/autodl_rtx_pro_6000_w14000_diffusion_only.log"
PID_FILE="${CONTROL_DIR}/launcher.pid"
MODE_FILE="${CONTROL_DIR}/last_mode"
LOCK_FILE="${CONTROL_DIR}/training.lock"

MEMORY_CEILING="${SCREW_AUTODL_MEMORY_CEILING:-0.90}"
PREFLIGHT_STEPS="${SCREW_AUTODL_PREFLIGHT_STEPS:-4000}"
MINIMUM_FREE_GIB="${SCREW_AUTODL_RUN_MIN_FREE_GIB:-110}"
EVAL_EPISODES="${SCREW_EVAL_EPISODES:-2000}"

PYTHON=""
MODE="full"
RUN_DIR=""
LAUNCH_LOG=""
PIPELINE_STAGE=""
PIPELINE_MODE=""
ACTIVE_CHILD=""
SYSTEM_MONITOR_PID=""
TERMINAL_RECORDED=0

usage() {
  cat <<'EOF'
Usage: ./autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh COMMAND

Commands:
  setup                   Validate the shared environment and this package.
  preflight               Run the 14,000-world Diffusion-IK-TQC gate.
  preflight-diffusion-only
                          Run the same gate with TQC disabled.
  start                   Start/resume full Diffusion-IK-TQC training.
  resume                  Alias for start.
  start-diffusion-only    Start/resume Diffusion-IK with zero residual.
  resume-diffusion-only   Alias for start-diffusion-only.
  status                  Show the active or most recently selected run.
  watch                   Refresh its status every 10 seconds.
  logs                    Follow its detached launcher log.
  stop                    Request a checkpoint-consistent stop.

`SCREW_ENABLE_TQC=0 ... start` is equivalent to start-diffusion-only.
The two modes use separate run directories, logs and checkpoints.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

set_mode() {
  case "$1" in
    full)
      MODE="full"
      RUN_DIR="${FULL_RUN_DIR}"
      LAUNCH_LOG="${FULL_LOG}"
      PIPELINE_STAGE="full"
      PIPELINE_MODE="diffusion-tqc"
      ;;
    diffusion-only)
      MODE="diffusion-only"
      RUN_DIR="${DIFFUSION_RUN_DIR}"
      LAUNCH_LOG="${DIFFUSION_LOG}"
      PIPELINE_STAGE="diffusion-full"
      PIPELINE_MODE="diffusion-only"
      ;;
    *)
      fail "invalid launcher mode: $1"
      ;;
  esac
}

mode_from_environment() {
  case "${SCREW_ENABLE_TQC:-1}" in
    1|true|TRUE|yes|YES) printf '%s\n' full ;;
    0|false|FALSE|no|NO) printf '%s\n' diffusion-only ;;
    *) fail "SCREW_ENABLE_TQC must be 0 or 1" ;;
  esac
}

last_mode() {
  local value=""
  if [[ -f "${MODE_FILE}" ]]; then
    value="$(tr -d '[:space:]' <"${MODE_FILE}")"
  fi
  case "${value}" in
    full|diffusion-only) printf '%s\n' "${value}" ;;
    *) mode_from_environment ;;
  esac
}

prepare_environment() {
  [[ "$(uname -s)" == "Linux" ]] || fail "AutoDL training requires Linux"
  screw_use_core_env
  PYTHON="${SCREW_PYTHON}"
  mkdir -p \
    "${CONTROL_DIR}" "${LOG_DIR}" \
    "${SCRIPT_DIR}/.cache/tmp" \
    "${SCRIPT_DIR}/.cache/xdg" \
    "${SCRIPT_DIR}/.cache/warp_rtx_pro_6000_w14000"
  export PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE=1
  export TMPDIR="${SCRIPT_DIR}/.cache/tmp"
  export XDG_CACHE_HOME="${SCRIPT_DIR}/.cache/xdg"
  export WARP_CACHE_PATH="${SCRIPT_DIR}/.cache/warp_rtx_pro_6000_w14000"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export CUDA_DEVICE_ORDER="PCI_BUS_ID"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-22}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-22}"
  export SCREW_ALLOCATION_ID="$(${PYTHON} -c 'import uuid; print(uuid.uuid4().hex)')"
  cd "${SCRIPT_DIR}"
}

validate_options() {
  if [[ ! "${PREFLIGHT_STEPS}" =~ ^[0-9]+$ ]] \
    || ((PREFLIGHT_STEPS < 4000)); then
    fail "SCREW_AUTODL_PREFLIGHT_STEPS must be an integer >= 4000"
  fi
  if [[ ! "${MINIMUM_FREE_GIB}" =~ ^[0-9]+$ ]] \
    || ((MINIMUM_FREE_GIB < 80)); then
    fail "SCREW_AUTODL_RUN_MIN_FREE_GIB must be an integer >= 80"
  fi
  "${PYTHON}" -c \
    'import math,sys; x=float(sys.argv[1]); raise SystemExit(0 if math.isfinite(x) and 0.5 <= x <= 0.90 else 1)' \
    "${MEMORY_CEILING}" \
    || fail "SCREW_AUTODL_MEMORY_CEILING must be in [0.5, 0.90]"
}

check_disk_space() {
  local available_kib required_kib
  available_kib="$(df -Pk "${SCRIPT_DIR}" | awk 'NR==2 {print $4}')"
  required_kib=$((MINIMUM_FREE_GIB * 1024 * 1024))
  [[ "${available_kib}" =~ ^[0-9]+$ ]] \
    || fail "could not determine AutoDL data-disk free space"
  ((available_kib >= required_kib)) \
    || fail "only $((available_kib / 1024 / 1024)) GiB is free; ${MINIMUM_FREE_GIB} GiB is required"
  printf 'AutoDL disk gate: %s GiB free (minimum %s GiB).\n' \
    "$((available_kib / 1024 / 1024))" "${MINIMUM_FREE_GIB}"
}

archive_unmanifested_run_artifacts() {
  [[ -d "${RUN_DIR}" ]] || return 0
  [[ ! -f "${RUN_DIR}/run_manifest.json" ]] || return 0

  local unexpected=""
  unexpected="$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 \
    ! -name status -print -quit)"
  [[ -n "${unexpected}" ]] || return 0

  local archive_dir="${RUN_DIR}.pre_manifest_failed_$(date -u +%Y%m%dT%H%M%SZ)"
  mv -- "${RUN_DIR}" "${archive_dir}"
  printf 'Archived unmanifested run artifacts to %s\n' "${archive_dir}"
}

launcher_pid() {
  local pid command_line
  [[ -f "${PID_FILE}" ]] || return 1
  pid="$(tr -cd '0-9' <"${PID_FILE}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  [[ "${command_line}" == *autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh* ]] \
    || return 1
  printf '%s\n' "${pid}"
}

record_job() {
  local state="$1" raw_exit_code="${2:-}" message="${3:-}"
  local arguments=(
    "${SCRIPT_DIR}/job_status.py"
    --run-dir "${RUN_DIR}"
    --state "${state}"
    --message "${message}"
  )
  [[ -z "${raw_exit_code}" ]] \
    || arguments+=(--raw-exit-code "${raw_exit_code}")
  "${PYTHON}" "${arguments[@]}" || true
  case "${state}" in
    completed|interrupted_resumable|failed) TERMINAL_RECORDED=1 ;;
  esac
}

run_managed() {
  "$@" &
  ACTIVE_CHILD=$!
  local status=0
  if wait "${ACTIVE_CHILD}"; then
    status=0
  else
    status=$?
    if kill -0 "${ACTIVE_CHILD}" 2>/dev/null; then
      wait "${ACTIVE_CHILD}" || status=$?
    fi
  fi
  ACTIVE_CHILD=""
  return "${status}"
}

handle_signal() {
  printf 'Stopping %s consistently.\n' "${MODE}" >&2
  if [[ -n "${ACTIVE_CHILD}" ]]; then
    kill -INT "${ACTIVE_CHILD}" 2>/dev/null || true
  else
    exit 130
  fi
}

stop_system_monitor() {
  if [[ -n "${SYSTEM_MONITOR_PID}" ]]; then
    kill "${SYSTEM_MONITOR_PID}" 2>/dev/null || true
    wait "${SYSTEM_MONITOR_PID}" 2>/dev/null || true
    SYSTEM_MONITOR_PID=""
  fi
}

finish_wrapper() {
  local exit_code="$1"
  trap - EXIT
  set +e
  stop_system_monitor
  if ((exit_code != 0)) && ((TERMINAL_RECORDED == 0)); then
    record_job failed "${exit_code}" \
      "AutoDL ${MODE} wrapper exited before normal completion."
  fi
  "${PYTHON}" "${SCRIPT_DIR}/collect_diagnostics.py" \
    --run-dir "${RUN_DIR}" \
    --exit-code "${exit_code}" \
    --job-id "${SCREW_ALLOCATION_ID}" \
    --allocation-log "${LAUNCH_LOG}" || true
  if [[ -f "${PID_FILE}" ]] \
    && [[ "$(tr -cd '0-9' <"${PID_FILE}")" == "$$" ]]; then
    rm -f -- "${PID_FILE}"
  fi
  exit "${exit_code}"
}

run_setup() {
  "${SCRIPT_DIR}/setup_autodl_rtx_pro_6000.sh"
  prepare_environment
  "${PYTHON}" "${SCRIPT_DIR}/preflight.py" \
    --mode static --config "${CONFIG}"
  printf 'Dedicated AutoDL w14000 Diffusion-IK-residual package is ready.\n'
}

run_preflight() {
  prepare_environment
  validate_options
  check_disk_space
  record_job preflight "" \
    "Running the 14000-world ${MODE} static, GPU, and training-stack gates."
  run_managed "${PYTHON}" "${SCRIPT_DIR}/autodl_rtx_pro_6000_preflight.py"
  run_managed "${PYTHON}" "${SCRIPT_DIR}/preflight.py" \
    --mode static --config "${CONFIG}"
  run_managed "${PYTHON}" "${SCRIPT_DIR}/preflight.py" \
    --mode gpu --config "${CONFIG}" --worlds 14000
  run_managed "${PYTHON}" "${SCRIPT_DIR}/training_stack_preflight.py" \
    --config "${CONFIG}" \
    --worlds 14000 \
    --max-memory-fraction "${MEMORY_CEILING}" \
    --physics-steps "${PREFLIGHT_STEPS}" \
    --pipeline-mode "${PIPELINE_MODE}"
}

foreground_run() {
  prepare_environment
  validate_options
  check_disk_space
  command -v flock >/dev/null 2>&1 || fail "flock is required"
  exec 9>>"${LOCK_FILE}"
  flock -n 9 || fail "another AutoDL training wrapper owns ${LOCK_FILE}"

  trap 'handle_signal TERM' TERM
  trap 'handle_signal HUP' HUP
  trap 'handle_signal INT' INT
  trap 'finish_wrapper $?' EXIT

  archive_unmanifested_run_artifacts
  run_preflight
  record_job running "" \
    "Formal 14000-world ${MODE} pipeline started after all gates passed."

  "${PYTHON}" "${SCRIPT_DIR}/system_monitor.py" \
    --run-dir "${RUN_DIR}" \
    --interval "${SCREW_SYSTEM_INTERVAL:-15}" \
    --history-interval "${SCREW_SYSTEM_HISTORY_INTERVAL:-15}" &
  SYSTEM_MONITOR_PID=$!

  local status=0
  set +e
  run_managed "${PYTHON}" "${SCRIPT_DIR}/pipeline.py" \
    --config "${CONFIG}" \
    --stage "${PIPELINE_STAGE}" \
    --resume auto \
    --episodes "${EVAL_EPISODES}" \
    --run-dir "${RUN_DIR}"
  status=$?
  set -e

  case "${status}" in
    0)
      record_job completed 0 "AutoDL ${MODE} pipeline completed."
      ;;
    130)
      record_job interrupted_resumable 130 \
        "AutoDL ${MODE} pipeline stopped at a resumable cursor."
      ;;
    *)
      record_job failed "${status}" \
        "AutoDL ${MODE} pipeline failed; last-good checkpoints were preserved."
      return "${status}"
      ;;
  esac
}

background_start() {
  prepare_environment
  validate_options
  check_disk_space
  local existing_pid=""
  if existing_pid="$(launcher_pid)"; then
    fail "training is already running as PID ${existing_pid}"
  fi
  rm -f -- "${PID_FILE}"
  printf '%s\n' "${MODE}" >"${MODE_FILE}"
  : >>"${LAUNCH_LOG}"
  nohup setsid \
    "${SCRIPT_DIR}/autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh" \
    run "${MODE}" >>"${LAUNCH_LOG}" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "${pid}" >"${PID_FILE}"
  sleep 1
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f -- "${PID_FILE}"
    tail -n 80 "${LAUNCH_LOG}" >&2 || true
    return 1
  fi
  printf '%s\n' \
    "AutoDL ${MODE} training launched as PID ${pid}." \
    "Worlds: 14000" \
    "Run: ${RUN_DIR}" \
    "Log: ${LAUNCH_LOG}" \
    "Status: ./autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh status" \
    "Stop: ./autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh stop"
}

show_status() {
  prepare_environment
  local pid=""
  if pid="$(launcher_pid)"; then
    printf 'launcher_pid=%s state=alive mode=%s\n' "${pid}" "${MODE}"
  else
    printf 'launcher_state=not_running selected_mode=%s\n' "${MODE}"
  fi
  if [[ -f "${RUN_DIR}/status/live.json" ]] \
    || [[ -f "${RUN_DIR}/status/job.json" ]]; then
    "${PYTHON}" "${SCRIPT_DIR}/status_hpc.py" \
      --run-dir "${RUN_DIR}" --no-scheduler || true
  elif [[ -f "${LAUNCH_LOG}" ]]; then
    tail -n 60 "${LAUNCH_LOG}"
  else
    printf 'No %s run has been started.\n' "${MODE}"
  fi
}

watch_status() {
  prepare_environment
  "${PYTHON}" "${SCRIPT_DIR}/status_hpc.py" \
    --run-dir "${RUN_DIR}" --no-scheduler --watch 10 || true
}

follow_logs() {
  prepare_environment
  touch "${LAUNCH_LOG}"
  tail -n 200 -F "${LAUNCH_LOG}"
}

request_stop() {
  prepare_environment
  if [[ -f "${RUN_DIR}/status/live.json" ]]; then
    if "${PYTHON}" "${SCRIPT_DIR}/status_hpc.py" \
      --run-dir "${RUN_DIR}" --no-scheduler --request-stop \
      --reason "requested through AutoDL w14000 wrapper"; then
      return 0
    fi
  fi
  local pid=""
  if pid="$(launcher_pid)"; then
    printf 'Sending SIGTERM to wrapper PID %s; it will forward a graceful stop.\n' \
      "${pid}"
    kill -TERM "${pid}"
    return 0
  fi
  fail "no live AutoDL training process was found"
}

COMMAND="${1:-}"
case "${COMMAND}" in
  setup)
    set_mode "$(mode_from_environment)"
    run_setup
    ;;
  preflight|check)
    set_mode "$(mode_from_environment)"
    run_preflight
    ;;
  preflight-diffusion-only)
    set_mode diffusion-only
    run_preflight
    ;;
  start|resume)
    set_mode "$(mode_from_environment)"
    background_start
    ;;
  start-diffusion-only|resume-diffusion-only)
    set_mode diffusion-only
    background_start
    ;;
  run)
    set_mode "${2:-full}"
    foreground_run
    ;;
  status|watch|logs|stop)
    set_mode "$(last_mode)"
    case "${COMMAND}" in
      status) show_status ;;
      watch) watch_status ;;
      logs) follow_logs ;;
      stop) request_stop ;;
    esac
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 64
    ;;
esac
