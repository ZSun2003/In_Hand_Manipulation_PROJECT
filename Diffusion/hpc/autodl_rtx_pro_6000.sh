#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=hpc_env.sh
source "${SCRIPT_DIR}/hpc_env.sh"
PYTHON=""
BASE_CONFIG="${SCRIPT_DIR}/configs/autodl_rtx_pro_6000_extreme.json"
CONTROL_DIR="${SCRIPT_DIR}/runs/autodl_rtx_pro_6000_extreme_v8_diffusion_only_control"
CAPACITY_DIR="${SCRIPT_DIR}/runs/autodl_rtx_pro_6000_extreme_v8_diffusion_only_capacity"
CAPACITY_PROFILE_ID="autodl_rtx_pro_6000_extreme_v8_diffusion_only_spawn60_c12k_step2k_first_unsafe"
CAPACITY_CANDIDATES=()
for ((candidate_worlds = 12000; candidate_worlds <= 100000; candidate_worlds += 2000)); do
  CAPACITY_CANDIDATES+=("${candidate_worlds}")
done
WORLD_REQUEST_FILE="${CONTROL_DIR}/world_request"
WORLD_REQUEST="${SCREW_AUTODL_WORLDS:-}"
if [[ -z "${WORLD_REQUEST}" && -f "${WORLD_REQUEST_FILE}" ]]; then
  WORLD_REQUEST="$(tr -d '[:space:]' < "${WORLD_REQUEST_FILE}")"
fi
WORLD_REQUEST="${WORLD_REQUEST:-12000}"
case "${WORLD_REQUEST}" in
  max)
    RUN_VARIANT="max"
    ;;
  *)
    if [[ "${WORLD_REQUEST}" =~ ^[0-9]+$ ]] \
      && ((WORLD_REQUEST >= 12000 && WORLD_REQUEST <= 100000)) \
      && ((WORLD_REQUEST % 2000 == 0)); then
      RUN_VARIANT="w${WORLD_REQUEST}"
    else
      printf 'ERROR: SCREW_AUTODL_WORLDS must be max or a 2000-world increment from 12000 through 100000; got %s\n' \
        "${WORLD_REQUEST}" >&2
      exit 64
    fi
    ;;
esac
SMOKE_DIR="${SCRIPT_DIR}/runs/autodl_rtx_pro_6000_extreme_v8_diffusion_only_${RUN_VARIANT}_smoke"
FULL_DIR="${SCRIPT_DIR}/runs/autodl_rtx_pro_6000_extreme_v8_diffusion_only_${RUN_VARIANT}_full"
LOG_DIR="${SCRIPT_DIR}/logs"
LAUNCH_LOG="${LOG_DIR}/autodl_rtx_pro_6000_extreme_v8_diffusion_only_${RUN_VARIANT}.log"
PID_FILE="${CONTROL_DIR}/launcher.pid"
LOCK_FILE="${CONTROL_DIR}/training.lock"
MEMORY_CEILING="${SCREW_AUTODL_MEMORY_CEILING:-0.90}"
MAXIMUM_MEMORY_CEILING="0.90"
MINIMUM_FREE_GIB="${SCREW_AUTODL_RUN_MIN_FREE_GIB:-80}"
PROBE_TIMEOUT="${SCREW_AUTODL_PROBE_TIMEOUT:-14400}"
SWEEP_PHYSICS_STEPS="${SCREW_AUTODL_SWEEP_PHYSICS_STEPS:-256}"
FORMAL_GATE_PHYSICS_STEPS="${SCREW_AUTODL_GATE_PHYSICS_STEPS:-3000}"
EVAL_EPISODES="${SCREW_EVAL_EPISODES:-2000}"

ACTIVE_CHILD=""
SYSTEM_MONITOR_PID=""
TERMINAL_RECORDED=0
SELECTED_WORLDS=""
SELECTED_CONFIG=""

usage() {
  cat <<'EOF'
Usage: ./autodl_rtx_pro_6000.sh COMMAND

Commands:
  start    Start or resume the qualified formal pipeline under nohup.
  resume   Alias for start; the same stable run directory is reused.
  run      Run in the foreground (normally used internally by start).
  check    Re-run hardware, static, and small MJWarp GPU qualification.
  status   Show one status snapshot.
  watch    Refresh training status every 10 seconds.
  logs     Follow the persistent launcher log.
  stop     Request a checkpoint-consistent graceful stop.

World selection:
  Default                      Use the qualified 12000-world profile.
  SCREW_AUTODL_WORLDS=max     Use the largest safely qualified candidate.
  Explicit candidates         2000-world increments from 12000 through the
                              first unsafe candidate's preceding safe value.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

prepare_environment() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    fail "AutoDL training requires Linux"
  fi
  screw_use_core_env
  PYTHON="${SCREW_PYTHON}"
  mkdir -p \
    "${CONTROL_DIR}" \
    "${CAPACITY_DIR%/*}" \
    "${LOG_DIR}" \
    "${SCRIPT_DIR}/.cache/tmp" \
    "${SCRIPT_DIR}/.cache/xdg" \
    "${SCRIPT_DIR}/.cache/warp_rtx_pro_6000"
  export PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE=1
  export TMPDIR="${SCRIPT_DIR}/.cache/tmp"
  export XDG_CACHE_HOME="${SCRIPT_DIR}/.cache/xdg"
  export WARP_CACHE_PATH="${SCRIPT_DIR}/.cache/warp_rtx_pro_6000"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export CUDA_DEVICE_ORDER="PCI_BUS_ID"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-22}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-22}"
  export SCREW_ALLOCATION_ID="$(
    "${PYTHON}" -c 'import uuid; print(uuid.uuid4().hex)'
  )"
  if [[ ! "${SWEEP_PHYSICS_STEPS}" =~ ^[0-9]+$ ]] \
    || ((SWEEP_PHYSICS_STEPS < 216)); then
    fail "SCREW_AUTODL_SWEEP_PHYSICS_STEPS must be an integer >= 216"
  fi
  if ! "${PYTHON}" -c \
    'import math,sys; v=float(sys.argv[1]); raise SystemExit(0 if math.isfinite(v) and 0.0 < v <= 0.90 else 1)' \
    "${MEMORY_CEILING}"; then
    fail "SCREW_AUTODL_MEMORY_CEILING must be finite, positive, and no greater than the hard 0.90 safety ceiling"
  fi
  if [[ ! "${FORMAL_GATE_PHYSICS_STEPS}" =~ ^[0-9]+$ ]] \
    || ((FORMAL_GATE_PHYSICS_STEPS < 3000)); then
    fail "SCREW_AUTODL_GATE_PHYSICS_STEPS must be an integer >= 3000 so the terminal task state is certified"
  fi
  cd "${SCRIPT_DIR}"
}

check_training_disk_space() {
  if [[ ! "${MINIMUM_FREE_GIB}" =~ ^[0-9]+$ ]] \
    || ((MINIMUM_FREE_GIB < 50)); then
    fail "SCREW_AUTODL_RUN_MIN_FREE_GIB must be an integer >= 50"
  fi
  local available_kib required_kib
  available_kib="$(df -Pk "${SCRIPT_DIR}" | awk 'NR==2 {print $4}')"
  required_kib=$((MINIMUM_FREE_GIB * 1024 * 1024))
  if [[ ! "${available_kib}" =~ ^[0-9]+$ ]]; then
    fail "could not determine free space on the AutoDL data disk"
  fi
  if ((available_kib < required_kib)); then
    fail "only $((available_kib / 1024 / 1024)) GiB is free; Extreme V8 Diffusion-only requires at least ${MINIMUM_FREE_GIB} GiB before start/resume"
  fi
  printf 'Extreme V8 Diffusion-only disk gate: %s GiB free (minimum %s GiB).\n' \
    "$((available_kib / 1024 / 1024))" "${MINIMUM_FREE_GIB}"
}

launcher_pid() {
  local pid command_line
  [[ -f "${PID_FILE}" ]] || return 1
  pid="$(tr -cd '0-9' < "${PID_FILE}")"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  if [[ ! -r "/proc/${pid}/cmdline" ]]; then
    return 1
  fi
  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  [[ "${command_line}" == *autodl_rtx_pro_6000.sh* ]] || return 1
  printf '%s\n' "${pid}"
}

record_job() {
  local state="$1"
  local raw_exit_code="${2:-}"
  local message="${3:-}"
  local arguments=(
    "${SCRIPT_DIR}/job_status.py"
    --run-dir "${FULL_DIR}"
    --state "${state}"
    --message "${message}"
  )
  if [[ -n "${raw_exit_code}" ]]; then
    arguments+=(--raw-exit-code "${raw_exit_code}")
  fi
  "${PYTHON}" "${arguments[@]}" || true
  case "${state}" in
    completed|interrupted_resumable|failed)
      TERMINAL_RECORDED=1
      ;;
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
    # A trapped signal can interrupt Bash's wait before the child has
    # finished handling the forwarded SIGINT.  Keep waiting so checkpoint
    # finalization is not abandoned with an orphaned Python process.
    if kill -0 "${ACTIVE_CHILD}" 2>/dev/null; then
      if wait "${ACTIVE_CHILD}"; then
        status=0
      else
        status=$?
      fi
    fi
  fi
  ACTIVE_CHILD=""
  return "${status}"
}

handle_signal() {
  local signal_name="$1"
  printf 'Received %s; asking the active stage to stop consistently.\n' \
    "${signal_name}" >&2
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

collect_diagnostics() {
  local exit_code="$1"
  "${PYTHON}" "${SCRIPT_DIR}/collect_diagnostics.py" \
    --run-dir "${CONTROL_DIR}" \
    --exit-code "${exit_code}" \
    --job-id "${SCREW_ALLOCATION_ID}" \
    --allocation-log "${LAUNCH_LOG}" || true
}

finish_wrapper() {
  local exit_code="$1"
  trap - EXIT
  set +e
  stop_system_monitor
  if ((exit_code != 0)) && ((TERMINAL_RECORDED == 0)); then
    record_job failed "${exit_code}" \
      "AutoDL wrapper exited before normal completion."
  fi
  collect_diagnostics "${exit_code}"
  if [[ -f "${PID_FILE}" ]] \
    && [[ "$(tr -cd '0-9' < "${PID_FILE}")" == "$$" ]]; then
    rm -f -- "${PID_FILE}"
  fi
  exit "${exit_code}"
}

run_platform_qualification() {
  record_job preflight "" \
    "Validating the corrected expert canary before capacity qualification; requested worlds=${WORLD_REQUEST}."
  run_managed "${PYTHON}" \
    "${SCRIPT_DIR}/autodl_rtx_pro_6000_preflight.py"
  run_managed "${PYTHON}" "${SCRIPT_DIR}/preflight.py" \
    --mode static \
    --config "${BASE_CONFIG}"
}

run_capacity_qualification() {
  if [[ ! -f "${CAPACITY_DIR}/selection.json" ]]; then
    if [[ -e "${CAPACITY_DIR}/capacity.json" ]] \
      || [[ -e "${CAPACITY_DIR}/selected_config.json" ]] \
      || [[ -e "${CAPACITY_DIR}/qualified_configs" ]]; then
      fail "incomplete capacity selection in ${CAPACITY_DIR}; move that directory aside and retry"
    fi
    run_managed "${PYTHON}" "${SCRIPT_DIR}/adaptive_capacity.py" probe \
      --base-config "${BASE_CONFIG}" \
      --run-dir "${CAPACITY_DIR}" \
      --worlds "${CAPACITY_CANDIDATES[@]}" \
      --stop-after-first-unsafe \
      --require-worlds 12000 \
      --profile-id "${CAPACITY_PROFILE_ID}" \
      --pipeline-mode diffusion-only \
      --memory-ceiling "${MEMORY_CEILING}" \
      --physics-steps "${SWEEP_PHYSICS_STEPS}" \
      --timeout-seconds "${PROBE_TIMEOUT}"
  fi

  run_managed "${PYTHON}" "${SCRIPT_DIR}/adaptive_capacity.py" gate \
    --run-dir "${CAPACITY_DIR}" \
    --worlds "${WORLD_REQUEST}" \
    --require-profile-id "${CAPACITY_PROFILE_ID}" \
    --require-pipeline-mode diffusion-only \
    --require-candidates "${CAPACITY_CANDIDATES[@]}" \
    --require-budget-policy diffusion_only_worlds_v1 \
    --require-min-sweep-physics-steps 216 \
    --maximum-memory-ceiling "${MAXIMUM_MEMORY_CEILING}" \
    --physics-steps "${FORMAL_GATE_PHYSICS_STEPS}" \
    --timeout-seconds "${PROBE_TIMEOUT}"

  local resolution
  resolution="$(
    "${PYTHON}" "${SCRIPT_DIR}/adaptive_capacity.py" resolve \
      --run-dir "${CAPACITY_DIR}" \
      --worlds "${WORLD_REQUEST}" \
      --require-profile-id "${CAPACITY_PROFILE_ID}" \
      --require-pipeline-mode diffusion-only \
      --require-candidates "${CAPACITY_CANDIDATES[@]}" \
      --require-budget-policy diffusion_only_worlds_v1 \
      --require-min-sweep-physics-steps 216 \
      --maximum-memory-ceiling "${MAXIMUM_MEMORY_CEILING}"
  )"
  SELECTED_WORLDS="${resolution%%$'\n'*}"
  SELECTED_CONFIG="${resolution#*$'\n'}"
  if [[ ! "${SELECTED_WORLDS}" =~ ^[0-9]+$ ]] \
    || ((SELECTED_WORLDS < 12000 || SELECTED_WORLDS > 100000)) \
    || ((SELECTED_WORLDS % 2000 != 0)) \
    || [[ ! -f "${SELECTED_CONFIG}" ]] \
    || [[ "${SELECTED_CONFIG}" == *$'\n'* ]]; then
    fail "capacity resolver returned an invalid world/config pair"
  fi
  printf '%s\n' \
    "Requested worlds: ${WORLD_REQUEST}" \
    "Selected qualified worlds: ${SELECTED_WORLDS}" \
    "Selected signed config: ${SELECTED_CONFIG}"
}

run_smoke_chain() {
  run_managed "${PYTHON}" "${SCRIPT_DIR}/pipeline.py" \
    --config "${SCRIPT_DIR}/configs/smoke.json" \
    --stage diffusion-smoke \
    --resume auto \
    --run-dir "${SMOKE_DIR}"
  run_managed "${PYTHON}" "${SCRIPT_DIR}/pipeline.py" \
    --config "${SCRIPT_DIR}/configs/smoke.json" \
    --stage diffusion-smoke \
    --resume auto \
    --run-dir "${SMOKE_DIR}"
}

foreground_run() {
  prepare_environment
  check_training_disk_space
  if ! command -v flock >/dev/null 2>&1; then
    fail "flock is required by the AutoDL run wrapper"
  fi
  exec 9>>"${LOCK_FILE}"
  if ! flock -n 9; then
    fail "another AutoDL training wrapper already owns ${LOCK_FILE}"
  fi

  trap 'handle_signal TERM' TERM
  trap 'handle_signal HUP' HUP
  trap 'handle_signal INT' INT
  trap 'finish_wrapper $?' EXIT

  run_platform_qualification
  run_smoke_chain
  run_capacity_qualification

  [[ -n "${SELECTED_WORLDS}" && -f "${SELECTED_CONFIG}" ]] \
    || fail "capacity qualification did not resolve a signed config"
  record_job running "" \
    "Diffusion-only smoke and resume checks passed; formal ${SELECTED_WORLDS}-world pipeline started."

  "${PYTHON}" "${SCRIPT_DIR}/system_monitor.py" \
    --run-dir "${FULL_DIR}" \
    --interval "${SCREW_SYSTEM_INTERVAL:-15}" \
    --history-interval "${SCREW_SYSTEM_HISTORY_INTERVAL:-15}" &
  SYSTEM_MONITOR_PID=$!

  local pipeline_status=0
  set +e
  run_managed "${PYTHON}" "${SCRIPT_DIR}/pipeline.py" \
    --config "${SELECTED_CONFIG}" \
    --stage diffusion-full \
    --resume auto \
    --episodes "${EVAL_EPISODES}" \
    --run-dir "${FULL_DIR}"
  pipeline_status=$?
  set -e

  case "${pipeline_status}" in
    0)
      record_job completed 0 \
        "AutoDL formal pipeline and deterministic evaluation completed."
      ;;
    130)
      record_job interrupted_resumable 130 \
        "AutoDL pipeline stopped at a consistent resumable cursor."
      printf '%s\n' \
        "Training stopped safely." \
        "Resume with: ./autodl_rtx_pro_6000.sh resume"
      ;;
    *)
      record_job failed "${pipeline_status}" \
        "AutoDL formal pipeline failed; the last-good checkpoint was preserved."
      return "${pipeline_status}"
      ;;
  esac
}

background_start() {
  prepare_environment
  check_training_disk_space
  local existing_pid=""
  if existing_pid="$(launcher_pid)"; then
    fail "training is already running as PID ${existing_pid}"
  fi
  rm -f -- "${PID_FILE}"
  printf '%s\n' "${WORLD_REQUEST}" >"${WORLD_REQUEST_FILE}"
  : >>"${LAUNCH_LOG}"
  nohup "${SCRIPT_DIR}/autodl_rtx_pro_6000.sh" run \
    >>"${LAUNCH_LOG}" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "${pid}" >"${PID_FILE}"
  sleep 1
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f -- "${PID_FILE}"
    printf 'Launcher exited during startup. Recent log:\n' >&2
    tail -n 80 "${LAUNCH_LOG}" >&2 || true
    return 1
  fi
  printf '%s\n' \
    "AutoDL training launched as PID ${pid}." \
    "World request: ${WORLD_REQUEST} (${RUN_VARIANT})" \
    "Log: ${LAUNCH_LOG}" \
    "Status: ./autodl_rtx_pro_6000.sh status" \
    "Watch:  ./autodl_rtx_pro_6000.sh watch" \
    "Stop:   ./autodl_rtx_pro_6000.sh stop"
}

run_check() {
  prepare_environment
  "${PYTHON}" "${SCRIPT_DIR}/autodl_rtx_pro_6000_preflight.py"
  "${PYTHON}" "${SCRIPT_DIR}/preflight.py" \
    --mode static \
    --config "${BASE_CONFIG}"
  "${PYTHON}" "${SCRIPT_DIR}/preflight.py" \
    --mode gpu \
    --config "${SCRIPT_DIR}/configs/smoke.json"
}

show_status() {
  prepare_environment
  local pid=""
  if pid="$(launcher_pid)"; then
    printf 'launcher_pid=%s state=alive\n' "${pid}"
  else
    printf '%s\n' "launcher_state=not_running"
  fi
  if [[ -f "${FULL_DIR}/status/live.json" ]] \
    || [[ -f "${FULL_DIR}/status/job.json" ]]; then
    "${PYTHON}" "${SCRIPT_DIR}/status_hpc.py" \
      --run-dir "${FULL_DIR}" \
      --no-scheduler || true
  elif [[ -f "${LAUNCH_LOG}" ]]; then
    printf '%s\n' "Formal pipeline has not started. Recent qualification log:"
    tail -n 40 "${LAUNCH_LOG}"
  else
    printf '%s\n' "No AutoDL run has been started."
  fi
}

watch_status() {
  prepare_environment
  "${PYTHON}" "${SCRIPT_DIR}/status_hpc.py" \
    --run-dir "${FULL_DIR}" \
    --no-scheduler \
    --watch 10 || true
}

follow_logs() {
  prepare_environment
  touch "${LAUNCH_LOG}"
  tail -n 200 -F "${LAUNCH_LOG}"
}

request_stop() {
  prepare_environment
  if [[ -f "${FULL_DIR}/status/live.json" ]]; then
    if "${PYTHON}" "${SCRIPT_DIR}/status_hpc.py" \
      --run-dir "${FULL_DIR}" \
      --no-scheduler \
      --request-stop \
      --reason "requested through AutoDL wrapper"; then
      return 0
    fi
  fi

  local pid=""
  if pid="$(launcher_pid)"; then
    printf '%s\n' \
      "The formal heartbeat is not active; sending SIGTERM to wrapper PID ${pid}."
    kill -TERM "${pid}"
    return 0
  fi
  fail "no live AutoDL training process was found"
}

COMMAND="${1:-}"
case "${COMMAND}" in
  start|resume)
    background_start
    ;;
  run)
    foreground_run
    ;;
  check)
    run_check
    ;;
  status)
    show_status
    ;;
  watch)
    watch_status
    ;;
  logs)
    follow_logs
    ;;
  stop)
    request_stop
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 64
    ;;
esac
