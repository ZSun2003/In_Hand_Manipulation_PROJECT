#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HPC_ROOT="${PACKAGE_ROOT}/hpc"
CONFIG="${HPC_ROOT}/configs/autodl_rtx_pro_6000_r100_w5000_tqc16_continuation.json"
DEFAULT_DIFFUSION="${HPC_ROOT}/runs/autodl_rtx_pro_6000_r100_success_selfdistill_w18000_tqc16/checkpoints/diffusion.pt"
SOURCE_DIFFUSION="${SCREW_W5000_DIFFUSION_SOURCE:-${DEFAULT_DIFFUSION}}"
RUN_DIR="${SCREW_W5000_RUN_DIR:-${HPC_ROOT}/runs/autodl_rtx_pro_6000_r100_w5000_tqc16_continuation}"
EVALUATION_ROOT="${SCREW_W5000_EVALUATION_ROOT:-${HPC_ROOT}/runs/autodl_rtx_pro_6000_r100_w5000_tqc16_continuation_eval2000}"
DIFFUSION="${RUN_DIR}/checkpoints/diffusion.pt"
TQC_FULL="${RUN_DIR}/checkpoints/residual_tqc.pt"
TQC_POLICY="${RUN_DIR}/checkpoints/residual_tqc_policy.pt"

CORE_ENV="${IRP_HPC_CORE_ENV:-${PACKAGE_ROOT}/../RL_env/profiles/core/RL_env}"
PYTHON="${CORE_ENV}/bin/python"
CONTROL_DIR="${PACKAGE_ROOT}/control/r100_w5000_tqc16_continuation"
LOG_DIR="${PACKAGE_ROOT}/logs"
CHAIN_LOG="${LOG_DIR}/autodl_r100_w5000_tqc16_continuation.log"
PID_FILE="${CONTROL_DIR}/launcher.pid"
LOCK_FILE="${CONTROL_DIR}/training.lock"
PREFLIGHT_MARKER="${CONTROL_DIR}/preflight.complete.json"
EVALUATION_MARKER="${CONTROL_DIR}/evaluation.complete.json"

MEMORY_CEILING="${SCREW_AUTODL_MEMORY_CEILING:-0.90}"
PREFLIGHT_STEPS="${SCREW_AUTODL_PREFLIGHT_STEPS:-4000}"
MAX_EXPERT_TRUNCATION_FRACTION="${SCREW_MAX_EXPERT_TRUNCATION_FRACTION:-0.005}"
MINIMUM_FREE_GIB="${SCREW_AUTODL_RUN_MIN_FREE_GIB:-80}"
EVALUATION_EPISODES="${SCREW_EVAL_EPISODES:-2000}"

ACTIVE_CHILD=""
SYSTEM_MONITOR_PID=""
STOP_REQUESTED=0

usage() {
  cat <<'EOF'
Usage: ./run_autodl_r100_w5000_tqc16_continuation.sh COMMAND

Commands:
  setup       Validate the shared environment, config and source Diffusion.
  validate    Run fail-closed static and checkpoint validation.
  start       Detach the resumable continuation chain.
  resume      Alias for start.
  run         Run the chain in the foreground.
  status      Show chain/checkpoint/live training status.
  watch       Refresh status every 10 seconds.
  logs        Follow the detached chain log.
  stop        Request a checkpoint-consistent stop.

The isolated chain is:
  completed 18,000-world Diffusion checkpoint
  -> copy and verify in a new run directory
  -> complete 5,000-world Diffusion-TQC training-stack preflight
     (zero physical failures; at most 0.5% expert-canary truncations)
  -> fresh/resumable 5,000-world residual TQC to 5M transitions
  -> 2,000-episode evaluation.

Only the frozen Diffusion runtime vector width changes. The validator rejects
any other source-checkpoint/config difference. No existing run is overwritten.
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_absolute_path() {
  local label="$1" value="$2"
  [[ "${value}" == /* ]] || fail "${label} must be an absolute path: ${value}"
}

prepare_environment() {
  [[ "$(uname -s)" == "Linux" ]] || fail "AutoDL training requires Linux"
  [[ -x "${PYTHON}" ]] || fail "shared Python is missing: ${PYTHON}"
  require_absolute_path SCREW_W5000_DIFFUSION_SOURCE "${SOURCE_DIFFUSION}"
  require_absolute_path SCREW_W5000_RUN_DIR "${RUN_DIR}"
  require_absolute_path SCREW_W5000_EVALUATION_ROOT "${EVALUATION_ROOT}"
  mkdir -p \
    "${CONTROL_DIR}" "${LOG_DIR}" "${RUN_DIR}/checkpoints" \
    "${PACKAGE_ROOT}/.cache/tmp" \
    "${PACKAGE_ROOT}/.cache/xdg" \
    "${PACKAGE_ROOT}/.cache/warp_rtx_pro_6000_w5000"
  export PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE=1
  export TMPDIR="${PACKAGE_ROOT}/.cache/tmp"
  export XDG_CACHE_HOME="${PACKAGE_ROOT}/.cache/xdg"
  export MPLCONFIGDIR="${PACKAGE_ROOT}/.cache/matplotlib"
  export WARP_CACHE_PATH="${PACKAGE_ROOT}/.cache/warp_rtx_pro_6000_w5000"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export CUDA_DEVICE_ORDER="PCI_BUS_ID"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-22}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-22}"
  cd "${PACKAGE_ROOT}"
}

validate_options() {
  [[ "${PREFLIGHT_STEPS}" =~ ^[0-9]+$ ]] \
    && ((PREFLIGHT_STEPS == 4000)) \
    || fail "SCREW_AUTODL_PREFLIGHT_STEPS must remain exactly 4000"
  [[ "${MINIMUM_FREE_GIB}" =~ ^[0-9]+$ ]] \
    && ((MINIMUM_FREE_GIB >= 40)) \
    || fail "SCREW_AUTODL_RUN_MIN_FREE_GIB must be an integer >= 40"
  [[ "${EVALUATION_EPISODES}" =~ ^[0-9]+$ ]] \
    && ((EVALUATION_EPISODES > 0)) \
    || fail "SCREW_EVAL_EPISODES must be a positive integer"
  "${PYTHON}" -c '
import math,sys
memory=float(sys.argv[1]); allowance=float(sys.argv[2])
ok=(math.isfinite(memory) and 0.5 <= memory < 1.0
    and math.isfinite(allowance) and 0.0 <= allowance <= 0.005)
raise SystemExit(0 if ok else 1)
' "${MEMORY_CEILING}" "${MAX_EXPERT_TRUNCATION_FRACTION}" \
    || fail "memory ceiling must be in [0.5,1.0) and expert timeout allowance in [0,0.005]"
}

check_disk_space() {
  local available_kib required_kib
  available_kib="$(df -Pk "${PACKAGE_ROOT}" | awk 'NR == 2 {print $4}')"
  required_kib=$((MINIMUM_FREE_GIB * 1024 * 1024))
  [[ "${available_kib}" =~ ^[0-9]+$ ]] \
    || fail "could not determine AutoDL data-disk free space"
  ((available_kib >= required_kib)) \
    || fail "only $((available_kib / 1024 / 1024)) GiB is free; ${MINIMUM_FREE_GIB} GiB is required"
  printf 'AutoDL disk gate: %s GiB free (minimum %s GiB).\n' \
    "$((available_kib / 1024 / 1024))" "${MINIMUM_FREE_GIB}"
}

launcher_pid() {
  local pid command_line
  [[ -f "${PID_FILE}" ]] || return 1
  pid="$(tr -cd '0-9' <"${PID_FILE}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  [[ "${command_line}" == *run_autodl_r100_w5000_tqc16_continuation.sh* ]] \
    || return 1
  printf '%s\n' "${pid}"
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
  ((STOP_REQUESTED == 0)) || return 130
  return "${status}"
}

handle_signal() {
  printf 'Stopping the w5000 continuation at the next safe cursor.\n' >&2
  STOP_REQUESTED=1
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
  local status="$1"
  trap - EXIT
  set +e
  stop_system_monitor
  if [[ -f "${PID_FILE}" ]] \
    && [[ "$(tr -cd '0-9' <"${PID_FILE}")" == "$$" ]]; then
    rm -f -- "${PID_FILE}"
  fi
  exit "${status}"
}

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

validate_artifact() {
  local stage="$1"
  local arguments=(
    "${PYTHON}" "${HPC_ROOT}/validate_w5000_tqc_continuation.py"
    --config "${CONFIG}" --stage "${stage}"
  )
  if [[ "${stage}" != "config" ]]; then
    arguments+=(--diffusion "${DIFFUSION}")
  fi
  if [[ "${stage}" == "tqc" ]]; then
    arguments+=(--tqc "${TQC_POLICY}")
  fi
  "${arguments[@]}"
}

validate_source_diffusion() {
  "${PYTHON}" "${HPC_ROOT}/validate_w5000_tqc_continuation.py" \
    --config "${CONFIG}" --diffusion "${SOURCE_DIFFUSION}" --stage diffusion
}

prepare_diffusion_checkpoint() {
  validate_source_diffusion >/dev/null
  local source_sha destination_sha temporary
  source_sha="$(file_sha256 "${SOURCE_DIFFUSION}")"
  if [[ "${SOURCE_DIFFUSION}" == "${DIFFUSION}" ]]; then
    validate_artifact diffusion
    return 0
  fi
  if [[ -f "${DIFFUSION}" ]]; then
    destination_sha="$(file_sha256 "${DIFFUSION}")"
    [[ "${destination_sha}" == "${source_sha}" ]] \
      || fail "new-run Diffusion checkpoint differs from the selected source"
  else
    temporary="${DIFFUSION}.copying.$$"
    cp -p -- "${SOURCE_DIFFUSION}" "${temporary}"
    [[ "$(file_sha256 "${temporary}")" == "${source_sha}" ]] \
      || fail "Diffusion checkpoint copy failed its SHA-256 check"
    mv -f -- "${temporary}" "${DIFFUSION}"
  fi
  validate_artifact diffusion
}

config_file_sha256() {
  file_sha256 "${CONFIG}"
}

preflight_marker_valid() {
  [[ -f "${PREFLIGHT_MARKER}" ]] || return 1
  "${PYTHON}" -c '
import hashlib,json,pathlib,sys
marker,config_sha,diffusion_sha,ceiling,steps,allowance=sys.argv[1:]
p=json.load(open(marker, encoding="utf-8"))
report=pathlib.Path(str(p.get("report", "")))
report_sha=(hashlib.sha256(report.read_bytes()).hexdigest()
            if report.is_file() else None)
ok=(p.get("stage")=="preflight" and p.get("status")=="complete"
    and int(p.get("worlds",-1))==5000
    and p.get("config_file_sha256")==config_sha
    and p.get("diffusion_sha256")==diffusion_sha
    and str(p.get("memory_ceiling"))==ceiling
    and int(p.get("physics_steps",-1))==int(steps)
    and str(p.get("max_expert_truncation_fraction"))==allowance
    and report_sha==p.get("report_sha256"))
raise SystemExit(0 if ok else 1)
' "${PREFLIGHT_MARKER}" "$(config_file_sha256)" \
    "$(file_sha256 "${DIFFUSION}")" "${MEMORY_CEILING}" \
    "${PREFLIGHT_STEPS}" "${MAX_EXPERT_TRUNCATION_FRACTION}"
}

write_preflight_marker() {
  local report="$1" temporary="${PREFLIGHT_MARKER}.tmp.$$"
  "${PYTHON}" -c '
import hashlib,json,os,sys
(target,report,config_sha,diffusion_sha,ceiling,steps,allowance)=sys.argv[1:]
payload=json.load(open(report, encoding="utf-8"))
if payload.get("status") != "pass" or int(payload.get("worlds",-1)) != 5000:
    raise SystemExit("preflight report is not a passing 5,000-world result")
if str(payload.get("max_expert_truncation_fraction")) != allowance:
    raise SystemExit("preflight report does not record the requested timeout allowance")
h=hashlib.sha256(open(report,"rb").read()).hexdigest()
marker={"stage":"preflight","status":"complete","worlds":5000,
        "memory_ceiling":ceiling,"physics_steps":int(steps),
        "max_expert_truncation_fraction":allowance,
        "config_file_sha256":config_sha,"diffusion_sha256":diffusion_sha,
        "report":os.path.abspath(report),"report_sha256":h}
with open(target,"w",encoding="utf-8") as f:
    json.dump(marker,f,sort_keys=True); f.write("\n")
' "${temporary}" "${report}" "$(config_file_sha256)" \
    "$(file_sha256 "${DIFFUSION}")" "${MEMORY_CEILING}" \
    "${PREFLIGHT_STEPS}" "${MAX_EXPERT_TRUNCATION_FRACTION}"
  mv -f -- "${temporary}" "${PREFLIGHT_MARKER}"
}

run_preflight_stage() {
  if preflight_marker_valid; then
    printf 'Stage preflight: matching 5,000-world gate already passed; skipping.\n'
    return 0
  fi
  local stamp report status
  stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
  report="${LOG_DIR}/w5000_diffusion_tqc_preflight_${stamp}.json"
  printf '%s\n' \
    "Stage preflight: worlds=5000, ceiling=${MEMORY_CEILING}, steps=${PREFLIGHT_STEPS}." \
    "Expert truncation allowance=${MAX_EXPERT_TRUNCATION_FRACTION}; physical failures remain forbidden."
  "${PYTHON}" "${HPC_ROOT}/training_stack_preflight.py" \
    --config "${CONFIG}" \
    --worlds 5000 \
    --max-memory-fraction "${MEMORY_CEILING}" \
    --physics-steps "${PREFLIGHT_STEPS}" \
    --pipeline-mode diffusion-tqc \
    --max-expert-truncation-fraction "${MAX_EXPERT_TRUNCATION_FRACTION}" \
    >"${report}" &
  ACTIVE_CHILD=$!
  if wait "${ACTIVE_CHILD}"; then
    status=0
  else
    status=$?
    if kill -0 "${ACTIVE_CHILD}" 2>/dev/null; then
      wait "${ACTIVE_CHILD}" || status=$?
    fi
  fi
  ACTIVE_CHILD=""
  ((STOP_REQUESTED == 0)) || return 130
  "${PYTHON}" -m json.tool "${report}" 2>/dev/null || true
  ((status == 0)) \
    || fail "5,000-world TQC preflight failed with status ${status}; TQC was not started"
  write_preflight_marker "${report}"
}

run_tqc_stage() {
  if validate_artifact tqc >/dev/null 2>&1; then
    printf 'Stage TQC: completed 5,000-world policy verified; skipping.\n'
    validate_artifact tqc
    return 0
  fi
  local arguments=(
    "${PYTHON}" "${HPC_ROOT}/train_residual_gpu.py"
    --config "${CONFIG}"
    --diffusion "${DIFFUSION}"
    --output "${TQC_FULL}"
    --log "${RUN_DIR}/logs/tqc.jsonl"
    --run-dir "${RUN_DIR}"
  )
  if [[ ! -f "${TQC_FULL}" ]]; then
    arguments+=(--no-resume)
    printf 'Stage TQC: starting fresh at 5,000 worlds and zero transitions.\n'
  else
    printf 'Stage TQC: resuming the exact 5,000-world replay/rollout cursor.\n'
  fi
  run_managed "${arguments[@]}"
  validate_artifact tqc \
    || fail "TQC exited without the complete 5M-transition policy"
}

evaluation_marker_valid() {
  [[ -f "${EVALUATION_MARKER}" ]] || return 1
  "${PYTHON}" -c '
import json,sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
ok=(p.get("stage")=="evaluation" and p.get("status")=="complete"
    and p.get("diffusion_sha256")==sys.argv[2]
    and p.get("tqc_sha256")==sys.argv[3]
    and int(p.get("episodes",-1))==int(sys.argv[4]))
raise SystemExit(0 if ok else 1)
' "${EVALUATION_MARKER}" "$(file_sha256 "${DIFFUSION}")" \
    "$(file_sha256 "${TQC_POLICY}")" "${EVALUATION_EPISODES}"
}

next_evaluation_attempt() {
  local index candidate
  for index in $(seq -w 1 999); do
    candidate="${EVALUATION_ROOT}/attempt_${index}"
    if [[ ! -e "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  fail "evaluation attempt namespace is exhausted"
}

write_evaluation_marker() {
  local attempt="$1" temporary="${EVALUATION_MARKER}.tmp.$$"
  "${PYTHON}" -c '
import json,os,sys
target,attempt,diffusion_sha,tqc_sha,episodes=sys.argv[1:]
payload={"stage":"evaluation","status":"complete",
         "attempt":os.path.abspath(attempt),
         "diffusion_sha256":diffusion_sha,"tqc_sha256":tqc_sha,
         "episodes":int(episodes)}
with open(target,"w",encoding="utf-8") as f:
    json.dump(payload,f,sort_keys=True); f.write("\n")
' "${temporary}" "${attempt}" "$(file_sha256 "${DIFFUSION}")" \
    "$(file_sha256 "${TQC_POLICY}")" "${EVALUATION_EPISODES}"
  mv -f -- "${temporary}" "${EVALUATION_MARKER}"
}

run_evaluation_stage() {
  if evaluation_marker_valid; then
    printf 'Stage evaluation: matching completed attempt verified; skipping.\n'
    return 0
  fi
  local attempt
  attempt="$(next_evaluation_attempt)"
  printf 'Stage evaluation: writing a fresh attempt to %s\n' "${attempt}"
  run_managed "${PYTHON}" "${HPC_ROOT}/evaluate_gpu.py" \
    --config "${CONFIG}" \
    --diffusion "${DIFFUSION}" \
    --tqc "${TQC_POLICY}" \
    --episodes "${EVALUATION_EPISODES}" \
    --log "${attempt}/logs/evaluation.jsonl" \
    --run-dir "${attempt}"
  write_evaluation_marker "${attempt}"
}

run_static_validation() {
  local required help_output
  for required in \
    "${CONFIG}" \
    "${HPC_ROOT}/validate_w5000_tqc_continuation.py" \
    "${HPC_ROOT}/training_stack_preflight.py" \
    "${HPC_ROOT}/train_residual_gpu.py" \
    "${HPC_ROOT}/evaluate_gpu.py"; do
    [[ -f "${required}" ]] || fail "required file is missing: ${required}"
  done
  validate_artifact config
  validate_source_diffusion
  help_output="$("${PYTHON}" "${HPC_ROOT}/training_stack_preflight.py" --help)"
  [[ "${help_output}" == *--max-expert-truncation-fraction* ]] \
    || fail "training-stack preflight lacks the formal expert-timeout admission CLI"
  "${PYTHON}" "${HPC_ROOT}/preflight.py" --mode static --config "${CONFIG}"
}

start_system_monitor() {
  mkdir -p "${RUN_DIR}/status"
  "${PYTHON}" "${HPC_ROOT}/system_monitor.py" \
    --run-dir "${RUN_DIR}" \
    --interval "${SCREW_SYSTEM_INTERVAL:-15}" \
    --history-interval "${SCREW_SYSTEM_HISTORY_INTERVAL:-15}" &
  SYSTEM_MONITOR_PID=$!
}

foreground_chain() {
  prepare_environment
  validate_options
  check_disk_space
  command -v flock >/dev/null 2>&1 || fail "flock is required"
  exec 9>>"${LOCK_FILE}"
  flock -n 9 || fail "another w5000 continuation owns ${LOCK_FILE}"
  trap 'handle_signal' TERM HUP INT
  trap 'finish_wrapper $?' EXIT

  run_static_validation
  prepare_diffusion_checkpoint
  start_system_monitor
  run_preflight_stage
  run_tqc_stage
  stop_system_monitor
  run_evaluation_stage
  printf '%s\n' \
    "Continuation chain finished." \
    "Diffusion: ${DIFFUSION}" \
    "TQC policy: ${TQC_POLICY}"
}

background_start() {
  prepare_environment
  validate_options
  check_disk_space
  local existing=""
  if existing="$(launcher_pid)"; then
    fail "the w5000 continuation is already running as PID ${existing}"
  fi
  rm -f -- "${PID_FILE}"
  : >>"${CHAIN_LOG}"
  nohup setsid "${PACKAGE_ROOT}/run_autodl_r100_w5000_tqc16_continuation.sh" \
    run >>"${CHAIN_LOG}" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "${pid}" >"${PID_FILE}"
  sleep 1
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f -- "${PID_FILE}"
    tail -n 100 "${CHAIN_LOG}" >&2 || true
    return 1
  fi
  printf '%s\n' \
    "AutoDL w5000 TQC continuation launched as PID ${pid}." \
    "Source Diffusion: ${SOURCE_DIFFUSION}" \
    "Run: ${RUN_DIR}" \
    "Log: ${CHAIN_LOG}" \
    "Status: ./run_autodl_r100_w5000_tqc16_continuation.sh status"
}

show_status() {
  prepare_environment
  local pid=""
  if pid="$(launcher_pid)"; then
    printf 'launcher_pid=%s state=alive\n' "${pid}"
  else
    printf 'launcher_state=not_running\n'
  fi
  printf 'source_diffusion=%s\nrun_dir=%s\n' "${SOURCE_DIFFUSION}" "${RUN_DIR}"
  printf 'preflight_marker=%s evaluation_marker=%s\n' \
    "$([[ -f "${PREFLIGHT_MARKER}" ]] && printf present || printf absent)" \
    "$([[ -f "${EVALUATION_MARKER}" ]] && printf present || printf absent)"
  if [[ -f "${RUN_DIR}/status/live.json" ]] \
    || [[ -f "${RUN_DIR}/status/job.json" ]]; then
    "${PYTHON}" "${HPC_ROOT}/status_hpc.py" \
      --run-dir "${RUN_DIR}" --no-scheduler || true
  elif [[ -f "${CHAIN_LOG}" ]]; then
    tail -n 40 "${CHAIN_LOG}"
  else
    printf 'No continuation stage has started yet.\n'
  fi
}

watch_status() {
  while true; do
    clear || true
    show_status
    sleep 10
  done
}

follow_logs() {
  prepare_environment
  touch "${CHAIN_LOG}"
  tail -n 200 -F "${CHAIN_LOG}"
}

request_stop() {
  prepare_environment
  if [[ -f "${RUN_DIR}/status/live.json" ]]; then
    "${PYTHON}" "${HPC_ROOT}/status_hpc.py" \
      --run-dir "${RUN_DIR}" --no-scheduler --request-stop \
      --reason "requested through the AutoDL w5000 TQC continuation wrapper" \
      || true
  fi
  local pid=""
  if pid="$(launcher_pid)"; then
    printf 'Sending SIGTERM to wrapper PID %s.\n' "${pid}"
    kill -TERM "${pid}"
    return 0
  fi
  fail "no live w5000 continuation process was found"
}

COMMAND="${1:-}"
case "${COMMAND}" in
  setup)
    "${HPC_ROOT}/setup_autodl_rtx_pro_6000.sh"
    prepare_environment
    validate_options
    run_static_validation
    ;;
  validate)
    prepare_environment
    validate_options
    check_disk_space
    run_static_validation
    ;;
  start|resume)
    background_start
    ;;
  run)
    foreground_chain
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
