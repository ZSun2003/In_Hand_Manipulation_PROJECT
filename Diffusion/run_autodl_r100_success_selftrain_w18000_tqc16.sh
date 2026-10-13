#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HPC_ROOT="${PACKAGE_ROOT}/hpc"
CONFIG="${HPC_ROOT}/configs/autodl_rtx_pro_6000_r100_success_selfdistill_w18000_tqc16.json"
SCAN_LAUNCHER="${PACKAGE_ROOT}/run_autodl_full_workspace_diffusion_reachability.sh"
DEFAULT_DATASET="${PACKAGE_ROOT}/results/workspace_policy_rollouts.successful_policy_rollouts"
DATASET_IS_EXTERNAL_OVERRIDE=0
if [[ -n "${SCREW_SUCCESS_DATASET:-}" ]]; then
  DATASET_IS_EXTERNAL_OVERRIDE=1
fi
DATASET="${SCREW_SUCCESS_DATASET:-${DEFAULT_DATASET}}"
RUN_DIR="${SCREW_W18000_RUN_DIR:-${HPC_ROOT}/runs/autodl_rtx_pro_6000_r100_success_selfdistill_w18000_tqc16}"
EVALUATION_ROOT="${SCREW_W18000_EVALUATION_ROOT:-${HPC_ROOT}/runs/autodl_rtx_pro_6000_r100_success_selfdistill_w18000_tqc16_eval2000}"

CORE_ENV="${IRP_HPC_CORE_ENV:-${PACKAGE_ROOT}/../RL_env/profiles/core/RL_env}"
PYTHON="${CORE_ENV}/bin/python"
CONTROL_DIR="${PACKAGE_ROOT}/control/r100_success_selftrain_w18000_tqc16"
LOG_DIR="${PACKAGE_ROOT}/logs"
CHAIN_LOG="${LOG_DIR}/autodl_r100_success_selftrain_w18000_tqc16.log"
PID_FILE="${CONTROL_DIR}/launcher.pid"
LOCK_FILE="${CONTROL_DIR}/training.lock"
PREFLIGHT_MARKER="${CONTROL_DIR}/preflight.complete.json"
EVALUATION_MARKER="${CONTROL_DIR}/evaluation.complete.json"

MEMORY_CEILING="${SCREW_AUTODL_MEMORY_CEILING:-0.90}"
PREFLIGHT_STEPS="${SCREW_AUTODL_PREFLIGHT_STEPS:-4000}"
MAX_EXPERT_TRUNCATION_FRACTION="${SCREW_MAX_EXPERT_TRUNCATION_FRACTION:-0.005}"
MINIMUM_FREE_GIB="${SCREW_AUTODL_RUN_MIN_FREE_GIB:-110}"
EVALUATION_EPISODES="${SCREW_EVAL_EPISODES:-2000}"
DATASET_POLL_SECONDS="${SCREW_DATASET_POLL_SECONDS:-60}"
DATASET_WAIT_TIMEOUT_SECONDS="${SCREW_DATASET_WAIT_TIMEOUT_SECONDS:-0}"

ACTIVE_CHILD=""
SYSTEM_MONITOR_PID=""
STOP_REQUESTED=0

usage() {
  cat <<'EOF'
Usage: ./run_autodl_r100_success_selftrain_w18000_tqc16.sh COMMAND

Commands:
  setup       Validate the shared environment and all static contracts.
  validate    Validate the 100 mm scanner and the 18,000-world config.
  start       Detach the complete resumable chain.
  resume      Alias for start; completed stages are verified and skipped.
  run         Run the chain in the foreground (internal/diagnostic use).
  status      Show chain, dataset, checkpoint and live training status.
  watch       Refresh status every 10 seconds.
  logs        Follow the detached chain log.
  stop        Request a checkpoint-consistent stop, then signal the wrapper.

The fixed chain is:
  exact-contact 100 mm scan (360 x 50 x 2)
  -> fresh Diffusion from successful policy rollouts
  -> complete 18,000-world Diffusion-TQC preflight
  -> fresh 18,000-world residual TQC
  -> 2,000-episode evaluation.

The default preflight ceiling is 0.90. It never reduces the world count.
To consume a scanner already running in another package, set the absolute
SCREW_SUCCESS_DATASET path before `start`; this wrapper waits for its finalized
metadata and arrays instead of launching a second scan.
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
  require_absolute_path SCREW_SUCCESS_DATASET "${DATASET}"
  require_absolute_path SCREW_W18000_RUN_DIR "${RUN_DIR}"
  require_absolute_path SCREW_W18000_EVALUATION_ROOT "${EVALUATION_ROOT}"
  mkdir -p \
    "${CONTROL_DIR}" "${LOG_DIR}" \
    "${PACKAGE_ROOT}/.cache/tmp" \
    "${PACKAGE_ROOT}/.cache/xdg" \
    "${PACKAGE_ROOT}/.cache/warp_rtx_pro_6000_w18000"
  export PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE=1
  export TMPDIR="${PACKAGE_ROOT}/.cache/tmp"
  export XDG_CACHE_HOME="${PACKAGE_ROOT}/.cache/xdg"
  export MPLCONFIGDIR="${PACKAGE_ROOT}/.cache/matplotlib"
  export WARP_CACHE_PATH="${PACKAGE_ROOT}/.cache/warp_rtx_pro_6000_w18000"
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
    && ((PREFLIGHT_STEPS >= 4000)) \
    || fail "SCREW_AUTODL_PREFLIGHT_STEPS must be an integer >= 4000"
  [[ "${MINIMUM_FREE_GIB}" =~ ^[0-9]+$ ]] \
    && ((MINIMUM_FREE_GIB >= 80)) \
    || fail "SCREW_AUTODL_RUN_MIN_FREE_GIB must be an integer >= 80"
  [[ "${EVALUATION_EPISODES}" =~ ^[0-9]+$ ]] \
    && ((EVALUATION_EPISODES > 0)) \
    || fail "SCREW_EVAL_EPISODES must be a positive integer"
  [[ "${DATASET_POLL_SECONDS}" =~ ^[0-9]+$ ]] \
    && ((DATASET_POLL_SECONDS >= 10)) \
    || fail "SCREW_DATASET_POLL_SECONDS must be an integer >= 10"
  [[ "${DATASET_WAIT_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] \
    || fail "SCREW_DATASET_WAIT_TIMEOUT_SECONDS must be a non-negative integer"
  "${PYTHON}" -c \
    'import math,sys; x=float(sys.argv[1]); raise SystemExit(0 if math.isfinite(x) and 0.5 <= x < 1.0 else 1)' \
    "${MEMORY_CEILING}" \
    || fail "SCREW_AUTODL_MEMORY_CEILING must be in [0.5, 1.0)"
  "${PYTHON}" -c \
    'import math,sys; x=float(sys.argv[1]); raise SystemExit(0 if math.isfinite(x) and 0.0 <= x <= 1.0 else 1)' \
    "${MAX_EXPERT_TRUNCATION_FRACTION}" \
    || fail "SCREW_MAX_EXPERT_TRUNCATION_FRACTION must be in [0, 1]"
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
  [[ "${command_line}" == *run_autodl_r100_success_selftrain_w18000_tqc16.sh* ]] \
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
  if ((STOP_REQUESTED)); then
    return 130
  fi
  return "${status}"
}

handle_signal() {
  printf 'Stopping the w18000 chain at the next safe child cursor.\n' >&2
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

dataset_status() {
  local output status
  if output="$("${PYTHON}" "${HPC_ROOT}/validate_success_dataset.py" \
    --dataset "${DATASET}" 2>&1)"; then
    printf '%s\n' "${output}"
    return 0
  else
    status=$?
  fi
  if ((status == 3)); then
    return 3
  fi
  printf '%s\n' "${output}" >&2
  return "${status}"
}

wait_for_dataset() {
  local started now elapsed next_report=0 status
  started="$(date +%s)"
  printf 'Waiting for finalized successful-policy dataset: %s\n' "${DATASET}"
  while true; do
    if dataset_status >/dev/null; then
      dataset_status
      return 0
    else
      status=$?
    fi
    if ((status != 3)); then
      fail "the requested successful-policy dataset is invalid"
    fi
    now="$(date +%s)"
    elapsed=$((now - started))
    if ((DATASET_WAIT_TIMEOUT_SECONDS > 0 \
      && elapsed >= DATASET_WAIT_TIMEOUT_SECONDS)); then
      fail "timed out waiting for ${DATASET} after ${elapsed} seconds"
    fi
    if ((elapsed >= next_report)); then
      printf 'Dataset is still being produced (waited %s seconds).\n' "${elapsed}"
      next_report=$((elapsed + 60))
    fi
    sleep "${DATASET_POLL_SECONDS}" &
    ACTIVE_CHILD=$!
    wait "${ACTIVE_CHILD}"
    ACTIVE_CHILD=""
    if ((STOP_REQUESTED)); then
      return 130
    fi
  done
}

local_scan_is_alive() {
  local pid_file="${PACKAGE_ROOT}/workspace_diffusion_reachability.pid" pid
  [[ -f "${pid_file}" ]] || return 1
  pid="$(tr -cd '0-9' <"${pid_file}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

run_scan_stage() {
  local status
  if dataset_status >/dev/null; then
    printf 'Stage scan: finalized dataset already exists; skipping.\n'
    dataset_status
    return 0
  else
    status=$?
  fi
  if ((status != 3)); then
    fail "existing success dataset is invalid: ${DATASET}"
  fi

  if ((DATASET_IS_EXTERNAL_OVERRIDE)); then
    printf 'Stage scan: external dataset requested; waiting without launching a scanner.\n'
    wait_for_dataset
    return 0
  fi
  if local_scan_is_alive; then
    printf 'Stage scan: an existing local scanner is alive; waiting for it.\n'
    wait_for_dataset
    return 0
  fi
  if [[ -d "${DATASET}" ]] && find "${DATASET}" -mindepth 1 -print -quit \
    | grep -q .; then
    fail "the local scan dataset is incomplete and no scanner is alive; preserve/move it for diagnosis before retrying"
  fi

  printf 'Stage scan: starting exact-contact 100 mm, 360 x 50 x 2 rollout.\n'
  run_managed env \
    CONTROLLER_MODE=diffusion-ik \
    EXPORT_SUCCESS_DEMOS=1 \
    ANGULAR_BINS=360 \
    RADIAL_BINS=50 \
    TRIALS_PER_CELL=2 \
    SPAWN_RADIUS_MM=100 \
    MAX_LIVE_WORLDS=7000 \
    "${SCAN_LAUNCHER}" run
  dataset_status \
    || fail "scan exited without a finalized successful-policy dataset"
}

validate_chain_artifact() {
  local stage="$1"
  "${PYTHON}" "${HPC_ROOT}/validate_w18000_chain.py" \
    --config "${CONFIG}" \
    --run-dir "${RUN_DIR}" \
    --dataset "${DATASET}" \
    --stage "${stage}"
}

run_diffusion_stage() {
  if validate_chain_artifact diffusion >/dev/null 2>&1; then
    printf 'Stage Diffusion: completed checkpoint verified; skipping.\n'
    validate_chain_artifact diffusion
    return 0
  fi
  local checkpoint="${RUN_DIR}/checkpoints/diffusion.pt"
  local arguments=(
    "${PYTHON}" "${HPC_ROOT}/train_diffusion_gpu.py"
    --config "${CONFIG}"
    --dataset "${DATASET}"
    --output "${checkpoint}"
    --log "${RUN_DIR}/logs/diffusion.jsonl"
    --run-dir "${RUN_DIR}"
    --allow-external-paths
  )
  if [[ ! -f "${checkpoint}" ]]; then
    arguments+=(--no-resume)
    printf 'Stage Diffusion: starting a fresh randomly initialized model.\n'
  else
    printf 'Stage Diffusion: resuming the exact dataset/checkpoint cursor.\n'
  fi
  run_managed "${arguments[@]}"
  validate_chain_artifact diffusion \
    || fail "Diffusion process exited without a completed checkpoint"
}

config_file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${CONFIG}" | awk '{print $1}'
  else
    shasum -a 256 "${CONFIG}" | awk '{print $1}'
  fi
}

preflight_marker_valid() {
  [[ -f "${PREFLIGHT_MARKER}" ]] || return 1
  "${PYTHON}" -c '
import hashlib,json,pathlib,sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
report=pathlib.Path(str(p.get("report", "")))
report_hash=(hashlib.sha256(report.read_bytes()).hexdigest()
             if report.is_file() else None)
ok=(p.get("stage")=="preflight" and p.get("status")=="complete"
    and p.get("config_file_sha256")==sys.argv[2]
    and str(p.get("memory_ceiling"))==sys.argv[3]
    and int(p.get("physics_steps",-1))==int(sys.argv[4])
    and int(p.get("worlds",-1))==18000
    and report_hash==p.get("report_sha256"))
raise SystemExit(0 if ok else 1)
' "${PREFLIGHT_MARKER}" "$(config_file_sha256)" \
    "${MEMORY_CEILING}" "${PREFLIGHT_STEPS}"
}

write_preflight_marker() {
  local report="$1" temporary="${PREFLIGHT_MARKER}.tmp.$$"
  "${PYTHON}" -c '
import hashlib,json,os,sys
target,report,config_sha,ceiling,steps=sys.argv[1:]
h=hashlib.sha256(open(report,"rb").read()).hexdigest()
payload={"stage":"preflight","status":"complete","worlds":18000,
         "memory_ceiling":ceiling,"physics_steps":int(steps),
         "config_file_sha256":config_sha,"report":os.path.abspath(report),
         "report_sha256":h}
with open(target,"w",encoding="utf-8") as f:
    json.dump(payload,f,sort_keys=True); f.write("\n")
' "${temporary}" "${report}" "$(config_file_sha256)" \
    "${MEMORY_CEILING}" "${PREFLIGHT_STEPS}"
  mv -f -- "${temporary}" "${PREFLIGHT_MARKER}"
}

run_preflight_stage() {
  if preflight_marker_valid; then
    printf 'Stage preflight: matching 18,000-world gate already passed; skipping.\n'
    return 0
  fi
  local stamp report status
  stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
  report="${LOG_DIR}/w18000_diffusion_tqc_preflight_${stamp}.json"
  printf '%s\n' \
    "Stage preflight: worlds=18000, ceiling=${MEMORY_CEILING}, steps=${PREFLIGHT_STEPS}." \
    "Expert-rollout pure-timeout allowance=${MAX_EXPERT_TRUNCATION_FRACTION}; physical failures remain forbidden." \
    "This gate never reduces the requested world count."
  "${PYTHON}" "${HPC_ROOT}/training_stack_preflight.py" \
    --config "${CONFIG}" \
    --worlds 18000 \
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
  if ((STOP_REQUESTED)); then
    return 130
  fi
  "${PYTHON}" -m json.tool "${report}" 2>/dev/null || true
  if ((status != 0)); then
    fail "18,000-world TQC preflight failed with status ${status}; training was not started and worlds were not reduced"
  fi
  write_preflight_marker "${report}"
}

run_tqc_stage() {
  if validate_chain_artifact tqc >/dev/null 2>&1; then
    printf 'Stage TQC: completed 18,000-world policy verified; skipping.\n'
    validate_chain_artifact tqc
    return 0
  fi
  local checkpoint="${RUN_DIR}/checkpoints/residual_tqc.pt"
  local arguments=(
    "${PYTHON}" "${HPC_ROOT}/train_residual_gpu.py"
    --config "${CONFIG}"
    --diffusion "${RUN_DIR}/checkpoints/diffusion.pt"
    --output "${checkpoint}"
    --log "${RUN_DIR}/logs/tqc.jsonl"
    --run-dir "${RUN_DIR}"
  )
  if [[ ! -f "${checkpoint}" ]]; then
    arguments+=(--no-resume)
    printf 'Stage TQC: starting fresh online residual training at 18,000 worlds.\n'
  else
    printf 'Stage TQC: resuming the exact 18,000-world replay/rollout cursor.\n'
  fi
  run_managed "${arguments[@]}"
  validate_chain_artifact tqc \
    || fail "TQC process exited without the complete 5M-transition policy"
}

model_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
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
' "${EVALUATION_MARKER}" \
    "$(model_sha256 "${RUN_DIR}/checkpoints/diffusion.pt")" \
    "$(model_sha256 "${RUN_DIR}/checkpoints/residual_tqc_policy.pt")" \
    "${EVALUATION_EPISODES}"
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
' "${temporary}" "${attempt}" \
    "$(model_sha256 "${RUN_DIR}/checkpoints/diffusion.pt")" \
    "$(model_sha256 "${RUN_DIR}/checkpoints/residual_tqc_policy.pt")" \
    "${EVALUATION_EPISODES}"
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
    --diffusion "${RUN_DIR}/checkpoints/diffusion.pt" \
    --tqc "${RUN_DIR}/checkpoints/residual_tqc_policy.pt" \
    --episodes "${EVALUATION_EPISODES}" \
    --log "${attempt}/logs/evaluation.jsonl" \
    --run-dir "${attempt}"
  write_evaluation_marker "${attempt}"
}

run_static_validation() {
  for required in \
    "${CONFIG}" \
    "${SCAN_LAUNCHER}" \
    "${SCREW_SCAN_DIFFUSION:-${PACKAGE_ROOT}/hpc/runs/fresh/checkpoints/diffusion.pt}" \
    "${SCREW_SCAN_TQC:-${PACKAGE_ROOT}/hpc/runs/fresh/checkpoints/residual_tqc_policy.pt}" \
    "${SCREW_SCAN_CONFIG:-${PACKAGE_ROOT}/hpc/configs/autodl_rtx_pro_6000_r100_success_selfdistill_w18000_tqc16.json}" \
    "${HPC_ROOT}/validate_success_dataset.py" \
    "${HPC_ROOT}/validate_w18000_chain.py"; do
    [[ -f "${required}" ]] || fail "required package file is missing: ${required}"
  done
  validate_chain_artifact config
  "${PYTHON}" "${HPC_ROOT}/preflight.py" --mode static --config "${CONFIG}"
  EXPORT_SUCCESS_DEMOS=0 \
  ANGULAR_BINS=360 RADIAL_BINS=50 TRIALS_PER_CELL=2 SPAWN_RADIUS_MM=100 \
  MAX_LIVE_WORLDS=7000 \
    "${SCAN_LAUNCHER}" validate
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
  flock -n 9 || fail "another w18000 chain owns ${LOCK_FILE}"
  trap 'handle_signal' TERM HUP INT
  trap 'finish_wrapper $?' EXIT

  run_static_validation
  run_scan_stage
  run_diffusion_stage
  start_system_monitor
  run_preflight_stage
  run_tqc_stage
  stop_system_monitor
  run_evaluation_stage
  printf '%s\n' \
    "Complete chain finished." \
    "Diffusion: ${RUN_DIR}/checkpoints/diffusion.pt" \
    "TQC policy: ${RUN_DIR}/checkpoints/residual_tqc_policy.pt"
}

background_start() {
  prepare_environment
  validate_options
  check_disk_space
  local existing=""
  if existing="$(launcher_pid)"; then
    fail "the w18000 chain is already running as PID ${existing}"
  fi
  rm -f -- "${PID_FILE}"
  : >>"${CHAIN_LOG}"
  nohup setsid "${PACKAGE_ROOT}/run_autodl_r100_success_selftrain_w18000_tqc16.sh" \
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
    "AutoDL r100 success-self-training chain launched as PID ${pid}." \
    "Formal/TQC worlds: 18000 (no automatic fallback)" \
    "Dataset: ${DATASET}" \
    "Run: ${RUN_DIR}" \
    "Log: ${CHAIN_LOG}" \
    "Status: ./run_autodl_r100_success_selftrain_w18000_tqc16.sh status"
}

show_status() {
  prepare_environment
  local pid="" dataset_result="" dataset_code
  if pid="$(launcher_pid)"; then
    printf 'launcher_pid=%s state=alive\n' "${pid}"
  else
    printf 'launcher_state=not_running\n'
  fi
  if dataset_result="$(dataset_status 2>&1)"; then
    dataset_code=0
  else
    dataset_code=$?
  fi
  printf 'dataset_status=%s path=%s\n' \
    "$([[ ${dataset_code} -eq 0 ]] && printf ready || printf incomplete_or_invalid)" \
    "${DATASET}"
  [[ -z "${dataset_result}" ]] || printf '%s\n' "${dataset_result}"
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
    printf 'No training stage has started yet.\n'
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
      --reason "requested through the AutoDL w18000 self-training wrapper" \
      || true
  fi
  local pid=""
  if pid="$(launcher_pid)"; then
    printf 'Sending SIGTERM to wrapper PID %s.\n' "${pid}"
    kill -TERM "${pid}"
    return 0
  fi
  fail "no live w18000 chain process was found"
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
