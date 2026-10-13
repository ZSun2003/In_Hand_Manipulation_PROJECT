#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_autodl_astar_spline_ik.sh canary   # 128 first episodes
  ./run_autodl_astar_spline_ik.sh formal   # 2000 first episodes; requires a validated canary
  ./run_autodl_astar_spline_ik.sh all      # canary, then formal if the canary is structurally valid

Optional environment variables:
  IRP_HPC_CORE_ENV       relocated CUDA Python environment
  ASTAR_CONFIG           evaluation config JSON
  ASTAR_RUN_ROOT         parent directory for unique result directories
  ASTAR_CANARY_SUMMARY   explicit canary summary used by formal mode
  ASTAR_MAX_STEPS        optional limit, no larger than config max_episode_steps
EOF
}

MODE="${1:-canary}"
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
case "$MODE" in
  canary|formal|all) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVALUATOR="$SCRIPT_DIR/hpc/evaluate_astar_spline_gpu.py"
DEFAULT_CONFIG="$SCRIPT_DIR/hpc/configs/autodl_rtx_pro_6000_r100_w5000_tqc16_continuation.json"
CONFIG_PATH="${ASTAR_CONFIG:-$DEFAULT_CONFIG}"
IRP_HPC_CORE_ENV="${IRP_HPC_CORE_ENV:-${SCRIPT_DIR}/../RL_env/profiles/core/RL_env}"
ASTAR_RUN_ROOT="${ASTAR_RUN_ROOT:-/root/autodl-tmp/astar_spline_ik_runs}"
PYTHON_BIN="$IRP_HPC_CORE_ENV/bin/python"
CANARY_POINTER="$ASTAR_RUN_ROOT/last_validated_canary_summary.txt"

export IRP_HPC_CORE_ENV
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$IRP_HPC_CORE_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "CUDA Python is not executable: $PYTHON_BIN" >&2
  echo "Set IRP_HPC_CORE_ENV to the relocated core environment." >&2
  exit 1
fi
if [[ ! -f "$EVALUATOR" ]]; then
  echo "Evaluator is missing: $EVALUATOR" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config is missing: $CONFIG_PATH" >&2
  exit 1
fi
mkdir -p "$ASTAR_RUN_ROOT"

validate_canary() {
  local summary_path="$1"
  if [[ ! -s "$summary_path" ]]; then
    echo "Canary summary is missing or empty: $summary_path" >&2
    return 1
  fi
  "$PYTHON_BIN" -c '
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve()
data = json.loads(path.read_text(encoding="utf-8"))
assert data.get("schema") == "screw-astar-spline-ik-summary-v1", data.get("schema")
assert data.get("control_stack") == "astar_spline_safety_analytic_ik"
assert data.get("residual") == "disabled_zero_16d"
assert int(data.get("episodes", -1)) == 128
counted = sum(int(data.get(key, -1)) for key in ("successes", "failures", "truncations"))
assert counted == 128, counted
manifest = pathlib.Path(data["initial_conditions_manifest"]["path"])
assert manifest.is_file(), manifest
assert data["provenance"]["checkpoints"] == {}
assert data["provenance"]["residual"]["mode"] == "disabled_zero"
print(
    "validated canary: success={}/128, failure={}, truncation={}".format(
        data["successes"], data["failures"], data["truncations"]
    )
)
' "$summary_path"
}

new_run_dir() {
  local label="$1"
  local stamp
  local candidate
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  candidate="$ASTAR_RUN_ROOT/astar_spline_ik_${label}_${stamp}"
  if [[ -e "$candidate" ]]; then
    candidate="${candidate}_pid$$"
  fi
  printf '%s\n' "$candidate"
}

run_evaluation() {
  local label="$1"
  local episodes="$2"
  local run_dir
  local launcher_log
  local evaluator_status
  local tee_status
  local -a command
  local -a pipeline_status
  run_dir="$(new_run_dir "$label")"
  launcher_log="$ASTAR_RUN_ROOT/.${run_dir##*/}_launcher_$$.log"
  LAST_RUN_DIR="$run_dir"
  command=(
    "$PYTHON_BIN" -u "$EVALUATOR"
    --config "$CONFIG_PATH"
    --episodes "$episodes"
    --worlds "$episodes"
    --run-dir "$run_dir"
    --log "$run_dir/logs/astar_evaluation.jsonl"
  )
  if [[ -n "${ASTAR_MAX_STEPS:-}" ]]; then
    command+=(--max-steps "$ASTAR_MAX_STEPS")
  fi

  # The run directory must remain completely absent until RunMonitor creates
  # its immutable manifest.  Keep the launcher transcript beside the run
  # directory temporarily, then move it inside only after monitor startup.
  set +e
  {
    echo "mode=$label"
    echo "episodes=$episodes"
    echo "config=$CONFIG_PATH"
    echo "run_dir=$run_dir"
    echo "python=$PYTHON_BIN"
    "${command[@]}"
  } 2>&1 | tee "$launcher_log"
  pipeline_status=("${PIPESTATUS[@]}")
  evaluator_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"
  set -e

  if [[ -d "$run_dir" ]]; then
    mv -f "$launcher_log" "$run_dir/launcher_console.log"
    {
      echo "mode=$label"
      echo "episodes=$episodes"
      echo "config=$CONFIG_PATH"
      echo "run_dir=$run_dir"
      echo "python=$PYTHON_BIN"
    } > "$run_dir/launcher_metadata.txt"
  else
    echo "Evaluator exited before creating its run manifest." >&2
    echo "Launcher transcript retained at: $launcher_log" >&2
  fi
  if [[ "$tee_status" -ne 0 ]]; then
    echo "Launcher transcript write failed with status $tee_status" >&2
    return "$tee_status"
  fi
  if [[ "$evaluator_status" -ne 0 ]]; then
    return "$evaluator_status"
  fi
}

run_canary() {
  local run_dir
  local summary_path
  run_evaluation canary 128
  run_dir="$LAST_RUN_DIR"
  summary_path="$run_dir/summary.json"
  validate_canary "$summary_path"
  printf '%s\n' "$summary_path" > "$CANARY_POINTER"
  echo "Canary passed structural validation: $summary_path"
  echo "Review its success/failure metrics before formal evaluation."
}

resolve_canary_summary() {
  local summary_path="${ASTAR_CANARY_SUMMARY:-}"
  if [[ -z "$summary_path" && -s "$CANARY_POINTER" ]]; then
    IFS= read -r summary_path < "$CANARY_POINTER"
  fi
  if [[ -z "$summary_path" ]]; then
    echo "No validated canary was found." >&2
    echo "Run '$0 canary' first, or set ASTAR_CANARY_SUMMARY." >&2
    return 1
  fi
  validate_canary "$summary_path" >&2
  printf '%s\n' "$summary_path"
}

run_formal() {
  local canary_summary
  local run_dir
  canary_summary="$(resolve_canary_summary)"
  echo "Formal gate uses canary: $canary_summary"
  run_evaluation formal 2000
  run_dir="$LAST_RUN_DIR"
  echo "Formal evaluation completed: $run_dir/summary.json"
}

case "$MODE" in
  canary)
    run_canary
    ;;
  formal)
    run_formal
    ;;
  all)
    run_canary
    run_formal
    ;;
esac
