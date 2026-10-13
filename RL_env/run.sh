#!/usr/bin/env bash
# Select a shared environment and project without making per-project venvs.
set -Eeuo pipefail
ENV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
IRP_ROOT="$(cd -- "$ENV_ROOT/.." && pwd -P)"
PROJECT="${1:---help}"
if [[ "$PROJECT" == --help || "$PROJECT" == -h ]]; then
  cat <<'EOF'
Usage: RL_env/run.sh PROJECT PYTHON_ARGS...
Projects: local, core, ik-residual, diffusion,
          diffusion-mps, diffusion-hpc, grab-g3, grab-g6
RL_BACKEND=local|mps|cpu|hpc overrides automatic platform selection.
RL_PYTHON=/absolute/python overrides the selected interpreter.
EOF
  exit 0
fi
shift
BACKEND="${RL_BACKEND:-}"
if [[ -z "$BACKEND" ]]; then
  if [[ "$(uname -s)" == Darwin ]]; then BACKEND=local; else BACKEND=hpc; fi
fi
case "$BACKEND" in local|mps|cpu|hpc) ;; *) echo "Invalid RL_BACKEND: $BACKEND" >&2; exit 2;; esac
WORK_DIR="$PWD"
PYTHON="$ENV_ROOT/bin/python"
SOURCE_PATH=""
case "$PROJECT" in
  local|mps) ;;
  core) PYTHON="$ENV_ROOT/profiles/core/RL_env/bin/python" ;;
  ik-residual) WORK_DIR="$IRP_ROOT/IK-residual" ;;
  diffusion|diffusion-mps|diffusion-hpc)
    WORK_DIR="$IRP_ROOT/Diffusion"
    if [[ "$PROJECT" == diffusion-hpc || ( "$PROJECT" == diffusion && "$BACKEND" == hpc ) ]]; then
      WORK_DIR="$WORK_DIR/hpc"
      PYTHON="$ENV_ROOT/profiles/core/RL_env/bin/python"
    fi
    ;;
  grab-g3|grab-g6)
    if [[ "$PROJECT" == grab-g3 ]]; then WORK_DIR="$IRP_ROOT/Diffusion_FutureWork/benchmark";
    else WORK_DIR="$IRP_ROOT/Diffusion_FutureWork/rl16_api"; fi
    if [[ "$BACKEND" == hpc ]]; then PYTHON="$ENV_ROOT/profiles/core/RL_env/bin/python"; fi
    ;;
  *) echo "Unknown project: $PROJECT (see --help)" >&2; exit 2 ;;
esac
PYTHON="${RL_PYTHON:-$PYTHON}"
[[ -x "$PYTHON" ]] || { echo "Missing interpreter: $PYTHON. See RL_env/README.md." >&2; exit 65; }
[[ -d "$WORK_DIR" ]] || { echo "Missing project: $WORK_DIR" >&2; exit 66; }
export IRP_HPC_ENV_ROOT="$ENV_ROOT" IRP_RL_ENV_ROOT="$ENV_ROOT"
export IRP_PYTHON_ENV="$(cd -- "$(dirname -- "$PYTHON")/.." && pwd -P)"
export VIRTUAL_ENV="$IRP_PYTHON_ENV"
export PATH="$(dirname -- "$PYTHON"):$PATH"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SOURCE_PATH:-$WORK_DIR}${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORK_DIR"
exec "$PYTHON" "$@"
