#!/usr/bin/env bash
set -Eeuo pipefail
ENV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: bash RL_env/build.sh --profile local|core'
  echo 'local: MuJoCo 3.7 + Torch 2.11 + SB3 on Mac/Linux; core: Linux CUDA/MJWarp.'
  exit 0
fi
PROFILE=local
if [[ "${1:-}" == --profile ]]; then PROFILE="${2:?--profile requires a value}"; shift 2; fi
case "$PROFILE" in
  core|all) exec bash "$ENV_ROOT/hpc/build.sh" --profile "$PROFILE" "$@" ;;
  local|mps) ;;
  *) echo 'Usage: build.sh --profile local|core' >&2; exit 2 ;;
esac
[[ $# == 0 ]] || { echo 'Unexpected argument' >&2; exit 2; }
TARGET="$ENV_ROOT"
REQS=local
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Requires Python 3.11 or 3.12"'
if [[ ! -x "$TARGET/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$TARGET"
fi
"$TARGET/bin/python" -m pip install -r "$ENV_ROOT/requirements/$REQS.txt"
"$TARGET/bin/python" -m pip check
"$TARGET/bin/python" -B "$ENV_ROOT/doctor.py"
