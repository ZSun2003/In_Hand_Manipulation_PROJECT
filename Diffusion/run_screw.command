#!/bin/zsh
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PYTHON="$SCRIPT_DIR/../RL_env/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  print -u2 "Shared Python missing: $PYTHON; see ../RL_env/README.md."
  exit 1
fi
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
exec "$PYTHON" "$SCRIPT_DIR/runtime.py" "$@"
