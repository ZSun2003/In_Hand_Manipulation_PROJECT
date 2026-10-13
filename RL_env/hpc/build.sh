#!/usr/bin/env bash
# Build the Linux/CUDA core runtime profile once, in their final persistent path.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROFILE="all"
PYTHON_BIN="${PYTHON_BIN:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
WHEELHOUSE="${WHEELHOUSE:-}"
REQUIRE_GPU=0

usage() {
  cat <<'EOF'
Usage: bash build.sh [--profile core|all] [--require-gpu]

Environment overrides:
  PYTHON_BIN=/absolute/path/to/python3.11-or-python3.12
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
  PYPI_INDEX_URL=https://pypi.org/simple
  WHEELHOUSE=/absolute/path/to/linux/wheels

Build this environment on Linux x86_64 in its final persistent directory.
Do not run this script on macOS and do not point it at a project's local
macOS RL_env.
EOF
}

while (($#)); do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || {
        printf '%s\n' "ERROR: --profile needs a value" >&2
        exit 2
      }
      PROFILE="$2"
      shift 2
      ;;
    --require-gpu)
      REQUIRE_GPU=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'ERROR: unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$PROFILE" in
  core|all) ;;
  *)
    printf 'ERROR: invalid profile: %s\n' "$PROFILE" >&2
    exit 2
    ;;
esac

[[ "$(uname -s)" == "Linux" ]] || {
  printf '%s\n' \
    "ERROR: the CUDA environment must be built on Linux, not macOS." >&2
  exit 64
}
case "$(uname -m)" in
  x86_64|amd64) ;;
  *)
    printf 'ERROR: x86_64 is required, found %s\n' "$(uname -m)" >&2
    exit 64
    ;;
esac

resolve_python() {
  local candidate resolved
  if [[ -n "$PYTHON_BIN" ]]; then
    command -v "$PYTHON_BIN"
    return
  fi
  for candidate in \
    python3.11 \
    python3.12 \
    /root/miniconda3/bin/python \
    /opt/conda/bin/python \
    python3; do
    resolved="$(command -v "$candidate" 2>/dev/null)" || continue
    if "$resolved" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(
    0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1
)
PY
    then
      printf '%s\n' "$resolved"
      return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(resolve_python)"; then
  printf '%s\n' \
    "ERROR: no supported CPython was found." \
    "Install CPython 3.11/3.12 or set PYTHON_BIN to its absolute path." >&2
  exit 69
fi
"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        "The shared bundle requires CPython 3.11 or 3.12; found "
        + ".".join(map(str, sys.version_info[:3]))
    )
print("bootstrap_python", sys.executable)
print("bootstrap_version", sys.version.split()[0])
PY

mkdir -p "$ROOT/../profiles" "$ROOT/state" "$ROOT/cache/pip"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/cache/pip}"
export PYTHONNOUSERSITE=1
umask 022

if [[ -n "$WHEELHOUSE" ]]; then
  [[ -d "$WHEELHOUSE" ]] || {
    printf 'ERROR: WHEELHOUSE does not exist: %s\n' "$WHEELHOUSE" >&2
    exit 66
  }
  WHEELHOUSE="$(cd -- "$WHEELHOUSE" && pwd -P)"
fi

torch_version() {
  case "$1" in
    core) printf '%s\n' "2.7.1" ;;
  esac
}

setuptools_version() {
  case "$1" in
    core) printf '%s\n' "83.0.0" ;;
  esac
}

torch_profile_ready() {
  local python="$1"
  local wanted="$2"
  "$python" - "$wanted" <<'PY' >/dev/null 2>&1
import sys

try:
    import torch
except Exception:
    raise SystemExit(1)

actual = torch.__version__.split("+", 1)[0]
raise SystemExit(
    0 if actual == sys.argv[1] and torch.version.cuda == "12.8" else 1
)
PY
}

install_profile() {
  local name="$1"
  # Keep the actual venv basename as RL_env. SCREW's existing fail-closed
  # runtime contract resolves symlinks and requires precisely that basename.
  local env_dir="$ROOT/../profiles/$name/RL_env"
  local requirements="$ROOT/requirements/$name.txt"
  local python="$env_dir/bin/python"
  local torch setuptools
  torch="$(torch_version "$name")"
  setuptools="$(setuptools_version "$name")"

  printf '\n=== Building profile: %s ===\n' "$name"
  if [[ -e "$env_dir" && ! -x "$python" ]]; then
    printf '%s\n' \
      "ERROR: $env_dir exists but is not a usable venv." \
      "Move it aside and rerun; this script will not delete it." >&2
    return 65
  fi
  if [[ ! -e "$env_dir" ]]; then
    "$PYTHON_BIN" -m venv --copies "$env_dir"
  fi

  if [[ -n "$WHEELHOUSE" ]]; then
    "$python" -m pip install \
      --no-index --find-links "$WHEELHOUSE" \
      "pip==26.1.2" "setuptools==$setuptools" "wheel==0.47.0"
    if torch_profile_ready "$python" "$torch"; then
      printf 'Reusing verified Torch %s + CUDA 12.8 in %s\n' \
        "$torch" "$env_dir"
    else
      "$python" -m pip install \
        --force-reinstall \
        --no-index --find-links "$WHEELHOUSE" \
        "torch==${torch}+cu128"
    fi
    "$python" -m pip install \
      --no-index --find-links "$WHEELHOUSE" \
      --requirement "$requirements"
  else
    "$python" -m pip install \
      --index-url "$PYPI_INDEX_URL" \
      "pip==26.1.2" "setuptools==$setuptools" "wheel==0.47.0"
    if torch_profile_ready "$python" "$torch"; then
      printf 'Reusing verified Torch %s + CUDA 12.8 in %s\n' \
        "$torch" "$env_dir"
    else
      "$python" -m pip install \
        --force-reinstall \
        --index-url "$TORCH_INDEX_URL" \
        "torch==${torch}+cu128"
    fi
    "$python" -m pip install \
      --index-url "$PYPI_INDEX_URL" \
      --extra-index-url "$TORCH_INDEX_URL" \
      --requirement "$requirements"
  fi

  "$python" -m pip check
  verify_args=("$name")
  if [[ "$REQUIRE_GPU" == "1" ]]; then
    verify_args+=(--require-gpu)
  fi
  "$python" "$ROOT/verify.py" "${verify_args[@]}"
  "$python" -m pip freeze --all \
    >"$ROOT/state/$name-installed-freeze.txt"
  {
    printf 'profile=%s\n' "$name"
    printf 'created_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'root=%s\n' "$ROOT"
    printf 'host=%s\n' "$(hostname)"
    printf 'kernel=%s\n' "$(uname -srmo)"
    printf 'python=%s\n' "$("$python" -c 'import platform; print(platform.python_version())')"
    printf 'torch=%s\n' "$torch"
    printf 'torch_index_url=%s\n' "$TORCH_INDEX_URL"
    sha256_file "$requirements"
  } >"$ROOT/state/$name-build-manifest.txt"
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path"
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path"
  else
    printf 'sha256=unavailable file=%s\n' "$path"
  fi
}

if [[ "$PROFILE" == "core" || "$PROFILE" == "all" ]]; then
  install_profile core
fi

printf '\n%s\n' \
  "Build complete: $ROOT" \
  "Verify on a GPU node: \"$ROOT/../profiles/core/RL_env/bin/python\" \"$ROOT/verify.py\" core --require-gpu" \
  "Run a project: \"$ROOT/../run.sh\" diffusion-hpc preflight.py --mode static --config configs/smoke.json"
