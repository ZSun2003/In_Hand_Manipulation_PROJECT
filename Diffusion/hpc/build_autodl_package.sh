#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PACKAGE_PARENT="$(dirname -- "${SCRIPT_DIR}")"
PACKAGE_NAME="$(basename -- "${SCRIPT_DIR}")"
STAMP="${SCREW_PACKAGE_STAMP:-$(date -u '+%Y%m%d')}"
DEFAULT_ARCHIVE_NAME="hpc_AUTODL_H110_MOUTH330_W18000_EP4000_BLEND100_HX1_COMPAT_${STAMP}.tar.gz"
ARCHIVE_NAME="${SCREW_PACKAGE_ARCHIVE_NAME:-${DEFAULT_ARCHIVE_NAME}}"
OUTPUT_DIR="${SCREW_PACKAGE_OUTPUT_DIR:-${PACKAGE_PARENT}}"
INCLUDE_MODELS="${SCREW_PACKAGE_INCLUDE_MODELS:-0}"
REQUIRE_MODELS="${SCREW_PACKAGE_REQUIRE_MODELS:-0}"

if [[ "$(basename -- "${ARCHIVE_NAME}")" != "${ARCHIVE_NAME}" ]] \
  || [[ "${ARCHIVE_NAME}" != *.tar.gz ]]; then
  printf 'ERROR: invalid archive filename: %s\n' "${ARCHIVE_NAME}" >&2
  exit 64
fi

case "${INCLUDE_MODELS}" in
  0|1) ;;
  *)
    printf '%s\n' \
      "ERROR: SCREW_PACKAGE_INCLUDE_MODELS must be 0 or 1." >&2
    exit 64
    ;;
esac
case "${REQUIRE_MODELS}" in
  0|1) ;;
  *)
    printf '%s\n' \
      "ERROR: SCREW_PACKAGE_REQUIRE_MODELS must be 0 or 1." >&2
    exit 64
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd -P)"
case "${OUTPUT_DIR}" in
  "${SCRIPT_DIR}"|"${SCRIPT_DIR}"/*)
    printf '%s\n' \
      "ERROR: SCREW_PACKAGE_OUTPUT_DIR must be outside ${SCRIPT_DIR}" >&2
    exit 64
    ;;
esac
ARCHIVE="${OUTPUT_DIR}/${ARCHIVE_NAME}"
CHECKSUM="${ARCHIVE}.sha256"
TEMP_ARCHIVE="$(mktemp "${OUTPUT_DIR}/.${ARCHIVE_NAME}.tmp.XXXXXX")"
TEMP_CHECKSUM="$(mktemp "${OUTPUT_DIR}/.${ARCHIVE_NAME}.sha256.tmp.XXXXXX")"
TEMP_MODEL_COUNT="$(mktemp "${OUTPUT_DIR}/.${ARCHIVE_NAME}.models.tmp.XXXXXX")"

cleanup() {
  if [[ -n "${TEMP_ARCHIVE:-}" ]]; then
    rm -f -- "${TEMP_ARCHIVE}"
  fi
  if [[ -n "${TEMP_CHECKSUM:-}" ]]; then
    rm -f -- "${TEMP_CHECKSUM}"
  fi
  if [[ -n "${TEMP_MODEL_COUNT:-}" ]]; then
    rm -f -- "${TEMP_MODEL_COUNT}"
  fi
}
trap cleanup EXIT

# Prevent macOS copyfile metadata from becoming hidden AppleDouble members
# when the archive is extracted on Linux.
export COPYFILE_DISABLE=1

TAR_METADATA_ARGS=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  # bsdtar otherwise synthesizes one ``._*`` member per source and stores
  # com.apple.provenance xattrs in PAX headers even when those names are
  # excluded below.
  TAR_METADATA_ARGS+=(--no-mac-metadata --no-xattrs)
fi

MODEL_EXCLUDE_ARGS=()
if [[ "${INCLUDE_MODELS}" == "0" ]]; then
  MODEL_EXCLUDE_ARGS+=(
    "--exclude=${PACKAGE_NAME}/runs"
    "--exclude=${PACKAGE_NAME}/checkpoints"
    "--exclude=${PACKAGE_NAME}/outputs"
  )
fi

# Detect arbitrarily named virtualenv/Conda roots before tar creation.  The
# marker alone is not enough: its entire parent directory must be excluded.
DISCOVERED_ENV_EXCLUDE_ARGS=()
DISCOVERED_ENV_PREFIXES=""
while IFS= read -r marker; do
  env_root="$(dirname -- "${marker}")"
  case "${env_root}" in
    "${SCRIPT_DIR}"/*) ;;
    *)
      printf 'ERROR: discovered environment escaped package: %s\n' \
        "${env_root}" >&2
      exit 65
      ;;
  esac
  relative_root="${PACKAGE_NAME}/${env_root#"${SCRIPT_DIR}/"}"
  DISCOVERED_ENV_EXCLUDE_ARGS+=(
    "--exclude=${relative_root}"
    "--exclude=${relative_root}/*"
  )
  DISCOVERED_ENV_PREFIXES+="${relative_root}"$'\n'
done < <(
  find "${SCRIPT_DIR}" \
    \( \
      -name RL_env -o \
      -name .venv -o \
      -name venv -o \
      -name env -o \
      -name .virtualenv -o \
      -name .python -o \
      -name _g6_python_runtime -o \
      -name .wheelhouse -o \
      -name wheelhouse \
    \) -prune -o \
    \( \
      \( -type f -name pyvenv.cfg \) -o \
      \( -type d -name conda-meta \) \
    \) -print
)

tar ${TAR_METADATA_ARGS[@]+"${TAR_METADATA_ARGS[@]}"} -C "${PACKAGE_PARENT}" \
  --exclude="${PACKAGE_NAME}/RL_env" \
  --exclude="${PACKAGE_NAME}/.venv" \
  --exclude="${PACKAGE_NAME}/venv" \
  --exclude="${PACKAGE_NAME}/env" \
  --exclude="${PACKAGE_NAME}/.virtualenv" \
  --exclude="${PACKAGE_NAME}/.python" \
  --exclude="${PACKAGE_NAME}/_g6_python_runtime" \
  --exclude="${PACKAGE_NAME}/.wheelhouse" \
  --exclude="${PACKAGE_NAME}/.cache" \
  --exclude="${PACKAGE_NAME}/.pytest_cache" \
  --exclude="${PACKAGE_NAME}/datasets" \
  --exclude="${PACKAGE_NAME}/logs" \
  --exclude="${PACKAGE_NAME}/wheelhouse" \
  --exclude="${PACKAGE_NAME}/environment-manifest.txt" \
  --exclude="*/RL_env" \
  --exclude="*/.venv" \
  --exclude="*/venv" \
  --exclude="*/env" \
  --exclude="*/.virtualenv" \
  --exclude="*/.python" \
  --exclude="*/_g6_python_runtime" \
  --exclude="*/.wheelhouse" \
  --exclude="*/wheelhouse" \
  --exclude="*/site-packages" \
  --exclude="*/conda-meta" \
  --exclude="*/pyvenv.cfg" \
  --exclude="*/__pycache__" \
  --exclude="*.pyc" \
  --exclude="*.pyo" \
  --exclude=".DS_Store" \
  --exclude="._*" \
  ${DISCOVERED_ENV_EXCLUDE_ARGS[@]+"${DISCOVERED_ENV_EXCLUDE_ARGS[@]}"} \
  ${MODEL_EXCLUDE_ARGS[@]+"${MODEL_EXCLUDE_ARGS[@]}"} \
  -czf "${TEMP_ARCHIVE}" \
  "${PACKAGE_NAME}"

AUDIT_PYTHON="${PYTHON_BOOTSTRAP:-python3}"
if ! command -v "${AUDIT_PYTHON}" >/dev/null 2>&1; then
  printf 'ERROR: archive audit Python not found: %s\n' \
    "${AUDIT_PYTHON}" >&2
  exit 69
fi
SCREW_ARCHIVE_TO_AUDIT="${TEMP_ARCHIVE}" \
SCREW_ARCHIVE_INCLUDES_MODELS="${INCLUDE_MODELS}" \
SCREW_ARCHIVE_REQUIRES_MODELS="${REQUIRE_MODELS}" \
SCREW_ARCHIVE_PACKAGE_NAME="${PACKAGE_NAME}" \
SCREW_ARCHIVE_MODEL_COUNT_FILE="${TEMP_MODEL_COUNT}" \
SCREW_DISCOVERED_ENV_PREFIXES="${DISCOVERED_ENV_PREFIXES}" \
  "${AUDIT_PYTHON}" - <<'PY'
import os
from pathlib import Path, PurePosixPath
import tarfile

archive = os.environ["SCREW_ARCHIVE_TO_AUDIT"]
include_models = os.environ["SCREW_ARCHIVE_INCLUDES_MODELS"] == "1"
require_models = os.environ["SCREW_ARCHIVE_REQUIRES_MODELS"] == "1"
package_name = os.environ["SCREW_ARCHIVE_PACKAGE_NAME"]
model_count_file = Path(os.environ["SCREW_ARCHIVE_MODEL_COUNT_FILE"])
discovered_prefixes = tuple(
    PurePosixPath(value)
    for value in os.environ.get(
        "SCREW_DISCOVERED_ENV_PREFIXES", ""
    ).splitlines()
    if value
)
generated = {
    ".python",
    ".venv",
    ".virtualenv",
    ".wheelhouse",
    ".cache",
    ".pytest_cache",
    "RL_env",
    "__pycache__",
    "datasets",
    "logs",
    "site-packages",
    "conda-meta",
    "venv",
    "env",
    "wheelhouse",
    "_g6_python_runtime",
}
if not include_models:
    generated.update({"checkpoints", "outputs", "runs"})

# Only native-HPC checkpoint files inside this hpc archive count as model
# artifacts.  In particular, sibling SCREW/checkpoints files produced by the
# macOS/MPS smoke workflow are outside the package and cannot satisfy
# SCREW_PACKAGE_REQUIRE_MODELS.
model_artifact_suffixes = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
}


def is_hpc_model_artifact(path):
    parts = path.parts
    if len(parts) < 3 or parts[0] != package_name:
        return False
    relative = parts[1:]
    if path.suffix.lower() not in model_artifact_suffixes:
        return False
    if relative[0] == "checkpoints":
        return True
    return (
        relative[0] == "runs"
        and "checkpoints" in relative[1:-1]
    )


problems = []
model_artifact_members = set()
with tarfile.open(archive, "r:gz") as stream:
    members = stream.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if member.isfile() and is_hpc_model_artifact(path):
            model_artifact_members.add(path.as_posix())
        if path.is_absolute() or ".." in path.parts:
            problems.append(f"unsafe path: {member.name}")
        if generated.intersection(path.parts):
            problems.append(f"generated tree: {member.name}")
        for prefix in discovered_prefixes:
            if path == prefix or prefix in path.parents:
                problems.append(
                    f"discovered environment leaked: {member.name}"
                )
        if path.name.startswith("._") or path.name == ".DS_Store":
            problems.append(f"macOS metadata member: {member.name}")
        if path.name == "environment-manifest.txt":
            problems.append(f"dynamic manifest: {member.name}")
        if path.name == "pyvenv.cfg":
            problems.append(f"virtualenv metadata: {member.name}")
        if path.suffix in {".pyc", ".pyo"}:
            problems.append(f"Python bytecode: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            problems.append(f"unsupported member type: {member.name}")
        xattrs = [
            key
            for key in member.pax_headers
            if "xattr" in key.lower()
        ]
        if xattrs:
            problems.append(
                f"extended attributes on {member.name}: {xattrs}"
            )
if not members:
    problems.append("archive is empty")
if require_models and not model_artifact_members:
    problems.append(
        "required model artifacts are missing from archive "
        "(model_artifact_members=0; "
        "SCREW_PACKAGE_REQUIRE_MODELS=1)"
    )
model_count_file.write_text(
    f"{len(model_artifact_members)}\n",
    encoding="utf-8",
)
print(
    f"model_artifact_members={len(model_artifact_members)}",
    flush=True,
)
if problems:
    raise SystemExit(
        "archive audit failed:\n  " + "\n  ".join(problems)
    )
print(f"archive_audit=pass members={len(members)}")
PY

MODEL_ARTIFACT_MEMBERS="$(
  tr -d '[:space:]' <"${TEMP_MODEL_COUNT}"
)"
case "${MODEL_ARTIFACT_MEMBERS}" in
  ""|*[!0-9]*)
    printf 'ERROR: invalid model artifact count: %s\n' \
      "${MODEL_ARTIFACT_MEMBERS}" >&2
    exit 70
    ;;
esac

if command -v sha256sum >/dev/null 2>&1; then
  DIGEST="$(sha256sum "${TEMP_ARCHIVE}" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  DIGEST="$(shasum -a 256 "${TEMP_ARCHIVE}" | awk '{print $1}')"
else
  printf '%s\n' "ERROR: sha256sum or shasum is required" >&2
  exit 69
fi
printf '%s  %s\n' "${DIGEST}" "${ARCHIVE_NAME}" >"${TEMP_CHECKSUM}"
chmod 0644 "${TEMP_ARCHIVE}" "${TEMP_CHECKSUM}"
mv -f -- "${TEMP_ARCHIVE}" "${ARCHIVE}"
TEMP_ARCHIVE=""
mv -f -- "${TEMP_CHECKSUM}" "${CHECKSUM}"
TEMP_CHECKSUM=""

printf '%s\n' \
  "Created: ${ARCHIVE}" \
  "Checksum: ${CHECKSUM}" \
  "Archive contents:"
tar -tzf "${ARCHIVE}"
