#!/usr/bin/env bash
# Resolve the shared, externally managed IRP core environment.
#
# This file is sourced by launchers and scheduler jobs.  It deliberately has
# no install/update operation: project code may read the central environment,
# but must never mutate it.

SCREW_HPC_PACKAGE_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P
)"
SCREW_DEFAULT_CORE_ENV="${SCREW_HPC_PACKAGE_DIR}/../../RL_env/profiles/core/RL_env"

screw_resolve_core_env() {
  local requested
  requested="${IRP_HPC_CORE_ENV:-${SCREW_DEFAULT_CORE_ENV}}"
  case "${requested}" in
    /*) ;;
    *)
      printf 'ERROR: IRP_HPC_CORE_ENV must be an absolute path: %s\n' \
        "${requested}" >&2
      return 64
      ;;
  esac
  if [[ ! -d "${requested}" ]]; then
    printf '%s\n' \
      "ERROR: shared IRP core environment is missing: ${requested}" \
      "Build the sibling RL_env HPC core profile once, or set" \
      "IRP_HPC_CORE_ENV to the compatible external core/RL_env." >&2
    return 65
  fi

  local resolved
  resolved="$(cd -- "${requested}" && pwd -P)"
  case "${resolved}" in
    "${SCREW_HPC_PACKAGE_DIR}"|"${SCREW_HPC_PACKAGE_DIR}"/*)
      printf '%s\n' \
        "ERROR: refusing a project-local HPC environment: ${resolved}" \
        "SCREW must reuse the external IRP core environment." >&2
      return 65
      ;;
  esac
  if [[ ! -x "${resolved}/bin/python" ]]; then
    printf 'ERROR: shared environment has no executable Python: %s\n' \
      "${resolved}/bin/python" >&2
    return 65
  fi
  printf '%s\n' "${resolved}"
}

screw_use_core_env() {
  SCREW_HPC_ENV="$(screw_resolve_core_env)" || return
  SCREW_PYTHON="${SCREW_HPC_ENV}/bin/python"
  IRP_HPC_CORE_ENV="${SCREW_HPC_ENV}"
  export SCREW_HPC_ENV SCREW_PYTHON IRP_HPC_CORE_ENV
  export VIRTUAL_ENV="${SCREW_HPC_ENV}"
  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1
  export PATH="${SCREW_HPC_ENV}/bin:${PATH}"
}

