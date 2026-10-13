#!/usr/bin/env bash
# Submit one allocation that performs capacity -> smoke -> formal training.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

# shellcheck source=hpc_env.sh
source "${SCRIPT_DIR}/hpc_env.sh"
screw_use_core_env

usage() {
  printf '%s\n' \
    "Usage:" \
    "  ./submit_cx3_adaptive_any.sh" \
    "  ./submit_cx3_adaptive_any.sh --resume ADAPTIVE_ID"
}

ADAPTIVE_ID=""
if (($#)); then
  if [[ "$#" -ne 2 || "$1" != "--resume" ]]; then
    usage >&2
    exit 64
  fi
  ADAPTIVE_ID="$2"
else
  ADAPTIVE_ID="anygpu_$(date -u '+%Y%m%dT%H%M%SZ')"
fi
if [[ ! "${ADAPTIVE_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'ERROR: invalid adaptive id: %s\n' "${ADAPTIVE_ID}" >&2
  exit 64
fi
if ! command -v qsub >/dev/null 2>&1; then
  printf '%s\n' "ERROR: qsub is unavailable; run this on a CX3 login node." >&2
  exit 69
fi
JOB_SCRIPT="${SCRIPT_DIR}/jobs/cx3_adaptive_any.pbs"
if [[ ! -f "${JOB_SCRIPT}" ]]; then
  printf 'ERROR: missing %s\n' "${JOB_SCRIPT}" >&2
  exit 66
fi

mkdir -p "${SCRIPT_DIR}/runs/submissions"
STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
RECORD="${SCRIPT_DIR}/runs/submissions/adaptive_any_${ADAPTIVE_ID}_${STAMP}.env"
TEMPORARY="${RECORD}.tmp.$$"
JOB_ID="$(
  qsub -v \
    "SCREW_ADAPTIVE_ID=${ADAPTIVE_ID},IRP_HPC_CORE_ENV=${IRP_HPC_CORE_ENV}" \
    "${JOB_SCRIPT}"
)"
{
  printf 'SUBMITTED_UTC=%q\n' "${STAMP}"
  printf 'ADAPTIVE_ID=%q\n' "${ADAPTIVE_ID}"
  printf 'ADAPTIVE_JOB=%q\n' "${JOB_ID}"
  printf 'SELECTION_DIR=%q\n' \
    "runs/cx3_adaptive_${ADAPTIVE_ID}_full/capacity_selection"
  printf 'SMOKE_RUN_DIR=%q\n' \
    "runs/cx3_adaptive_${ADAPTIVE_ID}_smoke"
  printf 'FULL_RUN_DIR=%q\n' \
    "runs/cx3_adaptive_${ADAPTIVE_ID}_full"
} >"${TEMPORARY}"
mv -f -- "${TEMPORARY}" "${RECORD}"
cp -f -- "${RECORD}" \
  "${SCRIPT_DIR}/runs/submissions/latest_adaptive_any.env"

printf '%s\n' \
  "adaptive any-GPU job: ${JOB_ID}" \
  "adaptive id:          ${ADAPTIVE_ID}" \
  "saved job record:     ${RECORD}" \
  "Reload later with:" \
  "  source runs/submissions/latest_adaptive_any.env" \
  "  qstat -f \"\$ADAPTIVE_JOB\"" \
  "Live training status:" \
  "  ./hpc_python.sh status_hpc.py --run-dir \"\$FULL_RUN_DIR\" --watch 10" \
  "Capacity result:" \
  "  ./hpc_python.sh -m json.tool \"\$SELECTION_DIR/capacity.json\"" \
  "Resume the exact selected config after a walltime stop:" \
  "  ./submit_cx3_adaptive_any.sh --resume ${ADAPTIVE_ID}"

qstat -u "${USER}"
