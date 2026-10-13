#!/usr/bin/env bash
# Submit the independent aggressive 8192-world CX3 dependency chain.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

# shellcheck source=hpc_env.sh
source "${SCRIPT_DIR}/hpc_env.sh"
screw_use_core_env

if ! command -v qsub >/dev/null 2>&1; then
  printf '%s\n' "ERROR: qsub is unavailable; run this on a CX3 login node." >&2
  exit 69
fi
for job in \
  cx3_capacity_8192.pbs \
  cx3_smoke_8192.pbs \
  cx3_full_8192.pbs; do
  if [[ ! -f "${SCRIPT_DIR}/jobs/${job}" ]]; then
    printf 'ERROR: missing jobs/%s\n' "${job}" >&2
    exit 66
  fi
done

mkdir -p "${SCRIPT_DIR}/runs/submissions"
STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
RECORD="${SCRIPT_DIR}/runs/submissions/cx3_8192_chain_${STAMP}.env"
TEMPORARY="${RECORD}.tmp.$$"
CAP_JOB=""
SMOKE_JOB=""
FULL_JOB=""

write_record() {
  {
    printf 'SUBMITTED_UTC=%q\n' "${STAMP}"
    printf 'CAP_JOB=%q\n' "${CAP_JOB}"
    printf 'SMOKE_JOB=%q\n' "${SMOKE_JOB}"
    printf 'FULL_JOB=%q\n' "${FULL_JOB}"
  } > "${TEMPORARY}"
  mv -f -- "${TEMPORARY}" "${RECORD}"
  cp -f -- "${RECORD}" \
    "${SCRIPT_DIR}/runs/submissions/latest_8192.env"
}

CAP_JOB="$(
  qsub -v "IRP_HPC_CORE_ENV=${IRP_HPC_CORE_ENV}" \
    "${SCRIPT_DIR}/jobs/cx3_capacity_8192.pbs"
)"
write_record
printf 'aggressive capacity job: %s\n' "${CAP_JOB}"
SMOKE_JOB="$(
  qsub -W "depend=afterok:${CAP_JOB}" \
    -v "IRP_HPC_CORE_ENV=${IRP_HPC_CORE_ENV}" \
    "${SCRIPT_DIR}/jobs/cx3_smoke_8192.pbs"
)"
write_record
printf 'aggressive smoke job:    %s (afterok:%s)\n' \
  "${SMOKE_JOB}" "${CAP_JOB}"
FULL_JOB="$(
  qsub -W "depend=afterok:${SMOKE_JOB}" \
    -v "IRP_HPC_CORE_ENV=${IRP_HPC_CORE_ENV}" \
    "${SCRIPT_DIR}/jobs/cx3_full_8192.pbs"
)"
write_record
printf 'aggressive formal job:   %s (afterok:%s)\n' \
  "${FULL_JOB}" "${SMOKE_JOB}"

printf '%s\n' \
  "Saved aggressive job IDs: ${RECORD}" \
  "Reload later with:" \
  "  source runs/submissions/latest_8192.env" \
  "  qstat -f \"\$FULL_JOB\"" \
  "Training status after the formal allocation starts:" \
  "  ./hpc_python.sh status_hpc.py --run-dir runs/cx3_l40s_8192_full --watch 10"

qstat -u "${USER}"
