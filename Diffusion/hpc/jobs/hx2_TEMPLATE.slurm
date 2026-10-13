#!/usr/bin/env bash
#SBATCH --job-name=screw_mjw_hx2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#
# INTENTIONALLY MISSING:
#   --partition=<confirmed HX2 GPU partition>
#   --gpus=<confirmed HX2 GPU resource syntax>
#
# HX2 is a pilot system and its site-specific resource directives must be
# confirmed with `sinfo`, `scontrol show partition`, and the current Imperial
# RCS documentation before this template is submitted.

set -euo pipefail

HX2_DIRECTIVES_CONFIRMED="NO_REPLACE_WITH_YES"
CONFIG_RELATIVE="configs/REPLACE_WITH_CONFIRMED_HX2_CONFIG.json"

if [[ "${HX2_DIRECTIVES_CONFIRMED}" != "YES" ]]; then
  printf '%s\n' \
    "STOP: this is a non-submittable HX2 template." \
    "Add the confirmed #SBATCH partition/GPU directives, choose a config," \
    "then set HX2_DIRECTIVES_CONFIRMED=YES."
  exit 64
fi

SUBMIT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
# shellcheck source=_common.sh
source "${SUBMIT_DIR}/jobs/_common.sh"
screw_hpc_init

CONFIG="${SCREW_HPC_ROOT}/${CONFIG_RELATIVE}"
RUN_DIR="${SCREW_HPC_ROOT}/runs/hx2_full"
screw_require_file "${SCREW_HPC_ROOT}/preflight.py"
screw_require_file "${SCREW_HPC_ROOT}/pipeline.py"
screw_require_file "${CONFIG}"

screw_start_live_monitor "${RUN_DIR}"
screw_record_job \
  "${RUN_DIR}" preflight "" "Running the confirmed HX2 CUDA preflight."

"${SCREW_PYTHON}" "${SCREW_HPC_ROOT}/preflight.py" \
  --mode gpu \
  --config "${CONFIG}" \
  --worlds 128

screw_record_job \
  "${RUN_DIR}" running "" "HX2 formal pipeline is running."

"${SCREW_PYTHON}" "${SCREW_HPC_ROOT}/pipeline.py" \
  --config "${CONFIG}" \
  --stage full \
  --resume auto \
  --run-dir "${RUN_DIR}"

screw_record_job \
  "${RUN_DIR}" completed 0 "HX2 formal pipeline completed."
