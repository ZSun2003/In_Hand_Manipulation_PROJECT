#!/usr/bin/env bash
# Source this script in bash or zsh.
_irp_activate_source="${BASH_SOURCE[0]:-${(%):-%x}}"
_irp_activate_root="$(cd -- "$(dirname -- "$_irp_activate_source")" && pwd -P)"
_irp_activate_profile="${1:-local}"
case "$_irp_activate_profile" in
  local|mps) _irp_activate_env="$_irp_activate_root" ;;
  core) _irp_activate_env="$_irp_activate_root/profiles/$_irp_activate_profile/RL_env" ;;
  *) printf 'Unknown profile: %s\n' "$_irp_activate_profile" >&2; return 2 ;;
esac
if [[ ! -x "$_irp_activate_env/bin/python" ]]; then
  printf 'Environment is not installed: %s\n' "$_irp_activate_env" >&2
  return 65
fi
export VIRTUAL_ENV="$_irp_activate_env" IRP_RL_ENV_ROOT="$_irp_activate_root"
export IRP_PYTHON_ENV="$_irp_activate_env"
export IRP_HPC_ENV_ROOT="$_irp_activate_root"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$_irp_activate_env/bin:$PATH"
unset _irp_activate_source _irp_activate_root _irp_activate_profile _irp_activate_env
