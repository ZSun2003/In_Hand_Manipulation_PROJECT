#!/usr/bin/env bash
# Build the isolated, source-only HX1 A100 14,000-world residual package.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
STAMP="${SCREW_PACKAGE_STAMP:-$(date -u '+%Y%m%d')}"
AUDIT_PYTHON="${PYTHON_BOOTSTRAP:-python3}"

if ! command -v "${AUDIT_PYTHON}" >/dev/null 2>&1; then
  printf 'ERROR: package-audit Python not found: %s\n' \
    "${AUDIT_PYTHON}" >&2
  exit 69
fi

SCREW_RESIDUAL_PACKAGE_ROOT="${SCRIPT_DIR}" \
  "${AUDIT_PYTHON}" - <<'PY'
import ast
import json
import os
from pathlib import Path

root = Path(os.environ["SCREW_RESIDUAL_PACKAGE_ROOT"])
config_path = root / "configs" / "hx1_a100_w14000_diffusion_ik_residual.json"
job_path = root / "jobs" / "hx1_w14000_diffusion_ik_residual_full.pbs"
contracts_path = root / "s_hpc" / "contracts.py"

config = json.loads(config_path.read_text(encoding="utf-8"))
expected = {
    "physics.worlds": (config["physics"]["worlds"], 14000),
    "physics.max_episode_steps": (
        config["physics"]["max_episode_steps"], 4000
    ),
    "tqc.critics": (config["tqc"]["critics"], 4),
    "tqc.quantiles": (config["tqc"]["quantiles"], 32),
    "tqc.top_quantiles_to_drop": (
        config["tqc"]["top_quantiles_to_drop"], 2
    ),
    "tqc.gradient_steps_per_vector_step": (
        config["tqc"]["gradient_steps_per_vector_step"], 8
    ),
}

contract_values = {}


def contract_expression(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return contract_values.get(node.id)
    if isinstance(node, ast.Tuple):
        values = tuple(contract_expression(item) for item in node.elts)
        return None if any(value is None for value in values) else values
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and not node.keywords
    ):
        value = contract_expression(node.args[0])
        return None if value is None else len(value)
    if isinstance(node, ast.BinOp):
        left = contract_expression(node.left)
        right = contract_expression(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    return None


for node in ast.parse(
    contracts_path.read_text(encoding="utf-8"),
    filename=str(contracts_path),
).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        expression = node.value
    elif isinstance(node, ast.AnnAssign):
        target = node.target
        expression = node.value
    else:
        continue
    if isinstance(target, ast.Name):
        value = contract_expression(expression)
        if value is not None:
            contract_values[target.id] = value
expected.update(
    {
        "contracts.OBS_DIM": (contract_values.get("OBS_DIM"), 191),
        "contracts.RESIDUAL_ACTION_DIM": (
            contract_values.get("RESIDUAL_ACTION_DIM"), 16
        ),
        "contracts.TQC_OBS_DIM": (
            contract_values.get("TQC_OBS_DIM"), 242
        ),
    }
)

problems = [
    f"{name}: expected {wanted!r}, found {actual!r}"
    for name, (actual, wanted) in expected.items()
    if actual != wanted
]
job = job_path.read_text(encoding="utf-8")
for required in (
    "SCREW_HX1_PREFLIGHT_STEPS:-4000",
    "--worlds 14000",
    "--pipeline-mode diffusion-tqc",
    "--stage full",
    "SCREW_MAX_RUNTIME:-4290m",
):
    if required not in job:
        problems.append(f"job is missing {required!r}")
if problems:
    raise SystemExit(
        "residual package contract audit failed:\n  "
        + "\n  ".join(problems)
    )
print("residual_package_contract_audit=pass")
PY

export SCREW_PACKAGE_INCLUDE_MODELS=0
export SCREW_PACKAGE_REQUIRE_MODELS=0
export SCREW_PACKAGE_ARCHIVE_NAME="hpc_HX1_A100_W14000_DIFFUSION_IK_RESIDUAL_TQC16_EP4000_BLEND20_CLEAR4_${STAMP}.tar.gz"

exec "${SCRIPT_DIR}/build_autodl_package.sh" "$@"
