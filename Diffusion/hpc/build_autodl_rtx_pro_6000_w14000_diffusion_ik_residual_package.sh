#!/usr/bin/env bash
# Build the isolated AutoDL RTX PRO 6000 14,000-world residual package.
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
import math
import os
from pathlib import Path

root = Path(os.environ["SCREW_RESIDUAL_PACKAGE_ROOT"])
config_path = (
    root
    / "configs"
    / "autodl_rtx_pro_6000_w14000_diffusion_ik_residual.json"
)
launcher_path = (
    root / "autodl_rtx_pro_6000_w14000_diffusion_ik_residual.sh"
)
contracts_path = root / "s_hpc" / "contracts.py"
control_path = root / "screw_control.py"
task_path = root / "screw_config.py"
env_path = root / "s_hpc" / "env.py"
workflows_path = root / "s_hpc" / "workflows.py"

for path in (
    config_path,
    launcher_path,
    contracts_path,
    control_path,
    task_path,
    env_path,
    workflows_path,
):
    if not path.is_file():
        raise SystemExit(f"required package file is missing: {path}")

config = json.loads(config_path.read_text(encoding="utf-8"))
expected = {
    "physics.worlds": (config["physics"]["worlds"], 14000),
    "physics.max_episode_steps": (
        config["physics"]["max_episode_steps"], 4000
    ),
    "demos.obs_horizon": (config["demos"]["obs_horizon"], 2),
    "tqc.total_transitions": (
        config["tqc"]["total_transitions"], 5_000_000
    ),
    "tqc.replay_insert_per_vector_step": (
        config["tqc"]["replay_insert_per_vector_step"], 256
    ),
    "tqc.gradient_steps_per_vector_step": (
        config["tqc"]["gradient_steps_per_vector_step"], 8
    ),
    "tqc.replay_ratio": (config["tqc"]["replay_ratio"], 8.0),
    "tqc.critics": (config["tqc"]["critics"], 4),
    "tqc.quantiles": (config["tqc"]["quantiles"], 32),
    "tqc.top_quantiles_to_drop": (
        config["tqc"]["top_quantiles_to_drop"], 2
    ),
}


def module_values(path):
    values = {}

    def expression(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return values.get(node.id)
        if isinstance(node, ast.Tuple):
            items = tuple(expression(item) for item in node.elts)
            return None if any(item is None for item in items) else items
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
        ):
            value = expression(node.args[0])
            return None if value is None else len(value)
        if isinstance(node, ast.BinOp):
            left = expression(node.left)
            right = expression(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
        return None

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, raw = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, raw = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name):
            value = expression(raw)
            if value is not None:
                values[target.id] = value
    return tree, values


_contracts_tree, contracts = module_values(contracts_path)
_control_tree, control = module_values(control_path)
expected.update(
    {
        "contracts.OBS_DIM": (contracts.get("OBS_DIM"), 191),
        "contracts.TQC_OBS_DIM": (contracts.get("TQC_OBS_DIM"), 242),
        "contracts.RESIDUAL_ACTION_DIM": (
            contracts.get("RESIDUAL_ACTION_DIM"), 16
        ),
        "control.DIFFUSION_TO_ANCHOR_BLEND_STEPS": (
            control.get("DIFFUSION_TO_ANCHOR_BLEND_STEPS"), 20
        ),
    }
)

task_tree = ast.parse(
    task_path.read_text(encoding="utf-8"), filename=str(task_path)
)
task_fields = {}
for node in task_tree.body:
    if not isinstance(node, ast.ClassDef) or node.name != "ScrewTaskConfig":
        continue
    for item in node.body:
        if (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and isinstance(item.value, ast.Constant)
        ):
            task_fields[item.target.id] = item.value.value

shaft_radius = task_fields.get("screw_shaft_radius_m")
hole_radius = task_fields.get("socket_hole_radius_m")
if not isinstance(shaft_radius, (int, float)) or not isinstance(
    hole_radius, (int, float)
):
    raise SystemExit("could not audit screw/bore radii")
diametral_clearance = 2.0 * (float(hole_radius) - float(shaft_radius))

problems = [
    f"{name}: expected {wanted!r}, found {actual!r}"
    for name, (actual, wanted) in expected.items()
    if actual != wanted
]
if not math.isclose(diametral_clearance, 0.004, abs_tol=1.0e-12):
    problems.append(
        "task bore diameter must equal screw diameter + 0.004 m; "
        f"found clearance {diametral_clearance!r} m"
    )

launcher = launcher_path.read_text(encoding="utf-8")
for required in (
    "nohup setsid",
    "start-diffusion-only",
    'PIPELINE_STAGE="full"',
    'PIPELINE_STAGE="diffusion-full"',
    'PIPELINE_MODE="diffusion-tqc"',
    'PIPELINE_MODE="diffusion-only"',
    "--worlds 14000",
    "--resume auto",
):
    if required not in launcher:
        problems.append(f"launcher is missing {required!r}")

workflows = workflows_path.read_text(encoding="utf-8")
for forbidden in (
    "action_dim=6",
    "(env.worlds, 6)",
    "residual_mask_from_observation",
    "trainer.act(observation",
):
    if forbidden in workflows:
        problems.append(
            f"residual workflow still contains legacy code {forbidden!r}"
        )
for required in (
    "obs_dim=TQC_OBS_DIM",
    "action_dim=RESIDUAL_ACTION_DIM",
    "env.step(residual_policy=residual_policy)",
    "replay_insert_per_vector_step",
    '"pending_replay"',
    "_terminal_tqc_observation(result)",
):
    if required not in workflows:
        problems.append(f"residual workflow is missing {required!r}")

env_source = env_path.read_text(encoding="utf-8")
ik_index = env_source.find("candidate_ik = self.ik.solve")
state_index = env_source.find("tqc_observation = self._tqc_observation")
residual_index = env_source.find("residual_step = self.actuator_residual.apply")
if not (0 <= ik_index < state_index < residual_index):
    problems.append(
        "controller order must be Diffusion/task-space safety -> nominal IK -> "
        "242-D TQC state -> bounded actuator residual"
    )

if problems:
    raise SystemExit(
        "AutoDL residual package contract audit failed:\n  "
        + "\n  ".join(problems)
    )
print(
    "autodl_residual_package_contract_audit=pass "
    f"obs={contracts['OBS_DIM']} tqc_obs={contracts['TQC_OBS_DIM']} "
    f"residual={contracts['RESIDUAL_ACTION_DIM']} "
    f"diametral_clearance_m={diametral_clearance:g}"
)
PY

export SCREW_PACKAGE_INCLUDE_MODELS=0
export SCREW_PACKAGE_REQUIRE_MODELS=0
export SCREW_PACKAGE_ARCHIVE_NAME="hpc_AUTODL_RTX_PRO_6000_W14000_DIFFUSION_IK_TQC16_OBS191_EP4000_BLEND20_CLEAR4_${STAMP}.tar.gz"

exec "${SCRIPT_DIR}/build_autodl_package.sh" "$@"
