"""Fast local model, dependency, IK, and experiment-contract preflight."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from typing import Any, Sequence

import gymnasium
import mujoco
import numpy as np

from . import config, rotations, trajectories
from .env import G3G6ManipulationEnv, _KinematicsAdapter
from .math3d import euler_xyz_deg_to_quat, quat_to_matrix
from .objects import FINGERS, OBJECTS


def _tip_targets(
    spec: Any,
    quaternion: np.ndarray,
    *,
    grasp_plane_z_m: float,
) -> dict[str, np.ndarray]:
    position = np.array((0.0, 0.0, float(grasp_plane_z_m)))
    rotation = quat_to_matrix(quaternion)
    local = spec.tip_centres_local_m(pregrasp=False)
    return {finger: position + rotation @ local[finger] for finger in FINGERS}


def run_preflight() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for module in (
        np,
        mujoco,
        gymnasium,
    ):
        module_path = Path(module.__file__).resolve()
        local = False
        try:
            module_path.relative_to(config.LOCAL_VENV.resolve())
            local = True
        except ValueError:
            pass
        checks.append(
            {
                "check": f"dependency:{module.__name__}",
                "ok": local,
                "path": str(module_path),
                "version": getattr(module, "__version__", "unknown"),
            }
        )

    timing_ok = bool(
        (config.APPROACH_TICKS, config.CONTACT_TICKS, config.LIFT_TICKS)
        == (120, 80, 160)
        and config.PRE_ACTION_HOLD_TICKS == 0
        and config.PRE_ACTION_TICKS == 360
        and config.PHASE1_TERMINAL_HOLD_TICKS == 25
        and config.PHASE1_EPISODE_TICKS == 385
        and (
            trajectories.LEAD_IN_TICKS,
            trajectories.START_SETTLE_TICKS,
            trajectories.ROUTE_TICKS,
            trajectories.TERMINAL_HOLD_TICKS,
        )
        == (300, 300, 1_800, 25)
        and trajectories.TOTAL_TICKS == 2_425
        and rotations.ROTATION_MOTION_TICKS == 2_400
        and config.FORMAL_EPISODE_TICKS == 2_785
    )
    checks.append(
        {
            "check": "phase_timing_and_25_tick_terminal_holds",
            "ok": timing_ok,
            "grasp_lift_prefix_ticks": [120, 80, 160],
            "phase1_terminal_hold_ticks": config.PHASE1_TERMINAL_HOLD_TICKS,
            "phase1_episode_ticks": config.PHASE1_EPISODE_TICKS,
            "phase2_action_ticks": [300, 300, 1_800, 25],
            "phase3_action_ticks": [2_400, 25],
            "phase2_phase3_episode_ticks": config.FORMAL_EPISODE_TICKS,
        }
    )

    for gripper, xml_path in config.MODEL_XML.items():
        model = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "root")
        table_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom"
        )
        table_body = int(model.geom_bodyid[table_geom])
        table_top = float(model.body_pos[table_body, 2] + model.geom_size[table_geom, 2])
        model_ok = bool(
            np.isclose(model.body_pos[root_id, 2], config.ROOT_HEIGHT_M)
            and np.isclose(table_top, config.TABLE_TOP_M)
            and np.isclose(model.opt.timestep, config.PHYSICS_TIMESTEP_S)
            and int(model.opt.noslip_iterations) == config.NO_SLIP_ITERATIONS
            and model.nu == (12 if gripper == "g3" else 16)
        )
        checks.append(
            {
                "check": f"model:{gripper}",
                "ok": model_ok,
                "root_height_m": float(model.body_pos[root_id, 2]),
                "table_top_m": table_top,
                "nu": int(model.nu),
                "noslip_iterations": int(model.opt.noslip_iterations),
            }
        )
        for object_name, spec in OBJECTS.items():
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, spec.body_name
            )
            mass = float(model.body_mass[body_id])
            checks.append(
                {
                    "check": f"mass:{gripper}:{object_name}",
                    "ok": bool(np.isclose(mass, config.OBJECT_MASS_KG)),
                    "mass_kg": mass,
                }
            )

        adapter = _KinematicsAdapter(gripper)
        identity = np.array((1.0, 0.0, 0.0, 0.0))
        for object_name, spec in OBJECTS.items():
            neutral = adapter.solve(
                _tip_targets(
                    spec,
                    identity,
                    grasp_plane_z_m=config.TARGET_GRASP_PLANE_Z_M,
                ),
                None,
                None,
            )
            checks.append(
                {
                    "check": f"lifted_identity_ik:{gripper}:{object_name}",
                    "ok": bool(neutral.success),
                    "failed_fingers": list(neutral.failed_fingers),
                }
            )
            rotation_neutral = adapter.solve(
                _tip_targets(
                    spec,
                    identity,
                    grasp_plane_z_m=config.ROTATION_TARGET_GRASP_PLANE_Z_M,
                ),
                None,
                None,
            )
            checks.append(
                {
                    "check": f"rotation_height_identity_ik:{gripper}:{object_name}",
                    "ok": bool(rotation_neutral.success),
                    "grasp_plane_z_m": config.ROTATION_TARGET_GRASP_PLANE_Z_M,
                    "failed_fingers": list(rotation_neutral.failed_fingers),
                }
            )
            for axes in config.ROTATION_AXIS_SETS:
                for signs in product((-45.0, 45.0), repeat=len(axes)):
                    by_axis = dict(zip(axes, signs))
                    euler = np.array(
                        [by_axis.get(axis, 0.0) for axis in "xyz"],
                        dtype=np.float64,
                    )
                    quaternion = euler_xyz_deg_to_quat(euler)
                    rotated = adapter.solve(
                        _tip_targets(
                            spec,
                            quaternion,
                            grasp_plane_z_m=(
                                config.ROTATION_TARGET_GRASP_PLANE_Z_M
                            ),
                        ),
                        (
                            np.asarray(rotation_neutral.ctrl)
                            if rotation_neutral.success
                            else None
                        ),
                        (
                            rotation_neutral.finger_solutions
                            if rotation_neutral.success
                            else None
                        ),
                    )
                    if not bool(rotated.success):
                        warnings.append(
                            {
                                "warning": "fixed_anchor_45deg_ik_unreachable",
                                "gripper": gripper,
                                "object": object_name,
                                "axes": axes,
                                "euler_xyz_deg": euler.tolist(),
                                "grasp_plane_z_m": (
                                    config.ROTATION_TARGET_GRASP_PLANE_Z_M
                                ),
                                "failed_fingers": list(rotated.failed_fingers),
                                "mitigation": "bounded pure-IK pose feedback plus feasible-pose projection; inspect recorded error",
                            }
                        )

        for object_name in OBJECTS:
            environment = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.GRASP_LIFT,
                object_name=object_name,
            )
            try:
                _observation, info = environment.reset(seed=config.DEFAULT_SEED)
                terminated = truncated = False
                while not (terminated or truncated):
                    _observation, _reward, terminated, truncated, info = (
                        environment.step_ik()
                    )
                checks.append(
                    {
                        "check": f"pure_ik_phase1:{gripper}:{object_name}",
                        "ok": bool(
                            info["is_success"]
                            and int(info["step"]) == config.PHASE1_EPISODE_TICKS
                        ),
                        "steps": int(info["step"]),
                        "position_error_m_per_axis": np.asarray(
                            info["position_error_m_per_axis"]
                        ).tolist(),
                        "angle_error_deg_per_axis": np.asarray(
                            info["angle_error_deg_per_axis"]
                        ).tolist(),
                        "tip_contact_count": int(info["tip_contact_count"]),
                        "termination_reason": info["termination_reason"],
                    }
                )
            finally:
                environment.close()

    checks.append(
        {
            "check": "fixed_phase_budgets_without_success_gate",
            "ok": config.PHASE_TIMESTEPS
            == {
                config.Phase.GRASP_LIFT: 500_000,
                config.Phase.TRANSLATE: 2_000_000,
                config.Phase.ROTATE: 2_000_000,
            },
            "success_rate_gate": False,
        }
    )
    return {
        "schema": "g3g6-local-preflight-v1",
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "warnings": warnings,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate the local G3/G6 training stack.")
    parser.add_argument("--json", type=Path, help="optional report path inside IK-residual")
    args = parser.parse_args(argv)
    report = run_preflight()
    if args.json is not None:
        destination = (
            (config.PROJECT_ROOT / args.json).resolve()
            if not args.json.is_absolute()
            else args.json.resolve()
        )
        try:
            destination.relative_to(config.PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise SystemExit("--json must stay inside IK-residual") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"preflight={'PASS' if report['ok'] else 'FAIL'}; "
        f"checks={len(report['checks'])}; reachability_warnings={len(report['warnings'])}"
    )
    for warning in report["warnings"][:12]:
        print(
            "warning: fixed-anchor 45deg IK unreachable "
            f"{warning['gripper']}/{warning['object']}/{warning['axes']} "
            f"euler={warning['euler_xyz_deg']} "
            f"z={warning['grasp_plane_z_m']:.3f}m"
        )
    if len(report["warnings"]) > 12:
        print(f"... {len(report['warnings']) - 12} more warnings (use --json for all)")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
