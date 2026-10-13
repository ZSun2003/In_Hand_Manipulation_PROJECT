"""MuJoCo/Gymnasium environment shared by the local G3 and G6 models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from . import config, trajectories
from .math3d import (
    euler_xyz_deg_to_quat,
    minimum_jerk,
    normalize_quat,
    orientation_error_xyz_deg,
    quat_conjugate,
    quat_multiply,
    quat_to_euler_xyz_deg,
    quat_to_matrix,
    slerp,
    wrap_degrees,
)
from .objects import FINGERS, OBJECTS, ObjectSpec, get_object_spec
from .residual import ActuatorResidualController
from .rotations import (
    build_coupled_rotation_quaternion_schedule,
    build_rotation_quaternion_schedule,
    sample_goal_euler_deg,
    validate_axis_set,
    validate_goal_euler_deg,
    validate_rotation_sampling_profile,
)


OBSERVATION_DIM = 149
MAX_ACTUATORS = 16


@dataclass(frozen=True)
class PoseTarget:
    position: np.ndarray
    quaternion: np.ndarray
    local_tip_centres: Mapping[str, np.ndarray]


class _KinematicsAdapter:
    def __init__(self, gripper: str) -> None:
        self.gripper = gripper
        project_xml = config.MODEL_XML[gripper].resolve()
        if gripper == config.GripperKind.G3.value:
            from .kinematics import g3 as module
        else:
            from .kinematics import g6 as module
        self.module = module
        self.ik = module.GripperIK.from_xml(project_xml)
        self.joint_kinds = tuple(module.JOINT_KINDS)

    @property
    def ctrl_ranges(self) -> np.ndarray:
        return np.asarray(self.ik.ctrl_ranges, dtype=np.float64)

    def solve(
        self,
        targets: Mapping[str, Sequence[float]],
        previous_ctrl: np.ndarray | None,
        previous_solutions: Mapping[str, Any] | None,
    ) -> Any:
        return self.ik.solve_tip_targets(
            targets,
            previous_ctrl=previous_ctrl,
            previous_solutions=previous_solutions,
            strict=False,
        )


class G3G6ManipulationEnv(gym.Env[np.ndarray, np.ndarray]):
    """One gripper, one random rigid object, and one fixed motion phase.

    Every Gym step commands exactly one 50 Hz control tick, which contains 40
    MuJoCo steps at 0.5 ms.  The RL interface uses each gripper's native
    actuator residual vector (12-D for G3, 16-D for G6); :meth:`step_ik`
    bypasses that interface and sends the analytic IK command directly.
    """

    metadata = {"render_modes": ("rgb_array",), "render_fps": config.CONTROL_HZ}

    def __init__(
        self,
        *,
        gripper: str = "g3",
        phase: int | config.Phase = config.Phase.GRASP_LIFT,
        object_name: str | None = None,
        path_name: str | None = None,
        rotation_axes: str | None = None,
        rotation_goal_max_deg: float = config.ROTATION_ANGLE_MAX_DEG,
        rotation_sampling_profile: str = (
            config.DEFAULT_ROTATION_SAMPLING_PROFILE
        ),
        coupled_rotation_axes: str | None = None,
        coupled_rotation_euler_deg: Sequence[float] | None = None,
        residual_nominal_feedback: bool = True,
        residual_after_preaction_only: bool = False,
        ik_position_feedback_gain: float = 1.0,
        ik_orientation_feedback_gain: float = 4.0,
        ik_max_position_correction_m: float = 0.015,
        ik_max_angle_correction_deg: float = 45.0,
        model_xml_path: str | Path | None = None,
        object_spec_override: ObjectSpec | None = None,
        render_mode: str | None = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        super().__init__()
        self.gripper = config.GripperKind(gripper).value
        self.phase = config.Phase(phase)
        self.task_target_grasp_plane_z_m = (
            config.ROTATION_TARGET_GRASP_PLANE_Z_M
            if self.phase == config.Phase.ROTATE
            else config.TARGET_GRASP_PLANE_Z_M
        )
        registered_fixed_object = (
            get_object_spec(object_name) if object_name is not None else None
        )
        if object_spec_override is not None:
            if not isinstance(object_spec_override, ObjectSpec):
                raise TypeError("object_spec_override must be an ObjectSpec")
            if registered_fixed_object is None:
                raise ValueError(
                    "object_spec_override requires a fixed object_name"
                )
            if object_spec_override.name != registered_fixed_object.name:
                raise ValueError(
                    "object_spec_override.name must match the fixed object_name: "
                    f"expected {registered_fixed_object.name!r}, got "
                    f"{object_spec_override.name!r}"
                )
        objects = dict(OBJECTS)
        if object_spec_override is not None:
            objects[object_spec_override.name] = object_spec_override
        self._objects: Mapping[str, ObjectSpec] = MappingProxyType(objects)
        self.fixed_object_name = (
            registered_fixed_object.name
            if registered_fixed_object is not None
            else None
        )
        self.fixed_path_name = (
            trajectories.normalize_path_name(path_name) if path_name is not None else None
        )
        self.fixed_rotation_axes = (
            validate_axis_set(rotation_axes) if rotation_axes is not None else None
        )
        self.rotation_sampling_profile = validate_rotation_sampling_profile(
            rotation_sampling_profile
        )
        if (
            self.rotation_sampling_profile
            == config.WEIGHTED_ROTATION_SAMPLING_PROFILE
            and self.phase != config.Phase.ROTATE
        ):
            raise ValueError(
                "weighted_0_90 rotation sampling is only available in Phase 3"
            )
        self.rotation_goal_max_deg = (
            config.ROTATION_SWEEP_ANGLE_MAX_DEG
            if self.rotation_sampling_profile
            == config.WEIGHTED_ROTATION_SAMPLING_PROFILE
            else float(rotation_goal_max_deg)
        )
        if (
            not np.isfinite(self.rotation_goal_max_deg)
            or self.rotation_goal_max_deg < config.ROTATION_ANGLE_MIN_DEG
            or self.rotation_goal_max_deg > config.ROTATION_SWEEP_ANGLE_MAX_DEG
        ):
            raise ValueError(
                "rotation_goal_max_deg must be finite and inside "
                f"[{config.ROTATION_ANGLE_MIN_DEG}, "
                f"{config.ROTATION_SWEEP_ANGLE_MAX_DEG:g}]"
            )
        if (
            self.phase != config.Phase.ROTATE
            and self.rotation_goal_max_deg != config.ROTATION_ANGLE_MAX_DEG
        ):
            raise ValueError(
                "an extended rotation goal limit is only available with the rotation phase"
            )
        self.fixed_coupled_rotation_axes = (
            validate_axis_set(coupled_rotation_axes)
            if coupled_rotation_axes is not None
            else None
        )
        if coupled_rotation_euler_deg is not None:
            if self.fixed_coupled_rotation_axes is None:
                raise ValueError(
                    "coupled_rotation_axes is required with "
                    "coupled_rotation_euler_deg"
                )
            self.fixed_coupled_rotation_euler_deg = validate_goal_euler_deg(
                self.fixed_coupled_rotation_axes, coupled_rotation_euler_deg
            )
        else:
            self.fixed_coupled_rotation_euler_deg = None
        if self.phase != config.Phase.TRANSLATE and (
            self.fixed_coupled_rotation_axes is not None
            or self.fixed_coupled_rotation_euler_deg is not None
        ):
            raise ValueError(
                "coupled rotation is only available with the translation phase"
            )
        self.residual_nominal_feedback = bool(residual_nominal_feedback)
        self.residual_after_preaction_only = bool(residual_after_preaction_only)
        feedback_values = np.asarray(
            (
                ik_position_feedback_gain,
                ik_orientation_feedback_gain,
                ik_max_position_correction_m,
                ik_max_angle_correction_deg,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(feedback_values)) or np.any(feedback_values < 0.0):
            raise ValueError("pure-IK feedback gains and limits must be finite and non-negative")
        self.ik_position_feedback_gain = float(ik_position_feedback_gain)
        self.ik_orientation_feedback_gain = float(ik_orientation_feedback_gain)
        self.ik_max_position_correction_m = float(ik_max_position_correction_m)
        self.ik_max_angle_correction_deg = float(ik_max_angle_correction_deg)
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode
        self.width = int(width)
        self.height = int(height)

        self.model_xml_path = (
            config.MODEL_XML[self.gripper].resolve()
            if model_xml_path is None
            else Path(model_xml_path).expanduser().resolve()
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.model_xml_path))
        self.data = mujoco.MjData(self.model)
        if not np.isclose(self.model.opt.timestep, config.PHYSICS_TIMESTEP_S):
            raise ValueError("model timestep does not match the 50 Hz/40-substep contract")
        if self.model.nu not in (12, 16):
            raise ValueError(f"unexpected actuator count: {self.model.nu}")

        self.kinematics = _KinematicsAdapter(self.gripper)
        self.ctrl_low = self.kinematics.ctrl_ranges[:, 0].copy()
        self.ctrl_high = self.kinematics.ctrl_ranges[:, 1].copy()
        self.action_dim = config.action_dim(self.gripper)
        self.residual_controller = ActuatorResidualController(
            self.gripper, self.ctrl_low, self.ctrl_high
        )

        self._actuator_joint_ids = self.model.actuator_trnid[:, 0].astype(np.int32)
        self._actuator_qpos_adrs = self.model.jnt_qposadr[
            self._actuator_joint_ids
        ].astype(np.int32)
        self._actuator_qvel_adrs = self.model.jnt_dofadr[
            self._actuator_joint_ids
        ].astype(np.int32)
        self._robot_qvel_adrs = np.asarray(
            sorted(
                {
                    int(self.model.jnt_dofadr[joint_id])
                    for joint_id in range(self.model.njnt)
                    if (
                        (name := mujoco.mj_id2name(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                        ))
                        is not None
                        and name.startswith("robot0:")
                    )
                }
            ),
            dtype=np.int32,
        )
        self._tip_body_ids = {
            finger: self._name_id(mujoco.mjtObj.mjOBJ_BODY, f"hard_tip_{finger}")
            for finger in FINGERS
        }
        self._tip_geom_ids = {
            finger: self._name_id(mujoco.mjtObj.mjOBJ_GEOM, f"hard_tip_{finger}")
            for finger in FINGERS
        }
        self._object_body_ids = {
            name: self._name_id(mujoco.mjtObj.mjOBJ_BODY, spec.body_name)
            for name, spec in self._objects.items()
        }
        self._object_joint_ids = {
            name: self._name_id(mujoco.mjtObj.mjOBJ_JOINT, spec.joint_name)
            for name, spec in self._objects.items()
        }
        self._object_geom_ids = {
            name: self._name_id(mujoco.mjtObj.mjOBJ_GEOM, spec.geom_name)
            for name, spec in self._objects.items()
        }
        self._object_contype = {
            name: int(self.model.geom_contype[geom_id])
            for name, geom_id in self._object_geom_ids.items()
        }
        self._object_conaffinity = {
            name: int(self.model.geom_conaffinity[geom_id])
            for name, geom_id in self._object_geom_ids.items()
        }

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self._renderer: mujoco.Renderer | None = None

        self.object_spec: ObjectSpec = self._objects[config.OBJECT_NAMES[0]]
        self.object_name = self.object_spec.name
        self.path_name: str | None = None
        self.rotation_axes: str | None = None
        self.coupled_rotation = False
        self.rotation_goal_euler_deg = np.zeros(3, dtype=np.float64)
        self._translation_schedule: np.ndarray | None = None
        self._rotation_quaternion_schedule: np.ndarray | None = None
        self._action_reference_captured = False
        self._action_reference_position = np.array(
            (0.0, 0.0, self.task_target_grasp_plane_z_m), dtype=np.float64
        )
        self._action_reference_quaternion = np.array(
            (1.0, 0.0, 0.0, 0.0), dtype=np.float64
        )
        self._previous_solutions: dict[str, Any] = {}
        self._last_feasible_position = np.zeros(3, dtype=np.float64)
        self._last_feasible_quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        self.nominal_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self.applied_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self.last_residual = np.zeros(self.action_dim, dtype=np.float64)
        self._last_requested_residual_action = np.zeros(
            self.action_dim, dtype=np.float64
        )
        self._last_executed_residual_action = np.zeros(
            self.action_dim, dtype=np.float64
        )
        self.target_position = np.zeros(3, dtype=np.float64)
        self.target_quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        self.ik_command_position = np.zeros(3, dtype=np.float64)
        self.ik_command_quaternion = np.array((1.0, 0.0, 0.0, 0.0))
        self._control_mode = "reset"
        self._residual_active = False
        self.step_count = 0
        self.episode_ticks = config.PHASE1_EPISODE_TICKS
        self._ik_success = True
        self._ik_projection_fraction = 1.0
        self._max_route_abs_position_error = np.zeros(3, dtype=np.float64)
        self._max_route_abs_angle_error = np.zeros(3, dtype=np.float64)
        self._termination_reason = "running"
        self._last_finite_qpos = np.zeros(self.model.nq, dtype=np.float64)
        self._last_finite_qvel = np.zeros(self.model.nv, dtype=np.float64)

    def _name_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        value = int(mujoco.mj_name2id(self.model, object_type, name))
        if value < 0:
            raise ValueError(f"model element not found: {name}")
        return value

    def _get_object_spec(self, name: str) -> ObjectSpec:
        try:
            return self._objects[str(name)]
        except KeyError as exc:
            raise ValueError(
                f"unknown object {name!r}; expected one of {tuple(self._objects)}"
            ) from exc

    @property
    def task_lift_distance_m(self) -> float:
        """Object-specific lift distance for this environment's active phase."""

        distance = (
            self.task_target_grasp_plane_z_m
            - self.object_spec.rest_grasp_plane_z_m
        )
        if not np.isfinite(distance) or distance <= 0.0:
            raise RuntimeError("task grasp-plane target must remain above the rest pose")
        return float(distance)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})
        object_name = options.get("object", self.fixed_object_name)
        if object_name is None:
            object_name = str(self.np_random.choice(config.OBJECT_NAMES))
        self.object_spec = self._get_object_spec(str(object_name))
        self.object_name = self.object_spec.name

        self.path_name = None
        self.rotation_axes = None
        self.coupled_rotation = False
        self.rotation_goal_euler_deg = np.zeros(3, dtype=np.float64)
        self._translation_schedule = None
        self._rotation_quaternion_schedule = None
        self._action_reference_captured = False
        self._action_reference_position[:] = (
            0.0,
            0.0,
            self.task_target_grasp_plane_z_m,
        )
        self._action_reference_quaternion[:] = (1.0, 0.0, 0.0, 0.0)
        if self.phase == config.Phase.TRANSLATE:
            path = options.get("path", self.fixed_path_name)
            if path is None:
                path = str(self.np_random.choice(trajectories.PATH_NAMES))
            self.path_name = trajectories.normalize_path_name(path)
            # Store offsets here.  Their measured world-frame origin is
            # captured after the lift, at the first Phase-2 action tick.
            self._translation_schedule = trajectories.scheduled_path(self.path_name)
            coupled_axes = options.get(
                "coupled_rotation_axes", self.fixed_coupled_rotation_axes
            )
            coupled_goal = options.get(
                "coupled_rotation_euler_deg",
                self.fixed_coupled_rotation_euler_deg,
            )
            if coupled_goal is not None and coupled_axes is None:
                raise ValueError(
                    "coupled_rotation_axes is required with "
                    "coupled_rotation_euler_deg"
                )
            if coupled_axes is not None:
                self.rotation_axes = validate_axis_set(str(coupled_axes))
                if coupled_goal is None:
                    _, self.rotation_goal_euler_deg = sample_goal_euler_deg(
                        self.np_random, self.rotation_axes
                    )
                else:
                    self.rotation_goal_euler_deg = validate_goal_euler_deg(
                        self.rotation_axes, coupled_goal
                    )
                self._rotation_quaternion_schedule = (
                    build_coupled_rotation_quaternion_schedule(
                        self.rotation_goal_euler_deg
                    )
                )
                self.coupled_rotation = True
        elif self.phase == config.Phase.ROTATE:
            axes = options.get("rotation_axes", self.fixed_rotation_axes)
            if "rotation_euler_deg" in options:
                if axes is None:
                    raise ValueError("rotation_axes is required with rotation_euler_deg")
                self.rotation_axes = validate_axis_set(str(axes))
                self.rotation_goal_euler_deg = validate_goal_euler_deg(
                    self.rotation_axes,
                    options["rotation_euler_deg"],
                    max_magnitude_deg=self.rotation_goal_max_deg,
                )
            else:
                self.rotation_axes, self.rotation_goal_euler_deg = sample_goal_euler_deg(
                    self.np_random,
                    None if axes is None else str(axes),
                    profile=self.rotation_sampling_profile,
                )
            self._rotation_quaternion_schedule = build_rotation_quaternion_schedule(
                self.rotation_goal_euler_deg
            )

        self.episode_ticks = (
            config.PHASE1_EPISODE_TICKS
            if self.phase == config.Phase.GRASP_LIFT
            else config.FORMAL_EPISODE_TICKS
        )
        self.step_count = 0
        self._termination_reason = "running"
        self._max_route_abs_position_error.fill(0.0)
        self._max_route_abs_angle_error.fill(0.0)
        self._previous_solutions = {}
        self.residual_controller.reset()
        self.last_residual.fill(0.0)
        self._last_requested_residual_action.fill(0.0)
        self._last_executed_residual_action.fill(0.0)
        self._control_mode = "reset"
        self._residual_active = False

        mujoco.mj_resetData(self.model, self.data)
        self._configure_objects_for_reset()
        identity = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        rest_position = np.array(
            (0.0, 0.0, self.object_spec.rest_grasp_plane_z_m),
            dtype=np.float64,
        )
        pregrasp = self.object_spec.tip_centres_local_m(pregrasp=True)
        tip_targets = self._world_tip_targets(rest_position, identity, pregrasp)
        solution = self.kinematics.solve(tip_targets, None, None)
        if not bool(solution.success):
            raise RuntimeError(
                f"pregrasp IK failed for {self.gripper}/{self.object_name}: "
                f"{solution.failed_fingers}"
            )
        self.nominal_ctrl = np.asarray(solution.ctrl, dtype=np.float64).copy()
        self.applied_ctrl = self.nominal_ctrl.copy()
        self._previous_solutions = dict(solution.finger_solutions)
        self._set_robot_qpos_from_solution(solution)
        self.data.ctrl[:] = self.nominal_ctrl
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_finite_qpos[:] = self.data.qpos
        self._last_finite_qvel[:] = self.data.qvel

        self._last_feasible_position = rest_position.copy()
        self._last_feasible_quaternion = identity.copy()
        self.target_position = rest_position.copy()
        self.target_quaternion = identity.copy()
        self.ik_command_position = rest_position.copy()
        self.ik_command_quaternion = identity.copy()
        self._ik_success = True
        self._ik_projection_fraction = 1.0
        observation = self._observation()
        return observation, self._info(endpoint=False)

    def _configure_objects_for_reset(self) -> None:
        for index, (name, spec) in enumerate(self._objects.items()):
            joint_id = self._object_joint_ids[name]
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            selected = name == self.object_name
            self.data.qpos[qpos_adr : qpos_adr + 7] = (
                (0.0, 0.0, spec.rest_grasp_plane_z_m, 1.0, 0.0, 0.0, 0.0)
                if selected
                else (1.0 + index, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
            )
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0
            body_id = self._object_body_ids[name]
            geom_id = self._object_geom_ids[name]
            self.model.body_gravcomp[body_id] = 0.0 if selected else 1.0
            self.model.geom_contype[geom_id] = (
                self._object_contype[name] if selected else 0
            )
            self.model.geom_conaffinity[geom_id] = (
                self._object_conaffinity[name] if selected else 0
            )

    def _set_robot_qpos_from_solution(self, solution: Any) -> None:
        for finger, finger_solution in solution.finger_solutions.items():
            for kind in self.kinematics.joint_kinds:
                joint_name = f"robot0:{kind}_joint_{finger}"
                joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                address = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[address] = finger_solution.joint_value(kind)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        value = np.asarray(action, dtype=np.float64).reshape(self.action_dim)
        if not np.all(np.isfinite(value)):
            raise ValueError("action must contain only finite values")
        return self._step_controller(
            action=value,
            pure_ik_feedback=self.residual_nominal_feedback,
        )

    def step_ik(
        self, *, feedback: bool = True
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one tick using only the analytic IK controller.

        No action or learned model participates in this path.  The optional
        bounded pose feedback changes the Cartesian cage target before solving
        IK; it never adds actuator-space residuals.
        """

        return self._step_controller(action=None, pure_ik_feedback=bool(feedback))

    def _step_controller(
        self,
        *,
        action: np.ndarray | None,
        pure_ik_feedback: bool,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:

        previous_nominal_ctrl = self.nominal_ctrl.copy()
        previous_applied_ctrl = self.applied_ctrl.copy()
        previous_last_residual = self.last_residual.copy()
        previous_internal_residual = self.residual_controller.residual.copy()

        pose_target = self._target_for_tick(self.step_count)
        self.target_position = pose_target.position.copy()
        self.target_quaternion = pose_target.quaternion.copy()
        if pure_ik_feedback:
            command_target = self._pure_ik_feedback_target(pose_target)
        else:
            command_target = pose_target
        self.ik_command_position = command_target.position.copy()
        self.ik_command_quaternion = command_target.quaternion.copy()
        nominal = self._nominal_for_target(command_target)
        self.nominal_ctrl = self._rate_limit_nominal(nominal)
        residual_active = bool(
            action is not None
            and (
                not self.residual_after_preaction_only
                or self.step_count >= config.PRE_ACTION_TICKS
            )
        )
        self._residual_active = residual_active
        if action is None:
            self._last_requested_residual_action.fill(0.0)
            self._last_executed_residual_action.fill(0.0)
        else:
            self._last_requested_residual_action[:] = action
            self._last_executed_residual_action[:] = (
                np.clip(action, -1.0, 1.0)
                if residual_active
                else np.zeros(self.action_dim, dtype=np.float64)
            )
        if action is None:
            self._control_mode = "pure_ik_feedback" if pure_ik_feedback else "pure_ik"
            self.residual_controller.reset()
            actual_residual = np.zeros(self.action_dim, dtype=np.float64)
            applied = self.nominal_ctrl.copy()
        elif not residual_active:
            self._control_mode = "residual_action"
            self.residual_controller.reset()
            actual_residual = np.zeros(self.action_dim, dtype=np.float64)
            applied = self.nominal_ctrl.copy()
        else:
            self._control_mode = "residual_action"
            applied, actual_residual = self.residual_controller.apply(
                self.nominal_ctrl, action
            )
        self.applied_ctrl = applied
        self.last_residual = actual_residual.copy()
        controller_finite = bool(
            np.all(np.isfinite(self.nominal_ctrl))
            and np.all(np.isfinite(self.applied_ctrl))
            and np.all(np.isfinite(self.last_residual))
            and np.all(np.isfinite(actual_residual))
        )
        if not controller_finite:
            # Fail closed before NaN controls enter MuJoCo, and restore every
            # controller field that is exposed through observation/reward.
            self.nominal_ctrl = previous_nominal_ctrl
            if action is None:
                self.applied_ctrl = previous_nominal_ctrl.copy()
                self.last_residual.fill(0.0)
                self.residual_controller.reset()
                actual_residual = np.zeros(self.action_dim, dtype=np.float64)
            else:
                self.applied_ctrl = previous_applied_ctrl
                self.last_residual = previous_last_residual
                self.residual_controller.residual[:] = previous_internal_residual
                actual_residual = previous_last_residual.copy()
            self.data.ctrl[:] = self.applied_ctrl
            self.step_count += 1
            failure = "non_finite_controller"
        else:
            self.data.ctrl[:] = self.applied_ctrl
            mujoco.mj_step(
                self.model,
                self.data,
                nstep=config.PHYSICS_SUBSTEPS,
            )
            self.step_count += 1

            failure = self._failure_reason()
            if failure in ("non_finite_state", "invalid_object_quaternion"):
                # Return one finite terminal observation instead of allowing a
                # NaN quaternion to fail inside observation normalization.
                self.data.qpos[:] = self._last_finite_qpos
                self.data.qvel[:] = self._last_finite_qvel
                self.data.ctrl[:] = self.applied_ctrl
                mujoco.mj_forward(self.model, self.data)
            else:
                self._last_finite_qpos[:] = self.data.qpos
                self._last_finite_qvel[:] = self.data.qvel
        position_error, angle_error = self._pose_errors()
        self._update_route_metrics(position_error, angle_error)
        horizon = self.step_count >= self.episode_ticks
        # This is an intrinsic finite-horizon task (time progress is part of
        # the Markov observation and endpoint success is defined here), not an
        # external Gym TimeLimit timeout.
        terminated = horizon or failure is not None
        if failure is not None:
            self._termination_reason = failure
        elif horizon:
            self._termination_reason = "horizon"
        endpoint_success = bool(
            horizon and failure is None and self._endpoint_success()
        )
        reward = self._reward(endpoint_success, failure, actual_residual)
        observation = self._observation()
        info = self._info(endpoint=horizon)
        info["is_success"] = endpoint_success
        return observation, float(reward), bool(terminated), False, info

    def _pure_ik_feedback_target(self, target: PoseTarget) -> PoseTarget:
        if self.step_count < config.PRE_ACTION_TICKS:
            return target

        actual_position, actual_quaternion = self._selected_pose()
        position_correction = np.clip(
            self.ik_position_feedback_gain
            * (np.asarray(target.position, dtype=np.float64) - actual_position),
            -self.ik_max_position_correction_m,
            self.ik_max_position_correction_m,
        )
        command_position = np.asarray(target.position, dtype=np.float64) + position_correction

        reference = self._action_reference_quaternion
        target_relative = quat_multiply(
            quat_conjugate(reference), target.quaternion
        )
        actual_relative = quat_multiply(
            quat_conjugate(reference), actual_quaternion
        )
        target_euler = quat_to_euler_xyz_deg(target_relative)
        actual_euler = quat_to_euler_xyz_deg(actual_relative)
        angle_correction = np.clip(
            self.ik_orientation_feedback_gain
            * wrap_degrees(target_euler - actual_euler),
            -self.ik_max_angle_correction_deg,
            self.ik_max_angle_correction_deg,
        )
        command_quaternion = quat_multiply(
            reference,
            euler_xyz_deg_to_quat(target_euler + angle_correction),
        )
        return PoseTarget(
            position=command_position,
            quaternion=command_quaternion,
            local_tip_centres=target.local_tip_centres,
        )

    def _target_for_tick(self, tick: int) -> PoseTarget:
        identity = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        rest = np.array((0.0, 0.0, self.object_spec.rest_grasp_plane_z_m))
        lifted = np.array((0.0, 0.0, self.task_target_grasp_plane_z_m))
        contact = self.object_spec.tip_centres_local_m(pregrasp=False)
        pregrasp = self.object_spec.tip_centres_local_m(pregrasp=True)
        if tick < config.APPROACH_TICKS:
            fraction = minimum_jerk((tick + 1) / config.APPROACH_TICKS)
            local = {
                finger: pregrasp[finger]
                + fraction * (contact[finger] - pregrasp[finger])
                for finger in FINGERS
            }
            return PoseTarget(rest, identity, local)
        if tick < config.APPROACH_TICKS + config.CONTACT_TICKS:
            return PoseTarget(rest, identity, contact)
        lift_start = config.APPROACH_TICKS + config.CONTACT_TICKS
        if tick < lift_start + config.LIFT_TICKS:
            fraction = minimum_jerk((tick - lift_start + 1) / config.LIFT_TICKS)
            return PoseTarget(rest + fraction * (lifted - rest), identity, contact)
        if tick < config.PRE_ACTION_TICKS or self.phase == config.Phase.GRASP_LIFT:
            return PoseTarget(lifted, identity, contact)

        self._capture_action_reference()
        action_tick = tick - config.PRE_ACTION_TICKS
        if self.phase == config.Phase.TRANSLATE:
            assert self._translation_schedule is not None
            position = self._action_reference_position + self._translation_schedule[
                action_tick
            ]
            quaternion = self._action_reference_quaternion.copy()
            if self.coupled_rotation:
                assert self._rotation_quaternion_schedule is not None
                quaternion = quat_multiply(
                    quaternion,
                    self._rotation_quaternion_schedule[action_tick],
                )
            return PoseTarget(position, quaternion, contact)
        assert self.phase == config.Phase.ROTATE
        assert self._rotation_quaternion_schedule is not None
        delta_quaternion = self._rotation_quaternion_schedule[action_tick]
        quaternion = quat_multiply(
            self._action_reference_quaternion, delta_quaternion
        )
        return PoseTarget(self._action_reference_position.copy(), quaternion, contact)

    def _capture_action_reference(self) -> None:
        if self._action_reference_captured:
            return
        position, quaternion = self._selected_pose()
        # Preserve the measured X/Y and orientation while retaining the active
        # phase's exact world grasp-plane target.
        self._action_reference_position[:] = (
            position[0],
            position[1],
            self.task_target_grasp_plane_z_m,
        )
        self._action_reference_quaternion[:] = quaternion
        self._action_reference_captured = True

    @staticmethod
    def _world_tip_targets(
        position: np.ndarray,
        quaternion: np.ndarray,
        local_tip_centres: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        rotation = quat_to_matrix(quaternion)
        return {
            finger: np.asarray(position, dtype=np.float64)
            + rotation @ np.asarray(local_tip_centres[finger], dtype=np.float64)
            for finger in FINGERS
        }

    def _nominal_for_target(self, target: PoseTarget) -> np.ndarray:
        command_position = target.position.copy()
        command_quaternion = target.quaternion
        cage_position = command_position.copy()
        cage_quaternion = command_quaternion
        lift_start = config.APPROACH_TICKS + config.CONTACT_TICKS
        if self.step_count >= lift_start:
            lift_fraction = float(
                np.clip(
                    (target.position[2] - self.object_spec.rest_grasp_plane_z_m)
                    / max(self.task_lift_distance_m, 1.0e-9),
                    0.0,
                    1.0,
                )
            )
            command_position[2] += (
                lift_fraction * config.NOMINAL_GRASP_Z_BIAS_M[self.gripper]
            )
            cage_position[2] = command_position[2]
        cone_preaction = (
            self.object_name == "cone"
            and self.step_count >= lift_start
            and (
                self.step_count < config.PRE_ACTION_TICKS
                or self.phase == config.Phase.GRASP_LIFT
            )
        )
        if cone_preaction:
            targets = self._cone_fast_lift_tip_targets(target)
        else:
            if (
                self.phase != config.Phase.GRASP_LIFT
                and self.step_count >= config.PRE_ACTION_TICKS
            ):
                # The task target is relative to the measured post-lift object
                # pose, while the analytic cage finished pre-action at its
                # nominal zero frame.  Remove the measured tracking offset so
                # Phase 2 commands only path offsets and Phase 3 only its
                # scheduled delta; otherwise the initial error is applied a
                # second time at the curriculum seam.
                cage_position[:2] -= self._action_reference_position[:2]
                cage_quaternion = quat_multiply(
                    quat_conjugate(self._action_reference_quaternion),
                    command_quaternion,
                )
            targets = self._world_tip_targets(
                cage_position, cage_quaternion, target.local_tip_centres
            )
        solution = self.kinematics.solve(
            targets, self.nominal_ctrl, self._previous_solutions
        )
        if bool(solution.success):
            self._previous_solutions = dict(solution.finger_solutions)
            self._last_feasible_position = cage_position.copy()
            self._last_feasible_quaternion = cage_quaternion.copy()
            self._ik_success = True
            self._ik_projection_fraction = 1.0
            return np.asarray(solution.ctrl, dtype=np.float64)

        # Keep a valid nominal cage even when fixed material anchors at the
        # requested X/Y rotation leave the analytic workspace.  Pure IK uses
        # the projected solution directly; the legacy action path may still
        # add an actuator residual after this nominal solve.
        low = 0.0
        high = 1.0
        best_solution: Any | None = None
        best_fraction = 0.0
        best_position = self._last_feasible_position.copy()
        best_quaternion = self._last_feasible_quaternion.copy()
        for _ in range(12):
            fraction = 0.5 * (low + high)
            position = self._last_feasible_position + fraction * (
                cage_position - self._last_feasible_position
            )
            quaternion = slerp(
                self._last_feasible_quaternion, cage_quaternion, fraction
            )
            candidate_targets = self._world_tip_targets(
                position, quaternion, target.local_tip_centres
            )
            candidate = self.kinematics.solve(
                candidate_targets, self.nominal_ctrl, self._previous_solutions
            )
            if bool(candidate.success):
                low = fraction
                best_fraction = fraction
                best_solution = candidate
                best_position = position
                best_quaternion = quaternion
            else:
                high = fraction
        self._ik_success = False
        self._ik_projection_fraction = best_fraction
        if best_solution is None:
            return self.nominal_ctrl.copy()
        self._previous_solutions = dict(best_solution.finger_solutions)
        self._last_feasible_position = best_position
        self._last_feasible_quaternion = best_quaternion
        return np.asarray(best_solution.ctrl, dtype=np.float64)

    def _cone_fast_lift_tip_targets(
        self, target: PoseTarget
    ) -> dict[str, np.ndarray]:
        """Use a desired-frame, ten-tick cone lift without pose feedback.

        Feeding the cone's measured quaternion back into the four fixed finger
        assignments creates a yaw-positive-feedback loop.  A short calibrated
        desired-frame cage motion reaches the requested grasp height, while
        MuJoCo's no-slip iterations stabilize the upward-tapered contact.
        """

        lift_start = config.APPROACH_TICKS + config.CONTACT_TICKS
        progress = minimum_jerk(
            (self.step_count - lift_start + 1) / config.CONE_FAST_LIFT_TICKS
        )
        rest_z = self.object_spec.rest_grasp_plane_z_m
        position = np.array(
            (
                target.position[0],
                target.position[1],
                rest_z
                + progress
                * (
                    self.task_target_grasp_plane_z_m
                    + config.CONE_NOMINAL_CAGE_Z_OFFSET_M
                    - rest_z
                ),
            ),
            dtype=np.float64,
        )
        return self._world_tip_targets(
            position, target.quaternion, target.local_tip_centres
        )

    def _rate_limit_nominal(self, requested: np.ndarray) -> np.ndarray:
        requested = np.clip(
            np.asarray(requested, dtype=np.float64), self.ctrl_low, self.ctrl_high
        )
        limits = np.full(self.model.nu, 0.055, dtype=np.float64)
        if self.gripper == config.GripperKind.G6.value:
            limits[list(config.G6_LINEAR_ACTUATOR_INDICES)] = 0.002
        delta = np.clip(requested - self.nominal_ctrl, -limits, limits)
        return np.clip(self.nominal_ctrl + delta, self.ctrl_low, self.ctrl_high)

    def _selected_pose(self) -> tuple[np.ndarray, np.ndarray]:
        body_id = self._object_body_ids[self.object_name]
        return (
            self.data.xpos[body_id].astype(np.float64).copy(),
            normalize_quat(self.data.xquat[body_id]),
        )

    def _pose_errors(self) -> tuple[np.ndarray, np.ndarray]:
        position, quaternion = self._selected_pose()
        return (
            position - self.target_position,
            orientation_error_xyz_deg(quaternion, self.target_quaternion),
        )

    def _contact_mask(self) -> np.ndarray:
        mask = np.zeros(4, dtype=np.float64)
        object_geom = self._object_geom_ids[self.object_name]
        tip_by_id = {geom_id: index for index, geom_id in enumerate(self._tip_geom_ids.values())}
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first = int(contact.geom1)
            second = int(contact.geom2)
            if first == object_geom and second in tip_by_id:
                mask[tip_by_id[second]] = 1.0
            elif second == object_geom and first in tip_by_id:
                mask[tip_by_id[first]] = 1.0
        return mask

    def _update_route_metrics(
        self, position_error: np.ndarray, angle_error: np.ndarray
    ) -> None:
        if self.phase != config.Phase.TRANSLATE:
            return
        action_tick = self.step_count - 1 - config.PRE_ACTION_TICKS
        route_start, route_stop = trajectories.PHASE_BOUNDS["route"]
        if route_start <= action_tick < route_stop:
            self._max_route_abs_position_error = np.maximum(
                self._max_route_abs_position_error, np.abs(position_error)
            )
            self._max_route_abs_angle_error = np.maximum(
                self._max_route_abs_angle_error, np.abs(angle_error)
            )

    def _endpoint_success(self) -> bool:
        position_error, angle_error = self._pose_errors()
        pose_ok = config.componentwise_pose_within_tolerance(
            position_error, angle_error
        )
        contacts_ok = int(self._contact_mask().sum()) >= config.REQUIRED_TIP_CONTACTS
        if self.phase == config.Phase.TRANSLATE:
            route_ok = bool(
                np.all(
                    self._max_route_abs_position_error
                    <= config.POSITION_TOLERANCE_M_PER_AXIS
                )
                and np.all(
                    self._max_route_abs_angle_error
                    <= config.ANGLE_TOLERANCE_DEG_PER_AXIS
                )
            )
        else:
            route_ok = True
        return pose_ok and contacts_ok and route_ok

    def _failure_reason(self) -> str | None:
        values = np.concatenate(
            (self.data.qpos, self.data.qvel, self.data.ctrl, self.applied_ctrl)
        )
        if not np.all(np.isfinite(values)):
            return "non_finite_state"
        object_joint = self._object_joint_ids[self.object_name]
        object_qpos = int(self.model.jnt_qposadr[object_joint])
        if float(np.linalg.norm(self.data.qpos[object_qpos + 3 : object_qpos + 7])) < 0.5:
            return "invalid_object_quaternion"
        position, _ = self._selected_pose()
        robot_qvel = self.data.qvel[self._robot_qvel_adrs]
        if float(np.max(np.abs(robot_qvel))) > config.MAX_ABS_QVEL:
            return "unstable_joint_velocity"
        object_dof = int(self.model.jnt_dofadr[object_joint])
        object_velocity = self.data.qvel[object_dof : object_dof + 6]
        if (
            float(np.linalg.norm(object_velocity[:3]))
            > config.MAX_OBJECT_LINEAR_SPEED_M_S
        ):
            return "unstable_object_linear_velocity"
        if (
            float(np.linalg.norm(object_velocity[3:]))
            > config.MAX_OBJECT_ANGULAR_SPEED_RAD_S
        ):
            return "unstable_object_angular_velocity"
        lift_end = config.APPROACH_TICKS + config.CONTACT_TICKS + config.LIFT_TICKS
        if self.step_count >= lift_end:
            minimum_held_height = (
                self.object_spec.rest_grasp_plane_z_m
                + config.DROP_MIN_LIFT_FRACTION * self.task_lift_distance_m
            )
            if position[2] < minimum_held_height:
                return "object_dropped"
        return None

    def _reward(
        self,
        endpoint_success: bool,
        failure: str | None,
        actual_residual: np.ndarray,
    ) -> float:
        position_error, angle_error = self._pose_errors()
        position_score = float(
            np.exp(
                -np.mean(np.abs(position_error)) / config.POSITION_REWARD_SCALE_M
            )
        )
        orientation_score = float(
            np.exp(
                -np.mean(np.abs(angle_error))
                / config.ORIENTATION_REWARD_SCALE_DEG
            )
        )
        contact_score = float(self._contact_mask().mean())
        position, _ = self._selected_pose()
        height_fraction = float(
            np.clip(
                (position[2] - self.object_spec.rest_grasp_plane_z_m)
                / max(self.task_lift_distance_m, 1.0e-6),
                0.0,
                1.0,
            )
        )
        residual_ratio = actual_residual / self.residual_controller.limits
        residual_mean_square = float(np.mean(np.square(residual_ratio)))
        residual_penalty = (
            config.G3_RESIDUAL_PENALTY_WEIGHT
            * float(np.sqrt(residual_mean_square))
            if self.gripper == config.GripperKind.G3.value
            else config.G6_RESIDUAL_PENALTY_WEIGHT * residual_mean_square
        )
        reward = (
            position_score
            + orientation_score
            + config.CONTACT_REWARD_WEIGHT * contact_score
            + config.HEIGHT_REWARD_WEIGHT * height_fraction
            - residual_penalty
        )
        if endpoint_success:
            reward += config.ENDPOINT_SUCCESS_BONUS
        if failure is not None:
            reward -= config.ENDPOINT_SUCCESS_BONUS
        return reward

    def _observation(self) -> np.ndarray:
        actuator_qpos = self.data.qpos[self._actuator_qpos_adrs]
        actuator_qvel = self.data.qvel[self._actuator_qvel_adrs]
        position, quaternion = self._selected_pose()
        object_joint = self._object_joint_ids[self.object_name]
        object_dof = int(self.model.jnt_dofadr[object_joint])
        object_velocity = self.data.qvel[object_dof : object_dof + 6]
        position_error, angle_error = self._pose_errors()
        tips = np.concatenate(
            [self.data.xpos[self._tip_body_ids[finger]] for finger in FINGERS]
        )

        def padded(value: np.ndarray) -> np.ndarray:
            result = np.zeros(MAX_ACTUATORS, dtype=np.float64)
            result[: value.size] = value
            return result

        phase_onehot = np.zeros(3, dtype=np.float64)
        phase_onehot[int(self.phase) - 1] = 1.0
        object_onehot = np.zeros(4, dtype=np.float64)
        object_onehot[config.OBJECT_NAMES.index(self.object_name)] = 1.0
        path_onehot = np.zeros(5, dtype=np.float64)
        if self.path_name is not None:
            path_onehot[trajectories.PATH_NAMES.index(self.path_name)] = 1.0
        axes_onehot = np.zeros(7, dtype=np.float64)
        if self.rotation_axes is not None:
            axes_onehot[config.ROTATION_AXIS_SETS.index(self.rotation_axes)] = 1.0
        gripper_onehot = np.array(
            (1.0, 0.0) if self.gripper == "g3" else (0.0, 1.0),
            dtype=np.float64,
        )
        observation = np.concatenate(
            (
                padded(actuator_qpos),
                padded(actuator_qvel),
                padded(self.nominal_ctrl),
                padded(self.applied_ctrl),
                padded(self.last_residual),
                tips,
                position,
                quaternion,
                object_velocity,
                self.target_position,
                self.target_quaternion,
                position_error,
                angle_error,
                self._contact_mask(),
                np.array((self.step_count / self.episode_ticks,)),
                phase_onehot,
                object_onehot,
                path_onehot,
                axes_onehot,
                self.rotation_goal_euler_deg / self.rotation_goal_max_deg,
                gripper_onehot,
                np.array((float(self._ik_success), self._ik_projection_fraction)),
            )
        ).astype(np.float32)
        if observation.shape != (OBSERVATION_DIM,):
            raise RuntimeError(f"observation shape changed: {observation.shape}")
        return observation

    def _info(self, *, endpoint: bool) -> dict[str, Any]:
        position, quaternion = self._selected_pose()
        position_error, angle_error = self._pose_errors()
        return {
            "gripper": self.gripper,
            "phase": int(self.phase),
            "object": self.object_name,
            "path": self.path_name,
            "rotation_axes": self.rotation_axes,
            "coupled_rotation": bool(self.coupled_rotation),
            "task": (
                "translate_rotate"
                if self.coupled_rotation
                else self.phase.name.lower()
            ),
            "rotation_goal_euler_deg": self.rotation_goal_euler_deg.copy(),
            "rotation_goal_max_deg": float(self.rotation_goal_max_deg),
            "rotation_sampling_profile": self.rotation_sampling_profile,
            "rotation_active_axis_mask": np.asarray(
                [
                    float(
                        self.rotation_axes is not None
                        and axis in self.rotation_axes
                    )
                    for axis in "xyz"
                ],
                dtype=np.float64,
            ),
            "task_target_grasp_plane_z_m": float(
                self.task_target_grasp_plane_z_m
            ),
            "control_mode": self._control_mode,
            "residual_nominal_feedback": bool(self.residual_nominal_feedback),
            "residual_after_preaction_only": bool(
                self.residual_after_preaction_only
            ),
            "residual_active": bool(self._residual_active),
            "residual_requested_action": (
                self._last_requested_residual_action.copy()
            ),
            "residual_executed_action": (
                self._last_executed_residual_action.copy()
            ),
            "actuator_residual": self.last_residual.copy(),
            "action_reference_captured": bool(self._action_reference_captured),
            "action_reference_position_m": self._action_reference_position.copy(),
            "action_reference_quaternion_wxyz": (
                self._action_reference_quaternion.copy()
            ),
            "step": self.step_count,
            "episode_ticks": self.episode_ticks,
            "object_grasp_frame_position_m": position,
            "object_quaternion_wxyz": quaternion,
            "target_position_m": self.target_position.copy(),
            "target_quaternion_wxyz": self.target_quaternion.copy(),
            "ik_command_position_m": self.ik_command_position.copy(),
            "ik_command_quaternion_wxyz": self.ik_command_quaternion.copy(),
            "ik_feasible_cage_position_m": self._last_feasible_position.copy(),
            "ik_feasible_cage_quaternion_wxyz": (
                self._last_feasible_quaternion.copy()
            ),
            "position_error_m_per_axis": position_error,
            "angle_error_deg_per_axis": angle_error,
            "position_tolerance_m_per_axis": config.POSITION_TOLERANCE_M_PER_AXIS,
            "angle_tolerance_deg_per_axis": config.ANGLE_TOLERANCE_DEG_PER_AXIS,
            "tip_contact_count": int(self._contact_mask().sum()),
            "ik_exact": bool(self._ik_success),
            "ik_projection_fraction": float(self._ik_projection_fraction),
            "max_abs_actuator_residual": float(
                np.max(np.abs(self.last_residual), initial=0.0)
            ),
            "max_route_abs_position_error_m": self._max_route_abs_position_error.copy(),
            "max_route_abs_angle_error_deg": self._max_route_abs_angle_error.copy(),
            "termination_reason": self._termination_reason,
            "endpoint": bool(endpoint),
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()


def make_env(
    gripper: str,
    phase: int | config.Phase,
    **kwargs: Any,
) -> G3G6ManipulationEnv:
    return G3G6ManipulationEnv(gripper=gripper, phase=phase, **kwargs)
