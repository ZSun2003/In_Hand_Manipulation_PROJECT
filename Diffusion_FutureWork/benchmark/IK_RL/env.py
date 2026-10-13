"""Gymnasium environment for the current G3 IK controller plus RL residuals."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from threading import Thread
from typing import Any, Callable, Mapping, Sequence

from .runtime import import_current_controller, validate_parent_contract

grab = import_current_controller()

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from _g3_objects import GraspPlane, ObjectKind, ObjectSpec, get_object_spec

from . import config
from .controller_bridge import (
    BridgeAborted,
    ControlTick,
    ControlTickBridge,
    ControllerDone,
)


OBSERVATION_DIM = 89
CONTROL_MODES = ("ik_only", "ik_residual", "ik_residual_no_plan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _initial_com(
    object_kind: str,
    body_position: Sequence[float],
    body_quaternion: Sequence[float],
) -> np.ndarray:
    spec = get_object_spec(object_kind)
    return (
        np.asarray(body_position, dtype=np.float64).reshape(3)
        + grab.quat_to_rot(body_quaternion)
        @ np.asarray(spec.local_com_m, dtype=np.float64).reshape(3)
    )


class _PlanningHandshake:
    """Apply the first policy action before the current banana IK precheck."""

    def __init__(
        self,
        delegate: Any,
        *,
        height_residual_m: float,
        yaw_residual_rad: float,
        before_plan: Callable[[], None],
    ) -> None:
        self.delegate = delegate
        self.height_residual_m = float(height_residual_m)
        self.yaw_residual_rad = float(yaw_residual_rad)
        self.before_plan = before_plan
        self.callback_called = False
        self.applied = False
        self.candidate_valid = False
        self.reason = ""
        self.nominal_yaw_rad = float("nan")
        self.candidate_yaw_rad = float("nan")
        self.candidate_height_m = float("nan")

    @property
    def is_zero(self) -> bool:
        return bool(
            abs(self.height_residual_m) <= config.PLAN_ZERO_EPS
            and abs(self.yaw_residual_rad) <= config.PLAN_ZERO_EPS
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def _capture_once(self) -> None:
        if not self.callback_called:
            self.before_plan()
            self.callback_called = True

    def plan(
        self,
        spec: str | ObjectKind | ObjectSpec,
        *,
        ik: Any = None,
        body_world_pos: Sequence[float] | None = None,
        body_world_quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        table_top_z: float = grab.TABLE_TOP_Z_M,
        table_clearance_m: float = grab.TIP_TABLE_CLEARANCE_M,
    ) -> Any:
        # The paired-state fingerprint is taken after current reset settling,
        # but before the policy-selected planning residual is applied.
        self._capture_once()
        object_spec = (
            spec if isinstance(spec, ObjectSpec) else get_object_spec(spec)
        )
        if object_spec.kind is not ObjectKind.BANANA:
            return self.delegate.plan(
                object_spec,
                ik=ik,
                body_world_pos=body_world_pos,
                body_world_quat=body_world_quat,
                table_top_z=table_top_z,
                table_clearance_m=table_clearance_m,
            )

        self.applied = True
        if self.is_zero:
            # Preserve the planner-selected nominal and its geometry/IK
            # validation outcome exactly for the zero-action baseline.
            return self.delegate.plan(
                object_spec,
                ik=ik,
                body_world_pos=body_world_pos,
                body_world_quat=body_world_quat,
                table_top_z=table_top_z,
                table_clearance_m=table_clearance_m,
            )

        geometry_nominal = self.delegate.plan(
            object_spec,
            ik=None,
            body_world_pos=body_world_pos,
            body_world_quat=body_world_quat,
            table_top_z=table_top_z,
            table_clearance_m=table_clearance_m,
        )
        self.nominal_yaw_rad = float(geometry_nominal.plane.yaw_rad)
        center = np.asarray(
            geometry_nominal.plane.center_local, dtype=np.float64
        ).copy()
        center[2] = float(
            np.clip(
                center[2] + self.height_residual_m,
                object_spec.local_bottom_z_m,
                object_spec.local_top_z_m,
            )
        )
        yaw = math.atan2(
            math.sin(self.nominal_yaw_rad + self.yaw_residual_rad),
            math.cos(self.nominal_yaw_rad + self.yaw_residual_rad),
        )
        plane = GraspPlane(
            center_local=center,
            roll_rad=float(geometry_nominal.plane.roll_rad),
            pitch_rad=float(geometry_nominal.plane.pitch_rad),
            yaw_rad=yaw,
        )
        candidate = self.delegate._build_candidate(
            object_spec,
            plane,
            ik=ik,
            body_world_pos=body_world_pos,
            body_world_quat=body_world_quat,
            table_top_z=table_top_z,
            table_clearance_m=table_clearance_m,
        )
        self.candidate_valid = bool(candidate.valid)
        self.reason = str(candidate.diagnostics.get("reason", ""))
        self.candidate_yaw_rad = float(candidate.plane.yaw_rad)
        self.candidate_height_m = float(candidate.plane.center_local[2])
        if not candidate.valid:
            reason = self.reason or "invalid_policy_grasp_plane"
            raise RuntimeError(
                "no feasible adaptive grasp plane for banana: " + reason
            )
        return candidate

    def record(self) -> dict[str, Any]:
        def finite_or_none(value: float) -> float | None:
            return float(value) if math.isfinite(float(value)) else None

        def degrees_or_none(value: float) -> float | None:
            return (
                math.degrees(float(value))
                if math.isfinite(float(value))
                else None
            )

        return {
            "role": "g3_banana_reachability_residual",
            "applied": self.applied,
            "height_residual_m": self.height_residual_m,
            "yaw_residual_rad": self.yaw_residual_rad,
            "yaw_residual_deg": math.degrees(self.yaw_residual_rad),
            "nominal_yaw_rad": finite_or_none(self.nominal_yaw_rad),
            "nominal_yaw_deg": degrees_or_none(self.nominal_yaw_rad),
            "candidate_yaw_rad": finite_or_none(self.candidate_yaw_rad),
            "candidate_yaw_deg": degrees_or_none(self.candidate_yaw_rad),
            "candidate_height_m": finite_or_none(self.candidate_height_m),
            "candidate_valid": self.candidate_valid,
            "reason": self.reason,
            "zero_action_preserves_current_nominal": self.is_zero,
            "state_captured_before_application": self.callback_called,
        }


class G3IKResidualEnv(gym.Env[np.ndarray, np.ndarray]):
    """Uniform six-object/two-task wrapper around the current G3 controller."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        control_mode: str = "ik_residual",
        record_trajectory: bool = False,
        fixed_object: str | None = None,
        fixed_task: str | None = None,
    ) -> None:
        super().__init__()
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {CONTROL_MODES}")
        if fixed_object is not None and fixed_object not in config.OBJECTS:
            raise ValueError(f"unknown fixed object: {fixed_object}")
        if fixed_task is not None and fixed_task not in config.TASKS:
            raise ValueError(f"unknown fixed task: {fixed_task}")

        config.validate()
        self.parent_contract = validate_parent_contract(grab)
        self.control_mode = control_mode
        self.record_trajectory = bool(record_trajectory)
        self.fixed_object = fixed_object
        self.fixed_task = fixed_task
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0e6,
            high=1.0e6,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

        self.grabber = grab.IKGrabber(render=False)
        self._actuator_qpos_adrs = np.asarray(
            [
                self.grabber.model.jnt_qposadr[int(joint_id)]
                for joint_id in self.grabber._actuator_joint_ids
            ],
            dtype=np.int32,
        )
        self._actuator_qvel_adrs = np.asarray(
            [
                self.grabber.model.jnt_dofadr[int(joint_id)]
                for joint_id in self.grabber._actuator_joint_ids
            ],
            dtype=np.int32,
        )
        self.bridge: ControlTickBridge | None = None
        self.worker: Thread | None = None
        self.worker_original_physics: Any = None
        self.handshake_pending = True
        self.episode_done = False
        self.initial_state_captured = False
        self.action_target_ready = False
        self.object_kind = config.OBJECTS[0]
        self.task = config.TASKS[0]
        self.episode_seed = 0
        self.plan_record: dict[str, Any] = {}
        self.precondition_record: dict[str, Any] = {}
        self.trajectory: list[dict[str, Any]] = []
        self.action_target_pos = np.zeros(3, dtype=np.float64)
        self.action_target_rotation = np.eye(3, dtype=np.float64)
        self.action_start_com = np.zeros(3, dtype=np.float64)
        self.action_start_body_pos = np.zeros(3, dtype=np.float64)
        self.action_start_rotation = np.eye(3, dtype=np.float64)
        self.overall_target_pos = np.zeros(3, dtype=np.float64)
        self.overall_target_rotation = np.eye(3, dtype=np.float64)
        self.initial_state_hash = ""
        self.integration_state_hash = ""
        self.controller_state_hash = ""
        self.previous_cost = 0.0
        self.last_tick: ControlTick | None = None
        self.last_endpoint = self._empty_endpoint()
        self.last_overall_endpoint = self._empty_endpoint()

    @staticmethod
    def _empty_endpoint() -> dict[str, Any]:
        return {
            "position_error_m": float("nan"),
            "position_error_xyz_m": [float("nan")] * 3,
            "euler_error_xyz_deg": [float("nan")] * 3,
            "euler_l2_error_deg": float("nan"),
            "position_tolerance_m": config.POSITION_TOLERANCE_M,
            "rotation_tolerance_deg": float("nan"),
            "position_ok": False,
            "orientation_ok": False,
            "success": False,
        }

    def _choose_pair(self, options: Mapping[str, Any]) -> None:
        object_option = options.get("object_kind", self.fixed_object)
        task_option = options.get("task", self.fixed_task)
        if object_option is None and task_option is None:
            pair_index = int(
                self.np_random.integers(
                    0, len(config.OBJECTS) * len(config.TASKS)
                )
            )
            object_option = config.OBJECTS[pair_index // len(config.TASKS)]
            task_option = config.TASKS[pair_index % len(config.TASKS)]
        else:
            if object_option is None:
                object_option = config.OBJECTS[
                    int(self.np_random.integers(0, len(config.OBJECTS)))
                ]
            if task_option is None:
                task_option = config.TASKS[
                    int(self.np_random.integers(0, len(config.TASKS)))
                ]
        if object_option not in config.OBJECTS:
            raise ValueError(f"object is outside the formal RL set: {object_option}")
        if task_option not in config.TASKS:
            raise ValueError(f"unknown task: {task_option}")
        self.object_kind = str(object_option)
        self.task = str(task_option)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._stop_worker()
        options_value: dict[str, Any] = {} if options is None else dict(options)
        self._choose_pair(options_value)
        self.episode_seed = int(
            options_value.get(
                "pair_seed",
                self.np_random.integers(0, np.iinfo(np.int32).max),
            )
        )
        self.handshake_pending = True
        self.episode_done = False
        self.initial_state_captured = False
        self.action_target_ready = False
        self.plan_record = {}
        self.precondition_record = {}
        self.trajectory = []
        self.bridge = None
        self.last_tick = None
        self.initial_state_hash = ""
        self.integration_state_hash = ""
        self.controller_state_hash = ""
        self.last_endpoint = self._empty_endpoint()
        self.last_overall_endpoint = self._empty_endpoint()
        self.previous_cost = 0.0
        return self._observation(), self._info(
            planning_handshake=True,
            control_tick=False,
        )

    def _capture_initial_state(self) -> None:
        """Fingerprint official MuJoCo integration state and controller state."""

        state_spec = grab.mujoco.mjtState.mjSTATE_INTEGRATION
        state_size = grab.mujoco.mj_stateSize(self.grabber.model, state_spec)
        integration_state = np.empty(state_size, dtype=np.float64)
        grab.mujoco.mj_getState(
            self.grabber.model,
            self.grabber.data,
            integration_state,
            state_spec,
        )
        controller_state = {
            "object_kind": self.object_kind,
            "task": self.task,
            "seed": self.episode_seed,
            "controller_phase": str(self.grabber.phase),
            "controller_phase_step": int(self.grabber.phase_step),
            "controller_trial_step": int(self.grabber.trial_step),
            "executed_ctrl": self.grabber.executed_ctrl.tolist(),
            "last_ctrl_delta": self.grabber.last_ctrl_delta.tolist(),
            "last_tip_targets_world": {
                finger: target.tolist()
                for finger, target in sorted(
                    self.grabber.last_tip_targets_world.items()
                )
            },
            "closure_fraction": self.grabber.closure_fraction.tolist(),
            "contact_locked": self.grabber.contact_locked.astype(int).tolist(),
            "initial_body_pos": self.grabber.initial_body_pos.tolist(),
            "initial_body_quat": self.grabber.initial_body_quat.tolist(),
            "goal_body_pos": self.grabber.goal_body_pos.tolist(),
            "commanded_lift_height_m": float(
                self.grabber.commanded_lift_height_m
            ),
        }
        controller_bytes = json.dumps(
            controller_state,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.integration_state_hash = hashlib.sha256(
            integration_state.tobytes(order="C")
        ).hexdigest()
        self.controller_state_hash = hashlib.sha256(controller_bytes).hexdigest()
        combined = hashlib.sha256()
        combined.update(integration_state.tobytes(order="C"))
        combined.update(controller_bytes)
        self.initial_state_hash = combined.hexdigest()
        self.initial_state_captured = True

        initial_rotation = grab.quat_to_rot(self.grabber.initial_body_quat)
        initial_com = _initial_com(
            self.object_kind,
            self.grabber.initial_body_pos,
            self.grabber.initial_body_quat,
        )
        self.overall_target_pos = initial_com + np.asarray(
            (0.0, 0.0, grab.LIFT_HEIGHT_M), dtype=np.float64
        )
        self.overall_target_rotation = initial_rotation.copy()
        if self.task == "translate_xy45_100mm":
            self.overall_target_pos += np.asarray(
                config.TRANSLATION_VECTOR_M, dtype=np.float64
            )
        else:
            self.overall_target_rotation = (
                initial_rotation
                @ grab._axis_rotation("z", config.ROTATION_ANGLE_RAD)
            )
        self._refresh_endpoints()

    def _set_action_target_after_lift(self) -> None:
        start_com = self.grabber._object_com_position()
        body_position, start_quaternion = self.grabber._object_pose()
        start_rotation = grab.quat_to_rot(start_quaternion)
        self.action_start_com = start_com.copy()
        self.action_start_body_pos = body_position.copy()
        self.action_start_rotation = start_rotation.copy()
        if self.task == "translate_xy45_100mm":
            self.action_target_pos = start_com + np.asarray(
                config.TRANSLATION_VECTOR_M, dtype=np.float64
            )
            self.action_target_rotation = start_rotation.copy()
        else:
            self.action_target_pos = start_com.copy()
            self.action_target_rotation = (
                start_rotation
                @ grab._axis_rotation("z", config.ROTATION_ANGLE_RAD)
            )
        self.action_target_ready = True

    def _endpoint(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
    ) -> dict[str, Any]:
        current_com = self.grabber._object_com_position()
        _body_position, current_quaternion = self.grabber._object_pose()
        xyz_error = current_com - np.asarray(target_position, dtype=np.float64)
        euler_error = grab._fixed_xyz_euler_error_deg(
            np.asarray(target_rotation, dtype=np.float64),
            current_quaternion,
        )
        position_error = float(np.linalg.norm(xyz_error))
        euler_l2 = float(np.linalg.norm(euler_error))
        finite = bool(
            np.all(np.isfinite(xyz_error))
            and np.all(np.isfinite(euler_error))
            and np.isfinite(position_error)
            and np.isfinite(euler_l2)
        )
        position_ok = bool(
            finite
            and position_error <= config.POSITION_TOLERANCE_M + 1.0e-12
        )
        orientation_tolerance = config.rotation_tolerance_deg(self.object_kind)
        orientation_ok = bool(
            finite and euler_l2 <= orientation_tolerance + 1.0e-10
        )
        return {
            "position_error_m": position_error,
            "position_error_xyz_m": xyz_error.tolist(),
            "euler_error_xyz_deg": euler_error.tolist(),
            "euler_l2_error_deg": euler_l2,
            "position_tolerance_m": config.POSITION_TOLERANCE_M,
            "rotation_tolerance_deg": orientation_tolerance,
            "position_ok": position_ok,
            "orientation_ok": orientation_ok,
            "success": bool(position_ok and orientation_ok),
        }

    def _refresh_endpoints(self) -> None:
        if not self.initial_state_captured:
            self.last_endpoint = self._empty_endpoint()
            self.last_overall_endpoint = self._empty_endpoint()
            return
        self.last_overall_endpoint = self._endpoint(
            self.overall_target_pos, self.overall_target_rotation
        )
        self.last_endpoint = (
            self._endpoint(self.action_target_pos, self.action_target_rotation)
            if self.action_target_ready
            else self._empty_endpoint()
        )

    @staticmethod
    def _tracking_cost(endpoint: Mapping[str, Any]) -> float:
        position_error = float(endpoint["position_error_m"])
        rotation_error = float(endpoint["euler_l2_error_deg"])
        rotation_tolerance = float(endpoint["rotation_tolerance_deg"])
        if not (
            np.isfinite(position_error)
            and np.isfinite(rotation_error)
            and np.isfinite(rotation_tolerance)
        ):
            return 100.0
        return float(
            position_error / config.POSITION_TOLERANCE_M
            + rotation_error / max(1.0, rotation_tolerance)
        )

    def _reward_endpoint(self) -> Mapping[str, Any]:
        return (
            self.last_endpoint
            if self.action_target_ready
            else self.last_overall_endpoint
        )

    def _snapshot(
        self,
        control_step: int,
        base_ctrl: np.ndarray,
        applied_ctrl: np.ndarray,
        residual: np.ndarray,
    ) -> dict[str, Any]:
        self._refresh_endpoints()
        body_position, body_quaternion = self.grabber._object_pose()
        linear_velocity, angular_velocity = self.grabber._object_velocity()
        reference_position = self.grabber.initial_body_pos.copy()
        reference_rotation = grab.quat_to_rot(self.grabber.initial_body_quat)
        if self.grabber.entered_lift:
            reference_rotation = grab.quat_to_rot(
                self.grabber.lift_start_body_quat
            )
            if self.grabber.phase == grab.Phase.LIFT:
                blend = grab._minimum_jerk_fraction(
                    self.grabber.phase_step + 1,
                    grab.LIFT_STEPS,
                )
                reference_position = (
                    self.grabber.lift_start_body_pos
                    + np.asarray(
                        (
                            0.0,
                            0.0,
                            blend * self.grabber.commanded_lift_height_m,
                        ),
                        dtype=np.float64,
                    )
                )
            elif self.grabber.phase == grab.Phase.HOLD:
                reference_position = self.grabber.goal_body_pos.copy()
            else:
                reference_position = self.grabber.lift_start_body_pos.copy()
        tracking_position_error = body_position - reference_position
        tracking_euler_error = grab._fixed_xyz_euler_error_deg(
            reference_rotation,
            body_quaternion,
        )
        contact_forces = np.asarray(
            self.grabber.last_contact_forces, dtype=np.float64
        )
        contact_count = int(
            np.count_nonzero(
                contact_forces >= grab.CONTACT_FORCE_THRESHOLD_N
            )
        )
        action_along_track_mm = float("nan")
        action_cross_track_mm = float("nan")
        action_net_rotation_deg = float("nan")
        action_axis_angle_deg = float("nan")
        action_center_drift_mm = float("nan")
        if self.action_target_ready:
            current_com = self.grabber._object_com_position()
            displacement = current_com - self.action_start_com
            direction = np.asarray(
                config.TRANSLATION_VECTOR_M, dtype=np.float64
            )
            direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
            along = float(displacement @ direction)
            action_along_track_mm = 1000.0 * along
            action_cross_track_mm = 1000.0 * float(
                np.linalg.norm(displacement - along * direction)
            )
            action_axis_angle_deg = math.degrees(
                grab._signed_local_axis_angle(
                    self.action_start_rotation,
                    body_quaternion,
                    "z",
                )
            )
            relative_rotation = (
                self.action_start_rotation.T
                @ grab.quat_to_rot(body_quaternion)
            )
            action_net_rotation_deg = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            0.5 * (np.trace(relative_rotation) - 1.0),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            action_center_drift_mm = 1000.0 * float(
                np.linalg.norm(current_com - self.action_start_com)
            )
        payload = {
            "control_step": int(control_step),
            "time_s": float((control_step + 1) / config.CONTROL_HZ),
            "segment": (
                "post_lift_action"
                if self.action_target_ready
                else "grasp_lift_precondition"
            ),
            "action_target_ready": self.action_target_ready,
            "phase": str(self.grabber.phase),
            "phase_step": int(self.grabber.phase_step),
            "trial_step": int(self.grabber.trial_step),
            "residual_active": self.control_mode != "ik_only",
            "position_error_m": self.last_endpoint["position_error_m"],
            "euler_l2_error_deg": self.last_endpoint["euler_l2_error_deg"],
            "overall_position_error_m": (
                self.last_overall_endpoint["position_error_m"]
            ),
            "overall_euler_l2_error_deg": (
                self.last_overall_endpoint["euler_l2_error_deg"]
            ),
            "object_pos": body_position.tolist(),
            "object_quat": body_quaternion.tolist(),
            "reference_position_m": reference_position.tolist(),
            "reference_quaternion_wxyz": grab.rot_to_quat(
                reference_rotation
            ).tolist(),
            "tracking_position_error_xyz_mm": (
                1000.0 * tracking_position_error
            ).tolist(),
            "tracking_euler_error_xyz_deg": tracking_euler_error.tolist(),
            "measured_lift_mm": 1000.0
            * float(body_position[2] - self.grabber.initial_body_pos[2]),
            "horizontal_drift_mm": 1000.0
            * float(
                np.linalg.norm(
                    body_position[:2] - self.grabber.initial_body_pos[:2]
                )
            ),
            "linear_speed_mm_s": 1000.0
            * float(np.linalg.norm(linear_velocity)),
            "angular_speed_deg_s": math.degrees(
                float(np.linalg.norm(angular_velocity))
            ),
            "contact_forces_n": contact_forces.tolist(),
            "contact_count": contact_count,
            "action_along_track_mm": action_along_track_mm,
            "action_cross_track_mm": action_cross_track_mm,
            "action_net_rotation_deg": action_net_rotation_deg,
            "action_axis_angle_deg": action_axis_angle_deg,
            "action_center_drift_mm": action_center_drift_mm,
            "endpoint_position_error_xyz_m": self.last_endpoint[
                "position_error_xyz_m"
            ],
            "endpoint_euler_error_xyz_deg": self.last_endpoint[
                "euler_error_xyz_deg"
            ],
            "ctrl": applied_ctrl.tolist(),
            "nominal_ctrl": base_ctrl.tolist(),
            "residual": residual.tolist(),
        }
        if self.record_trajectory:
            self.trajectory.append(payload)
        return payload

    def _run_ready_and_demo(
        self,
        episode_bridge: ControlTickBridge,
    ) -> dict[str, Any]:
        if self.task == "translate_xy45_100mm":
            ready = self.grabber.run_translation_ready_trial(
                self.object_kind,
                seed=self.episode_seed,
                visualize=False,
            )
        else:
            ready = self.grabber.run_rotation_ready_trial(
                self.object_kind,
                axes="z",
                seed=self.episode_seed,
                visualize=False,
            )
        self.precondition_record = _jsonable(asdict(ready))
        outcome: dict[str, Any] = {
            "precondition": dict(self.precondition_record),
            "demo": None,
        }
        if not ready.success:
            return outcome

        # Target A is captured exactly at the current Figure 16/20 boundary.
        self._set_action_target_after_lift()
        if self.task == "translate_xy45_100mm":
            demo = self.grabber.run_translation_demo()
        else:
            demo = self.grabber.run_rotation_demo("z")
        outcome["demo"] = _jsonable(asdict(demo))
        return outcome

    def _start_episode_worker(
        self,
        handshake: _PlanningHandshake,
    ) -> ControllerDone | None:
        self.bridge = ControlTickBridge(
            ctrl_low=self.grabber._ctrl_low,
            ctrl_high=self.grabber._ctrl_high,
            snapshot=self._snapshot,
        )
        episode_bridge = self.bridge
        original_physics = self.grabber._physics_ctrl
        original_planner = self.grabber.planner
        self.worker_original_physics = original_physics
        self.grabber.planner = handshake

        def bridged(ctrl: Sequence[float]) -> None:
            episode_bridge.wrapped_physics_ctrl(original_physics, ctrl)

        self.grabber._physics_ctrl = bridged

        def run() -> None:
            result: Any = None
            error: BaseException | None = None
            aborted = False
            try:
                result = self._run_ready_and_demo(episode_bridge)
            except BridgeAborted:
                aborted = True
            except BaseException as exc:
                error = exc
            finally:
                self.grabber._physics_ctrl = original_physics
                if not aborted:
                    episode_bridge.finish(result=result, error=error)

        self.worker = Thread(target=run, name="g3-current-controller", daemon=True)
        self.worker.start()
        try:
            done = episode_bridge.wait_until_ready()
        finally:
            # The first request proves reset/planning has completed; on an
            # early planning failure ControllerDone proves it has exited.
            self.grabber.planner = original_planner
            self.plan_record = handshake.record()
        return done

    def _stop_worker(self) -> None:
        if self.bridge is not None:
            self.bridge.abort()
        if self.worker is not None:
            self.worker.join(timeout=5.0)
            if self.worker.is_alive():
                raise RuntimeError(
                    "controller worker did not stop within the join timeout"
                )
        if self.worker_original_physics is not None:
            self.grabber._physics_ctrl = self.worker_original_physics
        self.bridge = None
        self.worker = None
        self.worker_original_physics = None

    def _planning_values(self, action: np.ndarray) -> tuple[float, float]:
        plan_enabled = self.control_mode == "ik_residual"
        if not plan_enabled or self.object_kind != "banana":
            return 0.0, 0.0
        return (
            float(action[config.ACTUATOR_DIM])
            * config.GRASP_HEIGHT_RESIDUAL_MAX_M,
            float(action[config.ACTUATOR_DIM + 1])
            * config.GRASP_YAW_RESIDUAL_MAX_RAD,
        )

    def _failure_from_done(self, done: ControllerDone) -> str:
        if done.error is not None:
            return f"controller_exception:{type(done.error).__name__}: {done.error}"
        result = done.result
        if isinstance(result, Mapping):
            precondition = result.get("precondition")
            if isinstance(precondition, Mapping) and not bool(
                precondition.get("success", False)
            ):
                return str(
                    precondition.get(
                        "termination_reason", "grasp_lift_precondition_failed"
                    )
                )
        return ""

    def _reward(self, *, terminal: bool, target_switched: bool) -> float:
        current_cost = self._tracking_cost(self._reward_endpoint())
        progress = 0.0 if target_switched else self.previous_cost - current_cost
        self.previous_cost = current_cost
        residual_norm = (
            0.0
            if self.bridge is None
            else float(
                np.linalg.norm(self.bridge.residual)
                / (
                    config.RESIDUAL_MAX_RAD
                    * math.sqrt(config.ACTUATOR_DIM)
                )
            )
        )
        reward = 2.0 * progress - 0.01 * residual_norm
        if terminal:
            final_success = bool(
                self.action_target_ready and self.last_endpoint["success"]
            )
            reward += 10.0 if final_success else -min(current_cost, 100.0)
        return float(reward)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.episode_done:
            raise RuntimeError("step called after episode end; call reset()")
        raw_action = np.asarray(action, dtype=np.float64).reshape(
            config.ACTION_DIM
        )
        if not np.all(np.isfinite(raw_action)):
            raise ValueError("action must contain only finite values")
        clipped_action = np.clip(raw_action, -1.0, 1.0)
        if self.control_mode == "ik_only":
            clipped_action = np.zeros(config.ACTION_DIM, dtype=np.float64)

        if self.handshake_pending:
            height_residual, yaw_residual = self._planning_values(clipped_action)
            handshake = _PlanningHandshake(
                self.grabber.planner,
                height_residual_m=height_residual,
                yaw_residual_rad=yaw_residual,
                before_plan=self._capture_initial_state,
            )
            done = self._start_episode_worker(handshake)
            self.handshake_pending = False
            self._refresh_endpoints()
            self.previous_cost = self._tracking_cost(self._reward_endpoint())
            if done is not None:
                self.episode_done = True
                failure_reason = self._failure_from_done(done)
                info = self._info(
                    planning_handshake=True,
                    control_tick=False,
                    failure_reason=failure_reason,
                    controller_result=done.result,
                )
                return self._observation(), -5.0, True, False, info
            info = self._info(
                planning_handshake=True,
                control_tick=False,
            )
            return self._observation(), 0.0, False, False, info

        if self.bridge is None:
            raise RuntimeError("prepared episode has no control bridge")
        actuator_action = (
            clipped_action[: config.ACTUATOR_DIM]
            if self.control_mode != "ik_only"
            else np.zeros(config.ACTUATOR_DIM, dtype=np.float64)
        )
        target_ready_before = self.action_target_ready
        try:
            tick, done = self.bridge.advance(actuator_action)
        except BaseException as exc:
            self.episode_done = True
            self._refresh_endpoints()
            failure_reason = f"controller_exception:{type(exc).__name__}: {exc}"
            info = self._info(
                planning_handshake=False,
                control_tick=False,
                failure_reason=failure_reason,
            )
            return self._observation(), -100.0, True, False, info

        self.last_tick = tick
        self._refresh_endpoints()
        target_switched = not target_ready_before and self.action_target_ready
        terminal = done is not None
        failure_reason = "" if done is None else self._failure_from_done(done)
        if terminal:
            self.episode_done = True
        reward = self._reward(
            terminal=terminal,
            target_switched=target_switched,
        )
        info = self._info(
            planning_handshake=False,
            control_tick=True,
            failure_reason=failure_reason,
            controller_result=(None if done is None else done.result),
        )
        return self._observation(), reward, terminal, False, info

    def _observation(self) -> np.ndarray:
        object_one_hot = np.zeros(len(config.OBJECTS), dtype=np.float64)
        object_one_hot[config.OBJECTS.index(self.object_kind)] = 1.0
        task_one_hot = np.zeros(len(config.TASKS), dtype=np.float64)
        task_one_hot[config.TASKS.index(self.task)] = 1.0
        if self.handshake_pending or not self.initial_state_captured:
            dynamic = np.zeros(79, dtype=np.float64)
            observation = np.concatenate(
                (object_one_hot, task_one_hot, (1.0, 0.0), dynamic)
            )
            return observation.astype(np.float32)

        body_position, body_quaternion = self.grabber._object_pose()
        linear_velocity, angular_velocity = self.grabber._object_velocity()
        current_com = self.grabber._object_com_position()
        target_position = (
            self.action_target_pos
            if self.action_target_ready
            else self.overall_target_pos
        )
        target_rotation = (
            self.action_target_rotation
            if self.action_target_ready
            else self.overall_target_rotation
        )
        position_error_xyz = current_com - target_position
        euler_error = grab._fixed_xyz_euler_error_deg(
            target_rotation, body_quaternion
        )
        qpos = np.asarray(
            self.grabber.data.qpos[self._actuator_qpos_adrs], dtype=np.float64
        )
        qvel = np.asarray(
            self.grabber.data.qvel[self._actuator_qvel_adrs], dtype=np.float64
        )
        residual = (
            np.zeros(config.ACTUATOR_DIM, dtype=np.float64)
            if self.bridge is None
            else self.bridge.residual
        )
        progress = (
            0.0
            if self.bridge is None
            else math.tanh(self.bridge.control_step / 1000.0)
        )
        observation = np.concatenate(
            (
                object_one_hot,
                task_one_hot,
                (0.0, float(self.action_target_ready)),
                body_position,
                body_quaternion,
                linear_velocity,
                angular_velocity,
                target_position,
                grab.rot_to_quat(target_rotation),
                position_error_xyz,
                euler_error / 180.0,
                qpos,
                qvel,
                np.asarray(self.grabber.data.ctrl, dtype=np.float64),
                residual,
                np.asarray(self.grabber.last_contact_forces, dtype=np.float64)
                / 10.0,
                (progress,),
            )
        )
        if observation.shape != (OBSERVATION_DIM,):
            raise RuntimeError(
                f"observation contract drifted: {observation.shape}"
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("non-finite observation produced by simulation")
        return observation.astype(np.float32)

    def _info(
        self,
        *,
        planning_handshake: bool,
        control_tick: bool,
        failure_reason: str = "",
        controller_result: Any = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "object_kind": self.object_kind,
            "task": self.task,
            "seed": self.episode_seed,
            "control_mode": self.control_mode,
            "planning_handshake": bool(planning_handshake),
            "planning_handshake_count": int(not self.handshake_pending),
            "control_tick": bool(control_tick),
            "control_step": (
                0 if self.bridge is None else int(self.bridge.control_step)
            ),
            "initial_state_hash": self.initial_state_hash,
            "integration_state_hash": self.integration_state_hash,
            "controller_state_hash": self.controller_state_hash,
            "parent_contract_sha256": self.parent_contract[
                "contract_sha256"
            ],
            "plan_residual": _jsonable(self.plan_record),
            "plan_residual_ablation": self.control_mode
            == "ik_residual_no_plan",
            "action_target_ready": self.action_target_ready,
            "target_reference": {
                "success_label": "post_lift_relative_action_target",
                "overall_diagnostic": (
                    "paired_initial_absolute_lift_plus_action_target"
                ),
                "grasp_lift_diagnostic": (
                    "current_figure15_compatible_precondition"
                ),
            },
            "targets": {
                "post_lift_relative_action": (
                    {
                        "position_m": self.action_target_pos.tolist(),
                        "quaternion_wxyz": grab.rot_to_quat(
                            self.action_target_rotation
                        ).tolist(),
                    }
                    if self.action_target_ready
                    else None
                ),
                "paired_initial_absolute_lift_plus_action": (
                    {
                        "position_m": self.overall_target_pos.tolist(),
                        "quaternion_wxyz": grab.rot_to_quat(
                            self.overall_target_rotation
                        ).tolist(),
                    }
                    if self.initial_state_captured
                    else None
                ),
            },
            "endpoint": _jsonable(self.last_endpoint),
            "overall_endpoint": _jsonable(self.last_overall_endpoint),
            "precondition": _jsonable(self.precondition_record),
            "failure_reason": str(failure_reason),
            # Final labels intentionally use only endpoint position/orientation.
            "is_success": bool(
                self.episode_done
                and self.action_target_ready
                and self.last_endpoint.get("success", False)
            ),
        }
        if controller_result is not None:
            info["controller_result"] = _jsonable(controller_result)
        if self.episode_done and self.record_trajectory:
            info["trajectory"] = list(self.trajectory)
        return info

    def close(self) -> None:
        self._stop_worker()
        self.grabber.close()
        super().close()


def make_env(
    *,
    control_mode: str = "ik_residual",
    record_trajectory: bool = False,
    fixed_object: str | None = None,
    fixed_task: str | None = None,
) -> G3IKResidualEnv:
    return G3IKResidualEnv(
        control_mode=control_mode,
        record_trajectory=record_trajectory,
        fixed_object=fixed_object,
        fixed_task=fixed_task,
    )


__all__ = [
    "CONTROL_MODES",
    "G3IKResidualEnv",
    "OBSERVATION_DIM",
    "make_env",
]
