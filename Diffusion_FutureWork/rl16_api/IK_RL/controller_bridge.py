"""One-control-tick thread bridge around the unchanged blocking G6 controller."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import math
import threading
import time
from typing import Any, Sequence

try:
    from . import config
    from .runtime import prepare_runtime, validate_parent_contract
except ImportError:  # Direct execution/import from the IK_RL directory.
    import config  # type: ignore
    from runtime import prepare_runtime, validate_parent_contract  # type: ignore


current = prepare_runtime()

import numpy as np


OBSERVATION_DIM = 154


class BridgeClosed(RuntimeError):
    """Internal signal used to stop a worker at its next control boundary."""


class ResidualControlLaw:
    """Blend and rate-limit the 16-D residual added after current IK control."""

    def __init__(
        self,
        ctrl_low: Sequence[float],
        ctrl_high: Sequence[float],
    ) -> None:
        self.ctrl_low = np.asarray(ctrl_low, dtype=np.float64).reshape(
            config.ACTION_DIM
        )
        self.ctrl_high = np.asarray(ctrl_high, dtype=np.float64).reshape(
            config.ACTION_DIM
        )
        self.maximum = np.full(
            config.ACTION_DIM,
            config.RESIDUAL_MAX_RAD,
            dtype=np.float64,
        )
        self.maximum[list(config.LINEAR_ACTUATOR_INDICES)] = (
            config.RESIDUAL_MAX_LINEAR_M
        )
        self.rate_limit = np.full(
            config.ACTION_DIM,
            config.RESIDUAL_RATE_LIMIT_RAD_PER_STEP,
            dtype=np.float64,
        )
        self.rate_limit[list(config.LINEAR_ACTUATOR_INDICES)] = (
            config.RESIDUAL_RATE_LIMIT_LINEAR_M_PER_STEP
        )
        self.residual = np.zeros(config.ACTION_DIM, dtype=np.float64)

    def reset(self) -> None:
        self.residual[:] = 0.0

    def apply(
        self,
        nominal_ctrl: Sequence[float],
        action: Sequence[float],
        *,
        enabled: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return applied control and residual without mutating IK bookkeeping."""
        nominal = np.asarray(nominal_ctrl, dtype=np.float64).reshape(
            config.ACTION_DIM
        )
        if not enabled:
            self.residual[:] = 0.0
            return nominal.copy(), self.residual.copy()

        clipped_action = np.clip(
            np.asarray(action, dtype=np.float64).reshape(config.ACTION_DIM),
            -1.0,
            1.0,
        )
        if not np.all(np.isfinite(clipped_action)):
            raise ValueError("residual action must contain only finite values")
        requested = clipped_action * self.maximum
        blend = float(np.clip(config.RESIDUAL_TARGET_BLEND, 0.0, 1.0))
        requested = (1.0 - blend) * self.residual + blend * requested
        delta = np.clip(
            requested - self.residual,
            -self.rate_limit,
            self.rate_limit,
        )
        self.residual = np.clip(
            self.residual + delta,
            -self.maximum,
            self.maximum,
        )
        applied = np.clip(
            nominal + self.residual,
            self.ctrl_low,
            self.ctrl_high,
        )
        return applied, self.residual.copy()


def _jsonable_result(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    return value


class ResidualControlBridge:
    """Expose every current-controller call to ``_physics_ctrl`` as one step.

    ``prepare_runtime`` first verifies that ``current``, its model XML, fruit
    capsule manifest/XML and planner all come from the parent GRAB_G6 directory.
    The worker then executes the unchanged blocking ready-trial and post-lift
    action.  The patched physics function pauses immediately before exactly one
    50 Hz control tick.  ``advance`` supplies one action, lets precisely that
    tick execute, and waits for the following boundary (or workflow completion).
    """

    def __init__(
        self,
        object_kind: str,
        task: str,
        seed: int,
        *,
        render: bool = False,
        wait_timeout_s: float = 120.0,
    ) -> None:
        if object_kind not in config.OBJECTS:
            raise ValueError(f"unsupported training object: {object_kind!r}")
        if task not in config.TASKS:
            raise ValueError(f"unsupported task: {task!r}")
        self.object_kind = object_kind
        self.task = task
        self.seed = int(seed)
        self.wait_timeout_s = float(wait_timeout_s)

        self.parent_contract = validate_parent_contract(current)
        self.grabber = current.IKGrabber(render=render)
        self._original_physics_ctrl = self.grabber._physics_ctrl
        self.grabber._physics_ctrl = self._physics_boundary
        self.control_law = ResidualControlLaw(
            self.grabber._ctrl_low,
            self.grabber._ctrl_high,
        )

        self._condition = threading.Condition()
        self._worker: threading.Thread | None = None
        self._closed = False
        self._waiting = False
        self._request_id = 0
        self._pending_action: np.ndarray | None = None
        self._worker_done = False
        self._worker_exception: BaseException | None = None

        self.control_step = 0
        self.action_control_step = 0
        self.action_started = False
        self.ready_result: Any = None
        self.demo_result: Any = None
        self.workflow_reason = ""

        self.nominal_ctrl = np.zeros(config.ACTION_DIM, dtype=np.float64)
        self.applied_ctrl = np.zeros(config.ACTION_DIM, dtype=np.float64)
        self.last_action = np.zeros(config.ACTION_DIM, dtype=np.float64)
        self.last_residual = np.zeros(config.ACTION_DIM, dtype=np.float64)
        self.last_sample: dict[str, Any] | None = None

        self.initial_com: np.ndarray | None = None
        self.initial_rotation: np.ndarray | None = None
        self.initial_quaternion: np.ndarray | None = None
        self.action_start_com: np.ndarray | None = None
        self.action_start_rotation: np.ndarray | None = None
        self.action_start_quaternion: np.ndarray | None = None
        self.action_target_com: np.ndarray | None = None
        self.action_target_rotation: np.ndarray | None = None
        self.overall_target_com: np.ndarray | None = None
        self.overall_target_rotation: np.ndarray | None = None
        self.initial_state_hash = ""
        self.action_ready_state_hash = ""
        self.endpoint: dict[str, Any] | None = None

        self._actuator_qpos_adrs = np.asarray(
            self.grabber.model.jnt_qposadr[
                self.grabber._actuator_joint_ids
            ],
            dtype=np.int32,
        )
        self._actuator_qvel_adrs = np.asarray(
            self.grabber.model.jnt_dofadr[
                self.grabber._actuator_joint_ids
            ],
            dtype=np.int32,
        )

    def start(self) -> tuple[np.ndarray, dict[str, Any]]:
        if self._worker is not None:
            raise RuntimeError("bridge has already been started")
        self._worker = threading.Thread(
            target=self._worker_main,
            name=f"g6-{self.object_kind}-{self.task}-{self.seed}",
            daemon=True,
        )
        self._worker.start()
        self._wait_for_boundary(previous_request_id=0)
        self._raise_worker_exception()
        return self.observation(), self.info()

    def advance(
        self,
        action: Sequence[float],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        value = np.asarray(action, dtype=np.float64).reshape(config.ACTION_DIM)
        if not np.all(np.isfinite(value)):
            raise ValueError("residual action must contain only finite values")
        with self._condition:
            if self._worker_done:
                raise RuntimeError("episode is already complete; call reset()")
            if not self._waiting:
                raise RuntimeError("worker is not waiting at a control boundary")
            request_id = self._request_id
            self._pending_action = value.copy()
            self._condition.notify_all()
        self._wait_for_boundary(previous_request_id=request_id)
        self._raise_worker_exception()

        terminated = bool(self._worker_done)
        observation = self.observation()
        info = self.info()
        reward = self._reward(terminated)
        return observation, reward, terminated, False, info

    def close(self) -> None:
        worker = self._worker
        with self._condition:
            self._closed = True
            self._pending_action = np.zeros(
                config.ACTION_DIM, dtype=np.float64
            )
            self._condition.notify_all()
        if (
            worker is not None
            and worker.is_alive()
            and worker is not threading.current_thread()
        ):
            worker.join(timeout=self.wait_timeout_s)
            if worker.is_alive():
                raise RuntimeError("G6 controller worker did not stop cleanly")
        self.grabber._physics_ctrl = self._original_physics_ctrl

    def _wait_for_boundary(self, *, previous_request_id: int) -> None:
        deadline = time.monotonic() + self.wait_timeout_s
        with self._condition:
            while not (
                self._worker_done
                or (
                    self._waiting
                    and self._request_id > previous_request_id
                )
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "timed out waiting for a G6 50 Hz control boundary"
                    )
                self._condition.wait(timeout=remaining)

    def _physics_boundary(self, nominal_ctrl: Sequence[float]) -> None:
        nominal = np.asarray(nominal_ctrl, dtype=np.float64).reshape(
            config.ACTION_DIM
        )
        with self._condition:
            if self._closed:
                raise BridgeClosed("bridge closed")
            self.nominal_ctrl = nominal.copy()
            self._request_id += 1
            self._waiting = True
            if self._request_id == 1:
                self._capture_initial_reference()
            self._condition.notify_all()
            while self._pending_action is None and not self._closed:
                self._condition.wait()
            if self._closed:
                raise BridgeClosed("bridge closed")
            action = self._pending_action
            self._pending_action = None
            self._waiting = False
            # No stage/phase gate: the learned correction covers grasp, lift,
            # and the post-lift action.  A zero action remains exactly nominal.
            residual_enabled = True
            manipulation_started = bool(self.action_started)

        assert action is not None
        applied, residual = self.control_law.apply(
            nominal,
            action,
            enabled=residual_enabled,
        )
        # This is the only physics call for this Gym step.  In particular,
        # executed_ctrl and IKTeacher are intentionally left untouched.
        self._original_physics_ctrl(applied)

        self.control_step += 1
        if manipulation_started:
            self.action_control_step += 1
        self.last_action = np.asarray(action, dtype=np.float64).copy()
        self.applied_ctrl = applied.copy()
        self.last_residual = residual.copy()
        self.last_sample = self.trajectory_sample()

    def _capture_initial_reference(self) -> None:
        self.initial_com = self.grabber._object_com_position()
        _position, quaternion = self.grabber._object_pose()
        self.initial_quaternion = quaternion.copy()
        self.initial_rotation = current.quat_to_rot(quaternion)
        lift = np.array(
            (0.0, 0.0, float(self.grabber.commanded_lift_height_m)),
            dtype=np.float64,
        )
        if self.task == config.TASK_TRANSLATE:
            self.overall_target_com = (
                self.initial_com
                + lift
                + np.asarray(config.TRANSLATION_VECTOR_M, dtype=np.float64)
            )
            self.overall_target_rotation = self.initial_rotation.copy()
        else:
            self.overall_target_com = self.initial_com + lift
            self.overall_target_rotation = (
                self.initial_rotation
                @ current._axis_rotation("z", config.ROTATION_ANGLE_RAD)
            )
        self.initial_state_hash = self._state_hash()

    def _capture_action_reference(self) -> None:
        self.action_start_com = self.grabber._object_com_position()
        _position, quaternion = self.grabber._object_pose()
        self.action_start_quaternion = quaternion.copy()
        self.action_start_rotation = current.quat_to_rot(quaternion)
        if self.task == config.TASK_TRANSLATE:
            self.action_target_com = (
                self.action_start_com
                + np.asarray(config.TRANSLATION_VECTOR_M, dtype=np.float64)
            )
            self.action_target_rotation = self.action_start_rotation.copy()
        else:
            self.action_target_com = self.action_start_com.copy()
            self.action_target_rotation = (
                self.action_start_rotation
                @ current._axis_rotation("z", config.ROTATION_ANGLE_RAD)
            )
        self.action_ready_state_hash = self._state_hash()

    def _worker_main(self) -> None:
        try:
            if self.task == config.TASK_TRANSLATE:
                self.ready_result = self.grabber.run_translation_ready_trial(
                    self.object_kind,
                    seed=self.seed,
                    visualize=False,
                )
            else:
                self.ready_result = self.grabber.run_rotation_ready_trial(
                    self.object_kind,
                    axes="z",
                    seed=self.seed,
                    visualize=False,
                )

            if not bool(getattr(self.ready_result, "success", False)):
                self.workflow_reason = "ready_trial_failed"
                return

            self._capture_action_reference()
            self.action_started = True
            if self.task == config.TASK_TRANSLATE:
                self.demo_result = self.grabber.run_translation_demo()
            else:
                self.demo_result = self.grabber.run_rotation_demo("z")
            self.workflow_reason = str(
                getattr(self.demo_result, "termination_reason", "completed")
            )
        except BridgeClosed:
            self.workflow_reason = "bridge_closed"
        except BaseException as exc:
            self._worker_exception = exc
            self.workflow_reason = f"worker_error:{type(exc).__name__}"
        finally:
            if not self._closed:
                try:
                    self.endpoint = self._endpoint_metrics()
                except BaseException as exc:
                    if self._worker_exception is None:
                        self._worker_exception = exc
            with self._condition:
                self._worker_done = True
                self._waiting = False
                self._condition.notify_all()

    def _raise_worker_exception(self) -> None:
        if self._worker_exception is not None:
            raise RuntimeError("G6 controller worker failed") from (
                self._worker_exception
            )

    def _state_hash(self) -> str:
        digest = hashlib.sha256()
        integration_spec = current.mujoco.mjtState.mjSTATE_INTEGRATION
        integration_state = np.empty(
            current.mujoco.mj_stateSize(
                self.grabber.model,
                integration_spec,
            ),
            dtype=np.float64,
        )
        current.mujoco.mj_getState(
            self.grabber.model,
            self.grabber.data,
            integration_state,
            integration_spec,
        )
        for value in (
            integration_state,
            self.grabber.executed_ctrl,
            self.grabber.last_ctrl_delta,
            self.grabber.closure_fraction,
            self.grabber.contact_locked.astype(np.float64),
            self.grabber.last_contact_forces,
            self.grabber.contact_center_trim_world,
            self.grabber.contact_pair_compression_m,
            self.grabber.initial_body_pos,
            self.grabber.initial_body_quat,
            self.grabber.goal_body_pos,
            np.asarray(
                (self.grabber.commanded_lift_height_m,),
                dtype=np.float64,
            ),
            *(
                self.grabber.last_tip_targets_world[finger]
                for finger in current.FINGERS
            ),
        ):
            array = np.asarray(value, dtype=np.float64)
            digest.update(array.tobytes(order="C"))
        controller_scalars = (
            str(self.grabber.phase),
            str(int(self.grabber.phase_step)),
            str(int(self.grabber.trial_step)),
            repr(getattr(self.grabber, "selected_plan", None)),
        )
        for value in controller_scalars:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        return digest.hexdigest()

    def _active_target(self) -> tuple[np.ndarray, np.ndarray]:
        if self.action_started and self.action_target_com is not None:
            assert self.action_target_rotation is not None
            return self.action_target_com, self.action_target_rotation
        if self.initial_com is None or self.initial_rotation is None:
            return np.zeros(3), np.eye(3)
        lift_target = self.initial_com + np.array(
            (0.0, 0.0, float(self.grabber.commanded_lift_height_m))
        )
        return lift_target, self.initial_rotation

    def _pose_metrics(
        self,
        target_com: np.ndarray,
        target_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        com = self.grabber._object_com_position()
        _position, quaternion = self.grabber._object_pose()
        position_error = com - target_com
        euler_error = current._fixed_xyz_euler_error_deg(
            target_rotation,
            quaternion,
        )
        return (
            position_error,
            euler_error,
            float(np.linalg.norm(position_error)),
            float(np.linalg.norm(euler_error)),
        )

    def _endpoint_metrics(self) -> dict[str, Any]:
        if self.action_target_com is None or self.action_target_rotation is None:
            endpoint: dict[str, Any] = {
                "position_error_m": float("nan"),
                "euler_l2_error_deg": float("nan"),
                "position_ok": False,
                "orientation_ok": False,
                "success": False,
                "target_reference": "post_lift_relative_action_target",
            }
        else:
            (
                position_error_xyz,
                euler_error_xyz,
                position_l2,
                euler_l2,
            ) = self._pose_metrics(
                self.action_target_com,
                self.action_target_rotation,
            )
            tolerance = config.rotation_tolerance_deg(self.object_kind)
            endpoint = {
                "position_error_xyz_m": position_error_xyz.tolist(),
                "position_error_m": position_l2,
                "euler_error_xyz_deg": euler_error_xyz.tolist(),
                "euler_l2_error_deg": euler_l2,
                "position_ok": bool(
                    math.isfinite(position_l2)
                    and position_l2 <= config.POSITION_TOLERANCE_M
                ),
                "orientation_ok": bool(
                    math.isfinite(euler_l2) and euler_l2 <= tolerance
                ),
                "success": config.endpoint_success(
                    self.object_kind, position_l2, euler_l2
                ),
                "position_tolerance_m": config.POSITION_TOLERANCE_M,
                "orientation_tolerance_deg": tolerance,
                "target_reference": "post_lift_relative_action_target",
                "target_com_m": self.action_target_com.tolist(),
                "target_quaternion_wxyz": current.rot_to_quat(
                    self.action_target_rotation
                ).tolist(),
            }
        if (
            self.overall_target_com is not None
            and self.overall_target_rotation is not None
        ):
            (
                overall_xyz,
                overall_euler_xyz,
                overall_l2,
                overall_euler_l2,
            ) = self._pose_metrics(
                self.overall_target_com,
                self.overall_target_rotation,
            )
            endpoint["overall_grasp_action"] = {
                "position_error_xyz_m": overall_xyz.tolist(),
                "position_error_m": overall_l2,
                "euler_error_xyz_deg": overall_euler_xyz.tolist(),
                "euler_l2_error_deg": overall_euler_l2,
                "target_reference": "paired_initial_absolute_target",
                "target_com_m": self.overall_target_com.tolist(),
                "target_quaternion_wxyz": current.rot_to_quat(
                    self.overall_target_rotation
                ).tolist(),
            }
        if (
            self.action_start_com is not None
            and self.initial_com is not None
            and self.initial_rotation is not None
        ):
            lift_target = self.initial_com + np.array(
                (0.0, 0.0, float(self.grabber.commanded_lift_height_m))
            )
            (
                lift_xyz,
                lift_euler_xyz,
                lift_l2,
                lift_euler_l2,
            ) = self._pose_metrics_at_pose(
                self.action_start_com,
                self.action_start_quaternion,
                lift_target,
                self.initial_rotation,
            )
            endpoint["grasp_lift"] = {
                "position_error_xyz_m": lift_xyz.tolist(),
                "position_error_m": lift_l2,
                "euler_error_xyz_deg": lift_euler_xyz.tolist(),
                "euler_l2_error_deg": lift_euler_l2,
                "target_reference": "paired_initial_lift_target",
            }
        return endpoint

    @staticmethod
    def _pose_metrics_at_pose(
        com: np.ndarray,
        quaternion: np.ndarray | None,
        target_com: np.ndarray,
        target_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        if quaternion is None:
            euler = np.full(3, np.nan)
        else:
            euler = current._fixed_xyz_euler_error_deg(
                target_rotation, quaternion
            )
        xyz = np.asarray(com) - np.asarray(target_com)
        return (
            xyz,
            euler,
            float(np.linalg.norm(xyz)),
            float(np.linalg.norm(euler)),
        )

    def observation(self) -> np.ndarray:
        com = self.grabber._object_com_position()
        _body_position, quaternion = self.grabber._object_pose()
        linear_velocity, angular_velocity = self.grabber._object_velocity()
        target_com, target_rotation = self._active_target()
        position_error, euler_error, position_l2, euler_l2 = (
            self._pose_metrics(target_com, target_rotation)
        )

        object_one_hot = np.zeros(len(config.OBJECTS), dtype=np.float64)
        object_one_hot[config.OBJECTS.index(self.object_kind)] = 1.0
        task_one_hot = np.zeros(len(config.TASKS), dtype=np.float64)
        task_one_hot[config.TASKS.index(self.task)] = 1.0

        actuator_qpos = np.asarray(
            self.grabber.data.qpos[self._actuator_qpos_adrs],
            dtype=np.float64,
        )
        actuator_qvel = np.asarray(
            self.grabber.data.qvel[self._actuator_qvel_adrs],
            dtype=np.float64,
        )
        forces = np.asarray(
            self.grabber._contact_forces(), dtype=np.float64
        )
        contact_flags = (
            forces >= current.CONTACT_FORCE_THRESHOLD_N
        ).astype(np.float64)
        tip_relative = np.concatenate(
            [
                np.asarray(
                    self.grabber.last_tip_targets_world[finger],
                    dtype=np.float64,
                )
                - com
                for finger in current.FINGERS
            ]
        )
        expected_action_steps = (
            current.DEMO_TRANSLATION_STEPS
            if self.task == config.TASK_TRANSLATE
            else current.DEMO_MAX_STEPS_PER_AXIS
        )
        progress = min(
            1.0,
            self.action_control_step / max(1, expected_action_steps),
        )
        scalars = np.asarray(
            (
                float(self.action_started),
                progress,
                position_l2 / config.POSITION_TOLERANCE_M,
                euler_l2 / config.rotation_tolerance_deg(self.object_kind),
            ),
            dtype=np.float64,
        )
        observation = np.concatenate(
            (
                object_one_hot,
                task_one_hot,
                com,
                quaternion,
                linear_velocity,
                angular_velocity,
                target_com,
                current.rot_to_quat(target_rotation),
                position_error,
                euler_error / 180.0,
                actuator_qpos,
                actuator_qvel,
                self.applied_ctrl,
                self.nominal_ctrl,
                self.last_residual,
                self.last_action,
                forces,
                contact_flags,
                tip_relative,
                scalars,
            )
        )
        if observation.shape != (OBSERVATION_DIM,):
            raise RuntimeError(
                f"observation shape drifted: {observation.shape}, "
                f"expected {(OBSERVATION_DIM,)}"
            )
        return np.nan_to_num(
            observation,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        ).astype(np.float32)

    def trajectory_sample(self) -> dict[str, Any]:
        target_com, target_rotation = self._active_target()
        position_xyz, euler_xyz, position_l2, euler_l2 = (
            self._pose_metrics(target_com, target_rotation)
        )
        com = self.grabber._object_com_position()
        body_position, quaternion = self.grabber._object_pose()
        linear_velocity, angular_velocity = self.grabber._object_velocity()
        forces = self.grabber._contact_forces()
        initial_body_position = np.asarray(
            self.grabber.initial_body_pos, dtype=np.float64
        )
        phase = str(self.grabber.phase)
        if phase == current.Phase.LIFT:
            lift_fraction = current._minimum_jerk_fraction(
                int(self.grabber.phase_step),
                int(current.LIFT_STEPS),
            )
            reference_position = initial_body_position.copy()
            reference_position[2] += (
                lift_fraction * float(self.grabber.commanded_lift_height_m)
            )
        elif phase == current.Phase.HOLD:
            lift_fraction = 1.0
            reference_position = np.asarray(
                self.grabber.goal_body_pos, dtype=np.float64
            ).copy()
        else:
            lift_fraction = 0.0
            reference_position = initial_body_position.copy()
        precondition_translation_error_mm = 1000.0 * (
            body_position - reference_position
        )
        if self.initial_rotation is None:
            initial_euler_error = np.zeros(3, dtype=np.float64)
        else:
            initial_euler_error = current._fixed_xyz_euler_error_deg(
                self.initial_rotation, quaternion
            )
        initial_com = (
            self.initial_com
            if self.initial_com is not None
            else com
        )
        return {
            "control_step": int(self.control_step),
            "time_s": float(self.control_step / config.CONTROL_HZ),
            "action_control_step": int(self.action_control_step),
            "residual_active": True,
            "action_stage_started": bool(self.action_started),
            "phase": phase,
            "phase_step": int(self.grabber.phase_step),
            "trial_step": int(self.grabber.trial_step),
            "position_error_xyz_m": position_xyz.tolist(),
            "position_error_m": position_l2,
            "euler_error_xyz_deg": euler_xyz.tolist(),
            "euler_l2_error_deg": euler_l2,
            "object_com_m": com.tolist(),
            "object_position_m": body_position.tolist(),
            "object_quaternion_wxyz": quaternion.tolist(),
            "object_linear_velocity_m_s": linear_velocity.tolist(),
            "object_angular_velocity_rad_s": angular_velocity.tolist(),
            "linear_speed_mm_s": float(
                1000.0 * np.linalg.norm(linear_velocity)
            ),
            "angular_speed_deg_s": float(
                math.degrees(np.linalg.norm(angular_velocity))
            ),
            "target_com_m": target_com.tolist(),
            "target_quaternion_wxyz": current.rot_to_quat(
                target_rotation
            ).tolist(),
            "nominal_ctrl": self.nominal_ctrl.tolist(),
            "ctrl": self.applied_ctrl.tolist(),
            "residual": self.last_residual.tolist(),
            "action": self.last_action.tolist(),
            "contact_forces_n": np.asarray(forces).tolist(),
            "contact_count": int(
                np.count_nonzero(
                    np.asarray(forces)
                    >= current.CONTACT_FORCE_THRESHOLD_N
                )
            ),
            "initial_com_m": np.asarray(initial_com).tolist(),
            "initial_position_m": initial_body_position.tolist(),
            "initial_quaternion_wxyz": (
                self.initial_quaternion.tolist()
                if self.initial_quaternion is not None
                else [1.0, 0.0, 0.0, 0.0]
            ),
            "reference_position_m": reference_position.tolist(),
            "reference_lift_fraction": float(lift_fraction),
            "signed_translation_tracking_error_x_mm": float(
                precondition_translation_error_mm[0]
            ),
            "signed_translation_tracking_error_y_mm": float(
                precondition_translation_error_mm[1]
            ),
            "signed_translation_tracking_error_z_mm": float(
                precondition_translation_error_mm[2]
            ),
            "translation_tracking_error_l2_mm": float(
                np.linalg.norm(precondition_translation_error_mm)
            ),
            "signed_euler_error_rx_deg": float(initial_euler_error[0]),
            "signed_euler_error_ry_deg": float(initial_euler_error[1]),
            "signed_euler_error_rz_deg": float(initial_euler_error[2]),
            "initial_euler_error_l2_deg": float(
                np.linalg.norm(initial_euler_error)
            ),
            "measured_lift_mm": float(
                1000.0
                * (body_position[2] - initial_body_position[2])
            ),
            "horizontal_drift_mm": float(
                1000.0
                * np.linalg.norm(
                    body_position[:2] - initial_body_position[:2]
                )
            ),
        }

    def info(self) -> dict[str, Any]:
        endpoint = self.endpoint
        if endpoint is None and self._worker_done:
            endpoint = self._endpoint_metrics()
        return {
            "object": self.object_kind,
            "task": self.task,
            "seed": self.seed,
            "control_step": int(self.control_step),
            "action_control_step": int(self.action_control_step),
            "residual_active": True,
            "action_stage_started": bool(self.action_started),
            "residual_enabled": True,
            "manipulation_started": bool(self.action_started),
            "initial_state_hash": self.initial_state_hash,
            "action_ready_state_hash": self.action_ready_state_hash,
            "parent_contract_sha256": self.parent_contract[
                "contract_sha256"
            ],
            "target_reference": {
                "success": "post_lift_relative_action_target",
                "overall": "paired_initial_absolute_target",
                "grasp_lift": "paired_initial_lift_target",
            },
            "trajectory_sample": self.last_sample,
            "endpoint": endpoint,
            "is_success": bool(
                endpoint is not None and endpoint.get("success", False)
            ),
            "pending_nominal_ctrl": self.nominal_ctrl.copy(),
            "last_applied_ctrl": self.applied_ctrl.copy(),
            "workflow_reason": self.workflow_reason,
            "ready_result": (
                _jsonable_result(self.ready_result)
                if self._worker_done
                else None
            ),
            "demo_result": (
                _jsonable_result(self.demo_result)
                if self._worker_done
                else None
            ),
        }

    def _reward(self, terminated: bool) -> float:
        target_com, target_rotation = self._active_target()
        _xyz, _euler, position_l2, euler_l2 = self._pose_metrics(
            target_com, target_rotation
        )
        reward = -min(
            4.0, position_l2 / config.POSITION_REWARD_SCALE_M
        )
        reward -= min(
            4.0, euler_l2 / config.ORIENTATION_REWARD_SCALE_DEG
        )
        normalized_residual = self.last_residual / np.maximum(
            self.control_law.maximum, 1e-12
        )
        reward -= config.RESIDUAL_PENALTY_WEIGHT * float(
            np.mean(np.square(normalized_residual))
        )
        if terminated:
            endpoint = self.endpoint or self._endpoint_metrics()
            if bool(endpoint.get("success", False)):
                reward += config.ENDPOINT_SUCCESS_BONUS
            else:
                reward -= config.ENDPOINT_FAILURE_PENALTY
        return float(reward)
