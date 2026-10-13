#!/usr/bin/env python3
"""Interactive IK-only controller for the local Gripper4 translation model.

Start with any Python environment that provides NumPy and MuJoCo:

    python3 T4_IKcontrol.py

Then enter cube-center targets as ``z r theta`` (metres, metres, degrees):

    0.32 0.03 90

On macOS the script uses the ``mjpython`` launcher belonging to the interpreter
chosen by the caller when a viewer is requested.  ``--no-render`` keeps
everything in the terminal and is useful for tests or remote runs.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from T4_kinematics import (
    ACTUATOR_NAMES,
    CUBE_BODY_NAME,
    CUBE_GEOM_NAME,
    CUBE_JOINT_NAME,
    DEFAULT_XML,
    FINGERS,
    IKTeacher,
    IKUnreachableError,
    PIN_SITE_PAIRS,
    TABLE_GEOM_NAME,
    TIP_GEOM_NAMES,
    GripperIK,
    cylindrical_to_xyz,
    format_vector,
    lerp,
    minimum_jerk,
    normalize_quat,
    quat_to_rot,
    xyz_to_cylindrical,
)

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:  # pragma: no cover - user-facing environment guard.
    raise SystemExit(
        "无法导入 MuJoCo。请在你选择的 Python 环境中安装 numpy 和 mujoco，"
        "再运行本脚本。\n"
        f"原始错误：{exc}"
    ) from exc


MODULE_DIR = Path(__file__).resolve().parent
MJPYTHON_RESTART_FLAG = "T4_MJPYTHON_RESTARTED"

# Safe first-grasp trajectory.  The initial q=0 hard tips are well above the
# object.  Moving every actuator directly toward the low pregrasp solution
# makes the intermediate linkage sweep inward through the cube top.  These
# limits instead make the commanded fingertip cross follow Cartesian stages.
SAFE_APPROACH_CLEARANCE_M = 0.005
SAFE_APPROACH_TARGET_STEP_M = 0.0009
SAFE_SHAPE_MIN_STEPS = 60
SAFE_DESCENT_MIN_STEPS = 100
SAFE_CLOSE_MIN_STEPS = 100
GRASP_SETTLE_STEPS = 20
GRASP_CONFIRMATION_STEPS = 10
GRASP_TRACKING_TOLERANCE_M = 0.005
GRASP_HEIGHT_TOLERANCE_M = 0.004
GRASP_MIN_NORMAL_FORCE_N = 0.1
GRASP_NORMAL_ALIGNMENT_MIN = math.cos(math.radians(30.0))
GRASP_SIDE_PLANE_TOLERANCE_M = 0.003
GRASP_SIDE_INTERIOR_LIMIT_M = 0.020


def restart_with_mjpython_if_needed(argv: Sequence[str]) -> None:
    """Use the caller-selected Python environment's macOS viewer launcher."""
    if platform.system() != "Darwin" or "--no-render" in argv:
        return
    if "-h" in argv or "--help" in argv:
        return
    if os.environ.get(MJPYTHON_RESTART_FLAG) == "1":
        return
    if Path(sys.executable).name == "mjpython":
        return

    selected_python = Path(sys.executable)
    selected_mjpython = selected_python.parent / "mjpython"
    if not selected_mjpython.is_file():
        raise SystemExit(
            "macOS MuJoCo viewer 需要当前 Python 环境提供 mjpython：\n"
            f"  {selected_mjpython}\n"
            "请用包含 mujoco 的解释器启动，或使用 --no-render 关闭窗口。"
        )

    print("正在通过当前 Python 环境的 mjpython 启动 MuJoCo 窗口……", flush=True)
    os.environ[MJPYTHON_RESTART_FLAG] = "1"
    script = Path(__file__).resolve()
    # Interpret the launcher with the selected Python to avoid relying on an
    # absolute shebang embedded in a copied environment.
    os.execv(
        str(selected_python),
        [str(selected_python), str(selected_mjpython), str(script), *argv],
    )


def require_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name!r}")
    return object_id


@dataclass(frozen=True)
class MotionStage:
    start_center: np.ndarray
    end_center: np.ndarray
    start_distance: float
    end_distance: float
    steps: int
    use_feedback: bool
    label: str


@dataclass(frozen=True)
class RunResult:
    target_xyz: np.ndarray
    final_xyz: np.ndarray
    position_error: np.ndarray
    success: bool
    steps: int
    max_analytic_tip_error: float
    max_analytic_loop_error: float
    max_sim_tip_tracking_error: float
    max_sim_pin_error: float
    elapsed_s: float


class T4IKController:
    """Analytic-IK position controller around a local MuJoCo simulation."""

    def __init__(
        self,
        xml_path: str | os.PathLike[str] = DEFAULT_XML,
        *,
        render: bool = True,
        control_hz: float = 50.0,
        contact_distance: Optional[float] = None,
        contact_compression: float = 0.0015,
        pregrasp_extra: float = 0.015,
        angular_rate_limit: float = 0.030,
        linear_rate_limit: float = 0.001,
        object_feedback_gain: float = 0.65,
        object_feedback_limit: float = 0.020,
        lift_clearance: float = 0.025,
        success_tolerance: float = 0.006,
        verbose: bool = False,
    ) -> None:
        self.xml_path = Path(xml_path).expanduser().resolve()
        self.ik = GripperIK.from_xml(self.xml_path)
        self.model_info = self.ik.model_info
        self.teacher = IKTeacher(self.ik)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._validate_mujoco_model()

        timestep = float(self.model.opt.timestep)
        self.frame_skip = max(1, int(round(1.0 / max(control_hz, 1e-9) / max(timestep, 1e-9))))
        self.control_hz = 1.0 / (self.frame_skip * timestep)
        self.angular_rate_limit = float(angular_rate_limit)
        self.linear_rate_limit = float(linear_rate_limit)
        if self.angular_rate_limit <= 0.0 or self.linear_rate_limit <= 0.0:
            raise ValueError("actuator rate limits must be positive")
        self.object_feedback_gain = float(object_feedback_gain)
        self.object_feedback_limit = float(object_feedback_limit)
        self.lift_clearance = float(lift_clearance)
        self.success_tolerance = float(success_tolerance)
        self.verbose = bool(verbose)

        nominal_contact = self.model_info.contact_distance
        if contact_distance is None:
            contact_distance = nominal_contact - max(0.0, float(contact_compression))
        self.contact_distance = float(contact_distance)
        if self.contact_distance <= self.model_info.cube_half_size:
            raise ValueError("contact distance is too small for the cube")
        self.pregrasp_distance = self.contact_distance + max(0.0, float(pregrasp_extra))

        self.cube_body_id = require_id(self.model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY_NAME)
        self.cube_joint_id = require_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT_NAME)
        self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.cube_geom_id = require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM_NAME)
        self.table_geom_id = require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM_NAME)
        self.tip_geom_ids = {
            finger: require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, TIP_GEOM_NAMES[finger])
            for finger in FINGERS
        }
        self.pin_site_ids = tuple(
            (
                require_id(self.model, mujoco.mjtObj.mjOBJ_SITE, first),
                require_id(self.model, mujoco.mjtObj.mjOBJ_SITE, second),
            )
            for first, second in PIN_SITE_PAIRS
        )
        self.actuator_joint_ids = np.asarray(self.model.actuator_trnid[:, 0], dtype=int)
        self.actuator_qpos_adrs = np.asarray(
            [int(self.model.jnt_qposadr[joint_id]) for joint_id in self.actuator_joint_ids],
            dtype=int,
        )
        self.actuator_is_slide = (
            np.asarray(self.model.jnt_type[self.actuator_joint_ids], dtype=int)
            == int(mujoco.mjtJoint.mjJNT_SLIDE)
        )
        self.actuator_rate_limits = np.where(
            self.actuator_is_slide,
            self.linear_rate_limit,
            self.angular_rate_limit,
        )
        self.ctrl_low = np.asarray(self.model.actuator_ctrlrange[:, 0], dtype=float)
        self.ctrl_high = np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=float)

        self.initial_qpos = np.asarray(self.data.qpos, dtype=float).copy()
        self.initial_qvel = np.asarray(self.data.qvel, dtype=float).copy()
        # Translation-only control keeps the XML's initial object orientation.
        # The four local-z slides and local-x armbases provide redundant XYZ
        # translation; object rotation remains outside this z/r/theta interface.
        self.translation_quat = self.cube_quat()
        self.data.ctrl[:] = self._ctrl_from_qpos(self.data.qpos)
        self.teacher.reset(self.data.ctrl)
        self.is_grasping = False

        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _validate_mujoco_model(self) -> None:
        expected_nu = len(ACTUATOR_NAMES)
        if self.model.nu != expected_nu:
            raise ValueError(f"expected {expected_nu} active actuators, found {self.model.nu}")
        actual_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
            for index in range(self.model.nu)
        )
        if actual_names != ACTUATOR_NAMES:
            raise ValueError(f"actuator order mismatch: expected {ACTUATOR_NAMES}, got {actual_names}")
        if not np.allclose(self.model.actuator_ctrlrange, self.ik.ctrl_ranges, atol=1e-10):
            raise ValueError("MuJoCo actuator limits do not match the XML-derived IK limits")
        for spec in self.model_info.actuator_specs:
            joint_id = require_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint_name)
            compiled_range = tuple(float(value) for value in self.model.jnt_range[joint_id])
            if not np.allclose(compiled_range, spec.joint_range, atol=1e-10):
                raise ValueError(
                    f"MuJoCo joint range mismatch for {spec.joint_name}: "
                    f"compiled={compiled_range}, parsed={spec.joint_range}"
                )

    def axis_summary(self) -> str:
        parts = []
        for finger in FINGERS:
            geometry = self.ik.fingers[finger]
            world = geometry.armbase_axis_world
            index = int(np.argmax(np.abs(world)))
            direction = "+" if float(world[index]) >= 0.0 else "-"
            linear_world = geometry.linear_axis_world
            linear_index = int(np.argmax(np.abs(linear_world)))
            linear_direction = "+" if float(linear_world[linear_index]) >= 0.0 else "-"
            parts.append(
                f"{finger}:slide +z -> world {linear_direction}{'XYZ'[linear_index]}, "
                f"armbase +x -> world {direction}{'XYZ'[index]}"
            )
        return ", ".join(parts)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _ctrl_from_qpos(self, qpos: Sequence[float]) -> np.ndarray:
        values = np.asarray(qpos, dtype=float)[self.actuator_qpos_adrs]
        return np.clip(values, self.ctrl_low, self.ctrl_high)

    def cube_pos(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self.cube_body_id], dtype=float).copy()

    def cube_quat(self) -> np.ndarray:
        return normalize_quat(np.asarray(self.data.xquat[self.cube_body_id], dtype=float))

    def current_ctrl(self) -> np.ndarray:
        return np.asarray(self.data.ctrl, dtype=float).copy()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.translation_quat = self.cube_quat()
        self.data.ctrl[:] = self._ctrl_from_qpos(self.data.qpos)
        self.teacher.reset(self.data.ctrl)
        self.is_grasping = False
        self._sync_viewer()

    def _sync_viewer(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()

    def _step_simulation(self) -> None:
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._sync_viewer()

    def _apply_ctrl(self, requested: Sequence[float]) -> np.ndarray:
        target = np.asarray(requested, dtype=float).reshape(self.model.nu)
        current = np.asarray(self.data.ctrl, dtype=float)
        delta = np.clip(
            target - current,
            -self.actuator_rate_limits,
            self.actuator_rate_limits,
        )
        commanded = np.clip(current + delta, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[:] = commanded
        return commanded

    def _validate_target(self, target_xyz: np.ndarray, object_quat: np.ndarray) -> None:
        min_z = self.model_info.minimum_cube_center_z
        if float(target_xyz[2]) < min_z - 1e-10:
            raise ValueError(
                f"目标 z={target_xyz[2]:.6f} m 低于桌面安全高度 {min_z:.6f} m"
            )
        half_x, half_y = float(self.model_info.table_half_size[0]), float(self.model_info.table_half_size[1])
        margin = self.model_info.cube_half_size
        if abs(float(target_xyz[0])) > half_x - margin + 1e-10:
            raise ValueError("目标 x 会使物体超出桌面")
        if abs(float(target_xyz[1])) > half_y - margin + 1e-10:
            raise ValueError("目标 y 会使物体超出桌面")
        result = self.ik.solve_cube_pose(
            target_xyz,
            object_quat,
            distance=self.contact_distance,
        )
        if not result.success:
            raise IKUnreachableError(
                "目标位置超出最新 XML 的闭链几何或物理 joint 限位；"
                f"失败手指：{', '.join(result.failed_fingers)}"
            )

    def _preferred_waypoints(
        self,
        start: np.ndarray,
        target: np.ndarray,
        object_quat: np.ndarray,
    ) -> List[np.ndarray]:
        if np.linalg.norm(np.asarray(target) - np.asarray(start)) <= 1e-8:
            return [np.asarray(start, dtype=float)]
        minimum_transit_z = self.model_info.minimum_cube_center_z + self.lift_clearance
        transit_z = max(float(start[2]), float(target[2]), minimum_transit_z)
        waypoints = [
            np.asarray(start, dtype=float),
            np.array([start[0], start[1], transit_z], dtype=float),
            np.array([target[0], target[1], transit_z], dtype=float),
            np.asarray(target, dtype=float),
        ]
        compact: List[np.ndarray] = [waypoints[0]]
        for waypoint in waypoints[1:]:
            if np.linalg.norm(waypoint - compact[-1]) > 1e-8:
                compact.append(waypoint)

        # Check several points, not just endpoints, because the workspace need
        # not be convex.  Fall back to a direct path if the lifted route exits it.
        if self._path_is_reachable(compact, object_quat):
            return compact
        direct = [np.asarray(start, dtype=float), np.asarray(target, dtype=float)]
        if self._path_is_reachable(direct, object_quat):
            return direct
        raise IKUnreachableError("目标本身可达，但从当前位置到目标的连续 IK 路径不可达")

    def _path_is_reachable(self, waypoints: Sequence[np.ndarray], object_quat: np.ndarray) -> bool:
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            distance = float(np.linalg.norm(end - start))
            samples = max(3, int(math.ceil(distance / 0.005)) + 1)
            for value in np.linspace(0.0, 1.0, samples):
                center = lerp(start, end, float(value))
                if not self.ik.cube_pose_feasible(
                    center,
                    object_quat,
                    distance=self.contact_distance,
                ):
                    return False
        return True

    @staticmethod
    def _allocate_segment_steps(waypoints: Sequence[np.ndarray], total_steps: int) -> List[int]:
        segment_count = len(waypoints) - 1
        if segment_count <= 0:
            return []
        lengths = np.asarray(
            [max(float(np.linalg.norm(end - start)), 0.005) for start, end in zip(waypoints[:-1], waypoints[1:])],
            dtype=float,
        )
        total_steps = max(total_steps, segment_count)
        raw = lengths / float(np.sum(lengths)) * total_steps
        steps = np.maximum(1, np.floor(raw).astype(int))
        while int(np.sum(steps)) < total_steps:
            index = int(np.argmax(raw - steps))
            steps[index] += 1
        while int(np.sum(steps)) > total_steps:
            candidates = np.where(steps > 1)[0]
            if candidates.size == 0:
                break
            index = int(candidates[np.argmax(steps[candidates] - raw[candidates])])
            steps[index] -= 1
        return [int(value) for value in steps]

    def _current_tip_cross(
        self,
        object_quat: Sequence[float],
    ) -> Tuple[np.ndarray, float, float]:
        """Approximate the current four tips by one symmetric target cross.

        The returned center and radius let the first commanded IK point start
        at the actual MuJoCo configuration instead of jumping directly to the
        low pregrasp pose.  ``residual`` reports the largest mismatch between
        the measured tips and that symmetric cross.
        """
        tip_positions = {
            finger: np.asarray(
                self.data.geom_xpos[self.tip_geom_ids[finger]],
                dtype=float,
            ).copy()
            for finger in FINGERS
        }
        center = np.mean(np.vstack(list(tip_positions.values())), axis=0)
        unit_targets = self.ik.cube_tip_targets(
            center,
            object_quat,
            distance=1.0,
        )
        directions = {
            finger: unit_targets[finger] - center
            for finger in FINGERS
        }
        distances = [
            float(np.dot(tip_positions[finger] - center, directions[finger]))
            for finger in FINGERS
        ]
        distance = float(np.mean(distances))
        residual = max(
            float(
                np.linalg.norm(
                    tip_positions[finger]
                    - (center + directions[finger] * distance)
                )
            )
            for finger in FINGERS
        )
        return center, distance, residual

    @staticmethod
    def _steps_for_cartesian_distance(
        distance: float,
        *,
        minimum: int,
    ) -> int:
        return max(
            int(minimum),
            int(math.ceil(max(0.0, float(distance)) / SAFE_APPROACH_TARGET_STEP_M)),
        )

    def _highest_reachable_pregrasp_center(
        self,
        object_center: np.ndarray,
        current_tip_center: np.ndarray,
        object_quat: Sequence[float],
    ) -> np.ndarray:
        def vertical_path_is_reachable(candidate: np.ndarray) -> bool:
            distance = abs(float(candidate[2] - object_center[2]))
            samples = max(
                2,
                int(math.ceil(distance / SAFE_APPROACH_TARGET_STEP_M)) + 1,
            )
            for value in np.linspace(0.0, 1.0, samples):
                center = lerp(candidate, object_center, float(value))
                if not self.ik.cube_pose_feasible(
                    center,
                    object_quat,
                    distance=self.pregrasp_distance,
                ):
                    return False
            return True

        object_rotation = quat_to_rot(object_quat)
        cube_vertical_half_extent = self.model_info.cube_half_size * float(
            np.sum(np.abs(object_rotation[2, :]))
        )
        minimum_safe_z = (
            float(object_center[2])
            + cube_vertical_half_extent
            + self.model_info.hard_tip_radius
            + SAFE_APPROACH_CLEARANCE_M
        )
        highest_z = max(float(current_tip_center[2]), minimum_safe_z)
        span = max(0.0, highest_z - minimum_safe_z)
        sample_count = int(math.ceil(span / SAFE_APPROACH_TARGET_STEP_M))
        for sample_index in range(sample_count + 1):
            z = max(
                minimum_safe_z,
                highest_z - sample_index * SAFE_APPROACH_TARGET_STEP_M,
            )
            candidate = np.array(
                [float(object_center[0]), float(object_center[1]), z],
                dtype=float,
            )
            if self.ik.cube_pose_feasible(
                candidate,
                object_quat,
                distance=self.pregrasp_distance,
            ) and vertical_path_is_reachable(candidate):
                return candidate
        raise IKUnreachableError(
            "找不到位于 cube 上方且满足当前 XML 限位的安全预抓取高度"
        )

    def _build_stages(
        self,
        start: np.ndarray,
        target: np.ndarray,
        object_quat: np.ndarray,
        total_steps: int,
    ) -> List[MotionStage]:
        total_steps = max(1, int(total_steps))
        stages: List[MotionStage] = []
        if not self.is_grasping:
            current_tip_center, current_tip_distance, cross_residual = self._current_tip_cross(
                object_quat
            )
            safe_center = self._highest_reachable_pregrasp_center(
                start,
                current_tip_center,
                object_quat,
            )

            current_targets = self.ik.cube_tip_targets(
                current_tip_center,
                object_quat,
                distance=current_tip_distance,
            )
            safe_targets = self.ik.cube_tip_targets(
                safe_center,
                object_quat,
                distance=self.pregrasp_distance,
            )
            shape_distance = max(
                float(np.linalg.norm(safe_targets[finger] - current_targets[finger]))
                for finger in FINGERS
            )
            shape_steps = self._steps_for_cartesian_distance(
                shape_distance + cross_residual,
                minimum=SAFE_SHAPE_MIN_STEPS,
            )
            descent_steps = self._steps_for_cartesian_distance(
                float(np.linalg.norm(safe_center - start)),
                minimum=SAFE_DESCENT_MIN_STEPS,
            )
            close_steps = self._steps_for_cartesian_distance(
                self.pregrasp_distance - self.contact_distance,
                minimum=SAFE_CLOSE_MIN_STEPS,
            )
            move_steps = total_steps
            stages.extend(
                [
                    MotionStage(
                        safe_center,
                        safe_center,
                        self.pregrasp_distance,
                        self.pregrasp_distance,
                        shape_steps,
                        False,
                        "pregrasp-shape",
                    ),
                    MotionStage(
                        safe_center,
                        start,
                        self.pregrasp_distance,
                        self.pregrasp_distance,
                        descent_steps,
                        False,
                        "pregrasp-descend",
                    ),
                    MotionStage(
                        start,
                        start,
                        self.pregrasp_distance,
                        self.contact_distance,
                        close_steps,
                        False,
                        "close",
                    ),
                    MotionStage(
                        start,
                        start,
                        self.contact_distance,
                        self.contact_distance,
                        GRASP_SETTLE_STEPS,
                        False,
                        "grasp-settle",
                    ),
                ]
            )
        else:
            move_steps = total_steps

        waypoints = self._preferred_waypoints(start, target, object_quat)
        segment_steps = self._allocate_segment_steps(waypoints, move_steps)
        for index, ((segment_start, segment_end), steps) in enumerate(
            zip(zip(waypoints[:-1], waypoints[1:]), segment_steps),
            start=1,
        ):
            stages.append(
                MotionStage(
                    segment_start,
                    segment_end,
                    self.contact_distance,
                    self.contact_distance,
                    steps,
                    True,
                    f"move-{index}",
                )
            )
        return stages

    def _valid_side_contact_fingers(self) -> set[str]:
        valid: set[str] = set()
        tip_id_to_finger = {
            geom_id: finger
            for finger, geom_id in self.tip_geom_ids.items()
        }
        cube_center = self.cube_pos()
        cube_rotation = quat_to_rot(self.cube_quat())
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self.cube_geom_id and geom2 in tip_id_to_finger:
                finger = tip_id_to_finger[geom2]
                outward_world = np.asarray(contact.frame[:3], dtype=float)
            elif geom2 == self.cube_geom_id and geom1 in tip_id_to_finger:
                finger = tip_id_to_finger[geom1]
                outward_world = -np.asarray(contact.frame[:3], dtype=float)
            else:
                continue
            if int(contact.efc_address) < 0:
                continue

            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            if float(force[0]) < GRASP_MIN_NORMAL_FORCE_N:
                continue

            expected = np.asarray(
                self.ik.contact_directions_world[finger],
                dtype=float,
            )
            outward_local = cube_rotation.T @ outward_world
            contact_local = cube_rotation.T @ (
                np.asarray(contact.pos, dtype=float) - cube_center
            )
            tangent = np.array([-expected[1], expected[0], 0.0], dtype=float)
            if float(np.dot(outward_local, expected)) < GRASP_NORMAL_ALIGNMENT_MIN:
                continue
            if abs(float(outward_local[2])) > 0.35:
                continue
            if abs(
                float(np.dot(contact_local, expected))
                - self.model_info.cube_half_size
            ) > GRASP_SIDE_PLANE_TOLERANCE_M:
                continue
            if abs(float(contact_local[2])) > GRASP_SIDE_INTERIOR_LIMIT_M:
                continue
            if abs(float(np.dot(contact_local, tangent))) > GRASP_SIDE_INTERIOR_LIMIT_M:
                continue
            valid.add(finger)
        return valid

    def _side_grasp_status(self) -> Tuple[bool, str]:
        """Check actual four-side contact before any lift or lateral motion."""
        cube_center = self.cube_pos()
        targets = self.ik.cube_tip_targets(
            cube_center,
            self.cube_quat(),
            distance=self.contact_distance,
        )
        tracking_errors: Dict[str, float] = {}
        height_errors: Dict[str, float] = {}
        for finger in FINGERS:
            actual = np.asarray(
                self.data.geom_xpos[self.tip_geom_ids[finger]],
                dtype=float,
            )
            tracking_errors[finger] = float(np.linalg.norm(actual - targets[finger]))
            height_errors[finger] = abs(float(actual[2] - targets[finger][2]))

        valid_contacts = self._valid_side_contact_fingers()
        missing = [finger for finger in FINGERS if finger not in valid_contacts]
        max_tracking = max(tracking_errors.values())
        max_height = max(height_errors.values())
        success = not (
            missing
            or max_tracking > GRASP_TRACKING_TOLERANCE_M
            or max_height > GRASP_HEIGHT_TOLERANCE_M
        )
        detail = (
            f"缺少有效侧面接触={missing or '无'}，"
            f"最大指尖跟踪误差={max_tracking * 1000.0:.2f} mm，"
            f"最大高度误差={max_height * 1000.0:.2f} mm"
        )
        return success, detail

    def _confirm_side_grasp(self, confirmation_streak: int, detail: str) -> None:
        if confirmation_streak < GRASP_CONFIRMATION_STEPS:
            self.is_grasping = False
            raise IKUnreachableError(
                "抓取确认失败，已停止搬运："
                f"{detail}；连续有效周期={confirmation_streak}/"
                f"{GRASP_CONFIRMATION_STEPS}。"
                "请 reset 后重试。"
            )

    def _tracking_metrics(self, targets: dict[str, np.ndarray]) -> Tuple[float, float]:
        tip_errors = [
            float(np.linalg.norm(self.data.geom_xpos[self.tip_geom_ids[finger]] - targets[finger]))
            for finger in FINGERS
        ]
        pin_errors = [
            float(np.linalg.norm(self.data.site_xpos[first] - self.data.site_xpos[second]))
            for first, second in self.pin_site_ids
        ]
        return max(tip_errors), max(pin_errors)

    def _solve_and_step(
        self,
        desired_center: np.ndarray,
        object_quat: np.ndarray,
        contact_distance: float,
        *,
        use_feedback: bool,
    ) -> Tuple[float, float, float, float]:
        ik_center = np.asarray(desired_center, dtype=float).copy()
        if use_feedback and self.object_feedback_gain > 0.0:
            correction = self.object_feedback_gain * (ik_center - self.cube_pos())
            correction = np.clip(
                correction,
                -self.object_feedback_limit,
                self.object_feedback_limit,
            )
            ik_center = ik_center + correction

        try:
            solution = self.teacher.solve_cube_pose(
                ik_center,
                object_quat,
                distance=contact_distance,
                strict=True,
                update=False,
            )
        except IKUnreachableError:
            # Feedback can push the virtual center just outside the analytic
            # workspace at a boundary.  The planned center is already checked.
            solution = self.teacher.solve_cube_pose(
                desired_center,
                object_quat,
                distance=contact_distance,
                strict=True,
                update=False,
            )
            ik_center = np.asarray(desired_center, dtype=float)

        commanded = self._apply_ctrl(solution.ctrl)
        self.teacher.commit(solution, ctrl=commanded)
        self._step_simulation()
        targets = {
            finger: solution.finger_solutions[finger].tip_world
            for finger in FINGERS
        }
        sim_tip_error, sim_pin_error = self._tracking_metrics(targets)
        return (
            solution.max_tip_error,
            solution.max_loop_error,
            sim_tip_error,
            sim_pin_error,
        )

    def run_to_xyz(
        self,
        target_xyz: Sequence[float],
        *,
        steps: int = 420,
        hold_steps: int = 80,
        reset_first: bool = False,
    ) -> RunResult:
        target = np.asarray(target_xyz, dtype=float).reshape(3)
        if reset_first:
            self.reset()
        start_time = time.monotonic()
        start = self.cube_pos()
        object_quat = self.translation_quat.copy()
        self._validate_target(target, object_quat)
        stages = self._build_stages(start, target, object_quat, int(steps))

        maxima = np.zeros(4, dtype=float)
        completed_steps = 0
        grasp_confirmation_streak = 0
        grasp_confirmation_detail = "尚未检测"
        for stage_index, stage in enumerate(stages, start=1):
            for step_index in range(stage.steps):
                value = minimum_jerk((step_index + 1) / max(1, stage.steps))
                center = lerp(stage.start_center, stage.end_center, value)
                distance = (1.0 - value) * stage.start_distance + value * stage.end_distance
                metrics = self._solve_and_step(
                    center,
                    object_quat,
                    distance,
                    use_feedback=stage.use_feedback,
                )
                maxima = np.maximum(maxima, np.asarray(metrics))
                completed_steps += 1
                if stage.label == "grasp-settle":
                    valid_grasp, grasp_confirmation_detail = self._side_grasp_status()
                    grasp_confirmation_streak = (
                        grasp_confirmation_streak + 1
                        if valid_grasp
                        else 0
                    )
                if self.verbose and (
                    step_index == 0
                    or step_index + 1 == stage.steps
                    or (step_index + 1) % 25 == 0
                ):
                    print(
                        f"[{stage_index}/{len(stages)} {stage.label}] "
                        f"{step_index + 1:03d}/{stage.steps:03d} "
                        f"desired={format_vector(center, 4)} "
                        f"cube={format_vector(self.cube_pos(), 4)} "
                        f"sim_tip={metrics[2] * 1000.0:.2f} mm",
                        flush=True,
                    )
            if stage.label == "grasp-settle":
                self._confirm_side_grasp(
                    grasp_confirmation_streak,
                    grasp_confirmation_detail,
                )
                self.is_grasping = True

        for hold_index in range(max(0, int(hold_steps))):
            metrics = self._solve_and_step(
                target,
                object_quat,
                self.contact_distance,
                use_feedback=True,
            )
            maxima = np.maximum(maxima, np.asarray(metrics))
            completed_steps += 1
            if self.verbose and (hold_index + 1) % 25 == 0:
                print(
                    f"[hold] {hold_index + 1:03d}/{hold_steps:03d} "
                    f"cube={format_vector(self.cube_pos(), 4)}",
                    flush=True,
                )

        final = self.cube_pos()
        error = final - target
        success = bool(np.all(np.abs(error) <= self.success_tolerance))
        return RunResult(
            target_xyz=target,
            final_xyz=final,
            position_error=error,
            success=success,
            steps=completed_steps,
            max_analytic_tip_error=float(maxima[0]),
            max_analytic_loop_error=float(maxima[1]),
            max_sim_tip_tracking_error=float(maxima[2]),
            max_sim_pin_error=float(maxima[3]),
            elapsed_s=time.monotonic() - start_time,
        )

    def release(self, *, steps: int = 80) -> None:
        if not self.is_grasping:
            print("夹爪当前未标记为抓取状态。")
            return
        center = self.cube_pos()
        quat = self.translation_quat.copy()
        for index in range(max(1, int(steps))):
            value = minimum_jerk((index + 1) / max(1, int(steps)))
            distance = (1.0 - value) * self.contact_distance + value * self.pregrasp_distance
            self._solve_and_step(center, quat, distance, use_feedback=False)
        self.is_grasping = False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Gripper4 解析 IK，在 MuJoCo 中抓取并移动方块。终端坐标顺序为 z r theta。"
    )
    parser.add_argument("target", nargs="*", type=float, help="可选单次目标：z r theta")
    parser.add_argument(
        "--xml",
        default=str(DEFAULT_XML),
        help="Project.xml；文件必须位于本脚本文件夹内",
    )
    parser.add_argument("--no-render", action="store_true", help="不打开 MuJoCo 窗口")
    parser.add_argument("--theta-radians", action="store_true", help="theta 使用弧度（默认角度）")
    parser.add_argument("--origin-x", type=float, default=0.0, help="柱坐标原点 x")
    parser.add_argument("--origin-y", type=float, default=0.0, help="柱坐标原点 y")
    parser.add_argument(
        "--steps",
        type=int,
        default=420,
        help="抓取完成后搬运路径的控制步数；首次安全预抓取阶段另计",
    )
    parser.add_argument("--hold-steps", type=int, default=80, help="到达后保持步数")
    parser.add_argument("--control-hz", type=float, default=50.0, help="外层 IK/控制频率")
    parser.add_argument("--reset-each", action="store_true", help="每次移动前重置仿真")
    parser.add_argument("--contact-distance", type=float, default=None, help="方块中心到硬指尖球心的接触距离")
    parser.add_argument("--contact-compression", type=float, default=0.0015, help="相对几何相切距离的压入量")
    parser.add_argument("--pregrasp-extra", type=float, default=0.015, help="预抓取额外张开距离")
    parser.add_argument("--angular-rate-limit", type=float, default=0.030, help="每控制步最大执行器目标变化量")
    parser.add_argument("--linear-rate-limit", type=float, default=0.001, help="每控制步最大直线执行器目标变化量（m）")
    parser.add_argument("--feedback-gain", type=float, default=0.65, help="物体位置外环反馈增益")
    parser.add_argument("--feedback-limit", type=float, default=0.020, help="IK 中心每轴最大反馈修正")
    parser.add_argument("--lift-clearance", type=float, default=0.025, help="横向移动前相对桌面的抬升余量")
    parser.add_argument("--success-tol", type=float, default=0.006, help="每轴成功误差阈值")
    parser.add_argument("--verbose", action="store_true", help="打印控制过程")
    return parser.parse_args(argv)


def build_controller(args: argparse.Namespace) -> T4IKController:
    return T4IKController(
        args.xml,
        render=not bool(args.no_render),
        control_hz=float(args.control_hz),
        contact_distance=args.contact_distance,
        contact_compression=float(args.contact_compression),
        pregrasp_extra=float(args.pregrasp_extra),
        angular_rate_limit=float(args.angular_rate_limit),
        linear_rate_limit=float(args.linear_rate_limit),
        object_feedback_gain=float(args.feedback_gain),
        object_feedback_limit=float(args.feedback_limit),
        lift_clearance=float(args.lift_clearance),
        success_tolerance=float(args.success_tol),
        verbose=bool(args.verbose),
    )


def print_help() -> None:
    print(
        "\n命令：\n"
        "  <z_m> <r_m> <theta_deg>       抓取/移动到柱坐标，例如 0.32 0.03 90\n"
        "  go <z_m> <r_m> <theta_deg>    同上\n"
        "  xyz <x_m> <y_m> <z_m>         使用世界直角坐标\n"
        "  where                         显示方块当前位置（XYZ 与 z/r/theta）\n"
        "  joints                        显示当前 16 维执行器目标及单位\n"
        "  maxr <theta>                  扫描该 theta 下最大 r 及对应 z（默认度）\n"
        "  release                       张开夹爪\n"
        "  reset                         恢复 XML 初始状态\n"
        "  set steps <n>                 修改移动步数\n"
        "  set hold <n>                  修改到达后保持步数\n"
        "  help                          显示帮助\n"
        "  q / quit / exit               退出\n"
    )


def print_result(result: RunResult) -> None:
    print("\n移动完成")
    print(f"  目标 XYZ:       {format_vector(result.target_xyz)}")
    print(f"  最终 XYZ:       {format_vector(result.final_xyz)}")
    print(f"  分轴误差:       {format_vector(result.position_error)}")
    print(f"  误差范数:       {np.linalg.norm(result.position_error) * 1000.0:.3f} mm")
    print(f"  解析 tip 最大误差:  {result.max_analytic_tip_error * 1000.0:.6f} mm")
    print(f"  解析闭链最大误差:   {result.max_analytic_loop_error * 1000.0:.6f} mm")
    print(f"  仿真 tip 最大跟踪差: {result.max_sim_tip_tracking_error * 1000.0:.3f} mm")
    print(f"  仿真 pin 最大误差:  {result.max_sim_pin_error * 1000.0:.3f} mm")
    print(f"  success={result.success}  steps={result.steps}  elapsed={result.elapsed_s:.2f}s\n")


def _run_target(
    controller: T4IKController,
    args: argparse.Namespace,
    values: Sequence[str | float],
) -> RunResult:
    if len(values) != 3:
        raise ValueError("目标格式：<z_m> <r_m> <theta_deg>")
    z_m, r_m, theta = (float(values[0]), float(values[1]), float(values[2]))
    target = cylindrical_to_xyz(
        z_m,
        r_m,
        theta,
        theta_radians=bool(args.theta_radians),
        origin_xy=(float(args.origin_x), float(args.origin_y)),
    )
    result = controller.run_to_xyz(
        target,
        steps=int(args.steps),
        hold_steps=int(args.hold_steps),
        reset_first=bool(args.reset_each),
    )
    print_result(result)
    return result


def execute_line(controller: T4IKController, args: argparse.Namespace, line: str) -> bool:
    parts = line.strip().replace(",", " ").split()
    if not parts:
        return True
    command = parts[0].lower()
    if command in {"q", "quit", "exit"}:
        return False
    if command in {"h", "help", "?"}:
        print_help()
        return True
    if command == "reset":
        controller.reset()
        print(f"已重置；方块 XYZ={format_vector(controller.cube_pos())}")
        return True
    if command == "release":
        controller.release()
        print("已张开夹爪。")
        return True
    if command == "where":
        position = controller.cube_pos()
        cylindrical = xyz_to_cylindrical(
            position,
            origin_xy=(float(args.origin_x), float(args.origin_y)),
            theta_radians=bool(args.theta_radians),
        )
        unit = "rad" if args.theta_radians else "deg"
        print(f"cube XYZ: {format_vector(position)}")
        print(f"cube z/r/theta({unit}): {format_vector(cylindrical)}")
        return True
    if command == "joints":
        print("actuator order: " + ", ".join(ACTUATOR_NAMES))
        for spec, value in zip(controller.model_info.actuator_specs, controller.current_ctrl()):
            unit = "m" if spec.kind == "linear" else "rad"
            print(f"  {spec.index:2d} {spec.name:10s} {float(value):+.6f} {unit}")
        return True
    if command == "maxr":
        if len(parts) != 2:
            raise ValueError("格式：maxr <theta>（默认单位为度）")
        theta = float(parts[1])
        limit = controller.ik.maximum_cylindrical_radius(
            theta,
            theta_radians=bool(args.theta_radians),
            origin_xy=(float(args.origin_x), float(args.origin_y)),
            quaternion_world=controller.translation_quat,
            distance=controller.contact_distance,
        )
        unit = "rad" if args.theta_radians else "deg"
        shown_theta = theta if args.theta_radians else limit.theta_deg
        print(f"theta={shown_theta:.6f} {unit} 的解析 IK 极限：")
        print(f"  max r = {limit.radius:.6f} m")
        print(f"  z     = {limit.z:.6f} m")
        print(f"  XYZ   = {format_vector(limit.xyz)}")
        print(f"  scan resolution = {limit.resolution:.1e} m")
        return True
    if command == "set":
        if len(parts) != 3:
            raise ValueError("格式：set steps <n> 或 set hold <n>")
        value = int(parts[2])
        if value < 0:
            raise ValueError("步数不能为负")
        key = parts[1].lower()
        if key == "steps":
            if value == 0:
                raise ValueError("steps 必须大于零")
            args.steps = value
        elif key in {"hold", "hold_steps", "hold-steps"}:
            args.hold_steps = value
        else:
            raise ValueError("只支持 steps 或 hold")
        print(f"{key} = {value}")
        return True
    if command == "xyz":
        if len(parts) != 4:
            raise ValueError("格式：xyz <x_m> <y_m> <z_m>")
        target = np.asarray([float(value) for value in parts[1:]], dtype=float)
        result = controller.run_to_xyz(
            target,
            steps=int(args.steps),
            hold_steps=int(args.hold_steps),
            reset_first=bool(args.reset_each),
        )
        print_result(result)
        return True
    if command == "go":
        parts = parts[1:]
    _run_target(controller, args, parts)
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    restart_with_mjpython_if_needed(raw_argv)
    args = parse_args(raw_argv)
    if args.target and len(args.target) != 3:
        raise SystemExit("单次目标必须正好提供三个数：z r theta")

    controller = build_controller(args)
    try:
        print("Gripper4 IK-only 交互控制")
        print(f"  model: {controller.xml_path}")
        print(f"  cube XYZ: {format_vector(controller.cube_pos())}")
        print(f"  armbase axes: {controller.axis_summary()}")
        print(f"  active ranges: {controller.ik.active_range_summary()}")
        print("  joint/actuator ranges: exact match")
        print(
            f"  contact={controller.contact_distance:.4f} m, "
            f"pregrasp={controller.pregrasp_distance:.4f} m, "
            f"control={controller.control_hz:.1f} Hz"
        )
        if args.target:
            try:
                result = _run_target(controller, args, args.target)
            except (ValueError, IKUnreachableError) as exc:
                print(f"错误：{exc}")
                return 2
            return 0 if result.success else 2

        print_help()
        while True:
            try:
                line = input("t4-ik> ")
            except EOFError:
                print()
                break
            try:
                if not execute_line(controller, args, line):
                    break
            except (ValueError, IKUnreachableError) as exc:
                print(f"错误：{exc}")
            except KeyboardInterrupt:
                print("\n当前命令已中断；输入 quit 可退出。")
        return 0
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
