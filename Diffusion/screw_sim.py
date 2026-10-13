#!/usr/bin/env python3
"""MuJoCo/IK integration for the two-turn screw task.

The simulator keeps three layers deliberately separate:

* :class:`~screw_control.ScrewCommand` describes the desired screw pose and
  legal regrasp contact mode.
* the local analytic T6 IK maps four fingertip centres to all 16 actuators;
* MuJoCo supplies rigid-body dynamics, contacts and the virtual helical
  constraint used after thread engagement.

The thread teeth drawn in the XML are visual.  Their mechanically meaningful
behaviour is the differentiable constraint

``depth - engagement_depth = pitch * accumulated_angle / (2*pi)``.

This is considerably more stable for imitation/RL training than resolving
sub-millimetre triangle-mesh thread contacts, while retaining insertion,
torque resistance, axial preload, jamming and alignment errors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import mujoco
import numpy as np

from grasp_plane import AdaptiveScrewGraspPlanePlanner
from kinematics import (
    ACTUATOR_COUNT,
    FINGERS,
    IKTeacher,
    GripperIK,
    TIP_BODY_NAMES,
    TIP_GEOM_NAMES,
)
from screw_config import (
    DEFAULT_CONFIG,
    SCREW_BODY_NAME,
    SCREW_FREE_JOINT_NAME,
    SCREW_GRASP_SITE_NAME,
    SCREW_HEAD_GEOM_NAME,
    SCREW_SHAFT_GEOM_NAME,
    SCREW_START_HOLDER_BODY_NAME,
    SCREW_START_HOLDER_EQUALITY_NAME,
    SCREW_TIP_SITE_NAME,
    SOCKET_BOTTOM_SITE_NAME,
    SOCKET_MOUTH_SITE_NAME,
    ScrewTaskConfig,
    validate_config,
)
from screw_control import (
    RELEASED_TIP_CENTER_DISTANCE_M,
    SCRIPTED_PREFIX_STEPS,
    SOCKET_YAW_EQUIVARIANCE_VERSION,
    SOCKET_YAW_ROLL_GAIN,
    ScrewCommand,
    ScrewExpert,
    TaskPhase,
    align_local_z_to,
    matrix_to_quaternion_wxyz,
    quaternion_slerp_wxyz,
    quaternion_to_matrix_wxyz,
    rotation_log,
    validate_contact_schedule,
)
from screw_observation import OmniscientObservationBuilder, TwistTracker
from screw_scene import MODULE_DIR, sample_spawn_xy, write_scene_xml


@dataclass(frozen=True)
class SimulationSettings:
    """Numerical settings, all independent of the task geometry."""

    control_dt_s: float = 0.020
    max_control_steps: int = 3_000
    strict_ik: bool = False
    ik_linear_search_step_m: float = 0.001
    # FLUID-style actuator smoothing below the analytic IK. The analytic
    # solution remains the target; bandwidth, acceleration and jerk limits
    # turn it into a command the closed-chain mechanism can track.
    ik_bandwidth_hz: float = 10.0
    hinge_rate_limit_rad_per_step: float = 0.15
    linear_rate_limit_m_per_step: float = 0.004
    hinge_acceleration_limit_rad_s2: float = 8.0
    hinge_jerk_limit_rad_s3: float = 80.0
    linear_acceleration_limit_m_s2: float = 4.0
    linear_jerk_limit_m_s3: float = 40.0

    # A light Cartesian guide is useful for clean expert demonstrations.  It
    # is a wrench, not a kinematic teleport, so contacts and failure modes stay
    # visible.  Curriculum training can reduce guide_scale to zero.
    # ``mocap`` is the deterministic expert-demonstration fixture. ``wrench``
    # is a compliant curriculum aid; ``none`` is the final physical task.
    guide_mode: str = "mocap"
    guide_scale: float = 1.0
    guide_position_kp: float = 90.0
    guide_position_kd: float = 5.0
    guide_rotation_kp: float = 0.70
    guide_rotation_kd: float = 0.035
    guide_force_limit_n: float = 18.0
    guide_torque_limit_nm: float = 0.30

    # MuJoCo's point-contact model lets a 50 mm cylindrical head slip from a
    # two-sphere pinch during the long 45-degree transfer. Once a real
    # opposing pair has closed, this bounded compliant wrench represents the
    # high-stiffness grasp that the physical fingers/soft pads would provide.
    # It is deliberately distinct from the optional curriculum guide.
    grasp_stabilization: bool = True
    grasp_position_kp: float = 500.0
    grasp_position_kd: float = 20.0
    grasp_rotation_kp: float = 4.0
    grasp_rotation_kd: float = 0.15
    grasp_force_limit_n: float = 40.0
    grasp_torque_limit_nm: float = 1.50

    helix_stiffness_n_per_m: float = 1_800.0
    helix_damping_ns_per_m: float = 28.0
    helix_force_limit_n: float = 25.0
    thread_lateral_stiffness_n_per_m: float = 1_000.0
    thread_lateral_damping_ns_per_m: float = 18.0
    thread_lateral_force_limit_n: float = 15.0
    thread_alignment_stiffness_nm_per_rad: float = 1.0
    thread_alignment_damping_nms_per_rad: float = 0.05
    thread_alignment_torque_limit_nm: float = 0.30
    resistance_velocity_scale_rad_s: float = 0.10
    insertion_command_lead_m: float = 0.0010
    engage_depth_tolerance_m: float = 0.0015
    engage_lateral_tolerance_m: float = 0.0030
    engage_axis_tolerance_rad: float = math.radians(5.0)

    def validate(self) -> "SimulationSettings":
        if self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        if self.max_control_steps <= 0:
            raise ValueError("max_control_steps must be positive")
        if self.ik_linear_search_step_m <= 0.0:
            raise ValueError("ik_linear_search_step_m must be positive")
        smoothing_limits = (
            self.ik_bandwidth_hz,
            self.hinge_rate_limit_rad_per_step,
            self.linear_rate_limit_m_per_step,
            self.hinge_acceleration_limit_rad_s2,
            self.hinge_jerk_limit_rad_s3,
            self.linear_acceleration_limit_m_s2,
            self.linear_jerk_limit_m_s3,
        )
        if any(value <= 0.0 for value in smoothing_limits):
            raise ValueError("all IK smoothing limits must be positive")
        if not 0.0 <= self.guide_scale <= 1.0:
            raise ValueError("guide_scale must lie in [0, 1]")
        if self.guide_mode not in ("mocap", "wrench", "none"):
            raise ValueError("guide_mode must be 'mocap', 'wrench' or 'none'")
        grasp_limits = (
            self.grasp_position_kp,
            self.grasp_position_kd,
            self.grasp_rotation_kp,
            self.grasp_rotation_kd,
            self.grasp_force_limit_n,
            self.grasp_torque_limit_nm,
        )
        if self.grasp_stabilization and any(
            value <= 0.0 for value in grasp_limits
        ):
            raise ValueError(
                "enabled grasp stabilization limits must be positive"
            )
        engagement_limits = (
            self.insertion_command_lead_m,
            self.engage_depth_tolerance_m,
            self.engage_lateral_tolerance_m,
            self.engage_axis_tolerance_rad,
        )
        if any(value <= 0.0 for value in engagement_limits):
            raise ValueError("all thread engagement tolerances must be positive")
        thread_limits = (
            self.helix_stiffness_n_per_m,
            self.helix_damping_ns_per_m,
            self.helix_force_limit_n,
            self.thread_lateral_stiffness_n_per_m,
            self.thread_lateral_damping_ns_per_m,
            self.thread_lateral_force_limit_n,
            self.thread_alignment_stiffness_nm_per_rad,
            self.thread_alignment_damping_nms_per_rad,
            self.thread_alignment_torque_limit_nm,
            self.resistance_velocity_scale_rad_s,
        )
        if any(value <= 0.0 for value in thread_limits):
            raise ValueError("all virtual thread limits must be positive")
        return self


@dataclass(frozen=True)
class StepDiagnostics:
    ik_success: bool
    ik_max_tip_error_m: float
    holder_active: bool
    engaged: bool
    insertion_depth_m: float
    twist_rad: float
    lateral_error_m: float
    axis_error_rad: float
    helical_error_m: float
    contact_count: int


def _name_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    identifier = int(mujoco.mj_name2id(model, object_type, name))
    if identifier < 0:
        raise KeyError(f"MuJoCo object not found: {name!r}")
    return identifier


def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > float(maximum) > 0.0:
        value = value * (float(maximum) / norm)
    return value


class ScrewSimulator:
    """One portable MuJoCo scene and its stateful analytic IK controller."""

    def __init__(
        self,
        *,
        config: ScrewTaskConfig = DEFAULT_CONFIG,
        settings: SimulationSettings = SimulationSettings(),
        screw_xy_m: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
        scene_path: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        self.config = validate_config(config)
        if not math.isclose(
            self.config.released_tip_center_distance_m,
            RELEASED_TIP_CENTER_DISTANCE_M,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "config and ScrewExpert released-distance contracts differ"
            )
        if (
            self.config.socket_yaw_equivariance_version
            != SOCKET_YAW_EQUIVARIANCE_VERSION
            or not math.isclose(
                self.config.socket_yaw_roll_gain,
                SOCKET_YAW_ROLL_GAIN,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "config and ScrewExpert yaw-equivariance contracts differ"
            )
        self.settings = settings.validate()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.spawn_xy_m = tuple(
            float(v)
            for v in (
                self.config.default_screw_xy_m
                if screw_xy_m is None
                else screw_xy_m
            )
        )

        if scene_path is None:
            runtime_name = (
                f"scene_{os.getpid()}_{id(self):x}.xml"
            )
            self.scene_path = write_scene_xml(
                Path(".runtime") / runtime_name,
                config=self.config,
                screw_xy_m=self.spawn_xy_m,
            )
        else:
            self.scene_path = Path(scene_path).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.physics_steps_per_control = max(
            1,
            int(round(self.settings.control_dt_s / float(self.model.opt.timestep))),
        )
        self.control_dt_s = (
            self.physics_steps_per_control * float(self.model.opt.timestep)
        )
        self.max_control_steps = int(self.settings.max_control_steps)
        self.control_step = 0
        self.ik_failure_count = 0

        self._cache_ids()
        mujoco.mj_forward(self.model, self.data)
        self._make_ik()
        self.grasp_plan = AdaptiveScrewGraspPlanePlanner(
            head_radius_m=self.config.screw_head_radius_m,
            head_bottom_local_z_m=0.5 * self.config.screw_shaft_length_m,
            head_height_m=self.config.screw_head_height_m,
            screw_tip_local_z_m=self.config.screw_tip_local_z_m,
            full_insertion_depth_m=self.config.target_insertion_depth_m,
        ).plan(
            body_world_pos=(
                self.spawn_xy_m[0],
                self.spawn_xy_m[1],
                self.config.screw_initial_center_z_m,
            ),
            table_top_z=self.config.table_top_z_m,
        )
        if not math.isclose(
            self.grasp_plan.grasp_height_from_tip_m,
            self.config.screw_grasp_height_from_tip_m,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "adaptive grasp plane and screw grasp site disagree"
            )

        self.expert = ScrewExpert(
            spawn_xy_m=self.spawn_xy_m,
            table_top_z_m=self.config.table_top_z_m,
            socket_mouth_world_m=self.config.socket_mouth_position_m,
            socket_axis_inward_world=self.config.socket_axis_world,
            contact_distance_m=self.grasp_plan.contact_radius_m,
            pregrasp_distance_m=self.grasp_plan.pregrasp_radius_m,
            released_distance_m=(
                self.config.released_tip_center_distance_m
            ),
            grasp_plane_yaw_rad=self.grasp_plan.plane.yaw_rad,
            engage_depth_m=self.config.thread_engagement_depth_m,
            final_depth_m=self.config.target_insertion_depth_m,
            pitch_m=self.config.thread_pitch_m,
            target_turns=self.config.target_turns,
            grasp_height_from_tip_m=(
                self.config.screw_grasp_height_from_tip_m
            ),
            scripted_grasp_plane_world_z_m=(
                self.config.scripted_grasp_plane_world_z_m
            ),
            scripted_center_xy_m=self.config.scripted_center_xy_m,
        )
        self.current_command = self.expert.reset()
        self.target_depth_m = self.config.target_insertion_depth_m
        self.target_twist_rad = self.config.target_rotation_rad
        self.thread_engaged = False
        self.thread_resistance_nm = self.config.rotation_resistance_min_nm
        self.axial_preload_n = self.config.axial_preload_n
        self._last_helical_lambda_n = 0.0
        self.grasp_latched = False

        reference_rotation = align_local_z_to(
            -np.asarray(self.config.socket_axis_world, dtype=np.float64)
        )
        self._twist_tracker = TwistTracker(
            reference_marker_world=reference_rotation[:, 0],
            outward_axis_world=-np.asarray(
                self.config.socket_axis_world, dtype=np.float64
            ),
        )
        self.marker_mod_angle_rad = 0.0
        self.twist_unwrapped_rad = 0.0

        self._initialize_gripper(self.current_command)
        # This command has just been solved with strict analytic IK and
        # committed by ``_initialize_gripper``.  Keep an explicit certified
        # anchor instead of assuming that the open-loop supervisor will
        # always remain reachable when an under-trained policy lags behind.
        self._last_certified_ik_command = self.current_command.copy()
        self.last_ik_projection_used_fallback = False
        self.ik_safety_stop = False
        self.ik_safety_reason = ""
        self.ik_anchor_fallback_count = 0
        self.ik_safety_stop_count = 0
        self._ctrl_velocity = np.zeros(ACTUATOR_COUNT, dtype=np.float64)
        self._ctrl_acceleration = np.zeros(
            ACTUATOR_COUNT, dtype=np.float64
        )
        self.observation_builder = OmniscientObservationBuilder(self)
        self._update_task_state(reset_twist=True)

    # ------------------------------------------------------------------
    # Construction and reset helpers
    # ------------------------------------------------------------------

    @classmethod
    def sample_reachable_spawn(
        cls,
        *,
        config: ScrewTaskConfig = DEFAULT_CONFIG,
        seed: Optional[int] = None,
        ik: Optional[GripperIK] = None,
    ) -> Tuple[float, float]:
        """Area-uniform reset with geometry and four-tip IK rejection."""

        cfg = validate_config(config)
        solver = ik or GripperIK.from_xml(MODULE_DIR / "Project.xml")
        rng = np.random.default_rng(seed)
        grasp_plan = AdaptiveScrewGraspPlanePlanner(
            head_radius_m=cfg.screw_head_radius_m,
            head_bottom_local_z_m=0.5 * cfg.screw_shaft_length_m,
            head_height_m=cfg.screw_head_height_m,
            screw_tip_local_z_m=cfg.screw_tip_local_z_m,
            full_insertion_depth_m=cfg.target_insertion_depth_m,
        ).plan()

        def reachable(grasp_world: Sequence[float]) -> bool:
            point = np.asarray(grasp_world, dtype=np.float64)
            yaw = grasp_plan.plane.yaw_rad
            radius = grasp_plan.pregrasp_radius_m
            targets = {
                finger: point
                + np.array(
                    (
                        radius
                        * math.cos(
                            angle + yaw
                        ),
                        radius
                        * math.sin(
                            angle + yaw
                        ),
                        0.0,
                    )
                )
                for finger, angle in {
                    "U": 0.5 * math.pi,
                    "D": -0.5 * math.pi,
                    "L": math.pi,
                    "R": 0.0,
                }.items()
            }
            return solver.solve_tip_targets(targets, strict=False).success

        def trajectory_reject(grasp_world: Sequence[float]) -> bool:
            xy = np.asarray(grasp_world, dtype=np.float64)[:2]
            expert = ScrewExpert(
                spawn_xy_m=xy,
                table_top_z_m=cfg.table_top_z_m,
                socket_mouth_world_m=cfg.socket_mouth_position_m,
                socket_axis_inward_world=cfg.socket_axis_world,
                contact_distance_m=grasp_plan.contact_radius_m,
                pregrasp_distance_m=grasp_plan.pregrasp_radius_m,
                released_distance_m=(
                    cfg.released_tip_center_distance_m
                ),
                grasp_plane_yaw_rad=grasp_plan.plane.yaw_rad,
                engage_depth_m=cfg.thread_engagement_depth_m,
                final_depth_m=cfg.target_insertion_depth_m,
                pitch_m=cfg.thread_pitch_m,
                target_turns=cfg.target_turns,
                grasp_height_from_tip_m=(
                    cfg.screw_grasp_height_from_tip_m
                ),
                scripted_grasp_plane_world_z_m=(
                    cfg.scripted_grasp_plane_world_z_m
                ),
                scripted_center_xy_m=cfg.scripted_center_xy_m,
            )
            # Every scripted pickup command is checked continuously with the
            # previous IK solution as its branch seed.  Once step 215 reaches
            # the common world-centre handoff, subsequent policy commands no
            # longer depend on the sampled spawn.
            previous_ctrl = np.zeros(ACTUATOR_COUNT, dtype=np.float64)
            command = expert.reset()
            for index in range(SCRIPTED_PREFIX_STEPS + 1):
                targets = command.fingertip_targets(
                    grasp_height_from_tip_m=cfg.screw_grasp_height_from_tip_m
                )
                solution = solver.solve_tip_targets(
                    targets,
                    prev_ctrl=previous_ctrl,
                    strict=False,
                )
                if not solution.success:
                    return True
                previous_ctrl = solution.ctrl
                if index < SCRIPTED_PREFIX_STEPS:
                    command = expert.advance()
            return False

        return sample_spawn_xy(
            rng,
            cfg,
            ik_reachable=reachable,
            reject=trajectory_reject,
        )

    def _cache_ids(self) -> None:
        model = self.model
        self.screw_body_id = _name_id(
            model, mujoco.mjtObj.mjOBJ_BODY, SCREW_BODY_NAME
        )
        self.screw_joint_id = _name_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, SCREW_FREE_JOINT_NAME
        )
        self.screw_qpos_adr = int(model.jnt_qposadr[self.screw_joint_id])
        self.screw_dof_adr = int(model.jnt_dofadr[self.screw_joint_id])
        self.screw_tip_site_id = _name_id(
            model, mujoco.mjtObj.mjOBJ_SITE, SCREW_TIP_SITE_NAME
        )
        self.screw_grasp_site_id = _name_id(
            model, mujoco.mjtObj.mjOBJ_SITE, SCREW_GRASP_SITE_NAME
        )
        self.socket_mouth_site_id = _name_id(
            model, mujoco.mjtObj.mjOBJ_SITE, SOCKET_MOUTH_SITE_NAME
        )
        self.socket_bottom_site_id = _name_id(
            model, mujoco.mjtObj.mjOBJ_SITE, SOCKET_BOTTOM_SITE_NAME
        )
        self.holder_eq_id = _name_id(
            model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            SCREW_START_HOLDER_EQUALITY_NAME,
        )
        self.holder_body_id = _name_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            SCREW_START_HOLDER_BODY_NAME,
        )
        self.holder_mocap_id = int(model.body_mocapid[self.holder_body_id])
        if self.holder_mocap_id < 0:
            raise RuntimeError("screw start holder must be a mocap body")
        self.tip_geom_ids = {
            finger: _name_id(
                model, mujoco.mjtObj.mjOBJ_GEOM, TIP_GEOM_NAMES[finger]
            )
            for finger in FINGERS
        }
        self.tip_body_ids = {
            finger: _name_id(
                model, mujoco.mjtObj.mjOBJ_BODY, TIP_BODY_NAMES[finger]
            )
            for finger in FINGERS
        }
        self.screw_geom_ids = {
            _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, SCREW_SHAFT_GEOM_NAME),
            _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, SCREW_HEAD_GEOM_NAME),
        }
        self.armbase_body_ids = {
            finger: _name_id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"robot0:armbase_{finger}",
            )
            for finger in FINGERS
        }
        self.ctrl_ranges = np.asarray(
            model.actuator_ctrlrange, dtype=np.float64
        ).copy()

    def _make_ik(self) -> None:
        self.ik = GripperIK.from_xml(
            self.scene_path,
            linear_search_step_m=self.settings.ik_linear_search_step_m,
        )
        poses = {
            finger: (
                np.asarray(self.data.xpos[body_id], dtype=np.float64).copy(),
                np.asarray(self.data.xmat[body_id], dtype=np.float64)
                .reshape(3, 3)
                .copy(),
            )
            for finger, body_id in self.armbase_body_ids.items()
        }
        self.ik.set_armbase_world_poses(poses)
        self.teacher = IKTeacher(self.ik)

    def _initialize_gripper(self, command: ScrewCommand) -> None:
        targets = command.fingertip_targets(
            grasp_height_from_tip_m=self.config.screw_grasp_height_from_tip_m
        )
        result = self.teacher.solve_tip_targets(
            targets,
            strict=True,
            update=True,
        )
        self.data.ctrl[:] = result.ctrl
        # Start both active and passive joints exactly on the selected
        # closed-chain branch. This avoids a violent connect-constraint settle.
        for finger, solution in result.finger_solutions.items():
            values = {
                "linear": solution.q_linear,
                "armbase": solution.q_armbase,
                "back": solution.q_back,
                "short": solution.q_short,
                "elbow": solution.q_elbow,
                "front": solution.q_front,
            }
            for kind, value in values.items():
                joint_name = f"robot0:{kind}_joint_{finger}"
                joint_id = _name_id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[qpos_adr] = float(value)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    # Public state accessors used by the omniscient observation
    # ------------------------------------------------------------------

    @property
    def holder_active(self) -> bool:
        return bool(self.data.eq_active[self.holder_eq_id])

    @property
    def insertion_depth_m(self) -> float:
        tip_delta = self.screw_tip_world() - self.socket_mouth_world
        return float(np.dot(tip_delta, self.socket_axis_world))

    @property
    def socket_axis_world(self) -> np.ndarray:
        return np.asarray(self.config.socket_axis_world, dtype=np.float64)

    @property
    def socket_mouth_world(self) -> np.ndarray:
        return np.asarray(
            self.data.site_xpos[self.socket_mouth_site_id], dtype=np.float64
        ).copy()

    @property
    def socket_rotation_world(self) -> np.ndarray:
        # Local socket +Z is the inward hole direction.
        return align_local_z_to(self.socket_axis_world)

    def screw_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.data.xpos[self.screw_body_id], dtype=np.float64).copy(),
            np.asarray(self.data.xquat[self.screw_body_id], dtype=np.float64).copy(),
        )

    def screw_rotation_world(self) -> np.ndarray:
        return (
            np.asarray(self.data.xmat[self.screw_body_id], dtype=np.float64)
            .reshape(3, 3)
            .copy()
        )

    def screw_velocity(self) -> np.ndarray:
        spatial = np.asarray(
            self.data.cvel[self.screw_body_id], dtype=np.float64
        )
        return np.concatenate((spatial[3:6], spatial[0:3])).copy()

    def screw_tip_world(self) -> np.ndarray:
        return np.asarray(
            self.data.site_xpos[self.screw_tip_site_id], dtype=np.float64
        ).copy()

    def diffusion_handoff_errors(self) -> Tuple[float, float]:
        """Measured position/orientation error at the scripted centre stage."""

        target_tip = np.array(
            (
                self.config.scripted_center_xy_m[0],
                self.config.scripted_center_xy_m[1],
                self.config.scripted_center_tip_world_z_m,
            ),
            dtype=np.float64,
        )
        position_error = float(
            np.linalg.norm(self.screw_tip_world() - target_tip)
        )
        upright_error = float(
            np.linalg.norm(
                rotation_log(self.screw_rotation_world().T)
            )
        )
        return position_error, upright_error

    def diffusion_handoff_ready(self) -> bool:
        """Whether measured pickup state is safe to hand to learned control."""

        position_error, upright_error = self.diffusion_handoff_errors()
        grasp_is_supported = (
            self.settings.guide_mode == "mocap"
            or (self.grasp_latched and not self.holder_active)
        )
        return bool(
            grasp_is_supported
            and position_error
            <= self.config.diffusion_handoff_position_tolerance_m
            and upright_error
            <= math.radians(
                self.config.diffusion_handoff_orientation_tolerance_deg
            )
        )

    def world_to_socket(self, point_world: Sequence[float]) -> np.ndarray:
        return self.socket_rotation_world.T @ (
            np.asarray(point_world, dtype=np.float64) - self.socket_mouth_world
        )

    def axis_alignment_error(self) -> np.ndarray:
        actual_rotation = self.screw_rotation_world()
        desired_rotation = align_local_z_to(-self.socket_axis_world)
        error_world = rotation_log(desired_rotation @ actual_rotation.T)
        error_socket = self.socket_rotation_world.T @ error_world
        return np.asarray(error_socket[:2], dtype=np.float64)

    def thread_ready(self) -> bool:
        """Return whether the measured pose can physically enter the thread.

        Depth alone is not sufficient: engaging a tilted or edge-loaded shaft
        would hide exactly the cross-threading/jamming behaviour that the
        residual controller is intended to learn.
        """

        tip_socket = self.world_to_socket(self.screw_tip_world())
        return bool(
            self.insertion_depth_m
            >= (
                self.config.thread_engagement_depth_m
                - self.settings.engage_depth_tolerance_m
            )
            and float(np.linalg.norm(tip_socket[:2]))
            <= self.settings.engage_lateral_tolerance_m
            and float(np.linalg.norm(self.axis_alignment_error()))
            <= self.settings.engage_axis_tolerance_rad
        )

    def helical_errors(self) -> Tuple[float, float]:
        if not self.thread_engaged:
            return 0.0, 0.0
        lead = self.config.thread_pitch_m / (2.0 * math.pi)
        error = (
            self.insertion_depth_m
            - self.config.thread_engagement_depth_m
            - lead * self.twist_unwrapped_rad
        )
        velocity = self.screw_velocity()
        velocity_error = float(
            np.dot(velocity[:3], self.socket_axis_world)
            - lead * np.dot(velocity[3:], self.socket_axis_world)
        )
        return float(error), velocity_error

    def fingertip_position(self, finger: str) -> np.ndarray:
        body_id = self.tip_body_ids[finger.upper()]
        return np.asarray(self.data.xpos[body_id], dtype=np.float64).copy()

    def fingertip_velocity(self, finger: str) -> np.ndarray:
        body_id = self.tip_body_ids[finger.upper()]
        return np.asarray(self.data.cvel[body_id, 3:6], dtype=np.float64).copy()

    def fingertip_contact_forces(self) -> Dict[str, float]:
        forces = {finger: 0.0 for finger in FINGERS}
        geom_to_finger = {
            geom_id: finger for finger, geom_id in self.tip_geom_ids.items()
        }
        wrench = np.zeros(6, dtype=np.float64)
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            finger = None
            if geom1 in geom_to_finger and geom2 in self.screw_geom_ids:
                finger = geom_to_finger[geom1]
            elif geom2 in geom_to_finger and geom1 in self.screw_geom_ids:
                finger = geom_to_finger[geom2]
            if finger is not None:
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                forces[finger] += float(np.linalg.norm(wrench[:3]))
        return forces

    def fingertip_contact_flags(
        self, threshold_n: float = 0.05
    ) -> Dict[str, bool]:
        return {
            finger: force >= float(threshold_n)
            for finger, force in self.fingertip_contact_forces().items()
        }

    def observation(self) -> np.ndarray:
        return self.observation_builder.build()

    # ------------------------------------------------------------------
    # Control and physics
    # ------------------------------------------------------------------

    def release_holder(self) -> None:
        self.data.eq_active[self.holder_eq_id] = 0

    def _rate_limit_ctrl(self, desired: np.ndarray) -> np.ndarray:
        """Track an IK target with bandwidth, acceleration and jerk limits."""

        previous = np.asarray(self.data.ctrl, dtype=np.float64).copy()
        target = np.asarray(desired, dtype=np.float64).copy()
        if target.shape != (ACTUATOR_COUNT,):
            raise ValueError(
                f"IK control must have shape {(ACTUATOR_COUNT,)}, "
                f"got {target.shape}"
            )
        dt = self.control_dt_s
        alpha = 1.0 - math.exp(
            -2.0 * math.pi * self.settings.ik_bandwidth_hz * dt
        )
        target = previous + alpha * (target - previous)
        omega = float(
            np.clip(
                2.0 * math.pi * self.settings.ik_bandwidth_hz,
                2.0,
                12.0,
            )
        )
        limited = previous.copy()
        for actuator_id in range(ACTUATOR_COUNT):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_type = int(self.model.jnt_type[joint_id])
            is_linear = joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE)
            maximum_step = (
                self.settings.linear_rate_limit_m_per_step
                if is_linear
                else self.settings.hinge_rate_limit_rad_per_step
            )
            acceleration_limit = (
                self.settings.linear_acceleration_limit_m_s2
                if is_linear
                else self.settings.hinge_acceleration_limit_rad_s2
            )
            jerk_limit = (
                self.settings.linear_jerk_limit_m_s3
                if is_linear
                else self.settings.hinge_jerk_limit_rad_s3
            )
            previous_velocity = self._ctrl_velocity[actuator_id]
            previous_acceleration = self._ctrl_acceleration[actuator_id]
            error = target[actuator_id] - previous[actuator_id]
            desired_acceleration = (
                omega * omega * error
                - 2.0 * omega * previous_velocity
            )
            acceleration = float(
                np.clip(
                    desired_acceleration,
                    previous_acceleration - jerk_limit * dt,
                    previous_acceleration + jerk_limit * dt,
                )
            )
            acceleration = float(
                np.clip(
                    acceleration,
                    -acceleration_limit,
                    acceleration_limit,
                )
            )
            velocity_limit = maximum_step / dt
            velocity = float(
                np.clip(
                    previous_velocity + acceleration * dt,
                    -velocity_limit,
                    velocity_limit,
                )
            )
            value = previous[actuator_id] + velocity * dt
            # Do not ring across the low-pass target.
            if error != 0.0 and (value - target[actuator_id]) * error > 0.0:
                value = target[actuator_id]
            value = float(
                np.clip(
                    value,
                    self.ctrl_ranges[actuator_id, 0],
                    self.ctrl_ranges[actuator_id, 1],
                )
            )
            realized_velocity = (value - previous[actuator_id]) / dt
            realized_acceleration = (
                realized_velocity - previous_velocity
            ) / dt
            self._ctrl_velocity[actuator_id] = realized_velocity
            self._ctrl_acceleration[actuator_id] = float(
                np.clip(
                    realized_acceleration,
                    -acceleration_limit,
                    acceleration_limit,
                )
            )
            limited[actuator_id] = value
        return limited

    def _desired_screw_body_pose(
        self, command: ScrewCommand
    ) -> Tuple[np.ndarray, np.ndarray]:
        tip_local = np.array(
            [0.0, 0.0, self.config.screw_tip_local_z_m],
            dtype=np.float64,
        )
        rotation = np.asarray(command.rotation_world, dtype=np.float64)
        position = np.asarray(command.tip_world, dtype=np.float64) - rotation @ tip_local
        return position, rotation

    def _apply_guide_wrench(self) -> None:
        if (
            self.settings.guide_mode != "wrench"
            or self.settings.guide_scale <= 0.0
            or self.holder_active
        ):
            return
        desired_position, desired_rotation = self._desired_screw_body_pose(
            self.current_command
        )
        actual_position = np.asarray(
            self.data.xpos[self.screw_body_id], dtype=np.float64
        )
        actual_rotation = self.screw_rotation_world()
        velocity = self.screw_velocity()
        force = (
            self.settings.guide_position_kp * (desired_position - actual_position)
            - self.settings.guide_position_kd * velocity[:3]
        )
        rotation_error = rotation_log(desired_rotation @ actual_rotation.T)
        torque = (
            self.settings.guide_rotation_kp * rotation_error
            - self.settings.guide_rotation_kd * velocity[3:]
        )
        scale = self.settings.guide_scale
        self.data.xfrc_applied[self.screw_body_id, :3] += scale * _clip_norm(
            force, self.settings.guide_force_limit_n
        )
        self.data.xfrc_applied[self.screw_body_id, 3:] += scale * _clip_norm(
            torque, self.settings.guide_torque_limit_nm
        )

    def _apply_grasp_stabilization_wrench(self) -> None:
        """Apply the bounded compliant pose coupling of a latched pinch."""

        if (
            not self.settings.grasp_stabilization
            or not self.grasp_latched
            or self.holder_active
            or self.settings.guide_mode == "mocap"
        ):
            return
        desired_position, desired_rotation = self._desired_screw_body_pose(
            self.current_command
        )
        actual_position = np.asarray(
            self.data.xpos[self.screw_body_id], dtype=np.float64
        )
        actual_rotation = self.screw_rotation_world()
        velocity = self.screw_velocity()
        force = (
            self.settings.grasp_position_kp
            * (desired_position - actual_position)
            - self.settings.grasp_position_kd * velocity[:3]
        )
        rotation_error = rotation_log(
            desired_rotation @ actual_rotation.T
        )
        torque = (
            self.settings.grasp_rotation_kp * rotation_error
            - self.settings.grasp_rotation_kd * velocity[3:]
        )
        self.data.xfrc_applied[self.screw_body_id, :3] += _clip_norm(
            force,
            self.settings.grasp_force_limit_n,
        )
        self.data.xfrc_applied[self.screw_body_id, 3:] += _clip_norm(
            torque,
            self.settings.grasp_torque_limit_nm,
        )

    def _apply_thread_wrench(self) -> None:
        self._last_helical_lambda_n = 0.0
        if not self.thread_engaged:
            return
        axis = self.socket_axis_world
        socket_rotation = self.socket_rotation_world
        velocity = self.screw_velocity()

        # Engaged threads constrain more than lead: they also keep the shaft
        # concentric and approximately coaxial.  Apply bounded compliant
        # radial/alignment wrenches rather than teleporting the pose.
        tip_socket = self.world_to_socket(self.screw_tip_world())
        linear_velocity_socket = socket_rotation.T @ velocity[:3]
        lateral_force_socket = np.array(
            [
                -self.settings.thread_lateral_stiffness_n_per_m
                * tip_socket[0]
                - self.settings.thread_lateral_damping_ns_per_m
                * linear_velocity_socket[0],
                -self.settings.thread_lateral_stiffness_n_per_m
                * tip_socket[1]
                - self.settings.thread_lateral_damping_ns_per_m
                * linear_velocity_socket[1],
                0.0,
            ],
            dtype=np.float64,
        )
        self.data.xfrc_applied[self.screw_body_id, :3] += _clip_norm(
            socket_rotation @ lateral_force_socket,
            self.settings.thread_lateral_force_limit_n,
        )
        alignment_error = self.axis_alignment_error()
        angular_velocity_socket = socket_rotation.T @ velocity[3:]
        alignment_torque_socket = np.array(
            [
                self.settings.thread_alignment_stiffness_nm_per_rad
                * alignment_error[0]
                - self.settings.thread_alignment_damping_nms_per_rad
                * angular_velocity_socket[0],
                self.settings.thread_alignment_stiffness_nm_per_rad
                * alignment_error[1]
                - self.settings.thread_alignment_damping_nms_per_rad
                * angular_velocity_socket[1],
                0.0,
            ],
            dtype=np.float64,
        )
        self.data.xfrc_applied[self.screw_body_id, 3:] += _clip_norm(
            socket_rotation @ alignment_torque_socket,
            self.settings.thread_alignment_torque_limit_nm,
        )

        helical_error, velocity_error = self.helical_errors()
        constraint_force = (
            self.settings.helix_stiffness_n_per_m * helical_error
            + self.settings.helix_damping_ns_per_m * velocity_error
        )
        constraint_force = float(
            np.clip(
                constraint_force,
                -self.settings.helix_force_limit_n,
                self.settings.helix_force_limit_n,
            )
        )
        self._last_helical_lambda_n = constraint_force
        lead = self.config.thread_pitch_m / (2.0 * math.pi)
        self.data.xfrc_applied[self.screw_body_id, :3] += -constraint_force * axis
        self.data.xfrc_applied[self.screw_body_id, 3:] += (
            lead * constraint_force * axis
        )

        progress = float(
            np.clip(
                self.twist_unwrapped_rad / max(self.target_twist_rad, 1e-9),
                0.0,
                1.0,
            )
        )
        self.thread_resistance_nm = (
            self.config.rotation_resistance_min_nm
            + progress
            * (
                self.config.rotation_resistance_max_nm
                - self.config.rotation_resistance_min_nm
            )
        )
        angular_speed = float(np.dot(velocity[3:], axis))
        resistance = -self.thread_resistance_nm * math.tanh(
            angular_speed
            / max(self.settings.resistance_velocity_scale_rad_s, 1e-6)
        )
        self.data.xfrc_applied[self.screw_body_id, 3:] += resistance * axis
        # Positive inward preload emulates the operator pressing the screw
        # into the thread while tightening.
        self.data.xfrc_applied[self.screw_body_id, :3] += (
            self.axial_preload_n * axis
        )

    def _physics_substep(self) -> None:
        self.data.xfrc_applied[:] = 0.0
        self._apply_grasp_stabilization_wrench()
        self._apply_guide_wrench()
        self._apply_thread_wrench()
        mujoco.mj_step(self.model, self.data)

    def _update_mocap_guide(self) -> None:
        desired_position, desired_rotation = self._desired_screw_body_pose(
            self.current_command
        )
        self.data.mocap_pos[self.holder_mocap_id] = desired_position
        self.data.mocap_quat[self.holder_mocap_id] = (
            matrix_to_quaternion_wxyz(desired_rotation)
        )
        self.data.eq_active[self.holder_eq_id] = 1

    def _update_task_state(self, *, reset_twist: bool = False) -> None:
        rotation = self.screw_rotation_world()
        marker_world = rotation[:, 0]
        if self.thread_engaged:
            self.twist_unwrapped_rad = self._twist_tracker.update(
                marker_world,
                reset=reset_twist,
            )
            self.marker_mod_angle_rad = self._twist_tracker.modular_angle(
                marker_world
            )
        else:
            self.marker_mod_angle_rad = 0.0
            self.twist_unwrapped_rad = 0.0

    def apply_command(self, command: ScrewCommand) -> StepDiagnostics:
        """Solve IK, apply one 20 ms command and return measured diagnostics."""

        validate_contact_schedule(
            command,
            contact_distance_m=self.expert.contact_distance_m,
        )
        previous_engaged = self.thread_engaged
        self.current_command = command.copy()
        if self.settings.guide_mode == "mocap":
            self._update_mocap_guide()
        targets = command.fingertip_targets(
            grasp_height_from_tip_m=self.config.screw_grasp_height_from_tip_m
        )
        result = self.teacher.solve_tip_targets(
            targets,
            strict=self.settings.strict_ik,
            update=False,
        )
        if result.success:
            ctrl = self._rate_limit_ctrl(result.ctrl)
            self.data.ctrl[:] = ctrl
            self.teacher.commit(result, ctrl=ctrl)
            self._last_certified_ik_command = command.copy()
        else:
            self.ik_failure_count += 1
            self._ctrl_velocity.fill(0.0)
            self._ctrl_acceleration.fill(0.0)

        contact_flags = self.fingertip_contact_flags(threshold_n=0.10)
        stable_pair = (
            contact_flags["U"] and contact_flags["D"]
        ) or (
            contact_flags["L"] and contact_flags["R"]
        )
        if (
            self.settings.guide_mode != "mocap"
            and
            self.holder_active
            and stable_pair
            and command.phase
            not in (TaskPhase.PREGRASP, TaskPhase.CLOSE)
        ):
            self.release_holder()
            self.grasp_latched = True
        elif (
            self.settings.guide_mode != "mocap"
            and not self.holder_active
            and stable_pair
        ):
            self.grasp_latched = True

        if command.engaged and self.thread_ready():
            self.thread_engaged = True

        for _ in range(self.physics_steps_per_control):
            self._physics_substep()
        self.control_step += 1
        self._update_task_state(
            reset_twist=self.thread_engaged and not previous_engaged
        )

        tip_socket = self.world_to_socket(self.screw_tip_world())
        lateral_error = float(np.linalg.norm(tip_socket[:2]))
        axis_error = float(np.linalg.norm(self.axis_alignment_error()))
        helical_error, _ = self.helical_errors()
        return StepDiagnostics(
            ik_success=bool(result.success),
            ik_max_tip_error_m=float(result.max_tip_error),
            holder_active=self.holder_active,
            engaged=self.thread_engaged,
            insertion_depth_m=self.insertion_depth_m,
            twist_rad=self.twist_unwrapped_rad,
            lateral_error_m=lateral_error,
            axis_error_rad=axis_error,
            helical_error_m=helical_error,
            contact_count=sum(self.fingertip_contact_flags().values()),
        )

    def apply_ik_safety_hold(self) -> StepDiagnostics:
        """Advance one control frame without issuing an uncertified IK target.

        This is the final containment path if even the remembered certified
        command can no longer be re-solved.  Existing actuator controls are
        held for one frame so Gym can return a normal terminal transition
        instead of raising from a worker process.
        """

        self.ik_failure_count += 1
        self._ctrl_velocity.fill(0.0)
        self._ctrl_acceleration.fill(0.0)
        for _ in range(self.physics_steps_per_control):
            self._physics_substep()
        self.control_step += 1
        self._update_task_state()

        tip_socket = self.world_to_socket(self.screw_tip_world())
        lateral_error = float(np.linalg.norm(tip_socket[:2]))
        axis_error = float(np.linalg.norm(self.axis_alignment_error()))
        helical_error, _ = self.helical_errors()
        return StepDiagnostics(
            ik_success=False,
            ik_max_tip_error_m=float("inf"),
            holder_active=self.holder_active,
            engaged=self.thread_engaged,
            insertion_depth_m=self.insertion_depth_m,
            twist_rad=self.twist_unwrapped_rad,
            lateral_error_m=lateral_error,
            axis_error_rad=axis_error,
            helical_error_m=helical_error,
            contact_count=sum(self.fingertip_contact_flags().values()),
        )

    def project_command_to_ik(
        self,
        safe_command: ScrewCommand,
        candidate_command: ScrewCommand,
        *,
        iterations: int = 20,
    ) -> Tuple[ScrewCommand, float]:
        """Project a learned task-space command along a certified IK segment.

        This mirrors the safe-anchor convex projection used by Gripper3's
        FLUID controller.  The deterministic supervisor is preferred as the
        safe endpoint.  If it has advanced beyond the physical/policy state
        and is no longer reachable, the last command that actually passed IK
        is used instead.  If no certified anchor can be re-solved, a safety
        stop is latched for the environment rather than raising an exception.
        """

        self.last_ik_projection_used_fallback = False
        self.ik_safety_stop = False
        self.ik_safety_reason = ""

        def solve(command: ScrewCommand):
            return self.teacher.solve_tip_targets(
                command.fingertip_targets(
                    grasp_height_from_tip_m=(
                        self.config.screw_grasp_height_from_tip_m
                    )
                ),
                strict=False,
                update=False,
            )

        candidate_solution = solve(candidate_command)
        if candidate_solution.success:
            return candidate_command.copy(), 1.0
        safe_solution = solve(safe_command)
        if not safe_solution.success:
            certified = self._last_certified_ik_command.copy()
            certified_solution = solve(certified)
            if certified_solution.success:
                self.last_ik_projection_used_fallback = True
                self.ik_safety_reason = "supervisor_anchor_unreachable"
                self.ik_anchor_fallback_count += 1
                return certified, 0.0
            self.ik_safety_stop = True
            self.ik_safety_reason = "certified_ik_anchor_lost"
            self.ik_safety_stop_count += 1
            # The environment sees ``ik_safety_stop`` before applying this
            # value and performs ``apply_ik_safety_hold``.  Returning a copy
            # preserves the two-value API and keeps callers exception-free.
            return certified, 0.0

        safe_q = safe_command.quaternion_wxyz
        candidate_q = candidate_command.quaternion_wxyz

        def interpolate(fraction: float) -> ScrewCommand:
            result = safe_command.copy()
            result.tip_world = (
                (1.0 - fraction)
                * np.asarray(safe_command.tip_world, dtype=np.float64)
                + fraction
                * np.asarray(candidate_command.tip_world, dtype=np.float64)
            )
            quaternion = quaternion_slerp_wxyz(
                safe_q, candidate_q, fraction
            )
            result.rotation_world = quaternion_to_matrix_wxyz(quaternion)
            result.twist_rad = (
                (1.0 - fraction) * float(safe_command.twist_rad)
                + fraction * float(candidate_command.twist_rad)
            )
            result.material_angles_rad = {
                finger: (
                    (1.0 - fraction)
                    * float(safe_command.material_angles_rad[finger])
                    + fraction
                    * float(candidate_command.material_angles_rad[finger])
                )
                for finger in FINGERS
            }
            result.tip_center_distances_m = {
                finger: (
                    (1.0 - fraction)
                    * float(safe_command.tip_center_distances_m[finger])
                    + fraction
                    * float(candidate_command.tip_center_distances_m[finger])
                )
                for finger in FINGERS
            }
            result.phase = candidate_command.phase
            result.engaged = candidate_command.engaged
            return result

        low, high = 0.0, 1.0
        chosen = safe_command.copy()
        for _ in range(max(1, int(iterations))):
            midpoint = 0.5 * (low + high)
            command = interpolate(midpoint)
            if solve(command).success:
                low = midpoint
                chosen = command
            else:
                high = midpoint
        return chosen, float(low)

    def step_expert(self) -> StepDiagnostics:
        return self.apply_command(self.expert.advance())

    # ------------------------------------------------------------------
    # Evaluation, rendering and controlled test helpers
    # ------------------------------------------------------------------

    def success_metrics(self) -> Dict[str, float | bool]:
        tip_socket = self.world_to_socket(self.screw_tip_world())
        velocity = self.screw_velocity()
        depth_error = abs(self.insertion_depth_m - self.target_depth_m)
        twist_error = abs(self.twist_unwrapped_rad - self.target_twist_rad)
        lateral_error = float(np.linalg.norm(tip_socket[:2]))
        axis_error = float(np.linalg.norm(self.axis_alignment_error()))
        speed = float(np.linalg.norm(velocity))
        success = bool(
            depth_error <= 0.002
            and twist_error <= math.radians(8.0)
            and lateral_error <= 0.003
            and axis_error <= math.radians(3.0)
            and speed <= 0.35
        )
        return {
            "success": success,
            "depth_m": self.insertion_depth_m,
            "depth_error_m": depth_error,
            "twist_rad": self.twist_unwrapped_rad,
            "turns": self.twist_unwrapped_rad / (2.0 * math.pi),
            "twist_error_rad": twist_error,
            "lateral_error_m": lateral_error,
            "axis_error_rad": axis_error,
            "speed": speed,
            "ik_failures": float(self.ik_failure_count),
            "ik_anchor_fallbacks": float(
                self.ik_anchor_fallback_count
            ),
            "ik_safety_stops": float(self.ik_safety_stop_count),
        }

    def set_screw_pose(
        self,
        position_world: Sequence[float],
        quaternion_wxyz: Sequence[float],
        *,
        zero_velocity: bool = True,
    ) -> None:
        """Explicit pose setter reserved for reset/unit tests."""

        qpos = self.screw_qpos_adr
        self.data.qpos[qpos : qpos + 3] = np.asarray(position_world, dtype=np.float64)
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
        self.data.qpos[qpos + 3 : qpos + 7] = quaternion
        if zero_velocity:
            dof = self.screw_dof_adr
            self.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._update_task_state(reset_twist=True)

    def render(
        self,
        *,
        width: int = 960,
        height: int = 720,
        camera: str = "camera_side",
    ) -> np.ndarray:
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(self.data, camera=camera)
            return np.asarray(renderer.render()).copy()
        finally:
            renderer.close()

    def close(self) -> None:
        # MjModel/MjData are reclaimed by Python. Runtime XMLs are intentionally
        # kept inside .runtime for reproducibility and safe multiprocessing.
        return None


__all__ = [
    "SimulationSettings",
    "ScrewSimulator",
    "StepDiagnostics",
]
