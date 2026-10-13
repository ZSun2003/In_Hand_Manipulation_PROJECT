#!/usr/bin/env python3
"""Gymnasium environment for Diffusion + IK + residual screw driving."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from policy_runtime import SafeDiffusionCommandAdapter
from residual_control import (
    RESIDUAL_ACTION_DIM,
    RESIDUAL_ACTION_NAMES,
    ResidualCommandProjector,
    residual_action_mask,
)
from screw_config import DEFAULT_CONFIG, ScrewTaskConfig
from screw_control import (
    ACTION_DIM,
    SOCKET_YAW_EQUIVARIANCE_VERSION,
    ScrewCommand,
    TaskPhase,
    rotation_log,
)
from screw_scene import sample_socket_mouth_xy
from screw_sim import SimulationSettings, ScrewSimulator, StepDiagnostics


class ScrewTaskEnv(gym.Env[np.ndarray, np.ndarray]):
    """Full-state residual environment.

    ``baseline="expert"`` is useful for residual pretraining and ablations.
    ``baseline="diffusion"`` loads a frozen diffusion checkpoint and is the
    intended final stack.  In both cases the residual policy sees every
    physical state supplied by :class:`OmniscientObservationBuilder`.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        config: ScrewTaskConfig = DEFAULT_CONFIG,
        baseline: str = "expert",
        diffusion_checkpoint: Optional[str | Path] = None,
        random_spawn: bool = True,
        random_socket_position: bool = True,
        random_socket_angle: bool = False,
        socket_angle_min_deg: float = 45.0,
        socket_angle_max_deg: float = 45.0,
        guide_mode: str = "none",
        guide_scale: float = 0.0,
        strict_ik: bool = False,
        max_steps: int = 3_000,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if baseline not in ("expert", "diffusion", "external_diffusion"):
            raise ValueError(
                "baseline must be 'expert', 'diffusion', or "
                "'external_diffusion'"
            )
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if baseline == "diffusion" and diffusion_checkpoint is None:
            raise ValueError("diffusion baseline requires a checkpoint")
        self.base_config = config
        self.config = config
        self.baseline = baseline
        self.diffusion_checkpoint = (
            None
            if diffusion_checkpoint is None
            else Path(diffusion_checkpoint).expanduser().resolve()
        )
        self.random_spawn = bool(random_spawn)
        self.random_socket_position = bool(random_socket_position)
        self.random_socket_angle = bool(random_socket_angle)
        self.socket_angle_min_deg = float(socket_angle_min_deg)
        self.socket_angle_max_deg = float(socket_angle_max_deg)
        if self.random_socket_angle:
            raise ValueError(
                "socket inclination is fixed at 45 degrees; "
                "random_socket_angle must be False"
            )
        if not (
            math.isclose(self.socket_angle_min_deg, 45.0, abs_tol=1.0e-9)
            and math.isclose(
                self.socket_angle_max_deg, 45.0, abs_tol=1.0e-9
            )
        ):
            raise ValueError(
                "socket angle bounds must both equal the fixed 45 degrees"
            )
        self.guide_mode = guide_mode
        self.guide_scale = float(guide_scale)
        self.strict_ik = bool(strict_ik)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.simulator: Optional[ScrewSimulator] = None
        self.diffusion_controller = None
        self.diffusion_adapter: Optional[SafeDiffusionCommandAdapter] = None
        self.residual_projector: Optional[ResidualCommandProjector] = None
        self._previous_depth = 0.0
        self._previous_twist = 0.0
        self._diffusion_active = False
        self._handoff_hold_steps = 0

        self._create_simulator(
            self.config.default_screw_xy_m,
            seed=seed,
        )
        assert self.simulator is not None
        observation = self.simulator.observation()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation.shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.concatenate(
                (
                    np.full(RESIDUAL_ACTION_DIM, -1.0, dtype=np.float32),
                    (
                        np.full(ACTION_DIM, -np.inf, dtype=np.float32)
                        if self.baseline == "external_diffusion"
                        else np.empty(0, dtype=np.float32)
                    ),
                )
            ),
            high=np.concatenate(
                (
                    np.full(RESIDUAL_ACTION_DIM, 1.0, dtype=np.float32),
                    (
                        np.full(ACTION_DIM, np.inf, dtype=np.float32)
                        if self.baseline == "external_diffusion"
                        else np.empty(0, dtype=np.float32)
                    ),
                )
            ),
            dtype=np.float32,
        )
        if self.baseline == "diffusion":
            self._load_diffusion()

    def _settings(self) -> SimulationSettings:
        return SimulationSettings(
            control_dt_s=0.020,
            max_control_steps=self.max_steps,
            strict_ik=self.strict_ik,
            guide_mode=self.guide_mode,
            guide_scale=self.guide_scale,
        )

    def _create_simulator(
        self,
        spawn_xy: Sequence[float],
        *,
        seed: Optional[int],
        config: Optional[ScrewTaskConfig] = None,
    ) -> None:
        if self.simulator is not None:
            self.simulator.close()
        if config is not None:
            self.config = config
        self.simulator = ScrewSimulator(
            config=self.config,
            settings=self._settings(),
            screw_xy_m=spawn_xy,
            seed=seed,
        )
        self.residual_projector = ResidualCommandProjector(
            socket_rotation_world=self.simulator.socket_rotation_world,
            pitch_m=self.config.thread_pitch_m,
        )
        self.diffusion_adapter = SafeDiffusionCommandAdapter.from_expert(
            self.simulator.expert
        )

    def _load_diffusion(self) -> None:
        from diffusion_policy import DiffusionController

        assert self.diffusion_checkpoint is not None
        self.diffusion_controller = DiffusionController.from_checkpoint(
            self.diffusion_checkpoint,
            device="auto",
        )

    @property
    def sim(self) -> ScrewSimulator:
        if self.simulator is None:
            raise RuntimeError("environment is closed")
        return self.simulator

    def _select_spawn(
        self,
        options: Optional[Dict[str, Any]],
        config: ScrewTaskConfig,
    ) -> Tuple[float, float]:
        if options and "spawn_xy" in options:
            value = np.asarray(options["spawn_xy"], dtype=np.float64).reshape(2)
            return float(value[0]), float(value[1])
        if not self.random_spawn:
            return config.default_screw_xy_m
        sample_seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
        return ScrewSimulator.sample_reachable_spawn(
            config=config,
            seed=sample_seed,
        )

    def _select_socket_angle(
        self,
        options: Optional[Dict[str, Any]],
    ) -> float:
        value = (
            float(options["socket_angle_deg"])
            if options and "socket_angle_deg" in options
            else float(self.base_config.socket_table_angle_deg)
        )
        if not math.isclose(value, 45.0, abs_tol=1.0e-9):
            raise ValueError("socket_angle_deg is fixed at 45 degrees")
        return 45.0

    def _select_socket_mouth(
        self,
        options: Optional[Dict[str, Any]],
    ) -> Tuple[float, float]:
        if options and "socket_mouth_xy" in options:
            value = np.asarray(
                options["socket_mouth_xy"], dtype=np.float64
            ).reshape(2)
            mouth = (float(value[0]), float(value[1]))
        elif not self.random_socket_position:
            mouth = self.base_config.socket_mouth_xy_m
        else:
            mouth = sample_socket_mouth_xy(
                self._rng, self.base_config
            )
        radius = math.hypot(*mouth)
        if not math.isclose(
            radius,
            self.base_config.socket_mouth_orbit_radius_m,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "socket_mouth_xy must lie on the configured orbit"
            )
        return mouth

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
        socket_angle_deg = self._select_socket_angle(options)
        socket_mouth_xy = self._select_socket_mouth(options)
        episode_config = self.base_config.with_socket_pose(
            socket_mouth_xy, socket_angle_deg
        )
        spawn = self._select_spawn(options, episode_config)
        episode_seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
        self._create_simulator(
            spawn, seed=episode_seed, config=episode_config
        )
        if self.baseline == "diffusion":
            # Keep the loaded network, reset only its observation history.
            if self.diffusion_controller is None:
                self._load_diffusion()
            if hasattr(self.diffusion_controller, "reset"):
                self.diffusion_controller.reset()
        self._diffusion_active = False
        self._handoff_hold_steps = 0
        self._previous_depth = self.sim.insertion_depth_m
        self._previous_twist = self.sim.twist_unwrapped_rad
        observation = self.sim.observation()
        return observation, {
            "spawn_xy": np.asarray(spawn, dtype=np.float32),
            "socket_mouth_xy": np.asarray(
                socket_mouth_xy, dtype=np.float32
            ),
            "socket_opening_azimuth_xy": np.asarray(
                episode_config.socket_opening_azimuth_xy,
                dtype=np.float32,
            ),
            "socket_angle_deg": socket_angle_deg,
            "socket_yaw_quadrant": self.sim.expert.socket_quadrant,
            "socket_yaw_residual_deg": math.degrees(
                self.sim.expert.socket_residual_yaw_rad
            ),
            "socket_yaw_roll_deg": math.degrees(
                self.sim.expert.socket_roll_rad
            ),
            "socket_yaw_equivariance_version": (
                SOCKET_YAW_EQUIVARIANCE_VERSION
            ),
            "observation_schema": self.sim.observation_builder.SCHEMA,
        }

    def _baseline_command(
        self,
        observation: np.ndarray,
        external_diffusion_action: Optional[np.ndarray] = None,
    ) -> Tuple[ScrewCommand, ScrewCommand]:
        current = self.sim.expert.current
        waiting_for_handoff = bool(
            self.sim.expert.step_index
            == self.sim.expert.diffusion_handoff_step
            and not self.sim.diffusion_handoff_ready()
        )
        if waiting_for_handoff:
            self._handoff_hold_steps += 1
        insert_target = (
            self.sim.socket_mouth_world
            + self.config.thread_engagement_depth_m
            * self.sim.socket_axis_world
        )
        # Once the deterministic schedule has reached its insertion target,
        # keep the pose in the residual-controllable INSERT phase until the
        # *measured* screw pose is deep, centred and aligned.  Without this
        # feedback gate, a contact jam could let the open-loop supervisor
        # continue into regrasp and tightening even though no thread engaged.
        waiting_for_thread_pose = bool(
            current.phase == TaskPhase.INSERT
            and np.linalg.norm(current.tip_world - insert_target) <= 1e-9
            and not self.sim.thread_ready()
        )
        supervisor = (
            current.copy()
            if waiting_for_handoff or waiting_for_thread_pose
            else self.sim.expert.advance()
        )
        diffusion_active = self.sim.expert.learned_control_active
        rising_handoff = diffusion_active and not self._diffusion_active
        self._diffusion_active = diffusion_active
        if (
            supervisor.phase == TaskPhase.INSERT
            and not self.sim.thread_engaged
        ):
            scheduled_socket = self.sim.world_to_socket(
                supervisor.tip_world
            )
            capped_depth = min(
                float(scheduled_socket[2]),
                self.sim.insertion_depth_m
                + self.sim.settings.insertion_command_lead_m,
            )
            supervisor.tip_world = (
                np.asarray(supervisor.tip_world, dtype=np.float64)
                + (capped_depth - float(scheduled_socket[2]))
                * self.sim.socket_axis_world
            )
        if self.baseline == "expert":
            return supervisor, supervisor
        # Pickup, lifting and the centre transfer are purely scripted.  The
        # learned baseline receives control only after measured handoff
        # readiness advances the schedule beyond the centre-stage index.
        if not diffusion_active:
            return supervisor, supervisor
        assert self.diffusion_adapter is not None
        if self.baseline == "external_diffusion":
            if external_diffusion_action is None:
                raise ValueError(
                    "external_diffusion baseline requires a 14-D policy action"
                )
            policy_action = external_diffusion_action
        else:
            assert self.diffusion_controller is not None
            if rising_handoff and hasattr(
                self.diffusion_controller, "reset"
            ):
                self.diffusion_controller.reset(observation)
            policy_action = self.diffusion_controller.act(observation)
        return (
            self.diffusion_adapter.adapt(
                policy_action,
                previous_command=self.sim.current_command,
                supervisor_command=supervisor,
            ),
            supervisor,
        )

    def _reward(
        self,
        diagnostics: StepDiagnostics,
        residual: np.ndarray,
    ) -> float:
        depth = float(diagnostics.insertion_depth_m)
        twist = float(diagnostics.twist_rad)
        depth_progress = (depth - self._previous_depth) / max(
            self.config.socket_hole_depth_m, 1e-9
        )
        twist_progress = (twist - self._previous_twist) / max(
            self.config.target_rotation_rad, 1e-9
        )
        self._previous_depth = depth
        self._previous_twist = twist

        command = self.sim.current_command
        position_error = float(
            np.linalg.norm(self.sim.screw_tip_world() - command.tip_world)
        )
        rotation_error = float(
            np.linalg.norm(
                rotation_log(
                    np.asarray(command.rotation_world)
                    @ self.sim.screw_rotation_world().T
                )
            )
        )
        helical_error, _ = self.sim.helical_errors()
        command_depth = float(
            self.sim.world_to_socket(command.tip_world)[2]
        )
        alignment_active = bool(
            command.engaged or command_depth >= -0.010
        )
        alignment_penalty = 0.0
        if alignment_active:
            # Scale state costs by the 20 ms control interval.  Otherwise the
            # 2,000+ step contact schedule overwhelms all telescoping depth
            # and rotation progress, even for a successful expert episode.
            alignment_penalty = self.sim.control_dt_s * (
                2.0 * diagnostics.lateral_error_m / 0.005
                + 0.5 * diagnostics.axis_error_rad / math.radians(5.0)
            )
        reward = (
            80.0 * depth_progress
            + 60.0 * twist_progress
            - alignment_penalty
            - 0.02 * abs(helical_error) / 0.002
            - 0.01 * position_error / 0.005
            - 0.005 * rotation_error / math.radians(5.0)
            - 0.001 * float(np.dot(residual, residual))
        )
        if self.residual_projector is not None:
            reward -= 0.25 * (
                1.0
                - self.residual_projector.last_ik_projection_fraction
            )
        if not diagnostics.ik_success:
            reward -= 5.0
        return float(reward)

    def _failure(self, diagnostics: StepDiagnostics) -> Tuple[bool, str]:
        position, _ = self.sim.screw_pose()
        if self.sim.ik_safety_stop:
            return True, self.sim.ik_safety_reason or "ik_safety_stop"
        if not np.all(np.isfinite(self.sim.data.qpos)):
            return True, "non_finite_state"
        if position[2] < self.config.table_top_z_m - 0.12:
            return True, "dropped"
        if diagnostics.lateral_error_m > 0.16:
            return True, "escaped_socket_workspace"
        if self.strict_ik and not diagnostics.ik_success:
            return True, "ik_unreachable"
        return False, ""

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        combined_action = np.asarray(action, dtype=np.float32)
        if combined_action.shape != self.action_space.shape:
            raise ValueError(
                f"action must have shape {self.action_space.shape}, "
                f"got {combined_action.shape}"
            )
        if not np.all(np.isfinite(combined_action)):
            raise ValueError("action contains NaN or infinity")
        residual = combined_action[:RESIDUAL_ACTION_DIM]
        external_diffusion_action = (
            combined_action[RESIDUAL_ACTION_DIM:]
            if self.baseline == "external_diffusion"
            else None
        )
        before = self.sim.observation()
        baseline, safe_anchor = self._baseline_command(
            before,
            external_diffusion_action=external_diffusion_action,
        )
        residual = residual * residual_action_mask(
            baseline.phase, baseline.engaged
        )
        if not self._diffusion_active:
            residual = np.zeros_like(residual)
        assert self.residual_projector is not None
        candidate = self.residual_projector.project(
            baseline,
            residual,
            previous_command=self.sim.current_command,
        )
        command, feasible_fraction = self.sim.project_command_to_ik(
            safe_anchor,
            candidate,
        )
        self.residual_projector.accept_ik_projection(feasible_fraction)
        if self.sim.ik_safety_stop:
            diagnostics = self.sim.apply_ik_safety_hold()
        else:
            diagnostics = self.sim.apply_command(command)
        observation = self.sim.observation()
        metrics = self.sim.success_metrics()
        reward = self._reward(diagnostics, residual)
        success = bool(metrics["success"]) and (
            self.sim.expert.complete
            or command.phase in (TaskPhase.FINAL_HOLD, TaskPhase.COMPLETE)
        )
        failure, failure_reason = self._failure(diagnostics)
        terminated = bool(success or failure)
        truncated = bool(
            not terminated and self.sim.control_step >= self.max_steps
        )
        if success:
            reward += 100.0
        elif failure:
            reward -= 50.0
        info: Dict[str, Any] = {
            **metrics,
            "phase": command.phase.name,
            "phase_id": int(command.phase),
            "ik_success": diagnostics.ik_success,
            "holder_active": diagnostics.holder_active,
            "grasp_latched": self.sim.grasp_latched,
            "thread_engaged": diagnostics.engaged,
            "contact_count": diagnostics.contact_count,
            "socket_angle_deg": float(
                self.config.socket_table_angle_deg
            ),
            "socket_mouth_xy": np.asarray(
                self.config.socket_mouth_xy_m, dtype=np.float32
            ),
            "socket_opening_azimuth_xy": np.asarray(
                self.config.socket_opening_azimuth_xy, dtype=np.float32
            ),
            "socket_yaw_quadrant": self.sim.expert.socket_quadrant,
            "socket_yaw_residual_deg": math.degrees(
                self.sim.expert.socket_residual_yaw_rad
            ),
            "socket_yaw_roll_deg": math.degrees(
                self.sim.expert.socket_roll_rad
            ),
            "socket_yaw_equivariance_version": (
                SOCKET_YAW_EQUIVARIANCE_VERSION
            ),
            "residual_action_names": RESIDUAL_ACTION_NAMES,
            "ik_feasible_fraction": feasible_fraction,
            "ik_anchor_fallback": (
                self.sim.last_ik_projection_used_fallback
            ),
            "ik_safety_stop": self.sim.ik_safety_stop,
            "ik_safety_reason": self.sim.ik_safety_reason,
            "external_diffusion": self.baseline == "external_diffusion",
            "diffusion_policy_loaded": self.diffusion_controller is not None,
            "diffusion_active": self._diffusion_active,
            "scripted_prefix_active": not self._diffusion_active,
            "diffusion_handoff_ready": (
                self.sim.diffusion_handoff_ready()
            ),
            "diffusion_handoff_step": (
                self.sim.expert.diffusion_handoff_step
            ),
            "expert_step": self.sim.expert.step_index,
            "diffusion_handoff_position_error_m": (
                self.sim.diffusion_handoff_errors()[0]
            ),
            "diffusion_handoff_orientation_error_deg": math.degrees(
                self.sim.diffusion_handoff_errors()[1]
            ),
            "diffusion_handoff_hold_steps": self._handoff_hold_steps,
            "failure_reason": failure_reason,
            "TimeLimit.truncated": truncated,
        }
        return observation, float(reward), terminated, truncated, info

    def diffusion_worker_status(self) -> Dict[str, Any]:
        """Small subprocess-safe audit used by the central-policy smoke test."""

        return {
            "baseline": self.baseline,
            "action_shape": tuple(self.action_space.shape),
            "diffusion_policy_loaded": self.diffusion_controller is not None,
            "diffusion_policy_module_loaded": "diffusion_policy" in sys.modules,
            "external_diffusion": self.baseline == "external_diffusion",
            "diffusion_active": self._diffusion_active,
        }

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        return self.sim.render()

    def close(self) -> None:
        if self.simulator is not None:
            self.simulator.close()
            self.simulator = None


def make_curriculum_env(
    *,
    curriculum_fraction: float,
    **kwargs: Any,
) -> ScrewTaskEnv:
    """Convenience factory: mocap expert -> wrench guide -> no guide."""

    fraction = float(np.clip(curriculum_fraction, 0.0, 1.0))
    if fraction < 0.25:
        guide_mode, guide_scale = "mocap", 1.0
    elif fraction < 0.75:
        guide_mode = "wrench"
        guide_scale = 1.0 - (fraction - 0.25) / 0.50
    else:
        guide_mode, guide_scale = "none", 0.0
    base_config = kwargs.pop("config", DEFAULT_CONFIG)
    return ScrewTaskEnv(
        config=base_config,
        random_socket_angle=False,
        socket_angle_min_deg=45.0,
        socket_angle_max_deg=45.0,
        guide_mode=guide_mode,
        guide_scale=guide_scale,
        **kwargs,
    )


__all__ = ["ScrewTaskEnv", "make_curriculum_env"]
