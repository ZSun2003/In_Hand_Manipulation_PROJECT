#!/usr/bin/env python3
"""Omniscient state observation used by Diffusion and residual learning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from screw_config import OBSERVATION_SCHEMA
from screw_control import FINGERS, PHASE_COUNT, ScrewCommand, TaskPhase


def wrap_to_pi(value: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(value) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class TwistTracker:
    """Unwrap the coloured marker orientation around the socket axis."""

    reference_marker_world: np.ndarray
    outward_axis_world: np.ndarray
    previous_mod_rad: float = 0.0
    unwrapped_rad: float = 0.0
    initialized: bool = False

    def __post_init__(self) -> None:
        self.outward_axis_world = self._normalize(self.outward_axis_world)
        marker = np.asarray(self.reference_marker_world, dtype=np.float64).reshape(3)
        marker -= self.outward_axis_world * float(
            np.dot(marker, self.outward_axis_world)
        )
        self.reference_marker_world = self._normalize(marker)

    @staticmethod
    def _normalize(vector: Sequence[float]) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(value))
        if norm <= 1e-12:
            raise ValueError("zero vector")
        return value / norm

    def modular_angle(self, marker_world: Sequence[float]) -> float:
        marker = np.asarray(marker_world, dtype=np.float64).reshape(3)
        marker -= self.outward_axis_world * float(
            np.dot(marker, self.outward_axis_world)
        )
        marker = self._normalize(marker)
        signed_ccw = math.atan2(
            float(
                np.dot(
                    self.outward_axis_world,
                    np.cross(self.reference_marker_world, marker),
                )
            ),
            float(np.dot(self.reference_marker_world, marker)),
        )
        # Positive task progress is clockwise when viewed from the head.
        return float(wrap_to_pi(-signed_ccw))

    def update(
        self,
        marker_world: Sequence[float],
        *,
        reset: bool = False,
    ) -> float:
        modular = self.modular_angle(marker_world)
        if reset or not self.initialized:
            self.previous_mod_rad = modular
            self.unwrapped_rad = 0.0
            self.initialized = True
            return self.unwrapped_rad
        increment = float(wrap_to_pi(modular - self.previous_mod_rad))
        self.unwrapped_rad += increment
        self.previous_mod_rad = modular
        return self.unwrapped_rad


class OmniscientObservationBuilder:
    """Build a fixed-width normalized vector from a running simulator.

    The simulator object is intentionally duck-typed to avoid coupling the
    observation schema to one environment wrapper.  It must expose the fields
    used below (model/data IDs, task geometry and current command).
    """

    SCHEMA = OBSERVATION_SCHEMA

    def __init__(self, simulator: object):
        self.simulator = simulator
        self.names: Tuple[str, ...] = self._field_names()

    @property
    def size(self) -> int:
        return len(self.names)

    def _field_names(self) -> Tuple[str, ...]:
        sim = self.simulator
        names: list[str] = []
        names.extend(f"qpos_{index}" for index in range(int(sim.model.nq)))
        names.extend(f"qvel_{index}" for index in range(int(sim.model.nv)))
        names.extend(f"ctrl_{index}" for index in range(int(sim.model.nu)))
        names.extend(f"actuator_force_{index}" for index in range(int(sim.model.nu)))
        names.extend(
            (
                "screw_x",
                "screw_y",
                "screw_z",
                "screw_qw",
                "screw_qx",
                "screw_qy",
                "screw_qz",
                "screw_vx",
                "screw_vy",
                "screw_vz",
                "screw_wx",
                "screw_wy",
                "screw_wz",
                "tip_socket_x",
                "tip_socket_y",
                "depth",
                "axis_error_x",
                "axis_error_y",
                "marker_sin",
                "marker_cos",
                "twist_unwrapped",
                "turn_count",
                "helical_error",
                "helical_velocity_error",
                "thread_resistance",
                "axial_preload",
                "socket_angle_sin",
                "socket_angle_cos",
                "socket_mouth_x",
                "socket_mouth_y",
                "socket_opening_x",
                "socket_opening_y",
            )
        )
        for finger in FINGERS:
            names.extend(
                (
                    f"force_{finger}",
                    f"contact_{finger}",
                    f"tip_rel_x_{finger}",
                    f"tip_rel_y_{finger}",
                    f"tip_rel_z_{finger}",
                    f"tip_vx_{finger}",
                    f"tip_vy_{finger}",
                    f"tip_vz_{finger}",
                    f"material_sin_{finger}",
                    f"material_cos_{finger}",
                    f"distance_{finger}",
                )
            )
        names.extend(f"phase_{phase.name.lower()}" for phase in TaskPhase)
        names.extend(
            (
                "target_depth",
                "target_twist",
                "depth_remaining",
                "twist_remaining",
                "episode_fraction",
                "holder_active",
                "engaged",
            )
        )
        return tuple(names)

    def build(self) -> np.ndarray:
        sim = self.simulator
        data = sim.data
        model = sim.model
        values: list[float] = []

        qpos = np.asarray(data.qpos, dtype=np.float64)
        qvel = np.asarray(data.qvel, dtype=np.float64)
        ctrl = np.asarray(data.ctrl, dtype=np.float64)
        actuator_force = np.asarray(data.actuator_force, dtype=np.float64)
        values.extend(np.clip(qpos, -10.0, 10.0))
        values.extend(np.clip(qvel / 10.0, -5.0, 5.0))
        ctrl_span = np.maximum(
            np.asarray(model.actuator_ctrlrange[:, 1])
            - np.asarray(model.actuator_ctrlrange[:, 0]),
            1e-6,
        )
        ctrl_mid = 0.5 * (
            np.asarray(model.actuator_ctrlrange[:, 1])
            + np.asarray(model.actuator_ctrlrange[:, 0])
        )
        values.extend(np.clip((ctrl - ctrl_mid) / (0.5 * ctrl_span), -2.0, 2.0))
        values.extend(np.tanh(actuator_force / 20.0))

        screw_position, screw_quaternion = sim.screw_pose()
        screw_velocity = sim.screw_velocity()
        tip_world = sim.screw_tip_world()
        tip_socket = sim.world_to_socket(tip_world)
        axis_error = sim.axis_alignment_error()
        marker_mod = float(sim.marker_mod_angle_rad)
        twist = float(sim.twist_unwrapped_rad)
        helical_error, helical_velocity_error = sim.helical_errors()
        values.extend(screw_position / 0.5)
        values.extend(screw_quaternion)
        values.extend(np.asarray(screw_velocity, dtype=np.float64) / 5.0)
        values.extend(
            (
                tip_socket[0] / 0.1,
                tip_socket[1] / 0.1,
                tip_socket[2] / max(float(sim.config.socket_hole_depth_m), 1e-6),
                axis_error[0] / math.radians(10.0),
                axis_error[1] / math.radians(10.0),
                math.sin(marker_mod),
                math.cos(marker_mod),
                twist / (4.0 * math.pi),
                twist / (2.0 * math.pi),
                helical_error / 0.005,
                helical_velocity_error / 0.05,
                float(sim.thread_resistance_nm) / 0.12,
                float(sim.axial_preload_n) / 5.0,
                math.sin(
                    math.radians(sim.config.socket_table_angle_deg)
                ),
                math.cos(
                    math.radians(sim.config.socket_table_angle_deg)
                ),
                float(sim.config.socket_mouth_xy_m[0])
                / sim.config.socket_mouth_orbit_radius_m,
                float(sim.config.socket_mouth_xy_m[1])
                / sim.config.socket_mouth_orbit_radius_m,
                float(sim.config.socket_opening_azimuth_xy[0]),
                float(sim.config.socket_opening_azimuth_xy[1]),
            )
        )

        contact_force = sim.fingertip_contact_forces()
        contact_flags = sim.fingertip_contact_flags()
        command: ScrewCommand = sim.current_command
        for finger in FINGERS:
            tip_position = sim.fingertip_position(finger)
            tip_velocity = sim.fingertip_velocity(finger)
            relative = tip_position - screw_position
            angle = float(command.material_angles_rad[finger])
            values.extend(
                (
                    math.tanh(float(contact_force[finger]) / 20.0),
                    float(bool(contact_flags[finger])),
                    relative[0] / 0.1,
                    relative[1] / 0.1,
                    relative[2] / 0.1,
                    tip_velocity[0] / 1.0,
                    tip_velocity[1] / 1.0,
                    tip_velocity[2] / 1.0,
                    math.sin(angle),
                    math.cos(angle),
                    command.tip_center_distances_m[finger] / 0.06,
                )
            )

        phase_one_hot = np.zeros(PHASE_COUNT, dtype=np.float64)
        phase_one_hot[int(command.phase)] = 1.0
        values.extend(phase_one_hot)
        target_depth = float(sim.target_depth_m)
        target_twist = float(sim.target_twist_rad)
        values.extend(
            (
                target_depth / max(float(sim.config.socket_hole_depth_m), 1e-6),
                target_twist / (4.0 * math.pi),
                (target_depth - float(sim.insertion_depth_m))
                / max(float(sim.config.socket_hole_depth_m), 1e-6),
                (target_twist - twist) / (4.0 * math.pi),
                float(sim.control_step) / max(float(sim.max_control_steps), 1.0),
                float(bool(sim.holder_active)),
                float(bool(sim.thread_engaged)),
            )
        )

        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.size,):
            raise RuntimeError(
                f"observation schema mismatch: {observation.shape} != {(self.size,)}"
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("observation contains non-finite values")
        return observation


__all__ = [
    "OmniscientObservationBuilder",
    "TwistTracker",
    "wrap_to_pi",
]
