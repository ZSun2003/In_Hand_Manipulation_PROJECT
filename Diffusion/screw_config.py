#!/usr/bin/env python3
"""Single source of truth for the movable SCREW MuJoCo task.

All dimensions use SI units.  This module intentionally depends only on the
Python standard library so configuration validation works before MuJoCo,
NumPy, or the learning stack is imported.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Tuple


Vec3 = Tuple[float, float, float]
RGBA = Tuple[float, float, float, float]

# Stable names shared by the scene, environment, tests, and training code.
SCREW_BODY_NAME = "screw"
SCREW_FREE_JOINT_NAME = "screw_free"
SCREW_SHAFT_GEOM_NAME = "screw_shaft_collision"
SCREW_HEAD_GEOM_NAME = "screw_head_collision"
SCREW_TIP_SITE_NAME = "screw_tip"
SCREW_GRASP_SITE_NAME = "screw_grasp"
SOCKET_BODY_NAME = "socket"
SOCKET_MOUTH_SITE_NAME = "socket_mouth"
SOCKET_BOTTOM_SITE_NAME = "socket_bottom"
SCREW_START_HOLDER_BODY_NAME = "screw_start_holder_body"
SCREW_START_HOLDER_EQUALITY_NAME = "screw_start_holder"
CAMERA_NAMES = ("camera_overhead", "camera_side")


class ScrewConfigError(ValueError):
    """A task dimension or coupled geometric constraint is inconsistent."""


@dataclass(frozen=True)
class ScrewTaskConfig:
    """Geometry and task constants for the two-turn screw experiment."""

    # Existing table geometry from the original gripper model.
    table_body_position_m: Vec3 = (0.0, 0.0, 0.250)
    table_half_size_m: Vec3 = (0.500, 0.500, 0.010)

    # Screw: an 80 mm shaft with a 40 mm-high head/base on top.
    screw_shaft_radius_m: float = 0.015
    screw_shaft_length_m: float = 0.080
    screw_head_radius_m: float = 0.025
    screw_head_height_m: float = 0.040
    screw_thread_length_m: float = 0.020
    screw_mass_kg: float = 0.120
    # Fixed at the 40 mm head's height centre, 20 mm below its top. At full
    # insertion the fingertip centres remain 20 mm outside the mouth.
    screw_grasp_height_from_tip_m: float = 0.100

    # The mouth is randomized uniformly on a 20 mm-radius orbit about the
    # world centre.  This deterministic default lies on +X; the outward
    # opening normal points back toward (0, 0).
    socket_mouth_xy_m: Tuple[float, float] = (0.020, 0.0)
    socket_mouth_orbit_radius_m: float = 0.020
    # User-specified world mouth height is 0.330 m; the table top is 0.260 m.
    socket_mouth_height_above_table_m: float = 0.070
    socket_axis_world: Vec3 = (
        0.7071067811865476,
        0.0,
        -0.7071067811865476,
    )
    socket_hole_radius_m: float = 0.020
    socket_hole_depth_m: float = 0.080
    socket_thread_length_m: float = 0.020
    socket_outer_width_m: float = 0.080
    socket_outer_height_m: float = 0.080
    # Retain 20 mm of blind backing behind the 80 mm bore.
    socket_outer_length_m: float = 0.100
    socket_segment_count: int = 24

    # The shaft is first inserted to 70 mm. Two 5 mm-pitch turns advance 10 mm.
    thread_pitch_m: float = 0.005
    thread_engagement_depth_m: float = 0.070
    target_turns: float = 2.0
    thread_right_handed: bool = True

    # Episode reset distribution and deterministic default scene.
    spawn_radius_m: float = 0.100
    spawn_clearance_m: float = 0.003
    spawn_max_attempts: int = 10_000
    default_screw_xy_m: Tuple[float, float] = (-0.075, -0.040)

    # Confirmed contact-domain parameters. These are consumed by the environment.
    rotation_resistance_min_nm: float = 0.030
    rotation_resistance_max_nm: float = 0.120
    axial_preload_n: float = 2.0

    # Versioned controller geometry. These fields participate in the task
    # hash, preventing demos/checkpoints made with the old 58 mm,
    # non-equivariant schedule from being resumed silently.
    released_tip_center_distance_m: float = 0.049
    socket_yaw_roll_gain: float = 0.8
    socket_yaw_equivariance_version: int = 1

    # Scripted pickup prefix.  The expert first raises the fixed head-centre
    # grasp plane to world z=0.420 m, then carries the upright screw to the
    # world centre.  Diffusion/residual control is disabled until the measured
    # screw pose reaches this handoff target.
    scripted_grasp_plane_world_z_m: float = 0.420
    scripted_center_xy_m: Tuple[float, float] = (0.0, 0.0)
    diffusion_handoff_position_tolerance_m: float = 0.003
    diffusion_handoff_orientation_tolerance_deg: float = 5.0

    # Visual/material parameters.
    screw_rgba: RGBA = (0.62, 0.64, 0.68, 1.0)
    screw_thread_rgba: RGBA = (0.30, 0.32, 0.35, 1.0)
    marker_rgba: RGBA = (0.92, 0.06, 0.04, 1.0)
    socket_rgba: RGBA = (0.30, 0.36, 0.43, 1.0)
    socket_thread_rgba: RGBA = (0.12, 0.16, 0.20, 1.0)

    @property
    def table_top_z_m(self) -> float:
        return self.table_body_position_m[2] + self.table_half_size_m[2]

    @property
    def screw_total_height_m(self) -> float:
        return self.screw_shaft_length_m + self.screw_head_height_m

    @property
    def screw_initial_center_z_m(self) -> float:
        """World z of the shaft-centred screw body when standing on the table."""
        return self.table_top_z_m + 0.5 * self.screw_shaft_length_m

    @property
    def screw_tip_local_z_m(self) -> float:
        return -0.5 * self.screw_shaft_length_m

    @property
    def screw_grasp_local_z_m(self) -> float:
        return self.screw_tip_local_z_m + self.screw_grasp_height_from_tip_m

    @property
    def screw_head_center_local_z_m(self) -> float:
        return (
            0.5 * self.screw_shaft_length_m
            + 0.5 * self.screw_head_height_m
        )

    @property
    def scripted_center_tip_world_z_m(self) -> float:
        return (
            self.scripted_grasp_plane_world_z_m
            - self.screw_grasp_height_from_tip_m
        )

    @property
    def socket_mouth_position_m(self) -> Vec3:
        return (
            self.socket_mouth_xy_m[0],
            self.socket_mouth_xy_m[1],
            self.table_top_z_m + self.socket_mouth_height_above_table_m,
        )

    @property
    def socket_thread_start_depth_m(self) -> float:
        return self.socket_hole_depth_m - self.socket_thread_length_m

    @property
    def socket_backing_length_m(self) -> float:
        return self.socket_outer_length_m - self.socket_hole_depth_m

    @property
    def target_rotation_rad(self) -> float:
        return self.target_turns * 2.0 * math.pi

    @property
    def target_advance_m(self) -> float:
        return self.thread_pitch_m * self.target_turns

    @property
    def target_insertion_depth_m(self) -> float:
        return self.thread_engagement_depth_m + self.target_advance_m

    @property
    def radial_clearance_m(self) -> float:
        return self.socket_hole_radius_m - self.screw_shaft_radius_m

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **changes: Any) -> "ScrewTaskConfig":
        return validate_config(replace(self, **changes))

    @property
    def socket_table_angle_deg(self) -> float:
        """Acute angle between the inward hole axis and the tabletop."""

        horizontal = math.hypot(
            self.socket_axis_world[0], self.socket_axis_world[1]
        )
        return math.degrees(
            math.atan2(-self.socket_axis_world[2], horizontal)
        )

    @property
    def socket_opening_azimuth_xy(self) -> Tuple[float, float]:
        """Tabletop opening direction, always pointing toward world centre."""

        inward = self.socket_inward_azimuth_xy
        return (-inward[0], -inward[1])

    @property
    def socket_inward_azimuth_xy(self) -> Tuple[float, float]:
        """Unit XY direction from world centre toward the socket mouth."""

        x, y = (float(value) for value in self.socket_mouth_xy_m)
        norm = math.hypot(x, y)
        if norm <= 1.0e-12:
            # A sampled point is exactly central with probability zero.  Keep
            # +X as the deterministic convention for explicit centre poses.
            return (1.0, 0.0)
        return (x / norm, y / norm)

    def spawn_is_on_opening_side(
        self, xy_m: Tuple[float, float], *, tolerance: float = 1.0e-12
    ) -> bool:
        # Split the 100 mm spawn disk through the world centre so the allowed
        # set remains a true semicircle as the socket mouth moves.
        delta = (float(xy_m[0]), float(xy_m[1]))
        return (
            delta[0] * self.socket_opening_azimuth_xy[0]
            + delta[1] * self.socket_opening_azimuth_xy[1]
            >= -float(tolerance)
        )

    def with_socket_table_angle(
        self, angle_deg: float
    ) -> "ScrewTaskConfig":
        return self.with_overrides(
            socket_axis_world=socket_axis_for_table_angle_deg(
                angle_deg, self.socket_inward_azimuth_xy
            )
        )

    def with_socket_pose(
        self,
        mouth_xy_m: Tuple[float, float],
        angle_deg: float,
    ) -> "ScrewTaskConfig":
        """Return one episode geometry with a centre-facing socket."""

        mouth = (float(mouth_xy_m[0]), float(mouth_xy_m[1]))
        norm = math.hypot(*mouth)
        inward = (
            self.socket_inward_azimuth_xy
            if norm <= 1.0e-12
            else (mouth[0] / norm, mouth[1] / norm)
        )
        opening = (-inward[0], -inward[1])
        default_radius = math.hypot(*self.default_screw_xy_m)
        return self.with_overrides(
            socket_mouth_xy_m=mouth,
            socket_axis_world=socket_axis_for_table_angle_deg(
                angle_deg, inward
            ),
            default_screw_xy_m=(
                default_radius * opening[0],
                default_radius * opening[1],
            ),
        )


def socket_axis_for_table_angle_deg(
    angle_deg: float,
    inward_azimuth_xy: Tuple[float, float] = (1.0, 0.0),
) -> Vec3:
    """Inward axis whose opening faces the workspace centre.

    The inward horizontal component follows ``inward_azimuth_xy`` from the
    workspace centre toward the sampled mouth, so the opening's outward
    horizontal component points back toward the centre.
    """

    value = float(angle_deg)
    if not math.isfinite(value) or not 0.0 <= value <= 90.0:
        raise ScrewConfigError(
            "socket/table angle must be finite and within 0-90 degrees"
        )
    angle = math.radians(value)
    azimuth_x, azimuth_y = (
        float(inward_azimuth_xy[0]),
        float(inward_azimuth_xy[1]),
    )
    azimuth_norm = math.hypot(azimuth_x, azimuth_y)
    if not math.isfinite(azimuth_norm) or azimuth_norm <= 1.0e-12:
        raise ScrewConfigError("socket inward azimuth must be finite and nonzero")
    azimuth_x /= azimuth_norm
    azimuth_y /= azimuth_norm
    horizontal = math.cos(angle)
    x = horizontal * azimuth_x
    y = horizontal * azimuth_y
    z = -math.sin(angle)
    if abs(x) < 1.0e-15:
        x = 0.0
    if abs(z) < 1.0e-15:
        z = 0.0
    if abs(y) < 1.0e-15:
        y = 0.0
    return (x, y, z)


def _positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ScrewConfigError(f"{name} must be finite and positive, got {value!r}")


def _finite_vector(name: str, values: Tuple[float, ...], length: int) -> None:
    if len(values) != length or any(not math.isfinite(float(v)) for v in values):
        raise ScrewConfigError(
            f"{name} must contain {length} finite values, got {values!r}"
        )


def validate_config(config: ScrewTaskConfig) -> ScrewTaskConfig:
    """Validate every coupled dimension and return *config* unchanged."""
    if not isinstance(config, ScrewTaskConfig):
        raise TypeError(f"expected ScrewTaskConfig, got {type(config).__name__}")

    _finite_vector("table_body_position_m", config.table_body_position_m, 3)
    _finite_vector("table_half_size_m", config.table_half_size_m, 3)
    _finite_vector("socket_axis_world", config.socket_axis_world, 3)
    _finite_vector("socket_mouth_xy_m", config.socket_mouth_xy_m, 2)
    _finite_vector("scripted_center_xy_m", config.scripted_center_xy_m, 2)
    for name in (
        "screw_shaft_radius_m",
        "screw_shaft_length_m",
        "screw_head_radius_m",
        "screw_head_height_m",
        "screw_thread_length_m",
        "screw_mass_kg",
        "screw_grasp_height_from_tip_m",
        "socket_mouth_orbit_radius_m",
        "socket_mouth_height_above_table_m",
        "socket_hole_radius_m",
        "socket_hole_depth_m",
        "socket_thread_length_m",
        "socket_outer_width_m",
        "socket_outer_height_m",
        "socket_outer_length_m",
        "thread_pitch_m",
        "thread_engagement_depth_m",
        "target_turns",
        "spawn_radius_m",
        "spawn_clearance_m",
        "rotation_resistance_min_nm",
        "rotation_resistance_max_nm",
        "axial_preload_n",
        "released_tip_center_distance_m",
        "socket_yaw_roll_gain",
        "scripted_grasp_plane_world_z_m",
        "diffusion_handoff_position_tolerance_m",
        "diffusion_handoff_orientation_tolerance_deg",
    ):
        _positive(name, float(getattr(config, name)))

    if any(v <= 0.0 for v in config.table_half_size_m):
        raise ScrewConfigError("all table half sizes must be positive")
    if config.screw_thread_length_m > config.screw_shaft_length_m:
        raise ScrewConfigError("screw thread cannot exceed shaft length")
    if config.socket_thread_length_m > config.socket_hole_depth_m:
        raise ScrewConfigError("socket thread cannot exceed hole depth")
    if config.screw_head_radius_m <= config.screw_shaft_radius_m:
        raise ScrewConfigError("screw head must be wider than the shaft")
    if config.socket_hole_radius_m <= config.screw_shaft_radius_m:
        raise ScrewConfigError("socket hole must be wider than the screw shaft")
    if 2.0 * config.socket_hole_radius_m >= min(
        config.socket_outer_width_m,
        config.socket_outer_height_m,
    ):
        raise ScrewConfigError("hole must leave positive socket wall thickness")
    if config.socket_outer_length_m <= config.socket_hole_depth_m:
        raise ScrewConfigError("blind socket needs positive material behind the hole")

    axis_norm = math.sqrt(sum(float(v) ** 2 for v in config.socket_axis_world))
    if not math.isclose(axis_norm, 1.0, abs_tol=1e-9):
        raise ScrewConfigError(
            f"socket_axis_world must be unit length, norm={axis_norm:.12g}"
        )
    mouth_radius = math.hypot(*config.socket_mouth_xy_m)
    if not math.isclose(
        mouth_radius,
        config.socket_mouth_orbit_radius_m,
        abs_tol=1.0e-9,
    ):
        raise ScrewConfigError(
            "socket mouth must lie on its configured randomization orbit"
        )
    horizontal_axis = math.hypot(
        config.socket_axis_world[0], config.socket_axis_world[1]
    )
    radial = config.socket_inward_azimuth_xy
    if (
        config.socket_axis_world[2] > 1.0e-12
        or (
            horizontal_axis > 1.0e-10
            and (
                abs(config.socket_axis_world[0] / horizontal_axis - radial[0])
                > 1.0e-9
                or abs(
                    config.socket_axis_world[1] / horizontal_axis - radial[1]
                )
                > 1.0e-9
            )
        )
    ):
        raise ScrewConfigError(
            "socket body axis must point radially away/down from the "
            "centre-facing opening"
        )
    table_angle = config.socket_table_angle_deg
    if not math.isclose(table_angle, 45.0, abs_tol=1.0e-9):
        raise ScrewConfigError("socket/table angle is fixed at 45 degrees")

    if config.socket_segment_count < 8 or config.socket_segment_count % 8 != 0:
        raise ScrewConfigError(
            "socket_segment_count must be >= 8 and divisible by 8 so square "
            "corners are segment boundaries"
        )
    if config.spawn_max_attempts <= 0:
        raise ScrewConfigError("spawn_max_attempts must be positive")
    if config.socket_yaw_equivariance_version != 1:
        raise ScrewConfigError(
            "socket_yaw_equivariance_version must equal 1"
        )
    if not math.isclose(
        config.socket_yaw_roll_gain, 0.8, abs_tol=1.0e-12
    ):
        raise ScrewConfigError("socket_yaw_roll_gain must equal 0.8")
    if not (
        config.screw_head_radius_m
        < config.released_tip_center_distance_m
        < config.socket_outer_width_m
    ):
        raise ScrewConfigError(
            "released fingertip distance must clear the screw head while "
            "remaining inside the controller workspace contract"
        )
    if config.rotation_resistance_max_nm < config.rotation_resistance_min_nm:
        raise ScrewConfigError("rotation resistance range is reversed")
    if config.scripted_center_tip_world_z_m <= config.table_top_z_m:
        raise ScrewConfigError(
            "scripted centre handoff must lift the screw tip above the table"
        )
    if not (
        config.screw_shaft_length_m
        < config.screw_grasp_height_from_tip_m
        < config.screw_total_height_m
    ):
        raise ScrewConfigError("screw grasp site must lie on the top head")

    spawn_radius = math.hypot(*config.default_screw_xy_m)
    if spawn_radius > config.spawn_radius_m + 1e-12:
        raise ScrewConfigError("default screw position lies outside spawn disk")
    if not config.spawn_is_on_opening_side(config.default_screw_xy_m):
        raise ScrewConfigError(
            "default screw position must lie in the opening-facing half disk"
        )
    if config.thread_engagement_depth_m < config.socket_thread_start_depth_m:
        raise ScrewConfigError("engagement must occur inside the socket thread region")
    if config.thread_engagement_depth_m >= config.socket_hole_depth_m:
        raise ScrewConfigError("engagement depth must be before the blind-hole bottom")
    if not math.isclose(
        config.target_insertion_depth_m,
        config.socket_hole_depth_m,
        abs_tol=1e-12,
    ):
        raise ScrewConfigError(
            "pitch × turns must advance from engagement depth to exactly full "
            f"insertion; got {config.target_insertion_depth_m:.6f} m"
        )
    if not math.isclose(
        config.screw_shaft_length_m,
        config.socket_hole_depth_m,
        abs_tol=1e-12,
    ):
        raise ScrewConfigError(
            "full insertion requires shaft length to equal blind-hole depth"
        )
    return config


DEFAULT_CONFIG = validate_config(ScrewTaskConfig())
OBSERVATION_SCHEMA = "screw-omniscient-state-v7"
# This is deliberately separate from the geometry dataclass: demonstrations
# and checkpoints also depend on how the same geometry is canonicalized and
# commanded.  Training imports ``TASK_GEOMETRY_HASH``, so changing any part of
# the yaw-equivariant controller invalidates stale artifacts automatically.
CONTROLLER_STRATEGY = {
    "version": DEFAULT_CONFIG.socket_yaw_equivariance_version,
    "released_distance_m": (
        DEFAULT_CONFIG.released_tip_center_distance_m
    ),
    "quadrant_canonicalization": "q=floor(yaw/(pi/2)),delta=yaw-q*pi/2",
    "pose_transform": "Q R Q^T",
    "tip_transform": "Q p",
    "finger_role_mapping": "logical-to-physical-quarter-turn",
    "residual_roll_gain": DEFAULT_CONFIG.socket_yaw_roll_gain,
    "residual_roll_application": (
        "right-multiply-after-original-minimum-jerk-interpolation"
    ),
    "align_boundary_step": 365,
    "align_steps": 150,
    "continuous_spawn_prefix_ik_steps": 216,
}
TASK_GEOMETRY_HASH = hashlib.sha256(
    json.dumps(
        {
            "root_height_m": 0.620,
            "task": DEFAULT_CONFIG.as_dict(),
            "controller_strategy": CONTROLLER_STRATEGY,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


__all__ = [
    "CAMERA_NAMES",
    "CONTROLLER_STRATEGY",
    "DEFAULT_CONFIG",
    "OBSERVATION_SCHEMA",
    "SCREW_BODY_NAME",
    "SCREW_FREE_JOINT_NAME",
    "SCREW_GRASP_SITE_NAME",
    "SCREW_HEAD_GEOM_NAME",
    "SCREW_SHAFT_GEOM_NAME",
    "SCREW_START_HOLDER_BODY_NAME",
    "SCREW_START_HOLDER_EQUALITY_NAME",
    "SCREW_TIP_SITE_NAME",
    "SOCKET_BODY_NAME",
    "SOCKET_BOTTOM_SITE_NAME",
    "SOCKET_MOUTH_SITE_NAME",
    "TASK_GEOMETRY_HASH",
    "ScrewConfigError",
    "ScrewTaskConfig",
    "socket_axis_for_table_angle_deg",
    "validate_config",
]
