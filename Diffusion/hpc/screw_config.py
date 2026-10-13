#!/usr/bin/env python3
"""Single source of truth for the movable SCREW MuJoCo task.

All dimensions use SI units.  This module intentionally depends only on the
Python standard library so configuration validation works before MuJoCo,
NumPy, or the learning stack is imported.
"""

from __future__ import annotations

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
    # The hard-tip grasp plane crosses the 40 mm head at its height centre.
    # With the upright screw tip on the 0.260 m table, its initial world
    # height is therefore exactly 0.370 m.  This H110 plane is retained until
    # measured thread engagement, then smoothly transitions to H112.
    screw_grasp_height_from_tip_m: float = 0.110
    # A 37.5 mm fingertip-centre radius gives 3.5 mm compliant preload against the
    # 25 mm head plus 16 mm hard-tip radius.  The H110 pickup plane lies in
    # the head;
    # the preload preserves contact when the spawn fixture
    # releases and the screw begins its lift in large MJWarp batches.
    continuous_grasp_distance_m: float = 0.0375

    # The mouth is randomized uniformly on a 20 mm-radius orbit about the
    # world centre.  This deterministic default lies on +X; the outward
    # opening normal points back toward (0, 0).
    socket_mouth_xy_m: Tuple[float, float] = (0.020, 0.0)
    socket_mouth_orbit_radius_m: float = 0.020
    # Test geometry: world mouth z=0.330 m; table top z=0.260 m.
    socket_mouth_height_above_table_m: float = 0.070
    socket_axis_world: Vec3 = (
        0.7071067811865476,
        0.0,
        -0.7071067811865476,
    )
    # The 30 mm-diameter screw shaft receives 2 mm radial clearance, giving
    # the requested hole diameter of screw diameter + 4 mm = 34 mm.
    socket_hole_radius_m: float = 0.017
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

    # Scripted pickup prefix.  The expert raises the fixed head-centre grasp
    # plane to world z=0.420 m directly above each sampled spawn.  There is no
    # centre-transfer waypoint: the learned ALIGN suffix then moves and
    # rotates the continuously held screw straight toward its socket.
    scripted_grasp_plane_world_z_m: float = 0.420
    # Retained only for checkpoint/config compatibility with the former
    # centre-transfer controller; it is not a trajectory waypoint.
    scripted_center_xy_m: Tuple[float, float] = (0.0, 0.0)
    diffusion_handoff_position_tolerance_m: float = 0.003
    diffusion_handoff_orientation_tolerance_deg: float = 5.0

    # Final measured pre-insertion pose.  The centre of the screw's leading
    # face is commanded onto the socket axis while remaining 5 mm outside the
    # mouth plane.  INSERT is not allowed to begin until the measured pose and
    # velocity satisfy every tolerance below.
    preinsert_standoff_m: float = 0.005
    # The 2 mm radial clearance requires tighter certified insertion gates.
    preinsert_lateral_tolerance_m: float = 0.0005
    # At the full 80 mm shaft length, 1 degree of tilt plus the 0.5 mm
    # lateral gate gives a worst-case radial excursion of 1.896 mm.  This
    # remains strictly inside the 2 mm radial clearance of the 34 mm bore,
    # while accommodating the measured closed-linkage equilibrium tilt.
    preinsert_orientation_tolerance_deg: float = 1.0
    preinsert_depth_tolerance_m: float = 0.002
    preinsert_tip_speed_tolerance_m_s: float = 0.020
    preinsert_lateral_speed_tolerance_m_s: float = 0.010
    preinsert_angular_speed_tolerance_deg_s: float = 12.0
    preinsert_stable_steps: int = 5
    preinsert_support_separation_deg: float = 150.0
    # A gain of three removes the 0.50--0.53 mm closed-linkage steady-state
    # offset observed at the full 14,000-world gate.  The correction remains
    # bounded by the unchanged 1.5 mm command limit and 0.4 mm/tick slew.
    preinsert_feedback_gain: float = 3.0
    preinsert_command_lateral_limit_m: float = 0.0015
    preinsert_feedback_depth_limit_m: float = 0.002
    preinsert_feedback_lateral_slew_m: float = 0.0004
    preinsert_feedback_depth_slew_m: float = 0.0005

    # Once the measured shaft is captured 30 mm inside the bore, the safety
    # contract may tolerate any two measured fingertip contacts.  The expert
    # still commands all four fingers at the 39 mm preload through the 70 mm
    # thread gate, and every material contact angle remains fixed until then.
    capture_depth_m: float = 0.030
    capture_lateral_tolerance_m: float = 0.0005
    # Retain the same certified axis envelope at the 30 mm capture handoff.
    # Only 50 mm of shaft remains outside the mouth here, so this is less
    # restrictive geometrically than the full-shaft pre-insertion proof.
    capture_orientation_tolerance_deg: float = 1.0
    capture_depth_tolerance_m: float = 0.002
    capture_tip_speed_tolerance_m_s: float = 0.020
    capture_lateral_speed_tolerance_m_s: float = 0.010
    capture_angular_speed_tolerance_deg_s: float = 12.0
    capture_head_bottom_clearance_m: float = 0.048
    capture_stable_steps: int = 5
    all_four_contact_stable_steps: int = 5
    # Four contacts are still required to release the pickup fixture.  During
    # tilted free-space transport, any three hard-tip contacts form a stable
    # grasp and avoid treating normal-force redistribution as a dropped screw.
    minimum_contact_count_before_capture: int = 3
    minimum_contact_count_after_capture: int = 2
    # The 70 mm engagement gate uses the same centre-line envelope as the
    # 30 mm capture gate.  The previous 3 mm / 5 degree legacy thresholds
    # exceeded the 2 mm radial clearance of the 34 mm bore.
    thread_lateral_tolerance_m: float = 0.0005
    thread_orientation_tolerance_deg: float = 1.0
    thread_stable_steps: int = 5
    thread_depth_tolerance_m: float = 0.0015
    # Contact acquisition remains strict, while already-established support
    # uses a lower Schmitt release threshold.  Removing the spawn fixture can
    # redistribute measured normal force for fewer than ten frames even while
    # the 2 mm preload and material point are unchanged; only a sustained loss
    # is a physical contract violation.
    contact_force_threshold_n: float = 0.10
    contact_force_release_threshold_n: float = 0.04
    contact_loss_grace_steps: int = 20

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
            return (1.0, 0.0)
        return (x / norm, y / norm)

    def spawn_is_on_opening_side(
        self, xy_m: Tuple[float, float], *, tolerance: float = 1.0e-12
    ) -> bool:
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
    """Inward axis whose horizontal opening direction faces the centre."""

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
        "continuous_grasp_distance_m",
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
        "scripted_grasp_plane_world_z_m",
        "diffusion_handoff_position_tolerance_m",
        "diffusion_handoff_orientation_tolerance_deg",
        "preinsert_standoff_m",
        "preinsert_lateral_tolerance_m",
        "preinsert_orientation_tolerance_deg",
        "preinsert_depth_tolerance_m",
        "preinsert_tip_speed_tolerance_m_s",
        "preinsert_lateral_speed_tolerance_m_s",
        "preinsert_angular_speed_tolerance_deg_s",
        "preinsert_support_separation_deg",
        "preinsert_feedback_gain",
        "preinsert_command_lateral_limit_m",
        "preinsert_feedback_depth_limit_m",
        "preinsert_feedback_lateral_slew_m",
        "preinsert_feedback_depth_slew_m",
        "capture_depth_m",
        "capture_lateral_tolerance_m",
        "capture_orientation_tolerance_deg",
        "capture_depth_tolerance_m",
        "capture_tip_speed_tolerance_m_s",
        "capture_lateral_speed_tolerance_m_s",
        "capture_angular_speed_tolerance_deg_s",
        "capture_head_bottom_clearance_m",
        "thread_lateral_tolerance_m",
        "thread_orientation_tolerance_deg",
        "thread_depth_tolerance_m",
        "contact_force_threshold_n",
        "contact_force_release_threshold_n",
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
            "socket mouth must lie on its 20 mm randomization orbit"
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
    if config.rotation_resistance_max_nm < config.rotation_resistance_min_nm:
        raise ScrewConfigError("rotation resistance range is reversed")
    if config.scripted_center_tip_world_z_m <= config.table_top_z_m:
        raise ScrewConfigError(
            "scripted lifted-spawn handoff must keep the tip above the table"
        )
    if config.preinsert_depth_tolerance_m >= config.preinsert_standoff_m:
        raise ScrewConfigError(
            "preinsert depth tolerance must keep the screw leading face "
            "strictly outside the socket mouth"
        )
    if config.preinsert_lateral_tolerance_m >= config.radial_clearance_m:
        raise ScrewConfigError(
            "preinsert lateral tolerance must be smaller than radial clearance"
        )
    if config.capture_lateral_tolerance_m >= config.radial_clearance_m:
        raise ScrewConfigError(
            "capture lateral tolerance must be smaller than radial clearance"
        )
    if config.preinsert_orientation_tolerance_deg >= 90.0:
        raise ScrewConfigError(
            "preinsert orientation tolerance must be below 90 degrees"
        )
    if config.capture_orientation_tolerance_deg >= 90.0:
        raise ScrewConfigError(
            "capture orientation tolerance must be below 90 degrees"
        )
    if config.thread_lateral_tolerance_m >= config.radial_clearance_m:
        raise ScrewConfigError(
            "thread lateral tolerance must be smaller than radial clearance"
        )
    if config.thread_orientation_tolerance_deg >= 90.0:
        raise ScrewConfigError(
            "thread orientation tolerance must be below 90 degrees"
        )
    if config.preinsert_support_separation_deg >= 180.0:
        raise ScrewConfigError(
            "preinsert support separation must be below 180 degrees"
        )
    if (
        config.preinsert_command_lateral_limit_m
        >= config.radial_clearance_m
    ):
        raise ScrewConfigError(
            "preinsert command lateral limit must remain inside radial clearance"
        )
    if (
        config.preinsert_feedback_depth_limit_m
        >= config.preinsert_standoff_m
    ):
        raise ScrewConfigError(
            "preinsert feedback depth limit must keep the nominal command outside"
        )
    if (
        config.preinsert_feedback_lateral_slew_m
        > config.preinsert_command_lateral_limit_m
        or config.preinsert_feedback_depth_slew_m
        > config.preinsert_feedback_depth_limit_m
    ):
        raise ScrewConfigError(
            "preinsert feedback slew cannot exceed its command limit"
        )
    worst_case_shaft_offset = (
        config.preinsert_lateral_tolerance_m
        + config.screw_shaft_length_m
        * math.sin(
            math.radians(config.preinsert_orientation_tolerance_deg)
        )
    )
    if worst_case_shaft_offset >= config.radial_clearance_m:
        raise ScrewConfigError(
            "preinsert lateral/orientation tolerances do not leave radial "
            "clearance for the full screw shaft"
        )
    for name in (
        "preinsert_stable_steps",
        "capture_stable_steps",
        "all_four_contact_stable_steps",
        "minimum_contact_count_after_capture",
        "contact_loss_grace_steps",
        "thread_stable_steps",
    ):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ScrewConfigError(f"{name} must be a positive integer")
    if not 2 <= config.minimum_contact_count_after_capture <= 4:
        raise ScrewConfigError(
            "minimum_contact_count_after_capture must be between 2 and 4"
        )
    if (
        config.contact_force_release_threshold_n
        >= config.contact_force_threshold_n
    ):
        raise ScrewConfigError(
            "contact release threshold must be lower than the acquisition "
            "threshold"
        )
    if not (
        config.screw_shaft_length_m
        < config.screw_grasp_height_from_tip_m
        < config.screw_total_height_m
    ):
        raise ScrewConfigError("screw grasp site must lie on the top head")
    if not (
        config.screw_head_radius_m
        < config.continuous_grasp_distance_m
        < config.screw_head_radius_m + 0.016
    ):
        raise ScrewConfigError(
            "continuous grasp distance must preload a 16 mm tip against "
            "the screw head without crossing the head centre"
        )
    if not (
        config.preinsert_standoff_m
        < config.capture_depth_m
        < config.thread_engagement_depth_m
    ):
        raise ScrewConfigError(
            "capture depth must lie between the outside-mouth gate and "
            "thread engagement"
        )
    if config.capture_depth_tolerance_m >= config.capture_depth_m:
        raise ScrewConfigError(
            "capture depth tolerance must be smaller than capture depth"
        )
    if (
        config.capture_depth_m + config.capture_depth_tolerance_m
        >= config.thread_engagement_depth_m
    ):
        raise ScrewConfigError(
            "capture depth tolerance must remain before thread engagement"
        )
    minimum_capture_head_clearance = (
        config.screw_shaft_length_m
        - config.capture_depth_m
        - config.capture_depth_tolerance_m
    )
    if (
        minimum_capture_head_clearance
        + 1.0e-12
        < config.capture_head_bottom_clearance_m
    ):
        raise ScrewConfigError(
            "capture gate does not preserve the configured screw-head "
            "bottom clearance outside the mouth"
        )
    worst_case_capture_offset = (
        config.capture_lateral_tolerance_m
        + config.screw_shaft_length_m
        * math.sin(
            math.radians(config.capture_orientation_tolerance_deg)
        )
    )
    if worst_case_capture_offset >= config.radial_clearance_m:
        raise ScrewConfigError(
            "capture lateral/orientation tolerances do not leave radial "
            "clearance for the full screw shaft"
        )
    worst_case_thread_offset = (
        config.thread_lateral_tolerance_m
        + config.screw_shaft_length_m
        * math.sin(
            math.radians(config.thread_orientation_tolerance_deg)
        )
    )
    if worst_case_thread_offset >= config.radial_clearance_m:
        raise ScrewConfigError(
            "thread lateral/orientation tolerances do not leave radial "
            "clearance for the full screw shaft"
        )

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
    if (
        config.thread_depth_tolerance_m
        >= config.socket_hole_depth_m - config.thread_engagement_depth_m
    ):
        raise ScrewConfigError(
            "thread depth tolerance must stay clear of the blind-hole bottom"
        )
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


__all__ = [
    "CAMERA_NAMES",
    "DEFAULT_CONFIG",
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
    "ScrewConfigError",
    "ScrewTaskConfig",
    "socket_axis_for_table_angle_deg",
    "validate_config",
]
