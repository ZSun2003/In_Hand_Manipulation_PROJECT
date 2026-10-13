#!/usr/bin/env python3
"""Relocatable geometry builder for the SCREW MuJoCo task.

The generated socket is the union of convex prism segments.  Their inner
polygon has a 20 mm inradius, so the approximation never makes the requested
40 mm opening smaller.  A square rear cap closes the final 20 mm of the
100 mm block, giving an exact 80 mm blind-hole depth.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Optional, Protocol, Sequence, Tuple

from screw_config import (
    CAMERA_NAMES,
    DEFAULT_CONFIG,
    SCREW_BODY_NAME,
    SCREW_FREE_JOINT_NAME,
    SCREW_GRASP_SITE_NAME,
    SCREW_HEAD_GEOM_NAME,
    SCREW_SHAFT_GEOM_NAME,
    SCREW_START_HOLDER_BODY_NAME,
    SCREW_TIP_SITE_NAME,
    SOCKET_BODY_NAME,
    SOCKET_BOTTOM_SITE_NAME,
    SOCKET_MOUTH_SITE_NAME,
    ScrewTaskConfig,
    validate_config,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_XML = MODULE_DIR / "Project.xml"
DEFAULT_RUNTIME_XML = MODULE_DIR / ".runtime" / "scene.xml"
Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


class RandomLike(Protocol):
    def random(self) -> float: ...


class SpawnSamplingError(RuntimeError):
    """No collision-free, IK-reachable reset point was found."""


@dataclass(frozen=True)
class ScrewPose:
    """World pose plus key points for an axis-aligned insertion depth."""

    position_m: Vec3
    quaternion_wxyz: Quat
    tip_position_m: Vec3
    grasp_position_m: Vec3
    insertion_depth_m: float
    accumulated_angle_rad: float


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Sequence[float]) -> Vec3:
    length = _norm(vector)
    if length <= 1e-15:
        raise ValueError("cannot normalize a zero vector")
    return tuple(float(v) / length for v in vector)  # type: ignore[return-value]


def _add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return tuple(float(x) + float(y) for x, y in zip(a, b))  # type: ignore[return-value]


def _scale(vector: Sequence[float], scalar: float) -> Vec3:
    return tuple(float(scalar) * float(v) for v in vector)  # type: ignore[return-value]


def _quat_normalize(quaternion: Sequence[float]) -> Quat:
    length = math.sqrt(sum(float(v) ** 2 for v in quaternion))
    if length <= 1e-15:
        return (1.0, 0.0, 0.0, 0.0)
    values = tuple(float(v) / length for v in quaternion)
    if values[0] < 0.0:
        values = tuple(-v for v in values)
    return values  # type: ignore[return-value]


def quat_multiply(left: Sequence[float], right: Sequence[float]) -> Quat:
    """Hamilton product for MuJoCo ``w x y z`` quaternions."""
    w1, x1, y1, z1 = _quat_normalize(left)
    w2, x2, y2, z2 = _quat_normalize(right)
    return _quat_normalize(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def axis_angle_quaternion(axis: Sequence[float], angle_rad: float) -> Quat:
    axis_unit = _normalize(axis)
    half = 0.5 * float(angle_rad)
    sine = math.sin(half)
    return _quat_normalize(
        (
            math.cos(half),
            axis_unit[0] * sine,
            axis_unit[1] * sine,
            axis_unit[2] * sine,
        )
    )


def quaternion_from_z_axis(direction: Sequence[float]) -> Quat:
    """Return the shortest rotation mapping local +Z to *direction*."""
    target = _normalize(direction)
    source = (0.0, 0.0, 1.0)
    cosine = max(-1.0, min(1.0, _dot(source, target)))
    if cosine > 1.0 - 1e-14:
        return (1.0, 0.0, 0.0, 0.0)
    if cosine < -1.0 + 1e-14:
        return (0.0, 1.0, 0.0, 0.0)
    axis = _normalize(_cross(source, target))
    return axis_angle_quaternion(axis, math.acos(cosine))


def quaternion_to_matrix(quaternion: Sequence[float]) -> Tuple[Vec3, Vec3, Vec3]:
    """Return a row-major 3×3 rotation matrix."""
    w, x, y, z = _quat_normalize(quaternion)
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * w),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def rotate_vector(quaternion: Sequence[float], vector: Sequence[float]) -> Vec3:
    matrix = quaternion_to_matrix(quaternion)
    return tuple(_dot(row, vector) for row in matrix)  # type: ignore[return-value]


def socket_quaternion(config: ScrewTaskConfig = DEFAULT_CONFIG) -> Quat:
    """Socket-local +Z points from the mouth into the blind hole."""
    validate_config(config)
    return quaternion_from_z_axis(config.socket_axis_world)


def socket_center_position(
    config: ScrewTaskConfig = DEFAULT_CONFIG,
) -> Vec3:
    validate_config(config)
    return _add(
        config.socket_mouth_position_m,
        _scale(config.socket_axis_world, 0.5 * config.socket_outer_length_m),
    )


def helical_insertion_depth(
    accumulated_angle_rad: float,
    config: ScrewTaskConfig = DEFAULT_CONFIG,
) -> float:
    """Hard screw relation after the 70 mm engagement event.

    Positive angle is the tightening direction. With an inward-pointing axis,
    positive right-handed rotation appears clockwise when viewed from outside.
    """
    validate_config(config)
    handedness = 1.0 if config.thread_right_handed else -1.0
    return (
        config.thread_engagement_depth_m
        + handedness
        * config.thread_pitch_m
        * float(accumulated_angle_rad)
        / (2.0 * math.pi)
    )


def screw_pose_for_insertion(
    insertion_depth_m: float,
    accumulated_angle_rad: float = 0.0,
    config: ScrewTaskConfig = DEFAULT_CONFIG,
) -> ScrewPose:
    """Return the shaft-centred screw pose for a depth along the socket axis."""
    validate_config(config)
    depth = float(insertion_depth_m)
    if not 0.0 <= depth <= config.socket_hole_depth_m + 1e-12:
        raise ValueError(
            f"insertion depth must be in [0, {config.socket_hole_depth_m}], "
            f"got {depth}"
        )

    inward = config.socket_axis_world
    outward = _scale(inward, -1.0)
    tip = _add(config.socket_mouth_position_m, _scale(inward, depth))
    shaft_center = _add(
        tip,
        _scale(outward, 0.5 * config.screw_shaft_length_m),
    )
    align = quaternion_from_z_axis(outward)
    spin = axis_angle_quaternion(inward, float(accumulated_angle_rad))
    quaternion = quat_multiply(spin, align)
    grasp_from_tip = _scale(outward, config.screw_grasp_height_from_tip_m)
    grasp = _add(tip, grasp_from_tip)
    return ScrewPose(
        position_m=shaft_center,
        quaternion_wxyz=quaternion,
        tip_position_m=tip,
        grasp_position_m=grasp,
        insertion_depth_m=depth,
        accumulated_angle_rad=float(accumulated_angle_rad),
    )


def socket_projected_aabb_xy(
    config: ScrewTaskConfig = DEFAULT_CONFIG,
) -> Tuple[float, float, float, float]:
    """XY AABB of the complete tilted 80×80×100 mm socket block."""
    validate_config(config)
    center = socket_center_position(config)
    rotation = quaternion_to_matrix(socket_quaternion(config))
    half = (
        0.5 * config.socket_outer_width_m,
        0.5 * config.socket_outer_height_m,
        0.5 * config.socket_outer_length_m,
    )
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = (sx * half[0], sy * half[1], sz * half[2])
                corners.append(_add(center, rotate_vector(socket_quaternion(config), local)))
    return (
        min(point[0] for point in corners),
        max(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[1] for point in corners),
    )


def upright_screw_overlaps_socket(
    screw_xy_m: Sequence[float],
    config: ScrewTaskConfig = DEFAULT_CONFIG,
) -> bool:
    """Conservative reset rejection against the socket's projected block."""
    validate_config(config)
    if len(screw_xy_m) != 2:
        raise ValueError("screw_xy_m must have two values")
    x, y = float(screw_xy_m[0]), float(screw_xy_m[1])
    xmin, xmax, ymin, ymax = socket_projected_aabb_xy(config)
    margin = config.screw_head_radius_m + config.spawn_clearance_m
    return bool(
        xmin - margin <= x <= xmax + margin
        and ymin - margin <= y <= ymax + margin
    )


def initial_grasp_position(
    screw_xy_m: Sequence[float],
    config: ScrewTaskConfig = DEFAULT_CONFIG,
) -> Vec3:
    """World position of ``screw_grasp`` for an upright reset screw."""
    if len(screw_xy_m) != 2:
        raise ValueError("screw_xy_m must have two values")
    return (
        float(screw_xy_m[0]),
        float(screw_xy_m[1]),
        config.table_top_z_m + config.screw_grasp_height_from_tip_m,
    )


def sample_socket_mouth_xy(
    rng: Optional[RandomLike] = None,
    config: ScrewTaskConfig = DEFAULT_CONFIG,
) -> Tuple[float, float]:
    """Sample a uniform azimuth on the configured mouth orbit."""

    validate_config(config)
    generator = rng if rng is not None else random.Random()
    azimuth = 2.0 * math.pi * float(generator.random())
    radius = config.socket_mouth_orbit_radius_m
    return (
        radius * math.cos(azimuth),
        radius * math.sin(azimuth),
    )


def sample_spawn_xy(
    rng: Optional[RandomLike] = None,
    config: ScrewTaskConfig = DEFAULT_CONFIG,
    ik_reachable: Optional[Callable[[Vec3], bool]] = None,
    reject: Optional[Callable[[Vec3], bool]] = None,
) -> Tuple[float, float]:
    """Area-uniform opening-side half-disk sampling with rejection gates.

    ``ik_reachable`` receives the upright head grasp-site world position.
    ``reject`` can add environment-specific collision tests and returns true
    when a candidate must be discarded.
    """
    validate_config(config)
    generator = rng if rng is not None else random.Random()
    for _attempt in range(config.spawn_max_attempts):
        radius = config.spawn_radius_m * math.sqrt(float(generator.random()))
        # Rotate the semicircle with the random socket azimuth.  Its dividing
        # line passes through world centre, so it remains a true half-disk.
        opening = config.socket_opening_azimuth_xy
        opening_angle = math.atan2(opening[1], opening[0])
        theta = (
            opening_angle
            - 0.5 * math.pi
            + math.pi * float(generator.random())
        )
        xy = (radius * math.cos(theta), radius * math.sin(theta))
        if not config.spawn_is_on_opening_side(xy):
            continue
        if upright_screw_overlaps_socket(xy, config):
            continue
        grasp = initial_grasp_position(xy, config)
        if reject is not None and bool(reject(grasp)):
            continue
        if ik_reachable is not None:
            try:
                reachable = bool(ik_reachable(grasp))
            except Exception:
                reachable = False
            if not reachable:
                continue
        return xy
    raise SpawnSamplingError(
        "failed to sample a collision-free, IK-reachable screw position "
        f"after {config.spawn_max_attempts} attempts"
    )


def _fmt(value: float) -> str:
    text = f"{float(value):.12g}"
    return "0" if text in ("-0", "-0.0") else text


def _vec(values: Iterable[float]) -> str:
    return " ".join(_fmt(value) for value in values)


def _rgba(values: Iterable[float]) -> str:
    return _vec(values)


def _resource_path(prefix: str, filename: str) -> str:
    prefix_path = PurePosixPath(str(prefix).replace("\\", "/"))
    if prefix_path.is_absolute():
        raise ValueError("resource_prefix must be relative")
    if str(prefix_path) in ("", "."):
        return filename
    return str(prefix_path / filename)


def _square_ray_point(
    angle_rad: float,
    half_width_m: float,
    half_height_m: float,
) -> Tuple[float, float]:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    tx = math.inf if abs(cosine) < 1e-15 else half_width_m / abs(cosine)
    ty = math.inf if abs(sine) < 1e-15 else half_height_m / abs(sine)
    distance = min(tx, ty)
    return distance * cosine, distance * sine


def _socket_segment_vertices(
    index: int,
    config: ScrewTaskConfig,
) -> Tuple[Vec3, ...]:
    count = config.socket_segment_count
    step = 2.0 * math.pi / float(count)
    theta0 = index * step
    theta1 = (index + 1) * step
    # Straight chords are tangent to the requested 20 mm-radius circle.
    inner_vertex_radius = config.socket_hole_radius_m / math.cos(0.5 * step)
    inner0 = (
        inner_vertex_radius * math.cos(theta0),
        inner_vertex_radius * math.sin(theta0),
    )
    inner1 = (
        inner_vertex_radius * math.cos(theta1),
        inner_vertex_radius * math.sin(theta1),
    )
    outer0 = _square_ray_point(
        theta0,
        0.5 * config.socket_outer_width_m,
        0.5 * config.socket_outer_height_m,
    )
    outer1 = _square_ray_point(
        theta1,
        0.5 * config.socket_outer_width_m,
        0.5 * config.socket_outer_height_m,
    )
    z0 = -0.5 * config.socket_outer_length_m
    z1 = 0.5 * config.socket_outer_length_m
    polygon = (inner0, outer0, outer1, inner1)
    return tuple(
        (point[0], point[1], z)
        for z in (z0, z1)
        for point in polygon
    )


def _socket_asset_xml(config: ScrewTaskConfig) -> str:
    lines = ["    <asset>"]
    for index in range(config.socket_segment_count):
        vertices = _socket_segment_vertices(index, config)
        flat = [coordinate for vertex in vertices for coordinate in vertex]
        lines.append(
            f'        <mesh name="socket_segment_{index:02d}_mesh" '
            f'vertex="{_vec(flat)}"/>'
        )
    lines.append("    </asset>")
    return "\n".join(lines)


def _screw_body_xml(
    screw_xy_m: Sequence[float],
    config: ScrewTaskConfig,
) -> str:
    x, y = float(screw_xy_m[0]), float(screw_xy_m[1])
    position = (x, y, config.screw_initial_center_z_m)
    shaft_mass = 2.0 * config.screw_mass_kg / 3.0
    head_mass = config.screw_mass_kg - shaft_mass
    head_z = config.screw_head_center_local_z_m
    top_z = 0.5 * config.screw_shaft_length_m + config.screw_head_height_m
    thread_start_z = config.screw_tip_local_z_m
    thread_ring_count = max(
        1,
        int(round(config.screw_thread_length_m / config.thread_pitch_m)),
    )
    lines = [
        f'        <body name="{SCREW_START_HOLDER_BODY_NAME}" '
        f'mocap="true" pos="{_vec(position)}"/>',
        f'        <body name="{SCREW_BODY_NAME}" pos="{_vec(position)}">',
        f'            <joint name="{SCREW_FREE_JOINT_NAME}" type="free" damping="0.01"/>',
        f'            <geom name="{SCREW_SHAFT_GEOM_NAME}" type="cylinder" '
        f'size="{_fmt(config.screw_shaft_radius_m)} '
        f'{_fmt(0.5 * config.screw_shaft_length_m)}" '
        f'mass="{_fmt(shaft_mass)}" rgba="{_rgba(config.screw_rgba)}" '
        'contype="4" conaffinity="11" condim="4" '
        'friction="1.0 0.01 0.001" solref="0.004 1" '
        'solimp="0.95 0.99 0.001"/>',
        f'            <geom name="{SCREW_HEAD_GEOM_NAME}" type="cylinder" '
        f'pos="0 0 {_fmt(head_z)}" size="{_fmt(config.screw_head_radius_m)} '
        f'{_fmt(0.5 * config.screw_head_height_m)}" mass="{_fmt(head_mass)}" '
        f'rgba="{_rgba(config.screw_rgba)}" contype="4" conaffinity="11" '
        'condim="4" friction="3.0 0.02 0.002" solref="0.004 1" '
        'solimp="0.95 0.99 0.001"/>',
    ]
    for ring in range(thread_ring_count):
        z = thread_start_z + (ring + 0.5) * config.thread_pitch_m
        lines.append(
            f'            <geom name="screw_thread_visual_{ring:02d}" '
            f'type="cylinder" pos="0 0 {_fmt(z)}" '
            f'size="{_fmt(config.screw_shaft_radius_m + 0.00035)} 0.00035" '
            f'rgba="{_rgba(config.screw_thread_rgba)}" '
            'contype="0" conaffinity="0" mass="0"/>'
        )
    lines.extend(
        [
            f'            <geom name="screw_marker_top" type="box" '
            f'pos="{_fmt(config.screw_head_radius_m * 0.45)} 0 '
            f'{_fmt(top_z + 0.00025)}" '
            f'size="{_fmt(config.screw_head_radius_m * 0.45)} '
            '0.003 0.00025" '
            f'rgba="{_rgba(config.marker_rgba)}" '
            'contype="0" conaffinity="0" mass="0"/>',
            f'            <geom name="screw_marker_side" type="box" '
            f'pos="{_fmt(config.screw_head_radius_m + 0.00025)} 0 '
            f'{_fmt(head_z)}" size="0.00035 0.004 '
            f'{_fmt(0.45 * config.screw_head_height_m)}" '
            f'rgba="{_rgba(config.marker_rgba)}" '
            'contype="0" conaffinity="0" mass="0"/>',
            f'            <site name="{SCREW_TIP_SITE_NAME}" '
            f'pos="0 0 {_fmt(config.screw_tip_local_z_m)}" '
            'size="0.002" rgba="1 0.8 0 1"/>',
            f'            <site name="{SCREW_GRASP_SITE_NAME}" '
            f'pos="0 0 {_fmt(config.screw_grasp_local_z_m)}" '
            'size="0.003" rgba="0 1 0 1"/>',
            "        </body>",
        ]
    )
    return "\n".join(lines)


def _socket_thread_visual_xml(config: ScrewTaskConfig) -> str:
    lines = []
    ring_count = max(
        1,
        int(round(config.socket_thread_length_m / config.thread_pitch_m)),
    )
    visual_segments = config.socket_segment_count
    angle_step = 2.0 * math.pi / visual_segments
    tangent_half = (
        config.socket_hole_radius_m * math.tan(0.5 * angle_step) * 1.03
    )
    for ring in range(ring_count):
        depth = (
            config.socket_thread_start_depth_m
            + (ring + 0.5) * config.thread_pitch_m
        )
        local_z = -0.5 * config.socket_outer_length_m + depth
        for segment in range(visual_segments):
            theta = (segment + 0.5) * angle_step
            radius = config.socket_hole_radius_m - 0.00025
            position = (
                radius * math.cos(theta),
                radius * math.sin(theta),
                local_z,
            )
            quaternion = axis_angle_quaternion((0.0, 0.0, 1.0), theta)
            lines.append(
                f'            <geom name="socket_thread_visual_{ring:02d}_'
                f'{segment:02d}" type="box" pos="{_vec(position)}" '
                f'quat="{_vec(quaternion)}" '
                f'size="0.00025 {_fmt(tangent_half)} 0.00035" '
                f'rgba="{_rgba(config.socket_thread_rgba)}" '
                'contype="0" conaffinity="0" group="1"/>'
            )
    return "\n".join(lines)


def _socket_body_xml(
    config: ScrewTaskConfig,
    *,
    mocap: bool = False,
) -> str:
    center = socket_center_position(config)
    quaternion = socket_quaternion(config)
    back_center_z = (
        -0.5 * config.socket_outer_length_m
        + config.socket_hole_depth_m
        + 0.5 * config.socket_backing_length_m
    )
    mocap_attribute = ' mocap="true"' if mocap else ""
    lines = [
        f'        <body name="{SOCKET_BODY_NAME}"'
        f'{mocap_attribute} pos="{_vec(center)}" '
        f'quat="{_vec(quaternion)}">',
    ]
    for index in range(config.socket_segment_count):
        lines.append(
            f'            <geom name="socket_wall_{index:02d}" type="mesh" '
            f'mesh="socket_segment_{index:02d}_mesh" '
            f'rgba="{_rgba(config.socket_rgba)}" contype="8" '
            'conaffinity="38" condim="4" friction="0.2 0.005 0.001" '
            'solref="0.004 1" solimp="0.95 0.99 0.001"/>'
        )
    lines.extend(
        [
            f'            <geom name="socket_bottom_collision" type="box" '
            f'pos="0 0 {_fmt(back_center_z)}" '
            f'size="{_fmt(0.5 * config.socket_outer_width_m)} '
            f'{_fmt(0.5 * config.socket_outer_height_m)} '
            f'{_fmt(0.5 * config.socket_backing_length_m)}" '
            f'rgba="{_rgba(config.socket_rgba)}" contype="8" '
            'conaffinity="38" condim="4" friction="1.0 0.01 0.001"/>',
            _socket_thread_visual_xml(config),
            f'            <site name="{SOCKET_MOUTH_SITE_NAME}" '
            f'pos="0 0 {_fmt(-0.5 * config.socket_outer_length_m)}" '
            f'size="{_fmt(config.socket_hole_radius_m)}" type="cylinder" '
            'rgba="0 0.7 1 0.18"/>',
            f'            <site name="{SOCKET_BOTTOM_SITE_NAME}" '
            f'pos="0 0 {_fmt(-0.5 * config.socket_outer_length_m + config.socket_hole_depth_m)}" '
            'size="0.002" rgba="1 0.8 0 1"/>',
            "        </body>",
        ]
    )
    return "\n".join(lines)


def _table_body_xml(config: ScrewTaskConfig) -> str:
    """Build the full tabletop used around the embedded random socket.

    The fixed-45-degree socket and its 80 mm bore intentionally extend through
    the 20 mm-thick table. Collision classes therefore
    keep hard-tip/long-link table contacts while filtering screw/table only;
    screw/socket-wall and screw/socket-bottom contacts remain physical.
    """

    table_pos = config.table_body_position_m
    table_size = config.table_half_size_m
    return "\n".join(
        (
            f'        <body name="table" pos="{_vec(table_pos)}">',
            f'            <geom name="table_geom" type="box" '
            f'size="{_vec(table_size)}" rgba="0.70 0.70 0.70 1" '
            'contype="0" conaffinity="0"/>',
            f'            <geom name="table_collision" type="box" '
            f'class="platform" size="{_vec(table_size)}" '
            'rgba="0 0 0 0" friction="1 0.005 0.0001"/>',
            "        </body>",
        )
    )


def build_scene_xml(
    config: ScrewTaskConfig = DEFAULT_CONFIG,
    screw_xy_m: Optional[Sequence[float]] = None,
    *,
    resource_prefix: str = ".",
    share_filename: str = "Share.xml",
    gripper_filename: str = "Gripper.xml",
    socket_mocap: bool = False,
) -> str:
    """Build a complete scene string.

    ``resource_prefix`` is relative to the eventual XML file. Use ``."`` for
    ``SCREW/Project.xml`` and ``..`` for ``SCREW/.runtime/scene.xml``.
    """
    validate_config(config)
    xy = config.default_screw_xy_m if screw_xy_m is None else screw_xy_m
    if len(xy) != 2 or any(not math.isfinite(float(v)) for v in xy):
        raise ValueError("screw_xy_m must contain two finite values")
    if math.hypot(float(xy[0]), float(xy[1])) > config.spawn_radius_m + 1e-12:
        raise ValueError("screw position is outside the 100 mm spawn disk")
    if not config.spawn_is_on_opening_side(
        (float(xy[0]), float(xy[1]))
    ):
        raise ValueError(
            "screw position is outside the opening-facing half disk"
        )
    if upright_screw_overlaps_socket(xy, config):
        raise ValueError("screw reset pose overlaps the tilted socket block")

    share_xml = _resource_path(resource_prefix, share_filename)
    gripper_xml = _resource_path(resource_prefix, gripper_filename)
    mesh_dir = _resource_path(resource_prefix, "mesh")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<mujoco model="SCREW_two_turn_task">\n'
        f'    <compiler angle="radian" meshdir="{mesh_dir}" '
        'balanceinertia="true" discardvisual="false" autolimits="true"/>\n'
        '    <option timestep="0.0005" solver="Newton" jacobian="dense" iterations="150" '
        'ls_iterations="20" tolerance="1e-10">\n'
        '        <flag warmstart="enable"/>\n'
        '    </option>\n'
        f'    <include file="{share_xml}"/>\n'
        f'{_socket_asset_xml(config)}\n'
        '    <worldbody>\n'
        '        <geom name="floor" pos="0 0 -0.1" size="15 15 1" '
        'type="plane" material="matplane" margin="0.001" condim="3" '
        'friction="1 0.005 0.0001" contype="0" conaffinity="0"/>\n'
        f'        <include file="{gripper_xml}"/>\n'
        f'{_table_body_xml(config)}\n'
        f'{_screw_body_xml(xy, config)}\n'
        f'{_socket_body_xml(config, mocap=socket_mocap)}\n'
        f'        <camera name="{CAMERA_NAMES[0]}" pos="0 -0.36 0.72" '
        'mode="targetbody" target="table"/>\n'
        f'        <camera name="{CAMERA_NAMES[1]}" pos="-0.40 0.28 0.47" '
        'mode="targetbody" target="table"/>\n'
        '        <light directional="true" diffuse="0.45 0.45 0.45" '
        'specular="0.25 0.25 0.25" castshadow="true" pos="0.1 0.1 0.8" '
        'dir="0.1 0.1 -1"/>\n'
        '    </worldbody>\n'
        '</mujoco>\n'
    )


def write_scene_xml(
    output_path: Optional[os.PathLike[str] | str] = None,
    config: ScrewTaskConfig = DEFAULT_CONFIG,
    screw_xy_m: Optional[Sequence[float]] = None,
    *,
    resource_prefix_override: Optional[str] = None,
    share_filename: str = "Share.xml",
    gripper_filename: str = "Gripper.xml",
    socket_mocap: bool = False,
) -> Path:
    """Write a runtime XML inside SCREW with correct relative dependencies."""
    path = (
        DEFAULT_RUNTIME_XML
        if output_path is None
        else Path(output_path).expanduser()
    )
    if not path.is_absolute():
        path = MODULE_DIR / path
    path = path.resolve()
    try:
        path.relative_to(MODULE_DIR)
    except ValueError as exc:
        raise ValueError(
            f"runtime scene must remain inside the movable SCREW folder: {path}"
        ) from exc
    resource_prefix = (
        os.path.relpath(MODULE_DIR, path.parent).replace(os.sep, "/")
        if resource_prefix_override is None
        else str(resource_prefix_override)
    )
    xml = build_scene_xml(
        config,
        screw_xy_m,
        resource_prefix=resource_prefix,
        share_filename=share_filename,
        gripper_filename=gripper_filename,
        socket_mocap=socket_mocap,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
    return path


def load_mujoco_model(
    config: ScrewTaskConfig = DEFAULT_CONFIG,
    screw_xy_m: Optional[Sequence[float]] = None,
):
    """Build and load the runtime model without depending on the caller's cwd."""
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "MuJoCo is unavailable; run with ../RL_env/bin/python"
        ) from exc
    return mujoco.MjModel.from_xml_path(
        str(write_scene_xml(config=config, screw_xy_m=screw_xy_m))
    )


if __name__ == "__main__":
    scene_path = write_scene_xml()
    print(scene_path)
