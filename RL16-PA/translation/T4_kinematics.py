#!/usr/bin/env python3
"""Gripper4 translation kinematics shared by control, RL and workspace tools.

This module is deliberately local and relocatable: the default ``Project.xml``
is resolved relative to this file, and XML includes are required to stay inside
the same project directory.  It does not import code from any other workspace.

Public entry points
-------------------
``GripperIK``
    XML-derived analytic forward/inverse kinematics for the four closed-chain
    fingers.
``IKTeacher``
    Stateful continuity wrapper suitable for an RL IK teacher.
``cube_tip_targets`` / ``cylindrical_to_xyz``
    Shared target generation for IK-only control and workspace scans.
``GripperIK.maximum_cylindrical_radius``
    XML-limit-aware search for the largest radius at a requested theta.
``parse_model_xml``
    Lightweight model metadata used by training and plotting programs.

Joint limits and actuator
order are read from the current XML rather than copied from the (potentially
stale) limits table in that document.  Every active actuator ``ctrlrange`` is
required to match its physical joint range exactly.

The current model contract is checked explicitly: every finger is attached as
``root -> linear -> armbase``; the linear joint translates along its own local
+z axis, every armbase joint rotates about module-local x, and every closed-chain
hinge rotates about module-local +y.  The slide plus the three rotary actuators
give each finger four active degrees of freedom.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_XML = MODULE_DIR / "Project.xml"

FINGERS: Tuple[str, ...] = ("U", "D", "L", "R")
JOINT_KINDS: Tuple[str, ...] = ("linear", "armbase", "back", "short", "elbow", "front")
ACTIVE_JOINT_KINDS: Tuple[str, ...] = ("linear", "armbase", "back", "front")
ANGULAR_JOINT_KINDS: Tuple[str, ...] = ("armbase", "back", "short", "elbow", "front")
PASSIVE_JOINT_KINDS: Tuple[str, ...] = ("short", "elbow")
LINEAR_IK_SEARCH_STEP_M = 0.001

TIP_GEOM_NAMES: Dict[str, str] = {finger: f"hard_tip_{finger}" for finger in FINGERS}
TIP_BODY_NAMES: Dict[str, str] = {finger: f"hard_tip_{finger}" for finger in FINGERS}
TIP_OFFSETS_UNIT: Dict[str, np.ndarray] = {
    "U": np.array([0.0, 1.0, 0.0], dtype=float),
    "D": np.array([0.0, -1.0, 0.0], dtype=float),
    "L": np.array([-1.0, 0.0, 0.0], dtype=float),
    "R": np.array([1.0, 0.0, 0.0], dtype=float),
}
PIN_SITE_PAIRS: Tuple[Tuple[str, str], ...] = tuple(
    (f"pin:{finger}_long", f"pin:{finger}_front") for finger in FINGERS
)

CUBE_BODY_NAME = "cube_5cm"
CUBE_JOINT_NAME = "joint:cube_5cm"
CUBE_GEOM_NAME = "cube_geom"
TABLE_BODY_NAME = "table"
TABLE_GEOM_NAME = "table_geom"

# Share.xml retains the historical twelve rotary actuator indices and appends
# the four linear stages.  Keeping the mapping here makes observation/action
# vectors stable for future RL programs.
EXPECTED_ACTUATOR_ORDER: Tuple[Tuple[str, str, str], ...] = (
    ("R", "back", "R_back"),
    ("R", "front", "R_front"),
    ("L", "front", "L_front"),
    ("L", "back", "L_back"),
    ("D", "back", "D_back"),
    ("D", "front", "D_front"),
    ("U", "front", "U_front"),
    ("U", "back", "U_back"),
    ("U", "armbase", "U_armbase"),
    ("D", "armbase", "D_armbase"),
    ("L", "armbase", "L_armbase"),
    ("R", "armbase", "R_armbase"),
    ("R", "linear", "R_linear"),
    ("U", "linear", "U_linear"),
    ("L", "linear", "L_linear"),
    ("D", "linear", "D_linear"),
)
CTRL_INDEX: Dict[Tuple[str, str], int] = {
    (finger, kind): index
    for index, (finger, kind, _name) in enumerate(EXPECTED_ACTUATOR_ORDER)
}
ACTUATOR_NAMES: Tuple[str, ...] = tuple(item[2] for item in EXPECTED_ACTUATOR_ORDER)
ACTUATOR_COUNT = len(ACTUATOR_NAMES)


class KinematicsError(RuntimeError):
    """Base error for invalid XML geometry or an IK failure."""


class IKUnreachableError(KinematicsError):
    """Raised when at least one requested fingertip target is unreachable."""


def _as_float_array(
    text: Optional[str],
    size: int,
    *,
    default: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if text is None:
        if default is None:
            raise KinematicsError(f"expected {size} numeric values, got no XML attribute")
        values = np.asarray(default, dtype=float)
    else:
        values = np.asarray([float(value) for value in text.replace(",", " ").split()], dtype=float)
    if values.size != size:
        raise KinematicsError(f"expected {size} numeric values, got {values.size}: {text!r}")
    return values.reshape(size)


def normalize(vector: Sequence[float], fallback: Optional[Sequence[float]] = None) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        if fallback is None:
            raise ValueError("zero-length vector cannot be normalized")
        return normalize(fallback)
    return value / norm


def normalize_quat(quaternion: Sequence[float]) -> np.ndarray:
    value = normalize(quaternion, fallback=(1.0, 0.0, 0.0, 0.0)).reshape(4)
    return -value if value[0] < 0.0 else value


def quat_to_rot(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a MuJoCo ``w x y z`` quaternion to a 3x3 rotation matrix."""
    w, x, y, z = normalize_quat(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quat(left)
    w2, x2, y2, z2 = normalize_quat(right)
    return normalize_quat(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def yaw_quat(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)


def rot2(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.array([[c, -s], [s, c]], dtype=float)


def rot_axis(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rodrigues rotation for a joint axis expressed in its local body frame."""
    x, y, z = normalize(axis)
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    wrapped = (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return float(wrapped) if np.ndim(angle) == 0 else wrapped


def minimum_jerk(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value * value)


def lerp(start: Sequence[float], end: Sequence[float], value: float) -> np.ndarray:
    return (1.0 - float(value)) * np.asarray(start, dtype=float) + float(value) * np.asarray(end, dtype=float)


def arange_inclusive(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("step must be positive")
    if stop < start:
        return np.zeros(0, dtype=float)
    count = int(math.floor((stop - start) / step + 1e-10))
    values = start + step * np.arange(count + 1, dtype=float)
    if values.size == 0 or values[-1] < stop - 1e-10:
        values = np.append(values, stop)
    return values


def cylindrical_to_xyz(
    z_m: float,
    r_m: float,
    theta: float,
    *,
    theta_radians: bool = False,
    origin_xy: Sequence[float] = (0.0, 0.0),
) -> np.ndarray:
    """Convert terminal input order ``z, r, theta`` to world XYZ."""
    if r_m < 0.0:
        raise ValueError("r must be non-negative")
    theta_rad = float(theta) if theta_radians else math.radians(float(theta))
    return np.array(
        [
            float(origin_xy[0]) + float(r_m) * math.cos(theta_rad),
            float(origin_xy[1]) + float(r_m) * math.sin(theta_rad),
            float(z_m),
        ],
        dtype=float,
    )


def xyz_to_cylindrical(
    xyz: Sequence[float],
    *,
    origin_xy: Sequence[float] = (0.0, 0.0),
    theta_radians: bool = False,
) -> Tuple[float, float, float]:
    value = np.asarray(xyz, dtype=float).reshape(3)
    dx = float(value[0]) - float(origin_xy[0])
    dy = float(value[1]) - float(origin_xy[1])
    theta = math.atan2(dy, dx)
    if not theta_radians:
        theta = math.degrees(theta)
    return float(value[2]), math.hypot(dx, dy), theta


def cube_tip_targets(
    center_world: Sequence[float],
    quaternion_world: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    *,
    distance: float,
    z_offset: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Return hard-tip sphere-center targets around a cube pose."""
    center = np.asarray(center_world, dtype=float).reshape(3)
    rotation = quat_to_rot(quaternion_world)
    vertical = rotation @ np.array([0.0, 0.0, float(z_offset)], dtype=float)
    return {
        finger: center + vertical + rotation @ (TIP_OFFSETS_UNIT[finger] * float(distance))
        for finger in FINGERS
    }


def circle_intersections(
    center_a: Sequence[float],
    radius_a: float,
    center_b: Sequence[float],
    radius_b: float,
    *,
    tolerance: float = 1e-10,
) -> List[np.ndarray]:
    a = np.asarray(center_a, dtype=float).reshape(2)
    b = np.asarray(center_b, dtype=float).reshape(2)
    delta = b - a
    distance = float(np.linalg.norm(delta))
    if distance < tolerance:
        return []
    if distance > radius_a + radius_b + tolerance:
        return []
    if distance < abs(radius_a - radius_b) - tolerance:
        return []
    along = (radius_a**2 - radius_b**2 + distance**2) / (2.0 * distance)
    height_sq = radius_a**2 - along**2
    if height_sq < -tolerance:
        return []
    height = math.sqrt(max(0.0, height_sq))
    midpoint = a + (along / distance) * delta
    normal = np.array([-delta[1], delta[0]], dtype=float) / distance
    if height <= tolerance:
        return [midpoint]
    return [midpoint + height * normal, midpoint - height * normal]


def _equivalent_angles_in_range(angle: float, limits: Sequence[float], tolerance: float = 1e-9) -> List[float]:
    low, high = float(limits[0]), float(limits[1])
    first = int(math.ceil((low - angle - tolerance) / (2.0 * math.pi)))
    last = int(math.floor((high - angle + tolerance) / (2.0 * math.pi)))
    values = []
    for turn in range(first, last + 1):
        value = angle + turn * 2.0 * math.pi
        if low - tolerance <= value <= high + tolerance:
            values.append(float(np.clip(value, low, high)))
    return values


def _require_element(parent: ET.Element, expression: str, label: str) -> ET.Element:
    element = parent.find(expression)
    if element is None:
        raise KinematicsError(f"XML element not found: {label}")
    return element


def _direct_child_body(parent: ET.Element, name: str) -> ET.Element:
    for child in parent.findall("body"):
        if child.get("name") == name:
            return child
    raise KinematicsError(f"body {name!r} is not a direct child of {parent.get('name')!r}")


def _body_pose(element: ET.Element) -> Tuple[np.ndarray, np.ndarray]:
    position = _as_float_array(element.get("pos"), 3, default=(0.0, 0.0, 0.0))
    quaternion = _as_float_array(element.get("quat"), 4, default=(1.0, 0.0, 0.0, 0.0))
    return position, quat_to_rot(quaternion)


def _safe_include_path(base_dir: Path, filename: str) -> Path:
    path = (base_dir / filename).resolve()
    try:
        path.relative_to(base_dir.resolve())
    except ValueError as exc:
        raise KinematicsError(f"external XML include is not allowed: {filename!r}") from exc
    if not path.is_file():
        raise KinematicsError(f"included XML file does not exist: {path}")
    return path


@dataclass(frozen=True)
class ActuatorSpec:
    name: str
    finger: str
    kind: str
    joint_name: str
    index: int
    ctrl_range: Tuple[float, float]
    joint_range: Tuple[float, float]
    effective_range: Tuple[float, float]


@dataclass(frozen=True)
class FingerGeometry:
    finger: str
    base_world: np.ndarray
    rotation_world: np.ndarray
    linear_axis: np.ndarray
    armbase_axis: np.ndarray
    O: np.ndarray
    M: np.ndarray
    l1: float
    l2: float
    l3: float
    c0: np.ndarray
    t0: np.ndarray
    alpha: float
    eta: float
    joint_names: Mapping[str, str]
    # Physical stops declared by each <joint range="...">.
    joint_ranges: Mapping[str, Tuple[float, float]]
    # Ranges actually usable by IK.  Active ranges are required to equal both
    # the joint range and actuator ctrlrange; passive ranges use the joint.
    effective_ranges: Mapping[str, Tuple[float, float]]

    @property
    def v0(self) -> np.ndarray:
        return self.t0 - self.c0

    @property
    def c_to_tip(self) -> float:
        return float(np.linalg.norm(self.v0))

    @property
    def armbase_axis_kind(self) -> str:
        if abs(float(self.armbase_axis[0])) > 1.0 - 1e-10:
            return "x"
        raise KinematicsError(
            f"{self.finger}: armbase axis must be module-local x, "
            f"got {self.armbase_axis.tolist()}"
        )

    @property
    def armbase_axis_world(self) -> np.ndarray:
        """Armbase hinge axis in the XML reference pose, expressed in world coordinates."""
        return self.rotation_world @ self.armbase_axis

    @property
    def linear_axis_world(self) -> np.ndarray:
        """Linear-stage axis expressed in world coordinates."""
        return self.rotation_world @ self.linear_axis


@dataclass(frozen=True)
class ModelInfo:
    project_xml: Path
    project_dir: Path
    gripper_xml: Path
    share_xml: Path
    timestep: float
    cube_initial_pos: np.ndarray
    cube_half_size: float
    hard_tip_radius: float
    table_top_z: float
    table_half_size: np.ndarray
    actuator_specs: Tuple[ActuatorSpec, ...]
    ctrl_ranges: np.ndarray
    fingers: Mapping[str, FingerGeometry]

    @property
    def contact_distance(self) -> float:
        return self.cube_half_size + self.hard_tip_radius

    @property
    def minimum_cube_center_z(self) -> float:
        return self.table_top_z + self.cube_half_size


@dataclass(frozen=True)
class FingerFK:
    finger: str
    A_local: np.ndarray
    B_local: np.ndarray
    C_long_local: np.ndarray
    C_front_local: np.ndarray
    tip_local: np.ndarray
    tip_world: np.ndarray
    loop_error: float


@dataclass(frozen=True)
class FingerSolution:
    finger: str
    q_linear: float
    q_armbase: float
    q_back: float
    q_short: float
    q_elbow: float
    q_front: float
    tip_world: np.ndarray
    tip_error: float
    loop_error: float
    score: float
    branch: Tuple[int, int, int, int]

    def joint_value(self, kind: str) -> float:
        return float(getattr(self, f"q_{kind}"))

    @property
    def joint_map(self) -> Dict[str, float]:
        return {kind: self.joint_value(kind) for kind in JOINT_KINDS}


@dataclass(frozen=True)
class IKSolution:
    ctrl: np.ndarray
    finger_solutions: Mapping[str, FingerSolution]
    solved_mask: np.ndarray
    failed_fingers: Tuple[str, ...]
    max_tip_error: float
    max_loop_error: float

    @property
    def success(self) -> bool:
        return not self.failed_fingers


@dataclass(frozen=True)
class CylindricalWorkspaceLimit:
    """Largest analytically feasible cylindrical radius found at one theta."""

    theta_deg: float
    radius: float
    z: float
    xyz: np.ndarray
    distance: float
    resolution: float


def _joint_name(finger: str, kind: str) -> str:
    return f"robot0:{kind}_joint_{finger}"


def _parse_joint(body: ET.Element, expected_name: str) -> ET.Element:
    for joint in body.findall("joint"):
        if joint.get("name") == expected_name:
            return joint
    raise KinematicsError(f"joint not found in body {body.get('name')!r}: {expected_name}")


def _joint_limits(joint: ET.Element) -> Tuple[float, float]:
    limits = _as_float_array(joint.get("range"), 2)
    return float(limits[0]), float(limits[1])


def _axis(joint: ET.Element) -> np.ndarray:
    return normalize(_as_float_array(joint.get("axis"), 3, default=(0.0, 0.0, 1.0)))


def _actuator_ctrl_limits(actuator: ET.Element, share_root: ET.Element) -> Tuple[float, float]:
    """Resolve an actuator ctrlrange, including a directly referenced default class."""
    ctrlrange = actuator.get("ctrlrange")
    if ctrlrange is None:
        class_name = actuator.get("class")
        if class_name:
            for default in share_root.iter("default"):
                if default.get("class") != class_name:
                    continue
                template = default.find(actuator.tag)
                if template is not None:
                    ctrlrange = template.get("ctrlrange")
                break
    if ctrlrange is None:
        raise KinematicsError(
            f"actuator {actuator.get('name')!r} has no explicit or class-derived ctrlrange"
        )
    limits = _as_float_array(ctrlrange, 2)
    if not float(limits[0]) < float(limits[1]):
        raise KinematicsError(
            f"actuator {actuator.get('name')!r} has invalid ctrlrange {limits.tolist()}"
        )
    return float(limits[0]), float(limits[1])


def parse_model_xml(project_xml: str | os.PathLike[str] = DEFAULT_XML) -> ModelInfo:
    """Parse all geometry needed by analytic IK from the local XML files."""
    project_path = Path(project_xml).expanduser().resolve()
    try:
        project_path.relative_to(next(parent for parent in Path(__file__).resolve().parents if (parent / "models" / "cad").is_dir()))
    except ValueError as exc:
        raise KinematicsError(
            f"Project XML must stay inside {MODULE_DIR}: {project_path}"
        ) from exc
    if not project_path.is_file():
        raise FileNotFoundError(f"Project XML not found: {project_path}")
    project_dir = project_path.parent
    project_root = ET.parse(project_path).getroot()

    include_paths: Dict[str, Path] = {}
    for include in project_root.iter("include"):
        filename = include.get("file")
        if filename:
            include_paths[Path(filename).name.lower()] = _safe_include_path(project_dir, filename)
    try:
        share_path = include_paths["share.xml"]
        gripper_path = include_paths["gripper.xml"]
    except KeyError as exc:
        raise KinematicsError("Project.xml must locally include Share.xml and Gripper.xml") from exc

    share_root = ET.parse(share_path).getroot()
    gripper_root = ET.parse(gripper_path).getroot()

    compiler = project_root.find("compiler")
    mesh_dir_name = compiler.get("meshdir", ".") if compiler is not None else "."
    mesh_dir = (project_dir / mesh_dir_name).resolve()
    try:
        mesh_dir.relative_to(next(parent for parent in Path(__file__).resolve().parents if (parent / "models" / "cad").is_dir()))
    except ValueError as exc:
        raise KinematicsError(
            f"mesh directory must stay inside {MODULE_DIR}: {mesh_dir}"
        ) from exc
    if not mesh_dir.is_dir():
        raise KinematicsError(f"mesh directory does not exist: {mesh_dir}")
    for mesh in share_root.findall(".//mesh"):
        filename = mesh.get("file")
        if not filename:
            continue
        mesh_path = (mesh_dir / filename).resolve()
        try:
            mesh_path.relative_to(next(parent for parent in Path(__file__).resolve().parents if (parent / "models" / "cad").is_dir()))
        except ValueError as exc:
            raise KinematicsError(
                f"mesh file must stay inside {MODULE_DIR}: {filename!r}"
            ) from exc
        if not mesh_path.is_file():
            raise KinematicsError(f"mesh file does not exist: {mesh_path}")

    option = project_root.find("option")
    timestep = float(option.get("timestep", "0.002")) if option is not None else 0.002
    worldbody = _require_element(project_root, "worldbody", "Project.xml/worldbody")
    table_body = _require_element(worldbody, f"body[@name='{TABLE_BODY_NAME}']", TABLE_BODY_NAME)
    table_geom = _require_element(table_body, f"geom[@name='{TABLE_GEOM_NAME}']", TABLE_GEOM_NAME)
    table_pos, table_rotation = _body_pose(table_body)
    table_geom_pos = _as_float_array(table_geom.get("pos"), 3, default=(0.0, 0.0, 0.0))
    table_size = _as_float_array(table_geom.get("size"), 3)
    table_center = table_pos + table_rotation @ table_geom_pos
    table_top_z = float(table_center[2] + table_size[2])

    cube_body = _require_element(worldbody, f"body[@name='{CUBE_BODY_NAME}']", CUBE_BODY_NAME)
    cube_geom = _require_element(cube_body, f"geom[@name='{CUBE_GEOM_NAME}']", CUBE_GEOM_NAME)
    cube_initial_pos = _as_float_array(cube_body.get("pos"), 3, default=(0.0, 0.0, 0.0))
    cube_size = _as_float_array(cube_geom.get("size"), 3)

    root_body = _require_element(gripper_root, ".//body[@name='root']", "Gripper.xml/root")
    root_pos, root_rotation = _body_pose(root_body)

    fingers: Dict[str, FingerGeometry] = {}
    joint_elements: Dict[str, ET.Element] = {}
    hard_tip_radius: Optional[float] = None

    for finger in FINGERS:
        linear_body = _direct_child_body(root_body, f"robot0:linear_{finger}")
        arm_body = _direct_child_body(linear_body, f"robot0:armbase_{finger}")
        back_body = _direct_child_body(arm_body, f"robot0:back_{finger}")
        front_body = _direct_child_body(arm_body, f"robot0:front_{finger}")
        short_body = _direct_child_body(back_body, f"robot0:short_{finger}")
        long_body = _direct_child_body(short_body, f"robot0:long_{finger}")
        hard_tip_body = _direct_child_body(long_body, f"hard_tip_{finger}")

        linear_pos, linear_rotation = _body_pose(linear_body)
        arm_pos, arm_rotation = _body_pose(arm_body)
        back_pos, back_rotation = _body_pose(back_body)
        front_pos, front_rotation = _body_pose(front_body)
        short_pos, _ = _body_pose(short_body)
        long_pos, long_rotation = _body_pose(long_body)
        hard_tip_pos, _ = _body_pose(hard_tip_body)

        long_site = _require_element(long_body, f"site[@name='pin:{finger}_long']", f"pin:{finger}_long")
        front_site = _require_element(front_body, f"site[@name='pin:{finger}_front']", f"pin:{finger}_front")
        long_site_pos = _as_float_array(long_site.get("pos"), 3, default=(0.0, 0.0, 0.0))
        front_site_pos = _as_float_array(front_site.get("pos"), 3, default=(0.0, 0.0, 0.0))

        tip_geom = _require_element(hard_tip_body, f"geom[@name='hard_tip_{finger}']", f"hard_tip_{finger}")
        radius = float(_as_float_array(tip_geom.get("size"), 1)[0])
        if hard_tip_radius is None:
            hard_tip_radius = radius
        elif not math.isclose(hard_tip_radius, radius, abs_tol=1e-12):
            raise KinematicsError("all hard-tip sphere radii must match")

        bodies = {
            "linear": linear_body,
            "armbase": arm_body,
            "back": back_body,
            "short": short_body,
            "elbow": long_body,
            "front": front_body,
        }
        names = {kind: _joint_name(finger, kind) for kind in JOINT_KINDS}
        joints = {kind: _parse_joint(bodies[kind], names[kind]) for kind in JOINT_KINDS}
        joint_elements.update({names[kind]: joints[kind] for kind in JOINT_KINDS})

        # Latest XML contract: each stage slides along its own +z axis, all
        # armbases use module-local x, and all planar closed-chain hinges use
        # module-local +y.
        if joints["linear"].get("type") != "slide":
            raise KinematicsError(f"{names['linear']} must be a slide joint")
        for kind in ANGULAR_JOINT_KINDS:
            if joints[kind].get("type") != "hinge":
                raise KinematicsError(f"{names[kind]} must be a hinge joint")
        linear_axis_in_stage = _axis(joints["linear"])
        if not np.allclose(linear_axis_in_stage, (0.0, 0.0, 1.0), atol=1e-10):
            raise KinematicsError(
                f"{names['linear']} axis must be the linear-stage local +z axis, "
                f"got {linear_axis_in_stage.tolist()}"
            )
        armbase_axis = _axis(joints["armbase"])
        axis_is_x = np.allclose(np.abs(armbase_axis), (1.0, 0.0, 0.0), atol=1e-10)
        if not axis_is_x:
            raise KinematicsError(
                f"{names['armbase']} axis must be parallel to module-local x, "
                f"got {armbase_axis.tolist()}"
            )
        for kind in ("back", "short", "elbow", "front"):
            if not np.allclose(_axis(joints[kind]), (0.0, 1.0, 0.0), atol=1e-10):
                raise KinematicsError(f"{names[kind]} axis must be local +y for the analytic IK")

        # These checks turn silent XML/derivation drift into an actionable error.
        if abs(short_pos[2]) > 1e-10 or abs(long_pos[2]) > 1e-10:
            raise KinematicsError(f"{finger}: short/long link offsets must lie along local x")
        if abs(front_site_pos[2]) > 1e-10:
            raise KinematicsError(f"{finger}: front pin offset must lie along local x")
        alpha = math.atan2(float(back_rotation[0, 2]), float(back_rotation[0, 0]))
        front_alpha = math.atan2(float(front_rotation[0, 2]), float(front_rotation[0, 0]))
        eta = math.atan2(float(long_rotation[0, 2]), float(long_rotation[0, 0]))
        if not math.isclose(alpha, front_alpha, abs_tol=1e-10):
            raise KinematicsError(f"{finger}: back and front fixed installation angles differ")

        # At q_linear=0, compose root -> linear-stage fixed pose -> armbase
        # fixed pose.  The slide axis is converted into the armbase reference
        # frame so FK/IK can use a single local coordinate system.
        base_world = root_pos + root_rotation @ (
            linear_pos + linear_rotation @ arm_pos
        )
        rotation_world = root_rotation @ linear_rotation @ arm_rotation
        linear_axis = arm_rotation.T @ linear_axis_in_stage
        joint_ranges = {kind: _joint_limits(joints[kind]) for kind in JOINT_KINDS}
        geometry = FingerGeometry(
            finger=finger,
            base_world=base_world,
            rotation_world=rotation_world,
            linear_axis=linear_axis,
            armbase_axis=armbase_axis,
            O=np.array([back_pos[0], back_pos[2]], dtype=float),
            M=np.array([front_pos[0], front_pos[2]], dtype=float),
            l1=float(short_pos[0]),
            l2=float(long_pos[0]),
            l3=float(front_site_pos[0]),
            c0=np.array([long_site_pos[0], long_site_pos[2]], dtype=float),
            t0=np.array([hard_tip_pos[0], hard_tip_pos[2]], dtype=float),
            alpha=float(alpha),
            eta=float(eta),
            joint_names=names,
            joint_ranges=joint_ranges,
            effective_ranges=joint_ranges.copy(),
        )

        # Verify the ignored y offsets really cancel as documented.
        a3 = back_pos + back_rotation @ short_pos
        b3 = a3 + back_rotation @ long_pos
        tip3 = b3 + back_rotation @ long_rotation @ hard_tip_pos
        c_long3 = b3 + back_rotation @ long_rotation @ long_site_pos
        c_front3 = front_pos + front_rotation @ front_site_pos
        if abs(float(tip3[1])) > 1e-9 or np.linalg.norm(c_long3 - c_front3) > 2e-9:
            raise KinematicsError(f"{finger}: XML y offsets or zero-pose closure no longer match the IK derivation")
        fingers[finger] = geometry

    # The controller and workspace tools assume four identical modules related
    # only by their base rotation.  Reject an accidental one-finger edit instead
    # of producing an apparently asymmetric workspace from stale geometry.
    reference = fingers[FINGERS[0]]
    vector_fields = ("linear_axis", "armbase_axis", "O", "M", "c0", "t0")
    scalar_fields = ("l1", "l2", "l3", "alpha", "eta")
    for finger in FINGERS[1:]:
        geometry = fingers[finger]
        for field in vector_fields:
            if not np.allclose(getattr(geometry, field), getattr(reference, field), atol=1e-10):
                raise KinematicsError(f"{finger}: local geometry field {field} breaks four-finger symmetry")
        for field in scalar_fields:
            if not math.isclose(getattr(geometry, field), getattr(reference, field), abs_tol=1e-10):
                raise KinematicsError(f"{finger}: local geometry field {field} breaks four-finger symmetry")
        for kind in JOINT_KINDS:
            if not np.allclose(geometry.joint_ranges[kind], reference.joint_ranges[kind], atol=1e-12):
                raise KinematicsError(f"{finger}: {kind} joint range breaks four-finger symmetry")

    actuator_parent = _require_element(share_root, "actuator", "Share.xml/actuator")
    actuator_specs: List[ActuatorSpec] = []
    effective_ranges = {
        finger: dict(geometry.joint_ranges)
        for finger, geometry in fingers.items()
    }
    expected_names = list(ACTUATOR_NAMES)
    actual_names = [element.get("name", "") for element in list(actuator_parent)]
    if actual_names != expected_names:
        raise KinematicsError(
            "Share.xml actuator order changed; expected "
            f"{expected_names}, got {actual_names}. Update EXPECTED_ACTUATOR_ORDER intentionally."
        )
    for index, (element, (finger, kind, name)) in enumerate(zip(list(actuator_parent), EXPECTED_ACTUATOR_ORDER)):
        joint_name = element.get("joint", "")
        expected_joint = _joint_name(finger, kind)
        if joint_name != expected_joint:
            raise KinematicsError(f"actuator {name!r} must control {expected_joint!r}, got {joint_name!r}")
        joint_limits = _joint_limits(joint_elements[joint_name])
        ctrl_limits = _actuator_ctrl_limits(element, share_root)
        if not np.allclose(ctrl_limits, joint_limits, atol=1e-12):
            raise KinematicsError(
                f"Share.xml actuator {name!r} ctrlrange {ctrl_limits} must exactly match "
                f"Gripper.xml joint range {joint_limits}"
            )
        effective_limits = joint_limits
        effective_ranges[finger][kind] = effective_limits
        actuator_specs.append(
            ActuatorSpec(
                name,
                finger,
                kind,
                joint_name,
                index,
                ctrl_limits,
                joint_limits,
                effective_limits,
            )
        )

    ctrl_range_by_kind: Dict[str, Tuple[float, float]] = {}
    for spec in actuator_specs:
        previous = ctrl_range_by_kind.setdefault(spec.kind, spec.ctrl_range)
        if not np.allclose(spec.ctrl_range, previous, atol=1e-12):
            raise KinematicsError(
                f"actuator {spec.name!r} ctrlrange breaks four-finger {spec.kind} symmetry"
            )

    fingers = {
        finger: replace(geometry, effective_ranges=effective_ranges[finger])
        for finger, geometry in fingers.items()
    }

    ctrl_ranges = np.asarray([spec.ctrl_range for spec in actuator_specs], dtype=float)
    return ModelInfo(
        project_xml=project_path,
        project_dir=project_dir,
        gripper_xml=gripper_path,
        share_xml=share_path,
        timestep=timestep,
        cube_initial_pos=cube_initial_pos,
        cube_half_size=float(cube_size[0]),
        hard_tip_radius=float(hard_tip_radius if hard_tip_radius is not None else 0.0),
        table_top_z=table_top_z,
        table_half_size=table_size.copy(),
        actuator_specs=tuple(actuator_specs),
        ctrl_ranges=ctrl_ranges,
        fingers=fingers,
    )


class GripperIK:
    """Analytic closed-chain IK built directly from the current local XML."""

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self.fingers = model_info.fingers
        self.ctrl_ranges = np.asarray(model_info.ctrl_ranges, dtype=float).copy()
        base_center_xy = np.mean(
            np.vstack([geometry.base_world[:2] for geometry in self.fingers.values()]),
            axis=0,
        )
        contact_directions: Dict[str, np.ndarray] = {}
        for finger, geometry in self.fingers.items():
            local_x_world = geometry.rotation_world @ np.array([1.0, 0.0, 0.0])
            horizontal = np.array([local_x_world[0], local_x_world[1], 0.0], dtype=float)
            if np.linalg.norm(horizontal) < 1e-12:
                radial = geometry.base_world[:2] - base_center_xy
                horizontal = np.array([radial[0], radial[1], 0.0], dtype=float)
            contact_directions[finger] = normalize(horizontal)
        self.contact_directions_world = contact_directions

    @classmethod
    def from_xml(cls, project_xml: str | os.PathLike[str] = DEFAULT_XML) -> "GripperIK":
        return cls(parse_model_xml(project_xml))

    def with_physical_joint_ranges(self) -> "GripperIK":
        """Return a diagnostic solver that ignores narrower actuator ctrlranges.

        Its solutions must never be sent directly to actuators.  The controller
        uses it only to explain whether a target is mechanically reachable but
        excluded by an actuator command limit.
        """
        physical_fingers = {
            finger: replace(geometry, effective_ranges=dict(geometry.joint_ranges))
            for finger, geometry in self.fingers.items()
        }
        return GripperIK(replace(self.model_info, fingers=physical_fingers))

    def active_range_summary(self) -> str:
        """Return the active XML joint/actuator limits with physical units."""
        geometry = self.fingers[FINGERS[0]]
        parts = []
        for kind in ACTIVE_JOINT_KINDS:
            low, high = geometry.effective_ranges[kind]
            if kind == "linear":
                parts.append(f"{kind}=[{low:+.4f}, {high:+.4f}] m")
            else:
                parts.append(
                    f"{kind}=[{math.degrees(low):+.1f}, {math.degrees(high):+.1f}] deg"
                )
        return ", ".join(parts)

    def _coerce_joint_map(self, joint_values: Mapping[str, float] | FingerSolution) -> Dict[str, float]:
        if isinstance(joint_values, FingerSolution):
            return joint_values.joint_map
        missing = [kind for kind in JOINT_KINDS if kind not in joint_values]
        if missing:
            raise ValueError(f"missing joint values: {missing}")
        return {kind: float(joint_values[kind]) for kind in JOINT_KINDS}

    def finger_forward(
        self,
        finger: str,
        joint_values: Mapping[str, float] | FingerSolution,
    ) -> FingerFK:
        geometry = self.fingers[finger.upper()]
        q = self._coerce_joint_map(joint_values)
        theta_back = -(geometry.alpha + q["back"])
        theta_short = -(geometry.alpha + q["back"] + q["short"])
        gamma = -(
            geometry.alpha
            + q["back"]
            + q["short"]
            + geometry.eta
            + q["elbow"]
        )
        theta_front = -(geometry.alpha + q["front"])
        point_a = geometry.O + rot2(theta_back) @ np.array([geometry.l1, 0.0])
        point_b = point_a + rot2(theta_short) @ np.array([geometry.l2, 0.0])
        c_long = point_b + rot2(gamma) @ geometry.c0
        tip = point_b + rot2(gamma) @ geometry.t0
        c_front = geometry.M + rot2(theta_front) @ np.array([geometry.l3, 0.0])

        point_pre_arm = rot_axis(geometry.armbase_axis, q["armbase"]) @ np.array(
            [tip[0], 0.0, tip[1]],
            dtype=float,
        )
        tip_world = geometry.base_world + geometry.rotation_world @ (
            geometry.linear_axis * q["linear"] + point_pre_arm
        )
        return FingerFK(
            finger=geometry.finger,
            A_local=point_a,
            B_local=point_b,
            C_long_local=c_long,
            C_front_local=c_front,
            tip_local=tip,
            tip_world=tip_world,
            loop_error=float(np.linalg.norm(c_long - c_front)),
        )

    def _previous_joint_values(
        self,
        finger: str,
        previous_ctrl: Optional[Sequence[float]],
        previous_solution: Optional[FingerSolution],
    ) -> Optional[Dict[str, float]]:
        previous: Dict[str, float] = {}
        if previous_solution is not None:
            previous.update(previous_solution.joint_map)
        if previous_ctrl is not None:
            ctrl = np.asarray(previous_ctrl, dtype=float).reshape(ACTUATOR_COUNT)
            for kind in ACTIVE_JOINT_KINDS:
                previous[kind] = float(ctrl[CTRL_INDEX[(finger, kind)]])
        return previous or None

    def _candidate_score(
        self,
        geometry: FingerGeometry,
        values: Mapping[str, float],
        previous: Optional[Mapping[str, float]],
    ) -> float:
        score = 0.0
        for kind in JOINT_KINDS:
            low, high = geometry.effective_ranges[kind]
            span = max(high - low, 1e-9)
            if previous is not None and kind in previous:
                delta = (
                    float(values[kind] - previous[kind])
                    if kind == "linear"
                    else float(wrap_to_pi(values[kind] - previous[kind]))
                )
                weight = 2.0 if kind in ACTIVE_JOINT_KINDS else 0.15
                score += weight * (delta / span) ** 2
            else:
                midpoint = 0.5 * (low + high)
                weight = 0.35 if kind in ACTIVE_JOINT_KINDS else 0.08
                score += weight * ((values[kind] - midpoint) / span) ** 2
        score += 0.01 * abs(float(values["armbase"]))
        return float(score)

    def _solve_with_arm_candidates(
        self,
        finger: str,
        geometry: FingerGeometry,
        target: np.ndarray,
        target_zero: np.ndarray,
        previous: Optional[Mapping[str, float]],
        q_linear: float,
        arm_candidates: Sequence[float],
        tolerance: float,
        *,
        linear_branch: int,
    ) -> List[FingerSolution]:
        candidates: List[FingerSolution] = []
        for arm_branch, q_armbase in enumerate(arm_candidates):
            planar = rot_axis(geometry.armbase_axis, -q_armbase) @ target_zero
            if abs(float(planar[1])) > tolerance:
                continue
            target_tip = np.array([planar[0], planar[2]], dtype=float)
            c_points = circle_intersections(geometry.M, geometry.l3, target_tip, geometry.c_to_tip)
            for circle_branch, point_c in enumerate(c_points):
                tip_vector = target_tip - point_c
                gamma = float(
                    wrap_to_pi(
                        math.atan2(float(tip_vector[1]), float(tip_vector[0]))
                        - math.atan2(float(geometry.v0[1]), float(geometry.v0[0]))
                    )
                )
                point_b = point_c - rot2(gamma) @ geometry.c0
                displacement = point_b - geometry.O
                cosine_short = (
                    float(displacement @ displacement) - geometry.l1**2 - geometry.l2**2
                ) / (2.0 * geometry.l1 * geometry.l2)
                if cosine_short < -1.0 - 1e-9 or cosine_short > 1.0 + 1e-9:
                    continue
                cosine_short = float(np.clip(cosine_short, -1.0, 1.0))
                sine_magnitude = math.sqrt(max(0.0, 1.0 - cosine_short**2))
                sine_options = [sine_magnitude] if sine_magnitude < 1e-12 else [sine_magnitude, -sine_magnitude]

                for short_branch, sine_short in enumerate(sine_options):
                    theta_short_relative = math.atan2(sine_short, cosine_short)
                    theta_back = math.atan2(float(displacement[1]), float(displacement[0])) - math.atan2(
                        geometry.l2 * sine_short,
                        geometry.l1 + geometry.l2 * cosine_short,
                    )
                    theta_front = math.atan2(
                        float(point_c[1] - geometry.M[1]),
                        float(point_c[0] - geometry.M[0]),
                    )
                    raw_values = {
                        "linear": q_linear,
                        "armbase": q_armbase,
                        "back": -theta_back - geometry.alpha,
                        "short": -theta_short_relative,
                        "front": -theta_front - geometry.alpha,
                    }
                    raw_values["elbow"] = (
                        -gamma
                        - geometry.alpha
                        - raw_values["back"]
                        - raw_values["short"]
                        - geometry.eta
                    )
                    options = []
                    for kind in JOINT_KINDS:
                        if kind == "linear":
                            low, high = geometry.effective_ranges[kind]
                            values_in_range = (
                                [float(q_linear)]
                                if low - 1e-12 <= q_linear <= high + 1e-12
                                else []
                            )
                        else:
                            values_in_range = _equivalent_angles_in_range(
                                raw_values[kind],
                                geometry.effective_ranges[kind],
                            )
                        options.append(values_in_range)
                    if any(not values for values in options):
                        continue
                    for equivalent in product(*options):
                        values = dict(zip(JOINT_KINDS, equivalent))
                        forward = self.finger_forward(finger, values)
                        tip_error = float(np.linalg.norm(forward.tip_world - target))
                        if tip_error > tolerance or forward.loop_error > tolerance:
                            continue
                        score = self._candidate_score(geometry, values, previous)
                        candidates.append(
                            FingerSolution(
                                finger=finger,
                                q_linear=values["linear"],
                                q_armbase=values["armbase"],
                                q_back=values["back"],
                                q_short=values["short"],
                                q_elbow=values["elbow"],
                                q_front=values["front"],
                                tip_world=forward.tip_world,
                                tip_error=tip_error,
                                loop_error=forward.loop_error,
                                score=score,
                                branch=(linear_branch, arm_branch, circle_branch, short_branch),
                            )
                        )

        return candidates

    def target_out_of_plane_error(self, finger: str, target_world: Sequence[float]) -> float:
        """Return the invariant plane error for the current spatial finger model.

        There is no configuration-independent plane constraint because the
        local-x armbase can tilt the local-y linkage plane; the local-z slide
        additionally shifts that plane.  Finite linear/rotary ranges are still
        enforced by the full IK solver.  This compatibility method therefore
        returns zero without claiming that every target is reachable.
        """
        finger = finger.upper()
        if finger not in self.fingers:
            raise KeyError(f"unknown finger {finger!r}")
        np.asarray(target_world, dtype=float).reshape(3)
        return 0.0

    def solve_finger_target(
        self,
        finger: str,
        target_world: Sequence[float],
        *,
        previous_ctrl: Optional[Sequence[float]] = None,
        previous_solution: Optional[FingerSolution] = None,
        tolerance: float = 2e-8,
    ) -> Optional[FingerSolution]:
        finger = finger.upper()
        geometry = self.fingers[finger]
        target = np.asarray(target_world, dtype=float).reshape(3)
        target_zero = geometry.rotation_world.T @ (target - geometry.base_world)
        previous = self._previous_joint_values(finger, previous_ctrl, previous_solution)
        low_linear, high_linear = geometry.effective_ranges["linear"]
        low_arm, high_arm = geometry.effective_ranges["armbase"]

        def solve_at_linear(q_linear: float, linear_branch: int) -> List[FingerSolution]:
            adjusted = target_zero - geometry.linear_axis * float(q_linear)
            yz_norm = math.hypot(float(adjusted[1]), float(adjusted[2]))
            if yz_norm < 1e-12:
                raw_arm_candidates = [
                    float(previous.get("armbase", 0.0)) if previous else 0.0
                ]
            else:
                axis_sign = float(np.sign(geometry.armbase_axis[0]))
                negative_z_angle = math.atan2(float(adjusted[1]), -float(adjusted[2]))
                raw_arm_candidates = [
                    negative_z_angle / axis_sign,
                    (negative_z_angle + math.pi) / axis_sign,
                ]

            arm_candidates: List[float] = []
            for raw in raw_arm_candidates:
                for value in _equivalent_angles_in_range(raw, (low_arm, high_arm)):
                    if not any(abs(value - existing) < 1e-10 for existing in arm_candidates):
                        arm_candidates.append(value)
            return self._solve_with_arm_candidates(
                finger,
                geometry,
                target,
                adjusted,
                previous,
                float(q_linear),
                arm_candidates,
                tolerance,
                linear_branch=linear_branch,
            )

        preferred_linear = (
            float(np.clip(previous["linear"], low_linear, high_linear))
            if previous is not None and "linear" in previous
            else high_linear
        )
        priority_values = (
            preferred_linear,
            high_linear,
            0.5 * (low_linear + high_linear),
            low_linear,
        )
        tried: List[float] = []
        branch = 0
        for q_linear in priority_values:
            if any(abs(q_linear - value) < 1e-12 for value in tried):
                continue
            tried.append(float(q_linear))
            candidates = solve_at_linear(float(q_linear), branch)
            branch += 1
            if candidates:
                return min(
                    candidates,
                    key=lambda candidate: (
                        candidate.score,
                        candidate.tip_error,
                        candidate.loop_error,
                    ),
                )

        # The slide makes the IK one-dimensionally redundant.  If the continuity
        # candidates fail, search the remaining 50 mm stroke on the same 1 mm
        # resolution used by the workspace scanner, ordered nearest to the
        # preferred command.  Every accepted candidate is still an exact
        # closed-chain analytic solution at its sampled slide coordinate.
        sample_count = max(
            1,
            int(math.ceil((high_linear - low_linear) / LINEAR_IK_SEARCH_STEP_M)),
        )
        fallback_values = sorted(
            np.linspace(low_linear, high_linear, sample_count + 1),
            key=lambda value: abs(float(value) - preferred_linear),
        )
        for q_linear_value in fallback_values:
            q_linear = float(q_linear_value)
            if any(abs(q_linear - value) < 1e-12 for value in tried):
                continue
            tried.append(q_linear)
            candidates = solve_at_linear(q_linear, branch)
            branch += 1
            if candidates:
                return min(
                    candidates,
                    key=lambda candidate: (
                        candidate.score,
                        candidate.tip_error,
                        candidate.loop_error,
                    ),
                )
        return None

    def finger_target_feasible(self, finger: str, target_world: Sequence[float]) -> bool:
        return self.solve_finger_target(finger, target_world) is not None

    def solve_tip_targets(
        self,
        targets_world: Mapping[str, Sequence[float]],
        *,
        previous_ctrl: Optional[Sequence[float]] = None,
        previous_solutions: Optional[Mapping[str, FingerSolution]] = None,
        strict: bool = False,
    ) -> IKSolution:
        if previous_ctrl is None:
            ctrl = np.clip(
                np.zeros(ACTUATOR_COUNT, dtype=float),
                self.ctrl_ranges[:, 0],
                self.ctrl_ranges[:, 1],
            )
        else:
            ctrl = np.clip(
                np.asarray(previous_ctrl, dtype=float).reshape(ACTUATOR_COUNT),
                self.ctrl_ranges[:, 0],
                self.ctrl_ranges[:, 1],
            )
        solutions: Dict[str, FingerSolution] = {}
        failed: List[str] = []
        solved_mask = np.zeros(len(FINGERS), dtype=bool)

        for index, finger in enumerate(FINGERS):
            if finger not in targets_world:
                raise ValueError(f"missing fingertip target for finger {finger}")
            solution = self.solve_finger_target(
                finger,
                targets_world[finger],
                previous_ctrl=ctrl,
                previous_solution=(previous_solutions or {}).get(finger),
            )
            if solution is None:
                failed.append(finger)
                continue
            solutions[finger] = solution
            solved_mask[index] = True
            for kind in ACTIVE_JOINT_KINDS:
                ctrl[CTRL_INDEX[(finger, kind)]] = solution.joint_value(kind)

        ctrl = np.clip(ctrl, self.ctrl_ranges[:, 0], self.ctrl_ranges[:, 1])
        max_tip = max((solution.tip_error for solution in solutions.values()), default=math.inf)
        max_loop = max((solution.loop_error for solution in solutions.values()), default=math.inf)
        result = IKSolution(
            ctrl=ctrl,
            finger_solutions=solutions,
            solved_mask=solved_mask,
            failed_fingers=tuple(failed),
            max_tip_error=float(max_tip),
            max_loop_error=float(max_loop),
        )
        if strict and failed:
            formatted = ", ".join(failed)
            raise IKUnreachableError(f"unreachable fingertip target(s): {formatted}")
        return result

    def solve_cube_pose(
        self,
        center_world: Sequence[float],
        quaternion_world: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        *,
        distance: Optional[float] = None,
        z_offset: float = 0.0,
        previous_ctrl: Optional[Sequence[float]] = None,
        previous_solutions: Optional[Mapping[str, FingerSolution]] = None,
        strict: bool = False,
    ) -> IKSolution:
        if distance is None:
            distance = self.model_info.contact_distance
        targets = self.cube_tip_targets(
            center_world,
            quaternion_world,
            distance=float(distance),
            z_offset=float(z_offset),
        )
        return self.solve_tip_targets(
            targets,
            previous_ctrl=previous_ctrl,
            previous_solutions=previous_solutions,
            strict=strict,
        )

    def cube_tip_targets(
        self,
        center_world: Sequence[float],
        quaternion_world: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        *,
        distance: float,
        z_offset: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Build cube targets from the current XML installation directions."""
        center = np.asarray(center_world, dtype=float).reshape(3)
        rotation = quat_to_rot(quaternion_world)
        vertical = rotation @ np.array([0.0, 0.0, float(z_offset)], dtype=float)
        return {
            finger: center
            + vertical
            + rotation @ (self.contact_directions_world[finger] * float(distance))
            for finger in FINGERS
        }

    def cube_pose_feasible(
        self,
        center_world: Sequence[float],
        quaternion_world: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        *,
        distance: Optional[float] = None,
        z_offset: float = 0.0,
    ) -> bool:
        target_distance = self.model_info.contact_distance if distance is None else float(distance)
        targets = self.cube_tip_targets(
            center_world,
            quaternion_world,
            distance=target_distance,
            z_offset=z_offset,
        )
        return all(
            self.finger_target_feasible(finger, targets[finger])
            for finger in FINGERS
        )

    def maximum_cylindrical_radius(
        self,
        theta: float,
        *,
        theta_radians: bool = False,
        origin_xy: Sequence[float] = (0.0, 0.0),
        quaternion_world: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        distance: Optional[float] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        r_max: Optional[float] = None,
        coarse_step: float = 0.005,
        resolution: float = 1e-6,
    ) -> CylindricalWorkspaceLimit:
        """Find the largest feasible ``r`` over ``z`` at a fixed ``theta``.

        The search uses the current XML geometry and every active/passive joint
        limit.  A coarse full-domain scan is followed by local refinements, so
        it does not assume that reachability is monotonic in either ``r`` or
        ``z``.  ``z_min`` defaults to the cube resting height above the table.
        """
        if coarse_step <= 0.0 or resolution <= 0.0:
            raise ValueError("workspace search steps must be positive")
        if resolution > coarse_step:
            raise ValueError("resolution must not exceed coarse_step")

        theta_rad = float(theta) if theta_radians else math.radians(float(theta))
        theta_deg = math.degrees(theta_rad)
        origin = np.asarray(origin_xy, dtype=float).reshape(2)
        quaternion = normalize_quat(quaternion_world)
        target_distance = (
            self.model_info.contact_distance if distance is None else float(distance)
        )
        if target_distance <= 0.0:
            raise ValueError("distance must be positive")

        # Triangle-inequality bounds derived from the current XML dimensions.
        # They deliberately overestimate the domain and therefore cannot clip
        # a valid IK solution before the actual joint-limit checks are applied.
        reach_by_finger = {
            finger: (
                float(np.linalg.norm(geometry.O))
                + geometry.l1
                + geometry.l2
                + float(np.linalg.norm(geometry.t0))
                + max(abs(value) for value in geometry.effective_ranges["linear"])
            )
            for finger, geometry in self.fingers.items()
        }
        if z_min is None:
            z_min = self.model_info.minimum_cube_center_z
        if z_max is None:
            z_max = max(
                float(self.fingers[finger].base_world[2]) + reach
                for finger, reach in reach_by_finger.items()
            ) + target_distance
        if r_max is None:
            r_max = max(
                float(np.linalg.norm(geometry.base_world[:2] - origin))
                + reach_by_finger[finger]
                + target_distance
                for finger, geometry in self.fingers.items()
            )

        global_z_low = float(z_min)
        global_z_high = float(z_max)
        global_r_high = float(r_max)
        if not global_z_low < global_z_high:
            raise ValueError("z_min must be smaller than z_max")
        if global_r_high <= 0.0:
            raise ValueError("r_max must be positive")

        def feasible(radius: float, z_value: float) -> bool:
            center = cylindrical_to_xyz(
                z_value,
                radius,
                theta_rad,
                theta_radians=True,
                origin_xy=origin,
            )
            return self.cube_pose_feasible(
                center,
                quaternion,
                distance=target_distance,
            )

        def search_level(
            r_low: float,
            r_high: float,
            z_low: float,
            z_high: float,
            step: float,
        ) -> Tuple[float, List[float]] | None:
            radii = arange_inclusive(r_low, r_high, step)[::-1]
            heights = arange_inclusive(z_low, z_high, step)
            for radius in radii:
                hits = [float(z_value) for z_value in heights if feasible(float(radius), float(z_value))]
                if hits:
                    return float(radius), hits
            return None

        step = float(coarse_step)
        found = search_level(0.0, global_r_high, global_z_low, global_z_high, step)
        if found is None:
            raise IKUnreachableError("no feasible cube center found in the requested workspace bounds")
        best_radius, best_heights = found

        while step > resolution:
            previous_step = step
            step = max(float(resolution), previous_step / 10.0)
            local_r_low = max(0.0, best_radius - previous_step)
            local_r_high = min(global_r_high, best_radius + 2.0 * previous_step)
            local_z_low = max(global_z_low, min(best_heights) - 2.0 * previous_step)
            local_z_high = min(global_z_high, max(best_heights) + 2.0 * previous_step)
            refined = search_level(local_r_low, local_r_high, local_z_low, local_z_high, step)
            if refined is None:
                raise KinematicsError("workspace refinement lost the feasible region")
            best_radius, best_heights = refined

        best_z = best_heights[len(best_heights) // 2]
        best_xyz = cylindrical_to_xyz(
            best_z,
            best_radius,
            theta_rad,
            theta_radians=True,
            origin_xy=origin,
        )
        return CylindricalWorkspaceLimit(
            theta_deg=float(theta_deg),
            radius=float(best_radius),
            z=float(best_z),
            xyz=best_xyz,
            distance=target_distance,
            resolution=float(step),
        )


class IKTeacher:
    """Stateful continuity wrapper for training environments and rollouts."""

    def __init__(
        self,
        ik: Optional[GripperIK] = None,
        *,
        project_xml: str | os.PathLike[str] = DEFAULT_XML,
        initial_ctrl: Optional[Sequence[float]] = None,
    ) -> None:
        self.ik = ik if ik is not None else GripperIK.from_xml(project_xml)
        self.reset(initial_ctrl)

    def reset(self, ctrl: Optional[Sequence[float]] = None) -> None:
        if ctrl is None:
            self.previous_ctrl = np.clip(
                np.zeros(ACTUATOR_COUNT, dtype=float),
                self.ik.ctrl_ranges[:, 0],
                self.ik.ctrl_ranges[:, 1],
            )
        else:
            self.previous_ctrl = np.clip(
                np.asarray(ctrl, dtype=float).reshape(ACTUATOR_COUNT),
                self.ik.ctrl_ranges[:, 0],
                self.ik.ctrl_ranges[:, 1],
            )
        self.previous_solutions: Dict[str, FingerSolution] = {}

    def commit(
        self,
        result: IKSolution,
        *,
        ctrl: Optional[Sequence[float]] = None,
    ) -> None:
        """Commit a successful IK result using the command actually sent.

        A rate-limited controller can pass its applied ``ctrl`` vector so the
        next continuity decision does not run ahead of the physical command.
        Passive-joint branch history is retained from ``result``.
        """
        if not result.success:
            raise ValueError("cannot commit an unsuccessful IK result")
        command = result.ctrl if ctrl is None else np.asarray(ctrl, dtype=float)
        self.previous_ctrl = np.clip(
            np.asarray(command, dtype=float).reshape(ACTUATOR_COUNT),
            self.ik.ctrl_ranges[:, 0],
            self.ik.ctrl_ranges[:, 1],
        )
        self.previous_solutions = dict(result.finger_solutions)

    def solve_tip_targets(
        self,
        targets_world: Mapping[str, Sequence[float]],
        *,
        strict: bool = False,
        update: bool = True,
    ) -> IKSolution:
        result = self.ik.solve_tip_targets(
            targets_world,
            previous_ctrl=self.previous_ctrl,
            previous_solutions=self.previous_solutions,
            strict=strict,
        )
        if update and result.success:
            self.commit(result)
        return result

    def solve_cube_pose(
        self,
        center_world: Sequence[float],
        quaternion_world: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        *,
        distance: Optional[float] = None,
        z_offset: float = 0.0,
        strict: bool = False,
        update: bool = True,
    ) -> IKSolution:
        targets = self.ik.cube_tip_targets(
            center_world,
            quaternion_world,
            distance=(self.ik.model_info.contact_distance if distance is None else float(distance)),
            z_offset=z_offset,
        )
        return self.solve_tip_targets(targets, strict=strict, update=update)


def format_vector(vector: Iterable[float], precision: int = 6) -> str:
    return "(" + ", ".join(f"{float(value):.{precision}f}" for value in vector) + ")"


__all__ = [
    "ACTIVE_JOINT_KINDS",
    "ACTUATOR_COUNT",
    "ACTUATOR_NAMES",
    "ANGULAR_JOINT_KINDS",
    "CUBE_BODY_NAME",
    "CUBE_GEOM_NAME",
    "CUBE_JOINT_NAME",
    "CTRL_INDEX",
    "CylindricalWorkspaceLimit",
    "DEFAULT_XML",
    "FINGERS",
    "IKSolution",
    "IKTeacher",
    "IKUnreachableError",
    "LINEAR_IK_SEARCH_STEP_M",
    "JOINT_KINDS",
    "ModelInfo",
    "PIN_SITE_PAIRS",
    "TABLE_GEOM_NAME",
    "TIP_GEOM_NAMES",
    "TIP_OFFSETS_UNIT",
    "FingerFK",
    "FingerGeometry",
    "FingerSolution",
    "GripperIK",
    "KinematicsError",
    "arange_inclusive",
    "cube_tip_targets",
    "cylindrical_to_xyz",
    "format_vector",
    "lerp",
    "minimum_jerk",
    "normalize",
    "normalize_quat",
    "parse_model_xml",
    "quat_multiply",
    "quat_to_rot",
    "wrap_to_pi",
    "xyz_to_cylindrical",
    "yaw_quat",
]


if __name__ == "__main__":
    info = parse_model_xml()
    ik = GripperIK(info)
    print(f"model: {info.project_xml}")
    print(f"timestep: {info.timestep:.6f} s, actuators: {len(info.actuator_specs)}")
    print(f"cube initial xyz: {format_vector(info.cube_initial_pos)}")
    print(f"table top z: {info.table_top_z:.6f} m")
    print(f"nominal contact distance: {info.contact_distance:.6f} m")
    print(f"active ranges: {ik.active_range_summary()}")
    for finger in FINGERS:
        geometry = info.fingers[finger]
        linear_low, linear_high = geometry.effective_ranges["linear"]
        arm_low, arm_high = geometry.effective_ranges["armbase"]
        print(
            f"{finger} linear: local={format_vector(geometry.linear_axis, 1)}, "
            f"world={format_vector(geometry.linear_axis_world, 1)}, "
            f"range=({linear_low:.4f}, {linear_high:.4f}) m; "
            f"armbase: local={format_vector(geometry.armbase_axis, 1)}, "
            f"world={format_vector(geometry.armbase_axis_world, 1)}, "
            f"range=({math.degrees(arm_low):.1f}, {math.degrees(arm_high):.1f}) deg"
        )
    for spec in info.actuator_specs:
        if np.allclose(spec.ctrl_range, spec.joint_range, atol=1e-12):
            continue
        if spec.kind == "linear":
            print(
                f"{spec.name}: joint={spec.joint_range} m, "
                f"ctrl={spec.ctrl_range} m, IK={spec.effective_range} m"
            )
        else:
            joint_deg = tuple(math.degrees(value) for value in spec.joint_range)
            ctrl_deg = tuple(math.degrees(value) for value in spec.ctrl_range)
            effective_deg = tuple(math.degrees(value) for value in spec.effective_range)
            print(
                f"{spec.name}: joint=({joint_deg[0]:.1f}, {joint_deg[1]:.1f}) deg, "
                f"ctrl=({ctrl_deg[0]:.1f}, {ctrl_deg[1]:.1f}) deg, "
                f"IK=({effective_deg[0]:.1f}, {effective_deg[1]:.1f}) deg"
            )
    initial_targets = ik.cube_tip_targets(
        info.cube_initial_pos,
        distance=info.contact_distance,
    )
    result = ik.solve_tip_targets(initial_targets)
    print(f"initial cube grasp analytically reachable: {result.success}")
    if result.success:
        print(f"ctrl: {format_vector(result.ctrl)}")
