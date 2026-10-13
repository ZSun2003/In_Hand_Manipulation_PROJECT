#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kinematics.py

Local analytic kinematics shared by the T6 controller and workspace scanner.
The module reads only Project.xml, Gripper.xml, Share.xml, meshes and optional
CSV data stored beside this file. Direct execution is pinned to the local
``RL_env`` virtual environment, so moving the whole SCREW folder does not
create a dependency on another gripper project.

The current XML hierarchy is ``root -> armbase -> linear -> fingers``:

  * armbase hinge is solved analytically from the linkage plane constraint;
  * the child linear stage moves along its own local +z axis and is searched on
    a configurable outer grid (default 1 mm);
  * back/front closed-chain IK is solved by two-circle intersections;
  * passive short/elbow joints are reconstructed to reject illegal branches.

All body, joint, site and actuator names are validated against the latest local
XML before the solver is constructed.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MODULE_DIR = Path(__file__).resolve().parent
LOCAL_ENV_DIR = MODULE_DIR.parent / "RL_env"
DEFAULT_XML = MODULE_DIR / "Project.xml"

# A direct invocation must use this folder's environment.  Importing this
# module never mutates the caller process; the two executable programs perform
# the same bootstrap before importing NumPy/MuJoCo.
if __name__ == "__main__" and Path(sys.prefix).resolve() != LOCAL_ENV_DIR.resolve():
    local_python = LOCAL_ENV_DIR / "bin" / "python"
    if not local_python.is_file():
        raise SystemExit(f"local virtual environment not found: {LOCAL_ENV_DIR}")
    os.execv(
        str(local_python),
        [str(local_python), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

import numpy as np

# =============================================================================
# Names and control order from Share.xml
# =============================================================================

FINGERS: Tuple[str, ...] = ("U", "D", "L", "R")
OBS_FINGER_ORDER: Tuple[str, ...] = ("U", "D", "L", "R")
JOINT_KINDS: Tuple[str, ...] = ("linear", "armbase", "back", "short", "elbow", "front")
ACTIVE_JOINT_KINDS: Tuple[str, ...] = ("linear", "armbase", "back", "front")
ANGULAR_JOINT_KINDS: Tuple[str, ...] = ("armbase", "back", "short", "elbow", "front")
PASSIVE_JOINT_KINDS: Tuple[str, ...] = ("short", "elbow")
LINEAR_IK_SEARCH_STEP_M = 0.001

# This order is the literal order in the latest local Share.xml and therefore
# also the order of MjData.ctrl / MjModel.actuator_* arrays.
EXPECTED_ACTUATOR_ORDER: Tuple[Tuple[str, str, str], ...] = (
    ("R", "linear", "R_linear"),
    ("R", "back", "R_back"),
    ("R", "front", "R_front"),
    ("L", "linear", "L_linear"),
    ("L", "front", "L_front"),
    ("L", "back", "L_back"),
    ("D", "linear", "D_linear"),
    ("D", "back", "D_back"),
    ("D", "front", "D_front"),
    ("U", "linear", "U_linear"),
    ("U", "front", "U_front"),
    ("U", "back", "U_back"),
    ("U", "armbase", "U_armbase"),
    ("D", "armbase", "D_armbase"),
    ("L", "armbase", "L_armbase"),
    ("R", "armbase", "R_armbase"),
)
ACTUATOR_NAMES: Tuple[str, ...] = tuple(spec[2] for spec in EXPECTED_ACTUATOR_ORDER)
ACTUATOR_COUNT = len(ACTUATOR_NAMES)

ACTUATOR_SPECS: Tuple[Tuple[str, str, str, int], ...] = (
    ("R", "linear", "robot0:linear_joint_R", 0),
    ("R", "back", "robot0:back_joint_R", 1),
    ("R", "front", "robot0:front_joint_R", 2),
    ("L", "linear", "robot0:linear_joint_L", 3),
    ("L", "front", "robot0:front_joint_L", 4),
    ("L", "back", "robot0:back_joint_L", 5),
    ("D", "linear", "robot0:linear_joint_D", 6),
    ("D", "back", "robot0:back_joint_D", 7),
    ("D", "front", "robot0:front_joint_D", 8),
    ("U", "linear", "robot0:linear_joint_U", 9),
    ("U", "front", "robot0:front_joint_U", 10),
    ("U", "back", "robot0:back_joint_U", 11),
    ("U", "armbase", "robot0:armbase_joint_U", 12),
    ("D", "armbase", "robot0:armbase_joint_D", 13),
    ("L", "armbase", "robot0:armbase_joint_L", 14),
    ("R", "armbase", "robot0:armbase_joint_R", 15),
)

CTRL_INDEX: Dict[Tuple[str, str], int] = {
    (finger, which): index for finger, which, _joint, index in ACTUATOR_SPECS
}
JOINT_TO_CTRL_INDEX: Dict[str, int] = {joint: index for _finger, _which, joint, index in ACTUATOR_SPECS}
LINEAR_CTRL_INDICES: Tuple[int, ...] = tuple(CTRL_INDEX[(f, "linear")] for f in OBS_FINGER_ORDER)
HINGE_CTRL_INDICES: Tuple[int, ...] = tuple(i for i in range(ACTUATOR_COUNT) if i not in LINEAR_CTRL_INDICES)

HARD_TIP_GEOM_NAMES: Dict[str, str] = {finger: f"hard_tip_{finger}" for finger in FINGERS}
HARD_TIP_BODY_NAMES: Dict[str, str] = {finger: f"hard_tip_{finger}" for finger in FINGERS}
TIP_GEOM_NAMES = HARD_TIP_GEOM_NAMES
TIP_BODY_NAMES = HARD_TIP_BODY_NAMES
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

# Conservative fallback ranges in the exact actuator order above.  Normal use
# always overwrites them from Share.xml and validates them against joint stops.
CTRL_RANGES_FALLBACK = np.array(
    [
        [-0.05, 0.0],
        [-2.007128640, 1.047197551],
        [-1.047197551, 1.745329252],
        [-0.05, 0.0],
        [-1.047197551, 1.745329252],
        [-2.007128640, 1.047197551],
        [-0.05, 0.0],
        [-2.007128640, 1.047197551],
        [-1.047197551, 1.745329252],
        [-0.05, 0.0],
        [-1.047197551, 1.745329252],
        [-2.007128640, 1.047197551],
        [-1.396263402, 1.396263402],
        [-1.396263402, 1.396263402],
        [-1.396263402, 1.396263402],
        [-1.396263402, 1.396263402],
    ],
    dtype=float,
)


class KinematicsError(RuntimeError):
    """The local XML contract or analytic geometry is invalid."""


class IKUnreachableError(KinematicsError):
    """At least one requested fingertip target has no valid IK branch."""

# =============================================================================
# General math helpers
# =============================================================================


def wrap_to_pi(theta: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(theta) + math.pi) % (2.0 * math.pi) - math.pi


def wrap_deg_180(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def normalize(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        if fallback is None:
            raise ValueError("zero-length vector cannot be normalized")
        return normalize(np.asarray(fallback, dtype=float))
    return v / n


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = q / n
    return -q if q[0] < 0.0 else q


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return normalize_quat(np.array([w, x, y, z], dtype=float))


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quat(q1)
    w2, x2, y2, z2 = normalize_quat(q2)
    return normalize_quat(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    q = quat_mul(quat_inv(q1), q2)
    w = float(np.clip(abs(q[0]), -1.0, 1.0))
    return 2.0 * math.acos(w)


def quat_to_euler_world_xyz_deg(q: np.ndarray) -> np.ndarray:
    """Extrinsic/fixed XYZ Euler angles in degrees for a relative quaternion."""
    R = quat_to_rot(q)
    sy = -float(R[2, 0])
    sy = float(np.clip(sy, -1.0, 1.0))
    beta = math.asin(sy)
    cb = math.cos(beta)
    if abs(cb) > 1e-9:
        alpha = math.atan2(float(R[2, 1]), float(R[2, 2]))
        gamma = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        alpha = 0.0
        gamma = math.atan2(-float(R[0, 1]), float(R[1, 1]))
    return np.asarray(wrap_deg_180(np.degrees([alpha, beta, gamma])), dtype=float)


def rot_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = normalize(np.asarray(axis, dtype=float).reshape(3), fallback=np.array([1.0, 0.0, 0.0]))
    x, y, z = [float(v) for v in axis]
    c = math.cos(float(theta))
    s = math.sin(float(theta))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def circle_intersections(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> List[np.ndarray]:
    delta = np.asarray(c2, dtype=float) - np.asarray(c1, dtype=float)
    d = float(np.linalg.norm(delta))
    if d < 1e-12 or d > r1 + r2 + 1e-12 or d < abs(r1 - r2) - 1e-12:
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h_sq = max(r1 * r1 - a * a, 0.0)
    h = math.sqrt(h_sq)
    p = np.asarray(c1, dtype=float) + (a / d) * delta
    n = np.array([-delta[1], delta[0]], dtype=float) / d
    if h < 1e-12:
        return [p]
    return [p + h * n, p - h * n]


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def minimum_jerk(t: float) -> float:
    """Quintic zero-velocity/zero-acceleration interpolation."""
    t = float(np.clip(t, 0.0, 1.0))
    return t**3 * (10.0 - 15.0 * t + 6.0 * t * t)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - float(t)) * np.asarray(a, dtype=float) + float(t) * np.asarray(b, dtype=float)


def arange_inclusive(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("step must be positive")
    if stop < start - 1e-12:
        return np.zeros(0, dtype=float)
    n = int(math.floor((stop - start) / step + 0.5))
    values = start + step * np.arange(n + 1, dtype=float)
    return values[values <= stop + 0.5 * step]


def clip_ctrl(ctrl: np.ndarray, ctrl_ranges: np.ndarray) -> np.ndarray:
    ranges = np.asarray(ctrl_ranges, dtype=float).reshape(ACTUATOR_COUNT, 2)
    return np.clip(
        np.asarray(ctrl, dtype=float).reshape(ACTUATOR_COUNT),
        ranges[:, 0],
        ranges[:, 1],
    )


def cylindrical_to_xyz(
    z_m: float,
    r_m: float,
    theta: float,
    *,
    theta_radians: bool = False,
    origin_xy: Sequence[float] = (0.0, 0.0),
) -> np.ndarray:
    """Convert terminal input order ``z, r, theta`` to world XYZ."""
    if float(r_m) < 0.0:
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
    return float(value[2]), math.hypot(dx, dy), float(theta)


def format_vector(vector: Iterable[float], precision: int = 6) -> str:
    return "(" + ", ".join(f"{float(value):.{precision}f}" for value in vector) + ")"

# =============================================================================
# XML parsing helpers
# =============================================================================


def _resolve_local_path(
    path_like: str | os.PathLike[str],
    *,
    label: str,
    must_exist: bool = True,
) -> Path:
    """Resolve a dependency inside this SCREW folder only."""
    candidate = Path(path_like).expanduser()
    if not candidate.is_absolute():
        candidate = MODULE_DIR / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(MODULE_DIR)
    except ValueError as exc:
        raise KinematicsError(
            f"{label} must stay inside {MODULE_DIR}: {candidate}"
        ) from exc
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    return candidate


def _parse_floats(text: Optional[str], n: Optional[int] = None, default: Optional[Sequence[float]] = None) -> np.ndarray:
    if text is None:
        arr = np.asarray(default if default is not None else [], dtype=float)
    else:
        arr = np.asarray([float(x) for x in str(text).replace(",", " ").split()], dtype=float)
    if n is not None and arr.size != n:
        if default is not None:
            arr = np.asarray(default, dtype=float)
        else:
            raise ValueError(f"Expected {n} floats, got {arr.size}: {text!r}")
    return arr.astype(float)


def _quat_from_element(elem: ET.Element) -> np.ndarray:
    if elem.get("quat") is not None:
        return normalize_quat(_parse_floats(elem.get("quat"), 4, [1.0, 0.0, 0.0, 0.0]))
    if elem.get("euler") is not None:
        vals = _parse_floats(elem.get("euler"), 3, [0.0, 0.0, 0.0])
        # MuJoCo default compiler uses xyz intrinsic-ish Euler; for these files quat is used.
        rx, ry, rz = [float(v) for v in vals]
        qx = np.array([math.cos(rx / 2), math.sin(rx / 2), 0.0, 0.0])
        qy = np.array([math.cos(ry / 2), 0.0, math.sin(ry / 2), 0.0])
        qz = np.array([math.cos(rz / 2), 0.0, 0.0, math.sin(rz / 2)])
        return quat_mul(quat_mul(qz, qy), qx)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def _find_xml(root: ET.Element, tag: str, **attrs: str) -> Optional[ET.Element]:
    for elem in root.iter(tag):
        ok = True
        for k, v in attrs.items():
            if elem.get(k) != v:
                ok = False
                break
        if ok:
            return elem
    return None


def _compose_pose(p_parent: np.ndarray, R_parent: np.ndarray, p_local: np.ndarray, R_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return p_parent + R_parent @ p_local, R_parent @ R_local


def _invert_pose(p: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Rt = np.asarray(R, dtype=float).T
    return -Rt @ np.asarray(p, dtype=float), Rt


def _pose_transform_point(p_pose: np.ndarray, R_pose: np.ndarray, point_local: np.ndarray) -> np.ndarray:
    return np.asarray(p_pose, dtype=float) + np.asarray(R_pose, dtype=float) @ np.asarray(point_local, dtype=float)


def _world_to_local(p_world: np.ndarray, p_body_world: np.ndarray, R_body_world: np.ndarray) -> np.ndarray:
    return np.asarray(R_body_world, dtype=float).T @ (np.asarray(p_world, dtype=float) - np.asarray(p_body_world, dtype=float))


@dataclass
class BodyNode:
    name: str
    pos: np.ndarray
    quat: np.ndarray
    parent: Optional[str]
    element: ET.Element
    children: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class JointXMLSpec:
    name: str
    joint_type: str
    body_name: str
    axis: np.ndarray
    pos: np.ndarray
    range: Tuple[float, float]


@dataclass(frozen=True)
class SiteXMLSpec:
    name: str
    body_name: str
    pos: np.ndarray


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
class XMLModelInfo:
    project_xml: Path
    project_dir: Path
    gripper_xml: Path
    share_xml: Path
    xml_timestep: float
    root_pos: np.ndarray
    root_quat: np.ndarray
    table_top_z: float
    table_bottom_z: float
    table_half_size: np.ndarray
    cube_pos: np.ndarray
    cube_quat: np.ndarray
    cube_half_size: float
    hard_tip_radius: float
    actuator_ctrlrange: np.ndarray
    actuator_names: Tuple[str, ...]
    actuator_joint_names: Tuple[str, ...]
    actuator_specs: Tuple[ActuatorSpec, ...]

    @property
    def timestep(self) -> float:
        return self.xml_timestep

    @property
    def cube_initial_pos(self) -> np.ndarray:
        return self.cube_pos

    @property
    def ctrl_ranges(self) -> np.ndarray:
        return self.actuator_ctrlrange

    @property
    def contact_distance(self) -> float:
        return self.cube_half_size + self.hard_tip_radius

    @property
    def minimum_cube_center_z(self) -> float:
        return self.table_top_z + self.cube_half_size


# Public metadata name used by the controller and workspace tools.
ModelInfo = XMLModelInfo


class XMLTreeModel:
    """Merged Project/Gripper/Share view with zero-pose transforms."""

    def __init__(self, project_xml: str | os.PathLike[str] = DEFAULT_XML):
        self.project_xml = _resolve_local_path(
            project_xml,
            label="Project XML",
        )
        if not self.project_xml.is_file():
            raise KinematicsError(f"Project XML is not a file: {self.project_xml}")
        self.model_dir = self.project_xml.parent
        self.project_root = ET.parse(self.project_xml).getroot()

        includes: Dict[str, Path] = {}
        for include in self.project_root.iter("include"):
            filename = include.get("file")
            if not filename:
                continue
            include_path = _resolve_local_path(
                self.model_dir / filename,
                label=f"XML include {filename!r}",
            )
            if not include_path.is_file():
                raise KinematicsError(f"XML include is not a file: {include_path}")
            includes[Path(filename).name.lower()] = include_path
        try:
            self.share_xml = includes["share.xml"]
            self.gripper_xml = includes["gripper.xml"]
        except KeyError as exc:
            raise KinematicsError(
                "Project.xml must include local Share.xml and Gripper.xml"
            ) from exc

        self.share_root = ET.parse(self.share_xml).getroot()
        self.gripper_root = ET.parse(self.gripper_xml).getroot()
        self._validate_local_mesh_dependencies()
        self.bodies: Dict[str, BodyNode] = {}
        self.joints: Dict[str, JointXMLSpec] = {}
        self.sites: Dict[str, SiteXMLSpec] = {}
        self.zero_world_pos: Dict[str, np.ndarray] = {}
        self.zero_world_rot: Dict[str, np.ndarray] = {}
        self._build_body_tree()
        self._compute_zero_world_transforms()
        self.info = self._parse_info()

    def _validate_local_mesh_dependencies(self) -> None:
        compiler = self.project_root.find("compiler")
        meshdir = compiler.get("meshdir", ".") if compiler is not None else "."
        mesh_dir = _resolve_local_path(
            self.model_dir / meshdir,
            label="mesh directory",
        )
        if not mesh_dir.is_dir():
            raise KinematicsError(f"mesh directory is not a directory: {mesh_dir}")
        for mesh in self.share_root.iter("mesh"):
            filename = mesh.get("file")
            if not filename:
                continue
            mesh_path = _resolve_local_path(
                mesh_dir / filename,
                label=f"mesh {filename!r}",
            )
            if not mesh_path.is_file():
                raise KinematicsError(f"mesh is not a file: {mesh_path}")

    def _build_body_tree(self) -> None:
        world_roots: List[ET.Element] = []
        worldbody = self.project_root.find("worldbody")
        if worldbody is not None:
            for child in list(worldbody):
                if child.tag == "body":
                    world_roots.append(child)
                elif (
                    child.tag == "include"
                    and Path(str(child.get("file", ""))).name.lower() == "gripper.xml"
                ):
                    for gr_child in list(self.gripper_root):
                        if gr_child.tag == "body":
                            world_roots.append(gr_child)
        # If Project.xml was not merged as expected, fall back to all top-level Gripper bodies.
        if not world_roots:
            world_roots = [e for e in list(self.gripper_root) if e.tag == "body"]

        def visit(elem: ET.Element, parent: Optional[str]) -> None:
            if elem.tag != "body":
                return
            name = elem.get("name") or f"unnamed_body_{len(self.bodies)}"
            pos = _parse_floats(elem.get("pos"), 3, [0.0, 0.0, 0.0])
            quat = _quat_from_element(elem)
            self.bodies[name] = BodyNode(name=name, pos=pos, quat=quat, parent=parent, element=elem)
            if parent is not None and parent in self.bodies:
                self.bodies[parent].children.append(name)
            for joint in elem.findall("joint"):
                jname = joint.get("name")
                if not jname:
                    continue
                jtype = joint.get("type", "hinge")
                axis_default = [0.0, 0.0, 1.0] if jtype == "slide" else [1.0, 0.0, 0.0]
                axis = normalize(_parse_floats(joint.get("axis"), 3, axis_default), fallback=np.asarray(axis_default, dtype=float))
                pos_j = _parse_floats(joint.get("pos"), 3, [0.0, 0.0, 0.0])
                rng_vals = _parse_floats(joint.get("range"), 2, [-np.inf, np.inf])
                self.joints[jname] = JointXMLSpec(
                    name=jname,
                    joint_type=jtype,
                    body_name=name,
                    axis=axis,
                    pos=pos_j,
                    range=(float(rng_vals[0]), float(rng_vals[1])),
                )
            for site in elem.findall("site"):
                sname = site.get("name")
                if sname:
                    self.sites[sname] = SiteXMLSpec(
                        name=sname,
                        body_name=name,
                        pos=_parse_floats(site.get("pos"), 3, [0.0, 0.0, 0.0]),
                    )
            for child in elem.findall("body"):
                visit(child, name)

        for root_body in world_roots:
            visit(root_body, None)

    def _compute_zero_world_transforms(self) -> None:
        def compute(name: str) -> Tuple[np.ndarray, np.ndarray]:
            if name in self.zero_world_pos:
                return self.zero_world_pos[name], self.zero_world_rot[name]
            node = self.bodies[name]
            R_local = quat_to_rot(node.quat)
            if node.parent is None:
                p, R = node.pos.astype(float), R_local
            else:
                p_parent, R_parent = compute(node.parent)
                p, R = _compose_pose(p_parent, R_parent, node.pos, R_local)
            self.zero_world_pos[name] = p.astype(float)
            self.zero_world_rot[name] = R.astype(float)
            return p, R

        for name in list(self.bodies):
            compute(name)

    def body_zero_pose(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if body_name not in self.zero_world_pos:
            raise KeyError(f"XML body not found: {body_name}")
        return self.zero_world_pos[body_name].copy(), self.zero_world_rot[body_name].copy()

    def point_world(self, body_name: str, local_point: Sequence[float]) -> np.ndarray:
        p, R = self.body_zero_pose(body_name)
        return _pose_transform_point(p, R, np.asarray(local_point, dtype=float))

    def point_in_body(self, point_body_name: str, local_point: Sequence[float], target_body_name: str) -> np.ndarray:
        pw = self.point_world(point_body_name, local_point)
        pt, Rt = self.body_zero_pose(target_body_name)
        return _world_to_local(pw, pt, Rt)

    def joint_origin_in_body(self, joint_name: str, target_body_name: str) -> np.ndarray:
        j = self.joints[joint_name]
        return self.point_in_body(j.body_name, j.pos, target_body_name)

    def site_in_body(self, site_name: str, target_body_name: str) -> np.ndarray:
        s = self.sites[site_name]
        return self.point_in_body(s.body_name, s.pos, target_body_name)

    def _parse_info(self) -> XMLModelInfo:
        timestep = 0.001
        option = self.project_root.find("option")
        if option is not None and option.get("timestep"):
            timestep = float(option.get("timestep"))

        if "root" not in self.bodies:
            raise KinematicsError(
                "Gripper.xml must define the root body; its XML pose is the "
                "single source of truth for all world-frame kinematics"
            )
        root_pos = self.bodies["root"].pos.astype(float)
        root_quat = self.bodies["root"].quat.astype(float)

        table_top = 0.260
        table_bottom = 0.240
        table_half = np.array([1.0, 1.0, 0.01], dtype=float)
        table_body = _find_xml(self.project_root, "body", name="table")
        if table_body is not None:
            tpos = _parse_floats(table_body.get("pos"), 3, [0.0, 0.0, 0.25])
            geom = table_body.find("geom")
            if geom is not None:
                size = _parse_floats(geom.get("size"), default=[1.0, 1.0, 0.01])
                if size.size >= 3:
                    table_half = size[:3].astype(float)
                    table_top = float(tpos[2] + table_half[2])
                    table_bottom = float(tpos[2] - table_half[2])

        cube_pos = np.array([0.0, 0.0, 0.285], dtype=float)
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        cube_half = 0.025
        cube_body = _find_xml(self.project_root, "body", name=CUBE_BODY_NAME)
        if cube_body is not None:
            cube_pos = _parse_floats(cube_body.get("pos"), 3, cube_pos)
            cube_quat = _quat_from_element(cube_body)
            geom = cube_body.find("geom")
            if geom is not None:
                size = _parse_floats(geom.get("size"), default=[0.025, 0.025, 0.025])
                if size.size >= 1:
                    cube_half = float(size[0])

        hard_tip_radius = 0.020
        for default in self.share_root.iter("default"):
            if default.get("class") == "Hard_tip":
                geom = default.find("geom")
                if geom is not None and geom.get("size"):
                    vals = _parse_floats(geom.get("size"), default=[hard_tip_radius])
                    if vals.size >= 1:
                        hard_tip_radius = float(vals[0])
        # Direct geom size in Gripper.xml overrides default if present.
        for geom in self.gripper_root.iter("geom"):
            name = str(geom.get("name") or "")
            if name.lower().startswith("hard_tip_") and geom.get("size"):
                vals = _parse_floats(geom.get("size"), default=[hard_tip_radius])
                if vals.size >= 1:
                    hard_tip_radius = float(vals[0])
                    break

        actuator_parent = self.share_root.find("actuator")
        if actuator_parent is None:
            raise KinematicsError("Share.xml is missing <actuator>")
        actuator_elements = [
            element for element in list(actuator_parent) if element.tag == "position"
        ]
        actual_names = tuple(str(element.get("name") or "") for element in actuator_elements)
        if actual_names != ACTUATOR_NAMES:
            raise KinematicsError(
                "Share.xml actuator order mismatch; expected "
                f"{ACTUATOR_NAMES}, got {actual_names}"
            )

        expected_joint_by_key = {
            (finger, kind): joint_name
            for finger, kind, joint_name, _index in ACTUATOR_SPECS
        }
        ctrl_ranges = CTRL_RANGES_FALLBACK.copy()
        joint_names: List[str] = []
        actuator_specs: List[ActuatorSpec] = []
        for index, (element, (finger, kind, actuator_name)) in enumerate(
            zip(actuator_elements, EXPECTED_ACTUATOR_ORDER)
        ):
            expected_joint = expected_joint_by_key[(finger, kind)]
            joint_name = str(element.get("joint") or "")
            if joint_name != expected_joint:
                raise KinematicsError(
                    f"actuator {actuator_name!r} must control "
                    f"{expected_joint!r}, got {joint_name!r}"
                )
            if joint_name not in self.joints:
                raise KinematicsError(
                    f"actuator {actuator_name!r} references missing joint {joint_name!r}"
                )

            ctrlrange_text = element.get("ctrlrange")
            class_name = element.get("class")
            if ctrlrange_text is None and class_name:
                for default in self.share_root.iter("default"):
                    if default.get("class") != class_name:
                        continue
                    template = default.find(element.tag)
                    if template is not None:
                        ctrlrange_text = template.get("ctrlrange")
                    break
            if ctrlrange_text is None:
                raise KinematicsError(
                    f"actuator {actuator_name!r} has no explicit or "
                    "class-derived ctrlrange"
                )
            ctrl_limits_array = _parse_floats(
                ctrlrange_text,
                2,
                ctrl_ranges[index],
            )
            ctrl_limits = (
                float(ctrl_limits_array[0]),
                float(ctrl_limits_array[1]),
            )
            if not ctrl_limits[0] < ctrl_limits[1]:
                raise KinematicsError(
                    f"actuator {actuator_name!r} has invalid ctrlrange {ctrl_limits}"
                )
            joint_limits = self.joints[joint_name].range
            if not np.allclose(ctrl_limits, joint_limits, atol=1e-12):
                raise KinematicsError(
                    f"actuator {actuator_name!r} ctrlrange {ctrl_limits} does not "
                    f"match joint {joint_name!r} range {joint_limits}"
                )
            ctrl_ranges[index] = ctrl_limits_array
            joint_names.append(joint_name)
            actuator_specs.append(
                ActuatorSpec(
                    name=actuator_name,
                    finger=finger,
                    kind=kind,
                    joint_name=joint_name,
                    index=index,
                    ctrl_range=ctrl_limits,
                    joint_range=joint_limits,
                    effective_range=joint_limits,
                )
            )

        return XMLModelInfo(
            project_xml=self.project_xml,
            project_dir=self.model_dir,
            gripper_xml=self.gripper_xml,
            share_xml=self.share_xml,
            xml_timestep=float(timestep),
            root_pos=root_pos.astype(float),
            root_quat=root_quat.astype(float),
            table_top_z=float(table_top),
            table_bottom_z=float(table_bottom),
            table_half_size=table_half.astype(float),
            cube_pos=cube_pos.astype(float),
            cube_quat=normalize_quat(cube_quat),
            cube_half_size=float(cube_half),
            hard_tip_radius=float(hard_tip_radius),
            actuator_ctrlrange=ctrl_ranges.astype(float),
            actuator_names=ACTUATOR_NAMES,
            actuator_joint_names=tuple(joint_names),
            actuator_specs=tuple(actuator_specs),
        )


def parse_model_xml(
    model_xml: str | os.PathLike[str] = DEFAULT_XML,
) -> XMLModelInfo:
    return XMLTreeModel(model_xml).info

# =============================================================================
# Finger closed-chain geometry and IK
# =============================================================================


@dataclass(frozen=True)
class FingerGeometry:
    finger: str
    p_A_world: np.ndarray
    R_A_world: np.ndarray
    p_AH0: np.ndarray
    R_AH0: np.ndarray
    armbase_axis_A: np.ndarray
    linear_axis_H: np.ndarray
    plane_normal_H: np.ndarray
    eX_H: np.ndarray
    eY_H: np.ndarray
    plane_offset: float
    O_H: np.ndarray
    M_H: np.ndarray
    A_H: np.ndarray
    B_H: np.ndarray
    C_front_H: np.ndarray
    C_long_H: np.ndarray
    C0_H: np.ndarray
    D0_H: np.ndarray
    O2: np.ndarray
    M2: np.ndarray
    A2: np.ndarray
    B2: np.ndarray
    C_front2: np.ndarray
    C_long2: np.ndarray
    C02: np.ndarray
    D02: np.ndarray
    dm: float
    l1: float
    l2: float
    l3: float
    dl: float
    r_tip: float
    kappa: float
    theta_back0: float
    theta_front0: float
    rel_short0: float
    rel_elbow0: float
    sigma_back: float
    sigma_front: float
    sigma_short: float
    sigma_elbow: float
    active_ranges: Dict[str, Tuple[float, float]]
    passive_ranges: Dict[str, Tuple[float, float]]
    pin_mismatch_m: float

    @property
    def base_world(self) -> np.ndarray:
        return self.p_A_world

    @property
    def rotation_world(self) -> np.ndarray:
        return self.R_A_world

    @property
    def armbase_axis(self) -> np.ndarray:
        return self.armbase_axis_A

    @property
    def linear_axis(self) -> np.ndarray:
        return self.linear_axis_H

    @property
    def armbase_axis_world(self) -> np.ndarray:
        return self.R_A_world @ self.armbase_axis_A

    @property
    def linear_axis_world(self) -> np.ndarray:
        return self.R_A_world @ self.R_AH0 @ self.linear_axis_H

    @property
    def joint_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {**self.active_ranges, **self.passive_ranges}

    @property
    def effective_ranges(self) -> Dict[str, Tuple[float, float]]:
        return self.joint_ranges

    def project2(self, p_H: np.ndarray) -> np.ndarray:
        v = np.asarray(p_H, dtype=float).reshape(3) - self.O_H
        return np.array([float(np.dot(self.eX_H, v)), float(np.dot(self.eY_H, v))], dtype=float)

    def in_active_range(self, which: str, value: float, tol: float = 1e-9) -> bool:
        lo, hi = self.active_ranges[which]
        return bool(float(value) >= lo - tol and float(value) <= hi + tol)

    def in_passive_range(self, which: str, value: float, tol: float = 1e-7) -> bool:
        lo, hi = self.passive_ranges[which]
        return bool(float(value) >= lo - tol and float(value) <= hi + tol)


@dataclass(frozen=True)
class GripperGeometryModel:
    xml_path: str
    xml_info: XMLModelInfo
    ctrl_ranges: np.ndarray
    fingers: Dict[str, FingerGeometry]

    @classmethod
    def from_xml(
        cls,
        model_xml: str | os.PathLike[str] = DEFAULT_XML,
    ) -> "GripperGeometryModel":
        tree = XMLTreeModel(model_xml)
        ctrl_ranges = tree.info.actuator_ctrlrange.copy()
        fingers = {finger: _build_finger_geometry(tree, finger, ctrl_ranges) for finger in OBS_FINGER_ORDER}
        reference = fingers[OBS_FINGER_ORDER[0]]
        vector_fields = (
            "p_AH0",
            "R_AH0",
            "armbase_axis_A",
            "linear_axis_H",
            "plane_normal_H",
            "O_H",
            "M_H",
            "A_H",
            "B_H",
            "C_front_H",
            "C_long_H",
            "D0_H",
        )
        scalar_fields = (
            "dm",
            "l1",
            "l2",
            "l3",
            "dl",
            "r_tip",
            "kappa",
            "theta_back0",
            "theta_front0",
            "rel_short0",
            "rel_elbow0",
        )
        for finger in OBS_FINGER_ORDER[1:]:
            geometry = fingers[finger]
            for field_name in vector_fields:
                if not np.allclose(
                    getattr(geometry, field_name),
                    getattr(reference, field_name),
                    atol=1e-10,
                ):
                    raise KinematicsError(
                        f"{finger}: local geometry field {field_name} "
                        "breaks four-finger symmetry"
                    )
            for field_name in scalar_fields:
                if not math.isclose(
                    float(getattr(geometry, field_name)),
                    float(getattr(reference, field_name)),
                    abs_tol=1e-10,
                ):
                    raise KinematicsError(
                        f"{finger}: local geometry field {field_name} "
                        "breaks four-finger symmetry"
                    )
        return cls(str(tree.project_xml), tree.info, ctrl_ranges, fingers)


def _joint_range_or_ctrl(tree: XMLTreeModel, joint_name: str, ctrl_idx: Optional[int] = None) -> Tuple[float, float]:
    if joint_name in tree.joints:
        lo, hi = tree.joints[joint_name].range
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return (float(lo), float(hi))
    if ctrl_idx is not None:
        rng = tree.info.actuator_ctrlrange[int(ctrl_idx)]
        return (float(rng[0]), float(rng[1]))
    return (-np.inf, np.inf)


def _angle2(v: np.ndarray) -> float:
    return math.atan2(float(v[1]), float(v[0]))


def _sign_from_derivative(delta: float) -> float:
    if not np.isfinite(delta) or abs(delta) < 1e-9:
        return 1.0
    return 1.0 if delta > 0.0 else -1.0


def _build_finger_geometry(tree: XMLTreeModel, finger: str, ctrl_ranges: np.ndarray) -> FingerGeometry:
    armbase_body = f"robot0:armbase_{finger}"
    # Despite its historical XML name, robot0:linear_* carries the current
    # handbase-module mesh and is the correct closed-chain reference frame.
    handbase_body = f"robot0:linear_{finger}"
    back_body = f"robot0:back_{finger}"
    front_body = f"robot0:front_{finger}"
    short_body = f"robot0:short_{finger}"
    long_body = f"robot0:long_{finger}"
    tip_body = f"hard_tip_{finger}"

    expected_parents = {
        armbase_body: "root",
        handbase_body: armbase_body,
        back_body: handbase_body,
        front_body: handbase_body,
        short_body: back_body,
        long_body: short_body,
        tip_body: long_body,
    }
    for body_name, expected_parent in expected_parents.items():
        if body_name not in tree.bodies:
            raise KinematicsError(f"required XML body not found: {body_name}")
        actual_parent = tree.bodies[body_name].parent
        if actual_parent != expected_parent:
            raise KinematicsError(
                f"{body_name} must be a direct child of {expected_parent}, "
                f"got parent {actual_parent!r}"
            )

    p_A_world, R_A_world = tree.body_zero_pose(armbase_body)
    p_H_world, R_H_world = tree.body_zero_pose(handbase_body)
    p_AH0 = R_A_world.T @ (p_H_world - p_A_world)
    R_AH0 = R_A_world.T @ R_H_world

    j_arm = tree.joints[f"robot0:armbase_joint_{finger}"]
    j_lin = tree.joints[f"robot0:linear_joint_{finger}"]
    j_back = tree.joints[f"robot0:back_joint_{finger}"]
    j_front = tree.joints[f"robot0:front_joint_{finger}"]
    j_short = tree.joints[f"robot0:short_joint_{finger}"]
    j_elbow = tree.joints[f"robot0:elbow_joint_{finger}"]

    if j_lin.joint_type != "slide":
        raise KinematicsError(f"{j_lin.name} must be a slide joint")
    for joint in (j_arm, j_back, j_front, j_short, j_elbow):
        if joint.joint_type != "hinge":
            raise KinematicsError(f"{joint.name} must be a hinge joint")
    if not np.allclose(j_lin.axis, (0.0, 0.0, 1.0), atol=1e-10):
        raise KinematicsError(
            f"{j_lin.name} must move along its local +z axis, got {j_lin.axis}"
        )
    if not np.allclose(np.abs(j_arm.axis), (1.0, 0.0, 0.0), atol=1e-10):
        raise KinematicsError(
            f"{j_arm.name} must rotate about local x, got {j_arm.axis}"
        )
    for joint in (j_back, j_front, j_short, j_elbow):
        if not np.allclose(joint.axis, (0.0, 1.0, 0.0), atol=1e-10):
            raise KinematicsError(
                f"{joint.name} must rotate about local +y, got {joint.axis}"
            )

    # Joint axes transformed into the handbase frame at zero.
    def axis_in_handbase(j: JointXMLSpec) -> np.ndarray:
        _p_body, R_body = tree.body_zero_pose(j.body_name)
        axis_w = R_body @ j.axis
        return normalize(R_H_world.T @ axis_w, fallback=np.array([0.0, 1.0, 0.0]))

    plane_normal = axis_in_handbase(j_back)
    # Ensure the normal points consistently; sign does not affect plane, but it affects eY orientation.
    if np.dot(plane_normal, np.array([0.0, 1.0, 0.0])) < 0.0:
        plane_normal = -plane_normal

    O_H = tree.joint_origin_in_body(j_back.name, handbase_body)
    M_H = tree.joint_origin_in_body(j_front.name, handbase_body)
    A_H = tree.joint_origin_in_body(j_short.name, handbase_body)
    B_H = tree.joint_origin_in_body(j_elbow.name, handbase_body)
    C_front_H = tree.site_in_body(f"pin:{finger}_front", handbase_body)
    C_long_H = tree.site_in_body(f"pin:{finger}_long", handbase_body)
    C0_H = 0.5 * (C_front_H + C_long_H)
    D0_H = tree.point_in_body(tip_body, [0.0, 0.0, 0.0], handbase_body)

    OM_plane = (M_H - O_H) - plane_normal * float(np.dot(plane_normal, M_H - O_H))
    eX = normalize(OM_plane, fallback=np.array([0.0, 0.0, -1.0]))
    eY = normalize(np.cross(plane_normal, eX), fallback=np.array([-1.0, 0.0, 0.0]))
    # Re-orthogonalize to avoid drift.
    eX = normalize(np.cross(eY, plane_normal), fallback=eX)

    def P(p: np.ndarray) -> np.ndarray:
        v = np.asarray(p, dtype=float) - O_H
        return np.array([float(np.dot(eX, v)), float(np.dot(eY, v))], dtype=float)

    O2 = np.zeros(2, dtype=float)
    M2 = P(M_H)
    A2 = P(A_H)
    B2 = P(B_H)
    C_front2 = P(C_front_H)
    C_long2 = P(C_long_H)
    C02 = P(C0_H)
    D02 = P(D0_H)

    dm = float(np.linalg.norm(M2 - O2))
    l1 = float(np.linalg.norm(A2 - O2))
    l2 = float(np.linalg.norm(B2 - A2))
    l3 = float(np.linalg.norm(C_front2 - M2))
    dl = float(np.linalg.norm(C_long2 - B2))
    r_tip = float(np.linalg.norm(D02 - C_long2))
    if min(dm, l1, l2, l3, dl, r_tip) <= 1e-8:
        raise ValueError(
            f"Degenerate finger geometry for {finger}: dm={dm}, l1={l1}, l2={l2}, l3={l3}, dl={dl}, r={r_tip}"
        )

    theta_back0 = _angle2(A2 - O2)
    theta_front0 = _angle2(C_front2 - M2)
    theta_ab0 = _angle2(B2 - A2)
    theta_bc0 = _angle2(C_long2 - B2)
    rel_short0 = float(wrap_to_pi(theta_ab0 - theta_back0))
    rel_elbow0 = float(wrap_to_pi(theta_bc0 - theta_ab0))
    kappa = float(wrap_to_pi(theta_bc0 - _angle2(D02 - C_long2)))

    eps = 1e-5
    axis_back_H = axis_in_handbase(j_back)
    axis_front_H = axis_in_handbase(j_front)
    axis_short_H = axis_in_handbase(j_short)
    axis_elbow_H = axis_in_handbase(j_elbow)

    A_eps = O_H + rot_axis(axis_back_H, eps) @ (A_H - O_H)
    sigma_back = _sign_from_derivative(float(wrap_to_pi(_angle2(P(A_eps) - O2) - theta_back0)) / eps)
    C_eps = M_H + rot_axis(axis_front_H, eps) @ (C_front_H - M_H)
    sigma_front = _sign_from_derivative(float(wrap_to_pi(_angle2(P(C_eps) - M2) - theta_front0)) / eps)
    B_eps = A_H + rot_axis(axis_short_H, eps) @ (B_H - A_H)
    rel_short_eps = float(wrap_to_pi(_angle2(P(B_eps) - A2) - theta_back0))
    sigma_short = _sign_from_derivative(float(wrap_to_pi(rel_short_eps - rel_short0)) / eps)
    C_elbow_eps = B_H + rot_axis(axis_elbow_H, eps) @ (C_long_H - B_H)
    rel_elbow_eps = float(wrap_to_pi(_angle2(P(C_elbow_eps) - B2) - theta_ab0))
    sigma_elbow = _sign_from_derivative(float(wrap_to_pi(rel_elbow_eps - rel_elbow0)) / eps)

    active_ranges = {
        "back": _joint_range_or_ctrl(tree, j_back.name, CTRL_INDEX[(finger, "back")]),
        "front": _joint_range_or_ctrl(tree, j_front.name, CTRL_INDEX[(finger, "front")]),
        "armbase": _joint_range_or_ctrl(tree, j_arm.name, CTRL_INDEX[(finger, "armbase")]),
        "linear": _joint_range_or_ctrl(tree, j_lin.name, CTRL_INDEX[(finger, "linear")]),
    }
    passive_ranges = {
        "short": _joint_range_or_ctrl(tree, j_short.name),
        "elbow": _joint_range_or_ctrl(tree, j_elbow.name),
    }

    return FingerGeometry(
        finger=finger,
        p_A_world=p_A_world.astype(float),
        R_A_world=R_A_world.astype(float),
        p_AH0=p_AH0.astype(float),
        R_AH0=R_AH0.astype(float),
        armbase_axis_A=normalize(j_arm.axis, fallback=np.array([1.0, 0.0, 0.0])),
        linear_axis_H=normalize(j_lin.axis, fallback=np.array([0.0, 0.0, 1.0])),
        plane_normal_H=plane_normal.astype(float),
        eX_H=eX.astype(float),
        eY_H=eY.astype(float),
        plane_offset=float(np.dot(plane_normal, D0_H)),
        O_H=O_H.astype(float),
        M_H=M_H.astype(float),
        A_H=A_H.astype(float),
        B_H=B_H.astype(float),
        C_front_H=C_front_H.astype(float),
        C_long_H=C_long_H.astype(float),
        C0_H=C0_H.astype(float),
        D0_H=D0_H.astype(float),
        O2=O2,
        M2=M2.astype(float),
        A2=A2.astype(float),
        B2=B2.astype(float),
        C_front2=C_front2.astype(float),
        C_long2=C_long2.astype(float),
        C02=C02.astype(float),
        D02=D02.astype(float),
        dm=float(dm),
        l1=float(l1),
        l2=float(l2),
        l3=float(l3),
        dl=float(dl),
        r_tip=float(r_tip),
        kappa=float(kappa),
        theta_back0=float(theta_back0),
        theta_front0=float(theta_front0),
        rel_short0=float(rel_short0),
        rel_elbow0=float(rel_elbow0),
        sigma_back=float(sigma_back),
        sigma_front=float(sigma_front),
        sigma_short=float(sigma_short),
        sigma_elbow=float(sigma_elbow),
        active_ranges=active_ranges,
        passive_ranges=passive_ranges,
        pin_mismatch_m=float(np.linalg.norm(C_front_H - C_long_H)),
    )


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
    branch: Tuple[int, ...] = ()

    def joint_value(self, kind: str) -> float:
        return float(getattr(self, f"q_{kind}"))

    @property
    def joint_map(self) -> Dict[str, float]:
        return {kind: self.joint_value(kind) for kind in JOINT_KINDS}


@dataclass
class IKSolutionSummary:
    ctrl: np.ndarray
    solved_mask: np.ndarray
    failed_fingers: Tuple[str, ...]
    finger_solutions: Dict[str, FingerSolution] = field(default_factory=dict)
    max_tip_error: float = math.inf
    max_loop_error: float = math.inf
    branch_debug: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return not self.failed_fingers


IKSolution = IKSolutionSummary


@dataclass(frozen=True)
class CylindricalWorkspaceLimit:
    theta_deg: float
    radius: float
    z: float
    xyz: np.ndarray
    distance: float
    resolution: float


class GripperIK:
    def __init__(
        self,
        geometry_model: GripperGeometryModel,
        linear_search_step_m: float = 0.001,
        linear_ref_m: float = 0.0,
        filter_passive_limits: bool = True,
    ):
        self.geometry_model = geometry_model
        self.ctrl_ranges = np.asarray(geometry_model.ctrl_ranges, dtype=float).reshape(16, 2).copy()
        self.linear_search_step_m = float(linear_search_step_m)
        self.linear_ref_m = float(linear_ref_m)
        self.filter_passive_limits = bool(filter_passive_limits)
        base_center_xy = np.mean(
            np.vstack(
                [
                    geometry.p_A_world[:2]
                    for geometry in geometry_model.fingers.values()
                ]
            ),
            axis=0,
        )
        self.contact_directions_world: Dict[str, np.ndarray] = {}
        for finger, geometry in geometry_model.fingers.items():
            radial_xy = geometry.p_A_world[:2] - base_center_xy
            if np.linalg.norm(radial_xy) < 1e-12:
                fallback = TIP_OFFSETS_UNIT[finger]
                direction = fallback
            else:
                direction = np.array(
                    [radial_xy[0], radial_xy[1], 0.0],
                    dtype=float,
                )
            self.contact_directions_world[finger] = normalize(
                direction,
                fallback=TIP_OFFSETS_UNIT[finger],
            )

    @property
    def model_info(self) -> XMLModelInfo:
        return self.geometry_model.xml_info

    @property
    def fingers(self) -> Dict[str, FingerGeometry]:
        return self.geometry_model.fingers

    @classmethod
    def from_xml(
        cls,
        model_xml: str | os.PathLike[str] = DEFAULT_XML,
        linear_search_step_m: float = 0.001,
        linear_ref_m: float = 0.0,
        filter_passive_limits: bool = True,
    ) -> "GripperIK":
        return cls(
            GripperGeometryModel.from_xml(model_xml),
            linear_search_step_m=linear_search_step_m,
            linear_ref_m=linear_ref_m,
            filter_passive_limits=filter_passive_limits,
        )

    def active_range_summary(self) -> str:
        geometry = self.fingers[FINGERS[0]]
        parts: List[str] = []
        for kind in ACTIVE_JOINT_KINDS:
            low, high = geometry.active_ranges[kind]
            if kind == "linear":
                parts.append(f"{kind}=[{low:+.4f}, {high:+.4f}] m")
            else:
                parts.append(
                    f"{kind}=[{math.degrees(low):+.1f}, "
                    f"{math.degrees(high):+.1f}] deg"
                )
        return ", ".join(parts)

    def with_physical_joint_ranges(self) -> "GripperIK":
        # The current XML contract requires every actuator ctrlrange to match
        # its physical joint range, so the diagnostic solver is identical.
        return self

    def set_armbase_world_poses(self, poses: Mapping[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """Update runtime armbase installation poses from MuJoCo at reset.

        Only the world pose of each armbase zero frame is updated.  The fixed
        handbase and closed-chain geometry parsed from XML are kept unchanged.
        This is useful when Project.xml/root pose changes without changing the
        finger subtree geometry.
        """
        fingers: Dict[str, FingerGeometry] = {}
        for finger, geom in self.geometry_model.fingers.items():
            if finger not in poses:
                fingers[finger] = geom
                continue
            p, R = poses[finger]
            fingers[finger] = FingerGeometry(
                **{**geom.__dict__, "p_A_world": np.asarray(p, dtype=float).reshape(3), "R_A_world": np.asarray(R, dtype=float).reshape(3, 3)}
            )
        self.geometry_model = GripperGeometryModel(
            xml_path=self.geometry_model.xml_path,
            xml_info=self.geometry_model.xml_info,
            ctrl_ranges=self.geometry_model.ctrl_ranges,
            fingers=fingers,
        )

    @staticmethod
    def _coerce_joint_map(
        joint_values: Mapping[str, float] | FingerSolution,
    ) -> Dict[str, float]:
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
        """Evaluate the current armbase->linear->closed-chain FK."""
        finger = finger.upper()
        geometry = self.fingers[finger]
        q = self._coerce_joint_map(joint_values)

        theta_back = geometry.theta_back0 + geometry.sigma_back * q["back"]
        theta_front = (
            geometry.theta_front0 + geometry.sigma_front * q["front"]
        )
        theta_ab = (
            theta_back
            + geometry.rel_short0
            + geometry.sigma_short * q["short"]
        )
        theta_bc = (
            theta_ab
            + geometry.rel_elbow0
            + geometry.sigma_elbow * q["elbow"]
        )

        point_a2 = geometry.O2 + geometry.l1 * np.array(
            [math.cos(theta_back), math.sin(theta_back)],
            dtype=float,
        )
        point_b2 = point_a2 + geometry.l2 * np.array(
            [math.cos(theta_ab), math.sin(theta_ab)],
            dtype=float,
        )
        c_long2 = point_b2 + geometry.dl * np.array(
            [math.cos(theta_bc), math.sin(theta_bc)],
            dtype=float,
        )
        c_front2 = geometry.M2 + geometry.l3 * np.array(
            [math.cos(theta_front), math.sin(theta_front)],
            dtype=float,
        )
        tip_angle = theta_bc - geometry.kappa
        tip2 = c_long2 + geometry.r_tip * np.array(
            [math.cos(tip_angle), math.sin(tip_angle)],
            dtype=float,
        )

        def embed(point2: np.ndarray, reference3: np.ndarray) -> np.ndarray:
            return (
                geometry.O_H
                + geometry.eX_H * float(point2[0])
                + geometry.eY_H * float(point2[1])
                + geometry.plane_normal_H
                * float(
                    np.dot(
                        geometry.plane_normal_H,
                        reference3 - geometry.O_H,
                    )
                )
            )

        point_a_h = embed(point_a2, geometry.A_H)
        point_b_h = embed(point_b2, geometry.B_H)
        c_long_h = embed(c_long2, geometry.C_long_H)
        c_front_h = embed(c_front2, geometry.C_front_H)
        tip_h = embed(tip2, geometry.D0_H)

        point_a = geometry.p_AH0 + geometry.R_AH0 @ (
            geometry.linear_axis_H * q["linear"] + point_a_h
        )
        point_b = geometry.p_AH0 + geometry.R_AH0 @ (
            geometry.linear_axis_H * q["linear"] + point_b_h
        )
        c_long = geometry.p_AH0 + geometry.R_AH0 @ (
            geometry.linear_axis_H * q["linear"] + c_long_h
        )
        c_front = geometry.p_AH0 + geometry.R_AH0 @ (
            geometry.linear_axis_H * q["linear"] + c_front_h
        )
        tip_a = geometry.p_AH0 + geometry.R_AH0 @ (
            geometry.linear_axis_H * q["linear"] + tip_h
        )
        arm_rotation = rot_axis(geometry.armbase_axis_A, q["armbase"])
        point_a_world = geometry.p_A_world + geometry.R_A_world @ (
            arm_rotation @ point_a
        )
        point_b_world = geometry.p_A_world + geometry.R_A_world @ (
            arm_rotation @ point_b
        )
        c_long_world = geometry.p_A_world + geometry.R_A_world @ (
            arm_rotation @ c_long
        )
        c_front_world = geometry.p_A_world + geometry.R_A_world @ (
            arm_rotation @ c_front
        )
        tip_world = geometry.p_A_world + geometry.R_A_world @ (
            arm_rotation @ tip_a
        )
        return FingerFK(
            finger=finger,
            A_local=point_a_world,
            B_local=point_b_world,
            C_long_local=c_long_world,
            C_front_local=c_front_world,
            tip_local=tip_h,
            tip_world=tip_world,
            loop_error=float(np.linalg.norm(c_long_world - c_front_world)),
        )

    def _linear_samples(self, finger: str, prev_linear: Optional[float] = None) -> np.ndarray:
        idx = CTRL_INDEX[(finger, "linear")]
        lo, hi = [float(v) for v in self.ctrl_ranges[idx]]
        step = max(1e-6, float(self.linear_search_step_m))
        vals = arange_inclusive(lo, hi, step)
        if vals.size == 0 or abs(vals[-1] - hi) > 0.5 * step:
            vals = np.unique(np.concatenate([vals, np.array([hi], dtype=float)]))
        if prev_linear is None or not np.isfinite(prev_linear):
            ref = float(np.clip(self.linear_ref_m, lo, hi))
        else:
            ref = float(np.clip(prev_linear, lo, hi))
        order = np.argsort(np.abs(vals - ref), kind="stable")
        return vals[order].astype(float)

    def _armbase_candidates(self, geom: FingerGeometry, p_A: np.ndarray, s: float, prev_arm: Optional[float]) -> List[float]:
        # Plane constraint: n^T R_AH0^T (R_axis(-q) p_A - p_AH0) - s n^T a_l = plane_offset.
        nH = geom.plane_normal_H
        axis = normalize(geom.armbase_axis_A, fallback=np.array([1.0, 0.0, 0.0]))
        p_A = np.asarray(p_A, dtype=float).reshape(3)
        w = geom.R_AH0 @ nH
        parallel = axis * float(np.dot(axis, p_A))
        perp = p_A - parallel
        cross = np.cross(axis, p_A)
        Acoef = float(np.dot(w, perp))
        Bcoef = float(np.dot(w, -cross))
        Ccoef = float(np.dot(w, parallel - geom.p_AH0) - float(s) * float(np.dot(nH, geom.linear_axis_H)) - geom.plane_offset)
        amp = math.hypot(Acoef, Bcoef)
        lo, hi = geom.active_ranges["armbase"]
        prev = 0.0 if prev_arm is None or not np.isfinite(prev_arm) else float(prev_arm)
        candidates: List[float] = []
        if amp < 1e-12:
            if abs(Ccoef) <= 1e-9:
                q = float(np.clip(prev, lo, hi))
                candidates.append(q)
            return candidates
        rhs = -Ccoef / amp
        if rhs < -1.0 - 1e-9 or rhs > 1.0 + 1e-9:
            return []
        rhs = float(np.clip(rhs, -1.0, 1.0))
        phi = math.atan2(Bcoef, Acoef)
        alpha = math.acos(rhs)
        raw = [phi + alpha, phi - alpha]
        for q0 in raw:
            # Wrap near previous to avoid selecting an equivalent angle outside the joint range.
            for k in range(-2, 3):
                q = float(q0 + 2.0 * math.pi * k)
                if lo - 1e-9 <= q <= hi + 1e-9:
                    candidates.append(q)
        # Deduplicate and order by continuity.
        unique: List[float] = []
        for q in candidates:
            if not any(abs(q - old) < 1e-8 for old in unique):
                unique.append(q)
        return sorted(unique, key=lambda q: abs(float(wrap_to_pi(q - prev))))

    def _target_handbase(self, geom: FingerGeometry, p_world: np.ndarray, q_arm: float, s: float) -> np.ndarray:
        p_A = geom.R_A_world.T @ (np.asarray(p_world, dtype=float).reshape(3) - geom.p_A_world)
        p_after_arm = rot_axis(geom.armbase_axis_A, -float(q_arm)) @ p_A
        return geom.R_AH0.T @ (p_after_arm - geom.p_AH0) - float(s) * geom.linear_axis_H

    def _candidate_score(
        self,
        finger: str,
        q_back: float,
        q_front: float,
        q_arm: float,
        q_linear: float,
        q_prev: Optional[Tuple[float, float, float, float]],
    ) -> float:
        if q_prev is None:
            return (
                abs(float(q_back))
                + abs(float(q_front))
                + 0.15 * abs(float(q_arm))
                + 10.0 * abs(float(q_linear) - float(self.linear_ref_m))
            )
        return (
            1.0 * abs(float(wrap_to_pi(q_back - q_prev[0])))
            + 1.0 * abs(float(wrap_to_pi(q_front - q_prev[1])))
            + 0.25 * abs(float(wrap_to_pi(q_arm - q_prev[2])))
            + 10.0 * abs(float(q_linear - q_prev[3]))
            + 1.0 * abs(float(q_linear - float(self.linear_ref_m)))
        )

    def solve_finger_target(
        self,
        finger: str,
        p_world: np.ndarray,
        q_prev: Optional[Tuple[float, float, float, float]] = None,
        return_all: bool = False,
        stop_at_first: bool = False,
    ) -> List[Dict[str, Any]]:
        geom = self.geometry_model.fingers[finger]
        p_world = np.asarray(p_world, dtype=float).reshape(3)
        p_A = geom.R_A_world.T @ (p_world - geom.p_A_world)
        prev_linear = None if q_prev is None else float(q_prev[3])
        prev_arm = None if q_prev is None else float(q_prev[2])
        solutions: List[Dict[str, Any]] = []

        for s in self._linear_samples(finger, prev_linear):
            if not geom.in_active_range("linear", float(s)):
                continue
            for q_arm in self._armbase_candidates(geom, p_A, float(s), prev_arm):
                if not geom.in_active_range("armbase", q_arm):
                    continue
                p_H = self._target_handbase(geom, p_world, q_arm, float(s))
                D = geom.project2(p_H)
                C_candidates = circle_intersections(geom.M2, geom.l3, D, geom.r_tip)
                for C in C_candidates:
                    if np.linalg.norm(D - C) < 1e-12:
                        continue
                    u_bc = rot2(geom.kappa) @ ((D - C) / geom.r_tip)
                    B = C - geom.dl * u_bc
                    rho2 = float(np.dot(B, B))
                    c2 = (rho2 - geom.l1 * geom.l1 - geom.l2 * geom.l2) / (2.0 * geom.l1 * geom.l2)
                    if c2 < -1.0 - 1e-10 or c2 > 1.0 + 1e-10:
                        continue
                    c2 = float(np.clip(c2, -1.0, 1.0))
                    for elbow_sign in (+1.0, -1.0):
                        s2 = elbow_sign * math.sqrt(max(0.0, 1.0 - c2 * c2))
                        theta_back = math.atan2(float(B[1]), float(B[0])) - math.atan2(
                            geom.l2 * s2, geom.l1 + geom.l2 * c2
                        )
                        A = geom.l1 * np.array([math.cos(theta_back), math.sin(theta_back)], dtype=float)
                        theta_ab = math.atan2(float(B[1] - A[1]), float(B[0] - A[0]))
                        theta_front = math.atan2(float(C[1] - geom.M2[1]), float(C[0] - geom.M2[0]))
                        theta_bc = math.atan2(float(C[1] - B[1]), float(C[0] - B[0]))

                        q_back = geom.sigma_back * float(wrap_to_pi(theta_back - geom.theta_back0))
                        q_front = geom.sigma_front * float(wrap_to_pi(theta_front - geom.theta_front0))
                        q_short = geom.sigma_short * float(wrap_to_pi((theta_ab - theta_back) - geom.rel_short0))
                        q_elbow = geom.sigma_elbow * float(wrap_to_pi((theta_bc - theta_ab) - geom.rel_elbow0))

                        if not geom.in_active_range("back", q_back):
                            continue
                        if not geom.in_active_range("front", q_front):
                            continue
                        if self.filter_passive_limits:
                            if not geom.in_passive_range("short", q_short):
                                continue
                            if not geom.in_passive_range("elbow", q_elbow):
                                continue
                        score = self._candidate_score(finger, q_back, q_front, q_arm, float(s), q_prev)
                        solutions.append(
                            {
                                "finger": finger,
                                "q_back": float(q_back),
                                "q_front": float(q_front),
                                "q_armbase": float(q_arm),
                                "q_linear": float(s),
                                "q_short": float(q_short),
                                "q_elbow": float(q_elbow),
                                "score": float(score),
                                "D2": D.astype(float),
                                "B2": B.astype(float),
                                "C2": C.astype(float),
                                "p_H": p_H.astype(float),
                            }
                        )
                        if stop_at_first:
                            return solutions
        return sorted(solutions, key=lambda sol: float(sol["score"]))

    def finger_target_feasible(
        self,
        finger: str,
        p_world: Sequence[float],
    ) -> bool:
        return bool(
            self.solve_finger_target(
                finger.upper(),
                np.asarray(p_world, dtype=float),
                q_prev=None,
                return_all=False,
                stop_at_first=True,
            )
        )

    def target_out_of_plane_error(
        self,
        finger: str,
        target_world: Sequence[float],
    ) -> float:
        # Armbase rotation plus the child slide makes the linkage plane
        # configuration-dependent; there is no invariant plane residual.
        if finger.upper() not in self.fingers:
            raise KeyError(f"unknown finger {finger!r}")
        np.asarray(target_world, dtype=float).reshape(3)
        return 0.0

    def solve_tip_targets(
        self,
        tip_targets: Mapping[str, Sequence[float]],
        prev_ctrl: Optional[np.ndarray] = None,
        fallback_ctrl: Optional[np.ndarray] = None,
        *,
        previous_ctrl: Optional[Sequence[float]] = None,
        previous_solutions: Optional[Mapping[str, FingerSolution]] = None,
        strict: bool = False,
    ) -> IKSolutionSummary:
        if previous_ctrl is not None:
            if prev_ctrl is not None:
                raise ValueError("use only one of prev_ctrl and previous_ctrl")
            prev_ctrl = np.asarray(previous_ctrl, dtype=float)
        if fallback_ctrl is None:
            ctrl = (
                np.asarray(prev_ctrl, dtype=float).copy()
                if prev_ctrl is not None
                else np.zeros(ACTUATOR_COUNT, dtype=float)
            )
        else:
            ctrl = np.asarray(fallback_ctrl, dtype=float).copy()
        ctrl = ctrl.reshape(ACTUATOR_COUNT)
        solved = np.zeros(ACTUATOR_COUNT, dtype=bool)
        failed: List[str] = []
        debug: Dict[str, Dict[str, Any]] = {}
        finger_solutions: Dict[str, FingerSolution] = {}
        for finger in OBS_FINGER_ORDER:
            if finger not in tip_targets:
                raise ValueError(f"missing fingertip target for finger {finger}")
            q_prev = (
                ctrl_to_finger_q(prev_ctrl, finger)
                if prev_ctrl is not None
                else None
            )
            if q_prev is None and previous_solutions and finger in previous_solutions:
                previous = previous_solutions[finger]
                q_prev = (
                    previous.q_back,
                    previous.q_front,
                    previous.q_armbase,
                    previous.q_linear,
                )
            target = np.asarray(tip_targets[finger], dtype=float).reshape(3)
            sols = self.solve_finger_target(
                finger,
                target,
                q_prev=q_prev,
            )
            if not sols:
                failed.append(finger)
                debug[finger] = {"failed": True, "reason": "no_valid_branch"}
                continue
            sol = sols[0]
            set_finger_ctrl(
                ctrl,
                finger,
                sol["q_back"],
                sol["q_front"],
                sol["q_armbase"],
                sol["q_linear"],
            )
            for which in ("back", "front", "armbase", "linear"):
                solved[CTRL_INDEX[(finger, which)]] = True
            debug[finger] = {k: v for k, v in sol.items() if k not in {"D2", "B2", "C2", "p_H"}}
            fk = self.finger_forward(
                finger,
                {
                    "linear": sol["q_linear"],
                    "armbase": sol["q_armbase"],
                    "back": sol["q_back"],
                    "short": sol["q_short"],
                    "elbow": sol["q_elbow"],
                    "front": sol["q_front"],
                },
            )
            finger_solutions[finger] = FingerSolution(
                finger=finger,
                q_linear=float(sol["q_linear"]),
                q_armbase=float(sol["q_armbase"]),
                q_back=float(sol["q_back"]),
                q_short=float(sol["q_short"]),
                q_elbow=float(sol["q_elbow"]),
                q_front=float(sol["q_front"]),
                tip_world=fk.tip_world.copy(),
                tip_error=float(np.linalg.norm(fk.tip_world - target)),
                loop_error=float(fk.loop_error),
                score=float(sol["score"]),
            )
        result = IKSolutionSummary(
            ctrl=clip_ctrl(ctrl, self.ctrl_ranges),
            solved_mask=solved,
            failed_fingers=tuple(failed),
            finger_solutions=finger_solutions,
            max_tip_error=max(
                (solution.tip_error for solution in finger_solutions.values()),
                default=math.inf,
            ),
            max_loop_error=max(
                (solution.loop_error for solution in finger_solutions.values()),
                default=math.inf,
            ),
            branch_debug=debug,
        )
        if strict and failed:
            raise IKUnreachableError(
                "unreachable fingertip target(s): " + ", ".join(failed)
            )
        return result

    def solve_cube_pose(
        self,
        center_world: Sequence[float],
        quat_world_object: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        distances: Optional[float | Mapping[str, float]] = None,
        z_offset: float = 0.0,
        prev_ctrl: Optional[np.ndarray] = None,
        fallback_ctrl: Optional[np.ndarray] = None,
        *,
        distance: Optional[float] = None,
        previous_ctrl: Optional[Sequence[float]] = None,
        previous_solutions: Optional[Mapping[str, FingerSolution]] = None,
        strict: bool = False,
    ) -> IKSolutionSummary:
        if distances is not None and distance is not None:
            raise ValueError("use only one of distances and distance")
        target_distances: float | Mapping[str, float]
        if distance is not None:
            target_distances = float(distance)
        elif distances is not None:
            target_distances = distances
        else:
            target_distances = self.model_info.contact_distance
        targets = self.cube_tip_targets(
            center_world,
            quat_world_object,
            distances=target_distances,
            z_offset=z_offset,
        )
        return self.solve_tip_targets(
            targets,
            prev_ctrl=prev_ctrl,
            fallback_ctrl=fallback_ctrl,
            previous_ctrl=previous_ctrl,
            previous_solutions=previous_solutions,
            strict=strict,
        )

    def cube_tip_targets(
        self,
        center_world: Sequence[float],
        quat_world_object: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        *,
        distances: Optional[float | Mapping[str, float]] = None,
        distance: Optional[float] = None,
        z_offset: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        if distances is not None and distance is not None:
            raise ValueError("use only one of distances and distance")
        value: float | Mapping[str, float]
        if distance is not None:
            value = float(distance)
        elif distances is not None:
            value = distances
        else:
            value = self.model_info.contact_distance
        center = np.asarray(center_world, dtype=float).reshape(3)
        rotation = quat_to_rot(np.asarray(quat_world_object, dtype=float))
        vertical = rotation @ np.array([0.0, 0.0, float(z_offset)])
        if isinstance(value, Mapping):
            distance_by_finger = {
                finger: float(value[finger]) for finger in FINGERS
            }
        else:
            distance_by_finger = {
                finger: float(value) for finger in FINGERS
            }
        return {
            finger: center
            + vertical
            + rotation
            @ (
                self.contact_directions_world[finger]
                * distance_by_finger[finger]
            )
            for finger in FINGERS
        }

    def cube_pose_feasible(
        self,
        center_world: Sequence[float],
        quat_world_object: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        *,
        distance: Optional[float] = None,
        z_offset: float = 0.0,
    ) -> bool:
        targets = self.cube_tip_targets(
            center_world,
            quat_world_object,
            distance=(
                self.model_info.contact_distance
                if distance is None
                else float(distance)
            ),
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
        """Find the largest feasible cylindrical radius at one azimuth."""
        if coarse_step <= 0.0 or resolution <= 0.0:
            raise ValueError("workspace search steps must be positive")
        if resolution > coarse_step:
            raise ValueError("resolution must not exceed coarse_step")

        theta_rad = float(theta) if theta_radians else math.radians(float(theta))
        target_distance = (
            self.model_info.contact_distance
            if distance is None
            else float(distance)
        )
        origin = np.asarray(origin_xy, dtype=float).reshape(2)
        quaternion = normalize_quat(np.asarray(quaternion_world, dtype=float))
        reach_by_finger: Dict[str, float] = {}
        for finger, geometry in self.fingers.items():
            linear_limits = geometry.active_ranges["linear"]
            reach_by_finger[finger] = (
                float(np.linalg.norm(geometry.p_AH0))
                + max(abs(value) for value in linear_limits)
                + float(np.linalg.norm(geometry.O_H))
                + geometry.l1
                + geometry.l2
                + geometry.dl
                + geometry.r_tip
            )
        if z_min is None:
            z_min = self.model_info.minimum_cube_center_z
        if z_max is None:
            z_max = max(
                float(geometry.p_A_world[2]) + reach_by_finger[finger]
                for finger, geometry in self.fingers.items()
            )
        if r_max is None:
            r_max = max(
                float(np.linalg.norm(geometry.p_A_world[:2] - origin))
                + reach_by_finger[finger]
                + target_distance
                for finger, geometry in self.fingers.items()
            )
        z_low = float(z_min)
        z_high = float(z_max)
        r_high = float(r_max)
        if not z_low < z_high:
            raise ValueError("z_min must be smaller than z_max")
        if r_high <= 0.0:
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
            radius_low: float,
            radius_high: float,
            height_low: float,
            height_high: float,
            step: float,
        ) -> Optional[Tuple[float, List[float]]]:
            for radius in arange_inclusive(
                radius_low,
                radius_high,
                step,
            )[::-1]:
                hits = [
                    float(height)
                    for height in arange_inclusive(
                        height_low,
                        height_high,
                        step,
                    )
                    if feasible(float(radius), float(height))
                ]
                if hits:
                    return float(radius), hits
            return None

        step = float(coarse_step)
        found = search_level(0.0, r_high, z_low, z_high, step)
        if found is None:
            raise IKUnreachableError(
                "no feasible cube center found in the requested workspace bounds"
            )
        best_radius, best_heights = found
        while step > resolution:
            previous_step = step
            step = max(float(resolution), previous_step / 10.0)
            refined = search_level(
                max(0.0, best_radius - previous_step),
                min(r_high, best_radius + 2.0 * previous_step),
                max(z_low, min(best_heights) - 2.0 * previous_step),
                min(z_high, max(best_heights) + 2.0 * previous_step),
                step,
            )
            if refined is None:
                raise KinematicsError(
                    "workspace refinement lost the feasible region"
                )
            best_radius, best_heights = refined
        best_z = best_heights[len(best_heights) // 2]
        return CylindricalWorkspaceLimit(
            theta_deg=math.degrees(theta_rad),
            radius=best_radius,
            z=best_z,
            xyz=cylindrical_to_xyz(
                best_z,
                best_radius,
                theta_rad,
                theta_radians=True,
                origin_xy=origin,
            ),
            distance=target_distance,
            resolution=step,
        )


class IKTeacher:
    """Stateful continuity wrapper shared by control and learning code."""

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
        initial = (
            np.zeros(ACTUATOR_COUNT, dtype=float)
            if ctrl is None
            else np.asarray(ctrl, dtype=float).reshape(ACTUATOR_COUNT)
        )
        self.previous_ctrl = clip_ctrl(initial, self.ik.ctrl_ranges)
        self.previous_solutions: Dict[str, FingerSolution] = {}

    def commit(
        self,
        result: IKSolutionSummary,
        *,
        ctrl: Optional[Sequence[float]] = None,
    ) -> None:
        if not result.success:
            raise ValueError("cannot commit an unsuccessful IK result")
        command = result.ctrl if ctrl is None else ctrl
        self.previous_ctrl = clip_ctrl(
            np.asarray(command, dtype=float),
            self.ik.ctrl_ranges,
        )
        self.previous_solutions = dict(result.finger_solutions)

    def solve_tip_targets(
        self,
        targets_world: Mapping[str, Sequence[float]],
        *,
        strict: bool = False,
        update: bool = True,
    ) -> IKSolutionSummary:
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
    ) -> IKSolutionSummary:
        result = self.ik.solve_cube_pose(
            center_world,
            quaternion_world,
            distance=distance,
            z_offset=z_offset,
            previous_ctrl=self.previous_ctrl,
            previous_solutions=self.previous_solutions,
            strict=strict,
        )
        if update and result.success:
            self.commit(result)
        return result


def rot2(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def ctrl_to_finger_q(ctrl: Optional[np.ndarray], finger: str) -> Optional[Tuple[float, float, float, float]]:
    if ctrl is None:
        return None
    ctrl = np.asarray(ctrl, dtype=float).reshape(ACTUATOR_COUNT)
    return (
        float(ctrl[CTRL_INDEX[(finger, "back")]]),
        float(ctrl[CTRL_INDEX[(finger, "front")]]),
        float(ctrl[CTRL_INDEX[(finger, "armbase")]]),
        float(ctrl[CTRL_INDEX[(finger, "linear")]]),
    )


def set_finger_ctrl(ctrl: np.ndarray, finger: str, q_back: float, q_front: float, q_armbase: float, q_linear: float) -> None:
    ctrl[CTRL_INDEX[(finger, "back")]] = float(q_back)
    ctrl[CTRL_INDEX[(finger, "front")]] = float(q_front)
    ctrl[CTRL_INDEX[(finger, "armbase")]] = float(q_armbase)
    ctrl[CTRL_INDEX[(finger, "linear")]] = float(q_linear)


def cube_tip_targets(
    center_world: Sequence[float],
    quat_world_object: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    distances: Optional[float | Mapping[str, float]] = None,
    z_offset: float = 0.0,
    *,
    distance: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    if distances is not None and distance is not None:
        raise ValueError("use only one of distances and distance")
    if distance is not None:
        distances = float(distance)
    if distances is None:
        raise ValueError("distance(s) must be provided")
    center_world = np.asarray(center_world, dtype=float).reshape(3)
    r_obj = quat_to_rot(quat_world_object)
    if isinstance(distances, Mapping):
        dist_map = {f: float(distances[f]) for f in OBS_FINGER_ORDER}
    else:
        dist_map = {f: float(distances) for f in OBS_FINGER_ORDER}
    offsets = {
        "U": np.array([0.0, +dist_map["U"], z_offset], dtype=float),
        "D": np.array([0.0, -dist_map["D"], z_offset], dtype=float),
        "L": np.array([-dist_map["L"], 0.0, z_offset], dtype=float),
        "R": np.array([+dist_map["R"], 0.0, z_offset], dtype=float),
    }
    return {finger: center_world + r_obj @ offset for finger, offset in offsets.items()}


def yaw_quat(yaw_rad: float) -> np.ndarray:
    yaw = float(yaw_rad)
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=float)

# =============================================================================
# Workspace CSV lookup
# =============================================================================


class NoValidWorkspaceTarget(RuntimeError):
    pass


@dataclass(frozen=True)
class CombinedBoundaryRow:
    axis_combination: str
    z_m: float
    angle_deg: float
    rmax_m: float


class CombinedBoundaryLookup:
    """Exact ``axis combination + z + max component angle -> rmax`` lookup.

    The boundary file is generated by the local T6 workspace scanner. Multi-axis
    rows use one common component magnitude, so an independently sampled target
    such as ``[30, -45, 15]`` intentionally queries the ``XYZ, 45 deg`` row.
    """

    VALID_COMBINATIONS = ("x", "y", "z", "xy", "xz", "yz", "xyz")

    def __init__(
        self,
        rows: Sequence[CombinedBoundaryRow],
        center_xy: Tuple[float, float] = (0.0, 0.0),
        z_resolution_m: float = 0.001,
        angle_resolution_deg: float = 1.0,
    ):
        if not rows:
            raise ValueError("CombinedBoundaryLookup requires at least one boundary row")
        self.rows = tuple(rows)
        self.center_xy = np.asarray(center_xy, dtype=float).reshape(2)
        self.z_resolution_m = float(z_resolution_m)
        self.angle_resolution_deg = float(angle_resolution_deg)
        self._lookup: Dict[Tuple[str, int, int], float] = {}
        for row in self.rows:
            combo = self.normalize_combination(row.axis_combination)
            key = (combo, self._z_key(row.z_m), self._angle_key(row.angle_deg))
            value = max(0.0, float(row.rmax_m))
            if key in self._lookup and abs(self._lookup[key] - value) > 1e-12:
                raise ValueError(f"Conflicting duplicate boundary row for {key}: {self._lookup[key]} vs {value}")
            self._lookup[key] = value
        self.z_values = np.asarray(sorted({self.quantize_z(row.z_m) for row in self.rows}), dtype=float)
        self.angle_values = np.asarray(
            sorted({self.quantize_angle(row.angle_deg) for row in self.rows}), dtype=float
        )
        self.r_max_grid = np.asarray(tuple(self._lookup.values()), dtype=float)
        self.z_min = float(np.min(self.z_values))
        self.z_max = float(np.max(self.z_values))
        self.angle_min = float(np.min(self.angle_values))
        self.angle_max = float(np.max(self.angle_values))

    @classmethod
    def from_csv(
        cls,
        csv_path: str | os.PathLike[str],
        center_xy: Tuple[float, float] = (0.0, 0.0),
        z_resolution_m: float = 0.001,
        angle_resolution_deg: float = 1.0,
    ) -> "CombinedBoundaryLookup":
        path = _resolve_local_path(
            csv_path,
            label="T6 boundary CSV",
            must_exist=False,
        )
        if not path.is_file():
            raise FileNotFoundError(f"T6 boundary CSV not found: {path}")
        parsed: List[CombinedBoundaryRow] = []
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {
                "axis_combination",
                "z_m",
                "object_rotation_angle_deg",
                "max_inner_circle_r_m",
            }
            fields = set(reader.fieldnames or ())
            missing = sorted(required - fields)
            if missing:
                raise ValueError(f"Boundary CSV {path} is missing columns: {missing}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    parsed.append(
                        CombinedBoundaryRow(
                            axis_combination=str(row["axis_combination"]),
                            z_m=float(row["z_m"]),
                            angle_deg=float(row["object_rotation_angle_deg"]),
                            rmax_m=float(row["max_inner_circle_r_m"]),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid boundary row at line {line_number}: {exc}") from exc
        return cls(
            parsed,
            center_xy=center_xy,
            z_resolution_m=z_resolution_m,
            angle_resolution_deg=angle_resolution_deg,
        )

    @classmethod
    def normalize_combination(cls, value: str | Sequence[str]) -> str:
        raw = "".join(value) if not isinstance(value, str) else value
        selected = {char.lower() for char in str(raw) if char.lower() in "xyz"}
        combo = "".join(axis for axis in "xyz" if axis in selected)
        if combo not in cls.VALID_COMBINATIONS:
            raise ValueError(f"Invalid axis combination {value!r}; expected one of {cls.VALID_COMBINATIONS}")
        return combo

    def _z_key(self, z_m: float) -> int:
        return int(round(float(z_m) / self.z_resolution_m))

    def _angle_key(self, angle_deg: float) -> int:
        return int(round(float(angle_deg) / self.angle_resolution_deg))

    def quantize_z(self, z_m: float) -> float:
        return round(self._z_key(z_m) * self.z_resolution_m, 6)

    def quantize_angle(self, angle_deg: float) -> float:
        return round(self._angle_key(angle_deg) * self.angle_resolution_deg, 6)

    def has(self, axis_combination: str, z_m: float, max_component_angle_deg: float) -> bool:
        combo = self.normalize_combination(axis_combination)
        return (combo, self._z_key(z_m), self._angle_key(abs(max_component_angle_deg))) in self._lookup

    def r_max(self, axis_combination: str, z_m: float, max_component_angle_deg: float) -> float:
        combo = self.normalize_combination(axis_combination)
        key = (combo, self._z_key(z_m), self._angle_key(abs(max_component_angle_deg)))
        if key not in self._lookup:
            raise NoValidWorkspaceTarget(
                "No boundary row for "
                f"axis_combination={combo}, z={self.quantize_z(z_m):.3f} m, "
                f"angle={self.quantize_angle(abs(max_component_angle_deg)):.1f} deg"
            )
        return float(self._lookup[key])

    def target_xyz_from_cyl(self, r_m: float, theta_rad: float, z_m: float) -> np.ndarray:
        return np.array(
            [
                float(self.center_xy[0]) + float(r_m) * math.cos(float(theta_rad)),
                float(self.center_xy[1]) + float(r_m) * math.sin(float(theta_rad)),
                float(z_m),
            ],
            dtype=float,
        )

    def cylindrical_from_xyz(self, xyz: np.ndarray) -> Tuple[float, float, float]:
        point = np.asarray(xyz, dtype=float).reshape(3)
        rel = point[:2] - self.center_xy
        radius = float(np.linalg.norm(rel))
        theta = float(math.atan2(float(rel[1]), float(rel[0]))) if radius > 1e-12 else 0.0
        return radius, theta, float(point[2])

    def summary(self) -> Dict[str, Any]:
        return {
            "rows": len(self._lookup),
            "axis_combinations": tuple(sorted({key[0] for key in self._lookup})),
            "z_min_m": self.z_min,
            "z_max_m": self.z_max,
            "angle_min_deg": self.angle_min,
            "angle_max_deg": self.angle_max,
            "rmax_max_m": float(np.max(self.r_max_grid)),
        }


@dataclass(frozen=True)
class WorkspaceRow:
    z_m: float
    rmax_m: float


class WorkspaceLookup:
    """Exact z-grid lookup for z -> max inner circle radius from CSV."""

    def __init__(self, rows: Sequence[WorkspaceRow], center_xy: Tuple[float, float] = (0.0, 0.0), z_resolution_m: float = 0.001):
        if not rows:
            raise ValueError("WorkspaceLookup requires at least one CSV row")
        self.rows = tuple(sorted(rows, key=lambda r: r.z_m))
        self.center_xy = np.asarray(center_xy, dtype=float).reshape(2)
        self.z_resolution_m = float(z_resolution_m)
        self.z_keys = np.asarray([self._key(r.z_m) for r in self.rows], dtype=np.int64)
        self.z_values = np.asarray([r.z_m for r in self.rows], dtype=float)
        self.r_max_grid = np.asarray([r.rmax_m for r in self.rows], dtype=float)
        self._lookup: Dict[int, float] = {int(k): float(r) for k, r in zip(self.z_keys, self.r_max_grid)}
        self.z_min = float(np.min(self.z_values))
        self.z_max = float(np.max(self.z_values))

    @classmethod
    def from_csv(
        cls,
        csv_path: str | os.PathLike[str],
        alternatives: Sequence[str | os.PathLike[str]] = (),
        center_xy: Tuple[float, float] = (0.0, 0.0),
        z_resolution_m: float = 0.001,
    ) -> "WorkspaceLookup":
        candidates = [
            _resolve_local_path(
                path,
                label="SCREW workspace CSV",
                must_exist=False,
            )
            for path in (csv_path, *alternatives)
        ]
        actual: Optional[Path] = None
        for p in candidates:
            if p.is_file():
                actual = p
                break
        if actual is None:
            searched = "\n  ".join(str(p) for p in candidates)
            raise FileNotFoundError(f"Workspace CSV not found. Searched:\n  {searched}")
        rows: List[WorkspaceRow] = []
        with actual.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"Workspace CSV has no header: {actual}")
            fields = {name.strip(): name for name in reader.fieldnames}
            z_col = fields.get("z_m") or fields.get("z") or fields.get("Z")
            r_col = fields.get("max_inner_circle_radius_m") or fields.get("rmax") or fields.get("r_max") or fields.get("max_r")
            if z_col is None or r_col is None:
                raise ValueError(
                    f"Workspace CSV columns must include z_m and max_inner_circle_radius_m, or z/rmax. Got {reader.fieldnames}"
                )
            for row in reader:
                try:
                    z = float(row[z_col])
                    r = float(row[r_col])
                except Exception:
                    continue
                if math.isfinite(z) and math.isfinite(r):
                    rows.append(WorkspaceRow(z_m=round(z, 6), rmax_m=max(0.0, float(r))))
        return cls(rows=rows, center_xy=center_xy, z_resolution_m=z_resolution_m)

    def _key(self, z_m: float) -> int:
        return int(round(float(z_m) / self.z_resolution_m))

    def quantize_z(self, z_m: float) -> float:
        return round(self._key(z_m) * self.z_resolution_m, 6)

    def has_z(self, z_m: float) -> bool:
        return int(self._key(z_m)) in self._lookup

    def r_max(self, z_m: float, scaled: bool = False, safety_scale: float = 0.95) -> float:
        key = self._key(z_m)
        if key not in self._lookup:
            return 0.0
        r = float(self._lookup[key])
        return r * float(safety_scale) if scaled else r

    def valid_z_values_in_bounds(self, z_low_m: float, z_high_m: float, min_rmax_m: float = 0.010) -> np.ndarray:
        low = min(float(z_low_m), float(z_high_m)) - 1e-9
        high = max(float(z_low_m), float(z_high_m)) + 1e-9
        mask = (self.z_values >= low) & (self.z_values <= high) & (self.r_max_grid >= float(min_rmax_m))
        return self.z_values[mask].copy()

    def scan_next_valid_z(
        self,
        start_z_m: float,
        z_low_m: float,
        z_high_m: float,
        min_rmax_m: float = 0.010,
        step_m: float = 0.001,
    ) -> Optional[float]:
        low = self.quantize_z(min(z_low_m, z_high_m))
        high = self.quantize_z(max(z_low_m, z_high_m))
        z = self.quantize_z(start_z_m)
        if z < low or z > high:
            z = low
        n = int(round((high - low) / float(step_m))) + 1
        for _ in range(max(1, n)):
            if self.has_z(z) and self.r_max(z, scaled=False) >= float(min_rmax_m):
                return float(z)
            z = self.quantize_z(z + float(step_m))
            if z > high + 1e-12:
                z = low
        return None

    def sample_valid_z(
        self,
        z_low_m: float,
        z_high_m: float,
        rng: np.random.Generator,
        min_rmax_m: float = 0.010,
        step_m: float = 0.001,
    ) -> float:
        low = self.quantize_z(min(z_low_m, z_high_m))
        high = self.quantize_z(max(z_low_m, z_high_m))
        if high < low:
            raise NoValidWorkspaceTarget(f"invalid z bounds: {z_low_m}, {z_high_m}")
        mask = (
            (self.z_values >= low - 1e-12)
            & (self.z_values <= high + 1e-12)
            & (self.r_max_grid >= float(min_rmax_m))
        )
        candidates = self.z_values[mask]
        if candidates.size <= 0:
            raise NoValidWorkspaceTarget(
                f"No z with rmax >= {min_rmax_m:.4f} in [{low:.6f}, {high:.6f}] according to workspace CSV."
            )
        return float(candidates[int(rng.integers(0, int(candidates.size)))])

    def target_xyz_from_cyl(self, r_m: float, theta_rad: float, z_m: float) -> np.ndarray:
        return np.array(
            [
                float(self.center_xy[0]) + float(r_m) * math.cos(float(theta_rad)),
                float(self.center_xy[1]) + float(r_m) * math.sin(float(theta_rad)),
                float(z_m),
            ],
            dtype=float,
        )

    def cylindrical_from_xyz(self, xyz: np.ndarray) -> Tuple[float, float, float]:
        xyz = np.asarray(xyz, dtype=float).reshape(3)
        rel = xyz[:2] - self.center_xy
        r = float(np.linalg.norm(rel))
        theta = float(math.atan2(float(rel[1]), float(rel[0]))) if r > 1e-12 else 0.0
        return r, theta, float(xyz[2])

    def target_inside_lookup_circle(self, xyz: np.ndarray, scaled: bool = True, safety_scale: float = 0.95) -> bool:
        r, _theta, z = self.cylindrical_from_xyz(xyz)
        rmax = self.r_max(z, scaled=scaled, safety_scale=safety_scale)
        return bool(r <= rmax + 1e-12 and rmax > 0.0)

    def segmented_path_reachable(
        self,
        target_xyz: np.ndarray,
        start_z_m: float,
        step_m: float = 0.001,
        safety_scale: float = 0.95,
        min_rmax_m: float = 0.010,
        lift_z_m: Optional[float] = None,
    ) -> bool:
        target = np.asarray(target_xyz, dtype=float).reshape(3)
        target_r, _theta, target_z = self.cylindrical_from_xyz(target)
        lift_z = self.quantize_z(start_z_m if lift_z_m is None else lift_z_m)
        step = max(float(step_m), float(self.z_resolution_m))

        def z_samples(a: float, b: float) -> List[float]:
            lo = self.quantize_z(min(float(a), float(b)))
            hi = self.quantize_z(max(float(a), float(b)))
            n = int(round((hi - lo) / step)) + 1
            vals = [self.quantize_z(lo + i * step) for i in range(max(1, n))]
            if not vals or abs(vals[-1] - hi) > 1e-12:
                vals.append(hi)
            return sorted(set(vals))

        def layer_has_clearance(z: float, radius: float) -> bool:
            zq = self.quantize_z(z)
            if not self.has_z(zq):
                return False
            if self.r_max(zq, scaled=False) < float(min_rmax_m):
                return False
            return bool(float(radius) <= self.r_max(zq, scaled=True, safety_scale=safety_scale) + 1e-12)

        for z in z_samples(start_z_m, lift_z):
            if not layer_has_clearance(z, 0.0):
                return False
        if not layer_has_clearance(lift_z, target_r):
            return False
        for z in z_samples(lift_z, target_z):
            if not layer_has_clearance(z, target_r):
                return False
        return True

    def workspace_summary(self) -> Dict[str, float]:
        return {
            "z_min_csv": float(self.z_min),
            "z_max_csv": float(self.z_max),
            "rmax_max_csv": float(np.max(self.r_max_grid)),
            "positive_layers": int(np.count_nonzero(self.r_max_grid > 0.0)),
        }


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    xml = here / "Project.xml"
    model = GripperGeometryModel.from_xml(xml)
    print("ctrl_ranges shape:", model.ctrl_ranges.shape)
    for f, g in model.fingers.items():
        print(
            f,
            "dm/l1/l2/l3/dl/r=",
            [round(v, 6) for v in (g.dm, g.l1, g.l2, g.l3, g.dl, g.r_tip)],
            "pin_mismatch=",
            f"{g.pin_mismatch_m:.6g}",
            "plane_normal=",
            np.round(g.plane_normal_H, 6),
        )
    ik = GripperIK(model, linear_search_step_m=0.001)
    info = model.xml_info
    contact = info.cube_half_size + info.hard_tip_radius - 0.003
    # Keep this executable self-test attached to the gripper root.  The table
    # and cube are independent world objects and need not move when root moves.
    test_center = np.array(
        [info.root_pos[0], info.root_pos[1], info.root_pos[2] - 0.240],
        dtype=float,
    )
    sol = ik.solve_cube_pose(test_center, info.cube_quat, distances=contact)
    print("root pose:", np.round(info.root_pos, 6))
    print("self-test center:", np.round(test_center, 6))
    print("self-test ctrl:", np.round(sol.ctrl, 6))
    print("failed:", sol.failed_fingers)
    if not sol.success:
        raise SystemExit("root-relative T6 kinematics self-test failed")
