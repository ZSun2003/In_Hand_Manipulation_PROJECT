#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gripper6 combined rotation/XY-translation workspace scan.

For every object-center height ``z`` and rotation-component magnitude, this
program finds the largest circle, centred on the gripper root XY position,
whose sampled radial prefixes are all analytically IK-feasible.

The implementation is self-contained in ``Rotation_G6``:

* ``Project.xml``, ``Gripper.xml`` and ``Share.xml`` are always loaded beside
  this file;
* analytic IK is imported only from the local ``R6_kinematics.py``;
* XML includes are already constrained by ``R6_kinematics`` to remain in this
  folder;
* on Linux/CX a direct launch uses ``CX_env`` created by ``R6_setup_cx.sh``;
  on macOS the existing local ``RL_env`` is used for development runs.

Rotation definition
-------------------
The convention deliberately matches Gripper2 ``both``.  The sampled angle is
assigned to every active *world-axis rotation-vector component*.  For example,
XY at 30 degrees means the rotation vector ``[30, 30, 0]`` degrees.  It is not
an Euler sequence and the total axis-angle magnitude is ``sqrt(2) * 30``.

By default, every +/- sign pattern for the active components must be feasible.
The three numerical aggregate conditions are:

* single axis: pointwise ``min(X, Y, Z)``;
* double axis: pointwise ``min(XY, XZ, YZ)``;
* triple axis: ``XYZ``.

Inscribed-circle definition
---------------------------
At fixed ``(z, rotation)`` every translation direction theta starts at r=0
and advances on the strict r grid.  The first infeasible sample closes that
ray; disconnected feasible islands farther out are intentionally ignored.
The result is the minimum contiguous-prefix radius over all theta rays.

Production defaults requested for root z=0.600 m:

* z: 0.285 .. 0.500 m, step 0.001 m;
* rotation component: 0 .. 90 deg, step 1 deg;
* translation r: 0 .. 0.180 m, step 0.001 m;
* translation theta: 0 .. <360 deg, step 9 deg (40 rays);
* 50 mm diameter disk grasp; hard-tip radius comes from the local XML.

Only analytic closed-chain IK and XML joint/actuator limits are evaluated.
Optional disk/hard-tip clearance against the XML table is enabled by default.
MuJoCo link collision, self-collision, contact forces and dynamics are outside
the scope of this workspace calculation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Relocatable, Rotation_G6-local Python bootstrap.
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent
CX_ENV_DIR = (Path(os.environ["IRP_PYTHON_ENV"]) if os.environ.get("IRP_PYTHON_ENV") else next((parent / "RL_env" for parent in Path(__file__).resolve().parents if (parent / "RL_env" / "build.sh").is_file()), Path(sys.prefix)))
MAC_ENV_DIR = (Path(os.environ["IRP_PYTHON_ENV"]) if os.environ.get("IRP_PYTHON_ENV") else next((parent / "RL_env" for parent in Path(__file__).resolve().parents if (parent / "RL_env" / "build.sh").is_file()), Path(sys.prefix)))


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        _bootstrap_envs = (CX_ENV_DIR,)
    elif sys.platform == "darwin":
        _bootstrap_envs = (MAC_ENV_DIR, CX_ENV_DIR)
    else:
        _bootstrap_envs = (CX_ENV_DIR, MAC_ENV_DIR)

    _already_local = any(
        env.is_dir() and _same_path(Path(sys.prefix), env)
        for env in _bootstrap_envs
    )
    if not _already_local:
        _local_python = next(
            (env / "bin" / "python" for env in _bootstrap_envs if (env / "bin" / "python").is_file()),
            None,
        )
        if _local_python is None:
            expected = CX_ENV_DIR if sys.platform.startswith("linux") else MAC_ENV_DIR
            raise SystemExit(
                f"Local virtual environment not found: {expected}\n"
                "On Imperial CX run: bash R6_setup_cx.sh"
            )
        os.execv(
            str(_local_python),
            [str(_local_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        )


import numpy as np  # noqa: E402

from R6_kinematics import (  # noqa: E402
    DEFAULT_XML,
    FINGERS,
    GripperIK,
    TIP_OFFSETS_UNIT,
    arange_inclusive,
)


AXIS_NAMES: Tuple[str, ...] = ("X", "Y", "Z")
AXIS_COMBINATIONS: Mapping[int, Tuple[Tuple[str, Tuple[int, ...]], ...]] = {
    1: (("X", (0,)), ("Y", (1,)), ("Z", (2,))),
    2: (("XY", (0, 1)), ("XZ", (0, 2)), ("YZ", (1, 2))),
    3: (("XYZ", (0, 1, 2)),),
}
COMBO_TO_AXES: Mapping[str, Tuple[int, ...]] = {
    label: axes
    for combinations in AXIS_COMBINATIONS.values()
    for label, axes in combinations
}
GROUP_NAMES: Mapping[int, str] = {1: "single", 2: "double", 3: "triple"}
GROUP_COMBOS: Mapping[int, Tuple[str, ...]] = {
    count: tuple(label for label, _axes in combinations)
    for count, combinations in AXIS_COMBINATIONS.items()
}

FORMAT_VERSION = 1
DEFAULT_ROOT_HEIGHT_M = 0.600
DEFAULT_Z_MIN_M = 0.285
DEFAULT_Z_MAX_M = 0.500
DEFAULT_Z_STEP_M = 0.001
DEFAULT_ANGLE_MIN_DEG = 0.0
DEFAULT_ANGLE_MAX_DEG = 90.0
DEFAULT_ANGLE_STEP_DEG = 1.0
DEFAULT_R_SEARCH_MAX_M = 0.180
DEFAULT_R_STEP_M = 0.001
DEFAULT_THETA_STEP_DEG = 9.0
DEFAULT_DISK_DIAMETER_M = 0.050
DEFAULT_OUTPUT_DIR = MODULE_DIR / "R6_rotation_translation_workspace_outputs"


@dataclass(frozen=True)
class ScanConfig:
    output_dir: str
    root_height_m: float
    xml_root_height_m: float
    root_height_offset_m: float
    center_xy_m: Tuple[float, float]
    table_center_xy_m: Tuple[float, float]
    table_top_z_m: float
    table_half_x_m: float
    table_half_y_m: float
    disk_diameter_m: float
    disk_radius_m: float
    hard_tip_radius_m: float
    tip_squeeze_margin_m: float
    tip_center_distance_m: float
    z_min_m: float
    z_max_m: float
    z_step_m: float
    angle_min_deg: float
    angle_max_deg: float
    angle_step_deg: float
    r_search_max_m: float
    r_step_m: float
    theta_step_deg: float
    sign_mode: str
    check_table_clearance: bool
    workers: int


@dataclass(frozen=True)
class SurfacePoint:
    axis_count: int
    axis_combination: str
    angle_deg: float
    z_m: float
    max_inner_circle_r_m: float
    limiting_theta_deg: float
    center_pose_feasible: bool
    search_limit_reached: bool
    pose_checks: int


@dataclass(frozen=True)
class ComboLayerResult:
    axis_combination: str
    z_m: float
    points: Tuple[SurfacePoint, ...]
    target_cache_hits: int
    target_cache_misses: int


@dataclass(frozen=True)
class AggregatePoint:
    axis_count: int
    group_name: str
    angle_deg: float
    z_m: float
    max_inner_circle_r_m: float
    limiting_axis_combinations: str
    center_pose_feasible: bool
    search_limit_reached: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def positive_float(value: str) -> float:
    number = finite_float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative_float(value: str) -> float:
    number = finite_float(value)
    if number < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def inclusive_values(start: float, stop: float, step: float) -> np.ndarray:
    values = arange_inclusive(float(start), float(stop), float(step))
    if values.size == 0:
        raise ValueError(f"empty scan range: {start} .. {stop}, step={step}")
    return np.round(np.asarray(values, dtype=float), 10)


def theta_values(step_deg: float) -> np.ndarray:
    values = np.arange(0.0, 360.0, float(step_deg), dtype=float)
    if values.size == 0:
        raise ValueError("theta scan contains no rays")
    return np.round(values, 10)


def rotation_vector_matrix(rotation_vector_deg: Sequence[float]) -> np.ndarray:
    """Convert world-axis rotation-vector components in degrees to a matrix."""

    rotvec = np.radians(np.asarray(rotation_vector_deg, dtype=float).reshape(3))
    total_angle = float(np.linalg.norm(rotvec))
    if total_angle <= 1e-14:
        return np.eye(3, dtype=float)
    axis = rotvec / total_angle
    x, y, z = (float(value) for value in axis)
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )
    return (
        np.eye(3, dtype=float)
        + math.sin(total_angle) * skew
        + (1.0 - math.cos(total_angle)) * (skew @ skew)
    )


def sign_patterns(axis_count: int, sign_mode: str) -> Tuple[Tuple[float, ...], ...]:
    if sign_mode == "positive":
        return (tuple(1.0 for _ in range(int(axis_count))),)
    return tuple(itertools.product((-1.0, 1.0), repeat=int(axis_count)))


def orientation_matrices(
    axes: Sequence[int],
    angle_deg: float,
    sign_mode: str,
) -> Tuple[np.ndarray, ...]:
    """Return unique orientation matrices required by one signed angle group."""

    matrices: List[np.ndarray] = []
    seen: set[Tuple[int, ...]] = set()
    for signs in sign_patterns(len(tuple(axes)), sign_mode):
        vector = np.zeros(3, dtype=float)
        for axis, sign in zip(axes, signs):
            vector[int(axis)] = float(sign) * float(angle_deg)
        matrix = rotation_vector_matrix(vector)
        # At angle=0 all sign patterns collapse to identity.  Deduplicating is
        # exact at the workspace precision and avoids 2^n redundant IK calls.
        key = tuple(int(round(float(value) / 1e-12)) for value in matrix.reshape(-1))
        if key not in seen:
            seen.add(key)
            matrices.append(matrix)
    return tuple(matrices)


def _xml_vector(element: ET.Element, attribute: str, default: Sequence[float]) -> np.ndarray:
    text = element.get(attribute)
    if text is None:
        return np.asarray(default, dtype=float)
    values = np.fromstring(text, sep=" ", dtype=float)
    if values.size != len(tuple(default)):
        raise ValueError(
            f"{element.tag} {attribute!r} must contain {len(tuple(default))} values"
        )
    return values


def model_placements() -> Tuple[float, Tuple[float, float]]:
    """Read root z and table XY centre from the local Rotation_G6 XML."""

    project_root = ET.parse(DEFAULT_XML).getroot()
    include_names = {
        Path(include.get("file", "")).name.lower(): include.get("file", "")
        for include in project_root.iter("include")
        if include.get("file")
    }
    gripper_name = include_names.get("gripper.xml")
    if not gripper_name:
        raise ValueError(f"{DEFAULT_XML} does not include local Gripper.xml")
    gripper_path = (MODULE_DIR / gripper_name).resolve()
    try:
        gripper_path.relative_to(next(parent for parent in Path(__file__).resolve().parents if (parent / "models" / "cad").is_dir()))
    except ValueError as exc:
        raise ValueError("Gripper.xml include escapes Rotation_G6") from exc

    gripper_root = ET.parse(gripper_path).getroot()
    root_body = gripper_root.find(".//body[@name='root']")
    if root_body is None:
        raise ValueError(f"root body not found in {gripper_path}")
    root_pos = _xml_vector(root_body, "pos", (0.0, 0.0, 0.0))

    worldbody = project_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"worldbody not found in {DEFAULT_XML}")
    table_body = worldbody.find("body[@name='table']")
    if table_body is None:
        raise ValueError(f"table body not found in {DEFAULT_XML}")
    table_pos = _xml_vector(table_body, "pos", (0.0, 0.0, 0.0))
    table_geom = table_body.find("geom[@name='table_geom']")
    table_geom_pos = (
        _xml_vector(table_geom, "pos", (0.0, 0.0, 0.0))
        if table_geom is not None
        else np.zeros(3, dtype=float)
    )
    table_center = table_pos + table_geom_pos
    return float(root_pos[2]), (float(table_center[0]), float(table_center[1]))


def resolved_worker_count(requested: int, task_count: int) -> int:
    if int(task_count) <= 0:
        return 1
    if int(requested) > 0:
        return max(1, min(int(requested), int(task_count)))

    allocation_candidates: List[int] = []
    for variable in ("PBS_NCPUS", "SLURM_CPUS_PER_TASK"):
        raw = os.environ.get(variable, "").strip()
        if raw:
            try:
                allocation_candidates.append(int(raw))
            except ValueError:
                pass
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_count = int(os.cpu_count() or 1)
    allocation_candidates.append(max(1, affinity_count))
    # PBS Pro can expose NCPUS as ompthreads (often 1), rather than the select
    # allocation's CPU count, so it is deliberately not used here.
    available = min(allocation_candidates)
    return max(1, min(int(available), int(task_count)))


class WorkspaceScanner:
    """Strict-grid analytic IK scanner used independently in each process."""

    def __init__(self, cfg: ScanConfig):
        self.cfg = cfg
        xml_ik = GripperIK.from_xml(DEFAULT_XML)
        root_shift = np.array([0.0, 0.0, cfg.root_height_offset_m], dtype=float)
        xml_ik.set_armbase_world_poses(
            {
                finger: (
                    np.asarray(geometry.base_world, dtype=float) + root_shift,
                    np.asarray(geometry.rotation_world, dtype=float),
                )
                for finger, geometry in xml_ik.fingers.items()
            }
        )
        self.ik = xml_ik
        self.r_values = inclusive_values(0.0, cfg.r_search_max_m, cfg.r_step_m)
        self.theta_values_deg = theta_values(cfg.theta_step_deg)
        self.angle_values_deg = inclusive_values(
            cfg.angle_min_deg,
            cfg.angle_max_deg,
            cfg.angle_step_deg,
        )
        self.orientation_lookup: Dict[str, Dict[float, Tuple[np.ndarray, ...]]] = {
            label: {
                float(angle): orientation_matrices(axes, float(angle), cfg.sign_mode)
                for angle in self.angle_values_deg
            }
            for label, axes in COMBO_TO_AXES.items()
        }
        self.target_cache: Dict[Tuple[str, int, int, int], bool] = {}
        # Do not merge neighbouring millimetre samples or near-origin theta
        # rays.  This resolution is only for stable hashing of identical math.
        self.target_cache_resolution_m = 1e-10
        self.target_cache_hits = 0
        self.target_cache_misses = 0

    def reset_task_cache(self) -> None:
        self.target_cache.clear()
        self.target_cache_hits = 0
        self.target_cache_misses = 0

    def _target_key(self, finger: str, target: np.ndarray) -> Tuple[str, int, int, int]:
        xyz = np.asarray(target, dtype=float).reshape(3)
        resolution = self.target_cache_resolution_m
        return (
            str(finger),
            int(round(float(xyz[0]) / resolution)),
            int(round(float(xyz[1]) / resolution)),
            int(round(float(xyz[2]) / resolution)),
        )

    def _target_feasible(self, finger: str, target: np.ndarray) -> bool:
        key = self._target_key(finger, target)
        if key in self.target_cache:
            self.target_cache_hits += 1
            return bool(self.target_cache[key])
        feasible = bool(self.ik.finger_target_feasible(finger, target))
        self.target_cache[key] = feasible
        self.target_cache_misses += 1
        return feasible

    def _inside_table_xy(self, center: np.ndarray, rotation: np.ndarray) -> bool:
        # Orthogonal projection of a zero-thickness circular grasp plane.
        x_extent = self.cfg.disk_radius_m * math.hypot(
            float(rotation[0, 0]), float(rotation[0, 1])
        )
        y_extent = self.cfg.disk_radius_m * math.hypot(
            float(rotation[1, 0]), float(rotation[1, 1])
        )
        dx = abs(float(center[0]) - self.cfg.table_center_xy_m[0])
        dy = abs(float(center[1]) - self.cfg.table_center_xy_m[1])
        return bool(
            dx + x_extent <= self.cfg.table_half_x_m + 1e-10
            and dy + y_extent <= self.cfg.table_half_y_m + 1e-10
        )

    def _table_clear(
        self,
        center: np.ndarray,
        rotation: np.ndarray,
        targets: Mapping[str, np.ndarray],
    ) -> bool:
        if not self.cfg.check_table_clearance:
            return True
        if not self._inside_table_xy(center, rotation):
            return False
        disk_vertical_extent = self.cfg.disk_radius_m * math.hypot(
            float(rotation[2, 0]), float(rotation[2, 1])
        )
        if (
            float(center[2]) - disk_vertical_extent
            < self.cfg.table_top_z_m - 1e-10
        ):
            return False
        return all(
            float(target[2]) - self.cfg.hard_tip_radius_m
            >= self.cfg.table_top_z_m - 1e-10
            for target in targets.values()
        )

    def orientation_feasible(self, center: np.ndarray, rotation: np.ndarray) -> bool:
        targets = {
            finger: np.asarray(center, dtype=float)
            + np.asarray(rotation, dtype=float)
            @ (
                np.asarray(TIP_OFFSETS_UNIT[finger], dtype=float)
                * self.cfg.tip_center_distance_m
            )
            for finger in FINGERS
        }
        if not self._table_clear(center, rotation, targets):
            return False
        for finger in FINGERS:
            if not self._target_feasible(finger, targets[finger]):
                return False
        return True

    def orientation_group_feasible(
        self,
        center: np.ndarray,
        rotations: Sequence[np.ndarray],
    ) -> bool:
        if self.cfg.sign_mode == "any":
            return any(self.orientation_feasible(center, rotation) for rotation in rotations)
        return all(self.orientation_feasible(center, rotation) for rotation in rotations)

    def scan_pose(
        self,
        axis_combination: str,
        z_m: float,
        angle_deg: float,
    ) -> SurfacePoint:
        rotations = self.orientation_lookup[axis_combination][float(angle_deg)]
        center_origin = np.array(
            [self.cfg.center_xy_m[0], self.cfg.center_xy_m[1], float(z_m)],
            dtype=float,
        )
        pose_checks = 1
        if not self.orientation_group_feasible(center_origin, rotations):
            return SurfacePoint(
                axis_count=len(COMBO_TO_AXES[axis_combination]),
                axis_combination=axis_combination,
                angle_deg=float(angle_deg),
                z_m=float(z_m),
                max_inner_circle_r_m=0.0,
                limiting_theta_deg=0.0,
                center_pose_feasible=False,
                search_limit_reached=False,
                pose_checks=pose_checks,
            )

        current_limit_index = len(self.r_values) - 1
        limiting_theta_deg = 0.0
        first_ray = True

        for theta_deg in self.theta_values_deg:
            theta_rad = math.radians(float(theta_deg))
            cos_theta = math.cos(theta_rad)
            sin_theta = math.sin(theta_rad)
            failed = False
            last_feasible_index = 0

            # The first ray establishes an upper bound.  Later rays only need
            # to reach the current global minimum; reaching it means that ray
            # cannot reduce the result.  This remains an exact strict-grid
            # prefix scan and does not assume monotonicity beyond first failure.
            upper_index = len(self.r_values) - 1 if first_ray else current_limit_index
            for radius_index in range(1, upper_index + 1):
                radius = float(self.r_values[radius_index])
                center = np.array(
                    [
                        self.cfg.center_xy_m[0] + radius * cos_theta,
                        self.cfg.center_xy_m[1] + radius * sin_theta,
                        float(z_m),
                    ],
                    dtype=float,
                )
                pose_checks += 1
                if self.orientation_group_feasible(center, rotations):
                    last_feasible_index = radius_index
                else:
                    failed = True
                    break

            if failed and last_feasible_index < current_limit_index:
                current_limit_index = last_feasible_index
                limiting_theta_deg = float(theta_deg)
            first_ray = False
            if current_limit_index == 0:
                break

        return SurfacePoint(
            axis_count=len(COMBO_TO_AXES[axis_combination]),
            axis_combination=axis_combination,
            angle_deg=float(angle_deg),
            z_m=float(z_m),
            max_inner_circle_r_m=float(self.r_values[current_limit_index]),
            limiting_theta_deg=float(limiting_theta_deg),
            center_pose_feasible=True,
            search_limit_reached=bool(current_limit_index == len(self.r_values) - 1),
            pose_checks=int(pose_checks),
        )

    def scan_combo_layer(self, axis_combination: str, z_m: float) -> ComboLayerResult:
        if axis_combination not in COMBO_TO_AXES:
            raise ValueError(f"unsupported axis combination: {axis_combination}")
        self.reset_task_cache()
        points = tuple(
            self.scan_pose(axis_combination, float(z_m), float(angle))
            for angle in self.angle_values_deg
        )
        return ComboLayerResult(
            axis_combination=axis_combination,
            z_m=float(z_m),
            points=points,
            target_cache_hits=int(self.target_cache_hits),
            target_cache_misses=int(self.target_cache_misses),
        )


_WORKER_SCANNER: Optional[WorkspaceScanner] = None


def initialize_worker(cfg: ScanConfig) -> None:
    global _WORKER_SCANNER
    _WORKER_SCANNER = WorkspaceScanner(cfg)


def scan_task(task: Tuple[str, float]) -> ComboLayerResult:
    if _WORKER_SCANNER is None:
        raise RuntimeError("workspace worker was not initialized")
    axis_combination, z_m = task
    return _WORKER_SCANNER.scan_combo_layer(axis_combination, float(z_m))


def task_key(axis_combination: str, z_m: float) -> Tuple[str, float]:
    return str(axis_combination), round(float(z_m), 10)


def result_to_checkpoint_record(result: ComboLayerResult) -> Dict[str, object]:
    return {
        "axis_combination": result.axis_combination,
        "z_m": float(result.z_m),
        "target_cache_hits": int(result.target_cache_hits),
        "target_cache_misses": int(result.target_cache_misses),
        "points": [asdict(point) for point in result.points],
    }


def result_from_checkpoint_record(record: Mapping[str, object]) -> ComboLayerResult:
    raw_points = record.get("points")
    if not isinstance(raw_points, list):
        raise ValueError("checkpoint record has no points list")
    points = tuple(SurfacePoint(**dict(raw)) for raw in raw_points)
    return ComboLayerResult(
        axis_combination=str(record["axis_combination"]),
        z_m=float(record["z_m"]),
        points=points,
        target_cache_hits=int(record.get("target_cache_hits", 0)),
        target_cache_misses=int(record.get("target_cache_misses", 0)),
    )


def checkpoint_result_is_valid(
    result: ComboLayerResult,
    cfg: ScanConfig,
    expected_angles: Sequence[float],
    expected_task_keys: set[Tuple[str, float]],
) -> bool:
    """Validate a parsed task record before treating it as completed."""

    key = task_key(result.axis_combination, result.z_m)
    if key not in expected_task_keys:
        return False
    if result.axis_combination not in COMBO_TO_AXES:
        return False
    if len(result.points) != len(expected_angles):
        return False

    expected_axis_count = len(COMBO_TO_AXES[result.axis_combination])
    for point, expected_angle in zip(result.points, expected_angles):
        numeric_values = (
            point.angle_deg,
            point.z_m,
            point.max_inner_circle_r_m,
            point.limiting_theta_deg,
        )
        if not all(math.isfinite(float(value)) for value in numeric_values):
            return False
        if point.axis_count != expected_axis_count:
            return False
        if point.axis_combination != result.axis_combination:
            return False
        if not math.isclose(
            point.z_m,
            result.z_m,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            return False
        if not math.isclose(
            point.angle_deg,
            float(expected_angle),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            return False
        if not (
            -1e-12
            <= point.max_inner_circle_r_m
            <= cfg.r_search_max_m + 1e-12
        ):
            return False
        if point.pose_checks < 1:
            return False
        if point.search_limit_reached and not math.isclose(
            point.max_inner_circle_r_m,
            cfg.r_search_max_m,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            return False
    return True


def load_checkpoint(path: Path) -> Dict[Tuple[str, float], ComboLayerResult]:
    results: Dict[Tuple[str, float], ComboLayerResult] = {}
    if not path.is_file():
        return results
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
                result = result_from_checkpoint_record(record)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(
                    f"[warning] ignoring incomplete checkpoint line {line_number}: {exc}",
                    flush=True,
                )
                continue
            results[task_key(result.axis_combination, result.z_m)] = result
    return results


def repair_incomplete_checkpoint_tail(path: Path) -> bool:
    """Drop only a non-newline-terminated JSONL tail left by interruption.

    ``append_checkpoint`` flushes and fsyncs every completed task.  A hard
    walltime kill can nevertheless stop inside the final write.  Without this
    repair, the next append would be concatenated to that partial JSON object
    and both task records would be unreadable on the following resume.
    """

    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return False

        end = handle.tell()
        cursor = end
        newline_position = -1
        chunk_size = 64 * 1024
        while cursor > 0:
            start = max(0, cursor - chunk_size)
            handle.seek(start)
            chunk = handle.read(cursor - start)
            index = chunk.rfind(b"\n")
            if index >= 0:
                newline_position = start + index
                break
            cursor = start

        new_size = newline_position + 1 if newline_position >= 0 else 0
        handle.truncate(new_size)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        f"[warning] removed an incomplete checkpoint tail from {path}",
        flush=True,
    )
    return True


def append_checkpoint(path: Path, result: ComboLayerResult) -> None:
    record = result_to_checkpoint_record(result)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def computation_signature(cfg: ScanConfig) -> Dict[str, object]:
    source_files = (
        Path(__file__).resolve(),
        MODULE_DIR / "R6_kinematics.py",
        MODULE_DIR / "Project.xml",
        MODULE_DIR / "Gripper.xml",
        MODULE_DIR / "Share.xml",
        MODULE_DIR / "R6_workspace_requirements.txt",
    )
    return {
        "format_version": FORMAT_VERSION,
        "program": "R6_rotation_translation_workspace",
        "source_sha256": {
            path.name: sha256_file(path)
            for path in source_files
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "model_xml": "Project.xml",
        "root_height_m": cfg.root_height_m,
        "xml_root_height_m": cfg.xml_root_height_m,
        "root_height_offset_m": cfg.root_height_offset_m,
        "center_xy_m": list(cfg.center_xy_m),
        "disk_diameter_m": cfg.disk_diameter_m,
        "hard_tip_radius_m": cfg.hard_tip_radius_m,
        "tip_squeeze_margin_m": cfg.tip_squeeze_margin_m,
        "tip_center_distance_m": cfg.tip_center_distance_m,
        "z_min_m": cfg.z_min_m,
        "z_max_m": cfg.z_max_m,
        "z_step_m": cfg.z_step_m,
        "angle_min_deg": cfg.angle_min_deg,
        "angle_max_deg": cfg.angle_max_deg,
        "angle_step_deg": cfg.angle_step_deg,
        "r_search_max_m": cfg.r_search_max_m,
        "r_step_m": cfg.r_step_m,
        "theta_step_deg": cfg.theta_step_deg,
        "theta_count": int(theta_values(cfg.theta_step_deg).size),
        "rotation_convention": "world_axis_rotation_vector_components",
        "sign_mode": cfg.sign_mode,
        "check_table_clearance": cfg.check_table_clearance,
        "object_geometry": "zero_thickness_circular_grasp_plane",
    }


def write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def scan_workspace(
    cfg: ScanConfig,
    *,
    resume: bool,
    force: bool,
) -> Tuple[List[SurfacePoint], List[ComboLayerResult], float, int]:
    output_dir = Path(cfg.output_dir)
    checkpoint_path = output_dir / "R6_workspace_checkpoint.jsonl"
    metadata_path = output_dir / "R6_workspace_run_metadata.json"
    signature = computation_signature(cfg)

    if force:
        for path in (checkpoint_path, metadata_path):
            if path.exists():
                path.unlink()

    prior_metadata: Dict[str, object] = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            prior_metadata = json.load(handle)
        if prior_metadata.get("signature") != signature:
            raise RuntimeError(
                "existing output checkpoint has a different computation signature; "
                "use another --output-dir or pass --force intentionally"
            )
    elif checkpoint_path.exists():
        raise RuntimeError(
            "checkpoint exists but its signature metadata is missing; refusing "
            "to mix unknown results. Use another --output-dir or pass --force "
            "intentionally"
        )

    if checkpoint_path.exists() and not resume:
        raise RuntimeError(
            "checkpoint already exists and --no-resume was requested; "
            "use another --output-dir or pass --force"
        )

    if resume:
        repair_incomplete_checkpoint_tail(checkpoint_path)
    completed = load_checkpoint(checkpoint_path) if resume else {}
    z_values = inclusive_values(cfg.z_min_m, cfg.z_max_m, cfg.z_step_m)
    angle_values = inclusive_values(
        cfg.angle_min_deg,
        cfg.angle_max_deg,
        cfg.angle_step_deg,
    )
    # Start expensive/high-sign-count combinations first to reduce the tail at
    # the end of a many-core CX allocation.
    combo_order = ("XYZ", "XY", "XZ", "YZ", "X", "Y", "Z")
    all_tasks = [
        (combo, float(z_m))
        for combo in combo_order
        for z_m in z_values
    ]
    expected_task_keys = {task_key(*task) for task in all_tasks}
    invalid_checkpoint_keys = [
        key
        for key, result in completed.items()
        if not checkpoint_result_is_valid(
            result,
            cfg,
            angle_values,
            expected_task_keys,
        )
    ]
    for key in invalid_checkpoint_keys:
        completed.pop(key, None)
    if invalid_checkpoint_keys:
        print(
            f"[warning] {len(invalid_checkpoint_keys)} structurally invalid "
            "checkpoint task(s) will be recomputed",
            flush=True,
        )
    missing_tasks = [
        task
        for task in all_tasks
        if task_key(*task) not in completed
    ]
    workers = resolved_worker_count(cfg.workers, len(missing_tasks))

    print("R6 combined rotation/translation analytic workspace", flush=True)
    print(f"  local model: {DEFAULT_XML}", flush=True)
    print(
        f"  root z: XML {cfg.xml_root_height_m:.3f} m -> workspace "
        f"{cfg.root_height_m:.3f} m (offset {cfg.root_height_offset_m:+.3f} m)",
        flush=True,
    )
    print(
        f"  z={cfg.z_min_m:.3f}..{cfg.z_max_m:.3f} step={cfg.z_step_m:.3f} m; "
        f"angle={cfg.angle_min_deg:g}..{cfg.angle_max_deg:g} "
        f"step={cfg.angle_step_deg:g} deg",
        flush=True,
    )
    print(
        f"  r=0..{cfg.r_search_max_m:.3f} step={cfg.r_step_m:.3f} m; "
        f"theta step={cfg.theta_step_deg:g} deg "
        f"({len(theta_values(cfg.theta_step_deg))} rays)",
        flush=True,
    )
    print(
        f"  signs={cfg.sign_mode}; rotation=world-axis rotation-vector components; "
        f"table clearance={cfg.check_table_clearance}",
        flush=True,
    )
    print(
        f"  tasks={len(all_tasks)}; resumed={len(completed)}; "
        f"remaining={len(missing_tasks)}; workers={workers}",
        flush=True,
    )

    started = time.monotonic()
    started_utc = str(prior_metadata.get("started_utc") or utc_now())
    run_metadata: Dict[str, object] = {
        "status": "running",
        "started_utc": started_utc,
        "updated_utc": utc_now(),
        "signature": signature,
        "task_count": len(all_tasks),
        "completed_task_count": len(completed),
    }
    write_json_atomic(metadata_path, run_metadata)

    def accept_result(result: ComboLayerResult, newly_done: int) -> None:
        key = task_key(result.axis_combination, result.z_m)
        completed[key] = result
        append_checkpoint(checkpoint_path, result)
        total_done = len(completed)
        if newly_done == 1 or total_done == len(all_tasks) or newly_done % 10 == 0:
            elapsed = time.monotonic() - started
            rate = newly_done / elapsed if elapsed > 0.0 else 0.0
            remaining = len(missing_tasks) - newly_done
            eta_s = remaining / rate if rate > 0.0 else float("inf")
            eta_text = f"{eta_s / 3600.0:.2f} h" if math.isfinite(eta_s) else "unknown"
            print(
                f"  [{total_done:4d}/{len(all_tasks)}] "
                f"{result.axis_combination} z={result.z_m:.3f} m; "
                f"elapsed={elapsed / 3600.0:.2f} h; ETA={eta_text}",
                flush=True,
            )
            run_metadata.update(
                {
                    "updated_utc": utc_now(),
                    "completed_task_count": total_done,
                }
            )
            write_json_atomic(metadata_path, run_metadata)

    if missing_tasks:
        if workers == 1:
            scanner = WorkspaceScanner(cfg)
            for newly_done, task in enumerate(missing_tasks, start=1):
                accept_result(scanner.scan_combo_layer(*task), newly_done)
        else:
            executor: Optional[ProcessPoolExecutor] = None
            try:
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=initialize_worker,
                    initargs=(cfg,),
                )
                futures = {
                    executor.submit(scan_task, task): task
                    for task in missing_tasks
                }
                for newly_done, future in enumerate(as_completed(futures), start=1):
                    accept_result(future.result(), newly_done)
            except (OSError, PermissionError, BrokenProcessPool) as exc:
                print(
                    f"[warning] multiprocessing failed ({exc}); "
                    "finishing uncheckpointed tasks serially",
                    flush=True,
                )
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor = None
                serial_missing = [
                    task
                    for task in all_tasks
                    if task_key(*task) not in completed
                ]
                scanner = WorkspaceScanner(cfg)
                already_new = len(missing_tasks) - len(serial_missing)
                for offset, task in enumerate(serial_missing, start=1):
                    accept_result(
                        scanner.scan_combo_layer(*task),
                        already_new + offset,
                    )
            except BaseException:
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor = None
                raise
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)

    absent = [
        task
        for task in all_tasks
        if task_key(*task) not in completed
    ]
    if absent:
        raise RuntimeError(f"scan finished with {len(absent)} missing tasks")

    ordered_results = [
        completed[task_key(*task)]
        for task in all_tasks
    ]
    points = [
        point
        for result in ordered_results
        for point in result.points
    ]
    points.sort(
        key=lambda row: (
            row.axis_count,
            row.axis_combination,
            row.z_m,
            row.angle_deg,
        )
    )
    elapsed_s = time.monotonic() - started
    resumed_count = len(all_tasks) - len(missing_tasks)
    return points, ordered_results, float(elapsed_s), int(resumed_count)


def aggregate_points(points: Sequence[SurfacePoint]) -> List[AggregatePoint]:
    lookup: Dict[Tuple[str, float, float], SurfacePoint] = {
        (row.axis_combination, round(row.z_m, 10), round(row.angle_deg, 10)): row
        for row in points
    }
    z_values = sorted({round(row.z_m, 10) for row in points})
    angle_values = sorted({round(row.angle_deg, 10) for row in points})
    aggregated: List[AggregatePoint] = []

    for axis_count in (1, 2, 3):
        combinations = GROUP_COMBOS[axis_count]
        for z_m in z_values:
            for angle_deg in angle_values:
                rows = [
                    lookup[(combo, z_m, angle_deg)]
                    for combo in combinations
                ]
                minimum = min(row.max_inner_circle_r_m for row in rows)
                limiting = "+".join(
                    row.axis_combination
                    for row in rows
                    if math.isclose(
                        row.max_inner_circle_r_m,
                        minimum,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )
                # The aggregate is capped only when the reported minimum itself
                # is the search ceiling for every tied limiting combination.
                limiting_rows = [
                    row
                    for row in rows
                    if math.isclose(
                        row.max_inner_circle_r_m,
                        minimum,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ]
                aggregated.append(
                    AggregatePoint(
                        axis_count=axis_count,
                        group_name=GROUP_NAMES[axis_count],
                        angle_deg=float(angle_deg),
                        z_m=float(z_m),
                        max_inner_circle_r_m=float(minimum),
                        limiting_axis_combinations=limiting,
                        center_pose_feasible=all(
                            row.center_pose_feasible for row in rows
                        ),
                        search_limit_reached=bool(
                            limiting_rows
                            and all(row.search_limit_reached for row in limiting_rows)
                        ),
                    )
                )
    return aggregated


def save_combo_csv(points: Sequence[SurfacePoint], output_dir: Path) -> Path:
    path = output_dir / "R6_workspace_combo_details.csv"
    fields = [
        "axis_count",
        "axis_combination",
        "angle_deg",
        "z_m",
        "max_inner_circle_r_m",
        "limiting_theta_deg",
        "center_pose_feasible",
        "search_limit_reached",
        "pose_checks",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in points:
            writer.writerow(
                {
                    "axis_count": row.axis_count,
                    "axis_combination": row.axis_combination,
                    "angle_deg": f"{row.angle_deg:.6f}",
                    "z_m": f"{row.z_m:.6f}",
                    "max_inner_circle_r_m": f"{row.max_inner_circle_r_m:.6f}",
                    "limiting_theta_deg": f"{row.limiting_theta_deg:.6f}",
                    "center_pose_feasible": int(row.center_pose_feasible),
                    "search_limit_reached": int(row.search_limit_reached),
                    "pose_checks": row.pose_checks,
                }
            )
    return path


def save_aggregate_csv(
    points: Sequence[AggregatePoint],
    output_dir: Path,
) -> Path:
    path = output_dir / "R6_workspace_aggregated.csv"
    fields = [
        "axis_count",
        "group_name",
        "angle_deg",
        "z_m",
        "max_inner_circle_r_m",
        "limiting_axis_combinations",
        "center_pose_feasible",
        "search_limit_reached",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in points:
            writer.writerow(
                {
                    "axis_count": row.axis_count,
                    "group_name": row.group_name,
                    "angle_deg": f"{row.angle_deg:.6f}",
                    "z_m": f"{row.z_m:.6f}",
                    "max_inner_circle_r_m": f"{row.max_inner_circle_r_m:.6f}",
                    "limiting_axis_combinations": row.limiting_axis_combinations,
                    "center_pose_feasible": int(row.center_pose_feasible),
                    "search_limit_reached": int(row.search_limit_reached),
                }
            )
    return path


def aggregate_surface_arrays(
    points: Sequence[AggregatePoint],
    axis_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in points if row.axis_count == int(axis_count)]
    if not selected:
        raise ValueError(f"no aggregate points for axis_count={axis_count}")
    angles = np.asarray(sorted({row.angle_deg for row in selected}), dtype=float)
    z_values = np.asarray(sorted({row.z_m for row in selected}), dtype=float)
    lookup = {
        (round(row.z_m, 10), round(row.angle_deg, 10)): row.max_inner_circle_r_m
        for row in selected
    }
    radius_grid = np.asarray(
        [
            [
                lookup[(round(float(z_m), 10), round(float(angle), 10))]
                for angle in angles
            ]
            for z_m in z_values
        ],
        dtype=float,
    )
    return angles, z_values, radius_grid




def save_summary(
    cfg: ScanConfig,
    points: Sequence[SurfacePoint],
    aggregate: Sequence[AggregatePoint],
    task_results: Sequence[ComboLayerResult],
    elapsed_s: float,
    resumed_tasks: int,
    outputs: Sequence[Path],
    output_dir: Path,
) -> Path:
    path = output_dir / "R6_workspace_summary.txt"
    capped_aggregate = [row for row in aggregate if row.search_limit_reached]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("R6 combined rotation/translation analytic workspace\n")
        handle.write(f"completed_utc: {utc_now()}\n")
        handle.write(f"project_xml: {DEFAULT_XML}\n")
        handle.write("external_project_code_dependencies: none\n")
        handle.write("environment_location: Rotation_G6/CX_env on Imperial CX\n")
        handle.write(f"xml_root_height_m: {cfg.xml_root_height_m:.6f}\n")
        handle.write(f"workspace_root_height_m: {cfg.root_height_m:.6f}\n")
        handle.write(f"root_height_offset_m: {cfg.root_height_offset_m:+.6f}\n")
        handle.write(f"workspace_center_xy_m: {cfg.center_xy_m}\n")
        handle.write(f"disk_diameter_m: {cfg.disk_diameter_m:.6f}\n")
        handle.write(f"hard_tip_radius_m: {cfg.hard_tip_radius_m:.6f}\n")
        handle.write(f"tip_squeeze_margin_m: {cfg.tip_squeeze_margin_m:.6f}\n")
        handle.write(f"tip_center_distance_m: {cfg.tip_center_distance_m:.6f}\n")
        handle.write(
            f"z_scan_m: {cfg.z_min_m:.6f} .. {cfg.z_max_m:.6f}, "
            f"step {cfg.z_step_m:.6f}\n"
        )
        handle.write(
            f"angle_scan_deg: {cfg.angle_min_deg:.6f} .. "
            f"{cfg.angle_max_deg:.6f}, step {cfg.angle_step_deg:.6f}\n"
        )
        handle.write(
            "rotation_convention: equal world-axis rotation-vector components\n"
        )
        handle.write(f"sign_mode: {cfg.sign_mode}\n")
        handle.write(
            f"translation_r_scan_m: 0 .. {cfg.r_search_max_m:.6f}, "
            f"step {cfg.r_step_m:.6f}\n"
        )
        handle.write(
            f"translation_theta_scan_deg: 0 .. <360, "
            f"step {cfg.theta_step_deg:.6f}, "
            f"count {len(theta_values(cfg.theta_step_deg))}\n"
        )
        handle.write(
            "inner_circle_definition: minimum first-failure contiguous radial "
            "prefix over theta\n"
        )
        handle.write(
            "single_surface: pointwise min(X,Y,Z)\n"
            "double_surface: pointwise min(XY,XZ,YZ)\n"
            "triple_surface: XYZ\n"
        )
        handle.write(
            f"table_disk_tip_clearance_checked: {cfg.check_table_clearance}\n"
        )
        handle.write("active_and_passive_joint_limits_checked: true\n")
        handle.write("mujoco_link_collision_or_dynamics_checked: false\n")
        handle.write(f"task_count: {len(task_results)}\n")
        handle.write(f"resumed_task_count: {resumed_tasks}\n")
        handle.write(
            f"pose_checks: {sum(row.pose_checks for row in points)}\n"
        )
        handle.write(
            f"target_cache_hits: "
            f"{sum(result.target_cache_hits for result in task_results)}\n"
        )
        handle.write(
            f"target_cache_misses: "
            f"{sum(result.target_cache_misses for result in task_results)}\n"
        )
        handle.write(f"scan_elapsed_this_invocation_s: {elapsed_s:.3f}\n")
        handle.write(
            f"aggregate_search_limit_reached_samples: "
            f"{len(capped_aggregate)}\n"
        )
        for axis_count in (1, 2, 3):
            values = [
                row.max_inner_circle_r_m
                for row in aggregate
                if row.axis_count == axis_count
            ]
            handle.write(
                f"{GROUP_NAMES[axis_count]}_surface_max_r_m: "
                f"{max(values, default=0.0):.6f}\n"
            )
            handle.write(
                f"{GROUP_NAMES[axis_count]}_surface_nonzero_samples: "
                f"{sum(value > 0.0 for value in values)}/{len(values)}\n"
            )
        for output in outputs:
            handle.write(f"output: {output}\n")
    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan R3 single/double/triple rotation versus z and maximum "
            "centred XY inner-circle translation radius."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root-height", type=finite_float, default=DEFAULT_ROOT_HEIGHT_M)
    parser.add_argument("--z-min", type=finite_float, default=DEFAULT_Z_MIN_M)
    parser.add_argument("--z-max", type=finite_float, default=DEFAULT_Z_MAX_M)
    parser.add_argument("--z-step", type=positive_float, default=DEFAULT_Z_STEP_M)
    parser.add_argument(
        "--angle-min-deg",
        type=nonnegative_float,
        default=DEFAULT_ANGLE_MIN_DEG,
    )
    parser.add_argument(
        "--angle-max-deg",
        type=nonnegative_float,
        default=DEFAULT_ANGLE_MAX_DEG,
    )
    parser.add_argument(
        "--angle-step-deg",
        type=positive_float,
        default=DEFAULT_ANGLE_STEP_DEG,
    )
    parser.add_argument(
        "--r-search-max",
        type=positive_float,
        default=DEFAULT_R_SEARCH_MAX_M,
    )
    parser.add_argument("--r-step", type=positive_float, default=DEFAULT_R_STEP_M)
    parser.add_argument(
        "--theta-step-deg",
        type=positive_float,
        default=DEFAULT_THETA_STEP_DEG,
    )
    parser.add_argument(
        "--disk-diameter",
        type=positive_float,
        default=DEFAULT_DISK_DIAMETER_M,
    )
    parser.add_argument(
        "--tip-squeeze-margin",
        type=nonnegative_float,
        default=0.0,
        help="subtract this distance from disk radius + hard-tip radius",
    )
    parser.add_argument(
        "--sign-mode",
        choices=("all", "any", "positive"),
        default="all",
        help=(
            "all=every +/- component pattern must work; "
            "any=one pattern may work; positive=only positive components"
        ),
    )
    parser.add_argument(
        "--ignore-table-clearance",
        action="store_true",
        help="evaluate analytic IK only, without disk/hard-tip table checks",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0=use the detected PBS/Slurm allocation or local CPU count",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="relative paths are resolved inside Rotation_G6",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume matching combo/z tasks from the output checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard only this output directory's known checkpoint metadata",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate local model/configuration and exit without scanning",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ScanConfig:
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if args.z_max < args.z_min - 1e-12:
        raise ValueError("--z-max must be >= --z-min")
    if args.angle_max_deg < args.angle_min_deg - 1e-12:
        raise ValueError("--angle-max-deg must be >= --angle-min-deg")
    if args.theta_step_deg > 360.0:
        raise ValueError("--theta-step-deg must be <= 360")
    if args.r_step > args.r_search_max + 1e-12:
        raise ValueError("--r-step must not exceed --r-search-max")

    xml_ik = GripperIK.from_xml(DEFAULT_XML)
    xml_root_height_m, table_center_xy_m = model_placements()
    root_height_m = float(args.root_height)
    root_height_offset_m = root_height_m - xml_root_height_m
    center_xy = np.mean(
        np.vstack(
            [
                np.asarray(xml_ik.fingers[finger].base_world[:2], dtype=float)
                for finger in FINGERS
            ]
        ),
        axis=0,
    )
    disk_diameter_m = float(args.disk_diameter)
    disk_radius_m = 0.5 * disk_diameter_m
    tip_center_distance_m = (
        disk_radius_m
        + float(xml_ik.model_info.hard_tip_radius)
        - float(args.tip_squeeze_margin)
    )
    if tip_center_distance_m <= 0.0:
        raise ValueError(
            "--tip-squeeze-margin is too large for the disk/hard-tip geometry"
        )

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = MODULE_DIR / output_dir
    output_dir = output_dir.resolve()

    cfg = ScanConfig(
        output_dir=str(output_dir),
        root_height_m=root_height_m,
        xml_root_height_m=float(xml_root_height_m),
        root_height_offset_m=float(root_height_offset_m),
        center_xy_m=(float(center_xy[0]), float(center_xy[1])),
        table_center_xy_m=table_center_xy_m,
        table_top_z_m=float(xml_ik.model_info.table_top_z),
        table_half_x_m=float(xml_ik.model_info.table_half_size[0]),
        table_half_y_m=float(xml_ik.model_info.table_half_size[1]),
        disk_diameter_m=disk_diameter_m,
        disk_radius_m=disk_radius_m,
        hard_tip_radius_m=float(xml_ik.model_info.hard_tip_radius),
        tip_squeeze_margin_m=float(args.tip_squeeze_margin),
        tip_center_distance_m=float(tip_center_distance_m),
        z_min_m=float(args.z_min),
        z_max_m=float(args.z_max),
        z_step_m=float(args.z_step),
        angle_min_deg=float(args.angle_min_deg),
        angle_max_deg=float(args.angle_max_deg),
        angle_step_deg=float(args.angle_step_deg),
        r_search_max_m=float(args.r_search_max),
        r_step_m=float(args.r_step),
        theta_step_deg=float(args.theta_step_deg),
        sign_mode=str(args.sign_mode),
        check_table_clearance=not bool(args.ignore_table_clearance),
        workers=int(args.workers),
    )

    # Materialize all grids now so invalid ranges fail on the login node or in
    # the PBS preflight, before allocating a long scan.
    inclusive_values(cfg.z_min_m, cfg.z_max_m, cfg.z_step_m)
    inclusive_values(cfg.angle_min_deg, cfg.angle_max_deg, cfg.angle_step_deg)
    inclusive_values(0.0, cfg.r_search_max_m, cfg.r_step_m)
    theta_values(cfg.theta_step_deg)
    return cfg


def print_validation(cfg: ScanConfig) -> None:
    print("R6 workspace configuration is valid")
    print(f"  program directory: {MODULE_DIR}")
    print(f"  Python prefix: {sys.prefix}")
    print(f"  model: {DEFAULT_XML}")
    print(
        f"  root: {cfg.xml_root_height_m:.6f} -> "
        f"{cfg.root_height_m:.6f} m"
    )
    print(
        f"  samples: z={len(inclusive_values(cfg.z_min_m, cfg.z_max_m, cfg.z_step_m))}, "
        f"angle={len(inclusive_values(cfg.angle_min_deg, cfg.angle_max_deg, cfg.angle_step_deg))}, "
        f"r={len(inclusive_values(0.0, cfg.r_search_max_m, cfg.r_step_m))}, "
        f"theta={len(theta_values(cfg.theta_step_deg))}"
    )
    print(f"  output: {cfg.output_dir}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    if args.validate_only:
        print_validation(cfg)
        return

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prevent two PBS jobs from appending to the same checkpoint concurrently.
    import fcntl

    lock_path = output_dir / ".R6_workspace.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another process is already using output directory {output_dir}"
            ) from exc

        points, task_results, elapsed_s, resumed_tasks = scan_workspace(
            cfg,
            resume=bool(args.resume),
            force=bool(args.force),
        )
        aggregate = aggregate_points(points)
        print("Scan complete; writing numerical workspace results...", flush=True)
        outputs: List[Path] = [
            save_combo_csv(points, output_dir),
            save_aggregate_csv(aggregate, output_dir),
        ]

        summary = save_summary(
            cfg,
            points,
            aggregate,
            task_results,
            elapsed_s,
            resumed_tasks,
            outputs,
            output_dir,
        )
        outputs.append(summary)

        metadata_path = output_dir / "R6_workspace_run_metadata.json"
        metadata = {
            "status": "complete",
            "completed_utc": utc_now(),
            "signature": computation_signature(cfg),
            "task_count": len(task_results),
            "resumed_task_count": resumed_tasks,
            "scan_elapsed_this_invocation_s": elapsed_s,
            "aggregate_search_limit_reached_samples": sum(
                row.search_limit_reached for row in aggregate
            ),
            "outputs": [str(path) for path in outputs],
        }
        write_json_atomic(metadata_path, metadata)

    print(f"Done. Outputs: {output_dir}")
    for path in outputs:
        print(f"  {path}")

    capped = [row for row in aggregate if row.search_limit_reached]
    if capped:
        examples = ", ".join(
            f"{row.group_name}:z={row.z_m:.3f},angle={row.angle_deg:g}"
            for row in capped[:8]
        )
        suffix = "..." if len(capped) > 8 else ""
        raise SystemExit(
            "r-search-max truncated one or more reported aggregate surfaces; "
            "increase --r-search-max and rerun with a new output directory "
            f"or --force. Samples: {examples}{suffix}"
        )


if __name__ == "__main__":
    main()
