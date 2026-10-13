#!/usr/bin/env python3
"""Scan the T6 four-finger analytic translation workspace for a flat disk grasp.

The program is self-contained inside its ``TRAnslation`` folder.  It reads the
current local ``Project.xml`` through the colocated :mod:`T6_kinematics`, runs
with the colocated ``RL_env`` virtual environment, and rejects project modules,
model files, output paths, or cache paths that escape this folder.

Grasp model
-----------
The disk is always parallel to the world xy plane.  Its center is the workspace
point ``(x, y, z)`` and the four hard-tip sphere centers are coplanar, spaced by
90 degrees.  By default the target cross is aligned with the gripper's U/D/L/R
axes.  Since a spherical hard tip touches the outside of the disk, its center is
at ``disk_radius + hard_tip_radius`` from the disk center.

For every z layer and theta ray, feasibility is checked from r=0 outwards on a
strict radial grid.  The first failed radius closes that ray's contiguous
feasible prefix.  The maximum centered inscribed-circle radius is the minimum
prefix radius over all theta rays.  This definition deliberately excludes any
disconnected feasible islands farther out on a ray.

Default world-coordinate scan
-----------------------------
* workspace-only root height: 0.600 m; the XML root pose is never modified
* z:     absolute world z 0.285 .. 0.500 m inclusive, step 0.001 m
* theta: 0 .. 360 deg inclusive, step 1 deg (360 duplicates 0)
* r:     strict 0.001 m grid
* disk diameter: 0.050 m (change ``DISK_DIAMETER_M`` or --disk-diameter)

Run from any directory with either of these relocatable commands::

    python /path/to/TRAnslation/T6_workspace.py
    /path/to/TRAnslation/RL_env/bin/python /path/to/TRAnslation/T6_workspace.py

Outputs are written to ``T6_workspace_outputs`` beside this script unless
``--output-dir`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


MODULE_DIR = Path(__file__).resolve().parent
LOCAL_ENV_DIR = (Path(os.environ["IRP_PYTHON_ENV"]) if os.environ.get("IRP_PYTHON_ENV") else next((parent / "RL_env" for parent in Path(__file__).resolve().parents if (parent / "RL_env" / "build.sh").is_file()), Path(sys.prefix)))
DEFAULT_XML = MODULE_DIR / "Project.xml"
LOCAL_MODEL_FILES = (DEFAULT_XML, MODULE_DIR / "Gripper.xml", MODULE_DIR / "Share.xml")

def _path_is_within(path: Path, root: Path) -> bool:
    resolved = path.expanduser().resolve()
    return resolved == root.resolve() or root.resolve() in resolved.parents

def _require_local_runtime(context: str = "T6_workspace") -> None:
    """The shared environment is selected by RL_env/run.sh."""

import numpy as np
import T6_kinematics as t6_kinematics

KINEMATICS_PATH = Path(t6_kinematics.__file__).resolve()
if KINEMATICS_PATH.parent != MODULE_DIR:
    raise ImportError(
        "T6_workspace 只能导入同目录的 T6_kinematics.py；"
        f"实际导入：{KINEMATICS_PATH}"
    )

FINGERS = t6_kinematics.FINGERS
GripperIK = t6_kinematics.GripperIK
arange_inclusive = t6_kinematics.arange_inclusive


# ---------------------------------------------------------------------------
# User-facing defaults.  The disk diameter can be changed here or on the CLI.
# ---------------------------------------------------------------------------

DISK_DIAMETER_M = 0.050
ROOT_HEIGHT_M = 0.600
# These are absolute world-frame sampling limits.  The XML root pose affects
# the IK geometry through a workspace-only query transform, but does not
# translate this requested observation window or modify any model file.
Z_MIN_M = 0.285
Z_MAX_M = 0.500
Z_STEP_M = 0.001
THETA_STEP_DEG = 1.0
R_STEP_M = 0.001
R_SEARCH_MAX_M = 0.300
VOXEL_SIZE_M = 0.001
CONTACT_YAW_DEG = 0.0
MAX_VOXEL_SURFACE_POINTS = 450_000
RANDOM_SEED = 7
DEFAULT_OUTPUT_DIR = MODULE_DIR / "T6_workspace_outputs"


def _validate_local_model_files() -> None:
    for model_file in LOCAL_MODEL_FILES:
        if not model_file.is_file():
            raise FileNotFoundError(f"缺少本地模型文件：{model_file}")
        resolved = model_file.resolve()
        if resolved.parent != MODULE_DIR:
            raise RuntimeError(
                "模型文件必须真实位于 T6_workspace.py 所在目录，不能链接到外部："
                f"{model_file} -> {resolved}"
            )


def _resolve_local_output_dir(value: str | os.PathLike[str]) -> Path:
    requested = Path(value).expanduser() if os.fspath(value) else DEFAULT_OUTPUT_DIR
    candidate = requested if requested.is_absolute() else MODULE_DIR / requested
    resolved = candidate.resolve()
    if not _path_is_within(resolved, MODULE_DIR):
        raise ValueError(
            "输出目录必须位于 T6_workspace.py 所在的 TRAnslation 文件夹内："
            f"{resolved}"
        )
    return resolved


def _output_path(cfg: "ScanConfig", filename: str) -> Path:
    return _resolve_local_output_dir(cfg.output_dir) / filename


@dataclass(frozen=True)
class ScanConfig:
    output_dir: Path
    disk_diameter: float
    hard_tip_radius: float
    tip_center_distance: float
    contact_yaw_deg: float
    root_x: float
    root_y: float
    xml_root_z: float
    root_height: float
    root_height_offset: float
    z_min: float
    z_max: float
    z_step: float
    theta_step_deg: float
    r_step: float
    r_search_max: float
    voxel_size: float
    use_fourfold_symmetry: bool
    workers: int


@dataclass(frozen=True)
class LayerResult:
    z: float
    max_inner_radius: float
    limiting_theta_deg: float
    center_feasible: bool
    search_limit_reached: bool
    pose_checks: int


@dataclass(frozen=True)
class WorkspaceResult:
    layers: Tuple[LayerResult, ...]
    elapsed_s: float

    @property
    def z_values(self) -> np.ndarray:
        return np.asarray([layer.z for layer in self.layers], dtype=float)

    @property
    def radii(self) -> np.ndarray:
        return np.asarray([layer.max_inner_radius for layer in self.layers], dtype=float)


@dataclass(frozen=True)
class PlaneBoundaryResult:
    z: float
    theta_degrees: np.ndarray
    max_radii: np.ndarray
    pose_checks: int
    elapsed_s: float


def _quantized_key(xyz: Sequence[float], resolution: float) -> Tuple[int, int, int]:
    return tuple(int(round(float(value) / resolution)) for value in xyz)  # type: ignore[return-value]


def _requested_theta_degrees(step_deg: float) -> np.ndarray:
    """Return the literal inclusive 0..360 degree scan grid."""
    return arange_inclusive(0.0, 360.0, float(step_deg))


def _independent_theta_degrees(cfg: ScanConfig) -> np.ndarray:
    """Remove the 360=0 duplicate and exact fourfold-equivalent directions."""
    requested = _requested_theta_degrees(cfg.theta_step_deg)
    period = 90.0 if cfg.use_fourfold_symmetry else 360.0
    representatives: Dict[int, float] = {}
    key_resolution = 1e-9
    for theta in requested:
        representative = float(theta) % period
        if math.isclose(representative, period, abs_tol=key_resolution):
            representative = 0.0
        key = int(round(representative / key_resolution))
        representatives.setdefault(key, representative)

    # Axis-aligned rays usually set the limiting radius for this symmetric
    # gripper.  Evaluating them first tightens the early-exit bound without
    # changing a single sampled feasibility decision.
    return np.asarray(sorted(representatives.values(), key=lambda value: (abs(value), value)), dtype=float)


def _fourfold_geometry_is_valid(ik: GripperIK, tolerance: float = 2e-8) -> bool:
    """Verify the XML geometry/limits needed for exact 90-degree replication."""
    geometries = ik.geometry_model.fingers
    reference = geometries["R"]
    numeric_fields = (
        "p_AH0",
        "R_AH0",
        "armbase_axis_A",
        "linear_axis_H",
        "plane_normal_H",
        "eX_H",
        "eY_H",
        "O_H",
        "M_H",
        "A_H",
        "B_H",
        "C_front_H",
        "C_long_H",
        "C0_H",
        "D0_H",
        "O2",
        "M2",
        "A2",
        "B2",
        "C_front2",
        "C_long2",
        "C02",
        "D02",
    )
    scalar_fields = (
        "plane_offset",
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
        "sigma_back",
        "sigma_front",
        "sigma_short",
        "sigma_elbow",
        "pin_mismatch_m",
    )
    for finger in FINGERS:
        geometry = geometries[finger]
        for field in numeric_fields:
            if not np.allclose(
                getattr(geometry, field),
                getattr(reference, field),
                atol=tolerance,
                rtol=0.0,
            ):
                return False
        for field in scalar_fields:
            if not math.isclose(
                float(getattr(geometry, field)),
                float(getattr(reference, field)),
                abs_tol=tolerance,
                rel_tol=0.0,
            ):
                return False
        for range_field in ("active_ranges", "passive_ranges"):
            reference_ranges = getattr(reference, range_field)
            geometry_ranges = getattr(geometry, range_field)
            if set(geometry_ranges) != set(reference_ranges):
                return False
            for kind, limits in reference_ranges.items():
                if not np.allclose(
                    geometry_ranges[kind],
                    limits,
                    atol=tolerance,
                    rtol=0.0,
                ):
                    return False

    bases = {
        finger: np.asarray(geometries[finger].p_A_world, dtype=float)
        for finger in FINGERS
    }
    center_xy = np.mean(np.vstack([value[:2] for value in bases.values()]), axis=0)
    radial_lengths = [float(np.linalg.norm(value[:2] - center_xy)) for value in bases.values()]
    if min(radial_lengths) <= tolerance or max(radial_lengths) - min(radial_lengths) > tolerance:
        return False
    if max(float(value[2]) for value in bases.values()) - min(float(value[2]) for value in bases.values()) > tolerance:
        return False

    expected = {
        "R": np.array([+1.0, 0.0]),
        "U": np.array([0.0, +1.0]),
        "L": np.array([-1.0, 0.0]),
        "D": np.array([0.0, -1.0]),
    }
    for finger, direction in expected.items():
        radial = bases[finger][:2] - center_xy
        radial /= max(float(np.linalg.norm(radial)), 1e-12)
        if not np.allclose(radial, direction, atol=tolerance, rtol=0.0):
            return False
        rotation = np.asarray(geometries[finger].R_A_world, dtype=float)
        expected_local_x_world = np.array([direction[0], direction[1], 0.0], dtype=float)
        if not np.allclose(
            rotation @ np.array([1.0, 0.0, 0.0]),
            expected_local_x_world,
            atol=tolerance,
            rtol=0.0,
        ):
            return False
        if not np.allclose(
            rotation @ np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            atol=tolerance,
            rtol=0.0,
        ):
            return False
    return True


class WorkspaceScanner:
    """Strict-grid analytic IK scanner for one process."""

    def __init__(self, cfg: ScanConfig):
        self.cfg = cfg
        _validate_local_model_files()
        self.ik = GripperIK.from_xml(DEFAULT_XML)
        self.theta_degrees = _independent_theta_degrees(cfg)
        self.r_values = arange_inclusive(0.0, cfg.r_search_max, cfg.r_step)
        # Cache only numerically identical sampled centers.  A millimetre cache
        # would incorrectly merge neighbouring theta rays near r=0.
        self.cache_resolution = 1e-10

        yaw = math.radians(cfg.contact_yaw_deg)
        c = math.cos(yaw)
        s = math.sin(yaw)
        yaw_rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        zero_center = np.zeros(3, dtype=float)
        nominal_offsets = t6_kinematics.cube_tip_targets(
            zero_center,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            distances=cfg.tip_center_distance,
        )
        self.tip_offsets = {
            finger: yaw_rotation @ np.asarray(nominal_offsets[finger], dtype=float)
            for finger in FINGERS
        }

    def _center_feasible(
        self,
        center: np.ndarray,
        cache: Dict[Tuple[int, int, int], bool],
    ) -> bool:
        key = _quantized_key(center, self.cache_resolution)
        cached = cache.get(key)
        if cached is not None:
            return cached

        # The IK geometry remains exactly as read from the local XML.  Moving
        # the workspace-only root by +dz is equivalent to querying that fixed
        # geometry with every world target shifted by -dz.  x/y are unchanged.
        query_center = np.asarray(center, dtype=float).copy()
        query_center[2] -= self.cfg.root_height_offset
        targets = {
            finger: query_center + self.tip_offsets[finger]
            for finger in FINGERS
        }
        feasible = all(
            self.ik.finger_target_feasible(finger, targets[finger])
            for finger in FINGERS
        )
        cache[key] = feasible
        return feasible

    def scan_layer(self, z: float) -> LayerResult:
        cache: Dict[Tuple[int, int, int], bool] = {}
        pose_checks = 0
        origin = np.array([self.cfg.root_x, self.cfg.root_y, float(z)], dtype=float)
        pose_checks += 1
        if not self._center_feasible(origin, cache):
            return LayerResult(
                z=float(z),
                max_inner_radius=0.0,
                limiting_theta_deg=0.0,
                center_feasible=False,
                search_limit_reached=False,
                pose_checks=pose_checks,
            )

        current_limit_index = len(self.r_values) - 1
        limiting_theta = 0.0
        first_ray = True

        for theta_deg in self.theta_degrees:
            theta = math.radians(float(theta_deg))
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            last_feasible_index = 0
            failed = False

            # The first ray finds a complete upper bound.  Every later ray only
            # needs to reach the current global minimum: if that point is still
            # feasible, the ray cannot reduce the inscribed circle.  This is an
            # exact early exit, not an adaptive or binary approximation.
            upper_index = len(self.r_values) - 1 if first_ray else current_limit_index
            for radius_index in range(1, upper_index + 1):
                radius = float(self.r_values[radius_index])
                center = np.array(
                    [
                        self.cfg.root_x + radius * cos_theta,
                        self.cfg.root_y + radius * sin_theta,
                        float(z),
                    ],
                    dtype=float,
                )
                pose_checks += 1
                if self._center_feasible(center, cache):
                    last_feasible_index = radius_index
                else:
                    failed = True
                    break

            if failed and last_feasible_index < current_limit_index:
                current_limit_index = last_feasible_index
                limiting_theta = float(theta_deg)
            first_ray = False

            if current_limit_index == 0:
                break

        search_limit_reached = current_limit_index == len(self.r_values) - 1
        return LayerResult(
            z=float(z),
            max_inner_radius=float(self.r_values[current_limit_index]),
            limiting_theta_deg=limiting_theta,
            center_feasible=True,
            search_limit_reached=bool(search_limit_reached),
            pose_checks=pose_checks,
        )

    def scan_plane_max_radii(self, z: float) -> PlaneBoundaryResult:
        """Return the center-connected feasible radius on every requested ray.

        Each ray starts at the common inscribed-circle origin and closes at its
        first infeasible radial sample.  Unlike :meth:`scan_layer`, every theta
        is scanned independently to its own boundary; the directional radii are
        not reduced to their global minimum (the inscribed-circle radius).
        """
        theta_degrees = _requested_theta_degrees(self.cfg.theta_step_deg)
        max_radii = np.zeros(theta_degrees.shape, dtype=float)
        cache: Dict[Tuple[int, int, int], bool] = {}
        pose_checks = 1
        start = time.monotonic()
        origin = np.array(
            [self.cfg.root_x, self.cfg.root_y, float(z)],
            dtype=float,
        )
        if not self._center_feasible(origin, cache):
            return PlaneBoundaryResult(
                z=float(z),
                theta_degrees=theta_degrees,
                max_radii=max_radii,
                pose_checks=pose_checks,
                elapsed_s=float(time.monotonic() - start),
            )

        for theta_index, theta_deg in enumerate(theta_degrees):
            theta = math.radians(float(theta_deg) % 360.0)
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            max_feasible_index = 0

            for radius_index in range(1, len(self.r_values)):
                radius_value = self.r_values[radius_index]
                radius = float(radius_value)
                center = np.array(
                    [
                        self.cfg.root_x + radius * cos_theta,
                        self.cfg.root_y + radius * sin_theta,
                        float(z),
                    ],
                    dtype=float,
                )
                pose_checks += 1
                if self._center_feasible(center, cache):
                    max_feasible_index = radius_index
                else:
                    break

            max_radii[theta_index] = float(self.r_values[max_feasible_index])
            if max_feasible_index == len(self.r_values) - 1:
                raise RuntimeError(
                    "r-search-max 截断了最优 z 平面从中心连续可达的 XY workspace；"
                    f"theta={float(theta_deg):.6f} deg, z={float(z):.6f} m。"
                    "请增大 --r-search-max 后重跑。"
                )

        return PlaneBoundaryResult(
            z=float(z),
            theta_degrees=theta_degrees,
            max_radii=max_radii,
            pose_checks=pose_checks,
            elapsed_s=float(time.monotonic() - start),
        )


_WORKER_SCANNER: Optional[WorkspaceScanner] = None


def _initialize_worker(cfg: ScanConfig) -> None:
    global _WORKER_SCANNER
    _require_local_runtime("T6 workspace multiprocessing 子进程")
    if Path(t6_kinematics.__file__).resolve().parent != MODULE_DIR:
        raise RuntimeError("multiprocessing 子进程未从本地导入 T6_kinematics")
    _validate_local_model_files()
    _WORKER_SCANNER = WorkspaceScanner(cfg)


def _scan_layer_worker(z: float) -> LayerResult:
    if _WORKER_SCANNER is None:
        raise RuntimeError("workspace worker was not initialized")
    return _WORKER_SCANNER.scan_layer(float(z))


def scan_workspace(cfg: ScanConfig) -> WorkspaceResult:
    z_values = arange_inclusive(cfg.z_min, cfg.z_max, cfg.z_step)
    if z_values.size == 0:
        raise ValueError("z 扫描范围为空")

    workers = int(cfg.workers)
    if workers <= 0:
        workers = max(1, min((os.cpu_count() or 2) - 1, int(z_values.size)))
    workers = max(1, min(workers, int(z_values.size)))

    requested_theta_count = len(_requested_theta_degrees(cfg.theta_step_deg))
    independent_theta_count = len(_independent_theta_degrees(cfg))
    print("开始扫描 T6 圆盘抓持 analytic workspace")
    print(f"  本地 XML: {DEFAULT_XML}")
    print(f"  圆盘直径: {cfg.disk_diameter:.6f} m")
    print(f"  hard-tip 半径（来自 XML）: {cfg.hard_tip_radius:.6f} m")
    print(f"  tip 球心到圆盘中心距离: {cfg.tip_center_distance:.6f} m")
    print(f"  root x/y（保持不变）: ({cfg.root_x:.6f}, {cfg.root_y:.6f}) m")
    print(f"  XML root z（只读）: {cfg.xml_root_z:.6f} m")
    print(f"  workspace root 高度: {cfg.root_height:.6f} m")
    print(f"  workspace-only root z 偏移: {cfg.root_height_offset:+.6f} m")
    print(f"  z: {cfg.z_min:.3f} .. {cfg.z_max:.3f} m, step={cfg.z_step:.3f} m")
    print(
        f"  theta: 0 .. 360 deg, step={cfg.theta_step_deg:g} deg "
        f"({requested_theta_count} 个含重复端点的采样；独立计算 {independent_theta_count} 个方向)"
    )
    print(f"  r: 0 .. {cfg.r_search_max:.3f} m, strict step={cfg.r_step:.3f} m")
    print(f"  四重对称加速: {cfg.use_fourfold_symmetry}")
    print(f"  并行进程: {workers}")

    start = time.monotonic()
    results: List[LayerResult] = []

    def report(done: int, layer: LayerResult) -> None:
        if done == 1 or done == len(z_values) or done % 10 == 0:
            print(
                f"  [{done:3d}/{len(z_values)}] z={layer.z:.3f} m, "
                f"max inner r={layer.max_inner_radius:.3f} m, "
                f"elapsed={time.monotonic() - start:.1f}s",
                flush=True,
            )

    def serial_scan() -> List[LayerResult]:
        scanner = WorkspaceScanner(cfg)
        serial_results: List[LayerResult] = []
        for index, z in enumerate(z_values, start=1):
            layer = scanner.scan_layer(float(z))
            serial_results.append(layer)
            report(index, layer)
        return serial_results

    if workers == 1:
        results = serial_scan()
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(cfg,),
            ) as executor:
                futures = {executor.submit(_scan_layer_worker, float(z)): float(z) for z in z_values}
                for done, future in enumerate(as_completed(futures), start=1):
                    layer = future.result()
                    results.append(layer)
                    report(done, layer)
            results.sort(key=lambda layer: layer.z)
        except (OSError, PermissionError, BrokenProcessPool) as exc:
            print(f"  并行扫描不可用（{exc}），自动改为单进程。", flush=True)
            results = serial_scan()

    elapsed = time.monotonic() - start
    capped = [layer for layer in results if layer.search_limit_reached]
    if capped:
        heights = ", ".join(f"{layer.z:.3f}" for layer in capped[:8])
        suffix = "..." if len(capped) > 8 else ""
        raise RuntimeError(
            "r-search-max 截断了最大内切圆；请增大 --r-search-max 后重跑。"
            f"受影响 z: {heights}{suffix}"
        )

    return WorkspaceResult(layers=tuple(results), elapsed_s=float(elapsed))


def scan_max_xy_radius_by_theta(
    result: WorkspaceResult,
    cfg: ScanConfig,
) -> PlaneBoundaryResult:
    if not result.radii.size:
        raise ValueError("没有 z 扫描结果，无法复扫最优平面")
    max_index = int(np.argmax(result.radii))
    z_at_maximum = float(result.z_values[max_index])
    print(
        f"正在 z={z_at_maximum:.6f} m（最大内切圆所在高度）"
        "从内切圆中心逐方向复扫连续可达的 XY workspace 边界……",
        flush=True,
    )
    boundary = WorkspaceScanner(cfg).scan_plane_max_radii(z_at_maximum)
    print(
        f"  平面复扫完成: {len(boundary.theta_degrees)} 个 theta，"
        f"{boundary.pose_checks} 次采样，耗时 {boundary.elapsed_s:.2f} s",
        flush=True,
    )
    return boundary




def make_inner_circle_ring_points(result: WorkspaceResult, cfg: ScanConfig) -> np.ndarray:
    theta = np.deg2rad(np.arange(0.0, 360.0, cfg.theta_step_deg, dtype=float))
    points: List[np.ndarray] = []
    for layer in result.layers:
        if not layer.center_feasible:
            continue
        if layer.max_inner_radius <= 0.0:
            points.append(np.array([[cfg.root_x, cfg.root_y, layer.z]], dtype=float))
            continue
        x = cfg.root_x + layer.max_inner_radius * np.cos(theta)
        y = cfg.root_y + layer.max_inner_radius * np.sin(theta)
        z = np.full(theta.shape, layer.z, dtype=float)
        points.append(np.column_stack([x, y, z]))
    return np.vstack(points) if points else np.zeros((0, 3), dtype=float)


def build_voxel_mask(
    result: WorkspaceResult,
    cfg: ScanConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_radius = max(float(np.max(result.radii)) if result.radii.size else 0.0, cfg.voxel_size)
    x_values = arange_inclusive(cfg.root_x - max_radius, cfg.root_x + max_radius, cfg.voxel_size)
    y_values = arange_inclusive(cfg.root_y - max_radius, cfg.root_y + max_radius, cfg.voxel_size)
    z_values = result.z_values.copy()
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    rr_squared = (xx - cfg.root_x) ** 2 + (yy - cfg.root_y) ** 2
    mask = np.zeros((x_values.size, y_values.size, z_values.size), dtype=bool)

    for index, layer in enumerate(result.layers):
        if not layer.center_feasible:
            continue
        if layer.max_inner_radius <= 0.0:
            ix = int(np.argmin(np.abs(x_values - cfg.root_x)))
            iy = int(np.argmin(np.abs(y_values - cfg.root_y)))
            mask[ix, iy, index] = True
        else:
            mask[:, :, index] = rr_squared <= layer.max_inner_radius**2 + 1e-12
    return mask, x_values, y_values, z_values


def voxel_surface_points(result: WorkspaceResult, cfg: ScanConfig) -> Tuple[np.ndarray, int]:
    mask, x_values, y_values, z_values = build_voxel_mask(result, cfg)
    eroded = np.zeros_like(mask, dtype=bool)
    if all(size >= 3 for size in mask.shape):
        center = mask[1:-1, 1:-1, 1:-1]
        eroded[1:-1, 1:-1, 1:-1] = (
            center
            & mask[:-2, 1:-1, 1:-1]
            & mask[2:, 1:-1, 1:-1]
            & mask[1:-1, :-2, 1:-1]
            & mask[1:-1, 2:, 1:-1]
            & mask[1:-1, 1:-1, :-2]
            & mask[1:-1, 1:-1, 2:]
        )
    surface_indices = np.argwhere(mask & ~eroded)
    occupied_count = int(np.count_nonzero(mask))
    if surface_indices.size == 0:
        return np.zeros((0, 3), dtype=float), occupied_count

    points = np.column_stack(
        [
            x_values[surface_indices[:, 0]],
            y_values[surface_indices[:, 1]],
            z_values[surface_indices[:, 2]],
        ]
    )
    if points.shape[0] > MAX_VOXEL_SURFACE_POINTS:
        rng = np.random.default_rng(RANDOM_SEED)
        keep = np.sort(rng.choice(points.shape[0], size=MAX_VOXEL_SURFACE_POINTS, replace=False))
        points = points[keep]
    return points, occupied_count


def save_radius_csv(result: WorkspaceResult, cfg: ScanConfig) -> Path:
    path = _output_path(cfg, "t6_workspace_max_inner_radius.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z_m", "max_inner_circle_radius_m"])
        for layer in result.layers:
            writer.writerow([f"{layer.z:.6f}", f"{layer.max_inner_radius:.6f}"])
    return path


def save_max_xy_radius_by_theta_csv(
    boundary: PlaneBoundaryResult,
    cfg: ScanConfig,
) -> Path:
    path = _output_path(cfg, "t6_workspace_max_xy_radius_by_theta.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["theta_deg", "max_radius_m"])
        for theta_deg, max_radius in zip(
            boundary.theta_degrees,
            boundary.max_radii,
        ):
            writer.writerow([f"{theta_deg:.6f}", f"{max_radius:.6f}"])
    return path










def save_summary(
    result: WorkspaceResult,
    boundary: PlaneBoundaryResult,
    cfg: ScanConfig,
    outputs: Dict[str, Path],
    occupied_voxels: int,
    rendered_voxel_points: int,
) -> Path:
    path = _output_path(cfg, "t6_workspace_summary.txt")
    max_index = int(np.argmax(result.radii)) if result.radii.size else 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("T6 disk-grasp analytic workspace summary\n")
        handle.write(f"project_xml: {DEFAULT_XML}\n")
        handle.write(f"kinematics_module: {KINEMATICS_PATH}\n")
        handle.write(f"runtime_python: {sys.executable}\n")
        handle.write(f"output_dir: {_resolve_local_output_dir(cfg.output_dir)}\n")
        handle.write("external_code_dependencies: none\n")
        handle.write(f"disk_diameter_m: {cfg.disk_diameter:.6f}\n")
        handle.write(f"hard_tip_radius_m: {cfg.hard_tip_radius:.6f}\n")
        handle.write(f"tip_center_distance_m: {cfg.tip_center_distance:.6f}\n")
        handle.write(f"contact_yaw_deg: {cfg.contact_yaw_deg:.6f}\n")
        handle.write(
            f"root_xy_m: ({cfg.root_x:.6f}, {cfg.root_y:.6f})\n"
        )
        handle.write(f"xml_root_z_m: {cfg.xml_root_z:.6f}\n")
        handle.write(f"workspace_root_height_m: {cfg.root_height:.6f}\n")
        handle.write(
            f"workspace_root_height_offset_m: {cfg.root_height_offset:+.6f}\n"
        )
        handle.write(
            "workspace_target_query_transform: "
            "xml_target_z = workspace_target_z - workspace_root_height_offset\n"
        )
        handle.write("xml_model_modified: false\n")
        handle.write("z_coordinate_frame: world_absolute\n")
        handle.write(f"z_scan_m: {cfg.z_min:.6f} .. {cfg.z_max:.6f}, step {cfg.z_step:.6f}\n")
        handle.write(f"theta_scan_deg: 0 .. 360 inclusive, step {cfg.theta_step_deg:.6f}\n")
        handle.write(f"r_step_m: {cfg.r_step:.6f}\n")
        handle.write(f"r_search_max_m: {cfg.r_search_max:.6f}\n")
        handle.write(f"voxel_size_m: {cfg.voxel_size:.6f}\n")
        handle.write(f"fourfold_symmetry_used: {cfg.use_fourfold_symmetry}\n")
        handle.write(f"z_layer_count: {len(result.layers)}\n")
        handle.write(f"analytic_pose_checks: {sum(layer.pose_checks for layer in result.layers)}\n")
        handle.write(f"scan_elapsed_s: {result.elapsed_s:.3f}\n")
        if result.radii.size:
            handle.write(f"maximum_radius_m: {result.radii[max_index]:.6f}\n")
            handle.write(f"z_at_maximum_radius_m: {result.z_values[max_index]:.6f}\n")
        handle.write(f"max_xy_boundary_z_m: {boundary.z:.6f}\n")
        handle.write(f"max_xy_boundary_theta_count: {len(boundary.theta_degrees)}\n")
        handle.write(f"max_xy_boundary_pose_checks: {boundary.pose_checks}\n")
        handle.write(f"max_xy_boundary_scan_elapsed_s: {boundary.elapsed_s:.3f}\n")
        handle.write(f"max_xy_boundary_min_radius_m: {np.min(boundary.max_radii):.6f}\n")
        handle.write(f"max_xy_boundary_max_radius_m: {np.max(boundary.max_radii):.6f}\n")
        handle.write(f"occupied_voxels: {occupied_voxels}\n")
        handle.write(f"rendered_surface_voxel_points: {rendered_voxel_points}\n")
        for name, output in outputs.items():
            handle.write(f"{name}: {output}\n")
    return path


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算 T6 四指同时触碰水平圆盘时的最大解析平移 workspace。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--disk-diameter", type=_positive_float, default=DISK_DIAMETER_M, help="圆盘直径（m）")
    parser.add_argument("--contact-yaw-deg", type=float, default=CONTACT_YAW_DEG, help="四个等间隔接触点相对 U/D/L/R 轴的固定旋转角")
    parser.add_argument(
        "--root-height",
        type=float,
        default=ROOT_HEIGHT_M,
        help="仅用于 workspace 计算的 root 世界 z 高度（m），不修改任何 XML",
    )
    parser.add_argument("--z-min", type=float, default=Z_MIN_M, help="圆盘中心 z 下界（m）")
    parser.add_argument("--z-max", type=float, default=Z_MAX_M, help="圆盘中心 z 上界（m）")
    parser.add_argument("--z-step", type=_positive_float, default=Z_STEP_M, help="z 步长（m）")
    parser.add_argument("--theta-step-deg", type=_positive_float, default=THETA_STEP_DEG, help="theta 步长（deg）")
    parser.add_argument("--r-step", type=_positive_float, default=R_STEP_M, help="r 步长（m）")
    parser.add_argument("--r-search-max", type=_positive_float, default=R_SEARCH_MAX_M, help="径向搜索保护上界（m）；若碰到上界程序会报错而不会静默截断")
    parser.add_argument("--voxel-size", type=_positive_float, default=VOXEL_SIZE_M, help="体素尺寸（m）")
    parser.add_argument("--workers", type=int, default=0, help="并行 z 层进程数；0=自动，1=关闭并行")
    parser.add_argument("--no-fourfold-symmetry", action="store_true", help="不利用经 XML 验证的四重对称性，逐一计算完整 360 度")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR.name,
        help="输出目录；相对路径按脚本所在 TRAnslation 文件夹解析，且不能指向该文件夹之外",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ScanConfig:
    _require_local_runtime()
    _validate_local_model_files()
    if args.z_max < args.z_min - 1e-12:
        raise ValueError("z-max 必须大于或等于 z-min")
    if args.theta_step_deg > 360.0:
        raise ValueError("theta-step-deg 不能大于 360")
    if args.r_step > args.r_search_max:
        raise ValueError("r-step 不能大于 r-search-max")

    ik = GripperIK.from_xml(DEFAULT_XML)
    hard_tip_radius = float(ik.geometry_model.xml_info.hard_tip_radius)
    geometries = ik.geometry_model.fingers
    root_position = np.asarray(ik.model_info.root_pos, dtype=float)
    if root_position.shape != (3,) or not np.all(np.isfinite(root_position)):
        raise ValueError("XML root pos 必须包含三个有限数值")
    xml_root_z = float(root_position[2])
    root_height = float(args.root_height)
    if not math.isfinite(root_height):
        raise ValueError("root-height 必须是有限数值")
    root_height_offset = root_height - xml_root_z
    root_xy = np.mean(
        np.vstack(
            [
                np.asarray(geometries[finger].p_A_world[:2], dtype=float)
                for finger in FINGERS
            ]
        ),
        axis=0,
    )

    requested_symmetry = not bool(args.no_fourfold_symmetry)
    use_symmetry = requested_symmetry and _fourfold_geometry_is_valid(ik)
    if requested_symmetry and not use_symmetry:
        print("[警告] 当前 XML 未通过四重对称性验证，将自动逐一计算完整 360 度。", flush=True)

    output_dir = _resolve_local_output_dir(args.output_dir)

    disk_diameter = float(args.disk_diameter)
    return ScanConfig(
        output_dir=output_dir,
        disk_diameter=disk_diameter,
        hard_tip_radius=hard_tip_radius,
        tip_center_distance=0.5 * disk_diameter + hard_tip_radius,
        contact_yaw_deg=float(args.contact_yaw_deg),
        root_x=float(root_xy[0]),
        root_y=float(root_xy[1]),
        xml_root_z=xml_root_z,
        root_height=root_height,
        root_height_offset=root_height_offset,
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        z_step=float(args.z_step),
        theta_step_deg=float(args.theta_step_deg),
        r_step=float(args.r_step),
        r_search_max=float(args.r_search_max),
        voxel_size=float(args.voxel_size),
        use_fourfold_symmetry=bool(use_symmetry),
        workers=int(args.workers),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Compute workspace radii without generating plots."""
    cfg = build_config(parse_args(argv))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result = scan_workspace(cfg)
    print(f"Numerical workspace: {save_radius_csv(result, cfg)}")
    if result.radii.size:
        index = int(np.argmax(result.radii))
        print(f"Maximum inscribed radius: {result.radii[index]:.6f} m at z={result.z_values[index]:.6f} m")
    print(f"Elapsed: {result.elapsed_s:.2f} s")


if __name__ == "__main__":
    main()
