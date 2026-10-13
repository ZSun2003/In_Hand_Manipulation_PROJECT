"""Collision-aware cached A* and cubic-spline nominal screw planner.

The planner replaces only the learned ``ALIGN`` suffix.  Pickup, measured
gates, insertion, thread engagement, and regrasp remain owned by the task
supervisor.  Planning is performed at reset time; :meth:`command` contains
only indexed Torch tensor operations and is therefore safe in the CUDA hot
loop.

The socket mouth moves on a 20 mm orbit while its elevation remains 45
degrees.  Rotating every reset into the mouth-radial frame makes the circular
workspace, obstacle, and staging goal identical for every socket azimuth.  A
reverse shortest-path field supplies an exact admissible A* heuristic.  The
actual A* result is cached by canonical start cell, so a large vector reset
does not run a full graph search once per world.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time
from typing import Dict, List, Mapping, Tuple

import numpy as np

from screw_config import DEFAULT_CONFIG
from screw_control import (
    DIRECT_ALIGN_STEPS,
    SCRIPTED_PREFIX_STEPS,
    TaskPhase,
)

from .supervisor import (
    CommandBatch,
    matrix_to_quaternion,
    quaternion_slerp,
    quaternion_to_matrix,
)


TRANSPORT_STEPS = 190
ROTATE_RAISE_STEPS = 80
DOCK_STEPS = 80
GRID_RESOLUTION_M = 0.002
STAGING_STANDOFF_M = 0.085
SPLINE_VALIDATION_SPACING_M = 0.00025

if TRANSPORT_STEPS + ROTATE_RAISE_STEPS + DOCK_STEPS != DIRECT_ALIGN_STEPS:
    raise RuntimeError("A* ALIGN segment lengths must sum to DIRECT_ALIGN_STEPS")


class AStarPlanError(RuntimeError):
    """Raised when a requested reset has no certified nominal A* path."""


@dataclass(frozen=True)
class _CachedGridPlan:
    """Canonical grid path shared by continuous starts in one grid cell."""

    indices: Tuple[int, ...]
    points_xy: np.ndarray
    expanded: int


def _torch():
    import torch

    return torch


def _minimum_jerk(value):
    """Fifth-order time scaling with zero endpoint velocity/acceleration."""

    return value * value * value * (
        10.0 + value * (-15.0 + 6.0 * value)
    )


class BatchedAStarSplinePlanner:
    """Prepare and serve batched A*--spline ``ALIGN`` commands.

    Parameters
    ----------
    config:
        Validated hpc configuration.  The planner currently certifies only
        the formal 45-degree, 100 mm-radius task.
    worlds:
        Number of vector environments.
    device:
        Torch device on which the command plans are retained.
    """

    _NEIGHBOURS: Tuple[Tuple[int, int, float], ...] = (
        (-1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, 0, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(self, config, worlds: int, device) -> None:
        torch = _torch()
        self.config = config
        self.worlds = int(worlds)
        if self.worlds <= 0:
            raise ValueError("worlds must be positive")
        self.device = torch.device(device)
        self.dtype = torch.float32
        task = DEFAULT_CONFIG
        self.task = task

        fixed_angle = float(config.curriculum.fixed_socket_angle_deg)
        if abs(fixed_angle - 45.0) > 1.0e-9:
            raise ValueError("A*--spline planner is certified only at 45 degrees")
        radius = float(config.curriculum.final_spawn_radius_m)
        if abs(radius - task.spawn_radius_m) > 1.0e-9:
            raise ValueError(
                "A*--spline planner requires the formal 100 mm workspace"
            )
        self.workspace_radius_m = radius
        self.grid_resolution_m = GRID_RESOLUTION_M
        self.staging_standoff_m = STAGING_STANDOFF_M

        elevation = math.radians(fixed_angle)
        cosine = max(math.cos(elevation), 0.0)
        sine = max(math.sin(elevation), 0.0)
        margin = task.screw_head_radius_m + task.spawn_clearance_m
        # Coordinates use +x along the socket inward tabletop direction.  The
        # canonical mouth is at (+orbit_radius, 0).
        self._obstacle_center_x = (
            task.socket_mouth_orbit_radius_m
            + 0.5 * task.socket_outer_length_m * cosine
        )
        self._obstacle_half_x = (
            0.5 * task.socket_outer_length_m * cosine
            + 0.5 * task.socket_outer_width_m * sine
            + margin
        )
        self._obstacle_half_y = 0.5 * task.socket_outer_height_m + margin

        count = int(round(2.0 * radius / self.grid_resolution_m)) + 1
        self._grid_values = np.linspace(-radius, radius, count, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(
            self._grid_values, self._grid_values, indexing="ij"
        )
        in_disc = grid_x * grid_x + grid_y * grid_y <= radius * radius + 1.0e-12
        in_obstacle = (
            np.abs(grid_x - self._obstacle_center_x)
            <= self._obstacle_half_x
        ) & (np.abs(grid_y) <= self._obstacle_half_y)
        self._free = in_disc & ~in_obstacle
        self._grid_shape = self._free.shape

        canonical_axis_x = cosine
        canonical_stage_x = (
            task.socket_mouth_orbit_radius_m
            - self.staging_standoff_m * canonical_axis_x
        )
        self._canonical_stage_xy = np.asarray(
            (canonical_stage_x, 0.0), dtype=np.float64
        )
        self._goal_ij = self._nearest_free_grid_cell(
            self._canonical_stage_xy, require_visible=False
        )
        if self._goal_ij is None:
            raise RuntimeError("canonical A* staging goal is not a free grid cell")
        self._goal_flat = self._flatten(*self._goal_ij)
        self._reverse_distance, self._successor = (
            self._build_reverse_shortest_path_tree()
        )

        self._grid_plan_cache: Dict[Tuple[int, int], _CachedGridPlan] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_reset_plans = 0
        self._total_planning_latency_s = 0.0
        self._total_spline_fallbacks = 0

        self._plan_tip = torch.zeros(
            (self.worlds, DIRECT_ALIGN_STEPS, 3),
            device=self.device,
            dtype=self.dtype,
        )
        self._plan_rotation = torch.eye(
            3, device=self.device, dtype=self.dtype
        ).reshape(1, 1, 3, 3).repeat(
            self.worlds, DIRECT_ALIGN_STEPS, 1, 1
        )
        self._plan_found = torch.zeros(
            self.worlds, device=self.device, dtype=torch.bool
        )
        self._spline_fallback = torch.zeros_like(self._plan_found)
        self._cache_hit = torch.zeros_like(self._plan_found)
        self._planning_latency_s = torch.zeros(
            self.worlds, device=self.device, dtype=self.dtype
        )
        self._path_length_m = torch.zeros_like(self._planning_latency_s)
        self._minimum_obstacle_clearance_m = torch.zeros_like(
            self._planning_latency_s
        )
        self._astar_expanded = torch.zeros(
            self.worlds, device=self.device, dtype=torch.long
        )
        self._start_grid_flat = torch.full(
            (self.worlds,), -1, device=self.device, dtype=torch.long
        )
        self._row = torch.arange(
            self.worlds, device=self.device, dtype=torch.long
        )

    # ------------------------------------------------------------------
    # Grid geometry and graph search
    # ------------------------------------------------------------------
    def _flatten(self, i: int, j: int) -> int:
        return int(i) * self._grid_shape[1] + int(j)

    def _unflatten(self, value: int) -> Tuple[int, int]:
        return divmod(int(value), self._grid_shape[1])

    def _grid_xy(self, i: int, j: int) -> np.ndarray:
        return np.asarray(
            (self._grid_values[int(i)], self._grid_values[int(j)]),
            dtype=np.float64,
        )

    def _points_free(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        # Reset coordinates originate as float32 CUDA tensors.  A point on the
        # exact 100 mm boundary can therefore be reconstructed a few nanometres
        # outside the mathematical disc after yaw canonicalisation.
        radius_with_roundoff = self.workspace_radius_m + 1.0e-6
        disc = np.sum(points * points, axis=1) <= (
            radius_with_roundoff * radius_with_roundoff
        )
        obstacle = (
            np.abs(points[:, 0] - self._obstacle_center_x)
            <= self._obstacle_half_x + 1.0e-12
        ) & (np.abs(points[:, 1]) <= self._obstacle_half_y + 1.0e-12)
        return disc & ~obstacle

    def _segment_free(self, first: np.ndarray, second: np.ndarray) -> bool:
        first = np.asarray(first, dtype=np.float64).reshape(2)
        second = np.asarray(second, dtype=np.float64).reshape(2)
        length = float(np.linalg.norm(second - first))
        count = max(2, int(math.ceil(length / SPLINE_VALIDATION_SPACING_M)) + 1)
        fraction = np.linspace(0.0, 1.0, count, dtype=np.float64)
        samples = first[None, :] + fraction[:, None] * (second - first)[None, :]
        return bool(self._points_free(samples).all())

    def _obstacle_clearance(self, points_xy: np.ndarray) -> np.ndarray:
        """Euclidean distance to the inflated rectangular C-obstacle."""

        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        delta_x = np.maximum(
            np.abs(points[:, 0] - self._obstacle_center_x)
            - self._obstacle_half_x,
            0.0,
        )
        delta_y = np.maximum(
            np.abs(points[:, 1]) - self._obstacle_half_y,
            0.0,
        )
        return np.hypot(delta_x, delta_y)

    def _nearest_free_grid_cell(
        self,
        point_xy: np.ndarray,
        *,
        require_visible: bool,
    ) -> Tuple[int, int] | None:
        point = np.asarray(point_xy, dtype=np.float64).reshape(2)
        base_i = int(round((point[0] + self.workspace_radius_m) / self.grid_resolution_m))
        base_j = int(round((point[1] + self.workspace_radius_m) / self.grid_resolution_m))
        candidates: List[Tuple[float, int, int]] = []
        # Four cells are sufficient for normal quantisation; a slightly wider
        # search makes points arbitrarily close to the inflated OBB robust.
        for radius in range(0, 5):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if radius > 0 and max(abs(di), abs(dj)) != radius:
                        continue
                    i, j = base_i + di, base_j + dj
                    if not (
                        0 <= i < self._grid_shape[0]
                        and 0 <= j < self._grid_shape[1]
                        and self._free[i, j]
                    ):
                        continue
                    grid_point = self._grid_xy(i, j)
                    if require_visible and not self._segment_free(point, grid_point):
                        continue
                    distance = float(np.linalg.norm(grid_point - point))
                    candidates.append((distance, i, j))
            if candidates:
                candidates.sort()
                return candidates[0][1], candidates[0][2]
        return None

    def _iter_neighbours(self, i: int, j: int):
        for di, dj, scale in self._NEIGHBOURS:
            ni, nj = i + di, j + dj
            if not (
                0 <= ni < self._grid_shape[0]
                and 0 <= nj < self._grid_shape[1]
                and self._free[ni, nj]
            ):
                continue
            # A diagonal may not pass between two occupied cardinal cells.
            if di and dj and not (
                self._free[i + di, j] and self._free[i, j + dj]
            ):
                continue
            yield ni, nj, scale * self.grid_resolution_m

    def _build_reverse_shortest_path_tree(self):
        node_count = self._grid_shape[0] * self._grid_shape[1]
        distance = np.full(node_count, np.inf, dtype=np.float64)
        successor = np.full(node_count, -1, dtype=np.int32)
        distance[self._goal_flat] = 0.0
        queue: List[Tuple[float, int]] = [(0.0, self._goal_flat)]
        while queue:
            cost, flat = heapq.heappop(queue)
            if cost != distance[flat]:
                continue
            i, j = self._unflatten(flat)
            for ni, nj, edge in self._iter_neighbours(i, j):
                neighbour = self._flatten(ni, nj)
                candidate = cost + edge
                if candidate + 1.0e-15 < distance[neighbour]:
                    distance[neighbour] = candidate
                    successor[neighbour] = flat
                    heapq.heappush(queue, (candidate, neighbour))
        successor[self._goal_flat] = self._goal_flat
        return distance, successor

    def _astar(self, start_ij: Tuple[int, int]) -> _CachedGridPlan:
        """Run real 8-neighbour A* using the reverse field as exact heuristic."""

        start = self._flatten(*start_ij)
        if not math.isfinite(float(self._reverse_distance[start])):
            raise AStarPlanError(
                f"canonical start grid cell {start_ij} is disconnected"
            )
        node_count = self._grid_shape[0] * self._grid_shape[1]
        g_score = np.full(node_count, np.inf, dtype=np.float64)
        parent = np.full(node_count, -1, dtype=np.int32)
        closed = np.zeros(node_count, dtype=np.bool_)
        g_score[start] = 0.0
        counter = 0
        # Exact reverse cost is an admissible, consistent heuristic.  Lower
        # h is the deterministic tie break, which prevents broad equal-f
        # expansion fronts.
        queue: List[Tuple[float, float, int, int]] = [
            (
                float(self._reverse_distance[start]),
                float(self._reverse_distance[start]),
                counter,
                start,
            )
        ]
        expanded = 0
        while queue:
            _f, _h, _order, flat = heapq.heappop(queue)
            if closed[flat]:
                continue
            closed[flat] = True
            expanded += 1
            if flat == self._goal_flat:
                break
            i, j = self._unflatten(flat)
            for ni, nj, edge in self._iter_neighbours(i, j):
                neighbour = self._flatten(ni, nj)
                if closed[neighbour]:
                    continue
                candidate = g_score[flat] + edge
                if candidate + 1.0e-15 >= g_score[neighbour]:
                    continue
                g_score[neighbour] = candidate
                parent[neighbour] = flat
                heuristic = float(self._reverse_distance[neighbour])
                counter += 1
                heapq.heappush(
                    queue,
                    (candidate + heuristic, heuristic, counter, neighbour),
                )
        if parent[self._goal_flat] < 0 and start != self._goal_flat:
            raise AStarPlanError(f"A* found no route from grid cell {start_ij}")
        reverse_path = [self._goal_flat]
        while reverse_path[-1] != start:
            previous = int(parent[reverse_path[-1]])
            if previous < 0:
                raise AStarPlanError("A* parent chain is incomplete")
            reverse_path.append(previous)
        path = tuple(reversed(reverse_path))
        points = np.stack(
            [self._grid_xy(*self._unflatten(flat)) for flat in path], axis=0
        )
        return _CachedGridPlan(path, points, expanded)

    # ------------------------------------------------------------------
    # Path simplification, spline construction, and time scaling
    # ------------------------------------------------------------------
    @staticmethod
    def _deduplicate(points: np.ndarray) -> np.ndarray:
        value = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(value) <= 1:
            return value.copy()
        keep = np.ones(len(value), dtype=np.bool_)
        keep[1:] = np.linalg.norm(np.diff(value, axis=0), axis=1) > 1.0e-10
        return value[keep]

    def _line_of_sight_shortcut(self, points: np.ndarray) -> np.ndarray:
        points = self._deduplicate(points)
        if len(points) <= 2:
            return points
        selected = [points[0]]
        index = 0
        while index < len(points) - 1:
            following = len(points) - 1
            while following > index + 1 and not self._segment_free(
                points[index], points[following]
            ):
                following -= 1
            selected.append(points[following])
            index = following
        return self._deduplicate(np.stack(selected, axis=0))

    @staticmethod
    def _natural_cubic_dense(points: np.ndarray) -> np.ndarray:
        """Return a dense natural cubic interpolation without SciPy."""

        points = BatchedAStarSplinePlanner._deduplicate(points)
        if len(points) < 3:
            length = float(np.linalg.norm(points[-1] - points[0]))
            count = max(
                2,
                int(math.ceil(length / SPLINE_VALIDATION_SPACING_M)) + 1,
            )
            fraction = np.linspace(0.0, 1.0, count, dtype=np.float64)
            return points[0] + fraction[:, None] * (points[-1] - points[0])

        chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if bool((chords <= 1.0e-10).any()):
            raise ValueError("cubic spline contains repeated waypoints")
        parameter = np.concatenate(([0.0], np.cumsum(chords)))
        interval = np.diff(parameter)
        size = len(points)
        matrix = np.zeros((size, size), dtype=np.float64)
        rhs = np.zeros((size, 2), dtype=np.float64)
        matrix[0, 0] = 1.0
        matrix[-1, -1] = 1.0
        for index in range(1, size - 1):
            left = interval[index - 1]
            right = interval[index]
            matrix[index, index - 1] = left
            matrix[index, index] = 2.0 * (left + right)
            matrix[index, index + 1] = right
            rhs[index] = 6.0 * (
                (points[index + 1] - points[index]) / right
                - (points[index] - points[index - 1]) / left
            )
        second = np.linalg.solve(matrix, rhs)

        dense: List[np.ndarray] = []
        for index, chord in enumerate(chords):
            # Oversample relative to the 0.25 mm validation requirement; the
            # spline derivative can exceed its waypoint chord derivative.
            count = max(
                3,
                2 * int(math.ceil(float(chord) / SPLINE_VALIDATION_SPACING_M))
                + 1,
            )
            local = np.linspace(0.0, 1.0, count, dtype=np.float64)
            if index:
                local = local[1:]
            h = interval[index]
            a = 1.0 - local
            b = local
            segment = (
                a[:, None] * points[index]
                + b[:, None] * points[index + 1]
                + (
                    (a**3 - a)[:, None] * second[index]
                    + (b**3 - b)[:, None] * second[index + 1]
                )
                * (h * h / 6.0)
            )
            dense.append(segment)
        result = np.concatenate(dense, axis=0)
        result[0] = points[0]
        result[-1] = points[-1]
        return result

    @staticmethod
    def _polyline_dense(points: np.ndarray) -> np.ndarray:
        points = BatchedAStarSplinePlanner._deduplicate(points)
        dense = []
        for index in range(len(points) - 1):
            length = float(np.linalg.norm(points[index + 1] - points[index]))
            count = max(
                2,
                int(math.ceil(length / SPLINE_VALIDATION_SPACING_M)) + 1,
            )
            fraction = np.linspace(0.0, 1.0, count, dtype=np.float64)
            if index:
                fraction = fraction[1:]
            dense.append(
                points[index]
                + fraction[:, None] * (points[index + 1] - points[index])
            )
        return np.concatenate(dense, axis=0)

    @staticmethod
    def _arc_length_resample(
        dense: np.ndarray, count: int
    ) -> Tuple[np.ndarray, float]:
        dense = np.asarray(dense, dtype=np.float64).reshape(-1, 2)
        segment = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment)))
        total = float(cumulative[-1])
        if not math.isfinite(total) or total <= 1.0e-10:
            raise AStarPlanError("planned transport path has zero length")
        fraction = np.arange(1, count + 1, dtype=np.float64) / float(count)
        target = _minimum_jerk(fraction) * total
        result = np.stack(
            (
                np.interp(target, cumulative, dense[:, 0]),
                np.interp(target, cumulative, dense[:, 1]),
            ),
            axis=-1,
        )
        result[-1] = dense[-1]
        return result, total

    @staticmethod
    def _safe_polyline_resample(
        points: np.ndarray, count: int
    ) -> Tuple[np.ndarray, float]:
        """Sample a polyline while retaining every collision-tested corner.

        Ordinary arc-length interpolation can put two consecutive commands on
        opposite sides of a corner.  The actuator executes their chord, which
        may cut the obstacle even though both endpoints lie on a safe
        polyline.  Allocating at least one interval to every segment preserves
        each waypoint exactly and makes every executed chord a subset of one
        already certified segment.
        """

        points = BatchedAStarSplinePlanner._deduplicate(points)
        segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total = float(segment.sum())
        if not math.isfinite(total) or total <= 1.0e-10:
            raise AStarPlanError("planned fallback polyline has zero length")
        # Keep individual translational increments below 1.9 mm.  This is
        # stricter than the task's 2 mm lateral and 3 mm axial limits.
        intervals = np.maximum(1, np.ceil(segment / 0.0019).astype(np.int64))
        required = int(intervals.sum())
        if required > count:
            raise AStarPlanError(
                f"collision-free polyline needs {required} commands, "
                f"but only {count} are available"
            )
        remaining = int(count - required)
        if remaining:
            ideal = remaining * segment / total
            extra = np.floor(ideal).astype(np.int64)
            intervals += extra
            remainder = remaining - int(extra.sum())
            if remainder:
                priority = np.argsort(-(ideal - extra), kind="stable")
                intervals[priority[:remainder]] += 1
        samples: List[np.ndarray] = []
        for index, amount in enumerate(intervals.tolist()):
            fraction = (
                np.arange(1, amount + 1, dtype=np.float64) / float(amount)
            )
            samples.append(
                points[index]
                + fraction[:, None] * (points[index + 1] - points[index])
            )
        result = np.concatenate(samples, axis=0)
        if len(result) != count:
            raise RuntimeError("polyline time allocation did not fill its horizon")
        result[-1] = points[-1]
        return result, total

    def _execution_chords_dense(
        self, continuous_start: np.ndarray, sampled: np.ndarray
    ) -> np.ndarray:
        executed = np.concatenate(
            (
                np.asarray(continuous_start, dtype=np.float64).reshape(1, 2),
                np.asarray(sampled, dtype=np.float64).reshape(-1, 2),
            ),
            axis=0,
        )
        return self._polyline_dense(executed)

    def _execution_chords_free(
        self, continuous_start: np.ndarray, sampled: np.ndarray
    ) -> bool:
        dense_chords = self._execution_chords_dense(
            continuous_start, sampled
        )
        return bool(self._points_free(dense_chords).all())

    def _transport_plan(
        self, continuous_start: np.ndarray
    ) -> Tuple[
        np.ndarray,
        float,
        float,
        bool,
        bool,
        int,
        Tuple[int, int],
    ]:
        start = np.asarray(continuous_start, dtype=np.float64).reshape(2)
        if not bool(self._points_free(start[None, :])[0]):
            raise AStarPlanError(
                f"continuous A* start {start.tolist()} lies in the C-obstacle"
            )
        start_ij = self._nearest_free_grid_cell(start, require_visible=True)
        if start_ij is None:
            raise AStarPlanError(
                f"continuous A* start {start.tolist()} has no visible free grid cell"
            )
        cached = self._grid_plan_cache.get(start_ij)
        cache_hit = cached is not None
        if cached is None:
            cached = self._astar(start_ij)
            self._grid_plan_cache[start_ij] = cached
            self._cache_misses += 1
        else:
            self._cache_hits += 1

        raw = np.concatenate(
            (
                start[None, :],
                cached.points_xy,
                self._canonical_stage_xy[None, :],
            ),
            axis=0,
        )
        waypoints = self._line_of_sight_shortcut(raw)
        spline_fallback = False
        try:
            dense = self._natural_cubic_dense(waypoints)
            if not bool(self._points_free(dense).all()):
                raise ValueError("cubic spline left certified free space")
            sampled, path_length = self._arc_length_resample(
                dense, TRANSPORT_STEPS
            )
            executed_dense = self._execution_chords_dense(start, sampled)
            if not bool(self._points_free(executed_dense).all()):
                raise ValueError("time-discretised spline chord cut free space")
        except (ValueError, np.linalg.LinAlgError):
            # LOS segments were each densely certified, so this is a genuine
            # collision-free fallback rather than a hidden straight-line or
            # expert-controller substitution.
            dense = self._polyline_dense(waypoints)
            if not bool(self._points_free(dense).all()):
                raise AStarPlanError("collision-free polyline fallback failed")
            sampled, path_length = self._safe_polyline_resample(
                waypoints, TRANSPORT_STEPS
            )
            executed_dense = self._execution_chords_dense(start, sampled)
            if not bool(self._points_free(executed_dense).all()):
                raise AStarPlanError(
                    "executed collision-free polyline chords failed validation"
                )
            spline_fallback = True
        if not bool(self._points_free(sampled).all()):
            raise AStarPlanError("time-resampled A* path is not collision-free")
        minimum_clearance = float(
            self._obstacle_clearance(executed_dense).min()
        )
        return (
            sampled,
            path_length,
            minimum_clearance,
            spline_fallback,
            cache_hit,
            cached.expanded,
            start_ij,
        )

    # ------------------------------------------------------------------
    # Public batched planner contract
    # ------------------------------------------------------------------
    def _masked_rows(self, value, ids, count: int, name: str):
        tensor = value
        if tensor.shape[0] == self.worlds:
            return tensor.index_select(0, ids)
        if tensor.shape[0] == count:
            return tensor
        raise ValueError(
            f"{name} leading dimension must be worlds ({self.worlds}) "
            f"or selected rows ({count}), got {tensor.shape[0]}"
        )

    def reset(
        self,
        mask,
        *,
        start_command: CommandBatch,
        goal_command: CommandBatch,
        socket_mouth,
        socket_axis,
    ) -> None:
        """Plan every selected world and upload all hot-loop tensors.

        Planning is atomic for the selected mask: if any world has no valid
        path, all selected ``plan_found`` flags are cleared and an explicit
        :class:`AStarPlanError` is raised.  Previously prepared, unselected
        worlds are not modified.
        """

        torch = _torch()
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if tuple(mask.shape) != (self.worlds,):
            raise ValueError(f"mask must have shape {(self.worlds,)}")
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        count = int(ids.numel())
        if count == 0:
            return
        start_tip = self._masked_rows(start_command.tip, ids, count, "start tip")
        start_rotation = self._masked_rows(
            start_command.rotation, ids, count, "start rotation"
        )
        goal_tip = self._masked_rows(goal_command.tip, ids, count, "goal tip")
        goal_rotation = self._masked_rows(
            goal_command.rotation, ids, count, "goal rotation"
        )
        mouth = self._masked_rows(socket_mouth, ids, count, "socket mouth")
        axis = self._masked_rows(socket_axis, ids, count, "socket axis")
        if mouth.shape[1] != 3 or axis.shape[1] != 3:
            raise ValueError("socket_mouth and socket_axis must end in xyz")

        # Reset-time synchronization is intentional.  No host operation is
        # performed by command(), which is the per-control-tick hot path.
        start_tip_cpu = start_tip.detach().to("cpu", torch.float64).numpy()
        goal_tip_cpu = goal_tip.detach().to("cpu", torch.float64).numpy()
        mouth_cpu = mouth.detach().to("cpu", torch.float64).numpy()
        axis_cpu = axis.detach().to("cpu", torch.float64).numpy()
        plan_tip_cpu = np.empty((count, DIRECT_ALIGN_STEPS, 3), dtype=np.float32)
        path_length = np.empty(count, dtype=np.float32)
        minimum_clearance = np.empty(count, dtype=np.float32)
        latency = np.empty(count, dtype=np.float32)
        fallback = np.zeros(count, dtype=np.bool_)
        cache_hit = np.zeros(count, dtype=np.bool_)
        expanded = np.zeros(count, dtype=np.int64)
        start_flat = np.full(count, -1, dtype=np.int64)

        try:
            for local in range(count):
                begin = time.perf_counter()
                radial = mouth_cpu[local, :2]
                radial_norm = float(np.linalg.norm(radial))
                if not math.isfinite(radial_norm) or abs(
                    radial_norm - self.task.socket_mouth_orbit_radius_m
                ) > 2.0e-5:
                    raise AStarPlanError(
                        "socket mouth must lie on the configured 20 mm orbit"
                    )
                inward = radial / radial_norm
                tangent = np.asarray((-inward[1], inward[0]), dtype=np.float64)
                canonical_start = np.asarray(
                    (
                        float(np.dot(start_tip_cpu[local, :2], inward)),
                        float(np.dot(start_tip_cpu[local, :2], tangent)),
                    )
                )
                (
                    transport_xy,
                    path_length[local],
                    minimum_clearance[local],
                    fallback[local],
                    cache_hit[local],
                    expanded[local],
                    start_ij,
                ) = self._transport_plan(canonical_start)
                start_flat[local] = self._flatten(*start_ij)

                world_transport_xy = (
                    transport_xy[:, :1] * inward[None, :]
                    + transport_xy[:, 1:] * tangent[None, :]
                )
                plan_tip_cpu[local, :TRANSPORT_STEPS, :2] = world_transport_xy
                plan_tip_cpu[local, :TRANSPORT_STEPS, 2] = start_tip_cpu[local, 2]

                stage = mouth_cpu[local] - self.staging_standoff_m * axis_cpu[local]
                transition_fraction = _minimum_jerk(
                    np.arange(1, ROTATE_RAISE_STEPS + 1, dtype=np.float64)
                    / float(ROTATE_RAISE_STEPS)
                )
                transition_start = np.asarray(
                    (stage[0], stage[1], start_tip_cpu[local, 2]),
                    dtype=np.float64,
                )
                transition = transition_start[None, :] + transition_fraction[:, None] * (
                    stage - transition_start
                )[None, :]
                transition[-1] = stage
                offset = TRANSPORT_STEPS
                plan_tip_cpu[
                    local, offset : offset + ROTATE_RAISE_STEPS
                ] = transition.astype(np.float32)

                dock_fraction = _minimum_jerk(
                    np.arange(1, DOCK_STEPS + 1, dtype=np.float64)
                    / float(DOCK_STEPS)
                )
                docking = stage[None, :] + dock_fraction[:, None] * (
                    goal_tip_cpu[local] - stage
                )[None, :]
                docking[-1] = goal_tip_cpu[local]
                plan_tip_cpu[local, offset + ROTATE_RAISE_STEPS :] = (
                    docking.astype(np.float32)
                )
                latency[local] = float(time.perf_counter() - begin)
        except BaseException as exc:
            self._plan_found.index_fill_(0, ids, False)
            if isinstance(exc, AStarPlanError):
                raise
            raise AStarPlanError(f"A* reset planning failed: {exc}") from exc

        tip_tensor = torch.as_tensor(
            plan_tip_cpu, device=self.device, dtype=self.dtype
        )
        self._plan_tip.index_copy_(0, ids, tip_tensor)

        start_q = matrix_to_quaternion(start_rotation)
        goal_q = matrix_to_quaternion(goal_rotation)
        transition_fraction_t = torch.as_tensor(
            _minimum_jerk(
                np.arange(1, ROTATE_RAISE_STEPS + 1, dtype=np.float32)
                / float(ROTATE_RAISE_STEPS)
            ),
            device=self.device,
            dtype=self.dtype,
        )
        transition_q = quaternion_slerp(
            start_q[:, None, :],
            goal_q[:, None, :],
            transition_fraction_t[None, :, None],
        )
        rotation = torch.empty(
            (count, DIRECT_ALIGN_STEPS, 3, 3),
            device=self.device,
            dtype=self.dtype,
        )
        rotation[:, :TRANSPORT_STEPS] = start_rotation[:, None]
        rotation[
            :, TRANSPORT_STEPS : TRANSPORT_STEPS + ROTATE_RAISE_STEPS
        ] = quaternion_to_matrix(transition_q)
        rotation[:, TRANSPORT_STEPS + ROTATE_RAISE_STEPS :] = (
            goal_rotation[:, None]
        )
        self._plan_rotation.index_copy_(0, ids, rotation)

        self._plan_found.index_fill_(0, ids, True)
        self._spline_fallback.index_copy_(
            0,
            ids,
            torch.as_tensor(fallback, device=self.device, dtype=torch.bool),
        )
        self._cache_hit.index_copy_(
            0,
            ids,
            torch.as_tensor(cache_hit, device=self.device, dtype=torch.bool),
        )
        self._planning_latency_s.index_copy_(
            0,
            ids,
            torch.as_tensor(latency, device=self.device, dtype=self.dtype),
        )
        self._path_length_m.index_copy_(
            0,
            ids,
            torch.as_tensor(path_length, device=self.device, dtype=self.dtype),
        )
        self._minimum_obstacle_clearance_m.index_copy_(
            0,
            ids,
            torch.as_tensor(
                minimum_clearance, device=self.device, dtype=self.dtype
            ),
        )
        self._astar_expanded.index_copy_(
            0,
            ids,
            torch.as_tensor(expanded, device=self.device, dtype=torch.long),
        )
        self._start_grid_flat.index_copy_(
            0,
            ids,
            torch.as_tensor(start_flat, device=self.device, dtype=torch.long),
        )
        self._total_reset_plans += count
        self._total_planning_latency_s += float(np.sum(latency, dtype=np.float64))
        self._total_spline_fallbacks += int(fallback.sum())

    def command(self, anchor: CommandBatch, *, next_step) -> CommandBatch:
        """Return the prepared nominal command using Torch operations only."""

        torch = _torch()
        step = torch.as_tensor(next_step, device=self.device, dtype=torch.long)
        if step.ndim == 0:
            step = step.expand(self.worlds)
        index = (
            step - (SCRIPTED_PREFIX_STEPS + 1)
        ).clamp(0, DIRECT_ALIGN_STEPS - 1)
        planned_tip = self._plan_tip[self._row, index]
        planned_rotation = self._plan_rotation[self._row, index]
        align = anchor.phase == int(TaskPhase.ALIGN)
        return CommandBatch(
            tip=torch.where(align[:, None], planned_tip, anchor.tip),
            rotation=torch.where(
                align[:, None, None], planned_rotation, anchor.rotation
            ),
            twist=anchor.twist,
            material=anchor.material,
            distance=anchor.distance,
            phase=anchor.phase,
            engaged=anchor.engaged,
        )

    def diagnostics(self) -> Dict[str, object]:
        denominator = max(self._total_reset_plans, 1)
        return {
            "plan_found": self._plan_found.detach().clone(),
            "spline_fallback": self._spline_fallback.detach().clone(),
            "cache_hit": self._cache_hit.detach().clone(),
            "planning_latency_s": self._planning_latency_s.detach().clone(),
            "planning_latency_ms": (
                1000.0 * self._planning_latency_s.detach().clone()
            ),
            "path_length_m": self._path_length_m.detach().clone(),
            "minimum_obstacle_clearance_m": (
                self._minimum_obstacle_clearance_m.detach().clone()
            ),
            "astar_expanded": self._astar_expanded.detach().clone(),
            "start_grid_flat": self._start_grid_flat.detach().clone(),
            "cache_hits": int(self._cache_hits),
            "cache_misses": int(self._cache_misses),
            "unique_plans": int(len(self._grid_plan_cache)),
            "total_reset_plans": int(self._total_reset_plans),
            "spline_fallbacks": int(self._total_spline_fallbacks),
            "mean_planning_latency_s": (
                self._total_planning_latency_s / denominator
            ),
            "grid_resolution_m": float(self.grid_resolution_m),
            "staging_standoff_m": float(self.staging_standoff_m),
            "transport_steps": TRANSPORT_STEPS,
            "rotate_raise_steps": ROTATE_RAISE_STEPS,
            "dock_steps": DOCK_STEPS,
        }

    def provenance(self) -> Dict[str, object]:
        """Return stable, JSON-serialisable planner parameters and semantics."""

        return {
            "schema": "screw-astar-spline-planner-v1",
            "algorithm": "cached-canonical-astar-natural-cubic",
            "workspace_radius_m": float(self.workspace_radius_m),
            "socket": {
                "table_angle_deg": 45.0,
                "mouth_orbit_radius_m": float(
                    self.task.socket_mouth_orbit_radius_m
                ),
            },
            "grid": {
                "resolution_m": float(self.grid_resolution_m),
                "connectivity": 8,
                "diagonal_corner_cutting": False,
            },
            "obstacle": {
                "model": "socket-projected-obb-minkowski-upright-screw-head",
                "canonical_center_x_m": float(self._obstacle_center_x),
                "canonical_half_x_m": float(self._obstacle_half_x),
                "canonical_half_y_m": float(self._obstacle_half_y),
                "screw_head_radius_m": float(
                    self.task.screw_head_radius_m
                ),
                "clearance_m": float(self.task.spawn_clearance_m),
                "total_inflation_m": float(
                    self.task.screw_head_radius_m
                    + self.task.spawn_clearance_m
                ),
            },
            "search": {
                "algorithm": "A*",
                "heuristic": "exact reverse shortest-path distance",
                "tie_break": "lower heuristic then insertion order",
                "cache_key": "canonical start grid cell",
            },
            "spline": {
                "kind": "natural cubic",
                "line_of_sight_shortcut": True,
                "validation_spacing_m": SPLINE_VALIDATION_SPACING_M,
                "executed_chord_validation": True,
                "fallback": "collision-tested waypoint-preserving polyline",
            },
            "schedule": {
                "transport_steps": TRANSPORT_STEPS,
                "rotate_raise_steps": ROTATE_RAISE_STEPS,
                "dock_steps": DOCK_STEPS,
                "align_steps": DIRECT_ALIGN_STEPS,
                "staging_standoff_m": float(self.staging_standoff_m),
                "time_scaling": "minimum jerk for spline/rotation/docking",
            },
        }

    def parameters(self) -> Dict[str, object]:
        """Compatibility alias for consumers that call parameters()."""

        return self.provenance()

    def state_dict(self) -> Dict[str, object]:
        return {
            "plan_tip": self._plan_tip.detach().clone(),
            "plan_rotation": self._plan_rotation.detach().clone(),
            "plan_found": self._plan_found.detach().clone(),
            "spline_fallback": self._spline_fallback.detach().clone(),
            "cache_hit": self._cache_hit.detach().clone(),
            "planning_latency_s": self._planning_latency_s.detach().clone(),
            "path_length_m": self._path_length_m.detach().clone(),
            "minimum_obstacle_clearance_m": (
                self._minimum_obstacle_clearance_m.detach().clone()
            ),
            "astar_expanded": self._astar_expanded.detach().clone(),
            "start_grid_flat": self._start_grid_flat.detach().clone(),
            "counters": {
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "total_reset_plans": self._total_reset_plans,
                "total_planning_latency_s": self._total_planning_latency_s,
                "total_spline_fallbacks": self._total_spline_fallbacks,
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        tensor_names = (
            "plan_tip",
            "plan_rotation",
            "plan_found",
            "spline_fallback",
            "cache_hit",
            "planning_latency_s",
            "path_length_m",
            "minimum_obstacle_clearance_m",
            "astar_expanded",
            "start_grid_flat",
        )
        for name in tensor_names:
            source = state.get(name)
            target = getattr(self, f"_{name}")
            if source is None or tuple(source.shape) != tuple(target.shape):
                raise ValueError(f"A* planner state {name!r} shape mismatch")
            target.copy_(source.to(device=self.device, dtype=target.dtype))
        counters = state.get("counters", {})
        self._cache_hits = int(counters.get("cache_hits", 0))
        self._cache_misses = int(counters.get("cache_misses", 0))
        self._total_reset_plans = int(counters.get("total_reset_plans", 0))
        self._total_planning_latency_s = float(
            counters.get("total_planning_latency_s", 0.0)
        )
        self._total_spline_fallbacks = int(
            counters.get("total_spline_fallbacks", 0)
        )

    def close(self) -> None:
        self._grid_plan_cache.clear()


__all__ = [
    "AStarPlanError",
    "BatchedAStarSplinePlanner",
    "DOCK_STEPS",
    "GRID_RESOLUTION_M",
    "ROTATE_RAISE_STEPS",
    "STAGING_STANDOFF_M",
    "TRANSPORT_STEPS",
]
