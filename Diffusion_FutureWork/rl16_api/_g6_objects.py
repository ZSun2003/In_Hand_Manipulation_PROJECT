"""Object geometry and adaptive parallelogram pre-grasp planning.

The seven XML objects are mirrored here with inexpensive distance queries.
The four YCB fruits use the same generated 5 mm medial capsule manifest as
MuJoCo; the remaining objects use direct analytic geometry.  Regular objects
retain strict world-XY-parallel pre-grasp/contact cages.  Fruit candidates
preserve each hard tip's own vertical coordinate; only their XY parallelogram
formation is required.

The YCB banana, pear, strawberry, and orange triangle meshes are used for
rendering.
The generated capsule proxies are shared by MuJoCo, IK-only, and IK+residual,
so planned contacts and simulated contacts stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from _g6_kinematics import FINGERS, GripperIK, quat_to_rot


HARD_TIP_RADIUS_M = 0.016
DEFAULT_PREGRASP_CLEARANCE_M = 0.012
OBJECT_MASS_KG = 0.200
NOMINAL_PINCH_TIP_FORCE_N = 3.0
STANDARD_GRAVITY_M_S2 = 9.81
MAX_PLANNED_NORMAL_SUPPORT_FRACTION = 0.20
MAX_PLANNED_PAIR_NORMAL_DOT = -0.50
MAX_GRASP_PLANE_BELOW_COM_M = 0.002
FRUIT_GRASP_PLANE_HEIGHT_FRACTIONS = (0.45, 0.50, 0.55, 0.60)
FRUIT_GRASP_PLANE_YAW_RAD = tuple(
    math.radians(value) for value in range(0, 180, 15)
)
IRREGULAR_RADIAL_LIMIT_M = 0.005
PLATE_HALF_SIZE_M = np.asarray((0.025, 0.025, 0.005), dtype=np.float64)
MAX_PAIR_RADIAL_ASYMMETRY_M = 0.004

# Retained for backwards-compatible imports by old notebooks.  The active YCB
# fruit geometry no longer uses the synthetic +/-5 mm Fourier perturbation.
PEAR_IRREGULAR_PHASE_RAD = math.radians(11.0)
BANANA_IRREGULAR_PHASE_RAD = 0.5 * math.pi

FRUIT_CAPSULE_MANIFEST_PATH = (
    Path(__file__).resolve().with_name("fruit_capsules_5mm.json")
)
FRUIT_CAPSULE_SCHEMA_VERSION = "ycb-fruit-capsules-5mm-v1"


def _load_fruit_capsule_manifest() -> tuple[
    dict[str, object],
    dict[str, np.ndarray],
    str,
]:
    payload = json.loads(
        FRUIT_CAPSULE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if payload.get("schema_version") != FRUIT_CAPSULE_SCHEMA_VERSION:
        raise ValueError("unsupported fruit capsule manifest schema")
    maximum_spacing = float(payload["maximum_capsule_spacing_m"])
    if maximum_spacing > 0.005 + 1e-12:
        raise ValueError("fruit capsule manifest exceeds 5 mm spacing")
    arrays: dict[str, np.ndarray] = {}
    for name in ("banana", "pear", "strawberry", "orange"):
        record = payload["objects"][name]
        source_path = (
            FRUIT_CAPSULE_MANIFEST_PATH.parent
            / str(record["source_path"])
        ).resolve()
        try:
            source_path.relative_to(
                FRUIT_CAPSULE_MANIFEST_PATH.parent.resolve()
            )
        except ValueError as exc:
            raise ValueError(
                f"{name} fruit capsule source escaped the GRAB directory"
            ) from exc
        source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_digest != str(record["source_sha256"]):
            raise ValueError(f"{name} YCB source hash mismatch")
        if (
            float(record["surface_vertices_within_5mm_fraction"])
            < 1.0 - 1e-12
        ):
            raise ValueError(
                f"{name} capsule proxy exceeds the 5 mm surface contract"
            )
        rows = [
            (
                *capsule["from_m"],
                *capsule["to_m"],
                capsule["radius_m"],
            )
            for capsule in record["capsules"]
        ]
        array = np.asarray(rows, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 7 or not len(array):
            raise ValueError(f"invalid {name} capsule records")
        if np.any(array[:, 6] <= 0.0):
            raise ValueError(f"non-positive {name} capsule radius")
        lengths = np.linalg.norm(array[:, 3:6] - array[:, :3], axis=1)
        measured_maximum_spacing = float(np.max(lengths))
        if measured_maximum_spacing > 0.005 + 1e-8:
            raise ValueError(f"{name} capsule centre spacing exceeds 5 mm")
        if not math.isclose(
            measured_maximum_spacing,
            float(record["maximum_center_spacing_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{name} capsule spacing metadata mismatch")
        if int(record["capsule_count"]) != len(array):
            raise ValueError(f"{name} capsule count mismatch")
        array.setflags(write=False)
        arrays[name] = array
    digest = hashlib.sha256(
        FRUIT_CAPSULE_MANIFEST_PATH.read_bytes()
    ).hexdigest()
    return payload, arrays, digest


(
    FRUIT_CAPSULE_MANIFEST,
    FRUIT_CAPSULES_BY_NAME,
    FRUIT_CAPSULE_MANIFEST_SHA256,
) = _load_fruit_capsule_manifest()


def irregular_radius_delta_m(
    angle_rad: float,
    *,
    phase_rad: float = 0.0,
) -> float:
    """Return a smooth, periodic radial perturbation bounded by +/-5 mm."""
    angle = float(angle_rad) + float(phase_rad)
    profile = (
        0.70 * math.cos(2.0 * angle)
        + 0.20 * math.cos(6.0 * angle)
        + 0.10 * math.cos(10.0 * angle)
    )
    value = IRREGULAR_RADIAL_LIMIT_M * profile
    return float(
        np.clip(
            value,
            -IRREGULAR_RADIAL_LIMIT_M,
            IRREGULAR_RADIAL_LIMIT_M,
        )
    )


# Generated 5 mm medial capsule chains.  Every final hard-tip centre is
# derived from the selected capsule surface and its recomputed normal.
BANANA_CAPSULES = FRUIT_CAPSULES_BY_NAME["banana"]
PEAR_CAPSULES = FRUIT_CAPSULES_BY_NAME["pear"]
STRAWBERRY_CAPSULES = FRUIT_CAPSULES_BY_NAME["strawberry"]
ORANGE_CAPSULES = FRUIT_CAPSULES_BY_NAME["orange"]
BANANA_SEGMENTS = BANANA_CAPSULES[:, :6].reshape((-1, 2, 3))
BANANA_CAPSULE_RADII_M = BANANA_CAPSULES[:, 6]
BANANA_RADIUS_M = float(np.max(BANANA_CAPSULE_RADII_M))
BANANA_COLLISION_CENTER_M = np.mean(
    0.5 * (BANANA_CAPSULES[:, :3] + BANANA_CAPSULES[:, 3:6]),
    axis=0,
)
BANANA_COLLISION_RADII_M = 0.5 * np.ptp(
    np.vstack((BANANA_CAPSULES[:, :3], BANANA_CAPSULES[:, 3:6])),
    axis=0,
)

# YCB 016 pear: two fitted overlapping ellipsoids. Candidate-section ray
# errors against the source triangle mesh stay below about 4.3 mm.
PEAR_ELLIPSOID_CENTERS_M = np.asarray(
    ((-0.0005, -0.009, 0.033), (-0.002, 0.015, 0.025)),
    dtype=np.float64,
)
PEAR_ELLIPSOID_RADII_SET_M = np.asarray(
    ((0.033, 0.034, 0.033), (0.025, 0.031, 0.020)),
    dtype=np.float64,
)
PEAR_ELLIPSOID_CENTER_M = PEAR_ELLIPSOID_CENTERS_M[0]
PEAR_ELLIPSOID_RADII_M = PEAR_ELLIPSOID_RADII_SET_M[0]
PEAR_TOTAL_HEIGHT_M = 0.065663
PEAR_JOIN_Z_M = PEAR_ELLIPSOID_CENTER_M[2]

# YCB 012 strawberry: COM-centred, bottom-aligned visual mesh with a fitted
# ellipsoid and a small flat support patch for deterministic table settling.
STRAWBERRY_ELLIPSOID_CENTERS_M = np.asarray(
    ((0.0005, 0.0015, 0.0228), (-0.014, -0.014, 0.014)),
    dtype=np.float64,
)
STRAWBERRY_ELLIPSOID_RADII_M = np.asarray(
    ((0.0225, 0.0215, 0.0228), (0.009, 0.009, 0.014)),
    dtype=np.float64,
)
STRAWBERRY_SUPPORT_CENTER_M = np.asarray(
    (0.0, 0.0, 0.002), dtype=np.float64
)
STRAWBERRY_SUPPORT_RADIUS_M = 0.010
STRAWBERRY_SUPPORT_HALF_HEIGHT_M = 0.002
STRAWBERRY_TOTAL_HEIGHT_M = 0.045675
ORANGE_TOTAL_HEIGHT_M = 0.071352

CONE_RADIUS_M = 0.040
CONE_HEIGHT_M = 0.120

class ObjectKind(str, Enum):
    CUBE = "cube"
    CYLINDER = "cylinder"
    BANANA = "banana"
    PEAR = "pear"
    STRAWBERRY = "strawberry"
    ORANGE = "orange"
    CONE = "cone"
    PLATE = "plate"


FRUIT_KINDS = frozenset(
    (
        ObjectKind.BANANA,
        ObjectKind.PEAR,
        ObjectKind.STRAWBERRY,
        ObjectKind.ORANGE,
    )
)


OBJECT_KIND_ORDER = (
    ObjectKind.CUBE,
    ObjectKind.CYLINDER,
    ObjectKind.BANANA,
    ObjectKind.PEAR,
    ObjectKind.STRAWBERRY,
    ObjectKind.ORANGE,
    ObjectKind.CONE,
    ObjectKind.PLATE,
)

DIRECT_GRASP_PLANE_HEIGHT_FRACTION = {
    # Four contacts are the centres of the cube's four vertical side faces.
    ObjectKind.CUBE: 0.5,
    ObjectKind.CYLINDER: 0.5,
    ObjectKind.CONE: 1.0 / 3.0,
}
DIRECT_GRASP_PLANE_YAW_RAD = {
    ObjectKind.CUBE: 0.0,
    ObjectKind.CYLINDER: 0.0,
    ObjectKind.CONE: 0.0,
}


def _box_sdf(point: np.ndarray, half_size: np.ndarray) -> float:
    q = np.abs(point) - half_size
    return float(
        np.linalg.norm(np.maximum(q, 0.0))
        + min(float(np.max(q)), 0.0)
    )


def _cylinder_sdf(point: np.ndarray, radius: float, half_height: float) -> float:
    q = np.array(
        [math.hypot(float(point[0]), float(point[1])) - radius,
         abs(float(point[2])) - half_height],
        dtype=np.float64,
    )
    return float(
        np.linalg.norm(np.maximum(q, 0.0))
        + min(float(np.max(q)), 0.0)
    )


def _capsule_sdf(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
) -> float:
    segment = np.asarray(end, dtype=np.float64) - np.asarray(
        start, dtype=np.float64
    )
    length_squared = float(segment @ segment)
    if length_squared <= 1e-16:
        closest = np.asarray(start, dtype=np.float64)
    else:
        fraction = float(
            np.clip(
                (np.asarray(point, dtype=np.float64) - start) @ segment
                / length_squared,
                0.0,
                1.0,
            )
        )
        closest = np.asarray(start, dtype=np.float64) + fraction * segment
    return float(np.linalg.norm(np.asarray(point) - closest) - radius)


def _ellipsoid_sdf(
    point: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
) -> float:
    """Euclidean signed distance to an axis-aligned ellipsoid.

    The exterior branch solves the closest-point Lagrange multiplier exactly.
    The inexpensive interior branch only supplies a negative bracket value;
    all fingertip offset roots are evaluated on the exact exterior branch.
    """
    value = np.asarray(point, dtype=np.float64) - np.asarray(
        center, dtype=np.float64
    )
    axes = np.asarray(radii, dtype=np.float64)
    normalized_squared = float(np.sum((value / axes) ** 2))
    if normalized_squared <= 1.0:
        return float(
            (math.sqrt(max(normalized_squared, 0.0)) - 1.0)
            * np.min(axes)
        )

    axes_squared = axes * axes
    weighted = axes_squared * value * value

    def equation(lagrange: float) -> float:
        return float(
            np.sum(weighted / (lagrange + axes_squared) ** 2) - 1.0
        )

    low = 0.0
    high = float(max(np.linalg.norm(value) * np.max(axes), 1e-9))
    while equation(high) > 0.0:
        high *= 2.0
    for _ in range(36):
        middle = 0.5 * (low + high)
        if equation(middle) > 0.0:
            low = middle
        else:
            high = middle
    lagrange = 0.5 * (low + high)
    closest = axes_squared * value / (lagrange + axes_squared)
    return float(np.linalg.norm(value - closest))


def _banana_segment_signed_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
) -> float:
    return _capsule_sdf(
        point,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
        float(radius),
    )


def _fruit_capsule_signed_distance(
    point: np.ndarray,
    capsules: np.ndarray,
) -> float:
    value = np.asarray(point, dtype=np.float64).reshape(1, 3)
    starts = capsules[:, :3]
    segments = capsules[:, 3:6] - starts
    length_squared = np.sum(segments * segments, axis=1)
    fractions = np.clip(
        np.sum((value - starts) * segments, axis=1)
        / np.maximum(length_squared, 1e-18),
        0.0,
        1.0,
    )
    closest = starts + fractions[:, None] * segments
    distances = (
        np.linalg.norm(value - closest, axis=1) - capsules[:, 6]
    )
    return float(np.min(distances))


def _cone_signed_distance(point: np.ndarray) -> float:
    """Exact Euclidean SDF of the closed finite 80 x 120 mm cone.

    Rotational symmetry reduces the query to the physical base and sloped
    edges of the cone's ``(radius, z)`` cross-section.  The axis is not a
    boundary of the revolved solid and is therefore intentionally excluded
    from the distance calculation.
    """
    value = np.asarray(point, dtype=np.float64).reshape(3)
    query = np.asarray(
        (math.hypot(float(value[0]), float(value[1])), float(value[2])),
        dtype=np.float64,
    )
    base_start = np.asarray((0.0, 0.0), dtype=np.float64)
    base_end = np.asarray((CONE_RADIUS_M, 0.0), dtype=np.float64)
    side_start = base_end
    side_end = np.asarray((0.0, CONE_HEIGHT_M), dtype=np.float64)

    def segment_distance(
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        segment = end - start
        fraction = float(
            np.clip(
                (query - start) @ segment
                / max(float(segment @ segment), 1e-18),
                0.0,
                1.0,
            )
        )
        return float(np.linalg.norm(query - (start + fraction * segment)))

    distance = min(
        segment_distance(base_start, base_end),
        segment_distance(side_start, side_end),
    )
    radius, z_value = map(float, query)
    inside = (
        0.0 <= z_value <= CONE_HEIGHT_M
        and radius
        <= CONE_RADIUS_M
        * (1.0 - z_value / CONE_HEIGHT_M)
        + 1e-15
    )
    return -distance if inside else distance


@dataclass(frozen=True)
class ObjectSpec:
    kind: ObjectKind
    body_name: str
    joint_name: str
    geom_names: tuple[str, ...]
    reference_site_name: str
    rest_body_z_m: float
    local_bottom_z_m: float
    local_top_z_m: float
    local_com_m: tuple[float, float, float]
    half_extent_xy_m: tuple[float, float]
    mass_kg: float = OBJECT_MASS_KG

    @property
    def height_m(self) -> float:
        return self.local_top_z_m - self.local_bottom_z_m

    @property
    def kind_index(self) -> int:
        return OBJECT_KIND_ORDER.index(self.kind)

    def one_hot(self) -> np.ndarray:
        value = np.zeros(len(OBJECT_KIND_ORDER), dtype=np.float32)
        value[self.kind_index] = 1.0
        return value

    def signed_distance(self, point_local: Sequence[float]) -> float:
        point = np.asarray(point_local, dtype=np.float64).reshape(3)
        if self.kind is ObjectKind.CUBE:
            return _box_sdf(point, np.array((0.025, 0.025, 0.025)))
        if self.kind is ObjectKind.CYLINDER:
            return _cylinder_sdf(point, 0.025, 0.050)
        if self.kind is ObjectKind.BANANA:
            return _fruit_capsule_signed_distance(
                point,
                BANANA_CAPSULES,
            )
        if self.kind is ObjectKind.PEAR:
            return _fruit_capsule_signed_distance(
                point,
                PEAR_CAPSULES,
            )
        if self.kind is ObjectKind.STRAWBERRY:
            return _fruit_capsule_signed_distance(
                point,
                STRAWBERRY_CAPSULES,
            )
        if self.kind is ObjectKind.ORANGE:
            return _fruit_capsule_signed_distance(
                point,
                ORANGE_CAPSULES,
            )
        if self.kind is ObjectKind.CONE:
            return _cone_signed_distance(point)
        if self.kind is ObjectKind.PLATE:
            return _box_sdf(point, PLATE_HALF_SIZE_M)
        raise ValueError(f"unsupported object kind: {self.kind}")

    def _pear_radius(self, z_value: float) -> float:
        relative_z = (
            float(z_value) - float(PEAR_ELLIPSOID_CENTER_M[2])
        ) / float(PEAR_ELLIPSOID_RADII_M[2])
        if abs(relative_z) > 1.0:
            return -1.0
        return float(
            max(PEAR_ELLIPSOID_RADII_M[:2])
            * math.sqrt(max(0.0, 1.0 - relative_z**2))
        )

    def _pear_surface_radius(
        self,
        z_value: float,
        angle_rad: float,
    ) -> float:
        base_radius = self._pear_radius(z_value)
        if base_radius < 0.0:
            return base_radius
        # Direction-dependent cross-section of the YCB pear ellipsoid proxy.
        cosine = math.cos(float(angle_rad))
        sine = math.sin(float(angle_rad))
        direction_radius = 1.0 / math.sqrt(
            (cosine / float(PEAR_ELLIPSOID_RADII_M[0])) ** 2
            + (sine / float(PEAR_ELLIPSOID_RADII_M[1])) ** 2
        )
        scale = base_radius / float(max(PEAR_ELLIPSOID_RADII_M[:2]))
        return max(1e-4, direction_radius * scale)

    def _pear_signed_distance(self, point: np.ndarray) -> float:
        return _ellipsoid_sdf(
            point,
            PEAR_ELLIPSOID_CENTER_M,
            PEAR_ELLIPSOID_RADII_M,
        )

    def contains(self, point_local: Sequence[float], tolerance: float = 0.0) -> bool:
        return self.signed_distance(point_local) <= float(tolerance)

    def surface_normal(self, point_local: Sequence[float]) -> np.ndarray:
        point = np.asarray(point_local, dtype=np.float64).reshape(3)
        epsilon = 2e-5
        gradient = np.empty(3, dtype=np.float64)
        for axis in range(3):
            step = np.zeros(3, dtype=np.float64)
            step[axis] = epsilon
            gradient[axis] = (
                self.signed_distance(point + step)
                - self.signed_distance(point - step)
            ) / (2.0 * epsilon)
        norm = float(np.linalg.norm(gradient))
        if norm < 1e-10:
            fallback = point - np.asarray(self.local_com_m, dtype=np.float64)
            norm = float(np.linalg.norm(fallback))
            return fallback / max(norm, 1e-12)
        return gradient / norm

    def ray_exit(
        self,
        origin_local: Sequence[float],
        direction_local: Sequence[float],
        *,
        max_distance_m: float = 0.25,
    ) -> Optional[tuple[float, np.ndarray]]:
        origin = np.asarray(origin_local, dtype=np.float64).reshape(3)
        direction = np.asarray(direction_local, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            raise ValueError("ray direction cannot be zero")
        direction /= norm
        if not self.contains(origin, tolerance=2e-6):
            return None

        low = 0.0
        high = 0.001
        while high <= max_distance_m and self.contains(
            origin + high * direction
        ):
            low = high
            high *= 1.5
        if high > max_distance_m:
            high = max_distance_m
            if self.contains(origin + high * direction):
                return None
        for _ in range(34):
            middle = 0.5 * (low + high)
            if self.contains(origin + middle * direction):
                low = middle
            else:
                high = middle
        distance = 0.5 * (low + high)
        point = origin + distance * direction
        return float(distance), point

    def height_at(self, x_local: float, y_local: float) -> Optional[float]:
        x_value = float(x_local)
        y_value = float(y_local)
        if self.kind is ObjectKind.CUBE:
            if abs(x_value) <= 0.025 and abs(y_value) <= 0.025:
                return 0.025
            return None
        if self.kind is ObjectKind.CYLINDER:
            if math.hypot(x_value, y_value) <= 0.025:
                return 0.050
            return None
        if self.kind is ObjectKind.PLATE:
            if abs(x_value) <= 0.025 and abs(y_value) <= 0.025:
                return 0.005
            return None
        if self.kind in FRUIT_KINDS:
            heights = np.linspace(
                self.local_bottom_z_m,
                self.local_top_z_m,
                401,
            )
            inside = np.asarray(
                [
                    self.contains((x_value, y_value, float(z_value)))
                    for z_value in heights
                ],
                dtype=bool,
            )
            valid_indices = np.flatnonzero(inside)
            if not valid_indices.size:
                return None
            last = int(valid_indices[-1])
            if last == len(heights) - 1:
                return float(heights[last])
            low = float(heights[last])
            high = float(heights[last + 1])
            for _ in range(24):
                middle = 0.5 * (low + high)
                if self.contains((x_value, y_value, middle)):
                    low = middle
                else:
                    high = middle
            return 0.5 * (low + high)
        if self.kind is ObjectKind.CONE:
            radius = math.hypot(x_value, y_value)
            if radius > CONE_RADIUS_M:
                return None
            return CONE_HEIGHT_M * (
                1.0 - radius / CONE_RADIUS_M
            )
        raise ValueError(f"unsupported object kind: {self.kind}")

    @lru_cache(maxsize=256)
    def candidate_xy_centers(
        self,
        z_local: float,
        *,
        limit: int = 4,
    ) -> tuple[tuple[float, float], ...]:
        """Return separated deep-interior points in a horizontal section."""
        preferred = np.asarray(self.local_com_m[:2], dtype=np.float64)
        if self.kind not in FRUIT_KINDS:
            return ((float(preferred[0]), float(preferred[1])),)

        z_value = float(z_local)
        if int(limit) <= 1 and self.contains(
            (preferred[0], preferred[1], z_value)
        ):
            return ((float(preferred[0]), float(preferred[1])),)
        extent_x, extent_y = self.half_extent_xy_m
        candidates: list[tuple[float, float, float]] = []
        if self.contains((preferred[0], preferred[1], z_value)):
            candidates.append(
                (
                    -self.signed_distance(
                        (preferred[0], preferred[1], z_value)
                    ),
                    float(preferred[0]),
                    float(preferred[1]),
                )
            )
        for y_value in np.linspace(-extent_y, extent_y, 23):
            for x_value in np.linspace(-extent_x, extent_x, 23):
                distance = self.signed_distance(
                    (float(x_value), float(y_value), z_value)
                )
                if distance <= -2e-4:
                    candidates.append(
                        (-distance, float(x_value), float(y_value))
                    )
        candidates.sort(reverse=True)
        selected: list[tuple[float, float]] = []
        if self.contains((preferred[0], preferred[1], z_value)):
            selected.append(
                (float(preferred[0]), float(preferred[1]))
            )
        minimum_spacing = max(
            0.006,
            0.18 * min(float(extent_x), float(extent_y)),
        )
        for _depth, x_value, y_value in candidates:
            if len(selected) >= int(limit):
                break
            if all(
                math.hypot(x_value - old_x, y_value - old_y)
                >= minimum_spacing
                for old_x, old_y in selected
            ):
                selected.append((x_value, y_value))
                if len(selected) >= int(limit):
                    break
        return tuple(selected)

    def height_map(
        self,
        size: int = 16,
        *,
        extent_m: float | Sequence[float] | None = None,
        height_normalization_m: float = PEAR_TOTAL_HEIGHT_M,
    ) -> np.ndarray:
        if size < 2:
            raise ValueError("height-map size must be at least two")
        if extent_m is None:
            extent_x = 2.20 * self.half_extent_xy_m[0]
            extent_y = 2.20 * self.half_extent_xy_m[1]
        elif np.isscalar(extent_m):
            extent_x = extent_y = float(extent_m)
        else:
            extent_value = np.asarray(extent_m, dtype=np.float64).reshape(2)
            extent_x, extent_y = map(float, extent_value)
        x_coordinates = np.linspace(-0.5 * extent_x, 0.5 * extent_x, size)
        y_coordinates = np.linspace(-0.5 * extent_y, 0.5 * extent_y, size)
        result = np.zeros((size, size), dtype=np.float32)
        for row, y_value in enumerate(y_coordinates):
            for column, x_value in enumerate(x_coordinates):
                top = self.height_at(float(x_value), float(y_value))
                if top is None:
                    continue
                normalized = (top - self.local_bottom_z_m) / max(
                    float(height_normalization_m), 1e-9
                )
                result[row, column] = float(np.clip(normalized, 0.0, 1.5))
        return result

    def directional_profiles(
        self,
        height_samples: int = 10,
        *,
        distance_normalization_m: float = 0.090,
    ) -> np.ndarray:
        if height_samples < 2:
            raise ValueError("height_samples must be at least two")
        heights = np.linspace(
            self.local_bottom_z_m + 1e-5,
            self.local_top_z_m - 1e-5,
            height_samples,
        )
        directions = {
            "U": np.array((0.0, 1.0, 0.0)),
            "D": np.array((0.0, -1.0, 0.0)),
            "L": np.array((-1.0, 0.0, 0.0)),
            "R": np.array((1.0, 0.0, 0.0)),
        }
        result = np.zeros((len(FINGERS), height_samples), dtype=np.float32)
        for finger_index, finger in enumerate(FINGERS):
            for height_index, z_value in enumerate(heights):
                origin = np.array((0.0, 0.0, float(z_value)))
                hit = self.ray_exit(origin, directions[finger])
                if hit is not None:
                    result[finger_index, height_index] = float(
                        np.clip(
                            hit[0]
                            / max(float(distance_normalization_m), 1e-9),
                            0.0,
                            1.5,
                        )
                    )
        return result


OBJECT_SPECS: Dict[ObjectKind, ObjectSpec] = {
    ObjectKind.CUBE: ObjectSpec(
        ObjectKind.CUBE,
        "object_cube",
        "joint:object_cube",
        ("object_cube_geom",),
        "object_cube_reference",
        0.285,
        -0.025,
        0.025,
        (0.0, 0.0, 0.0),
        (0.025, 0.025),
    ),
    ObjectKind.CYLINDER: ObjectSpec(
        ObjectKind.CYLINDER,
        "object_cylinder",
        "joint:object_cylinder",
        ("object_cylinder_geom",),
        "object_cylinder_reference",
        0.310,
        -0.050,
        0.050,
        (0.0, 0.0, 0.0),
        (0.025, 0.025),
    ),
    ObjectKind.BANANA: ObjectSpec(
        ObjectKind.BANANA,
        "object_banana",
        "joint:object_banana",
        (
            "object_banana_visual",
            *tuple(
                f"object_banana_collision_{index:03d}"
                for index in range(len(BANANA_CAPSULES))
            ),
        ),
        "object_banana_reference",
        0.260,
        0.0,
        0.036740,
        (0.0, 0.0, 0.0175339),
        (0.075883, 0.100640),
        mass_kg=0.066,
    ),
    ObjectKind.PEAR: ObjectSpec(
        ObjectKind.PEAR,
        "object_pear",
        "joint:object_pear",
        (
            "object_pear_visual",
            *tuple(
                f"object_pear_collision_{index:03d}"
                for index in range(len(PEAR_CAPSULES))
            ),
        ),
        "object_pear_reference",
        0.260,
        0.0,
        PEAR_TOTAL_HEIGHT_M,
        (0.0, 0.0, 0.0302195),
        (0.034050, 0.057146),
        mass_kg=0.049,
    ),
    ObjectKind.STRAWBERRY: ObjectSpec(
        ObjectKind.STRAWBERRY,
        "object_strawberry",
        "joint:object_strawberry",
        (
            "object_strawberry_visual",
            *tuple(
                f"object_strawberry_collision_{index:03d}"
                for index in range(len(STRAWBERRY_CAPSULES))
            ),
        ),
        "object_strawberry_reference",
        0.260,
        0.0,
        STRAWBERRY_TOTAL_HEIGHT_M,
        (0.0, 0.0, 0.0213315),
        (0.023243, 0.022987),
        mass_kg=0.018,
    ),
    ObjectKind.ORANGE: ObjectSpec(
        ObjectKind.ORANGE,
        "object_orange",
        "joint:object_orange",
        (
            "object_orange_visual",
            *tuple(
                f"object_orange_collision_{index:03d}"
                for index in range(len(ORANGE_CAPSULES))
            ),
        ),
        "object_orange_reference",
        0.260,
        0.0,
        ORANGE_TOTAL_HEIGHT_M,
        (0.0, 0.0, 0.0338816),
        (0.036286, 0.037001),
        mass_kg=0.047,
    ),
    ObjectKind.CONE: ObjectSpec(
        ObjectKind.CONE,
        "object_cone",
        "joint:object_cone",
        (
            "object_cone_visual",
            "object_cone_collision_0",
        ),
        "object_cone_reference",
        0.260,
        0.0,
        CONE_HEIGHT_M,
        (0.0, 0.0, 0.030),
        (CONE_RADIUS_M, CONE_RADIUS_M),
    ),
    ObjectKind.PLATE: ObjectSpec(
        ObjectKind.PLATE,
        "object_plate",
        "joint:object_plate",
        ("object_plate_geom",),
        "object_plate_reference",
        0.265,
        -0.005,
        0.005,
        (0.0, 0.0, 0.0),
        (0.025, 0.025),
    ),
}


def get_object_spec(kind: str | ObjectKind) -> ObjectSpec:
    try:
        key = kind if isinstance(kind, ObjectKind) else ObjectKind(str(kind))
    except ValueError as exc:
        valid = ", ".join(value.value for value in OBJECT_KIND_ORDER)
        raise KeyError(f"unknown object {kind!r}; expected one of {valid}") from exc
    return OBJECT_SPECS[key]


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rz @ ry @ rx


@dataclass(frozen=True)
class GraspPlane:
    center_local: np.ndarray
    roll_rad: float
    pitch_rad: float
    yaw_rad: float = 0.0

    @property
    def rotation_local(self) -> np.ndarray:
        return _rotation_matrix(
            float(self.roll_rad),
            float(self.pitch_rad),
            float(self.yaw_rad),
        )

    @property
    def u_local(self) -> np.ndarray:
        return self.rotation_local[:, 0]

    @property
    def v_local(self) -> np.ndarray:
        return self.rotation_local[:, 1]

    @property
    def normal_local(self) -> np.ndarray:
        return self.rotation_local[:, 2]

    def vector5(self) -> np.ndarray:
        return np.array(
            [
                *np.asarray(self.center_local, dtype=np.float64).reshape(3),
                self.roll_rad,
                self.pitch_rad,
            ],
            dtype=np.float64,
        )

    def vector6(self) -> np.ndarray:
        return np.array(
            [
                *np.asarray(self.center_local, dtype=np.float64).reshape(3),
                self.roll_rad,
                self.pitch_rad,
                self.yaw_rad,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class GraspPlan:
    object_kind: ObjectKind
    plane: GraspPlane
    pregrasp_targets_local: Mapping[str, np.ndarray]
    contact_targets_local: Mapping[str, np.ndarray]
    surface_points_local: Mapping[str, np.ndarray]
    contact_normals_local: Mapping[str, np.ndarray]
    outward_directions_local: Mapping[str, np.ndarray]
    score: float
    valid: bool
    diagnostics: Mapping[str, float | str]

    @property
    def tip_targets_local(self) -> Mapping[str, np.ndarray]:
        return self.contact_targets_local

    def targets_world(
        self,
        body_world_pos: Sequence[float],
        body_world_quat: Sequence[float],
        *,
        pregrasp: bool = False,
    ) -> Dict[str, np.ndarray]:
        position = np.asarray(body_world_pos, dtype=np.float64).reshape(3)
        rotation = quat_to_rot(body_world_quat)
        local = (
            self.pregrasp_targets_local
            if pregrasp
            else self.contact_targets_local
        )
        return {
            finger: position + rotation @ np.asarray(local[finger])
            for finger in FINGERS
        }


class AdaptiveGraspPlanePlanner:
    """Deterministic geometry teacher with a residual-safe candidate API."""

    def __init__(
        self,
        *,
        hard_tip_radius_m: float = HARD_TIP_RADIUS_M,
        pregrasp_clearance_m: float = DEFAULT_PREGRASP_CLEARANCE_M,
        maximum_tilt_rad: float = math.radians(12.0),
        enforce_anti_support: bool = True,
    ) -> None:
        self.hard_tip_radius_m = float(hard_tip_radius_m)
        self.pregrasp_clearance_m = float(pregrasp_clearance_m)
        self.maximum_tilt_rad = float(maximum_tilt_rad)
        self.enforce_anti_support = bool(enforce_anti_support)

    def _offset_exit(
        self,
        spec: ObjectSpec,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> Optional[float]:
        """Ray distance to the Euclidean hard-tip-radius offset surface."""
        target = self.hard_tip_radius_m
        low = 0.0
        if spec.signed_distance(origin) > target:
            return None
        high = 0.001
        while (
            high <= 0.35
            and spec.signed_distance(origin + high * direction) < target
        ):
            low = high
            high *= 1.5
        if high > 0.35:
            return None
        for _ in range(36):
            middle = 0.5 * (low + high)
            if (
                spec.signed_distance(origin + middle * direction)
                < target
            ):
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    def _balanced_horizontal_section(
        self,
        spec: ObjectSpec,
        seed_center: np.ndarray,
        yaw_world_rad: float,
        body_world_quat: Sequence[float],
    ) -> Optional[
        tuple[
            np.ndarray,
            Dict[str, np.ndarray],
            Dict[str, float],
            int,
        ]
    ]:
        """Find four centred roots on the radius-offset object silhouette."""
        rotation_world = quat_to_rot(body_world_quat)
        cosine = math.cos(float(yaw_world_rad))
        sine = math.sin(float(yaw_world_rad))
        u_world = np.asarray((cosine, sine, 0.0), dtype=np.float64)
        v_world = np.asarray((-sine, cosine, 0.0), dtype=np.float64)
        u_local = rotation_world.T @ u_world
        v_local = rotation_world.T @ v_world
        directions = {
            "R": u_local,
            "L": -u_local,
            "U": v_local,
            "D": -v_local,
        }
        center = np.asarray(seed_center, dtype=np.float64).copy()
        roots: Dict[str, float] = {}
        for iteration in range(40):
            roots = {}
            for finger in FINGERS:
                root = self._offset_exit(
                    spec,
                    center,
                    directions[finger],
                )
                if root is None:
                    return None
                roots[finger] = float(root)
            delta = 0.5 * (
                (roots["R"] - roots["L"]) * u_local
                + (roots["U"] - roots["D"]) * v_local
            )
            if float(np.linalg.norm(delta)) <= 2.5e-5:
                break
            center += 0.35 * delta
            if float(np.linalg.norm(center - seed_center)) > 0.12:
                return None
        else:
            return None

        roots = {}
        for finger in FINGERS:
            root = self._offset_exit(
                spec,
                center,
                directions[finger],
            )
            if root is None:
                return None
            roots[finger] = float(root)
        pair_asymmetry = max(
            abs(roots["R"] - roots["L"]),
            abs(roots["U"] - roots["D"]),
        )
        if pair_asymmetry > 1.0e-4:
            return None
        if not spec.contains(center, tolerance=0.001):
            return None
        return center, directions, roots, iteration + 1

    def _build_candidate(
        self,
        spec: ObjectSpec,
        plane: GraspPlane,
        *,
        ik: Optional[GripperIK],
        body_world_pos: Optional[Sequence[float]],
        body_world_quat: Sequence[float],
        table_top_z: float,
        table_clearance_m: float,
    ) -> GraspPlan:
        seed_center = np.asarray(
            plane.center_local, dtype=np.float64
        ).reshape(3)

        def invalid(
            reason: str,
            **diagnostics: float | str,
        ) -> GraspPlan:
            return GraspPlan(
                spec.kind,
                plane,
                {},
                {},
                {},
                {},
                {},
                -math.inf,
                False,
                {"reason": reason, **diagnostics},
            )

        if (
            abs(float(plane.roll_rad)) > 1e-12
            or abs(float(plane.pitch_rad)) > 1e-12
        ):
            return invalid("grasp_plane_not_parallel_to_xy")
        if not spec.contains(seed_center, tolerance=1e-6):
            return invalid("plane_center_outside_object")
        com = np.asarray(spec.local_com_m, dtype=np.float64)
        grasp_plane_below_com_m = max(
            float(com[2] - seed_center[2]),
            0.0,
        )
        if (
            self.enforce_anti_support
            and grasp_plane_below_com_m > MAX_GRASP_PLANE_BELOW_COM_M
        ):
            return invalid(
                "grasp_plane_too_far_below_com",
                grasp_plane_below_com_m=grasp_plane_below_com_m,
                maximum_grasp_plane_below_com_m=(
                    MAX_GRASP_PLANE_BELOW_COM_M
                ),
            )

        balanced = self._balanced_horizontal_section(
            spec,
            seed_center,
            plane.yaw_rad,
            body_world_quat,
        )
        if balanced is None:
            return invalid("no_balanced_hard_tip_offset_section")
        center, directions, offset_roots, balance_iterations = balanced
        grasp_plane_below_com_m = max(
            float(com[2] - center[2]),
            0.0,
        )
        if (
            self.enforce_anti_support
            and grasp_plane_below_com_m > MAX_GRASP_PLANE_BELOW_COM_M
        ):
            return invalid(
                "balanced_grasp_plane_too_far_below_com",
                grasp_plane_below_com_m=grasp_plane_below_com_m,
                maximum_grasp_plane_below_com_m=(
                    MAX_GRASP_PLANE_BELOW_COM_M
                ),
            )
        plane = GraspPlane(
            center.copy(),
            0.0,
            0.0,
            float(plane.yaw_rad),
        )

        u_contact_half = 0.5 * (
            offset_roots["R"] + offset_roots["L"]
        )
        v_contact_half = 0.5 * (
            offset_roots["U"] + offset_roots["D"]
        )
        contact = {
            "R": center + u_contact_half * directions["R"],
            "L": center + u_contact_half * directions["L"],
            "U": center + v_contact_half * directions["U"],
            "D": center + v_contact_half * directions["D"],
        }
        clearance_errors = {
            finger: abs(
                spec.signed_distance(contact[finger])
                - self.hard_tip_radius_m
            )
            for finger in FINGERS
        }
        max_clearance_error = max(clearance_errors.values())
        if max_clearance_error > 3.0e-4:
            return invalid("hard_tip_clearance_error")

        normals: Dict[str, np.ndarray] = {}
        surface: Dict[str, np.ndarray] = {}
        alignments: Dict[str, float] = {}
        for finger in FINGERS:
            normal = spec.surface_normal(contact[finger])
            if float(normal @ directions[finger]) < 0.0:
                normal = -normal
            normals[finger] = normal
            alignments[finger] = float(normal @ directions[finger])
            surface[finger] = (
                contact[finger] - self.hard_tip_radius_m * normal
            )
            if alignments[finger] < 0.35:
                return invalid(f"{finger}_surface_normal_misaligned")

        rotation_world = quat_to_rot(body_world_quat)
        normals_world = {
            finger: rotation_world @ normals[finger]
            for finger in FINGERS
        }
        planned_pair_normal_dot_rl = float(
            normals_world["R"] @ normals_world["L"]
        )
        planned_pair_normal_dot_ud = float(
            normals_world["U"] @ normals_world["D"]
        )
        if (
            self.enforce_anti_support
            and planned_pair_normal_dot_rl > MAX_PLANNED_PAIR_NORMAL_DOT
        ):
            return invalid(
                "planned_rl_contact_normals_not_opposed",
                planned_pair_normal_dot_rl=planned_pair_normal_dot_rl,
                maximum_planned_pair_normal_dot=(
                    MAX_PLANNED_PAIR_NORMAL_DOT
                ),
            )
        if (
            self.enforce_anti_support
            and planned_pair_normal_dot_ud > MAX_PLANNED_PAIR_NORMAL_DOT
        ):
            return invalid(
                "planned_ud_contact_normals_not_opposed",
                planned_pair_normal_dot_ud=planned_pair_normal_dot_ud,
                maximum_planned_pair_normal_dot=(
                    MAX_PLANNED_PAIR_NORMAL_DOT
                ),
            )
        planned_normal_vertical_sum = float(
            sum(
                normals_world[finger][2]
                for finger in FINGERS
            )
        )
        predicted_normal_support_n = max(
            -NOMINAL_PINCH_TIP_FORCE_N
            * planned_normal_vertical_sum,
            0.0,
        )
        predicted_normal_support_fraction = (
            predicted_normal_support_n
            / max(spec.mass_kg * STANDARD_GRAVITY_M_S2, 1e-12)
        )
        if (
            self.enforce_anti_support
            and predicted_normal_support_fraction
            > MAX_PLANNED_NORMAL_SUPPORT_FRACTION
        ):
            return invalid(
                "predicted_normal_support_fraction_exceeded",
                predicted_normal_support_n=predicted_normal_support_n,
                predicted_normal_support_fraction=(
                    predicted_normal_support_fraction
                ),
                maximum_predicted_normal_support_fraction=(
                    MAX_PLANNED_NORMAL_SUPPORT_FRACTION
                ),
            )

        symmetry_r = abs(offset_roots["R"] - offset_roots["L"])
        symmetry_u = abs(offset_roots["U"] - offset_roots["D"])
        u_half = u_contact_half + self.pregrasp_clearance_m
        v_half = v_contact_half + self.pregrasp_clearance_m
        pregrasp = {
            "R": center + u_half * directions["R"],
            "L": center + u_half * directions["L"],
            "U": center + v_half * directions["U"],
            "D": center + v_half * directions["D"],
        }
        # Regular-object cages are strict horizontal parallelograms.  Fruits
        # keep the same XY formation but deliberately do not make a
        # coplanarity contract: each hard tip's Z is an independent geometric
        # coordinate throughout planning and control.
        pregrasp_diagonal_error = float(
            np.linalg.norm(
                pregrasp["R"] + pregrasp["L"]
                - pregrasp["U"] - pregrasp["D"]
            )
        )
        contact_diagonal_error = float(
            np.linalg.norm(
                contact["R"] + contact["L"]
                - contact["U"] - contact["D"]
            )
        )
        pregrasp_diagonal_error_xy = float(
            np.linalg.norm(
                (
                    pregrasp["R"] + pregrasp["L"]
                    - pregrasp["U"] - pregrasp["D"]
                )[:2]
            )
        )
        contact_diagonal_error_xy = float(
            np.linalg.norm(
                (
                    contact["R"] + contact["L"]
                    - contact["U"] - contact["D"]
                )[:2]
            )
        )
        contact_z_span = float(
            np.ptp(
                [
                    float((rotation_world @ contact[finger])[2])
                    for finger in FINGERS
                ]
            )
        )
        diagonal_tolerance_error = (
            pregrasp_diagonal_error_xy
            if spec.kind in FRUIT_KINDS
            else pregrasp_diagonal_error
        )
        if diagonal_tolerance_error > 1e-10:
            return invalid("pregrasp_not_parallelogram")
        diagonal_tolerance_error = (
            contact_diagonal_error_xy
            if spec.kind in FRUIT_KINDS
            else contact_diagonal_error
        )
        if diagonal_tolerance_error > 1e-10:
            return invalid("contact_not_parallelogram")
        if (
            spec.kind not in FRUIT_KINDS
            and contact_z_span > 1e-10
        ):
            return invalid("contact_not_parallel_to_xy")

        if body_world_pos is not None:
            body_position = np.asarray(body_world_pos, dtype=np.float64)
            for target_set, label in ((pregrasp, "pregrasp"), (contact, "contact")):
                minimum_center_z = min(
                    float((body_position + rotation_world @ target_set[f])[2])
                    for f in FINGERS
                )
                if (
                    minimum_center_z - self.hard_tip_radius_m
                    < float(table_top_z) + float(table_clearance_m)
                ):
                    return invalid(f"{label}_tip_table_clearance")

        ik_margin = 1.0
        if ik is not None and body_world_pos is not None:
            body_position = np.asarray(body_world_pos, dtype=np.float64)
            for fraction in (0.0, 0.5, 1.0):
                local_targets = {
                    finger: (
                        (1.0 - fraction) * pregrasp[finger]
                        + fraction * contact[finger]
                    )
                    for finger in FINGERS
                }
                world_targets = {
                    finger: body_position
                    + rotation_world @ local_targets[finger]
                    for finger in FINGERS
                }
                solution = ik.solve_tip_targets(world_targets)
                if not solution.success:
                    return invalid(f"ik_unreachable_at_closure_{fraction:.1f}")
                normalized_margin = np.minimum(
                    solution.ctrl - ik.ctrl_ranges[:, 0],
                    ik.ctrl_ranges[:, 1] - solution.ctrl,
                ) / np.maximum(
                    ik.ctrl_ranges[:, 1] - ik.ctrl_ranges[:, 0],
                    1e-9,
                )
                ik_margin = min(ik_margin, float(np.min(normalized_margin)))

        symmetry_error = symmetry_r + symmetry_u
        com_offset = float(np.linalg.norm((center - com)[:2]))
        vertical_com_offset = abs(float(center[2] - com[2]))
        mean_alignment = float(np.mean(tuple(alignments.values())))
        mean_normal_z = float(
            np.mean([normal[2] for normal in normals.values()])
        )
        anti_support_penalty = (
            4.0 * predicted_normal_support_fraction
            if self.enforce_anti_support
            else 0.0
        )
        tilt = math.hypot(float(plane.roll_rad), float(plane.pitch_rad))
        score = (
            3.0 * mean_alignment
            + 1.5 * ik_margin
            - 100.0 * symmetry_error
            - 30.0 * com_offset
            - 20.0 * vertical_com_offset
            - 4.0 * (u_contact_half + v_contact_half)
            - 0.5 * tilt / max(self.maximum_tilt_rad, 1e-9)
            - anti_support_penalty
        )
        return GraspPlan(
            object_kind=spec.kind,
            plane=plane,
            pregrasp_targets_local={
                key: value.copy() for key, value in pregrasp.items()
            },
            contact_targets_local={
                key: value.copy() for key, value in contact.items()
            },
            surface_points_local={
                key: value.copy() for key, value in surface.items()
            },
            contact_normals_local={
                key: value.copy() for key, value in normals.items()
            },
            outward_directions_local={
                key: value.copy() for key, value in directions.items()
            },
            score=float(score),
            valid=True,
            diagnostics={
                "mean_normal_alignment": mean_alignment,
                "mean_normal_z": mean_normal_z,
                "anti_support_enforced": self.enforce_anti_support,
                "anti_support_penalty": anti_support_penalty,
                "predicted_normal_support_n": (
                    predicted_normal_support_n
                ),
                "predicted_normal_support_fraction": (
                    predicted_normal_support_fraction
                ),
                "maximum_predicted_normal_support_fraction": (
                    MAX_PLANNED_NORMAL_SUPPORT_FRACTION
                ),
                "planned_pair_normal_dot_rl": (
                    planned_pair_normal_dot_rl
                ),
                "planned_pair_normal_dot_ud": (
                    planned_pair_normal_dot_ud
                ),
                "maximum_planned_pair_normal_dot": (
                    MAX_PLANNED_PAIR_NORMAL_DOT
                ),
                "grasp_plane_below_com_m": grasp_plane_below_com_m,
                "maximum_grasp_plane_below_com_m": (
                    MAX_GRASP_PLANE_BELOW_COM_M
                ),
                **{
                    f"planned_world_normal_z_{finger}": float(
                        normals_world[finger][2]
                    )
                    for finger in FINGERS
                },
                "ik_margin": ik_margin,
                "symmetry_error_m": symmetry_error,
                "com_xy_offset_m": com_offset,
                "com_vertical_offset_m": vertical_com_offset,
                "pregrasp_diagonal_error_m": pregrasp_diagonal_error,
                "contact_diagonal_error_m": contact_diagonal_error,
                "pregrasp_xy_diagonal_error_m": (
                    pregrasp_diagonal_error_xy
                ),
                "contact_xy_diagonal_error_m": (
                    contact_diagonal_error_xy
                ),
                "contact_z_span_m": contact_z_span,
                "world_contact_z_span_m": contact_z_span,
                "tip_coplanarity_required": (
                    "no" if spec.kind in FRUIT_KINDS else "yes"
                ),
                "max_tip_clearance_error_m": max_clearance_error,
                "balance_iterations": float(balance_iterations),
                "pair_root_asymmetry_m": max(
                    symmetry_r,
                    symmetry_u,
                ),
                "opposite_pair_max_asymmetry_m": max(
                    symmetry_r,
                    symmetry_u,
                ),
                "u_half_diagonal_m": u_half,
                "v_half_diagonal_m": v_half,
                "u_contact_half_diagonal_m": u_contact_half,
                "v_contact_half_diagonal_m": v_contact_half,
            },
        )

    def _candidate_planes(self, spec: ObjectSpec) -> Iterable[GraspPlane]:
        if spec.kind in DIRECT_GRASP_PLANE_HEIGHT_FRACTION:
            # These regular solids use one explicit, axis-aligned horizontal
            # grasp plane rather than an adaptive height/yaw candidate:
            # cube side-face centres, cylinder mid-height, cone one-third
            # height measured from the base.
            height_fractions = (
                DIRECT_GRASP_PLANE_HEIGHT_FRACTION[spec.kind],
            )
            yaw_values = (DIRECT_GRASP_PLANE_YAW_RAD[spec.kind],)
        elif spec.kind in FRUIT_KINDS:
            # Search one shared, finite side-pinch grid.  Candidate scoring is
            # allowed to choose object-specific planes, but no fruit receives
            # a privileged lower-surface/supporting grasp.
            height_fractions = FRUIT_GRASP_PLANE_HEIGHT_FRACTIONS
            yaw_values = FRUIT_GRASP_PLANE_YAW_RAD
        elif spec.kind is ObjectKind.PLATE:
            # A mid-plane side grasp is intentionally evaluated and rejected
            # by the table-clearance test: a 10 mm flat plate is thinner than
            # the 16 mm spherical fingertip radius.  This object demonstrates
            # the gripper's thin-object limitation instead of using the table
            # or an artificial top-edge trick to lift it.
            height_fractions = (0.50,)
            yaw_values = (0.0,)
        else:
            height_fractions = (0.55,)
            yaw_values = (math.radians(45.0),)
        tilts = (0.0,)
        for fraction in height_fractions:
            z_value = spec.local_bottom_z_m + fraction * spec.height_m
            xy_centers = spec.candidate_xy_centers(
                z_value,
                limit=1,
            )
            for yaw in yaw_values:
                for center_x, center_y in xy_centers:
                    for roll in tilts:
                        for pitch in tilts:
                            # Avoid redundant coupled extremes in the first teacher.
                            if roll != 0.0 and pitch != 0.0:
                                continue
                            yield GraspPlane(
                                np.array(
                                    (
                                        center_x,
                                        center_y,
                                        z_value,
                                    ),
                                    dtype=np.float64,
                                ),
                                roll,
                                pitch,
                                yaw,
                            )

    def plan(
        self,
        spec: str | ObjectKind | ObjectSpec,
        *,
        ik: Optional[GripperIK] = None,
        body_world_pos: Optional[Sequence[float]] = None,
        body_world_quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        table_top_z: float = 0.260,
        table_clearance_m: float = 0.002,
    ) -> GraspPlan:
        object_spec = (
            spec if isinstance(spec, ObjectSpec) else get_object_spec(spec)
        )
        candidates = [
            self._build_candidate(
                object_spec,
                plane,
                ik=ik,
                body_world_pos=body_world_pos,
                body_world_quat=body_world_quat,
                table_top_z=table_top_z,
                table_clearance_m=table_clearance_m,
            )
            for plane in self._candidate_planes(object_spec)
        ]
        valid = [candidate for candidate in candidates if candidate.valid]
        if not valid:
            reasons = sorted(
                {
                    str(candidate.diagnostics.get("reason", "unknown"))
                    for candidate in candidates
                }
            )
            raise RuntimeError(
                f"no feasible adaptive grasp plane for {object_spec.kind.value}: "
                + ", ".join(reasons)
            )
        return max(valid, key=lambda candidate: candidate.score)

    def apply_residual(
        self,
        nominal: GraspPlan,
        residual: Sequence[float],
        *,
        ik: Optional[GripperIK] = None,
        body_world_pos: Optional[Sequence[float]] = None,
        body_world_quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        table_top_z: float = 0.260,
        table_clearance_m: float = 0.002,
    ) -> GraspPlan:
        """Apply physical ``(dx,dy,dz,droll,dpitch,dyaw)`` residuals."""
        value = np.asarray(residual, dtype=np.float64).reshape(6)
        center = np.asarray(nominal.plane.center_local) + value[:3]
        roll = float(
            np.clip(
                nominal.plane.roll_rad + value[3],
                -self.maximum_tilt_rad,
                self.maximum_tilt_rad,
            )
        )
        pitch = float(
            np.clip(
                nominal.plane.pitch_rad + value[4],
                -self.maximum_tilt_rad,
                self.maximum_tilt_rad,
            )
        )
        yaw = math.atan2(
            math.sin(nominal.plane.yaw_rad + float(value[5])),
            math.cos(nominal.plane.yaw_rad + float(value[5])),
        )
        plane = GraspPlane(center, roll, pitch, yaw)
        return self._build_candidate(
            get_object_spec(nominal.object_kind),
            plane,
            ik=ik,
            body_world_pos=body_world_pos,
            body_world_quat=body_world_quat,
            table_top_z=table_top_z,
            table_clearance_m=table_clearance_m,
        )

    def project_residual(
        self,
        nominal: GraspPlan,
        residual: Sequence[float],
        **kwargs: object,
    ) -> tuple[GraspPlan, float]:
        requested = np.asarray(residual, dtype=np.float64).reshape(6)
        direct = self.apply_residual(nominal, requested, **kwargs)
        if direct.valid:
            return direct, 1.0
        low = 0.0
        high = 1.0
        best = nominal
        for _ in range(24):
            middle = 0.5 * (low + high)
            candidate = self.apply_residual(
                nominal, middle * requested, **kwargs
            )
            if candidate.valid:
                low = middle
                best = candidate
            else:
                high = middle
        return best, float(low)


__all__ = [
    "AdaptiveGraspPlanePlanner",
    "BANANA_RADIUS_M",
    "BANANA_SEGMENTS",
    "BANANA_IRREGULAR_PHASE_RAD",
    "DIRECT_GRASP_PLANE_HEIGHT_FRACTION",
    "DIRECT_GRASP_PLANE_YAW_RAD",
    "DEFAULT_PREGRASP_CLEARANCE_M",
    "FRUIT_CAPSULE_MANIFEST",
    "FRUIT_CAPSULE_MANIFEST_PATH",
    "FRUIT_CAPSULE_MANIFEST_SHA256",
    "FRUIT_CAPSULE_SCHEMA_VERSION",
    "FRUIT_CAPSULES_BY_NAME",
    "FRUIT_KINDS",
    "GraspPlan",
    "GraspPlane",
    "HARD_TIP_RADIUS_M",
    "IRREGULAR_RADIAL_LIMIT_M",
    "OBJECT_KIND_ORDER",
    "OBJECT_MASS_KG",
    "OBJECT_SPECS",
    "ORANGE_CAPSULES",
    "ORANGE_TOTAL_HEIGHT_M",
    "ObjectKind",
    "ObjectSpec",
    "PEAR_IRREGULAR_PHASE_RAD",
    "PEAR_JOIN_Z_M",
    "PEAR_TOTAL_HEIGHT_M",
    "PLATE_HALF_SIZE_M",
    "get_object_spec",
    "irregular_radius_delta_m",
]
