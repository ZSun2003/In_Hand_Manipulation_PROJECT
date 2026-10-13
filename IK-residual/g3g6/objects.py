"""Four exact rigid-object profiles and their four hard-tip grasp maps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from . import config


FINGERS = ("U", "D", "L", "R")
RADIAL = {
    "U": np.array((0.0, 1.0, 0.0)),
    "D": np.array((0.0, -1.0, 0.0)),
    "L": np.array((-1.0, 0.0, 0.0)),
    "R": np.array((1.0, 0.0, 0.0)),
}


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    body_name: str
    joint_name: str
    geom_name: str
    dimensions_m: tuple[float, ...]
    rest_grasp_plane_z_m: float
    surface_points_local_m: Mapping[str, np.ndarray]
    surface_normals_local: Mapping[str, np.ndarray]
    contact_indentation_m: float = config.CONTACT_INDENTATION_M

    @property
    def lift_distance_m(self) -> float:
        return config.TARGET_GRASP_PLANE_Z_M - self.rest_grasp_plane_z_m

    def tip_centres_local_m(self, *, pregrasp: bool = False) -> dict[str, np.ndarray]:
        offset = (
            config.HARD_TIP_RADIUS_M + config.PREGRASP_CLEARANCE_M
            if pregrasp
            else config.HARD_TIP_RADIUS_M - self.contact_indentation_m
        )
        return {
            finger: np.asarray(self.surface_points_local_m[finger], dtype=np.float64)
            + offset * np.asarray(self.surface_normals_local[finger], dtype=np.float64)
            for finger in FINGERS
        }


def _frozen(values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    copied: dict[str, np.ndarray] = {}
    for key, value in values.items():
        array = np.asarray(value, dtype=np.float64).reshape(3).copy()
        array.setflags(write=False)
        copied[key] = array
    return MappingProxyType(copied)


def _radial_surface(radius: float) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    return (
        _frozen({finger: radius * direction for finger, direction in RADIAL.items()}),
        _frozen(RADIAL),
    )


def _triangle_grasp() -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    # Rotate the regular triangle -15 degrees.  Its face normals become
    # +45, +165 and -75 degrees: R/U share the +45-degree face, while L and D
    # each occupy one remaining rectangular face.  The shared points are
    # separated by 33 mm, clearing two 16 mm hard-tip spheres.
    circumradius = 0.060 / math.sqrt(3.0)
    vertices = np.array(
        ((circumradius, 0.0), (-circumradius / 2.0, 0.030), (-circumradius / 2.0, -0.030)),
        dtype=np.float64,
    )
    yaw = math.radians(-15.0)
    rotation = np.array(((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw))))
    vertices = vertices @ rotation.T
    pairs = ((0, 1), (1, 2), (2, 0))
    centres: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for first, second in pairs:
        centre = 0.5 * (vertices[first] + vertices[second])
        edge = vertices[second] - vertices[first]
        normal = np.array((edge[1], -edge[0]), dtype=np.float64)
        normal /= np.linalg.norm(normal)
        if float(np.dot(normal, centre)) < 0.0:
            normal = -normal
        centres.append(centre)
        normals.append(normal)
    tangent = np.array((-normals[0][1], normals[0][0]))
    half_separation = 0.0165
    points_xy = {
        "R": centres[0] - half_separation * tangent,
        "U": centres[0] + half_separation * tangent,
        "L": centres[1],
        "D": centres[2],
    }
    normal_xy = {"R": normals[0], "U": normals[0], "L": normals[1], "D": normals[2]}
    points = {finger: np.array((*points_xy[finger], 0.0)) for finger in FINGERS}
    normal3 = {finger: np.array((*normal_xy[finger], 0.0)) for finger in FINGERS}
    return _frozen(points), _frozen(normal3)


def _cone_grasp() -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    # At H/3 from the base, r = R*(1-1/3) = 20 mm.  Contact points share that
    # plane; sphere centres are displaced along the true sloped surface normal.
    # A -5.625 degree half-facet offset places all contacts at the centres of
    # the 32-sided collision mesh facets instead of numerically ambiguous
    # edges; the four azimuths remain exactly 90 degrees apart.
    yaw = -math.pi / 32.0
    rotate = np.array(
        ((math.cos(yaw), -math.sin(yaw), 0.0),
         (math.sin(yaw), math.cos(yaw), 0.0),
         (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    directions = {
        finger: rotate @ direction for finger, direction in RADIAL.items()
    }
    points = {finger: 0.020 * direction for finger, direction in directions.items()}
    normals = {}
    for finger, direction in directions.items():
        normal = direction + np.array((0.0, 0.0, 0.3))
        normals[finger] = normal / np.linalg.norm(normal)
    return _frozen(points), _frozen(normals)


_cube_points, _cube_normals = _radial_surface(0.025)
_cylinder_points, _cylinder_normals = _radial_surface(0.025)
_triangle_points, _triangle_normals = _triangle_grasp()
_cone_points, _cone_normals = _cone_grasp()

OBJECTS: Mapping[str, ObjectSpec] = MappingProxyType(
    {
        "cube": ObjectSpec(
            "cube", "cube_5cm", "joint:cube_5cm", "cube_geom", (0.050, 0.050, 0.050),
            config.TABLE_TOP_M + 0.025, _cube_points, _cube_normals,
        ),
        "cylinder": ObjectSpec(
            "cylinder", "cylinder_50x80", "joint:cylinder_50x80", "cylinder_geom", (0.050, 0.080),
            config.TABLE_TOP_M + 0.040, _cylinder_points, _cylinder_normals,
        ),
        "triangular_prism": ObjectSpec(
            "triangular_prism", "triangular_prism_60x80", "joint:triangular_prism_60x80", "triangular_prism_geom", (0.060, 0.080),
            config.TABLE_TOP_M + 0.040, _triangle_points, _triangle_normals,
        ),
        "cone": ObjectSpec(
            "cone", "cone_60x100", "joint:cone_60x100", "cone_geom", (0.060, 0.100),
            config.TABLE_TOP_M + 0.100 / 3.0, _cone_points, _cone_normals,
            contact_indentation_m=0.001,
        ),
    }
)


def get_object_spec(name: str) -> ObjectSpec:
    try:
        return OBJECTS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown object {name!r}; expected one of {tuple(OBJECTS)}") from exc
