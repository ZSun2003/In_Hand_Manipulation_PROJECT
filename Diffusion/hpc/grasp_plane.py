#!/usr/bin/env python3
"""Fixed continuous four-finger grasp-plane planner for the composite screw.

This file is intentionally self-contained inside ``hpc``.  It mirrors the
GRAB raycast/normal/parallelogram/IK/score checks used by the MPS profile,
while fixing the plane at the head height centre and adding the screw's
full-insertion fingertip-clearance gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


FINGERS = ("U", "D", "L", "R")
HARD_TIP_RADIUS_M = 0.016
PREGRASP_CLEARANCE_M = 0.012


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rz @ ry @ rx


def _quat_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class GraspPlane:
    center_local: np.ndarray
    roll_rad: float
    pitch_rad: float
    yaw_rad: float = 0.0

    @property
    def rotation_local(self) -> np.ndarray:
        return _rotation_matrix(self.roll_rad, self.pitch_rad, self.yaw_rad)

    @property
    def u_local(self) -> np.ndarray:
        return self.rotation_local[:, 0]

    @property
    def v_local(self) -> np.ndarray:
        return self.rotation_local[:, 1]


@dataclass(frozen=True)
class ScrewGraspPlan:
    plane: GraspPlane
    screw_tip_local_z_m: float
    pregrasp_targets_local: Mapping[str, np.ndarray]
    contact_targets_local: Mapping[str, np.ndarray]
    surface_points_local: Mapping[str, np.ndarray]
    outward_directions_local: Mapping[str, np.ndarray]
    score: float
    valid: bool
    diagnostics: Mapping[str, float | str]

    def targets_world(
        self,
        body_world_pos: Sequence[float],
        body_world_quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        *,
        pregrasp: bool = False,
    ) -> Dict[str, np.ndarray]:
        position = np.asarray(body_world_pos, dtype=np.float64).reshape(3)
        rotation = _quat_to_rotation(body_world_quat)
        local = (
            self.pregrasp_targets_local
            if pregrasp
            else self.contact_targets_local
        )
        return {
            finger: position + rotation @ np.asarray(local[finger])
            for finger in FINGERS
        }

    @property
    def grasp_height_from_tip_m(self) -> float:
        return float(
            self.plane.center_local[2] - self.screw_tip_local_z_m
        )

    @property
    def contact_radius_m(self) -> float:
        return float(
            np.mean(
                [
                    np.linalg.norm(
                        np.asarray(self.contact_targets_local[finger])[:2]
                    )
                    for finger in FINGERS
                ]
            )
        )

    @property
    def pregrasp_radius_m(self) -> float:
        return float(
            np.mean(
                [
                    np.linalg.norm(
                        np.asarray(self.pregrasp_targets_local[finger])[:2]
                    )
                    for finger in FINGERS
                ]
            )
        )


class AdaptiveScrewGraspPlanePlanner:
    """GRAB-compatible deterministic planner for the cylindrical screw head."""

    def __init__(
        self,
        *,
        head_radius_m: float = 0.025,
        head_bottom_local_z_m: float = 0.040,
        head_height_m: float = 0.040,
        screw_tip_local_z_m: float = -0.040,
        full_insertion_depth_m: float = 0.080,
        hard_tip_radius_m: float = HARD_TIP_RADIUS_M,
        pregrasp_clearance_m: float = PREGRASP_CLEARANCE_M,
        maximum_tilt_rad: float = math.radians(12.0),
    ) -> None:
        self.head_radius_m = float(head_radius_m)
        self.head_bottom_local_z_m = float(head_bottom_local_z_m)
        self.head_height_m = float(head_height_m)
        self.screw_tip_local_z_m = float(screw_tip_local_z_m)
        self.full_insertion_depth_m = float(full_insertion_depth_m)
        self.hard_tip_radius_m = float(hard_tip_radius_m)
        self.pregrasp_clearance_m = float(pregrasp_clearance_m)
        self.maximum_tilt_rad = float(maximum_tilt_rad)

    @property
    def head_top_local_z_m(self) -> float:
        return self.head_bottom_local_z_m + self.head_height_m

    def _contains(self, point: Sequence[float], tolerance: float = 1.0e-9) -> bool:
        x, y, z = np.asarray(point, dtype=np.float64).reshape(3)
        return bool(
            x * x + y * y <= (self.head_radius_m + tolerance) ** 2
            and self.head_bottom_local_z_m - tolerance
            <= z
            <= self.head_top_local_z_m + tolerance
        )

    def _ray_exit(
        self, origin: Sequence[float], direction: Sequence[float]
    ) -> Optional[tuple[float, np.ndarray]]:
        point = np.asarray(origin, dtype=np.float64).reshape(3)
        ray = np.asarray(direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(ray))
        if norm <= 1.0e-12 or not self._contains(point, tolerance=1.0e-7):
            return None
        ray = ray / norm
        candidates: list[float] = []
        radial_a = float(ray[0] ** 2 + ray[1] ** 2)
        if radial_a > 1.0e-14:
            radial_b = 2.0 * float(point[0] * ray[0] + point[1] * ray[1])
            radial_c = float(
                point[0] ** 2 + point[1] ** 2 - self.head_radius_m**2
            )
            discriminant = max(radial_b * radial_b - 4.0 * radial_a * radial_c, 0.0)
            root = (-radial_b + math.sqrt(discriminant)) / (2.0 * radial_a)
            if root > 1.0e-10:
                hit = point + root * ray
                if (
                    self.head_bottom_local_z_m - 1.0e-8
                    <= hit[2]
                    <= self.head_top_local_z_m + 1.0e-8
                ):
                    candidates.append(root)
        if abs(float(ray[2])) > 1.0e-14:
            cap_z = (
                self.head_top_local_z_m
                if ray[2] > 0.0
                else self.head_bottom_local_z_m
            )
            root = float((cap_z - point[2]) / ray[2])
            if root > 1.0e-10:
                hit = point + root * ray
                if float(hit[0] ** 2 + hit[1] ** 2) <= self.head_radius_m**2 + 1.0e-9:
                    candidates.append(root)
        if not candidates:
            return None
        distance = min(candidates)
        return distance, point + distance * ray

    def _surface_normal(self, point: Sequence[float]) -> np.ndarray:
        x, y, z = np.asarray(point, dtype=np.float64).reshape(3)
        radial_error = abs(math.hypot(x, y) - self.head_radius_m)
        bottom_error = abs(z - self.head_bottom_local_z_m)
        top_error = abs(z - self.head_top_local_z_m)
        if min(bottom_error, top_error) + 1.0e-8 < radial_error:
            return np.array(
                (0.0, 0.0, -1.0 if bottom_error < top_error else 1.0),
                dtype=np.float64,
            )
        radial = np.array((x, y, 0.0), dtype=np.float64)
        return radial / max(float(np.linalg.norm(radial)), 1.0e-12)

    def candidate_planes(self) -> Iterable[GraspPlane]:
        # The 40 mm head spans local z=40..80 mm.  Use its height centre at
        # local z=60 mm (100 mm above the screw tip), restoring the requested
        # upright hard-tip grasp plane at world z=0.360 m.
        yield GraspPlane(
            np.array(
                (
                    0.0,
                    0.0,
                    self.head_bottom_local_z_m
                    + 0.5 * self.head_height_m,
                ),
                dtype=np.float64,
            ),
            0.0,
            0.0,
            0.0,
        )

    def _invalid(self, plane: GraspPlane, reason: str) -> ScrewGraspPlan:
        return ScrewGraspPlan(
            plane=plane,
            screw_tip_local_z_m=self.screw_tip_local_z_m,
            pregrasp_targets_local={},
            contact_targets_local={},
            surface_points_local={},
            outward_directions_local={},
            score=-math.inf,
            valid=False,
            diagnostics={"reason": reason},
        )

    def _build_candidate(
        self,
        plane: GraspPlane,
        *,
        ik=None,
        body_world_pos: Optional[Sequence[float]],
        body_world_quat: Sequence[float],
        table_top_z: float,
        table_clearance_m: float,
    ) -> ScrewGraspPlan:
        center = np.asarray(plane.center_local, dtype=np.float64).reshape(3)
        if not self._contains(center, tolerance=1.0e-6):
            return self._invalid(plane, "plane_center_outside_head")
        outside_after_full_insertion = (
            center[2]
            - self.screw_tip_local_z_m
            - self.full_insertion_depth_m
        )
        if outside_after_full_insertion + 1.0e-10 < self.hard_tip_radius_m:
            return self._invalid(plane, "full_insertion_fingertip_socket_clearance")

        directions = {
            "R": plane.u_local,
            "L": -plane.u_local,
            "U": plane.v_local,
            "D": -plane.v_local,
        }
        hits: Dict[str, tuple[float, np.ndarray]] = {}
        normals: Dict[str, np.ndarray] = {}
        alignments: Dict[str, float] = {}
        for finger in FINGERS:
            hit = self._ray_exit(center, directions[finger])
            if hit is None:
                return self._invalid(plane, f"missing_{finger}_surface")
            normal = self._surface_normal(hit[1])
            if float(normal @ directions[finger]) < 0.0:
                normal = -normal
            alignment = float(normal @ directions[finger])
            if alignment < 0.35:
                return self._invalid(plane, f"{finger}_surface_normal_misaligned")
            hits[finger] = hit
            normals[finger] = normal
            alignments[finger] = alignment

        surface = {finger: hits[finger][1] for finger in FINGERS}
        contact = {
            finger: surface[finger] + self.hard_tip_radius_m * normals[finger]
            for finger in FINGERS
        }
        u_half = max(hits["R"][0], hits["L"][0]) + (
            self.hard_tip_radius_m + self.pregrasp_clearance_m
        )
        v_half = max(hits["U"][0], hits["D"][0]) + (
            self.hard_tip_radius_m + self.pregrasp_clearance_m
        )
        pregrasp = {
            "R": center + u_half * plane.u_local,
            "L": center - u_half * plane.u_local,
            "U": center + v_half * plane.v_local,
            "D": center - v_half * plane.v_local,
        }
        diagonal_error = float(
            np.linalg.norm(
                pregrasp["R"] + pregrasp["L"]
                - pregrasp["U"] - pregrasp["D"]
            )
        )
        if diagonal_error > 1.0e-10:
            return self._invalid(plane, "pregrasp_not_parallelogram")

        rotation_world = _quat_to_rotation(body_world_quat)
        if body_world_pos is not None:
            body_position = np.asarray(body_world_pos, dtype=np.float64).reshape(3)
            for target_set, label in ((pregrasp, "pregrasp"), (contact, "contact")):
                minimum_center_z = min(
                    float((body_position + rotation_world @ target_set[f])[2])
                    for f in FINGERS
                )
                if (
                    minimum_center_z - self.hard_tip_radius_m
                    < float(table_top_z) + float(table_clearance_m)
                ):
                    return self._invalid(plane, f"{label}_tip_table_clearance")

        ik_margin = 1.0
        if ik is not None and body_world_pos is not None:
            body_position = np.asarray(body_world_pos, dtype=np.float64).reshape(3)
            for fraction in (0.0, 0.5, 1.0):
                local_targets = {
                    finger: (
                        (1.0 - fraction) * pregrasp[finger]
                        + fraction * contact[finger]
                    )
                    for finger in FINGERS
                }
                world_targets = {
                    finger: body_position + rotation_world @ local_targets[finger]
                    for finger in FINGERS
                }
                solution = ik.solve_tip_targets(world_targets, strict=False)
                if not solution.success:
                    return self._invalid(
                        plane, f"ik_unreachable_at_closure_{fraction:.1f}"
                    )
                ctrl_ranges = np.asarray(ik.ctrl_ranges, dtype=np.float64)
                normalized_margin = np.minimum(
                    solution.ctrl - ctrl_ranges[:, 0],
                    ctrl_ranges[:, 1] - solution.ctrl,
                ) / np.maximum(ctrl_ranges[:, 1] - ctrl_ranges[:, 0], 1.0e-9)
                ik_margin = min(ik_margin, float(np.min(normalized_margin)))

        symmetry_error = abs(hits["R"][0] - hits["L"][0]) + abs(
            hits["U"][0] - hits["D"][0]
        )
        mean_alignment = float(np.mean(tuple(alignments.values())))
        mean_normal_z = float(np.mean([normal[2] for normal in normals.values()]))
        lift_support = max(-mean_normal_z, 0.0) - 1.5 * max(mean_normal_z, 0.0)
        tilt = math.hypot(plane.roll_rad, plane.pitch_rad)
        score = (
            3.0 * mean_alignment
            + 1.5 * ik_margin
            - 8.0 * symmetry_error
            - 0.5 * tilt / max(self.maximum_tilt_rad, 1.0e-9)
            + lift_support
            + 10.0 * outside_after_full_insertion
        )
        return ScrewGraspPlan(
            plane=plane,
            screw_tip_local_z_m=self.screw_tip_local_z_m,
            pregrasp_targets_local={
                key: value.copy() for key, value in pregrasp.items()
            },
            contact_targets_local={
                key: value.copy() for key, value in contact.items()
            },
            surface_points_local={
                key: value.copy() for key, value in surface.items()
            },
            outward_directions_local={
                key: value.copy() for key, value in normals.items()
            },
            score=float(score),
            valid=True,
            diagnostics={
                "mean_normal_alignment": mean_alignment,
                "mean_normal_z": mean_normal_z,
                "lift_support_score": lift_support,
                "ik_margin": ik_margin,
                "symmetry_error_m": symmetry_error,
                "pregrasp_diagonal_error_m": diagonal_error,
                "u_half_diagonal_m": u_half,
                "v_half_diagonal_m": v_half,
                "outside_socket_at_full_insertion_m": outside_after_full_insertion,
            },
        )

    def plan(
        self,
        *,
        ik=None,
        body_world_pos: Optional[Sequence[float]] = None,
        body_world_quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        table_top_z: float = 0.260,
        table_clearance_m: float = 0.002,
    ) -> ScrewGraspPlan:
        candidates = [
            self._build_candidate(
                plane,
                ik=ik,
                body_world_pos=body_world_pos,
                body_world_quat=body_world_quat,
                table_top_z=table_top_z,
                table_clearance_m=table_clearance_m,
            )
            for plane in self.candidate_planes()
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
                "no feasible adaptive screw grasp plane: " + ", ".join(reasons)
            )
        return max(valid, key=lambda candidate: candidate.score)


def default_screw_grasp_plan() -> ScrewGraspPlan:
    return AdaptiveScrewGraspPlanePlanner().plan()


__all__ = [
    "AdaptiveScrewGraspPlanePlanner",
    "FINGERS",
    "GraspPlane",
    "HARD_TIP_RADIUS_M",
    "PREGRASP_CLEARANCE_M",
    "ScrewGraspPlan",
    "default_screw_grasp_plan",
]
