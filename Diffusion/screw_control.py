#!/usr/bin/env python3
"""Task-space command representation and deterministic screw expert.

The learned policies never command the sixteen actuators directly.  They
produce bounded task-space increments which are projected onto this expert's
legal contact schedule and then passed through the analytic T6 IK.

The screw local frame is:

* local ``+Z``: from the shaft tip towards the head (out of the socket);
* local ``-Z``: insertion direction;
* local ``+X``: the red marker direction at zero accumulated rotation.

``twist_rad`` is positive for the requested clockwise motion as viewed from
the screw head.  Consequently the physical local-Z rotation is ``-twist``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


FINGERS: Tuple[str, ...] = ("U", "D", "L", "R")
BASE_FINGER_ANGLES_RAD: Dict[str, float] = {
    "U": 0.5 * math.pi,
    "D": -0.5 * math.pi,
    "L": math.pi,
    "R": 0.0,
}


class TaskPhase(IntEnum):
    PREGRASP = 0
    CLOSE = 1
    LIFT = 2
    ALIGN = 3
    INSERT = 4
    ENGAGE = 5
    POWER = 6
    LR_OPEN = 7
    LR_RESET = 8
    LR_CLOSE = 9
    UD_OPEN = 10
    UD_RESET = 11
    UD_CLOSE = 12
    FINAL_HOLD = 13
    COMPLETE = 14


PHASE_COUNT = len(TaskPhase)

# Diffusion action: object tip displacement in the socket frame, incremental
# object rotation in the socket frame, four material-contact phase increments,
# and four fingertip centre-distance increments.
ACTION_NAMES: Tuple[str, ...] = (
    "tip_dx_socket",
    "tip_dy_socket",
    "tip_ds_socket",
    "rotation_dx_socket",
    "rotation_dy_socket",
    "rotation_dz_socket",
    "phase_U",
    "phase_D",
    "phase_L",
    "phase_R",
    "distance_U",
    "distance_D",
    "distance_L",
    "distance_R",
)
ACTION_DIM = len(ACTION_NAMES)

# Fixed scripted prefix requested by the task.  Keeping the step counts
# explicit lets both dataset samplers exclude pickup from Diffusion training.
SCRIPTED_PREGRASP_STEPS = 25
SCRIPTED_CLOSE_STEPS = 40
SCRIPTED_PICKUP_LIFT_STEPS = 50
SCRIPTED_CENTER_TRANSFER_STEPS = 100
SCRIPTED_PREFIX_STEPS = (
    SCRIPTED_PREGRASP_STEPS
    + SCRIPTED_CLOSE_STEPS
    + SCRIPTED_PICKUP_LIFT_STEPS
    + SCRIPTED_CENTER_TRANSFER_STEPS
)

# Versioned task/controller contract for the full 20 mm socket orbit.  The
# fourfold gripper symmetry is used to keep the canonical socket residual yaw
# inside one positive 90-degree sector.  A local roll is applied only after
# the original ScrewExpert interpolation, so the validated ALIGN trajectory
# is not replaced by a different endpoint slerp.
SOCKET_YAW_EQUIVARIANCE_VERSION = 1
SOCKET_YAW_QUADRANT_RAD = 0.5 * math.pi
SOCKET_YAW_ROLL_GAIN = 0.8
SOCKET_ALIGN_STEPS = 150
RELEASED_TIP_CENTER_DISTANCE_M = 0.049

_FINGER_QUADRANT_CYCLE: Tuple[str, ...] = ("R", "U", "L", "D")
_FINGER_CYCLE_INDEX = {
    finger: index for index, finger in enumerate(_FINGER_QUADRANT_CYCLE)
}


def finger_quadrant_permutation(quadrant: int) -> Dict[str, str]:
    """Map canonical logical fingers to the rotated physical fingers."""

    shift = int(quadrant) % 4
    return {
        finger: _FINGER_QUADRANT_CYCLE[
            (_FINGER_CYCLE_INDEX[finger] + shift) % 4
        ]
        for finger in FINGERS
    }


def permute_regrasp_phase(
    phase: TaskPhase, quadrant: int
) -> TaskPhase:
    """Keep phase/holding-pair semantics aligned with physical fingers."""

    value = TaskPhase(phase)
    if int(quadrant) % 2 == 0:
        return value
    swaps = {
        TaskPhase.LR_OPEN: TaskPhase.UD_OPEN,
        TaskPhase.LR_RESET: TaskPhase.UD_RESET,
        TaskPhase.LR_CLOSE: TaskPhase.UD_CLOSE,
        TaskPhase.UD_OPEN: TaskPhase.LR_OPEN,
        TaskPhase.UD_RESET: TaskPhase.LR_RESET,
        TaskPhase.UD_CLOSE: TaskPhase.LR_CLOSE,
    }
    return swaps.get(value, value)


def _normalize(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return value / norm


def skew(vector: Sequence[float]) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_exp(rotvec: Sequence[float]) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    if theta <= 1e-12:
        return np.eye(3, dtype=np.float64) + skew(vector)
    axis = vector / theta
    K = skew(axis)
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(theta) * K
        + (1.0 - math.cos(theta)) * (K @ K)
    )


def rotation_log(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip(0.5 * (np.trace(matrix) - 1.0), -1.0, 1.0))
    theta = math.acos(cosine)
    if theta <= 1e-10:
        return np.array(
            [
                0.5 * (matrix[2, 1] - matrix[1, 2]),
                0.5 * (matrix[0, 2] - matrix[2, 0]),
                0.5 * (matrix[1, 0] - matrix[0, 1]),
            ],
            dtype=np.float64,
        )
    if math.pi - theta <= 1e-6:
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (matrix + np.eye(3, dtype=np.float64))
        )
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        return _normalize(axis) * theta
    scale = theta / (2.0 * math.sin(theta))
    return scale * np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )


def rot_z(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def align_local_z_to(direction_world: Sequence[float]) -> np.ndarray:
    """Return a deterministic right-handed frame with local Z on direction."""

    z_axis = _normalize(direction_world)
    reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(reference, z_axis))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = _normalize(np.cross(reference, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def matrix_to_quaternion_wxyz(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * math.sqrt(
                max(1e-12, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            )
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * math.sqrt(
                max(1e-12, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            )
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * math.sqrt(
                max(1e-12, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            )
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion.astype(np.float64)


def quaternion_slerp_wxyz(
    first: Sequence[float],
    second: Sequence[float],
    fraction: float,
) -> np.ndarray:
    q0 = np.asarray(first, dtype=np.float64).reshape(4)
    q1 = np.asarray(second, dtype=np.float64).reshape(4)
    q0 /= max(float(np.linalg.norm(q0)), 1e-12)
    q1 /= max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    value = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        result = q0 + value * (q1 - q0)
        return result / max(float(np.linalg.norm(result)), 1e-12)
    theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
    denominator = math.sin(theta)
    return (
        math.sin((1.0 - value) * theta) / denominator * q0
        + math.sin(value * theta) / denominator * q1
    )


def quaternion_to_matrix_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def minimum_jerk(fraction: float) -> float:
    value = float(np.clip(fraction, 0.0, 1.0))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


@dataclass
class ScrewCommand:
    tip_world: np.ndarray
    rotation_world: np.ndarray
    twist_rad: float
    material_angles_rad: Dict[str, float]
    tip_center_distances_m: Dict[str, float]
    phase: TaskPhase
    engaged: bool = False

    def copy(self) -> "ScrewCommand":
        return ScrewCommand(
            tip_world=np.asarray(self.tip_world, dtype=np.float64).copy(),
            rotation_world=np.asarray(self.rotation_world, dtype=np.float64).copy(),
            twist_rad=float(self.twist_rad),
            material_angles_rad=dict(self.material_angles_rad),
            tip_center_distances_m=dict(self.tip_center_distances_m),
            phase=TaskPhase(self.phase),
            engaged=bool(self.engaged),
        )

    @property
    def quaternion_wxyz(self) -> np.ndarray:
        return matrix_to_quaternion_wxyz(self.rotation_world)

    def fingertip_targets(
        self,
        *,
        grasp_height_from_tip_m: float,
    ) -> Dict[str, np.ndarray]:
        targets: Dict[str, np.ndarray] = {}
        for finger in FINGERS:
            angle = float(self.material_angles_rad[finger])
            distance = float(self.tip_center_distances_m[finger])
            local = np.array(
                [
                    distance * math.cos(angle),
                    distance * math.sin(angle),
                    float(grasp_height_from_tip_m),
                ],
                dtype=np.float64,
            )
            targets[finger] = (
                np.asarray(self.tip_world, dtype=np.float64)
                + np.asarray(self.rotation_world, dtype=np.float64) @ local
            )
        return targets


@dataclass(frozen=True)
class _Segment:
    start: ScrewCommand
    end: ScrewCommand
    steps: int


class ScrewExpert:
    """Deterministic teacher for demonstrations and the residual baseline."""

    def __init__(
        self,
        *,
        spawn_xy_m: Sequence[float],
        table_top_z_m: float,
        socket_mouth_world_m: Sequence[float],
        socket_axis_inward_world: Sequence[float],
        contact_distance_m: float = 0.0410,
        pregrasp_distance_m: float = 0.053,
        released_distance_m: float = RELEASED_TIP_CENTER_DISTANCE_M,
        grasp_plane_yaw_rad: float = 0.0,
        engage_depth_m: float = 0.070,
        final_depth_m: float = 0.080,
        pitch_m: float = 0.005,
        target_turns: float = 2.0,
        stroke_deg: float = 40.0,
        grasp_height_from_tip_m: float = 0.100,
        scripted_grasp_plane_world_z_m: float = 0.420,
        scripted_center_xy_m: Sequence[float] = (0.0, 0.0),
    ) -> None:
        actual_spawn_xy = np.asarray(
            spawn_xy_m, dtype=np.float64
        ).reshape(2)
        actual_mouth = np.asarray(
            socket_mouth_world_m, dtype=np.float64
        ).reshape(3)
        actual_axis = _normalize(socket_axis_inward_world)
        actual_scripted_center = np.asarray(
            scripted_center_xy_m, dtype=np.float64
        ).reshape(2)

        radial = actual_mouth[:2] - actual_scripted_center
        if float(np.linalg.norm(radial)) <= 1.0e-12:
            socket_yaw = 0.0
        else:
            socket_yaw = math.atan2(float(radial[1]), float(radial[0]))
            if socket_yaw < 0.0:
                socket_yaw += 2.0 * math.pi
        quadrant_coordinate = socket_yaw / SOCKET_YAW_QUADRANT_RAD
        nearest_cardinal = round(quadrant_coordinate)
        if math.isclose(
            quadrant_coordinate,
            nearest_cardinal,
            abs_tol=1.0e-12,
        ):
            quadrant = int(nearest_cardinal) % 4
            residual_yaw = 0.0
        else:
            quadrant = int(math.floor(quadrant_coordinate)) % 4
            residual_yaw = (
                socket_yaw - quadrant * SOCKET_YAW_QUADRANT_RAD
            )
        quadrant_yaw = quadrant * SOCKET_YAW_QUADRANT_RAD

        self.socket_yaw_rad = float(socket_yaw)
        self.socket_quadrant = int(quadrant)
        self.socket_quadrant_yaw_rad = float(quadrant_yaw)
        self.socket_residual_yaw_rad = float(residual_yaw)
        self.socket_roll_rad = (
            SOCKET_YAW_ROLL_GAIN * self.socket_residual_yaw_rad
        )
        self.quadrant_rotation_world = rot_z(
            self.socket_quadrant_yaw_rad
        )
        self.finger_permutation = finger_quadrant_permutation(
            self.socket_quadrant
        )

        canonical_rotation = self.quadrant_rotation_world.T
        self.spawn_xy_m = (
            canonical_rotation[:2, :2] @ actual_spawn_xy
        )
        self.table_top_z_m = float(table_top_z_m)
        self.mouth = canonical_rotation @ actual_mouth
        self.axis_inward = _normalize(canonical_rotation @ actual_axis)
        self.contact_distance_m = float(contact_distance_m)
        self.pregrasp_distance_m = float(pregrasp_distance_m)
        self.released_distance_m = float(released_distance_m)
        self.grasp_plane_yaw_rad = float(grasp_plane_yaw_rad)
        self.engage_depth_m = float(engage_depth_m)
        self.final_depth_m = float(final_depth_m)
        self.pitch_m = float(pitch_m)
        self.grasp_height_from_tip_m = float(grasp_height_from_tip_m)
        self.scripted_grasp_plane_world_z_m = float(
            scripted_grasp_plane_world_z_m
        )
        self.scripted_center_xy_m = (
            canonical_rotation[:2, :2] @ actual_scripted_center
        )
        self.scripted_center_tip_world_z_m = (
            self.scripted_grasp_plane_world_z_m
            - self.grasp_height_from_tip_m
        )
        if self.scripted_center_tip_world_z_m <= self.table_top_z_m:
            raise ValueError(
                "scripted centre handoff must lift the screw tip above the table"
            )
        self.target_twist_rad = float(target_turns) * 2.0 * math.pi
        self.stroke_rad = math.radians(float(stroke_deg))
        if not math.isclose(
            self.engage_depth_m
            + self.pitch_m * self.target_twist_rad / (2.0 * math.pi),
            self.final_depth_m,
            abs_tol=1e-10,
        ):
            raise ValueError("pitch, turns and engagement depth do not reach final depth")
        if self.stroke_rad <= 0.0:
            raise ValueError("stroke must be positive")
        stroke_count = self.target_twist_rad / self.stroke_rad
        if not math.isclose(stroke_count, round(stroke_count), abs_tol=1e-9):
            raise ValueError("target twist must be an integer number of strokes")
        self.stroke_count = int(round(stroke_count))
        self._canonical_socket_frame_world = align_local_z_to(
            self.axis_inward
        )
        self.socket_frame_world = (
            self.quadrant_rotation_world
            @ self._canonical_socket_frame_world
            @ self.quadrant_rotation_world.T
        )
        self._segments = self._build_segments()
        self._segment_index = 0
        self._step_in_segment = 0
        self._elapsed_steps = 0
        self._last_command = self._segments[0].start.copy()

    def _distances(self, value: float) -> Dict[str, float]:
        return {finger: float(value) for finger in FINGERS}

    def _initial_command(self, distance: float, phase: TaskPhase) -> ScrewCommand:
        return ScrewCommand(
            tip_world=np.array(
                [self.spawn_xy_m[0], self.spawn_xy_m[1], self.table_top_z_m],
                dtype=np.float64,
            ),
            rotation_world=np.eye(3, dtype=np.float64),
            twist_rad=0.0,
            material_angles_rad={
                finger: angle + self.grasp_plane_yaw_rad
                for finger, angle in BASE_FINGER_ANGLES_RAD.items()
            },
            tip_center_distances_m=self._distances(distance),
            phase=phase,
            engaged=False,
        )

    @staticmethod
    def _changed(command: ScrewCommand, *, phase: TaskPhase, **changes) -> ScrewCommand:
        result = command.copy()
        result.phase = phase
        for key, value in changes.items():
            setattr(result, key, value)
        return result

    def _append(
        self,
        segments: list[_Segment],
        end: ScrewCommand,
        steps: int,
    ) -> ScrewCommand:
        start = segments[-1].end.copy() if segments else self._initial_command(
            self.pregrasp_distance_m, TaskPhase.PREGRASP
        )
        segments.append(_Segment(start=start, end=end.copy(), steps=max(1, int(steps))))
        return end

    def _build_segments(self) -> Tuple[_Segment, ...]:
        segments: list[_Segment] = []
        current = self._initial_command(
            self.pregrasp_distance_m, TaskPhase.PREGRASP
        )
        segments.append(
            _Segment(
                current.copy(),
                current.copy(),
                SCRIPTED_PREGRASP_STEPS,
            )
        )

        current = self._changed(
            current,
            phase=TaskPhase.CLOSE,
            tip_center_distances_m=self._distances(self.contact_distance_m),
        )
        current = self._append(segments, current, SCRIPTED_CLOSE_STEPS)

        # Scripted prefix requested by the task: first raise the fixed
        # head-centre grasp plane to world z=0.420 m while the screw remains
        # upright, then translate the upright screw to the world centre.
        lifted_tip = current.tip_world.copy()
        lifted_tip[2] = self.scripted_center_tip_world_z_m
        current = self._changed(
            current,
            phase=TaskPhase.LIFT,
            tip_world=lifted_tip,
        )
        current = self._append(
            segments, current, SCRIPTED_PICKUP_LIFT_STEPS
        )

        centred_tip = np.array(
            (
                self.scripted_center_xy_m[0],
                self.scripted_center_xy_m[1],
                self.scripted_center_tip_world_z_m,
            ),
            dtype=np.float64,
        )
        current = self._changed(
            current,
            phase=TaskPhase.LIFT,
            tip_world=centred_tip,
            rotation_world=np.eye(3, dtype=np.float64),
        )
        current = self._append(
            segments, current, SCRIPTED_CENTER_TRANSFER_STEPS
        )
        # This exact table index is a feedback boundary.  The environment
        # holds here until the measured screw pose reaches the centre target;
        # Diffusion and the residual policy are masked before that event.
        self.diffusion_handoff_step = sum(
            segment.steps for segment in segments
        )
        if self.diffusion_handoff_step != SCRIPTED_PREFIX_STEPS:
            raise RuntimeError("scripted prefix step contract drifted")

        # The T6 installation cannot keep all four fingertips on the head
        # while changing from vertical to a tilted socket pose.  Perform
        # a deliberate pair-wise pre-alignment regrasp. U/D make a small
        # common +25 degree shift while L/R hold; then L/R are shifted by
        # -70 degrees while U/D hold. This avoids crossing U/D around the
        # object (which can eject it on contact), while retaining the
        # IK-reachable grasp throughout the complete approach.
        ud_open = dict(current.tip_center_distances_m)
        for finger in ("U", "D"):
            ud_open[finger] = self.released_distance_m
        current = self._changed(
            current,
            phase=TaskPhase.UD_OPEN,
            tip_center_distances_m=ud_open,
        )
        current = self._append(segments, current, 10)

        ud_shifted = dict(current.material_angles_rad)
        # Keep the validated GRAB/regrasp schedule fixed.  Only the socket
        # mouth azimuth is randomized; it must not silently change grasp
        # material angles or any other controller setting.
        prealign_ud_shift = math.radians(25.0)
        for finger in ("U", "D"):
            ud_shifted[finger] += prealign_ud_shift
        current = self._changed(
            current,
            phase=TaskPhase.UD_RESET,
            material_angles_rad=ud_shifted,
        )
        current = self._append(segments, current, 25)

        ud_closed = dict(current.tip_center_distances_m)
        for finger in ("U", "D"):
            ud_closed[finger] = self.contact_distance_m
        current = self._changed(
            current,
            phase=TaskPhase.UD_CLOSE,
            tip_center_distances_m=ud_closed,
        )
        # Allow the jerk-limited closed-chain controller to establish the new
        # opposing contact pair before the currently holding pair releases.
        current = self._append(segments, current, 40)

        lr_open = dict(current.tip_center_distances_m)
        for finger in ("L", "R"):
            lr_open[finger] = self.released_distance_m
        lr_shifted = dict(current.material_angles_rad)
        prealign_shift = math.radians(-70.0)
        for finger in ("L", "R"):
            lr_shifted[finger] += prealign_shift
        current = self._changed(
            current,
            phase=TaskPhase.LR_OPEN,
            tip_center_distances_m=lr_open,
            material_angles_rad=lr_shifted,
        )
        current = self._append(segments, current, 25)

        lr_closed = dict(current.tip_center_distances_m)
        for finger in ("L", "R"):
            lr_closed[finger] = self.contact_distance_m
        current = self._changed(
            current,
            phase=TaskPhase.LR_CLOSE,
            tip_center_distances_m=lr_closed,
        )
        current = self._append(segments, current, 40)

        # L/R now hold the object during the large pose change. U/D remain
        # clear until the screw reaches the 70 mm engagement pose.
        ud_open = dict(current.tip_center_distances_m)
        for finger in ("U", "D"):
            ud_open[finger] = self.released_distance_m
        current = self._changed(
            current,
            phase=TaskPhase.UD_OPEN,
            tip_center_distances_m=ud_open,
        )
        current = self._append(segments, current, 10)

        # Keep the tip outside the socket while the screw is aligned, but do
        # not carry over the old 40 mm standoff: with the fixed 100 mm
        # tip-to-grasp-plane distance, that standoff puts the L/R holding pair
        # outside the T6 workspace before insertion begins.  A 15 mm
        # standoff still leaves the shaft tip clearly outside the mouth.
        preinsert_tip = self.mouth - 0.015 * self.axis_inward
        aligned_rotation = align_local_z_to(-self.axis_inward)
        # The yaw-equivalent roll is deliberately applied later by
        # ``_to_world_command`` to the already-interpolated command.  Record
        # the original ALIGN interpolation boundary here so its exact
        # minimum-jerk schedule remains unchanged.
        self._align_roll_boundary_step = sum(
            segment.steps for segment in segments
        )
        current = self._changed(
            current,
            phase=TaskPhase.ALIGN,
            tip_world=preinsert_tip,
            rotation_world=aligned_rotation,
        )
        current = self._append(
            segments, current, SOCKET_ALIGN_STEPS
        )
        self._align_roll_end_step = (
            self._align_roll_boundary_step + SOCKET_ALIGN_STEPS
        )

        engaged_tip = self.mouth + self.engage_depth_m * self.axis_inward
        current = self._changed(
            current,
            phase=TaskPhase.INSERT,
            tip_world=engaged_tip,
        )
        current = self._append(segments, current, 110)

        ud_closed = dict(current.tip_center_distances_m)
        for finger in ("U", "D"):
            ud_closed[finger] = self.contact_distance_m
        current = self._changed(
            current,
            phase=TaskPhase.UD_CLOSE,
            tip_center_distances_m=ud_closed,
        )
        current = self._append(segments, current, 15)

        current = self._changed(
            current,
            phase=TaskPhase.ENGAGE,
            engaged=True,
        )
        current = self._append(segments, current, 25)

        cumulative_twist = 0.0
        for _cycle in range(self.stroke_count):
            cumulative_twist += self.stroke_rad
            depth = (
                self.engage_depth_m
                + self.pitch_m * cumulative_twist / (2.0 * math.pi)
            )
            current = self._changed(
                current,
                phase=TaskPhase.POWER,
                tip_world=self.mouth + depth * self.axis_inward,
                rotation_world=aligned_rotation @ rot_z(-cumulative_twist),
                twist_rad=cumulative_twist,
                engaged=True,
            )
            current = self._append(segments, current, 30)

            lr_open = dict(current.tip_center_distances_m)
            for finger in ("L", "R"):
                lr_open[finger] = self.released_distance_m
            current = self._changed(
                current,
                phase=TaskPhase.LR_OPEN,
                tip_center_distances_m=lr_open,
            )
            current = self._append(segments, current, 10)

            lr_angles = dict(current.material_angles_rad)
            for finger in ("L", "R"):
                lr_angles[finger] += self.stroke_rad
            current = self._changed(
                current,
                phase=TaskPhase.LR_RESET,
                material_angles_rad=lr_angles,
            )
            current = self._append(segments, current, 15)

            lr_closed = dict(current.tip_center_distances_m)
            for finger in ("L", "R"):
                lr_closed[finger] = self.contact_distance_m
            current = self._changed(
                current,
                phase=TaskPhase.LR_CLOSE,
                tip_center_distances_m=lr_closed,
            )
            current = self._append(segments, current, 10)

            ud_open = dict(current.tip_center_distances_m)
            for finger in ("U", "D"):
                ud_open[finger] = self.released_distance_m
            current = self._changed(
                current,
                phase=TaskPhase.UD_OPEN,
                tip_center_distances_m=ud_open,
            )
            current = self._append(segments, current, 10)

            ud_angles = dict(current.material_angles_rad)
            for finger in ("U", "D"):
                ud_angles[finger] += self.stroke_rad
            current = self._changed(
                current,
                phase=TaskPhase.UD_RESET,
                material_angles_rad=ud_angles,
            )
            current = self._append(segments, current, 15)

            ud_closed = dict(current.tip_center_distances_m)
            for finger in ("U", "D"):
                ud_closed[finger] = self.contact_distance_m
            current = self._changed(
                current,
                phase=TaskPhase.UD_CLOSE,
                tip_center_distances_m=ud_closed,
            )
            current = self._append(segments, current, 10)

        current = self._changed(current, phase=TaskPhase.FINAL_HOLD)
        self._append(segments, current, 50)
        return tuple(segments)

    @property
    def total_steps(self) -> int:
        return sum(segment.steps for segment in self._segments)

    @property
    def complete(self) -> bool:
        return self._segment_index >= len(self._segments)

    @property
    def step_index(self) -> int:
        return self._elapsed_steps

    @property
    def learned_control_active(self) -> bool:
        return self._elapsed_steps > self.diffusion_handoff_step

    def _socket_roll_at_step(self, step: int) -> float:
        value = int(step)
        if value <= self._align_roll_boundary_step:
            return 0.0
        fraction = (
            value - self._align_roll_boundary_step
        ) / float(SOCKET_ALIGN_STEPS)
        return self.socket_roll_rad * minimum_jerk(fraction)

    def _to_world_command(
        self, canonical: ScrewCommand, step: int
    ) -> ScrewCommand:
        """Apply quadrant equivariance after the original interpolation."""

        Q = self.quadrant_rotation_world
        rotation = Q @ canonical.rotation_world @ Q.T
        roll = self._socket_roll_at_step(step)
        if abs(roll) > 1.0e-15:
            rotation = rotation @ rot_z(roll)

        material: Dict[str, float] = {}
        distances: Dict[str, float] = {}
        for logical in FINGERS:
            physical = self.finger_permutation[logical]
            material[physical] = (
                float(canonical.material_angles_rad[logical])
                + self.socket_quadrant_yaw_rad
            )
            distances[physical] = float(
                canonical.tip_center_distances_m[logical]
            )

        return ScrewCommand(
            tip_world=Q @ np.asarray(
                canonical.tip_world, dtype=np.float64
            ),
            rotation_world=rotation,
            twist_rad=float(canonical.twist_rad),
            material_angles_rad=material,
            tip_center_distances_m=distances,
            phase=permute_regrasp_phase(
                canonical.phase, self.socket_quadrant
            ),
            engaged=bool(canonical.engaged),
        )

    @property
    def current(self) -> ScrewCommand:
        return self._to_world_command(
            self._last_command.copy(), self._elapsed_steps
        )

    def reset(self) -> ScrewCommand:
        self._segment_index = 0
        self._step_in_segment = 0
        self._elapsed_steps = 0
        self._last_command = self._segments[0].start.copy()
        return self.current

    def _interpolate(
        self,
        first: ScrewCommand,
        second: ScrewCommand,
        fraction: float,
    ) -> ScrewCommand:
        value = minimum_jerk(fraction)
        first_quaternion = first.quaternion_wxyz
        second_quaternion = second.quaternion_wxyz
        quaternion = quaternion_slerp_wxyz(
            first_quaternion, second_quaternion, value
        )
        return ScrewCommand(
            tip_world=(1.0 - value) * first.tip_world + value * second.tip_world,
            rotation_world=quaternion_to_matrix_wxyz(quaternion),
            twist_rad=(1.0 - value) * first.twist_rad + value * second.twist_rad,
            material_angles_rad={
                finger: (
                    (1.0 - value) * first.material_angles_rad[finger]
                    + value * second.material_angles_rad[finger]
                )
                for finger in FINGERS
            },
            tip_center_distances_m={
                finger: (
                    (1.0 - value) * first.tip_center_distances_m[finger]
                    + value * second.tip_center_distances_m[finger]
                )
                for finger in FINGERS
            },
            phase=second.phase,
            engaged=second.engaged,
        )

    def advance(self) -> ScrewCommand:
        if self.complete:
            result = self._last_command.copy()
            result.phase = TaskPhase.COMPLETE
            return self._to_world_command(result, self._elapsed_steps)
        segment = self._segments[self._segment_index]
        self._step_in_segment += 1
        fraction = self._step_in_segment / segment.steps
        self._last_command = self._interpolate(
            segment.start, segment.end, fraction
        )
        if self._step_in_segment >= segment.steps:
            self._segment_index += 1
            self._step_in_segment = 0
        self._elapsed_steps += 1
        return self.current

    def phase_one_hot(self) -> np.ndarray:
        vector = np.zeros(PHASE_COUNT, dtype=np.float32)
        phase = TaskPhase.COMPLETE if self.complete else self.current.phase
        vector[int(phase)] = 1.0
        return vector

    def command_action(
        self,
        previous: ScrewCommand,
        following: ScrewCommand,
    ) -> np.ndarray:
        """Encode a command transition in the socket frame."""

        socket_rotation = self.socket_frame_world
        delta_position = socket_rotation.T @ (
            np.asarray(following.tip_world) - np.asarray(previous.tip_world)
        )
        world_delta_rotation = (
            np.asarray(following.rotation_world)
            @ np.asarray(previous.rotation_world).T
        )
        socket_delta_rotation = (
            socket_rotation.T @ world_delta_rotation @ socket_rotation
        )
        delta_rotation = rotation_log(socket_delta_rotation)
        phase_delta = np.array(
            [
                following.material_angles_rad[finger]
                - previous.material_angles_rad[finger]
                for finger in FINGERS
            ],
            dtype=np.float64,
        )
        distance_delta = np.array(
            [
                following.tip_center_distances_m[finger]
                - previous.tip_center_distances_m[finger]
                for finger in FINGERS
            ],
            dtype=np.float64,
        )
        return np.concatenate(
            (delta_position, delta_rotation, phase_delta, distance_delta)
        ).astype(np.float32)


def holding_fingers(phase: TaskPhase) -> Tuple[str, ...]:
    if phase in (TaskPhase.LR_OPEN, TaskPhase.LR_RESET, TaskPhase.LR_CLOSE):
        return ("U", "D")
    if phase in (TaskPhase.UD_OPEN, TaskPhase.UD_RESET, TaskPhase.UD_CLOSE):
        return ("L", "R")
    return FINGERS


def validate_contact_schedule(
    command: ScrewCommand,
    *,
    contact_distance_m: float,
    holding_tolerance_m: float = 0.002,
) -> None:
    """Raise when a command would release every opposing holding pair."""

    # The start fixture, not fingertip friction, constrains the screw during
    # pregrasp and the minimum-jerk closing transition.
    if command.phase in (TaskPhase.PREGRASP, TaskPhase.CLOSE):
        return
    closed = {
        finger
        for finger in FINGERS
        if command.tip_center_distances_m[finger]
        <= float(contact_distance_m) + float(holding_tolerance_m)
    }
    if not (
        {"U", "D"}.issubset(closed)
        or {"L", "R"}.issubset(closed)
    ):
        raise ValueError("at least one opposing finger pair must remain closed")


__all__ = [
    "ACTION_DIM",
    "ACTION_NAMES",
    "BASE_FINGER_ANGLES_RAD",
    "FINGERS",
    "PHASE_COUNT",
    "RELEASED_TIP_CENTER_DISTANCE_M",
    "SOCKET_ALIGN_STEPS",
    "SOCKET_YAW_EQUIVARIANCE_VERSION",
    "SOCKET_YAW_QUADRANT_RAD",
    "SOCKET_YAW_ROLL_GAIN",
    "ScrewCommand",
    "ScrewExpert",
    "TaskPhase",
    "align_local_z_to",
    "finger_quadrant_permutation",
    "holding_fingers",
    "matrix_to_quaternion_wxyz",
    "minimum_jerk",
    "permute_regrasp_phase",
    "rotation_exp",
    "rotation_log",
    "validate_contact_schedule",
]
