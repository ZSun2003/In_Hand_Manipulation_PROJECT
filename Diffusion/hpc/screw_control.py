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

# Collision-tested continuous-grasp trajectory.  Keeping every boundary
# explicit makes measured gates and checkpoint compatibility unambiguous.
SCRIPTED_PREGRASP_STEPS = 65
SCRIPTED_CLOSE_STEPS = 50
SCRIPTED_CLOSE_HOLD_STEPS = 10
SCRIPTED_PICKUP_LIFT_STEPS = 50
# Enter the first near-zero-motion LIFT command before waiting for measured
# fixture release. Holding at the preceding CLOSE step would deadlock because
# the fixture is intentionally forbidden from releasing in CLOSE.
PICKUP_RELEASE_GATE_STEP = (
    SCRIPTED_PREGRASP_STEPS
    + SCRIPTED_CLOSE_STEPS
    + SCRIPTED_CLOSE_HOLD_STEPS
    + 1
)
PICKUP_RELEASE_GATE_STABLE_STEPS = 10
# Kept as a zero-valued compatibility name: the former centre-transfer
# waypoint has been removed.  ALIGN now travels directly from the lifted
# spawn pose to the socket mouth.
SCRIPTED_CENTER_TRANSFER_STEPS = 0
SCRIPTED_PREFIX_STEPS = (
    SCRIPTED_PREGRASP_STEPS
    + SCRIPTED_CLOSE_STEPS
    + SCRIPTED_CLOSE_HOLD_STEPS
    + SCRIPTED_PICKUP_LIFT_STEPS
)

# The four physical finger roles and material angles remain fixed for every
# socket azimuth.  Runtime constructs the deterministic final frame directly
# from the actual socket axis; there is no quadrant permutation or extra roll.
CONTINUOUS_GRASP_STRATEGY_VERSION = (
    "pickup-h110-thread-regrasp-h112-mouth330-direct-min3-min2-thread70-blend20-v22"
)
CONTINUOUS_GRASP_DISTANCE_M = 0.0375
# Planned Diffusion-to-supervisor takeovers use a minimum-jerk task-space
# blend.  The validated old-model setting is restored: at the fixed 20 ms
# control period, 20 frames give a 0.4 s transition.
DIFFUSION_TO_ANCHOR_BLEND_STEPS = 20
RELEASED_TIP_CENTER_DISTANCE_M = 0.049
# Once the thread supports the screw, a slightly smaller reset radius avoids
# the analytic-IK boundary at two diagonal socket azimuths.  The 49 mm value
# is retained as a compatibility/controller bound; it is not commanded before
# measured thread engagement.
THREAD_REGRASP_RELEASED_DISTANCE_M = 0.047
DIRECT_ALIGN_STEPS = 350
CAPTURE_INSERT_STEPS = 110
# Keep the four established contacts during this measured in-bore dwell.  The
# post-capture *constraint* is relaxed to at least two contacts, but opening a
# pair early creates an unnecessary single-pair failure point before thread
# support is available.
CAPTURE_SUPPORT_DWELL_STEPS = 20
# Compatibility name for consumers that only use the boundary arithmetic.
CAPTURE_RELEASE_STEPS = CAPTURE_SUPPORT_DWELL_STEPS
THREAD_INSERT_STEPS = 120
THREAD_ENGAGE_STEPS = 1
THREAD_REGRASP_GRASP_HEIGHT_FROM_TIP_M = 0.112
THREAD_REGRASP_CLOSE_STEPS = 15

PREINSERT_GATE_STEP = SCRIPTED_PREFIX_STEPS + DIRECT_ALIGN_STEPS
CAPTURE_GATE_STEP = PREINSERT_GATE_STEP + CAPTURE_INSERT_STEPS
THREAD_GATE_STEP = (
    CAPTURE_GATE_STEP
    + CAPTURE_SUPPORT_DWELL_STEPS
    + THREAD_INSERT_STEPS
)
THREAD_REGRASP_STEP = THREAD_GATE_STEP + THREAD_ENGAGE_STEPS

# Compatibility aliases retained for code that serializes the previous
# controller contract.  Their values now describe the direct-alignment
# schedule, and the zero roll gain explicitly disables early roll.
YAW_EQUIVARIANT_STRATEGY_VERSION = CONTINUOUS_GRASP_STRATEGY_VERSION
YAW_EQUIVARIANT_RELEASED_DISTANCE_M = RELEASED_TIP_CENTER_DISTANCE_M
YAW_EQUIVARIANT_ROLL_GAIN = 0.0
YAW_EQUIVARIANT_ALIGN_BOUNDARY_STEP = SCRIPTED_PREFIX_STEPS
YAW_EQUIVARIANT_ALIGN_STEPS = DIRECT_ALIGN_STEPS
YAW_EQUIVARIANT_ROLL_BOUNDARY_STEP = SCRIPTED_PREFIX_STEPS
YAW_EQUIVARIANT_ROLL_STEPS = DIRECT_ALIGN_STEPS


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
        contact_distance_m: float = CONTINUOUS_GRASP_DISTANCE_M,
        pregrasp_distance_m: float = 0.053,
        released_distance_m: float = RELEASED_TIP_CENTER_DISTANCE_M,
        grasp_plane_yaw_rad: float = 0.0,
        engage_depth_m: float = 0.070,
        capture_depth_m: float = 0.030,
        final_depth_m: float = 0.080,
        pitch_m: float = 0.005,
        target_turns: float = 2.0,
        stroke_deg: float = 40.0,
        grasp_height_from_tip_m: float = 0.110,
        scripted_grasp_plane_world_z_m: float = 0.420,
        scripted_center_xy_m: Sequence[float] = (0.0, 0.0),
        preinsert_standoff_m: float = 0.005,
    ) -> None:
        self.spawn_xy_m = np.asarray(spawn_xy_m, dtype=np.float64).reshape(2)
        self.table_top_z_m = float(table_top_z_m)
        self.mouth = np.asarray(socket_mouth_world_m, dtype=np.float64).reshape(3)
        self.axis_inward = _normalize(socket_axis_inward_world)
        self.contact_distance_m = float(contact_distance_m)
        self.pregrasp_distance_m = float(pregrasp_distance_m)
        self.released_distance_m = float(released_distance_m)
        self.grasp_plane_yaw_rad = float(grasp_plane_yaw_rad)
        self.engage_depth_m = float(engage_depth_m)
        self.capture_depth_m = float(capture_depth_m)
        self.final_depth_m = float(final_depth_m)
        self.pitch_m = float(pitch_m)
        self.grasp_height_from_tip_m = float(grasp_height_from_tip_m)
        self.scripted_grasp_plane_world_z_m = float(
            scripted_grasp_plane_world_z_m
        )
        self.scripted_center_xy_m = np.asarray(
            scripted_center_xy_m, dtype=np.float64
        ).reshape(2)
        self.preinsert_standoff_m = float(preinsert_standoff_m)
        self.scripted_center_tip_world_z_m = (
            self.scripted_grasp_plane_world_z_m
            - self.grasp_height_from_tip_m
        )
        if self.scripted_center_tip_world_z_m <= self.table_top_z_m:
            raise ValueError(
                "scripted lifted-spawn handoff must keep the tip above the table"
            )
        if (
            not math.isfinite(self.preinsert_standoff_m)
            or self.preinsert_standoff_m <= 0.0
        ):
            raise ValueError("preinsert standoff must be finite and positive")
        if not 0.0 < self.capture_depth_m < self.engage_depth_m:
            raise ValueError(
                "capture depth must be positive and precede engagement"
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
        self.socket_frame_world = align_local_z_to(self.axis_inward)
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

        # Establish five-plus measured all-four contact frames before the
        # start fixture may release.  This zero-motion segment is explicit so
        # the schedule cannot accidentally shorten the preload dwell.
        current = self._append(
            segments, current, SCRIPTED_CLOSE_HOLD_STEPS
        )

        # Raise the H=110 mm grasp plane to world z=0.420 m, leaving the screw
        # tip at z=0.310 m.  There is no intermediate centre-transfer waypoint.
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

        # The learned suffix begins only after the fixed pickup/lift is
        # complete.  ALIGN then moves directly from this lifted spawn pose.
        self.diffusion_handoff_step = sum(
            segment.steps for segment in segments
        )
        if self.diffusion_handoff_step != SCRIPTED_PREFIX_STEPS:
            raise RuntimeError("scripted prefix step contract drifted")

        # Keep all four physical fingers at fixed material angles and fixed
        # 39 mm preload while position and orientation change together.
        preinsert_tip = (
            self.mouth - self.preinsert_standoff_m * self.axis_inward
        )
        aligned_rotation = align_local_z_to(-self.axis_inward)
        current = self._changed(
            current,
            phase=TaskPhase.ALIGN,
            tip_world=preinsert_tip,
            rotation_world=aligned_rotation,
        )
        current = self._append(segments, current, DIRECT_ALIGN_STEPS)
        self.preinsert_gate_step = sum(
            segment.steps for segment in segments
        )
        if self.preinsert_gate_step != PREINSERT_GATE_STEP:
            raise RuntimeError("preinsert gate step contract drifted")

        # Enter the bore with all four contacts intact, then hold the endpoint
        # until the measured 30 mm capture gate is stable.
        capture_tip = (
            self.mouth + self.capture_depth_m * self.axis_inward
        )
        current = self._changed(
            current,
            phase=TaskPhase.INSERT,
            tip_world=capture_tip,
        )
        current = self._append(segments, current, CAPTURE_INSERT_STEPS)
        self.capture_gate_step = sum(
            segment.steps for segment in segments
        )
        if self.capture_gate_step != CAPTURE_GATE_STEP:
            raise RuntimeError("capture gate step contract drifted")

        # Capture grants permission to rely on any two measured contacts, but
        # there is no clearance reason to open a pair yet: at the 70 mm thread
        # gate the 110 mm grasp plane remains 40 mm outside the mouth and the
        # 16 mm tips retain 14 mm axial clearance.  Preserve all four material
        # points and the 39 mm preload until measured thread engagement.
        current = self._append(
            segments, current, CAPTURE_SUPPORT_DWELL_STEPS
        )

        # Continue on the same centre line with all four fingers commanded
        # closed.  The runtime safety contract may tolerate any two measured
        # contacts after capture.  Leaving this INSERT segment is guarded by
        # the measured thread-ready gate.
        engaged_tip = self.mouth + self.engage_depth_m * self.axis_inward
        current = self._changed(
            current,
            phase=TaskPhase.INSERT,
            tip_world=engaged_tip,
        )
        current = self._append(segments, current, THREAD_INSERT_STEPS)
        self.thread_gate_step = sum(
            segment.steps for segment in segments
        )
        if self.thread_gate_step != THREAD_GATE_STEP:
            raise RuntimeError("thread gate step contract drifted")

        # Engagement must be active before any close/reset/regrasp command.
        current = self._changed(
            current,
            phase=TaskPhase.ENGAGE,
            engaged=True,
        )
        current = self._append(
            segments, current, THREAD_ENGAGE_STEPS
        )
        self.thread_regrasp_step = sum(
            segment.steps for segment in segments
        )
        if self.thread_regrasp_step != THREAD_REGRASP_STEP:
            raise RuntimeError("thread regrasp step contract drifted")

        ud_closed = dict(current.tip_center_distances_m)
        for finger in ("U", "D"):
            ud_closed[finger] = self.contact_distance_m
        current = self._changed(
            current,
            phase=TaskPhase.UD_CLOSE,
            tip_center_distances_m=ud_closed,
        )
        current = self._append(
            segments, current, THREAD_REGRASP_CLOSE_STEPS
        )

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
                lr_open[finger] = THREAD_REGRASP_RELEASED_DISTANCE_M
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
                ud_open[finger] = THREAD_REGRASP_RELEASED_DISTANCE_M
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

    @property
    def current(self) -> ScrewCommand:
        return self._last_command.copy()

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
            return result
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
    "CAPTURE_GATE_STEP",
    "CAPTURE_INSERT_STEPS",
    "CAPTURE_SUPPORT_DWELL_STEPS",
    "CAPTURE_RELEASE_STEPS",
    "CONTINUOUS_GRASP_DISTANCE_M",
    "THREAD_REGRASP_GRASP_HEIGHT_FROM_TIP_M",
    "CONTINUOUS_GRASP_STRATEGY_VERSION",
    "DIFFUSION_TO_ANCHOR_BLEND_STEPS",
    "DIRECT_ALIGN_STEPS",
    "FINGERS",
    "PHASE_COUNT",
    "PICKUP_RELEASE_GATE_STABLE_STEPS",
    "PICKUP_RELEASE_GATE_STEP",
    "SCRIPTED_CLOSE_STEPS",
    "SCRIPTED_CLOSE_HOLD_STEPS",
    "SCRIPTED_CENTER_TRANSFER_STEPS",
    "SCRIPTED_PICKUP_LIFT_STEPS",
    "SCRIPTED_PREGRASP_STEPS",
    "SCRIPTED_PREFIX_STEPS",
    "PREINSERT_GATE_STEP",
    "RELEASED_TIP_CENTER_DISTANCE_M",
    "ScrewCommand",
    "ScrewExpert",
    "TaskPhase",
    "THREAD_ENGAGE_STEPS",
    "THREAD_GATE_STEP",
    "THREAD_INSERT_STEPS",
    "THREAD_REGRASP_CLOSE_STEPS",
    "THREAD_REGRASP_RELEASED_DISTANCE_M",
    "THREAD_REGRASP_STEP",
    "YAW_EQUIVARIANT_ALIGN_BOUNDARY_STEP",
    "YAW_EQUIVARIANT_ALIGN_STEPS",
    "YAW_EQUIVARIANT_RELEASED_DISTANCE_M",
    "YAW_EQUIVARIANT_ROLL_GAIN",
    "YAW_EQUIVARIANT_ROLL_BOUNDARY_STEP",
    "YAW_EQUIVARIANT_ROLL_STEPS",
    "YAW_EQUIVARIANT_STRATEGY_VERSION",
    "align_local_z_to",
    "holding_fingers",
    "matrix_to_quaternion_wxyz",
    "minimum_jerk",
    "rotation_exp",
    "rotation_log",
    "validate_contact_schedule",
]
