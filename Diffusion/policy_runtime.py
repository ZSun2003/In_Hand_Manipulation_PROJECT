#!/usr/bin/env python3
"""Safety adapter from a 14-D Diffusion action to :class:`ScrewCommand`.

The learned policy proposes task-space *increments* in ``ACTION_NAMES`` order.
It never owns the discrete task phase.  A deterministic supervisor supplies
the next legal phase/contact mode, while this module:

* clamps every per-step action and every absolute fingertip distance;
* keeps the candidate inside a trust region around the supervisor trajectory;
* freezes object motion while either opposing finger pair is regrasping;
* forces the designated opposing pair to remain closed during regrasp;
* after thread engagement, projects axial advance onto
  ``ds = pitch * d_twist / (2*pi)``;
* prevents learned tightening commands from reversing the screw by default.

Only NumPy and the local ``screw_control`` module are required.  The adapter
therefore sits between ``DiffusionController.act()`` and the IK/controller
without importing PyTorch or MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from screw_control import (
    ACTION_DIM,
    ACTION_NAMES,
    BASE_FINGER_ANGLES_RAD,
    FINGERS,
    RELEASED_TIP_CENTER_DISTANCE_M,
    ScrewCommand,
    TaskPhase,
    holding_fingers,
    rotation_exp,
    rotation_log,
    validate_contact_schedule,
)


REGRASP_PHASES: Tuple[TaskPhase, ...] = (
    TaskPhase.LR_OPEN,
    TaskPhase.LR_RESET,
    TaskPhase.LR_CLOSE,
    TaskPhase.UD_OPEN,
    TaskPhase.UD_RESET,
    TaskPhase.UD_CLOSE,
)

OBJECT_HOLD_PHASES: Tuple[TaskPhase, ...] = (
    *REGRASP_PHASES,
    TaskPhase.FINAL_HOLD,
)


@dataclass(frozen=True)
class PolicySafetyLimits:
    """Per-control-step and supervisor-trust limits in SI units."""

    max_tip_xy_step_m: float = 0.0020
    max_tip_axial_step_m: float = 0.0030
    max_tilt_step_rad: float = math.radians(2.0)
    max_twist_step_rad: float = math.radians(4.0)
    max_material_step_rad: float = math.radians(6.0)
    max_distance_step_m: float = 0.0040

    min_tip_center_distance_m: float = 0.0385
    max_tip_center_distance_m: float = 0.0610
    holding_tolerance_m: float = 0.0020

    supervisor_tip_xy_trust_m: float = 0.0060
    supervisor_tip_axial_trust_m: float = 0.0060
    supervisor_tilt_trust_rad: float = math.radians(5.0)
    supervisor_twist_trust_rad: float = math.radians(8.0)
    supervisor_material_trust_rad: float = math.radians(12.0)
    supervisor_distance_trust_m: float = 0.0050

    prevent_reverse_when_engaged: bool = True
    freeze_object_during_regrasp: bool = True

    def __post_init__(self) -> None:
        positive = (
            "max_tip_xy_step_m",
            "max_tip_axial_step_m",
            "max_tilt_step_rad",
            "max_twist_step_rad",
            "max_material_step_rad",
            "max_distance_step_m",
            "min_tip_center_distance_m",
            "max_tip_center_distance_m",
            "holding_tolerance_m",
            "supervisor_tip_xy_trust_m",
            "supervisor_tip_axial_trust_m",
            "supervisor_tilt_trust_rad",
            "supervisor_twist_trust_rad",
            "supervisor_material_trust_rad",
            "supervisor_distance_trust_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_tip_center_distance_m >= self.max_tip_center_distance_m:
            raise ValueError("tip-centre distance range is empty")

    @property
    def action_limit_vector(self) -> np.ndarray:
        return np.asarray(
            (
                self.max_tip_xy_step_m,
                self.max_tip_xy_step_m,
                self.max_tip_axial_step_m,
                self.max_tilt_step_rad,
                self.max_tilt_step_rad,
                self.max_twist_step_rad,
                self.max_material_step_rad,
                self.max_material_step_rad,
                self.max_material_step_rad,
                self.max_material_step_rad,
                self.max_distance_step_m,
                self.max_distance_step_m,
                self.max_distance_step_m,
                self.max_distance_step_m,
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Machine-readable account of the last safety projection."""

    phase: TaskPhase
    raw_action: np.ndarray
    clipped_action: np.ndarray
    applied_action: np.ndarray
    per_step_clipped: bool
    supervisor_trust_projected: bool
    helical_projected: bool
    object_frozen: bool
    forced_closed_fingers: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.name,
            "raw_action": self.raw_action.tolist(),
            "clipped_action": self.clipped_action.tolist(),
            "applied_action": self.applied_action.tolist(),
            "per_step_clipped": self.per_step_clipped,
            "supervisor_trust_projected": self.supervisor_trust_projected,
            "helical_projected": self.helical_projected,
            "object_frozen": self.object_frozen,
            "forced_closed_fingers": list(self.forced_closed_fingers),
        }


def _validate_rotation(name: str, rotation: Any) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3x3 rotation matrix")
    if not np.allclose(value.T @ value, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{name} is not orthonormal")
    if not math.isclose(float(np.linalg.det(value)), 1.0, abs_tol=1.0e-6):
        raise ValueError(f"{name} must have determinant +1")
    return value


def _validate_command(name: str, command: ScrewCommand) -> None:
    if not isinstance(command, ScrewCommand):
        raise TypeError(f"{name} must be a ScrewCommand")
    tip = np.asarray(command.tip_world, dtype=np.float64)
    if tip.shape != (3,) or not np.all(np.isfinite(tip)):
        raise ValueError(f"{name}.tip_world must be a finite 3-vector")
    _validate_rotation(f"{name}.rotation_world", command.rotation_world)
    if not math.isfinite(float(command.twist_rad)):
        raise ValueError(f"{name}.twist_rad must be finite")
    for field_name, values in (
        ("material_angles_rad", command.material_angles_rad),
        ("tip_center_distances_m", command.tip_center_distances_m),
    ):
        if set(values) != set(FINGERS):
            raise ValueError(f"{name}.{field_name} must contain exactly {FINGERS}")
        if any(not math.isfinite(float(values[finger])) for finger in FINGERS):
            raise ValueError(f"{name}.{field_name} contains a non-finite value")
    TaskPhase(command.phase)


class SafeDiffusionCommandAdapter:
    """Project one Diffusion action into the supervisor's legal command mode."""

    def __init__(
        self,
        *,
        socket_frame_world: Sequence[Sequence[float]],
        thread_pitch_m: float,
        contact_distance_m: float,
        released_distance_m: float = RELEASED_TIP_CENTER_DISTANCE_M,
        limits: Optional[PolicySafetyLimits] = None,
    ) -> None:
        self.socket_frame_world = _validate_rotation(
            "socket_frame_world", socket_frame_world
        ).copy()
        self.thread_pitch_m = float(thread_pitch_m)
        self.contact_distance_m = float(contact_distance_m)
        self.released_distance_m = float(released_distance_m)
        if (
            not math.isfinite(self.thread_pitch_m)
            or self.thread_pitch_m <= 0.0
        ):
            raise ValueError("thread_pitch_m must be finite and positive")
        if (
            not math.isfinite(self.contact_distance_m)
            or self.contact_distance_m <= 0.0
        ):
            raise ValueError("contact_distance_m must be finite and positive")
        if (
            not math.isfinite(self.released_distance_m)
            or self.released_distance_m <= self.contact_distance_m
        ):
            raise ValueError(
                "released_distance_m must be finite and greater than contact distance"
            )
        self.limits = limits or PolicySafetyLimits()
        if not (
            self.limits.min_tip_center_distance_m
            <= self.contact_distance_m
            <= self.limits.max_tip_center_distance_m
        ):
            raise ValueError("contact distance lies outside safety limits")
        if self.released_distance_m > self.limits.max_tip_center_distance_m:
            raise ValueError("released distance lies outside safety limits")
        self.last_diagnostics: Optional[ProjectionDiagnostics] = None

    @classmethod
    def from_expert(
        cls,
        expert: Any,
        *,
        limits: Optional[PolicySafetyLimits] = None,
    ) -> "SafeDiffusionCommandAdapter":
        """Construct from the matching attributes of a local ``ScrewExpert``."""

        return cls(
            socket_frame_world=expert.socket_frame_world,
            thread_pitch_m=expert.pitch_m,
            contact_distance_m=expert.contact_distance_m,
            released_distance_m=expert.released_distance_m,
            limits=limits,
        )

    def _relative_rotation_socket(
        self,
        previous_rotation_world: np.ndarray,
        following_rotation_world: np.ndarray,
    ) -> np.ndarray:
        world_delta = following_rotation_world @ previous_rotation_world.T
        socket_delta = (
            self.socket_frame_world.T
            @ world_delta
            @ self.socket_frame_world
        )
        return rotation_log(socket_delta)

    def _advance_rotation(
        self,
        previous_rotation_world: np.ndarray,
        delta_socket: np.ndarray,
    ) -> np.ndarray:
        world_delta = (
            self.socket_frame_world
            @ rotation_exp(delta_socket)
            @ self.socket_frame_world.T
        )
        return world_delta @ previous_rotation_world

    def _clamp_rotation_with_supervisor(
        self,
        previous: ScrewCommand,
        supervisor: ScrewCommand,
        proposed_delta: np.ndarray,
        *,
        engaged: bool,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Return ``(rotation, actual_delta, trust_was_projected)``."""

        candidate = self._advance_rotation(
            np.asarray(previous.rotation_world), proposed_delta
        )
        error = self._relative_rotation_socket(
            np.asarray(supervisor.rotation_world), candidate
        )
        trust = np.asarray(
            (
                self.limits.supervisor_tilt_trust_rad,
                self.limits.supervisor_tilt_trust_rad,
                self.limits.supervisor_twist_trust_rad,
            ),
            dtype=np.float64,
        )
        projected_error = np.clip(error, -trust, trust)
        trust_projected = not np.allclose(error, projected_error, atol=1.0e-14)
        candidate = self._advance_rotation(
            np.asarray(supervisor.rotation_world), projected_error
        )

        # Reapply the hard increment bound.  This takes priority if the
        # supervisor trust set and the one-step reach set do not intersect.
        actual_delta = self._relative_rotation_socket(
            np.asarray(previous.rotation_world), candidate
        )
        step_limit = np.asarray(
            (
                self.limits.max_tilt_step_rad,
                self.limits.max_tilt_step_rad,
                self.limits.max_twist_step_rad,
            )
        )
        actual_delta = np.clip(actual_delta, -step_limit, step_limit)

        if engaged:
            twist_min = (
                float(previous.twist_rad)
                if self.limits.prevent_reverse_when_engaged
                else -math.inf
            )
            twist_low = max(
                twist_min,
                float(supervisor.twist_rad)
                - self.limits.supervisor_twist_trust_rad,
            )
            twist_high = (
                float(supervisor.twist_rad)
                + self.limits.supervisor_twist_trust_rad
            )
            if twist_low > twist_high:
                # An inconsistent supervisor must never make tightening reverse.
                twist_low = twist_min
                twist_high = max(twist_min, twist_high)
            proposed_twist = float(previous.twist_rad) + float(actual_delta[2])
            safe_twist = float(np.clip(proposed_twist, twist_low, twist_high))
            actual_delta[2] = safe_twist - float(previous.twist_rad)
        candidate = self._advance_rotation(
            np.asarray(previous.rotation_world), actual_delta
        )
        return candidate, actual_delta, trust_projected

    def _project_position(
        self,
        previous: ScrewCommand,
        supervisor: ScrewCommand,
        proposed_delta: np.ndarray,
        *,
        engaged: bool,
        twist_delta: float,
    ) -> Tuple[np.ndarray, bool]:
        candidate = (
            np.asarray(previous.tip_world)
            + self.socket_frame_world @ proposed_delta
        )
        error = self.socket_frame_world.T @ (
            candidate - np.asarray(supervisor.tip_world)
        )
        trust = np.asarray(
            (
                self.limits.supervisor_tip_xy_trust_m,
                self.limits.supervisor_tip_xy_trust_m,
                self.limits.supervisor_tip_axial_trust_m,
            )
        )
        projected_error = np.clip(error, -trust, trust)
        trust_projected = not np.allclose(error, projected_error, atol=1.0e-14)
        candidate = (
            np.asarray(supervisor.tip_world)
            + self.socket_frame_world @ projected_error
        )

        actual_delta = self.socket_frame_world.T @ (
            candidate - np.asarray(previous.tip_world)
        )
        actual_delta[:2] = np.clip(
            actual_delta[:2],
            -self.limits.max_tip_xy_step_m,
            self.limits.max_tip_xy_step_m,
        )
        if engaged:
            # This final assignment is deliberately after every other clamp:
            # no later projection may break the helical kinematic invariant.
            actual_delta[2] = (
                self.thread_pitch_m * twist_delta / (2.0 * math.pi)
            )
        else:
            actual_delta[2] = float(
                np.clip(
                    actual_delta[2],
                    -self.limits.max_tip_axial_step_m,
                    self.limits.max_tip_axial_step_m,
                )
            )
        return (
            np.asarray(previous.tip_world)
            + self.socket_frame_world @ actual_delta,
            trust_projected,
        )

    def _project_scalar_finger_fields(
        self,
        previous: ScrewCommand,
        supervisor: ScrewCommand,
        clipped_action: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, float], bool]:
        phase = TaskPhase(supervisor.phase)
        material: Dict[str, float] = {}
        distances: Dict[str, float] = {}
        trust_projected = False
        for index, finger in enumerate(FINGERS):
            proposed_material = (
                float(previous.material_angles_rad[finger])
                + float(clipped_action[6 + index])
            )
            material_low = (
                float(supervisor.material_angles_rad[finger])
                - self.limits.supervisor_material_trust_rad
            )
            material_high = (
                float(supervisor.material_angles_rad[finger])
                + self.limits.supervisor_material_trust_rad
            )
            trusted_material = float(
                np.clip(proposed_material, material_low, material_high)
            )
            final_material = float(
                np.clip(
                    trusted_material,
                    float(previous.material_angles_rad[finger])
                    - self.limits.max_material_step_rad,
                    float(previous.material_angles_rad[finger])
                    + self.limits.max_material_step_rad,
                )
            )
            trust_projected |= not math.isclose(
                proposed_material, trusted_material, abs_tol=1.0e-14
            )

            proposed_distance = (
                float(previous.tip_center_distances_m[finger])
                + float(clipped_action[10 + index])
            )
            distance_low = max(
                self.limits.min_tip_center_distance_m,
                float(supervisor.tip_center_distances_m[finger])
                - self.limits.supervisor_distance_trust_m,
            )
            distance_high = min(
                self.limits.max_tip_center_distance_m,
                float(supervisor.tip_center_distances_m[finger])
                + self.limits.supervisor_distance_trust_m,
            )
            trusted_distance = float(
                np.clip(proposed_distance, distance_low, distance_high)
            )
            final_distance = float(
                np.clip(
                    trusted_distance,
                    float(previous.tip_center_distances_m[finger])
                    - self.limits.max_distance_step_m,
                    float(previous.tip_center_distances_m[finger])
                    + self.limits.max_distance_step_m,
                )
            )
            final_distance = float(
                np.clip(
                    final_distance,
                    self.limits.min_tip_center_distance_m,
                    self.limits.max_tip_center_distance_m,
                )
            )
            trust_projected |= not math.isclose(
                proposed_distance, trusted_distance, abs_tol=1.0e-14
            )
            material[finger] = final_material
            distances[finger] = final_distance

        if phase in REGRASP_PHASES:
            # The holding pair is stationary in the screw material frame.  A
            # policy cannot silently slide those fingers while calling them
            # "closed".
            for finger in holding_fingers(phase):
                material[finger] = float(
                    previous.material_angles_rad[finger]
                )
        return material, distances, trust_projected

    def _encode_transition(
        self, previous: ScrewCommand, following: ScrewCommand
    ) -> np.ndarray:
        position_delta = self.socket_frame_world.T @ (
            np.asarray(following.tip_world)
            - np.asarray(previous.tip_world)
        )
        rotation_delta = self._relative_rotation_socket(
            np.asarray(previous.rotation_world),
            np.asarray(following.rotation_world),
        )
        material_delta = np.asarray(
            [
                following.material_angles_rad[finger]
                - previous.material_angles_rad[finger]
                for finger in FINGERS
            ]
        )
        distance_delta = np.asarray(
            [
                following.tip_center_distances_m[finger]
                - previous.tip_center_distances_m[finger]
                for finger in FINGERS
            ]
        )
        return np.concatenate(
            (position_delta, rotation_delta, material_delta, distance_delta)
        ).astype(np.float64)

    def adapt(
        self,
        action: Sequence[float] | np.ndarray,
        *,
        previous_command: ScrewCommand,
        supervisor_command: ScrewCommand,
    ) -> ScrewCommand:
        """Return the safe next command for one controller action.

        ``supervisor_command.phase`` is authoritative.  Invalid shapes or
        non-finite values raise immediately instead of reaching IK.
        """

        _validate_command("previous_command", previous_command)
        _validate_command("supervisor_command", supervisor_command)
        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (ACTION_DIM,):
            raise ValueError(
                f"Diffusion action must have shape ({ACTION_DIM},), got {raw.shape}"
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError("Diffusion action contains NaN or infinity")

        limits = self.limits.action_limit_vector
        clipped = np.clip(raw, -limits, limits)
        per_step_clipped = not np.allclose(raw, clipped, atol=1.0e-14)
        phase = TaskPhase(supervisor_command.phase)
        engaged = bool(previous_command.engaged or supervisor_command.engaged)

        # COMPLETE is a terminal supervisor decision; no learned field is used.
        if phase is TaskPhase.COMPLETE:
            result = supervisor_command.copy()
            applied = self._encode_transition(previous_command, result)
            self.last_diagnostics = ProjectionDiagnostics(
                phase=phase,
                raw_action=raw.copy(),
                clipped_action=clipped.copy(),
                applied_action=applied,
                per_step_clipped=per_step_clipped,
                supervisor_trust_projected=False,
                helical_projected=False,
                object_frozen=True,
                forced_closed_fingers=(),
            )
            return result

        object_frozen = (
            self.limits.freeze_object_during_regrasp
            and phase in OBJECT_HOLD_PHASES
        )
        trust_projected = False
        if object_frozen:
            rotation_world = np.asarray(
                previous_command.rotation_world, dtype=np.float64
            ).copy()
            tip_world = np.asarray(
                previous_command.tip_world, dtype=np.float64
            ).copy()
            twist_rad = float(previous_command.twist_rad)
            actual_rotation_delta = np.zeros(3, dtype=np.float64)
        else:
            proposed_rotation = clipped[3:6].copy()
            if engaged and phase is not TaskPhase.POWER:
                # ENGAGE and contact-mode transitions may correct tilt, but may
                # not consume any of the two-turn tightening budget.
                proposed_rotation[2] = 0.0
            if (
                engaged
                and self.limits.prevent_reverse_when_engaged
                and proposed_rotation[2] < 0.0
            ):
                proposed_rotation[2] = 0.0
            (
                rotation_world,
                actual_rotation_delta,
                rotation_trust_projected,
            ) = self._clamp_rotation_with_supervisor(
                previous_command,
                supervisor_command,
                proposed_rotation,
                engaged=engaged,
            )
            trust_projected |= rotation_trust_projected
            if engaged:
                twist_rad = (
                    float(previous_command.twist_rad)
                    + float(actual_rotation_delta[2])
                )
            else:
                # Before engagement, general alignment rotations are not thread
                # progress.  The supervisor owns the zero/reference twist.
                twist_rad = float(supervisor_command.twist_rad)
            (
                tip_world,
                position_trust_projected,
            ) = self._project_position(
                previous_command,
                supervisor_command,
                clipped[:3].copy(),
                engaged=engaged,
                twist_delta=(
                    float(actual_rotation_delta[2]) if engaged else 0.0
                ),
            )
            trust_projected |= position_trust_projected

        material, distances, finger_trust_projected = (
            self._project_scalar_finger_fields(
                previous_command, supervisor_command, clipped
            )
        )
        trust_projected |= finger_trust_projected

        forced_closed: list[str] = []
        if phase not in (TaskPhase.PREGRASP, TaskPhase.CLOSE):
            required = holding_fingers(phase)
            closed_limit = (
                self.contact_distance_m
                + self.limits.holding_tolerance_m
            )
            for finger in required:
                if distances[finger] > closed_limit:
                    distances[finger] = closed_limit
                    forced_closed.append(finger)

        result = ScrewCommand(
            tip_world=np.asarray(tip_world, dtype=np.float64),
            rotation_world=np.asarray(rotation_world, dtype=np.float64),
            twist_rad=float(twist_rad),
            material_angles_rad=material,
            tip_center_distances_m=distances,
            phase=phase,
            engaged=engaged,
        )
        validate_contact_schedule(
            result,
            contact_distance_m=self.contact_distance_m,
            holding_tolerance_m=self.limits.holding_tolerance_m,
        )
        applied = self._encode_transition(previous_command, result)
        self.last_diagnostics = ProjectionDiagnostics(
            phase=phase,
            raw_action=raw.copy(),
            clipped_action=clipped.copy(),
            applied_action=applied,
            per_step_clipped=per_step_clipped,
            supervisor_trust_projected=trust_projected,
            helical_projected=engaged and not object_frozen,
            object_frozen=object_frozen,
            forced_closed_fingers=tuple(forced_closed),
        )
        return result


def _test_command(
    *,
    phase: TaskPhase,
    engaged: bool,
    tip: Sequence[float] = (0.0, 0.0, 0.090),
    twist: float = 0.0,
    distances: Optional[Mapping[str, float]] = None,
) -> ScrewCommand:
    return ScrewCommand(
        tip_world=np.asarray(tip, dtype=np.float64),
        rotation_world=np.eye(3, dtype=np.float64),
        twist_rad=float(twist),
        material_angles_rad=dict(BASE_FINGER_ANGLES_RAD),
        tip_center_distances_m=(
            {finger: 0.0405 for finger in FINGERS}
            if distances is None
            else {finger: float(distances[finger]) for finger in FINGERS}
        ),
        phase=phase,
        engaged=engaged,
    )


def run_self_test() -> Dict[str, Any]:
    """Pure NumPy safety regression test; raises on the first failed invariant."""

    adapter = SafeDiffusionCommandAdapter(
        socket_frame_world=np.eye(3),
        thread_pitch_m=0.005,
        contact_distance_m=0.0405,
        released_distance_m=RELEASED_TIP_CENTER_DISTANCE_M,
    )

    # Large learned values must be bounded, phase must remain supervised, and
    # an engaged axial step must exactly follow the applied twist.
    previous = _test_command(phase=TaskPhase.ENGAGE, engaged=True)
    supervisor = _test_command(phase=TaskPhase.POWER, engaged=True)
    raw = np.full(ACTION_DIM, 1.0, dtype=np.float64)
    tightened = adapter.adapt(
        raw,
        previous_command=previous,
        supervisor_command=supervisor,
    )
    applied = adapter.last_diagnostics.applied_action
    expected_ds = 0.005 * applied[5] / (2.0 * math.pi)
    assert tightened.phase is TaskPhase.POWER
    assert tightened.engaged
    assert math.isclose(applied[2], expected_ds, abs_tol=1.0e-12)
    assert applied[5] <= adapter.limits.max_twist_step_rad + 1.0e-12
    assert np.max(np.abs(applied[:2])) <= (
        adapter.limits.max_tip_xy_step_m + 1.0e-12
    )
    assert adapter.last_diagnostics.per_step_clipped
    assert adapter.last_diagnostics.helical_projected
    assert np.allclose(
        tightened.rotation_world.T @ tightened.rotation_world,
        np.eye(3),
        atol=1.0e-10,
    )

    # During LR regrasp, U/D must stay closed and fixed in material phase even
    # when the policy explicitly requests opening/sliding every finger.
    regrasp_supervisor = tightened.copy()
    regrasp_supervisor.phase = TaskPhase.LR_RESET
    regrasp_action = np.zeros(ACTION_DIM, dtype=np.float64)
    regrasp_action[:6] = 1.0
    regrasp_action[6:10] = 1.0
    regrasp_action[10:14] = 1.0
    regrasp = adapter.adapt(
        regrasp_action,
        previous_command=tightened,
        supervisor_command=regrasp_supervisor,
    )
    closed_limit = (
        adapter.contact_distance_m + adapter.limits.holding_tolerance_m
    )
    assert regrasp.phase is TaskPhase.LR_RESET
    assert np.array_equal(regrasp.tip_world, tightened.tip_world)
    assert np.array_equal(regrasp.rotation_world, tightened.rotation_world)
    assert regrasp.twist_rad == tightened.twist_rad
    assert regrasp.tip_center_distances_m["U"] <= closed_limit
    assert regrasp.tip_center_distances_m["D"] <= closed_limit
    assert (
        regrasp.material_angles_rad["U"]
        == tightened.material_angles_rad["U"]
    )
    assert (
        regrasp.material_angles_rad["D"]
        == tightened.material_angles_rad["D"]
    )
    assert adapter.last_diagnostics.object_frozen

    # Negative twist cannot reverse an engaged right-handed tightening state.
    reverse_supervisor = regrasp.copy()
    reverse_supervisor.phase = TaskPhase.POWER
    reverse_action = np.zeros(ACTION_DIM, dtype=np.float64)
    reverse_action[5] = -1.0
    no_reverse = adapter.adapt(
        reverse_action,
        previous_command=regrasp,
        supervisor_command=reverse_supervisor,
    )
    assert no_reverse.twist_rad >= regrasp.twist_rad
    assert math.isclose(
        float(
            (
                adapter.socket_frame_world.T
                @ (no_reverse.tip_world - regrasp.tip_world)
            )[2]
        ),
        0.005
        * (no_reverse.twist_rad - regrasp.twist_rad)
        / (2.0 * math.pi),
        abs_tol=1.0e-12,
    )

    # Invalid learned output is rejected before it can reach IK.
    rejected_nonfinite = False
    try:
        bad = np.zeros(ACTION_DIM)
        bad[0] = np.nan
        adapter.adapt(
            bad,
            previous_command=previous,
            supervisor_command=supervisor,
        )
    except ValueError:
        rejected_nonfinite = True
    assert rejected_nonfinite

    return {
        "status": "passed",
        "action_dim": ACTION_DIM,
        "action_names": list(ACTION_NAMES),
        "phase_authority": "supervisor",
        "per_step_clamping": True,
        "helical_projection": True,
        "regrasp_opposing_pair": True,
        "object_frozen_during_regrasp": True,
        "reverse_prevention": True,
        "nonfinite_rejected": True,
    }


def self_test() -> Dict[str, Any]:
    """Alias used by lightweight project test runners."""

    return run_self_test()


__all__ = [
    "OBJECT_HOLD_PHASES",
    "PolicySafetyLimits",
    "ProjectionDiagnostics",
    "REGRASP_PHASES",
    "SafeDiffusionCommandAdapter",
    "run_self_test",
    "self_test",
]


if __name__ == "__main__":
    print(json.dumps(run_self_test(), indent=2, sort_keys=True))
