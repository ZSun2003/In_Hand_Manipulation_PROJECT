#!/usr/bin/env python3
"""Bounded residual layer between Diffusion and analytic IK."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np

from screw_control import (
    FINGERS,
    ScrewCommand,
    TaskPhase,
    rotation_exp,
    validate_contact_schedule,
)


RESIDUAL_ACTION_NAMES: Tuple[str, ...] = (
    "lateral_x",
    "lateral_y",
    "axial",
    "tilt_x",
    "tilt_y",
    "tighten",
)
RESIDUAL_ACTION_DIM = len(RESIDUAL_ACTION_NAMES)


def residual_action_mask(
    phase: TaskPhase, engaged: bool
) -> np.ndarray:
    """Executable residual dimensions for the current task phase."""

    regrasp = phase in (
        TaskPhase.LR_OPEN,
        TaskPhase.LR_RESET,
        TaskPhase.LR_CLOSE,
        TaskPhase.UD_OPEN,
        TaskPhase.UD_RESET,
        TaskPhase.UD_CLOSE,
    )
    if regrasp:
        return np.zeros(RESIDUAL_ACTION_DIM, dtype=np.float32)
    axial = (not engaged) and phase in (
        TaskPhase.ALIGN,
        TaskPhase.INSERT,
        TaskPhase.ENGAGE,
    )
    return np.asarray(
        (True, True, axial, True, True, bool(engaged)),
        dtype=np.float32,
    )


@dataclass(frozen=True)
class ResidualLimits:
    # Absolute task-space residual envelopes. As in Gripper3/FLUID, zero
    # action exactly recovers the teacher and the policy never controls grip.
    lateral_m: float = 0.0040
    lateral_delta_m_per_step: float = 0.0004
    # Five millimetres leaves enough authority to unload/retract a wedged
    # shaft while the adaptive supervisor keeps only a 1 mm inward lead.
    axial_m: float = 0.0050
    axial_delta_m_per_step: float = 0.0005
    tilt_rad: float = math.radians(2.0)
    tilt_delta_rad_per_step: float = math.radians(0.5)
    twist_rad: float = math.radians(2.0)
    contact_distance_m: float = 0.0410
    holding_tolerance_m: float = 0.0020
    filter_alpha: float = 0.50


class ResidualCommandProjector:
    """Apply a normalized residual without violating regrasp/thread safety."""

    def __init__(
        self,
        *,
        socket_rotation_world: Sequence[Sequence[float]],
        pitch_m: float,
        limits: ResidualLimits = ResidualLimits(),
    ) -> None:
        self.socket_rotation_world = np.asarray(
            socket_rotation_world, dtype=np.float64
        ).reshape(3, 3)
        self.pitch_m = float(pitch_m)
        self.limits = limits
        self.last_applied = np.zeros(RESIDUAL_ACTION_DIM, dtype=np.float32)
        self.lateral_state_m = np.zeros(2, dtype=np.float64)
        self.axial_state_m = 0.0
        self.tilt_state_rad = np.zeros(2, dtype=np.float64)
        self.last_ik_projection_fraction = 1.0

    def reset(self) -> None:
        self.lateral_state_m.fill(0.0)
        self.axial_state_m = 0.0
        self.tilt_state_rad.fill(0.0)
        self.last_applied.fill(0.0)
        self.last_ik_projection_fraction = 1.0

    def accept_ik_projection(self, feasible_fraction: float) -> None:
        """Synchronize the residual filter with the command IK accepted.

        The simulator projects between the certified supervisor command
        (fraction 0) and the learned candidate (fraction 1).  Feeding the
        accepted fraction back prevents the residual filter from integrating
        an unreachable offset behind the safety layer.
        """

        fraction = float(np.clip(feasible_fraction, 0.0, 1.0))
        self.lateral_state_m *= fraction
        self.axial_state_m *= fraction
        self.tilt_state_rad *= fraction
        self.last_applied *= np.float32(fraction)
        self.last_ik_projection_fraction = fraction

    def project(
        self,
        base: ScrewCommand,
        action: Sequence[float],
        *,
        previous_command: ScrewCommand | None = None,
    ) -> ScrewCommand:
        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        if raw.shape != (RESIDUAL_ACTION_DIM,):
            raise ValueError(
                f"residual action must have shape {(RESIDUAL_ACTION_DIM,)}, "
                f"got {raw.shape}"
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError("residual action contains NaN or infinity")
        normalized = np.clip(raw, -1.0, 1.0) * residual_action_mask(
            base.phase, base.engaged
        )
        # Preserve FLUID's most useful invariant: the all-zero learned action
        # is exactly the deterministic IK supervisor, even after a prior
        # non-zero action.  The actuator-space controller below this layer
        # performs the required smooth return to the safe anchor.
        if not np.any(normalized):
            self.reset()
            result = base.copy()
            validate_contact_schedule(
                result,
                contact_distance_m=self.limits.contact_distance_m,
                holding_tolerance_m=self.limits.holding_tolerance_m,
            )
            return result
        limits = self.limits
        result = base.copy()
        requested_lateral = normalized[:2] * limits.lateral_m
        filtered_lateral = self.lateral_state_m + limits.filter_alpha * (
            requested_lateral - self.lateral_state_m
        )
        lateral_delta = np.clip(
            filtered_lateral - self.lateral_state_m,
            -limits.lateral_delta_m_per_step,
            limits.lateral_delta_m_per_step,
        )
        applied_lateral = np.clip(
            self.lateral_state_m + lateral_delta,
            -limits.lateral_m,
            limits.lateral_m,
        )

        requested_axial = float(normalized[2] * limits.axial_m)
        filtered_axial = self.axial_state_m + limits.filter_alpha * (
            requested_axial - self.axial_state_m
        )
        axial_delta = float(
            np.clip(
                filtered_axial - self.axial_state_m,
                -limits.axial_delta_m_per_step,
                limits.axial_delta_m_per_step,
            )
        )
        applied_axial = float(
            np.clip(
                self.axial_state_m + axial_delta,
                -limits.axial_m,
                limits.axial_m,
            )
        )

        requested_tilt = normalized[3:5] * limits.tilt_rad
        filtered_tilt = self.tilt_state_rad + limits.filter_alpha * (
            requested_tilt - self.tilt_state_rad
        )
        tilt_delta = np.clip(
            filtered_tilt - self.tilt_state_rad,
            -limits.tilt_delta_rad_per_step,
            limits.tilt_delta_rad_per_step,
        )
        applied_tilt = np.clip(
            self.tilt_state_rad + tilt_delta,
            -limits.tilt_rad,
            limits.tilt_rad,
        )
        twist_offset = float(normalized[5] * limits.twist_rad)

        socket_delta = np.array(
            [applied_lateral[0], applied_lateral[1], applied_axial],
            dtype=np.float64,
        )
        rotation_delta = np.array(
            [
                applied_tilt[0],
                applied_tilt[1],
                twist_offset,
            ],
            dtype=np.float64,
        )
        # During a regrasp the object pose is frozen by the safety supervisor.
        regrasp = result.phase in (
            TaskPhase.LR_OPEN,
            TaskPhase.LR_RESET,
            TaskPhase.LR_CLOSE,
            TaskPhase.UD_OPEN,
            TaskPhase.UD_RESET,
            TaskPhase.UD_CLOSE,
        )
        if regrasp:
            socket_delta[:] = 0.0
            rotation_delta[:] = 0.0
            applied_lateral[:] = 0.0
            applied_axial = 0.0
            applied_tilt[:] = 0.0
            twist_offset = 0.0

        if result.engaged:
            # Tightening may speed up or slow down the baseline but cannot
            # reverse actual progress. Axial residual obeys the exact lead.
            if previous_command is not None:
                minimum_offset = (
                    float(previous_command.twist_rad)
                    - float(base.twist_rad)
                )
                twist_offset = max(twist_offset, minimum_offset)
                rotation_delta[2] = twist_offset
            socket_delta[2] = (
                self.pitch_m * twist_offset / (2.0 * math.pi)
            )
        else:
            # Rotation about the screw axis is owned by the supervisor before
            # thread engagement; the sixth action is inactive.
            rotation_delta[2] = 0.0
            twist_offset = 0.0
            axial_active = result.phase in (
                TaskPhase.ALIGN,
                TaskPhase.INSERT,
                TaskPhase.ENGAGE,
            )
            if not axial_active:
                socket_delta[2] = 0.0
                applied_axial = 0.0

        result.tip_world = (
            np.asarray(result.tip_world, dtype=np.float64)
            + self.socket_rotation_world @ socket_delta
        )
        world_increment = (
            self.socket_rotation_world
            @ rotation_exp(rotation_delta)
            @ self.socket_rotation_world.T
        )
        result.rotation_world = world_increment @ np.asarray(
            result.rotation_world, dtype=np.float64
        )
        result.twist_rad += twist_offset

        validate_contact_schedule(
            result,
            contact_distance_m=limits.contact_distance_m,
            holding_tolerance_m=limits.holding_tolerance_m,
        )
        self.lateral_state_m = applied_lateral.copy()
        self.axial_state_m = applied_axial
        self.tilt_state_rad = applied_tilt.copy()
        self.last_applied = normalized.astype(np.float32)
        return result


__all__ = [
    "RESIDUAL_ACTION_DIM",
    "RESIDUAL_ACTION_NAMES",
    "ResidualCommandProjector",
    "ResidualLimits",
    "residual_action_mask",
]
