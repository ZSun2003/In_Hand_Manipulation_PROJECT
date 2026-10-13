"""Unit-aware direct actuator residual law for G3 and G6."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from . import config


class ActuatorResidualController:
    def __init__(self, gripper: str, ctrl_low: Iterable[float], ctrl_high: Iterable[float]) -> None:
        self.gripper = config.GripperKind(gripper).value
        self.indices = np.asarray(
            config.RESIDUAL_ACTUATOR_INDICES[self.gripper], dtype=np.int32
        )
        self.action_dim = config.action_dim(self.gripper)
        self.ctrl_low = np.asarray(tuple(ctrl_low), dtype=np.float64)
        self.ctrl_high = np.asarray(tuple(ctrl_high), dtype=np.float64)
        if self.ctrl_low.shape != self.ctrl_high.shape:
            raise ValueError("control range shapes differ")
        if self.indices.size != self.action_dim:
            raise ValueError("residual actuator mapping does not match action width")
        if np.any(self.indices < 0) or np.any(self.indices >= self.ctrl_low.size):
            raise ValueError("residual actuator mapping is outside ctrl range")

        self.limits = np.full(
            self.action_dim, config.ANGULAR_RESIDUAL_LIMIT_RAD, dtype=np.float64
        )
        self.rates = np.full(
            self.action_dim,
            config.ANGULAR_RESIDUAL_RATE_RAD_PER_TICK,
            dtype=np.float64,
        )
        self.blends = np.full(
            self.action_dim, config.ANGULAR_RESIDUAL_BLEND, dtype=np.float64
        )
        self.units = np.full(self.action_dim, "rad", dtype=object)
        if self.gripper == config.GripperKind.G6.value:
            linear = np.asarray(config.G6_LINEAR_ACTUATOR_INDICES, dtype=np.int32)
            self.limits[linear] = config.LINEAR_RESIDUAL_LIMIT_M
            self.rates[linear] = config.LINEAR_RESIDUAL_RATE_M_PER_TICK
            self.blends[linear] = config.LINEAR_RESIDUAL_BLEND
            self.units[linear] = "m"
        self.residual = np.zeros(self.action_dim, dtype=np.float64)

    def reset(self) -> None:
        self.residual.fill(0.0)

    def apply(self, nominal_ctrl: Iterable[float], action: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        nominal = np.asarray(tuple(nominal_ctrl), dtype=np.float64).reshape(self.ctrl_low.shape)
        value = np.asarray(tuple(action), dtype=np.float64).reshape(self.action_dim)
        if not np.all(np.isfinite(value)):
            raise ValueError("action must contain only finite values")
        requested = np.clip(value, -1.0, 1.0) * self.limits
        requested = (1.0 - self.blends) * self.residual + self.blends * requested
        delta = np.clip(
            requested - self.residual,
            -self.rates,
            self.rates,
        )
        self.residual = np.clip(
            self.residual + delta,
            -self.limits,
            self.limits,
        )
        applied = nominal.copy()
        applied[self.indices] += self.residual
        applied = np.clip(applied, self.ctrl_low, self.ctrl_high)
        # Anti-windup: only the unit-aware actuator offset that can actually be
        # realized is carried into the next tick.  Otherwise a hidden saturated
        # residual can accumulate at a ctrlrange bound and be released in one
        # jump when the nominal target moves away from that bound.
        self.residual = applied[self.indices] - nominal[self.indices]
        return applied, self.residual.copy()


# Backward-compatible import name for existing callers.  The implementation is
# unit-aware and includes all four G6 linear slides.
AngularResidualController = ActuatorResidualController
