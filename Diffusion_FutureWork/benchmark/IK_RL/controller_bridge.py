"""One-Gym-step/one-50 Hz-control-tick bridge for the current G3 controller."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Lock
from typing import Any, Callable, Sequence

import numpy as np

from . import config


class BridgeAborted(RuntimeError):
    """Raised inside the controller worker when an environment resets."""


@dataclass(frozen=True)
class ControlTick:
    control_step: int
    base_ctrl: np.ndarray
    applied_ctrl: np.ndarray
    residual: np.ndarray
    payload: dict[str, Any]


@dataclass(frozen=True)
class ControllerDone:
    result: Any = None
    error: BaseException | None = None


@dataclass(frozen=True)
class _ControlRequest:
    control_step: int
    base_ctrl: np.ndarray


_ABORT = object()


class ControlTickBridge:
    """Synchronize an internally-looping demo with external Gym actions.

    The wrapper is installed at ``IKGrabber._physics_ctrl``.  The authoritative
    parent controller, model XML, fruit manifest and planner have already passed
    :func:`IK_RL.runtime.validate_parent_contract` when this hook is installed.
    The controller has produced its filtered/rate-limited IK command when this
    hook is called.  We add the bounded residual, invoke the original physics
    function exactly once, and intentionally do not write the residual into
    ``executed_ctrl`` or the IK teacher cache.
    """

    def __init__(
        self,
        *,
        ctrl_low: Sequence[float],
        ctrl_high: Sequence[float],
        snapshot: Callable[
            [int, np.ndarray, np.ndarray, np.ndarray],
            dict[str, Any],
        ],
    ) -> None:
        self.ctrl_low = np.asarray(ctrl_low, dtype=np.float64).reshape(
            config.ACTUATOR_DIM
        )
        self.ctrl_high = np.asarray(ctrl_high, dtype=np.float64).reshape(
            config.ACTUATOR_DIM
        )
        self.snapshot = snapshot
        self.residual = np.zeros(config.ACTUATOR_DIM, dtype=np.float64)
        self._events: Queue[object] = Queue()
        self._actions: Queue[object] = Queue()
        self._pending: _ControlRequest | None = None
        self._control_step = 0
        self._lock = Lock()
        self._closed = False

    @property
    def control_step(self) -> int:
        return self._control_step

    @property
    def waiting(self) -> bool:
        return self._pending is not None

    def wrapped_physics_ctrl(
        self,
        original: Callable[[Sequence[float]], None],
        base_ctrl: Sequence[float],
    ) -> None:
        base = np.asarray(base_ctrl, dtype=np.float64).reshape(
            config.ACTUATOR_DIM
        )
        if not np.all(np.isfinite(base)):
            raise ValueError("the current IK controller produced non-finite controls")
        with self._lock:
            if self._closed:
                raise BridgeAborted("control bridge aborted")
        request = _ControlRequest(self._control_step, base.copy())
        self._events.put(request)
        action_value = self._actions.get()
        if action_value is _ABORT:
            raise BridgeAborted("control bridge aborted")

        action = np.asarray(action_value, dtype=np.float64).reshape(
            config.ACTUATOR_DIM
        )
        if not np.all(np.isfinite(action)):
            raise ValueError("residual action must contain only finite values")
        action = np.clip(action, -1.0, 1.0)
        target = action * config.RESIDUAL_MAX_RAD
        blended = (
            (1.0 - config.RESIDUAL_TARGET_BLEND) * self.residual
            + config.RESIDUAL_TARGET_BLEND * target
        )
        delta = np.clip(
            blended - self.residual,
            -config.RESIDUAL_RATE_LIMIT_RAD_PER_STEP,
            config.RESIDUAL_RATE_LIMIT_RAD_PER_STEP,
        )
        self.residual = np.clip(
            self.residual + delta,
            -config.RESIDUAL_MAX_RAD,
            config.RESIDUAL_MAX_RAD,
        )
        applied = np.clip(base + self.residual, self.ctrl_low, self.ctrl_high)

        original(applied)
        payload = self.snapshot(
            self._control_step,
            base.copy(),
            applied.copy(),
            self.residual.copy(),
        )
        tick = ControlTick(
            control_step=self._control_step,
            base_ctrl=base.copy(),
            applied_ctrl=applied.copy(),
            residual=self.residual.copy(),
            payload=payload,
        )
        self._control_step += 1
        self._events.put(tick)

    def wait_until_ready(self) -> ControllerDone | None:
        event = self._events.get()
        if isinstance(event, _ControlRequest):
            with self._lock:
                self._pending = event
            return None
        if isinstance(event, ControllerDone):
            self._pending = None
            return event
        raise RuntimeError(f"unexpected bridge event while waiting: {event!r}")

    def advance(
        self,
        action: Sequence[float],
    ) -> tuple[ControlTick, ControllerDone | None]:
        if self._pending is None:
            raise RuntimeError("advance called without a pending control request")
        self._pending = None
        self._actions.put(
            np.asarray(action, dtype=np.float64).reshape(config.ACTUATOR_DIM)
        )
        completed = self._events.get()
        if isinstance(completed, ControllerDone):
            if completed.error is not None:
                raise completed.error
            raise RuntimeError(
                "controller completed before the requested physics tick"
            )
        if not isinstance(completed, ControlTick):
            raise RuntimeError(f"expected completed control tick, got {completed!r}")
        following = self.wait_until_ready()
        return completed, following

    def finish(self, *, result: Any = None, error: BaseException | None = None) -> None:
        self._events.put(ControllerDone(result=result, error=error))

    def abort(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            # Always enqueue cancellation.  This also closes the small race
            # where the worker has published a request but the consumer has
            # not yet copied it into ``_pending``.
            self._actions.put(_ABORT)


def apply_residual_once(
    previous_residual: Sequence[float],
    normalized_action: Sequence[float],
) -> np.ndarray:
    """Pure helper used by contract tests."""

    previous = np.asarray(previous_residual, dtype=np.float64).reshape(
        config.ACTUATOR_DIM
    )
    action = np.asarray(normalized_action, dtype=np.float64).reshape(
        config.ACTUATOR_DIM
    )
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(action)):
        raise ValueError("residual inputs must contain only finite values")
    action = np.clip(action, -1.0, 1.0)
    target = action * config.RESIDUAL_MAX_RAD
    blended = (
        (1.0 - config.RESIDUAL_TARGET_BLEND) * previous
        + config.RESIDUAL_TARGET_BLEND * target
    )
    delta = np.clip(
        blended - previous,
        -config.RESIDUAL_RATE_LIMIT_RAD_PER_STEP,
        config.RESIDUAL_RATE_LIMIT_RAD_PER_STEP,
    )
    return np.clip(
        previous + delta,
        -config.RESIDUAL_MAX_RAD,
        config.RESIDUAL_MAX_RAD,
    )


__all__ = [
    "BridgeAborted",
    "ControlTick",
    "ControlTickBridge",
    "ControllerDone",
    "apply_residual_once",
]
