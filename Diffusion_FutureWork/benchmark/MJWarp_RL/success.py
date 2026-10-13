"""Endpoint pose-error metrics and per-axis success classification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from . import contract


@dataclass(frozen=True)
class SuccessThreshold:
    position_axis_abs_m: float
    euler_axis_abs_deg: float


ROTATION_THRESHOLD = SuccessThreshold(
    position_axis_abs_m=(
        contract.ROTATION_POSITION_AXIS_ABS_TOLERANCE_M
    ),
    euler_axis_abs_deg=contract.ROTATION_EULER_AXIS_ABS_TOLERANCE_DEG,
)
TRANSLATION_THRESHOLD = SuccessThreshold(
    position_axis_abs_m=(
        contract.TRANSLATION_POSITION_AXIS_ABS_TOLERANCE_M
    ),
    euler_axis_abs_deg=(
        contract.TRANSLATION_EULER_AXIS_ABS_TOLERANCE_DEG
    ),
)


def threshold_for_object(
    object_kind: str,
    task: str = contract.TASK_ROTATE,
) -> SuccessThreshold:
    kind = contract.validate_object(object_kind)
    selected_task = contract.validate_task(task)
    if kind not in contract.FRUIT_OBJECTS:
        raise ValueError(f"active object is not a fruit: {kind!r}")
    if selected_task == contract.TASK_TRANSLATE:
        return TRANSLATION_THRESHOLD
    return ROTATION_THRESHOLD


def _vector3(values: Iterable[object], name: str) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a numeric length-3 iterable")
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric length-3 iterable") from exc
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain three finite values")
    return vector  # type: ignore[return-value]


def euclidean_position_error_m(
    actual_xyz_m: Iterable[object],
    target_xyz_m: Iterable[object],
) -> float:
    actual = _vector3(actual_xyz_m, "actual_xyz_m")
    target = _vector3(target_xyz_m, "target_xyz_m")
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(actual, target)))


def wrapped_euler_delta_deg(
    actual_euler_deg: Iterable[object],
    target_euler_deg: Iterable[object],
) -> tuple[float, float, float]:
    actual = _vector3(actual_euler_deg, "actual_euler_deg")
    target = _vector3(target_euler_deg, "target_euler_deg")
    return tuple(
        (left - right + 180.0) % 360.0 - 180.0
        for left, right in zip(actual, target)
    )  # type: ignore[return-value]


def euler_l2_error_deg(
    actual_euler_deg: Iterable[object],
    target_euler_deg: Iterable[object],
) -> float:
    delta = wrapped_euler_delta_deg(actual_euler_deg, target_euler_deg)
    return math.sqrt(sum(value * value for value in delta))


def endpoint_success(
    object_kind: str,
    position_error_xyz_m: Iterable[object],
    euler_error_xyz_deg: Iterable[object],
    task: str = contract.TASK_ROTATE,
) -> bool:
    """Classify one endpoint using closed, independent XYZ bounds.

    The Euler vector must use the wrapped fixed-XYZ convention returned by
    :func:`wrapped_euler_delta_deg`.  L2 norms are deliberately not accepted:
    one scalar cannot determine membership in an axis-aligned tolerance box.
    """

    threshold = threshold_for_object(object_kind, task)
    try:
        position_error = _vector3(
            position_error_xyz_m, "position_error_xyz_m"
        )
        euler_error = _vector3(
            euler_error_xyz_deg, "euler_error_xyz_deg"
        )
    except (TypeError, ValueError):
        return False
    return bool(
        all(
            abs(value) <= threshold.position_axis_abs_m
            for value in position_error
        )
        and all(
            abs(value) <= threshold.euler_axis_abs_deg
            for value in euler_error
        )
    )


def pose_success(
    object_kind: str,
    actual_xyz_m: Iterable[object],
    target_xyz_m: Iterable[object],
    actual_euler_deg: Iterable[object],
    target_euler_deg: Iterable[object],
    task: str = contract.TASK_ROTATE,
) -> bool:
    return endpoint_success(
        object_kind,
        tuple(
            actual - target
            for actual, target in zip(
                _vector3(actual_xyz_m, "actual_xyz_m"),
                _vector3(target_xyz_m, "target_xyz_m"),
            )
        ),
        wrapped_euler_delta_deg(actual_euler_deg, target_euler_deg),
        task,
    )


__all__ = [
    "ROTATION_THRESHOLD",
    "SuccessThreshold",
    "TRANSLATION_THRESHOLD",
    "endpoint_success",
    "euler_l2_error_deg",
    "euclidean_position_error_m",
    "pose_success",
    "threshold_for_object",
    "wrapped_euler_delta_deg",
]
