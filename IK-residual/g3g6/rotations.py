"""Phase-3 single-, double-, and triple-axis rotation targets."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from . import config
from .math3d import euler_xyz_deg_to_quat, minimum_jerk
from .trajectories import PHASE_BOUNDS, TERMINAL_HOLD_TICKS, TOTAL_TICKS


ROTATION_MOTION_TICKS = TOTAL_TICKS - TERMINAL_HOLD_TICKS
if ROTATION_MOTION_TICKS != config.ACTION_MOTION_TICKS:
    raise AssertionError("rotation motion tick contract diverged")


def validate_axis_set(axes: str) -> str:
    value = str(axes).lower()
    if value not in config.ROTATION_AXIS_SETS:
        raise ValueError(f"unknown axes {axes!r}; expected {config.ROTATION_AXIS_SETS}")
    return value


def validate_rotation_sampling_profile(profile: str) -> str:
    value = str(profile).lower()
    if value not in config.ROTATION_SAMPLING_PROFILES:
        raise ValueError(
            f"unknown rotation sampling profile {profile!r}; expected "
            f"{config.ROTATION_SAMPLING_PROFILES}"
        )
    return value


def sample_goal_euler_deg(
    rng: np.random.Generator,
    axes: str | None = None,
    *,
    profile: str = config.DEFAULT_ROTATION_SAMPLING_PROFILE,
) -> tuple[str, np.ndarray]:
    sampling_profile = validate_rotation_sampling_profile(profile)
    selected = (
        validate_axis_set(axes)
        if axes is not None
        else str(rng.choice(config.ROTATION_AXIS_SETS))
    )
    goal = np.zeros(3, dtype=np.float64)

    if sampling_profile == config.WEIGHTED_ROTATION_SAMPLING_PROFILE:
        bin_index = int(
            rng.choice(
                len(config.WEIGHTED_ROTATION_MAGNITUDE_BINS_DEG),
                p=config.WEIGHTED_ROTATION_BIN_PROBABILITIES,
            )
        )
        lower, upper = config.WEIGHTED_ROTATION_MAGNITUDE_BINS_DEG[bin_index]
        # One common absolute component magnitude is shared by every active
        # axis.  Signs remain independent, so multi-axis tasks cover every
        # signed octant without changing their commanded component norm.
        magnitude = int(rng.integers(lower, upper + 1))
        for index, axis in enumerate("xyz"):
            if axis in selected:
                goal[index] = magnitude * (
                    -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
                )
        return selected, goal

    # Preserve the original 10--45 degree sampler exactly: each active axis
    # receives its own independently sampled integer magnitude and sign.
    for index, axis in enumerate("xyz"):
        if axis in selected:
            magnitude = int(
                rng.integers(
                    config.ROTATION_ANGLE_MIN_DEG,
                    config.ROTATION_ANGLE_MAX_DEG + 1,
                )
            )
            goal[index] = magnitude * (-1.0 if int(rng.integers(0, 2)) == 0 else 1.0)
    return selected, goal


def validate_goal_euler_deg(
    axes: str,
    euler_deg: Iterable[float],
    *,
    max_magnitude_deg: float = config.ROTATION_ANGLE_MAX_DEG,
) -> np.ndarray:
    selected = validate_axis_set(axes)
    maximum = float(max_magnitude_deg)
    if (
        not np.isfinite(maximum)
        or maximum < config.ROTATION_ANGLE_MIN_DEG
        or maximum > config.ROTATION_SWEEP_ANGLE_MAX_DEG
    ):
        raise ValueError(
            "maximum rotation magnitude must be finite and inside "
            f"[{config.ROTATION_ANGLE_MIN_DEG}, "
            f"{config.ROTATION_SWEEP_ANGLE_MAX_DEG:g}] degrees"
        )
    value = np.asarray(tuple(euler_deg), dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(value)):
        raise ValueError("rotation goal must be finite")
    for index, axis in enumerate("xyz"):
        magnitude = abs(float(value[index]))
        if axis in selected:
            if not config.ROTATION_ANGLE_MIN_DEG <= magnitude <= maximum:
                raise ValueError(
                    "active-axis magnitudes must be inside "
                    f"[{config.ROTATION_ANGLE_MIN_DEG}, {maximum:g}] degrees"
                )
        elif magnitude > 1.0e-12:
            raise ValueError("inactive rotation axes must be zero")
    return value


def build_rotation_euler_schedule(goal_euler_deg: Iterable[float]) -> np.ndarray:
    goal = np.asarray(tuple(goal_euler_deg), dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(goal)):
        raise ValueError("rotation goal must be finite")
    progress = np.array(
        [minimum_jerk((tick + 1) / ROTATION_MOTION_TICKS) for tick in range(ROTATION_MOTION_TICKS)],
        dtype=np.float64,
    )
    schedule = np.empty((TOTAL_TICKS, 3), dtype=np.float64)
    schedule[:ROTATION_MOTION_TICKS] = progress[:, None] * goal[None, :]
    schedule[ROTATION_MOTION_TICKS:] = goal
    return schedule


def build_rotation_quaternion_schedule(goal_euler_deg: Iterable[float]) -> np.ndarray:
    euler_schedule = build_rotation_euler_schedule(goal_euler_deg)
    return np.stack(
        [euler_xyz_deg_to_quat(euler) for euler in euler_schedule], axis=0
    )


def build_coupled_rotation_euler_schedule(
    goal_euler_deg: Iterable[float],
) -> np.ndarray:
    """Synchronize a rotation with the 1,800-tick translation route.

    Lead-in and settle retain the measured post-lift orientation, rotation is
    minimum-jerk over the route itself, and the terminal hold retains the
    final goal.  This gives analysis a genuine, monotonic translation-distance
    horizontal axis instead of assigning a fictitious distance to Phase 3.
    """

    goal = np.asarray(tuple(goal_euler_deg), dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(goal)):
        raise ValueError("rotation goal must be finite")
    route_start, route_stop = PHASE_BOUNDS["route"]
    route_ticks = route_stop - route_start
    progress = np.array(
        [minimum_jerk((tick + 1) / route_ticks) for tick in range(route_ticks)],
        dtype=np.float64,
    )
    schedule = np.zeros((TOTAL_TICKS, 3), dtype=np.float64)
    schedule[route_start:route_stop] = progress[:, None] * goal[None, :]
    schedule[route_stop:] = goal
    return schedule


def build_coupled_rotation_quaternion_schedule(
    goal_euler_deg: Iterable[float],
) -> np.ndarray:
    euler_schedule = build_coupled_rotation_euler_schedule(goal_euler_deg)
    return np.stack(
        [euler_xyz_deg_to_quat(euler) for euler in euler_schedule], axis=0
    )
