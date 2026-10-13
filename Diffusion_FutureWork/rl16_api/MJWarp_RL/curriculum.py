"""Dependency-free seeded path and rotation curriculum sampling."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Protocol

from . import contract


PHASE = "all"
PHASES = (PHASE,)
REGION_ALL = "all"
# Pre-v19 readers may import these names, but they are identical aliases and
# are intentionally omitted from ``__all__``.  New runtime code must use
# ``PHASE``/``REGION_ALL`` and can never schedule a second run through them.
PHASE1 = PHASE
PHASE2 = PHASE
REGION_PHASE1 = REGION_ALL
REGION_PHASE2_MAIN = REGION_ALL

class RandomSource(Protocol):
    def random(self) -> float: ...

    def randrange(self, start: int, stop: int | None = None) -> int: ...


@dataclass(frozen=True)
class TranslationSample:
    requested_phase: str
    source_region: str
    path_seed: int
    path_shape_index: int
    path_step_m: float
    path_length_m: float
    waypoints_xy_m: tuple[tuple[float, float], ...]

    @property
    def delta_m(self) -> tuple[float, float, float]:
        endpoint = self.waypoints_xy_m[-1]
        return (endpoint[0], endpoint[1], contract.TRANSLATION_DZ_M)


@dataclass(frozen=True)
class RotationSample:
    requested_phase: str
    source_region: str
    rotation_angle_deg: int


def validate_phase(phase: str) -> str:
    value = str(phase)
    if value not in PHASES:
        raise ValueError(f"unsupported curriculum phase: {value!r}")
    return value


def _closed_integer(rng: RandomSource, low: int, high: int) -> int:
    return rng.randrange(low, high + 1)


def _validate_closed_integer(
    value: object,
    *,
    name: str,
    low: int,
    high: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    integer = int(numeric)
    if integer < low or integer > high:
        raise ValueError(
            f"{name} must be in the closed integer interval [{low}, {high}]"
        )
    return integer


def rotation_angle_bounds_deg(phase: str) -> tuple[int, int]:
    validate_phase(phase)
    return (
        contract.ROTATION_ANGLE_MIN_DEG,
        contract.ROTATION_ANGLE_MAX_DEG,
    )


def validate_rotation_angle_deg(
    phase: str,
    rotation_angle_deg: object,
) -> int:
    low, high = rotation_angle_bounds_deg(phase)
    return _validate_closed_integer(
        rotation_angle_deg,
        name="rotation_angle_deg",
        low=low,
        high=high,
    )


def sample_rotation(
    phase: str,
    rng: RandomSource | None = None,
) -> RotationSample:
    selected_phase = validate_phase(phase)
    source: RandomSource = random if rng is None else rng
    low, high = rotation_angle_bounds_deg(selected_phase)
    sample = RotationSample(
        requested_phase=selected_phase,
        source_region=REGION_ALL,
        rotation_angle_deg=_closed_integer(source, low, high),
    )
    validate_rotation_sample(sample)
    return sample


def validate_rotation_sample(sample: RotationSample) -> None:
    phase = validate_phase(sample.requested_phase)
    if sample.source_region != REGION_ALL:
        raise ValueError("rotation source region must be the full task")
    validate_rotation_angle_deg(phase, sample.rotation_angle_deg)


def _path_shape_index(value: object) -> int:
    return _validate_closed_integer(
        value,
        name="path_shape_index",
        low=0,
        high=len(contract.TRANSLATION_PATH_SHAPES) - 1,
    )


def translation_path_step_m(phase: str, path_shape_index: object) -> float:
    """Return fixed arc length per interval; phase does not change geometry."""

    validate_phase(phase)
    return contract.TRANSLATION_PATH_STEP_M_BY_SHAPE[
        _path_shape_index(path_shape_index)
    ]


def translation_path_length_m(phase: str, path_shape_index: object) -> float:
    """Return exact canonical arc length; phase does not change geometry."""

    validate_phase(phase)
    return contract.TRANSLATION_PATH_LENGTH_M_BY_SHAPE[
        _path_shape_index(path_shape_index)
    ]


def translation_path_waypoints_m(
    phase: str,
    path_shape_index: object,
    path_seed: object,
) -> tuple[tuple[float, float], ...]:
    validate_phase(phase)
    shape = _path_shape_index(path_shape_index)
    _validate_closed_integer(
        path_seed,
        name="path_seed",
        low=contract.TRANSLATION_PATH_SEED_MIN,
        high=contract.TRANSLATION_PATH_SEED_MAX,
    )
    return contract.TRANSLATION_PATH_WAYPOINTS_XY_M[shape]


def sample_translation(
    phase: str,
    rng: RandomSource | None = None,
) -> TranslationSample:
    """Sample one of the five deterministic full-task paths."""

    selected_phase = validate_phase(phase)
    source: RandomSource = random if rng is None else rng
    shape = source.randrange(len(contract.TRANSLATION_PATH_SHAPES))
    seed = contract.TRANSLATION_PATH_SEED_MIN
    step = translation_path_step_m(selected_phase, shape)
    sample = TranslationSample(
        requested_phase=selected_phase,
        source_region=REGION_ALL,
        path_seed=seed,
        path_shape_index=shape,
        path_step_m=step,
        path_length_m=translation_path_length_m(selected_phase, shape),
        waypoints_xy_m=translation_path_waypoints_m(
            selected_phase, shape, seed
        ),
    )
    validate_translation_sample(sample)
    return sample


def validate_translation_sample(sample: TranslationSample) -> None:
    phase = validate_phase(sample.requested_phase)
    if sample.source_region != REGION_ALL:
        raise ValueError("translation source region must be the full task")
    seed = _validate_closed_integer(
        sample.path_seed,
        name="path_seed",
        low=contract.TRANSLATION_PATH_SEED_MIN,
        high=contract.TRANSLATION_PATH_SEED_MAX,
    )
    shape = _path_shape_index(sample.path_shape_index)
    step = translation_path_step_m(phase, shape)
    length = translation_path_length_m(phase, shape)
    if not math.isclose(sample.path_step_m, step, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("translation path step does not match its fixed shape")
    if not math.isclose(sample.path_length_m, length, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("translation path length does not match its fixed shape")
    expected = translation_path_waypoints_m(phase, shape, seed)
    if len(sample.waypoints_xy_m) != contract.TRANSLATION_PATH_WAYPOINTS:
        raise ValueError("translation path must contain exactly 21 waypoints")
    if any(
        len(point) != 2
        or not all(math.isfinite(float(value)) for value in point)
        for point in sample.waypoints_xy_m
    ):
        raise ValueError("translation path waypoints must be finite XY pairs")
    if any(
        not math.isclose(float(actual), target, rel_tol=0.0, abs_tol=1e-12)
        for actual_point, expected_point in zip(sample.waypoints_xy_m, expected)
        for actual, target in zip(actual_point, expected_point)
    ):
        raise ValueError("translation path waypoints do not match seed/shape")
    if max(math.hypot(*point) for point in expected) > (
        contract.TRANSLATION_PATH_GENERATION_RADIUS_M + 1e-12
    ):
        raise ValueError("translation path leaves the 100 mm disk")
    if math.hypot(*expected[0]) > (
        contract.TRANSLATION_PATH_START_RADIUS_MAX_M + 1e-12
    ):
        raise ValueError("translation lead-in start radius is too large")
    if not math.isclose(
        step * contract.TRANSLATION_PATH_SEGMENTS,
        length,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("translation path arc-length table is inconsistent")


def summarize_samples(
    samples: Iterable[TranslationSample],
) -> dict[str, object]:
    values = tuple(samples)
    if not values:
        raise ValueError("at least one sample is required")
    for sample in values:
        validate_translation_sample(sample)
    return {
        "count": len(values),
        "source_region_counts": {
            REGION_ALL: sum(
                sample.source_region == REGION_ALL for sample in values
            )
        },
        "shape_counts": {
            name: sum(sample.path_shape_index == index for sample in values)
            for index, name in enumerate(contract.TRANSLATION_PATH_SHAPES)
        },
        "path_length_min_m": min(sample.path_length_m for sample in values),
        "path_length_max_m": max(sample.path_length_m for sample in values),
        "maximum_waypoint_radius_m": max(
            math.hypot(*point)
            for sample in values
            for point in sample.waypoints_xy_m
        ),
    }


def summarize_rotation_samples(
    samples: Iterable[RotationSample],
) -> dict[str, object]:
    values = tuple(samples)
    if not values:
        raise ValueError("at least one sample is required")
    for sample in values:
        validate_rotation_sample(sample)
    angles = tuple(sample.rotation_angle_deg for sample in values)
    return {
        "count": len(values),
        "angle_min_deg": min(angles),
        "angle_max_deg": max(angles),
        "angle_unique_count": len(set(angles)),
        "source_region_counts": {
            REGION_ALL: sum(
                sample.source_region == REGION_ALL for sample in values
            )
        },
    }


__all__ = [
    "PHASE",
    "PHASES",
    "REGION_ALL",
    "RotationSample",
    "TranslationSample",
    "rotation_angle_bounds_deg",
    "sample_rotation",
    "sample_translation",
    "summarize_rotation_samples",
    "summarize_samples",
    "translation_path_length_m",
    "translation_path_step_m",
    "translation_path_waypoints_m",
    "validate_phase",
    "validate_rotation_angle_deg",
    "validate_rotation_sample",
    "validate_translation_sample",
]
