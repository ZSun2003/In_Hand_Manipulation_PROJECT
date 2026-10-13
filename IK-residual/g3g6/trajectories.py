"""Canonical phase-2 translation paths and their fixed tick schedule.

The five paths in this module are a local, dependency-free reimplementation
of the geometry used by ``GRAB_G6``.  Geometry is expressed as an XY offset
from the post-lift object centre; Z is unchanged.  Path progress is normalized
arc length, not a generic curve parameter, so the 21 points at ``k / 20`` are
equal-arc-length observation anchors.

The control schedule has exactly 2,425 targets::

    [0, 300)     minimum-jerk lead from the centre to the first path point
    [300, 600)   settle at the first path point
    [600, 2400)  minimum-jerk route
    [2400, 2425) terminal hold at the last path point

The reference progress tables contain their geometric zero sample as well as
the commanded samples.  Consequently, the 300 lead targets use table entries
1..300 and the 1,800 route targets use entries 1..1800.  The preceding centre
or settle sample already supplies table entry zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import config


FloatArray = NDArray[np.float64]

PATH_NAMES: Final[tuple[str, ...]] = ("square", "circle", "S", "U", "N")
REFERENCE_PATH_NAMES: Final[tuple[str, ...]] = (
    "square_100mm",
    "circle_r50mm",
    "letter_s_300mm",
    "letter_u_300mm",
    "letter_n_300mm",
)

SQUARE_SIDE_M: Final = 0.100
CIRCLE_RADIUS_M: Final = 0.050
LETTER_PATH_LENGTH_M: Final = 0.300
U_RADIUS_M: Final = 0.045
S_TERMINAL_EXTRA_TURN_RAD: Final = math.pi / 4.0
S_ARC_SWEEP_RAD: Final = math.pi + S_TERMINAL_EXTRA_TURN_RAD
S_ARC_LENGTH_M: Final = LETTER_PATH_LENGTH_M / 2.0
S_RADIUS_M: Final = S_ARC_LENGTH_M / S_ARC_SWEEP_RAD

PATH_LENGTHS_M: Final[tuple[float, ...]] = (
    4.0 * SQUARE_SIDE_M,
    2.0 * math.pi * CIRCLE_RADIUS_M,
    LETTER_PATH_LENGTH_M,
    LETTER_PATH_LENGTH_M,
    LETTER_PATH_LENGTH_M,
)
PATH_LENGTH_M_BY_NAME: Final[dict[str, float]] = dict(
    zip(PATH_NAMES, PATH_LENGTHS_M)
)

ANCHOR_INTERVALS: Final = 20
ANCHOR_COUNT: Final = ANCHOR_INTERVALS + 1

LEAD_IN_TICKS: Final = 300
START_SETTLE_TICKS: Final = 300
ROUTE_TICKS: Final = 1_800
TERMINAL_HOLD_TICKS: Final = config.POST_ACTION_HOLD_TICKS
# Short aliases match the phase labels used in the 2,425-tick schedule.
LEAD_TICKS: Final = LEAD_IN_TICKS
SETTLE_TICKS: Final = START_SETTLE_TICKS
HOLD_TICKS: Final = TERMINAL_HOLD_TICKS
TOTAL_TICKS: Final = (
    LEAD_IN_TICKS
    + START_SETTLE_TICKS
    + ROUTE_TICKS
    + TERMINAL_HOLD_TICKS
)
if TOTAL_TICKS != config.ACTION_SCHEDULE_TICKS:
    raise AssertionError("translation/action schedule tick contract diverged")

SQUARE_STROKE_TICKS: Final[tuple[int, ...]] = (450, 450, 450, 450)
N_STROKE_TICKS: Final[tuple[int, ...]] = (555, 690, 555)
ROUTE_STROKE_TICKS: Final[tuple[tuple[int, ...], ...]] = (
    SQUARE_STROKE_TICKS,
    (ROUTE_TICKS,),
    (ROUTE_TICKS,),
    (ROUTE_TICKS,),
    N_STROKE_TICKS,
)
ROUTE_STROKE_FRACTIONS: Final[tuple[tuple[float, ...], ...]] = (
    (0.25, 0.25, 0.25, 0.25),
    (1.0,),
    (1.0,),
    (1.0,),
    (4.0 / 13.0, 5.0 / 13.0, 4.0 / 13.0),
)

PHASE_BOUNDS: Final[dict[str, tuple[int, int]]] = {
    "lead_in": (0, LEAD_IN_TICKS),
    "settle": (LEAD_IN_TICKS, LEAD_IN_TICKS + START_SETTLE_TICKS),
    "route": (
        LEAD_IN_TICKS + START_SETTLE_TICKS,
        LEAD_IN_TICKS + START_SETTLE_TICKS + ROUTE_TICKS,
    ),
    "hold": (
        LEAD_IN_TICKS + START_SETTLE_TICKS + ROUTE_TICKS,
        TOTAL_TICKS,
    ),
}
PHASE_SLICES: Final[dict[str, slice]] = {
    name: slice(start, stop) for name, (start, stop) in PHASE_BOUNDS.items()
}

QUANTIZATION_DECIMALS: Final = 12
QUANTIZATION_SCALE: Final = 10**QUANTIZATION_DECIMALS

_ALIASES: Final[dict[str, int]] = {
    "square": 0,
    "square_100mm": 0,
    "circle": 1,
    "circle_r50mm": 1,
    "s": 2,
    "letter_s": 2,
    "letter_s_300mm": 2,
    "u": 3,
    "letter_u": 3,
    "letter_u_300mm": 3,
    "n": 4,
    "letter_n": 4,
    "letter_n_300mm": 4,
}


def _path_index(path: str | int) -> int:
    """Return the canonical table index for a supported name or index."""

    if isinstance(path, bool):
        raise ValueError("path must be a path name or an integer index")
    if isinstance(path, Integral):
        index = int(path)
        if 0 <= index < len(PATH_NAMES):
            return index
        raise ValueError(f"path index must be in [0, {len(PATH_NAMES) - 1}]")
    if not isinstance(path, str):
        raise ValueError("path must be a path name or an integer index")
    key = path.strip().casefold()
    try:
        return _ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(PATH_NAMES)
        raise ValueError(f"unknown path {path!r}; expected one of {supported}") from exc


def normalize_path_name(path: str | int) -> str:
    """Normalize aliases such as ``letter_s_300mm`` to ``S``."""

    return PATH_NAMES[_path_index(path)]


def _quantize(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("trajectory values must be finite")
    return np.rint(array * QUANTIZATION_SCALE) / QUANTIZATION_SCALE


def minimum_jerk(progress: ArrayLike) -> float | FloatArray:
    """Evaluate ``10t^3 - 15t^4 + 6t^5`` on finite ``t`` in ``[0, 1]``."""

    value = np.asarray(progress, dtype=np.float64)
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("minimum-jerk progress must be finite and inside [0, 1]")
    result = value**3 * (10.0 + value * (-15.0 + 6.0 * value))
    if result.ndim == 0:
        return float(result)
    return result


def canonical_point_xy(path: str | int, progress: ArrayLike) -> FloatArray:
    """Evaluate a path at normalized arc-length ``progress``.

    A scalar progress returns shape ``(2,)``.  An input with shape ``S``
    returns shape ``S + (2,)``.
    """

    index = _path_index(path)
    value = np.asarray(progress, dtype=np.float64)
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("path progress must be finite and inside [0, 1]")

    if index == 0:
        side = SQUARE_SIDE_M
        half = side / 2.0
        distance = 4.0 * side * value
        x = np.where(
            distance <= side,
            half - distance,
            np.where(
                distance <= 2.0 * side,
                -half,
                np.where(distance <= 3.0 * side, -half + distance - 2.0 * side, half),
            ),
        )
        y = np.where(
            distance <= side,
            half,
            np.where(
                distance <= 2.0 * side,
                half - (distance - side),
                np.where(distance <= 3.0 * side, -half, -half + distance - 3.0 * side),
            ),
        )
        return np.stack((x, y), axis=-1)

    if index == 1:
        angle = 2.0 * math.pi * value
        x = CIRCLE_RADIUS_M * np.cos(angle)
        y = CIRCLE_RADIUS_M * np.sin(angle)
        # Pin the closed endpoint instead of retaining sin(2*pi) roundoff.
        x = np.where(value == 1.0, CIRCLE_RADIUS_M, x)
        y = np.where(value == 1.0, 0.0, y)
        return np.stack((x, y), axis=-1)

    distance = LETTER_PATH_LENGTH_M * value

    if index == 2:
        # Two tangent-continuous 225-degree arcs form a more recognisable S.
        # Relative to the former two-semicircle construction, the upper and
        # lower outer terminals each curl an additional 45 degrees.  Shrinking
        # the common radius keeps each lobe at exactly 150 mm and the complete
        # route at exactly 300 mm.
        first = distance <= S_ARC_LENGTH_M
        angle_first = (
            S_TERMINAL_EXTRA_TURN_RAD + distance / S_RADIUS_M
        )
        angle_second = (
            math.pi / 2.0
            - (distance - S_ARC_LENGTH_M) / S_RADIUS_M
        )
        x = np.where(
            first,
            S_RADIUS_M * np.cos(angle_first),
            S_RADIUS_M * np.cos(angle_second),
        )
        y = np.where(
            first,
            S_RADIUS_M + S_RADIUS_M * np.sin(angle_first),
            -S_RADIUS_M + S_RADIUS_M * np.sin(angle_second),
        )
        terminal_x = S_RADIUS_M / math.sqrt(2.0)
        terminal_y = S_RADIUS_M * (1.0 + 1.0 / math.sqrt(2.0))
        x = np.where(value == 0.0, terminal_x, x)
        y = np.where(value == 0.0, terminal_y, y)
        x = np.where(value == 0.5, 0.0, x)
        y = np.where(value == 0.5, 0.0, y)
        x = np.where(value == 1.0, -terminal_x, x)
        y = np.where(value == 1.0, -terminal_y, y)
        return np.stack((x, y), axis=-1)

    if index == 3:
        radius = U_RADIUS_M
        vertical = (LETTER_PATH_LENGTH_M - math.pi * radius) / 2.0
        center_y = (radius - vertical) / 2.0
        in_left = distance <= vertical
        in_arc = distance <= vertical + math.pi * radius
        angle = math.pi + (distance - vertical) / radius
        x = np.where(
            in_left,
            -radius,
            np.where(in_arc, radius * np.cos(angle), radius),
        )
        y = np.where(
            in_left,
            center_y + vertical - distance,
            np.where(
                in_arc,
                center_y + radius * np.sin(angle),
                center_y + distance - vertical - math.pi * radius,
            ),
        )
        return np.stack((x, y), axis=-1)

    scale = LETTER_PATH_LENGTH_M / 13.0
    width = 3.0 * scale
    height = 4.0 * scale
    diagonal = 5.0 * scale
    in_left = distance <= height
    in_diagonal = distance <= height + diagonal
    along = distance - height
    x = np.where(
        in_left,
        -width / 2.0,
        np.where(in_diagonal, -width / 2.0 + width * along / diagonal, width / 2.0),
    )
    y = np.where(
        in_left,
        -height / 2.0 + distance,
        np.where(
            in_diagonal,
            height / 2.0 - height * along / diagonal,
            -height / 2.0 + distance - height - diagonal,
        ),
    )
    return np.stack((x, y), axis=-1)


# Concise alias for callers that already know the path is planar.
canonical_point = canonical_point_xy


def canonical_anchors_xy(path: str | int) -> FloatArray:
    """Return the fixed 21 equal-arc-length XY anchors, quantized to 1 pm."""

    progress = np.linspace(0.0, 1.0, ANCHOR_COUNT, dtype=np.float64)
    return _quantize(canonical_point_xy(path, progress))


canonical_anchors = canonical_anchors_xy


def lead_in_progress() -> FloatArray:
    """Return the 301-entry inclusive minimum-jerk lead progress table."""

    tau = np.arange(LEAD_IN_TICKS + 1, dtype=np.float64) / LEAD_IN_TICKS
    result = _quantize(minimum_jerk(tau))
    result[0] = 0.0
    result[-1] = 1.0
    return result


def lead_in_points_xy(path: str | int) -> FloatArray:
    """Return the inclusive 301-point centre-to-path-start lead table."""

    start = _quantize(canonical_point_xy(path, 0.0))
    # Match the authority table's composition order: both the start and the
    # minimum-jerk fraction are quantized before their product is quantized.
    return _quantize(lead_in_progress()[:, None] * start[None, :])


def route_progress(path: str | int) -> FloatArray:
    """Return the inclusive 1,801-entry per-stroke minimum-jerk progress."""

    index = _path_index(path)
    chunks: list[FloatArray] = [np.zeros(1, dtype=np.float64)]
    base = 0.0
    for ticks, fraction in zip(
        ROUTE_STROKE_TICKS[index], ROUTE_STROKE_FRACTIONS[index]
    ):
        tau = np.arange(1, ticks + 1, dtype=np.float64) / ticks
        chunks.append(_quantize(base + fraction * minimum_jerk(tau)))
        base += fraction
    result = np.concatenate(chunks)
    if result.shape != (ROUTE_TICKS + 1,):
        raise RuntimeError("internal route stroke schedule does not total 1,800 ticks")
    result[0] = 0.0
    result[-1] = 1.0
    return result


def route_points_xy(path: str | int) -> FloatArray:
    """Return the inclusive 1,801-point quantized route geometry."""

    return _quantize(canonical_point_xy(path, route_progress(path)))


route_points = route_points_xy


def build_schedule(
    path: str | int,
    origin_xyz: ArrayLike = (0.0, 0.0, 0.0),
) -> FloatArray:
    """Build the fixed ``(2425, 3)`` world-frame target schedule.

    ``origin_xyz`` is the post-lift centre about which the canonical XY
    offsets are applied.  The returned Z axis therefore equals the supplied
    origin Z for every tick.
    """

    origin = np.asarray(origin_xyz, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin_xyz must contain exactly three finite values")

    lead = lead_in_points_xy(path)
    route = route_points_xy(path)
    start = route[0]
    endpoint = route[-1]

    offsets_xy = np.concatenate(
        (
            lead[1:],
            np.repeat(start[None, :], START_SETTLE_TICKS, axis=0),
            route[1:],
            np.repeat(endpoint[None, :], TERMINAL_HOLD_TICKS, axis=0),
        ),
        axis=0,
    )
    if offsets_xy.shape != (TOTAL_TICKS, 2):
        raise RuntimeError("internal translation schedule does not total 2,425 ticks")

    schedule = np.empty((TOTAL_TICKS, 3), dtype=np.float64)
    schedule[:, :2] = offsets_xy + origin[:2]
    schedule[:, 2] = origin[2]
    return schedule


def scheduled_path(
    path: str | int,
    center_xyz: ArrayLike | None = None,
) -> FloatArray:
    """Return the 2,425 targets relative to, or centred at, ``center_xyz``.

    ``None`` deliberately means a relative-offset schedule.  The named
    parameter mirrors the environment API while :func:`build_schedule` keeps
    the more explicit ``origin_xyz`` spelling.
    """

    if center_xyz is None:
        center_xyz = (0.0, 0.0, 0.0)
    return build_schedule(path, center_xyz)


build_translation_schedule = scheduled_path


def phase_for_tick(tick: int) -> str:
    """Return ``lead_in``, ``settle``, ``route``, or ``hold`` for a tick."""

    if isinstance(tick, bool) or not isinstance(tick, Integral):
        raise ValueError("tick must be an integer")
    value = int(tick)
    if not 0 <= value < TOTAL_TICKS:
        raise ValueError(f"tick must be in [0, {TOTAL_TICKS - 1}]")
    for phase, (start, stop) in PHASE_BOUNDS.items():
        if start <= value < stop:
            return phase
    raise RuntimeError("internal phase bounds do not cover the schedule")


@dataclass(frozen=True)
class CanonicalTrajectory:
    """Convenient immutable-by-contract bundle for one canonical path."""

    name: str
    reference_name: str
    length_m: float
    anchors_xy: FloatArray
    route_xy: FloatArray
    schedule_xyz: FloatArray


def make_trajectory(
    path: str | int,
    origin_xyz: ArrayLike = (0.0, 0.0, 0.0),
) -> CanonicalTrajectory:
    """Return geometry, anchors, and the complete schedule in one bundle."""

    index = _path_index(path)
    return CanonicalTrajectory(
        name=PATH_NAMES[index],
        reference_name=REFERENCE_PATH_NAMES[index],
        length_m=PATH_LENGTHS_M[index],
        anchors_xy=canonical_anchors_xy(index),
        route_xy=route_points_xy(index),
        schedule_xyz=build_schedule(index, origin_xyz),
    )


__all__ = [
    "ANCHOR_COUNT",
    "ANCHOR_INTERVALS",
    "CIRCLE_RADIUS_M",
    "CanonicalTrajectory",
    "HOLD_TICKS",
    "LEAD_IN_TICKS",
    "LEAD_TICKS",
    "LETTER_PATH_LENGTH_M",
    "N_STROKE_TICKS",
    "PATH_LENGTHS_M",
    "PATH_LENGTH_M_BY_NAME",
    "PATH_NAMES",
    "PHASE_BOUNDS",
    "PHASE_SLICES",
    "REFERENCE_PATH_NAMES",
    "ROUTE_STROKE_TICKS",
    "ROUTE_TICKS",
    "S_ARC_LENGTH_M",
    "S_ARC_SWEEP_RAD",
    "S_RADIUS_M",
    "S_TERMINAL_EXTRA_TURN_RAD",
    "SQUARE_SIDE_M",
    "SQUARE_STROKE_TICKS",
    "START_SETTLE_TICKS",
    "SETTLE_TICKS",
    "TERMINAL_HOLD_TICKS",
    "TOTAL_TICKS",
    "U_RADIUS_M",
    "build_schedule",
    "build_translation_schedule",
    "canonical_anchors",
    "canonical_anchors_xy",
    "canonical_point",
    "canonical_point_xy",
    "lead_in_points_xy",
    "lead_in_progress",
    "make_trajectory",
    "minimum_jerk",
    "normalize_path_name",
    "phase_for_tick",
    "route_points_xy",
    "route_points",
    "route_progress",
    "scheduled_path",
]
