"""Dependency-free readers and aggregators for live GRAB training metrics.

The training process owns ``training.jsonl``.  This module is deliberately
read-only and tolerant of an incomplete final line, malformed historical
records, and schema additions.  It can therefore be used while a run is
actively appending to the log.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MAX_JSONL_LINE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class JsonlReadResult:
    """Records and data-quality counters from a JSONL snapshot."""

    records: tuple[dict[str, Any], ...]
    invalid_lines: int = 0
    oversized_lines: int = 0
    ignored_partial_line: bool = False
    error: str | None = None


@dataclass
class _WeightedMean:
    total: float = 0.0
    weight: float = 0.0
    samples: int = 0
    exact_weight: bool = True

    def add(
        self,
        value: object,
        *,
        weight: object = 1.0,
        exact_weight: bool = True,
    ) -> None:
        number = _finite_float(value)
        sample_weight = _finite_float(weight)
        if (
            number is None
            or sample_weight is None
            or sample_weight <= 0.0
        ):
            return
        self.total += number * sample_weight
        self.weight += sample_weight
        self.samples += 1
        self.exact_weight = self.exact_weight and exact_weight

    @property
    def mean(self) -> float | None:
        return self.total / self.weight if self.weight > 0.0 else None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _decode_json_object(text: str) -> dict[str, Any] | None:
    value = json.loads(text, parse_constant=_reject_json_constant)
    return value if isinstance(value, dict) else None


def read_jsonl(path: Path | str) -> JsonlReadResult:
    """Read a safe point-in-time JSONL snapshot without locking the writer."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    invalid = 0
    oversized = 0
    ignored_partial = False
    try:
        with source.open("rb") as stream:
            for raw_line in stream:
                if not raw_line.endswith(b"\n"):
                    ignored_partial = True
                    continue
                if len(raw_line) > MAX_JSONL_LINE_BYTES:
                    oversized += 1
                    continue
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    text = stripped.decode("utf-8")
                    record = _decode_json_object(text)
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    ValueError,
                    RecursionError,
                ):
                    invalid += 1
                    continue
                if record is None:
                    invalid += 1
                    continue
                records.append(record)
    except FileNotFoundError:
        return JsonlReadResult(records=())
    except OSError as exc:
        return JsonlReadResult(
            records=tuple(records),
            invalid_lines=invalid,
            oversized_lines=oversized,
            ignored_partial_line=ignored_partial,
            error=f"{type(exc).__name__}: {exc}",
        )
    return JsonlReadResult(
        records=tuple(records),
        invalid_lines=invalid,
        oversized_lines=oversized,
        ignored_partial_line=ignored_partial,
    )


def read_json_object(path: Path | str) -> tuple[dict[str, Any], str | None]:
    """Read an optional JSON object, returning an error instead of raising."""

    source = Path(path)
    try:
        if source.stat().st_size > MAX_JSONL_LINE_BYTES:
            return {}, "file exceeds safe JSON size limit"
        text = source.read_text(encoding="utf-8")
        value = _decode_json_object(text)
        if value is None:
            return {}, "top-level JSON value is not an object"
        return value, None
    except FileNotFoundError:
        return {}, None
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _path_get(mapping: Mapping[str, Any], path: str) -> object:
    current: object = mapping
    for name in path.split("."):
        if not isinstance(current, Mapping) or name not in current:
            return None
        current = current[name]
    return current


def _first(mapping: Mapping[str, Any], paths: Sequence[str]) -> object:
    for path in paths:
        value = _path_get(mapping, path)
        if value is not None:
            return value
    return None


def _name(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


def _finite_values(value: object) -> list[float]:
    values: list[float] = []
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            pending.extend(reversed(item))
            continue
        number = _finite_float(item)
        if number is not None:
            values.append(number)
    return values


SUCCESS_RATE_PATHS = (
    "endpoint_success_rate",
    "success_rate",
    "metrics.endpoint_success_rate",
    "metrics.success_rate",
    "evaluation.success_rate",
)
LATEST_SUCCESS_RATE_PATHS = (
    "latest_endpoint_success_rate",
    "metrics.latest_endpoint_success_rate",
)
CUMULATIVE_SUCCESS_RATE_PATHS = (
    "cumulative_endpoint_success_rate",
    "metrics.cumulative_endpoint_success_rate",
)
CUMULATIVE_SUCCESS_COUNT_PATHS = (
    "cumulative_endpoint_success_count",
    "metrics.cumulative_endpoint_success_count",
)
CUMULATIVE_EPISODE_COUNT_PATHS = (
    "cumulative_endpoint_episode_count",
    "metrics.cumulative_endpoint_episode_count",
)
SUCCESS_COUNT_PATHS = (
    "endpoint_success_count",
    "success_count",
    "successful_episodes",
    "metrics.success_count",
)
EPISODE_COUNT_PATHS = (
    "endpoint_episode_count",
    "endpoint_count",
    "completed_episode_count",
    "completed_episodes",
    "episode_count",
    "episodes",
    "metrics.episode_count",
)
OBJECT_NAME_PATHS = (
    "object",
    "object_name",
    "curriculum.object",
    "metrics.object",
)
TASK_NAME_PATHS = (
    "task",
    "task_name",
    "curriculum.task",
    "metrics.task",
)
POSITION_ERROR_PATHS = (
    "position_error_m",
    "pose.position_error_m",
    "metrics.position_error_m",
    "evaluation.position_error_m",
)
EULER_ERROR_PATHS = (
    "euler_l2_error_deg",
    "pose.euler_l2_error_deg",
    "metrics.euler_l2_error_deg",
    "evaluation.euler_l2_error_deg",
)
TRANSLATION_DESIRED_DISPLACEMENT_PATHS = (
    "translation_endpoint_desired_final_displacement_l2_mean_m",
    "desired_final_displacement_l2_m",
    "metrics.desired_final_displacement_l2_m",
)
TRANSLATION_ACTUAL_DISPLACEMENT_PATHS = (
    "translation_endpoint_actual_final_displacement_l2_mean_m",
    "actual_final_displacement_l2_m",
    "metrics.actual_final_displacement_l2_m",
)
TRANSLATION_DISPLACEMENT_ERROR_PATHS = (
    "translation_endpoint_final_displacement_error_l2_mean_m",
    "final_displacement_error_l2_m",
    "metrics.final_displacement_error_l2_m",
)
TRANSLATION_FINAL_EULER_PATHS = (
    "translation_endpoint_final_euler_error_l2_mean_deg",
    "final_euler_error_l2_deg",
    "metrics.final_euler_error_l2_deg",
)
PATH_TRACKING_RMSE_PATHS = (
    "translation_endpoint_path_tracking_rmse_mean_m",
    "path_tracking_rmse_m",
    "metrics.path_tracking_rmse_m",
)
PATH_TRACKING_MAX_PATHS = (
    "translation_endpoint_path_tracking_max_mean_m",
    "translation_endpoint_path_tracking_max_m",
    "path_tracking_max_error_m",
    "metrics.path_tracking_max_error_m",
)
PLANE_RATE_PATHS = (
    "grasp_plane_compliance_rate",
    "plane_compliance_rate",
    "grasp_plane_within_bounds_rate",
    "metrics.grasp_plane_compliance_rate",
    "metrics.plane_compliance_rate",
)
PLANE_BOOL_PATHS = (
    "grasp_plane_within_bounds",
    "grasp_plane_compliant",
    "plane_within_bounds",
    "metrics.grasp_plane_within_bounds",
    "metrics.grasp_plane_compliant",
)
PLANE_FRACTION_PATHS = (
    "grasp_plane_fraction",
    "grasp_plane_height_fraction",
    "metrics.grasp_plane_fraction",
)
PLANE_COUNT_PATHS = (
    "grasp_plane_sample_count",
    "plane_sample_count",
    "metrics.grasp_plane_sample_count",
)
DIFFUSION_LOSS_PATHS = (
    "diffusion_loss",
    "update.diffusion_loss",
    "diffusion.loss",
    "metrics.diffusion_loss",
    "train.diffusion_loss",
)
DIFFUSION_LR_PATHS = (
    "diffusion_lr",
    "diffusion_learning_rate",
    "update.diffusion_lr",
    "diffusion.lr",
    "diffusion.learning_rate",
    "metrics.diffusion_lr",
)
STEP_PATHS = (
    "total_steps",
    "steps",
    "global_step",
    "completed_steps",
    "progress.completed_steps",
    "progress.current",
)
ELAPSED_PATHS = (
    "elapsed_s",
    "elapsed_seconds",
    "runtime.elapsed_s",
    "progress.elapsed_s",
)


def _rate_weight(record: Mapping[str, Any]) -> tuple[float | None, float, bool]:
    count = _finite_float(_first(record, EPISODE_COUNT_PATHS))
    successes = _finite_float(_first(record, SUCCESS_COUNT_PATHS))
    rate = _finite_float(_first(record, SUCCESS_RATE_PATHS))
    if count is not None and count > 0.0:
        if successes is not None:
            return successes / count, count, True
        if rate is not None:
            return rate, count, True
    return rate, 1.0, False


def _mapping_rate_weight(value: object) -> tuple[float | None, float, bool]:
    if not isinstance(value, Mapping):
        return _finite_float(value), 1.0, False
    count = _finite_float(
        _first(
            value,
            (
                "episode_count",
                "episodes",
                "count",
                "endpoint_count",
                "sample_count",
            ),
        )
    )
    successes = _finite_float(
        _first(value, ("success_count", "successes", "successful_episodes"))
    )
    rate = _finite_float(
        _first(
            value,
            ("success_rate", "endpoint_success_rate", "rate", "mean"),
        )
    )
    if count is not None and count > 0.0:
        if successes is not None:
            return successes / count, count, True
        if rate is not None:
            return rate, count, True
    return rate, 1.0, False


def _metric_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[str],
    window: int,
    conditional_loss: bool = False,
) -> dict[str, Any]:
    def value(record: Mapping[str, Any]) -> float | None:
        found = _finite_float(_first(record, paths))
        if found is not None:
            return found
        if conditional_loss:
            stage = _name(_first(record, ("stage", "phase", "mode")))
            if stage is not None and "diffusion" in stage.lower():
                return _finite_float(
                    _first(record, ("loss", "train_loss", "update.loss"))
                )
        return None

    all_values = [number for record in records if (number := value(record)) is not None]
    recent_records = records[-window:] if window > 0 else records
    recent_values = [
        number
        for record in recent_records
        if (number := value(record)) is not None
    ]
    return {
        "latest": all_values[-1] if all_values else None,
        "recent_mean": (
            sum(recent_values) / len(recent_values) if recent_values else None
        ),
        "cumulative_mean": (
            sum(all_values) / len(all_values) if all_values else None
        ),
        "samples": len(all_values),
        "recent_samples": len(recent_values),
    }


def _success_summary(
    records: Sequence[Mapping[str, Any]], window: int
) -> dict[str, Any]:
    cumulative = _WeightedMean()
    recent = _WeightedMean()
    latest: float | None = None
    for index, record in enumerate(records):
        rate, weight, exact = _rate_weight(record)
        if rate is None:
            continue
        cumulative.add(rate, weight=weight, exact_weight=exact)
        if index >= max(0, len(records) - window):
            recent.add(rate, weight=weight, exact_weight=exact)
        latest = rate
    authoritative_cumulative: float | None = None
    authoritative_latest: float | None = None
    for record in reversed(records):
        if authoritative_latest is None:
            authoritative_latest = _finite_float(
                _first(record, LATEST_SUCCESS_RATE_PATHS)
            )
        if authoritative_cumulative is None:
            count = _finite_float(
                _first(record, CUMULATIVE_EPISODE_COUNT_PATHS)
            )
            successes = _finite_float(
                _first(record, CUMULATIVE_SUCCESS_COUNT_PATHS)
            )
            if count is not None and count > 0.0 and successes is not None:
                authoritative_cumulative = successes / count
            else:
                authoritative_cumulative = _finite_float(
                    _first(record, CUMULATIVE_SUCCESS_RATE_PATHS)
                )
        if (
            authoritative_latest is not None
            and authoritative_cumulative is not None
        ):
            break
    exact = cumulative.samples > 0 and cumulative.exact_weight
    return {
        "latest": (
            authoritative_latest
            if authoritative_latest is not None
            else latest
        ),
        "recent": recent.mean,
        "cumulative": (
            authoritative_cumulative
            if authoritative_cumulative is not None
            else cumulative.mean
        ),
        "samples": cumulative.samples,
        "recent_samples": recent.samples,
        "weighting": "episode" if exact else "log_record",
    }


def _grouped_success(
    records: Sequence[Mapping[str, Any]],
    *,
    group_paths: Sequence[str],
    mapping_paths: Sequence[str],
    cumulative_mapping_paths: Sequence[str] = (),
    window: int,
) -> dict[str, dict[str, Any]]:
    accumulators: dict[str, _WeightedMean] = {}
    recent_accumulators: dict[str, _WeightedMean] = {}

    def add(
        group: str,
        rate: object,
        weight: object,
        exact: bool,
        *,
        recent: bool,
    ) -> None:
        accumulator = accumulators.setdefault(group, _WeightedMean())
        accumulator.add(rate, weight=weight, exact_weight=exact)
        if recent:
            recent_accumulator = recent_accumulators.setdefault(
                group, _WeightedMean()
            )
            recent_accumulator.add(
                rate, weight=weight, exact_weight=exact
            )

    recent_start = max(0, len(records) - window)
    for record_index, record in enumerate(records):
        is_recent = record_index >= recent_start
        found_mapping = False
        for path in mapping_paths:
            mapping = _path_get(record, path)
            if not isinstance(mapping, Mapping):
                continue
            found_mapping = True
            for raw_group, value in mapping.items():
                group = _name(raw_group)
                if group is None:
                    continue
                rate, weight, exact = _mapping_rate_weight(value)
                add(
                    group,
                    rate,
                    weight,
                    exact,
                    recent=is_recent,
                )
            break
        if not found_mapping:
            group = _name(_first(record, group_paths))
            if group is not None:
                rate, weight, exact = _rate_weight(record)
                add(
                    group,
                    rate,
                    weight,
                    exact,
                    recent=is_recent,
                )
    authoritative: dict[str, tuple[float, float, bool]] = {}
    for record in reversed(records):
        for path in cumulative_mapping_paths:
            mapping = _path_get(record, path)
            if not isinstance(mapping, Mapping):
                continue
            for raw_group, value in mapping.items():
                group = _name(raw_group)
                if group is None or group in authoritative:
                    continue
                rate, weight, exact = _mapping_rate_weight(value)
                if rate is not None:
                    authoritative[group] = (rate, weight, exact)
            break
        if authoritative:
            break
    groups = set(accumulators) | set(authoritative)
    return {
        group: {
            "success_rate": (
                authoritative[group][0]
                if group in authoritative
                else accumulators[group].mean
            ),
            "cumulative": (
                authoritative[group][0]
                if group in authoritative
                else accumulators[group].mean
            ),
            "recent": recent_accumulators.get(
                group, _WeightedMean()
            ).mean,
            "weight": (
                authoritative[group][1]
                if group in authoritative
                else accumulators[group].weight
            ),
            "samples": (
                accumulators.get(group, _WeightedMean()).samples
            ),
            "weighting": (
                "episode"
                if (
                    group in authoritative
                    and authoritative[group][2]
                )
                or (
                    group in accumulators
                    and accumulators[group].samples > 0
                    and accumulators[group].exact_weight
                )
                else "log_record"
            ),
        }
        for group in sorted(groups)
        if (
            group in authoritative
            or accumulators[group].mean is not None
        )
    }


def _plane_value(record: Mapping[str, Any]) -> float | None:
    rate = _finite_float(_first(record, PLANE_RATE_PATHS))
    if rate is not None:
        return rate
    within = _first(record, PLANE_BOOL_PATHS)
    if isinstance(within, Sequence) and not isinstance(
        within, (str, bytes, bytearray)
    ):
        values = _finite_values(within)
        if values:
            return sum(values) / len(values)
        return None
    if isinstance(within, bool):
        return float(within)
    within_number = _finite_float(within)
    if within_number is not None:
        return within_number
    fraction = _first(record, PLANE_FRACTION_PATHS)
    if isinstance(fraction, Sequence) and not isinstance(
        fraction, (str, bytes, bytearray)
    ):
        values = _finite_values(fraction)
        if values:
            return sum(0.4 <= item <= 0.6 for item in values) / len(values)
        return None
    height_fraction = _finite_float(fraction)
    if height_fraction is not None:
        return float(0.4 <= height_fraction <= 0.6)
    return None


def _plane_summary(
    records: Sequence[Mapping[str, Any]], window: int
) -> dict[str, Any]:
    cumulative = _WeightedMean()
    recent = _WeightedMean()
    latest: float | None = None
    for index, record in enumerate(records):
        value = _plane_value(record)
        if value is None:
            continue
        weight = _finite_float(_first(record, PLANE_COUNT_PATHS))
        if weight is None:
            fractions = _first(record, PLANE_FRACTION_PATHS)
            if isinstance(fractions, Sequence) and not isinstance(
                fractions, (str, bytes, bytearray)
            ):
                valid_fractions = _finite_values(fractions)
                if valid_fractions:
                    weight = float(len(valid_fractions))
        exact = weight is not None and weight > 0.0
        sample_weight = weight if exact else 1.0
        cumulative.add(value, weight=sample_weight, exact_weight=exact)
        if index >= max(0, len(records) - window):
            recent.add(value, weight=sample_weight, exact_weight=exact)
        latest = value
    return {
        "latest": latest,
        "recent": recent.mean,
        "cumulative": cumulative.mean,
        "samples": cumulative.samples,
        "recent_samples": recent.samples,
        "weighting": (
            "sample"
            if cumulative.samples > 0 and cumulative.exact_weight
            else "log_record"
        ),
        "required_fraction_range": [0.4, 0.6],
    }


def _step_elapsed(
    record: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    return (
        _finite_float(_first(record, STEP_PATHS)),
        _finite_float(_first(record, ELAPSED_PATHS)),
    )


def _target_steps(
    run_status: Mapping[str, Any], explicit: int | None
) -> float | None:
    if explicit is not None and explicit > 0:
        return float(explicit)
    return _finite_float(
        _first(
            run_status,
            (
                "target_steps",
                "max_steps",
                "total_training_steps",
                "config.total_steps",
                "config.max_steps",
                "progress.target_steps",
                "progress.total_steps",
                "stage.target_steps",
            ),
        )
    )


def _progress_summary(
    records: Sequence[Mapping[str, Any]],
    run_status: Mapping[str, Any],
    *,
    window: int,
    total_steps: int | None,
) -> dict[str, Any]:
    points = [
        point
        for record in records
        if (point := _step_elapsed(record))[0] is not None
        and point[1] is not None
    ]
    latest_step = points[-1][0] if points else None
    latest_elapsed = points[-1][1] if points else None
    if latest_step is None:
        latest_step = _finite_float(
            _first(
                run_status,
                (
                    "current_steps",
                    "completed_steps",
                    "progress.current_steps",
                    "progress.completed_steps",
                    "progress.current",
                ),
            )
        )
    if latest_elapsed is None:
        latest_elapsed = _finite_float(
            _first(run_status, ELAPSED_PATHS)
        )

    recent_rate: float | None = None
    usable = points[-max(2, window) :] if points else []
    if len(usable) >= 2:
        step_delta = usable[-1][0] - usable[0][0]  # type: ignore[operator]
        elapsed_delta = usable[-1][1] - usable[0][1]  # type: ignore[operator]
        if step_delta >= 0.0 and elapsed_delta > 0.0:
            recent_rate = step_delta / elapsed_delta
    cumulative_rate = None
    if (
        latest_step is not None
        and latest_elapsed is not None
        and latest_elapsed > 0.0
    ):
        cumulative_rate = latest_step / latest_elapsed
    status_rate = _finite_float(
        _first(
            run_status,
            (
                "steps_per_second",
                "throughput_steps_s",
                "progress.steps_per_second",
                "progress.throughput_steps_s",
            ),
        )
    )
    if recent_rate is None:
        recent_rate = status_rate
    target = _target_steps(run_status, total_steps)
    remaining = (
        max(0.0, target - latest_step)
        if target is not None and latest_step is not None
        else None
    )
    eta = (
        remaining / recent_rate
        if remaining is not None
        and recent_rate is not None
        and recent_rate > 0.0
        else None
    )
    fraction = (
        min(1.0, max(0.0, latest_step / target))
        if latest_step is not None and target is not None and target > 0.0
        else None
    )
    return {
        "completed_steps": latest_step,
        "target_steps": target,
        "fraction": fraction,
        "elapsed_s": latest_elapsed,
        "recent_steps_per_s": recent_rate,
        "cumulative_steps_per_s": cumulative_rate,
        "eta_s": eta,
        "phase": _name(
            _first(
                records[-1] if records else run_status,
                ("phase", "stage", "progress.phase", "progress.stage"),
            )
        ),
    }


def _use_status_fallback(
    summary: dict[str, Any],
    run_status: Mapping[str, Any],
    *,
    paths: Sequence[str],
) -> None:
    if summary["latest"] is not None:
        return
    value = _finite_float(_first(run_status, paths))
    if value is not None:
        summary.update(
            latest=value,
            recent_mean=value,
            cumulative_mean=value,
            samples=1,
            recent_samples=1,
        )


def build_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    run_status: Mapping[str, Any] | None = None,
    window: int = 20,
    total_steps: int | None = None,
) -> dict[str, Any]:
    """Build one stable live-metrics payload from heterogeneous log records."""

    if window <= 0:
        raise ValueError("window must be positive")
    rows = [dict(record) for record in records]
    status = dict(run_status or {})
    success = _success_summary(rows, window)
    if success["latest"] is None:
        latest_rate = _finite_float(
            _first(status, LATEST_SUCCESS_RATE_PATHS)
        )
        cumulative_rate = _finite_float(
            _first(status, CUMULATIVE_SUCCESS_RATE_PATHS)
        )
        rate, weight, exact = _rate_weight(status)
        if latest_rate is not None:
            rate = latest_rate
        if rate is not None:
            success.update(
                latest=rate,
                recent=rate,
                cumulative=(
                    cumulative_rate
                    if cumulative_rate is not None
                    else rate
                ),
                samples=1,
                recent_samples=1,
                weighting="episode" if exact and weight > 0 else "log_record",
            )
    plane = _plane_summary(rows, window)
    if plane["latest"] is None:
        value = _plane_value(status)
        if value is not None:
            plane.update(
                latest=value,
                recent=value,
                cumulative=value,
                samples=1,
                recent_samples=1,
            )
    position = _metric_summary(
        rows, paths=POSITION_ERROR_PATHS, window=window
    )
    euler = _metric_summary(rows, paths=EULER_ERROR_PATHS, window=window)
    translation_desired_displacement = _metric_summary(
        rows,
        paths=TRANSLATION_DESIRED_DISPLACEMENT_PATHS,
        window=window,
    )
    translation_actual_displacement = _metric_summary(
        rows,
        paths=TRANSLATION_ACTUAL_DISPLACEMENT_PATHS,
        window=window,
    )
    translation_displacement_error = _metric_summary(
        rows,
        paths=TRANSLATION_DISPLACEMENT_ERROR_PATHS,
        window=window,
    )
    translation_final_euler = _metric_summary(
        rows,
        paths=TRANSLATION_FINAL_EULER_PATHS,
        window=window,
    )
    path_tracking_rmse = _metric_summary(
        rows, paths=PATH_TRACKING_RMSE_PATHS, window=window
    )
    path_tracking_max = _metric_summary(
        rows, paths=PATH_TRACKING_MAX_PATHS, window=window
    )
    diffusion_loss = _metric_summary(
        rows,
        paths=DIFFUSION_LOSS_PATHS,
        window=window,
        conditional_loss=True,
    )
    diffusion_lr = _metric_summary(
        rows, paths=DIFFUSION_LR_PATHS, window=window
    )
    _use_status_fallback(
        position, status, paths=POSITION_ERROR_PATHS
    )
    _use_status_fallback(euler, status, paths=EULER_ERROR_PATHS)
    _use_status_fallback(
        translation_desired_displacement,
        status,
        paths=TRANSLATION_DESIRED_DISPLACEMENT_PATHS,
    )
    _use_status_fallback(
        translation_actual_displacement,
        status,
        paths=TRANSLATION_ACTUAL_DISPLACEMENT_PATHS,
    )
    _use_status_fallback(
        translation_displacement_error,
        status,
        paths=TRANSLATION_DISPLACEMENT_ERROR_PATHS,
    )
    _use_status_fallback(
        translation_final_euler,
        status,
        paths=TRANSLATION_FINAL_EULER_PATHS,
    )
    _use_status_fallback(
        path_tracking_rmse, status, paths=PATH_TRACKING_RMSE_PATHS
    )
    _use_status_fallback(
        path_tracking_max, status, paths=PATH_TRACKING_MAX_PATHS
    )
    _use_status_fallback(
        diffusion_loss, status, paths=DIFFUSION_LOSS_PATHS
    )
    _use_status_fallback(diffusion_lr, status, paths=DIFFUSION_LR_PATHS)
    current = rows[-1] if rows else status

    def latest_finite(paths: Sequence[str]) -> float | None:
        for record in reversed(rows):
            value = _finite_float(_first(record, paths))
            if value is not None:
                return value
        return _finite_float(_first(status, paths))

    return {
        "record_count": len(rows),
        "window": window,
        "success": success,
        "success_by_object": _grouped_success(
            rows,
            group_paths=OBJECT_NAME_PATHS,
            mapping_paths=(
                "success_by_object",
                "endpoint_success_by_object",
                "object_success_rates",
                "metrics.success_by_object",
            ),
            cumulative_mapping_paths=(
                "cumulative_success_by_object",
                "metrics.cumulative_success_by_object",
            ),
            window=window,
        ),
        "success_by_task": _grouped_success(
            rows,
            group_paths=TASK_NAME_PATHS,
            mapping_paths=(
                "success_by_task",
                "endpoint_success_by_task",
                "task_success_rates",
                "metrics.success_by_task",
            ),
            cumulative_mapping_paths=(
                "cumulative_success_by_task",
                "metrics.cumulative_success_by_task",
            ),
            window=window,
        ),
        "success_by_rotation_target_angle_deg": _grouped_success(
            rows,
            group_paths=("rotation_angle_deg",),
            mapping_paths=(
                "success_by_rotation_target_angle_deg",
                "metrics.success_by_rotation_target_angle_deg",
            ),
            cumulative_mapping_paths=(
                "phase_cumulative_success_by_rotation_target_angle_deg",
                "metrics.phase_cumulative_success_by_rotation_target_angle_deg",
            ),
            window=window,
        ),
        "grasp_plane_compliance": plane,
        "position_error_m": position,
        "euler_l2_error_deg": euler,
        "translation_desired_final_displacement_m": (
            translation_desired_displacement
        ),
        "translation_actual_final_displacement_m": (
            translation_actual_displacement
        ),
        "translation_final_displacement_error_m": (
            translation_displacement_error
        ),
        "translation_final_euler_error_deg": translation_final_euler,
        "translation_path_tracking_rmse_m": path_tracking_rmse,
        "translation_path_tracking_max_error_m": path_tracking_max,
        "diffusion_loss": diffusion_loss,
        "diffusion_lr": diffusion_lr,
        "progress": _progress_summary(
            rows,
            status,
            window=window,
            total_steps=total_steps,
        ),
        "training": {
            "stage": _name(_first(current, ("stage", "phase"))),
            "certified_trajectory_count": latest_finite(
                ("certified_trajectory_count", "certified_demo_count")
            ),
            "valid_group_rate": latest_finite(("valid_group_rate",)),
            "cumulative_valid_group_rate": latest_finite(
                ("cumulative_valid_group_rate",)
            ),
            "valid_group_count": latest_finite(("valid_group_count",)),
            "valid_group_episode_count": latest_finite(
                ("valid_group_episode_count",)
            ),
            "epoch": _finite_float(_first(current, ("epoch",))),
            "diffusion_epochs": _finite_float(
                _first(current, ("diffusion_epochs",))
            ),
            "batch_in_epoch": _finite_float(
                _first(current, ("batch_in_epoch",))
            ),
            "batches_per_epoch": _finite_float(
                _first(current, ("batches_per_epoch",))
            ),
            "diffusion_updates": _finite_float(
                _first(current, ("diffusion_updates",))
            ),
            "diffusion_gradient_norm": _finite_float(
                _first(current, ("diffusion_gradient_norm",))
            ),
            "diffusion_eta_s": _finite_float(
                _first(current, ("diffusion_eta_s",))
            ),
            "reward_mean": latest_finite(("reward_mean",)),
            "commanded_grasp_plane_compliance_rate": latest_finite(
                ("commanded_grasp_plane_compliance_rate",)
            ),
            "gpu_memory_allocated_gb": _finite_float(
                _first(current, ("gpu_memory_allocated_gb",))
            ),
            "gpu_memory_reserved_gb": _finite_float(
                _first(current, ("gpu_memory_reserved_gb",))
            ),
        },
    }
