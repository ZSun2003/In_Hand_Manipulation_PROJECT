"""Strict, shared validation for device-wide CUDA memory certificates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


GIB = 1024**3
POST_RELEASE_DEVICE_TOLERANCE_FRACTION = 0.01
POST_RELEASE_DEVICE_TOLERANCE_MIN_BYTES = 256 * 1024**2
POST_RELEASE_TORCH_ALLOCATED_TOLERANCE_BYTES = 64 * 1024**2
POST_RELEASE_TORCH_RESERVED_TOLERANCE_BYTES = 256 * 1024**2


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _same_number(actual: Any, expected: float) -> bool:
    return _finite_number(actual) and math.isclose(
        float(actual), float(expected), rel_tol=1.0e-12, abs_tol=1.0e-12
    )


def memory_sample_is_valid(
    value: Any,
    *,
    total_bytes: int,
    expected_label: str | None = None,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    label = value.get("label")
    if not isinstance(label, str) or not label:
        return False
    if expected_label is not None and label != expected_label:
        return False
    fields = (
        "free_bytes",
        "device_used_bytes",
        "torch_allocated_bytes",
        "torch_reserved_bytes",
    )
    if not all(_nonnegative_int(value.get(name)) for name in fields):
        return False
    free = int(value["free_bytes"])
    used = int(value["device_used_bytes"])
    allocated = int(value["torch_allocated_bytes"])
    reserved = int(value["torch_reserved_bytes"])
    return (
        0 <= free <= total_bytes
        and used == total_bytes - free
        and allocated <= reserved <= total_bytes
    )


def memory_segment_is_valid(value: Any, *, total_bytes: int) -> bool:
    if not isinstance(value, Mapping):
        return False
    free = value.get("minimum_free_bytes")
    used = value.get("actual_peak_device_used_bytes")
    if not _nonnegative_int(free) or not _nonnegative_int(used):
        return False
    free = int(free)
    used = int(used)
    return (
        0 <= free <= total_bytes
        and used == total_bytes - free
        and _same_number(value.get("actual_peak_device_used_gb"), used / GIB)
        and _same_number(
            value.get("actual_peak_device_memory_fraction"),
            used / total_bytes,
        )
    )


def stage_has_actual_memory_certificate(
    stage: Any,
    *,
    expected_stage: str,
    expected_total_bytes: int,
) -> bool:
    """Validate all redundant stage/detail fields and device capacity binding."""

    if (
        not isinstance(stage, Mapping)
        or stage.get("passed") is not True
        or not _nonnegative_int(expected_total_bytes)
        or expected_total_bytes <= 0
    ):
        return False
    details = stage.get("actual_device_memory")
    if not isinstance(details, Mapping):
        return False
    if not (
        details.get("measurement")
        == "background torch.cuda.mem_get_info device-wide polling"
        and details.get("includes_non_torch_cuda_allocations") is True
        and details.get("stage") == expected_stage
        and details.get("total_bytes") == expected_total_bytes
        and details.get("valid") is True
        and details.get("validation_error") is None
        and details.get("poll_thread_errors") == []
        and _same_number(details.get("limit_fraction"), 0.90)
        and _finite_number(details.get("poll_interval_s"))
        and float(details["poll_interval_s"]) > 0.0
    ):
        return False
    baseline_free = details.get("baseline_free_bytes")
    minimum_free = details.get("minimum_free_bytes")
    peak_used = details.get("actual_peak_device_used_bytes")
    stage_increment = details.get("stage_peak_increment_bytes")
    if not all(
        _nonnegative_int(value)
        for value in (baseline_free, minimum_free, peak_used, stage_increment)
    ):
        return False
    baseline_free = int(baseline_free)
    minimum_free = int(minimum_free)
    peak_used = int(peak_used)
    if not (
        0 <= minimum_free <= baseline_free <= expected_total_bytes
        and peak_used == expected_total_bytes - minimum_free
        and int(stage_increment) == max(0, baseline_free - minimum_free)
        and _same_number(details.get("actual_peak_device_used_gb"), peak_used / GIB)
        and _same_number(
            details.get("actual_peak_device_memory_fraction"),
            peak_used / expected_total_bytes,
        )
        and _same_number(
            details.get("stage_peak_increment_gb"),
            int(stage_increment) / GIB,
        )
        and peak_used / expected_total_bytes <= 0.90
        and stage.get("actual_peak_device_used_bytes") == peak_used
        and _same_number(stage.get("actual_peak_device_used_gb"), peak_used / GIB)
        and _same_number(
            stage.get("actual_peak_device_memory_fraction"),
            peak_used / expected_total_bytes,
        )
    ):
        return False

    background_count = details.get("background_sample_count")
    checkpoint_count = details.get("checkpoint_sample_count")
    sample_count = details.get("sample_count")
    checkpoints = details.get("checkpoint_samples")
    if not (
        _nonnegative_int(background_count)
        and int(background_count) > 0
        and _nonnegative_int(checkpoint_count)
        and _nonnegative_int(sample_count)
        and isinstance(checkpoints, Sequence)
        and not isinstance(checkpoints, (str, bytes))
        and int(checkpoint_count) == len(checkpoints)
        and int(sample_count) == int(background_count) + len(checkpoints)
        and len(checkpoints) > 0
        and memory_sample_is_valid(
            checkpoints[0],
            total_bytes=expected_total_bytes,
            expected_label="baseline",
        )
        and int(checkpoints[0]["free_bytes"]) == baseline_free
        and all(
            memory_sample_is_valid(
                sample, total_bytes=expected_total_bytes
            )
            for sample in checkpoints
        )
        and minimum_free
        <= min(int(sample["free_bytes"]) for sample in checkpoints)
    ):
        return False

    segments = details.get("segments")
    if not isinstance(segments, Mapping) or not segments:
        return False
    return all(
        isinstance(name, str)
        and bool(name)
        and memory_segment_is_valid(segment, total_bytes=expected_total_bytes)
        and int(segment["minimum_free_bytes"]) >= minimum_free
        for name, segment in segments.items()
    )


def post_release_certificate_is_valid(
    value: Any,
    *,
    label: str,
    expected_total_bytes: int,
    checkpoint_samples: Any,
) -> bool:
    """Validate a release certificate against the tracker's real checkpoints."""

    if not isinstance(value, Mapping) or not isinstance(
        checkpoint_samples, Sequence
    ):
        return False
    baseline = value.get("baseline")
    current = value.get("post_release")
    if not (
        value.get("passed") is True
        and value.get("label") == label
        and value.get("measurement")
        == "actual device free plus current Torch allocation"
        and value.get("total_device_bytes") == expected_total_bytes
        and memory_sample_is_valid(
            baseline,
            total_bytes=expected_total_bytes,
            expected_label=f"pre_allocation:{label}",
        )
        and memory_sample_is_valid(
            current,
            total_bytes=expected_total_bytes,
            expected_label=f"post_release:{label}",
        )
        and any(dict(sample) == dict(baseline) for sample in checkpoint_samples)
        and any(dict(sample) == dict(current) for sample in checkpoint_samples)
    ):
        return False
    device_unrecovered = max(
        0, int(baseline["free_bytes"]) - int(current["free_bytes"])
    )
    allocated_unrecovered = max(
        0,
        int(current["torch_allocated_bytes"])
        - int(baseline["torch_allocated_bytes"]),
    )
    reserved_unrecovered = max(
        0,
        int(current["torch_reserved_bytes"])
        - int(baseline["torch_reserved_bytes"]),
    )
    device_tolerance = max(
        POST_RELEASE_DEVICE_TOLERANCE_MIN_BYTES,
        int(expected_total_bytes * POST_RELEASE_DEVICE_TOLERANCE_FRACTION),
    )
    return (
        value.get("device_unrecovered_bytes") == device_unrecovered
        and value.get("device_tolerance_bytes") == device_tolerance
        and device_unrecovered <= device_tolerance
        and value.get("torch_allocated_unrecovered_bytes")
        == allocated_unrecovered
        and value.get("torch_allocated_tolerance_bytes")
        == POST_RELEASE_TORCH_ALLOCATED_TOLERANCE_BYTES
        and allocated_unrecovered
        <= POST_RELEASE_TORCH_ALLOCATED_TOLERANCE_BYTES
        and value.get("torch_reserved_unrecovered_bytes")
        == reserved_unrecovered
        and value.get("torch_reserved_tolerance_bytes")
        == POST_RELEASE_TORCH_RESERVED_TOLERANCE_BYTES
        and reserved_unrecovered
        <= POST_RELEASE_TORCH_RESERVED_TOLERANCE_BYTES
    )


__all__ = [
    "GIB",
    "POST_RELEASE_DEVICE_TOLERANCE_FRACTION",
    "POST_RELEASE_DEVICE_TOLERANCE_MIN_BYTES",
    "POST_RELEASE_TORCH_ALLOCATED_TOLERANCE_BYTES",
    "POST_RELEASE_TORCH_RESERVED_TOLERANCE_BYTES",
    "memory_sample_is_valid",
    "memory_segment_is_valid",
    "post_release_certificate_is_valid",
    "stage_has_actual_memory_certificate",
]
