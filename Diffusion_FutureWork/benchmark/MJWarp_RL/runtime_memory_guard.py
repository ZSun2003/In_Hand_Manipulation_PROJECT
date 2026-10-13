"""Fail-closed device-wide CUDA memory guard for formal training.

The guard intentionally uses ``cudaMemGetInfo`` through ``torch.cuda`` rather
than Torch allocator statistics.  The resulting limit includes allocations
owned by Warp/MJWarp, CUDA graphs, and any other CUDA allocator in the process.
"""

from __future__ import annotations

import json
import math
import threading
from copy import deepcopy
from typing import Any


_GIB = 1024**3


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class TrainingDeviceMemoryGuard:
    """Continuously enforce a device-wide CUDA memory ceiling.

    A synchronized checkpoint is used for every admission decision and explicit
    safety check.  A daemon thread polls between checkpoints so allocations
    outside Torch's caching allocator are still visible.  Once any violation or
    sampling error is observed, the guard remains invalid for its lifetime.
    """

    def __init__(
        self,
        torch: Any,
        device: Any,
        limit_fraction: float = 0.90,
        poll_interval_s: float = 0.02,
    ) -> None:
        if (
            isinstance(limit_fraction, bool)
            or not isinstance(limit_fraction, (int, float))
            or not math.isfinite(float(limit_fraction))
            or not 0.0 < float(limit_fraction) <= 1.0
        ):
            raise ValueError("limit_fraction must be finite and in (0, 1]")
        if (
            isinstance(poll_interval_s, bool)
            or not isinstance(poll_interval_s, (int, float))
            or not math.isfinite(float(poll_interval_s))
            or float(poll_interval_s) <= 0.0
        ):
            raise ValueError("poll_interval_s must be finite and positive")

        self.torch = torch
        self.device = device
        self.limit_fraction = float(limit_fraction)
        self.poll_interval_s = float(poll_interval_s)
        self._state_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._report_lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._report_cache: dict[str, Any] | None = None
        self._background_sample_count = 0
        self._checkpoint_sample_count = 0
        self._checkpoint_labels: list[str] = []
        self._poll_errors: list[str] = []
        self._checkpoint_errors: list[str] = []
        self._total_change: dict[str, int] | None = None
        self._limit_violation: dict[str, Any] | None = None
        self._admission_rejection_count = 0

        try:
            self.torch.cuda.set_device(self.device)
            self.torch.cuda.synchronize(self.device)
            free_bytes, total_bytes = self._query_memory(synchronize=False)
        except BaseException as exc:
            raise RuntimeError(
                "failed to initialize the CUDA training memory guard: "
                + _format_exception(exc)
            ) from exc

        self.total_bytes = int(total_bytes)
        self.limit_bytes = int(self.limit_fraction * self.total_bytes)
        self.baseline_free_bytes = int(free_bytes)
        self.baseline_used_bytes = self.total_bytes - self.baseline_free_bytes
        self.minimum_free_bytes = self.baseline_free_bytes
        self._record_observation(
            free_bytes=self.baseline_free_bytes,
            total_bytes=self.total_bytes,
            label="baseline",
            background=False,
        )

        self._thread = threading.Thread(
            target=self._poll_loop,
            name="training-device-memory-guard",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _validated_memory_info(value: Any) -> tuple[int, int]:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise RuntimeError("cuda mem_get_info returned an invalid result")
        free_bytes, total_bytes = value
        if (
            not _is_nonnegative_int(free_bytes)
            or not _is_nonnegative_int(total_bytes)
            or int(total_bytes) <= 0
            or int(free_bytes) > int(total_bytes)
        ):
            raise RuntimeError(
                "cuda mem_get_info returned invalid free/total byte counts"
            )
        return int(free_bytes), int(total_bytes)

    def _query_memory(self, *, synchronize: bool) -> tuple[int, int]:
        with self._io_lock:
            if synchronize:
                self.torch.cuda.synchronize(self.device)
            value = self.torch.cuda.mem_get_info(self.device)
        return self._validated_memory_info(value)

    def _record_observation(
        self,
        *,
        free_bytes: int,
        total_bytes: int,
        label: str,
        background: bool,
    ) -> None:
        with self._state_lock:
            if background:
                self._background_sample_count += 1
            else:
                self._checkpoint_sample_count += 1
                self._checkpoint_labels.append(str(label))

            if int(total_bytes) != self.total_bytes:
                if self._total_change is None:
                    self._total_change = {
                        "expected_total_bytes": self.total_bytes,
                        "observed_total_bytes": int(total_bytes),
                    }
                self._stop.set()
                return

            free = int(free_bytes)
            self.minimum_free_bytes = min(self.minimum_free_bytes, free)
            used = self.total_bytes - free
            if used > self.limit_bytes:
                if self._limit_violation is None:
                    self._limit_violation = {
                        "label": str(label),
                        "device_used_bytes": used,
                        "device_memory_fraction": used / self.total_bytes,
                    }
                self._stop.set()

    def _poll_loop(self) -> None:
        try:
            self.torch.cuda.set_device(self.device)
            while not self._stop.is_set():
                free_bytes, total_bytes = self._query_memory(
                    synchronize=False
                )
                self._record_observation(
                    free_bytes=free_bytes,
                    total_bytes=total_bytes,
                    label="background",
                    background=True,
                )
                if self._stop.wait(self.poll_interval_s):
                    break
        except BaseException as exc:
            with self._state_lock:
                self._poll_errors.append(_format_exception(exc))
            self._stop.set()

    def _ensure_open(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("CUDA training memory guard is closed")

    def _synchronous_sample(self, label: str) -> dict[str, Any]:
        if not isinstance(label, str) or not label:
            raise ValueError("memory checkpoint label must be a non-empty string")
        # Serialize checkpoints against close() so a CUDA query can never run
        # after another thread has completed shutdown.
        with self._close_lock:
            self._ensure_open()
            try:
                free_bytes, total_bytes = self._query_memory(synchronize=True)
            except BaseException as exc:
                message = _format_exception(exc)
                with self._state_lock:
                    self._checkpoint_errors.append(f"{label}: {message}")
                self._stop.set()
                raise RuntimeError(
                    f"CUDA memory checkpoint {label!r} failed: {message}"
                ) from exc
            self._record_observation(
                free_bytes=free_bytes,
                total_bytes=total_bytes,
                label=label,
                background=False,
            )
        used = int(total_bytes) - int(free_bytes)
        return {
            "label": label,
            "free_bytes": int(free_bytes),
            "device_used_bytes": used,
            "device_memory_fraction": used / int(total_bytes),
            "total_bytes": int(total_bytes),
        }

    def _failure_messages(self) -> list[str]:
        with self._state_lock:
            errors = list(self._poll_errors)
            errors.extend(self._checkpoint_errors)
            if self._total_change is not None:
                errors.append(
                    "CUDA device total memory changed from "
                    f"{self._total_change['expected_total_bytes']} to "
                    f"{self._total_change['observed_total_bytes']} bytes"
                )
            if self._limit_violation is not None:
                errors.append(
                    "CUDA device memory reached "
                    f"{self._limit_violation['device_memory_fraction']:.2%}, "
                    f"exceeding the {self.limit_fraction:.2%} limit"
                )
            return errors

    def admit(self, certified_increment_bytes: int) -> dict[str, Any]:
        """Admit an allocation only when its certified peak fits the limit."""

        if isinstance(certified_increment_bytes, bool) or not isinstance(
            certified_increment_bytes, int
        ):
            raise TypeError(
                "certified_increment_bytes must be a non-boolean "
                "non-negative integer"
            )
        if certified_increment_bytes < 0:
            raise ValueError("certified_increment_bytes must be non-negative")
        sample = self._synchronous_sample("admit")
        failures = self._failure_messages()
        if failures:
            raise RuntimeError(
                "CUDA memory guard is already invalid: " + "; ".join(failures)
            )
        increment = int(certified_increment_bytes)
        projected = int(sample["device_used_bytes"]) + increment
        passed = projected <= self.limit_bytes
        certificate = {
            "passed": passed,
            "measurement": "synchronized torch.cuda.mem_get_info admission",
            "device": str(self.device),
            "total_bytes": self.total_bytes,
            "limit_fraction": self.limit_fraction,
            "limit_bytes": self.limit_bytes,
            "current_free_bytes": int(sample["free_bytes"]),
            "current_device_used_bytes": int(sample["device_used_bytes"]),
            "certified_increment_bytes": increment,
            "projected_device_used_bytes": projected,
            "projected_device_memory_fraction": projected / self.total_bytes,
            "available_limit_headroom_bytes": max(
                0, self.limit_bytes - int(sample["device_used_bytes"])
            ),
        }
        # Exercise JSON serialization here so callers never receive a
        # certificate containing device-specific scalar objects.
        json.dumps(certificate, allow_nan=False)
        if not passed:
            with self._state_lock:
                self._admission_rejection_count += 1
            raise RuntimeError(
                "certified CUDA allocation would exceed the training memory "
                "limit: "
                + json.dumps(certificate, sort_keys=True, allow_nan=False)
            )
        return certificate

    def raise_if_violated(self, label: str) -> None:
        """Synchronize and raise if any current or historical fault exists."""

        self._synchronous_sample(label)
        failures = self._failure_messages()
        if failures:
            raise RuntimeError(
                f"CUDA memory guard failed at {label!r}: "
                + "; ".join(failures)
            )

    def close(self) -> None:
        """Stop the polling thread.  Repeated calls are safe."""

        with self._close_lock:
            with self._state_lock:
                if self._closed:
                    return
            self._stop.set()
            self._thread.join(timeout=max(1.0, self.poll_interval_s * 10.0))
            if self._thread.is_alive():
                with self._state_lock:
                    self._poll_errors.append(
                        "RuntimeError: CUDA memory polling thread did not stop"
                    )
            with self._state_lock:
                self._closed = True

    def report(self) -> dict[str, Any]:
        """Stop polling and return a JSON-serializable lifetime report."""

        with self._report_lock:
            with self._state_lock:
                cached = self._report_cache
                closed = self._closed
            if cached is not None:
                return deepcopy(cached)
            if not closed:
                try:
                    self._synchronous_sample("report")
                except RuntimeError:
                    # The checkpoint error is retained and makes the report
                    # invalid without preventing diagnostic serialization.
                    pass
            self.close()
            with self._state_lock:
                minimum_free = int(self.minimum_free_bytes)
                background_count = int(self._background_sample_count)
                checkpoint_count = int(self._checkpoint_sample_count)
                checkpoint_labels = list(self._checkpoint_labels)
                poll_errors = list(self._poll_errors)
                checkpoint_errors = list(self._checkpoint_errors)
                total_change = (
                    dict(self._total_change)
                    if self._total_change is not None
                    else None
                )
                limit_violation = (
                    dict(self._limit_violation)
                    if self._limit_violation is not None
                    else None
                )
                admission_rejections = int(self._admission_rejection_count)
            peak_used = self.total_bytes - minimum_free
            errors = self._failure_messages()
            result = {
                "measurement": (
                    "background and synchronized torch.cuda.mem_get_info "
                    "device-wide polling"
                ),
                "includes_non_torch_cuda_allocations": True,
                "device": str(self.device),
                "total_bytes": self.total_bytes,
                "baseline_free_bytes": self.baseline_free_bytes,
                "baseline_device_used_bytes": self.baseline_used_bytes,
                "minimum_free_bytes": minimum_free,
                "peak_device_used_bytes": peak_used,
                "peak_device_used_gb": peak_used / _GIB,
                "peak_device_memory_fraction": peak_used / self.total_bytes,
                "limit_fraction": self.limit_fraction,
                "limit_bytes": self.limit_bytes,
                "poll_interval_s": self.poll_interval_s,
                "background_sample_count": background_count,
                "checkpoint_sample_count": checkpoint_count,
                "sample_count": background_count + checkpoint_count,
                "checkpoint_labels": checkpoint_labels,
                "poll_errors": poll_errors,
                "checkpoint_errors": checkpoint_errors,
                "limit_violated": limit_violation is not None,
                "first_limit_violation": limit_violation,
                "total_memory_changed": total_change is not None,
                "total_memory_change": total_change,
                "admission_rejection_count": admission_rejections,
                "closed": True,
                "errors": errors,
                "valid": not errors,
            }
            json.dumps(result, allow_nan=False)
            with self._state_lock:
                self._report_cache = deepcopy(result)
            return deepcopy(result)

    def __enter__(self) -> "TrainingDeviceMemoryGuard":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = ["TrainingDeviceMemoryGuard"]
