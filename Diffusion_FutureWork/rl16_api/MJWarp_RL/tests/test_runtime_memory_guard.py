from __future__ import annotations

import json
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Callable

from MJWarp_RL.runtime_memory_guard import TrainingDeviceMemoryGuard


class _FakeCuda:
    def __init__(
        self,
        *,
        main_info: tuple[int, int] = (500, 1_000),
        background_info: tuple[int, int] | None = None,
        background_error: BaseException | None = None,
    ) -> None:
        self.main_info = main_info
        self.background_info = background_info or main_info
        self.background_error = background_error
        self.background_sampled = threading.Event()
        self.set_device_calls = 0
        self.synchronize_calls = 0
        self.mem_get_info_calls = 0
        self._lock = threading.Lock()

    @staticmethod
    def _is_background() -> bool:
        return threading.current_thread().name == (
            "training-device-memory-guard"
        )

    def set_device(self, device: object) -> None:
        del device
        with self._lock:
            self.set_device_calls += 1

    def synchronize(self, device: object) -> None:
        del device
        with self._lock:
            self.synchronize_calls += 1

    def mem_get_info(self, device: object) -> tuple[int, int]:
        del device
        background = self._is_background()
        with self._lock:
            self.mem_get_info_calls += 1
            response = self.background_info if background else self.main_info
            error = self.background_error if background else None
        if background:
            self.background_sampled.set()
        if error is not None:
            raise error
        return response


def _fake_torch(cuda: _FakeCuda) -> SimpleNamespace:
    return SimpleNamespace(cuda=cuda)


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return bool(predicate())


class TrainingDeviceMemoryGuardTests(unittest.TestCase):
    def test_normal_poll_admission_checkpoint_and_report(self) -> None:
        cuda = _FakeCuda(main_info=(500, 1_000))
        guard = TrainingDeviceMemoryGuard(
            _fake_torch(cuda), "cuda:0", poll_interval_s=0.001
        )
        self.assertTrue(cuda.background_sampled.wait(1.0))
        certificate = guard.admit(300)
        self.assertTrue(certificate["passed"])
        self.assertEqual(certificate["projected_device_used_bytes"], 800)
        json.dumps(certificate, allow_nan=False)
        guard.raise_if_violated("safe-checkpoint")
        report = guard.report()
        self.assertTrue(report["valid"])
        self.assertEqual(report["baseline_free_bytes"], 500)
        self.assertEqual(report["minimum_free_bytes"], 500)
        self.assertEqual(report["peak_device_used_bytes"], 500)
        self.assertGreater(report["background_sample_count"], 0)
        self.assertIn("baseline", report["checkpoint_labels"])
        self.assertIn("admit", report["checkpoint_labels"])
        self.assertIn("safe-checkpoint", report["checkpoint_labels"])
        self.assertIn("report", report["checkpoint_labels"])
        self.assertEqual(report["sample_count"], (
            report["background_sample_count"]
            + report["checkpoint_sample_count"]
        ))
        json.dumps(report, allow_nan=False)

    def test_admission_rejects_projected_limit_and_invalid_sizes(self) -> None:
        cuda = _FakeCuda(main_info=(250, 1_000))
        guard = TrainingDeviceMemoryGuard(
            _fake_torch(cuda), "cuda:0", poll_interval_s=0.001
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "would exceed"):
                guard.admit(151)
            with self.assertRaises(TypeError):
                guard.admit(True)
            with self.assertRaises(ValueError):
                guard.admit(-1)
            report = guard.report()
            self.assertTrue(report["valid"])
            self.assertEqual(report["admission_rejection_count"], 1)
        finally:
            guard.close()

    def test_background_limit_violation_is_sticky(self) -> None:
        cuda = _FakeCuda(
            main_info=(200, 1_000),
            background_info=(50, 1_000),
        )
        guard = TrainingDeviceMemoryGuard(
            _fake_torch(cuda), "cuda:0", poll_interval_s=0.001
        )
        self.assertTrue(cuda.background_sampled.wait(1.0))
        self.assertTrue(
            _wait_until(lambda: bool(guard._failure_messages()))
        )
        with self.assertRaisesRegex(RuntimeError, "exceeding"):
            guard.raise_if_violated("main-is-safe")
        report = guard.report()
        self.assertFalse(report["valid"])
        self.assertTrue(report["limit_violated"])
        self.assertEqual(
            report["first_limit_violation"]["label"], "background"
        )
        self.assertEqual(report["minimum_free_bytes"], 50)

    def test_synchronous_limit_violation_is_detected(self) -> None:
        cuda = _FakeCuda(
            main_info=(200, 1_000),
            background_info=(200, 1_000),
        )
        guard = TrainingDeviceMemoryGuard(
            _fake_torch(cuda), "cuda:0", poll_interval_s=1.0
        )
        self.assertTrue(cuda.background_sampled.wait(1.0))
        cuda.main_info = (50, 1_000)
        with self.assertRaisesRegex(RuntimeError, "exceeding"):
            guard.raise_if_violated("synchronous-over-limit")
        report = guard.report()
        self.assertFalse(report["valid"])
        self.assertEqual(
            report["first_limit_violation"]["label"],
            "synchronous-over-limit",
        )

    def test_background_total_change_is_fail_closed(self) -> None:
        cuda = _FakeCuda(
            main_info=(200, 1_000),
            background_info=(200, 2_000),
        )
        guard = TrainingDeviceMemoryGuard(
            _fake_torch(cuda), "cuda:0", poll_interval_s=0.001
        )
        self.assertTrue(cuda.background_sampled.wait(1.0))
        self.assertTrue(
            _wait_until(lambda: bool(guard._failure_messages()))
        )
        with self.assertRaisesRegex(RuntimeError, "total memory changed"):
            guard.raise_if_violated("after-total-change")
        report = guard.report()
        self.assertFalse(report["valid"])
        self.assertTrue(report["total_memory_changed"])
        self.assertEqual(
            report["total_memory_change"],
            {
                "expected_total_bytes": 1_000,
                "observed_total_bytes": 2_000,
            },
        )

    def test_background_poll_error_and_close_are_fail_closed_idempotent(self) -> None:
        cuda = _FakeCuda(
            main_info=(200, 1_000),
            background_error=RuntimeError("poll boom"),
        )
        guard = TrainingDeviceMemoryGuard(
            _fake_torch(cuda), "cuda:0", poll_interval_s=0.001
        )
        self.assertTrue(cuda.background_sampled.wait(1.0))
        self.assertTrue(
            _wait_until(lambda: bool(guard._failure_messages()))
        )
        with self.assertRaisesRegex(RuntimeError, "poll boom"):
            guard.raise_if_violated("after-poll-error")
        guard.close()
        guard.close()
        self.assertFalse(guard._thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            guard.admit(0)
        report = guard.report()
        self.assertFalse(report["valid"])
        self.assertTrue(any("poll boom" in item for item in report["poll_errors"]))
        guard.close()


if __name__ == "__main__":
    unittest.main()
