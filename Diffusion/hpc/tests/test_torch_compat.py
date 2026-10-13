from __future__ import annotations

import unittest

from s_hpc.torch_compat import unique_cuda_index_writes


class _FakeTorch:
    def __init__(self, *, enabled: bool, warn_only: bool) -> None:
        self.enabled = enabled
        self.warn_only = warn_only
        self.calls: list[tuple[bool, bool]] = []

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.enabled

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.warn_only

    def use_deterministic_algorithms(
        self,
        enabled: bool,
        *,
        warn_only: bool = False,
    ) -> None:
        self.enabled = enabled
        self.warn_only = warn_only
        self.calls.append((enabled, warn_only))


class TorchCompatibilityTest(unittest.TestCase):
    def test_cuda_scope_disables_and_restores_warn_only_mode(self):
        torch = _FakeTorch(enabled=True, warn_only=True)
        with unique_cuda_index_writes(torch, device_type="cuda"):
            self.assertFalse(torch.enabled)
        self.assertTrue(torch.enabled)
        self.assertTrue(torch.warn_only)
        self.assertEqual(torch.calls, [(False, False), (True, True)])

    def test_scope_restores_after_write_failure(self):
        torch = _FakeTorch(enabled=True, warn_only=False)
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            with unique_cuda_index_writes(torch, device_type="cuda"):
                raise RuntimeError("write failed")
        self.assertTrue(torch.enabled)
        self.assertFalse(torch.warn_only)

    def test_non_cuda_or_nondeterministic_mode_is_unchanged(self):
        for device_type, enabled in (("cpu", True), ("cuda", False)):
            with self.subTest(device_type=device_type, enabled=enabled):
                torch = _FakeTorch(enabled=enabled, warn_only=True)
                with unique_cuda_index_writes(
                    torch, device_type=device_type
                ):
                    pass
                self.assertEqual(torch.calls, [])
