"""Regression tests for binding reports to the G6 platform adapter bytes."""

from __future__ import annotations

import hashlib
import re
import unittest

from MJWarp_RL import contract
from MJWarp_RL import platform_adapter as adapter


class PlatformAdapterFingerprintTests(unittest.TestCase):
    def test_current_adapter_sha256_and_task_bundle_schema(self) -> None:
        expected = hashlib.sha256(
            adapter.PLATFORM_ADAPTER_PATH.read_bytes()
        ).hexdigest()
        self.assertRegex(expected, re.compile(r"\A[0-9a-f]{64}\Z"))
        self.assertEqual(adapter._sha256(adapter.PLATFORM_ADAPTER_PATH), expected)
        self.assertEqual(tuple(adapter.MODEL_XMLS_BY_TASK), contract.TASKS)
        self.assertEqual(tuple(adapter.TASK_GRIPPER_XMLS), contract.TASKS)
        self.assertEqual(tuple(adapter.TASK_ROOT_HEIGHT_M), contract.TASKS)
        for task in contract.TASKS:
            self.assertEqual(
                tuple(adapter.model_xmls_for_task(task)), contract.OBJECTS
            )
            self.assertTrue(adapter.TASK_GRIPPER_XMLS[task].is_file())
            self.assertTrue(
                all(
                    path.is_file()
                    for path in adapter.model_xmls_for_task(task).values()
                )
            )
        expected_heights = {
            "gripper3": {
                contract.TASK_TRANSLATE: 0.560,
                contract.TASK_ROTATE: 0.560,
            },
            "gripper6": {
                contract.TASK_TRANSLATE: 0.560,
                contract.TASK_ROTATE: 0.560,
            },
        }
        self.assertEqual(adapter.TASK_ROOT_HEIGHT_M, expected_heights[adapter.PLATFORM])
        for spec_type in (adapter.PlatformSpec, adapter.ShardedPlatformSpec):
            fields = spec_type.__dataclass_fields__
            for field in (
                "task",
                "task_index",
                "root_height_m",
                "platform_adapter_fingerprint_sha256",
            ):
                self.assertIn(field, fields)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
