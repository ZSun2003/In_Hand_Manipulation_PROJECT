from __future__ import annotations

import json
import math
import unittest

import numpy as np

from IK_RL.eval import endpoint_success, json_safe


class EvaluationSchemaTests(unittest.TestCase):
    def test_nonfinite_values_become_strict_json_null(self) -> None:
        value = {
            "plain": float("nan"),
            "numpy": np.asarray((1.0, np.inf)),
            "nested": [float("-inf"), {"ok": 2.0}],
        }
        safe = json_safe(value)
        rendered = json.dumps(safe, allow_nan=False)
        decoded = json.loads(rendered)
        self.assertIsNone(decoded["plain"])
        self.assertIsNone(decoded["numpy"][1])
        self.assertIsNone(decoded["nested"][0])
        self.assertTrue(math.isfinite(decoded["nested"][1]["ok"]))

    def test_eval_reclassifies_rigid_and_fruit_thresholds(self) -> None:
        rigid_endpoint = {
            "position_error_m": 0.005,
            "euler_l2_error_deg": 5.0,
        }
        fruit_endpoint = {
            "position_error_m": 0.005,
            "euler_l2_error_deg": 10.0,
        }
        self.assertTrue(
            endpoint_success(
                rigid_endpoint,
                object_kind="cone",
                target_ready=True,
            )
        )
        self.assertFalse(
            endpoint_success(
                fruit_endpoint,
                object_kind="cube",
                target_ready=True,
            )
        )
        self.assertTrue(
            endpoint_success(
                fruit_endpoint,
                object_kind="banana",
                target_ready=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
