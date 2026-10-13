from __future__ import annotations

import unittest

from capacity_probe import _capacity_summary


class CapacitySummaryTest(unittest.TestCase):
    def test_summary_selects_only_qualified_measured_world_counts(self):
        total = 48 * 1024**3
        results = [
            {
                "worlds": 1024,
                "status": "pass",
                "preflight": {
                    "cuda_peak_device_used_bytes": 20 * 1024**3,
                    "cuda_total_memory_bytes": total,
                    "cuda_peak_device_used_fraction": 20 / 48,
                },
            },
            {
                "worlds": 2048,
                "status": "pass",
                "preflight": {
                    "cuda_peak_device_used_bytes": 36 * 1024**3,
                    "cuda_total_memory_bytes": total,
                    "cuda_peak_device_used_fraction": 36 / 48,
                },
            },
            {
                "worlds": 4096,
                "status": "fail",
                "preflight": None,
            },
        ]
        summary = _capacity_summary(
            results, [1024, 2048, 4096, 8192]
        )
        self.assertEqual(summary["qualified_worlds"], [1024, 2048])
        self.assertEqual(summary["maximum_passed_worlds"], 2048)
        self.assertEqual(summary["first_failed_worlds"], 4096)
        self.assertFalse(summary["all_requested_passed"])
        self.assertEqual(
            summary["recommendations"][
                "largest_tested_at_or_below_70_percent"
            ],
            1024,
        )
        self.assertEqual(
            summary["recommendations"][
                "largest_tested_at_or_below_80_percent"
            ],
            2048,
        )
        self.assertEqual(
            summary["recommendations"][
                "largest_tested_at_or_below_90_percent"
            ],
            2048,
        )


if __name__ == "__main__":
    unittest.main()
