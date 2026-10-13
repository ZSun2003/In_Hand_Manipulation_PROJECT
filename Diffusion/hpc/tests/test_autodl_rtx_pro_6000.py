from __future__ import annotations

import unittest

from autodl_rtx_pro_6000_preflight import (
    is_target_gpu_name,
    normalize_gpu_name,
    numeric_version,
    version_at_least,
)


class RTXPro6000HardwarePolicyTest(unittest.TestCase):
    def test_blackwell_names_are_distinct_from_older_rtx_6000_cards(self):
        for name in (
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_target_gpu_name(name))
        for name in (
            "Quadro RTX 6000",
            "NVIDIA RTX 6000 Ada Generation",
            "NVIDIA RTX PRO 6000",
            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
            "NVIDIA H100 80GB HBM3",
        ):
            with self.subTest(name=name):
                self.assertFalse(is_target_gpu_name(name))

    def test_normalization_and_driver_comparison(self):
        self.assertEqual(
            normalize_gpu_name("NVIDIA RTX PRO 6000 Blackwell"),
            "NVIDIARTXPRO6000BLACKWELL",
        )
        self.assertEqual(numeric_version("595.71.05"), (595, 71, 5))
        self.assertTrue(version_at_least("595.71.05", "575.51.03"))
        self.assertTrue(version_at_least("580.82.09", "575.51.03"))
        self.assertTrue(version_at_least("575.51.3", "575.51.03"))
        self.assertFalse(version_at_least("570.26", "575.51.03"))


if __name__ == "__main__":
    unittest.main()
