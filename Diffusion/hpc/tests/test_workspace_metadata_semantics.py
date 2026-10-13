from __future__ import annotations
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.workspace_policy_rollouts import (FULL_DISC_TRAINING_REGION, HALF_DISC_TRAINING_REGION, training_distribution_semantics)

class WorkspaceMetadataSemanticsTest(unittest.TestCase):
    def test_semantics_follow_selected_config_boolean(self) -> None:
            full = training_distribution_semantics(
                {
                    "curriculum": {
                        "final_spawn_radius_m": 0.1,
                        "restrict_screw_spawn_to_opening_half_disk": False,
                    }
                }
            )
            self.assertEqual(full["training_region"], FULL_DISC_TRAINING_REGION)
            self.assertFalse(full["opposite_half_disc_is_ood"])
            half = training_distribution_semantics(
                {
                    "curriculum": {
                        "final_spawn_radius_m": 0.1,
                        "restrict_screw_spawn_to_opening_half_disk": True,
                    }
                }
            )
            self.assertEqual(half["training_region"], HALF_DISC_TRAINING_REGION)
            self.assertTrue(half["opposite_half_disc_is_ood"])
