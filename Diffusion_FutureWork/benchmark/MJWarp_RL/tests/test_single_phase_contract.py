from __future__ import annotations

import unittest

from MJWarp_RL import contract
from MJWarp_RL.capacity_scan import (
    ROTATION_DIAGNOSTIC_MIN_WORLDS,
    _validate_diagnostic_config,
    _worker_config,
)
from MJWarp_RL.curriculum import (
    PHASE,
    PHASES,
    REGION_ALL,
    rotation_angle_bounds_deg,
    sample_rotation,
    sample_translation,
    validate_phase,
)
from MJWarp_RL.hpc_config import DEFAULT_CONFIG


class SinglePhaseContractTests(unittest.TestCase):
    def test_formal_budget_is_one_complete_60m_run(self) -> None:
        self.assertEqual(PHASES, ("all",))
        self.assertEqual(PHASE, "all")
        self.assertEqual(contract.TOTAL_TRAINING_STEPS, 60_000_000)
        self.assertEqual(DEFAULT_CONFIG.num_envs, 20_000)
        self.assertEqual(DEFAULT_CONFIG.episode_length, 3_000)
        self.assertEqual(DEFAULT_CONFIG.control_ticks, 3_000)
        self.assertEqual(DEFAULT_CONFIG.remainder_worlds, 0)
        self.assertEqual(DEFAULT_CONFIG.maximum_demo_episodes, 2_000)
        self.assertNotIn("phase1_steps", DEFAULT_CONFIG.as_dict())
        self.assertNotIn("phase2_steps", DEFAULT_CONFIG.as_dict())

    def test_full_task_distribution_covers_all_paths_and_angles(self) -> None:
        self.assertEqual(rotation_angle_bounds_deg(PHASE), (10, 45))
        self.assertEqual(sample_rotation(PHASE).source_region, REGION_ALL)
        self.assertEqual(sample_translation(PHASE).source_region, REGION_ALL)
        with self.assertRaises(ValueError):
            validate_phase("phase1")
        with self.assertRaises(ValueError):
            validate_phase("phase2")

    def test_grasp_plane_contract_is_forty_to_sixty_percent(self) -> None:
        self.assertEqual(contract.GRASP_PLANE_MIN_FRACTION, 0.40)
        self.assertEqual(contract.GRASP_PLANE_NOMINAL_FRACTION, 0.50)
        self.assertEqual(contract.GRASP_PLANE_MAX_FRACTION, 0.60)
        self.assertEqual(
            contract.decode_action(contract.ACTION_LOW, "banana")
            .tip_height_fraction,
            (0.40,) * 4,
        )
        self.assertEqual(
            contract.decode_action(contract.ACTION_HIGH, "banana")
            .tip_height_fraction,
            (0.60,) * 4,
        )

    def test_rotation_diagnostic_rejects_incomplete_angle_coverage(self) -> None:
        self.assertEqual(ROTATION_DIAGNOSTIC_MIN_WORLDS, 1_440)
        with self.assertRaisesRegex(ValueError, "at least 1440 worlds"):
            _worker_config(
                num_envs=1_430,
                device="cpu",
                task=contract.TASK_ROTATE,
                diagnostic_only=True,
            )
        config = _worker_config(
            num_envs=1_440,
            device="cpu",
            task=contract.TASK_ROTATE,
            diagnostic_only=True,
        )
        self.assertEqual(config.object_world_counts, (360,) * 4)
        self.assertEqual(config.object_group_counts, (36,) * 4)
        self.assertEqual(config.control_ticks, config.episode_length)
        _validate_diagnostic_config(config)


if __name__ == "__main__":
    unittest.main()
