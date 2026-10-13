import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from s_hpc.config import load_config
from s_hpc.contracts import ACTION_DIM, OBS_DIM
from s_hpc.demos import DemoStore
from validate_success_dataset import validate_ready


class W18000SelfTrainingChainTests(unittest.TestCase):
    def test_formal_config_is_isolated_full_disc_w18000(self):
        config = load_config(
            ROOT
            / "configs"
            / "autodl_rtx_pro_6000_r100_success_selfdistill_w18000_tqc16.json"
        )
        self.assertEqual(config.physics.worlds, 18_000)
        self.assertEqual(config.physics.max_episode_steps, 4_000)
        self.assertAlmostEqual(config.curriculum.final_spawn_radius_m, 0.1)
        self.assertFalse(
            config.curriculum.restrict_screw_spawn_to_opening_half_disk
        )
        self.assertEqual(config.tqc.total_transitions, 5_000_000)
        self.assertEqual(config.tqc.replay_insert_per_vector_step, 256)

        old = load_config(
            ROOT
            / "configs"
            / "autodl_rtx_pro_6000_w14000_diffusion_ik_residual.json"
        )
        self.assertEqual(old.physics.worlds, 14_000)
        self.assertTrue(
            old.curriculum.restrict_screw_spawn_to_opening_half_disk
        )

    def test_success_dataset_ready_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "successful_policy_rollouts"
            policy_start = 20
            horizon = 16
            steps = 40
            provenance = {
                "dataset_role": "successful_policy_rollout_self_distillation",
                "expert_demonstrations": False,
                "controller": "Diffusion--Safety--IK",
                "policy_start_step": policy_start,
                "action_horizon": horizon,
                "fixed_start_grid": {
                    "angular_bins": 360,
                    "radial_bins": 50,
                    "trials_per_cell": 2,
                    "spawn_radius_mm": 100.0,
                    "socket_exclusion_semantics": (
                        "exact initial CPU-MuJoCo screw-vs-socket physical contact"
                    ),
                },
            }
            store = DemoStore(
                root,
                episodes=1,
                steps=steps,
                create=True,
                provenance=provenance,
                variable_episode_lengths=True,
            )
            store.write(
                0,
                np.zeros((1, steps + 1, OBS_DIM), dtype=np.float16),
                np.zeros((1, steps, ACTION_DIM), dtype=np.float16),
                np.zeros((1, 2), dtype=np.float32),
                socket_mouth_xy=np.zeros((1, 2), dtype=np.float32),
                socket_angle_deg=np.full((1,), 45.0, dtype=np.float32),
                valid_steps=np.full((1,), steps, dtype=np.int32),
                commit=False,
            )
            store.commit(0, 1)
            result = validate_ready(root)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["episodes"], 1)
            self.assertEqual(
                result["training_windows"],
                steps - policy_start - horizon + 1,
            )

    def test_success_dataset_rejects_140_mm_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "format": DemoStore.FORMAT,
                "episodes": 1,
                "completed": 1,
                "steps": 40,
                "provenance": {
                    "dataset_role": (
                        "successful_policy_rollout_self_distillation"
                    ),
                    "expert_demonstrations": False,
                    "controller": "Diffusion--Safety--IK",
                    "policy_start_step": 20,
                    "action_horizon": 16,
                    "fixed_start_grid": {
                        "angular_bins": 360,
                        "radial_bins": 50,
                        "trials_per_cell": 2,
                        "spawn_radius_mm": 140.0,
                    },
                },
            }
            # The radius check happens before array validation.
            from s_hpc.contracts import SCHEMA_HASH, TASK_GEOMETRY_HASH

            metadata.update(
                {
                    "schema_hash": SCHEMA_HASH,
                    "task_geometry_hash": TASK_GEOMETRY_HASH,
                    "variable_episode_lengths": True,
                    "valid_steps_file": "valid_steps.npy",
                }
            )
            (root / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "spawn_radius_mm"):
                validate_ready(root)

    def test_launcher_pins_worlds_radius_and_memory_gate(self):
        launcher = (
            ROOT.parent
            / "run_autodl_r100_success_selftrain_w18000_tqc16.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--worlds 18000", launcher)
        self.assertIn("SPAWN_RADIUS_MM=100", launcher)
        self.assertIn("MAX_LIVE_WORLDS=7000", launcher)
        self.assertIn('MEMORY_CEILING="${SCREW_AUTODL_MEMORY_CEILING:-0.90}"', launcher)
        self.assertNotIn("fallback_worlds", launcher.lower())

    def test_explicit_dataset_override_never_launches_a_scanner(self):
        launcher = (
            ROOT.parent
            / "run_autodl_r100_success_selftrain_w18000_tqc16.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("DATASET_IS_EXTERNAL_OVERRIDE=1", launcher)
        external_branch = launcher.index("if ((DATASET_IS_EXTERNAL_OVERRIDE)); then")
        local_branch = launcher.index("if local_scan_is_alive; then")
        self.assertLess(external_branch, local_branch)
        self.assertIn(
            "external dataset requested; waiting without launching a scanner",
            launcher,
        )


if __name__ == "__main__":
    unittest.main()
