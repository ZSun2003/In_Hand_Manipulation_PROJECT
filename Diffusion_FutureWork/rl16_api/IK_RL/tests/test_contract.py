from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from IK_RL import config
from IK_RL.controller_bridge import ResidualControlLaw, current
from IK_RL.runtime import validate_parent_contract


IK_RL_DIR = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_parent_model_controller_planner_and_fruit_manifest_are_shared(
        self,
    ) -> None:
        contract = validate_parent_contract(current)
        self.assertEqual(
            Path(current.MODEL_XML).resolve(),
            config.PARENT_MODEL_XML.resolve(),
        )
        self.assertEqual(
            Path(current.__file__).resolve(),
            config.PARENT_CONTROLLER.resolve(),
        )
        self.assertEqual(
            Path(contract["planner"]),
            config.PARENT_PLANNER.resolve(),
        )
        self.assertEqual(
            Path(contract["fruit_manifest"]),
            config.PARENT_FRUIT_MANIFEST.resolve(),
        )
        self.assertEqual(len(contract["fruit_manifest_sha256"]), 64)
        self.assertEqual(len(contract["contract_sha256"]), 64)

        copied_geometry = [
            path
            for pattern in (
                "*.xml",
                "*.obj",
                "*.ply",
                "*.stl",
                "*.mtl",
                "texture_map.png",
            )
            for path in config.PACKAGE_DIR.rglob(pattern)
        ]
        self.assertEqual(copied_geometry, [])

    def test_training_objects_and_plate_exclusion(self) -> None:
        self.assertEqual(
            config.OBJECTS,
            (
                "cube",
                "cylinder",
                "banana",
                "pear",
                "strawberry",
                "cone",
            ),
        )
        self.assertNotIn(config.PLATE_OBJECT, config.OBJECTS)
        self.assertEqual(len(config.OBJECTS), 6)

    def test_uniform_task_support_contract(self) -> None:
        self.assertEqual(
            config.TASKS,
            (
                "translate_xy45_100mm",
                "rotate_z_45deg",
            ),
        )

    def test_endpoint_thresholds_require_both_pose_errors(self) -> None:
        self.assertEqual(config.POSITION_TOLERANCE_M, 0.005)
        for object_kind in ("cube", "cylinder", "cone"):
            self.assertEqual(
                config.rotation_tolerance_deg(object_kind), 5.0
            )
        for object_kind in ("banana", "pear", "strawberry"):
            self.assertEqual(
                config.rotation_tolerance_deg(object_kind), 10.0
            )
        self.assertTrue(config.endpoint_success("cube", 0.005, 5.0))
        self.assertFalse(config.endpoint_success("cube", 0.005001, 5.0))
        self.assertFalse(config.endpoint_success("cube", 0.005, 5.001))
        self.assertTrue(config.endpoint_success("pear", 0.005, 10.0))
        self.assertFalse(config.endpoint_success("pear", 0.005, 10.001))

    def test_residual_limits_and_indices(self) -> None:
        self.assertEqual(config.ACTION_DIM, 16)
        self.assertEqual(config.LINEAR_ACTUATOR_INDICES, (0, 3, 6, 9))
        self.assertEqual(config.RESIDUAL_MAX_RAD, 0.005)
        self.assertEqual(config.RESIDUAL_MAX_LINEAR_M, 0.0005)
        self.assertEqual(
            config.RESIDUAL_RATE_LIMIT_RAD_PER_STEP, 0.0001
        )
        self.assertEqual(
            config.RESIDUAL_RATE_LIMIT_LINEAR_M_PER_STEP, 0.00002
        )
        self.assertEqual(config.RESIDUAL_TARGET_BLEND, 0.35)

    def test_first_residual_step_obeys_per_actuator_rate_limit(self) -> None:
        law = ResidualControlLaw(
            np.full(16, -1.0),
            np.full(16, 1.0),
        )
        nominal = np.zeros(16)
        applied, residual = law.apply(
            nominal, np.ones(16), enabled=True
        )
        expected = np.full(16, 0.0001)
        expected[list(config.LINEAR_ACTUATOR_INDICES)] = 0.00002
        np.testing.assert_array_equal(residual, expected)
        np.testing.assert_array_equal(applied, expected)

    def test_zero_residual_exactly_matches_a_nominal_trace(self) -> None:
        rng = np.random.default_rng(182)
        trace = rng.uniform(-0.3, 0.3, size=(64, 16))
        low = np.full(16, -0.5)
        high = np.full(16, 0.5)
        law = ResidualControlLaw(low, high)
        applied_trace = []
        for nominal in trace:
            applied, residual = law.apply(
                nominal,
                np.zeros(16),
                enabled=True,
            )
            applied_trace.append(applied)
            np.testing.assert_array_equal(residual, np.zeros(16))
        np.testing.assert_array_equal(np.asarray(applied_trace), trace)

    def test_banana_current_contract_preserves_independent_tip_z(self) -> None:
        targets = {
            "U": np.asarray((0.0, 0.04, 0.024)),
            "R": np.asarray((0.03, 0.0, 0.021)),
            "D": np.asarray((0.0, -0.04, 0.016)),
            "L": np.asarray((-0.03, 0.0, 0.013)),
        }
        projected = current._formation_targets(
            current.ObjectKind.BANANA,
            targets,
        )
        for finger in current.FINGERS:
            self.assertEqual(projected[finger][2], targets[finger][2])
        self.assertTrue(
            current._tip_formation_within_tolerance(
                current.ObjectKind.BANANA,
                10.0 * current.TIP_COPLANARITY_TOLERANCE_M,
                0.0,
            )
        )
        self.assertFalse(
            current._tip_formation_within_tolerance(
                current.ObjectKind.BANANA,
                0.0,
                2.0 * current.TIP_PARALLELOGRAM_TOLERANCE_M,
            )
        )

    def test_nonfinite_action_is_rejected(self) -> None:
        law = ResidualControlLaw(
            np.full(16, -1.0),
            np.full(16, 1.0),
        )
        action = np.zeros(16)
        action[5] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            law.apply(np.zeros(16), action, enabled=True)

    def test_training_is_one_learn_call_with_two_million_default(self) -> None:
        self.assertEqual(config.TOTAL_TIMESTEPS, 2_000_000)
        source = (IK_RL_DIR / "train.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "learn"
        ]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("CheckpointCallback", source)
        self.assertNotIn("EvalCallback", source)
        self.assertNotIn("StopTraining", source)
        self.assertNotIn("tensorboard", source.lower())


if __name__ == "__main__":
    unittest.main()
