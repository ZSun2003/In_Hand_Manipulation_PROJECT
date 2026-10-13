from __future__ import annotations

import ast
from pathlib import Path
import unittest

from IK_RL import config
from IK_RL.runtime import import_current_controller, validate_parent_contract


class ConfigAndTrainContractTests(unittest.TestCase):
    def test_parent_model_controller_planner_and_fruit_manifest_are_shared(
        self,
    ) -> None:
        current = import_current_controller()
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

    def test_formal_task_and_tolerances(self) -> None:
        self.assertEqual(len(config.OBJECTS), 6)
        self.assertNotIn("plate", config.OBJECTS)
        self.assertEqual(
            config.TASKS,
            ("translate_xy45_100mm", "rotate_z_45deg"),
        )
        self.assertEqual(config.TOTAL_TIMESTEPS, 2_000_000)
        self.assertEqual(config.POSITION_TOLERANCE_M, 0.005)
        self.assertEqual(config.rotation_tolerance_deg("cube"), 5.0)
        self.assertEqual(config.rotation_tolerance_deg("cylinder"), 5.0)
        self.assertEqual(config.rotation_tolerance_deg("cone"), 5.0)
        for fruit in ("banana", "pear", "strawberry"):
            self.assertEqual(config.rotation_tolerance_deg(fruit), 10.0)

    def test_residual_contract(self) -> None:
        self.assertEqual(config.ACTUATOR_DIM, 12)
        self.assertEqual(config.ACTION_DIM, 14)
        self.assertEqual(config.RESIDUAL_MAX_RAD, 0.005)
        self.assertEqual(
            config.RESIDUAL_RATE_LIMIT_RAD_PER_STEP,
            0.0001,
        )
        self.assertEqual(config.RESIDUAL_TARGET_BLEND, 0.35)

    def test_training_has_one_learn_and_no_training_gate(self) -> None:
        source_path = Path(config.PACKAGE_DIR) / "train.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        learn_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "learn"
        ]
        self.assertEqual(len(learn_calls), 1)
        keywords = {item.arg for item in learn_calls[0].keywords}
        self.assertNotIn("callback", keywords)
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("save_replay_buffer", called_attributes)
        self.assertNotIn("load_replay_buffer", called_attributes)


if __name__ == "__main__":
    unittest.main()
