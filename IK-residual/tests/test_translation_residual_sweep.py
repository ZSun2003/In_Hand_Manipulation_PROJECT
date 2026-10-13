"""Lightweight contracts for deterministic Phase-2 residual evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from g3g6 import config
from g3g6 import ik_control
from g3g6 import rotation_residual_training as training
from g3g6 import translation_residual_sweep as sweep


class TranslationResidualSweepTests(unittest.TestCase):
    @staticmethod
    def _artifact_run(root: Path, gripper: str = "g3") -> Path:
        run = root / gripper
        run.mkdir(parents=True)
        model = run / "final_model.zip"
        normalizer = run / "vec_normalize.pkl"
        model.write_bytes(b"translation-model-contract")
        normalizer.write_bytes(b"translation-normalizer-contract")
        manifest = training.formal_config(gripper, device="cpu")
        manifest.update(
            {
                "status": "complete",
                "artifacts": {
                    "model": {
                        "path": model.name,
                        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    },
                    "vec_normalize": {
                        "path": normalizer.name,
                        "sha256": hashlib.sha256(
                            normalizer.read_bytes()
                        ).hexdigest(),
                    },
                },
            }
        )
        (run / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return run

    def test_artifact_loader_requires_exact_phase_two_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as temporary:
            run = self._artifact_run(Path(temporary), "g6")
            artifacts = sweep.load_training_artifacts(run)
            self.assertEqual(artifacts.gripper, "g6")
            self.assertEqual(artifacts.action_dim, 16)
            self.assertTrue(artifacts.residual_after_preaction_only)

            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["phase_tasks"]["2"]["target_grasp_plane_z_m"] = 0.325
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Phase-2 task contract"):
                sweep.load_training_artifacts(run)

    def test_matching_ik_manifest_preserves_specs_and_seed(self) -> None:
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as temporary:
            run = Path(temporary) / "pure"
            run.mkdir()
            specs = ik_control.build_experiment_specs(
                grippers=("g3",),
                objects=("cube",),
                paths=("S",),
                axes=("x", "yz"),
                suite="full",
                rotation_angle_deg=30.0,
                seed=431,
            )
            (run / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": ik_control.RUN_SCHEMA,
                        "experiments": [
                            {
                                "episode_id": spec.episode_id,
                                "gripper": spec.gripper,
                                "object": spec.object_name,
                                "path": spec.path_name,
                                "task": spec.task,
                                "rotation_axes": spec.rotation_axes,
                                "rotation_goal_euler_deg": list(
                                    spec.rotation_goal_euler_deg
                                ),
                                "seed": spec.seed,
                            }
                            for spec in specs
                        ],
                    }
                ),
                encoding="utf-8",
            )
            loaded = sweep.load_matching_ik_specs((run,))
            self.assertEqual(loaded, specs)
            self.assertEqual([spec.seed for spec in loaded], [431, 432, 433])
            self.assertEqual(
                [spec.rotation_axes for spec in loaded], [None, "x", "yz"]
            )

    def test_residual_trace_is_strict_superset_of_ik_trace(self) -> None:
        spec = ik_control.ExperimentSpec(
            gripper="g3",
            object_name="cube",
            path_name="circle",
            rotation_axes=None,
            rotation_goal_euler_deg=(0.0, 0.0, 0.0),
            seed=7,
        )
        env = SimpleNamespace(
            nominal_ctrl=np.zeros(12, dtype=np.float64),
            applied_ctrl=np.zeros(12, dtype=np.float64),
        )
        info = {
            "phase": int(config.Phase.TRANSLATE),
            "task": "translate",
            "object": "cube",
            "path": "circle",
            "rotation_axes": None,
            "rotation_goal_euler_deg": np.zeros(3),
            "control_mode": "residual_action",
            "residual_nominal_feedback": True,
            "residual_after_preaction_only": True,
            "residual_active": False,
            "task_target_grasp_plane_z_m": config.TARGET_GRASP_PLANE_Z_M,
            "residual_requested_action": np.zeros(12),
            "residual_executed_action": np.zeros(12),
            "actuator_residual": np.zeros(12),
            "step": 1,
            "object_grasp_frame_position_m": np.array((0.0, 0.0, 0.3)),
            "target_position_m": np.array((0.0, 0.0, 0.3)),
            "ik_command_position_m": np.array((0.0, 0.0, 0.3)),
            "object_quaternion_wxyz": np.array((1.0, 0.0, 0.0, 0.0)),
            "target_quaternion_wxyz": np.array((1.0, 0.0, 0.0, 0.0)),
            "ik_command_quaternion_wxyz": np.array((1.0, 0.0, 0.0, 0.0)),
            "action_reference_quaternion_wxyz": np.array(
                (1.0, 0.0, 0.0, 0.0)
            ),
            "position_error_m_per_axis": np.zeros(3),
            "angle_error_deg_per_axis": np.zeros(3),
            "tip_contact_count": 0,
            "ik_exact": True,
            "ik_projection_fraction": 1.0,
            "max_abs_actuator_residual": 0.0,
            "is_success": False,
            "termination_reason": "running",
        }
        artifacts = SimpleNamespace(
            action_dim=12,
            model_sha256="m" * 64,
            vec_normalize_sha256="v" * 64,
        )
        ik_row = ik_control._record_tick(spec, env, info, terminated=False)
        residual_row = sweep._record_tick(
            spec,
            env,
            artifacts,
            info,
            np.zeros(12),
            terminated=False,
        )
        self.assertTrue(set(ik_row).issubset(residual_row))
        for key, value in ik_row.items():
            if key != "schema":
                self.assertEqual(residual_row[key], value)
        self.assertEqual(residual_row["schema"], sweep.TRACE_SCHEMA)
        self.assertEqual(residual_row["episode_id"], spec.episode_id)
        self.assertEqual(residual_row["source_ik_episode_id"], spec.episode_id)
        self.assertEqual(residual_row["residual_active"], 0)
        self.assertEqual(residual_row["actuator_residual_15"], 0.0)

    def test_suite_writes_complete_manifest_and_closes_controller(self) -> None:
        spec = ik_control.ExperimentSpec(
            gripper="g3",
            object_name="cube",
            path_name="circle",
            rotation_axes=None,
            rotation_goal_euler_deg=(0.0, 0.0, 0.0),
            seed=7,
        )
        artifact = SimpleNamespace(
            gripper="g3",
            run_dir=config.PROJECT_ROOT / "fake_training",
            manifest={"schema": training.TRAINING_SCHEMA},
            action_dim=12,
            observation_dim=149,
            residual_nominal_feedback=True,
            residual_after_preaction_only=True,
            model_sha256="a" * 64,
            vec_normalize_sha256="b" * 64,
        )

        class Controller:
            def __init__(self) -> None:
                self.normalizer = SimpleNamespace(training=False, norm_reward=False)
                self.closed = False

            def close(self) -> None:
                self.closed = True

        controller = Controller()
        summary = {
            "episode_id": spec.episode_id,
            "source_ik_episode_id": spec.episode_id,
            "gripper": "g3",
            "object": "cube",
            "path": "circle",
            "task": "translation",
            "rotation_axes": None,
            "rotation_goal_euler_deg": [0.0, 0.0, 0.0],
            "seed": 7,
            "trace": f"traces/{spec.episode_id}.csv",
        }
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as temporary:
            destination = Path(temporary) / "result"
            with patch.object(
                sweep,
                "run_residual_translation_episode",
                return_value=summary,
            ):
                manifest = sweep.run_residual_translation_suite(
                    specs=(spec,),
                    training_runs={"g3": config.PROJECT_ROOT / "fake_training"},
                    output_dir=destination,
                    device="cpu",
                    artifact_loader=lambda _path: artifact,
                    controller_loader=lambda _artifacts, *, device: controller,
                )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["schema"], sweep.RUN_SCHEMA)
            self.assertEqual(manifest["trace_schema"], sweep.TRACE_SCHEMA)
            self.assertEqual(manifest["completed_experiments"], 1)
            self.assertTrue(manifest["deterministic_policy"])
            self.assertTrue(manifest["normalization_frozen"])
            self.assertTrue(controller.closed)
            on_disk = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(on_disk["status"], "complete")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
