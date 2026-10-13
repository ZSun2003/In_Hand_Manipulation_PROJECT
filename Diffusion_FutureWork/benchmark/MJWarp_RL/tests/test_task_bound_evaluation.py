from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest

from MJWarp_RL import contract
from MJWarp_RL.evaluate import (
    SCHEMA as EVALUATION_SCHEMA,
    TRAINING_SCHEMA,
    _cases,
    _checkpoint_trained_task,
    _config_from_mapping,
    _translation_path_case_coverage,
    _validate_checkpoint,
    _validate_task_binding,
)
from MJWarp_RL.hpc_config import DEFAULT_CONFIG
from MJWarp_RL.env import OBSERVATION_DIM
from MJWarp_RL.runtime_fingerprint import runtime_code_fingerprint_sha256
from MJWarp_RL.sharded_env import ordered_shard_manifest


class TaskBoundEvaluationTests(unittest.TestCase):

    def test_training_schema_is_certified_expert_v5(self) -> None:
        self.assertEqual(
            TRAINING_SCHEMA,
            "native-mjwarp-sharded-certified-ik-diffusion-training-v5",
        )

    def test_config_roundtrip_ignores_forged_derived_task_index(self) -> None:
        raw = replace(DEFAULT_CONFIG, task=contract.TASK_ROTATE).as_dict()
        raw["task_index"] = 0
        restored = _config_from_mapping(raw)
        self.assertEqual(restored.task, contract.TASK_ROTATE)
        self.assertEqual(restored.task_index, 1)

    def test_cases_can_be_task_specific_without_changing_case_identity(self) -> None:
        paired = _cases("all", episodes_per_case=3, seed=123)
        rotation = _cases(
            "all",
            episodes_per_case=3,
            seed=123,
            task=contract.TASK_ROTATE,
        )
        expected = [
            case for case in paired if case["task"] == contract.TASK_ROTATE
        ]
        self.assertEqual(rotation, expected)
        self.assertEqual(len(rotation), len(contract.OBJECTS) * 3)
        self.assertEqual({case["task_index"] for case in rotation}, {1})
        coverage = _translation_path_case_coverage(rotation)
        self.assertIs(coverage["active"], False)
        self.assertIsNone(coverage["all_shapes_covered"])

    def test_translation_only_coverage_remains_formally_active(self) -> None:
        cases = _cases(
            "all",
            episodes_per_case=contract.RESET_TEMPLATES_PER_OBJECT,
            seed=456,
            task=contract.TASK_TRANSLATE,
        )
        coverage = _translation_path_case_coverage(cases)
        self.assertIs(coverage["active"], True)
        self.assertIs(coverage["all_shapes_covered"], True)

    def test_checkpoint_task_inference_rejects_requested_or_index_drift(self) -> None:
        config = replace(DEFAULT_CONFIG, task=contract.TASK_ROTATE)
        metadata = {
            "trained_task": contract.TASK_ROTATE,
            "task": contract.TASK_ROTATE,
            "task_index": 1,
            "root_height_m": 123.0,
            "training_config": config.as_dict(),
        }
        checkpoint = {"metadata": metadata}
        self.assertEqual(
            _checkpoint_trained_task(checkpoint, requested_task=None),
            contract.TASK_ROTATE,
        )
        with self.assertRaisesRegex(ValueError, "requested evaluation task"):
            _checkpoint_trained_task(
                checkpoint, requested_task=contract.TASK_TRANSLATE
            )
        drifted = copy.deepcopy(checkpoint)
        drifted["metadata"]["task_index"] = 0
        with self.assertRaisesRegex(ValueError, "task_index"):
            _checkpoint_trained_task(drifted, requested_task=None)

    def test_manifest_checkpoint_and_platform_binding_is_exact(self) -> None:
        config = replace(DEFAULT_CONFIG, task=contract.TASK_ROTATE)
        platform = SimpleNamespace(
            platform="gripper-test",
            task=contract.TASK_ROTATE,
            task_index=1,
            root_height_m=0.525,
        )
        record = {
            "platform": "gripper-test",
            "trained_task": contract.TASK_ROTATE,
            "task": contract.TASK_ROTATE,
            "task_index": 1,
            "root_height_m": 0.525,
        }
        _validate_task_binding(
            record,
            label="training manifest",
            config=config,
            platform_spec=platform,
        )
        for field, value in (
            ("trained_task", contract.TASK_TRANSLATE),
            ("task_index", 0),
            ("root_height_m", 0.526),
        ):
            drifted = dict(record)
            drifted[field] = value
            with self.assertRaisesRegex(ValueError, field):
                _validate_task_binding(
                    drifted,
                    label="training manifest",
                    config=config,
                    platform_spec=platform,
                )

    def test_checkpoint_model_shards_bind_task_and_root(self) -> None:
        config = replace(DEFAULT_CONFIG, task=contract.TASK_TRANSLATE)
        root_height_m = 0.525
        platform = SimpleNamespace(
            platform="gripper-test",
            task=config.task,
            task_index=config.task_index,
            root_height_m=root_height_m,
            platform_adapter_fingerprint_sha256="f" * 64,
            shards=tuple(
                SimpleNamespace(
                    fixed_object_index=index,
                    task=config.task,
                    task_index=config.task_index,
                    root_height_m=root_height_m,
                    model_xml=f"/models/{object_kind}.xml",
                    model_xml_sha256=(format(index + 1, "x") * 64),
                    host_model=SimpleNamespace(nq=10, nv=9, nu=8),
                )
                for index, object_kind in enumerate(contract.OBJECTS)
            ),
        )
        settings = config.diffusion
        metadata = {
            "algorithm": contract.ALGORITHM,
            "algorithm_signature": contract.ALGORITHM_SIGNATURE,
            "runtime_code_fingerprint_sha256": (
                runtime_code_fingerprint_sha256()
            ),
            "platform": platform.platform,
            "trained_task": config.task,
            "task": config.task,
            "task_index": config.task_index,
            "root_height_m": float(platform.root_height_m),
            "platform_adapter_fingerprint_sha256": (
                platform.platform_adapter_fingerprint_sha256
            ),
            "physics_layout": contract.PHYSICS_LAYOUT,
            "translation_path_contract": (
                contract.translation_path_contract_payload()
            ),
            "rotation_angle_distribution_deg": [
                contract.ROTATION_ANGLE_MIN_DEG,
                contract.ROTATION_ANGLE_MAX_DEG,
            ],
            "global_worlds": config.num_envs,
            "object_world_counts": list(config.object_world_counts),
            "remainder_worlds": config.remainder_worlds,
            "remainder_object_world_counts": list(
                config.remainder_object_world_counts
            ),
            "model_shards": ordered_shard_manifest(
                platform, config.object_world_counts
            ),
            "total_mjwarp_expert_steps": contract.TOTAL_TRAINING_STEPS,
            "epoch": settings.epochs,
            "epochs": settings.epochs,
            "updates": 1,
        }
        policy = SimpleNamespace(
            model_config=lambda: {
                "obs_dim": OBSERVATION_DIM,
                "action_dim": contract.ACTION_DIM,
                "obs_horizon": settings.obs_horizon,
                "horizon": settings.action_horizon,
                "num_diffusion_steps": settings.diffusion_steps,
                "hidden_dim": settings.hidden_dim,
                "depth": settings.depth,
                "dropout": settings.dropout,
            }
        )
        checkpoint = {"metadata": metadata, "ema_state": {}}
        _validate_checkpoint(
            checkpoint,
            policy=policy,
            config=config,
            platform_spec=platform,
        )
        drifted = copy.deepcopy(checkpoint)
        drifted["metadata"]["model_shards"][0]["task"] = (
            contract.TASK_ROTATE
        )
        with self.assertRaisesRegex(ValueError, "model shard"):
            _validate_checkpoint(
                drifted,
                policy=policy,
                config=config,
                platform_spec=platform,
            )


if __name__ == "__main__":
    unittest.main()
