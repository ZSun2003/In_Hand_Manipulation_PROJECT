"""Static and mocked tests for the full three-phase residual trainer.

No test in this module constructs MuJoCo, starts subprocess workers, or calls a
real Stable-Baselines3 ``learn`` implementation.
"""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from g3g6 import config
from g3g6 import rotation_residual_training as training


class RotationResidualTrainingContractTests(unittest.TestCase):
    def test_formal_contract_matches_requested_g3_and_g6_settings(self) -> None:
        training.validate_training_contract()
        g3 = training.formal_config("g3", device="cpu")
        g6 = training.formal_config("g6", device="mps")

        for payload, gripper, action_dim, sde_frequency in (
            (g3, "g3", 12, 64),
            (g6, "g6", 16, 64),
        ):
            with self.subTest(gripper=gripper):
                self.assertEqual(payload["schema"], training.TRAINING_SCHEMA)
                self.assertEqual(payload["algorithm"], "TQC")
                self.assertEqual(payload["phase"], int(config.Phase.ROTATE))
                self.assertEqual(payload["total_timesteps"], 4_500_000)
                self.assertEqual(payload["expected_executed_timesteps"], 4_500_072)
                self.assertEqual(
                    payload["phase_budgets"],
                    {"1": 500_000, "2": 2_000_000, "3": 2_000_000},
                )
                self.assertEqual(payload["learn_calls"], 3)
                self.assertEqual(payload["n_envs"], 72)
                self.assertEqual(
                    payload["runtime_requirements"],
                    {
                        "requested_nofile_soft_limit": 65_536,
                        "close_previous_phase_workers_before_spawn": True,
                    },
                )
                self.assertEqual(payload["seed"], 7)
                self.assertEqual(payload["observation_dimension"], 149)
                self.assertEqual(
                    payload["task"]["target_grasp_plane_z_m"], 0.325
                )
                self.assertEqual(
                    payload["task"]["sampling_profile"], "weighted_0_90"
                )
                self.assertEqual(
                    [
                        item["probability"]
                        for item in payload["task"][
                            "component_angle_integer_deg"
                        ]["bins"]
                    ],
                    [0.2, 0.5, 0.3],
                )
                self.assertEqual(payload["residual"]["action_dimension"], action_dim)
                self.assertEqual(
                    payload["residual"]["activation_by_phase"],
                    {
                        "1": "all_ticks",
                        "2": "after_preaction_tick_360",
                        "3": "after_preaction_tick_360",
                    },
                )
                self.assertTrue(payload["residual"]["nominal_ik_pose_feedback"])
                self.assertEqual(
                    payload["optimizer"]["sde_sample_freq"], sde_frequency
                )
                self.assertEqual(payload["policy"]["net_arch"], [256, 256])
                self.assertEqual(payload["policy"]["n_critics"], 4)
                self.assertEqual(payload["policy"]["n_quantiles"], 32)
                self.assertEqual(payload["optimizer"]["gradient_steps"], 8)
                self.assertEqual(payload["optimizer"]["train_freq"], 1)
                self.assertFalse(payload["optimizer"]["progress_bar"])

        self.assertEqual(g3["residual"]["linear"]["count"], 0)
        self.assertEqual(g3["residual"]["reward_penalty_weight"], 0.01)
        self.assertEqual(g6["residual"]["linear"]["count"], 4)
        self.assertEqual(g6["residual"]["reward_penalty_weight"], 0.002)
        self.assertEqual(g6["residual"]["linear"]["limit"], 0.0005)
        self.assertEqual(
            g6["residual"]["linear"]["rate_per_control_tick"], 0.00002
        )

    def test_environment_factory_binds_phase_specific_residual_options(self) -> None:
        sentinels = [mock.Mock(), mock.Mock(), mock.Mock()]
        with mock.patch.object(
            training, "G3G6ManipulationEnv", side_effect=sentinels
        ) as env_class:
            results = [
                training._phase_env("g6", phase, rank=3, seed=7)
                for phase in config.Phase
            ]

        self.assertEqual(results, sentinels)
        self.assertEqual(
            env_class.call_args_list,
            [
                mock.call(
                    gripper="g6",
                    phase=config.Phase.GRASP_LIFT,
                    residual_nominal_feedback=True,
                    residual_after_preaction_only=False,
                ),
                mock.call(
                    gripper="g6",
                    phase=config.Phase.TRANSLATE,
                    residual_nominal_feedback=True,
                    residual_after_preaction_only=True,
                ),
                mock.call(
                    gripper="g6",
                    phase=config.Phase.ROTATE,
                    residual_nominal_feedback=True,
                    residual_after_preaction_only=True,
                    rotation_goal_max_deg=90.0,
                    rotation_sampling_profile="weighted_0_90",
                ),
            ],
        )
        for sentinel in sentinels:
            sentinel.action_space.seed.assert_called_once_with(10)

    def test_source_contains_one_learn_call_and_only_vanilla_tqc(self) -> None:
        source_path = config.PROJECT_ROOT / "g3g6" / "rotation_residual_training.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        learn_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "learn"
        ]
        self.assertEqual(len(learn_calls), 1)
        source = source_path.read_text(encoding="utf-8")
        self.assertNotIn("SafeWarmupTQC", source)
        self.assertNotIn("model.load", source)

    def test_mocked_training_runs_all_phases_and_writes_bound_artifacts(self) -> None:
        class FakeVecNormalize:
            def __init__(self, label: str) -> None:
                self.label = label
                self.obs_rms = {"label": f"{label}-obs"}
                self.ret_rms = {"label": f"{label}-ret"}
                self.save_calls: list[str] = []
                self.closed = False

            def save(self, path: str) -> None:
                self.save_calls.append(path)
                Path(path).write_bytes(b"normalizer-test-bytes")

            def close(self) -> None:
                self.closed = True

        class FakeModel:
            def __init__(self) -> None:
                self.learn_calls: list[dict[str, object]] = []
                self.save_calls: list[Path] = []
                self.set_env_calls: list[FakeVecNormalize] = []
                self.num_timesteps = 0
                self.replay_buffer = None

            def learn(self, **kwargs: object) -> None:
                self.learn_calls.append(dict(kwargs))
                if bool(kwargs["reset_num_timesteps"]):
                    self.num_timesteps = 0
                self.num_timesteps += training._round_up_to_vector_step(
                    int(kwargs["total_timesteps"])
                )

            def set_env(self, environment: FakeVecNormalize) -> None:
                self.set_env_calls.append(environment)

            def save(self, path: Path) -> None:
                value = Path(path)
                self.save_calls.append(value)
                value.write_bytes(b"model-test-bytes")

        vecs = [FakeVecNormalize(f"phase{phase}") for phase in range(1, 4)]
        model = FakeModel()
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as temporary:
            destination = Path(temporary) / "formal" / "g6"
            with (
                mock.patch.object(
                    training, "_build_phase_vec_env", side_effect=vecs
                ) as vec_builder,
                mock.patch.object(training, "_build_model", return_value=model),
            ):
                result = training.train_one_gripper(
                    gripper="g6",
                    output_dir=destination,
                    device="cpu",
                )

            self.assertEqual(result, destination.resolve())
            self.assertEqual(len(model.learn_calls), 3)
            self.assertEqual(
                [call["total_timesteps"] for call in model.learn_calls],
                [500_000, 2_000_000, 2_000_000],
            )
            self.assertEqual(
                [call["reset_num_timesteps"] for call in model.learn_calls],
                [True, False, False],
            )
            self.assertEqual(
                [call["progress_bar"] for call in model.learn_calls],
                [False, False, False],
            )
            self.assertEqual(model.set_env_calls, vecs[1:])
            self.assertEqual(
                [call.args[:2] for call in vec_builder.call_args_list],
                [
                    ("g6", config.Phase.GRASP_LIFT),
                    ("g6", config.Phase.TRANSLATE),
                    ("g6", config.Phase.ROTATE),
                ],
            )
            self.assertIsNone(vec_builder.call_args_list[0].args[2])
            self.assertEqual(
                vec_builder.call_args_list[1].args[2],
                (vecs[0].obs_rms, vecs[0].ret_rms),
            )
            self.assertEqual(
                vec_builder.call_args_list[2].args[2],
                (vecs[1].obs_rms, vecs[1].ret_rms),
            )

            manifest_path = destination / "manifest.json"
            model_path = destination / "final_model.zip"
            normalizer_path = destination / "vec_normalize.pkl"
            self.assertEqual(
                model.save_calls,
                [
                    destination / "phase1_model.zip",
                    destination / "phase2_model.zip",
                    destination / "phase3_model.zip",
                    model_path,
                ],
            )
            self.assertEqual(
                vecs[0].save_calls,
                [str(destination / "phase1_vec_normalize.pkl")],
            )
            self.assertEqual(
                vecs[1].save_calls,
                [str(destination / "phase2_vec_normalize.pkl")],
            )
            self.assertEqual(
                vecs[2].save_calls,
                [
                    str(destination / "phase3_vec_normalize.pkl"),
                    str(normalizer_path),
                ],
            )
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["gripper"], "g6")
            self.assertEqual(len(manifest["completed_phases"]), 3)
            self.assertEqual(
                [item["actual_steps"] for item in manifest["completed_phases"]],
                [500_040, 2_000_016, 2_000_016],
            )
            self.assertEqual(
                manifest["completed_phases"][-1]["cumulative_steps"],
                4_500_072,
            )
            self.assertEqual(
                manifest["artifacts"]["model"]["sha256"],
                hashlib.sha256(model_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["artifacts"]["vec_normalize"]["sha256"],
                hashlib.sha256(normalizer_path.read_bytes()).hexdigest(),
            )
            self.assertTrue(all(vec.closed for vec in vecs))

    def test_file_descriptor_soft_limit_is_raised_before_worker_creation(self) -> None:
        infinity = training.resource.RLIM_INFINITY
        with (
            mock.patch.object(
                training.resource,
                "getrlimit",
                side_effect=[(256, infinity), (65_536, infinity)],
            ),
            mock.patch.object(training.resource, "setrlimit") as setter,
        ):
            result = training.raise_file_descriptor_limit()

        self.assertEqual(result, (65_536, infinity))
        setter.assert_called_once_with(
            training.resource.RLIMIT_NOFILE,
            (65_536, infinity),
        )

    def test_existing_destination_is_rejected_before_worker_or_model_creation(self) -> None:
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as temporary:
            destination = Path(temporary) / "existing"
            destination.mkdir()
            with (
                mock.patch.object(training, "SubprocVecEnv") as subproc_class,
                mock.patch.object(training, "TQC") as tqc_class,
            ):
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    training.train_one_gripper(
                        gripper="g3", output_dir=destination, device="cpu"
                    )
            subproc_class.assert_not_called()
            tqc_class.assert_not_called()

    def test_print_config_never_creates_output_or_calls_training(self) -> None:
        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as temporary:
            output = Path(temporary) / "must_not_exist"
            stream = io.StringIO()
            with (
                mock.patch.object(training, "train_one_gripper") as train_mock,
                redirect_stdout(stream),
            ):
                training.main(
                    [
                        "--gripper",
                        "both",
                        "--device",
                        "cpu",
                        "--output-dir",
                        str(output),
                        "--print-config",
                    ]
                )
            train_mock.assert_not_called()
            self.assertFalse(output.exists())
            payload = json.loads(stream.getvalue())
            self.assertFalse(payload["training_started"])
            self.assertFalse(payload["output_created"])
            self.assertEqual(
                [item["gripper"] for item in payload["grippers"]], ["g3", "g6"]
            )


if __name__ == "__main__":
    unittest.main()
