from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pipeline
from s_hpc import workflows


class DiffusionOnlyPipelineTest(unittest.TestCase):
    def test_formal_stage_never_constructs_or_trains_tqc(self):
        config = SimpleNamespace(
            name="diffusion-only-test",
            runtime=SimpleNamespace(
                device="cpu",
                status_interval_seconds=1.0,
            ),
            physics=SimpleNamespace(worlds=18_000),
            to_dict=lambda: {"name": "diffusion-only-test"},
        )
        monitor = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            run_dir = root / "run"
            argv = (
                "pipeline.py",
                "--config",
                str(config_path),
                "--stage",
                "diffusion-full",
                "--run-dir",
                str(run_dir),
                "--episodes",
                "2000",
            )
            with (
                patch("sys.argv", list(argv)),
                patch(
                    "pipeline._local_path",
                    side_effect=lambda value, **_kwargs: Path(value),
                ),
                patch("pipeline.load_config", return_value=config),
                patch("pipeline.RunMonitor", return_value=monitor),
                patch("pipeline.generate_demos") as generate,
                patch("pipeline.train_diffusion") as diffusion,
                patch("pipeline.train_residual") as residual,
                patch("pipeline.evaluate") as evaluate,
                patch("pipeline._release_cuda_stage_memory"),
            ):
                pipeline.main()

        generate.assert_called_once()
        diffusion.assert_called_once()
        residual.assert_not_called()
        evaluate.assert_called_once()
        self.assertIsNone(evaluate.call_args.kwargs["tqc_checkpoint"])
        self.assertEqual(evaluate.call_args.kwargs["episodes"], 2000)

    def test_pure_evaluation_builds_sixteen_dimensional_zero_residual(self):
        import torch

        config = SimpleNamespace(
            name="diffusion-only-evaluation-test",
            runtime=SimpleNamespace(
                seed=17,
                deterministic=False,
                device="cpu",
                log_every_vector_steps=50,
            ),
            physics=SimpleNamespace(worlds=2),
            diffusion=SimpleNamespace(
                inference_steps=4,
                replan_interval=2,
            ),
            to_dict=lambda: {"name": "diffusion-only-evaluation-test"},
        )
        tracker = MagicMock()
        tracker.snapshot.return_value = {
            "successes": 1,
            "failures": 0,
            "truncations": 0,
            "success_rate": 1.0,
            "mean_return": 1.0,
            "mean_depth_m": 0.08,
            "mean_turns": 2.0,
        }

        class FakeEnv:
            def __init__(self, worlds):
                self.device = torch.device("cpu")
                self.worlds = worlds
                self.obs_history = torch.zeros(
                    (worlds, 2, workflows.OBS_DIM)
                )
                self.received_residual = None

            def step(self, residual):
                self.received_residual = residual.clone()
                return SimpleNamespace(
                    done=torch.ones(
                        self.worlds, dtype=torch.bool
                    ),
                    observation=torch.zeros(
                        (self.worlds, workflows.OBS_DIM)
                    ),
                    info={
                        "executed_diffusion_action": torch.zeros(
                            (self.worlds, workflows.ACTION_DIM)
                        )
                    },
                )

            def reset(self, _mask):
                return self.obs_history[:, -1]

            def close(self):
                return None

        fake_env = FakeEnv(worlds=1)
        with (
            patch("s_hpc.workflows._seed_everything"),
            patch(
                "s_hpc.workflows._load",
                return_value={"kind": "diffusion"},
            ),
            patch("s_hpc.workflows.validate_checkpoint"),
            patch("s_hpc.workflows._validate_checkpoint_config"),
            patch("s_hpc.workflows._validate_frozen_diffusion_config"),
            patch(
                "s_hpc.workflows.sha256_file",
                return_value="diffusion-sha256",
            ),
            patch(
                "s_hpc.workflows.BatchedDiffusionController.from_checkpoint",
                return_value=MagicMock(),
            ),
            patch(
                "s_hpc.workflows.MJWarpScrewEnv",
                return_value=fake_env,
            ),
            patch(
                "s_hpc.workflows._episode_tracker",
                return_value=tracker,
            ),
            patch(
                "s_hpc.workflows._track_terminal_episodes",
                return_value=1,
            ),
            patch(
                "s_hpc.workflows._assert_backend_health",
                return_value={"finite": True},
            ),
            patch(
                "s_hpc.workflows._materialize_rollout_metrics",
                return_value=({}, None),
            ),
        ):
            result = workflows.evaluate(
                config,
                diffusion_checkpoint=Path("diffusion.pt"),
                tqc_checkpoint=None,
                episodes=1,
            )

        self.assertEqual(result["episodes"], 1.0)
        self.assertEqual(
            tuple(fake_env.received_residual.shape), (1, 16)
        )
        self.assertTrue(
            torch.equal(
                fake_env.received_residual,
                torch.zeros_like(fake_env.received_residual),
            )
        )


if __name__ == "__main__":
    unittest.main()
