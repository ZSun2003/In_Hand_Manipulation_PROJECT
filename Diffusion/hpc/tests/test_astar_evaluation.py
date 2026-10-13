from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import torch


HPC_ROOT = Path(__file__).resolve().parents[1]
if str(HPC_ROOT) not in sys.path:
    sys.path.insert(0, str(HPC_ROOT))

from evaluate_astar_spline_gpu import build_parser
from s_hpc.astar_evaluation import evaluate_astar_spline
from s_hpc.contracts import RESIDUAL_ACTION_DIM


class FakeConfig:
    name = "astar_test"

    def __init__(self, *, worlds: int = 4, max_steps: int = 8) -> None:
        self.physics = SimpleNamespace(
            worlds=worlds,
            max_episode_steps=max_steps,
        )
        self.runtime = SimpleNamespace(
            seed=73,
            device="cpu",
            status_interval_seconds=1.0,
            log_every_vector_steps=2,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "physics": {
                "worlds": self.physics.worlds,
                "max_episode_steps": self.physics.max_episode_steps,
            },
            "runtime": {
                "seed": self.runtime.seed,
                "device": self.runtime.device,
                "log_every_vector_steps": (
                    self.runtime.log_every_vector_steps
                ),
            },
        }


class FakePlanner:
    def __init__(self, worlds: int) -> None:
        self.worlds = worlds

    def provenance(self):
        return {
            "algorithm": "astar_spline",
            "grid_resolution_m": 0.002,
            "connectivity": 8,
        }

    def diagnostics(self):
        return {
            "plan_found": torch.ones(self.worlds, dtype=torch.bool),
            "planning_latency_ms": torch.arange(
                self.worlds, dtype=torch.float32
            ),
            "path_length_m": torch.tensor(
                [0.10, 0.20, 0.30], dtype=torch.float32
            )[: self.worlds],
            "minimum_obstacle_clearance_m": torch.tensor(
                [0.010, 0.020, 0.030], dtype=torch.float32
            )[: self.worlds],
            "astar_expanded": torch.tensor(
                [10, 20, 30], dtype=torch.long
            )[: self.worlds],
            "spline_fallback": torch.tensor(
                [False, True, False], dtype=torch.bool
            )[: self.worlds],
            # Non-scalar paths must not leak into per-episode JSON.
            "path_points": torch.zeros((self.worlds, 5, 3)),
            # Global counters belong only in the summary.
            "total_reset_plans": 17,
            "cache_hits": 5,
        }


class FakeResult:
    def __init__(self, done, info) -> None:
        self.done = done
        self.info = info


class FakeMonitor:
    def __init__(self) -> None:
        self.heartbeats = []
        self.stages = []

    @property
    def due(self):
        return True

    def set_stage(self, name, **kwargs):
        self.stages.append((name, kwargs))

    def heartbeat(self, **kwargs):
        self.heartbeats.append(kwargs)


class FakeEnv:
    def __init__(
        self,
        config,
        *,
        worlds,
        baseline,
        nominal_planner,
        curriculum_progress,
        manifest_path: Path,
        never_done: bool = False,
    ) -> None:
        assert baseline == "astar_spline"
        assert nominal_planner.worlds == worlds
        assert curriculum_progress == 1.0
        assert config.runtime.seed == 100_073
        self.config = config
        self.worlds = worlds
        self.device = torch.device("cpu")
        self.state = SimpleNamespace(
            spawn_xy=torch.stack(
                (
                    torch.arange(worlds, dtype=torch.float32) / 100.0,
                    -torch.arange(worlds, dtype=torch.float32) / 200.0,
                ),
                dim=-1,
            ),
            socket_mouth=torch.tensor(
                [[0.02, 0.0, 0.33]] * worlds, dtype=torch.float32
            ),
            socket_angle_deg=torch.full((worlds,), 45.0),
        )
        self.manifest_path = manifest_path
        self.never_done = never_done
        self.steps = 0
        self.reset_masks: list[list[bool]] = []
        self.residuals: list[torch.Tensor] = []
        self.closed = False
        self.terminal_in_physics = torch.zeros(
            worlds, dtype=torch.bool
        )

    def step(self, *, residual_action):
        # The central contract: exact initial states exist on disk before the
        # first call that advances physics.
        assert self.manifest_path.is_file()
        assert residual_action.shape == (
            self.worlds,
            RESIDUAL_ACTION_DIM,
        )
        assert bool((residual_action == 0).all())
        # A correct evaluator resets every terminal row before the next tick,
        # including already-inactive rows that finish a later episode.
        assert not bool(self.terminal_in_physics.any())
        self.residuals.append(residual_action.detach().clone())
        self.steps += 1
        terminal_tick = torch.arange(1, self.worlds + 1)
        done = (
            torch.zeros(self.worlds, dtype=torch.bool)
            if self.never_done
            else terminal_tick <= self.steps
        )
        self.terminal_in_physics |= done
        success = done & (torch.arange(self.worlds) == 0)
        failure = done & (torch.arange(self.worlds) == 1)
        truncated = done & ~(success | failure)
        ids = torch.arange(self.worlds, dtype=torch.float32)
        info = {
            "success": success,
            "failure": failure,
            "truncated": truncated,
            "elapsed_steps": torch.full(
                (self.worlds,), self.steps, dtype=torch.long
            ),
            "task_phase": torch.full((self.worlds,), 3, dtype=torch.long),
            "depth": 0.08 + ids * 1.0e-5,
            "twist": torch.full((self.worlds,), 4.0 * torch.pi),
            "lateral_error": 0.001 + ids * 1.0e-5,
            "axis_error": torch.deg2rad(
                torch.full((self.worlds,), 0.2)
            ),
            "speed": torch.full((self.worlds,), 0.01),
            "ik_failures": torch.arange(self.worlds, dtype=torch.long),
            "ik_anchor_fallbacks": torch.zeros(
                self.worlds, dtype=torch.long
            ),
            "ik_safety_stops": failure.to(torch.long),
            "planner_ik_failure": failure,
            "astar_planner_active": torch.ones(
                self.worlds, dtype=torch.bool
            ),
        }
        return FakeResult(done, info)

    def reset(self, mask):
        self.reset_masks.append(mask.detach().cpu().tolist())
        self.terminal_in_physics[mask.detach().cpu()] = False

    def close(self):
        self.closed = True


class AStarEvaluationTest(unittest.TestCase):
    def _run(
        self,
        root: Path,
        *,
        never_done: bool = False,
        max_steps=6,
        monitor=None,
        health_check=None,
    ):
        config = FakeConfig(worlds=3, max_steps=8)
        holder = {}
        manifest = root / "initial_conditions_manifest.json"

        def planner_factory(_config, *, worlds, device):
            self.assertEqual(device, "cpu")
            self.assertEqual(_config.runtime.seed, 100_073)
            holder["planner_config"] = _config
            planner = FakePlanner(worlds)
            holder["planner"] = planner
            return planner

        def env_factory(_config, **kwargs):
            env = FakeEnv(
                _config,
                **kwargs,
                manifest_path=manifest,
                never_done=never_done,
            )
            holder["env"] = env
            return env

        summary = evaluate_astar_spline(
            config,
            episodes=3,
            worlds=3,
            max_steps=max_steps,
            run_dir=root,
            log_path=root / "episodes.jsonl",
            planner_factory=planner_factory,
            env_factory=env_factory,
            monitor=monitor,
            health_check=health_check,
        )
        holder["base_config"] = config
        return summary, holder

    def test_records_only_first_episode_and_sha_bound_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, holder = self._run(root)
            self.assertEqual(summary["episodes"], 3)
            self.assertEqual(summary["successes"], 1)
            self.assertEqual(summary["failures"], 1)
            self.assertEqual(summary["truncations"], 1)
            self.assertEqual(summary["planner_ik_failure_episodes"], 1)
            env = holder["env"]
            self.assertTrue(env.closed)
            self.assertEqual(len(env.reset_masks), 2)
            self.assertEqual(env.reset_masks[0], [True, False, False])
            self.assertEqual(env.reset_masks[1], [True, True, False])
            self.assertEqual(holder["base_config"].runtime.seed, 73)
            self.assertEqual(holder["planner_config"].runtime.seed, 100_073)
            self.assertIs(holder["planner_config"], env.config)
            manifest = root / "initial_conditions_manifest.json"
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(
                summary["initial_conditions_manifest"]["sha256"], digest
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["rows"]), 3)
            self.assertEqual(payload["rows"][2]["world_id"], 2)
            self.assertEqual(payload["base_seed"], 73)
            self.assertEqual(payload["evaluation_seed"], 100_073)
            self.assertEqual(payload["rows"][0]["root_seed"], 100_073)

            events = [
                json.loads(line)
                for line in (root / "episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            episodes = [
                event["episode"]
                for event in events
                if event.get("event") == "episode"
            ]
            self.assertEqual(
                [episode["world_id"] for episode in episodes], [0, 1, 2]
            )
            self.assertTrue(episodes[0]["planner_diagnostics"]["plan_found"])
            self.assertNotIn(
                "path_points", episodes[0]["planner_diagnostics"]
            )
            self.assertNotIn(
                "total_reset_plans",
                episodes[0]["planner_diagnostics"],
            )
            self.assertEqual(
                summary["planner_diagnostics"]["final_global"][
                    "total_reset_plans"
                ],
                17,
            )
            self.assertEqual(
                summary["planner_metrics"]["plan_found"]["count"], 3
            )
            self.assertEqual(
                summary["planner_metrics"]["spline_fallback"]["count"],
                1,
            )
            self.assertAlmostEqual(
                summary["planner_metrics"]["path_length_m"]["mean"],
                0.2,
                places=6,
            )
            self.assertAlmostEqual(
                summary["planner_metrics"][
                    "minimum_obstacle_clearance_m"
                ]["min"],
                0.01,
                places=6,
            )
            self.assertEqual(
                summary["planner_metrics"]["astar_expanded"]["max"],
                30.0,
            )
            self.assertEqual(summary["provenance"]["checkpoints"], {})
            self.assertEqual(
                summary["provenance"]["residual"]["mode"],
                "disabled_zero",
            )
            self.assertEqual(summary["provenance"]["base_seed"], 73)
            self.assertEqual(
                summary["provenance"]["evaluation_seed"], 100_073
            )
            self.assertIn("schema_hash", summary["provenance"]["contracts"])
            self.assertIn(
                "task_geometry_hash", summary["provenance"]["contracts"]
            )
            self.assertEqual(
                set(summary["provenance"]["source"]),
                {
                    "backend",
                    "cli",
                    "env",
                    "evaluator",
                    "ik",
                    "planner",
                    "planner_grid_base",
                    "runtime_environment",
                    "supervisor",
                    "task",
                },
            )
            self.assertEqual(summary["execution"]["vector_steps"], 3)
            self.assertEqual(
                summary["health"]["check"], "skipped_no_backend"
            )

            summary_payload = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary_payload, summary)
            summary_sidecar = json.loads(
                (root / "summary.sha256.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary_sidecar["sha256"],
                hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
            )
            log_sidecar = json.loads(
                (root / "episodes.jsonl.sha256.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                log_sidecar["sha256"],
                hashlib.sha256((root / "episodes.jsonl").read_bytes()).hexdigest(),
            )

    def test_evaluation_limit_forces_one_truncation_per_world(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, holder = self._run(
                root, never_done=True, max_steps=1
            )
            self.assertEqual(summary["successes"], 0)
            self.assertEqual(summary["failures"], 0)
            self.assertEqual(summary["truncations"], 3)
            self.assertEqual(
                summary["failure_reasons"], {"evaluation_step_limit": 3}
            )
            self.assertEqual(holder["env"].reset_masks, [])

    def test_health_and_heartbeat_do_not_depend_on_terminal_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            monitor = FakeMonitor()
            health_contexts = []

            def health_check(_env, *, context):
                health_contexts.append(context)
                return {"finite": True, "context": context}

            summary, _holder = self._run(
                root,
                never_done=True,
                max_steps=3,
                monitor=monitor,
                health_check=health_check,
            )
            progress_ticks = [
                item["metrics"]["vector_steps"]
                for item in monitor.heartbeats
                if "vector_steps" in item.get("metrics", {})
                and item["metrics"]["vector_steps"] > 0
            ]
            self.assertEqual(progress_ticks, [1, 2, 3])
            self.assertTrue(
                any("vector step 1" in value for value in health_contexts)
            )
            self.assertEqual(summary["execution"]["vector_steps"], 3)
            self.assertTrue(summary["health"]["finite"])

    def test_world_capacity_must_cover_all_committed_initial_states(self):
        config = FakeConfig(worlds=2, max_steps=8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "at least --episodes"):
                evaluate_astar_spline(
                    config,
                    episodes=3,
                    worlds=2,
                    run_dir=root,
                    log_path=root / "episodes.jsonl",
                    planner_factory=lambda *_args, **_kwargs: None,
                    env_factory=lambda *_args, **_kwargs: None,
                )

    def test_cli_exposes_required_and_capacity_flags(self):
        args = build_parser().parse_args(
            [
                "--config",
                "config.json",
                "--episodes",
                "32",
                "--run-dir",
                "run",
                "--worlds",
                "64",
                "--max-steps",
                "1200",
            ]
        )
        self.assertEqual(args.episodes, 32)
        self.assertEqual(args.worlds, 64)
        self.assertEqual(args.max_steps, 1200)


if __name__ == "__main__":
    unittest.main()
