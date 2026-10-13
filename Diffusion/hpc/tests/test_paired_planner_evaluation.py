from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from s_hpc.astar_evaluation import (
    EVALUATION_SEED_OFFSET,
    INITIAL_CONDITIONS_SCHEMA,
    canonical_sha256,
)
from s_hpc.config import load_config
from s_hpc.paired_planner_evaluation import (
    MeasuredAlignPathRecorder,
    TimedAStarPlanner,
    TimedDiffusionController,
    aggregate_latency_events,
    aggregate_paired_rows,
    assert_environment_matches_manifest,
    build_paired_rows,
    load_paired_initial_conditions,
    summarise_measured_path,
)
from s_hpc.supervisor import CommandBatch
from screw_control import TaskPhase


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "autodl_rtx_pro_6000_r100_w5000_tqc16_continuation.json"


class _State:
    pass


class _Environment:
    pass


class _FakeDiffusion:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.worlds = 2
        self.replan_interval = 4
        self.cursor = 4
        self.sample_count = 0

    def act(self, observation):
        if self.cursor >= self.replan_interval:
            self.sample_count += 1
            self.cursor = 0
        self.cursor += 1
        return observation[:, 0, :1]

    def reset(self, mask, observation_history=None):
        if observation_history is not None and bool(mask.any()):
            self.sample_count += 1

    def state_dict(self):
        return {"sample_count": self.sample_count}

    def load_state_dict(self, state):
        self.sample_count = int(state["sample_count"])


class _FakePlanner:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.worlds = 2
        self.calls = 0

    def reset(self, mask, **kwargs):
        self.calls += 1

    def command(self, value, **_kwargs):
        return value

    def provenance(self):
        return {"schema": "fake-planner"}

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = int(state["calls"])


class PairedPlannerEvaluationTests(unittest.TestCase):
    def test_path_summary_uses_planar_straightness_and_separate_3d_length(self):
        points = np.asarray(
            ((0.0, 0.0, 0.0), (0.5, 0.2, 0.5), (1.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        result = summarise_measured_path(points)
        expected_xy = 2.0 * math.hypot(0.5, 0.2)
        expected_xyz = 2.0 * math.sqrt(0.5**2 + 0.2**2 + 0.5**2)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["path_length_xy_m"], expected_xy)
        self.assertAlmostEqual(result["path_length_xyz_m"], expected_xyz)
        self.assertAlmostEqual(result["straight_distance_xy_m"], 1.0)
        self.assertAlmostEqual(result["max_straight_line_deviation_xy_m"], 0.2)

    def test_align_recorder_keeps_only_first_contiguous_segment(self):
        recorder = MeasuredAlignPathRecorder(1, 5, "cpu")
        active = torch.tensor([True])
        learned = torch.tensor([True])
        recorder.update(
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            first_episode_active=active,
            learned_active=learned,
            phase=torch.tensor([int(TaskPhase.ALIGN)]),
        )
        recorder.update(
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[2.0, 0.0, 0.0]]),
            first_episode_active=active,
            learned_active=learned,
            phase=torch.tensor([int(TaskPhase.INSERT)]),
        )
        recorder.update(
            torch.tensor([[2.0, 0.0, 0.0]]),
            torch.tensor([[3.0, 0.0, 0.0]]),
            first_episode_active=active,
            learned_active=learned,
            phase=torch.tensor([int(TaskPhase.ALIGN)]),
        )
        result = recorder.summaries()[0]
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["start_xyz_m"], [0.0, 0.0, 0.0])
        self.assertEqual(result["end_xyz_m"], [1.0, 0.0, 0.0])

    def test_manifest_config_hash_and_row_identity_are_enforced(self):
        config = load_config(CONFIG)
        seed = config.runtime.seed + EVALUATION_SEED_OFFSET
        evaluation = replace(config, runtime=replace(config.runtime, seed=seed))
        rows = [
            {
                "episode_index": index,
                "world_id": index,
                "root_seed": seed,
                "row_seed": seed + index,
                "spawn_x_m": 0.01 * (index + 1),
                "spawn_y_m": -0.02,
                "socket_mouth_x_m": 0.02,
                "socket_mouth_y_m": 0.0,
                "socket_angle_deg": 45.0,
            }
            for index in range(2)
        ]
        payload = {
            "schema": INITIAL_CONDITIONS_SCHEMA,
            "root_seed": seed,
            "evaluation_config_canonical_sha256": canonical_sha256(
                evaluation.to_dict()
            ),
            "rows": rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initial.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded, detached, selected = load_paired_initial_conditions(
                path, config=config
            )
        self.assertEqual(loaded["rows"], rows)
        self.assertEqual(detached.runtime.seed, seed)
        self.assertEqual(selected, rows)
        self.assertEqual(config.runtime.seed + EVALUATION_SEED_OFFSET, seed)

    def test_environment_rows_are_compared_before_physics(self):
        rows = [
            {
                "spawn_x_m": 0.01,
                "spawn_y_m": -0.02,
                "socket_mouth_x_m": 0.02,
                "socket_mouth_y_m": 0.0,
                "socket_angle_deg": 45.0,
            },
            {
                "spawn_x_m": -0.03,
                "spawn_y_m": 0.04,
                "socket_mouth_x_m": 0.0,
                "socket_mouth_y_m": 0.02,
                "socket_angle_deg": 45.0,
            },
        ]
        env = _Environment()
        env.worlds = 2
        env.state = _State()
        env.state.spawn_xy = torch.tensor(
            [[0.01, -0.02], [-0.03, 0.04]], dtype=torch.float32
        )
        env.state.socket_mouth = torch.tensor(
            [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0]], dtype=torch.float32
        )
        env.state.socket_angle_deg = torch.tensor([45.0, 45.0])
        result = assert_environment_matches_manifest(env, rows)
        self.assertTrue(result["matched"])
        env.state.spawn_xy[0, 0] += 1.0e-3
        with self.assertRaisesRegex(RuntimeError, "spawn rows"):
            assert_environment_matches_manifest(env, rows)

    def test_timed_wrappers_report_batch_scope(self):
        diffusion = TimedDiffusionController(_FakeDiffusion())
        active = torch.tensor([True, False])
        diffusion.set_first_episode_active(active)
        observation = torch.zeros((2, 2, 1))
        diffusion.act(observation)
        diffusion.reset(torch.tensor([False, True]), observation)
        self.assertEqual(len(diffusion.events), 1)
        self.assertTrue(diffusion.events[0]["comparison_relevant"])
        self.assertEqual(diffusion.background_only_samples_suppressed, 1)
        aggregate = aggregate_latency_events(diffusion.events[:1])
        self.assertTrue(aggregate["available"])
        self.assertEqual(aggregate["events"], 1)

        planner = TimedAStarPlanner(_FakePlanner())
        planner.reset(torch.tensor([True, True]))
        planner.reset(torch.tensor([True, True]))
        planner.mark_paired_initial_state()
        self.assertEqual(planner.paired_initial_event_index, 1)
        self.assertTrue(planner.events[1]["paired_initial_state"])

    def test_timed_astar_can_skip_only_the_throwaway_graph_warmup_reset(self):
        inner = _FakePlanner()
        planner = TimedAStarPlanner(inner, skip_initial_resets=1)
        full = torch.tensor([True, True])
        planner.reset(full, start_command=object())
        self.assertEqual(inner.calls, 0)
        self.assertEqual(len(planner.events), 0)
        self.assertEqual(
            planner.skipped_reset_events,
            [
                {
                    "kind": "planner_reset_skipped",
                    "batch_rows": 2,
                    "reason": "non_evaluation_cuda_graph_warmup_state",
                    "physics_environment_reset_preserved": True,
                }
            ],
        )

        planner.reset(full, start_command=object())
        self.assertEqual(inner.calls, 1)
        planner.mark_paired_initial_state()
        self.assertEqual(planner.paired_initial_event_index, 0)
        policy = planner.provenance()["evaluation_reset_policy"]
        self.assertTrue(
            policy["non_evaluation_cuda_graph_warmup_planning_skipped"]
        )
        self.assertTrue(policy["physics_environment_resets_preserved"])
        self.assertTrue(policy["manifest_matched_reset_planned"])

    def test_timed_astar_rejects_partial_warmup_skip(self):
        planner = TimedAStarPlanner(_FakePlanner(), skip_initial_resets=1)
        with self.assertRaisesRegex(RuntimeError, "requires a full reset"):
            planner.reset(torch.tensor([True, False]))

    def test_timed_astar_suppresses_only_background_post_episode_plans(self):
        inner = _FakePlanner()
        planner = TimedAStarPlanner(inner)
        full = torch.tensor([True, True])
        planner.reset(full, start_command=object())
        planner.mark_paired_initial_state()
        active = torch.tensor([True, True])
        planner.set_first_episode_active(active)

        active[1] = False
        handoff = CommandBatch(
            tip=torch.full((2, 3), 3.0),
            rotation=torch.full((2, 3, 3), 3.0),
            twist=torch.full((2,), 3.0),
            material=torch.full((2, 4), 3.0),
            distance=torch.full((2, 4), 3.0),
            phase=torch.full((2,), int(TaskPhase.ALIGN), dtype=torch.long),
            engaged=torch.zeros(2, dtype=torch.bool),
        )
        planner.reset(torch.tensor([False, True]), start_command=handoff)
        self.assertEqual(inner.calls, 1)
        self.assertEqual(len(planner.events), 1)
        self.assertEqual(
            planner.background_reset_events,
            [
                {
                    "kind": "planner_reset_skipped",
                    "batch_rows": 1,
                    "reason": "background_only_post_first_episode_state",
                    "first_episode_rows_in_reset": 0,
                    "physics_environment_reset_preserved": True,
                    "certified_handoff_hold_installed": True,
                }
            ],
        )

        planned = CommandBatch(
            tip=torch.full((2, 3), 7.0),
            rotation=torch.full((2, 3, 3), 7.0),
            twist=torch.full((2,), 7.0),
            material=torch.full((2, 4), 7.0),
            distance=torch.full((2, 4), 7.0),
            phase=torch.full((2,), int(TaskPhase.ALIGN), dtype=torch.long),
            engaged=torch.zeros(2, dtype=torch.bool),
        )
        result = planner.command(planned, next_step=torch.ones(2))
        for name in ("tip", "rotation", "twist", "material", "distance"):
            self.assertTrue(
                torch.equal(getattr(result, name)[0], getattr(planned, name)[0])
            )
            self.assertTrue(
                torch.equal(getattr(result, name)[1], getattr(handoff, name)[1])
            )
        self.assertTrue(torch.equal(result.phase, planned.phase))
        self.assertTrue(torch.equal(result.engaged, planned.engaged))

        with self.assertRaisesRegex(RuntimeError, "one-shot global planning"):
            planner.reset(torch.tensor([True, False]), start_command=handoff)
        self.assertEqual(inner.calls, 1)
        policy = planner.provenance()["evaluation_reset_policy"]
        self.assertTrue(policy["background_only_planning_skipped"])
        self.assertEqual(policy["background_only_skipped_resets"], 1)
        self.assertEqual(policy["background_only_skipped_rows"], 1)
        self.assertTrue(policy["first_episode_planning_preserved"])
        self.assertTrue(policy["background_rows_held_at_certified_handoff"])

    def test_paired_rows_require_exact_initial_conditions(self):
        base = {
            "world_id": 0,
            "initial_condition": {"world_id": 0, "spawn_x_m": 0.1},
            "success": True,
            "reached_insert": True,
            "first_insert_tick": 8,
            "terminal_steps": 10,
            "measured_align_path": {
                "path_length_xy_m": 0.2,
                "path_length_xyz_m": 0.3,
                "path_to_straight_ratio_xy": 1.1,
                "max_straight_line_deviation_xy_m": 0.01,
            },
        }
        astar = {
            **base,
            "success": False,
            "reached_insert": False,
            "first_insert_tick": None,
            "terminal_steps": 20,
            "measured_align_path": {
                **base["measured_align_path"],
                "path_length_xy_m": 0.1,
            },
        }
        rows = build_paired_rows([base], [astar])
        self.assertEqual(rows[0]["paired_delta"]["success_diffusion_minus_astar"], 1)
        self.assertEqual(
            rows[0]["paired_delta"]["reached_insert_diffusion_minus_astar"],
            1,
        )
        self.assertIsNone(
            rows[0]["paired_delta"]["first_insert_tick_diffusion_minus_astar"]
        )
        self.assertAlmostEqual(
            rows[0]["paired_delta"]["path_length_xy_m_diffusion_minus_astar"],
            0.1,
        )
        summary = aggregate_paired_rows(rows)
        self.assertEqual(summary["outcome_pair_counts"]["diffusion_only_success"], 1)
        self.assertEqual(summary["both_reached_insert"], 0)


if __name__ == "__main__":
    unittest.main()
