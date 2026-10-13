from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from screw_control import (
    PREINSERT_GATE_STEP,
    SCRIPTED_PREFIX_STEPS,
    TaskPhase,
)
from s_hpc.astar_spline import (
    AStarPlanError,
    BatchedAStarSplinePlanner,
    DOCK_STEPS,
    ROTATE_RAISE_STEPS,
    TRANSPORT_STEPS,
)
from s_hpc.config import HPCConfig
from s_hpc.supervisor import BatchedSupervisor


class BatchedAStarSplinePlannerTest(unittest.TestCase):
    def setUp(self):
        self.worlds = 3
        self.supervisor = BatchedSupervisor(device="cpu")
        self.config = HPCConfig()
        self.angle = torch.full((self.worlds,), 45.0)
        self.mouth_xy = torch.tensor(
            (
                (0.020, 0.0),
                (0.0, 0.020),
                (-0.020, 0.0),
            ),
            dtype=torch.float32,
        )
        # Canonical positions are rotated with each mouth.  The middle start
        # lies behind and to the side of the socket, forcing a non-straight
        # route around the inflated socket rectangle.
        canonical_spawn = torch.tensor(
            (
                (-0.075, 0.0),
                (0.070, 0.070),
                (-0.060, -0.030),
            ),
            dtype=torch.float32,
        )
        yaw = torch.atan2(self.mouth_xy[:, 1], self.mouth_xy[:, 0])
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        rotation = torch.stack(
            (
                torch.stack((cosine, -sine), dim=-1),
                torch.stack((sine, cosine), dim=-1),
            ),
            dim=-2,
        )
        self.spawn = torch.einsum("bij,bj->bi", rotation, canonical_spawn)
        self.start = self.supervisor.at(
            torch.full((self.worlds,), SCRIPTED_PREFIX_STEPS),
            self.spawn,
            self.angle,
            self.mouth_xy,
        )
        self.goal = self.supervisor.at(
            torch.full((self.worlds,), PREINSERT_GATE_STEP),
            self.spawn,
            self.angle,
            self.mouth_xy,
        )
        (
            _frame,
            self.axis,
            self.mouth,
            _aligned,
        ) = self.supervisor.socket_geometry(self.angle, self.mouth_xy)
        self.planner = BatchedAStarSplinePlanner(
            self.config, self.worlds, "cpu"
        )

    def tearDown(self):
        self.planner.close()

    def _reset_all(self):
        self.planner.reset(
            torch.ones(self.worlds, dtype=torch.bool),
            start_command=self.start,
            goal_command=self.goal,
            socket_mouth=self.mouth,
            socket_axis=self.axis,
        )

    def _command_at(self, step: int):
        next_step = torch.full((self.worlds,), step, dtype=torch.long)
        anchor = self.supervisor.at(
            next_step, self.spawn, self.angle, self.mouth_xy
        )
        return self.planner.command(anchor, next_step=next_step)

    def test_geometry_and_segment_contract(self):
        self.assertEqual(
            TRANSPORT_STEPS + ROTATE_RAISE_STEPS + DOCK_STEPS,
            350,
        )
        self.assertAlmostEqual(self.planner.grid_resolution_m, 0.002)
        self.assertAlmostEqual(self.planner._obstacle_center_x, 0.0553553391)
        self.assertAlmostEqual(self.planner._obstacle_half_x, 0.0916396103)
        self.assertAlmostEqual(self.planner._obstacle_half_y, 0.068)
        stage = self.planner._canonical_stage_xy
        self.assertAlmostEqual(float(stage[0]), -0.0401040764)
        self.assertAlmostEqual(float(stage[1]), 0.0)
        front = (
            self.planner._obstacle_center_x
            - self.planner._obstacle_half_x
        )
        self.assertGreater(front - float(stage[0]), 0.003)

    def test_true_astar_is_connected_and_has_no_corner_cutting(self):
        start = self.planner._nearest_free_grid_cell(
            np.asarray((0.070, 0.070)), require_visible=True
        )
        self.assertIsNotNone(start)
        result = self.planner._astar(start)
        self.assertGreater(result.expanded, 1)
        self.assertEqual(result.indices[0], self.planner._flatten(*start))
        self.assertEqual(result.indices[-1], self.planner._goal_flat)
        self.assertTrue(self.planner._points_free(result.points_xy).all())
        for first, second in zip(result.indices, result.indices[1:]):
            i, j = self.planner._unflatten(first)
            ni, nj = self.planner._unflatten(second)
            self.assertLessEqual(abs(ni - i), 1)
            self.assertLessEqual(abs(nj - j), 1)
            if ni != i and nj != j:
                self.assertTrue(self.planner._free[ni, j])
                self.assertTrue(self.planner._free[i, nj])

    def test_reset_builds_collision_free_curved_transport_and_exact_endpoints(self):
        self._reset_all()
        diagnostics = self.planner.diagnostics()
        self.assertEqual(diagnostics["plan_found"].tolist(), [True] * 3)
        self.assertEqual(diagnostics["unique_plans"], 3)
        self.assertTrue(bool((diagnostics["path_length_m"] > 0.0).all()))
        self.assertTrue(
            bool(
                (
                    diagnostics["minimum_obstacle_clearance_m"]
                    >= 0.0
                ).all()
            )
        )

        # Convert the yaw=90 degree world back into canonical coordinates and
        # densely check its prepared transport samples against the C-obstacle.
        world_path = self.planner._plan_tip[1, :TRANSPORT_STEPS, :2].numpy()
        inward = np.asarray((0.0, 1.0))
        tangent = np.asarray((-1.0, 0.0))
        canonical = np.stack(
            (world_path @ inward, world_path @ tangent), axis=-1
        )
        self.assertTrue(self.planner._points_free(canonical).all())
        self.assertTrue(
            self.planner._execution_chords_free(
                np.asarray((0.070, 0.070)), canonical
            )
        )

        # The rear-side direct segment crosses the inflated rectangle; A*
        # therefore has to deviate measurably from it.
        start = canonical[0]
        goal = canonical[-1]
        direction = goal - start
        distance = np.abs(
            direction[0] * (canonical[:, 1] - start[1])
            - direction[1] * (canonical[:, 0] - start[0])
        ) / np.linalg.norm(direction)
        self.assertGreater(float(distance.max()), 0.005)

        # Including the exact lifted handoff, every translational increment
        # remains below the existing 2 mm lateral / 3 mm axial task limits.
        for world in range(self.worlds):
            full_tip = torch.cat(
                (
                    self.start.tip[world : world + 1],
                    self.planner._plan_tip[world],
                ),
                dim=0,
            )
            increment = torch.linalg.vector_norm(
                torch.diff(full_tip, dim=0), dim=-1
            )
            self.assertLessEqual(float(increment.max()), 0.00201)

        first = self._command_at(SCRIPTED_PREFIX_STEPS + 1)
        transport_end = self._command_at(
            SCRIPTED_PREFIX_STEPS + TRANSPORT_STEPS
        )
        transition_end = self._command_at(
            SCRIPTED_PREFIX_STEPS
            + TRANSPORT_STEPS
            + ROTATE_RAISE_STEPS
        )
        final = self._command_at(PREINSERT_GATE_STEP)
        torch.testing.assert_close(
            first.rotation, self.start.rotation, atol=1.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            transport_end.tip[:, 2], self.start.tip[:, 2], atol=1.0e-7, rtol=0.0
        )
        expected_stage = self.mouth - 0.085 * self.axis
        torch.testing.assert_close(
            transition_end.tip, expected_stage, atol=2.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            transition_end.rotation,
            self.goal.rotation,
            atol=2.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(final.tip, self.goal.tip, atol=2.0e-6, rtol=0.0)
        torch.testing.assert_close(
            final.rotation, self.goal.rotation, atol=2.0e-6, rtol=0.0
        )

    def test_command_preserves_non_align_anchor_and_contact_fields(self):
        self._reset_all()
        step = torch.tensor(
            (
                SCRIPTED_PREFIX_STEPS + 1,
                PREINSERT_GATE_STEP + 1,
                PREINSERT_GATE_STEP,
            )
        )
        anchor = self.supervisor.at(
            step, self.spawn, self.angle, self.mouth_xy
        )
        command = self.planner.command(anchor, next_step=step)
        self.assertEqual(int(anchor.phase[1]), int(TaskPhase.INSERT))
        torch.testing.assert_close(command.tip[1], anchor.tip[1])
        torch.testing.assert_close(command.rotation[1], anchor.rotation[1])
        torch.testing.assert_close(command.material, anchor.material)
        torch.testing.assert_close(command.distance, anchor.distance)
        torch.testing.assert_close(command.twist, anchor.twist)
        self.assertTrue(torch.equal(command.phase, anchor.phase))
        self.assertTrue(torch.equal(command.engaged, anchor.engaged))

    def test_async_masked_reset_preserves_other_world_plans(self):
        self._reset_all()
        old_tip = self.planner._plan_tip[1:].clone()
        old_rotation = self.planner._plan_rotation[1:].clone()
        replacement = self.start.clone()
        replacement.tip[0, :2] = torch.tensor((-0.090, -0.010))
        self.planner.reset(
            torch.tensor((True, False, False)),
            start_command=replacement,
            goal_command=self.goal,
            socket_mouth=self.mouth,
            socket_axis=self.axis,
        )
        torch.testing.assert_close(self.planner._plan_tip[1:], old_tip)
        torch.testing.assert_close(self.planner._plan_rotation[1:], old_rotation)
        self.assertTrue(bool(self.planner.diagnostics()["plan_found"].all()))

    def test_cache_is_keyed_by_canonical_start_cell_across_socket_yaw(self):
        # Worlds 0 and 1 receive the same canonical start in different yaws.
        canonical = torch.tensor((-0.075, 0.010))
        spawn = self.spawn.clone()
        spawn[0] = canonical
        spawn[1] = torch.tensor((-canonical[1], canonical[0]))
        start = self.supervisor.at(
            torch.full((self.worlds,), SCRIPTED_PREFIX_STEPS),
            spawn,
            self.angle,
            self.mouth_xy,
        )
        goal = self.supervisor.at(
            torch.full((self.worlds,), PREINSERT_GATE_STEP),
            spawn,
            self.angle,
            self.mouth_xy,
        )
        self.planner.reset(
            torch.tensor((True, True, False)),
            start_command=start,
            goal_command=goal,
            socket_mouth=self.mouth,
            socket_axis=self.axis,
        )
        diagnostics = self.planner.diagnostics()
        self.assertFalse(bool(diagnostics["cache_hit"][0]))
        self.assertTrue(bool(diagnostics["cache_hit"][1]))
        self.assertEqual(diagnostics["unique_plans"], 1)
        self.assertEqual(diagnostics["cache_hits"], 1)

    def test_invalid_start_raises_without_hidden_fallback(self):
        bad = self.start.clone()
        # The canonical origin is inside the inflated socket rectangle.
        bad.tip[0, :2] = torch.tensor((0.0, 0.0))
        with self.assertRaises(AStarPlanError):
            self.planner.reset(
                torch.tensor((True, False, False)),
                start_command=bad,
                goal_command=self.goal,
                socket_mouth=self.mouth,
                socket_axis=self.axis,
            )
        self.assertFalse(bool(self.planner.diagnostics()["plan_found"][0]))

    def test_invalid_cubic_uses_certified_polyline_and_records_fallback(self):
        original = self.planner._natural_cubic_dense

        def leave_workspace(points):
            dense = original(points)
            dense[len(dense) // 2] = np.asarray((0.2, 0.2))
            return dense

        with patch.object(
            self.planner, "_natural_cubic_dense", side_effect=leave_workspace
        ):
            self.planner.reset(
                torch.tensor((False, True, False)),
                start_command=self.start,
                goal_command=self.goal,
                socket_mouth=self.mouth,
                socket_axis=self.axis,
            )
        diagnostics = self.planner.diagnostics()
        self.assertTrue(bool(diagnostics["plan_found"][1]))
        self.assertTrue(bool(diagnostics["spline_fallback"][1]))
        self.assertEqual(diagnostics["spline_fallbacks"], 1)

    def test_chord_validator_rejects_free_endpoints_that_cut_corner(self):
        start = np.asarray((-0.040, 0.0))
        following = np.asarray(((-0.030, 0.070),))
        self.assertTrue(bool(self.planner._points_free(start[None])[0]))
        self.assertTrue(bool(self.planner._points_free(following)[0]))
        self.assertFalse(
            self.planner._execution_chords_free(start, following)
        )

    def test_provenance_is_stable_and_planner_specific(self):
        value = self.planner.provenance()
        self.assertEqual(value["schema"], "screw-astar-spline-planner-v1")
        self.assertEqual(value["search"]["algorithm"], "A*")
        self.assertEqual(value["grid"]["connectivity"], 8)
        self.assertFalse(value["grid"]["diagonal_corner_cutting"])
        self.assertTrue(value["spline"]["executed_chord_validation"])
        self.assertEqual(value["schedule"]["transport_steps"], 190)
        self.assertEqual(value["schedule"]["rotate_raise_steps"], 80)
        self.assertEqual(value["schedule"]["dock_steps"], 80)
        self.assertEqual(self.planner.parameters(), value)

    def test_state_round_trip_restores_hot_loop_tensors(self):
        self._reset_all()
        state = self.planner.state_dict()
        restored = BatchedAStarSplinePlanner(
            self.config, self.worlds, "cpu"
        )
        try:
            restored.load_state_dict(state)
            torch.testing.assert_close(restored._plan_tip, self.planner._plan_tip)
            torch.testing.assert_close(
                restored._plan_rotation, self.planner._plan_rotation
            )
            self.assertTrue(
                torch.equal(restored._plan_found, self.planner._plan_found)
            )
            self.assertEqual(
                restored.diagnostics()["total_reset_plans"], self.worlds
            )
        finally:
            restored.close()


if __name__ == "__main__":
    unittest.main()
