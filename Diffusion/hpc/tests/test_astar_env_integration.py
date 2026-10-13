from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from screw_control import TaskPhase
from s_hpc.env import MJWarpScrewEnv, _select_ik_routes
from s_hpc.supervisor import BatchedSupervisor


class _Planner:
    def __init__(self, command=None, reset_error=None):
        self.planned_command = command
        self.reset_error = reset_error
        self.reset_calls = []
        self.command_calls = []

    def reset(
        self,
        mask,
        *,
        start_command,
        goal_command,
        socket_mouth,
        socket_axis,
    ):
        self.reset_calls.append(
            (
                mask.clone(),
                start_command.clone(),
                goal_command.clone(),
                socket_mouth.clone(),
                socket_axis.clone(),
            )
        )
        if self.reset_error is not None:
            raise self.reset_error

    def command(self, anchor, *, next_step):
        self.command_calls.append((anchor.clone(), next_step.clone()))
        return self.planned_command.clone()


class AStarEnvironmentIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.supervisor = BatchedSupervisor(device="cpu")
        self.spawn = torch.tensor(
            ((-0.075, -0.040), (0.060, 0.055)), dtype=torch.float32
        )
        self.socket_angle = self.supervisor.fixed_angle_batch(2)
        _frame, axis, mouth, _aligned = self.supervisor.socket_geometry(
            self.socket_angle
        )
        self.socket_mouth = mouth
        self.socket_axis = axis

    def _bare_env(self, planner):
        env = object.__new__(MJWarpScrewEnv)
        env.baseline = "astar_spline"
        env.nominal_planner = planner
        env.supervisor = self.supervisor
        env.worlds = 2
        env.device = torch.device("cpu")
        env.state = SimpleNamespace(
            spawn_xy=self.spawn,
            socket_angle_deg=self.socket_angle,
            socket_mouth=self.socket_mouth,
            socket_axis=self.socket_axis,
        )
        return env

    def test_masked_reset_receives_exact_supervisor_endpoints(self):
        planner = _Planner()
        env = self._bare_env(planner)
        mask = torch.tensor((False, True))

        env._reset_nominal_planner(mask)

        self.assertEqual(len(planner.reset_calls), 1)
        actual_mask, handoff, preinsert, mouth, axis = planner.reset_calls[0]
        torch.testing.assert_close(actual_mask, mask)
        torch.testing.assert_close(mouth, self.socket_mouth)
        torch.testing.assert_close(axis, self.socket_axis)
        handoff_step = torch.full(
            (2,), self.supervisor.diffusion_handoff_step, dtype=torch.long
        )
        preinsert_step = torch.full(
            (2,), self.supervisor.preinsert_gate_step, dtype=torch.long
        )
        expected_handoff = self.supervisor.at(
            handoff_step,
            self.spawn,
            self.socket_angle,
            self.socket_mouth[:, :2],
        )
        expected_preinsert = self.supervisor.at(
            preinsert_step,
            self.spawn,
            self.socket_angle,
            self.socket_mouth[:, :2],
        )
        torch.testing.assert_close(handoff.tip, expected_handoff.tip)
        torch.testing.assert_close(handoff.rotation, expected_handoff.rotation)
        torch.testing.assert_close(preinsert.tip, expected_preinsert.tip)
        torch.testing.assert_close(
            preinsert.rotation, expected_preinsert.rotation
        )
        self.assertEqual(
            handoff.phase.tolist(),
            [int(TaskPhase.LIFT), int(TaskPhase.LIFT)],
        )
        self.assertEqual(
            preinsert.phase.tolist(),
            [int(TaskPhase.ALIGN), int(TaskPhase.ALIGN)],
        )

    def test_planning_failure_is_not_replaced_by_a_straight_plan(self):
        planner = _Planner(reset_error=RuntimeError("no A* path"))
        env = self._bare_env(planner)

        with self.assertRaisesRegex(RuntimeError, "no A\* path"):
            env._reset_nominal_planner(torch.ones(2, dtype=torch.bool))

    def test_only_align_rows_use_planner_command(self):
        steps = torch.tensor((200, 600), dtype=torch.long)
        anchor = self.supervisor.at(
            steps,
            self.spawn,
            self.socket_angle,
            self.socket_mouth[:, :2],
        )
        planned = anchor.clone()
        planned.tip += torch.tensor((0.001, -0.002, 0.003))
        planned.phase.fill_(int(TaskPhase.COMPLETE))
        planned.engaged.logical_not_()
        planner = _Planner(command=planned)
        env = self._bare_env(planner)

        selected, active = env._astar_nominal_command(
            anchor,
            steps,
            torch.ones(2, dtype=torch.bool),
        )

        torch.testing.assert_close(active, torch.tensor((True, False)))
        torch.testing.assert_close(selected.tip[0], planned.tip[0])
        torch.testing.assert_close(selected.tip[1], anchor.tip[1])
        torch.testing.assert_close(selected.phase, anchor.phase)
        torch.testing.assert_close(selected.engaged, anchor.engaged)
        self.assertEqual(len(planner.command_calls), 1)

    def test_non_align_batch_does_not_call_planner(self):
        steps = torch.tensor((600, 800), dtype=torch.long)
        anchor = self.supervisor.at(
            steps,
            self.spawn,
            self.socket_angle,
            self.socket_mouth[:, :2],
        )
        planner = _Planner(command=anchor)
        env = self._bare_env(planner)

        selected, active = env._astar_nominal_command(
            anchor,
            steps,
            torch.ones(2, dtype=torch.bool),
        )

        self.assertFalse(bool(active.any()))
        torch.testing.assert_close(selected.tip, anchor.tip)
        self.assertEqual(planner.command_calls, [])

    def test_measured_safety_guard_returns_align_row_to_anchor(self):
        steps = torch.tensor((200, 250), dtype=torch.long)
        anchor = self.supervisor.at(
            steps,
            self.spawn,
            self.socket_angle,
            self.socket_mouth[:, :2],
        )
        planned = anchor.clone()
        planned.tip += torch.tensor((0.004, -0.003, 0.002))
        planner = _Planner(command=planned)
        env = self._bare_env(planner)
        guard = torch.tensor((True, False))

        selected, active = env._astar_nominal_command(
            anchor,
            steps,
            torch.ones(2, dtype=torch.bool),
            safety_guard=guard,
        )

        torch.testing.assert_close(active, torch.tensor((False, True)))
        torch.testing.assert_close(selected.tip[0], anchor.tip[0])
        torch.testing.assert_close(selected.tip[1], planned.tip[1])
        self.assertEqual(len(planner.command_calls), 1)

    def test_all_guarded_align_rows_do_not_query_planner(self):
        steps = torch.tensor((200, 250), dtype=torch.long)
        anchor = self.supervisor.at(
            steps,
            self.spawn,
            self.socket_angle,
            self.socket_mouth[:, :2],
        )
        planner = _Planner(command=anchor)
        env = self._bare_env(planner)

        selected, active = env._astar_nominal_command(
            anchor,
            steps,
            torch.ones(2, dtype=torch.bool),
            safety_guard=torch.ones(2, dtype=torch.bool),
        )

        self.assertFalse(bool(active.any()))
        torch.testing.assert_close(selected.tip, anchor.tip)
        self.assertEqual(planner.command_calls, [])

    def test_planner_ik_failure_cannot_use_anchor_ik(self):
        candidate_success = torch.tensor((False, False, True))
        anchor_success = torch.tensor((True, True, False))
        planner_active = torch.tensor((True, False, True))

        candidate, anchor, stop, planner_failure = _select_ik_routes(
            candidate_success, anchor_success, planner_active
        )

        torch.testing.assert_close(candidate, torch.tensor((False, False, True)))
        torch.testing.assert_close(anchor, torch.tensor((False, True, False)))
        torch.testing.assert_close(stop, torch.tensor((True, False, False)))
        torch.testing.assert_close(
            planner_failure, torch.tensor((True, False, False))
        )

    def test_astar_baseline_has_no_actuator_residual_authority(self):
        env = object.__new__(MJWarpScrewEnv)
        active = torch.tensor((True, True, False))
        guard = torch.tensor((False, True, False))
        candidate = torch.tensor((True, True, True))
        stop = torch.tensor((False, False, False))

        env.baseline = "astar_spline"
        astar = env._residual_authority(active, guard, candidate, stop)
        torch.testing.assert_close(astar, torch.zeros(3, dtype=torch.bool))

        env.baseline = "diffusion"
        diffusion = env._residual_authority(
            active, guard, candidate, stop
        )
        torch.testing.assert_close(
            diffusion, torch.tensor((True, False, False))
        )


if __name__ == "__main__":
    unittest.main()
