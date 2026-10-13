from __future__ import annotations

import unittest

import numpy as np
import torch

from kinematics import GripperIK
from screw_config import DEFAULT_CONFIG
from screw_control import (
    CAPTURE_GATE_STEP,
    FINGERS,
    PREINSERT_GATE_STEP,
    ScrewExpert,
    THREAD_GATE_STEP,
    THREAD_REGRASP_STEP,
)
from s_hpc.analytic_ik import BatchedAnalyticIK
from s_hpc.env import MJWarpScrewEnv
from s_hpc.supervisor import BatchedSupervisor


class BatchedAnalyticIKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = DEFAULT_CONFIG
        cls.expert = ScrewExpert(
            spawn_xy_m=c.default_screw_xy_m,
            table_top_z_m=c.table_top_z_m,
            socket_mouth_world_m=c.socket_mouth_position_m,
            socket_axis_inward_world=c.socket_axis_world,
            contact_distance_m=c.continuous_grasp_distance_m,
            engage_depth_m=c.thread_engagement_depth_m,
            capture_depth_m=c.capture_depth_m,
            final_depth_m=c.target_insertion_depth_m,
            pitch_m=c.thread_pitch_m,
            target_turns=c.target_turns,
            grasp_height_from_tip_m=c.screw_grasp_height_from_tip_m,
            preinsert_standoff_m=c.preinsert_standoff_m,
        )
        cls.batched = BatchedAnalyticIK(device="cpu", linear_samples=51)
        cls.oracle = GripperIK.from_xml()
        cls.supervisor = BatchedSupervisor(device="cpu")

    def _command_at(self, step):
        command = self.expert.reset()
        for _ in range(step):
            command = self.expert.advance()
        return command

    def test_matches_cpu_oracle_across_all_task_modes(self):
        previous = np.zeros(16, dtype=np.float64)
        for step in (
            0,
            65,
            115,
            125,
            175,
            176,
            PREINSERT_GATE_STEP,
            PREINSERT_GATE_STEP + 1,
            CAPTURE_GATE_STEP,
            CAPTURE_GATE_STEP + 1,
            CAPTURE_GATE_STEP + 65,
            THREAD_GATE_STEP,
            THREAD_GATE_STEP + 1,
            THREAD_REGRASP_STEP,
            THREAD_REGRASP_STEP + 1,
            1000,
            1500,
            2000,
            2400,
            2665,
        ):
            command = self._command_at(step)
            grasp_height = float(
                self.supervisor.grasp_height_at(
                    torch.tensor((step,))
                )[0]
            )
            targets = command.fingertip_targets(
                grasp_height_from_tip_m=grasp_height
            )
            ordered = np.stack([targets[finger] for finger in FINGERS])[None]
            gpu = self.batched.solve(
                torch.as_tensor(ordered, dtype=torch.float32),
                torch.as_tensor(previous[None], dtype=torch.float32),
            )
            cpu = self.oracle.solve_tip_targets(targets, prev_ctrl=previous)
            self.assertEqual(bool(gpu.success[0]), cpu.success, step)
            if cpu.success:
                np.testing.assert_allclose(
                    gpu.ctrl.numpy()[0], cpu.ctrl, atol=1.0e-4, rtol=0.0
                )
                previous = gpu.ctrl.numpy()[0]

    def test_unreachable_world_falls_back_without_poisoning_batch(self):
        command = self._command_at(0)
        targets = command.fingertip_targets(
            grasp_height_from_tip_m=DEFAULT_CONFIG.screw_grasp_height_from_tip_m
        )
        ordered = np.stack([targets[finger] for finger in FINGERS])
        batch = np.stack((ordered, ordered))
        batch[1, 0] = np.array((5.0, 5.0, 5.0))
        previous = torch.zeros((2, 16))
        result = self.batched.solve(torch.tensor(batch), previous)
        self.assertTrue(bool(result.success[0]))
        self.assertFalse(bool(result.success[1]))
        # Only the failed U finger retains its four previous controls; the
        # environment discards the whole-world candidate when success=False.
        self.assertTrue(
            torch.equal(result.ctrl[1, (9, 11, 10, 12)], previous[1, (9, 11, 10, 12)])
        )
        self.assertTrue(torch.isfinite(result.ctrl[1]).all())

    def test_fixed_45_full_trajectory_reaches_thirteen_key_azimuths(self):
        degrees = torch.tensor(
            (
                0.0,
                45.0,
                73.7,
                89.9,
                90.0,
                135.0,
                161.2,
                180.0,
                225.0,
                270.0,
                287.4,
                315.0,
                359.9,
            )
        )
        yaw = torch.deg2rad(degrees)
        mouth = 0.020 * torch.stack(
            (torch.cos(yaw), torch.sin(yaw)), dim=-1
        )
        canonical_spawn = torch.tensor((-0.070, 0.015))
        rotation = torch.stack(
            (
                torch.stack((torch.cos(yaw), -torch.sin(yaw)), dim=-1),
                torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1),
            ),
            dim=-2,
        )
        spawn = torch.einsum(
            "bij,j->bi", rotation, canonical_spawn
        )
        socket_angle = torch.full_like(degrees, 45.0)
        previous = torch.zeros((degrees.numel(), 16))
        feasible = torch.ones(degrees.numel(), dtype=torch.bool)
        for step_index in range(self.supervisor.length):
            command = self.supervisor.at(
                torch.full(
                    (degrees.numel(),), step_index, dtype=torch.long
                ),
                spawn,
                socket_angle,
                mouth,
            )
            result = self.batched.solve(
                command.fingertip_targets(
                    self.supervisor.grasp_height_at(
                        torch.full(
                            (degrees.numel(),),
                            step_index,
                            dtype=torch.long,
                        )
                    )
                ),
                previous,
            )
            valid = result.success & torch.isfinite(result.ctrl).all(
                dim=1
            )
            feasible &= valid
            previous = torch.where(
                valid[:, None], result.ctrl, previous
            )
        self.assertTrue(
            bool(feasible.all()),
            f"failed azimuths: {degrees[~feasible].tolist()}",
        )

    def test_h100_prefix_certifies_boundary_and_interior_spawns(self):
        # Retain one H100-qualified 99 mm boundary sample and one interior
        # sample.  Other boundary draws remain eligible for normal reset-time
        # rejection without shrinking the configured 100 mm distribution.
        spawn = torch.tensor(
            (
                (-0.08852126449346542, 0.044328153133392334),
                (-0.07478377968072891, -0.010381440632045269),
            )
        )
        mouth = torch.tensor(
            (
                (0.008947346359491348, 0.01788700744509697),
                (0.010893166065216064, -0.01677316054701805),
            )
        )
        env = object.__new__(MJWarpScrewEnv)
        env.device = torch.device("cpu")
        env.supervisor = self.supervisor
        env.ik = self.batched
        feasible, command, ctrl, passive = (
            env._certify_scripted_prefix(
                spawn,
                torch.full((2,), 45.0),
                mouth,
            )
        )
        self.assertEqual(feasible.tolist(), [True, True])
        self.assertEqual(command.tip.shape, (2, 3))
        self.assertTrue(torch.isfinite(ctrl).all())
        self.assertTrue(torch.isfinite(passive).all())


if __name__ == "__main__":
    unittest.main()
