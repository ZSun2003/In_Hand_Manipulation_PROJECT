from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from MJWarp_RL import contract, success
from MJWarp_RL.env import MJWarpIKResidualEnv
from MJWarp_RL.evaluate import SCHEMA as EVALUATION_SCHEMA
from MJWarp_RL.evaluate import _success_tolerances


class PerAxisSuccessTests(unittest.TestCase):
    def test_public_metadata_names_the_axis_semantics(self) -> None:
        self.assertEqual(
            EVALUATION_SCHEMA,
            "mjwarp-task-bound-paired-ik-diffusion-eval-v6",
        )
        self.assertEqual(
            _success_tolerances(),
            {
                "translate": {
                    "position_axis_abs_tolerance_m": 0.010,
                    "euler_axis_abs_tolerance_deg": 20.0,
                },
                "rotate": {
                    "position_axis_abs_tolerance_m": 0.010,
                    "euler_axis_abs_tolerance_deg": 10.0,
                },
            },
        )


    def test_public_classifier_is_a_box_not_an_l2_ball(self) -> None:
        position = (0.010, -0.010, 0.010)
        euler = (10.0, -10.0, 10.0)
        self.assertGreater(
            math.sqrt(sum(value * value for value in position)),
            contract.ROTATION_POSITION_AXIS_ABS_TOLERANCE_M,
        )
        self.assertGreater(
            math.sqrt(sum(value * value for value in euler)),
            contract.ROTATION_EULER_AXIS_ABS_TOLERANCE_DEG,
        )
        self.assertTrue(
            success.endpoint_success("orange", position, euler, "rotate")
        )
        self.assertFalse(
            success.endpoint_success(
                "orange", (0.010001, 0.0, 0.0), euler, "rotate"
            )
        )
        self.assertFalse(
            success.endpoint_success(
                "orange", position, (0.0, -10.001, 0.0), "rotate"
            )
        )

    def test_vector_environment_uses_the_same_closed_axis_bounds(self) -> None:
        import torch

        worlds = 2
        xipos = torch.zeros((worlds, 1, 3))
        xquat = torch.zeros((worlds, 1, 4))
        xquat[:, :, 0] = 1.0

        class FakeBackend:
            def tensor(self, name: str) -> torch.Tensor:
                return {"xipos": xipos, "xquat": xquat}[name]

        environment = MJWarpIKResidualEnv.__new__(MJWarpIKResidualEnv)
        environment.backend = FakeBackend()
        environment._rows = torch.arange(worlds)
        environment._object_body_ids = torch.zeros(1, dtype=torch.long)
        environment.object_index = torch.zeros(worlds, dtype=torch.long)
        environment.task_index = torch.tensor(
            [
                contract.TASKS.index(contract.TASK_TRANSLATE),
                contract.TASKS.index(contract.TASK_ROTATE),
            ]
        )
        environment.goal_pos = torch.tensor(
            ((0.010, -0.010, 0.010), (0.010, -0.010, 0.010))
        )
        environment.goal_quat = torch.zeros((worlds, 4))
        environment.goal_quat[:, 0] = 1.0

        position_l2, euler_l2, passed, euler_xyz = (
            environment._pose_metrics()
        )
        self.assertTrue(bool(passed.all()))
        self.assertTrue(
            bool(
                (
                    position_l2
                    > contract.ROTATION_POSITION_AXIS_ABS_TOLERANCE_M
                ).all()
            )
        )
        self.assertTrue(bool((euler_l2 == 0.0).all()))
        self.assertTrue(bool((euler_xyz == 0.0).all()))

        environment.goal_pos[1, 2] = 0.010001
        _, _, passed, _ = environment._pose_metrics()
        self.assertTrue(bool(passed[0]))
        self.assertFalse(bool(passed[1]))


if __name__ == "__main__":
    unittest.main()
