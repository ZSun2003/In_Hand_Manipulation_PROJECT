from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from MJWarp_RL import contract
from MJWarp_RL.curriculum import PHASE
from MJWarp_RL.env import OBSERVATION_DIM, OBSERVATION_SLICES
from MJWarp_RL.hpc_config import DEFAULT_CONFIG
from MJWarp_RL.ik_expert import MJWarpIKExpert
from MJWarp_RL.train import _episode_freeze_certificate


class MJWarpIKExpertTests(unittest.TestCase):
    def _config(self, task: str):
        return replace(DEFAULT_CONFIG, task=task, num_envs=80, device="cpu")

    def _platform(self, task: str):
        shards = []
        for object_index in range(len(contract.OBJECTS)):
            fraction = np.full(
                (
                    len(contract.OBJECTS),
                    contract.RESET_TEMPLATES_PER_OBJECT,
                ),
                np.nan,
            )
            yaw = np.full_like(fraction, np.nan)
            fraction[object_index] = np.linspace(
                contract.GRASP_PLANE_MIN_FRACTION,
                contract.GRASP_PLANE_MAX_FRACTION,
                contract.RESET_TEMPLATES_PER_OBJECT,
            )
            contact = np.full(
                (
                    len(contract.OBJECTS),
                    contract.RESET_TEMPLATES_PER_OBJECT,
                    len(contract.TIP_ORDER),
                    3,
                ),
                np.nan,
            )
            base = fraction[object_index]
            contact[object_index, :, :, 2] = np.stack(
                (
                    base,
                    0.5 + 0.8 * (base - 0.5),
                    0.5 + 0.6 * (base - 0.5),
                    0.5 + 0.4 * (base - 0.5),
                ),
                axis=1,
            )
            yaw[object_index] = np.linspace(
                -0.25, 0.25, contract.RESET_TEMPLATES_PER_OBJECT
            )
            shards.append(
                SimpleNamespace(
                    fixed_object_index=object_index,
                    planned_grasp_plane_fraction=fraction,
                    planned_grasp_plane_yaw_rad=yaw,
                    contact_targets_local=contact,
                    object_local_bottom_z_m=np.zeros(len(contract.OBJECTS)),
                    object_local_top_z_m=np.ones(len(contract.OBJECTS)),
                )
            )
        return SimpleNamespace(task=task, shards=tuple(shards))

    def _expert(self, task: str) -> MJWarpIKExpert:
        config = self._config(task)
        return MJWarpIKExpert(
            config,
            platform_spec=self._platform(task),
            device="cpu",
        )

    def test_frozen_planner_yaw_is_not_limited_like_policy_residual_yaw(
        self,
    ) -> None:
        platform = self._platform(contract.TASK_TRANSLATE)
        # The canonical G6 pear planner selects 135 degrees for reset
        # templates 12 and 23.  This is an absolute anchor orientation, not a
        # policy residual command.
        native_planner_yaw = np.deg2rad(135.0)
        platform.shards[2].planned_grasp_plane_yaw_rad[2, 12] = (
            native_planner_yaw
        )
        expert = MJWarpIKExpert(
            self._config(contract.TASK_TRANSLATE),
            platform_spec=platform,
            device="cpu",
        )
        self.assertAlmostEqual(
            float(expert.planned_grasp_plane_yaw_rad[2, 12]),
            native_planner_yaw,
            places=6,
        )

        platform.shards[2].planned_grasp_plane_yaw_rad[2, 12] = (
            contract.FROZEN_PLANNER_YAW_ABS_LIMIT_RAD + 0.01
        )
        with self.assertRaisesRegex(
            ValueError,
            r"pear template 12: .* rad",
        ):
            MJWarpIKExpert(
                self._config(contract.TASK_TRANSLATE),
                platform_spec=platform,
                device="cpu",
            )

    def _assert_group_frozen(self, expert, goals) -> None:
        for value in goals.values():
            grouped = value.reshape(expert.groups, expert.population)
            torch.testing.assert_close(
                grouped,
                grouped[:, :1].expand_as(grouped),
            )

    def test_translation_goals_and_planned_plane_are_frozen(self) -> None:
        config = self._config(contract.TASK_TRANSLATE)
        expert = self._expert(contract.TASK_TRANSLATE)
        goals = expert.begin_episode(PHASE, 3)

        self._assert_group_frozen(expert, goals)
        self.assertTrue(
            bool(
                (
                    goals["task_index"]
                    == contract.TASKS.index(contract.TASK_TRANSLATE)
                ).all()
            )
        )
        self.assertTrue(bool((goals["path_seed"] > 0).all()))
        self.assertTrue(bool((goals["rotation_angle_deg"] == 0).all()))
        expected_fraction = expert.planned_tip_height_fraction[
            goals["object_index"], goals["template_index"]
        ]
        offset = expected_fraction - contract.GRASP_PLANE_NOMINAL_FRACTION
        expected_action = offset / torch.where(
            offset <= 0.0,
            torch.full_like(
                offset,
                contract.GRASP_PLANE_NOMINAL_FRACTION
                - contract.GRASP_PLANE_MIN_FRACTION,
            ),
            torch.full_like(
                offset,
                contract.GRASP_PLANE_MAX_FRACTION
                - contract.GRASP_PLANE_NOMINAL_FRACTION,
            ),
        )
        torch.testing.assert_close(
            expert.episode_action[:, contract.TIP_HEIGHT_SLICE],
            expected_action,
        )
        self.assertTrue(
            bool((expert.episode_action[:, contract.YAW_INDEX] == 0.0).all())
        )
        self.assertFalse(hasattr(expert, "generator"))
        for row, object_kind in enumerate(contract.OBJECTS):
            action = expert.episode_action[
                goals["object_index"] == row
            ][0].tolist()
            decoded = contract.decode_action(
                action, object_kind
            )
            expected = expected_fraction[
                goals["object_index"] == row
            ][0].tolist()
            for value, expected_value in zip(
                decoded.tip_height_fraction, expected
            ):
                self.assertAlmostEqual(value, expected_value, places=6)
            self.assertEqual(decoded.yaw_rad, 0.0)

    def test_translation_paths_use_seeded_uniform_five_shape_permutations(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            task=contract.TASK_TRANSLATE,
            num_envs=200,
            device="cpu",
        )
        expert = MJWarpIKExpert(
            config,
            platform_spec=self._platform(contract.TASK_TRANSLATE),
            device="cpu",
        )
        full = expert._group_goals(PHASE, 0)
        group_object = full["object_index"][:: expert.population]
        full_shape = full["path_shape_index"][:: expert.population]
        expected = torch.arange(len(contract.TRANSLATION_PATH_SHAPES))
        for object_index in range(len(contract.OBJECTS)):
            observed = full_shape[group_object == object_index].sort().values
            torch.testing.assert_close(observed, expected)

        repeated = expert._group_goals(PHASE, 0)
        torch.testing.assert_close(
            repeated["path_shape_index"], full["path_shape_index"]
        )
        next_episode = expert._group_goals(PHASE, 1)
        self.assertFalse(
            torch.equal(
                next_episode["path_shape_index"],
                full["path_shape_index"],
            )
        )

    def test_rotation_goals_zero_path_fields_and_use_closed_angles(self) -> None:
        config = self._config(contract.TASK_ROTATE)
        expert = self._expert(contract.TASK_ROTATE)
        goals = expert.begin_episode(PHASE, 5)

        self._assert_group_frozen(expert, goals)
        self.assertTrue(
            bool(
                (
                    goals["task_index"]
                    == contract.TASKS.index(contract.TASK_ROTATE)
                ).all()
            )
        )
        self.assertTrue(bool((goals["path_seed"] == 0).all()))
        self.assertTrue(bool((goals["path_shape_index"] == 0).all()))
        angles = goals["rotation_angle_deg"]
        self.assertTrue(
            bool(
                (
                    angles
                    >= contract.ROTATION_ANGLE_MIN_DEG
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    angles
                    <= contract.ROTATION_ANGLE_MAX_DEG
                ).all()
            )
        )

    def test_action_is_deterministic_bounded_tip_error_feedback_only(self) -> None:
        config = self._config(contract.TASK_TRANSLATE)
        expert = self._expert(contract.TASK_TRANSLATE)
        goals = expert.begin_episode(PHASE, 0)
        frozen_goals = {name: value.clone() for name, value in goals.items()}
        observation = torch.zeros((config.num_envs, OBSERVATION_DIM))
        object_slice = OBSERVATION_SLICES["object_one_hot"]
        observation[
            torch.arange(config.num_envs),
            object_slice.start + goals["object_index"],
        ] = 1.0
        phase_slice = OBSERVATION_SLICES["phase_one_hot"]
        observation[:, phase_slice.start] = 1.0
        error = torch.linspace(-2.0, 2.0, 12).repeat(config.num_envs, 1)
        observation[:, OBSERVATION_SLICES["tip_target_error_world"]] = error

        first = expert.action(0, observation)
        second = expert.action(99, observation)

        torch.testing.assert_close(first, second)
        torch.testing.assert_close(
            first[:, contract.TIP_XYZ_RESIDUAL_SLICE],
            error.clamp(-1.0, 1.0),
        )
        inactive = torch.ones(contract.ACTION_DIM, dtype=torch.bool)
        inactive[contract.TIP_XYZ_RESIDUAL_SLICE] = False
        torch.testing.assert_close(
            first[:, inactive], expert.episode_action[:, inactive]
        )
        self.assertLessEqual(float(first.abs().max()), 1.0)
        for name, expected in frozen_goals.items():
            torch.testing.assert_close(expert.goals[name], expected)

    def test_pre_action_feedback_uses_initial_object_frame_and_zero_baseline(self) -> None:
        config = self._config(contract.TASK_TRANSLATE)
        expert = self._expert(contract.TASK_TRANSLATE)
        expert.begin_episode(PHASE, 0)
        observation = torch.zeros((config.num_envs, OBSERVATION_DIM))
        half_sqrt = float(2.0**-0.5)
        observation[:, OBSERVATION_SLICES["initial_quaternion_world"]] = (
            torch.tensor((half_sqrt, 0.0, 0.0, half_sqrt))
        )
        # A stale action-start value must be ignored until the reference is
        # valid.  World +X becomes object-local -Y under a +90-degree yaw.
        observation[:, OBSERVATION_SLICES["action_start_action"]] = 0.75
        world_error = torch.tensor((0.4, 0.0, 0.0)).repeat(4)
        observation[:, OBSERVATION_SLICES["tip_target_error_world"]] = (
            world_error
        )

        result = expert.action(0, observation)
        expected_tip = torch.tensor((0.0, -0.4, 0.0)).repeat(4)
        torch.testing.assert_close(
            result[:, contract.TIP_XYZ_RESIDUAL_SLICE],
            expected_tip[None, :].expand(config.num_envs, -1),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_translation_action_feedback_uses_reference_frame_and_frozen_baseline(self) -> None:
        config = self._config(contract.TASK_TRANSLATE)
        expert = self._expert(contract.TASK_TRANSLATE)
        expert.begin_episode(PHASE, 0)
        observation = torch.zeros((config.num_envs, OBSERVATION_DIM))
        half_sqrt = float(2.0**-0.5)
        observation[:, OBSERVATION_SLICES[
            "action_reference_quaternion_world"
        ]] = torch.tensor((half_sqrt, 0.0, 0.0, half_sqrt))
        observation[:, OBSERVATION_SLICES["action_reference_valid"]] = 1.0
        baseline = torch.tensor((0.10, 0.20, -0.10)).repeat(4)
        observation[:, OBSERVATION_SLICES["action_start_action"]][
            :, contract.TIP_XYZ_RESIDUAL_SLICE
        ] = baseline
        world_error = torch.tensor((0.30, 0.0, 0.0)).repeat(4)
        observation[:, OBSERVATION_SLICES["tip_target_error_world"]] = (
            world_error
        )

        first = expert.action(0, observation)
        observation[:, OBSERVATION_SLICES["previous_action"]] = -0.91
        second = expert.action(1, observation)
        expected_tip = baseline + torch.tensor((0.0, -0.30, 0.0)).repeat(4)
        torch.testing.assert_close(
            first[:, contract.TIP_XYZ_RESIDUAL_SLICE],
            expected_tip[None, :].expand(config.num_envs, -1),
            atol=1.0e-6,
            rtol=0.0,
        )
        # Dynamic commands are reconstructed from the frozen boundary on
        # every tick; previous_action is never accumulated.
        torch.testing.assert_close(first, second)

    def test_rotation_action_feedback_uses_slerped_desired_frame(self) -> None:
        config = self._config(contract.TASK_ROTATE)
        expert = self._expert(contract.TASK_ROTATE)
        expert.begin_episode(PHASE, 0)
        observation = torch.zeros((config.num_envs, OBSERVATION_DIM))
        observation[:, OBSERVATION_SLICES[
            "action_reference_quaternion_world"
        ]] = torch.tensor((1.0, 0.0, 0.0, 0.0))
        observation[:, OBSERVATION_SLICES["goal_quaternion_world"]] = (
            torch.tensor((0.0, 0.0, 0.0, 1.0))
        )
        observation[:, OBSERVATION_SLICES["action_reference_valid"]] = 1.0
        observation[:, OBSERVATION_SLICES["motion_progress"]][:, 1] = 0.5
        baseline = torch.tensor((-0.10, 0.05, 0.20)).repeat(4)
        observation[:, OBSERVATION_SLICES["action_start_action"]][
            :, contract.TIP_XYZ_RESIDUAL_SLICE
        ] = baseline
        world_error = torch.tensor((0.25, 0.0, 0.0)).repeat(4)
        observation[:, OBSERVATION_SLICES["tip_target_error_world"]] = (
            world_error
        )

        result = expert.action(0, observation)
        expected_tip = baseline + torch.tensor((0.0, -0.25, 0.0)).repeat(4)
        torch.testing.assert_close(
            result[:, contract.TIP_XYZ_RESIDUAL_SLICE],
            expected_tip[None, :].expand(config.num_envs, -1),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_first_successful_plane_compliant_replica_wins_without_reward(self) -> None:
        config = self._config(contract.TASK_TRANSLATE)
        expert = self._expert(contract.TASK_TRANSLATE)
        expert.begin_episode(PHASE, 0)
        success = torch.zeros(config.num_envs, dtype=torch.bool)
        compliance = torch.zeros_like(success)
        returns = torch.arange(config.num_envs, dtype=torch.float32)
        for group in range(expert.groups):
            base = group * expert.population
            success[base + 2] = True
            compliance[base + 2] = True
            success[base + 7] = True
            compliance[base + 7] = True
        # The final group has no certified replica and must fall back to slot
        # zero; train.py's second certification check then stores nothing.
        final_base = (expert.groups - 1) * expert.population
        success[final_base : final_base + expert.population] = False
        compliance[final_base : final_base + expert.population] = False

        winner = expert.select_winners(returns, success, compliance)
        expected = (
            torch.arange(expert.groups) * expert.population + 2
        )
        expected[-1] = final_base
        torch.testing.assert_close(winner, expected)

        before = expert.episode_action.clone()
        expert.update_centroids(winner)
        torch.testing.assert_close(expert.episode_action, before)

    def test_clear_episode_state_releases_every_episode_tensor(self) -> None:
        expert = self._expert(contract.TASK_TRANSLATE)
        expert.begin_episode(PHASE, 0)
        expert.clear_episode_state()
        for name in (
            "episode_action",
            "episode_planned_grasp_plane_fraction",
            "episode_planned_grasp_plane_yaw_rad",
            "dynamic_knots",
            "goals",
            "group_object",
            "group_task",
            "group_template",
        ):
            self.assertIsNone(getattr(expert, name))


class EpisodeFreezeCertificateTests(unittest.TestCase):
    worlds = 2
    steps = 3

    def _fixture(self):
        observation = torch.zeros(
            (self.worlds, self.steps + 1, OBSERVATION_DIM),
            dtype=torch.float32,
        )
        endpoint_path = torch.zeros(
            (
                self.worlds,
                contract.TRANSLATION_PATH_WAYPOINTS,
                2,
            )
        )
        endpoint_path[:, :, 0] = torch.linspace(
            0.0, 0.02, contract.TRANSLATION_PATH_WAYPOINTS
        )
        normalized_path = (
            endpoint_path / contract.TRANSLATION_PATH_REGION_RADIUS_M
        ).reshape(self.worlds, -1)
        observation[:, :, OBSERVATION_SLICES[
            "translation_path_waypoints_xy"
        ]] = normalized_path[:, None, :]
        phase_slice = OBSERVATION_SLICES["phase_one_hot"]
        observation[:, 0, phase_slice.start] = 1.0

        episode_action = torch.zeros(self.worlds, contract.ACTION_DIM)
        episode_action[:, contract.TIP_HEIGHT_SLICE] = torch.tensor(
            (-0.5, -0.5, -0.5, -0.5)
        )
        action = torch.zeros(
            self.worlds, self.steps, contract.ACTION_DIM
        )
        indices = list(contract.EPISODE_LEVEL_ACTION_INDICES)
        action[:, 0, indices] = episode_action[:, indices]
        action_start_slice = OBSERVATION_SLICES["action_start_action"]
        observation[:, -1, action_start_slice] = episode_action

        valid_slice = OBSERVATION_SLICES["action_reference_valid"]
        observation[:, 2:, valid_slice] = 1.0
        anchor_slice = OBSERVATION_SLICES[
            "action_start_cage_object_local"
        ]
        anchor = torch.linspace(-0.03, 0.03, 12).repeat(self.worlds, 1)
        observation[:, 2:, anchor_slice] = anchor[:, None, :]

        info = {
            "path_waypoints_xy_m": endpoint_path,
            "task_index": torch.full(
                (self.worlds,),
                contract.TASKS.index(contract.TASK_TRANSLATE),
            ),
            "requested_tip_height_fraction": torch.full(
                (self.worlds, len(contract.TIP_ORDER)), 0.35
            ),
            "grasp_plane_fraction": torch.full(
                (self.worlds,), 0.35
            ),
        }
        planner_yaw = torch.tensor((0.2, -0.2))
        return observation, action, info, episode_action, planner_yaw

    def _certify(self, fixture):
        observation, action, info, episode_action, planner_yaw = fixture
        return _episode_freeze_certificate(
            observation,
            action,
            info,
            episode_action,
            planner_yaw,
            expected_task_index=contract.TASKS.index(
                contract.TASK_TRANSLATE
            ),
            torch=torch,
        )

    def test_accepts_frozen_path_plane_and_four_anchors(self) -> None:
        certificate = self._certify(self._fixture())
        self.assertTrue(bool(certificate["trajectory_frozen"].all()))
        self.assertEqual(
            tuple(
                certificate[
                    "frozen_hard_tip_anchors_object_local_m"
                ].shape
            ),
            (self.worlds, len(contract.TIP_ORDER), 3),
        )
        torch.testing.assert_close(
            certificate["frozen_grasp_plane_yaw_rad"],
            torch.tensor((0.2, -0.2)),
        )

    def test_rejects_path_plane_or_anchor_mutation(self) -> None:
        path_fixture = self._fixture()
        path_fixture[0][0, 1, OBSERVATION_SLICES[
            "translation_path_waypoints_xy"
        ].start] += 0.1
        self.assertFalse(bool(self._certify(path_fixture)["path_frozen"][0]))

        plane_fixture = self._fixture()
        plane_fixture[1][0, 0, contract.TIP_HEIGHT_SLICE.start] += 0.2
        self.assertFalse(
            bool(self._certify(plane_fixture)["plane_action_frozen"][0])
        )

        anchor_fixture = self._fixture()
        anchor_fixture[0][0, 2, OBSERVATION_SLICES[
            "action_start_cage_object_local"
        ].start] += 0.01
        self.assertFalse(
            bool(self._certify(anchor_fixture)["anchors_frozen"][0])
        )


if __name__ == "__main__":
    unittest.main()
