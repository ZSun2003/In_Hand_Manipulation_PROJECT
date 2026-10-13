from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
from unittest import mock

import torch

from MJWarp_RL import contract
from MJWarp_RL.curriculum import PHASE
from MJWarp_RL.ik_expert import MJWarpIKExpert
from MJWarp_RL.env import OBSERVATION_DIM, StepBatch
from MJWarp_RL.hpc_config import (
    DEFAULT_CONFIG,
    balanced_object_world_counts,
    balanced_object_world_slices,
)
from MJWarp_RL.newton_regularization import (
    formal_newton_regularization_metadata,
)
from MJWarp_RL.sharded_env import (
    ShardedMJWarpEnv,
    formal_sharded_health_is_valid,
    ordered_shard_manifest,
)


class _FakeBackend:
    control_dt = 0.02

    def __init__(self, worlds: int) -> None:
        self.worlds = worlds
        self._graph_ready = False
        self._fields = {
            "qpos": torch.zeros((worlds, 3)),
            "qvel": torch.zeros((worlds, 2)),
            "time": torch.zeros(worlds),
        }

    def prepare_cuda_graph(self) -> None:
        self._graph_ready = True

    def has_field(self, name: str) -> bool:
        return name in self._fields

    def tensor(self, name: str) -> torch.Tensor:
        return self._fields[name]

    def health(self) -> dict[str, object]:
        return {
            **formal_newton_regularization_metadata(),
            "finite": True,
            "model_nv": 26,
            "solver_option": 2,
            "solver_name": contract.FORMAL_SOLVER,
            "requested_solver_tolerance": (
                contract.FORMAL_XML_SOLVER_TOLERANCE
            ),
            "effective_solver_tolerance": (
                contract.FORMAL_EFFECTIVE_SOLVER_TOLERANCE
            ),
            "jacobian_option": 0,
            "resolved_sparse_jacobian": False,
            "sticky_nonfinite": 0,
            "sticky_overflow": 0,
            "sticky_contacts_full": 0,
            "sticky_constraints_full": 0,
            "sticky_external_nonfinite": 0,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value in self._fields.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, value in state.items():
            self._fields[name].copy_(value)


class _FakeChildEnv:
    def __init__(
        self,
        *,
        fixed_object: int,
        worlds: int,
        platform_spec: object,
        **_: object,
    ) -> None:
        self.fixed_object = int(fixed_object)
        self.worlds = int(worlds)
        self.platform = platform_spec
        self.torch = torch
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.backend = _FakeBackend(self.worlds)
        self.object_index = torch.full(
            (self.worlds,), self.fixed_object, dtype=torch.long
        )
        self.closed = False
        self.last_action: torch.Tensor | None = None
        self.observation = torch.zeros((self.worlds, OBSERVATION_DIM))

    def reset(
        self,
        mask: torch.Tensor | None = None,
        goals: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        selected = (
            torch.ones(self.worlds, dtype=torch.bool)
            if mask is None
            else mask.to(torch.bool)
        )
        if goals is not None and "object_index" in goals and bool(
            selected.any()
        ):
            values = torch.as_tensor(goals["object_index"]).reshape(-1)
            if values.numel() == self.worlds:
                values = values[selected]
            if values.numel() and not bool(
                (values == self.fixed_object).all()
            ):
                raise ValueError("fake child received a cross-object goal")
        self.observation[:, 0] = float(self.fixed_object)
        return self.observation

    def step(self, action: torch.Tensor) -> StepBatch:
        self.last_action = action
        self.backend.tensor("time").add_(self.backend.control_dt)
        return StepBatch(
            observation=self.observation,
            reward=torch.full((self.worlds,), float(self.fixed_object)),
            done=torch.zeros(self.worlds, dtype=torch.bool),
            info={
                "success": torch.zeros(self.worlds, dtype=torch.bool),
                "object_index": self.object_index,
                "uniform_scalar": 7,
            },
        )

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {"object_index": self.object_index}

    def close(self) -> None:
        self.closed = True


def _platform(
    task: str = contract.TASK_TRANSLATE,
    *,
    platform: str = "gripper3",
) -> SimpleNamespace:
    task = contract.validate_task(task)
    task_index = contract.TASKS.index(task)
    root_height_m = {
        ("gripper3", contract.TASK_TRANSLATE): 0.525,
        ("gripper3", contract.TASK_ROTATE): 0.525,
        ("gripper6", contract.TASK_TRANSLATE): 0.525,
        ("gripper6", contract.TASK_ROTATE): 0.525,
    }[(platform, task)]

    def contact_bank(index: int):
        bank = torch.full(
            (
                len(contract.OBJECTS),
                contract.RESET_TEMPLATES_PER_OBJECT,
                len(contract.TIP_ORDER),
                3,
            ),
            float("nan"),
        )
        bank[index, :, :, 2] = 0.5
        return bank.numpy()

    shards = tuple(
        SimpleNamespace(
            platform=platform,
            task=task,
            task_index=task_index,
            root_height_m=root_height_m,
            fixed_object_index=index,
            model_xml=Path(f"/fake/{name}.xml"),
            model_xml_sha256=f"{index + 1:x}" * 64,
            host_model=SimpleNamespace(nq=27, nv=26, nu=12),
            planned_grasp_plane_fraction=torch.full(
                (
                    len(contract.OBJECTS),
                    contract.RESET_TEMPLATES_PER_OBJECT,
                ),
                float("nan"),
            ).index_fill(0, torch.tensor([index]), 0.5).numpy(),
            planned_grasp_plane_yaw_rad=torch.full(
                (
                    len(contract.OBJECTS),
                    contract.RESET_TEMPLATES_PER_OBJECT,
                ),
                float("nan"),
            ).index_fill(0, torch.tensor([index]), 0.0).numpy(),
            contact_targets_local=contact_bank(index),
            object_local_bottom_z_m=torch.zeros(
                len(contract.OBJECTS)
            ).numpy(),
            object_local_top_z_m=torch.ones(
                len(contract.OBJECTS)
            ).numpy(),
        )
        for index, name in enumerate(contract.OBJECTS)
    )
    return SimpleNamespace(
        platform=platform,
        task=task,
        task_index=task_index,
        root_height_m=root_height_m,
        parent_model_fingerprint_sha256="a" * 64,
        parent_controller_fingerprint_sha256="b" * 64,
        platform_adapter_fingerprint_sha256="c" * 64,
        shards=shards,
    )


class ObjectShardAllocationTests(unittest.TestCase):
    def test_formal_and_arbitrary_allocations_are_balanced_and_stable(self) -> None:
        self.assertEqual(
            balanced_object_world_counts(20_000, population=10),
            contract.FORMAL_OBJECT_WORLD_COUNTS,
        )
        self.assertEqual(contract.FORMAL_OBJECT_WORLD_COUNTS, (5_000,) * 4)
        self.assertEqual(DEFAULT_CONFIG.control_ticks, 3_000)
        self.assertEqual(DEFAULT_CONFIG.remainder_worlds, 0)
        self.assertEqual(
            DEFAULT_CONFIG.remainder_object_world_counts,
            (0,) * 4,
        )
        self.assertEqual(
            balanced_object_world_counts(17), (5, 4, 4, 4)
        )
        slices = balanced_object_world_slices(17)
        self.assertEqual(
            [(value.start, value.stop) for value in slices],
            [(0, 5), (5, 9), (9, 13), (13, 17)],
        )

    def test_ik_expert_populations_are_object_sharded_and_single_task(self) -> None:
        for task in contract.TASKS:
            config = replace(DEFAULT_CONFIG, task=task, device="cpu")
            expert = MJWarpIKExpert(
                config,
                platform_spec=_platform(task),
                device="cpu",
            )
            goals = expert._group_goals(PHASE, 0)
            objects = goals["object_index"].reshape(
                -1, config.elite.population
            )
            self.assertTrue(bool((objects == objects[:, :1]).all()))
            expected_group_objects = torch.repeat_interleave(
                torch.arange(len(contract.OBJECTS)),
                torch.tensor(config.object_group_counts),
            )
            self.assertTrue(torch.equal(objects[:, 0], expected_group_objects))
            self.assertTrue(
                bool(
                    (
                        goals["task_index"]
                        == contract.TASKS.index(task)
                    ).all()
                )
            )
            if task == contract.TASK_TRANSLATE:
                self.assertTrue(bool((goals["path_seed"] > 0).all()))
                self.assertTrue(bool((goals["rotation_angle_deg"] == 0).all()))
                continue
            full_goals = expert._group_goals(PHASE, 0)
            self.assertTrue(bool((full_goals["path_seed"] == 0).all()))
            angles = full_goals["rotation_angle_deg"]
            self.assertEqual(int(angles.min()), 10)
            self.assertEqual(int(angles.max()), 45)
            self.assertEqual(set(full_goals["source_region"].tolist()), {0})


class ShardedEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(DEFAULT_CONFIG, device="cpu")
        self.platform = _platform()
        self.environment = ShardedMJWarpEnv(
            curriculum_phase=PHASE,
            config=self.config,
            platform_spec=self.platform,
            worlds=40,
            device="cpu",
            require_linux=False,
            require_cuda=False,
            validate_official_config=False,
            child_env_type=_FakeChildEnv,
        )

    def tearDown(self) -> None:
        self.environment.close()

    def test_global_actions_are_object_major_views_and_outputs_are_stable(
        self,
    ) -> None:
        self.assertEqual(self.environment.world_counts, (10,) * 4)
        goals = torch.repeat_interleave(torch.arange(4), 10)
        observation = self.environment.reset(
            goals={"object_index": goals}
        )
        self.assertTrue(torch.equal(observation[:, 0].to(torch.long), goals))
        action = torch.arange(
            40 * contract.ACTION_DIM, dtype=torch.float32
        ).reshape(40, contract.ACTION_DIM)
        first = self.environment.step(action)
        first_pointers = (
            first.observation.data_ptr(),
            first.reward.data_ptr(),
            first.done.data_ptr(),
            first.info["object_index"].data_ptr(),
        )
        storage_pointer = action.untyped_storage().data_ptr()
        for shard in self.environment.shards:
            self.assertIsNotNone(shard.last_action)
            self.assertEqual(
                shard.last_action.untyped_storage().data_ptr(),
                storage_pointer,
            )
        second = self.environment.step(action)
        self.assertEqual(
            first_pointers,
            (
                second.observation.data_ptr(),
                second.reward.data_ptr(),
                second.done.data_ptr(),
                second.info["object_index"].data_ptr(),
            ),
        )

    def test_goal_routing_is_fail_closed_but_ignores_unselected_shards(
        self,
    ) -> None:
        bad = torch.repeat_interleave(torch.arange(4), 10)
        bad[0] = 1
        with self.assertRaisesRegex(ValueError, "non-orange"):
            self.environment.reset(goals={"object_index": bad})

        mask = torch.zeros(40, dtype=torch.bool)
        mask[10:20] = True
        # A scalar banana goal is valid for the selected banana shard and is
        # not consumed by the other three zero-selection children.
        self.environment.reset(
            mask=mask,
            goals={"object_index": torch.tensor(1)},
        )

    def test_four_graphs_health_state_and_manifest_are_aggregate(self) -> None:
        self.environment.backend.prepare_cuda_graph()
        self.assertTrue(self.environment.backend._graph_ready)
        self.assertTrue(
            all(shard.backend._graph_ready for shard in self.environment.shards)
        )
        health = self.environment.backend.health()
        self.assertTrue(
            formal_sharded_health_is_valid(
                health, expected_world_counts=(10,) * 4
            )
        )
        sorted_round_trip = json.loads(
            json.dumps(health, sort_keys=True)
        )
        self.assertTrue(
            formal_sharded_health_is_valid(
                sorted_round_trip, expected_world_counts=(10,) * 4
            )
        )
        wrong_world_total = dict(sorted_round_trip)
        wrong_world_total["worlds"] = 39
        self.assertFalse(
            formal_sharded_health_is_valid(
                wrong_world_total, expected_world_counts=(10,) * 4
            )
        )
        state = self.environment.backend.state_dict()
        self.assertEqual(tuple(state["qpos"].shape), (40, 3))
        manifest = ordered_shard_manifest(
            self.platform, self.environment.world_counts
        )
        self.assertEqual(
            [entry["object"] for entry in manifest],
            list(contract.OBJECTS),
        )
        self.assertEqual([entry["worlds"] for entry in manifest], [10] * 4)
        self.environment.shards[0].backend._fields["qpos"] = torch.zeros(
            (1, 3)
        )
        with self.assertRaisesRegex(ValueError, "not world-major"):
            self.environment.backend.tensor("qpos")

    def test_child_constructor_oom_closes_current_and_all_prior_children(
        self,
    ) -> None:
        class PartiallyFailingChild:
            instances: list["PartiallyFailingChild"] = []

            def __init__(self, *, fixed_object: int, **_: object) -> None:
                self.fixed_object = int(fixed_object)
                self.close_calls = 0
                self.__class__.instances.append(self)
                if self.fixed_object == 2:
                    raise MemoryError("child constructor OOM")

            def close(self) -> None:
                self.close_calls += 1
                if self.fixed_object == 1:
                    raise RuntimeError("mock child close failure")

        with self.assertRaisesRegex(MemoryError, "child constructor OOM"):
            ShardedMJWarpEnv(
                curriculum_phase=PHASE,
                config=self.config,
                platform_spec=self.platform,
                worlds=40,
                device="cpu",
                require_linux=False,
                require_cuda=False,
                validate_official_config=False,
                child_env_type=PartiallyFailingChild,
            )
        self.assertEqual(len(PartiallyFailingChild.instances), 3)
        self.assertEqual(
            [child.close_calls for child in PartiallyFailingChild.instances],
            [1, 1, 1],
        )

    def test_aggregate_constructor_oom_closes_all_despite_close_errors(
        self,
    ) -> None:
        class TrackingChild(_FakeChildEnv):
            instances: list["TrackingChild"] = []

            def __init__(self, **kwargs: object) -> None:
                super().__init__(**kwargs)
                self.close_calls = 0
                self.__class__.instances.append(self)

            def close(self) -> None:
                self.close_calls += 1
                super().close()
                if self.fixed_object == 3:
                    raise RuntimeError("mock aggregate close failure")

        with mock.patch(
            "MJWarp_RL.sharded_env.ShardedBackendFacade",
            side_effect=MemoryError("aggregate constructor OOM"),
        ):
            with self.assertRaisesRegex(
                MemoryError, "aggregate constructor OOM"
            ):
                ShardedMJWarpEnv(
                    curriculum_phase=PHASE,
                    config=self.config,
                    platform_spec=self.platform,
                    worlds=40,
                    device="cpu",
                    require_linux=False,
                    require_cuda=False,
                    validate_official_config=False,
                    child_env_type=TrackingChild,
                )
        self.assertEqual(len(TrackingChild.instances), len(contract.OBJECTS))
        self.assertTrue(
            all(child.close_calls == 1 for child in TrackingChild.instances)
        )
        self.assertTrue(all(child.closed for child in TrackingChild.instances))


if __name__ == "__main__":
    unittest.main()
