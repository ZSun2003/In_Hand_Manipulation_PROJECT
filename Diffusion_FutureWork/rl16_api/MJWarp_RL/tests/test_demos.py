from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from MJWarp_RL import contract
from MJWarp_RL.demos import (
    CertifiedIKDemoStore,
    DATA_FILES,
    DEMO_FORMAT,
    DemoWindowSampler,
)
from MJWarp_RL.env import OBSERVATION_DIM
from MJWarp_RL.hpc_config import DEFAULT_CONFIG
from MJWarp_RL.train import _validate_demo_distribution


class CertifiedIKDemoStoreTests(unittest.TestCase):
    steps = 3
    episodes = 4
    provenance = {"test": "certified-ik-demo-v5"}

    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        observations = np.zeros(
            (self.episodes, self.steps + 1, OBSERVATION_DIM),
            dtype=np.float32,
        )
        for episode in range(self.episodes):
            observations[episode].fill(float(episode))
        actions = np.zeros(
            (self.episodes, self.steps, contract.ACTION_DIM),
            dtype=np.float32,
        )
        return observations, actions

    def _metadata(self) -> dict[str, np.ndarray]:
        normalized_plane = np.asarray((-1.0, -0.5, 0.5, 1.0), dtype=np.float32)
        frozen_episode_action = np.zeros(
            (self.episodes, contract.ACTION_DIM), dtype=np.float32
        )
        frozen_episode_action[:, contract.TIP_HEIGHT_SLICE] = (
            normalized_plane[:, None]
        )
        planned_fraction = (
            contract.GRASP_PLANE_NOMINAL_FRACTION
            + normalized_plane
            * np.where(
                normalized_plane <= 0.0,
                contract.GRASP_PLANE_NOMINAL_FRACTION
                - contract.GRASP_PLANE_MIN_FRACTION,
                contract.GRASP_PLANE_MAX_FRACTION
                - contract.GRASP_PLANE_NOMINAL_FRACTION,
            )
        )
        frozen_path = np.zeros(
            (
                self.episodes,
                contract.TRANSLATION_PATH_WAYPOINTS,
                2,
            ),
            dtype=np.float32,
        )
        frozen_path[0] = np.asarray(
            contract.TRANSLATION_PATH_WAYPOINTS_XY_M[1], dtype=np.float32
        )
        frozen_path[2] = np.asarray(
            contract.TRANSLATION_PATH_WAYPOINTS_XY_M[4], dtype=np.float32
        )
        return {
            "returns": np.asarray((1.0, 2.0, 3.0, 4.0)),
            "success": np.ones(self.episodes, dtype=np.bool_),
            "object_index": np.asarray((0, 1, 2, 3)),
            "task_index": np.asarray((0, 1, 0, 1)),
            "curriculum_phase": np.zeros(self.episodes, dtype=np.int8),
            "template_index": np.asarray((0, 1, 2, 3)),
            "source_region": np.zeros(self.episodes, dtype=np.int8),
            "path_seed": np.asarray((1, 0, 1, 0)),
            "path_shape_index": np.asarray((1, 0, 4, 0)),
            "path_tracking_rmse_m": np.asarray((0.01, 0.0, 0.02, 0.0)),
            "path_tracking_max_error_m": np.asarray((0.02, 0.0, 0.03, 0.0)),
            "rotation_angle_deg": np.asarray((0, 30, 0, 45)),
            "position_error_m": np.asarray((0.1, 0.2, 0.3, 0.4)),
            "euler_error_deg": np.asarray((1.0, 2.0, 3.0, 4.0)),
            "position_error_xyz_m": np.zeros((self.episodes, 3)),
            "euler_error_xyz_deg": np.zeros((self.episodes, 3)),
            "plane_compliance": np.ones(self.episodes, dtype=np.bool_),
            "object_dropped": np.zeros(self.episodes, dtype=np.bool_),
            "path_frozen": np.ones(self.episodes, dtype=np.bool_),
            "plane_action_frozen": np.ones(self.episodes, dtype=np.bool_),
            "anchors_frozen": np.ones(self.episodes, dtype=np.bool_),
            "trajectory_certified": np.ones(self.episodes, dtype=np.bool_),
            "frozen_path_waypoints_xy_m": frozen_path,
            "frozen_episode_action": frozen_episode_action,
            "frozen_tip_height_fraction": np.repeat(
                planned_fraction[:, None], len(contract.TIP_ORDER), axis=1
            ),
            "frozen_grasp_plane_fraction": planned_fraction,
            "frozen_grasp_plane_yaw_rad": np.asarray(
                (0.1, -0.1, 0.2, -0.2), dtype=np.float32
            ),
            "frozen_hard_tip_anchors_object_local_m": np.zeros(
                (self.episodes, len(contract.TIP_ORDER), 3),
                dtype=np.float32,
            ),
        }

    def _store(self, root: Path) -> CertifiedIKDemoStore:
        return CertifiedIKDemoStore(
            root,
            capacity=self.episodes,
            steps=self.steps,
            create=True,
            provenance=self.provenance,
        )

    def test_provenance_is_hashed_and_coverage_is_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            observations, actions = self._arrays()
            store.append(observations, actions, **self._metadata())
            original_distribution = store.distribution()
            hashes = store.finalize()
            metadata = json.loads(
                (root / "metadata.json").read_text(encoding="utf-8")
            )

            self.assertEqual(metadata["format"], DEMO_FORMAT)
            self.assertEqual(
                metadata["source_region_names"],
                {"0": "all"},
            )
            for name in (
                "source_region",
                "path_seed",
                "path_shape_index",
                "path_tracking_rmse_m",
                "path_tracking_max_error_m",
                "rotation_angle_deg",
            ):
                self.assertIn(name, DATA_FILES)
                self.assertIn(DATA_FILES[name], hashes)
            self.assertEqual(metadata["distribution"], original_distribution)
            coverage = metadata["coverage_distribution"]
            self.assertEqual(
                coverage["run_counts"],
                {"all": 4},
            )
            self.assertEqual(
                coverage["source_region_counts"],
                {"all": 4},
            )
            self.assertEqual(coverage["object_counts"]["orange"], 1)
            self.assertEqual(coverage["task_counts"]["translate"], 2)
            self.assertEqual(
                coverage["template_counts"]["orange"]["0"],
                1,
            )
            self.assertEqual(
                coverage["source_region_by_object_task"]["orange"][
                    "translate"
                ],
                {"all": 1},
            )
            self.assertEqual(
                coverage["source_region_by_object_task"]["banana"][
                    "rotate"
                ],
                {"all": 1},
            )
            geometry = coverage["translation_path_by_object"]
            self.assertEqual(
                geometry["orange"]["shape_counts"]["circle_r50mm"],
                1,
            )
            self.assertEqual(
                geometry["pear"]["shape_counts"]["letter_n_200mm"],
                1,
            )
            self.assertEqual(
                geometry["pear"]["seed_unique_count"],
                1,
            )
            self.assertAlmostEqual(
                geometry["pear"]["tracking_rmse_mean_m"],
                0.02,
            )
            rotation = coverage["rotation_angle"]
            self.assertEqual(rotation["angle_histogram"], {"30": 1, "45": 1})
            self.assertEqual(rotation["maximum_target_count"], 1)
            self.assertEqual(
                rotation["maximum_target_success_count"], 1
            )

            sampler = DemoWindowSampler(root, seed=17)
            observation, action = sampler.sample_stratified(
                8,
                2,
                device="cpu",
            )
            identifiers = Counter(
                int(value) for value in observation[:, 1, 0].tolist()
            )
            self.assertEqual(identifiers, Counter({0: 2, 1: 2, 2: 2, 3: 2}))
            self.assertEqual(
                tuple(action.shape),
                (8, 2, contract.ACTION_DIM),
            )

    def test_nested_tuple_provenance_round_trips_as_json(self) -> None:
        provenance = {
            "config": {
                "object_world_counts": (10_000,) * 4,
                "remainder_object_world_counts": (0, 0, 0, 0),
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            created = CertifiedIKDemoStore(
                root,
                capacity=self.episodes,
                steps=self.steps,
                create=True,
                provenance=provenance,
            )
            self.assertEqual(created.completed, 0)
            metadata = json.loads(
                (root / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["provenance"]["config"]["object_world_counts"],
                [10_000] * 4,
            )
            reopened = CertifiedIKDemoStore(
                root,
                capacity=self.episodes,
                steps=self.steps,
                create=False,
                provenance=provenance,
            )
            self.assertEqual(reopened.completed, 0)
            changed = {
                "config": {
                    **provenance["config"],
                    "object_world_counts": (9_999, 10_000, 10_000, 10_000),
                }
            }
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                CertifiedIKDemoStore(
                    root,
                    capacity=self.episodes,
                    steps=self.steps,
                    create=False,
                    provenance=changed,
                )

    def test_store_accepts_native_planner_yaw_beyond_policy_residual_limit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(Path(directory))
            observations, actions = self._arrays()
            metadata = self._metadata()
            metadata["frozen_grasp_plane_yaw_rad"][0] = np.deg2rad(135.0)
            store.append(observations, actions, **metadata)
            self.assertEqual(store.completed, self.episodes)

    def test_store_rejects_nonprincipal_frozen_planner_yaw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(Path(directory))
            observations, actions = self._arrays()
            metadata = self._metadata()
            metadata["frozen_grasp_plane_yaw_rad"][0] = np.pi + 0.01
            with self.assertRaisesRegex(
                ValueError,
                "frozen planner grasp-plane yaw is outside",
            ):
                store.append(observations, actions, **metadata)

    def test_append_rejects_inconsistent_single_run_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            observations, actions = self._arrays()
            metadata = self._metadata()
            metadata["source_region"] = np.asarray((0, 0, 0, 1))
            with self.assertRaisesRegex(
                ValueError,
                "source_region",
            ):
                store.append(observations, actions, **metadata)
            self.assertEqual(store.completed, 0)

    def test_distribution_gates_only_the_trained_task(self) -> None:
        coverage = {
            "template_counts": {
                object_name: {
                    str(index): 1
                    for index in range(contract.RESET_TEMPLATES_PER_OBJECT)
                }
                for object_name in contract.OBJECTS
            },
            "translation_path_by_object": {
                object_name: {
                    "shape_success_counts": {
                        shape_name: 1
                        for shape_name in contract.TRANSLATION_PATH_SHAPES
                    }
                }
                for object_name in contract.OBJECTS
            },
            "rotation_angle_by_object": {
                object_name: {
                    "maximum_target_deg": 45,
                    "maximum_target_success_count": 0,
                }
                for object_name in contract.OBJECTS
            },
        }

        class Store:
            completed = DEFAULT_CONFIG.elite.minimum_successful_demos

            def __init__(self, task: str) -> None:
                self.task = task

            def distribution(self):
                return {
                    f"all/{object_name}/{self.task}": 3
                    for object_name in contract.OBJECTS
                }

            @staticmethod
            def coverage_distribution():
                return coverage

        translation_config = replace(
            DEFAULT_CONFIG, task=contract.TASK_TRANSLATE
        )
        _validate_demo_distribution(
            Store(contract.TASK_TRANSLATE), translation_config
        )
        for values in coverage["rotation_angle_by_object"].values():
            values["maximum_target_success_count"] = 1
        for values in coverage["translation_path_by_object"].values():
            for shape in values["shape_success_counts"]:
                values["shape_success_counts"][shape] = 0
        rotation_config = replace(DEFAULT_CONFIG, task=contract.TASK_ROTATE)
        _validate_demo_distribution(
            Store(contract.TASK_ROTATE), rotation_config
        )

    def test_append_rejects_wrong_run_source_and_rotation_angle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            observations, actions = self._arrays()
            metadata = self._metadata()
            metadata["source_region"] = np.asarray((0, 0, 1, 0))
            with self.assertRaisesRegex(
                ValueError, "source_region"
            ):
                store.append(observations, actions, **metadata)
            metadata = self._metadata()
            metadata["rotation_angle_deg"] = np.asarray((0, 46, 0, 45))
            with self.assertRaisesRegex(
                ValueError, "rotation_angle_deg is outside the full contract"
            ):
                store.append(observations, actions, **metadata)
            self.assertEqual(store.completed, 0)


if __name__ == "__main__":
    unittest.main()
