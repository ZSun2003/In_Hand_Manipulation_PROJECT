"""G6-only tests for fixed-object dense-Newton MJWarp shards."""

from __future__ import annotations

import hashlib
import unittest
from unittest import mock
from xml.etree import ElementTree

import mujoco
import numpy as np

import _g6_grab
from _g6_objects import get_object_spec
from MJWarp_RL import contract
from MJWarp_RL import platform_adapter as adapter


def _synthetic_parent_templates(task: str) -> dict[str, np.ndarray]:
    """Fast finite reset bank; real parent-IK scanning belongs to HPC smoke."""

    model = mujoco.MjModel.from_xml_path(str(adapter.PARENT_MODEL_XML.resolve()))
    templates = contract.RESET_TEMPLATES_PER_OBJECT
    qpos = np.broadcast_to(
        np.asarray(model.qpos0),
        (len(contract.OBJECTS), templates, model.nq),
    ).copy()
    ctrl = np.zeros((len(contract.OBJECTS), templates, model.nu))
    initial_position = np.empty((len(contract.OBJECTS), templates, 3))
    initial_quaternion = np.empty((len(contract.OBJECTS), templates, 4))
    fractions = np.full((len(contract.OBJECTS), templates), 0.5)
    if task == contract.TASK_ROTATE:
        fractions[:, 0] = 0.495
    yaw = np.zeros_like(fractions)
    pregrasp = np.zeros((len(contract.OBJECTS), templates, 4, 3))
    contact = np.zeros_like(pregrasp)
    outward = np.zeros_like(pregrasp)
    bottom = np.empty(len(contract.OBJECTS))
    top = np.empty(len(contract.OBJECTS))
    for object_index, object_kind in enumerate(contract.OBJECTS):
        spec = get_object_spec(object_kind)
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint_name
        )
        address = int(model.jnt_qposadr[joint_id])
        initial_position[object_index] = qpos[object_index, :, address : address + 3]
        initial_quaternion[object_index] = qpos[
            object_index, :, address + 3 : address + 7
        ]
        bottom[object_index] = float(spec.local_bottom_z_m)
        top[object_index] = float(spec.local_top_z_m)
        contact[object_index, :, :, 2] = (
            bottom[object_index]
            + fractions[object_index] * float(spec.height_m)
        )[:, None]
        pregrasp[object_index] = contact[object_index]
        outward[object_index, :, :, 0] = 1.0
    return {
        "qpos": qpos,
        "ctrl": ctrl,
        "initial_position": initial_position,
        "initial_quaternion": initial_quaternion,
        "pregrasp": pregrasp,
        "contact": contact,
        "outward": outward,
        "planned_fraction": fractions,
        "planned_yaw": yaw,
        "bottom": bottom,
        "top": top,
    }


class G6RLShardXMLTests(unittest.TestCase):
    def test_nominal_reset_planes_follow_all_fruit_contract(self) -> None:
        expected = {
            "orange": (0.5, None),
            "banana": (0.5, None),
            "pear": (0.5, None),
            "strawberry": (0.5, None),
        }
        for object_kind, (fraction, yaw_rad) in expected.items():
            spec = get_object_spec(object_kind)
            plane_z_m, actual_yaw = (
                adapter._nominal_grasp_plane_parameters(_g6_grab, spec)
            )
            actual_fraction = (
                plane_z_m - float(spec.local_bottom_z_m)
            ) / float(spec.height_m)
            kwargs = adapter._nominal_grasp_plane_reset_kwargs(
                _g6_grab, spec
            )
            with self.subTest(object=object_kind):
                self.assertAlmostEqual(actual_fraction, fraction, places=15)
                self.assertEqual(actual_yaw, yaw_rad)
                self.assertEqual(kwargs["demo_grasp_plane_z_m"], plane_z_m)
                if yaw_rad is None:
                    self.assertNotIn("demo_grasp_plane_yaw_rad", kwargs)
                else:
                    self.assertEqual(
                        kwargs["demo_grasp_plane_yaw_rad"], yaw_rad
                    )

    def test_canonical_planner_is_unchanged(self) -> None:
        parent = ElementTree.parse(adapter.PARENT_MODEL_XML).getroot()
        option = parent.find("option")
        self.assertIsNotNone(option)
        self.assertEqual(option.get("solver"), "CG")
        self.assertEqual(option.get("jacobian"), "sparse")
        parent_objects = {
            body.get("name")
            for body in parent.findall("./worldbody/body")
            if str(body.get("name", "")).startswith("object_")
        }
        self.assertEqual(
            parent_objects,
            {
                *(f"object_{object_kind}" for object_kind in contract.OBJECTS),
                *(f"object_{object_kind}" for object_kind in contract.INFEASIBLE_OBJECTS),
                "object_cube",
                "object_plate",
            },
        )

    def test_shard_xml_order_solver_and_exact_object_membership(self) -> None:
        self.assertEqual(tuple(adapter.MODEL_XMLS_BY_TASK), contract.TASKS)
        self.assertEqual(tuple(adapter.TASK_GRIPPER_XMLS), contract.TASKS)
        for task in contract.TASKS:
            model_xmls = adapter.model_xmls_for_task(task)
            self.assertEqual(tuple(model_xmls), contract.OBJECTS)
            self.assertTrue(adapter.TASK_GRIPPER_XMLS[task].is_file())
            for fixed_object_index, object_kind in enumerate(contract.OBJECTS):
                path = model_xmls[object_kind]
                with self.subTest(task=task, object=object_kind):
                    self.assertTrue(path.is_file())
                    root = ElementTree.parse(path).getroot()
                    option = root.find("option")
                    self.assertIsNotNone(option)
                    self.assertEqual(option.get("solver"), "Newton")
                    self.assertEqual(option.get("jacobian"), "dense")
                    self.assertEqual(option.get("iterations"), "150")
                    self.assertEqual(option.get("ls_iterations"), "20")
                    self.assertEqual(float(option.get("tolerance")), 1.0e-10)
                    object_bodies = {
                        body.get("name")
                        for body in root.findall("./worldbody/body")
                        if str(body.get("name", "")).startswith("object_")
                    }
                    self.assertEqual(object_bodies, {f"object_{object_kind}"})

                    model = mujoco.MjModel.from_xml_path(str(path.resolve()))
                    self.assertEqual((model.nq, model.nv, model.nu), (31, 30, 16))
                    self.assertLessEqual(model.nv, 60)
                    self.assertEqual(
                        int(model.opt.solver), int(mujoco.mjtSolver.mjSOL_NEWTON)
                    )
                    self.assertEqual(
                        int(model.opt.jacobian),
                        int(mujoco.mjtJacobian.mjJAC_DENSE),
                    )
                    self.assertFalse(bool(mujoco.mj_isSparse(model)))
                    adapter._validate_rl_model(
                        mujoco,
                        model,
                        fixed_object_index,
                        expected_root_height_m=adapter.TASK_ROOT_HEIGHT_M[task],
                    )
                    for candidate in (
                        *contract.OBJECTS,
                        *contract.INFEASIBLE_OBJECTS,
                        "plate",
                    ):
                        expected = (
                            candidate == object_kind and candidate != "plate"
                        )
                        actual = (
                            mujoco.mj_name2id(
                                model,
                                mujoco.mjtObj.mjOBJ_BODY,
                                f"object_{candidate}",
                            )
                            >= 0
                        )
                        self.assertEqual(actual, expected)

    def test_projection_rejects_wrong_fixed_object_contract(self) -> None:
        planner = mujoco.MjModel.from_xml_path(
            str(adapter.PARENT_MODEL_XML.resolve())
        )
        orange = mujoco.MjModel.from_xml_path(
            str(
                adapter.model_xmls_for_task(contract.TASK_TRANSLATE)[
                    "orange"
                ].resolve()
            )
        )
        projection = adapter._qpos_projection(mujoco, planner, orange, 0)
        self.assertEqual(len(projection), orange.nq)
        with self.assertRaisesRegex(ValueError, "omitted the wrong"):
            adapter._qpos_projection(mujoco, planner, orange, 1)


class G6PlatformShardProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        platforms = {}
        for task in contract.TASKS:
            with mock.patch.object(
                adapter,
                "_collect_parent_templates",
                return_value=_synthetic_parent_templates(task),
            ):
                platforms[task] = adapter.build_platform_spec(task)
        cls.platforms = platforms

    def test_aggregate_order_fingerprints_and_single_ik_generation(self) -> None:
        self.assertEqual(set(self.platforms), set(contract.TASKS))
        self.assertEqual(
            sum(len(platform.shards) for platform in self.platforms.values()),
            8,
        )
        for task, aggregate in self.platforms.items():
            with self.subTest(task=task):
                self.assertIsInstance(aggregate, adapter.ShardedPlatformSpec)
                self.assertEqual(aggregate.platform, adapter.PLATFORM)
                self.assertEqual(aggregate.task, task)
                self.assertEqual(
                    aggregate.task_index, contract.TASKS.index(task)
                )
                self.assertEqual(
                    aggregate.root_height_m, adapter.TASK_ROOT_HEIGHT_M[task]
                )
                self.assertEqual(len(aggregate.shards), len(contract.OBJECTS))
                self.assertEqual(
                    tuple(
                        shard.fixed_object_index for shard in aggregate.shards
                    ),
                    tuple(range(len(contract.OBJECTS))),
                )
                self.assertIs(
                    aggregate.host_model, aggregate.shards[0].host_model
                )
                child_hashes = [
                    shard.rl_model_fingerprint_sha256
                    for shard in aggregate.shards
                ]
                aggregate_hash = adapter._ordered_shard_fingerprint(
                    child_hashes,
                    task=task,
                    root_height_m=adapter.TASK_ROOT_HEIGHT_M[task],
                )
                self.assertEqual(
                    aggregate.rl_model_fingerprint_sha256, aggregate_hash
                )
                self.assertEqual(
                    aggregate.planner_model_fingerprint_sha256,
                    _g6_grab.MODEL_FINGERPRINT_SHA256,
                )
                self.assertEqual(
                    aggregate.parent_model_fingerprint_sha256,
                    adapter._bound_model_fingerprint(
                        _g6_grab.MODEL_FINGERPRINT_SHA256,
                        aggregate_hash,
                        task=task,
                        root_height_m=adapter.TASK_ROOT_HEIGHT_M[task],
                    ),
                )
                self.assertTrue(
                    all(
                        shard.parent_controller_fingerprint_sha256
                        == aggregate.parent_controller_fingerprint_sha256
                        for shard in aggregate.shards
                    )
                )

    def test_learned_fruit_planes_reach_all_32_parent_resets(self) -> None:
        non_nominal_rotation_plane = False
        for task, platform in self.platforms.items():
            for object_index, object_kind in enumerate(contract.OBJECTS):
                planned_fraction = platform.shards[
                    object_index
                ].planned_grasp_plane_fraction[object_index]
                planned_yaw = platform.shards[
                    object_index
                ].planned_grasp_plane_yaw_rad[object_index]
                with self.subTest(task=task, object=object_kind):
                    self.assertEqual(
                        planned_fraction.shape,
                        (contract.RESET_TEMPLATES_PER_OBJECT,),
                    )
                    self.assertTrue(np.isfinite(planned_fraction).all())
                    self.assertTrue(np.isfinite(planned_yaw).all())
                    self.assertTrue(
                        bool(
                            (
                                planned_fraction
                                >= contract.GRASP_PLANE_MIN_FRACTION
                            ).all()
                        )
                    )
                    self.assertTrue(
                        bool(
                            (
                                planned_fraction
                                <= contract.GRASP_PLANE_MAX_FRACTION
                            ).all()
                        )
                    )
                if task == contract.TASK_ROTATE:
                    non_nominal_rotation_plane |= bool(
                        (np.abs(planned_fraction - 0.5) > 1.0e-6).any()
                    )
        self.assertTrue(non_nominal_rotation_plane)
        candidates = adapter._PLANE_FRACTION_CANDIDATES
        self.assertEqual(candidates[0], 0.5)
        self.assertIn(contract.GRASP_PLANE_MIN_FRACTION, candidates)
        self.assertIn(contract.GRASP_PLANE_MAX_FRACTION, candidates)

    def test_each_child_has_only_one_valid_four_slot_reset_bank(self) -> None:
        for task, platform in self.platforms.items():
            model_xmls = adapter.model_xmls_for_task(task)
            for fixed_object_index, object_kind in enumerate(contract.OBJECTS):
                shard = platform.shards[fixed_object_index]
                model = shard.host_model
                unsupported = [
                    index
                    for index in range(len(contract.OBJECTS))
                    if index != fixed_object_index
                ]
                with self.subTest(task=task, object=object_kind):
                    self.assertEqual(shard.task, task)
                    self.assertEqual(
                        shard.root_height_m, adapter.TASK_ROOT_HEIGHT_M[task]
                    )
                    self.assertEqual(shard.fixed_object_index, fixed_object_index)
                    self.assertEqual(shard.model_xml, model_xmls[object_kind])
                    xml_hash = adapter._task_shard_fingerprint(
                        shard.model_xml,
                        adapter.TASK_GRIPPER_XMLS[task],
                        task=task,
                        root_height_m=adapter.TASK_ROOT_HEIGHT_M[task],
                    )
                    self.assertEqual(shard.model_xml_sha256, hashlib.sha256(
                        shard.model_xml.read_bytes()
                    ).hexdigest())
                    self.assertEqual(shard.rl_model_fingerprint_sha256, xml_hash)
                    self.assertEqual((model.nq, model.nv, model.nu), (31, 30, 16))
                    self.assertEqual(
                        shard.qpos_templates.shape,
                        (
                            len(contract.OBJECTS),
                            contract.RESET_TEMPLATES_PER_OBJECT,
                            31,
                        ),
                    )
                    self.assertEqual(
                        shard.ctrl_templates.shape,
                        (
                            len(contract.OBJECTS),
                            contract.RESET_TEMPLATES_PER_OBJECT,
                            16,
                        ),
                    )
                    for bank in (
                        shard.qpos_templates,
                        shard.ctrl_templates,
                        shard.initial_object_pos,
                        shard.initial_object_quat,
                        shard.pregrasp_targets_local,
                        shard.contact_targets_local,
                        shard.outward_directions_local,
                        shard.planned_grasp_plane_fraction,
                        shard.planned_grasp_plane_yaw_rad,
                    ):
                        self.assertTrue(np.isfinite(bank[fixed_object_index]).all())
                        self.assertFalse(np.isfinite(bank[unsupported]).any())
                    self.assertTrue(
                        np.isfinite(
                            shard.object_local_bottom_z_m[fixed_object_index]
                        )
                    )
                    self.assertTrue(
                        np.isfinite(shard.object_local_top_z_m[fixed_object_index])
                    )
                    self.assertFalse(
                        np.isfinite(
                            shard.object_local_bottom_z_m[unsupported]
                        ).any()
                    )
                    self.assertFalse(
                        np.isfinite(
                            shard.object_local_top_z_m[unsupported]
                        ).any()
                    )

                    qpos_address = int(
                        shard.object_qpos_adrs[fixed_object_index]
                    )
                    np.testing.assert_allclose(
                        shard.qpos_templates[
                            fixed_object_index,
                            :,
                            qpos_address : qpos_address + 3,
                        ],
                        shard.initial_object_pos[fixed_object_index],
                        rtol=0.0,
                        atol=2.0e-6,
                    )
                    np.testing.assert_allclose(
                        shard.qpos_templates[
                            fixed_object_index,
                            :,
                            qpos_address + 3 : qpos_address + 7,
                        ],
                        shard.initial_object_quat[fixed_object_index],
                        rtol=0.0,
                        atol=2.0e-6,
                    )

    def test_child_ids_and_geom_masks_refer_to_its_active_object(self) -> None:
        for task, platform in self.platforms.items():
            for fixed_object_index, object_kind in enumerate(contract.OBJECTS):
                shard = platform.shards[fixed_object_index]
                model = shard.host_model
                object_spec = get_object_spec(object_kind)
                with self.subTest(task=task, object=object_kind):
                    body_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, object_spec.body_name
                    )
                    joint_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, object_spec.joint_name
                    )
                    expected_geoms = {
                        mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
                        )
                        for geom_name in object_spec.geom_names
                    }
                    self.assertTrue(np.all(shard.object_body_ids == body_id))
                    self.assertTrue(
                        np.all(
                            shard.object_qpos_adrs
                            == int(model.jnt_qposadr[joint_id])
                        )
                    )
                    self.assertTrue(
                        np.all(
                            shard.object_qvel_adrs
                            == int(model.jnt_dofadr[joint_id])
                        )
                    )
                    for mask in shard.object_geom_mask:
                        self.assertEqual(set(np.flatnonzero(mask)), expected_geoms)


if __name__ == "__main__":
    unittest.main()
