"""Fast contract tests for the self-contained G3/G6 manipulation project.

The tests in this module never run a full environment episode.  Most stop at
reset; termination/recovery cases use one physics-patched step.  No test starts
reinforcement-learning work.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from g3g6.kinematics import g6 as t6
from g3g6 import config, trajectories
from g3g6.env import G3G6ManipulationEnv, OBSERVATION_DIM
from g3g6.math3d import euler_xyz_deg_to_quat, quat_multiply
from g3g6.objects import FINGERS, OBJECTS, RADIAL
from g3g6.residual import AngularResidualController
from g3g6.rotations import (
    ROTATION_MOTION_TICKS,
    build_rotation_euler_schedule,
    sample_goal_euler_deg,
    validate_axis_set,
    validate_goal_euler_deg,
    validate_rotation_sampling_profile,
)


def _vector(element: ET.Element, attribute: str, length: int) -> np.ndarray:
    text = element.get(attribute)
    if text is None:
        raise AssertionError(
            f"<{element.tag}> has no required {attribute!r} attribute"
        )
    value = np.fromstring(text.replace(",", " "), sep=" ", dtype=np.float64)
    if value.shape != (length,):
        raise AssertionError(
            f"<{element.tag}> {attribute!r} expected {length} values, got {text!r}"
        )
    return value


def _required(root: ET.Element, query: str) -> ET.Element:
    element = root.find(query)
    if element is None:
        raise AssertionError(f"required XML element not found: {query}")
    return element


def _set_selected_object_pose(
    env: G3G6ManipulationEnv,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Set one free body directly for a no-episode action-reference test."""

    joint_id = env._object_joint_ids[env.object_name]
    qpos_address = int(env.model.jnt_qposadr[joint_id])
    dof_address = int(env.model.jnt_dofadr[joint_id])
    env.data.qpos[qpos_address : qpos_address + 7] = (
        *np.asarray(position, dtype=np.float64),
        *np.asarray(quaternion, dtype=np.float64),
    )
    env.data.qvel[dof_address : dof_address + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    return env._selected_pose()


class LocalAssetContractTests(unittest.TestCase):
    def assertInsideProject(self, path: Path) -> Path:  # noqa: N802 - unittest style
        resolved = path.resolve()
        try:
            resolved.relative_to(config.PROJECT_ROOT.resolve())
        except ValueError:
            self.fail(f"path escapes the G3G6 project: {resolved}")
        return resolved

    def _xml_dependency_closure(self, project_xml: Path) -> set[Path]:
        pending = [self.assertInsideProject(project_xml)]
        visited: set[Path] = set()
        while pending:
            xml_path = pending.pop()
            if xml_path in visited:
                continue
            self.assertTrue(xml_path.is_file(), xml_path)
            visited.add(xml_path)
            root = ET.parse(xml_path).getroot()
            for include in root.iter("include"):
                raw_name = include.get("file")
                self.assertTrue(raw_name, f"include without file in {xml_path}")
                relative = Path(str(raw_name))
                self.assertFalse(relative.is_absolute(), relative)
                dependency = self.assertInsideProject(xml_path.parent / relative)
                self.assertTrue(dependency.is_file(), dependency)
                pending.append(dependency)
        return visited

    def test_all_configured_paths_includes_and_meshes_are_project_local(self) -> None:
        root = config.PROJECT_ROOT.resolve()
        self.assertEqual(root, Path(__file__).resolve().parents[1])
        self.assertEqual(config.LOCAL_VENV.resolve(), root.parent / "RL_env")
        self.assertTrue((config.LOCAL_VENV / "pyvenv.cfg").is_file())
        self.assertEqual(set(config.MODEL_XML), {"g3", "g6"})

        for gripper, project_xml in config.MODEL_XML.items():
            with self.subTest(gripper=gripper):
                project_xml = self.assertInsideProject(project_xml)
                xml_files = self._xml_dependency_closure(project_xml)
                self.assertEqual(
                    {path.name for path in xml_files},
                    {"Project.xml", "Share.xml", "Gripper.xml"},
                )

                project_root = ET.parse(project_xml).getroot()
                compiler = _required(project_root, "compiler")
                raw_meshdir = Path(compiler.get("meshdir", "."))
                self.assertFalse(raw_meshdir.is_absolute(), raw_meshdir)
                meshdir = self.assertInsideProject(project_xml.parent / raw_meshdir)
                self.assertTrue(meshdir.is_dir(), meshdir)

                mesh_references: list[Path] = []
                for xml_file in xml_files:
                    xml_root = ET.parse(xml_file).getroot()
                    for mesh in xml_root.iter("mesh"):
                        raw_file = mesh.get("file")
                        if raw_file is None:  # Inline triangle/cone meshes.
                            continue
                        relative = Path(raw_file)
                        self.assertFalse(relative.is_absolute(), relative)
                        mesh_path = self.assertInsideProject(meshdir / relative)
                        self.assertTrue(mesh_path.is_file(), mesh_path)
                        mesh_references.append(mesh_path)
                self.assertGreater(len(mesh_references), 0)

    def test_authored_python_and_xml_do_not_embed_external_absolute_paths(self) -> None:
        authored_files = [
            *sorted(config.PROJECT_ROOT.glob("*.py")),
            *sorted((config.PROJECT_ROOT / "g3g6").glob("*.py")),
            *sorted((config.PROJECT_ROOT / "g3g6" / "kinematics").glob("*.py")),
            *sorted((config.PROJECT_ROOT / "model").rglob("*.xml")),
        ]
        for path in authored_files:
            with self.subTest(path=path.relative_to(config.PROJECT_ROOT)):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("/Users/", text)
                self.assertNotIn("/home/", text)


class XmlGeometryContractTests(unittest.TestCase):
    def _roots(self, gripper: str) -> tuple[ET.Element, ET.Element]:
        folder = config.MODEL_XML[gripper].parent
        return (
            ET.parse(folder / "Project.xml").getroot(),
            ET.parse(folder / "Gripper.xml").getroot(),
        )

    def test_both_models_have_exact_root_and_table_heights(self) -> None:
        for gripper in ("g3", "g6"):
            with self.subTest(gripper=gripper):
                project, gripper_xml = self._roots(gripper)
                option = _required(project, "option")
                self.assertEqual(config.NO_SLIP_ITERATIONS, 10)
                self.assertEqual(
                    int(option.get("noslip_iterations", "-1")),
                    config.NO_SLIP_ITERATIONS,
                )
                root_body = _required(gripper_xml, ".//body[@name='root']")
                self.assertAlmostEqual(
                    float(_vector(root_body, "pos", 3)[2]),
                    config.ROOT_HEIGHT_M,
                    delta=1.0e-12,
                )

                table = _required(project, ".//body[@name='table']")
                table_geom = _required(table, "geom[@name='table_geom']")
                table_position = _vector(table, "pos", 3)
                table_geom_position = np.fromstring(
                    table_geom.get("pos", "0 0 0"), sep=" ", dtype=np.float64
                )
                table_size = _vector(table_geom, "size", 3)
                table_top = table_position[2] + table_geom_position[2] + table_size[2]
                self.assertAlmostEqual(
                    float(table_top), config.TABLE_TOP_M, delta=1.0e-12
                )

    def test_both_models_have_four_point_two_kg_objects_with_exact_sizes(self) -> None:
        geom_names = {
            "cube": "cube_geom",
            "cylinder": "cylinder_geom",
            "triangular_prism": "triangular_prism_geom",
            "cone": "cone_geom",
        }
        for gripper in ("g3", "g6"):
            project, _ = self._roots(gripper)
            for object_name, geom_name in geom_names.items():
                with self.subTest(gripper=gripper, object=object_name):
                    geom = _required(project, f".//geom[@name='{geom_name}']")
                    self.assertAlmostEqual(
                        float(geom.get("mass", "nan")),
                        config.OBJECT_MASS_KG,
                        delta=1.0e-12,
                    )

            cube = _required(project, ".//geom[@name='cube_geom']")
            self.assertEqual(cube.get("type"), "box")
            np.testing.assert_allclose(
                2.0 * _vector(cube, "size", 3),
                (0.050, 0.050, 0.050),
                rtol=0.0,
                atol=1.0e-12,
            )

            cylinder = _required(project, ".//geom[@name='cylinder_geom']")
            self.assertEqual(cylinder.get("type"), "cylinder")
            cylinder_size = _vector(cylinder, "size", 2)
            np.testing.assert_allclose(
                (2.0 * cylinder_size[0], 2.0 * cylinder_size[1]),
                (0.050, 0.080),
                rtol=0.0,
                atol=1.0e-12,
            )

            triangle_mesh = _required(project, ".//mesh[@name='triangular_prism_mesh']")
            triangle_vertices = _vector(triangle_mesh, "vertex", 18).reshape(-1, 3)
            triangle_xy = np.unique(triangle_vertices[:, :2], axis=0)
            self.assertEqual(triangle_xy.shape, (3, 2))
            triangle_sides = sorted(
                float(np.linalg.norm(triangle_xy[first] - triangle_xy[second]))
                for first, second in combinations(range(3), 2)
            )
            np.testing.assert_allclose(
                triangle_sides, (0.060, 0.060, 0.060), rtol=0.0, atol=1.0e-9
            )
            self.assertAlmostEqual(
                float(np.ptp(triangle_vertices[:, 2])), 0.080, delta=1.0e-12
            )

            cone_mesh = _required(project, ".//mesh[@name='cone_60x100_mesh']")
            cone_vertex_values = np.fromstring(
                cone_mesh.get("vertex", ""), sep=" ", dtype=np.float64
            )
            self.assertEqual(cone_vertex_values.size % 3, 0)
            cone_vertices = cone_vertex_values.reshape(-1, 3)
            base_z = float(np.min(cone_vertices[:, 2]))
            apex_z = float(np.max(cone_vertices[:, 2]))
            base = cone_vertices[np.isclose(cone_vertices[:, 2], base_z)]
            base_radii = np.linalg.norm(base[:, :2], axis=1)
            np.testing.assert_allclose(
                base_radii, np.full(base_radii.shape, 0.030), rtol=0.0, atol=1.0e-9
            )
            self.assertAlmostEqual(apex_z - base_z, 0.100, delta=1.0e-12)

    def test_cone_body_origin_is_the_one_third_height_grasp_frame(self) -> None:
        for gripper in ("g3", "g6"):
            with self.subTest(gripper=gripper):
                project, _ = self._roots(gripper)
                cone_body = _required(project, ".//body[@name='cone_60x100']")
                cone_geom = _required(cone_body, "geom[@name='cone_geom']")
                cone_site = _required(cone_body, "site[@name='grasp_frame_cone']")
                body_z = float(_vector(cone_body, "pos", 3)[2])
                geom_z = float(_vector(cone_geom, "pos", 3)[2])
                np.testing.assert_allclose(
                    _vector(cone_site, "pos", 3), np.zeros(3), rtol=0.0, atol=0.0
                )
                self.assertAlmostEqual(
                    body_z, config.TABLE_TOP_M + 0.100 / 3.0, delta=1.0e-9
                )
                # The inline cone mesh is centred at mid-height.  Moving it
                # H/6 above the body frame puts the body origin H/3 above base.
                self.assertAlmostEqual(geom_z, 0.100 / 6.0, delta=1.0e-9)
                self.assertAlmostEqual(geom_z - 0.050, -0.100 / 3.0, delta=1.0e-9)


class ObjectGraspContractTests(unittest.TestCase):
    def test_dimensions_grasp_planes_and_object_specific_lift_distances(self) -> None:
        expected_dimensions = {
            "cube": (0.050, 0.050, 0.050),
            "cylinder": (0.050, 0.080),
            "triangular_prism": (0.060, 0.080),
            "cone": (0.060, 0.100),
        }
        expected_rest_z = {
            "cube": 0.285,
            "cylinder": 0.300,
            "triangular_prism": 0.300,
            "cone": config.TABLE_TOP_M + 0.100 / 3.0,
        }
        expected_lift = {
            "cube": 0.075,
            "cylinder": 0.060,
            "triangular_prism": 0.060,
            "cone": 0.100 - 0.100 / 3.0,
        }
        self.assertEqual(tuple(OBJECTS), config.OBJECT_NAMES)
        for name, spec in OBJECTS.items():
            with self.subTest(object=name):
                self.assertEqual(spec.dimensions_m, expected_dimensions[name])
                self.assertAlmostEqual(
                    spec.rest_grasp_plane_z_m, expected_rest_z[name], delta=1.0e-12
                )
                self.assertAlmostEqual(
                    spec.lift_distance_m, expected_lift[name], delta=1.0e-12
                )
                self.assertAlmostEqual(
                    spec.rest_grasp_plane_z_m + spec.lift_distance_m,
                    config.TARGET_GRASP_PLANE_Z_M,
                    delta=1.0e-12,
                )

    def test_cube_and_cylinder_have_cardinal_quadrature_grasp_points(self) -> None:
        for name in ("cube", "cylinder"):
            spec = OBJECTS[name]
            for finger in FINGERS:
                with self.subTest(object=name, finger=finger):
                    np.testing.assert_allclose(
                        spec.surface_points_local_m[finger],
                        0.025 * RADIAL[finger],
                        rtol=0.0,
                        atol=1.0e-12,
                    )

    def test_cone_points_are_rotated_facet_centres_at_h_over_three(self) -> None:
        spec = OBJECTS["cone"]
        angles: list[float] = []
        deflections: list[float] = []
        for finger in FINGERS:
            point = np.asarray(spec.surface_points_local_m[finger])
            with self.subTest(finger=finger):
                self.assertEqual(float(point[2]), 0.0)
                self.assertAlmostEqual(
                    float(np.linalg.norm(point[:2])), 0.020, delta=1.0e-12
                )
                direction = point[:2] / np.linalg.norm(point[:2])
                radial = np.asarray(RADIAL[finger][:2], dtype=np.float64)
                deflection = float(
                    np.arctan2(
                        radial[0] * direction[1] - radial[1] * direction[0],
                        np.dot(radial, direction),
                    )
                )
                deflections.append(deflection)
                angles.append(float(np.mod(np.arctan2(point[1], point[0]), 2.0 * np.pi)))

        # A 32-facet mesh is contacted at each facet midpoint, a common
        # half-facet turn from the nominal finger rays, while relative azimuths
        # remain exactly one quarter-turn apart.
        np.testing.assert_allclose(
            deflections,
            np.full(4, -np.pi / 32.0),
            rtol=0.0,
            atol=1.0e-12,
        )
        sorted_angles = np.sort(np.asarray(angles))
        adjacent = np.diff(
            np.concatenate((sorted_angles, sorted_angles[:1] + 2.0 * np.pi))
        )
        np.testing.assert_allclose(
            adjacent, np.full(4, np.pi / 2.0), rtol=0.0, atol=1.0e-12
        )

        self.assertAlmostEqual(spec.contact_indentation_m, 0.001, delta=0.0)
        for name in ("cube", "cylinder", "triangular_prism"):
            self.assertAlmostEqual(
                OBJECTS[name].contact_indentation_m,
                config.CONTACT_INDENTATION_M,
                delta=0.0,
            )
        contact_centres = spec.tip_centres_local_m(pregrasp=False)
        for finger in FINGERS:
            realized_offset = float(
                np.dot(
                    contact_centres[finger] - spec.surface_points_local_m[finger],
                    spec.surface_normals_local[finger],
                )
            )
            self.assertAlmostEqual(
                realized_offset,
                config.HARD_TIP_RADIUS_M - 0.001,
                delta=1.0e-12,
            )

    def test_triangle_face_assignment_is_two_plus_one_plus_one_without_overlap(self) -> None:
        spec = OBJECTS["triangular_prism"]
        groups: list[list[str]] = []
        for finger in FINGERS:
            normal = spec.surface_normals_local[finger]
            for group in groups:
                if np.allclose(
                    normal,
                    spec.surface_normals_local[group[0]],
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    group.append(finger)
                    break
            else:
                groups.append([finger])

        self.assertEqual(sorted(map(len, groups)), [1, 1, 2])
        doubled = next(group for group in groups if len(group) == 2)
        self.assertEqual(set(doubled), {"U", "R"})

        tip_centres = spec.tip_centres_local_m(pregrasp=False)
        doubled_distance = float(
            np.linalg.norm(tip_centres[doubled[0]] - tip_centres[doubled[1]])
        )
        self.assertGreater(doubled_distance, 2.0 * config.HARD_TIP_RADIUS_M)
        self.assertAlmostEqual(doubled_distance, 0.033, delta=1.0e-12)
        for first, second in combinations(FINGERS, 2):
            with self.subTest(first=first, second=second):
                self.assertGreaterEqual(
                    float(np.linalg.norm(tip_centres[first] - tip_centres[second])),
                    2.0 * config.HARD_TIP_RADIUS_M,
                )


class ResidualControllerContractTests(unittest.TestCase):
    @staticmethod
    def _controller(gripper: str) -> tuple[AngularResidualController, np.ndarray]:
        actuator_count = 12 if gripper == "g3" else 16
        nominal = np.linspace(-0.4, 0.4, actuator_count, dtype=np.float64)
        controller = AngularResidualController(
            gripper,
            np.full(actuator_count, -10.0),
            np.full(actuator_count, 10.0),
        )
        return controller, nominal

    def test_native_action_dimensions_and_unit_aware_actuator_mapping(self) -> None:
        self.assertEqual(config.ACTION_DIM, 12)
        self.assertEqual(config.G3_ACTION_DIM, 12)
        self.assertEqual(config.G6_ACTION_DIM, 16)
        self.assertEqual(config.MAX_ACTION_DIM, 16)
        self.assertEqual(config.action_dim("g3"), 12)
        self.assertEqual(config.action_dim("g6"), 16)
        self.assertEqual(config.G3_ANGULAR_ACTUATOR_INDICES, tuple(range(12)))
        self.assertEqual(
            config.G6_ANGULAR_ACTUATOR_INDICES,
            (1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15),
        )
        self.assertEqual(config.G6_LINEAR_ACTUATOR_INDICES, (0, 3, 6, 9))
        self.assertEqual(
            set(config.G6_ANGULAR_ACTUATOR_INDICES)
            | set(config.G6_LINEAR_ACTUATOR_INDICES),
            set(range(16)),
        )
        self.assertEqual(config.RESIDUAL_ACTUATOR_INDICES["g3"], tuple(range(12)))
        self.assertEqual(config.RESIDUAL_ACTUATOR_INDICES["g6"], tuple(range(16)))

        g3, _nominal = self._controller("g3")
        np.testing.assert_array_equal(g3.units, np.full(12, "rad", dtype=object))
        g6, _nominal = self._controller("g6")
        angular = np.asarray(config.G6_ANGULAR_ACTUATOR_INDICES)
        linear = np.asarray(config.G6_LINEAR_ACTUATOR_INDICES)
        np.testing.assert_array_equal(g6.units[angular], np.full(12, "rad", dtype=object))
        np.testing.assert_array_equal(g6.units[linear], np.full(4, "m", dtype=object))
        np.testing.assert_array_equal(
            g6.limits[angular], np.full(12, config.ANGULAR_RESIDUAL_LIMIT_RAD)
        )
        np.testing.assert_array_equal(
            g6.limits[linear], np.full(4, config.LINEAR_RESIDUAL_LIMIT_M)
        )
        np.testing.assert_array_equal(
            g6.rates[linear], np.full(4, config.LINEAR_RESIDUAL_RATE_M_PER_TICK)
        )
        np.testing.assert_array_equal(
            g6.blends, np.full(16, 0.35)
        )

    def test_zero_action_and_reset_produce_zero_residual(self) -> None:
        for gripper in ("g3", "g6"):
            with self.subTest(gripper=gripper):
                controller, nominal = self._controller(gripper)
                action_dim = config.action_dim(gripper)
                applied, residual = controller.apply(nominal, np.zeros(action_dim))
                np.testing.assert_array_equal(residual, np.zeros(action_dim))
                np.testing.assert_array_equal(applied, nominal)

                controller.apply(nominal, np.ones(action_dim))
                controller.reset()
                applied, residual = controller.apply(nominal, np.zeros(action_dim))
                np.testing.assert_array_equal(residual, np.zeros(action_dim))
                np.testing.assert_array_equal(applied, nominal)

    def test_residual_obeys_per_tick_rate_and_absolute_limit(self) -> None:
        for gripper in ("g3", "g6"):
            with self.subTest(gripper=gripper):
                controller, nominal = self._controller(gripper)
                action_dim = config.action_dim(gripper)
                previous = np.zeros(action_dim, dtype=np.float64)
                for _ in range(256):
                    _applied, residual = controller.apply(
                        nominal, np.ones(action_dim)
                    )
                    self.assertTrue(
                        np.all(
                            np.abs(residual - previous)
                            <= controller.rates + 1.0e-15
                        )
                    )
                    self.assertTrue(
                        np.all(np.abs(residual) <= controller.limits + 1.0e-15)
                    )
                    previous = residual
                np.testing.assert_allclose(
                    residual,
                    controller.limits,
                    rtol=0.0,
                    atol=1.0e-14,
                )

                controller.reset()
                for _ in range(256):
                    _applied, residual = controller.apply(
                        nominal, -np.ones(action_dim)
                    )
                np.testing.assert_allclose(
                    residual,
                    -controller.limits,
                    rtol=0.0,
                    atol=1.0e-14,
                )

    def test_ctrlrange_saturation_has_rate_limited_actual_residual_and_no_windup(self) -> None:
        for gripper, actuator_count in (("g3", 12), ("g6", 16)):
            residual_indices = np.asarray(config.RESIDUAL_ACTUATOR_INDICES[gripper])
            for sign in (-1.0, 1.0):
                with self.subTest(gripper=gripper, sign=sign):
                    controller = AngularResidualController(
                        gripper,
                        np.full(actuator_count, -1.0),
                        np.full(actuator_count, 1.0),
                    )
                    nominal_at_bound = np.zeros(actuator_count, dtype=np.float64)
                    nominal_at_bound[residual_indices] = sign
                    previous_actual = np.zeros(actuator_count, dtype=np.float64)

                    # Repeated outward requests cannot accumulate a hidden
                    # residual behind a saturated actuator ctrlrange.
                    for _ in range(64):
                        applied, actual = controller.apply(
                            nominal_at_bound, np.full(actuator_count, sign)
                        )
                        np.testing.assert_array_equal(
                            applied[residual_indices],
                            nominal_at_bound[residual_indices],
                        )
                        np.testing.assert_array_equal(
                            actual, np.zeros(actuator_count)
                        )
                        self.assertTrue(
                            np.all(
                                np.abs(actual - previous_actual)
                                <= controller.rates + 1.0e-15
                            )
                        )
                        previous_actual = actual

                    # Moving the nominal command into range with zero action
                    # must reveal no stored offset in a single release jump.
                    released_nominal = nominal_at_bound.copy()
                    released_nominal[residual_indices] -= sign * 0.2
                    applied, actual = controller.apply(
                        released_nominal, np.zeros(actuator_count)
                    )
                    np.testing.assert_array_equal(applied, released_nominal)
                    np.testing.assert_array_equal(actual, np.zeros(actuator_count))

                    # A new outward command starts from zero and the realized
                    # actuator-angle offset changes by no more than the
                    # configured 0.1 deg/tick.
                    previous_actual = actual
                    for _ in range(4):
                        applied, actual = controller.apply(
                            released_nominal, np.full(actuator_count, sign)
                        )
                        np.testing.assert_allclose(
                            actual,
                            applied[residual_indices]
                            - released_nominal[residual_indices],
                            rtol=0.0,
                            atol=1.0e-15,
                        )
                        self.assertTrue(
                            np.all(
                                np.abs(actual - previous_actual)
                                <= controller.rates + 1.0e-15
                            )
                        )
                        previous_actual = actual

    def test_g6_maps_all_hinge_and_linear_residuals_in_native_order(self) -> None:
        controller, nominal = self._controller("g6")
        applied, residual = controller.apply(nominal, np.ones(16))
        angular = np.asarray(config.G6_ANGULAR_ACTUATOR_INDICES)
        linear = np.asarray(config.G6_LINEAR_ACTUATOR_INDICES)
        np.testing.assert_allclose(
            applied - nominal, residual, rtol=0.0, atol=1.0e-15
        )
        np.testing.assert_allclose(
            residual[angular],
            np.full(12, config.ANGULAR_RESIDUAL_RATE_RAD_PER_TICK),
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            residual[linear],
            np.full(4, config.LINEAR_RESIDUAL_RATE_M_PER_TICK),
            rtol=0.0,
            atol=1.0e-15,
        )


class G6ContinuityIKContractTests(unittest.TestCase):
    @staticmethod
    def _tip_targets(
        ik: t6.GripperIK, center: tuple[float, float, float]
    ) -> dict[str, np.ndarray]:
        pregrasp_distance = (
            0.025 + config.HARD_TIP_RADIUS_M + config.PREGRASP_CLEARANCE_M
        )
        return ik.cube_tip_targets(center, distance=pregrasp_distance)

    def assertSolutionContract(  # noqa: N802 - unittest assertion style
        self,
        ik: t6.GripperIK,
        result: t6.IKSolutionSummary,
        targets: dict[str, np.ndarray],
    ) -> None:
        self.assertTrue(result.success, result.failed_fingers)
        self.assertEqual(result.failed_fingers, ())
        self.assertTrue(result.solved_mask.all())
        self.assertEqual(set(result.finger_solutions), set(t6.OBS_FINGER_ORDER))
        for finger in t6.OBS_FINGER_ORDER:
            with self.subTest(finger=finger):
                solution = result.finger_solutions[finger]
                measured_tip_error = float(
                    np.linalg.norm(solution.tip_world - targets[finger])
                )
                self.assertAlmostEqual(
                    solution.tip_error, measured_tip_error, delta=1.0e-15
                )
                forward = ik.finger_forward(finger, solution)
                np.testing.assert_allclose(
                    forward.tip_world,
                    solution.tip_world,
                    rtol=0.0,
                    atol=1.0e-15,
                )
                self.assertAlmostEqual(
                    solution.loop_error, forward.loop_error, delta=1.0e-15
                )
                self.assertLessEqual(solution.tip_error, 1.0e-9)
                self.assertLessEqual(solution.loop_error, 1.0e-9)
        self.assertAlmostEqual(
            result.max_tip_error,
            max(value.tip_error for value in result.finger_solutions.values()),
            delta=1.0e-15,
        )
        self.assertAlmostEqual(
            result.max_loop_error,
            max(value.loop_error for value in result.finger_solutions.values()),
            delta=1.0e-15,
        )

    def test_small_target_sequence_prefers_local_linear_neighbours(self) -> None:
        ik = t6.GripperIK.from_xml(config.MODEL_XML["g6"])
        centers = (
            (0.0, 0.0, 0.2850),
            (0.0002, -0.0001, 0.2852),
            (0.0004, -0.0002, 0.2854),
            (0.0006, -0.0003, 0.2856),
        )
        targets = self._tip_targets(ik, centers[0])
        previous = ik.solve_tip_targets(targets, strict=True)
        self.assertSolutionContract(ik, previous, targets)

        step = ik.linear_search_step_m
        for center in centers[1:]:
            targets = self._tip_targets(ik, center)
            with mock.patch.object(
                ik, "solve_finger_target", wraps=ik.solve_finger_target
            ) as search:
                result = ik.solve_tip_targets(
                    targets,
                    previous_ctrl=previous.ctrl,
                    previous_solutions=previous.finger_solutions,
                    strict=True,
                )
            self.assertSolutionContract(ik, result, targets)
            self.assertEqual(len(search.call_args_list), len(t6.OBS_FINGER_ORDER))
            self.assertTrue(
                all(
                    call.kwargs.get("linear_samples") is not None
                    for call in search.call_args_list
                ),
                "a small target move unexpectedly required the full slide grid",
            )
            for call in search.call_args_list:
                finger = call.args[0]
                previous_linear = previous.finger_solutions[finger].q_linear
                np.testing.assert_allclose(
                    call.kwargs["linear_samples"],
                    (
                        previous_linear,
                        previous_linear - step,
                        previous_linear + step,
                    ),
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertLessEqual(
                    abs(
                        result.finger_solutions[finger].q_linear
                        - previous_linear
                    ),
                    step + 1.0e-15,
                )
            previous = result

    def test_forced_local_miss_invokes_full_grid_fallback_and_still_solves(self) -> None:
        ik = t6.GripperIK.from_xml(config.MODEL_XML["g6"])
        initial_targets = self._tip_targets(ik, (0.0, 0.0, 0.2850))
        previous = ik.solve_tip_targets(initial_targets, strict=True)
        targets = self._tip_targets(ik, (0.0008, -0.0004, 0.2858))

        original_search = ik.solve_finger_target
        search_modes: list[tuple[str, str]] = []

        def force_local_miss(
            finger: str,
            target: np.ndarray,
            *args: object,
            **kwargs: object,
        ) -> list[dict[str, object]]:
            if kwargs.get("linear_samples") is not None:
                search_modes.append((finger, "local"))
                return []
            search_modes.append((finger, "full"))
            return original_search(finger, target, *args, **kwargs)

        with mock.patch.object(
            ik, "solve_finger_target", side_effect=force_local_miss
        ):
            result = ik.solve_tip_targets(
                targets,
                previous_ctrl=previous.ctrl,
                previous_solutions=previous.finger_solutions,
                strict=True,
            )

        self.assertSolutionContract(ik, result, targets)
        self.assertEqual(
            search_modes,
            [
                mode
                for finger in t6.OBS_FINGER_ORDER
                for mode in ((finger, "local"), (finger, "full"))
            ],
        )


class PhaseTimingContractTests(unittest.TestCase):
    def test_phase_specific_grasp_plane_targets_and_lift_distances(self) -> None:
        self.assertEqual(config.TARGET_GRASP_PLANE_Z_M, 0.360)
        self.assertEqual(config.ROTATION_TARGET_GRASP_PLANE_Z_M, 0.325)
        expected_target_by_phase = {
            config.Phase.GRASP_LIFT: config.TARGET_GRASP_PLANE_Z_M,
            config.Phase.TRANSLATE: config.TARGET_GRASP_PLANE_Z_M,
            config.Phase.ROTATE: config.ROTATION_TARGET_GRASP_PLANE_Z_M,
        }
        for gripper in ("g3", "g6"):
            for phase, expected_target_z in expected_target_by_phase.items():
                env = G3G6ManipulationEnv(gripper=gripper, phase=phase)
                try:
                    for object_name, spec in OBJECTS.items():
                        options: dict[str, object] = {"object": object_name}
                        if phase == config.Phase.TRANSLATE:
                            options["path"] = "square"
                        elif phase == config.Phase.ROTATE:
                            options.update(
                                {
                                    "rotation_axes": "x",
                                    "rotation_euler_deg": (10.0, 0.0, 0.0),
                                }
                            )
                        _observation, info = env.reset(seed=50, options=options)
                        with self.subTest(
                            gripper=gripper,
                            phase=int(phase),
                            object=object_name,
                        ):
                            self.assertEqual(
                                env.task_target_grasp_plane_z_m,
                                expected_target_z,
                            )
                            self.assertEqual(
                                info["task_target_grasp_plane_z_m"],
                                expected_target_z,
                            )
                            self.assertAlmostEqual(
                                env.task_lift_distance_m,
                                expected_target_z - spec.rest_grasp_plane_z_m,
                                delta=1.0e-12,
                            )
                            self.assertEqual(
                                float(env._action_reference_position[2]),
                                expected_target_z,
                            )
                            final_lift = env._target_for_tick(
                                config.PRE_ACTION_TICKS - 1
                            )
                            self.assertEqual(
                                float(final_lift.position[2]), expected_target_z
                            )
                            if phase != config.Phase.GRASP_LIFT:
                                first_action = env._target_for_tick(
                                    config.PRE_ACTION_TICKS
                                )
                                self.assertEqual(
                                    float(first_action.position[2]),
                                    expected_target_z,
                                )
                finally:
                    env.close()

    def test_cone_fast_lift_cage_is_relative_to_each_phase_target(self) -> None:
        lift_start = config.APPROACH_TICKS + config.CONTACT_TICKS
        for phase in config.Phase:
            env = G3G6ManipulationEnv(gripper="g3", phase=phase, object_name="cone")
            try:
                options: dict[str, object] = {}
                if phase == config.Phase.TRANSLATE:
                    options["path"] = "square"
                elif phase == config.Phase.ROTATE:
                    options.update(
                        {
                            "rotation_axes": "x",
                            "rotation_euler_deg": (10.0, 0.0, 0.0),
                        }
                    )
                env.reset(seed=50, options=options)
                env.step_count = lift_start + config.CONE_FAST_LIFT_TICKS - 1
                target = env._target_for_tick(env.step_count)
                tip_targets = env._cone_fast_lift_tip_targets(target)
                expected_cage = np.array(
                    (
                        0.0,
                        0.0,
                        env.task_target_grasp_plane_z_m
                        + config.CONE_NOMINAL_CAGE_Z_OFFSET_M,
                    ),
                    dtype=np.float64,
                )
                for finger in FINGERS:
                    with self.subTest(phase=int(phase), finger=finger):
                        np.testing.assert_allclose(
                            tip_targets[finger] - target.local_tip_centres[finger],
                            expected_cage,
                            rtol=0.0,
                            atol=1.0e-12,
                        )
            finally:
                env.close()

    def test_weighted_profile_is_phase3_only_and_visible_in_info(self) -> None:
        with self.assertRaisesRegex(ValueError, "only available in Phase 3"):
            G3G6ManipulationEnv(
                gripper="g3",
                phase=config.Phase.TRANSLATE,
                rotation_sampling_profile=(
                    config.WEIGHTED_ROTATION_SAMPLING_PROFILE
                ),
            )

        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.ROTATE,
                object_name="cube",
                rotation_sampling_profile=(
                    config.WEIGHTED_ROTATION_SAMPLING_PROFILE
                ),
            )
            try:
                observation, info = env.reset(seed=20260813)
                active = np.asarray(info["rotation_active_axis_mask"], dtype=bool)
                goal = np.asarray(info["rotation_goal_euler_deg"])
                with self.subTest(gripper=gripper):
                    self.assertEqual(observation.shape, (149,))
                    self.assertEqual(
                        info["rotation_sampling_profile"], "weighted_0_90"
                    )
                    self.assertEqual(info["rotation_goal_max_deg"], 90.0)
                    self.assertGreaterEqual(int(active.sum()), 1)
                    self.assertEqual(len(set(np.abs(goal[active]).tolist())), 1)
                    np.testing.assert_array_equal(goal[~active], 0.0)
            finally:
                env.close()

    def test_rotation_lift_fraction_uses_the_shorter_phase3_distance(self) -> None:
        env = G3G6ManipulationEnv(
            gripper="g3",
            phase=config.Phase.ROTATE,
            object_name="cube",
            rotation_axes="x",
        )
        try:
            env.reset(seed=50, options={"rotation_euler_deg": (10.0, 0.0, 0.0)})
            lift_start = config.APPROACH_TICKS + config.CONTACT_TICKS
            env.step_count = lift_start + config.LIFT_TICKS // 2 - 1
            target = env._target_for_tick(env.step_count)
            expected_fraction = 0.5
            self.assertAlmostEqual(
                (target.position[2] - env.object_spec.rest_grasp_plane_z_m)
                / env.task_lift_distance_m,
                expected_fraction,
                delta=1.0e-12,
            )
            successful_solution = mock.Mock(
                success=True,
                ctrl=env.nominal_ctrl.copy(),
                finger_solutions={},
            )
            with mock.patch.object(
                env.kinematics, "solve", return_value=successful_solution
            ) as solve:
                env._nominal_for_target(target)
            expected_position = target.position.copy()
            expected_position[2] += (
                expected_fraction * config.NOMINAL_GRASP_Z_BIAS_M[env.gripper]
            )
            expected_targets = env._world_tip_targets(
                expected_position,
                target.quaternion,
                target.local_tip_centres,
            )
            for finger in FINGERS:
                np.testing.assert_allclose(
                    solve.call_args.args[0][finger],
                    expected_targets[finger],
                    rtol=0.0,
                    atol=1.0e-12,
                )
        finally:
            env.close()

    def test_exact_lift_action_and_terminal_hold_lengths(self) -> None:
        lift_end = config.APPROACH_TICKS + config.CONTACT_TICKS + config.LIFT_TICKS
        self.assertEqual(
            (config.APPROACH_TICKS, config.CONTACT_TICKS, config.LIFT_TICKS),
            (120, 80, 160),
        )
        self.assertEqual(lift_end, 360)
        self.assertEqual(config.PRE_ACTION_TICKS, lift_end)
        self.assertEqual(config.PRE_ACTION_HOLD_TICKS, 0)

        self.assertEqual(config.PHASE1_TERMINAL_HOLD_TICKS, 25)
        self.assertEqual(config.PHASE1_EPISODE_TICKS, lift_end + 25)
        self.assertEqual(config.PHASE1_EPISODE_TICKS, 385)

        self.assertEqual(
            (
                trajectories.LEAD_IN_TICKS,
                trajectories.START_SETTLE_TICKS,
                trajectories.ROUTE_TICKS,
                trajectories.TERMINAL_HOLD_TICKS,
            ),
            (300, 300, 1_800, 25),
        )
        self.assertEqual(trajectories.TOTAL_TICKS, 2_425)
        self.assertEqual(
            trajectories.PHASE_BOUNDS,
            {
                "lead_in": (0, 300),
                "settle": (300, 600),
                "route": (600, 2_400),
                "hold": (2_400, 2_425),
            },
        )
        for tick, expected in (
            (0, "lead_in"),
            (299, "lead_in"),
            (300, "settle"),
            (599, "settle"),
            (600, "route"),
            (2_399, "route"),
            (2_400, "hold"),
            (2_424, "hold"),
        ):
            with self.subTest(tick=tick):
                self.assertEqual(trajectories.phase_for_tick(tick), expected)
        with self.assertRaises(ValueError):
            trajectories.phase_for_tick(2_425)

        self.assertEqual(ROTATION_MOTION_TICKS, 2_400)
        self.assertEqual(
            config.FORMAL_EPISODE_TICKS,
            config.PRE_ACTION_TICKS + trajectories.TOTAL_TICKS,
        )
        self.assertEqual(config.FORMAL_EPISODE_TICKS, 2_785)

    def test_phase1_has_exactly_25_targets_after_the_final_lift_tick(self) -> None:
        env = G3G6ManipulationEnv(
            gripper="g3",
            phase=config.Phase.GRASP_LIFT,
            object_name="cube",
        )
        try:
            env.reset(seed=51)
            lift_end = config.APPROACH_TICKS + config.CONTACT_TICKS + config.LIFT_TICKS
            final_lift = env._target_for_tick(lift_end - 1)
            first_hold = env._target_for_tick(lift_end)
            final_hold = env._target_for_tick(env.episode_ticks - 1)
            for target in (final_lift, first_hold, final_hold):
                np.testing.assert_array_equal(
                    target.position,
                    np.array(
                        (0.0, 0.0, config.TARGET_GRASP_PLANE_Z_M),
                        dtype=np.float64,
                    ),
                )
            self.assertEqual(env.episode_ticks - lift_end, 25)
        finally:
            env.close()

    def test_phase2_starts_lead_on_tick_immediately_after_lift(self) -> None:
        env = G3G6ManipulationEnv(
            gripper="g3",
            phase=config.Phase.TRANSLATE,
            object_name="cube",
            path_name="square",
        )
        try:
            env.reset(seed=52)
            lift_end = config.APPROACH_TICKS + config.CONTACT_TICKS + config.LIFT_TICKS
            final_lift = env._target_for_tick(lift_end - 1)
            self.assertFalse(env._action_reference_captured)

            reference_input = np.array((0.013, -0.021, 0.348), dtype=np.float64)
            reference_quaternion = euler_xyz_deg_to_quat((7.0, -9.0, 16.0))
            actual_position, actual_quaternion = _set_selected_object_pose(
                env, reference_input, reference_quaternion
            )
            first_action = env._target_for_tick(lift_end)
            self.assertTrue(env._action_reference_captured)
            expected_reference = np.array(
                (
                    actual_position[0],
                    actual_position[1],
                    config.TARGET_GRASP_PLANE_Z_M,
                ),
                dtype=np.float64,
            )
            np.testing.assert_array_equal(
                final_lift.position,
                np.array(
                    (0.0, 0.0, config.TARGET_GRASP_PLANE_Z_M),
                    dtype=np.float64,
                ),
            )
            np.testing.assert_allclose(
                first_action.position,
                expected_reference + env._translation_schedule[0],
                rtol=0.0,
                atol=1.0e-15,
            )
            np.testing.assert_allclose(
                first_action.quaternion,
                actual_quaternion,
                rtol=0.0,
                atol=1.0e-15,
            )

            for action_tick in (299, 300, 599, 600, 2_399, 2_400, 2_424):
                target = env._target_for_tick(lift_end + action_tick)
                with self.subTest(action_tick=action_tick):
                    np.testing.assert_allclose(
                        target.position,
                        expected_reference + env._translation_schedule[action_tick],
                        rtol=0.0,
                        atol=1.0e-15,
                    )
            np.testing.assert_array_equal(
                env._translation_schedule[2_400:],
                np.repeat(
                    env._translation_schedule[2_400][None, :], 25, axis=0
                ),
            )
        finally:
            env.close()

    def test_phase3_starts_motion_after_lift_and_holds_goal_for_25_ticks(self) -> None:
        goal = np.array((25.0, -15.0, 35.0), dtype=np.float64)
        euler_schedule = build_rotation_euler_schedule(goal)
        self.assertEqual(euler_schedule.shape, (2_425, 3))
        np.testing.assert_allclose(
            euler_schedule[2_399], goal, rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_array_equal(
            euler_schedule[2_400:], np.repeat(goal[None, :], 25, axis=0)
        )

        env = G3G6ManipulationEnv(
            gripper="g3",
            phase=config.Phase.ROTATE,
            object_name="cube",
            rotation_axes="xyz",
        )
        try:
            env.reset(seed=53, options={"rotation_euler_deg": goal})
            lift_end = config.APPROACH_TICKS + config.CONTACT_TICKS + config.LIFT_TICKS
            env._target_for_tick(lift_end - 1)
            self.assertFalse(env._action_reference_captured)
            reference_quaternion = euler_xyz_deg_to_quat((8.0, 12.0, -19.0))
            _set_selected_object_pose(
                env,
                np.array((-0.014, 0.018, 0.347), dtype=np.float64),
                reference_quaternion,
            )

            first_action = env._target_for_tick(lift_end)
            self.assertTrue(env._action_reference_captured)
            np.testing.assert_allclose(
                first_action.quaternion,
                quat_multiply(reference_quaternion, env._rotation_quaternion_schedule[0]),
                rtol=0.0,
                atol=1.0e-15,
            )
            self.assertFalse(
                np.array_equal(first_action.quaternion, reference_quaternion)
            )

            final_motion = env._target_for_tick(lift_end + 2_399)
            first_hold = env._target_for_tick(lift_end + 2_400)
            final_hold = env._target_for_tick(lift_end + 2_424)
            np.testing.assert_allclose(
                final_motion.quaternion, first_hold.quaternion, rtol=0.0, atol=1.0e-15
            )
            np.testing.assert_allclose(
                first_hold.quaternion, final_hold.quaternion, rtol=0.0, atol=1.0e-15
            )
        finally:
            env.close()


class CurriculumToleranceAndRotationContractTests(unittest.TestCase):
    def test_fixed_phase_budgets_and_no_success_rate_gate(self) -> None:
        self.assertEqual(
            config.PHASE_TIMESTEPS,
            {
                config.Phase.GRASP_LIFT: 500_000,
                config.Phase.TRANSLATE: 2_000_000,
                config.Phase.ROTATE: 2_000_000,
            },
        )
        self.assertEqual(trajectories.PATH_NAMES, ("square", "circle", "S", "U", "N"))

        training_path = config.PROJECT_ROOT / "g3g6" / "training.py"
        training_source = training_path.read_text(encoding="utf-8")
        tree = ast.parse(training_source, filename=str(training_path))
        gate_values: list[bool] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "success_rate_gate"
                    and isinstance(value, ast.Constant)
                ):
                    gate_values.append(value.value)
        self.assertEqual(gate_values, [False])

        gated_conditions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While)):
                condition = ast.unparse(node.test).lower()
                if "success" in condition and ("rate" in condition or "gate" in condition):
                    gated_conditions.append(condition)
        self.assertEqual(gated_conditions, [])

    def test_replay_curriculum_boundary_marks_last_transition_for_every_env(self) -> None:
        from g3g6 import training

        class FakeReplay:
            def __init__(self, stored: int) -> None:
                self.pos = 0  # Exercise wraparound to buffer_size - 1.
                self.buffer_size = 5
                self.dones = np.zeros((self.buffer_size, 3), dtype=np.float32)
                self.timeouts = np.ones((self.buffer_size, 3), dtype=np.float32)
                self._stored = stored

            def size(self) -> int:
                return self._stored

        class FakeModel:
            def __init__(self, replay: FakeReplay | None) -> None:
                self.replay_buffer = replay

        replay = FakeReplay(stored=4)
        training._terminate_replay_curriculum_boundary(FakeModel(replay))
        np.testing.assert_array_equal(replay.dones[-1], np.ones(3))
        np.testing.assert_array_equal(replay.timeouts[-1], np.zeros(3))
        np.testing.assert_array_equal(replay.dones[:-1], np.zeros((4, 3)))
        np.testing.assert_array_equal(replay.timeouts[:-1], np.ones((4, 3)))

        empty = FakeReplay(stored=0)
        empty.dones[:] = 0.25
        empty.timeouts[:] = 0.75
        before_dones = empty.dones.copy()
        before_timeouts = empty.timeouts.copy()
        training._terminate_replay_curriculum_boundary(FakeModel(empty))
        np.testing.assert_array_equal(empty.dones, before_dones)
        np.testing.assert_array_equal(empty.timeouts, before_timeouts)
        training._terminate_replay_curriculum_boundary(FakeModel(None))

    def test_tolerance_is_componentwise_and_boundaries_are_inclusive(self) -> None:
        self.assertEqual(config.POSITION_TOLERANCE_M_PER_AXIS, 0.005)
        self.assertEqual(config.ANGLE_TOLERANCE_DEG_PER_AXIS, 5.0)
        self.assertTrue(
            config.componentwise_pose_within_tolerance(
                (0.005, -0.005, 0.005), (5.0, -5.0, 5.0)
            )
        )
        for axis in range(3):
            position = np.zeros(3)
            position[axis] = 0.005 + 1.0e-12
            with self.subTest(kind="position", axis=axis):
                self.assertFalse(
                    config.componentwise_pose_within_tolerance(position, np.zeros(3))
                )

            angle = np.zeros(3)
            angle[axis] = 5.0 + 1.0e-12
            with self.subTest(kind="angle", axis=axis):
                self.assertFalse(
                    config.componentwise_pose_within_tolerance(np.zeros(3), angle)
                )

    def test_all_seven_rotation_axis_groups_and_inclusive_10_to_45_degree_range(self) -> None:
        expected = ("x", "y", "z", "xy", "xz", "yz", "xyz")
        self.assertEqual(config.ROTATION_AXIS_SETS, expected)
        cardinalities: defaultdict[int, int] = defaultdict(int)
        rng = np.random.default_rng(20260810)
        for axes in expected:
            with self.subTest(axes=axes):
                self.assertEqual(validate_axis_set(axes), axes)
                cardinalities[len(axes)] += 1
                low = np.array(
                    [10.0 if axis in axes else 0.0 for axis in "xyz"],
                    dtype=np.float64,
                )
                high = np.array(
                    [-45.0 if axis in axes else 0.0 for axis in "xyz"],
                    dtype=np.float64,
                )
                np.testing.assert_array_equal(validate_goal_euler_deg(axes, low), low)
                np.testing.assert_array_equal(validate_goal_euler_deg(axes, high), high)

                selected, sampled = sample_goal_euler_deg(rng, axes)
                self.assertEqual(selected, axes)
                for index, axis in enumerate("xyz"):
                    magnitude = abs(float(sampled[index]))
                    if axis in axes:
                        self.assertGreaterEqual(magnitude, 10.0)
                        self.assertLessEqual(magnitude, 45.0)
                    else:
                        self.assertEqual(magnitude, 0.0)

                active_index = "xyz".index(axes[0])
                below = low.copy()
                below[active_index] = 10.0 - 1.0e-9
                above = high.copy()
                above[active_index] = -(45.0 + 1.0e-9)
                with self.assertRaises(ValueError):
                    validate_goal_euler_deg(axes, below)
                with self.assertRaises(ValueError):
                    validate_goal_euler_deg(axes, above)
        self.assertEqual(dict(cardinalities), {1: 3, 2: 3, 3: 1})

        seen: set[str] = set()
        rng = np.random.default_rng(7)
        for _ in range(256):
            axes, _goal = sample_goal_euler_deg(rng)
            seen.add(axes)
        self.assertEqual(seen, set(expected))

    def test_weighted_0_90_profile_has_disjoint_integer_bins_and_common_magnitude(self) -> None:
        profile = config.WEIGHTED_ROTATION_SAMPLING_PROFILE
        self.assertEqual(profile, "weighted_0_90")
        self.assertEqual(validate_rotation_sampling_profile(profile), profile)
        self.assertEqual(
            config.WEIGHTED_ROTATION_MAGNITUDE_BINS_DEG,
            ((0, 30), (31, 60), (61, 90)),
        )
        self.assertEqual(
            config.WEIGHTED_ROTATION_BIN_PROBABILITIES,
            (0.20, 0.50, 0.30),
        )
        with self.assertRaises(ValueError):
            validate_rotation_sampling_profile("weighted_10_90")

        rng = np.random.default_rng(20260811)
        counts = np.zeros(3, dtype=np.int64)
        signs: set[tuple[int, int, int]] = set()
        observed_magnitudes: set[int] = set()
        samples = 60_000
        for _ in range(samples):
            axes, goal = sample_goal_euler_deg(
                rng, "xyz", profile=profile
            )
            self.assertEqual(axes, "xyz")
            magnitudes = np.abs(goal).astype(np.int64)
            self.assertTrue(np.equal(goal, np.round(goal)).all())
            self.assertEqual(len(set(magnitudes.tolist())), 1)
            magnitude = int(magnitudes[0])
            observed_magnitudes.add(magnitude)
            if magnitude <= 30:
                counts[0] += 1
            elif magnitude <= 60:
                self.assertGreaterEqual(magnitude, 31)
                counts[1] += 1
            else:
                self.assertGreaterEqual(magnitude, 61)
                self.assertLessEqual(magnitude, 90)
                counts[2] += 1
            if magnitude > 0:
                signs.add(tuple(np.sign(goal).astype(np.int64).tolist()))

        np.testing.assert_allclose(
            counts / samples,
            config.WEIGHTED_ROTATION_BIN_PROBABILITIES,
            rtol=0.0,
            atol=0.01,
        )
        self.assertIn(0, observed_magnitudes)
        self.assertIn(30, observed_magnitudes)
        self.assertIn(31, observed_magnitudes)
        self.assertIn(60, observed_magnitudes)
        self.assertIn(61, observed_magnitudes)
        self.assertIn(90, observed_magnitudes)
        self.assertEqual(len(signs), 8)

    def test_weighted_profile_keeps_the_existing_seven_axis_choice_uniform(self) -> None:
        rng = np.random.default_rng(20260812)
        counts = {axes: 0 for axes in config.ROTATION_AXIS_SETS}
        samples = 35_000
        for _ in range(samples):
            axes, goal = sample_goal_euler_deg(
                rng,
                profile=config.WEIGHTED_ROTATION_SAMPLING_PROFILE,
            )
            counts[axes] += 1
            active_magnitudes = [
                abs(float(goal[index]))
                for index, axis in enumerate("xyz")
                if axis in axes
            ]
            self.assertEqual(len(set(active_magnitudes)), 1)
            for index, axis in enumerate("xyz"):
                if axis not in axes:
                    self.assertEqual(goal[index], 0.0)
        expected = 1.0 / len(config.ROTATION_AXIS_SETS)
        for axes, count in counts.items():
            with self.subTest(axes=axes):
                self.assertAlmostEqual(count / samples, expected, delta=0.01)


class SafeWarmupTrainingContractTests(unittest.TestCase):
    @staticmethod
    def _uninitialized_algorithm(training: object) -> object:
        """Build only the state exercised by ``_sample_action``."""

        algorithm = object.__new__(training.SafeWarmupTQC)
        action_space = mock.Mock()
        action_space.shape = (config.ACTION_DIM,)
        action_space.dtype = np.dtype(np.float32)
        policy = mock.Mock()
        policy.scale_action.side_effect = lambda action: np.array(action, copy=True)
        algorithm.action_space = action_space
        algorithm.policy = policy
        return algorithm

    def test_safe_warmup_executes_and_stores_exact_zero_through_boundary(self) -> None:
        from g3g6 import training

        learning_starts = 20_000
        n_envs = 3
        algorithm = self._uninitialized_algorithm(training)
        for num_timesteps in (0, learning_starts - 1, learning_starts):
            with self.subTest(num_timesteps=num_timesteps):
                algorithm.num_timesteps = num_timesteps
                action, buffer_action = algorithm._sample_action(
                    learning_starts,
                    action_noise=object(),
                    n_envs=n_envs,
                )
                self.assertEqual(action.shape, (n_envs, config.ACTION_DIM))
                self.assertEqual(buffer_action.shape, (n_envs, config.ACTION_DIM))
                self.assertEqual(action.dtype, np.dtype(np.float32))
                self.assertEqual(buffer_action.dtype, np.dtype(np.float32))
                np.testing.assert_array_equal(action, np.zeros_like(action))
                np.testing.assert_array_equal(
                    buffer_action, np.zeros_like(buffer_action)
                )

        self.assertEqual(algorithm.policy.scale_action.call_count, 3)
        for call in algorithm.policy.scale_action.call_args_list:
            np.testing.assert_array_equal(
                call.args[0], np.zeros((n_envs, config.ACTION_DIM), dtype=np.float32)
            )

    def test_first_tick_after_warmup_delegates_to_parent_sampler(self) -> None:
        from g3g6 import training

        learning_starts = 20_000
        n_envs = 2
        action_noise = object()
        parent_action = np.full(
            (n_envs, config.ACTION_DIM), 0.25, dtype=np.float32
        )
        parent_buffer_action = np.full(
            (n_envs, config.ACTION_DIM), -0.5, dtype=np.float32
        )
        algorithm = self._uninitialized_algorithm(training)
        algorithm.num_timesteps = learning_starts + 1

        with mock.patch.object(
            training.TQC,
            "_sample_action",
            autospec=True,
            return_value=(parent_action, parent_buffer_action),
        ) as parent_sampler:
            action, buffer_action = algorithm._sample_action(
                learning_starts,
                action_noise=action_noise,
                n_envs=n_envs,
            )

        parent_sampler.assert_called_once_with(
            algorithm, learning_starts, action_noise, n_envs
        )
        self.assertIs(action, parent_action)
        self.assertIs(buffer_action, parent_buffer_action)
        algorithm.policy.scale_action.assert_not_called()

    def test_actor_mean_helper_zeros_only_the_mean_output_layer(self) -> None:
        import torch

        from g3g6 import training

        mean_network = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, config.ACTION_DIM),
        )
        with torch.no_grad():
            mean_network[0].weight.fill_(0.125)
            mean_network[0].bias.fill_(-0.25)
            mean_network[2].weight.fill_(0.375)
            mean_network[2].bias.fill_(0.5)
        hidden_weight_before = mean_network[0].weight.detach().clone()
        hidden_bias_before = mean_network[0].bias.detach().clone()

        class Holder:
            pass

        model = Holder()
        model.policy = Holder()
        model.policy.actor = Holder()
        model.policy.actor.mu = mean_network
        training._zero_initialize_actor_mean(model)

        self.assertTrue(torch.equal(mean_network[0].weight, hidden_weight_before))
        self.assertTrue(torch.equal(mean_network[0].bias, hidden_bias_before))
        self.assertEqual(torch.count_nonzero(mean_network[2].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(mean_network[2].bias).item(), 0)
        output = mean_network(torch.ones((2, 4), dtype=torch.float32))
        self.assertEqual(torch.count_nonzero(output).item(), 0)

    def test_manifest_records_safe_residual_and_exploration_targets(self) -> None:
        from g3g6 import training

        self.assertAlmostEqual(
            float(np.degrees(config.ANGULAR_RESIDUAL_LIMIT_RAD)),
            float(np.degrees(0.005)),
            places=12,
        )
        self.assertAlmostEqual(
            float(np.degrees(config.ANGULAR_RESIDUAL_RATE_RAD_PER_TICK)),
            float(np.degrees(0.0001)),
            places=12,
        )

        class StopAfterManifest(RuntimeError):
            pass

        budgets = tuple(config.PHASE_TIMESTEPS[phase] for phase in config.Phase)
        with tempfile.TemporaryDirectory(
            prefix=".training-contract-", dir=config.PROJECT_ROOT
        ) as temporary:
            for n_envs, expected_safe_transitions in ((1, 20_001), (16, 20_016)):
                with self.subTest(n_envs=n_envs):
                    output_dir = Path(temporary) / f"n_envs_{n_envs}"
                    with mock.patch.object(
                        training,
                        "_make_vec_env",
                        side_effect=StopAfterManifest("manifest captured"),
                    ):
                        with self.assertRaisesRegex(
                            StopAfterManifest, "manifest captured"
                        ):
                            training.train_one_gripper(
                                gripper="g3",
                                phase_steps=budgets,
                                n_envs=n_envs,
                                vec_env_type="dummy",
                                seed=17,
                                device="cpu",
                                output_dir=output_dir,
                                progress_bar=False,
                                checkpoint_every=10_000,
                            )

                    manifest = json.loads(
                        (output_dir / "g3" / "manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(
                        manifest["schema"],
                        "g3g6-local-tqc-v2-hold25-safe-residual",
                    )
                    self.assertTrue(manifest["formal_fixed_budget"])
                    self.assertEqual(manifest["n_envs"], n_envs)
                    self.assertEqual(
                        manifest["episode_ticks"],
                        {"phase1": 385, "phase2": 2_785, "phase3": 2_785},
                    )
                    self.assertEqual(
                        manifest["timing_segments"],
                        {
                            "grasp_lift_prefix": {
                                "approach": 120,
                                "contact": 80,
                                "lift": 160,
                                "post_lift_wait_phase2_phase3": 0,
                            },
                            "phase1_terminal_hold": 25,
                            "phase2_action": {
                                "lead": 300,
                                "settle": 300,
                                "route": 1_800,
                                "terminal_hold": 25,
                            },
                            "phase3_action": {
                                "motion": 2_400,
                                "terminal_hold": 25,
                            },
                        },
                    )
                    self.assertEqual(
                        manifest["action"],
                        {
                            "dimension": config.ACTION_DIM,
                            "meaning": (
                                "normalized_direct_rotary_actuator_angle_residual"
                            ),
                            "limit_deg": float(np.degrees(0.005)),
                            "rate_deg_per_tick": float(np.degrees(0.0001)),
                            "g6_linear_actuators": "nominal_ik_only",
                        },
                    )
                    self.assertEqual(
                        manifest["exploration"],
                        {
                            "warmup_action": "exact_zero_residual",
                            "learning_starts": 20_000,
                            "zero_warmup_includes_boundary_vector_batch": True,
                            "safe_warmup_transitions": expected_safe_transitions,
                            "actor_mean_initialization": "zero",
                            "log_std_init": -5.0,
                            "ent_coef": "auto_0.01",
                            "gSDE_noise_refresh": "once_per_one_tick_rollout",
                        },
                    )
                    self.assertEqual(
                        manifest["residual_penalty"],
                        {
                            "normalization": "actual_residual_divided_by_limit",
                            "weight": config.RESIDUAL_PENALTY_WEIGHT,
                        },
                    )

    def test_tqc_constructor_uses_safe_exploration_parameters(self) -> None:
        training_path = config.PROJECT_ROOT / "g3g6" / "training.py"
        tree = ast.parse(
            training_path.read_text(encoding="utf-8"), filename=str(training_path)
        )
        constructor_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SafeWarmupTQC"
        ]
        self.assertEqual(len(constructor_calls), 1)
        keywords = {
            keyword.arg: keyword.value
            for keyword in constructor_calls[0].keywords
            if keyword.arg is not None
        }
        self.assertEqual(ast.literal_eval(keywords["ent_coef"]), "auto_0.01")
        self.assertEqual(ast.literal_eval(keywords["use_sde"]), True)
        # With a one-tick rollout, -1 refreshes gSDE exactly once per tick;
        # a nominal frequency of 16 would still be reset at every rollout.
        self.assertEqual(ast.literal_eval(keywords["train_freq"]), (1, "step"))
        self.assertEqual(ast.literal_eval(keywords["sde_sample_freq"]), -1)
        self.assertEqual(ast.literal_eval(keywords["use_sde_at_warmup"]), False)

        policy_kwargs = keywords["policy_kwargs"]
        self.assertIsInstance(policy_kwargs, ast.Dict)
        policy_values = {
            ast.literal_eval(key): value
            for key, value in zip(policy_kwargs.keys, policy_kwargs.values)
        }
        self.assertEqual(ast.literal_eval(policy_values["log_std_init"]), -5.0)

        zero_init_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_zero_initialize_actor_mean"
        ]
        self.assertEqual(len(zero_init_calls), 1)
        self.assertEqual(ast.unparse(zero_init_calls[0]), "_zero_initialize_actor_mean(model)")

    def test_smoke_crosses_safe_warmup_and_exercises_gradient_updates(self) -> None:
        from g3g6 import training

        expected = config.PROJECT_ROOT / "runs" / "_smoke-contract" / "g3"
        with mock.patch.object(
            training, "train_one_gripper", return_value=expected
        ) as train, mock.patch("builtins.print"):
            training.main(
                [
                    "--smoke",
                    "--gripper",
                    "g3",
                    "--output-dir",
                    "runs/_smoke-contract",
                ]
            )
        kwargs = train.call_args.kwargs
        self.assertEqual(kwargs["phase_steps"], (128, 32, 32))
        self.assertEqual(kwargs["n_envs"], 1)
        self.assertEqual(kwargs["vec_env_type"], "dummy")
        self.assertEqual(kwargs["device"], "cpu")
        self.assertGreater(kwargs["phase_steps"][0], 100)


class Phase3GoalObservationContractTests(unittest.TestCase):
    def test_signed_rotation_goal_is_normalized_visible_and_keeps_inactive_axis_zero(self) -> None:
        self.assertEqual(OBSERVATION_DIM, 149)
        goal_slice = slice(-7, -4)  # goal(3), gripper one-hot(2), IK status(2)
        first_goal = np.array((10.0, 0.0, -45.0), dtype=np.float64)
        second_goal = np.array((-30.0, 0.0, 20.0), dtype=np.float64)
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.ROTATE,
                object_name="cube",
                rotation_axes="xz",
            )
            try:
                first_observation, first_info = env.reset(
                    seed=61,
                    options={"rotation_euler_deg": first_goal},
                )
                second_observation, second_info = env.reset(
                    seed=61,
                    options={"rotation_euler_deg": second_goal},
                )
                with self.subTest(gripper=gripper):
                    self.assertEqual(first_observation.shape, (149,))
                    self.assertEqual(second_observation.shape, (149,))
                    np.testing.assert_allclose(
                        first_observation[goal_slice],
                        first_goal / config.ROTATION_ANGLE_MAX_DEG,
                        rtol=0.0,
                        atol=1.0e-7,
                    )
                    np.testing.assert_allclose(
                        second_observation[goal_slice],
                        second_goal / config.ROTATION_ANGLE_MAX_DEG,
                        rtol=0.0,
                        atol=1.0e-7,
                    )
                    self.assertEqual(float(first_observation[goal_slice][1]), 0.0)
                    self.assertEqual(float(second_observation[goal_slice][1]), 0.0)
                    self.assertFalse(
                        np.array_equal(
                            first_observation[goal_slice],
                            second_observation[goal_slice],
                        )
                    )
                    np.testing.assert_array_equal(
                        first_info["rotation_goal_euler_deg"], first_goal
                    )
                    np.testing.assert_array_equal(
                        second_info["rotation_goal_euler_deg"], second_goal
                    )
            finally:
                env.close()


class ResidualEnvironmentContractTests(unittest.TestCase):
    @staticmethod
    def _rotation_options() -> dict[str, object]:
        return {
            "object": "cube",
            "rotation_axes": "x",
            "rotation_euler_deg": (30.0, 0.0, 0.0),
        }

    def test_common_149d_observation_pads_only_the_g3_residual(self) -> None:
        residual_slice = slice(64, 80)
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.GRASP_LIFT,
                object_name="cube",
            )
            try:
                env.reset(seed=900)
                native = np.linspace(0.001, 0.001 * env.action_dim, env.action_dim)
                env.last_residual = native.copy()
                observation = env._observation()
                expected = np.zeros(16, dtype=np.float32)
                expected[: env.action_dim] = native
                with self.subTest(gripper=gripper):
                    self.assertEqual(observation.shape, (149,))
                    np.testing.assert_allclose(
                        observation[residual_slice], expected, rtol=0.0, atol=1.0e-9
                    )
            finally:
                env.close()

    def test_zero_residual_uses_the_same_feedback_nominal_as_step_ik(self) -> None:
        for gripper in ("g3", "g6"):
            for tick in (0, config.PRE_ACTION_TICKS):
                residual_env = G3G6ManipulationEnv(
                    gripper=gripper,
                    phase=config.Phase.ROTATE,
                    rotation_axes="x",
                    residual_nominal_feedback=True,
                    residual_after_preaction_only=True,
                )
                ik_env = G3G6ManipulationEnv(
                    gripper=gripper,
                    phase=config.Phase.ROTATE,
                    rotation_axes="x",
                )
                try:
                    residual_env.reset(seed=901, options=self._rotation_options())
                    ik_env.reset(seed=901, options=self._rotation_options())
                    residual_env.step_count = tick
                    ik_env.step_count = tick
                    with mock.patch("g3g6.env.mujoco.mj_step", return_value=None):
                        residual_result = residual_env.step(
                            np.zeros(residual_env.action_dim, dtype=np.float32)
                        )
                        ik_result = ik_env.step_ik(feedback=True)
                    residual_info = residual_result[4]
                    ik_info = ik_result[4]
                    with self.subTest(gripper=gripper, tick=tick):
                        np.testing.assert_allclose(
                            residual_env.ik_command_position,
                            ik_env.ik_command_position,
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                        np.testing.assert_allclose(
                            residual_env.ik_command_quaternion,
                            ik_env.ik_command_quaternion,
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                        np.testing.assert_allclose(
                            residual_env.nominal_ctrl,
                            ik_env.nominal_ctrl,
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                        np.testing.assert_array_equal(
                            residual_env.applied_ctrl, residual_env.nominal_ctrl
                        )
                        np.testing.assert_array_equal(
                            residual_info["actuator_residual"],
                            np.zeros(residual_env.action_dim),
                        )
                        self.assertEqual(
                            residual_info["residual_active"],
                            tick >= config.PRE_ACTION_TICKS,
                        )
                        self.assertEqual(
                            residual_info["position_error_m_per_axis"].shape,
                            ik_info["position_error_m_per_axis"].shape,
                        )
                finally:
                    residual_env.close()
                    ik_env.close()

    def test_preaction_mask_and_info_distinguish_requested_and_executed_action(self) -> None:
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.ROTATE,
                rotation_axes="x",
                residual_nominal_feedback=True,
                residual_after_preaction_only=True,
                rotation_sampling_profile=(
                    config.WEIGHTED_ROTATION_SAMPLING_PROFILE
                ),
            )
            try:
                env.reset(seed=902, options=self._rotation_options())
                requested = np.linspace(-2.0, 2.0, env.action_dim)
                with (
                    mock.patch("g3g6.env.mujoco.mj_step", return_value=None),
                    mock.patch.object(
                        env,
                        "_nominal_for_target",
                        return_value=env.nominal_ctrl.copy(),
                    ),
                ):
                    _obs, _reward, _terminated, _truncated, inactive = env.step(
                        requested
                    )
                with self.subTest(gripper=gripper, stage="preaction"):
                    self.assertFalse(inactive["residual_active"])
                    np.testing.assert_array_equal(
                        inactive["residual_requested_action"], requested
                    )
                    np.testing.assert_array_equal(
                        inactive["residual_executed_action"],
                        np.zeros(env.action_dim),
                    )
                    np.testing.assert_array_equal(
                        inactive["actuator_residual"], np.zeros(env.action_dim)
                    )
                    self.assertEqual(
                        inactive["rotation_sampling_profile"], "weighted_0_90"
                    )

                env.reset(seed=902, options=self._rotation_options())
                env.step_count = config.PRE_ACTION_TICKS
                with (
                    mock.patch("g3g6.env.mujoco.mj_step", return_value=None),
                    mock.patch.object(
                        env,
                        "_nominal_for_target",
                        return_value=env.nominal_ctrl.copy(),
                    ),
                ):
                    _obs, _reward, _terminated, _truncated, active = env.step(
                        requested
                    )
                with self.subTest(gripper=gripper, stage="action"):
                    self.assertTrue(active["residual_active"])
                    np.testing.assert_array_equal(
                        active["residual_executed_action"],
                        np.clip(requested, -1.0, 1.0),
                    )
                    self.assertGreater(
                        float(np.max(np.abs(active["actuator_residual"]))), 0.0
                    )
            finally:
                env.close()


class PostLiftActionReferenceContractTests(unittest.TestCase):
    @staticmethod
    def _phase_options(phase: config.Phase) -> dict[str, object]:
        if phase == config.Phase.TRANSLATE:
            return {"object": "cube", "path": "square"}
        return {
            "object": "cube",
            "rotation_axes": "xyz",
            "rotation_euler_deg": (10.0, -20.0, 30.0),
        }

    def test_phase2_schedule_remains_relative_to_the_measured_post_lift_pose(self) -> None:
        actual_input_position = np.array((0.017, -0.023, 0.341), dtype=np.float64)
        actual_input_quaternion = euler_xyz_deg_to_quat((12.0, -8.0, 23.0))
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.TRANSLATE,
                object_name="cube",
                path_name="square",
            )
            try:
                _observation, reset_info = env.reset(seed=71)
                self.assertFalse(reset_info["action_reference_captured"])
                relative_schedule = trajectories.scheduled_path("square")
                np.testing.assert_array_equal(env._translation_schedule, relative_schedule)
                np.testing.assert_array_equal(
                    env._translation_schedule[:, 2],
                    np.zeros(trajectories.TOTAL_TICKS, dtype=np.float64),
                )

                actual_position, actual_quaternion = _set_selected_object_pose(
                    env, actual_input_position, actual_input_quaternion
                )
                env.step_count = config.PRE_ACTION_TICKS
                target = env._target_for_tick(env.step_count)

                expected_reference = np.array(
                    (
                        actual_position[0],
                        actual_position[1],
                        config.TARGET_GRASP_PLANE_Z_M,
                    ),
                    dtype=np.float64,
                )
                self.assertTrue(env._action_reference_captured)
                np.testing.assert_array_equal(
                    env._action_reference_position, expected_reference
                )
                np.testing.assert_allclose(
                    env._action_reference_quaternion,
                    actual_quaternion,
                    rtol=0.0,
                    atol=1.0e-15,
                )
                np.testing.assert_allclose(
                    target.position,
                    expected_reference + relative_schedule[0],
                    rtol=0.0,
                    atol=1.0e-15,
                )
                self.assertEqual(
                    float(target.position[2]), config.TARGET_GRASP_PLANE_Z_M
                )
                np.testing.assert_allclose(
                    target.quaternion, actual_quaternion, rtol=0.0, atol=1.0e-15
                )

                # Capturing is one-shot: later object drift must not rebase
                # the remainder of the trajectory.
                captured_position = env._action_reference_position.copy()
                captured_quaternion = env._action_reference_quaternion.copy()
                _set_selected_object_pose(
                    env,
                    np.array((-0.091, 0.084, 0.390), dtype=np.float64),
                    euler_xyz_deg_to_quat((-25.0, 14.0, -9.0)),
                )
                env.step_count += 1
                next_target = env._target_for_tick(env.step_count)
                np.testing.assert_array_equal(
                    env._action_reference_position, captured_position
                )
                np.testing.assert_array_equal(
                    env._action_reference_quaternion, captured_quaternion
                )
                np.testing.assert_allclose(
                    next_target.position,
                    captured_position + relative_schedule[1],
                    rtol=0.0,
                    atol=1.0e-15,
                )
                self.assertEqual(
                    float(next_target.position[2]), config.TARGET_GRASP_PLANE_Z_M
                )
            finally:
                env.close()

    def test_phase3_first_action_tick_captures_xy_and_orientation_but_pins_z(self) -> None:
        actual_input_position = np.array((-0.031, 0.026, 0.349), dtype=np.float64)
        actual_input_quaternion = euler_xyz_deg_to_quat((-17.0, 11.0, 28.0))
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.ROTATE,
                object_name="cube",
                rotation_axes="xyz",
            )
            try:
                env.reset(
                    seed=72,
                    options={"rotation_euler_deg": (10.0, -20.0, 30.0)},
                )
                actual_position, actual_quaternion = _set_selected_object_pose(
                    env, actual_input_position, actual_input_quaternion
                )
                env.step_count = config.PRE_ACTION_TICKS
                target = env._target_for_tick(env.step_count)

                expected_position = np.array(
                    (
                        actual_position[0],
                        actual_position[1],
                        config.ROTATION_TARGET_GRASP_PLANE_Z_M,
                    ),
                    dtype=np.float64,
                )
                expected_quaternion = quat_multiply(
                    actual_quaternion, env._rotation_quaternion_schedule[0]
                )
                self.assertTrue(env._action_reference_captured)
                np.testing.assert_array_equal(
                    env._action_reference_position, expected_position
                )
                np.testing.assert_allclose(
                    env._action_reference_quaternion,
                    actual_quaternion,
                    rtol=0.0,
                    atol=1.0e-15,
                )
                np.testing.assert_array_equal(target.position, expected_position)
                self.assertEqual(
                    float(target.position[2]),
                    config.ROTATION_TARGET_GRASP_PLANE_Z_M,
                )
                np.testing.assert_allclose(
                    target.quaternion,
                    expected_quaternion,
                    rtol=0.0,
                    atol=1.0e-15,
                )
            finally:
                env.close()

    def test_reset_clears_a_previously_captured_action_reference(self) -> None:
        moved_position = np.array((0.014, 0.019, 0.337), dtype=np.float64)
        moved_quaternion = euler_xyz_deg_to_quat((9.0, 13.0, -21.0))
        for gripper in ("g3", "g6"):
            for phase in (config.Phase.TRANSLATE, config.Phase.ROTATE):
                with self.subTest(gripper=gripper, phase=int(phase)):
                    env = G3G6ManipulationEnv(gripper=gripper, phase=phase)
                    options = self._phase_options(phase)
                    try:
                        env.reset(seed=73, options=options)
                        _set_selected_object_pose(
                            env, moved_position, moved_quaternion
                        )
                        env.step_count = config.PRE_ACTION_TICKS
                        env._target_for_tick(env.step_count)
                        self.assertTrue(env._action_reference_captured)

                        _observation, info = env.reset(seed=74, options=options)
                        self.assertFalse(env._action_reference_captured)
                        self.assertFalse(info["action_reference_captured"])
                        expected_target_z = (
                            config.ROTATION_TARGET_GRASP_PLANE_Z_M
                            if phase == config.Phase.ROTATE
                            else config.TARGET_GRASP_PLANE_Z_M
                        )
                        np.testing.assert_array_equal(
                            env._action_reference_position,
                            np.array(
                                (0.0, 0.0, expected_target_z),
                                dtype=np.float64,
                            ),
                        )
                        np.testing.assert_array_equal(
                            env._action_reference_quaternion,
                            np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
                        )
                        self.assertEqual(env.step_count, 0)
                    finally:
                        env.close()

    def test_cone_action_cage_uses_pose_relative_to_captured_post_lift_pose(self) -> None:
        reference_quaternion = euler_xyz_deg_to_quat((11.0, -17.0, 23.0))
        for gripper in ("g3", "g6"):
            for phase in (config.Phase.TRANSLATE, config.Phase.ROTATE):
                with self.subTest(gripper=gripper, phase=int(phase)):
                    reference_position = np.array(
                        (
                            0.012,
                            -0.018,
                            config.ROTATION_TARGET_GRASP_PLANE_Z_M
                            if phase == config.Phase.ROTATE
                            else config.TARGET_GRASP_PLANE_Z_M,
                        ),
                        dtype=np.float64,
                    )
                    if phase == config.Phase.TRANSLATE:
                        env = G3G6ManipulationEnv(
                            gripper=gripper,
                            phase=phase,
                            object_name="cone",
                            path_name="square",
                        )
                        reset_options: dict[str, object] = {}
                        action_tick = 0
                    else:
                        env = G3G6ManipulationEnv(
                            gripper=gripper,
                            phase=phase,
                            object_name="cone",
                            rotation_axes="xyz",
                        )
                        reset_options = {
                            "rotation_euler_deg": (25.0, -15.0, 35.0)
                        }
                        action_tick = 1_200
                    try:
                        env.reset(seed=75, options=reset_options)
                        env._action_reference_captured = True
                        env._action_reference_position[:] = reference_position
                        env._action_reference_quaternion[:] = reference_quaternion
                        env.step_count = config.PRE_ACTION_TICKS + action_tick
                        target = env._target_for_tick(env.step_count)

                        if phase == config.Phase.TRANSLATE:
                            expected_cage_quaternion = np.array(
                                (1.0, 0.0, 0.0, 0.0), dtype=np.float64
                            )
                            np.testing.assert_allclose(
                                target.quaternion,
                                reference_quaternion,
                                rtol=0.0,
                                atol=1.0e-15,
                            )
                        else:
                            expected_cage_quaternion = (
                                env._rotation_quaternion_schedule[action_tick].copy()
                            )
                            np.testing.assert_allclose(
                                target.quaternion,
                                quat_multiply(
                                    reference_quaternion,
                                    expected_cage_quaternion,
                                ),
                                rtol=0.0,
                                atol=1.0e-15,
                            )

                        successful_solution = mock.Mock(
                            success=True,
                            ctrl=env.nominal_ctrl.copy(),
                            finger_solutions={},
                        )
                        with mock.patch.object(
                            env.kinematics,
                            "solve",
                            return_value=successful_solution,
                        ) as solve:
                            env._nominal_for_target(target)
                        ik_targets = {
                            finger: np.asarray(value).copy()
                            for finger, value in solve.call_args.args[0].items()
                        }

                        command_position = target.position.copy()
                        command_position[2] += config.NOMINAL_GRASP_Z_BIAS_M[
                            gripper
                        ]
                        command_position[:2] -= reference_position[:2]
                        expected_targets = env._world_tip_targets(
                            command_position,
                            expected_cage_quaternion,
                            target.local_tip_centres,
                        )
                        wrongly_reapplied_reference = env._world_tip_targets(
                            command_position,
                            target.quaternion,
                            target.local_tip_centres,
                        )
                        for finger in FINGERS:
                            np.testing.assert_allclose(
                                ik_targets[finger],
                                expected_targets[finger],
                                rtol=0.0,
                                atol=1.0e-12,
                            )
                        self.assertGreater(
                            max(
                                float(
                                    np.linalg.norm(
                                        ik_targets[finger]
                                        - wrongly_reapplied_reference[finger]
                                    )
                                )
                                for finger in FINGERS
                            ),
                            1.0e-3,
                            "captured facet yaw was applied to the cone cage twice",
                        )
                    finally:
                        env.close()


class StabilityGuardContractTests(unittest.TestCase):
    @staticmethod
    def _selected_free_joint_addresses(
        env: G3G6ManipulationEnv,
    ) -> tuple[int, int]:
        joint_id = env._object_joint_ids[env.object_name]
        return (
            int(env.model.jnt_qposadr[joint_id]),
            int(env.model.jnt_dofadr[joint_id]),
        )

    def test_guard_covers_passive_robot_object_and_global_nonfinite_state(self) -> None:
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.GRASP_LIFT,
                object_name="cube",
            )
            try:
                env.reset(seed=91)
                passive_qvel = sorted(
                    set(map(int, env._robot_qvel_adrs))
                    - set(map(int, env._actuator_qvel_adrs))
                )
                self.assertGreater(len(passive_qvel), 0)
                env.data.qvel[passive_qvel[0]] = config.MAX_ABS_QVEL + 1.0
                with self.subTest(gripper=gripper, state="passive_robot_qvel"):
                    self.assertEqual(
                        env._failure_reason(), "unstable_joint_velocity"
                    )

                env.reset(seed=92)
                _object_qpos, object_dof = self._selected_free_joint_addresses(env)
                env.data.qvel[object_dof] = (
                    config.MAX_OBJECT_LINEAR_SPEED_M_S + 1.0
                )
                with self.subTest(gripper=gripper, state="object_linear_qvel"):
                    self.assertEqual(
                        env._failure_reason(),
                        "unstable_object_linear_velocity",
                    )

                env.reset(seed=93)
                _object_qpos, object_dof = self._selected_free_joint_addresses(env)
                env.data.qvel[object_dof + 3] = (
                    config.MAX_OBJECT_ANGULAR_SPEED_RAD_S + 1.0
                )
                with self.subTest(gripper=gripper, state="object_angular_qvel"):
                    self.assertEqual(
                        env._failure_reason(),
                        "unstable_object_angular_velocity",
                    )

                # A NaN in an inactive/offstage object still belongs to the
                # global simulator state and must not escape the guard.
                env.reset(seed=94)
                offstage_joint = env._object_joint_ids["cone"]
                offstage_qpos = int(env.model.jnt_qposadr[offstage_joint])
                env.data.qpos[offstage_qpos] = np.nan
                with self.subTest(gripper=gripper, state="global_qpos_nan"):
                    self.assertEqual(env._failure_reason(), "non_finite_state")

                env.reset(seed=95)
                offstage_joint = env._object_joint_ids["cone"]
                offstage_dof = int(env.model.jnt_dofadr[offstage_joint])
                env.data.qvel[offstage_dof] = np.nan
                with self.subTest(gripper=gripper, state="global_qvel_nan"):
                    self.assertEqual(env._failure_reason(), "non_finite_state")

                env.reset(seed=96)
                object_qpos, _object_dof = self._selected_free_joint_addresses(env)
                env.data.qpos[object_qpos + 3 : object_qpos + 7] = 0.0
                with self.subTest(gripper=gripper, state="invalid_quaternion"):
                    self.assertEqual(
                        env._failure_reason(), "invalid_object_quaternion"
                    )
            finally:
                env.close()

    def test_nan_from_physics_returns_a_finite_terminal_observation(self) -> None:
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.GRASP_LIFT,
                object_name="cube",
            )
            try:
                env.reset(seed=97)
                saved_qpos = env._last_finite_qpos.copy()
                saved_qvel = env._last_finite_qvel.copy()

                def inject_nan(*_args: object, **_kwargs: object) -> None:
                    env.data.qpos[0] = np.nan
                    env.data.qvel[-1] = np.nan

                with (
                    mock.patch(
                        "g3g6.env.mujoco.mj_step", side_effect=inject_nan
                    ),
                    mock.patch.object(
                        env,
                        "_nominal_for_target",
                        return_value=env.nominal_ctrl.copy(),
                    ),
                ):
                    observation, reward, terminated, truncated, info = env.step(
                        np.zeros(env.action_dim, dtype=np.float32)
                    )

                with self.subTest(gripper=gripper):
                    self.assertTrue(terminated)
                    self.assertFalse(truncated)
                    self.assertEqual(info["termination_reason"], "non_finite_state")
                    self.assertFalse(info["is_success"])
                    self.assertTrue(np.isfinite(observation).all())
                    self.assertTrue(np.isfinite(reward))
                    self.assertTrue(np.isfinite(env.data.qpos).all())
                    self.assertTrue(np.isfinite(env.data.qvel).all())
                    np.testing.assert_allclose(
                        env.data.qpos, saved_qpos, rtol=0.0, atol=1.0e-12
                    )
                    np.testing.assert_allclose(
                        env.data.qvel, saved_qvel, rtol=0.0, atol=1.0e-12
                    )
            finally:
                env.close()

    def test_nan_nominal_fails_before_physics_and_restores_all_controller_state(self) -> None:
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.GRASP_LIFT,
                object_name="cube",
            )
            try:
                env.reset(seed=98)
                angular = np.asarray(
                    config.ANGULAR_ACTUATOR_INDICES[gripper], dtype=np.int32
                )
                env.nominal_ctrl = 0.5 * (env.ctrl_low + env.ctrl_high)
                distinctive_residual = np.linspace(
                    -0.25 * config.ANGULAR_RESIDUAL_RATE_RAD_PER_TICK,
                    0.25 * config.ANGULAR_RESIDUAL_RATE_RAD_PER_TICK,
                    env.action_dim,
                    dtype=np.float64,
                )
                env.applied_ctrl = env.nominal_ctrl.copy()
                residual_indices = np.asarray(
                    config.RESIDUAL_ACTUATOR_INDICES[gripper], dtype=np.int32
                )
                env.applied_ctrl[residual_indices] += distinctive_residual
                env.last_residual = distinctive_residual.copy()
                env.residual_controller.residual[:] = distinctive_residual
                env.data.ctrl[:] = env.applied_ctrl

                saved_nominal = env.nominal_ctrl.copy()
                saved_applied = env.applied_ctrl.copy()
                saved_last = env.last_residual.copy()
                saved_internal = env.residual_controller.residual.copy()
                saved_qpos = env.data.qpos.copy()
                saved_qvel = env.data.qvel.copy()

                with (
                    mock.patch.object(
                        env,
                        "_nominal_for_target",
                        return_value=np.full(env.model.nu, np.nan),
                    ),
                    mock.patch("g3g6.env.mujoco.mj_step") as physics_step,
                ):
                    observation, reward, terminated, truncated, info = env.step(
                        np.ones(env.action_dim, dtype=np.float32)
                    )

                with self.subTest(gripper=gripper):
                    physics_step.assert_not_called()
                    self.assertTrue(terminated)
                    self.assertFalse(truncated)
                    self.assertEqual(
                        info["termination_reason"], "non_finite_controller"
                    )
                    self.assertFalse(info["is_success"])
                    self.assertTrue(np.isfinite(reward))
                    self.assertTrue(np.isfinite(observation).all())
                    np.testing.assert_array_equal(env.nominal_ctrl, saved_nominal)
                    np.testing.assert_array_equal(env.applied_ctrl, saved_applied)
                    np.testing.assert_array_equal(env.last_residual, saved_last)
                    np.testing.assert_array_equal(
                        env.residual_controller.residual, saved_internal
                    )
                    np.testing.assert_array_equal(env.data.ctrl, saved_applied)
                    np.testing.assert_array_equal(env.data.qpos, saved_qpos)
                    np.testing.assert_array_equal(env.data.qvel, saved_qvel)
                    self.assertTrue(np.isfinite(env.nominal_ctrl).all())
                    self.assertTrue(np.isfinite(env.applied_ctrl).all())
                    self.assertTrue(np.isfinite(env.last_residual).all())
                    self.assertTrue(
                        np.isfinite(env.residual_controller.residual).all()
                    )
                    self.assertTrue(np.isfinite(env.data.ctrl).all())
            finally:
                env.close()


class DropAndEpisodeEndContractTests(unittest.TestCase):
    @staticmethod
    def _static_step(
        env: G3G6ManipulationEnv,
        *,
        endpoint_success: bool,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        # The test is about Gym termination flags, not integration or IK.  A
        # no-op physics call keeps the directly injected free-body pose exact.
        with (
            mock.patch(
                "g3g6.env.mujoco.mj_step",
                return_value=None,
            ),
            mock.patch.object(
                env,
                "_nominal_for_target",
                return_value=env.nominal_ctrl.copy(),
            ),
            mock.patch.object(
                env,
                "_endpoint_success",
                return_value=endpoint_success,
            ),
        ):
            return env.step(np.zeros(env.action_dim, dtype=np.float32))

    def test_drop_height_is_object_specific_and_inactive_until_lift_completion(self) -> None:
        self.assertEqual(config.DROP_MIN_LIFT_FRACTION, 0.25)
        lift_end = config.APPROACH_TICKS + config.CONTACT_TICKS + config.LIFT_TICKS
        identity = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper, phase=config.Phase.GRASP_LIFT
            )
            try:
                for object_name, spec in OBJECTS.items():
                    with self.subTest(gripper=gripper, object=object_name):
                        env.reset(seed=81, options={"object": object_name})
                        threshold = (
                            spec.rest_grasp_plane_z_m
                            + 0.25 * spec.lift_distance_m
                        )
                        below = threshold - 1.0e-8

                        _set_selected_object_pose(
                            env, np.array((0.0, 0.0, below)), identity
                        )
                        env.step_count = lift_end - 1
                        self.assertIsNone(env._failure_reason())

                        env.step_count = lift_end
                        self.assertEqual(env._failure_reason(), "object_dropped")

                        _set_selected_object_pose(
                            env, np.array((0.0, 0.0, threshold)), identity
                        )
                        self.assertIsNone(env._failure_reason())
                        _set_selected_object_pose(
                            env,
                            np.array((0.0, 0.0, threshold + 1.0e-8)),
                            identity,
                        )
                        self.assertIsNone(env._failure_reason())
            finally:
                env.close()

    def test_rotation_drop_height_uses_the_phase3_lift_distance(self) -> None:
        lift_end = config.APPROACH_TICKS + config.CONTACT_TICKS + config.LIFT_TICKS
        identity = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.ROTATE,
                rotation_axes="x",
            )
            try:
                for object_name, spec in OBJECTS.items():
                    env.reset(
                        seed=81,
                        options={
                            "object": object_name,
                            "rotation_euler_deg": (10.0, 0.0, 0.0),
                        },
                    )
                    threshold = (
                        spec.rest_grasp_plane_z_m
                        + config.DROP_MIN_LIFT_FRACTION * env.task_lift_distance_m
                    )
                    legacy_threshold = (
                        spec.rest_grasp_plane_z_m
                        + config.DROP_MIN_LIFT_FRACTION * spec.lift_distance_m
                    )
                    with self.subTest(gripper=gripper, object=object_name):
                        self.assertLess(threshold, legacy_threshold)
                        _set_selected_object_pose(
                            env,
                            np.array((0.0, 0.0, threshold - 1.0e-8)),
                            identity,
                        )
                        env.step_count = lift_end
                        self.assertEqual(env._failure_reason(), "object_dropped")
                        _set_selected_object_pose(
                            env, np.array((0.0, 0.0, threshold)), identity
                        )
                        self.assertIsNone(env._failure_reason())
            finally:
                env.close()

    def test_rotation_height_reward_uses_the_phase3_lift_distance(self) -> None:
        identity = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.ROTATE,
                rotation_axes="x",
            )
            try:
                for object_name, spec in OBJECTS.items():
                    env.reset(
                        seed=81,
                        options={
                            "object": object_name,
                            "rotation_euler_deg": (10.0, 0.0, 0.0),
                        },
                    )
                    position = np.array(
                        (
                            0.0,
                            0.0,
                            spec.rest_grasp_plane_z_m
                            + 0.5 * env.task_lift_distance_m,
                        ),
                        dtype=np.float64,
                    )
                    with (
                        mock.patch.object(
                            env,
                            "_pose_errors",
                            return_value=(np.zeros(3), np.zeros(3)),
                        ),
                        mock.patch.object(
                            env, "_contact_mask", return_value=np.zeros(4)
                        ),
                        mock.patch.object(
                            env, "_selected_pose", return_value=(position, identity)
                        ),
                    ):
                        reward = env._reward(
                            endpoint_success=False,
                            failure=None,
                            actual_residual=np.zeros(env.action_dim),
                        )
                    with self.subTest(gripper=gripper, object=object_name):
                        self.assertAlmostEqual(
                            reward,
                            2.0 + 0.5 * config.HEIGHT_REWARD_WEIGHT,
                            delta=1.0e-12,
                        )
            finally:
                env.close()

    def test_residual_regularizer_matches_each_translation_reference(self) -> None:
        identity = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.GRASP_LIFT,
                object_name="cube",
            )
            try:
                env.reset(seed=811)
                position = np.array(
                    (0.0, 0.0, env.task_target_grasp_plane_z_m),
                    dtype=np.float64,
                )
                normalized_ratio = np.full(env.action_dim, 0.5)
                residual = normalized_ratio * env.residual_controller.limits
                with (
                    mock.patch.object(
                        env,
                        "_pose_errors",
                        return_value=(np.zeros(3), np.zeros(3)),
                    ),
                    mock.patch.object(
                        env, "_contact_mask", return_value=np.zeros(4)
                    ),
                    mock.patch.object(
                        env, "_selected_pose", return_value=(position, identity)
                    ),
                ):
                    reward = env._reward(False, None, residual)
                base_reward = 2.0 + config.HEIGHT_REWARD_WEIGHT
                expected_penalty = (
                    config.G3_RESIDUAL_PENALTY_WEIGHT * 0.5
                    if gripper == "g3"
                    else config.G6_RESIDUAL_PENALTY_WEIGHT * 0.25
                )
                with self.subTest(gripper=gripper):
                    self.assertAlmostEqual(
                        reward, base_reward - expected_penalty, delta=1.0e-12
                    )
            finally:
                env.close()

    def test_intrinsic_horizon_and_drop_terminate_but_success_never_ends_early(self) -> None:
        identity = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        lift_end = config.APPROACH_TICKS + config.CONTACT_TICKS + config.LIFT_TICKS
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(
                gripper=gripper,
                phase=config.Phase.GRASP_LIFT,
                object_name="cube",
            )
            try:
                # Even if the endpoint predicate would report success, it is
                # not an early-termination condition before the fixed horizon.
                env.reset(seed=82)
                _set_selected_object_pose(
                    env,
                    np.array((0.0, 0.0, config.TARGET_GRASP_PLANE_Z_M)),
                    identity,
                )
                env.step_count = lift_end
                _obs, _reward, terminated, truncated, info = self._static_step(
                    env, endpoint_success=True
                )
                with self.subTest(gripper=gripper, outcome="early_success"):
                    self.assertFalse(terminated)
                    self.assertFalse(truncated)
                    self.assertFalse(info["endpoint"])
                    self.assertFalse(info["is_success"])
                    self.assertEqual(info["termination_reason"], "running")

                # Time progress is part of the observation and endpoint
                # success exists only here, so this is an intrinsic terminal
                # state rather than an external TimeLimit truncation.
                env.reset(seed=83)
                _set_selected_object_pose(
                    env,
                    np.array((0.0, 0.0, config.TARGET_GRASP_PLANE_Z_M)),
                    identity,
                )
                env.step_count = env.episode_ticks - 1
                _obs, _reward, terminated, truncated, info = self._static_step(
                    env, endpoint_success=True
                )
                with self.subTest(gripper=gripper, outcome="horizon"):
                    self.assertTrue(terminated)
                    self.assertFalse(truncated)
                    self.assertTrue(info["endpoint"])
                    self.assertTrue(info["is_success"])
                    self.assertEqual(info["termination_reason"], "horizon")

                # A drop has failure precedence over endpoint success even
                # when it coincides with the intrinsic final tick.
                env.reset(seed=84)
                drop_threshold = (
                    env.object_spec.rest_grasp_plane_z_m
                    + config.DROP_MIN_LIFT_FRACTION
                    * env.object_spec.lift_distance_m
                )
                _set_selected_object_pose(
                    env,
                    np.array((0.0, 0.0, drop_threshold - 0.001)),
                    identity,
                )
                env.step_count = env.episode_ticks - 1
                _obs, _reward, terminated, truncated, info = self._static_step(
                    env, endpoint_success=True
                )
                with self.subTest(gripper=gripper, outcome="drop"):
                    self.assertTrue(terminated)
                    self.assertFalse(truncated)
                    self.assertTrue(info["endpoint"])
                    self.assertFalse(info["is_success"])
                    self.assertEqual(info["termination_reason"], "object_dropped")
            finally:
                env.close()


class EnvironmentResetContractTests(unittest.TestCase):
    def test_g3_and_g6_reset_every_object_and_random_sampling_covers_all_four(self) -> None:
        # These seeds map to cone, cylinder, triangular prism, and cube with
        # NumPy's Generator and the authoritative OBJECT_NAMES ordering.
        coverage_seeds = (0, 1, 4, 11)
        for gripper in ("g3", "g6"):
            env = G3G6ManipulationEnv(gripper=gripper, phase=config.Phase.GRASP_LIFT)
            try:
                self.assertEqual(
                    env.action_space.shape, (config.action_dim(gripper),)
                )
                for object_name in config.OBJECT_NAMES:
                    with self.subTest(gripper=gripper, object=object_name):
                        observation, info = env.reset(
                            seed=12345, options={"object": object_name}
                        )
                        self.assertEqual(env.object_name, object_name)
                        self.assertEqual(info["object"], object_name)
                        self.assertEqual(observation.shape, (OBSERVATION_DIM,))
                        self.assertTrue(np.isfinite(observation).all())
                        np.testing.assert_allclose(
                            info["object_grasp_frame_position_m"],
                            (0.0, 0.0, OBJECTS[object_name].rest_grasp_plane_z_m),
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                        np.testing.assert_allclose(
                            info["target_position_m"],
                            (0.0, 0.0, OBJECTS[object_name].rest_grasp_plane_z_m),
                            rtol=0.0,
                            atol=1.0e-12,
                        )

                sampled = {
                    env.reset(seed=seed)[1]["object"] for seed in coverage_seeds
                }
                self.assertEqual(sampled, set(config.OBJECT_NAMES))
                self.assertEqual(env.step_count, 0)
            finally:
                env.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
