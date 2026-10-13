#!/usr/bin/env python3
"""Geometry, XML-contract, and relocation tests for the SCREW task."""

from __future__ import annotations

import importlib.util
import math
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from screw_config import (
    CAMERA_NAMES,
    CONTROLLER_STRATEGY,
    DEFAULT_CONFIG,
    SCREW_BODY_NAME,
    SCREW_FREE_JOINT_NAME,
    SCREW_GRASP_SITE_NAME,
    SCREW_HEAD_GEOM_NAME,
    SCREW_SHAFT_GEOM_NAME,
    SCREW_START_HOLDER_EQUALITY_NAME,
    SCREW_TIP_SITE_NAME,
    SOCKET_BOTTOM_SITE_NAME,
    SOCKET_MOUTH_SITE_NAME,
    ScrewConfigError,
    socket_axis_for_table_angle_deg,
)
from grasp_plane import (
    AdaptiveScrewGraspPlanePlanner,
    GraspPlane,
    default_screw_grasp_plan,
)
from screw_scene import (
    DEFAULT_PROJECT_XML,
    SpawnSamplingError,
    _socket_segment_vertices,
    build_scene_xml,
    helical_insertion_depth,
    initial_grasp_position,
    quaternion_to_matrix,
    rotate_vector,
    sample_socket_mouth_xy,
    sample_spawn_xy,
    screw_pose_for_insertion,
    socket_center_position,
    socket_projected_aabb_xy,
    socket_quaternion,
    upright_screw_overlaps_socket,
    write_scene_xml,
)


HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
MODULE_DIR = Path(__file__).resolve().parent


def assert_vector_close(
    testcase: unittest.TestCase,
    actual,
    expected,
    tolerance: float = 1e-10,
) -> None:
    testcase.assertEqual(len(actual), len(expected))
    for got, wanted in zip(actual, expected):
        testcase.assertAlmostEqual(float(got), float(wanted), delta=tolerance)


class SequenceRNG:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


class ConfigGeometryTests(unittest.TestCase):
    def test_confirmed_dimensions_and_full_insertion_coupling(self):
        cfg = DEFAULT_CONFIG
        self.assertAlmostEqual(2.0 * cfg.screw_shaft_radius_m, 0.030)
        self.assertAlmostEqual(cfg.screw_shaft_length_m, 0.080)
        self.assertAlmostEqual(2.0 * cfg.screw_head_radius_m, 0.050)
        self.assertAlmostEqual(cfg.screw_head_height_m, 0.040)
        self.assertAlmostEqual(cfg.screw_total_height_m, 0.120)
        self.assertAlmostEqual(cfg.screw_thread_length_m, 0.020)
        self.assertAlmostEqual(2.0 * cfg.socket_hole_radius_m, 0.040)
        self.assertAlmostEqual(cfg.socket_hole_depth_m, 0.080)
        self.assertAlmostEqual(cfg.socket_thread_length_m, 0.020)
        self.assertEqual(
            (
                cfg.socket_outer_width_m,
                cfg.socket_outer_height_m,
                cfg.socket_outer_length_m,
            ),
            (0.080, 0.080, 0.100),
        )
        self.assertAlmostEqual(cfg.thread_pitch_m, 0.005)
        self.assertAlmostEqual(cfg.target_turns, 2.0)
        self.assertAlmostEqual(cfg.target_advance_m, 0.010)
        self.assertAlmostEqual(cfg.thread_engagement_depth_m, 0.070)
        self.assertAlmostEqual(cfg.target_insertion_depth_m, 0.080)
        self.assertAlmostEqual(cfg.radial_clearance_m, 0.005)
        self.assertAlmostEqual(cfg.socket_mouth_orbit_radius_m, 0.020)
        self.assertAlmostEqual(
            cfg.socket_mouth_height_above_table_m, 0.070
        )
        self.assertAlmostEqual(cfg.scripted_grasp_plane_world_z_m, 0.420)
        self.assertAlmostEqual(cfg.scripted_center_tip_world_z_m, 0.320)
        self.assertEqual(cfg.scripted_center_xy_m, (0.0, 0.0))
        self.assertAlmostEqual(
            cfg.released_tip_center_distance_m, 0.049
        )
        self.assertAlmostEqual(cfg.socket_yaw_roll_gain, 0.8)
        self.assertEqual(cfg.socket_yaw_equivariance_version, 1)
        self.assertEqual(
            CONTROLLER_STRATEGY["pose_transform"], "Q R Q^T"
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["align_boundary_step"], 365
        )
        self.assertEqual(CONTROLLER_STRATEGY["align_steps"], 150)

    def test_invalid_pitch_turn_depth_contract_is_rejected(self):
        with self.assertRaisesRegex(
            ScrewConfigError,
            "pitch × turns",
        ):
            DEFAULT_CONFIG.with_overrides(thread_pitch_m=0.004)

    def test_table_mouth_and_axis(self):
        cfg = DEFAULT_CONFIG
        self.assertAlmostEqual(cfg.table_top_z_m, 0.260)
        assert_vector_close(
            self, cfg.socket_mouth_position_m, (0.020, 0.0, 0.330)
        )
        assert_vector_close(
            self,
            cfg.socket_axis_world,
            (math.sqrt(0.5), 0.0, -math.sqrt(0.5)),
        )
        expected_center = (
            0.020 + 0.050 * math.sqrt(0.5),
            0.0,
            0.330 - 0.050 * math.sqrt(0.5),
        )
        assert_vector_close(self, socket_center_position(cfg), expected_center)

    def test_fixed_45_degree_socket_embedding_and_screw_clearance(self):
        cfg = DEFAULT_CONFIG
        socket_center = np.asarray(socket_center_position(cfg))
        lowest_outer_corner = min(
            float(
                (
                    socket_center
                    + np.asarray(
                        rotate_vector(socket_quaternion(cfg), vertex)
                    )
                )[2]
            )
            for index in range(cfg.socket_segment_count)
            for vertex in _socket_segment_vertices(index, cfg)
        )
        angle = math.radians(cfg.socket_table_angle_deg)
        lowest_bore_surface = (
            cfg.socket_mouth_position_m[2]
            - cfg.socket_hole_depth_m * math.sin(angle)
            - cfg.socket_hole_radius_m * math.cos(angle)
        )
        lowest_screw_envelope = (
            cfg.socket_mouth_position_m[2]
            - cfg.target_insertion_depth_m * math.sin(angle)
            - cfg.screw_shaft_radius_m * math.cos(angle)
        )
        self.assertAlmostEqual(lowest_outer_corner, 0.231005, places=6)
        self.assertLess(lowest_outer_corner, cfg.table_top_z_m)
        self.assertLess(lowest_bore_surface, cfg.table_top_z_m)
        self.assertGreater(lowest_screw_envelope, cfg.table_top_z_m)

    def test_fixed_45_degree_socket_keeps_opening_facing_centre(self):
        cfg = DEFAULT_CONFIG.with_socket_table_angle(45.0)
        expected_axis = (math.sqrt(0.5), 0.0, -math.sqrt(0.5))
        assert_vector_close(
            self,
            socket_axis_for_table_angle_deg(45.0),
            expected_axis,
        )
        self.assertAlmostEqual(cfg.socket_table_angle_deg, 45.0)
        center = socket_center_position(cfg)
        # Its outward normal is the vector from body centre to mouth and its
        # horizontal component points toward world centre.
        body_to_mouth = tuple(
            mouth - body
            for mouth, body in zip(cfg.socket_mouth_position_m, center)
        )
        assert_vector_close(
            self,
            body_to_mouth,
            tuple(
                -0.5 * cfg.socket_outer_length_m * component
                for component in cfg.socket_axis_world
            ),
        )

        with self.assertRaisesRegex(
            ScrewConfigError, "fixed at 45 degrees"
        ):
            DEFAULT_CONFIG.with_socket_table_angle(90.0)

        north = DEFAULT_CONFIG.with_socket_pose((0.0, 0.020), 45.0)
        assert_vector_close(
            self,
            north.socket_axis_world,
            (0.0, math.sqrt(0.5), -math.sqrt(0.5)),
        )
        assert_vector_close(
            self, north.socket_opening_azimuth_xy, (0.0, -1.0)
        )
        self.assertAlmostEqual(
            math.hypot(*north.socket_mouth_xy_m), 0.020
        )

    def test_grab_style_plane_respects_full_insertion_clearance(self):
        planner = AdaptiveScrewGraspPlanePlanner()
        plan = default_screw_grasp_plan()
        self.assertEqual(len(list(planner.candidate_planes())), 1)
        self.assertTrue(plan.valid)
        self.assertAlmostEqual(plan.plane.center_local[2], 0.060)
        self.assertAlmostEqual(plan.grasp_height_from_tip_m, 0.100)
        self.assertAlmostEqual(plan.plane.roll_rad, 0.0)
        self.assertAlmostEqual(plan.plane.pitch_rad, 0.0)
        self.assertAlmostEqual(plan.plane.yaw_rad, 0.0)
        self.assertAlmostEqual(
            plan.diagnostics["outside_socket_at_full_insertion_m"],
            0.020,
        )
        self.assertAlmostEqual(plan.contact_radius_m, 0.041)
        self.assertAlmostEqual(plan.pregrasp_radius_m, 0.053)
        self.assertAlmostEqual(
            plan.diagnostics["pregrasp_diagonal_error_m"], 0.0
        )
        lower = GraspPlane(
            center_local=np.array((0.0, 0.0, 0.040)),
            roll_rad=0.0,
            pitch_rad=0.0,
        )
        rejected = planner._build_candidate(
            lower,
            ik=None,
            body_world_pos=None,
            body_world_quat=(1.0, 0.0, 0.0, 0.0),
            table_top_z=DEFAULT_CONFIG.table_top_z_m,
            table_clearance_m=0.002,
        )
        self.assertFalse(rejected.valid)
        self.assertEqual(
            rejected.diagnostics["reason"],
            "full_insertion_fingertip_socket_clearance",
        )

    def test_socket_quaternion_maps_local_z_to_inward_axis(self):
        mapped = rotate_vector(socket_quaternion(), (0.0, 0.0, 1.0))
        assert_vector_close(self, mapped, DEFAULT_CONFIG.socket_axis_world)
        matrix = quaternion_to_matrix(socket_quaternion())
        # Rotation matrix remains right-handed and orthonormal.
        for row in matrix:
            self.assertAlmostEqual(sum(value * value for value in row), 1.0)

    def test_upright_head_grasp_geometry(self):
        cfg = DEFAULT_CONFIG
        self.assertAlmostEqual(cfg.screw_head_center_local_z_m, 0.060)
        self.assertAlmostEqual(cfg.screw_grasp_local_z_m, 0.060)
        self.assertAlmostEqual(cfg.screw_grasp_height_from_tip_m, 0.100)
        grasp = initial_grasp_position((-0.075, -0.040), cfg)
        assert_vector_close(self, grasp, (-0.075, -0.040, 0.360))

    def test_two_turn_helix_and_fully_inserted_pose(self):
        cfg = DEFAULT_CONFIG
        angle = 4.0 * math.pi
        self.assertAlmostEqual(helical_insertion_depth(angle, cfg), 0.080)
        pose = screw_pose_for_insertion(0.080, angle, cfg)
        expected_tip = tuple(
            mouth + axis * cfg.socket_hole_depth_m
            for mouth, axis in zip(
                cfg.socket_mouth_position_m,
                cfg.socket_axis_world,
            )
        )
        assert_vector_close(self, pose.tip_position_m, expected_tip)

        outward = tuple(-value for value in cfg.socket_axis_world)
        head_underface = tuple(
            center + outward_component * 0.5 * cfg.screw_shaft_length_m
            for center, outward_component in zip(pose.position_m, outward)
        )
        assert_vector_close(
            self,
            head_underface,
            cfg.socket_mouth_position_m,
        )
        grasp_offset = tuple(
            grasp - mouth
            for grasp, mouth in zip(
                pose.grasp_position_m,
                cfg.socket_mouth_position_m,
            )
        )
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in grasp_offset)),
            0.020,
        )

    def test_convex_segments_are_square_outside_and_20mm_inradius(self):
        cfg = DEFAULT_CONFIG
        step = 2.0 * math.pi / cfg.socket_segment_count
        all_vertices = []
        for index in range(cfg.socket_segment_count):
            vertices = _socket_segment_vertices(index, cfg)
            self.assertEqual(len(vertices), 8)
            all_vertices.extend(vertices)
            inner0 = vertices[0]
            inner1 = vertices[3]
            midpoint = (
                0.5 * (inner0[0] + inner1[0]),
                0.5 * (inner0[1] + inner1[1]),
            )
            self.assertAlmostEqual(math.hypot(*midpoint), cfg.socket_hole_radius_m)
            self.assertAlmostEqual(
                math.hypot(inner0[0], inner0[1]),
                cfg.socket_hole_radius_m / math.cos(0.5 * step),
            )
        self.assertAlmostEqual(min(v[0] for v in all_vertices), -0.040)
        self.assertAlmostEqual(max(v[0] for v in all_vertices), 0.040)
        self.assertAlmostEqual(min(v[1] for v in all_vertices), -0.040)
        self.assertAlmostEqual(max(v[1] for v in all_vertices), 0.040)
        self.assertAlmostEqual(min(v[2] for v in all_vertices), -0.050)
        self.assertAlmostEqual(max(v[2] for v in all_vertices), 0.050)


class SpawnTests(unittest.TestCase):
    def test_socket_mouth_samples_uniform_azimuth_on_20mm_orbit(self):
        assert_vector_close(
            self,
            sample_socket_mouth_xy(SequenceRNG((0.0,))),
            (0.020, 0.0),
        )
        assert_vector_close(
            self,
            sample_socket_mouth_xy(SequenceRNG((0.25,))),
            (0.0, 0.020),
        )

    def test_default_reset_is_in_disk_and_clear_of_socket(self):
        cfg = DEFAULT_CONFIG
        self.assertLessEqual(math.hypot(*cfg.default_screw_xy_m), 0.100)
        self.assertTrue(cfg.spawn_is_on_opening_side(cfg.default_screw_xy_m))
        self.assertFalse(upright_screw_overlaps_socket(cfg.default_screw_xy_m))
        xmin, xmax, ymin, ymax = socket_projected_aabb_xy()
        self.assertLess(xmin, xmax)
        self.assertLess(ymin, ymax)

    def test_radial_draw_uses_sqrt_for_area_uniformity(self):
        # sqrt(0.81) × 0.1 = 0.09; theta=pi puts it clear of the block.
        xy = sample_spawn_xy(SequenceRNG((0.81, 0.50)))
        assert_vector_close(self, xy, (-0.090, 0.0))

    def test_ik_rejection_resamples_and_receives_head_grasp_site(self):
        # First candidate (-90, 0) mm is rejected by IK. The second remains
        # in the opening-facing half disk at theta=3*pi/4.
        rng = SequenceRNG((0.81, 0.50, 0.64, 0.25))
        calls = []

        def reachable(grasp):
            calls.append(grasp)
            return len(calls) >= 2

        xy = sample_spawn_xy(rng, ik_reachable=reachable)
        assert_vector_close(
            self,
            xy,
            (-0.080 / math.sqrt(2.0), 0.080 / math.sqrt(2.0)),
        )
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(calls[0][2], 0.360)
        self.assertAlmostEqual(calls[1][2], 0.360)

    def test_impossible_reachability_fails_boundedly(self):
        cfg = DEFAULT_CONFIG.with_overrides(spawn_max_attempts=2)
        rng = SequenceRNG((0.81, 0.50, 0.64, 0.25))
        with self.assertRaises(SpawnSamplingError):
            sample_spawn_xy(rng, cfg, ik_reachable=lambda _grasp: False)

    def test_spawn_never_crosses_into_socket_body_half(self):
        for angular_draw in (0.0, 0.25, 0.5, 0.75, 1.0):
            xy = sample_spawn_xy(SequenceRNG((1.0, angular_draw)))
            self.assertLessEqual(xy[0], 1.0e-12)
            self.assertTrue(DEFAULT_CONFIG.spawn_is_on_opening_side(xy))


class XMLContractTests(unittest.TestCase):
    def test_static_project_is_canonical_builder_output(self):
        self.assertEqual(
            DEFAULT_PROJECT_XML.read_text(encoding="utf-8"),
            build_scene_xml(),
        )

    def test_xml_names_markers_and_collision_prefixes(self):
        root = ET.fromstring(build_scene_xml())
        screw = root.find(".//body[@name='screw']")
        socket = root.find(".//body[@name='socket']")
        table = root.find(".//body[@name='table']")
        self.assertIsNotNone(screw)
        self.assertIsNotNone(socket)
        self.assertIsNotNone(table)
        self.assertIsNotNone(root.find(".//joint[@name='screw_free']"))
        for site_name in (
            SCREW_TIP_SITE_NAME,
            SCREW_GRASP_SITE_NAME,
            SOCKET_MOUTH_SITE_NAME,
            SOCKET_BOTTOM_SITE_NAME,
        ):
            self.assertIsNotNone(root.find(f".//site[@name='{site_name}']"))
        self.assertIsNotNone(
            root.find(f".//geom[@name='{SCREW_SHAFT_GEOM_NAME}']")
        )
        head = root.find(f".//geom[@name='{SCREW_HEAD_GEOM_NAME}']")
        self.assertIsNotNone(head)
        self.assertEqual(head.get("pos"), "0 0 0.06")
        marker_top = root.find(".//geom[@name='screw_marker_top']")
        marker_side = root.find(".//geom[@name='screw_marker_side']")
        self.assertEqual(marker_top.get("type"), "box")
        self.assertEqual(marker_side.get("type"), "box")
        self.assertNotEqual(marker_top.get("rgba"), DEFAULT_CONFIG.screw_rgba)

        collision_socket_geoms = [
            geom
            for geom in socket.findall("geom")
            if geom.get("contype") != "0"
        ]
        self.assertEqual(
            len(
                [
                    geom
                    for geom in collision_socket_geoms
                    if geom.get("name", "").startswith("socket_")
                ]
            ),
            len(collision_socket_geoms),
        )
        self.assertEqual(len(root.findall(".//camera")), 2)
        table_visual = table.find("geom[@name='table_geom']")
        self.assertEqual(table_visual.get("contype"), "0")
        self.assertEqual(table_visual.get("conaffinity"), "0")
        table_collision = table.find("geom[@name='table_collision']")
        self.assertIsNotNone(table_collision)
        self.assertEqual(
            table_collision.get("size"), table_visual.get("size")
        )
        self.assertEqual(
            tuple(camera.get("name") for camera in root.findall(".//camera")),
            CAMERA_NAMES,
        )

    @unittest.skipUnless(HAS_MUJOCO, "MuJoCo is not installed")
    def test_embedded_socket_collision_masks_preserve_only_required_contacts(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(DEFAULT_PROJECT_XML))
        tip_id = model.geom("hard_tip_U").id
        long_link_id = model.geom("long_U").id
        ordinary_link_id = model.geom("short_U").id
        screw_ids = (
            model.geom(SCREW_SHAFT_GEOM_NAME).id,
            model.geom(SCREW_HEAD_GEOM_NAME).id,
        )
        socket_wall_id = model.geom("socket_wall_00").id
        socket_bottom_id = model.geom("socket_bottom_collision").id
        table_id = model.geom("table_collision").id

        def pair_enabled(first: int, second: int) -> bool:
            return bool(
                (
                    int(model.geom_contype[first])
                    & int(model.geom_conaffinity[second])
                )
                or (
                    int(model.geom_contype[second])
                    & int(model.geom_conaffinity[first])
                )
            )

        self.assertEqual(int(model.geom_contype[tip_id]), 2)
        self.assertEqual(int(model.geom_conaffinity[tip_id]), 7)
        self.assertEqual(int(model.geom_contype[table_id]), 64)
        self.assertEqual(int(model.geom_conaffinity[table_id]), 34)
        root_body_id = model.body("root").id

        def belongs_to_robot(geom_id: int) -> bool:
            body_id = int(model.geom_bodyid[geom_id])
            while body_id > 0:
                if body_id == root_body_id:
                    return True
                body_id = int(model.body_parentid[body_id])
            return False

        robot_geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if belongs_to_robot(geom_id)
        ]
        self.assertGreater(len(robot_geom_ids), 0)
        for obstacle_id in (socket_wall_id, socket_bottom_id, table_id):
            if obstacle_id != table_id:
                self.assertEqual(int(model.geom_contype[obstacle_id]), 8)
                self.assertEqual(int(model.geom_conaffinity[obstacle_id]), 38)
            for robot_geom_id in robot_geom_ids:
                name = model.geom(robot_geom_id).name
                should_collide = name.startswith("long_") or name.startswith(
                    "hard_tip_"
                )
                self.assertEqual(
                    pair_enabled(robot_geom_id, obstacle_id),
                    should_collide,
                    (name, model.geom(obstacle_id).name),
                )
        for screw_id in screw_ids:
            self.assertFalse(pair_enabled(screw_id, table_id))
            self.assertTrue(pair_enabled(screw_id, socket_wall_id))
            self.assertTrue(pair_enabled(screw_id, socket_bottom_id))
            self.assertTrue(pair_enabled(tip_id, screw_id))

        self.assertTrue(pair_enabled(tip_id, table_id))
        self.assertTrue(pair_enabled(long_link_id, table_id))
        self.assertTrue(pair_enabled(tip_id, socket_wall_id))
        self.assertTrue(pair_enabled(long_link_id, socket_wall_id))
        self.assertFalse(pair_enabled(ordinary_link_id, table_id))
        self.assertFalse(pair_enabled(ordinary_link_id, socket_wall_id))
        # The low socket deliberately intersects the tabletop without
        # generating a static-static collision seam.
        self.assertFalse(pair_enabled(socket_wall_id, table_id))

    @unittest.skipUnless(HAS_MUJOCO, "MuJoCo is not installed")
    def test_confirmed_root_and_stationary_screw_heights(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(DEFAULT_PROJECT_XML))
        self.assertAlmostEqual(model.body("root").pos[2], 0.620)
        screw_qadr = model.joint(SCREW_FREE_JOINT_NAME).qposadr[0]
        self.assertAlmostEqual(model.qpos0[screw_qadr + 2], 0.300)
        self.assertAlmostEqual(DEFAULT_CONFIG.socket_mouth_position_m[2], 0.330)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert_vector_close(
            self,
            data.site(SOCKET_MOUTH_SITE_NAME).xpos,
            DEFAULT_CONFIG.socket_mouth_position_m,
        )

    @unittest.skipUnless(HAS_MUJOCO, "MuJoCo is not installed")
    def test_static_project_loads_and_holder_starts_active(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(DEFAULT_PROJECT_XML))
        self.assertGreaterEqual(model.body(SCREW_BODY_NAME).id, 0)
        self.assertGreaterEqual(model.joint(SCREW_FREE_JOINT_NAME).id, 0)
        for site_name in (
            SCREW_TIP_SITE_NAME,
            SCREW_GRASP_SITE_NAME,
            SOCKET_MOUTH_SITE_NAME,
            SOCKET_BOTTOM_SITE_NAME,
        ):
            self.assertGreaterEqual(model.site(site_name).id, 0)
        equality_id = model.equality(SCREW_START_HOLDER_EQUALITY_NAME).id
        self.assertEqual(int(model.eq_active0[equality_id]), 1)
        self.assertEqual(model.ncam, 2)

    @unittest.skipUnless(HAS_MUJOCO, "MuJoCo is not installed")
    def test_runtime_xml_loads_from_unrelated_cwd(self):
        import mujoco

        with tempfile.TemporaryDirectory(prefix=".geometry_test_", dir=MODULE_DIR) as tmp:
            scene_path = write_scene_xml(Path(tmp) / "nested" / "scene.xml")
            original_cwd = Path.cwd()
            try:
                os.chdir("/")
                model = mujoco.MjModel.from_xml_path(str(scene_path))
            finally:
                os.chdir(original_cwd)
            self.assertGreaterEqual(model.body(SCREW_BODY_NAME).id, 0)
            xml = scene_path.read_text(encoding="utf-8")
            self.assertIn('include file="../../Share.xml"', xml)
            self.assertIn('meshdir="../../mesh"', xml)


if __name__ == "__main__":
    unittest.main(verbosity=2)
