from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from screw_control import (
    CAPTURE_GATE_STEP,
    PREINSERT_GATE_STEP,
    SCRIPTED_PREFIX_STEPS,
    THREAD_GATE_STEP,
    THREAD_REGRASP_RELEASED_DISTANCE_M,
    THREAD_REGRASP_STEP,
    YAW_EQUIVARIANT_ALIGN_BOUNDARY_STEP,
    YAW_EQUIVARIANT_ALIGN_STEPS,
    YAW_EQUIVARIANT_RELEASED_DISTANCE_M,
    YAW_EQUIVARIANT_ROLL_BOUNDARY_STEP,
    YAW_EQUIVARIANT_ROLL_GAIN,
    YAW_EQUIVARIANT_ROLL_STEPS,
    YAW_EQUIVARIANT_STRATEGY_VERSION,
)
from s_hpc.config import (
    HPCConfig,
    MIN_RELIABLE_CURRICULUM_SPAWN_RADIUS_M,
    canonicalize_config_payload,
    load_config,
    validate_config,
)
from preflight import _assert_smoke_physics_contract
from s_hpc.contracts import (
    ACTION_DIM,
    CONTROLLER_STRATEGY,
    DIFFUSION_SCHEMA_HASH,
    OBSERVATION_NAMES,
    OBS_DIM,
    RESIDUAL_ACTION_DIM,
    SCHEMA_HASH,
    TQC_CONTROLLER_STATE_NAMES,
    TQC_OBSERVATION_NAMES,
    TQC_OBS_DIM,
    TQC_SCHEMA_HASH,
    TASK_ASSET_HASHES,
    TASK_GEOMETRY_HASH,
    checkpoint_metadata,
    validate_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PRE_CAPTURE_TASK_HASH = (
    "6c58a20628994b80be59b012b45ba938c0616dfea3bd732175147f4a73ee0cd1"
)


class ContractsAndConfigTest(unittest.TestCase):
    @staticmethod
    def _collides(first, second):
        first_type, first_affinity = first
        second_type, second_affinity = second
        return bool(
            (first_type & second_affinity)
            or (second_type & first_affinity)
        )

    def test_contract_dimensions_and_hash_are_stable(self):
        self.assertEqual(len(OBSERVATION_NAMES), OBS_DIM)
        self.assertEqual(ACTION_DIM, 14)
        self.assertEqual(OBS_DIM, 191)
        self.assertEqual(RESIDUAL_ACTION_DIM, 16)
        self.assertEqual(TQC_OBS_DIM, 242)
        self.assertEqual(len(TQC_OBSERVATION_NAMES), TQC_OBS_DIM)
        self.assertEqual(
            TQC_OBSERVATION_NAMES[-3:], TQC_CONTROLLER_STATE_NAMES
        )
        self.assertEqual(SCHEMA_HASH, DIFFUSION_SCHEMA_HASH)
        self.assertNotEqual(DIFFUSION_SCHEMA_HASH, TQC_SCHEMA_HASH)
        self.assertEqual(len(SCHEMA_HASH), 64)
        self.assertEqual(len(TASK_GEOMETRY_HASH), 64)
        self.assertEqual(
            set(TASK_ASSET_HASHES),
            {"Project_hpc.xml", "Gripper.xml", "Share_hpc.xml"},
        )
        self.assertTrue(
            all(len(value) == 64 for value in TASK_ASSET_HASHES.values())
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["version"],
            YAW_EQUIVARIANT_STRATEGY_VERSION,
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["released_distance_m"],
            YAW_EQUIVARIANT_RELEASED_DISTANCE_M,
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["roll_gain"],
            YAW_EQUIVARIANT_ROLL_GAIN,
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["align_boundary_step"],
            YAW_EQUIVARIANT_ALIGN_BOUNDARY_STEP,
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["align_steps"],
            YAW_EQUIVARIANT_ALIGN_STEPS,
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["roll_boundary_step"],
            YAW_EQUIVARIANT_ROLL_BOUNDARY_STEP,
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["roll_steps"],
            YAW_EQUIVARIANT_ROLL_STEPS,
        )
        expected_roll_application = (
            "disabled-direct-final-frame"
            if YAW_EQUIVARIANT_ROLL_GAIN == 0.0
            else "post-interpolation-right-multiply"
        )
        self.assertIn(
            expected_roll_application,
            CONTROLLER_STRATEGY["roll_application"],
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["yaw_decomposition"],
            "diagnostic-only;actual-socket-yaw-used-directly",
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["finger_role_mapping"],
            (
                "identity-physical-U/D/L/R;yaw-invariant;"
                "material-fixed-until-measured-thread-engagement"
            ),
        )
        self.assertEqual(
            CONTROLLER_STRATEGY[
                "thread_regrasp_released_distance_m"
            ],
            THREAD_REGRASP_RELEASED_DISTANCE_M,
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["scripted_grasp_height_from_tip_m"],
            0.110,
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["diffusion_to_anchor_blend"],
            {
                "steps": 20,
                "profile": "minimum-jerk",
                "rotation": "so3-geodesic",
                "routes": (
                    "preinsert-gate",
                    "capture-gate",
                    "thread-gate",
                    "thread-regrasp-gate",
                    "regrasp-phases",
                    "contact-recovery",
                    "unmeasured-engage",
                ),
            },
        )
        self.assertEqual(
            CONTROLLER_STRATEGY["thread_regrasp_height"],
            {
                "from_tip_m": 0.112,
                "switch_step": THREAD_REGRASP_STEP + 1,
                "mode": "direct-at-regrasp",
            },
        )
        capture = CONTROLLER_STRATEGY["capture_handoff"]
        self.assertEqual(
            capture["preinsert_gate_step"], PREINSERT_GATE_STEP
        )
        self.assertEqual(capture["capture_gate_step"], CAPTURE_GATE_STEP)
        self.assertEqual(capture["thread_gate_step"], THREAD_GATE_STEP)
        self.assertEqual(
            capture["thread_regrasp_step"], THREAD_REGRASP_STEP
        )
        self.assertLess(PREINSERT_GATE_STEP, CAPTURE_GATE_STEP)
        self.assertLess(CAPTURE_GATE_STEP, THREAD_GATE_STEP)
        self.assertLess(THREAD_GATE_STEP, THREAD_REGRASP_STEP)
        self.assertEqual(capture["capture_depth_m"], 0.030)
        self.assertEqual(capture["head_bottom_standoff_m"], 0.050)
        self.assertEqual(capture["all_four_contact_stable_steps"], 5)
        self.assertEqual(capture["contact_force_threshold_n"], 0.10)
        self.assertEqual(
            capture["contact_force_release_threshold_n"], 0.04
        )
        self.assertEqual(capture["contact_loss_grace_steps"], 20)
        self.assertEqual(
            capture["minimum_contact_count_before_capture"], 3
        )
        self.assertEqual(capture["capture_stable_steps"], 5)
        self.assertEqual(
            capture["minimum_contact_count_after_capture"], 2
        )
        self.assertEqual(
            capture["commanded_prethread_grasp"],
            "all-four-at-0.039m",
        )
        self.assertEqual(
            capture["diagnostic_primary_pair"], "physical-L/R"
        )
        self.assertEqual(
            capture["material_angles"],
            "fixed-from-pickup-through-thread-gate",
        )
        self.assertEqual(capture["thread_engagement_depth_m"], 0.070)
        self.assertEqual(capture["thread_depth_tolerance_m"], 0.0015)
        self.assertEqual(capture["thread_stable_steps"], 5)
        self.assertIs(capture["prethread_regrasp"], False)
        self.assertNotEqual(
            TASK_GEOMETRY_HASH, LEGACY_PRE_CAPTURE_TASK_HASH
        )

    def test_both_configs_validate(self):
        smoke = load_config(ROOT / "configs" / "smoke.json")
        formal = load_config(ROOT / "configs" / "hx1_a100.json")
        cx3 = load_config(ROOT / "configs" / "cx3_l40s.json")
        aggressive = load_config(
            ROOT / "configs" / "cx3_l40s_8192.json"
        )
        autodl = load_config(
            ROOT / "configs" / "autodl_rtx_pro_6000_blackwell.json"
        )
        extreme = load_config(
            ROOT / "configs" / "autodl_rtx_pro_6000_extreme.json"
        )
        autodl_w18000 = load_config(
            ROOT / "configs" / "autodl_rtx_pro_6000_h105_w18000.json"
        )
        hx1_w14000 = load_config(
            ROOT / "configs" / "hx1_a100_w14000_diffusion_only.json"
        )
        self.assertEqual(smoke.physics.worlds, 8)
        self.assertEqual(formal.physics.worlds, 1024)
        self.assertEqual(formal.physics.physics_substeps, 40)
        self.assertEqual(smoke.physics.physics_substeps, 40)
        self.assertEqual(smoke.physics.max_episode_steps, 4000)
        self.assertEqual(formal.physics.max_episode_steps, 4000)
        self.assertEqual(cx3.physics.max_episode_steps, 4000)
        self.assertEqual(autodl.physics.max_episode_steps, 4000)
        self.assertEqual(extreme.physics.max_episode_steps, 4000)
        self.assertEqual(autodl_w18000.physics.worlds, 18000)
        self.assertEqual(hx1_w14000.physics.worlds, 14000)
        self.assertEqual(hx1_w14000.physics.max_episode_steps, 4000)
        self.assertEqual(hx1_w14000.demos, autodl_w18000.demos)
        self.assertEqual(hx1_w14000.diffusion, autodl_w18000.diffusion)
        self.assertTrue(smoke.physics.cuda_graph)
        self.assertTrue(formal.physics.cuda_graph)
        self.assertEqual(smoke.runtime.status_interval_seconds, 2.0)
        self.assertEqual(formal.runtime.status_interval_seconds, 10.0)
        self.assertEqual(cx3.name, "cx3_l40s")
        self.assertEqual(cx3.physics.worlds, 1024)
        self.assertEqual(aggressive.name, "cx3_l40s_8192")
        self.assertEqual(aggressive.physics.worlds, 8192)
        self.assertEqual(aggressive.diffusion, cx3.diffusion)
        self.assertEqual(aggressive.tqc, cx3.tqc)
        self.assertEqual(cx3.curriculum.fixed_socket_angle_deg, 45.0)
        self.assertEqual(formal.curriculum.fixed_socket_angle_deg, 45.0)
        self.assertEqual(smoke.curriculum.fixed_socket_angle_deg, 45.0)
        for profile in (
            smoke,
            formal,
            cx3,
            aggressive,
            autodl,
            extreme,
            autodl_w18000,
            hx1_w14000,
        ):
            self.assertEqual(
                profile.curriculum.start_spawn_radius_m,
                MIN_RELIABLE_CURRICULUM_SPAWN_RADIUS_M,
            )
        self.assertEqual(cx3.diffusion.epochs, 30)
        self.assertEqual(cx3.diffusion.batch_size, 2048)
        self.assertEqual(cx3.diffusion.ema_decay, 0.999)
        # These three archived profiles retain their original capacity-study
        # TQC settings; the new formal actuator-residual profile is isolated.
        self.assertEqual(cx3.tqc.batch_size, 1024)
        self.assertEqual(cx3.tqc.replay_ratio, 1.0)
        self.assertEqual(cx3.tqc.critics, 2)
        self.assertEqual(cx3.tqc.quantiles, 25)
        self.assertEqual(cx3.tqc.hidden_dim, 512)
        self.assertEqual(cx3.tqc.actor_lr, 3.0e-4)
        self.assertEqual(
            autodl.name, "autodl_rtx_pro_6000_blackwell"
        )
        self.assertEqual(autodl.physics.worlds, 1024)
        self.assertEqual(autodl.diffusion, cx3.diffusion)
        self.assertEqual(autodl.tqc, cx3.tqc)
        self.assertEqual(
            extreme.name,
            "autodl_rtx_pro_6000_extreme_v7_diffusion_only_spawn60_all4_min2_threadgate",
        )
        self.assertEqual(extreme.physics.worlds, 1000)
        self.assertEqual(extreme.demos.episodes, 8192)
        self.assertEqual(extreme.diffusion.epochs, 60)
        self.assertEqual(extreme.diffusion.batch_size, 4096)
        self.assertEqual(extreme.tqc.batch_size, 1000)
        self.assertEqual(
            smoke.physics.contact_sensor_maxmatch,
            extreme.physics.contact_sensor_maxmatch,
        )
        self.assertEqual(
            smoke.ik.linear_samples, extreme.ik.linear_samples
        )
        _assert_smoke_physics_contract(extreme, smoke)

    def test_smoke_physics_contract_rejects_ik_drift(self):
        smoke = load_config(ROOT / "configs" / "smoke.json")
        formal = load_config(
            ROOT / "configs" / "autodl_rtx_pro_6000_extreme.json"
        )
        changed = replace(
            smoke,
            ik=replace(
                smoke.ik,
                linear_samples=smoke.ik.linear_samples - 2,
            ),
        )
        with self.assertRaisesRegex(
            RuntimeError, "smoke physical contract"
        ):
            _assert_smoke_physics_contract(formal, changed)

    def test_default_profile_uses_formal_actuator_residual_tqc(self):
        default = HPCConfig()
        self.assertEqual(default.tqc.batch_size, 256)
        self.assertEqual(default.tqc.replay_ratio, 8.0)
        self.assertEqual(default.tqc.replay_insert_per_vector_step, 256)
        self.assertEqual(default.tqc.gradient_steps_per_vector_step, 8)
        self.assertEqual(default.tqc.critics, 4)
        self.assertEqual(default.tqc.quantiles, 32)
        self.assertEqual(default.tqc.hidden_dim, 256)
        self.assertEqual(default.tqc.actor_lr, 1.0e-4)
        self.assertEqual(
            default.curriculum.start_spawn_radius_m,
            MIN_RELIABLE_CURRICULUM_SPAWN_RADIUS_M,
        )
        self.assertTrue(
            default.curriculum.restrict_screw_spawn_to_opening_half_disk
        )

    def test_full_workspace_spawn_switch_loads_and_validates_strict_bool(self):
        payload = HPCConfig().to_dict()
        payload["curriculum"][
            "restrict_screw_spawn_to_opening_half_disk"
        ] = False
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "full_disc.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_config(path)
        self.assertFalse(
            loaded.curriculum.restrict_screw_spawn_to_opening_half_disk
        )

        changed = replace(
            HPCConfig(),
            curriculum=replace(
                HPCConfig().curriculum,
                restrict_screw_spawn_to_opening_half_disk="false",
            ),
        )
        with self.assertRaisesRegex(ValueError, "must be bool"):
            validate_config(changed)

    def test_legacy_spawn_config_canonicalizes_to_half_disc_only(self):
        current = HPCConfig().to_dict()
        legacy = json.loads(json.dumps(current))
        legacy["curriculum"].pop(
            "restrict_screw_spawn_to_opening_half_disk"
        )
        self.assertEqual(
            canonicalize_config_payload(legacy),
            canonicalize_config_payload(current),
        )
        full_disc = json.loads(json.dumps(current))
        full_disc["curriculum"][
            "restrict_screw_spawn_to_opening_half_disk"
        ] = False
        self.assertNotEqual(
            canonicalize_config_payload(legacy),
            canonicalize_config_payload(full_disc),
        )

    def test_enabled_curriculum_rejects_unreliable_start_radius(self):
        config = HPCConfig()
        changed = replace(
            config,
            curriculum=replace(
                config.curriculum,
                start_spawn_radius_m=0.059,
            ),
        )
        with self.assertRaisesRegex(
            ValueError, r"must be at least 0\.060 m"
        ):
            validate_config(changed)

    def test_socket_is_a_movable_physical_body(self):
        project = ET.parse(ROOT / "assets" / "Project_hpc.xml").getroot()
        share = ET.parse(ROOT / "assets" / "Share_hpc.xml").getroot()
        socket = project.find(".//body[@name='socket']")
        self.assertIsNotNone(socket)
        self.assertEqual(socket.get("mocap"), "true")
        collision = [
            geom
            for geom in socket.findall("geom")
            if geom.get("contype") != "0"
        ]
        self.assertGreater(len(collision), 0)
        self.assertTrue(
            all(geom.get("conaffinity") == "38" for geom in collision)
        )
        table = project.find(".//body[@name='table']")
        table_visual = table.find("geom[@name='table_geom']")
        table_collision = table.find("geom[@name='table_collision']")
        self.assertIsNotNone(table_collision)
        self.assertEqual(
            table_collision.get("size"), table_visual.get("size")
        )
        defaults = {
            default.get("class"): default.find("geom")
            for default in share.findall(".//default")
            if default.get("class")
        }
        self.assertEqual(
            (
                defaults["arm_link"].get("contype"),
                defaults["arm_link"].get("conaffinity"),
            ),
            ("1", "19"),
        )
        self.assertEqual(
            (
                defaults["long_link"].get("contype"),
                defaults["long_link"].get("conaffinity"),
            ),
            ("32", "95"),
        )
        self.assertEqual(
            (
                defaults["Hard_tip"].get("contype"),
                defaults["Hard_tip"].get("conaffinity"),
            ),
            ("2", "7"),
        )
        self.assertEqual(
            (
                defaults["platform"].get("contype"),
                defaults["platform"].get("conaffinity"),
            ),
            ("64", "34"),
        )
        self.assertEqual(table_collision.get("class"), "platform")

        masks = {
            "screw": (4, 11),
            "socket": (8, 38),
            "hard_tip": (2, 7),
            "long_link": (32, 95),
            "table": (64, 34),
        }
        for first, second in (
            ("screw", "socket"),
            ("hard_tip", "socket"),
            ("long_link", "socket"),
            ("hard_tip", "table"),
            ("long_link", "table"),
        ):
            self.assertTrue(
                self._collides(masks[first], masks[second]),
                f"{first}/{second} contact must be enabled",
            )
        for first, second in (
            ("screw", "table"),
            ("socket", "table"),
        ):
            self.assertFalse(
                self._collides(masks[first], masks[second]),
                f"{first}/{second} contact must be disabled",
            )

    def test_root_mouth_and_screw_heights_are_independent(self):
        gripper = ET.parse(ROOT / "assets" / "Gripper.xml").getroot()
        root_body = gripper.find(".//body[@name='root']")
        self.assertEqual(root_body.get("pos"), "0 0 0.62")
        project = ET.parse(ROOT / "assets" / "Project_hpc.xml").getroot()
        screw = project.find(".//body[@name='screw']")
        self.assertEqual(screw.get("pos").split()[2], "0.3")
        socket = project.find(".//body[@name='socket']")
        formal = load_config(ROOT / "configs" / "cx3_l40s.json")
        from screw_config import DEFAULT_CONFIG

        self.assertAlmostEqual(
            DEFAULT_CONFIG.socket_mouth_height_above_table_m, 0.070
        )
        self.assertAlmostEqual(DEFAULT_CONFIG.socket_mouth_position_m[2], 0.330)
        self.assertAlmostEqual(DEFAULT_CONFIG.socket_table_angle_deg, 45.0)
        expected_socket_center = tuple(
            mouth
            + 0.5
            * DEFAULT_CONFIG.socket_outer_length_m
            * axis
            for mouth, axis in zip(
                DEFAULT_CONFIG.socket_mouth_position_m,
                DEFAULT_CONFIG.socket_axis_world,
            )
        )
        for actual, expected in zip(
            map(float, socket.get("pos").split()),
            expected_socket_center,
        ):
            self.assertAlmostEqual(actual, expected, places=11)
        self.assertAlmostEqual(
            DEFAULT_CONFIG.screw_grasp_height_from_tip_m, 0.110
        )
        self.assertAlmostEqual(
            DEFAULT_CONFIG.socket_mouth_orbit_radius_m, 0.020
        )
        self.assertAlmostEqual(DEFAULT_CONFIG.screw_shaft_length_m, 0.080)
        self.assertAlmostEqual(2.0 * DEFAULT_CONFIG.screw_shaft_radius_m, 0.030)
        self.assertAlmostEqual(2.0 * DEFAULT_CONFIG.socket_hole_radius_m, 0.034)
        self.assertAlmostEqual(DEFAULT_CONFIG.screw_head_height_m, 0.040)
        self.assertAlmostEqual(DEFAULT_CONFIG.socket_hole_depth_m, 0.080)
        self.assertAlmostEqual(DEFAULT_CONFIG.socket_outer_length_m, 0.100)
        self.assertAlmostEqual(
            DEFAULT_CONFIG.thread_engagement_depth_m, 0.070
        )
        self.assertAlmostEqual(
            DEFAULT_CONFIG.scripted_grasp_plane_world_z_m, 0.420
        )
        self.assertAlmostEqual(
            DEFAULT_CONFIG.scripted_center_tip_world_z_m, 0.310
        )
        self.assertGreater(SCRIPTED_PREFIX_STEPS, 0)
        self.assertTrue(
            DEFAULT_CONFIG.spawn_is_on_opening_side(
                DEFAULT_CONFIG.default_screw_xy_m
            )
        )
        self.assertEqual(formal.curriculum.fixed_socket_angle_deg, 45.0)

    def test_embedded_socket_keeps_entrance_clear(self):
        from screw_config import DEFAULT_CONFIG as task

        mouth_z = task.socket_mouth_position_m[2]
        table_z = task.table_top_z_m
        self.assertGreater(mouth_z, table_z)
        angle = math.radians(45.0)
        sine = math.sin(angle)
        cosine = math.cos(angle)
        entrance_lowest = (
            mouth_z - 0.5 * task.socket_outer_height_m * cosine
        )
        outer_lowest = (
            mouth_z
            - task.socket_outer_length_m * sine
            - 0.5 * task.socket_outer_height_m * cosine
        )
        bore_lowest = (
            mouth_z
            - task.socket_hole_depth_m * sine
            - task.socket_hole_radius_m * cosine
        )
        inserted_screw_lowest = (
            mouth_z
            - task.target_insertion_depth_m * sine
            - task.screw_shaft_radius_m * cosine
        )
        self.assertGreater(entrance_lowest, table_z)
        self.assertLess(outer_lowest, table_z)
        self.assertGreater(bore_lowest, table_z)
        # At the 330 mm mouth height the socket/table collision pair remains
        # filtered; the narrowed 34 mm bore and the fully inserted screw both
        # clear the tabletop while the outer backing remains embedded.
        self.assertGreater(inserted_screw_lowest, table_z)

    def test_xml_and_substeps_define_20ms_control_period(self):
        formal = load_config(ROOT / "configs" / "hx1_a100.json")
        option = ET.parse(
            ROOT / "assets" / "Project_hpc.xml"
        ).getroot().find("option")
        self.assertIsNotNone(option)
        physics_dt = float(option.get("timestep"))
        self.assertAlmostEqual(
            physics_dt * formal.physics.physics_substeps,
            0.020,
            places=12,
        )

    def test_unknown_config_key_is_rejected(self):
        payload = json.loads(
            (ROOT / "configs" / "smoke.json").read_text(encoding="utf-8")
        )
        payload["physics"]["typo"] = 1
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(source)

    def test_hpc_config_rejects_any_non_45_degree_socket(self):
        payload = json.loads(
            (ROOT / "configs" / "smoke.json").read_text(encoding="utf-8")
        )
        payload["curriculum"]["fixed_socket_angle_deg"] = 46.0
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad_angle.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed 45 degree"):
                load_config(source)

    def test_tqc_drop_is_per_critic(self):
        payload = json.loads(
            (ROOT / "configs" / "smoke.json").read_text(encoding="utf-8")
        )
        payload["tqc"]["top_quantiles_to_drop"] = payload["tqc"]["quantiles"]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad_drop.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(source)

    def test_mps_checkpoint_cannot_silently_load(self):
        payload = checkpoint_metadata("tqc", {})
        self.assertEqual(payload["schema_hash"], TQC_SCHEMA_HASH)
        self.assertEqual(payload["obs_dim"], TQC_OBS_DIM)
        self.assertEqual(payload["action_dim"], RESIDUAL_ACTION_DIM)
        validate_checkpoint(payload, kind="tqc")
        payload["backend"] = "mps"
        with self.assertRaises(ValueError):
            validate_checkpoint(payload, kind="tqc")

    def test_pre_capture_checkpoint_hash_cannot_silently_load(self):
        payload = checkpoint_metadata("diffusion", {})
        self.assertEqual(payload["schema_hash"], DIFFUSION_SCHEMA_HASH)
        self.assertEqual(payload["obs_dim"], OBS_DIM)
        self.assertEqual(payload["residual_action_dim"], 6)
        payload["task_geometry_hash"] = LEGACY_PRE_CAPTURE_TASK_HASH
        with self.assertRaisesRegex(
            ValueError, "task geometry hash mismatch"
        ):
            validate_checkpoint(payload, kind="diffusion")


if __name__ == "__main__":
    unittest.main()
