#!/usr/bin/env python3
"""Static or live-GPU qualification checks for the movable hpc package."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import xml.etree.ElementTree as ET

from screw_config import DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.config import load_config
from s_hpc.contracts import (
    CONTROLLER_STRATEGY,
    OBS_DIM,
    RESIDUAL_ACTION_DIM,
    SCHEMA_HASH,
    TASK_GEOMETRY_HASH,
)
from s_hpc.paths import add_external_path_flag, resolve_package_path


GENERATED_TREE_PARTS = frozenset(
    {
        "RL_env",
        ".cache",
        "runs",
        "datasets",
        "checkpoints",
        "outputs",
        "logs",
        "wheelhouse",
        "__pycache__",
    }
)


def _inside_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def _mask_pair_collides(
    first: tuple[int, int], second: tuple[int, int]
) -> bool:
    """Mirror MuJoCo's symmetric contype/conaffinity compatibility test."""

    first_type, first_affinity = first
    second_type, second_affinity = second
    return bool(
        (first_type & second_affinity)
        or (second_type & first_affinity)
    )


def _time_advance_tolerance(physics_dt: float) -> float:
    """Allow float32 accumulation error while still detecting one lost tick."""

    physics_dt = float(physics_dt)
    if not math.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError("physics_dt must be finite and positive")
    return max(1.0e-6, 0.25 * physics_dt)


def _assert_smoke_physics_contract(formal, smoke) -> None:
    """Allow small budgets, but forbid a different physical canary."""

    physics_fields = (
        "physics_substeps",
        "nconmax",
        "njmax",
        "max_episode_steps",
        "cuda_graph",
        "contact_sensor_maxmatch",
    )
    runtime_fields = ("seed", "device", "dtype", "masked_reset")
    demo_fields = ("obs_horizon", "action_horizon", "seed")
    mismatches = []
    for section, fields in (
        ("physics", physics_fields),
        ("runtime", runtime_fields),
        ("demos", demo_fields),
    ):
        formal_section = getattr(formal, section)
        smoke_section = getattr(smoke, section)
        for field in fields:
            formal_value = getattr(formal_section, field)
            smoke_value = getattr(smoke_section, field)
            if formal_value != smoke_value:
                mismatches.append(
                    f"{section}.{field}: "
                    f"{smoke_value!r} != {formal_value!r}"
                )
    if smoke.ik != formal.ik:
        mismatches.append("ik configuration differs")
    if smoke.curriculum != formal.curriculum:
        mismatches.append("curriculum configuration differs")
    if mismatches:
        raise RuntimeError(
            "smoke physical contract differs from formal config: "
            + "; ".join(mismatches)
        )


def static_checks(config_path: Path) -> dict:
    config = load_config(config_path)
    if config.name.startswith("autodl_rtx_pro_6000_extreme_"):
        _assert_smoke_physics_contract(
            config, load_config(ROOT / "configs" / "smoke.json")
        )
    required = (
        "assets/Project_hpc.xml",
        "assets/Share_hpc.xml",
        "assets/Gripper.xml",
        "s_hpc/env.py",
        "s_hpc/analytic_ik.py",
        "s_hpc/wrench_kernel.py",
        "s_hpc/tqc.py",
        "s_hpc/monitoring.py",
        "s_hpc/rollout_audit.py",
        "pipeline.py",
        "capacity_probe.py",
        "adaptive_capacity.py",
        "training_stack_preflight.py",
        "autodl_rtx_pro_6000_preflight.py",
        "autodl_warp_kernel_smoke.py",
        "status_hpc.py",
        "system_monitor.py",
        "job_status.py",
        "submit_cx3_chain.sh",
        "submit_cx3_adaptive_any.sh",
        "hpc_env.sh",
        "hpc_python.sh",
        "run_hpc.sh",
        "setup_hpc.sh",
        "setup_autodl_rtx_pro_6000.sh",
        "autodl_rtx_pro_6000.sh",
        "build_autodl_package.sh",
        "build_hpc_package.sh",
        "requirements-hpc.txt",
        "README_AUTODL_RTX_PRO_6000.md",
        "configs/autodl_rtx_pro_6000_blackwell.json",
        "configs/autodl_rtx_pro_6000_extreme.json",
        "configs/cx3_l40s.json",
        "jobs/cx3_capacity.pbs",
        "jobs/cx3_smoke.pbs",
        "jobs/cx3_full.pbs",
        "jobs/cx3_adaptive_any.pbs",
        "jobs/cx3_eval.pbs",
    )
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing hpc files: {missing}")

    project_path = ROOT / "assets" / "Project_hpc.xml"
    project = ET.parse(project_path).getroot()
    share = ET.parse(ROOT / "assets" / "Share_hpc.xml").getroot()
    option = project.find("option")
    if option is None or option.get("solver") != "Newton":
        raise RuntimeError("Project_hpc.xml must explicitly use Newton")
    if option.get("jacobian") != "dense":
        raise RuntimeError("Project_hpc.xml must explicitly use jacobian=dense")
    physics_dt = float(option.get("timestep", "0"))
    control_dt = physics_dt * config.physics.physics_substeps
    if abs(control_dt - 0.020) > 1.0e-9:
        raise RuntimeError(
            f"XML/config control period must be 0.020 s, got {control_dt}"
        )
    socket = project.find(".//body[@name='socket']")
    if socket is None or socket.get("mocap") != "true":
        raise RuntimeError("socket must be a movable physical mocap body")
    socket_collision = [
        geom
        for geom in socket.findall("geom")
        if geom.get("contype") != "0"
    ]
    if not socket_collision or any(
        (
            geom.get("contype") != "8"
            or geom.get("conaffinity") != "38"
        )
        for geom in socket_collision
    ):
        raise RuntimeError(
            "socket collisions must use contype=8 and accept object, "
            "hard-tip, and long-link bits"
        )
    table = project.find(".//body[@name='table']")
    table_visual = (
        None if table is None else table.find("geom[@name='table_geom']")
    )
    table_collision = (
        None
        if table is None
        else table.find("geom[@name='table_collision']")
    )
    if (
        table_visual is None
        or table_collision is None
        or table_collision.get("size") != table_visual.get("size")
        or table_collision.get("class") != "platform"
    ):
        raise RuntimeError(
            "random socket poses require a complete platform-class table "
            "collision box"
        )
    collision_defaults = {
        default.get("class"): default.find("geom")
        for default in share.findall(".//default")
        if default.get("class")
    }
    expected_masks = {
        "arm_link": ("1", "19"),
        "long_link": ("32", "95"),
        "Hard_tip": ("2", "7"),
        "platform": ("64", "34"),
    }
    for class_name, (contype, conaffinity) in expected_masks.items():
        geom = collision_defaults.get(class_name)
        if (
            geom is None
            or geom.get("contype") != contype
            or geom.get("conaffinity") != conaffinity
        ):
            raise RuntimeError(
                f"{class_name} collision mask must be "
                f"contype={contype}, conaffinity={conaffinity}"
            )
    screw = project.find(".//body[@name='screw']")
    if screw is None or abs(float(screw.get("pos").split()[2]) - 0.300) > 1.0e-9:
        raise RuntimeError("standing reset screw centre must be z=0.300 m")
    screw_collision = [
        geom
        for geom in screw.findall("geom")
        if geom.get("contype") != "0"
    ]
    if not screw_collision or any(
        (
            geom.get("contype") != "4"
            or geom.get("conaffinity") != "11"
        )
        for geom in screw_collision
    ):
        raise RuntimeError(
            "screw collisions must use contype=4/conaffinity=11"
        )
    gripper = ET.parse(ROOT / "assets" / "Gripper.xml").getroot()
    root_body = gripper.find(".//body[@name='root']")
    if root_body is None or root_body.get("pos") != "0 0 0.62":
        raise RuntimeError("gripper root must be fixed at z=0.620 m")
    for finger in ("U", "D", "L", "R"):
        long_geom = gripper.find(f".//geom[@name='long_{finger}']")
        if long_geom is None or long_geom.get("class") != "long_link":
            raise RuntimeError(f"long_{finger} must use long_link collision class")
    if (
        abs(config.curriculum.fixed_socket_angle_deg - 45.0) > 1.0e-9
    ):
        raise RuntimeError("formal training must keep the socket fixed at 45 degrees")
    if config.physics.max_episode_steps < 4000:
        raise RuntimeError(
            "episodes need at least 4000 steps for the expert trajectory, "
            "20-step anchor blends, "
            "plus measured-gate timing margin"
        )
    if abs(DEFAULT_CONFIG.socket_mouth_position_m[2] - 0.330) > 1.0e-9:
        raise RuntimeError("socket mouth height must remain world z=0.330 m")
    expected_socket_center = tuple(
        DEFAULT_CONFIG.socket_mouth_position_m[index]
        + 0.5
        * DEFAULT_CONFIG.socket_outer_length_m
        * DEFAULT_CONFIG.socket_axis_world[index]
        for index in range(3)
    )
    asset_socket_center = tuple(
        float(value) for value in socket.get("pos", "").split()
    )
    if (
        len(asset_socket_center) != 3
        or any(
            abs(actual - expected) > 1.0e-9
            for actual, expected in zip(
                asset_socket_center, expected_socket_center
            )
        )
    ):
        raise RuntimeError(
            "Project_hpc.xml socket pose does not match the fixed-45-degree "
            "0.330 m mouth-height contract"
        )
    if abs(DEFAULT_CONFIG.screw_grasp_height_from_tip_m - 0.110) > 1.0e-9:
        raise RuntimeError(
            "the pickup grasp plane must remain H110 "
            "(110 mm from the screw tip; initial world z=0.370 m)"
        )
    if (
        abs(DEFAULT_CONFIG.scripted_grasp_plane_world_z_m - 0.420)
        > 1.0e-9
        or abs(DEFAULT_CONFIG.scripted_center_tip_world_z_m - 0.310)
        > 1.0e-9
    ):
        raise RuntimeError(
            "scripted lifted-spawn handoff must use grasp-plane z=0.420 m "
            "and H110 screw-tip z=0.310 m"
        )
    expected_preinsert = {
        "preinsert_standoff_m": 0.005,
        "preinsert_lateral_tolerance_m": 0.0005,
        "preinsert_orientation_tolerance_deg": 1.0,
        "preinsert_depth_tolerance_m": 0.002,
        "preinsert_tip_speed_tolerance_m_s": 0.020,
        "preinsert_lateral_speed_tolerance_m_s": 0.010,
        "preinsert_angular_speed_tolerance_deg_s": 12.0,
        "preinsert_support_separation_deg": 150.0,
    }
    for name, expected in expected_preinsert.items():
        if abs(float(getattr(DEFAULT_CONFIG, name)) - expected) > 1.0e-9:
            raise RuntimeError(
                f"5 mm preinsert contract drifted: {name} "
                f"must remain {expected}"
            )
    if DEFAULT_CONFIG.preinsert_stable_steps != 5:
        raise RuntimeError(
            "5 mm preinsert alignment must remain stable for five steps"
        )
    optional_preinsert = {
        "preinsert_feedback_gain": 3.0,
        "preinsert_command_lateral_limit_m": 0.0015,
        "preinsert_feedback_depth_limit_m": 0.002,
        "preinsert_feedback_lateral_slew_m": 0.0004,
        "preinsert_feedback_depth_slew_m": 0.0005,
    }
    for name, expected in optional_preinsert.items():
        if (
            hasattr(DEFAULT_CONFIG, name)
            and abs(float(getattr(DEFAULT_CONFIG, name)) - expected)
            > 1.0e-9
        ):
            raise RuntimeError(
                f"optional measured preinsert feedback contract drifted: "
                f"{name} must remain {expected}"
            )
    expected_capture = {
        "capture_depth_m": 0.030,
        "capture_lateral_tolerance_m": 0.0005,
        "capture_orientation_tolerance_deg": 1.0,
        "capture_depth_tolerance_m": 0.002,
        "capture_tip_speed_tolerance_m_s": 0.020,
        "capture_lateral_speed_tolerance_m_s": 0.010,
        "capture_angular_speed_tolerance_deg_s": 12.0,
        "thread_depth_tolerance_m": 0.0015,
        "contact_force_threshold_n": 0.10,
        "contact_force_release_threshold_n": 0.04,
    }
    for name, expected in expected_capture.items():
        if abs(float(getattr(DEFAULT_CONFIG, name)) - expected) > 1.0e-9:
            raise RuntimeError(
                f"30 mm shaft-capture contract drifted: {name} "
                f"must remain {expected}"
            )
    if (
        DEFAULT_CONFIG.capture_stable_steps != 5
        or DEFAULT_CONFIG.all_four_contact_stable_steps != 5
        or DEFAULT_CONFIG.minimum_contact_count_before_capture != 3
        or DEFAULT_CONFIG.minimum_contact_count_after_capture != 2
        or DEFAULT_CONFIG.contact_loss_grace_steps != 20
        or DEFAULT_CONFIG.thread_stable_steps != 5
    ):
        raise RuntimeError(
            "four-finger pickup, two-contact post-capture support, "
            "contact-loss grace, 30 mm capture, and measured thread gates "
            "drifted from the formal contract"
        )
    head_bottom_standoff = (
        DEFAULT_CONFIG.screw_shaft_length_m
        - DEFAULT_CONFIG.capture_depth_m
    )
    if abs(head_bottom_standoff - 0.050) > 1.0e-9:
        raise RuntimeError(
            "30 mm shaft capture must leave the screw-head bottom "
            "50 mm outside the mouth plane"
        )
    capture_contract = CONTROLLER_STRATEGY.get("capture_handoff", {})
    if (
        capture_contract.get("minimum_contact_count_before_capture") != 3
        or
        capture_contract.get("minimum_contact_count_after_capture") != 2
        or capture_contract.get("commanded_prethread_grasp")
        != "all-four-at-0.039m"
        or capture_contract.get("diagnostic_primary_pair")
        != "physical-L/R"
        or capture_contract.get("material_angles")
        != "fixed-from-pickup-through-thread-gate"
        or capture_contract.get("prethread_regrasp") is not False
        or abs(
            float(capture_contract.get("capture_depth_m", -1.0))
            - DEFAULT_CONFIG.capture_depth_m
        )
        > 1.0e-9
        or abs(
            float(
                capture_contract.get(
                    "thread_engagement_depth_m", -1.0
                )
            )
            - DEFAULT_CONFIG.thread_engagement_depth_m
        )
        > 1.0e-9
        or abs(
            float(
                capture_contract.get(
                    "thread_depth_tolerance_m", -1.0
                )
            )
            - DEFAULT_CONFIG.thread_depth_tolerance_m
        )
        > 1.0e-9
    ):
        raise RuntimeError(
            "four-finger -> captured physical L/R -> measured-thread "
            "regrasp controller contract drifted"
        )
    if abs(
        sum(value * value for value in DEFAULT_CONFIG.socket_mouth_xy_m) ** 0.5
        - 0.020
    ) > 1.0e-9:
        raise RuntimeError("default socket mouth must lie on the 20 mm orbit")
    if (
        abs(DEFAULT_CONFIG.screw_shaft_length_m - 0.080) > 1.0e-9
        or abs(DEFAULT_CONFIG.screw_head_height_m - 0.040) > 1.0e-9
        or abs(2.0 * DEFAULT_CONFIG.screw_shaft_radius_m - 0.030) > 1.0e-9
        or abs(2.0 * DEFAULT_CONFIG.socket_hole_radius_m - 0.034) > 1.0e-9
        or abs(DEFAULT_CONFIG.socket_hole_depth_m - 0.080) > 1.0e-9
        or abs(DEFAULT_CONFIG.socket_outer_length_m - 0.100) > 1.0e-9
        or abs(DEFAULT_CONFIG.thread_engagement_depth_m - 0.070)
        > 1.0e-9
    ):
        raise RuntimeError(
            "30 mm screw / 34 mm hole / 80/40/80/100/70 mm task geometry mismatch"
        )

    # The socket is deliberately embedded in the tabletop. Its entrance and
    # fully inserted screw remain clear, while the backing block extends below
    # the table at the fixed 45-degree elevation.  With the narrowed 34 mm
    # bore, the bore rim itself remains above the tabletop.
    mouth_z = DEFAULT_CONFIG.socket_mouth_position_m[2]
    table_z = DEFAULT_CONFIG.table_top_z_m
    if mouth_z <= table_z:
        raise RuntimeError("socket mouth centre must remain above the tabletop")
    for angle_deg in (45.0,):
        angle = math.radians(angle_deg)
        sine = math.sin(angle)
        cosine = math.cos(angle)
        entrance_lowest = (
            mouth_z
            - 0.5 * DEFAULT_CONFIG.socket_outer_height_m * cosine
        )
        outer_lowest = (
            mouth_z
            - DEFAULT_CONFIG.socket_outer_length_m * sine
            - 0.5 * DEFAULT_CONFIG.socket_outer_height_m * cosine
        )
        bore_lowest = (
            mouth_z
            - DEFAULT_CONFIG.socket_hole_depth_m * sine
            - DEFAULT_CONFIG.socket_hole_radius_m * cosine
        )
        screw_lowest = (
            mouth_z
            - DEFAULT_CONFIG.target_insertion_depth_m * sine
            - DEFAULT_CONFIG.screw_shaft_radius_m * cosine
        )
        if entrance_lowest <= table_z:
            raise RuntimeError(
                f"socket entrance clips the table at {angle_deg:g} degrees"
            )
        if not (outer_lowest < table_z and bore_lowest > table_z):
            raise RuntimeError(
                "socket must retain embedded backing while the narrowed "
                f"34 mm bore clears the table at {angle_deg:g} degrees"
            )
        if screw_lowest <= table_z:
            raise RuntimeError(
                "the fully inserted screw must remain clear of the tabletop "
                f"at {angle_deg:g} degrees"
            )

    from screw_control import (
        CAPTURE_GATE_STEP,
        DIFFUSION_TO_ANCHOR_BLEND_STEPS,
        PREINSERT_GATE_STEP,
        SCRIPTED_PREFIX_STEPS,
        THREAD_GATE_STEP,
        THREAD_REGRASP_GRASP_HEIGHT_FROM_TIP_M,
        THREAD_REGRASP_STEP,
        YAW_EQUIVARIANT_ALIGN_BOUNDARY_STEP,
        YAW_EQUIVARIANT_ALIGN_STEPS,
        YAW_EQUIVARIANT_RELEASED_DISTANCE_M,
        YAW_EQUIVARIANT_ROLL_GAIN,
        YAW_EQUIVARIANT_ROLL_BOUNDARY_STEP,
        YAW_EQUIVARIANT_ROLL_STEPS,
        YAW_EQUIVARIANT_STRATEGY_VERSION,
    )

    expected_roll_application = (
        "disabled-direct-final-frame"
        if YAW_EQUIVARIANT_ROLL_GAIN == 0.0
        else (
            "post-interpolation-right-multiply-Rz("
            f"{YAW_EQUIVARIANT_ROLL_GAIN:g}*delta*minimum_jerk("
            f"(step-{YAW_EQUIVARIANT_ROLL_BOUNDARY_STEP})/"
            f"{YAW_EQUIVARIANT_ROLL_STEPS}));"
            "controller-boundaries-imported-from-screw-control"
        )
    )
    expected_controller_values = {
        "version": YAW_EQUIVARIANT_STRATEGY_VERSION,
        "released_distance_m": YAW_EQUIVARIANT_RELEASED_DISTANCE_M,
        "roll_gain": YAW_EQUIVARIANT_ROLL_GAIN,
        "scripted_grasp_height_from_tip_m": 0.110,
        "diffusion_to_anchor_blend": {
            "steps": DIFFUSION_TO_ANCHOR_BLEND_STEPS,
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
        "thread_regrasp_height": {
            "from_tip_m": THREAD_REGRASP_GRASP_HEIGHT_FROM_TIP_M,
            "switch_step": THREAD_REGRASP_STEP + 1,
            "mode": "direct-at-regrasp",
        },
        "align_boundary_step": YAW_EQUIVARIANT_ALIGN_BOUNDARY_STEP,
        "align_steps": YAW_EQUIVARIANT_ALIGN_STEPS,
        "roll_boundary_step": YAW_EQUIVARIANT_ROLL_BOUNDARY_STEP,
        "roll_steps": YAW_EQUIVARIANT_ROLL_STEPS,
        "roll_application": expected_roll_application,
    }
    if (
        SCRIPTED_PREFIX_STEPS <= 0
        or not YAW_EQUIVARIANT_STRATEGY_VERSION
        or any(
            CONTROLLER_STRATEGY.get(name) != expected
            for name, expected in expected_controller_values.items()
        )
    ):
        raise RuntimeError(
            "controller strategy must be imported from screw_control "
            "without stale hard-coded boundary/version values"
        )
    capture_boundaries = CONTROLLER_STRATEGY["capture_handoff"]
    if (
        capture_boundaries.get("preinsert_gate_step")
        != PREINSERT_GATE_STEP
        or capture_boundaries.get("capture_gate_step")
        != CAPTURE_GATE_STEP
        or capture_boundaries.get("thread_gate_step")
        != THREAD_GATE_STEP
        or capture_boundaries.get("thread_regrasp_step")
        != THREAD_REGRASP_STEP
        or not (
            PREINSERT_GATE_STEP
            < CAPTURE_GATE_STEP
            < THREAD_GATE_STEP
            < THREAD_REGRASP_STEP
        )
    ):
        raise RuntimeError(
            "measured preinsert/capture/thread/regrasp boundaries drifted"
        )

    masks = {
        "screw": (4, 11),
        "socket": (8, 38),
        "hard_tip": (2, 7),
        "long_link": (32, 95),
        "table": (64, 34),
    }
    required_contacts = (
        ("screw", "socket"),
        ("hard_tip", "socket"),
        ("long_link", "socket"),
        ("hard_tip", "table"),
        ("long_link", "table"),
    )
    forbidden_contacts = (
        ("screw", "table"),
        ("socket", "table"),
    )
    if any(
        not _mask_pair_collides(masks[first], masks[second])
        for first, second in required_contacts
    ):
        raise RuntimeError(
            "collision masks do not enable all required screw/socket and "
            "hard-tip/long-link table/socket contacts"
        )
    if any(
        _mask_pair_collides(masks[first], masks[second])
        for first, second in forbidden_contacts
    ):
        raise RuntimeError(
            "embedded socket requires screw-table and socket-table contacts "
            "to remain disabled"
        )
    size = share.find("size")
    forbidden = [
        key
        for key in (
            "nuser_jnt", "nuser_site", "nuser_sensor", "nuser_actuator"
        )
        if size is not None and size.get(key) is not None
    ]
    if forbidden:
        raise RuntimeError(f"MJWarp-incompatible XML attributes: {forbidden}")
    for include in project.iter("include"):
        source = (project_path.parent / str(include.get("file"))).resolve()
        if not source.is_file() or not _inside_root(source):
            raise RuntimeError(f"invalid/nonlocal XML include: {source}")
    compiler = project.find("compiler")
    meshdir = "mesh" if compiler is None else compiler.get("meshdir", "mesh")
    for mesh in share.iter("mesh"):
        filename = mesh.get("file")
        if filename:
            source = (project_path.parent / meshdir / filename).resolve()
            if not source.is_file() or not _inside_root(source):
                raise RuntimeError(f"invalid/nonlocal mesh: {source}")

    # Split the token so this validator does not flag its own source.
    absolute_macos = re.compile("/" + r"Users/[^/]+/")
    offenders = []
    for source in ROOT.rglob("*"):
        if (
            not source.is_file()
            or GENERATED_TREE_PARTS.intersection(source.parts)
            or source.name.startswith("._")
            or source.suffix not in {".py", ".sh", ".pbs", ".slurm", ".json", ".md", ".txt"}
        ):
            continue
        text = source.read_text(encoding="utf-8", errors="ignore")
        if absolute_macos.search(text):
            offenders.append(str(source.relative_to(ROOT)))
        if source.suffix == ".py":
            compile(text, str(source), "exec")
    if offenders:
        raise RuntimeError(f"host-specific absolute paths found: {offenders}")
    return {
        "mode": "static",
        "status": "pass",
        "config": config.name,
        "root": str(ROOT),
        "obs_dim": OBS_DIM,
        "schema_hash": SCHEMA_HASH,
        "task_geometry_hash": TASK_GEOMETRY_HASH,
        "controller_strategy": CONTROLLER_STRATEGY["version"],
        "fixed_socket_angle_deg": config.curriculum.fixed_socket_angle_deg,
        "restrict_screw_spawn_to_opening_half_disk": (
            config.curriculum.restrict_screw_spawn_to_opening_half_disk
        ),
        "screw_spawn_region": (
            "opening-facing half-disc"
            if config.curriculum.restrict_screw_spawn_to_opening_half_disk
            else "full workspace disc minus socket exclusion"
        ),
        "socket_mouth_world_z_m": DEFAULT_CONFIG.socket_mouth_position_m[2],
        "physics_dt_s": physics_dt,
        "control_dt_s": control_dt,
        "files_checked": len(list(ROOT.rglob("*"))),
    }


def gpu_checks(config_path: Path, worlds: int | None) -> dict:
    import torch

    from s_hpc.env import MJWarpScrewEnv
    from s_hpc.rollout_audit import (
        ExpertRolloutAudit,
        certify_expert_rollout,
    )

    versions = {"torch": importlib.metadata.version("torch")}
    expected_versions = {
        "torch": "2.7.1",
        "mujoco": "3.10.0",
        "mujoco-warp": "3.10.0.3",
        "warp-lang": "1.14.0",
    }
    for package, expected in expected_versions.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required GPU package is missing: {package}"
            ) from exc
        versions[package] = actual
        normalized = actual.split("+", 1)[0]
        if normalized != expected:
            raise RuntimeError(
                f"{package} version must be {expected}, got {actual}"
            )
    config = load_config(config_path)
    if worlds is not None:
        config = replace(
            config, physics=replace(config.physics, worlds=int(worlds))
        )
    device = torch.device(config.runtime.device)
    free_before_env, total_device_memory = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats()
    env = MJWarpScrewEnv(config, baseline="expert")
    zero = torch.zeros(
        (env.worlds, RESIDUAL_ACTION_DIM), device=env.device
    )
    try:
        torch.cuda.synchronize(env.device)
        free_after_init, observed_total = torch.cuda.mem_get_info(env.device)
        if int(observed_total) != int(total_device_memory):
            raise RuntimeError("CUDA total memory changed during preflight")
        minimum_free_memory = int(free_after_init)
        # Exercise the exact lowest-radius GPU sampler used at curriculum
        # progress zero.  The environment starts its expert canary at full
        # difficulty, so without this explicit gate a lower-bound geometry
        # regression could otherwise remain hidden until a later workflow.
        start_spawn_radius = float(
            config.curriculum.start_spawn_radius_m
        )
        start_socket_mouth = env.supervisor.sample_socket_mouth(
            env.worlds,
            generator=env.generator,
        )
        start_socket_angle = env.supervisor.fixed_angle_batch(env.worlds)
        start_spawn = env.supervisor.sample_spawn(
            env.worlds,
            socket_mouth_xy=start_socket_mouth,
            socket_angle_deg=start_socket_angle,
            radius_m=start_spawn_radius,
            generator=env.generator,
        )
        if start_spawn.shape != (env.worlds, 2):
            raise RuntimeError(
                "GPU curriculum-start spawn sampler returned the wrong shape"
            )
        if not torch.isfinite(start_spawn).all():
            raise RuntimeError(
                "GPU curriculum-start spawn sampler returned NaN/Inf"
            )
        if bool(
            (
                torch.linalg.vector_norm(start_spawn, dim=-1)
                > start_spawn_radius + 1.0e-6
            ).any()
        ):
            raise RuntimeError(
                "GPU curriculum-start spawn escaped its configured radius"
            )
        expected_angle = config.curriculum.fixed_socket_angle_deg
        if not torch.allclose(
            env.state.socket_angle_deg,
            torch.full_like(env.state.socket_angle_deg, expected_angle),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise RuntimeError("GPU reset produced a non-45-degree socket")
        expected_reset_z = float(
            DEFAULT_CONFIG.screw_initial_center_z_m
        )
        screw_center_z = env.backend.tensor("qpos")[
            :, env.backend.index.screw_qpos + 2
        ]
        if not torch.allclose(
            screw_center_z,
            torch.full_like(screw_center_z, expected_reset_z),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise RuntimeError(
                "GPU reset placed the screw at the wrong centre height"
            )
        if (
            env.backend.index.holder_mocap >= 0
            and env.backend.has_field("mocap_pos")
        ):
            holder_center_z = env.backend.tensor("mocap_pos")[
                :, env.backend.index.holder_mocap, 2
            ]
            if not torch.allclose(
                holder_center_z,
                torch.full_like(holder_center_z, expected_reset_z),
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise RuntimeError(
                    "GPU reset placed the screw holder at the wrong height"
                )
        reset_tip_error = torch.linalg.vector_norm(
            env.state.screw_tip() - env.state.desired_command.tip,
            dim=-1,
        )
        if not bool((reset_tip_error <= 1.0e-6).all()):
            raise RuntimeError(
                "GPU reset screw tip does not match the step-zero expert "
                f"command; max error={float(reset_tip_error.max()):.9f} m"
            )
        if config.physics.cuda_graph and not env.backend._graph_ready:
            raise RuntimeError("CUDA Graph was requested but not captured")
        initial_time = env.backend.tensor("time").clone()
        ik_success = []
        health = {}
        steps_run = min(96, config.physics.max_episode_steps)
        rollout_audit = ExpertRolloutAudit(
            torch, worlds=env.worlds, device=env.device
        )
        for step_index in range(steps_run):
            result = env.step(zero)
            rollout_audit.update(
                result, step_number=step_index + 1
            )
            ik_success.append(result.ik_feasible_fraction)
            sample = env.backend.health()
            if step_index % 8 == 7 or step_index + 1 == steps_run:
                free_now, observed_total = torch.cuda.mem_get_info(env.device)
                if int(observed_total) != int(total_device_memory):
                    raise RuntimeError(
                        "CUDA total memory changed during preflight"
                    )
                minimum_free_memory = min(
                    minimum_free_memory, int(free_now)
                )
            if not sample["finite"] or sample.get("sticky_nonfinite", 0):
                raise RuntimeError(f"non-finite MJWarp state: {sample}")
            if sample.get("sticky_external_nonfinite", 0):
                raise RuntimeError(
                    f"non-finite policy/task tensor: {sample}"
                )
            if (
                sample.get("overflow_worlds", 0) != 0
                or sample.get("sticky_overflow", 0)
            ):
                raise RuntimeError(f"MJWarp overflow detected: {sample}")
            if (
                sample.get("contacts_total", 0)
                >= sample.get("contact_capacity_total", 1)
                or sample.get("sticky_contacts_full", 0)
            ):
                raise RuntimeError(f"contact capacity reached: {sample}")
            if (
                sample.get("max_nefc", 0) >= config.physics.njmax
                or sample.get("sticky_constraints_full", 0)
            ):
                raise RuntimeError(f"constraint capacity reached: {sample}")
            for key, value in sample.items():
                if isinstance(value, bool):
                    health[key] = health.get(key, True) and value
                elif isinstance(value, (int, float)):
                    health[key] = max(health.get(key, value), value)
                else:
                    health[key] = value
        task_audit = certify_expert_rollout(
            env,
            result,
            rollout_audit,
            context="GPU preflight expert prefix",
            require_terminal=False,
        )
        elapsed = env.backend.tensor("time") - initial_time
        expected_elapsed = env.backend.control_dt * steps_run
        elapsed_tolerance = _time_advance_tolerance(
            env.backend.physics_dt
        )
        if not torch.allclose(
            elapsed,
            torch.full_like(elapsed, expected_elapsed),
            rtol=0.0,
            atol=elapsed_tolerance,
        ):
            elapsed_min = float(elapsed.min())
            elapsed_max = float(elapsed.max())
            raise RuntimeError(
                "MJWarp time advance does not match the configured control "
                f"period: expected {expected_elapsed}s, observed "
                f"{elapsed_min}..{elapsed_max}s, tolerance "
                f"{elapsed_tolerance}s"
            )
        elapsed_min = float(elapsed.min())
        elapsed_max = float(elapsed.max())
        observation = env.state.observation()
        if observation.shape != (env.worlds, OBS_DIM):
            raise RuntimeError("GPU observation contract mismatch")
        if not torch.isfinite(observation).all():
            raise RuntimeError("GPU observation contains NaN/Inf")
        ik_fraction = float(torch.stack(ik_success).mean())
        physics_dt = env.backend.physics_dt
        control_dt = env.backend.control_dt
        graph_ready = bool(env.backend._graph_ready)
        peak_allocated = int(
            torch.cuda.max_memory_allocated(env.device)
        )
        peak_reserved = int(
            torch.cuda.max_memory_reserved(env.device)
        )
        peak_device_used = int(total_device_memory) - minimum_free_memory
        peak_device_fraction = (
            peak_device_used / int(total_device_memory)
        )
    finally:
        env.close()
    return {
        "mode": "gpu",
        "status": "pass",
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "worlds": config.physics.worlds,
        "physics_substeps": config.physics.physics_substeps,
        "physics_dt_s": physics_dt,
        "control_dt_s": control_dt,
        "preflight_elapsed_expected_s": expected_elapsed,
        "preflight_elapsed_min_s": elapsed_min,
        "preflight_elapsed_max_s": elapsed_max,
        "preflight_elapsed_tolerance_s": elapsed_tolerance,
        "fixed_socket_angle_deg": expected_angle,
        "curriculum_start_spawn_radius_m": start_spawn_radius,
        "curriculum_start_spawn_worlds_checked": env.worlds,
        "restrict_screw_spawn_to_opening_half_disk": (
            config.curriculum.restrict_screw_spawn_to_opening_half_disk
        ),
        "screw_spawn_region": (
            "opening-facing half-disc"
            if config.curriculum.restrict_screw_spawn_to_opening_half_disk
            else "full workspace disc minus socket exclusion"
        ),
        "reset_screw_center_z_m": expected_reset_z,
        "reset_tip_error_max_m": float(reset_tip_error.max()),
        "cuda_graph_ready": graph_ready,
        "torch_peak_allocated_bytes": peak_allocated,
        "torch_peak_reserved_bytes": peak_reserved,
        "cuda_total_memory_bytes": int(total_device_memory),
        "cuda_free_before_env_bytes": int(free_before_env),
        "cuda_min_free_memory_bytes": minimum_free_memory,
        "cuda_peak_device_used_bytes": peak_device_used,
        "cuda_peak_device_used_fraction": peak_device_fraction,
        "versions": versions,
        "mean_ik_feasible_fraction": ik_fraction,
        "task_audit": task_audit,
        "health": health,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "gpu"), required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "smoke.json"
    )
    parser.add_argument("--worlds", type=int, default=None)
    add_external_path_flag(parser)
    args = parser.parse_args()
    try:
        config_path = resolve_package_path(
            args.config,
            allow_external=args.allow_external_paths,
            label="config",
        )
    except ValueError as exc:
        parser.error(str(exc))
    result = (
        static_checks(config_path)
        if args.mode == "static"
        else gpu_checks(config_path, args.worlds)
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
