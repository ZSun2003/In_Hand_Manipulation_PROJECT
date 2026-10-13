"""Gripper6 host-side adapter for the shared sharded MJWarp controller.

The canonical multi-object model remains the single IK/planning source of
truth.  Physics is split into one dense-Newton model per trained object,
keeping every GPU shard at nq=31/nv=30/nu=16.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .contract import (
    FIXED_GRASP_PLANE_FRACTIONS,
    FIXED_GRASP_PLANE_YAW_RAD,
    GRASP_PLANE_MAX_FRACTION,
    GRASP_PLANE_MIN_FRACTION,
    GRASP_PLANE_NOMINAL_FRACTION,
    INITIAL_OBJECT_COM_Z_M,
    INFEASIBLE_OBJECTS,
    LEARNED_GRASP_PLANE_OBJECTS,
    OBJECTS,
    RESET_TEMPLATES_PER_OBJECT,
    TASK_ROTATE,
    TASK_TRANSLATE,
    TASKS,
    TIP_ORDER,
    YAW_LIMIT_RAD,
    reset_template_seed,
    validate_task,
)


PLATFORM = "gripper6"
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
PLATFORM_ADAPTER_PATH = Path(__file__).resolve()
TASK_ROOT_HEIGHT_M = {
    TASK_TRANSLATE: 0.560,
    TASK_ROTATE: 0.560,
}
TASK_GRIPPER_XMLS = {
    task: PROJECT_ROOT / f"_g6_gripper_{task}.xml" for task in TASKS
}
MODEL_XMLS_BY_TASK = {
    task: {
        object_kind: PROJECT_ROOT / f"_g6_project_rl_{task}_{object_kind}.xml"
        for object_kind in OBJECTS
    }
    for task in TASKS
}
PARENT_MODEL_XML = PROJECT_ROOT / "_g6_project.xml"
PARENT_MODULE = "_g6_grab"
EXPECTED_ACTUATORS = 16
EXPECTED_SHARD_NQ = 31
EXPECTED_SHARD_NV = 30
# Legacy aliases retained for the G6-only model test/API transition.
EXPECTED_RL_NQ = EXPECTED_SHARD_NQ
EXPECTED_RL_NV = EXPECTED_SHARD_NV


def model_xmls_for_task(task: str) -> dict[str, Path]:
    """Return a copy of the ordered four-shard XML mapping for one task."""

    selected = validate_task(task)
    result = dict(MODEL_XMLS_BY_TASK[selected])
    if tuple(result) != tuple(OBJECTS):
        raise RuntimeError("G6 task shard order differs from contract.OBJECTS")
    return result


@dataclass(frozen=True)
class PlatformSpec:
    """One fixed-object model and its four-slot controller compatibility bank."""

    platform: str
    task: str
    task_index: int
    root_height_m: float
    fixed_object_index: int
    model_xml: Path
    model_xml_sha256: str
    platform_adapter_fingerprint_sha256: str
    parent_model_fingerprint_sha256: str
    parent_controller_fingerprint_sha256: str
    planner_model_fingerprint_sha256: str
    rl_model_fingerprint_sha256: str
    host_model: Any
    qpos_templates: np.ndarray
    ctrl_templates: np.ndarray
    initial_object_pos: np.ndarray
    initial_object_quat: np.ndarray
    pregrasp_targets_local: np.ndarray
    contact_targets_local: np.ndarray
    outward_directions_local: np.ndarray
    planned_grasp_plane_fraction: np.ndarray
    planned_grasp_plane_yaw_rad: np.ndarray
    object_local_bottom_z_m: np.ndarray
    object_local_top_z_m: np.ndarray
    object_body_ids: np.ndarray
    object_qpos_adrs: np.ndarray
    object_qvel_adrs: np.ndarray
    object_geom_mask: np.ndarray
    tip_body_ids: np.ndarray
    tip_geom_ids: np.ndarray
    actuator_qpos_adrs: np.ndarray
    actuator_dof_adrs: np.ndarray
    passive_qpos_adrs: np.ndarray
    passive_dof_adrs: np.ndarray
    passive_qpos_low: np.ndarray
    passive_qpos_high: np.ndarray
    pin_site_ids: np.ndarray
    pin_body_ids: np.ndarray
    actuator_finger_mask: np.ndarray
    finger_actuator_indices: np.ndarray
    finger_passive_indices: np.ndarray
    ctrl_low: np.ndarray
    ctrl_high: np.ndarray


@dataclass(frozen=True)
class ShardedPlatformSpec:
    """Ordered four-model physics bundle presented to the shared trainer."""

    platform: str
    task: str
    task_index: int
    root_height_m: float
    platform_adapter_fingerprint_sha256: str
    parent_model_fingerprint_sha256: str
    parent_controller_fingerprint_sha256: str
    planner_model_fingerprint_sha256: str
    rl_model_fingerprint_sha256: str
    shards: tuple[PlatformSpec, ...]

    @property
    def host_model(self) -> Any:
        """Representative model for legacy dimension-only callers."""

        if not self.shards:
            raise RuntimeError("sharded platform has no child models")
        return self.shards[0].host_model


def _parent_module() -> Any:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(PARENT_MODULE)


def _finger_from_actuator(name: str) -> str:
    prefix = name.split("_", 1)[0]
    if prefix not in TIP_ORDER:
        raise ValueError(f"cannot assign actuator {name!r} to a finger")
    return prefix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_shard_fingerprint(
    model_xml: Path,
    task_gripper_xml: Path,
    *,
    task: str,
    root_height_m: float,
) -> str:
    """Bind one top-level shard to its task-specific gripper include."""

    digest = hashlib.sha256()
    digest.update(b"mjwarp-task-object-shard-v2\0")
    for label, value in (
        ("task", validate_task(task)),
        ("root-height-m", format(float(root_height_m), ".17g")),
        ("model-xml", _sha256(model_xml)),
        ("task-gripper-xml", _sha256(task_gripper_xml)),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _bound_model_fingerprint(
    planner_fingerprint: str,
    rl_fingerprint: str,
    *,
    task: str,
    root_height_m: float,
) -> str:
    """Bind planner and RL dependencies to an exact task/root-height pair."""

    digest = hashlib.sha256()
    for label, value in (
        ("task", validate_task(task)),
        ("root-height-m", format(float(root_height_m), ".17g")),
        ("canonical-planner", str(planner_fingerprint)),
        ("dense-newton-rl", str(rl_fingerprint)),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _ordered_shard_fingerprint(
    model_fingerprints: tuple[str, ...] | list[str],
    *,
    task: str,
    root_height_m: float,
) -> str:
    """Hash one task bundle in the immutable contract object order."""

    if len(model_fingerprints) != len(OBJECTS):
        raise ValueError(
            "ordered shard fingerprint requires one hash per contract object"
        )
    digest = hashlib.sha256()
    digest.update(b"mjwarp-task-object-shards-v2\0")
    digest.update(validate_task(task).encode("ascii"))
    digest.update(b"\0")
    digest.update(format(float(root_height_m), ".17g").encode("ascii"))
    digest.update(b"\0")
    for object_kind, fingerprint in zip(OBJECTS, model_fingerprints):
        if len(fingerprint) != 64:
            raise ValueError(f"invalid SHA-256 for {object_kind!r}")
        digest.update(object_kind.encode("utf-8"))
        digest.update(b"\0")
        digest.update(fingerprint.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _expected_omitted_joint_names(
    fixed_object_index: int,
) -> frozenset[str]:
    if not 0 <= int(fixed_object_index) < len(OBJECTS):
        raise ValueError("fixed_object_index is outside contract.OBJECTS")
    active = OBJECTS[int(fixed_object_index)]
    return frozenset(
        {"joint:object_plate", "joint:object_cube"}
        | {
            f"joint:object_{object_kind}"
            for object_kind in OBJECTS
            if object_kind != active
        }
        | {
            f"joint:object_{object_kind}"
            for object_kind in INFEASIBLE_OBJECTS
        }
    )


def _require_id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"RL model object not found: {name}")
    return value


def _joint_qpos_width(mujoco: Any, joint_type: int) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    if joint_type in {
        int(mujoco.mjtJoint.mjJNT_SLIDE),
        int(mujoco.mjtJoint.mjJNT_HINGE),
    }:
        return 1
    raise ValueError(f"unsupported MuJoCo joint type: {joint_type}")


def _qpos_projection(
    mujoco: Any,
    parent_model: Any,
    rl_model: Any,
    fixed_object_index: int,
) -> np.ndarray:
    """Map every RL qpos element to the same named canonical joint."""

    def joint_names(model: Any) -> tuple[str, ...]:
        values = tuple(
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, index
            )
            for index in range(int(model.njnt))
        )
        if any(not name for name in values):
            raise ValueError("every canonical and RL joint must have a name")
        return tuple(str(name) for name in values)

    parent_names = joint_names(parent_model)
    rl_names = joint_names(rl_model)
    unexpected = set(rl_names) - set(parent_names)
    omitted = set(parent_names) - set(rl_names)
    expected_omitted = _expected_omitted_joint_names(fixed_object_index)
    if unexpected or omitted != set(expected_omitted):
        raise ValueError(
            "G6 object-shard joint set omitted the wrong canonical joints; "
            f"expected={sorted(expected_omitted)}, omitted={sorted(omitted)}, "
            f"unexpected={sorted(unexpected)}"
        )

    projection = np.full(int(rl_model.nq), -1, dtype=np.int32)
    for target_joint in range(int(rl_model.njnt)):
        name = mujoco.mj_id2name(
            rl_model, mujoco.mjtObj.mjOBJ_JOINT, target_joint
        )
        if not name:
            raise ValueError("every RL joint must have a stable name")
        source_joint = int(
            mujoco.mj_name2id(
                parent_model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
        )
        if source_joint < 0:
            raise ValueError(
                f"canonical planner model lacks RL joint {name!r}"
            )
        target_type = int(rl_model.jnt_type[target_joint])
        source_type = int(parent_model.jnt_type[source_joint])
        if source_type != target_type:
            raise ValueError(f"joint type changed for {name!r}")
        width = _joint_qpos_width(mujoco, target_type)
        target_adr = int(rl_model.jnt_qposadr[target_joint])
        source_adr = int(parent_model.jnt_qposadr[source_joint])
        projection[target_adr : target_adr + width] = np.arange(
            source_adr, source_adr + width, dtype=np.int32
        )
    if np.any(projection < 0):
        raise ValueError("RL qpos projection is incomplete")
    return projection


def _actuator_projection(
    mujoco: Any, parent_model: Any, rl_model: Any
) -> np.ndarray:
    """Map RL controls to identically named canonical actuators."""

    result: list[int] = []
    for target_actuator in range(int(rl_model.nu)):
        name = mujoco.mj_id2name(
            rl_model, mujoco.mjtObj.mjOBJ_ACTUATOR, target_actuator
        )
        if not name:
            raise ValueError("every RL actuator must have a stable name")
        source_actuator = int(
            mujoco.mj_name2id(
                parent_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
        )
        if source_actuator < 0:
            raise ValueError(
                f"canonical planner model lacks RL actuator {name!r}"
            )
        result.append(source_actuator)
    if len(set(result)) != int(rl_model.nu):
        raise ValueError("RL actuator projection is not one-to-one")
    return np.asarray(result, dtype=np.int32)


def _project_qpos(
    parent_qpos: np.ndarray,
    rl_qpos0: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """Project one canonical reset state onto the dense RL model."""

    source = np.asarray(parent_qpos, dtype=np.float64)
    target = np.asarray(rl_qpos0, dtype=np.float64).copy()
    indices = np.asarray(projection, dtype=np.int32)
    if tuple(indices.shape) != tuple(target.shape):
        raise ValueError("qpos projection and RL qpos0 shapes disagree")
    target[:] = source[indices]
    if not np.isfinite(target).all():
        raise FloatingPointError("projected RL qpos template is non-finite")
    return target


def _validate_fixed_root_height(
    mujoco: Any,
    model: Any,
    *,
    expected_root_height_m: float,
    label: str,
) -> None:
    """Require one fixed root at the exact task-specific world height."""

    root_id = _require_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "root")
    if int(model.body_jntnum[root_id]) != 0:
        raise ValueError(f"{label} root must be fixed and have no joint")
    actual = np.asarray(model.body_pos[root_id], dtype=np.float64)
    expected = np.asarray(
        (0.0, 0.0, float(expected_root_height_m)), dtype=np.float64
    )
    if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"{label} root position mismatch: expected {expected.tolist()}, "
            f"got {actual.tolist()}"
        )


def _validate_rl_model(
    mujoco: Any,
    model: Any,
    fixed_object_index: int,
    *,
    expected_root_height_m: float,
) -> None:
    if not 0 <= int(fixed_object_index) < len(OBJECTS):
        raise ValueError("fixed_object_index is outside contract.OBJECTS")
    _validate_fixed_root_height(
        mujoco,
        model,
        expected_root_height_m=expected_root_height_m,
        label="G6 RL shard",
    )
    object_kind = OBJECTS[int(fixed_object_index)]
    actual_dimensions = (int(model.nq), int(model.nv), int(model.nu))
    expected_dimensions = (
        EXPECTED_SHARD_NQ,
        EXPECTED_SHARD_NV,
        EXPECTED_ACTUATORS,
    )
    if actual_dimensions != expected_dimensions:
        raise ValueError(
            f"G6 {object_kind} RL shard must contain one object and resolve "
            f"nq/nv/nu={expected_dimensions}, got {actual_dimensions}"
        )
    if int(model.nv) > 60:
        raise ValueError("MJWarp dense Newton requires nv <= 60")
    if int(model.opt.solver) != int(mujoco.mjtSolver.mjSOL_NEWTON):
        raise ValueError("G6 RL model must explicitly use Newton")
    if (
        int(model.opt.jacobian)
        != int(mujoco.mjtJacobian.mjJAC_DENSE)
        or bool(mujoco.mj_isSparse(model))
    ):
        raise ValueError(
            "G6 RL model must explicitly resolve a dense constraint Jacobian"
        )
    expected_options = {
        "iterations": 150,
        "ls_iterations": 20,
        "tolerance": 1.0e-10,
    }
    actual_options = {
        "iterations": int(model.opt.iterations),
        "ls_iterations": int(model.opt.ls_iterations),
        "tolerance": float(model.opt.tolerance),
    }
    if actual_options != expected_options:
        raise ValueError(
            "G6 RL dense Newton options changed: "
            f"expected {expected_options}, got {actual_options}"
        )
    present = tuple(
        candidate
        for candidate in OBJECTS
        if int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"object_{candidate}",
            )
        )
        >= 0
    )
    if present != (object_kind,):
        raise ValueError(
            f"G6 {object_kind} RL shard must contain exactly its active "
            f"training object, found {present}"
        )
    if int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "object_plate"
        )
    ) >= 0:
        raise ValueError("G6 object shards must not contain the plate")
    free_joint_names = tuple(
        str(
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
        )
        for joint_id in range(int(model.njnt))
        if int(model.jnt_type[joint_id])
        == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    if free_joint_names != (f"joint:object_{object_kind}",):
        raise ValueError(
            "G6 object shard must expose exactly one matching free joint, "
            f"got {free_joint_names}"
        )


def _nominal_grasp_plane_parameters(
    current: Any, object_spec: Any
) -> tuple[float, float | None]:
    """Resolve the RL reset plane from the canonical parent contract."""

    object_kind = str(object_spec.kind.value)
    if object_kind in FIXED_GRASP_PLANE_FRACTIONS:
        object_module = importlib.import_module(
            type(object_spec.kind).__module__
        )
        height_fractions = getattr(
            current, "DIRECT_GRASP_PLANE_HEIGHT_FRACTION", None
        )
        yaw_values = getattr(current, "DIRECT_GRASP_PLANE_YAW_RAD", None)
        if height_fractions is None:
            height_fractions = getattr(
                object_module, "DIRECT_GRASP_PLANE_HEIGHT_FRACTION", None
            )
        if yaw_values is None:
            yaw_values = getattr(
                object_module, "DIRECT_GRASP_PLANE_YAW_RAD", None
            )
        if height_fractions is None or yaw_values is None:
            raise ValueError(
                "canonical G6 direct grasp-plane constants are unavailable"
            )
        try:
            fraction = float(height_fractions[object_spec.kind])
            yaw_rad = float(yaw_values[object_spec.kind])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"canonical G6 direct grasp plane is missing {object_kind!r}"
            ) from exc
        expected_fraction = float(
            FIXED_GRASP_PLANE_FRACTIONS[object_kind]
        )
        expected_yaw = float(FIXED_GRASP_PLANE_YAW_RAD)
        if fraction != expected_fraction or yaw_rad != expected_yaw:
            raise ValueError(
                f"canonical G6 direct grasp plane changed for {object_kind!r}"
            )
    elif object_kind in LEARNED_GRASP_PLANE_OBJECTS:
        fraction = float(GRASP_PLANE_NOMINAL_FRACTION)
        yaw_rad = None
        if fraction != 0.5:
            raise ValueError("G6 fruit RL reset plane must start at 50%")
    else:
        raise ValueError(f"unsupported G6 RL reset object {object_kind!r}")

    plane_z_m = float(
        object_spec.local_bottom_z_m + fraction * object_spec.height_m
    )
    return plane_z_m, yaw_rad


def _nominal_grasp_plane_z(
    object_spec: Any, current: Any | None = None
) -> float:
    """Return the platform's explicit nominal plane height."""

    parent = _parent_module() if current is None else current
    return _nominal_grasp_plane_parameters(parent, object_spec)[0]


def _nominal_grasp_plane_reset_kwargs(
    current: Any, object_spec: Any
) -> dict[str, float]:
    plane_z_m, yaw_rad = _nominal_grasp_plane_parameters(
        current, object_spec
    )
    result = {"demo_grasp_plane_z_m": plane_z_m}
    if yaw_rad is not None:
        result["demo_grasp_plane_yaw_rad"] = yaw_rad
    return result


_PLANE_FRACTION_SCAN_STEP = 0.005
_PLANE_YAW_SCAN_STEP_RAD = math.radians(5.0)


def _symmetric_scan_values(
    center: float, lower: float, upper: float, step: float
) -> tuple[float, ...]:
    """Return center, center-step, center+step, ... with stable rounding."""

    values = [float(center)]
    maximum_steps = int(math.ceil(max(center - lower, upper - center) / step))
    for ordinal in range(1, maximum_steps + 1):
        for sign in (-1.0, 1.0):
            candidate = round(center + sign * ordinal * step, 12)
            if lower - 1.0e-12 <= candidate <= upper + 1.0e-12:
                values.append(candidate)
    return tuple(dict.fromkeys(values))


_PLANE_FRACTION_CANDIDATES = _symmetric_scan_values(
    GRASP_PLANE_NOMINAL_FRACTION,
    GRASP_PLANE_MIN_FRACTION,
    GRASP_PLANE_MAX_FRACTION,
    _PLANE_FRACTION_SCAN_STEP,
)
_PLANE_YAW_OFFSET_CANDIDATES = tuple(
    offset
    for offset in _symmetric_scan_values(
        0.0, -YAW_LIMIT_RAD, YAW_LIMIT_RAD, _PLANE_YAW_SCAN_STEP_RAD
    )
    if abs(offset) > 1.0e-15
)


def _candidate_rejection(exc: RuntimeError) -> bool:
    return str(exc).startswith("rotation demo grasp plane is invalid:")


def _reset_on_feasible_grasp_plane(
    current: Any,
    grabber: Any,
    *,
    object_kind: str,
    seed: int,
) -> tuple[float, float]:
    """Freeze the first deterministic 40--60% plane accepted by parent IK."""

    object_spec = current.get_object_spec(object_kind)
    rejections: list[str] = []
    nominal_yaw_rad: float | None = None

    def attempt(fraction: float, yaw_rad: float | None) -> tuple[float, float] | None:
        nonlocal nominal_yaw_rad
        kwargs: dict[str, float] = {
            "demo_grasp_plane_z_m": float(
                object_spec.local_bottom_z_m + fraction * object_spec.height_m
            )
        }
        if yaw_rad is not None:
            kwargs["demo_grasp_plane_yaw_rad"] = float(yaw_rad)
        try:
            accepted = bool(
                grabber.reset(
                    object_kind,
                    seed=seed,
                    initial_com_z_m=INITIAL_OBJECT_COM_Z_M,
                    **kwargs,
                )
            )
        except RuntimeError as exc:
            if not _candidate_rejection(exc):
                raise RuntimeError(
                    f"unexpected {object_kind} grasp-plane planner failure for "
                    f"seed {seed}, fraction {fraction}, yaw {yaw_rad}"
                ) from exc
            accepted = False
            rejections.append(str(exc))
        plan = grabber.selected_plan
        if plan is not None and nominal_yaw_rad is None:
            nominal_yaw_rad = float(plan.plane.yaw_rad)
        if not accepted:
            reason = str(getattr(grabber, "termination_reason", ""))
            if reason:
                rejections.append(reason)
            return None
        if plan is None:
            raise RuntimeError("accepted parent reset omitted its grasp plan")
        initial_com_z_m = float(grabber._object_com_position()[2])
        if not math.isclose(
            initial_com_z_m,
            INITIAL_OBJECT_COM_Z_M,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise RuntimeError(
                f"{object_kind} parent reset COM z {initial_com_z_m} does not "
                f"match {INITIAL_OBJECT_COM_Z_M} m"
            )
        anchors = np.stack(
            [
                np.asarray(plan.contact_targets_local[finger], dtype=np.float64)
                for finger in TIP_ORDER
            ]
        )
        if anchors.shape != (len(TIP_ORDER), 3) or not np.isfinite(anchors).all():
            raise FloatingPointError("planned hard-tip anchors are non-finite")
        planned_fraction = float(
            (anchors[:, 2].mean() - object_spec.local_bottom_z_m)
            / object_spec.height_m
        )
        planned_yaw = float(plan.plane.yaw_rad)
        if (
            not math.isfinite(planned_fraction)
            or not math.isfinite(planned_yaw)
            or planned_fraction < GRASP_PLANE_MIN_FRACTION - 1.0e-9
            or planned_fraction > GRASP_PLANE_MAX_FRACTION + 1.0e-9
        ):
            raise ValueError("accepted grasp plane escaped the fruit contract")
        return planned_fraction, planned_yaw

    for fraction in _PLANE_FRACTION_CANDIDATES:
        result = attempt(fraction, None)
        if result is not None:
            return result
    if nominal_yaw_rad is None or not math.isfinite(nominal_yaw_rad):
        raise RuntimeError(
            f"{object_kind} seed {seed} did not expose a nominal planner yaw"
        )
    for yaw_offset in _PLANE_YAW_OFFSET_CANDIDATES:
        yaw_rad = math.atan2(
            math.sin(nominal_yaw_rad + yaw_offset),
            math.cos(nominal_yaw_rad + yaw_offset),
        )
        for fraction in _PLANE_FRACTION_CANDIDATES:
            result = attempt(fraction, yaw_rad)
            if result is not None:
                return result
    tail = rejections[-5:]
    raise RuntimeError(
        f"no deterministic 40--60% parent-IK grasp plane for {object_kind}, "
        f"seed {seed}; last rejections={tail!r}"
    )


def _single_slot_bank(
    fixed_object_index: int,
    active_values: np.ndarray | float,
) -> np.ndarray:
    """Place one object's float data in its global four-object slot."""

    active = np.asarray(active_values, dtype=np.float32)
    result = np.full(
        (len(OBJECTS), *active.shape), np.nan, dtype=np.float32
    )
    result[int(fixed_object_index)] = active
    return result


def _collect_parent_templates(current: Any, grabber: Any) -> dict[str, np.ndarray]:
    """Run the canonical IK planner for all active-object reset templates."""

    fields: dict[str, list[np.ndarray | float]] = {
        "qpos": [],
        "ctrl": [],
        "initial_position": [],
        "initial_quaternion": [],
        "pregrasp": [],
        "contact": [],
        "outward": [],
        "planned_fraction": [],
        "planned_yaw": [],
        "bottom": [],
        "top": [],
    }
    for object_index, object_kind in enumerate(OBJECTS):
        object_spec = current.get_object_spec(object_kind)
        fields["bottom"].append(float(object_spec.local_bottom_z_m))
        fields["top"].append(float(object_spec.local_top_z_m))
        rows: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "qpos",
                "ctrl",
                "initial_position",
                "initial_quaternion",
                "pregrasp",
                "contact",
                "outward",
                "planned_fraction",
                "planned_yaw",
            )
        }
        for template_index in range(RESET_TEMPLATES_PER_OBJECT):
            seed = reset_template_seed(object_index, template_index)
            planned_fraction, planned_yaw = _reset_on_feasible_grasp_plane(
                current,
                grabber,
                object_kind=object_kind,
                seed=seed,
            )
            plan = grabber.selected_plan
            if plan is None:
                raise RuntimeError(
                    f"no grasp plan for {object_kind!r}, reset template "
                    f"{template_index}"
                )
            rows["qpos"].append(np.asarray(grabber.data.qpos).copy())
            rows["ctrl"].append(np.asarray(grabber.data.ctrl).copy())
            rows["initial_position"].append(
                np.asarray(grabber.initial_body_pos, dtype=np.float64).copy()
            )
            rows["initial_quaternion"].append(
                np.asarray(grabber.initial_body_quat, dtype=np.float64).copy()
            )
            rows["pregrasp"].append(
                np.stack(
                    [plan.pregrasp_targets_local[finger] for finger in TIP_ORDER]
                )
            )
            rows["contact"].append(
                np.stack(
                    [plan.contact_targets_local[finger] for finger in TIP_ORDER]
                )
            )
            rows["outward"].append(
                np.stack(
                    [plan.outward_directions_local[finger] for finger in TIP_ORDER]
                )
            )
            rows["planned_fraction"].append(planned_fraction)
            rows["planned_yaw"].append(planned_yaw)
        for name, values in rows.items():
            fields[name].append(np.stack(values))
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in fields.items()
    }


def _build_object_shard(
    *,
    current: Any,
    parent_model: Any,
    model: Any,
    fixed_object_index: int,
    parent_templates: dict[str, np.ndarray],
    planner_fingerprint: str,
    controller_fingerprint: str,
    platform_adapter_fingerprint_sha256: str,
    task: str,
    root_height_m: float,
    model_xml: Path,
    task_gripper_xml: Path,
) -> PlatformSpec:
    object_kind = OBJECTS[fixed_object_index]
    object_spec = current.get_object_spec(object_kind)
    _validate_rl_model(
        current.mujoco,
        model,
        fixed_object_index,
        expected_root_height_m=root_height_m,
    )
    if int(model.nu) != EXPECTED_ACTUATORS:
        raise ValueError(
            f"{PLATFORM} expected {EXPECTED_ACTUATORS} actuators, got {model.nu}"
        )
    qpos_projection = _qpos_projection(
        current.mujoco, parent_model, model, fixed_object_index
    )
    actuator_projection = _actuator_projection(
        current.mujoco, parent_model, model
    )
    parent_actuator_names = tuple(
        current.mujoco.mj_id2name(
            parent_model, current.mujoco.mjtObj.mjOBJ_ACTUATOR, index
        )
        for index in range(int(parent_model.nu))
    )
    shard_actuator_names = tuple(
        current.mujoco.mj_id2name(
            model, current.mujoco.mjtObj.mjOBJ_ACTUATOR, index
        )
        for index in range(int(model.nu))
    )
    if parent_actuator_names != shard_actuator_names:
        raise ValueError(
            "canonical planner and RL shard actuator order must be identical"
        )
    if not np.array_equal(
        np.asarray(parent_model.actuator_ctrlrange),
        np.asarray(model.actuator_ctrlrange),
    ):
        raise ValueError(
            "canonical planner and RL actuator ranges must be identical"
        )
    qpos_active = np.stack(
        [
            _project_qpos(parent_qpos, model.qpos0, qpos_projection)
            for parent_qpos in parent_templates["qpos"][fixed_object_index]
        ]
    )
    ctrl_active = parent_templates["ctrl"][fixed_object_index][
        :, actuator_projection
    ]

    joint_id = _require_id(
        current.mujoco,
        model,
        current.mujoco.mjtObj.mjOBJ_JOINT,
        object_spec.joint_name,
    )
    body_id = _require_id(
        current.mujoco,
        model,
        current.mujoco.mjtObj.mjOBJ_BODY,
        object_spec.body_name,
    )
    geom_ids = np.asarray(
        [
            _require_id(
                current.mujoco,
                model,
                current.mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
            for name in object_spec.geom_names
        ],
        dtype=np.int32,
    )
    geom_mask = np.zeros(int(model.ngeom), dtype=bool)
    geom_mask[geom_ids] = True
    model_mass = float(model.body_mass[body_id])
    if not np.isclose(
        model_mass, float(object_spec.mass_kg), rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            f"RL model mass mismatch for {object_kind!r}: {model_mass}"
        )

    actuator_names = shard_actuator_names
    if any(name is None for name in actuator_names):
        raise ValueError("every actuator must have a stable name")
    actuator_joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
    actuator_qpos_adrs = np.asarray(
        model.jnt_qposadr[actuator_joint_ids], dtype=np.int32
    )
    actuator_dof_adrs = np.asarray(
        model.jnt_dofadr[actuator_joint_ids], dtype=np.int32
    )
    actuator_finger_mask = np.zeros(
        (len(TIP_ORDER), int(model.nu)), dtype=np.float32
    )
    for actuator_index, actuator_name in enumerate(actuator_names):
        finger = _finger_from_actuator(str(actuator_name))
        actuator_finger_mask[TIP_ORDER.index(finger), actuator_index] = 1.0
    actuators_per_finger = actuator_finger_mask.sum(axis=1).astype(np.int32)
    if not np.all(actuators_per_finger == actuators_per_finger[0]):
        raise ValueError("every finger must expose the same number of actuators")
    finger_actuator_indices = np.stack(
        [
            np.flatnonzero(actuator_finger_mask[index]).astype(np.int32)
            for index in range(len(TIP_ORDER))
        ]
    )

    tip_body_ids = np.asarray(
        [
            _require_id(
                current.mujoco,
                model,
                current.mujoco.mjtObj.mjOBJ_BODY,
                f"hard_tip_{finger}",
            )
            for finger in TIP_ORDER
        ],
        dtype=np.int32,
    )
    tip_geom_ids = np.asarray(
        [
            _require_id(
                current.mujoco,
                model,
                current.mujoco.mjtObj.mjOBJ_GEOM,
                f"hard_tip_{finger}",
            )
            for finger in TIP_ORDER
        ],
        dtype=np.int32,
    )
    passive_joint_ids = np.asarray(
        [
            _require_id(
                current.mujoco,
                model,
                current.mujoco.mjtObj.mjOBJ_JOINT,
                f"robot0:{joint}_joint_{finger}",
            )
            for finger in TIP_ORDER
            for joint in ("short", "elbow")
        ],
        dtype=np.int32,
    )
    passive_dof_adrs = np.asarray(
        model.jnt_dofadr[passive_joint_ids], dtype=np.int32
    )
    passive_qpos_adrs = np.asarray(
        model.jnt_qposadr[passive_joint_ids], dtype=np.int32
    )
    passive_qpos_low = np.asarray(
        model.jnt_range[passive_joint_ids, 0], dtype=np.float32
    )
    passive_qpos_high = np.asarray(
        model.jnt_range[passive_joint_ids, 1], dtype=np.float32
    )
    finger_passive_indices = np.arange(
        passive_joint_ids.size, dtype=np.int32
    ).reshape(len(TIP_ORDER), -1)
    pin_site_ids = np.asarray(
        [
            [
                _require_id(
                    current.mujoco,
                    model,
                    current.mujoco.mjtObj.mjOBJ_SITE,
                    f"pin:{finger}_{branch}",
                )
                for branch in ("long", "front")
            ]
            for finger in TIP_ORDER
        ],
        dtype=np.int32,
    )
    pin_body_ids = np.asarray(model.site_bodyid[pin_site_ids], dtype=np.int32)
    model_xml_sha256 = _sha256(model_xml)
    rl_fingerprint = _task_shard_fingerprint(
        model_xml,
        task_gripper_xml,
        task=task,
        root_height_m=root_height_m,
    )
    child_bound_fingerprint = _bound_model_fingerprint(
        planner_fingerprint,
        rl_fingerprint,
        task=task,
        root_height_m=root_height_m,
    )
    object_body_ids = np.full(len(OBJECTS), body_id, dtype=np.int32)
    object_qpos_adrs = np.full(
        len(OBJECTS), int(model.jnt_qposadr[joint_id]), dtype=np.int32
    )
    object_qvel_adrs = np.full(
        len(OBJECTS), int(model.jnt_dofadr[joint_id]), dtype=np.int32
    )
    object_geom_mask = np.repeat(
        geom_mask[None, :], len(OBJECTS), axis=0
    )

    return PlatformSpec(
        platform=PLATFORM,
        task=task,
        task_index=TASKS.index(task),
        root_height_m=float(root_height_m),
        fixed_object_index=fixed_object_index,
        model_xml=model_xml,
        model_xml_sha256=model_xml_sha256,
        platform_adapter_fingerprint_sha256=(
            platform_adapter_fingerprint_sha256
        ),
        parent_model_fingerprint_sha256=child_bound_fingerprint,
        parent_controller_fingerprint_sha256=controller_fingerprint,
        planner_model_fingerprint_sha256=planner_fingerprint,
        rl_model_fingerprint_sha256=rl_fingerprint,
        host_model=model,
        qpos_templates=_single_slot_bank(fixed_object_index, qpos_active),
        ctrl_templates=_single_slot_bank(fixed_object_index, ctrl_active),
        initial_object_pos=_single_slot_bank(
            fixed_object_index,
            parent_templates["initial_position"][fixed_object_index],
        ),
        initial_object_quat=_single_slot_bank(
            fixed_object_index,
            parent_templates["initial_quaternion"][fixed_object_index],
        ),
        pregrasp_targets_local=_single_slot_bank(
            fixed_object_index,
            parent_templates["pregrasp"][fixed_object_index],
        ),
        contact_targets_local=_single_slot_bank(
            fixed_object_index,
            parent_templates["contact"][fixed_object_index],
        ),
        outward_directions_local=_single_slot_bank(
            fixed_object_index,
            parent_templates["outward"][fixed_object_index],
        ),
        planned_grasp_plane_fraction=_single_slot_bank(
            fixed_object_index,
            parent_templates["planned_fraction"][fixed_object_index],
        ),
        planned_grasp_plane_yaw_rad=_single_slot_bank(
            fixed_object_index,
            parent_templates["planned_yaw"][fixed_object_index],
        ),
        object_local_bottom_z_m=_single_slot_bank(
            fixed_object_index, parent_templates["bottom"][fixed_object_index]
        ),
        object_local_top_z_m=_single_slot_bank(
            fixed_object_index, parent_templates["top"][fixed_object_index]
        ),
        object_body_ids=object_body_ids,
        object_qpos_adrs=object_qpos_adrs,
        object_qvel_adrs=object_qvel_adrs,
        object_geom_mask=object_geom_mask,
        tip_body_ids=tip_body_ids,
        tip_geom_ids=tip_geom_ids,
        actuator_qpos_adrs=actuator_qpos_adrs,
        actuator_dof_adrs=actuator_dof_adrs,
        passive_qpos_adrs=passive_qpos_adrs,
        passive_dof_adrs=passive_dof_adrs,
        passive_qpos_low=passive_qpos_low,
        passive_qpos_high=passive_qpos_high,
        pin_site_ids=pin_site_ids,
        pin_body_ids=pin_body_ids,
        actuator_finger_mask=actuator_finger_mask,
        finger_actuator_indices=finger_actuator_indices,
        finger_passive_indices=finger_passive_indices,
        ctrl_low=np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float32),
        ctrl_high=np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float32),
    )


def build_platform_spec(task: str) -> ShardedPlatformSpec:
    """Build one explicit task's four fixed-object dense-Newton specs."""

    selected_task = validate_task(task)
    root_height_m = float(TASK_ROOT_HEIGHT_M[selected_task])
    model_xmls = model_xmls_for_task(selected_task)
    task_gripper_xml = TASK_GRIPPER_XMLS[selected_task]
    for path in (PARENT_MODEL_XML, task_gripper_xml, *model_xmls.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    if GRASP_PLANE_NOMINAL_FRACTION != 0.50:
        raise ValueError("G6 fruit reset bank must start at the exact 50% plane")

    current = _parent_module()
    if Path(current.MODEL_XML).resolve() != PARENT_MODEL_XML.resolve():
        raise ValueError(
            "canonical G6 controller is not bound to _g6_project.xml"
        )
    grabber = current.IKGrabber(
        render=False, root_height_m=root_height_m
    )
    grabber.planner.enforce_anti_support = False
    parent_model = grabber.model
    try:
        _validate_fixed_root_height(
            current.mujoco,
            parent_model,
            expected_root_height_m=root_height_m,
            label="G6 canonical planner",
        )
        if (
            int(parent_model.opt.jacobian)
            != int(current.mujoco.mjtJacobian.mjJAC_SPARSE)
            or not bool(current.mujoco.mj_isSparse(parent_model))
        ):
            raise ValueError(
                "canonical parent planner must retain its sparse constraint "
                "Jacobian"
            )
        parent_templates = _collect_parent_templates(current, grabber)
        parent_model.geom_contype[:] = grabber._original_geom_contype
        parent_model.geom_conaffinity[:] = grabber._original_geom_conaffinity
    finally:
        grabber.close()

    models = {
        object_kind: current.mujoco.MjModel.from_xml_path(str(model_xml))
        for object_kind, model_xml in model_xmls.items()
    }
    planner_fingerprint = str(current.MODEL_FINGERPRINT_SHA256)
    controller_fingerprint = str(current.CONTROLLER_FINGERPRINT_SHA256)
    platform_adapter_fingerprint = _sha256(PLATFORM_ADAPTER_PATH)
    shards = tuple(
        _build_object_shard(
            current=current,
            parent_model=parent_model,
            model=models[object_kind],
            fixed_object_index=object_index,
            parent_templates=parent_templates,
            planner_fingerprint=planner_fingerprint,
            controller_fingerprint=controller_fingerprint,
            platform_adapter_fingerprint_sha256=(
                platform_adapter_fingerprint
            ),
            task=selected_task,
            root_height_m=root_height_m,
            model_xml=model_xmls[object_kind],
            task_gripper_xml=task_gripper_xml,
        )
        for object_index, object_kind in enumerate(OBJECTS)
    )
    if tuple(shard.fixed_object_index for shard in shards) != tuple(
        range(len(OBJECTS))
    ):
        raise RuntimeError("object shard order differs from contract.OBJECTS")
    aggregate_rl_fingerprint = _ordered_shard_fingerprint(
        [shard.rl_model_fingerprint_sha256 for shard in shards],
        task=selected_task,
        root_height_m=root_height_m,
    )
    return ShardedPlatformSpec(
        platform=PLATFORM,
        task=selected_task,
        task_index=TASKS.index(selected_task),
        root_height_m=root_height_m,
        platform_adapter_fingerprint_sha256=platform_adapter_fingerprint,
        parent_model_fingerprint_sha256=_bound_model_fingerprint(
            planner_fingerprint,
            aggregate_rl_fingerprint,
            task=selected_task,
            root_height_m=root_height_m,
        ),
        parent_controller_fingerprint_sha256=controller_fingerprint,
        planner_model_fingerprint_sha256=planner_fingerprint,
        rl_model_fingerprint_sha256=aggregate_rl_fingerprint,
        shards=shards,
    )


__all__ = [
    "EXPECTED_ACTUATORS",
    "EXPECTED_RL_NQ",
    "EXPECTED_RL_NV",
    "EXPECTED_SHARD_NQ",
    "EXPECTED_SHARD_NV",
    "MODEL_XMLS_BY_TASK",
    "PACKAGE_DIR",
    "PARENT_MODEL_XML",
    "PLATFORM",
    "PLATFORM_ADAPTER_PATH",
    "PROJECT_ROOT",
    "TASK_GRIPPER_XMLS",
    "TASK_ROOT_HEIGHT_M",
    "PlatformSpec",
    "ShardedPlatformSpec",
    "build_platform_spec",
    "model_xmls_for_task",
]
