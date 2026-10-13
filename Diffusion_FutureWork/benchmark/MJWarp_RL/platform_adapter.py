"""Gripper3 host-side adapter for the shared sharded MJWarp controller.

The canonical multi-object model remains the single IK/planning source of
truth.  Physics is split into one dense-Newton MuJoCo model per trained
object, keeping every GPU simulation shard at nq=27/nv=26/nu=12.
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


PLATFORM = "gripper3"
PACKAGE_DIR = Path(__file__).resolve().parent
PLATFORM_ADAPTER_PATH = Path(__file__).resolve()
PLATFORM_ADAPTER_FINGERPRINT_SHA256 = hashlib.sha256(
    PLATFORM_ADAPTER_PATH.read_bytes()
).hexdigest()
PROJECT_ROOT = PACKAGE_DIR.parent
PARENT_MODEL_XML = PROJECT_ROOT / "G3_Project.xml"
TASK_ROOT_HEIGHT_M = {
    TASK_TRANSLATE: 0.525,
    TASK_ROTATE: 0.525,
}
TASK_GRIPPER_XMLS = {
    task: PROJECT_ROOT / f"G3_Gripper_{task}.xml" for task in TASKS
}
MODEL_XMLS_BY_TASK = {
    task: {
        object_kind: PROJECT_ROOT / f"G3_Project_RL_{task}_{object_kind}.xml"
        for object_kind in OBJECTS
    }
    for task in TASKS
}
PARENT_MODULE = "G3_grab"
EXPECTED_ACTUATORS = 12
EXPECTED_SHARD_NQ = 27
EXPECTED_SHARD_NV = 26


def model_xmls_for_task(task: str) -> dict[str, Path]:
    """Return a copy of the ordered four-shard XML mapping for one task."""

    selected = validate_task(task)
    result = dict(MODEL_XMLS_BY_TASK[selected])
    if tuple(result) != tuple(OBJECTS):
        raise RuntimeError("G3 task shard order differs from contract.OBJECTS")
    return result


@dataclass(frozen=True)
class PlatformSpec:
    """One fixed-object model plus its fail-closed four-slot host arrays."""

    platform: str
    task: str
    task_index: int
    root_height_m: float
    fixed_object_index: int
    model_xml: Path
    model_xml_sha256: str
    parent_model_fingerprint_sha256: str
    parent_controller_fingerprint_sha256: str
    platform_adapter_fingerprint_sha256: str
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
    """Ordered four-object platform whose children own separate GPU systems."""

    platform: str
    task: str
    task_index: int
    root_height_m: float
    parent_model_fingerprint_sha256: str
    parent_controller_fingerprint_sha256: str
    platform_adapter_fingerprint_sha256: str
    planner_model_fingerprint_sha256: str
    rl_model_fingerprint_sha256: str
    shards: tuple[PlatformSpec, ...]

    @property
    def host_model(self) -> Any:
        """Representative model for legacy dimension/introspection callers."""

        if not self.shards:
            raise RuntimeError("sharded platform has no child models")
        return self.shards[0].host_model


@dataclass(frozen=True)
class _CanonicalResetBank:
    """All active-object IK resets generated once from the canonical planner."""

    planner_model: Any
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


def _parent_module() -> Any:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(PARENT_MODULE)


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
                "canonical G3 direct grasp-plane constants are unavailable"
            )
        try:
            fraction = float(height_fractions[object_spec.kind])
            yaw_rad = float(yaw_values[object_spec.kind])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"canonical G3 direct grasp plane is missing {object_kind!r}"
            ) from exc
        expected_fraction = float(
            FIXED_GRASP_PLANE_FRACTIONS[object_kind]
        )
        expected_yaw = float(FIXED_GRASP_PLANE_YAW_RAD)
        if fraction != expected_fraction or yaw_rad != expected_yaw:
            raise ValueError(
                f"canonical G3 direct grasp plane changed for {object_kind!r}"
            )
    elif object_kind in LEARNED_GRASP_PLANE_OBJECTS:
        fraction = float(GRASP_PLANE_NOMINAL_FRACTION)
        yaw_rad = None
        if fraction != 0.5:
            raise ValueError("G3 fruit RL reset plane must start at 50%")
    else:
        raise ValueError(f"unsupported G3 RL reset object {object_kind!r}")

    plane_z_m = float(
        object_spec.local_bottom_z_m + fraction * object_spec.height_m
    )
    return plane_z_m, yaw_rad


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
            accepted = bool(grabber.reset(object_kind, seed=seed, **kwargs))
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
            if not rejections or rejections[-1] != str(
                getattr(grabber, "termination_reason", "")
            ):
                reason = str(getattr(grabber, "termination_reason", ""))
                if reason:
                    rejections.append(reason)
            return None
        if plan is None:
            raise RuntimeError("accepted parent reset omitted its grasp plan")
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


def _finger_from_actuator(name: str) -> str:
    prefix = name.split("_", 1)[0]
    if prefix not in TIP_ORDER:
        raise ValueError(f"cannot assign actuator {name!r} to a finger")
    return prefix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
        ("canonical-planner", planner_fingerprint),
        ("dense-newton-rl", rl_fingerprint),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _qpos_width(mujoco: Any, joint_type: int) -> int:
    widths = {
        int(mujoco.mjtJoint.mjJNT_FREE): 7,
        int(mujoco.mjtJoint.mjJNT_BALL): 4,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }
    try:
        return widths[int(joint_type)]
    except KeyError as exc:
        raise ValueError(f"unsupported MuJoCo joint type: {joint_type}") from exc


def _expected_omitted_joint_names(fixed_object_index: int) -> frozenset[str]:
    if not 0 <= int(fixed_object_index) < len(OBJECTS):
        raise ValueError(f"invalid fixed object index: {fixed_object_index}")
    current = _parent_module()
    # The canonical G3 planner retains cube because its legacy kinematics
    # parser still uses that body as a geometry reference.  Cube is not a
    # training object and every single-object RL shard must omit it.
    omitted = {"joint:object_plate", "joint:object_cube"}
    omitted.update(
        current.get_object_spec(object_kind).joint_name
        for object_index, object_kind in enumerate(OBJECTS)
        if object_index != int(fixed_object_index)
    )
    omitted.update(
        current.get_object_spec(object_kind).joint_name
        for object_kind in INFEASIBLE_OBJECTS
    )
    return frozenset(omitted)


def _qpos_projection(
    mujoco: Any,
    planner_model: Any,
    rl_model: Any,
    fixed_object_index: int,
) -> tuple[tuple[int, int, int, str], ...]:
    """Return a strict name-based canonical-to-single-object projection."""

    planner_names: set[str] = set()
    for planner_joint in range(int(planner_model.njnt)):
        planner_name = mujoco.mj_id2name(
            planner_model, mujoco.mjtObj.mjOBJ_JOINT, planner_joint
        )
        if planner_name is None or planner_name in planner_names:
            raise ValueError(
                "every canonical planner joint must have a unique stable name"
            )
        planner_names.add(planner_name)

    projection: list[tuple[int, int, int, str]] = []
    rl_names: set[str] = set()
    for rl_joint in range(int(rl_model.njnt)):
        name = mujoco.mj_id2name(
            rl_model, mujoco.mjtObj.mjOBJ_JOINT, rl_joint
        )
        if name is None or name in rl_names:
            raise ValueError("every RL joint must have a unique stable name")
        rl_names.add(name)
        planner_joint = int(
            mujoco.mj_name2id(
                planner_model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
        )
        if planner_joint < 0:
            raise ValueError(f"RL joint is absent from planner model: {name!r}")
        planner_type = int(planner_model.jnt_type[planner_joint])
        rl_type = int(rl_model.jnt_type[rl_joint])
        if planner_type != rl_type:
            raise ValueError(f"joint type differs between models: {name!r}")
        projection.append(
            (
                int(planner_model.jnt_qposadr[planner_joint]),
                int(rl_model.jnt_qposadr[rl_joint]),
                _qpos_width(mujoco, rl_type),
                name,
            )
        )

    omitted = planner_names - rl_names
    expected = _expected_omitted_joint_names(fixed_object_index)
    if omitted != expected:
        raise ValueError(
            "single-object RL model must omit exactly the plate and every "
            f"inactive object; expected {sorted(expected)!r}, got "
            f"{sorted(omitted)!r}"
        )
    return tuple(projection)


def _project_qpos(
    planner_qpos: np.ndarray,
    rl_qpos0: np.ndarray,
    projection: tuple[tuple[int, int, int, str], ...],
) -> np.ndarray:
    result = np.asarray(rl_qpos0, dtype=np.float64).copy()
    source = np.asarray(planner_qpos, dtype=np.float64)
    for planner_address, rl_address, width, _name in projection:
        result[rl_address : rl_address + width] = source[
            planner_address : planner_address + width
        ]
    if not np.isfinite(result).all():
        raise FloatingPointError("projected RL qpos template is non-finite")
    return result


def _require_id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"RL model object not found: {name!r}")
    return value


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
    """Fail closed unless this is exactly one dense-Newton object shard."""

    current = _parent_module()
    _validate_fixed_root_height(
        mujoco,
        model,
        expected_root_height_m=expected_root_height_m,
        label="G3 RL shard",
    )
    if int(model.opt.solver) != int(mujoco.mjtSolver.mjSOL_NEWTON):
        raise ValueError("G3 MJWarp RL shard must use the Newton solver")
    if (
        int(model.opt.jacobian) != int(mujoco.mjtJacobian.mjJAC_DENSE)
        or bool(mujoco.mj_isSparse(model))
    ):
        raise ValueError(
            "G3 MJWarp RL shard must explicitly resolve a dense Jacobian"
        )
    if int(model.nv) > 60:
        raise ValueError(f"G3 RL shard exceeds nv<=60: nv={model.nv}")
    if (
        int(model.nq),
        int(model.nv),
        int(model.nu),
    ) != (EXPECTED_SHARD_NQ, EXPECTED_SHARD_NV, EXPECTED_ACTUATORS):
        raise ValueError(
            "G3 RL shard must contain exactly the gripper and one training "
            f"object, got nq/nv/nu={model.nq}/{model.nv}/{model.nu}"
        )
    if int(fixed_object_index) not in range(len(OBJECTS)):
        raise ValueError(f"invalid fixed object index: {fixed_object_index}")
    for object_index, object_kind in enumerate(OBJECTS):
        object_spec = current.get_object_spec(object_kind)
        body_id = int(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, object_spec.body_name
            )
        )
        joint_id = int(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, object_spec.joint_name
            )
        )
        should_exist = object_index == int(fixed_object_index)
        if (body_id >= 0) != should_exist or (joint_id >= 0) != should_exist:
            raise ValueError(
                f"RL shard object membership mismatch for {object_kind!r}"
            )
    if int(
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "joint:object_plate"
        )
    ) >= 0:
        raise ValueError("plate free joint must be absent from every RL shard")


def _build_canonical_reset_bank(
    current: Any, *, root_height_m: float
) -> _CanonicalResetBank:
    """Run exactly OBJECTS*32 canonical IK resets once."""

    grabber = current.IKGrabber(
        render=False, root_height_m=float(root_height_m)
    )
    grabber.planner.enforce_anti_support = False
    planner_model = grabber.model
    _validate_fixed_root_height(
        current.mujoco,
        planner_model,
        expected_root_height_m=root_height_m,
        label="G3 canonical planner",
    )
    if (
        int(planner_model.opt.jacobian)
        != int(current.mujoco.mjtJacobian.mjJAC_SPARSE)
        or not bool(current.mujoco.mj_isSparse(planner_model))
    ):
        raise ValueError(
            "canonical G3 planner must resolve a sparse constraint Jacobian "
            "because nv > 60"
        )

    qpos_templates: list[np.ndarray] = []
    ctrl_templates: list[np.ndarray] = []
    initial_positions: list[np.ndarray] = []
    initial_quaternions: list[np.ndarray] = []
    pregrasp_targets: list[np.ndarray] = []
    contact_targets: list[np.ndarray] = []
    outward_directions: list[np.ndarray] = []
    planned_fractions: list[np.ndarray] = []
    planned_yaws: list[np.ndarray] = []
    object_local_bottom_z_m: list[float] = []
    object_local_top_z_m: list[float] = []
    try:
        for object_index, object_kind in enumerate(OBJECTS):
            object_spec = current.get_object_spec(object_kind)
            object_local_bottom_z_m.append(float(object_spec.local_bottom_z_m))
            object_local_top_z_m.append(float(object_spec.local_top_z_m))
            object_qpos: list[np.ndarray] = []
            object_ctrl: list[np.ndarray] = []
            object_initial_pos: list[np.ndarray] = []
            object_initial_quat: list[np.ndarray] = []
            object_pregrasp: list[np.ndarray] = []
            object_contact: list[np.ndarray] = []
            object_outward: list[np.ndarray] = []
            object_fractions: list[float] = []
            object_yaws: list[float] = []
            for template_index in range(RESET_TEMPLATES_PER_OBJECT):
                seed = reset_template_seed(object_index, template_index)
                fraction, yaw_rad = _reset_on_feasible_grasp_plane(
                    current,
                    grabber,
                    object_kind=object_kind,
                    seed=seed,
                )
                if grabber.selected_plan is None:
                    raise RuntimeError(
                        f"no grasp plan for {object_kind!r}, reset template "
                        f"{template_index}"
                    )
                object_qpos.append(np.asarray(grabber.data.qpos).copy())
                object_ctrl.append(np.asarray(grabber.data.ctrl).copy())
                object_initial_pos.append(
                    np.asarray(grabber.initial_body_pos, dtype=np.float64).copy()
                )
                object_initial_quat.append(
                    np.asarray(grabber.initial_body_quat, dtype=np.float64).copy()
                )
                object_pregrasp.append(
                    np.stack(
                        [
                            grabber.selected_plan.pregrasp_targets_local[finger]
                            for finger in TIP_ORDER
                        ]
                    )
                )
                object_contact.append(
                    np.stack(
                        [
                            grabber.selected_plan.contact_targets_local[finger]
                            for finger in TIP_ORDER
                        ]
                    )
                )
                object_outward.append(
                    np.stack(
                        [
                            grabber.selected_plan.outward_directions_local[finger]
                            for finger in TIP_ORDER
                        ]
                    )
                )
                object_fractions.append(fraction)
                object_yaws.append(yaw_rad)
            qpos_templates.append(np.stack(object_qpos))
            ctrl_templates.append(np.stack(object_ctrl))
            initial_positions.append(np.stack(object_initial_pos))
            initial_quaternions.append(np.stack(object_initial_quat))
            pregrasp_targets.append(np.stack(object_pregrasp))
            contact_targets.append(np.stack(object_contact))
            outward_directions.append(np.stack(object_outward))
            planned_fractions.append(
                np.asarray(object_fractions, dtype=np.float64)
            )
            planned_yaws.append(np.asarray(object_yaws, dtype=np.float64))
    finally:
        grabber.close()

    return _CanonicalResetBank(
        planner_model=planner_model,
        qpos_templates=np.asarray(qpos_templates, dtype=np.float64),
        ctrl_templates=np.asarray(ctrl_templates, dtype=np.float64),
        initial_object_pos=np.asarray(initial_positions, dtype=np.float64),
        initial_object_quat=np.asarray(initial_quaternions, dtype=np.float64),
        pregrasp_targets_local=np.asarray(pregrasp_targets, dtype=np.float64),
        contact_targets_local=np.asarray(contact_targets, dtype=np.float64),
        outward_directions_local=np.asarray(
            outward_directions, dtype=np.float64
        ),
        planned_grasp_plane_fraction=np.asarray(
            planned_fractions, dtype=np.float64
        ),
        planned_grasp_plane_yaw_rad=np.asarray(
            planned_yaws, dtype=np.float64
        ),
        object_local_bottom_z_m=np.asarray(
            object_local_bottom_z_m, dtype=np.float64
        ),
        object_local_top_z_m=np.asarray(
            object_local_top_z_m, dtype=np.float64
        ),
    )


def _fail_closed_bank(
    active: np.ndarray,
    fixed_object_index: int,
) -> np.ndarray:
    """Place active rows in one slot and poison every unsupported slot."""

    source = np.asarray(active)
    bank = np.full((len(OBJECTS), *source.shape), np.nan, dtype=np.float32)
    bank[int(fixed_object_index)] = source.astype(np.float32)
    return bank


def _actuator_metadata(current: Any, planner_model: Any, model: Any) -> dict[str, np.ndarray]:
    actuator_names = tuple(
        current.mujoco.mj_id2name(
            model, current.mujoco.mjtObj.mjOBJ_ACTUATOR, index
        )
        for index in range(int(model.nu))
    )
    if any(name is None for name in actuator_names):
        raise ValueError("every actuator must have a stable name")
    planner_names = tuple(
        current.mujoco.mj_id2name(
            planner_model, current.mujoco.mjtObj.mjOBJ_ACTUATOR, index
        )
        for index in range(int(planner_model.nu))
    )
    if planner_names != actuator_names:
        raise ValueError(
            "canonical planner and RL shard actuator order must be identical"
        )
    if not np.array_equal(
        np.asarray(planner_model.actuator_ctrlrange),
        np.asarray(model.actuator_ctrlrange),
    ):
        raise ValueError(
            "canonical planner and RL actuator ranges must be identical"
        )

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
    return {
        "actuator_qpos_adrs": actuator_qpos_adrs,
        "actuator_dof_adrs": actuator_dof_adrs,
        "actuator_finger_mask": actuator_finger_mask,
        "finger_actuator_indices": finger_actuator_indices,
    }


def _build_platform_shard(
    current: Any,
    canonical: _CanonicalResetBank,
    fixed_object_index: int,
    planner_fingerprint: str,
    controller_fingerprint: str,
    *,
    task: str,
    root_height_m: float,
    model_xml: Path,
    task_gripper_xml: Path,
) -> PlatformSpec:
    object_kind = OBJECTS[int(fixed_object_index)]
    model = current.mujoco.MjModel.from_xml_path(str(model_xml.resolve()))
    _validate_rl_model(
        current.mujoco,
        model,
        fixed_object_index,
        expected_root_height_m=root_height_m,
    )
    projection = _qpos_projection(
        current.mujoco,
        canonical.planner_model,
        model,
        fixed_object_index,
    )
    active_qpos = np.stack(
        [
            _project_qpos(source, np.asarray(model.qpos0), projection)
            for source in canonical.qpos_templates[fixed_object_index]
        ]
    )

    object_spec = current.get_object_spec(object_kind)
    joint_id = _require_id(
        current.mujoco,
        model,
        current.mujoco.mjtObj.mjOBJ_JOINT,
        object_spec.joint_name,
    )
    if int(model.jnt_type[joint_id]) != int(
        current.mujoco.mjtJoint.mjJNT_FREE
    ):
        raise ValueError(f"RL training object is not free: {object_kind!r}")
    body_id = _require_id(
        current.mujoco,
        model,
        current.mujoco.mjtObj.mjOBJ_BODY,
        object_spec.body_name,
    )
    geom_ids = tuple(
        _require_id(
            current.mujoco,
            model,
            current.mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        for name in object_spec.geom_names
    )
    geom_mask = np.zeros(int(model.ngeom), dtype=bool)
    geom_mask[np.asarray(geom_ids, dtype=np.int32)] = True
    if not np.isclose(
        float(model.body_mass[body_id]),
        float(object_spec.mass_kg),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"RL model mass differs from {object_kind!r} spec")

    actuator = _actuator_metadata(current, canonical.planner_model, model)
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
    model_xml_sha256 = _sha256(model_xml)
    rl_fingerprint = _task_shard_fingerprint(
        model_xml,
        task_gripper_xml,
        task=task,
        root_height_m=root_height_m,
    )
    object_slots = len(OBJECTS)
    return PlatformSpec(
        platform=PLATFORM,
        task=task,
        task_index=TASKS.index(task),
        root_height_m=float(root_height_m),
        fixed_object_index=int(fixed_object_index),
        model_xml=model_xml,
        model_xml_sha256=model_xml_sha256,
        parent_model_fingerprint_sha256=_bound_model_fingerprint(
            planner_fingerprint,
            rl_fingerprint,
            task=task,
            root_height_m=root_height_m,
        ),
        parent_controller_fingerprint_sha256=controller_fingerprint,
        platform_adapter_fingerprint_sha256=(
            PLATFORM_ADAPTER_FINGERPRINT_SHA256
        ),
        planner_model_fingerprint_sha256=planner_fingerprint,
        rl_model_fingerprint_sha256=rl_fingerprint,
        host_model=model,
        qpos_templates=_fail_closed_bank(active_qpos, fixed_object_index),
        ctrl_templates=_fail_closed_bank(
            canonical.ctrl_templates[fixed_object_index], fixed_object_index
        ),
        initial_object_pos=_fail_closed_bank(
            canonical.initial_object_pos[fixed_object_index], fixed_object_index
        ),
        initial_object_quat=_fail_closed_bank(
            canonical.initial_object_quat[fixed_object_index], fixed_object_index
        ),
        pregrasp_targets_local=_fail_closed_bank(
            canonical.pregrasp_targets_local[fixed_object_index],
            fixed_object_index,
        ),
        contact_targets_local=_fail_closed_bank(
            canonical.contact_targets_local[fixed_object_index],
            fixed_object_index,
        ),
        outward_directions_local=_fail_closed_bank(
            canonical.outward_directions_local[fixed_object_index],
            fixed_object_index,
        ),
        planned_grasp_plane_fraction=_fail_closed_bank(
            canonical.planned_grasp_plane_fraction[fixed_object_index],
            fixed_object_index,
        ),
        planned_grasp_plane_yaw_rad=_fail_closed_bank(
            canonical.planned_grasp_plane_yaw_rad[fixed_object_index],
            fixed_object_index,
        ),
        object_local_bottom_z_m=_fail_closed_bank(
            canonical.object_local_bottom_z_m[fixed_object_index : fixed_object_index + 1],
            fixed_object_index,
        ).reshape(object_slots, 1)[:, 0],
        object_local_top_z_m=_fail_closed_bank(
            canonical.object_local_top_z_m[fixed_object_index : fixed_object_index + 1],
            fixed_object_index,
        ).reshape(object_slots, 1)[:, 0],
        object_body_ids=np.full(object_slots, body_id, dtype=np.int32),
        object_qpos_adrs=np.full(
            object_slots, int(model.jnt_qposadr[joint_id]), dtype=np.int32
        ),
        object_qvel_adrs=np.full(
            object_slots, int(model.jnt_dofadr[joint_id]), dtype=np.int32
        ),
        object_geom_mask=np.repeat(
            geom_mask[np.newaxis, :], object_slots, axis=0
        ),
        tip_body_ids=tip_body_ids,
        tip_geom_ids=tip_geom_ids,
        actuator_qpos_adrs=actuator["actuator_qpos_adrs"],
        actuator_dof_adrs=actuator["actuator_dof_adrs"],
        passive_qpos_adrs=passive_qpos_adrs,
        passive_dof_adrs=passive_dof_adrs,
        passive_qpos_low=np.asarray(
            model.jnt_range[passive_joint_ids, 0], dtype=np.float32
        ),
        passive_qpos_high=np.asarray(
            model.jnt_range[passive_joint_ids, 1], dtype=np.float32
        ),
        pin_site_ids=pin_site_ids,
        pin_body_ids=np.asarray(model.site_bodyid[pin_site_ids], dtype=np.int32),
        actuator_finger_mask=actuator["actuator_finger_mask"],
        finger_actuator_indices=actuator["finger_actuator_indices"],
        finger_passive_indices=np.arange(
            passive_joint_ids.size, dtype=np.int32
        ).reshape(len(TIP_ORDER), -1),
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
    current = _parent_module()
    if Path(current.MODEL_XML).resolve() != PARENT_MODEL_XML.resolve():
        raise ValueError(
            "G3_grab must retain canonical G3_Project.xml as its planner model"
        )
    canonical = _build_canonical_reset_bank(
        current, root_height_m=root_height_m
    )
    planner_fingerprint = str(current.MODEL_FINGERPRINT_SHA256)
    controller_fingerprint = str(current.CONTROLLER_FINGERPRINT_SHA256)
    shards = tuple(
        _build_platform_shard(
            current,
            canonical,
            object_index,
            planner_fingerprint,
            controller_fingerprint,
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
        parent_model_fingerprint_sha256=_bound_model_fingerprint(
            planner_fingerprint,
            aggregate_rl_fingerprint,
            task=selected_task,
            root_height_m=root_height_m,
        ),
        parent_controller_fingerprint_sha256=controller_fingerprint,
        platform_adapter_fingerprint_sha256=(
            PLATFORM_ADAPTER_FINGERPRINT_SHA256
        ),
        planner_model_fingerprint_sha256=planner_fingerprint,
        rl_model_fingerprint_sha256=aggregate_rl_fingerprint,
        shards=shards,
    )


__all__ = [
    "EXPECTED_ACTUATORS",
    "EXPECTED_SHARD_NQ",
    "EXPECTED_SHARD_NV",
    "MODEL_XMLS_BY_TASK",
    "PACKAGE_DIR",
    "PARENT_MODEL_XML",
    "PLATFORM",
    "PLATFORM_ADAPTER_FINGERPRINT_SHA256",
    "PLATFORM_ADAPTER_PATH",
    "PROJECT_ROOT",
    "TASK_GRIPPER_XMLS",
    "TASK_ROOT_HEIGHT_M",
    "PlatformSpec",
    "ShardedPlatformSpec",
    "build_platform_spec",
    "model_xmls_for_task",
]
