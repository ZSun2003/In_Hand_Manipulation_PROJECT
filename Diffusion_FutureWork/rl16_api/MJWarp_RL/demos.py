"""Hash-bound certified IK-expert trajectories for action-chunk Diffusion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from . import contract
from .env import OBSERVATION_DIM


DEMO_FORMAT = "grab-mjwarp-certified-ik-expert-demo-v6"
SOURCE_REGION_NAMES = {0: "all"}
DATA_FILES = {
    "observations": "observations.npy",
    "actions": "actions.npy",
    "returns": "returns.npy",
    "success": "success.npy",
    "object_index": "object_index.npy",
    "task_index": "task_index.npy",
    "curriculum_phase": "curriculum_phase.npy",
    "template_index": "template_index.npy",
    "source_region": "source_region.npy",
    "path_seed": "path_seed.npy",
    "path_shape_index": "path_shape_index.npy",
    "path_tracking_rmse_m": "path_tracking_rmse_m.npy",
    "path_tracking_max_error_m": "path_tracking_max_error_m.npy",
    "rotation_angle_deg": "rotation_angle_deg.npy",
    "position_error_m": "position_error_m.npy",
    "euler_error_deg": "euler_error_deg.npy",
    "position_error_xyz_m": "position_error_xyz_m.npy",
    "euler_error_xyz_deg": "euler_error_xyz_deg.npy",
    "plane_compliance": "plane_compliance.npy",
    "object_dropped": "object_dropped.npy",
    "path_frozen": "path_frozen.npy",
    "plane_action_frozen": "plane_action_frozen.npy",
    "anchors_frozen": "anchors_frozen.npy",
    "trajectory_certified": "trajectory_certified.npy",
    "frozen_path_waypoints_xy_m": "frozen_path_waypoints_xy_m.npy",
    "frozen_episode_action": "frozen_episode_action.npy",
    "frozen_tip_height_fraction": "frozen_tip_height_fraction.npy",
    "frozen_grasp_plane_fraction": "frozen_grasp_plane_fraction.npy",
    "frozen_grasp_plane_yaw_rad": "frozen_grasp_plane_yaw_rad.npy",
    "frozen_hard_tip_anchors_object_local_m": (
        "frozen_hard_tip_anchors_object_local_m.npy"
    ),
}


def _atomic_json(payload: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_native(value: Any) -> Any:
    """Return the exact JSON value persisted by the provenance contract."""

    return json.loads(
        json.dumps(value, sort_keys=True, allow_nan=False)
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_vector(
    values: Any,
    *,
    name: str,
    count: int,
    dtype: Any,
    integer: bool = False,
) -> np.ndarray:
    """Coerce one per-episode vector without silently truncating indices."""

    raw = np.asarray(values)
    if raw.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if integer:
        if np.issubdtype(raw.dtype, np.bool_):
            raise ValueError(f"{name} must contain integers, not booleans")
        try:
            numeric = raw.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain integers") from exc
        if not np.isfinite(numeric).all() or not np.equal(
            numeric, np.floor(numeric)
        ).all():
            raise ValueError(f"{name} must contain finite integers")
        limits = np.iinfo(dtype)
        if bool(((numeric < limits.min) | (numeric > limits.max)).any()):
            raise ValueError(f"{name} cannot be represented as {np.dtype(dtype)}")
    try:
        return raw.astype(dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be stored as {np.dtype(dtype)}") from exc


def _episode_array(
    values: Any,
    *,
    name: str,
    count: int,
    trailing_shape: tuple[int, ...],
    dtype: Any,
) -> np.ndarray:
    """Coerce one fixed-shape per-episode artifact without broadcasting."""

    try:
        result = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be stored as {np.dtype(dtype)}") from exc
    expected = (count, *trailing_shape)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    return result


def _require_closed_interval(
    values: np.ndarray,
    *,
    name: str,
    low: int,
    high: int,
) -> None:
    if bool(((values < low) | (values > high)).any()):
        raise ValueError(f"{name} must be in the closed interval [{low}, {high}]")


class CertifiedIKDemoStore:
    """Append-only certified expert store with an atomic completion cursor."""

    def __init__(
        self,
        root: str | Path,
        *,
        capacity: int,
        steps: int,
        create: bool,
        provenance: Mapping[str, Any],
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root / "metadata.json"
        normalized_provenance = _json_native(dict(provenance))
        if create and not self.metadata_path.exists():
            metadata = {
                "format": DEMO_FORMAT,
                "algorithm_signature": contract.ALGORITHM_SIGNATURE,
                "action_dim": contract.ACTION_DIM,
                "observation_dim": OBSERVATION_DIM,
                "capacity": int(capacity),
                "steps": int(steps),
                "completed": 0,
                "successful": 0,
                "storage_dtype": "float16",
                "source_region_names": {
                    str(index): name
                    for index, name in SOURCE_REGION_NAMES.items()
                },
                "provenance": normalized_provenance,
            }
            _atomic_json(metadata, self.metadata_path)
            self._create_arrays(int(capacity), int(steps))
        if not self.metadata_path.is_file():
            raise FileNotFoundError(self.metadata_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._validate(capacity, steps, normalized_provenance)
        self._open_arrays(mode="r+")

    def _create_arrays(self, capacity: int, steps: int) -> None:
        shapes = {
            "observations": (capacity, steps + 1, OBSERVATION_DIM),
            "actions": (capacity, steps, contract.ACTION_DIM),
            "returns": (capacity,),
            "success": (capacity,),
            "object_index": (capacity,),
            "task_index": (capacity,),
            "curriculum_phase": (capacity,),
            "template_index": (capacity,),
            "source_region": (capacity,),
            "path_seed": (capacity,),
            "path_shape_index": (capacity,),
            "path_tracking_rmse_m": (capacity,),
            "path_tracking_max_error_m": (capacity,),
            "rotation_angle_deg": (capacity,),
            "position_error_m": (capacity,),
            "euler_error_deg": (capacity,),
            "position_error_xyz_m": (capacity, 3),
            "euler_error_xyz_deg": (capacity, 3),
            "plane_compliance": (capacity,),
            "object_dropped": (capacity,),
            "path_frozen": (capacity,),
            "plane_action_frozen": (capacity,),
            "anchors_frozen": (capacity,),
            "trajectory_certified": (capacity,),
            "frozen_path_waypoints_xy_m": (
                capacity,
                contract.TRANSLATION_PATH_WAYPOINTS,
                2,
            ),
            "frozen_episode_action": (capacity, contract.ACTION_DIM),
            "frozen_tip_height_fraction": (
                capacity,
                len(contract.TIP_ORDER),
            ),
            "frozen_grasp_plane_fraction": (capacity,),
            "frozen_grasp_plane_yaw_rad": (capacity,),
            "frozen_hard_tip_anchors_object_local_m": (
                capacity,
                len(contract.TIP_ORDER),
                3,
            ),
        }
        dtypes = {
            "observations": np.float16,
            "actions": np.float16,
            "returns": np.float32,
            "success": np.bool_,
            "object_index": np.int16,
            "task_index": np.int16,
            "curriculum_phase": np.int8,
            "template_index": np.int16,
            "source_region": np.int8,
            "path_seed": np.int64,
            "path_shape_index": np.int16,
            "path_tracking_rmse_m": np.float32,
            "path_tracking_max_error_m": np.float32,
            "rotation_angle_deg": np.int16,
            "position_error_m": np.float32,
            "euler_error_deg": np.float32,
            "position_error_xyz_m": np.float32,
            "euler_error_xyz_deg": np.float32,
            "plane_compliance": np.bool_,
            "object_dropped": np.bool_,
            "path_frozen": np.bool_,
            "plane_action_frozen": np.bool_,
            "anchors_frozen": np.bool_,
            "trajectory_certified": np.bool_,
            "frozen_path_waypoints_xy_m": np.float32,
            "frozen_episode_action": np.float32,
            "frozen_tip_height_fraction": np.float32,
            "frozen_grasp_plane_fraction": np.float32,
            "frozen_grasp_plane_yaw_rad": np.float32,
            "frozen_hard_tip_anchors_object_local_m": np.float32,
        }
        for key, filename in DATA_FILES.items():
            np.lib.format.open_memmap(
                self.root / filename,
                mode="w+",
                dtype=dtypes[key],
                shape=shapes[key],
            ).flush()

    def _open_arrays(self, *, mode: str) -> None:
        for key, filename in DATA_FILES.items():
            setattr(self, key, np.load(self.root / filename, mmap_mode=mode))

    def _validate(
        self,
        capacity: int,
        steps: int,
        provenance: Mapping[str, Any],
    ) -> None:
        metadata = self.metadata
        if metadata.get("format") != DEMO_FORMAT:
            raise ValueError("unsupported certified IK demonstration format")
        if metadata.get("algorithm_signature") != contract.ALGORITHM_SIGNATURE:
            raise ValueError("demonstration action/algorithm contract mismatch")
        if int(metadata.get("action_dim", -1)) != contract.ACTION_DIM:
            raise ValueError("demonstration action dimension mismatch")
        if int(metadata.get("observation_dim", -1)) != OBSERVATION_DIM:
            raise ValueError("demonstration observation dimension mismatch")
        if int(metadata.get("capacity", -1)) != int(capacity):
            raise ValueError("demonstration capacity mismatch")
        if int(metadata.get("steps", -1)) != int(steps):
            raise ValueError("demonstration episode length mismatch")
        if metadata.get("source_region_names") != {
            str(index): name
            for index, name in SOURCE_REGION_NAMES.items()
        }:
            raise ValueError("demonstration source-region encoding mismatch")
        if metadata.get("provenance") != _json_native(dict(provenance)):
            raise ValueError("demonstration provenance mismatch")

    @property
    def completed(self) -> int:
        return int(self.metadata["completed"])

    @property
    def capacity(self) -> int:
        return int(self.metadata["capacity"])

    def append(
        self,
        observations: Any,
        actions: Any,
        *,
        returns: Any,
        success: Any,
        object_index: Any,
        task_index: Any,
        curriculum_phase: Any,
        template_index: Any,
        source_region: Any,
        path_seed: Any,
        path_shape_index: Any,
        path_tracking_rmse_m: Any,
        path_tracking_max_error_m: Any,
        rotation_angle_deg: Any,
        position_error_m: Any,
        euler_error_deg: Any,
        position_error_xyz_m: Any,
        euler_error_xyz_deg: Any,
        plane_compliance: Any,
        object_dropped: Any,
        path_frozen: Any,
        plane_action_frozen: Any,
        anchors_frozen: Any,
        trajectory_certified: Any,
        frozen_path_waypoints_xy_m: Any,
        frozen_episode_action: Any,
        frozen_tip_height_fraction: Any,
        frozen_grasp_plane_fraction: Any,
        frozen_grasp_plane_yaw_rad: Any,
        frozen_hard_tip_anchors_object_local_m: Any,
    ) -> tuple[int, int]:
        observations = np.asarray(observations, dtype=np.float16)
        actions = np.asarray(actions, dtype=np.float16)
        count = int(observations.shape[0])
        expected_observation = (
            count,
            int(self.metadata["steps"]) + 1,
            OBSERVATION_DIM,
        )
        expected_action = (
            count,
            int(self.metadata["steps"]),
            contract.ACTION_DIM,
        )
        if observations.shape != expected_observation:
            raise ValueError(
                f"observation batch must be {expected_observation}, "
                f"got {observations.shape}"
            )
        if actions.shape != expected_action:
            raise ValueError(
                f"action batch must be {expected_action}, got {actions.shape}"
            )
        start = self.completed
        stop = start + count
        if stop > self.capacity:
            raise ValueError("certified demonstration store capacity exceeded")
        fields = {
            "returns": _episode_vector(
                returns,
                name="returns",
                count=count,
                dtype=np.float32,
            ),
            "success": _episode_vector(
                success,
                name="success",
                count=count,
                dtype=np.bool_,
            ),
            "object_index": _episode_vector(
                object_index,
                name="object_index",
                count=count,
                dtype=np.int16,
                integer=True,
            ),
            "task_index": _episode_vector(
                task_index,
                name="task_index",
                count=count,
                dtype=np.int16,
                integer=True,
            ),
            "curriculum_phase": _episode_vector(
                curriculum_phase,
                name="curriculum_phase",
                count=count,
                dtype=np.int8,
                integer=True,
            ),
            "template_index": _episode_vector(
                template_index,
                name="template_index",
                count=count,
                dtype=np.int16,
                integer=True,
            ),
            "source_region": _episode_vector(
                source_region,
                name="source_region",
                count=count,
                dtype=np.int8,
                integer=True,
            ),
            "path_seed": _episode_vector(
                path_seed,
                name="path_seed",
                count=count,
                dtype=np.int64,
                integer=True,
            ),
            "path_shape_index": _episode_vector(
                path_shape_index,
                name="path_shape_index",
                count=count,
                dtype=np.int16,
                integer=True,
            ),
            "path_tracking_rmse_m": _episode_vector(
                path_tracking_rmse_m,
                name="path_tracking_rmse_m",
                count=count,
                dtype=np.float32,
            ),
            "path_tracking_max_error_m": _episode_vector(
                path_tracking_max_error_m,
                name="path_tracking_max_error_m",
                count=count,
                dtype=np.float32,
            ),
            "rotation_angle_deg": _episode_vector(
                rotation_angle_deg,
                name="rotation_angle_deg",
                count=count,
                dtype=np.int16,
                integer=True,
            ),
            "position_error_m": _episode_vector(
                position_error_m,
                name="position_error_m",
                count=count,
                dtype=np.float32,
            ),
            "euler_error_deg": _episode_vector(
                euler_error_deg,
                name="euler_error_deg",
                count=count,
                dtype=np.float32,
            ),
            "position_error_xyz_m": _episode_array(
                position_error_xyz_m,
                name="position_error_xyz_m",
                count=count,
                trailing_shape=(3,),
                dtype=np.float32,
            ),
            "euler_error_xyz_deg": _episode_array(
                euler_error_xyz_deg,
                name="euler_error_xyz_deg",
                count=count,
                trailing_shape=(3,),
                dtype=np.float32,
            ),
            "plane_compliance": _episode_vector(
                plane_compliance,
                name="plane_compliance",
                count=count,
                dtype=np.bool_,
            ),
            "object_dropped": _episode_vector(
                object_dropped,
                name="object_dropped",
                count=count,
                dtype=np.bool_,
            ),
            "path_frozen": _episode_vector(
                path_frozen,
                name="path_frozen",
                count=count,
                dtype=np.bool_,
            ),
            "plane_action_frozen": _episode_vector(
                plane_action_frozen,
                name="plane_action_frozen",
                count=count,
                dtype=np.bool_,
            ),
            "anchors_frozen": _episode_vector(
                anchors_frozen,
                name="anchors_frozen",
                count=count,
                dtype=np.bool_,
            ),
            "trajectory_certified": _episode_vector(
                trajectory_certified,
                name="trajectory_certified",
                count=count,
                dtype=np.bool_,
            ),
            "frozen_path_waypoints_xy_m": _episode_array(
                frozen_path_waypoints_xy_m,
                name="frozen_path_waypoints_xy_m",
                count=count,
                trailing_shape=(contract.TRANSLATION_PATH_WAYPOINTS, 2),
                dtype=np.float32,
            ),
            "frozen_episode_action": _episode_array(
                frozen_episode_action,
                name="frozen_episode_action",
                count=count,
                trailing_shape=(contract.ACTION_DIM,),
                dtype=np.float32,
            ),
            "frozen_tip_height_fraction": _episode_array(
                frozen_tip_height_fraction,
                name="frozen_tip_height_fraction",
                count=count,
                trailing_shape=(len(contract.TIP_ORDER),),
                dtype=np.float32,
            ),
            "frozen_grasp_plane_fraction": _episode_vector(
                frozen_grasp_plane_fraction,
                name="frozen_grasp_plane_fraction",
                count=count,
                dtype=np.float32,
            ),
            "frozen_grasp_plane_yaw_rad": _episode_vector(
                frozen_grasp_plane_yaw_rad,
                name="frozen_grasp_plane_yaw_rad",
                count=count,
                dtype=np.float32,
            ),
            "frozen_hard_tip_anchors_object_local_m": _episode_array(
                frozen_hard_tip_anchors_object_local_m,
                name="frozen_hard_tip_anchors_object_local_m",
                count=count,
                trailing_shape=(len(contract.TIP_ORDER), 3),
                dtype=np.float32,
            ),
        }
        scalar_fields = (
            "returns",
            "success",
            "object_index",
            "task_index",
            "curriculum_phase",
            "template_index",
            "source_region",
            "path_seed",
            "path_shape_index",
            "path_tracking_rmse_m",
            "path_tracking_max_error_m",
            "rotation_angle_deg",
            "position_error_m",
            "euler_error_deg",
            "plane_compliance",
            "object_dropped",
            "path_frozen",
            "plane_action_frozen",
            "anchors_frozen",
            "trajectory_certified",
            "frozen_grasp_plane_fraction",
            "frozen_grasp_plane_yaw_rad",
        )
        for name in scalar_fields:
            values = fields[name]
            if values.shape != (count,):
                raise ValueError(f"{name} must have shape ({count},)")
        for name in (
            "returns",
            "position_error_m",
            "euler_error_deg",
            "path_tracking_rmse_m",
            "path_tracking_max_error_m",
            "position_error_xyz_m",
            "euler_error_xyz_deg",
            "frozen_path_waypoints_xy_m",
            "frozen_episode_action",
            "frozen_tip_height_fraction",
            "frozen_grasp_plane_fraction",
            "frozen_grasp_plane_yaw_rad",
            "frozen_hard_tip_anchors_object_local_m",
        ):
            if not np.isfinite(fields[name]).all():
                raise ValueError(f"{name} must contain only finite values")
        if not np.isfinite(observations).all():
            raise ValueError("observations must contain only finite values")
        if not np.isfinite(actions).all():
            raise ValueError("actions must contain only finite values")
        if bool((np.abs(fields["frozen_episode_action"]) > 1.0 + 1.0e-6).any()):
            raise ValueError("frozen_episode_action must remain in [-1, 1]")
        if bool(
            (
                (fields["frozen_tip_height_fraction"] < 0.0)
                | (fields["frozen_tip_height_fraction"] > 1.0)
            ).any()
        ):
            raise ValueError("frozen tip-height fractions must be in [0, 1]")
        if bool(
            (
                (fields["frozen_grasp_plane_fraction"] < 0.0)
                | (fields["frozen_grasp_plane_fraction"] > 1.0)
            ).any()
        ):
            raise ValueError("frozen grasp-plane fractions must be in [0, 1]")
        if bool(
            (
                np.abs(fields["frozen_grasp_plane_yaw_rad"])
                > contract.FROZEN_PLANNER_YAW_ABS_LIMIT_RAD + 1.0e-6
            ).any()
        ):
            raise ValueError(
                "frozen planner grasp-plane yaw is outside the principal-angle "
                "contract"
            )
        if not bool(fields["success"].all()):
            raise ValueError("only successful trajectories may be certified")
        if not bool(fields["plane_compliance"].all()):
            raise ValueError("only plane-compliant trajectories may be certified")
        if bool(fields["object_dropped"].any()):
            raise ValueError("dropped-object trajectories cannot be certified")
        for name in (
            "path_frozen",
            "plane_action_frozen",
            "anchors_frozen",
            "trajectory_certified",
        ):
            if not bool(fields[name].all()):
                raise ValueError(f"all appended trajectories require {name}=true")
        _require_closed_interval(
            fields["object_index"],
            name="object_index",
            low=0,
            high=len(contract.OBJECTS) - 1,
        )
        _require_closed_interval(
            fields["task_index"],
            name="task_index",
            low=0,
            high=len(contract.TASKS) - 1,
        )
        trained_task = self.metadata.get("provenance", {}).get("trained_task")
        if trained_task in contract.TASKS:
            expected_task_index = contract.TASKS.index(trained_task)
            if bool((fields["task_index"] != expected_task_index).any()):
                raise ValueError(
                    "certified dataset cannot mix tasks with trained_task"
                )
        _require_closed_interval(
            fields["curriculum_phase"],
            name="curriculum_phase",
            low=0,
            high=0,
        )
        _require_closed_interval(
            fields["template_index"],
            name="template_index",
            low=0,
            high=contract.RESET_TEMPLATES_PER_OBJECT - 1,
        )
        _require_closed_interval(
            fields["source_region"],
            name="source_region",
            low=min(SOURCE_REGION_NAMES),
            high=max(SOURCE_REGION_NAMES),
        )
        full = fields["curriculum_phase"] == 0
        if bool((fields["source_region"][full] != 0).any()):
            raise ValueError("all episodes must use source_region=0")
        translating = fields["task_index"] == contract.TASKS.index(
            contract.TASK_TRANSLATE
        )
        rotating = ~translating
        normalized_height = fields[
            "frozen_episode_action"
        ][:, contract.TIP_HEIGHT_SLICE]
        expected_tip_height_fraction = (
            contract.GRASP_PLANE_NOMINAL_FRACTION
            + normalized_height
            * np.where(
                normalized_height <= 0.0,
                contract.GRASP_PLANE_NOMINAL_FRACTION
                - contract.GRASP_PLANE_MIN_FRACTION,
                contract.GRASP_PLANE_MAX_FRACTION
                - contract.GRASP_PLANE_NOMINAL_FRACTION,
            )
        )
        if not np.allclose(
            fields["frozen_tip_height_fraction"],
            expected_tip_height_fraction,
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError(
                "frozen tip-height fractions disagree with episode action"
            )
        if not np.allclose(
            fields["frozen_episode_action"][:, contract.YAW_INDEX],
            0.0,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                "episode yaw must remain zero because planner anchors embed yaw"
            )
        if not np.allclose(
            fields["frozen_grasp_plane_fraction"],
            fields["frozen_tip_height_fraction"].mean(axis=1),
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError(
                "frozen grasp-plane fraction disagrees with hard-tip fractions"
            )
        if bool((fields["path_seed"][rotating] != 0).any()):
            raise ValueError(
                "rotation episodes must store path_seed=0"
            )
        if bool((fields["path_shape_index"][rotating] != 0).any()):
            raise ValueError("rotation episodes must store path_shape_index=0")
        if bool(
            (
                (fields["path_tracking_rmse_m"][rotating] != 0.0)
                | (fields["path_tracking_max_error_m"][rotating] != 0.0)
            ).any()
        ):
            raise ValueError(
                "rotation episodes must store zero path-tracking metrics"
            )
        if bool(
            (
                (fields["path_seed"][translating]
                 < contract.TRANSLATION_PATH_SEED_MIN)
                | (fields["path_seed"][translating]
                   > contract.TRANSLATION_PATH_SEED_MAX)
            ).any()
        ):
            raise ValueError("translation path_seed is outside the contract")
        if bool(
            (
                (fields["path_shape_index"][translating] < 0)
                | (
                    fields["path_shape_index"][translating]
                    >= len(contract.TRANSLATION_PATH_SHAPES)
                )
            ).any()
        ):
            raise ValueError("translation path_shape_index is outside the contract")
        if bool(
            (
                (fields["path_tracking_rmse_m"] < 0.0)
                | (fields["path_tracking_max_error_m"] < 0.0)
                | (
                    fields["path_tracking_rmse_m"]
                    > fields["path_tracking_max_error_m"] + 1.0e-6
                )
            ).any()
        ):
            raise ValueError("translation path tracking metrics are inconsistent")
        if bool((fields["rotation_angle_deg"][translating] != 0).any()):
            raise ValueError(
                "translation episodes must store rotation_angle_deg=0"
            )
        rotating_paths = fields["frozen_path_waypoints_xy_m"][rotating]
        if rotating_paths.size and bool((np.abs(rotating_paths) > 1.0e-7).any()):
            raise ValueError("rotation episodes must store zero frozen path")
        translation_waypoints = fields["frozen_path_waypoints_xy_m"][translating]
        if translation_waypoints.size:
            waypoint_radius = np.linalg.norm(translation_waypoints, axis=-1)
            if bool(
                (
                    waypoint_radius
                    > contract.TRANSLATION_PATH_GENERATION_RADIUS_M + 1.0e-5
                ).any()
            ):
                raise ValueError("translation frozen path leaves generation region")
        full_rotating = full & rotating
        if bool(
            (
                (fields["rotation_angle_deg"][full_rotating]
                 < contract.ROTATION_ANGLE_MIN_DEG)
                | (fields["rotation_angle_deg"][full_rotating]
                   > contract.ROTATION_ANGLE_MAX_DEG)
            ).any()
        ):
            raise ValueError("rotation_angle_deg is outside the full contract")
        self.observations[start:stop] = observations
        self.actions[start:stop] = actions
        for name, values in fields.items():
            getattr(self, name)[start:stop] = values
        for key in DATA_FILES:
            getattr(self, key).flush()
        self.metadata["completed"] = stop
        self.metadata["successful"] = int(
            self.metadata.get("successful", 0)
        ) + int(fields["success"].sum())
        self.metadata.pop("content_sha256", None)
        self.metadata.pop("distribution", None)
        self.metadata.pop("coverage_distribution", None)
        _atomic_json(self.metadata, self.metadata_path)
        return start, stop

    def distribution(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in range(self.completed):
            key = (
                "all/"
                f"{contract.OBJECTS[int(self.object_index[row])]}/"
                f"{contract.TASKS[int(self.task_index[row])]}"
            )
            result[key] = result.get(key, 0) + 1
        return result

    def coverage_distribution(self) -> dict[str, Any]:
        """Return complete, zero-filled audit counts for certified coverage.

        ``distribution()`` remains the non-empty object/task map used by
        training.  This structured companion additionally exposes object and
        task counts,
        object-scoped reset-template counts, seeded translation-path shape
        coverage and tracking errors, and maximum-angle rotation coverage.
        """

        completed = self.completed
        phase = np.asarray(self.curriculum_phase[:completed], dtype=np.int64)
        objects = np.asarray(self.object_index[:completed], dtype=np.int64)
        tasks = np.asarray(self.task_index[:completed], dtype=np.int64)
        successful = np.asarray(self.success[:completed], dtype=np.bool_)
        templates = np.asarray(self.template_index[:completed], dtype=np.int64)
        regions = np.asarray(self.source_region[:completed], dtype=np.int64)
        run_counts = {"all": int((phase == 0).sum())}
        object_counts = {
            name: int((objects == index).sum())
            for index, name in enumerate(contract.OBJECTS)
        }
        task_counts = {
            name: int((tasks == index).sum())
            for index, name in enumerate(contract.TASKS)
        }
        template_counts = {
            object_name: {
                str(template): int(
                    (
                        (objects == object_index)
                        & (templates == template)
                    ).sum()
                )
                for template in range(contract.RESET_TEMPLATES_PER_OBJECT)
            }
            for object_index, object_name in enumerate(contract.OBJECTS)
        }
        full = phase == 0
        source_region_counts = {
            SOURCE_REGION_NAMES[0]: int((full & (regions == 0)).sum())
        }
        source_region_by_object_task = {
            object_name: {
                task_name: {
                    SOURCE_REGION_NAMES[0]: int(
                        (
                            full
                            & (objects == object_index)
                            & (tasks == task_index)
                            & (regions == 0)
                        ).sum()
                    )
                }
                for task_index, task_name in enumerate(contract.TASKS)
            }
            for object_index, object_name in enumerate(contract.OBJECTS)
        }
        translating = tasks == contract.TASKS.index(contract.TASK_TRANSLATE)
        path_seed = np.asarray(self.path_seed[:completed], dtype=np.int64)
        path_shape = np.asarray(
            self.path_shape_index[:completed], dtype=np.int64
        )
        path_rmse = np.asarray(
            self.path_tracking_rmse_m[:completed], dtype=np.float64
        )
        path_max = np.asarray(
            self.path_tracking_max_error_m[:completed], dtype=np.float64
        )
        rotation_angle = np.asarray(
            self.rotation_angle_deg[:completed], dtype=np.int64
        )

        def histogram(values: np.ndarray) -> dict[str, int]:
            unique, counts = np.unique(values, return_counts=True)
            return {
                str(int(value)): int(count)
                for value, count in zip(unique, counts)
            }

        def path_geometry(selected: np.ndarray) -> dict[str, Any]:
            selected_shape = path_shape[selected]
            selected_success = successful[selected]
            selected_rmse = path_rmse[selected]
            selected_max = path_max[selected]
            return {
                "count": int(selected.sum()),
                "successful_count": int(selected_success.sum()),
                "seed_unique_count": int(np.unique(path_seed[selected]).size),
                "shape_counts": {
                    name: int((selected_shape == index).sum())
                    for index, name in enumerate(
                        contract.TRANSLATION_PATH_SHAPES
                    )
                },
                "shape_success_counts": {
                    name: int(
                        ((selected_shape == index) & selected_success).sum()
                    )
                    for index, name in enumerate(
                        contract.TRANSLATION_PATH_SHAPES
                    )
                },
                "tracking_rmse_mean_m": (
                    float(selected_rmse.mean()) if selected_rmse.size else None
                ),
                "tracking_rmse_max_m": (
                    float(selected_rmse.max()) if selected_rmse.size else None
                ),
                "tracking_max_error_m": (
                    float(selected_max.max()) if selected_max.size else None
                ),
            }

        translation_path_by_object = {
            object_name: path_geometry(
                translating & full & (objects == object_index)
            )
            for object_index, object_name in enumerate(contract.OBJECTS)
        }
        rotating = tasks == contract.TASKS.index(contract.TASK_ROTATE)

        def rotation_geometry(selected: np.ndarray, maximum: int) -> dict[str, Any]:
            selected_angles = rotation_angle[selected]
            selected_success = successful[selected]
            return {
                "count": int(selected.sum()),
                "successful_count": int(selected_success.sum()),
                "angle_min_deg": (
                    int(selected_angles.min()) if selected_angles.size else None
                ),
                "angle_max_deg": (
                    int(selected_angles.max()) if selected_angles.size else None
                ),
                "angle_unique_count": int(np.unique(selected_angles).size),
                "angle_histogram": histogram(selected_angles),
                "maximum_target_deg": int(maximum),
                "maximum_target_count": int(
                    (selected_angles == int(maximum)).sum()
                ),
                "maximum_target_success_count": int(
                    (
                        (selected_angles == int(maximum))
                        & selected_success
                    ).sum()
                ),
            }

        full_rotation_geometry = rotation_geometry(
            rotating & full, contract.ROTATION_ANGLE_MAX_DEG
        )
        rotation_angle_by_object = {
            object_name: rotation_geometry(
                rotating & full & (objects == object_index),
                contract.ROTATION_ANGLE_MAX_DEG,
            )
            for object_index, object_name in enumerate(contract.OBJECTS)
        }
        return {
            "run_counts": run_counts,
            "object_counts": object_counts,
            "task_counts": task_counts,
            "template_counts": template_counts,
            "source_region_counts": source_region_counts,
            "source_region_by_object_task": (
                source_region_by_object_task
            ),
            "translation_path_by_object": translation_path_by_object,
            "rotation_angle": full_rotation_geometry,
            "rotation_angle_by_object": rotation_angle_by_object,
            "object_task_counts": self.distribution(),
        }

    def finalize(self) -> dict[str, str]:
        if self.completed <= 0:
            raise ValueError("cannot finalize an empty demonstration store")
        hashes = {
            filename: sha256_file(self.root / filename)
            for filename in DATA_FILES.values()
        }
        self.metadata["content_sha256"] = hashes
        self.metadata["distribution"] = self.distribution()
        self.metadata["coverage_distribution"] = self.coverage_distribution()
        _atomic_json(self.metadata, self.metadata_path)
        return hashes


class DemoWindowSampler:
    """History/action-chunk windows from a finalized mmap dataset.

    ``sample()`` preserves uniform episode sampling.  Use
    ``sample_stratified()`` to cycle evenly through every observed
    ``(object, task)`` stratum; episodes within a stratum and valid
    chunk start times are then sampled uniformly with replacement.
    """

    def __init__(self, root: str | Path, *, seed: int) -> None:
        self.root = Path(root).expanduser().resolve()
        self.metadata = json.loads(
            (self.root / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("format") != DEMO_FORMAT:
            raise ValueError("not a GRAB certified IK demonstration store")
        if self.metadata.get("algorithm_signature") != contract.ALGORITHM_SIGNATURE:
            raise ValueError("dataset contract mismatch")
        if self.metadata.get("source_region_names") != {
            str(index): name
            for index, name in SOURCE_REGION_NAMES.items()
        }:
            raise ValueError("dataset source-region encoding mismatch")
        self.episodes = int(self.metadata["completed"])
        self.steps = int(self.metadata["steps"])
        if self.episodes <= 0:
            raise ValueError("dataset contains no certified trajectories")
        expected = self.metadata.get("content_sha256")
        if not isinstance(expected, dict):
            raise ValueError("dataset has not been finalized")
        actual = {
            filename: sha256_file(self.root / filename)
            for filename in DATA_FILES.values()
        }
        if actual != expected:
            raise ValueError("dataset content checksum mismatch")
        self.observations = np.load(
            self.root / DATA_FILES["observations"], mmap_mode="r"
        )
        self.actions = np.load(
            self.root / DATA_FILES["actions"], mmap_mode="r"
        )
        self.curriculum_phase = np.load(
            self.root / DATA_FILES["curriculum_phase"], mmap_mode="r"
        )
        self.object_index = np.load(
            self.root / DATA_FILES["object_index"], mmap_mode="r"
        )
        self.task_index = np.load(
            self.root / DATA_FILES["task_index"], mmap_mode="r"
        )
        self.rng = np.random.default_rng(int(seed))
        labels = np.stack(
            (
                np.asarray(self.curriculum_phase[: self.episodes]),
                np.asarray(self.object_index[: self.episodes]),
                np.asarray(self.task_index[: self.episodes]),
            ),
            axis=1,
        )
        self.strata = tuple(
            tuple(int(value) for value in row)
            for row in np.unique(labels, axis=0)
        )
        self._stratum_episodes = tuple(
            np.flatnonzero((labels == np.asarray(key)).all(axis=1))
            for key in self.strata
        )
        self._stratum_order = self.rng.permutation(len(self.strata))
        self._stratum_cursor = 0
        self.device_observations = None
        self.device_actions = None

    def normalization(self, *, chunk_episodes: int = 8) -> dict[str, np.ndarray]:
        obs_sum = np.zeros(OBSERVATION_DIM, dtype=np.float64)
        obs_square = np.zeros(OBSERVATION_DIM, dtype=np.float64)
        action_sum = np.zeros(contract.ACTION_DIM, dtype=np.float64)
        action_square = np.zeros(contract.ACTION_DIM, dtype=np.float64)
        obs_count = 0
        action_count = 0
        for start in range(0, self.episodes, chunk_episodes):
            stop = min(self.episodes, start + chunk_episodes)
            observation = np.asarray(
                self.observations[start:stop], dtype=np.float32
            ).reshape(-1, OBSERVATION_DIM)
            action = np.asarray(
                self.actions[start:stop], dtype=np.float32
            ).reshape(-1, contract.ACTION_DIM)
            obs_sum += observation.sum(axis=0, dtype=np.float64)
            obs_square += np.square(observation, dtype=np.float64).sum(axis=0)
            action_sum += action.sum(axis=0, dtype=np.float64)
            action_square += np.square(action, dtype=np.float64).sum(axis=0)
            obs_count += int(observation.shape[0])
            action_count += int(action.shape[0])
        obs_mean = obs_sum / obs_count
        action_mean = action_sum / action_count
        obs_variance = np.maximum(obs_square / obs_count - obs_mean**2, 1e-8)
        action_variance = np.maximum(
            action_square / action_count - action_mean**2, 1e-8
        )
        return {
            "obs_mean": obs_mean.astype(np.float32),
            "obs_std": np.sqrt(obs_variance).astype(np.float32),
            "action_mean": action_mean.astype(np.float32),
            "action_std": np.sqrt(action_variance).astype(np.float32),
        }

    def cache_on_device(self, device: str, *, chunk_episodes: int = 32) -> None:
        import torch

        selected = torch.device(device)
        self.device_observations = torch.empty(
            (self.episodes, self.steps + 1, OBSERVATION_DIM),
            device=selected,
            dtype=torch.float16,
        )
        self.device_actions = torch.empty(
            (self.episodes, self.steps, contract.ACTION_DIM),
            device=selected,
            dtype=torch.float16,
        )
        for start in range(0, self.episodes, chunk_episodes):
            stop = min(self.episodes, start + chunk_episodes)
            observation = np.array(
                self.observations[start:stop], dtype=np.float16, copy=True
            )
            action = np.array(
                self.actions[start:stop], dtype=np.float16, copy=True
            )
            self.device_observations[start:stop].copy_(
                torch.from_numpy(observation).to(selected)
            )
            self.device_actions[start:stop].copy_(
                torch.from_numpy(action).to(selected)
            )

    def _balanced_episode_indices(self, batch_size: int) -> np.ndarray:
        """Cycle through observed object/task strata without bias."""

        episode = np.empty(int(batch_size), dtype=np.int64)
        for row in range(int(batch_size)):
            if self._stratum_cursor >= len(self._stratum_order):
                self._stratum_order = self.rng.permutation(len(self.strata))
                self._stratum_cursor = 0
            stratum = int(self._stratum_order[self._stratum_cursor])
            self._stratum_cursor += 1
            candidates = self._stratum_episodes[stratum]
            episode[row] = candidates[
                self.rng.integers(0, len(candidates))
            ]
        return episode

    def _sample_episode_windows(
        self,
        episode: np.ndarray,
        horizon: int,
        *,
        device: str,
    ) -> tuple[Any, Any]:
        import torch

        if horizon <= 0 or horizon > self.steps:
            raise ValueError("action horizon is outside the episode")
        timestep = self.rng.integers(
            0, self.steps - int(horizon) + 1, size=int(episode.size)
        )
        previous = np.maximum(timestep - 1, 0)
        if self.device_observations is not None:
            selected = torch.device(device)
            episode_index = torch.from_numpy(episode).to(selected)
            timestep_index = torch.from_numpy(timestep).to(selected)
            previous_index = torch.from_numpy(previous).to(selected)
            observation = torch.stack(
                (
                    self.device_observations[episode_index, previous_index],
                    self.device_observations[episode_index, timestep_index],
                ),
                dim=1,
            ).to(torch.float32)
            offsets = torch.arange(horizon, device=selected)[None, :]
            action = self.device_actions[
                episode_index[:, None], timestep_index[:, None] + offsets
            ].to(torch.float32)
            return observation, action
        observation = np.stack(
            (
                self.observations[episode, previous],
                self.observations[episode, timestep],
            ),
            axis=1,
        ).astype(np.float32)
        action = np.stack(
            [
                self.actions[episode, timestep + offset]
                for offset in range(horizon)
            ],
            axis=1,
        ).astype(np.float32)
        return (
            torch.from_numpy(observation).to(device, non_blocking=True),
            torch.from_numpy(action).to(device, non_blocking=True),
        )

    def sample(
        self,
        batch_size: int,
        horizon: int,
        *,
        device: str,
    ) -> tuple[Any, Any]:
        """Sample windows with the original uniform-over-episodes behavior."""

        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        episode = self.rng.integers(
            0, self.episodes, size=int(batch_size)
        )
        return self._sample_episode_windows(
            episode,
            horizon,
            device=device,
        )

    def sample_stratified(
        self,
        batch_size: int,
        horizon: int,
        *,
        device: str,
    ) -> tuple[Any, Any]:
        """Sample an object/task-balanced batch of trajectory windows.

        Consecutive samples cycle through a freshly shuffled ordering of all
        observed strata.  This guarantees stratum counts differ by at most
        one across any complete cycle while retaining random episodes and
        chunk start times inside each stratum.
        """

        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        episode = self._balanced_episode_indices(int(batch_size))
        return self._sample_episode_windows(
            episode,
            horizon,
            device=device,
        )


# Compatibility for callers that have not yet migrated their import name.  The
# dataset format and append gates remain the certified IK contract above.
EliteDemoStore = CertifiedIKDemoStore


__all__ = [
    "DATA_FILES",
    "DEMO_FORMAT",
    "DemoWindowSampler",
    "CertifiedIKDemoStore",
    "EliteDemoStore",
    "SOURCE_REGION_NAMES",
    "sha256_file",
]
