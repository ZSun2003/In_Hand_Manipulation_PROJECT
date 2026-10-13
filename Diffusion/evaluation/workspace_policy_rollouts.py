#!/usr/bin/env python3
"""Measure empirical Diffusion reachability over the complete screw workspace.

This is deliberately a policy-rollout audit, not expert-demonstration
generation and not an analytical-IK pre-filter.  One centre point from every
initially screw--socket-contact-free polar bin is injected directly at reset,
step-zero IK is used only to
initialise the simulator, and the actual controller is allowed to terminate
with success, failure, or time-limit truncation.  The default controller is
Diffusion--Safety--IK with a zero 16-D actuator residual.  ``--with-tqc``
enables the paired post-IK residual policy as an optional second experiment.

The checkpoint training region is derived from the selected configuration.
When ``restrict_screw_spawn_to_opening_half_disk`` is disabled, neither
angular half-disc is labelled out of distribution merely because of its
direction; radial extension beyond the configured training radius remains a
separate distribution shift.

``--export-success-demos`` optionally records the executed 14-D task-space
commands and observations to bounded-RAM, time-major disk memory maps, then
compacts formal successes into a variable-length DemoStore corpus.  It is
labelled self-distillation policy-rollout data, never expert demonstrations.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
HPC_ROOT = PROJECT_ROOT / "hpc"
if not HPC_ROOT.is_dir():
    raise RuntimeError(f"hpc source tree is missing: {HPC_ROOT}")
for location in (str(PROJECT_ROOT), str(HPC_ROOT)):
    if location not in sys.path:
        sys.path.insert(0, location)

from evaluation.evaluate_tqc_azimuth import (  # noqa: E402
    PHASE_NAMES,
    _mapping_difference_paths,
    failure_reasons,
)


SCHEMA = "screw-full-workspace-diffusion-reachability-v2"
SEMANTIC_CORRECTION_SCHEMA = (
    "screw-workspace-distribution-metadata-correction-v1"
)
DEFAULT_STEM = "workspace_policy_rollouts"
ALLOWED_DEPLOYMENT_DIFFERENCES = frozenset({"name", "physics.worlds"})
INITIAL_SOCKET_CONTACT_TOLERANCE_M = 1.0e-9
DEFAULT_MAX_LIVE_WORLDS = 7000
BATCH_SEED_STRIDE = 10_000_019
SCREW_COLLISION_GEOM_NAMES = frozenset(
    {"screw_shaft_collision", "screw_head_collision"}
)
SOCKET_COLLISION_GEOM_PREFIX = "socket_wall_"
SOCKET_COLLISION_GEOM_NAMES = frozenset({"socket_bottom_collision"})
HALF_DISC_TRAINING_REGION = (
    "opening-facing half-disc up to the checkpoint training spawn radius"
)
FULL_DISC_TRAINING_REGION = (
    "full workspace disc up to the checkpoint training spawn radius"
)


def trajectory_storage_estimate(
    *,
    worlds: int,
    steps: int,
    obs_dim: int,
    action_dim: int,
    final_episode_cap: int | None = None,
) -> dict[str, int]:
    """Return conservative byte counts for optional successful-demo export."""

    if min(worlds, steps, obs_dim, action_dim) <= 0:
        raise ValueError("trajectory storage dimensions must be positive")
    if final_episode_cap is not None and final_episode_cap <= 0:
        raise ValueError("final episode cap must be positive")
    final_episodes = (
        worlds
        if final_episode_cap is None
        else min(worlds, int(final_episode_cap))
    )
    # Staging is time-major so each vector step is one contiguous write.  The
    # final DemoStore is episode-major.  In the all-success worst case both
    # coexist until the final store has content hashes.
    observation_bytes = (steps + 1) * worlds * obs_dim * 2
    action_bytes = steps * worlds * action_dim * 2
    final_observation_bytes = (steps + 1) * final_episodes * obs_dim * 2
    final_action_bytes = steps * final_episodes * action_dim * 2
    geometry_bytes = final_episodes * (2 + 2 + 1) * 4
    valid_steps_bytes = final_episodes * 4
    staging_bytes = observation_bytes + action_bytes
    worst_final_bytes = (
        final_observation_bytes
        + final_action_bytes
        + geometry_bytes
        + valid_steps_bytes
    )
    return {
        "staging_bytes": int(staging_bytes),
        "final_episode_capacity": int(final_episodes),
        "worst_case_final_dataset_bytes": int(worst_final_bytes),
        "worst_case_peak_bytes_before_reserve": int(
            staging_bytes + worst_final_bytes
        ),
        "final_bytes_per_success_at_step_limit": int(
            (steps + 1) * obs_dim * 2
            + steps * action_dim * 2
            + (2 + 2 + 1) * 4
            + 4
        ),
    }


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"no existing parent for {path}")
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


class TimeMajorTrajectoryStager:
    """Bounded-RAM, float16 disk staging for vector-policy trajectories."""

    def __init__(
        self,
        root: Path,
        *,
        worlds: int,
        steps: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.worlds = int(worlds)
        self.steps = int(steps)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.observation_path = self.root / "observations_time_major.npy"
        self.action_path = self.root / "actions_time_major.npy"
        self.observations = np.lib.format.open_memmap(
            self.observation_path,
            mode="w+",
            dtype=np.float16,
            shape=(self.steps + 1, self.worlds, self.obs_dim),
        )
        self.actions = np.lib.format.open_memmap(
            self.action_path,
            mode="w+",
            dtype=np.float16,
            shape=(self.steps, self.worlds, self.action_dim),
        )

    def write_initial(self, observation: np.ndarray) -> None:
        self.write_initial_batch(0, observation)

    def write_initial_batch(
        self, world_start: int, observation: np.ndarray
    ) -> None:
        value = np.asarray(observation, dtype=np.float16)
        world_start = int(world_start)
        world_stop = world_start + len(value)
        if (
            value.ndim != 2
            or value.shape[1] != self.obs_dim
            or world_start < 0
            or world_stop > self.worlds
        ):
            raise ValueError("initial staged observation shape mismatch")
        self.observations[0, world_start:world_stop] = value

    def write_step(
        self,
        step: int,
        observation: np.ndarray,
        action: np.ndarray,
    ) -> None:
        self.write_step_batch(step, 0, observation, action)

    def write_step_batch(
        self,
        step: int,
        world_start: int,
        observation: np.ndarray,
        action: np.ndarray,
    ) -> None:
        if not 1 <= int(step) <= self.steps:
            raise ValueError("staged trajectory step is outside allocation")
        observation = np.asarray(observation, dtype=np.float16)
        action = np.asarray(action, dtype=np.float16)
        world_start = int(world_start)
        world_stop = world_start + len(observation)
        if (
            observation.ndim != 2
            or observation.shape[1] != self.obs_dim
            or world_start < 0
            or world_stop > self.worlds
        ):
            raise ValueError("staged observation shape mismatch")
        if action.shape != (len(observation), self.action_dim):
            raise ValueError("staged action shape mismatch")
        self.observations[int(step), world_start:world_stop] = observation
        self.actions[int(step) - 1, world_start:world_stop] = action

    def flush(self) -> None:
        if self.observations is not None:
            self.observations.flush()
        if self.actions is not None:
            self.actions.flush()

    def close(self) -> None:
        self.flush()
        self.observations = None
        self.actions = None


def rollout_batch_slices(
    total_worlds: int, max_live_worlds: int
) -> tuple[slice, ...]:
    """Partition one stable global rollout order into bounded live batches."""

    total_worlds = int(total_worlds)
    max_live_worlds = int(max_live_worlds)
    if total_worlds <= 0:
        raise ValueError("total rollout worlds must be positive")
    if max_live_worlds <= 0:
        raise ValueError("maximum live rollout worlds must be positive")
    return tuple(
        slice(start, min(start + max_live_worlds, total_worlds))
        for start in range(0, total_worlds, max_live_worlds)
    )


def release_cuda_batch_memory(torch, device) -> dict[str, object]:
    """Synchronously return Torch and Warp batch allocations to CUDA."""

    gc.collect()
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {"cuda": False}
    cuda_device = torch.device(device)
    # MJWarp owns its large per-world arrays through Warp's CUDA mempool.
    # Torch synchronization/empty_cache alone does not force those deferred
    # frees back to the driver before the next batch constructs another env.
    torch.cuda.synchronize(cuda_device)
    torch.cuda.empty_cache()
    import warp as wp

    wp.set_mempool_release_threshold(str(cuda_device), 0)
    wp.synchronize_device(str(cuda_device))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(cuda_device)
    return {
        "cuda": True,
        "torch_allocated_bytes": int(
            torch.cuda.memory_allocated(cuda_device)
        ),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved(cuda_device)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _load_checkpoint(torch, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a dictionary: {path}")
    return payload


def _checkpoint_config(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    value = payload.get("config")
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} checkpoint has no config mapping")
    result = copy.deepcopy(dict(value))
    curriculum = result.get("curriculum")
    if isinstance(curriculum, Mapping):
        curriculum = dict(curriculum)
        curriculum.setdefault(
            "restrict_screw_spawn_to_opening_half_disk", True
        )
        result["curriculum"] = curriculum
    return result


def _deployment_projection(
    raw: Mapping[str, Any], *, diffusion: bool
) -> dict[str, Any]:
    result = copy.deepcopy(dict(raw))
    curriculum = result.get("curriculum")
    if isinstance(curriculum, Mapping):
        curriculum = dict(curriculum)
        curriculum.setdefault(
            "restrict_screw_spawn_to_opening_half_disk", True
        )
        result["curriculum"] = curriculum
    result.pop("name", None)
    physics = result.get("physics")
    if not isinstance(physics, Mapping):
        raise ValueError("checkpoint config is missing physics")
    physics = dict(physics)
    physics.pop("worlds", None)
    result["physics"] = physics
    # TQC settings embedded in a Diffusion checkpoint were not used to train
    # or execute that Diffusion model and may legitimately predate TQC16.
    if diffusion:
        result.pop("tqc", None)
    return result


def _assert_checkpoint_config(
    selected: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    label: str,
    diffusion: bool,
) -> list[str]:
    raw_differences = _mapping_difference_paths(selected, checkpoint)
    expected = _deployment_projection(selected, diffusion=diffusion)
    actual = _deployment_projection(checkpoint, diffusion=diffusion)
    effective = _mapping_difference_paths(expected, actual)
    if effective:
        raise ValueError(
            f"{label} checkpoint differs from the selected config in "
            f"effective fields: {effective}"
        )
    return raw_differences


def _atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows:
        if list(row) != fields:
            raise ValueError("CSV rows do not share an identical schema")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def training_distribution_semantics(
    selected_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive directional training-distribution semantics from config."""

    curriculum = selected_config.get("curriculum")
    if not isinstance(curriculum, Mapping):
        raise ValueError("selected config is missing curriculum")
    restricted = curriculum.get(
        "restrict_screw_spawn_to_opening_half_disk"
    )
    if not isinstance(restricted, bool):
        raise ValueError(
            "curriculum.restrict_screw_spawn_to_opening_half_disk must "
            "be a JSON boolean"
        )
    radius_m = float(curriculum.get("final_spawn_radius_m"))
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("curriculum.final_spawn_radius_m must be positive")
    return {
        "training_region": (
            HALF_DISC_TRAINING_REGION
            if restricted
            else FULL_DISC_TRAINING_REGION
        ),
        "opposite_half_disc_is_ood": restricted,
        "training_restrict_screw_spawn_to_opening_half_disk": restricted,
        "training_spawn_radius_mm": radius_m * 1000.0,
    }






def build_polar_grid(
    *, angular_bins: int, radial_bins: int, radius_m: float
) -> dict[str, np.ndarray]:
    theta_edges = np.linspace(
        -math.pi, math.pi, angular_bins + 1, dtype=np.float64
    )
    radius_edges = np.linspace(
        0.0, radius_m, radial_bins + 1, dtype=np.float64
    )
    theta = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    radius = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    radius_grid, theta_grid = np.meshgrid(radius, theta, indexing="ij")
    xy = np.stack(
        (
            radius_grid * np.cos(theta_grid),
            radius_grid * np.sin(theta_grid),
        ),
        axis=-1,
    )
    return {
        "theta_edges": theta_edges,
        "radius_edges": radius_edges,
        "theta_grid": theta_grid,
        "radius_grid": radius_grid,
        "xy": xy,
    }


def socket_outer_obb_center_mask(
    spawn_xy: np.ndarray,
    *,
    screw_center_z_m: float,
    socket_center: np.ndarray,
    socket_frame: np.ndarray,
    socket_outer_dimensions_m: Sequence[float],
    tolerance_m: float = INITIAL_SOCKET_CONTACT_TOLERANCE_M,
) -> np.ndarray:
    """Return a diagnostic point-in-OBB mask for upright screw centres.

    The OBB is aligned with the socket's three local axes and has exactly the
    socket's outer width, height, and length.  It is deliberately *not*
    inflated by the screw radius or reset clearance.  Consequently this mask
    documents the tight socket envelope but is not used as the physical
    collision decision: a finite-radius screw can touch a socket face while
    its centre remains outside this box.
    """

    xy = np.asarray(spawn_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("spawn_xy must have shape [count, 2]")
    center = np.asarray(socket_center, dtype=np.float64)
    frame = np.asarray(socket_frame, dtype=np.float64)
    dimensions = np.asarray(socket_outer_dimensions_m, dtype=np.float64)
    if center.shape != (3,) or frame.shape != (3, 3):
        raise ValueError("socket centre/frame must have shapes [3] and [3,3]")
    if dimensions.shape != (3,) or bool(np.any(dimensions <= 0.0)):
        raise ValueError("socket outer dimensions must be three positive values")
    if tolerance_m < 0.0:
        raise ValueError("OBB tolerance must be non-negative")
    points = np.column_stack(
        (xy, np.full(len(xy), float(screw_center_z_m), dtype=np.float64))
    )
    # ``socket_frame`` stores the local axes as columns, hence row-vector
    # world displacements are transformed with ``@ frame``.
    local = (points - center[None, :]) @ frame
    half_extent = 0.5 * dimensions + float(tolerance_m)
    return np.all(np.abs(local) <= half_extent[None, :], axis=1)


def classify_initial_screw_socket_contacts(
    spawn_xy: np.ndarray,
    *,
    screw_center_z_m: float,
    socket_position: np.ndarray,
    socket_quaternion_wxyz: np.ndarray,
    xml_path: Path,
    contact_tolerance_m: float = INITIAL_SOCKET_CONTACT_TOLERANCE_M,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Classify exact initial screw--socket contacts with CPU MuJoCo.

    Each grid centre is evaluated with the screw upright on the table and the
    fixed socket mocap pose used by the rollout.  Only contacts between the two
    physical screw collision geoms and the socket wall/bottom collision geoms
    count.  Table, holder, gripper, and visual-geom contacts are ignored.
    No IK or inflated reset-clearance geometry participates in this test.
    """

    if contact_tolerance_m < 0.0:
        raise ValueError("contact tolerance must be non-negative")
    xy = np.asarray(spawn_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("spawn_xy must have shape [count, 2]")
    socket_position = np.asarray(socket_position, dtype=np.float64)
    socket_quaternion = np.asarray(
        socket_quaternion_wxyz, dtype=np.float64
    )
    if socket_position.shape != (3,) or socket_quaternion.shape != (4,):
        raise ValueError("socket pose must have shapes [3] and [4]")

    import mujoco

    resolved_xml = Path(xml_path).expanduser().resolve()
    if not resolved_xml.is_file():
        raise FileNotFoundError(f"MuJoCo XML is missing: {resolved_xml}")
    model = mujoco.MjModel.from_xml_path(str(resolved_xml))
    data = mujoco.MjData(model)

    def object_id(object_type, name: str) -> int:
        identifier = int(mujoco.mj_name2id(model, object_type, name))
        if identifier < 0:
            raise ValueError(f"MuJoCo object is missing: {name}")
        return identifier

    screw_joint = object_id(mujoco.mjtObj.mjOBJ_JOINT, "screw_free")
    screw_qpos = int(model.jnt_qposadr[screw_joint])
    socket_body = object_id(mujoco.mjtObj.mjOBJ_BODY, "socket")
    socket_mocap = int(model.body_mocapid[socket_body])
    if socket_mocap < 0:
        raise ValueError("socket body is not mocap-controlled")

    geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(model.ngeom)
    ]
    screw_geom_ids = {
        geom_id
        for geom_id, name in enumerate(geom_names)
        if name in SCREW_COLLISION_GEOM_NAMES
    }
    socket_geom_ids = {
        geom_id
        for geom_id, name in enumerate(geom_names)
        if name in SOCKET_COLLISION_GEOM_NAMES
        or (name is not None and name.startswith(SOCKET_COLLISION_GEOM_PREFIX))
    }
    if len(screw_geom_ids) != len(SCREW_COLLISION_GEOM_NAMES):
        raise ValueError("MuJoCo XML does not contain both screw collision geoms")
    if not socket_geom_ids:
        raise ValueError("MuJoCo XML contains no physical socket collision geoms")

    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    # Equality constraints do not define collision geometry.  Disabling the
    # spawn-holder weld avoids solving a large artificial constraint error as
    # the fixed grid point changes, without changing any geom pose/contact.
    if model.neq:
        data.eq_active[:] = 0
    data.mocap_pos[socket_mocap] = socket_position
    data.mocap_quat[socket_mocap] = socket_quaternion
    data.qpos[screw_qpos + 2] = float(screw_center_z_m)
    data.qpos[screw_qpos + 3 : screw_qpos + 7] = (1.0, 0.0, 0.0, 0.0)

    collision = np.zeros(len(xy), dtype=bool)
    contact_pair_counts: dict[str, int] = {}
    minimum_contact_distance_m: float | None = None
    for point_index, (spawn_x, spawn_y) in enumerate(xy):
        data.qpos[screw_qpos : screw_qpos + 2] = (spawn_x, spawn_y)
        mujoco.mj_forward(model, data)
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            screw_socket_pair = (
                geom1 in screw_geom_ids and geom2 in socket_geom_ids
            ) or (geom2 in screw_geom_ids and geom1 in socket_geom_ids)
            if not screw_socket_pair or float(contact.dist) > contact_tolerance_m:
                continue
            collision[point_index] = True
            distance = float(contact.dist)
            minimum_contact_distance_m = (
                distance
                if minimum_contact_distance_m is None
                else min(minimum_contact_distance_m, distance)
            )
            screw_id = geom1 if geom1 in screw_geom_ids else geom2
            socket_id = geom2 if geom2 in socket_geom_ids else geom1
            pair_name = f"{geom_names[screw_id]}::{geom_names[socket_id]}"
            contact_pair_counts[pair_name] = contact_pair_counts.get(pair_name, 0) + 1
            # Classification is per position; one qualifying physical pair is
            # sufficient and avoids counting mesh-segment multiplicity.
            break

    provenance: dict[str, Any] = {
        "method": "cpu_mujoco_mj_forward_filtered_contacts",
        "mujoco_version": str(mujoco.__version__),
        "xml_path": str(resolved_xml),
        "xml_sha256": sha256_file(resolved_xml),
        "contact_tolerance_m": float(contact_tolerance_m),
        "clearance_inflation_m": 0.0,
        "screw_pose": "upright, shaft centre fixed at configured table reset height",
        "socket_pose": "fixed mocap position and wxyz quaternion from supervisor",
        "included_screw_geoms": sorted(
            geom_names[geom_id] for geom_id in screw_geom_ids
        ),
        "included_socket_geoms": sorted(
            geom_names[geom_id] for geom_id in socket_geom_ids
        ),
        "ignored_contact_classes": [
            "table",
            "gripper/fingertips",
            "spawn holder",
            "visual-only geoms",
        ],
        "contacting_grid_centres": int(collision.sum()),
        "minimum_qualifying_contact_distance_m": minimum_contact_distance_m,
        "first_contact_pair_counts": dict(sorted(contact_pair_counts.items())),
    }
    return collision, provenance


def aggregate_cells(
    *,
    grid: Mapping[str, np.ndarray],
    collision: np.ndarray,
    trials: Sequence[Mapping[str, object]],
    trials_per_cell: int,
) -> list[dict[str, object]]:
    xy = np.asarray(grid["xy"]).reshape(-1, 2)
    radius = np.asarray(grid["radius_grid"]).reshape(-1)
    theta = np.asarray(grid["theta_grid"]).reshape(-1)
    collision = np.asarray(collision, dtype=bool).reshape(-1)
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in trials:
        grouped.setdefault(int(row["cell_id"]), []).append(row)
    rows: list[dict[str, object]] = []
    angular_bins = int(np.asarray(grid["theta_grid"]).shape[1])
    for cell_id in range(len(xy)):
        radial_index = cell_id // angular_bins
        angular_index = cell_id % angular_bins
        cell_trials = grouped.get(cell_id, [])
        if collision[cell_id]:
            classification = "socket_excluded"
            successes = failures = truncations = handoffs = 0
            success_rate = handoff_rate = None
            mean_steps = mean_success_steps = None
            reasons = ""
        else:
            if len(cell_trials) != trials_per_cell:
                raise ValueError(
                    f"cell {cell_id} has {len(cell_trials)} trials; "
                    f"expected {trials_per_cell}"
                )
            successes = sum(bool(row["success"]) for row in cell_trials)
            failures = sum(bool(row["failure"]) for row in cell_trials)
            truncations = sum(bool(row["truncated"]) for row in cell_trials)
            handoffs = sum(
                bool(row["diffusion_ever_active"]) for row in cell_trials
            )
            success_rate = successes / trials_per_cell
            handoff_rate = handoffs / trials_per_cell
            mean_steps = float(
                np.mean([int(row["terminal_steps"]) for row in cell_trials])
            )
            successful_steps = [
                int(row["terminal_steps"])
                for row in cell_trials
                if bool(row["success"])
            ]
            mean_success_steps = (
                float(np.mean(successful_steps)) if successful_steps else None
            )
            reason_set = sorted(
                {
                    item
                    for row in cell_trials
                    for item in str(row["failure_reasons"]).split(";")
                    if item
                }
            )
            reasons = ";".join(reason_set)
            if successes:
                classification = "demonstrated_reachable"
            elif handoffs:
                classification = "handoff_reached_no_task_success"
            else:
                classification = "pickup_handoff_not_reached"
        rows.append(
            {
                "cell_id": cell_id,
                "radial_index": radial_index,
                "angular_index": angular_index,
                "spawn_x_m": float(xy[cell_id, 0]),
                "spawn_y_m": float(xy[cell_id, 1]),
                "spawn_radius_mm": float(radius[cell_id] * 1000.0),
                "spawn_azimuth_deg": float(np.degrees(theta[cell_id])),
                "socket_excluded": bool(collision[cell_id]),
                "tested_trials": len(cell_trials),
                "successes": successes,
                "failures": failures,
                "truncations": truncations,
                "handoff_reached_trials": handoffs,
                "success_rate": success_rate,
                "handoff_rate": handoff_rate,
                "mean_terminal_steps": mean_steps,
                "mean_success_steps": mean_success_steps,
                "classification": classification,
                "failure_reasons": reasons,
            }
        )
    return rows




def _select_info(info: Mapping[str, Any], indices):
    selected: dict[str, Any] = {}
    for key, value in info.items():
        if hasattr(value, "shape") and len(value.shape) >= 1:
            selected[key] = value.index_select(0, indices).detach().cpu()
        else:
            selected[key] = value
    return selected


def export_successful_policy_rollouts(
    *,
    stager: TimeMajorTrajectoryStager,
    target: Path,
    trial_rows: Sequence[Mapping[str, object]],
    trial_csv: Path,
    fixed_spawn: np.ndarray,
    fixed_mouths: np.ndarray,
    socket_angle_deg: float,
    policy_start_step: int,
    action_horizon: int,
    shard_episodes: int,
    max_episodes: int,
    selection_seed: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact successful source rollouts into a variable-length DemoStore."""

    from s_hpc.demos import DemoStore

    if shard_episodes <= 0:
        raise ValueError("success-demo shard size must be positive")
    if max_episodes <= 0:
        raise ValueError("success-demo maximum episodes must be positive")
    all_successful_indices = np.asarray(
        [index for index, row in enumerate(trial_rows) if bool(row["success"])],
        dtype=np.int64,
    )
    if all_successful_indices.size == 0:
        # No corpus can be finalized.  The terminal trial CSV is already
        # durable, so the all-failure staging allocation has no retained data.
        stager.close()
        shutil.rmtree(stager.root)
        return {
            "enabled": True,
            "status": "no_successful_rollouts",
            "dataset_role": "successful_policy_rollout_self_distillation",
            "expert_demonstrations": False,
            "successful_episodes": 0,
            "staging_removed": True,
        }
    if all_successful_indices.size > max_episodes:
        selection_rng = np.random.default_rng(int(selection_seed))
        successful_indices = np.sort(
            selection_rng.choice(
                all_successful_indices,
                size=int(max_episodes),
                replace=False,
            )
        )
        selection_method = (
            "fixed-seed uniform sampling without replacement, then source-order sort"
        )
    else:
        successful_indices = all_successful_indices
        selection_method = "all successful source rollouts in source order"

    valid_steps = np.asarray(
        [int(trial_rows[index]["terminal_steps"]) for index in successful_indices],
        dtype=np.int32,
    )
    minimum_trainable_steps = int(policy_start_step) + int(action_horizon)
    if np.any(valid_steps < minimum_trainable_steps):
        raise ValueError(
            "a successful rollout is too short to contain one post-prefix "
            "Diffusion training window"
        )
    dataset_steps = int(valid_steps.max())
    if dataset_steps > stager.steps:
        raise ValueError("successful trajectory exceeds staging allocation")
    if target.exists() and (
        not target.is_dir() or any(target.iterdir())
    ):
        raise FileExistsError(
            f"successful-rollout dataset is not empty: {target}"
        )

    dataset_provenance = {
        **dict(provenance),
        "dataset_role": "successful_policy_rollout_self_distillation",
        "expert_demonstrations": False,
        "selection_rule": "formal task success at first terminal transition",
        "selection_method": selection_method,
        "selection_seed": int(selection_seed),
        "source_successful_rollouts": int(all_successful_indices.size),
        "retained_successful_rollouts": int(successful_indices.size),
        "retained_episode_cap": int(max_episodes),
        "action_semantics": (
            "14D socket-frame task-space command actually selected by the "
            "Diffusion-Safety-IK controller for each recorded transition; "
            "this is StepBatch.expert_action for the executed command, not "
            "an expert-policy relabelling and not actuator control"
        ),
        "variable_episode_lengths": True,
        "action_horizon": int(action_horizon),
        "padding_semantics": (
            "terminal observation repeat and zero action after valid_steps; "
            "DemoWindowSampler excludes every padded window"
        ),
        "source_trial_csv": {
            "path": str(trial_csv),
            "sha256": sha256_file(trial_csv),
        },
        "source_world_order": (
            "dataset episode order is ascending source trial-CSV row/world index"
        ),
        "source_terminal_steps": {
            "minimum": int(valid_steps.min()),
            "maximum": dataset_steps,
            "mean": float(valid_steps.mean()),
        },
    }
    store = DemoStore(
        target,
        episodes=int(successful_indices.size),
        steps=dataset_steps,
        create=True,
        provenance=dataset_provenance,
        variable_episode_lengths=True,
    )
    completed = 0
    for offset in range(0, successful_indices.size, shard_episodes):
        stop = min(successful_indices.size, offset + shard_episodes)
        source_indices = successful_indices[offset:stop]
        lengths = valid_steps[offset:stop]
        observations = np.asarray(
            stager.observations[: dataset_steps + 1, source_indices, :],
            dtype=np.float16,
        ).transpose(1, 0, 2).copy()
        actions = np.asarray(
            stager.actions[:dataset_steps, source_indices, :],
            dtype=np.float16,
        ).transpose(1, 0, 2).copy()
        for local_index, valid_length in enumerate(lengths):
            valid_length = int(valid_length)
            if valid_length < dataset_steps:
                observations[local_index, valid_length + 1 :] = observations[
                    local_index, valid_length
                ]
                actions[local_index, valid_length:] = np.float16(0.0)
        count = stop - offset
        store.write(
            completed,
            observations,
            actions,
            np.asarray(fixed_spawn[source_indices], dtype=np.float32),
            socket_mouth_xy=np.asarray(
                fixed_mouths[source_indices], dtype=np.float32
            ),
            socket_angle_deg=np.full(
                count, float(socket_angle_deg), dtype=np.float32
            ),
            valid_steps=lengths,
            commit=False,
        )
        completed += count
    store.commit(0, completed)
    content_hashes = store.metadata.get("content_sha256")
    if not isinstance(content_hashes, Mapping) or (
        "valid_steps.npy" not in content_hashes
    ):
        raise RuntimeError(
            "successful-rollout DemoStore was not finalized with hashes"
        )
    metadata_path = target / "metadata.json"
    final_bytes = sum(
        path.stat().st_size for path in target.iterdir() if path.is_file()
    )

    # The temporary all-world allocation is removed only after the compact
    # corpus is committed and every retained array is bound into metadata by
    # SHA-256.  An exception before this point intentionally preserves it.
    stager.close()
    shutil.rmtree(stager.root)
    return {
        "enabled": True,
        "status": "complete",
        "dataset_role": "successful_policy_rollout_self_distillation",
        "expert_demonstrations": False,
        "path": str(target),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "successful_episodes": int(successful_indices.size),
        "source_successful_rollouts": int(all_successful_indices.size),
        "retained_episode_cap": int(max_episodes),
        "selection_method": selection_method,
        "dataset_steps": dataset_steps,
        "valid_steps_min": int(valid_steps.min()),
        "valid_steps_max": int(valid_steps.max()),
        "valid_steps_mean": float(valid_steps.mean()),
        "final_dataset_bytes": int(final_bytes),
        "content_sha256": dict(content_hashes),
        "staging_removed": True,
    }


def run_rollouts(args: argparse.Namespace) -> dict[str, Path]:
    if args.angular_bins <= 0 or args.radial_bins <= 0:
        raise ValueError("polar grid dimensions must be positive")
    if args.trials_per_cell <= 0:
        raise ValueError("--trials-per-cell must be positive")
    if args.spawn_radius_mm <= 0.0:
        raise ValueError("--spawn-radius-mm must be positive")
    if args.success_demo_shard_episodes <= 0:
        raise ValueError("--success-demo-shard-episodes must be positive")
    if args.success_demo_max_episodes <= 0:
        raise ValueError("--success-demo-max-episodes must be positive")
    if args.success_demo_flush_every <= 0:
        raise ValueError("--success-demo-flush-every must be positive")
    if args.success_demo_reserve_gib < 0.0:
        raise ValueError("--success-demo-reserve-gib must be non-negative")
    if args.max_live_worlds <= 0:
        raise ValueError("--max-live-worlds must be positive")

    config_path = _resolved_file(args.config, label="config")
    diffusion_path = _resolved_file(
        args.diffusion, label="Diffusion checkpoint"
    )
    tqc_path = None
    if args.tqc is not None:
        tqc_path = _resolved_file(args.tqc, label="TQC checkpoint")
    if args.with_tqc and tqc_path is None:
        raise ValueError("--with-tqc requires --tqc")

    import torch

    from s_hpc.config import load_config
    from s_hpc.contracts import (
        ACTION_DIM,
        OBS_DIM,
        TASK_ASSET_HASHES,
        validate_checkpoint,
    )
    from s_hpc.diffusion_runtime import BatchedDiffusionController
    from s_hpc.env import MJWarpScrewEnv
    from s_hpc.supervisor import BatchedSupervisor
    from s_hpc.tqc import TQCTrainer
    from screw_control import SCRIPTED_PREFIX_STEPS

    selected_config = load_config(config_path)
    selected_payload = selected_config.to_dict()
    diffusion_payload = _load_checkpoint(torch, diffusion_path)
    validate_checkpoint(diffusion_payload, kind="diffusion")
    diffusion_config = _checkpoint_config(
        diffusion_payload, label="Diffusion"
    )
    diffusion_differences = _assert_checkpoint_config(
        selected_payload,
        diffusion_config,
        label="Diffusion",
        diffusion=True,
    )
    diffusion_sha256 = sha256_file(diffusion_path)

    tqc_payload = None
    tqc_sha256 = None
    tqc_differences: list[str] = []
    if tqc_path is not None:
        tqc_payload = _load_checkpoint(torch, tqc_path)
        validate_checkpoint(tqc_payload, kind="tqc")
        tqc_config = _checkpoint_config(tqc_payload, label="TQC")
        tqc_differences = _assert_checkpoint_config(
            selected_payload,
            tqc_config,
            label="TQC",
            diffusion=False,
        )
        tqc_sha256 = sha256_file(tqc_path)
        if tqc_payload.get("diffusion_sha256") != diffusion_sha256:
            raise ValueError(
                "TQC checkpoint does not pin the supplied Diffusion "
                "checkpoint by SHA-256"
            )

    evaluation_seed = (
        int(selected_config.runtime.seed) + 200_000
        if args.seed is None
        else int(args.seed)
    )
    # This explicit override is evaluation-only.  Checkpoint lineage above is
    # validated against the original selected training configuration.
    runtime_config = replace(
        selected_config,
        curriculum=replace(
            selected_config.curriculum,
            enabled=False,
            restrict_screw_spawn_to_opening_half_disk=False,
        ),
        runtime=replace(selected_config.runtime, seed=evaluation_seed),
    )
    max_steps = (
        int(runtime_config.physics.max_episode_steps)
        if args.max_steps is None
        else int(args.max_steps)
    )
    if max_steps <= 0 or max_steps > runtime_config.physics.max_episode_steps:
        raise ValueError(
            "--max-steps must be in [1, configured max_episode_steps]"
        )

    spawn_radius_m = args.spawn_radius_mm / 1000.0
    supervisor_cpu = BatchedSupervisor(
        device="cpu", curriculum=runtime_config.curriculum
    )
    if spawn_radius_m > supervisor_cpu.config.spawn_radius_m + 1.0e-12:
        raise ValueError("requested radius exceeds the configured workspace")
    grid = build_polar_grid(
        angular_bins=args.angular_bins,
        radial_bins=args.radial_bins,
        radius_m=spawn_radius_m,
    )
    mouth_azimuth_rad = math.radians(args.socket_mouth_azimuth_deg)
    mouth_xy = np.asarray(
        [
            supervisor_cpu.config.socket_mouth_orbit_radius_m
            * math.cos(mouth_azimuth_rad),
            supervisor_cpu.config.socket_mouth_orbit_radius_m
            * math.sin(mouth_azimuth_rad),
        ],
        dtype=np.float32,
    )
    flat_xy = np.asarray(grid["xy"], dtype=np.float32).reshape(-1, 2)
    grid_count = len(flat_xy)
    repeated_mouth_xy = np.repeat(mouth_xy[None, :], grid_count, axis=0)
    legacy_conservative_collision = (
        supervisor_cpu.spawn_overlaps_socket(
            torch.from_numpy(flat_xy),
            socket_mouth_xy=torch.from_numpy(repeated_mouth_xy),
            socket_angle_deg=torch.full(
                (grid_count,), float(args.socket_angle_deg)
            ),
        )
        .cpu()
        .numpy()
        .astype(bool)
    )
    pose_angle = torch.full((1,), float(args.socket_angle_deg))
    pose_mouth = torch.from_numpy(mouth_xy[None, :])
    socket_frame_tensor, _, _, _ = supervisor_cpu.socket_geometry(
        pose_angle, pose_mouth
    )
    socket_position_tensor, socket_quaternion_tensor = (
        supervisor_cpu.socket_mocap_pose(pose_angle, pose_mouth)
    )
    socket_frame = socket_frame_tensor[0].detach().cpu().numpy()
    socket_position = socket_position_tensor[0].detach().cpu().numpy()
    socket_quaternion = (
        socket_quaternion_tensor[0].detach().cpu().numpy()
    )
    socket_outer_dimensions_m = (
        supervisor_cpu.config.socket_outer_width_m,
        supervisor_cpu.config.socket_outer_height_m,
        supervisor_cpu.config.socket_outer_length_m,
    )
    socket_obb_center = socket_outer_obb_center_mask(
        flat_xy,
        screw_center_z_m=supervisor_cpu.config.screw_initial_center_z_m,
        socket_center=socket_position,
        socket_frame=socket_frame,
        socket_outer_dimensions_m=socket_outer_dimensions_m,
    )
    collision, collision_provenance = (
        classify_initial_screw_socket_contacts(
            flat_xy,
            screw_center_z_m=supervisor_cpu.config.screw_initial_center_z_m,
            socket_position=socket_position,
            socket_quaternion_wxyz=socket_quaternion,
            xml_path=HPC_ROOT / "assets" / "Project_hpc.xml",
        )
    )
    collision_comparison = {
        "exact_initial_physical_contact_cells": int(collision.sum()),
        "exact_socket_outer_obb_center_cells": int(socket_obb_center.sum()),
        "legacy_inflated_projected_obb_cells": int(
            legacy_conservative_collision.sum()
        ),
        "cells_restored_vs_legacy": int(
            legacy_conservative_collision.sum() - collision.sum()
        ),
        "exact_contacts_outside_uninflated_socket_obb_center_mask": int(
            np.logical_and(collision, ~socket_obb_center).sum()
        ),
    }
    obb_half = 0.5 * np.asarray(socket_outer_dimensions_m, dtype=np.float64)
    obb_local_corners = np.asarray(
        [
            (x, y, z)
            for x in (-obb_half[0], obb_half[0])
            for y in (-obb_half[1], obb_half[1])
            for z in (-obb_half[2], obb_half[2])
        ],
        dtype=np.float64,
    )
    obb_world_corners = (
        obb_local_corners @ socket_frame.T + socket_position[None, :]
    )
    socket_obb_provenance = {
        "role": (
            "exact uninflated outer-envelope diagnostic; final exclusion is "
            "the filtered MuJoCo physical-contact mask"
        ),
        "local_axis_order": ["outer_width", "outer_height", "outer_length"],
        "dimensions_m": [float(value) for value in socket_outer_dimensions_m],
        "center_world_m": [float(value) for value in socket_position],
        "frame_columns_world": socket_frame.astype(float).tolist(),
        "quaternion_wxyz": [float(value) for value in socket_quaternion],
        "mouth_azimuth_deg": float(args.socket_mouth_azimuth_deg),
        "elevation_deg": float(args.socket_angle_deg),
        "clearance_inflation_m": 0.0,
        "screw_radius_inflation_m": 0.0,
        "projected_plan_length_m": float(
            supervisor_cpu.config.socket_outer_length_m
            * math.cos(math.radians(args.socket_angle_deg))
            + supervisor_cpu.config.socket_outer_height_m
            * math.sin(math.radians(args.socket_angle_deg))
        ),
        "projected_plan_width_m": float(
            supervisor_cpu.config.socket_outer_width_m
        ),
        "bottom_height_above_table_m": float(
            obb_world_corners[:, 2].min()
            - supervisor_cpu.config.table_top_z_m
        ),
        "top_height_above_table_m": float(
            obb_world_corners[:, 2].max()
            - supervisor_cpu.config.table_top_z_m
        ),
        "grid_centres_inside": int(socket_obb_center.sum()),
        "finite_screw_warning": (
            "Point-in-OBB is not a contact-free guarantee because the upright "
            "screw has finite 15 mm shaft and 25 mm head radii."
        ),
    }
    candidate_ids = np.flatnonzero(~collision)
    if len(candidate_ids) == 0:
        raise RuntimeError("socket exclusion removed every grid cell")
    world_cell_ids = np.repeat(candidate_ids, args.trials_per_cell)
    world_trial_ids = np.tile(
        np.arange(args.trials_per_cell, dtype=np.int64), len(candidate_ids)
    )
    fixed_spawn = flat_xy[world_cell_ids]
    worlds = len(fixed_spawn)
    fixed_mouths = np.repeat(mouth_xy[None, :], worlds, axis=0)
    rollout_batches = rollout_batch_slices(worlds, args.max_live_worlds)
    live_worlds_peak = max(
        int(batch.stop - batch.start) for batch in rollout_batches
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    success_demo_target = (
        Path(args.success_demo_output).expanduser().resolve()
        if args.success_demo_output is not None
        else output_dir / f"{args.stem}.successful_policy_rollouts"
    )
    storage_estimate = trajectory_storage_estimate(
        worlds=worlds,
        steps=max_steps,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        final_episode_cap=args.success_demo_max_episodes,
    )
    storage_estimate["reserve_bytes"] = int(
        args.success_demo_reserve_gib * (1024**3)
    )
    storage_estimate["required_free_bytes"] = int(
        storage_estimate["worst_case_peak_bytes_before_reserve"]
        + storage_estimate["reserve_bytes"]
    )
    storage_root = _nearest_existing_directory(output_dir)
    storage_estimate["available_bytes"] = int(
        shutil.disk_usage(storage_root).free
    )
    storage_estimate["preflight_pass"] = bool(
        storage_estimate["available_bytes"]
        >= storage_estimate["required_free_bytes"]
    )
    if args.export_success_demos:
        if success_demo_target.exists() and (
            not success_demo_target.is_dir()
            or any(success_demo_target.iterdir())
        ):
            raise FileExistsError(
                "successful-rollout output must be absent or empty: "
                f"{success_demo_target}"
            )
        if not storage_estimate["preflight_pass"]:
            raise OSError(
                "insufficient free disk for bounded-memory successful-rollout "
                f"export: need {storage_estimate['required_free_bytes']} bytes, "
                f"have {storage_estimate['available_bytes']} bytes"
            )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "event": "workspace_reachability_validation",
                    "status": "pass",
                    "grid_cells": grid_count,
                    "socket_excluded_cells": int(collision.sum()),
                    "socket_exclusion_semantics": (
                        "exact initial CPU-MuJoCo screw-vs-socket physical contact"
                    ),
                    "socket_exclusion_comparison": collision_comparison,
                    "socket_outer_obb": socket_obb_provenance,
                    "socket_contact_classifier": collision_provenance,
                    "tested_cells": int(len(candidate_ids)),
                    "trials_per_cell": int(args.trials_per_cell),
                    "rollout_worlds": int(worlds),
                    "rollout_batches": int(len(rollout_batches)),
                    "max_live_worlds": int(args.max_live_worlds),
                    "peak_live_worlds": int(live_worlds_peak),
                    "batch_world_counts": [
                        int(batch.stop - batch.start)
                        for batch in rollout_batches
                    ],
                    "controller": (
                        "diffusion-safety-ik-tqc"
                        if args.with_tqc
                        else "diffusion-safety-ik"
                    ),
                    "prefix_ik_prefilter_used": False,
                    "successful_policy_rollout_export": {
                        "enabled": bool(args.export_success_demos),
                        "dataset_role": (
                            "successful_policy_rollout_self_distillation"
                        ),
                        "expert_demonstrations": False,
                        "output": str(success_demo_target),
                        "policy_start_step": int(SCRIPTED_PREFIX_STEPS),
                        "storage": storage_estimate,
                    },
                    "diffusion_sha256": diffusion_sha256,
                    "tqc_sha256": tqc_sha256,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return {}

    class FixedSpawnEnv(MJWarpScrewEnv):
        """Inject fixed starts without the normal prefix-IK rejection gate."""

        def __init__(
            self,
            *env_args,
            fixed_spawn_xy: np.ndarray,
            fixed_socket_mouth_xy: np.ndarray,
            fixed_socket_angle_deg: float,
            **env_kwargs,
        ) -> None:
            self._fixed_spawn_np = np.asarray(
                fixed_spawn_xy, dtype=np.float32
            )
            self._fixed_mouth_np = np.asarray(
                fixed_socket_mouth_xy, dtype=np.float32
            )
            self._fixed_angle_deg = float(fixed_socket_angle_deg)
            self._fixed_spawn_tensor = None
            self._fixed_mouth_tensor = None
            self._initial_command_cache = None
            self._initial_ctrl_cache = None
            self._initial_passive_cache = None
            self.initial_step0_ik_success = None
            self.initial_step0_failed_fingers = None
            super().__init__(*env_args, **env_kwargs)

        def _fixed_tensors(self):
            torch_local = torch
            if self._fixed_spawn_tensor is None:
                self._fixed_spawn_tensor = torch_local.as_tensor(
                    self._fixed_spawn_np,
                    device=self.device,
                    dtype=torch_local.float32,
                )
                self._fixed_mouth_tensor = torch_local.as_tensor(
                    self._fixed_mouth_np,
                    device=self.device,
                    dtype=torch_local.float32,
                )
            return self._fixed_spawn_tensor, self._fixed_mouth_tensor

        def _sample_reachable(self, mask):
            fixed_spawn_tensor, fixed_mouth_tensor = self._fixed_tensors()
            slots = torch.nonzero(mask, as_tuple=False).flatten()
            spawn = self.state.spawn_xy.clone()
            socket_angle = self.state.socket_angle_deg.clone()
            socket_mouth_xy = self.state.socket_mouth[:, :2].clone()
            spawn[slots] = fixed_spawn_tensor.index_select(0, slots)
            socket_mouth_xy[slots] = fixed_mouth_tensor.index_select(0, slots)
            socket_angle[slots] = self._fixed_angle_deg

            # Compute the step-zero reset solution once for every fixed world.
            # Completed worlds are reset on every later vector step so they
            # remain frozen before Diffusion activation; caching avoids
            # repeating this IK solve thousands of times.
            if self._initial_ctrl_cache is None:
                all_step = torch.zeros(
                    self.worlds, device=self.device, dtype=torch.long
                )
                initial_command = self.supervisor.at(
                    all_step,
                    fixed_spawn_tensor,
                    torch.full(
                        (self.worlds,),
                        self._fixed_angle_deg,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    fixed_mouth_tensor,
                )
                targets = initial_command.fingertip_targets(
                    self.supervisor.grasp_height_at(all_step)
                )
                initial = self.ik.solve(
                    targets,
                    torch.zeros(
                        (self.worlds, 16),
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
                self._initial_command_cache = initial_command
                self._initial_ctrl_cache = initial.ctrl.clone()
                self._initial_passive_cache = initial.passive_q.clone()
                self.initial_step0_ik_success = initial.success.clone()
                self.initial_step0_failed_fingers = (
                    ~initial.finger_success
                ).sum(dim=-1)

            command = self.current_command.clone()
            self._assign_command_rows(
                command,
                slots,
                self._initial_command_cache,
                slots,
            )
            ctrl = torch.zeros(
                (self.worlds, 16), device=self.device, dtype=torch.float32
            )
            ctrl[slots] = self._initial_ctrl_cache.index_select(0, slots)
            passive = torch.zeros(
                (self.worlds, *self._initial_passive_cache.shape[1:]),
                device=self.device,
                dtype=self._initial_passive_cache.dtype,
            )
            passive[slots] = self._initial_passive_cache.index_select(0, slots)
            return (
                spawn,
                socket_angle,
                socket_mouth_xy,
                command,
                ctrl,
                passive,
            )

    torch.manual_seed(evaluation_seed)
    torch.cuda.manual_seed_all(evaluation_seed)
    trainer = None
    if args.with_tqc:
        if tqc_payload is None:
            raise RuntimeError("TQC payload was not loaded")
        trainer = TQCTrainer(runtime_config, device=runtime_config.runtime.device)
        trainer.actor.load_state_dict(tqc_payload["actor"], strict=True)
        trainer.actor.eval()

    trajectory_stager = None
    rollout_completed_cleanly = False
    if args.export_success_demos:
        output_dir.mkdir(parents=True, exist_ok=True)
        staging_root = output_dir / (
            f".{args.stem}.trajectory_staging.{os.getpid()}"
        )
        trajectory_stager = TimeMajorTrajectoryStager(
            staging_root,
            worlds=worlds,
            steps=max_steps,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
        )
    trial_rows: list[dict[str, object] | None] = [None] * worlds
    started = time.monotonic()

    def make_trial_row(
        *,
        selected: Mapping[str, Any],
        selected_index: int,
        local_world_index: int,
        global_world_index: int,
        env: FixedSpawnEnv,
        ever_diffusion,
        first_diffusion_step,
        batch_index: int,
        batch_seed: int,
        forced_truncation: bool,
    ) -> dict[str, object]:
        phase_id = int(selected["task_phase"][selected_index].item())
        phase = PHASE_NAMES.get(phase_id, f"phase_{phase_id}")
        if forced_truncation:
            success = False
            failure = False
            truncated = True
            terminal_steps = max_steps
        else:
            success = bool(selected["success"][selected_index].item())
            failure = bool(selected["failure"][selected_index].item())
            truncated = bool(selected["truncated"][selected_index].item())
            terminal_steps = int(
                selected["elapsed_steps"][selected_index].item()
            )
        reasons = failure_reasons(
            selected,
            selected_index,
            phase,
            forced_truncation=forced_truncation,
        )
        spawn = fixed_spawn[global_world_index]
        return {
            "rollout_index": int(global_world_index),
            "rollout_batch": int(batch_index),
            "rollout_batch_seed": int(batch_seed),
            "cell_id": int(world_cell_ids[global_world_index]),
            "trial_id": int(world_trial_ids[global_world_index]),
            "root_seed": evaluation_seed,
            "spawn_x_m": float(spawn[0]),
            "spawn_y_m": float(spawn[1]),
            "spawn_radius_mm": float(np.linalg.norm(spawn) * 1000.0),
            "spawn_azimuth_deg": float(
                math.degrees(math.atan2(spawn[1], spawn[0]))
            ),
            "initial_step0_ik_success": bool(
                env.initial_step0_ik_success[local_world_index].item()
            ),
            "initial_step0_failed_fingers": int(
                env.initial_step0_failed_fingers[local_world_index].item()
            ),
            "diffusion_ever_active": bool(
                ever_diffusion[local_world_index].item()
            ),
            "first_diffusion_step": int(
                first_diffusion_step[local_world_index].item()
            ),
            "terminal_steps": int(terminal_steps),
            "terminal_phase": phase,
            "success": success,
            "failure": failure,
            "truncated": truncated,
            "empirically_reached": success,
            "failure_reasons": ";".join(reasons),
            "episode_return": float(
                selected["episode_return"][selected_index].item()
            ),
            "final_depth_mm": float(
                selected["depth"][selected_index].item() * 1000.0
            ),
            "final_turns": float(
                selected["twist"][selected_index].item()
                / (2.0 * math.pi)
            ),
            "final_lateral_error_mm": float(
                selected["lateral_error"][selected_index].item() * 1000.0
            ),
            "final_axis_error_deg": float(
                math.degrees(selected["axis_error"][selected_index].item())
            ),
            "final_speed_m_s": float(
                selected["speed"][selected_index].item()
            ),
            "ik_failures": int(
                selected["ik_failures"][selected_index].item()
            ),
            "ik_anchor_fallbacks": int(
                selected["ik_anchor_fallbacks"][selected_index].item()
            ),
            "ik_safety_stops": int(
                selected["ik_safety_stops"][selected_index].item()
            ),
            "shaft_captured": bool(
                selected["shaft_captured"][selected_index].item()
            ),
            "thread_engaged": bool(
                selected["thread_engaged"][selected_index].item()
            ),
        }

    try:
        for batch_index, batch in enumerate(rollout_batches):
            batch_start = int(batch.start)
            batch_stop = int(batch.stop)
            batch_worlds = batch_stop - batch_start
            batch_seed = evaluation_seed + batch_index * BATCH_SEED_STRIDE
            torch.manual_seed(batch_seed)
            torch.cuda.manual_seed_all(batch_seed)
            controller = BatchedDiffusionController.from_checkpoint(
                diffusion_path,
                worlds=batch_worlds,
                inference_steps=runtime_config.diffusion.inference_steps,
                replan_interval=runtime_config.diffusion.replan_interval,
                device=runtime_config.runtime.device,
                seed=batch_seed,
            )
            env = None
            result = None
            last_info: Mapping[str, Any] | None = None
            try:
                env = FixedSpawnEnv(
                    runtime_config,
                    worlds=batch_worlds,
                    baseline="diffusion",
                    diffusion_controller=controller,
                    curriculum_progress=1.0,
                    fixed_spawn_xy=fixed_spawn[batch],
                    fixed_socket_mouth_xy=fixed_mouths[batch],
                    fixed_socket_angle_deg=args.socket_angle_deg,
                )
                if trajectory_stager is not None:
                    trajectory_stager.write_initial_batch(
                        batch_start,
                        env.obs_history[:, 1]
                        .detach()
                        .to(device="cpu", dtype=torch.float16)
                        .numpy(),
                    )
                active = torch.ones(
                    batch_worlds, device=env.device, dtype=torch.bool
                )
                ever_diffusion = torch.zeros_like(active)
                first_diffusion_step = torch.full(
                    (batch_worlds,),
                    -1,
                    device=env.device,
                    dtype=torch.long,
                )
                zero_residual = torch.zeros(
                    (batch_worlds, 16),
                    device=env.device,
                    dtype=torch.float32,
                )
                print(
                    json.dumps(
                        {
                            "event": "workspace_reachability_batch_start",
                            "batch_index": batch_index,
                            "batch_count": len(rollout_batches),
                            "global_world_start": batch_start,
                            "global_world_stop": batch_stop,
                            "live_worlds": batch_worlds,
                            "batch_seed": batch_seed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                for vector_step in range(1, max_steps + 1):
                    if trainer is None:
                        result = env.step(residual_action=zero_residual)
                    else:
                        def residual_policy(tqc_state, action_mask):
                            return trainer.act(
                                tqc_state,
                                action_mask=action_mask,
                                deterministic=True,
                            )

                        result = env.step(residual_policy=residual_policy)
                    if trajectory_stager is not None:
                        trajectory_stager.write_step_batch(
                            vector_step,
                            batch_start,
                            result.observation.detach()
                            .to(device="cpu", dtype=torch.float16)
                            .numpy(),
                            result.expert_action.detach()
                            .to(device="cpu", dtype=torch.float16)
                            .numpy(),
                        )
                        if vector_step % args.success_demo_flush_every == 0:
                            trajectory_stager.flush()
                    last_info = result.info
                    diffusion_now = result.info["diffusion_active"] & active
                    newly_active = diffusion_now & ~ever_diffusion
                    first_diffusion_step[newly_active] = vector_step
                    ever_diffusion |= diffusion_now
                    first_done = result.done & active
                    if bool(first_done.any()):
                        indices = torch.nonzero(
                            first_done, as_tuple=False
                        ).flatten()
                        selected = _select_info(result.info, indices)
                        local_indices = (
                            indices.detach().cpu().numpy().astype(int)
                        )
                        for selected_index, local_world_index in enumerate(
                            local_indices
                        ):
                            global_world_index = (
                                batch_start + int(local_world_index)
                            )
                            trial_rows[global_world_index] = make_trial_row(
                                selected=selected,
                                selected_index=selected_index,
                                local_world_index=int(local_world_index),
                                global_world_index=global_world_index,
                                env=env,
                                ever_diffusion=ever_diffusion,
                                first_diffusion_step=first_diffusion_step,
                                batch_index=batch_index,
                                batch_seed=batch_seed,
                                forced_truncation=False,
                            )
                        active[indices] = False
                    if vector_step == 1 or vector_step % args.log_every == 0:
                        completed = (
                            batch_start
                            + batch_worlds
                            - int(active.sum().item())
                        )
                        print(
                            json.dumps(
                                {
                                    "event": "workspace_reachability_progress",
                                    "batch_index": batch_index,
                                    "batch_count": len(rollout_batches),
                                    "vector_step": vector_step,
                                    "completed_trials": completed,
                                    "total_trials": worlds,
                                    "progress": completed / worlds,
                                    "elapsed_seconds": (
                                        time.monotonic() - started
                                    ),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    if not bool(active.any()):
                        break
                    # The bounded batch keeps the full-forward allocation
                    # below the GPU memory gate while completed rows are
                    # deterministically frozen at their step-zero reset.
                    reset_mask = ~active
                    if bool(reset_mask.any()):
                        env.reset(reset_mask)

                if bool(active.any()):
                    if last_info is None:
                        raise RuntimeError(
                            "rollout batch produced no transition"
                        )
                    indices = torch.nonzero(
                        active, as_tuple=False
                    ).flatten()
                    selected = _select_info(last_info, indices)
                    local_indices = (
                        indices.detach().cpu().numpy().astype(int)
                    )
                    for selected_index, local_world_index in enumerate(
                        local_indices
                    ):
                        global_world_index = (
                            batch_start + int(local_world_index)
                        )
                        trial_rows[global_world_index] = make_trial_row(
                            selected=selected,
                            selected_index=selected_index,
                            local_world_index=int(local_world_index),
                            global_world_index=global_world_index,
                            env=env,
                            ever_diffusion=ever_diffusion,
                            first_diffusion_step=first_diffusion_step,
                            batch_index=batch_index,
                            batch_seed=batch_seed,
                            forced_truncation=True,
                        )
                if trajectory_stager is not None:
                    trajectory_stager.flush()
                print(
                    json.dumps(
                        {
                            "event": "workspace_reachability_batch_complete",
                            "batch_index": batch_index,
                            "batch_count": len(rollout_batches),
                            "completed_trials": batch_stop,
                            "total_trials": worlds,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                if env is not None:
                    env.close()
                result = None
                last_info = None
                active = None
                ever_diffusion = None
                first_diffusion_step = None
                zero_residual = None
                indices = None
                reset_mask = None
                diffusion_now = None
                newly_active = None
                first_done = None
                env = None
                controller = None
                release = release_cuda_batch_memory(
                    torch, runtime_config.runtime.device
                )
                print(
                    json.dumps(
                        {
                            "event": (
                                "workspace_reachability_batch_memory_released"
                            ),
                            "batch_index": batch_index,
                            **release,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        rollout_completed_cleanly = True
    finally:
        if trajectory_stager is not None and not rollout_completed_cleanly:
            # Preserve the incomplete staging files for diagnosis/recovery;
            # only a finalized compact corpus authorizes their deletion.
            trajectory_stager.close()

    if any(row is None for row in trial_rows):
        raise RuntimeError("at least one rollout has no terminal record")
    complete_trials = [row for row in trial_rows if row is not None]
    cell_rows = aggregate_cells(
        grid=grid,
        collision=collision,
        trials=complete_trials,
        trials_per_cell=args.trials_per_cell,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    trial_csv = output_dir / f"{args.stem}.trials.csv"
    cell_csv = output_dir / f"{args.stem}.cells.csv"
    metadata_path = output_dir / f"{args.stem}.metadata.json"
    _atomic_write_csv(trial_csv, complete_trials)
    _atomic_write_csv(cell_csv, cell_rows)
    controller_label = (
        "Diffusion--Safety--IK--TQC"
        if args.with_tqc
        else "Diffusion--Safety--IK"
    )
    training_semantics = training_distribution_semantics(selected_payload)
    radial_extension_is_ood = bool(
        args.spawn_radius_mm
        > float(training_semantics["training_spawn_radius_mm"]) + 1.0e-9
    )
    success_demo_export: dict[str, Any] = {
        "enabled": bool(args.export_success_demos),
        "status": "disabled",
        "dataset_role": "successful_policy_rollout_self_distillation",
        "expert_demonstrations": False,
        "storage_preflight": storage_estimate,
    }
    if trajectory_stager is not None:
        success_demo_export = export_successful_policy_rollouts(
            stager=trajectory_stager,
            target=success_demo_target,
            trial_rows=complete_trials,
            trial_csv=trial_csv,
            fixed_spawn=fixed_spawn,
            fixed_mouths=fixed_mouths,
            socket_angle_deg=args.socket_angle_deg,
            policy_start_step=SCRIPTED_PREFIX_STEPS,
            action_horizon=runtime_config.demos.action_horizon,
            shard_episodes=args.success_demo_shard_episodes,
            max_episodes=args.success_demo_max_episodes,
            selection_seed=evaluation_seed + 1_000_003,
            provenance={
                "backend": "mjwarp-native-v3",
                "task_geometry_hash": diffusion_payload.get(
                    "task_geometry_hash"
                ),
                "policy_start_step": int(SCRIPTED_PREFIX_STEPS),
                "root_seed": evaluation_seed,
                "controller": controller_label,
                "controller_residual": (
                    "paired deterministic TQC actor"
                    if args.with_tqc
                    else "16-dimensional all-zero post-IK residual"
                ),
                "fixed_start_grid": {
                    "angular_bins": args.angular_bins,
                    "radial_bins": args.radial_bins,
                    "trials_per_cell": args.trials_per_cell,
                    "spawn_radius_mm": args.spawn_radius_mm,
                    "socket_excluded_cells": int(collision.sum()),
                    "socket_exclusion_semantics": (
                        "exact initial CPU-MuJoCo screw-vs-socket physical contact"
                    ),
                    "socket_exclusion_comparison": collision_comparison,
                    "tested_cells": int(len(candidate_ids)),
                    "total_rollouts": int(worlds),
                    "rollout_batches": int(len(rollout_batches)),
                    "max_live_worlds": int(args.max_live_worlds),
                    "peak_live_worlds": int(live_worlds_peak),
                    "batch_world_counts": [
                        int(batch.stop - batch.start)
                        for batch in rollout_batches
                    ],
                    "global_rollout_order": (
                        "ascending collision-free cell_id, then trial_id"
                    ),
                },
                "distribution_shift": {
                    "training_region": training_semantics["training_region"],
                    "rollout_region": (
                        "full workspace disc minus exact initial physical "
                        "screw--socket contacts"
                    ),
                    "opposite_half_disc_is_ood": training_semantics[
                        "opposite_half_disc_is_ood"
                    ],
                    "training_restrict_screw_spawn_to_opening_half_disk": (
                        training_semantics[
                            "training_restrict_screw_spawn_to_opening_half_disk"
                        ]
                    ),
                    "training_spawn_radius_mm": training_semantics[
                        "training_spawn_radius_mm"
                    ],
                    "rollout_spawn_radius_mm": float(args.spawn_radius_mm),
                    "radial_extension_is_ood": radial_extension_is_ood,
                },
                "initial_socket_exclusion": {
                    "classifier": collision_provenance,
                    "outer_obb": socket_obb_provenance,
                    "comparison": collision_comparison,
                },
                "config": runtime_config.to_dict(),
                "selected_checkpoint_config": selected_payload,
                "checkpoints": {
                    "diffusion_sha256": diffusion_sha256,
                    "tqc_sha256": tqc_sha256 if args.with_tqc else None,
                },
                "source_script": {
                    "path": str(SCRIPT_PATH),
                    "sha256": sha256_file(SCRIPT_PATH),
                },
            },
        )
    tested_cells = [row for row in cell_rows if not row["socket_excluded"]]
    demonstrated_cells = [
        row for row in tested_cells if int(row["successes"]) > 0
    ]
    robust_cells = [
        row
        for row in tested_cells
        if int(row["successes"]) == args.trials_per_cell
    ]
    total_successes = sum(int(row["successes"]) for row in tested_cells)
    metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "control_stack": controller_label,
        "terminology": (
            "These are fixed-start policy rollouts, not expert training "
            "demonstrations. A success demonstrates empirical reachability; "
            "a failed finite sample does not prove geometric impossibility."
        ),
        "distribution_shift": {
            "training_region": training_semantics["training_region"],
            "evaluation_region": (
                "full workspace disc minus exact initial physical "
                "screw--socket contacts"
            ),
            "out_of_distribution_opposite_half_disc": training_semantics[
                "opposite_half_disc_is_ood"
            ],
            "training_restrict_screw_spawn_to_opening_half_disk": (
                training_semantics[
                    "training_restrict_screw_spawn_to_opening_half_disk"
                ]
            ),
            "training_spawn_radius_mm": training_semantics[
                "training_spawn_radius_mm"
            ],
            "evaluation_spawn_radius_mm": float(args.spawn_radius_mm),
            "radial_extension_is_ood": radial_extension_is_ood,
            "prefix_ik_prefilter_used": False,
            "step0_ik_used_only_for_simulator_initialisation": True,
        },
        "grid": {
            "angular_bins": args.angular_bins,
            "radial_bins": args.radial_bins,
            "angular_resolution_deg": 360.0 / args.angular_bins,
            "radial_resolution_mm": args.spawn_radius_mm / args.radial_bins,
            "sampling_location": "centre point of each polar bin",
            "cells": grid_count,
            "spawn_radius_mm": args.spawn_radius_mm,
            "socket_excluded_cells": int(collision.sum()),
            "socket_exclusion_semantics": (
                "exact initial CPU-MuJoCo screw-vs-socket physical contact"
            ),
            "socket_exclusion_comparison": collision_comparison,
            "tested_cells": len(tested_cells),
            "trials_per_cell": args.trials_per_cell,
            "total_rollouts": worlds,
        },
        "socket": {
            "mouth_azimuth_deg": args.socket_mouth_azimuth_deg,
            "mouth_orbit_radius_mm": (
                supervisor_cpu.config.socket_mouth_orbit_radius_m * 1000.0
            ),
            "elevation_deg": args.socket_angle_deg,
            "initial_contact_exclusion": {
                "classifier": collision_provenance,
                "outer_obb": socket_obb_provenance,
                "comparison": collision_comparison,
            },
        },
        "results": {
            "demonstrated_reachable_cells": len(demonstrated_cells),
            "demonstrated_reachable_fraction": (
                len(demonstrated_cells) / len(tested_cells)
            ),
            "all_trials_success_cells": len(robust_cells),
            "all_trials_success_fraction": len(robust_cells) / len(tested_cells),
            "successful_rollouts": total_successes,
            "rollout_success_fraction": total_successes / worlds,
            "handoff_reached_rollouts": sum(
                int(row["handoff_reached_trials"]) for row in tested_cells
            ),
            "failures": sum(bool(row["failure"]) for row in complete_trials),
            "truncations": sum(
                bool(row["truncated"]) for row in complete_trials
            ),
        },
        "runtime": {
            "root_seed": evaluation_seed,
            "trial_sampling": (
                f"{args.trials_per_cell} distinct policy-noise rows per cell; "
                "stable contiguous batches use deterministic substream seeds "
                f"root_seed + batch_index * {BATCH_SEED_STRIDE}"
            ),
            "max_steps": max_steps,
            "worlds": worlds,
            "total_rollout_worlds": worlds,
            "max_live_worlds": int(args.max_live_worlds),
            "peak_live_worlds": int(live_worlds_peak),
            "rollout_batches": int(len(rollout_batches)),
            "batch_world_counts": [
                int(batch.stop - batch.start) for batch in rollout_batches
            ],
            "batch_seed_stride": BATCH_SEED_STRIDE,
            "batching_changes_grid_or_trial_count": False,
            "global_rollout_order": (
                "ascending collision-free cell_id, then trial_id"
            ),
            "elapsed_seconds": time.monotonic() - started,
        },
        "successful_policy_rollout_export": success_demo_export,
        "success_contract": {
            "source": "hpc/s_hpc/task.py::reward_done",
            "target_depth_mm": (
                supervisor_cpu.config.target_insertion_depth_m * 1000.0
            ),
            "target_turns": (
                supervisor_cpu.config.target_rotation_rad / (2.0 * math.pi)
            ),
            "depth_absolute_tolerance_mm": 2.0,
            "rotation_absolute_tolerance_deg": 8.0,
            "lateral_error_max_mm": 3.0,
            "axis_error_max_deg": 3.0,
            "speed_max_m_s": 0.35,
            "required_terminal_phase": ["final_hold", "complete"],
            "failure_veto": True,
            "episode_step_limit": runtime_config.physics.max_episode_steps,
        },
        "diffusion_inference": {
            "ddim_eta": 0.0,
            "inference_steps": runtime_config.diffusion.inference_steps,
            "replan_interval": runtime_config.diffusion.replan_interval,
            "residual_action": (
                "paired deterministic TQC actor"
                if args.with_tqc
                else "16-dimensional all-zero post-IK residual"
            ),
        },
        "checkpoints": {
            "diffusion": {
                "path": str(diffusion_path),
                "sha256": diffusion_sha256,
                "config_differences": diffusion_differences,
            },
            "tqc": (
                {
                    "path": str(tqc_path),
                    "sha256": tqc_sha256,
                    "enabled": bool(args.with_tqc),
                    "config_differences": tqc_differences,
                }
                if tqc_path is not None
                else None
            ),
            "selected_config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
                "canonical_sha256": canonical_sha256(selected_payload),
            },
        },
        "contracts": {
            "task_geometry_hash": diffusion_payload.get("task_geometry_hash"),
            "diffusion_schema_hash": diffusion_payload.get("schema_hash"),
            "asset_sha256": TASK_ASSET_HASHES,
        },
        "outputs": {
            "trial_csv": {
                "path": trial_csv.name,
                "sha256": sha256_file(trial_csv),
            },
            "cell_csv": {
                "path": cell_csv.name,
                "sha256": sha256_file(cell_csv),
            },
            "successful_policy_rollout_dataset": (
                {
                    "path": str(success_demo_target),
                    "metadata_sha256": success_demo_export.get(
                        "metadata_sha256"
                    ),
                    "dataset_role": (
                        "successful_policy_rollout_self_distillation"
                    ),
                    "expert_demonstrations": False,
                }
                if success_demo_export.get("status") == "complete"
                else None
            ),
        },
        "source": {
            "script": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)}
        },
    }
    _atomic_write_json(metadata_path, metadata)
    print(json.dumps(metadata["results"], sort_keys=True), flush=True)
    outputs = {
        "trial_csv": trial_csv,
        "cell_csv": cell_csv,
        "metadata": metadata_path,
    }
    if success_demo_export.get("status") == "complete":
        outputs["successful_policy_rollout_dataset"] = success_demo_target
    return outputs




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute GPU policy rollouts")
    run.add_argument("--diffusion", required=True)
    run.add_argument("--tqc")
    run.add_argument("--with-tqc", action="store_true")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--stem", default=DEFAULT_STEM)
    run.add_argument("--angular-bins", type=int, default=360)
    run.add_argument("--radial-bins", type=int, default=50)
    run.add_argument("--trials-per-cell", type=int, default=2)
    run.add_argument("--spawn-radius-mm", type=float, default=100.0)
    run.add_argument(
        "--max-live-worlds",
        type=int,
        default=DEFAULT_MAX_LIVE_WORLDS,
        help=(
            "maximum simultaneous CUDA/MJWarp rollout worlds; the complete "
            "fixed grid is processed in stable contiguous batches"
        ),
    )
    run.add_argument("--socket-mouth-azimuth-deg", type=float, default=0.0)
    run.add_argument("--socket-angle-deg", type=float, default=45.0)
    run.add_argument("--seed", type=int)
    run.add_argument("--max-steps", type=int)
    run.add_argument("--log-every", type=int, default=25)
    run.add_argument(
        "--export-success-demos",
        action="store_true",
        help=(
            "optionally stage trajectories on disk and export only formal "
            "task successes as variable-length self-distillation DemoStore data"
        ),
    )
    run.add_argument(
        "--success-demo-output",
        help=(
            "successful policy-rollout dataset directory; defaults beside "
            "the reachability outputs"
        ),
    )
    run.add_argument(
        "--success-demo-shard-episodes",
        type=int,
        default=8,
        help="episodes copied from time-major staging per bounded-RAM shard",
    )
    run.add_argument(
        "--success-demo-max-episodes",
        type=int,
        default=8192,
        help=(
            "maximum retained successful rollouts; excess successes are "
            "sampled reproducibly without replacement"
        ),
    )
    run.add_argument(
        "--success-demo-flush-every",
        type=int,
        default=100,
        help="flush time-major trajectory memmaps every N vector steps",
    )
    run.add_argument(
        "--success-demo-reserve-gib",
        type=float,
        default=2.0,
        help="free-space reserve added to the all-success peak estimate",
    )
    run.add_argument(
        "--validate-only",
        action="store_true",
        help="validate checkpoints/grid without creating a CUDA environment",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "log_every", 1) <= 0:
        parser.error("--log-every must be positive")
    if args.command == "run":
        outputs = run_rollouts(args)
    print(
        json.dumps(
            {key: str(path) for key, path in outputs.items()}, sort_keys=True
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
