"""Deterministic closed-loop IK expert for Diffusion demonstrations.

The environment already owns the two pieces that must remain immutable for
one episode: goal/path tensors are materialized by ``reset()``, and the
platform reset bank supplies deterministic feasible object-local
pregrasp/contact anchors and their 40%-60% planned grasp-plane fraction. This
module replaces reward-guided action search only.  It keeps
the former teacher interface so the training and capacity loops can switch
implementations without changing their allocation lifecycle.

The episode command encodes the platform planner's frozen feasible fraction
for all four hard tips, while keeping action yaw zero because the frozen anchor
geometry already contains planner yaw. Per-control-tick commands are a bounded
deterministic feedback residual derived from the observation's four hard-tip
target errors. Diffusion therefore imitates a closed-loop high-level IK command;
actuator commands remain exclusively owned by the environment's constrained IK
solver.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import contract
from .curriculum import PHASE
from .hpc_config import HPCConfig


# ``tip_target_error_world`` is normalized by 30 mm in the observation while
# the executable dynamic residual is normalized to +/-3 mm.  A gain of one in
# observation space therefore asks for a conservative 10% tracking-error
# compensation and remains inside the existing per-tip residual contract.
TIP_ERROR_FEEDBACK_GAIN = 1.0


class MJWarpIKExpert:
    """Frozen-plan, deterministic expert with the legacy teacher interface."""

    def __init__(
        self,
        config: HPCConfig,
        *,
        platform_spec: Any,
        device: str,
    ) -> None:
        import torch

        self.torch = torch
        self.config = config
        self.device = torch.device(device)
        self.task = contract.validate_task(config.task)
        self.task_index = contract.TASKS.index(self.task)
        self.population = int(config.elite.population)
        self.groups = int(config.groups_per_vector_batch)
        self.object_world_counts = tuple(config.object_world_counts)
        self.object_group_counts = tuple(config.object_group_counts)
        if any(worlds % self.population for worlds in self.object_world_counts):
            raise ValueError("an IK-expert population crosses an object shard")
        if getattr(platform_spec, "task", None) != self.task:
            raise ValueError("IK expert and platform_spec tasks differ")
        shards = tuple(getattr(platform_spec, "shards", ()))
        if len(shards) != len(contract.OBJECTS):
            raise ValueError("IK expert requires one ordered shard per object")
        planned_fraction = np.empty(
            (len(contract.OBJECTS), contract.RESET_TEMPLATES_PER_OBJECT),
            dtype=np.float32,
        )
        planned_tip_fraction = np.empty(
            (
                len(contract.OBJECTS),
                contract.RESET_TEMPLATES_PER_OBJECT,
                len(contract.TIP_ORDER),
            ),
            dtype=np.float32,
        )
        planned_yaw = np.empty_like(planned_fraction)
        for object_index, shard in enumerate(shards):
            if int(getattr(shard, "fixed_object_index", -1)) != object_index:
                raise ValueError("IK expert platform shards are not object ordered")
            fraction_bank = np.asarray(
                shard.planned_grasp_plane_fraction, dtype=np.float64
            )
            yaw_bank = np.asarray(
                shard.planned_grasp_plane_yaw_rad, dtype=np.float64
            )
            contact_bank = np.asarray(
                shard.contact_targets_local, dtype=np.float64
            )
            bottom_bank = np.asarray(
                shard.object_local_bottom_z_m, dtype=np.float64
            )
            top_bank = np.asarray(
                shard.object_local_top_z_m, dtype=np.float64
            )
            expected_bank_shape = (
                len(contract.OBJECTS),
                contract.RESET_TEMPLATES_PER_OBJECT,
            )
            if (
                fraction_bank.shape != expected_bank_shape
                or yaw_bank.shape != expected_bank_shape
                or contact_bank.shape
                != (
                    *expected_bank_shape,
                    len(contract.TIP_ORDER),
                    3,
                )
                or bottom_bank.shape != (len(contract.OBJECTS),)
                or top_bank.shape != (len(contract.OBJECTS),)
            ):
                raise ValueError("platform planned grasp-plane bank has wrong shape")
            fraction_row = fraction_bank[object_index]
            yaw_row = yaw_bank[object_index]
            object_height = top_bank[object_index] - bottom_bank[object_index]
            if not np.isfinite(object_height) or object_height <= 0.0:
                raise ValueError("platform object height is invalid")
            tip_fraction_row = (
                contact_bank[object_index, :, :, 2]
                - bottom_bank[object_index]
            ) / object_height
            if (
                not np.isfinite(fraction_row).all()
                or bool(
                    (
                        (fraction_row < contract.GRASP_PLANE_MIN_FRACTION)
                        | (fraction_row > contract.GRASP_PLANE_MAX_FRACTION)
                    ).any()
                )
            ):
                raise ValueError("platform planned grasp-plane fraction is invalid")
            if (
                not np.isfinite(tip_fraction_row).all()
                or bool(
                    (
                        (
                            tip_fraction_row
                            < contract.GRASP_PLANE_MIN_FRACTION
                        )
                        | (
                            tip_fraction_row
                            > contract.GRASP_PLANE_MAX_FRACTION
                        )
                    ).any()
                )
            ):
                raise ValueError(
                    "platform planned per-tip grasp anchors are invalid"
                )
            if (
                not np.isfinite(yaw_row).all()
                or bool(
                    (
                        np.abs(yaw_row)
                        > contract.FROZEN_PLANNER_YAW_ABS_LIMIT_RAD + 1.0e-12
                    ).any()
                )
            ):
                invalid = np.flatnonzero(
                    ~np.isfinite(yaw_row)
                    | (
                        np.abs(yaw_row)
                        > contract.FROZEN_PLANNER_YAW_ABS_LIMIT_RAD + 1.0e-12
                    )
                )
                template_index = int(invalid[0])
                raise ValueError(
                    "platform planned grasp-plane yaw is invalid for "
                    f"{contract.OBJECTS[object_index]} template "
                    f"{template_index}: {yaw_row[template_index]!r} rad"
                )
            planned_fraction[object_index] = fraction_row
            planned_tip_fraction[object_index] = tip_fraction_row
            planned_yaw[object_index] = yaw_row
        self.planned_grasp_plane_fraction = torch.as_tensor(
            planned_fraction, device=self.device, dtype=torch.float32
        )
        self.planned_tip_height_fraction = torch.as_tensor(
            planned_tip_fraction, device=self.device, dtype=torch.float32
        )
        self.planned_grasp_plane_yaw_rad = torch.as_tensor(
            planned_yaw, device=self.device, dtype=torch.float32
        )

        # These names deliberately match MJWarpEliteTeacher's episode-owned
        # attributes.  Capacity/training cleanup can therefore release the
        # expert through the same explicit lifecycle boundary.
        self.episode_action: Any | None = None
        self.episode_planned_grasp_plane_fraction: Any | None = None
        self.episode_planned_grasp_plane_yaw_rad: Any | None = None
        self.dynamic_knots: None = None
        self.goals: dict[str, Any] | None = None
        self.group_object: Any | None = None
        self.group_task: Any | None = None
        self.group_template: Any | None = None
        self.phase_index = 0

    def clear_episode_state(self) -> None:
        """Release all world-sized tensors while retaining no learned state."""

        self.episode_action = None
        self.episode_planned_grasp_plane_fraction = None
        self.episode_planned_grasp_plane_yaw_rad = None
        self.dynamic_knots = None
        self.goals = None
        self.group_object = None
        self.group_task = None
        self.group_template = None

    def _group_goals(self, phase: str, episode_index: int) -> dict[str, Any]:
        """Build one deterministic, task-specific frozen goal per group."""

        torch = self.torch
        object_index = torch.repeat_interleave(
            torch.arange(
                len(contract.OBJECTS),
                device=self.device,
                dtype=torch.long,
            ),
            torch.tensor(
                self.object_group_counts,
                device=self.device,
                dtype=torch.long,
            ),
        )
        local_sequence = torch.cat(
            tuple(
                torch.arange(count, device=self.device, dtype=torch.long)
                + int(episode_index) * count
                for count in self.object_group_counts
            )
        )
        sequence = object_index + len(contract.OBJECTS) * local_sequence
        task_index = torch.full_like(local_sequence, self.task_index)
        template_index = torch.remainder(
            local_sequence, contract.RESET_TEMPLATES_PER_OBJECT
        )
        if phase != PHASE:
            raise ValueError(f"unsupported curriculum phase {phase!r}")
        source_region = torch.zeros_like(sequence)
        rotation_low = contract.ROTATION_ANGLE_MIN_DEG
        rotation_high = contract.ROTATION_ANGLE_MAX_DEG
        self.phase_index = 0

        path_seed = 1 + torch.remainder(
            (sequence + 1) * 1_000_003
            + (int(episode_index) + 1) * 97_409
            + int(self.config.seed) * 65_537,
            contract.TRANSLATION_PATH_SEED_MAX,
        )
        # Draw paths from a seeded, object/episode-specific permutation.
        # Every consecutive block of five groups contains each fixed path
        # exactly once, while the offset and traversal order change
        # reproducibly.  This gives uniform randomised sampling without
        # sacrificing fail-closed path coverage in small diagnostics.
        path_shape_count = len(contract.TRANSLATION_PATH_SHAPES)
        path_cycle_offset = torch.remainder(
            (object_index + 1) * 1_000_003
            + (int(episode_index) + 1) * 97_409
            + int(self.config.seed) * 65_537,
            path_shape_count,
        )
        path_cycle_stride = 1 + torch.remainder(
            (object_index + 1) * 4_099
            + (int(episode_index) + 1) * 1_009
            + int(self.config.seed) * 257,
            path_shape_count - 1,
        )
        path_shape_index = torch.remainder(
            path_cycle_offset + path_cycle_stride * local_sequence,
            path_shape_count,
        )
        rotation_angle_deg = rotation_low + torch.remainder(
            local_sequence, rotation_high - rotation_low + 1
        )
        if self.task == contract.TASK_TRANSLATE:
            rotation_angle_deg = torch.zeros_like(rotation_angle_deg)
        else:
            path_seed = torch.zeros_like(path_seed)
            path_shape_index = torch.zeros_like(path_shape_index)

        self.group_object = object_index
        self.group_task = task_index
        self.group_template = template_index

        def worlds(value: Any) -> Any:
            # Every former population member receives the exact same frozen
            # task/path/plan.  Population slots now provide certification
            # redundancy only; they are never randomized or reward-ranked.
            return value.repeat_interleave(self.population)

        return {
            "object_index": worlds(object_index),
            "template_index": worlds(template_index),
            "task_index": worlds(task_index),
            "path_seed": worlds(path_seed),
            "path_shape_index": worlds(path_shape_index),
            "rotation_angle_deg": worlds(rotation_angle_deg),
            "source_region": worlds(source_region),
        }

    def begin_episode(self, phase: str, episode_index: int) -> dict[str, Any]:
        """Freeze the task/path and platform-planned grasp plane."""

        self.clear_episode_state()
        goals = self._group_goals(phase, episode_index)
        self.episode_action = self.torch.zeros(
            (self.config.num_envs, contract.ACTION_DIM),
            device=self.device,
            dtype=self.torch.float32,
        )
        object_index = goals["object_index"].to(self.torch.long)
        template_index = goals["template_index"].to(self.torch.long)
        fraction = self.planned_grasp_plane_fraction[
            object_index, template_index
        ]
        tip_fraction = self.planned_tip_height_fraction[
            object_index, template_index
        ]
        planner_yaw = self.planned_grasp_plane_yaw_rad[
            object_index, template_index
        ]
        offset = tip_fraction - contract.GRASP_PLANE_NOMINAL_FRACTION
        normalized_fraction = offset / self.torch.where(
            offset <= 0.0,
            self.torch.full_like(
                offset,
                contract.GRASP_PLANE_NOMINAL_FRACTION
                - contract.GRASP_PLANE_MIN_FRACTION,
            ),
            self.torch.full_like(
                offset,
                contract.GRASP_PLANE_MAX_FRACTION
                - contract.GRASP_PLANE_NOMINAL_FRACTION,
            ),
        )
        if not bool(
            self.torch.isfinite(normalized_fraction).all()
            and (normalized_fraction.abs() <= 1.0 + 1.0e-6).all()
        ):
            raise ValueError("planned grasp plane cannot be encoded as an action")
        self.episode_action[:, contract.TIP_HEIGHT_SLICE] = normalized_fraction
        # Planner yaw is already embedded in the frozen pregrasp/contact
        # anchors. Encoding it again in the action would rotate them twice.
        self.episode_action[:, contract.YAW_INDEX] = 0.0
        self.episode_planned_grasp_plane_fraction = fraction
        self.episode_planned_grasp_plane_yaw_rad = planner_yaw
        self.goals = goals
        return goals

    def action(self, tick: int, observation: Any) -> Any:
        """Return deterministic bounded high-level hard-tip IK feedback."""

        del tick  # The closed-loop command depends on measured state only.
        if self.episode_action is None or self.goals is None:
            raise RuntimeError("begin_episode must be called before action")
        from .env import (
            OBSERVATION_DIM,
            OBSERVATION_SLICES,
            _quat_conjugate,
            _quat_rotate,
            _quat_slerp,
            action_mask_from_observation,
        )

        value = self.torch.as_tensor(
            observation,
            device=self.device,
            dtype=self.torch.float32,
        )
        expected = (self.config.num_envs, OBSERVATION_DIM)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"observation must have shape {expected}, got {tuple(value.shape)}"
            )
        reference_valid = (
            value[:, OBSERVATION_SLICES["action_reference_valid"]]
            .squeeze(-1)
            > 0.5
        )
        initial_quaternion = value[
            :, OBSERVATION_SLICES["initial_quaternion_world"]
        ]
        reference_quaternion = value[
            :, OBSERVATION_SLICES["action_reference_quaternion_world"]
        ]
        if self.task == contract.TASK_TRANSLATE:
            active_quaternion = reference_quaternion
        else:
            goal_quaternion = value[
                :, OBSERVATION_SLICES["goal_quaternion_world"]
            ]
            action_progress = value[
                :, OBSERVATION_SLICES["motion_progress"]
            ][:, 1]
            active_quaternion = _quat_slerp(
                reference_quaternion,
                goal_quaternion,
                action_progress.clamp(0.0, 1.0),
            )
        desired_quaternion = self.torch.where(
            reference_valid[:, None],
            active_quaternion,
            initial_quaternion,
        )

        # The observation reports world-frame target error, whereas the
        # executable per-tip residual is defined in the desired object frame.
        # Rotate each vector into that frame before adding it to the frozen
        # action-boundary residual.  The latter is required because the
        # environment executes ``dynamic - action_start_dynamic`` after the
        # action reference has been captured.
        error_world = value[
            :, OBSERVATION_SLICES["tip_target_error_world"]
        ].reshape(self.config.num_envs, len(contract.TIP_ORDER), 3)
        feedback_local = _quat_rotate(
            _quat_conjugate(desired_quaternion),
            error_world,
        ).reshape(self.config.num_envs, -1) * TIP_ERROR_FEEDBACK_GAIN
        action_start_residual = value[
            :, OBSERVATION_SLICES["action_start_action"]
        ][:, contract.TIP_XYZ_RESIDUAL_SLICE]
        baseline = self.torch.where(
            reference_valid[:, None],
            action_start_residual,
            self.torch.zeros_like(action_start_residual),
        )
        dynamic_command = self.torch.nan_to_num(
            baseline + feedback_local,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(
            contract.NORMALIZED_ACTION_MIN,
            contract.NORMALIZED_ACTION_MAX,
        )
        result = self.episode_action.clone()
        result[:, contract.TIP_XYZ_RESIDUAL_SLICE] = dynamic_command
        return result * action_mask_from_observation(value).to(result.dtype)

    def select_winners(
        self,
        episode_return: Any,
        success: Any,
        plane_compliance: Any,
    ) -> Any:
        """Select the first certified replica; reward never affects choice."""

        del episode_return
        torch = self.torch
        valid = (
            torch.as_tensor(success, device=self.device, dtype=torch.bool)
            & torch.as_tensor(
                plane_compliance, device=self.device, dtype=torch.bool
            )
        )
        expected = (self.config.num_envs,)
        if tuple(valid.shape) != expected:
            raise ValueError(
                f"success/compliance must have shape {expected}"
            )
        grouped = valid.reshape(self.groups, self.population)
        # argmax on a bool/int row returns the first True.  An all-false row
        # deliberately returns slot zero; train.py re-checks certification
        # and therefore stores no trajectory for that group.
        offset = grouped.to(torch.int8).argmax(dim=1)
        return (
            torch.arange(self.groups, device=self.device) * self.population
            + offset
        )

    def update_centroids(self, winner: Any) -> None:
        """Compatibility no-op: the deterministic expert has no learned state."""

        del winner


# Transitional alias: callers can change only their import module while
# preserving the historic class name and all call sites.
MJWarpEliteTeacher = MJWarpIKExpert


__all__ = [
    "MJWarpEliteTeacher",
    "MJWarpIKExpert",
    "TIP_ERROR_FEEDBACK_GAIN",
]
