"""GPU-resident expert schedule and task-space safety projections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

from grasp_plane import default_screw_grasp_plan
from screw_config import DEFAULT_CONFIG
from screw_control import (
    CAPTURE_GATE_STEP,
    CONTINUOUS_GRASP_STRATEGY_VERSION,
    DIRECT_ALIGN_STEPS,
    FINGERS,
    PREINSERT_GATE_STEP,
    RELEASED_TIP_CENTER_DISTANCE_M,
    SCRIPTED_PREFIX_STEPS,
    ScrewExpert,
    TaskPhase,
    THREAD_GATE_STEP,
    THREAD_REGRASP_GRASP_HEIGHT_FROM_TIP_M,
    THREAD_REGRASP_STEP,
)

from .contracts import ACTION_DIM


REGRASP_PHASES = (
    int(TaskPhase.LR_OPEN),
    int(TaskPhase.LR_RESET),
    int(TaskPhase.LR_CLOSE),
    int(TaskPhase.UD_OPEN),
    int(TaskPhase.UD_RESET),
    int(TaskPhase.UD_CLOSE),
)

def residual_action_mask(phase, engaged):
    """Return the six executable residual dimensions for each task phase."""

    torch = _torch()
    phase = torch.as_tensor(phase)
    engaged = torch.as_tensor(
        engaged, device=phase.device, dtype=torch.bool
    )
    regrasp = torch.zeros_like(engaged)
    for value in REGRASP_PHASES:
        regrasp |= phase == value
    waiting_for_measured_thread = (
        (phase == int(TaskPhase.ENGAGE)) & ~engaged
    )
    active = ~regrasp & ~waiting_for_measured_thread
    axial = active & ~engaged & (
        (phase == int(TaskPhase.ALIGN))
        | (phase == int(TaskPhase.INSERT))
        | (phase == int(TaskPhase.ENGAGE))
    )
    # Once the shaft is inside the bore, its orientation is hard-projected to
    # the certified socket-axis anchor until measured thread engagement.
    # Advertising tilt residuals here would make TQC assign probability and
    # replay credit to actions that the safety layer always discards.
    tilt = active & ~(
        (phase == int(TaskPhase.INSERT)) & ~engaged
    )
    twist = active & engaged
    return torch.stack(
        (active, active, axial, tilt, tilt, twist), dim=-1
    )


def _torch():
    import torch

    return torch


def _normalize_quaternion(quaternion):
    torch = _torch()
    return quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-9)


def quaternion_slerp(first, second, fraction):
    """Shortest-path batched SLERP for ``wxyz`` quaternions."""

    torch = _torch()
    q0 = _normalize_quaternion(first)
    q1 = _normalize_quaternion(second)
    value = torch.as_tensor(
        fraction, device=q0.device, dtype=q0.dtype
    )
    while value.ndim < q0.ndim:
        value = value.unsqueeze(-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(0.0, 1.0)
    theta = torch.acos(dot)
    sine = torch.sin(theta)
    spherical = (
        torch.sin((1.0 - value) * theta)
        / sine.clamp_min(1.0e-7)
        * q0
        + torch.sin(value * theta)
        / sine.clamp_min(1.0e-7)
        * q1
    )
    linear = _normalize_quaternion(
        (1.0 - value) * q0 + value * q1
    )
    return _normalize_quaternion(
        torch.where(dot > 0.9995, linear, spherical)
    )


def quaternion_to_matrix(quaternion):
    torch = _torch()
    q = _normalize_quaternion(quaternion)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quaternion(matrix):
    """Numerically stable batched rotation-matrix to ``wxyz`` conversion."""

    torch = _torch()
    m = matrix
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1.0 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2],
                    1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                    1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                    1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
                ),
                dim=-1,
            ),
            min=0.0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack(
                (
                    q_abs[..., 0].square(),
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    m[..., 2, 1] - m[..., 1, 2],
                    q_abs[..., 1].square(),
                    m[..., 1, 0] + m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] + m[..., 0, 1],
                    q_abs[..., 2].square(),
                    m[..., 2, 1] + m[..., 1, 2],
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 2, 0] + m[..., 0, 2],
                    m[..., 2, 1] + m[..., 1, 2],
                    q_abs[..., 3].square(),
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )
    candidates = candidates / (
        2.0 * q_abs[..., :, None].clamp_min(1.0e-6)
    )
    selected = q_abs.argmax(dim=-1)
    gather_index = selected[..., None, None].expand(
        *selected.shape, 1, 4
    )
    return _normalize_quaternion(
        candidates.gather(-2, gather_index).squeeze(-2)
    )


def rotation_exp(rotvec):
    torch = _torch()
    theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    axis = rotvec / theta.clamp_min(1.0e-9)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*rotvec.shape[:-1], 3, 3)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    eye = eye.expand(*rotvec.shape[:-1], 3, 3)
    small = theta[..., 0] < 1.0e-5
    sine_over_theta = torch.where(
        small,
        1.0 - theta[..., 0].square() / 6.0,
        torch.sin(theta[..., 0]) / theta[..., 0],
    )
    one_minus_cos_over_theta2 = torch.where(
        small,
        0.5 - theta[..., 0].square() / 24.0,
        (1.0 - torch.cos(theta[..., 0])) / theta[..., 0].square(),
    )
    raw_skew = skew * theta.unsqueeze(-1)
    return (
        eye
        + sine_over_theta.unsqueeze(-1).unsqueeze(-1) * raw_skew
        + one_minus_cos_over_theta2.unsqueeze(-1).unsqueeze(-1)
        * (raw_skew @ raw_skew)
    )


def rotation_log(matrix):
    torch = _torch()
    cosine = (
        (matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    ).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    vector = torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )
    denominator = 2.0 * torch.sin(theta)
    scale = torch.where(
        theta < 1.0e-5,
        0.5 + theta.square() / 12.0,
        theta / denominator.clamp_min(1.0e-7),
    )
    return vector * scale.unsqueeze(-1)


def rotation_z(angle):
    """Return batched Z-axis rotations for a tensor of angles."""

    torch = _torch()
    angle = torch.as_tensor(angle)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    return torch.stack(
        (
            cosine,
            -sine,
            zero,
            sine,
            cosine,
            zero,
            zero,
            zero,
            one,
        ),
        dim=-1,
    ).reshape(*angle.shape, 3, 3)


@dataclass
class CommandBatch:
    tip: object
    rotation: object
    twist: object
    material: object
    distance: object
    phase: object
    engaged: object

    def clone(self) -> "CommandBatch":
        return CommandBatch(
            tip=self.tip.clone(),
            rotation=self.rotation.clone(),
            twist=self.twist.clone(),
            material=self.material.clone(),
            distance=self.distance.clone(),
            phase=self.phase.clone(),
            engaged=self.engaged.clone(),
        )

    def state_dict(self):
        """Return a detached snapshot without changing zero-copy storage."""

        return {
            name: getattr(self, name).detach().clone()
            for name in (
                "tip",
                "rotation",
                "twist",
                "material",
                "distance",
                "phase",
                "engaged",
            )
        }

    def load_state_dict(self, state) -> None:
        """Restore in-place so persistent Torch/Warp aliases remain valid."""

        for name, target in (
            ("tip", self.tip),
            ("rotation", self.rotation),
            ("twist", self.twist),
            ("material", self.material),
            ("distance", self.distance),
            ("phase", self.phase),
            ("engaged", self.engaged),
        ):
            source = state.get(name)
            if source is None:
                raise ValueError(f"command checkpoint is missing {name!r}")
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"command state {name!r} shape mismatch: "
                    f"{tuple(source.shape)} != {tuple(target.shape)}"
                )
            target.copy_(
                source.to(device=target.device, dtype=target.dtype)
            )

    def fingertip_targets(
        self,
        grasp_height_m=DEFAULT_CONFIG.screw_grasp_height_from_tip_m,
    ):
        torch = _torch()
        height = torch.as_tensor(
            grasp_height_m,
            device=self.distance.device,
            dtype=self.distance.dtype,
        )
        if height.ndim == 0:
            height = height.expand_as(self.distance)
        elif tuple(height.shape) == (self.distance.shape[0],):
            height = height[:, None].expand_as(self.distance)
        elif tuple(height.shape) != tuple(self.distance.shape):
            raise ValueError(
                "grasp height must be scalar, [world], or [world, finger]"
            )
        local = torch.stack(
            (
                self.distance * torch.cos(self.material),
                self.distance * torch.sin(self.material),
                height,
            ),
            dim=-1,
        )
        return self.tip[:, None, :] + torch.einsum(
            "bij,bfj->bfi", self.rotation, local
        )


class BatchedSupervisor:
    """Fixed-45-degree continuous-grasp expert for every socket azimuth."""

    _trajectory_cache = {}

    def __init__(self, *, device: str, curriculum=None) -> None:
        torch = _torch()
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.config = DEFAULT_CONFIG
        self.curriculum = curriculum
        self.restrict_screw_spawn_to_opening_half_disk = (
            True
            if curriculum is None
            else bool(
                curriculum.restrict_screw_spawn_to_opening_half_disk
            )
        )
        self.grasp_plan = default_screw_grasp_plan()
        self.contact_distance_m = float(
            self.config.continuous_grasp_distance_m
        )
        self.pregrasp_distance_m = float(self.grasp_plan.pregrasp_radius_m)
        self.grasp_plane_yaw_rad = float(self.grasp_plan.plane.yaw_rad)
        self.released_distance_m = float(
            RELEASED_TIP_CENTER_DISTANCE_M
        )
        self.yaw_strategy_version = CONTINUOUS_GRASP_STRATEGY_VERSION
        self.fixed_socket_angle_deg = (
            float(curriculum.fixed_socket_angle_deg)
            if curriculum is not None
            else float(self.config.socket_table_angle_deg)
        )
        # Keep these aliases for diagnostics that report an observed range.
        # The production table contains exactly one fixed-angle trajectory.
        self.angle_min_deg = self.fixed_socket_angle_deg
        self.angle_max_deg = self.fixed_socket_angle_deg
        self.angle_step_deg = 1.0
        self.angle_grid_deg = np.asarray(
            (self.fixed_socket_angle_deg,), dtype=np.float64
        )
        cache_key = (
            tuple(self.angle_grid_deg.tolist()),
            self.contact_distance_m,
            self.pregrasp_distance_m,
            self.released_distance_m,
            self.grasp_plane_yaw_rad,
            self.yaw_strategy_version,
            self.config.screw_shaft_length_m,
            self.config.screw_head_height_m,
            self.config.socket_hole_depth_m,
            self.config.thread_engagement_depth_m,
            self.config.socket_mouth_orbit_radius_m,
            self.config.socket_mouth_position_m[2],
            self.config.scripted_grasp_plane_world_z_m,
            self.config.screw_grasp_height_from_tip_m,
            self.config.capture_depth_m,
            self.config.preinsert_standoff_m,
        )
        arrays = self._trajectory_cache.get(cache_key)
        if arrays is None:
            arrays = self._build_angle_trajectories()
            self._trajectory_cache[cache_key] = arrays
        for name, value in arrays.items():
            dtype = (
                torch.long
                if name == "phase"
                else torch.bool
                if name == "engaged"
                else self.dtype
            )
            setattr(
                self,
                name,
                torch.as_tensor(value, device=self.device, dtype=dtype),
            )
        self.length = int(self.tip_base.shape[1])
        boundary_expert = self._expert(
            (0.0, 0.0), float(self.angle_grid_deg[0])
        )
        self.diffusion_handoff_step = int(
            boundary_expert.diffusion_handoff_step
        )
        self.preinsert_gate_step = int(
            boundary_expert.preinsert_gate_step
        )
        self.capture_gate_step = int(boundary_expert.capture_gate_step)
        self.thread_gate_step = int(boundary_expert.thread_gate_step)
        self.thread_regrasp_step = int(
            boundary_expert.thread_regrasp_step
        )
        expected_boundaries = (
            PREINSERT_GATE_STEP,
            CAPTURE_GATE_STEP,
            THREAD_GATE_STEP,
            THREAD_REGRASP_STEP,
        )
        actual_boundaries = (
            self.preinsert_gate_step,
            self.capture_gate_step,
            self.thread_gate_step,
            self.thread_regrasp_step,
        )
        if actual_boundaries != expected_boundaries:
            raise RuntimeError(
                "continuous-grasp boundary contract drifted: "
                f"{actual_boundaries} != {expected_boundaries}"
            )
        if self.thread_regrasp_step - self.thread_gate_step != 1:
            raise RuntimeError(
                "thread engagement must hand directly to H112 regrasp"
            )
        if not (
            int(self.phase[self.diffusion_handoff_step])
            == int(TaskPhase.LIFT)
            and bool(
                (
                    self.phase[
                        self.diffusion_handoff_step
                        + 1 : self.preinsert_gate_step
                        + 1
                    ]
                    == int(TaskPhase.ALIGN)
                ).all()
            )
            and bool(
                (
                    self.phase[
                        self.preinsert_gate_step
                        + 1 : self.capture_gate_step
                        + 1
                    ]
                    == int(TaskPhase.INSERT)
                ).all()
            )
            and bool(
                (
                    self.phase[
                        self.capture_gate_step
                        + 1 : self.thread_gate_step
                        + 1
                    ]
                    == int(TaskPhase.INSERT)
                ).all()
            )
            and bool(
                (
                    self.phase[
                        self.thread_gate_step
                        + 1 : self.thread_regrasp_step
                        + 1
                    ]
                    == int(TaskPhase.ENGAGE)
                ).all()
            )
            and int(self.phase[self.thread_regrasp_step + 1])
            == int(TaskPhase.UD_CLOSE)
        ):
            raise RuntimeError(
                "continuous-grasp phase schedule no longer matches its gates"
            )
        prethread_distance = self.distance_table[
            :,
            SCRIPTED_PREFIX_STEPS : self.thread_gate_step + 1,
            :,
        ]
        if not bool(torch.isclose(
            prethread_distance,
            torch.full_like(prethread_distance, self.contact_distance_m),
            atol=2.0e-7,
            rtol=0.0,
        ).all()):
            raise RuntimeError(
                "all four fingertips must retain the pickup preload"
            )
        prethread_material = self.material_table[
            :,
            SCRIPTED_PREFIX_STEPS : self.thread_gate_step + 1,
            :,
        ]
        material_reference = prethread_material[:, :1, :]
        if not bool(
            torch.isclose(
                prethread_material,
                material_reference,
                atol=2.0e-7,
                rtol=0.0,
            ).all()
        ):
            raise RuntimeError(
                "fingertip material angles must remain fixed from pickup "
                "through the measured thread gate"
            )
        default_angle = torch.full(
            (1,),
            self.fixed_socket_angle_deg,
            device=self.device,
            dtype=self.dtype,
        )
        default_mouth = torch.as_tensor(
            (self.config.socket_mouth_xy_m,),
            device=self.device,
            dtype=self.dtype,
        )
        frame, axis, mouth, _aligned = self.socket_geometry(
            default_angle, default_mouth
        )
        # Fixed-45-degree references for diagnostics and reset geometry.
        self.socket_frame = frame[0]
        self.socket_axis = axis[0]
        self.socket_mouth = mouth[0]

    def grasp_height_at(self, step):
        """Return H110 through thread, then direct H112 for regrasp."""

        torch = _torch()
        step = torch.as_tensor(
            step, device=self.device, dtype=self.dtype
        )
        pickup_height = float(
            self.config.screw_grasp_height_from_tip_m
        )
        return torch.where(
            step > float(self.thread_regrasp_step),
            torch.full_like(step, THREAD_REGRASP_GRASP_HEIGHT_FROM_TIP_M),
            torch.full_like(step, pickup_height),
        )

    @staticmethod
    def _axis_for_angle(angle_deg: float) -> np.ndarray:
        angle = math.radians(float(angle_deg))
        return np.array(
            (math.cos(angle), 0.0, -math.sin(angle)), dtype=np.float64
        )

    def _expert(
        self, spawn: Tuple[float, float], angle_deg: float = 45.0
    ) -> ScrewExpert:
        c = self.config
        return ScrewExpert(
            spawn_xy_m=spawn,
            table_top_z_m=c.table_top_z_m,
            # Canonical tables use a central mouth and +X inward azimuth.
            # Runtime translates/rotates the complete command equivariantly.
            socket_mouth_world_m=(
                0.0,
                0.0,
                c.socket_mouth_position_m[2],
            ),
            socket_axis_inward_world=self._axis_for_angle(angle_deg),
            contact_distance_m=self.contact_distance_m,
            pregrasp_distance_m=self.pregrasp_distance_m,
            released_distance_m=self.released_distance_m,
            grasp_plane_yaw_rad=self.grasp_plane_yaw_rad,
            engage_depth_m=c.thread_engagement_depth_m,
            capture_depth_m=c.capture_depth_m,
            final_depth_m=c.target_insertion_depth_m,
            pitch_m=c.thread_pitch_m,
            target_turns=c.target_turns,
            grasp_height_from_tip_m=c.screw_grasp_height_from_tip_m,
            scripted_grasp_plane_world_z_m=(
                c.scripted_grasp_plane_world_z_m
            ),
            # Legacy compatibility only: the direct controller no longer
            # visits a centre-transfer waypoint.
            scripted_center_xy_m=(
                -c.socket_mouth_orbit_radius_m,
                0.0,
            ),
            preinsert_standoff_m=c.preinsert_standoff_m,
        )

    @staticmethod
    def _roll(expert: ScrewExpert):
        commands = [expert.reset()]
        for _ in range(expert.total_steps):
            commands.append(expert.advance())
        return commands

    def _build_affine_trajectory(self, angle_deg: float):
        base_expert = self._expert((0.0, 0.0), angle_deg)
        base = self._roll(base_expert)
        x_basis = self._roll(self._expert((1.0, 0.0), angle_deg))
        y_basis = self._roll(self._expert((0.0, 1.0), angle_deg))
        tips = np.stack([command.tip_world for command in base])
        x_tip = np.stack([command.tip_world for command in x_basis]) - tips
        y_tip = np.stack([command.tip_world for command in y_basis]) - tips
        rotations = np.stack([command.rotation_world for command in base])
        twist = np.asarray([command.twist_rad for command in base])
        material = np.asarray(
            [[command.material_angles_rad[f] for f in FINGERS] for command in base]
        )
        distance = np.asarray(
            [[command.tip_center_distances_m[f] for f in FINGERS] for command in base]
        )
        phase = np.asarray([int(command.phase) for command in base], dtype=np.int64)
        engaged = np.asarray([command.engaged for command in base], dtype=np.bool_)
        actions = []
        for index in range(len(base) - 1):
            actions.append(base_expert.command_action(base[index], base[index + 1]))
        actions.append(np.zeros(ACTION_DIM, dtype=np.float32))
        action_coeff = np.zeros((len(base), 2, ACTION_DIM), dtype=np.float32)
        socket_frame = np.asarray(base_expert.socket_frame_world)
        for index in range(len(base) - 1):
            action_coeff[index, 0, :3] = (
                socket_frame.T @ (x_tip[index + 1] - x_tip[index])
            )
            action_coeff[index, 1, :3] = (
                socket_frame.T @ (y_tip[index + 1] - y_tip[index])
            )
        return {
            "tip_base": tips.astype(np.float32),
            "tip_coeff": np.stack((x_tip, y_tip), axis=1).astype(np.float32),
            "rotation_table": rotations.astype(np.float32),
            "twist_table": twist.astype(np.float32),
            "material_table": material.astype(np.float32),
            "distance_table": distance.astype(np.float32),
            "phase": phase,
            "engaged": engaged,
            "expert_action": np.asarray(actions, dtype=np.float32),
            "expert_action_coeff": action_coeff,
        }

    def _build_angle_trajectories(self):
        per_angle = [
            self._build_affine_trajectory(float(angle))
            for angle in self.angle_grid_deg
        ]
        reference_phase = per_angle[0]["phase"]
        reference_engaged = per_angle[0]["engaged"]
        for value in per_angle[1:]:
            if not np.array_equal(value["phase"], reference_phase):
                raise RuntimeError("angle expert tables have different phase schedules")
            if not np.array_equal(value["engaged"], reference_engaged):
                raise RuntimeError("angle expert tables have different thread schedules")
        return {
            name: (
                reference_phase
                if name == "phase"
                else reference_engaged
                if name == "engaged"
                else np.stack([value[name] for value in per_angle], axis=0)
            )
            for name in per_angle[0]
        }

    def _angle_index(self, angle_deg):
        torch = _torch()
        angle = torch.as_tensor(
            angle_deg, device=self.device, dtype=self.dtype
        )
        # Production has one expert table.  Keep the angle argument in the
        # low-level API so geometry diagnostics can still be explicit without
        # reintroducing reset-time angle randomization.
        return torch.zeros_like(angle, dtype=torch.long)

    def _yaw_transform(self, socket_mouth_xy):
        """Return yaw, cosine/sine, and Rz for the 20 mm socket orbit."""

        torch = _torch()
        mouth_xy = torch.as_tensor(
            socket_mouth_xy, device=self.device, dtype=self.dtype
        )
        radius = torch.linalg.vector_norm(
            mouth_xy, dim=-1, keepdim=True
        ).clamp_min(1.0e-9)
        unit = mouth_xy / radius
        cosine = unit[:, 0]
        sine = unit[:, 1]
        zero = torch.zeros_like(cosine)
        one = torch.ones_like(cosine)
        rotation = torch.stack(
            (
                cosine,
                -sine,
                zero,
                sine,
                cosine,
                zero,
                zero,
                zero,
                one,
            ),
            dim=-1,
        ).reshape(-1, 3, 3)
        return torch.atan2(sine, cosine), cosine, sine, rotation

    def yaw_strategy(self, socket_mouth_xy):
        """Return yaw diagnostics and the geometric world-Z rotation.

        Quadrant and residual values remain available for monitoring only;
        they never change physical finger roles, material angles, or pose.
        """

        torch = _torch()
        yaw, cosine, sine, rotation = self._yaw_transform(socket_mouth_xy)
        quarter_turn = 0.5 * torch.pi
        scaled = yaw / quarter_turn
        # atan2/float32 can represent the negative-X axis one ulp below 2.
        # Snap only numerical quarter-turn boundaries; 89.9 degrees remains
        # in the preceding quadrant as required by the mathematical floor.
        nearest = torch.round(scaled)
        scaled = torch.where(
            (scaled - nearest).abs() <= 1.0e-6,
            nearest,
            scaled,
        )
        quadrant = torch.floor(scaled).to(torch.long)
        residual = yaw - quadrant.to(self.dtype) * quarter_turn
        residual = residual.clamp(0.0, float(0.5 * math.pi))
        return yaw, quadrant, residual, cosine, sine, rotation

    def _canonical_spawn(self, spawn_xy, socket_mouth_xy):
        torch = _torch()
        spawn = torch.as_tensor(
            spawn_xy, device=self.device, dtype=self.dtype
        )
        mouth = torch.as_tensor(
            socket_mouth_xy, device=self.device, dtype=self.dtype
        )
        _yaw, cosine, sine, _rotation = self._yaw_transform(mouth)
        delta = spawn - mouth
        return torch.stack(
            (
                cosine * delta[:, 0] + sine * delta[:, 1],
                -sine * delta[:, 0] + cosine * delta[:, 1],
            ),
            dim=-1,
        )

    def primary_hold_mask(self, socket_mouth_xy):
        """Return the fixed physical L/R holding mask in ``[U,D,L,R]`` order."""

        torch = _torch()
        mouth = torch.as_tensor(
            socket_mouth_xy, device=self.device, dtype=self.dtype
        )
        if mouth.ndim < 1 or mouth.shape[-1] != 2:
            raise ValueError("socket_mouth_xy must end in a two-vector")
        mask = torch.zeros(
            (*mouth.shape[:-1], len(FINGERS)),
            device=self.device,
            dtype=torch.bool,
        )
        mask[..., 2:] = True
        return mask

    def _direct_rotation(self, step, aligned, twist):
        """SLERP identity directly to the actual-yaw aligned socket frame."""

        torch = _torch()
        progress = (
            step.to(self.dtype) - float(self.diffusion_handoff_step)
        ) / float(DIRECT_ALIGN_STEPS)
        progress = progress.clamp(0.0, 1.0)
        blend = progress**3 * (
            10.0 - 15.0 * progress + 6.0 * progress.square()
        )
        identity = torch.zeros(
            (*step.shape, 4), device=self.device, dtype=self.dtype
        )
        identity[..., 0] = 1.0
        aligned_quaternion = matrix_to_quaternion(aligned)
        rotation = quaternion_to_matrix(
            quaternion_slerp(identity, aligned_quaternion, blend)
        )
        return rotation @ rotation_z(-twist)

    def at(
        self,
        step,
        spawn_xy,
        angle_deg=None,
        socket_mouth_xy=None,
    ) -> CommandBatch:
        torch = _torch()
        step = torch.as_tensor(step, device=self.device, dtype=torch.long)
        spawn_xy = torch.as_tensor(
            spawn_xy, device=self.device, dtype=self.dtype
        )
        if angle_deg is None:
            angle_deg = torch.full_like(
                step, self.fixed_socket_angle_deg, dtype=self.dtype
            )
        if socket_mouth_xy is None:
            socket_mouth_xy = torch.as_tensor(
                self.config.socket_mouth_xy_m,
                device=self.device,
                dtype=self.dtype,
            ).expand(spawn_xy.shape[0], 2)
        socket_mouth_xy = torch.as_tensor(
            socket_mouth_xy, device=self.device, dtype=self.dtype
        )
        local_spawn = self._canonical_spawn(
            spawn_xy, socket_mouth_xy
        )
        _yaw, _cosine, _sine, yaw_rotation = self._yaw_transform(
            socket_mouth_xy
        )
        angle_index = self._angle_index(angle_deg)
        index = step.clamp(0, self.length - 1)
        canonical_tip = self.tip_base[angle_index, index] + torch.einsum(
            "bij,bi->bj",
            self.tip_coeff[angle_index, index],
            local_spawn,
        )
        tip = torch.einsum(
            "bij,bj->bi", yaw_rotation, canonical_tip
        )
        tip[:, :2] += socket_mouth_xy
        _frame, _axis, _mouth, aligned = self.socket_geometry(
            angle_deg, socket_mouth_xy
        )
        twist = self.twist_table[angle_index, index]
        rotation = self._direct_rotation(
            index, aligned, twist
        )
        return CommandBatch(
            tip=tip,
            rotation=rotation,
            twist=twist,
            material=self.material_table[angle_index, index],
            distance=self.distance_table[angle_index, index],
            phase=self.phase[index],
            engaged=self.engaged[index],
        )

    def action_at(
        self,
        step,
        spawn_xy,
        angle_deg=None,
        socket_mouth_xy=None,
    ):
        torch = _torch()
        step = torch.as_tensor(step, device=self.device, dtype=torch.long)
        spawn_xy = torch.as_tensor(
            spawn_xy, device=self.device, dtype=self.dtype
        )
        if angle_deg is None:
            angle_deg = torch.full_like(
                step, self.fixed_socket_angle_deg, dtype=self.dtype
            )
        if socket_mouth_xy is None:
            socket_mouth_xy = torch.as_tensor(
                self.config.socket_mouth_xy_m,
                device=self.device,
                dtype=self.dtype,
            ).expand(spawn_xy.shape[0], 2)
        socket_mouth_xy = torch.as_tensor(
            socket_mouth_xy, device=self.device, dtype=self.dtype
        )
        index = step.clamp(0, self.length - 1)
        following_index = (index + 1).clamp(max=self.length - 1)
        current = self.at(
            index,
            spawn_xy,
            angle_deg,
            socket_mouth_xy,
        )
        following = self.at(
            following_index,
            spawn_xy,
            angle_deg,
            socket_mouth_xy,
        )
        socket_frame, _axis, _mouth, _aligned = self.socket_geometry(
            angle_deg, socket_mouth_xy
        )
        delta_position = torch.einsum(
            "bji,bj->bi",
            socket_frame,
            following.tip - current.tip,
        )
        world_delta_rotation = (
            following.rotation @ current.rotation.transpose(-1, -2)
        )
        socket_delta_rotation = (
            socket_frame.transpose(-1, -2)
            @ world_delta_rotation
            @ socket_frame
        )
        delta_rotation = rotation_log(socket_delta_rotation)
        result = torch.cat(
            (
                delta_position,
                delta_rotation,
                following.material - current.material,
                following.distance - current.distance,
            ),
            dim=-1,
        )
        terminal = index >= self.length - 1
        return torch.where(
            terminal[:, None], torch.zeros_like(result), result
        )

    def socket_geometry(self, angle_deg, socket_mouth_xy=None):
        """Return per-world frame, inward axis, mouth, and aligned screw pose."""

        torch = _torch()
        angle = torch.deg2rad(
            torch.as_tensor(
                angle_deg, device=self.device, dtype=self.dtype
            )
        )
        sine = torch.sin(angle)
        cosine = torch.cos(angle)
        zero = torch.zeros_like(angle)
        one = torch.ones_like(angle)
        canonical_axis = torch.stack((cosine, zero, -sine), dim=-1)
        canonical_x = torch.stack((-sine, zero, -cosine), dim=-1)
        frame_y = torch.stack((zero, one, zero), dim=-1)
        canonical_frame = torch.stack(
            (canonical_x, frame_y, canonical_axis), dim=-1
        )
        if socket_mouth_xy is None:
            socket_mouth_xy = torch.as_tensor(
                self.config.socket_mouth_xy_m,
                device=self.device,
                dtype=self.dtype,
            ).expand(angle.shape[0], 2)
        socket_mouth_xy = torch.as_tensor(
            socket_mouth_xy, device=self.device, dtype=self.dtype
        )
        _yaw, _cosine, _sine, yaw_rotation = self._yaw_transform(
            socket_mouth_xy
        )
        frame = yaw_rotation @ canonical_frame
        # ``frame[:, :, 2]`` is a strided column view.  The native wrench
        # kernel keeps a persistent zero-copy Torch/Warp alias, which requires
        # a contiguous [world, 3] tensor on CUDA.
        axis = frame[:, :, 2].contiguous()
        outward = -axis
        reference_y = torch.zeros_like(outward)
        reference_y[:, 1] = 1.0
        reference_x = torch.zeros_like(outward)
        reference_x[:, 0] = 1.0
        use_x = (
            (reference_y * outward).sum(dim=-1).abs() > 0.95
        )
        reference = torch.where(
            use_x[:, None], reference_x, reference_y
        )
        aligned_x = torch.linalg.cross(
            reference, outward, dim=-1
        )
        aligned_x = aligned_x / torch.linalg.vector_norm(
            aligned_x, dim=-1, keepdim=True
        ).clamp_min(1.0e-9)
        aligned_y = torch.linalg.cross(
            outward, aligned_x, dim=-1
        )
        aligned_y = aligned_y / torch.linalg.vector_norm(
            aligned_y, dim=-1, keepdim=True
        ).clamp_min(1.0e-9)
        aligned = torch.stack((aligned_x, aligned_y, outward), dim=-1)
        mouth_z = torch.full(
            (angle.shape[0], 1),
            float(self.config.socket_mouth_position_m[2]),
            device=self.device,
            dtype=self.dtype,
        )
        # Native Warp keeps a persistent zero-copy alias to this tensor.
        mouth = torch.cat((socket_mouth_xy, mouth_z), dim=-1).contiguous()
        return frame, axis, mouth, aligned

    def socket_mocap_pose(self, angle_deg, socket_mouth_xy=None):
        frame, axis, mouth, _aligned = self.socket_geometry(
            angle_deg, socket_mouth_xy
        )
        center = mouth + 0.5 * self.config.socket_outer_length_m * axis
        return center, matrix_to_quaternion(frame)

    def curriculum_parameters(self, progress: float) -> Tuple[float, float]:
        if self.curriculum is None or not self.curriculum.enabled:
            return (
                self.config.spawn_radius_m,
                self.fixed_socket_angle_deg,
            )
        value = min(max(float(progress), 0.0), 1.0)
        start = self.curriculum.warmup_fraction
        end = self.curriculum.full_difficulty_fraction
        blend = min(max((value - start) / max(end - start, 1.0e-9), 0.0), 1.0)
        spawn_radius = (
            (1.0 - blend) * self.curriculum.start_spawn_radius_m
            + blend * self.curriculum.final_spawn_radius_m
        )
        return (
            float(spawn_radius),
            self.fixed_socket_angle_deg,
        )

    def fixed_angle_batch(self, count: int):
        """Return the formal task angle without consuming RNG state."""

        torch = _torch()
        return torch.full(
            (count,),
            self.fixed_socket_angle_deg,
            device=self.device,
            dtype=self.dtype,
        )

    def sample_socket_mouth(self, count: int, *, generator=None):
        """Uniform azimuth samples on the fixed 20 mm socket orbit."""

        torch = _torch()
        azimuth = 2.0 * torch.pi * torch.rand(
            count, device=self.device, generator=generator
        )
        radius = self.config.socket_mouth_orbit_radius_m
        return radius * torch.stack(
            (torch.cos(azimuth), torch.sin(azimuth)), dim=-1
        )

    def _spawn_overlaps_socket_obb(
        self,
        proposal,
        mouth,
        inward,
        tangent,
        cosine,
        sine,
    ):
        """Conservative upright-screw overlap against the rotated socket OBB."""

        centre = (
            mouth
            + 0.5
            * self.config.socket_outer_length_m
            * cosine[:, None]
            * inward
        )
        relative = proposal - centre
        margin = (
            self.config.screw_head_radius_m
            + self.config.spawn_clearance_m
        )
        radial_extent = (
            0.5 * self.config.socket_outer_length_m * cosine
            + 0.5 * self.config.socket_outer_width_m * sine
            + margin
        )
        tangent_extent = (
            0.5 * self.config.socket_outer_height_m + margin
        )
        radial_coordinate = (relative * inward).sum(dim=-1).abs()
        tangent_coordinate = (relative * tangent).sum(dim=-1).abs()
        return (
            (radial_coordinate <= radial_extent)
            & (tangent_coordinate <= tangent_extent)
        )

    def spawn_overlaps_socket(
        self,
        spawn_xy,
        *,
        socket_mouth_xy=None,
        socket_angle_deg=None,
    ):
        """Return the exact conservative socket-exclusion mask used at reset."""

        torch = _torch()
        proposal = torch.as_tensor(
            spawn_xy, device=self.device, dtype=self.dtype
        )
        if proposal.ndim != 2 or proposal.shape[1] != 2:
            raise ValueError("spawn_xy must be [count,2]")
        count = int(proposal.shape[0])
        if socket_mouth_xy is None:
            socket_mouth_xy = torch.as_tensor(
                self.config.socket_mouth_xy_m,
                device=self.device,
                dtype=self.dtype,
            ).expand(count, 2)
        mouth = torch.as_tensor(
            socket_mouth_xy, device=self.device, dtype=self.dtype
        )
        if tuple(mouth.shape) != (count, 2):
            raise ValueError("socket_mouth_xy must be [count,2]")
        if socket_angle_deg is None:
            socket_angle_deg = torch.full(
                (count,),
                self.fixed_socket_angle_deg,
                device=self.device,
                dtype=self.dtype,
            )
        elevation = torch.deg2rad(
            torch.as_tensor(
                socket_angle_deg,
                device=self.device,
                dtype=self.dtype,
            )
        )
        if tuple(elevation.shape) != (count,):
            raise ValueError("socket_angle_deg must be [count]")
        mouth_radius = torch.linalg.vector_norm(
            mouth, dim=-1, keepdim=True
        )
        inward = mouth / mouth_radius.clamp_min(1.0e-9)
        tangent = torch.stack((-inward[:, 1], inward[:, 0]), dim=-1)
        cosine = torch.cos(elevation).clamp_min(0.0)
        sine = torch.sin(elevation).clamp_min(0.0)
        return self._spawn_overlaps_socket_obb(
            proposal,
            mouth,
            inward,
            tangent,
            cosine,
            sine,
        )

    def sample_spawn(
        self,
        count: int,
        *,
        socket_mouth_xy=None,
        socket_angle_deg=None,
        radius_m=None,
        generator=None,
    ):
        """Sample the configured workspace region with socket OBB rejection.

        The default is the original dynamic opening-facing half-disc.  When
        the curriculum switch is disabled, the proposal distribution is the
        complete workspace disc; radius, socket clearance, IK qualification,
        and all later reset gates remain unchanged.
        """

        torch = _torch()
        radius_limit = (
            self.config.spawn_radius_m
            if radius_m is None
            else float(radius_m)
        )
        if socket_mouth_xy is None:
            socket_mouth_xy = torch.as_tensor(
                self.config.socket_mouth_xy_m,
                device=self.device,
                dtype=self.dtype,
            ).expand(count, 2)
        mouth = torch.as_tensor(
            socket_mouth_xy, device=self.device, dtype=self.dtype
        )
        if socket_angle_deg is None:
            socket_angle_deg = torch.full(
                (count,), 45.0, device=self.device, dtype=self.dtype
            )
        elevation = torch.deg2rad(
            torch.as_tensor(
                socket_angle_deg,
                device=self.device,
                dtype=self.dtype,
            )
        )
        mouth_radius = torch.linalg.vector_norm(
            mouth, dim=-1, keepdim=True
        )
        inward = mouth / mouth_radius.clamp_min(1.0e-9)
        tangent = torch.stack((-inward[:, 1], inward[:, 0]), dim=-1)
        opening_angle = torch.atan2(-inward[:, 1], -inward[:, 0])
        if not math.isfinite(radius_limit) or radius_limit <= 0.0:
            raise ValueError(
                f"spawn radius must be finite and positive, got {radius_limit!r}"
            )

        # In opening-frame coordinates x points away from the socket and y is
        # tangent to its orbit.  The inflated projected OBB rejects
        # x <= radial_escape and |y| <= tangent_escape.  If the configured
        # disc radius cannot cross either boundary, its collision-free support
        # has zero area and no rejection-attempt budget can make sampling
        # succeed.
        cosine = torch.cos(elevation).clamp_min(0.0)
        sine = torch.sin(elevation).clamp_min(0.0)
        margin = (
            self.config.screw_head_radius_m
            + self.config.spawn_clearance_m
        )
        radial_escape = (
            0.5 * self.config.socket_outer_width_m * sine
            + margin
            - mouth_radius[:, 0]
        ).clamp_min(0.0)
        tangent_escape = torch.full_like(
            radial_escape,
            0.5 * self.config.socket_outer_height_m + margin,
        )
        support_boundary = torch.minimum(
            radial_escape, tangent_escape
        )
        unsupported = radius_limit <= support_boundary + 1.0e-9
        if bool(unsupported.any()):
            required = float(support_boundary[unsupported].max().item())
            affected = int(unsupported.sum().item())
            raise ValueError(
                f"spawn radius {radius_limit:.9f} m has no positive-area "
                f"collision-free support for {affected}/{count} slot(s); "
                "the rotated inflated-socket geometry requires radius > "
                f"{required:.9f} m, so increasing the rejection-attempt "
                "budget cannot help"
            )

        result = torch.empty((count, 2), device=self.device, dtype=self.dtype)
        pending = torch.ones(count, device=self.device, dtype=torch.bool)
        max_attempts = (
            128
            if self.restrict_screw_spawn_to_opening_half_disk
            else 256
        )
        for _ in range(max_attempts):
            slots = torch.nonzero(pending, as_tuple=False).flatten()
            needed = int(slots.numel())
            if needed == 0:
                return result
            radius = radius_limit * torch.sqrt(
                torch.rand(needed, device=self.device, generator=generator)
            )
            if self.restrict_screw_spawn_to_opening_half_disk:
                # Keep the legacy branch byte-for-byte equivalent so seeded
                # half-disc runs retain their original proposal sequence.
                angle = (
                    opening_angle[slots]
                    - 0.5 * torch.pi
                    + torch.pi * torch.rand(
                        needed, device=self.device, generator=generator
                    )
                )
            else:
                angle = (
                    opening_angle[slots]
                    - torch.pi
                    + 2.0
                    * torch.pi
                    * torch.rand(
                        needed, device=self.device, generator=generator
                    )
                )
            proposal = torch.stack(
                (radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1
            )

            # Project the rotated 80x80x100 mm box into its radial/tangential
            # XY axes and inflate it by the upright screw-head clearance.
            invalid = self._spawn_overlaps_socket_obb(
                proposal,
                mouth[slots],
                inward[slots],
                tangent[slots],
                cosine[slots],
                sine[slots],
            )
            accepted_slots = slots[~invalid]
            result[accepted_slots] = proposal[~invalid]
            pending[accepted_slots] = False
        region = (
            "opening-facing half-disc"
            if self.restrict_screw_spawn_to_opening_half_disk
            else "full workspace disc"
        )
        raise RuntimeError(
            "GPU spawn rejection sampler exhausted "
            f"{max_attempts} attempts in {region} mode"
        )


class TaskSpaceSafety:
    """Vectorized Diffusion and residual projections around the supervisor."""

    def __init__(self, supervisor: BatchedSupervisor) -> None:
        torch = _torch()
        self.supervisor = supervisor
        self.device = supervisor.device
        self.dtype = supervisor.dtype
        self.action_limits = torch.as_tensor(
            (
                0.002, 0.002, 0.003,
                math.radians(2), math.radians(2), math.radians(4),
                *([math.radians(6)] * 4),
                *([0.004] * 4),
            ),
            device=self.device,
            dtype=self.dtype,
        )

    def diffusion(
        self,
        previous: CommandBatch,
        anchor: CommandBatch,
        physical_action,
        *,
        socket_frame=None,
        socket_axis=None,
        socket_mouth=None,
    ) -> CommandBatch:
        torch = _torch()
        action = torch.as_tensor(
            physical_action, device=self.device, dtype=self.dtype
        )
        action = torch.maximum(
            torch.minimum(action, self.action_limits), -self.action_limits
        )
        worlds = action.shape[0]
        if socket_frame is None:
            socket_frame = self.supervisor.socket_frame.expand(worlds, 3, 3)
        if socket_axis is None:
            socket_axis = self.supervisor.socket_axis.expand(worlds, 3)
        if socket_mouth is None:
            socket_mouth = self.supervisor.socket_mouth.expand(worlds, 3)
        delta_world = torch.einsum(
            "bij,bj->bi", socket_frame, action[:, :3]
        )
        tip = previous.tip + delta_world
        tip_error_socket = torch.einsum(
            "bji,bj->bi",
            socket_frame,
            tip - anchor.tip,
        )
        trust_position = torch.as_tensor(
            (0.006, 0.006, 0.006), device=self.device, dtype=self.dtype
        )
        tip = anchor.tip + torch.einsum(
            "bij,bj->bi",
            socket_frame,
            torch.maximum(
                torch.minimum(tip_error_socket, trust_position), -trust_position
            ),
        )

        socket_delta = rotation_exp(action[:, 3:6])
        frame = socket_frame
        world_delta = frame @ socket_delta @ frame.transpose(-1, -2)
        rotation = world_delta @ previous.rotation
        rotation_error = rotation_log(
            anchor.rotation.transpose(-1, -2) @ rotation
        )
        tilt_trust = torch.as_tensor(
            (
                math.radians(5), math.radians(5), math.radians(8)
            ),
            device=self.device,
            dtype=self.dtype,
        )
        correction = torch.maximum(
            torch.minimum(rotation_error, tilt_trust), -tilt_trust
        )
        rotation = anchor.rotation @ rotation_exp(correction)

        material = previous.material + action[:, 6:10]
        material = anchor.material + (
            material - anchor.material
        ).clamp(-math.radians(12), math.radians(12))
        distance = (previous.distance + action[:, 10:14]).clamp(0.0385, 0.061)
        distance = anchor.distance + (
            distance - anchor.distance
        ).clamp(-0.005, 0.005)
        twist = previous.twist + action[:, 5].clamp_min(0.0)
        twist = anchor.twist + (
            twist - anchor.twist
        ).clamp(-math.radians(8), math.radians(8))

        engaged = anchor.engaged
        target_depth = (
            self.supervisor.config.thread_engagement_depth_m
            + self.supervisor.config.thread_pitch_m
            * twist
            / (2.0 * math.pi)
        )
        helical_tip = (
            socket_mouth
            + target_depth[:, None] * socket_axis
        )
        tip = torch.where(engaged[:, None], helical_tip, tip)

        regrasp = torch.zeros_like(anchor.engaged)
        for phase in REGRASP_PHASES:
            regrasp |= anchor.phase == phase
        tip = torch.where(regrasp[:, None], anchor.tip, tip)
        rotation = torch.where(regrasp[:, None, None], anchor.rotation, rotation)
        twist = torch.where(regrasp, anchor.twist, twist)

        # The designated holding pair is always forced to contact distance.
        lr = (
            (anchor.phase == int(TaskPhase.LR_OPEN))
            | (anchor.phase == int(TaskPhase.LR_RESET))
            | (anchor.phase == int(TaskPhase.LR_CLOSE))
        )
        ud = (
            (anchor.phase == int(TaskPhase.UD_OPEN))
            | (anchor.phase == int(TaskPhase.UD_RESET))
            | (anchor.phase == int(TaskPhase.UD_CLOSE))
        )
        # FINGERS order U,D,L,R.
        distance[:, 0:2] = torch.where(
            lr[:, None],
            torch.full_like(
                distance[:, 0:2], self.supervisor.contact_distance_m
            ),
            distance[:, 0:2],
        )
        distance[:, 2:4] = torch.where(
            ud[:, None],
            torch.full_like(
                distance[:, 2:4], self.supervisor.contact_distance_m
            ),
            distance[:, 2:4],
        )
        return CommandBatch(
            tip=tip,
            rotation=rotation,
            twist=twist,
            material=material,
            distance=distance,
            phase=anchor.phase,
            engaged=anchor.engaged,
        )

    def residual(
        self,
        base: CommandBatch,
        normalized_action,
        *,
        previous: CommandBatch | None = None,
        socket_frame=None,
    ) -> CommandBatch:
        torch = _torch()
        action = torch.as_tensor(
            normalized_action, device=self.device, dtype=self.dtype
        ).clamp(-1.0, 1.0)
        action = action * residual_action_mask(
            base.phase, base.engaged
        ).to(self.dtype)
        if socket_frame is None:
            socket_frame = self.supervisor.socket_frame.expand(
                action.shape[0], 3, 3
            )
        result = base.clone()
        regrasp = torch.zeros_like(base.engaged)
        for phase in REGRASP_PHASES:
            regrasp |= base.phase == phase
        active = ~regrasp
        prethread_axial = (
            (base.phase == int(TaskPhase.ALIGN))
            | (base.phase == int(TaskPhase.INSERT))
            | (base.phase == int(TaskPhase.ENGAGE))
        )
        delta_socket = torch.stack(
            (
                0.004 * action[:, 0],
                0.004 * action[:, 1],
                torch.where(
                    prethread_axial,
                    0.005 * action[:, 2],
                    torch.zeros_like(action[:, 2]),
                ),
            ),
            dim=-1,
        )
        requested_twist_delta = math.radians(2.0) * action[:, 5]
        if previous is None:
            minimum_twist_delta = torch.zeros_like(requested_twist_delta)
        else:
            minimum_twist_delta = previous.twist - base.twist
        # A negative residual may slow the supervisor's next screw increment,
        # but may never command a lower absolute twist than the last certified
        # command.  This keeps the virtual thread one-way without forcing the
        # residual to accelerate every baseline step.
        twist_delta = torch.maximum(
            requested_twist_delta, minimum_twist_delta
        )
        twist_delta = torch.where(
            base.engaged, twist_delta, torch.zeros_like(twist_delta)
        )
        delta_socket[:, 2] = torch.where(
            base.engaged,
            self.supervisor.config.thread_pitch_m
            * twist_delta
            / (2.0 * math.pi),
            delta_socket[:, 2],
        )
        delta_world = torch.einsum(
            "bij,bj->bi", socket_frame, delta_socket
        )
        result.tip = torch.where(active[:, None], base.tip + delta_world, base.tip)
        rotvec = torch.stack(
            (
                math.radians(2.0) * action[:, 3],
                math.radians(2.0) * action[:, 4],
                twist_delta,
            ),
            dim=-1,
        )
        frame = socket_frame
        result.rotation = torch.where(
            active[:, None, None],
            (frame @ rotation_exp(rotvec) @ frame.transpose(-1, -2))
            @ base.rotation,
            base.rotation,
        )
        result.twist = torch.where(active, base.twist + twist_delta, base.twist)
        return result


__all__ = [
    "BatchedSupervisor",
    "CommandBatch",
    "REGRASP_PHASES",
    "TaskSpaceSafety",
    "matrix_to_quaternion",
    "quaternion_to_matrix",
    "rotation_exp",
    "rotation_log",
    "residual_action_mask",
]
