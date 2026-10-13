"""Vectorized screw state, virtual thread, observations, and rewards."""

from __future__ import annotations

import math

from screw_config import DEFAULT_CONFIG
from screw_control import TaskPhase

from .contracts import OBS_DIM
from .supervisor import (
    CommandBatch,
    quaternion_to_matrix,
    rotation_log,
)
from .wrench_kernel import NativeWrenchKernel


def _torch():
    import torch

    return torch


def _wrap_pi(value):
    torch = _torch()
    return torch.remainder(value + torch.pi, 2.0 * torch.pi) - torch.pi


class BatchedTaskState:
    """Per-world non-MuJoCo state kept entirely in Torch CUDA tensors."""

    def __init__(self, backend, supervisor, config) -> None:
        torch = _torch()
        self.backend = backend
        self.supervisor = supervisor
        self.hpc_config = config
        self.task = DEFAULT_CONFIG
        self.device = backend.device
        self.dtype = torch.float32
        n = backend.worlds
        self.spawn_xy = torch.zeros((n, 2), device=self.device)
        self.socket_angle_deg = torch.full(
            (n,),
            supervisor.fixed_socket_angle_deg,
            device=self.device,
        )
        self.step_count = torch.zeros(n, device=self.device, dtype=torch.long)
        self.elapsed_steps = torch.zeros(n, device=self.device, dtype=torch.long)
        self.episode_return = torch.zeros(n, device=self.device)
        self.episode_id = torch.zeros(n, device=self.device, dtype=torch.long)
        self.thread_engaged = torch.zeros(n, device=self.device, dtype=torch.bool)
        self.holder_active = torch.ones(n, device=self.device, dtype=torch.bool)
        self.grasp_latched = torch.zeros(n, device=self.device, dtype=torch.bool)
        self.stable_grasp_support = torch.zeros(
            n, device=self.device, dtype=torch.bool
        )
        self.pickup_material_ref = torch.zeros(
            (n, 4), device=self.device, dtype=self.dtype
        )
        self.primary_hold_mask = self._primary_hold_mask(n)
        self.all_four_contact = torch.zeros(
            n, device=self.device, dtype=torch.bool
        )
        self.primary_pair_contact = torch.zeros(
            n, device=self.device, dtype=torch.bool
        )
        self.two_finger_contact = torch.zeros(
            n, device=self.device, dtype=torch.bool
        )
        self.fingertip_contact_force_n = torch.zeros(
            (n, 4), device=self.device, dtype=self.dtype
        )
        self.all_four_contact_stable_steps = torch.zeros(
            n, device=self.device, dtype=torch.long
        )
        self.primary_pair_contact_stable_steps = torch.zeros(
            n, device=self.device, dtype=torch.long
        )
        self.capture_alignment_stable_steps = torch.zeros(
            n, device=self.device, dtype=torch.long
        )
        self.thread_ready_stable_steps = torch.zeros(
            n, device=self.device, dtype=torch.long
        )
        self.shaft_captured = torch.zeros(
            n, device=self.device, dtype=torch.bool
        )
        self.secondary_released = torch.zeros(
            n, device=self.device, dtype=torch.bool
        )
        self.contact_constraint_violation = torch.zeros(
            n, device=self.device, dtype=torch.bool
        )
        self.contact_violation_before_capture = torch.zeros_like(
            self.contact_constraint_violation
        )
        self.contact_violation_before_thread = torch.zeros_like(
            self.contact_constraint_violation
        )
        self.contact_violation_premature_regrasp = torch.zeros_like(
            self.contact_constraint_violation
        )
        self.contact_violation_material_drift = torch.zeros_like(
            self.contact_constraint_violation
        )
        self.before_capture_contact_loss_steps = torch.zeros(
            n, device=self.device, dtype=torch.long
        )
        self.before_thread_two_finger_loss_steps = torch.zeros(
            n, device=self.device, dtype=torch.long
        )
        self.previous_marker_mod = torch.zeros(n, device=self.device)
        self.marker_mod = torch.zeros(n, device=self.device)
        self.twist = torch.zeros(n, device=self.device)
        self.thread_resistance = torch.full(
            (n,), self.task.rotation_resistance_min_nm, device=self.device
        )
        self.axial_preload = torch.full(
            (n,), self.task.axial_preload_n, device=self.device
        )
        self.previous_depth = torch.zeros(n, device=self.device)
        self.previous_twist = torch.zeros(n, device=self.device)
        self.ik_failures = torch.zeros(n, device=self.device, dtype=torch.long)
        self.ik_anchor_fallbacks = torch.zeros(
            n, device=self.device, dtype=torch.long
        )
        self.ik_safety_stops = torch.zeros(n, device=self.device, dtype=torch.long)
        self.diffusion_handoff_hold_steps = torch.zeros(
            n, device=self.device, dtype=torch.long
        )
        self.preinsert_alignment_stable_steps = torch.zeros(
            n, device=self.device, dtype=torch.long
        )
        self.preinsert_feedback_offset_socket = torch.zeros(
            (n, 3), device=self.device, dtype=self.dtype
        )
        (
            self.socket_frame,
            self.socket_axis,
            self.socket_mouth,
            self.aligned_rotation,
        ) = supervisor.socket_geometry(self.socket_angle_deg)
        self.reference_marker = self.aligned_rotation[
            :, :, 0
        ].clone()
        self.desired_command = supervisor.at(
            self.step_count,
            self.spawn_xy,
            self.socket_angle_deg,
            self.socket_mouth[:, :2],
        )
        self.wrench_kernel = NativeWrenchKernel(backend, self)
        self.backend.register_pre_step(self.wrench_kernel.launch)

    def _primary_hold_mask(self, worlds: int):
        """Return the fixed physical L/R pair retained for diagnostics."""

        torch = _torch()
        expected = torch.as_tensor(
            (False, False, True, True),
            device=self.device,
            dtype=torch.bool,
        )
        supplied = getattr(self.supervisor, "primary_hold_mask", None)
        if callable(supplied):
            supplied = supplied(
                torch.zeros(
                    (worlds, 2),
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        if supplied is None:
            mask = expected.expand(worlds, 4).clone()
        else:
            mask = torch.as_tensor(
                supplied, device=self.device, dtype=torch.bool
            )
            if mask.shape == (4,):
                mask = mask.expand(worlds, 4).clone()
            elif mask.shape == (worlds, 4):
                mask = mask.clone()
            else:
                raise ValueError(
                    "supervisor primary_hold_mask must have shape (4,) "
                    f"or ({worlds}, 4), got {tuple(mask.shape)}"
                )
        if not bool((mask == expected).all()):
            raise ValueError(
                "diagnostic primary_hold_mask must identify physical L/R"
            )
        return mask

    def reset(
        self,
        mask,
        spawn_xy,
        socket_angle_deg,
        socket_mouth_xy,
        command: CommandBatch,
    ) -> None:
        torch = _torch()
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        proposed_angle = torch.as_tensor(
            socket_angle_deg,
            device=self.device,
            dtype=self.dtype,
        )
        if bool(
            (
                mask
                & (
                    proposed_angle
                    - self.supervisor.fixed_socket_angle_deg
                ).abs().gt(1.0e-6)
            ).any()
        ):
            raise ValueError(
                "formal task reset requires a fixed 45 degree socket"
            )
        self.spawn_xy[mask] = spawn_xy[mask]
        self.socket_angle_deg[mask] = (
            self.supervisor.fixed_socket_angle_deg
        )
        frame, axis, mouth, aligned = self.supervisor.socket_geometry(
            self.socket_angle_deg,
            socket_mouth_xy,
        )
        self.socket_frame[mask] = frame[mask]
        self.socket_axis[mask] = axis[mask]
        self.socket_mouth[mask] = mouth[mask]
        self.aligned_rotation[mask] = aligned[mask]
        self.reference_marker[mask] = aligned[mask, :, 0]
        self.step_count[mask] = 0
        self.elapsed_steps[mask] = 0
        self.episode_return[mask] = 0.0
        self.episode_id[mask] += 1
        self.thread_engaged[mask] = False
        self.holder_active[mask] = True
        self.grasp_latched[mask] = False
        self.stable_grasp_support[mask] = False
        self.pickup_material_ref[mask] = 0.0
        self.all_four_contact[mask] = False
        self.primary_pair_contact[mask] = False
        self.two_finger_contact[mask] = False
        self.fingertip_contact_force_n[mask] = 0.0
        self.all_four_contact_stable_steps[mask] = 0
        self.primary_pair_contact_stable_steps[mask] = 0
        self.capture_alignment_stable_steps[mask] = 0
        self.thread_ready_stable_steps[mask] = 0
        self.shaft_captured[mask] = False
        self.secondary_released[mask] = False
        self.contact_constraint_violation[mask] = False
        self.contact_violation_before_capture[mask] = False
        self.contact_violation_before_thread[mask] = False
        self.contact_violation_premature_regrasp[mask] = False
        self.contact_violation_material_drift[mask] = False
        self.before_capture_contact_loss_steps[mask] = 0
        self.before_thread_two_finger_loss_steps[mask] = 0
        self.previous_marker_mod[mask] = 0.0
        self.marker_mod[mask] = 0.0
        self.twist[mask] = 0.0
        self.thread_resistance[mask] = self.task.rotation_resistance_min_nm
        self.axial_preload[mask] = self.task.axial_preload_n
        self.previous_depth[mask] = 0.0
        self.previous_twist[mask] = 0.0
        self.ik_failures[mask] = 0
        self.ik_anchor_fallbacks[mask] = 0
        self.ik_safety_stops[mask] = 0
        self.diffusion_handoff_hold_steps[mask] = 0
        self.preinsert_alignment_stable_steps[mask] = 0
        self.preinsert_feedback_offset_socket[mask] = 0.0
        self._assign_command(mask, command)
        self.update_measured(reset_twist=mask)
        self.previous_depth[mask] = self.insertion_depth()[mask]

    def state_dict(self):
        """Snapshot every non-MuJoCo world variable needed for continuation."""

        names = (
            "spawn_xy",
            "socket_angle_deg",
            "socket_frame",
            "socket_axis",
            "socket_mouth",
            "aligned_rotation",
            "reference_marker",
            "step_count",
            "elapsed_steps",
            "episode_return",
            "episode_id",
            "thread_engaged",
            "holder_active",
            "grasp_latched",
            "stable_grasp_support",
            "pickup_material_ref",
            "primary_hold_mask",
            "all_four_contact",
            "primary_pair_contact",
            "two_finger_contact",
            "all_four_contact_stable_steps",
            "primary_pair_contact_stable_steps",
            "capture_alignment_stable_steps",
            "thread_ready_stable_steps",
            "shaft_captured",
            "secondary_released",
            "contact_constraint_violation",
            "contact_violation_before_capture",
            "contact_violation_before_thread",
            "contact_violation_premature_regrasp",
            "contact_violation_material_drift",
            "before_capture_contact_loss_steps",
            "before_thread_two_finger_loss_steps",
            "previous_marker_mod",
            "marker_mod",
            "twist",
            "thread_resistance",
            "axial_preload",
            "previous_depth",
            "previous_twist",
            "ik_failures",
            "ik_anchor_fallbacks",
            "ik_safety_stops",
            "diffusion_handoff_hold_steps",
            "preinsert_alignment_stable_steps",
            "preinsert_feedback_offset_socket",
        )
        return {
            "version": 5,
            "worlds": int(self.backend.worlds),
            "tensors": {
                name: getattr(self, name).detach().clone()
                for name in names
            },
            "desired_command": self.desired_command.state_dict(),
        }

    def load_state_dict(self, state) -> None:
        if int(state.get("version", -1)) != 5:
            raise ValueError("unsupported task-state checkpoint version")
        if int(state.get("worlds", -1)) != self.backend.worlds:
            raise ValueError("task-state world count differs from environment")
        tensors = state.get("tensors")
        if not isinstance(tensors, dict):
            raise ValueError("task-state checkpoint has no tensor payload")
        for name, source in tensors.items():
            if not hasattr(self, name):
                raise ValueError(f"unknown task-state tensor {name!r}")
            target = getattr(self, name)
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"task-state {name!r} shape mismatch: "
                    f"{tuple(source.shape)} != {tuple(target.shape)}"
                )
            if name == "socket_angle_deg" and bool(
                (
                    source.to(device=self.device, dtype=self.dtype)
                    - self.supervisor.fixed_socket_angle_deg
                ).abs().gt(1.0e-6).any()
            ):
                raise ValueError(
                    "task-state checkpoint contains a non-45-degree socket"
                )
            # Several of these tensors are persistently aliased by the native
            # Warp wrench kernel.  Never replace their storage on restore.
            target.copy_(
                source.to(device=self.device, dtype=target.dtype)
            )
        required = {
            "spawn_xy",
            "socket_angle_deg",
            "socket_frame",
            "socket_axis",
            "socket_mouth",
            "aligned_rotation",
            "reference_marker",
            "step_count",
            "elapsed_steps",
            "episode_return",
            "episode_id",
            "thread_engaged",
            "holder_active",
            "grasp_latched",
            "stable_grasp_support",
            "pickup_material_ref",
            "primary_hold_mask",
            "all_four_contact",
            "primary_pair_contact",
            "two_finger_contact",
            "all_four_contact_stable_steps",
            "primary_pair_contact_stable_steps",
            "capture_alignment_stable_steps",
            "thread_ready_stable_steps",
            "shaft_captured",
            "secondary_released",
            "contact_constraint_violation",
            "contact_violation_before_capture",
            "contact_violation_before_thread",
            "contact_violation_premature_regrasp",
            "contact_violation_material_drift",
            "before_capture_contact_loss_steps",
            "before_thread_two_finger_loss_steps",
            "previous_marker_mod",
            "marker_mod",
            "twist",
            "thread_resistance",
            "axial_preload",
            "previous_depth",
            "previous_twist",
            "ik_failures",
            "ik_anchor_fallbacks",
            "ik_safety_stops",
            "diffusion_handoff_hold_steps",
            "preinsert_alignment_stable_steps",
            "preinsert_feedback_offset_socket",
        }
        missing = required.difference(tensors)
        if missing:
            raise ValueError(
                f"task-state checkpoint is missing {sorted(missing)}"
            )
        expected_primary = _torch().as_tensor(
            (False, False, True, True),
            device=self.device,
            dtype=_torch().bool,
        )
        if not bool((self.primary_hold_mask == expected_primary).all()):
            raise ValueError(
                "task-state primary_hold_mask is not fixed physical L/R"
            )
        self.desired_command.load_state_dict(
            state.get("desired_command", {})
        )

    def _assign_command(self, mask, command: CommandBatch) -> None:
        current = self.desired_command
        current.tip[mask] = command.tip[mask]
        current.rotation[mask] = command.rotation[mask]
        current.twist[mask] = command.twist[mask]
        current.material[mask] = command.material[mask]
        current.distance[mask] = command.distance[mask]
        current.phase[mask] = command.phase[mask]
        current.engaged[mask] = command.engaged[mask]

    def set_command(self, command: CommandBatch) -> None:
        mask = _torch().ones(
            self.backend.worlds, device=self.device, dtype=_torch().bool
        )
        self._assign_command(mask, command)

    def screw_pose(self):
        qpos = self.backend.tensor("qpos")
        qadr = self.backend.index.screw_qpos
        position = qpos[:, qadr : qadr + 3]
        quaternion = qpos[:, qadr + 3 : qadr + 7]
        return position, quaternion, quaternion_to_matrix(quaternion)

    def screw_velocity(self):
        # MJWarp cvel is [angular, linear] at the root c-frame (subtree COM).
        # Convert its linear component to the screw body origin before exposing
        # the task contract's [linear, angular] ordering.
        spatial = self.backend.tensor("cvel")[
            :, self.backend.index.screw_body
        ]
        position = self.screw_pose()[0]
        linear = self._point_velocity(
            spatial,
            self.backend.index.screw_root_body,
            position,
        )
        return _torch().cat((linear, spatial[:, 0:3]), dim=-1)

    def _point_velocity(self, spatial, root_body, point):
        torch = _torch()
        subtree_com = self.backend.tensor("subtree_com")
        if isinstance(root_body, tuple):
            roots = torch.as_tensor(
                root_body, device=self.device, dtype=torch.long
            )
            center = subtree_com.index_select(1, roots)
        else:
            center = subtree_com[:, int(root_body)]
        angular = spatial[..., 0:3]
        linear_at_com = spatial[..., 3:6]
        return linear_at_com + torch.linalg.cross(
            angular, point - center, dim=-1
        )

    def screw_tip_velocity(self):
        spatial = self.backend.tensor("cvel")[
            :, self.backend.index.screw_body
        ]
        return self._point_velocity(
            spatial,
            self.backend.index.screw_root_body,
            self.screw_tip(),
        )

    def screw_tip(self):
        position, _quaternion, rotation = self.screw_pose()
        local = _torch().as_tensor(
            (0.0, 0.0, self.task.screw_tip_local_z_m),
            device=self.device,
            dtype=self.dtype,
        )
        return position + _torch().einsum("bij,j->bi", rotation, local)

    def diffusion_handoff_errors(self):
        """Per-world measured error at the upright lifted-spawn handoff."""

        torch = _torch()
        _position, _quaternion, rotation = self.screw_pose()
        target = torch.cat(
            (
                self.spawn_xy,
                torch.full(
                    (self.backend.worlds, 1),
                    self.task.scripted_center_tip_world_z_m,
                    device=self.device,
                    dtype=self.dtype,
                ),
            ),
            dim=-1,
        ).to(
            device=self.device,
            dtype=self.dtype,
        )
        position_error = torch.linalg.vector_norm(
            self.screw_tip() - target, dim=-1
        )
        upright_error = torch.linalg.vector_norm(
            rotation_log(rotation.transpose(-1, -2)), dim=-1
        )
        return position_error, upright_error

    def diffusion_handoff_ready(self):
        """Measured pickup state that permits learned-control activation."""

        position_error, upright_error = self.diffusion_handoff_errors()
        return (
            self.grasp_latched
            & ~self.holder_active
            & (
                position_error
                <= self.task.diffusion_handoff_position_tolerance_m
            )
            & (
                upright_error
                <= math.radians(
                    self.task.diffusion_handoff_orientation_tolerance_deg
                )
            )
        )

    def tip_socket(self):
        return _torch().einsum(
            "bji,bj->bi",
            self.socket_frame,
            self.screw_tip() - self.socket_mouth,
        )

    def insertion_depth(self):
        return self.tip_socket()[:, 2]

    def axis_error(self):
        _position, _quaternion, rotation = self.screw_pose()
        error_world = rotation_log(
            self.aligned_rotation
            @ rotation.transpose(-1, -2)
        )
        return _torch().einsum(
            "bji,bj->bi", self.socket_frame, error_world
        )[:, :2]

    def preinsert_alignment_errors(self):
        """Measured errors for the 5 mm outside-mouth alignment pose."""

        torch = _torch()
        tip_socket = self.tip_socket()
        tip_velocity = self.screw_tip_velocity()
        tip_velocity_socket = torch.einsum(
            "bji,bj->bi", self.socket_frame, tip_velocity
        )
        screw_velocity = self.screw_velocity()
        _position, _quaternion, rotation = self.screw_pose()
        screw_axis_outward = rotation[:, :, 2]
        target_axis_outward = -self.socket_axis
        axis_cross = torch.linalg.vector_norm(
            torch.linalg.cross(
                screw_axis_outward, target_axis_outward, dim=-1
            ),
            dim=-1,
        )
        axis_dot = (
            screw_axis_outward * target_axis_outward
        ).sum(dim=-1).clamp(-1.0, 1.0)
        full_axis_angle = torch.atan2(axis_cross, axis_dot)
        return (
            torch.linalg.vector_norm(tip_socket[:, :2], dim=-1),
            full_axis_angle,
            (
                tip_socket[:, 2] + self.task.preinsert_standoff_m
            ).abs(),
            torch.linalg.vector_norm(tip_velocity, dim=-1),
            torch.linalg.vector_norm(
                tip_velocity_socket[:, :2], dim=-1
            ),
            torch.linalg.vector_norm(screw_velocity[:, 3:], dim=-1),
        )

    def preinsert_alignment_in_tolerance(self):
        """Instantaneous physical qualification before entering the bore."""

        torch = _torch()
        (
            lateral,
            axis,
            depth_error,
            tip_speed,
            lateral_speed,
            angular_speed,
        ) = self.preinsert_alignment_errors()
        finite = torch.stack(
            (
                lateral,
                axis,
                depth_error,
                tip_speed,
                lateral_speed,
                angular_speed,
            ),
            dim=-1,
        ).isfinite().all(dim=-1)
        all_four_contact = (
            self.all_four_contact
            if hasattr(self, "all_four_contact")
            else self.stable_grasp_support
        )
        return (
            finite
            & self.grasp_latched
            & all_four_contact
            & ~self.holder_active
            & (lateral <= self.task.preinsert_lateral_tolerance_m)
            & (
                axis
                <= math.radians(
                    self.task.preinsert_orientation_tolerance_deg
                )
            )
            & (depth_error <= self.task.preinsert_depth_tolerance_m)
            & (
                tip_speed
                <= self.task.preinsert_tip_speed_tolerance_m_s
            )
            & (
                lateral_speed
                <= self.task.preinsert_lateral_speed_tolerance_m_s
            )
            & (
                angular_speed
                <= math.radians(
                    self.task.preinsert_angular_speed_tolerance_deg_s
                )
            )
        )

    def preinsert_alignment_ready(self):
        """Require a stable run of qualified frames before INSERT."""

        return self.preinsert_alignment_in_tolerance() & (
            self.preinsert_alignment_stable_steps
            >= self.task.preinsert_stable_steps
        )

    def capture_alignment_errors(self):
        """Measured errors at the in-bore two-finger capture handoff."""

        torch = _torch()
        tip_socket = self.tip_socket()
        tip_velocity = self.screw_tip_velocity()
        tip_velocity_socket = torch.einsum(
            "bji,bj->bi", self.socket_frame, tip_velocity
        )
        screw_velocity = self.screw_velocity()
        _position, _quaternion, rotation = self.screw_pose()
        screw_axis_outward = rotation[:, :, 2]
        target_axis_outward = -self.socket_axis
        axis_cross = torch.linalg.vector_norm(
            torch.linalg.cross(
                screw_axis_outward, target_axis_outward, dim=-1
            ),
            dim=-1,
        )
        axis_dot = (
            screw_axis_outward * target_axis_outward
        ).sum(dim=-1).clamp(-1.0, 1.0)
        full_axis_angle = torch.atan2(axis_cross, axis_dot)
        return (
            torch.linalg.vector_norm(tip_socket[:, :2], dim=-1),
            full_axis_angle,
            (tip_socket[:, 2] - self.task.capture_depth_m).abs(),
            torch.linalg.vector_norm(tip_velocity, dim=-1),
            torch.linalg.vector_norm(
                tip_velocity_socket[:, :2], dim=-1
            ),
            torch.linalg.vector_norm(screw_velocity[:, 3:], dim=-1),
        )

    def head_bottom_standoff(self):
        """Axial clearance from the mouth plane to the screw-head bottom."""

        return self.task.screw_shaft_length_m - self.insertion_depth()

    def capture_alignment_in_tolerance(self):
        """Instantaneous four-finger qualification for shaft capture."""

        torch = _torch()
        (
            lateral,
            axis,
            depth_error,
            tip_speed,
            lateral_speed,
            angular_speed,
        ) = self.capture_alignment_errors()
        finite = torch.stack(
            (
                lateral,
                axis,
                depth_error,
                tip_speed,
                lateral_speed,
                angular_speed,
                self.head_bottom_standoff(),
            ),
            dim=-1,
        ).isfinite().all(dim=-1)
        return (
            finite
            & self.grasp_latched
            & self.all_four_contact
            & ~self.holder_active
            & (lateral <= self.task.capture_lateral_tolerance_m)
            & (
                axis
                <= math.radians(
                    self.task.capture_orientation_tolerance_deg
                )
            )
            & (depth_error <= self.task.capture_depth_tolerance_m)
            & (
                tip_speed
                <= self.task.capture_tip_speed_tolerance_m_s
            )
            & (
                lateral_speed
                <= self.task.capture_lateral_speed_tolerance_m_s
            )
            & (
                angular_speed
                <= math.radians(
                    self.task.capture_angular_speed_tolerance_deg_s
                )
            )
            & (
                self.head_bottom_standoff()
                >= self.task.capture_head_bottom_clearance_m
            )
        )

    def capture_alignment_ready(self):
        """Require consecutive qualified frames before the two-contact gate."""

        return self.capture_alignment_in_tolerance() & (
            self.capture_alignment_stable_steps
            >= self.task.capture_stable_steps
        )

    def fingertip_state(self):
        xpos = self.backend.tensor("xpos")
        cvel = self.backend.tensor("cvel")
        ids = _torch().as_tensor(
            self.backend.index.tip_bodies,
            device=self.device,
            dtype=_torch().long,
        )
        position = xpos.index_select(1, ids)
        spatial = cvel.index_select(1, ids)
        velocity = self._point_velocity(
            spatial,
            self.backend.index.tip_body_roots,
            position,
        )
        sensors = self.backend.tensor("sensordata")
        sensor_ids = _torch().as_tensor(
            self.backend.index.touch_sensor_adr,
            device=self.device,
            dtype=_torch().long,
        )
        force = sensors.index_select(1, sensor_ids).clamp_min(0.0)
        return position, velocity, force

    def helical_errors(self):
        torch = _torch()
        lead = self.task.thread_pitch_m / (2.0 * math.pi)
        depth = self.insertion_depth()
        velocity = self.screw_velocity()
        tip_linear_velocity = self.screw_tip_velocity()
        error = (
            depth - self.task.thread_engagement_depth_m - lead * self.twist
        )
        velocity_error = (
            (tip_linear_velocity * self.socket_axis).sum(-1)
            - lead * (velocity[:, 3:] * self.socket_axis).sum(-1)
        )
        zero = torch.zeros_like(error)
        return (
            torch.where(self.thread_engaged, error, zero),
            torch.where(self.thread_engaged, velocity_error, zero),
        )

    def thread_ready(self):
        torch = _torch()
        return (
            (
                (
                    self.insertion_depth()
                    - self.task.thread_engagement_depth_m
                ).abs()
                <= self.task.thread_depth_tolerance_m
            )
            & (
                torch.linalg.vector_norm(self.tip_socket()[:, :2], dim=-1)
                <= self.task.thread_lateral_tolerance_m
            )
            & (
                torch.linalg.vector_norm(self.axis_error(), dim=-1)
                <= math.radians(
                    self.task.thread_orientation_tolerance_deg
                )
            )
        )

    def thread_engagement_errors(self):
        """Measured pose/velocity errors at the 70 mm thread gate."""

        torch = _torch()
        tip_socket = self.tip_socket()
        tip_velocity = self.screw_tip_velocity()
        tip_velocity_socket = torch.einsum(
            "bji,bj->bi", self.socket_frame, tip_velocity
        )
        screw_velocity = self.screw_velocity()
        _position, _quaternion, rotation = self.screw_pose()
        screw_axis_outward = rotation[:, :, 2]
        target_axis_outward = -self.socket_axis
        axis_cross = torch.linalg.vector_norm(
            torch.linalg.cross(
                screw_axis_outward, target_axis_outward, dim=-1
            ),
            dim=-1,
        )
        axis_dot = (
            screw_axis_outward * target_axis_outward
        ).sum(dim=-1).clamp(-1.0, 1.0)
        full_axis_angle = torch.atan2(axis_cross, axis_dot)
        lateral = torch.linalg.vector_norm(
            tip_socket[:, :2], dim=-1
        )
        tip_speed = torch.linalg.vector_norm(tip_velocity, dim=-1)
        lateral_speed = torch.linalg.vector_norm(
            tip_velocity_socket[:, :2], dim=-1
        )
        angular_speed = torch.linalg.vector_norm(
            screw_velocity[:, 3:], dim=-1
        )
        return (
            lateral,
            full_axis_angle,
            (
                tip_socket[:, 2]
                - self.task.thread_engagement_depth_m
            ).abs(),
            tip_speed,
            lateral_speed,
            angular_speed,
        )

    def thread_engagement_in_tolerance(self):
        """Measured low-velocity qualification with at least two contacts."""

        torch = _torch()
        tip_socket = self.tip_socket()
        (
            lateral,
            full_axis_angle,
            depth_error,
            tip_speed,
            lateral_speed,
            angular_speed,
        ) = self.thread_engagement_errors()
        finite = torch.stack(
            (
                tip_socket[:, 2],
                lateral,
                full_axis_angle,
                depth_error,
                tip_speed,
                lateral_speed,
                angular_speed,
            ),
            dim=-1,
        ).isfinite().all(dim=-1)
        return (
            finite
            & self.shaft_captured
            & self.two_finger_contact
            & (depth_error <= self.task.thread_depth_tolerance_m)
            & (lateral <= self.task.thread_lateral_tolerance_m)
            & (
                full_axis_angle
                <= math.radians(
                    self.task.thread_orientation_tolerance_deg
                )
            )
            & (
                tip_speed
                <= self.task.capture_tip_speed_tolerance_m_s
            )
            & (
                lateral_speed
                <= self.task.capture_lateral_speed_tolerance_m_s
            )
            & (
                angular_speed
                <= math.radians(
                    self.task.capture_angular_speed_tolerance_deg_s
                )
            )
        )

    def thread_engagement_ready(self):
        """Measured thread pose sustained long enough to enter ENGAGE."""

        return self.thread_engagement_in_tolerance() & (
            self.thread_ready_stable_steps
            >= self.task.thread_stable_steps
        )

    def update_measured(self, *, reset_twist=None) -> None:
        torch = _torch()
        if reset_twist is None:
            reset_twist = torch.zeros(
                self.backend.worlds, device=self.device, dtype=torch.bool
            )
        _position, _quaternion, rotation = self.screw_pose()
        marker = rotation[:, :, 0]
        outward = -self.socket_axis
        marker = marker - (
            marker * outward
        ).sum(-1, keepdim=True) * outward
        marker = marker / torch.linalg.vector_norm(
            marker, dim=-1, keepdim=True
        ).clamp_min(1.0e-9)
        reference = self.reference_marker
        signed_ccw = torch.atan2(
            (
                torch.linalg.cross(
                    reference, marker, dim=-1
                )
                * outward
            ).sum(-1),
            (reference * marker).sum(-1),
        )
        modular = _wrap_pi(-signed_ccw)
        engagement_phase = (
            (self.desired_command.phase == int(TaskPhase.INSERT))
            | (self.desired_command.phase == int(TaskPhase.ENGAGE))
        )
        newly_engaged = (
            engagement_phase
            & self.thread_engagement_ready()
            & ~self.thread_engaged
        )
        self.thread_engaged |= newly_engaged
        reset = reset_twist | newly_engaged
        increment = _wrap_pi(modular - self.previous_marker_mod)
        new_twist = torch.where(
            self.thread_engaged & ~reset, self.twist + increment, self.twist
        )
        new_twist = torch.where(
            reset, torch.zeros_like(new_twist), new_twist
        )
        # The native Warp pre-step kernel aliases these tensors once during
        # construction, so updates must preserve their storage.
        self.twist.copy_(new_twist)
        self.marker_mod = torch.where(
            self.thread_engaged, modular, torch.zeros_like(modular)
        )
        self.previous_marker_mod = torch.where(
            self.thread_engaged, modular, self.previous_marker_mod
        )
        progress = (
            self.twist / max(self.task.target_rotation_rad, 1.0e-9)
        ).clamp(0.0, 1.0)
        self.thread_resistance.copy_(
            self.task.rotation_resistance_min_nm
            + progress
            * (
                self.task.rotation_resistance_max_nm
                - self.task.rotation_resistance_min_nm
            )
        )

    def _desired_screw_pose(self):
        local_tip = _torch().as_tensor(
            (0.0, 0.0, self.task.screw_tip_local_z_m),
            device=self.device,
            dtype=self.dtype,
        )
        position = self.desired_command.tip - _torch().einsum(
            "bij,j->bi", self.desired_command.rotation, local_tip
        )
        return position, self.desired_command.rotation

    def _ensure_contact_contract_state(self, contact):
        """Initialize contact tensors for lightweight state test doubles."""

        torch = _torch()
        n = int(contact.shape[0])
        device = contact.device
        if not hasattr(self, "primary_hold_mask"):
            self.primary_hold_mask = torch.as_tensor(
                (False, False, True, True),
                device=device,
                dtype=torch.bool,
            ).expand(n, 4).clone()
        bool_names = (
            "all_four_contact",
            "primary_pair_contact",
            "two_finger_contact",
            "shaft_captured",
            "secondary_released",
            "contact_constraint_violation",
            "contact_violation_before_capture",
            "contact_violation_before_thread",
            "contact_violation_premature_regrasp",
            "contact_violation_material_drift",
        )
        for name in bool_names:
            if not hasattr(self, name):
                setattr(
                    self,
                    name,
                    torch.zeros(n, device=device, dtype=torch.bool),
                )
        counter_names = (
            "all_four_contact_stable_steps",
            "primary_pair_contact_stable_steps",
            "capture_alignment_stable_steps",
            "thread_ready_stable_steps",
            "before_capture_contact_loss_steps",
            "before_thread_two_finger_loss_steps",
        )
        for name in counter_names:
            if not hasattr(self, name):
                setattr(
                    self,
                    name,
                    torch.zeros(n, device=device, dtype=torch.long),
                )
        if not hasattr(self, "pickup_material_ref"):
            self.pickup_material_ref = torch.zeros(
                (n, 4),
                device=device,
                dtype=self.desired_command.material.dtype,
            )
        if not hasattr(self, "fingertip_contact_force_n"):
            self.fingertip_contact_force_n = torch.zeros(
                (n, 4),
                device=device,
                dtype=self.desired_command.material.dtype,
            )

    def post_physics(self) -> None:
        torch = _torch()
        _position, _velocity, contact_force = self.fingertip_state()
        acquired_contact = (
            contact_force
            >= getattr(self.task, "contact_force_threshold_n", 0.10)
        )
        maintained_contact = (
            contact_force
            >= getattr(
                self.task,
                "contact_force_release_threshold_n",
                0.04,
            )
        )
        # A firm 0.10 N contact is required before releasing the fixture.
        # Once that grasp is established, use a lower Schmitt off threshold:
        # MJWarp can briefly redistribute normal force between still-closed
        # fingers, and that is not equivalent to an intentional release.
        contact = torch.where(
            self.holder_active[:, None],
            acquired_contact,
            maintained_contact,
        )
        self._ensure_contact_contract_state(contact)
        self.fingertip_contact_force_n.copy_(contact_force)
        all_four = contact.all(dim=-1)
        primary_pair = torch.where(
            self.primary_hold_mask,
            contact,
            torch.ones_like(contact),
        ).all(dim=-1)
        two_finger = (
            contact.sum(dim=-1)
            >= self.task.minimum_contact_count_after_capture
        )
        pre_capture_support = (
            contact.sum(dim=-1)
            >= self.task.minimum_contact_count_before_capture
        )
        self.all_four_contact.copy_(all_four)
        self.primary_pair_contact.copy_(primary_pair)
        self.two_finger_contact.copy_(two_finger)
        self.all_four_contact_stable_steps.copy_(
            torch.where(
                all_four,
                self.all_four_contact_stable_steps + 1,
                torch.zeros_like(self.all_four_contact_stable_steps),
            )
        )
        self.primary_pair_contact_stable_steps.copy_(
            torch.where(
                primary_pair,
                self.primary_pair_contact_stable_steps + 1,
                torch.zeros_like(
                    self.primary_pair_contact_stable_steps
                ),
            )
        )
        material = self.desired_command.material
        stable_support = torch.zeros_like(all_four)
        for first in range(4):
            for second in range(first + 1, 4):
                separation = (
                    _wrap_pi(
                        material[:, first] - material[:, second]
                    )
                    .abs()
                )
                stable_support |= (
                    contact[:, first]
                    & contact[:, second]
                    & (
                        separation
                        >= math.radians(
                            self.task.preinsert_support_separation_deg
                        )
                    )
                )
        self.stable_grasp_support.copy_(stable_support)
        phase = self.desired_command.phase
        may_release = (
            phase != int(TaskPhase.PREGRASP)
        ) & (phase != int(TaskPhase.CLOSE))
        release = (
            self.holder_active
            & all_four
            & (
                self.all_four_contact_stable_steps
                >= getattr(
                    self.task, "all_four_contact_stable_steps", 5
                )
            )
            & may_release
        )
        # Formal physics never releases the spawn fixture on elapsed time:
        # every fingertip must sustain measured contact for five frames.
        self.pickup_material_ref[release] = material[release]
        self.backend.release_holder(release)
        self.holder_active[release] = False
        self.grasp_latched |= all_four & ~self.holder_active

        preinsert_qualified = (
            (phase == int(TaskPhase.ALIGN))
            & self.preinsert_alignment_in_tolerance()
        )
        self.preinsert_alignment_stable_steps.copy_(
            torch.where(
                preinsert_qualified,
                self.preinsert_alignment_stable_steps + 1,
                torch.zeros_like(self.preinsert_alignment_stable_steps),
            )
        )
        capture_phase = phase == int(TaskPhase.INSERT)
        if hasattr(self, "device"):
            capture_in_tolerance = self.capture_alignment_in_tolerance()
        elif bool(capture_phase.any()):
            # Lightweight CPU test doubles do not expose the measured pose.
            capture_in_tolerance = self.capture_alignment_in_tolerance()
        else:
            capture_in_tolerance = torch.zeros_like(capture_phase)
        capture_qualified = capture_phase & capture_in_tolerance
        self.capture_alignment_stable_steps.copy_(
            torch.where(
                capture_qualified,
                self.capture_alignment_stable_steps + 1,
                torch.zeros_like(self.capture_alignment_stable_steps),
            )
        )
        newly_captured = (
            ~self.shaft_captured
            & capture_phase
            & capture_in_tolerance
            & (
                self.capture_alignment_stable_steps
                >= getattr(self.task, "capture_stable_steps", 5)
            )
        )
        self.shaft_captured |= newly_captured

        thread_count_phase = (
            (phase == int(TaskPhase.INSERT))
            | (phase == int(TaskPhase.ENGAGE))
        )
        if hasattr(self, "device"):
            thread_qualified = (
                thread_count_phase
                & self.shaft_captured
                & self.thread_engagement_in_tolerance()
            )
        elif bool(thread_count_phase.any()):
            thread_qualified = (
                thread_count_phase
                & self.shaft_captured
                & self.thread_engagement_in_tolerance()
            )
        else:
            thread_qualified = torch.zeros_like(thread_count_phase)
        self.thread_ready_stable_steps.copy_(
            torch.where(
                thread_qualified,
                self.thread_ready_stable_steps + 1,
                torch.zeros_like(self.thread_ready_stable_steps),
            )
        )
        self.update_measured()

        before_capture_missing = (
            ~self.holder_active
            & ~self.shaft_captured
            & ~pre_capture_support
        )
        before_thread_two_finger_missing = (
            self.shaft_captured
            & ~self.thread_engaged
            & ~two_finger
        )
        self.before_capture_contact_loss_steps.copy_(
            torch.where(
                before_capture_missing,
                self.before_capture_contact_loss_steps + 1,
                torch.zeros_like(
                    self.before_capture_contact_loss_steps
                ),
            )
        )
        self.before_thread_two_finger_loss_steps.copy_(
            torch.where(
                before_thread_two_finger_missing,
                self.before_thread_two_finger_loss_steps + 1,
                torch.zeros_like(
                    self.before_thread_two_finger_loss_steps
                ),
            )
        )
        loss_grace = int(
            getattr(self.task, "contact_loss_grace_steps", 20)
        )
        before_capture_loss = (
            self.before_capture_contact_loss_steps >= loss_grace
        )
        before_thread_two_finger_loss = (
            self.before_thread_two_finger_loss_steps >= loss_grace
        )
        legal_secondary_open = (
            (phase == int(TaskPhase.UD_OPEN))
            & self.shaft_captured
        )
        # Latch only when the trajectory actually starts its one monotonic
        # U/D release; capture merely grants permission for that release.
        self.secondary_released |= legal_secondary_open

        regrasp = torch.zeros_like(phase, dtype=torch.bool)
        for value in (
            TaskPhase.LR_OPEN,
            TaskPhase.LR_RESET,
            TaskPhase.LR_CLOSE,
            TaskPhase.UD_RESET,
            TaskPhase.UD_CLOSE,
        ):
            regrasp |= phase == int(value)
        premature_regrasp = (
            (regrasp & ~self.thread_engaged)
            | (
                (phase == int(TaskPhase.UD_OPEN))
                & ~self.shaft_captured
            )
        )
        # Releasing pressure is not permission to slide the fingers around
        # the head.  Until measured thread engagement, every material angle
        # must remain at the value captured when the start fixture released.
        prethread_material_drift = (
            self.grasp_latched
            & ~self.thread_engaged
            & (
                _wrap_pi(material - self.pickup_material_ref)
                .abs()
                .amax(dim=-1)
                > math.radians(0.05)
            )
        )
        self.contact_violation_before_capture |= before_capture_loss
        self.contact_violation_before_thread |= (
            before_thread_two_finger_loss
        )
        self.contact_violation_premature_regrasp |= premature_regrasp
        self.contact_violation_material_drift |= (
            prethread_material_drift
        )
        self.contact_constraint_violation |= (
            self.contact_violation_before_capture
            | self.contact_violation_before_thread
            | self.contact_violation_premature_regrasp
            | self.contact_violation_material_drift
        )

    def contact_recovery_pending(self):
        """Freeze the task schedule during a recoverable contact-force dip."""

        grace = int(
            getattr(self.task, "contact_loss_grace_steps", 20)
        )
        return (
            (
                (self.before_capture_contact_loss_steps > 0)
                & (self.before_capture_contact_loss_steps < grace)
            )
            | (
                (self.before_thread_two_finger_loss_steps > 0)
                & (self.before_thread_two_finger_loss_steps < grace)
            )
        )

    def observation(self):
        torch = _torch()
        qpos = self.backend.tensor("qpos")
        qvel = self.backend.tensor("qvel")
        ctrl = self.backend.tensor("ctrl")
        if self.backend.has_field("actuator_force"):
            actuator_force = self.backend.tensor("actuator_force")
        else:
            qfrc = self.backend.tensor("qfrc_actuator")
            indices = torch.as_tensor(
                self.backend.index.actuator_dof,
                device=self.device,
                dtype=torch.long,
            )
            actuator_force = qfrc.index_select(1, indices)
        values = [
            qpos.clamp(-10.0, 10.0),
            (qvel / 10.0).clamp(-5.0, 5.0),
        ]
        span = (self.backend.ctrl_range[:, 1] - self.backend.ctrl_range[:, 0]).clamp_min(
            1.0e-6
        )
        midpoint = 0.5 * (
            self.backend.ctrl_range[:, 1] + self.backend.ctrl_range[:, 0]
        )
        values.append(((ctrl - midpoint) / (0.5 * span)).clamp(-2.0, 2.0))
        values.append(torch.tanh(actuator_force / 20.0))

        position, quaternion, _rotation = self.screw_pose()
        velocity = self.screw_velocity()
        tip_socket = self.tip_socket()
        axis = self.axis_error()
        helical_error, helical_velocity_error = self.helical_errors()
        global_state = torch.cat(
            (
                position / 0.5,
                quaternion,
                velocity / 5.0,
                tip_socket[:, 0:1] / 0.1,
                tip_socket[:, 1:2] / 0.1,
                tip_socket[:, 2:3] / self.task.socket_hole_depth_m,
                axis / math.radians(10.0),
                torch.sin(self.marker_mod)[:, None],
                torch.cos(self.marker_mod)[:, None],
                self.twist[:, None] / (4.0 * math.pi),
                self.twist[:, None] / (2.0 * math.pi),
                helical_error[:, None] / 0.005,
                helical_velocity_error[:, None] / 0.05,
                self.thread_resistance[:, None] / 0.12,
                self.axial_preload[:, None] / 5.0,
                torch.sin(torch.deg2rad(self.socket_angle_deg))[:, None],
                torch.cos(torch.deg2rad(self.socket_angle_deg))[:, None],
                self.socket_mouth[:, 0:1]
                / self.task.socket_mouth_orbit_radius_m,
                self.socket_mouth[:, 1:2]
                / self.task.socket_mouth_orbit_radius_m,
                -self.socket_mouth[:, 0:1]
                / self.task.socket_mouth_orbit_radius_m,
                -self.socket_mouth[:, 1:2]
                / self.task.socket_mouth_orbit_radius_m,
            ),
            dim=-1,
        )
        values.append(global_state)

        tip_position, tip_velocity, contact_force = self.fingertip_state()
        relative = tip_position - position[:, None, :]
        contact = contact_force >= 0.05
        per_finger = []
        for finger in range(4):
            angle = self.desired_command.material[:, finger]
            per_finger.append(
                torch.cat(
                    (
                        torch.tanh(contact_force[:, finger : finger + 1] / 20.0),
                        contact[:, finger : finger + 1].to(self.dtype),
                        relative[:, finger] / 0.1,
                        tip_velocity[:, finger],
                        torch.sin(angle)[:, None],
                        torch.cos(angle)[:, None],
                        self.desired_command.distance[:, finger : finger + 1] / 0.06,
                    ),
                    dim=-1,
                )
            )
        values.append(torch.cat(per_finger, dim=-1))
        phase_one_hot = torch.nn.functional.one_hot(
            self.desired_command.phase.clamp(0, len(TaskPhase) - 1),
            num_classes=len(TaskPhase),
        ).to(self.dtype)
        values.append(phase_one_hot)
        depth = self.insertion_depth()
        target = torch.cat(
            (
                torch.full_like(depth[:, None], self.task.target_insertion_depth_m)
                / self.task.socket_hole_depth_m,
                torch.full_like(depth[:, None], self.task.target_rotation_rad)
                / (4.0 * math.pi),
                (self.task.target_insertion_depth_m - depth)[:, None]
                / self.task.socket_hole_depth_m,
                (self.task.target_rotation_rad - self.twist)[:, None]
                / (4.0 * math.pi),
                self.elapsed_steps[:, None].to(self.dtype)
                / max(float(self.hpc_config.physics.max_episode_steps), 1.0),
                self.holder_active[:, None].to(self.dtype),
                self.thread_engaged[:, None].to(self.dtype),
            ),
            dim=-1,
        )
        values.append(target)
        observation = torch.cat(values, dim=-1).to(torch.float32)
        if observation.shape != (self.backend.worlds, OBS_DIM):
            raise RuntimeError(
                f"GPU observation shape {tuple(observation.shape)} != "
                f"{(self.backend.worlds, OBS_DIM)}"
            )
        return observation

    def reward_done(self, residual_action, ik_success, safety_stop):
        torch = _torch()
        depth = self.insertion_depth()
        twist = self.twist
        depth_progress = (depth - self.previous_depth) / self.task.socket_hole_depth_m
        twist_progress = (
            twist - self.previous_twist
        ) / self.task.target_rotation_rad
        self.previous_depth.copy_(depth)
        self.previous_twist.copy_(twist)
        lateral = torch.linalg.vector_norm(self.tip_socket()[:, :2], dim=-1)
        axis = torch.linalg.vector_norm(self.axis_error(), dim=-1)
        helical_error, _velocity_error = self.helical_errors()
        actual_position, _quaternion, actual_rotation = self.screw_pose()
        desired_position, desired_rotation = self._desired_screw_pose()
        position_error = torch.linalg.vector_norm(
            desired_position - actual_position, dim=-1
        )
        rotation_error = torch.linalg.vector_norm(
            rotation_log(desired_rotation @ actual_rotation.transpose(-1, -2)),
            dim=-1,
        )
        alignment_active = self.desired_command.engaged | (
            torch.einsum(
                "bj,bj->b",
                self.socket_axis,
                self.desired_command.tip - self.socket_mouth,
            )
            >= -0.010
        )
        alignment_penalty = torch.where(
            alignment_active,
            0.020
            * (
                2.0 * lateral / 0.005
                + 0.5 * axis / math.radians(5.0)
            ),
            torch.zeros_like(lateral),
        )
        reward = (
            80.0 * depth_progress
            + 60.0 * twist_progress
            - alignment_penalty
            - 0.02 * helical_error.abs() / 0.002
            - 0.01 * position_error / 0.005
            - 0.005 * rotation_error / math.radians(5.0)
            # ``residual_action`` is the realised actuator correction divided
            # by its mixed-unit authority.  Penalise the mean enabled-channel
            # use so the certified Diffusion--IK command remains dominant.
            - 0.002 * residual_action.square().mean(-1)
            - (~ik_success).to(self.dtype) * 5.0
        )
        velocity = self.screw_velocity()
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        success = (
            ((depth - self.task.target_insertion_depth_m).abs() <= 0.002)
            & ((twist - self.task.target_rotation_rad).abs() <= math.radians(8.0))
            & (lateral <= 0.003)
            & (axis <= math.radians(3.0))
            & (speed <= 0.35)
            & (
                (self.desired_command.phase == int(TaskPhase.FINAL_HOLD))
                | (self.desired_command.phase == int(TaskPhase.COMPLETE))
            )
        )
        position, _quat, _rot = self.screw_pose()
        nonfinite = ~torch.isfinite(self.backend.tensor("qpos")).all(dim=-1)
        fell_below_table = position[:, 2] < self.task.table_top_z_m - 0.12
        lateral_escape = lateral > 0.16
        failure = (
            nonfinite
            | fell_below_table
            | lateral_escape
            | safety_stop
            | self.contact_constraint_violation
        )
        # A safety/physics failure wins if both predicates happen on the same
        # control step.  Terminal outcome counters must remain mutually
        # exclusive; otherwise cumulative and recent success rates disagree.
        success = success & ~failure
        truncated = (
            self.elapsed_steps >= self.hpc_config.physics.max_episode_steps
        ) & ~success & ~failure
        reward = reward + success.to(self.dtype) * 100.0 - failure.to(self.dtype) * 50.0
        done = success | failure | truncated
        self.episode_return += reward
        info = {
            "success": success,
            "failure": failure,
            "truncated": truncated,
            "nonfinite": nonfinite,
            "fell_below_table": fell_below_table,
            "lateral_escape": lateral_escape,
            "safety_stop": safety_stop,
            "contact_constraint_violation": (
                self.contact_constraint_violation.clone()
            ),
            "contact_violation_before_capture": (
                self.contact_violation_before_capture.clone()
            ),
            "contact_violation_before_thread": (
                self.contact_violation_before_thread.clone()
            ),
            "contact_violation_premature_regrasp": (
                self.contact_violation_premature_regrasp.clone()
            ),
            "contact_violation_material_drift": (
                self.contact_violation_material_drift.clone()
            ),
            "before_capture_contact_loss_steps": (
                self.before_capture_contact_loss_steps.clone()
            ),
            "before_thread_two_finger_loss_steps": (
                self.before_thread_two_finger_loss_steps.clone()
            ),
            "contact_recovery_pending": (
                self.contact_recovery_pending().clone()
            ),
            "fingertip_contact_force_n": (
                self.fingertip_contact_force_n.clone()
            ),
            "task_phase": self.desired_command.phase.clone(),
            "depth": depth,
            "twist": twist,
            "lateral_error": lateral,
            "axis_error": axis,
            "speed": speed,
            "episode_return": self.episode_return.clone(),
            "elapsed_steps": self.elapsed_steps.clone(),
            "ik_failures": self.ik_failures.clone(),
            "ik_anchor_fallbacks": self.ik_anchor_fallbacks.clone(),
            "ik_safety_stops": self.ik_safety_stops.clone(),
            "thread_engaged": self.thread_engaged.clone(),
            "holder_active": self.holder_active.clone(),
            "grasp_latched": self.grasp_latched.clone(),
            "stable_grasp_support": self.stable_grasp_support.clone(),
            "pickup_material_ref": self.pickup_material_ref.clone(),
            "primary_hold_mask": self.primary_hold_mask.clone(),
            "all_four_contact": self.all_four_contact.clone(),
            "primary_pair_contact": self.primary_pair_contact.clone(),
            "two_finger_contact": self.two_finger_contact.clone(),
            "all_four_contact_stable_steps": (
                self.all_four_contact_stable_steps.clone()
            ),
            "primary_pair_contact_stable_steps": (
                self.primary_pair_contact_stable_steps.clone()
            ),
            "capture_alignment_ready": self.capture_alignment_ready(),
            "capture_alignment_stable_steps": (
                self.capture_alignment_stable_steps.clone()
            ),
            "capture_head_bottom_standoff": (
                self.head_bottom_standoff().clone()
            ),
            "shaft_captured": self.shaft_captured.clone(),
            "secondary_released": self.secondary_released.clone(),
            "thread_engagement_ready": self.thread_engagement_ready(),
            "thread_ready_stable_steps": (
                self.thread_ready_stable_steps.clone()
            ),
            "preinsert_alignment_ready": (
                self.preinsert_alignment_ready()
            ),
            "preinsert_alignment_stable_steps": (
                self.preinsert_alignment_stable_steps.clone()
            ),
            "socket_angle_deg": self.socket_angle_deg.clone(),
            "socket_mouth_xy": self.socket_mouth[:, :2].clone(),
            "socket_opening_azimuth_xy": (
                -self.socket_mouth[:, :2]
                / self.task.socket_mouth_orbit_radius_m
            ).clone(),
        }
        return reward, done, info


__all__ = ["BatchedTaskState"]
