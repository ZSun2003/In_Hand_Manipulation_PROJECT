"""Native vector environment: MJWarp + nominal planning + analytic IK."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

from screw_control import (
    DIFFUSION_TO_ANCHOR_BLEND_STEPS,
    PICKUP_RELEASE_GATE_STABLE_STEPS,
    PICKUP_RELEASE_GATE_STEP,
    TaskPhase,
)

from .analytic_ik import BatchedAnalyticIK
from .actuator_residual import BatchedActuatorResidualMapper
from .contracts import ACTION_DIM, OBS_DIM, RESIDUAL_ACTION_DIM
from .mjwarp_backend import MJWarpBackend
from .supervisor import (
    BatchedSupervisor,
    CommandBatch,
    REGRASP_PHASES,
    TaskSpaceSafety,
    rotation_exp,
    rotation_log,
)
from .task import BatchedTaskState
from .torch_compat import unique_cuda_index_writes


RESET_IK_RESAMPLE_BATCHES = 32


def _torch():
    import torch

    return torch


def _select_command(mask, first: CommandBatch, second: CommandBatch) -> CommandBatch:
    torch = _torch()
    return CommandBatch(
        tip=torch.where(mask[:, None], first.tip, second.tip),
        rotation=torch.where(mask[:, None, None], first.rotation, second.rotation),
        twist=torch.where(mask, first.twist, second.twist),
        material=torch.where(mask[:, None], first.material, second.material),
        distance=torch.where(mask[:, None], first.distance, second.distance),
        phase=torch.where(mask, first.phase, second.phase),
        engaged=torch.where(mask, first.engaged, second.engaged),
    )


def _select_ik_routes(candidate_success, anchor_success, planner_active):
    """Choose candidate/anchor IK without hiding an A* planning failure."""

    use_candidate = candidate_success
    planner_ik_failure = planner_active & ~use_candidate
    use_anchor = (
        ~use_candidate
        & anchor_success
        & ~planner_ik_failure
    )
    safety_stop = ~use_candidate & ~use_anchor
    return use_candidate, use_anchor, safety_stop, planner_ik_failure


def _blend_command(
    source: CommandBatch,
    target: CommandBatch,
    progress,
) -> CommandBatch:
    """Minimum-jerk task-space interpolation with geodesic SO(3) rotation."""

    torch = _torch()
    progress = torch.as_tensor(
        progress, device=source.tip.device, dtype=source.tip.dtype
    )
    if progress.ndim == 0:
        progress = progress.expand(source.tip.shape[0])
    if tuple(progress.shape) != (source.tip.shape[0],):
        raise ValueError(
            "command blend progress must have one value per world"
        )
    t = progress.clamp(0.0, 1.0)
    weight = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
    rotation_delta = rotation_log(
        source.rotation.transpose(-1, -2) @ target.rotation
    )
    rotation = source.rotation @ rotation_exp(
        weight[:, None] * rotation_delta
    )
    material_delta = torch.remainder(
        target.material - source.material + torch.pi,
        2.0 * torch.pi,
    ) - torch.pi
    return CommandBatch(
        tip=source.tip + weight[:, None] * (target.tip - source.tip),
        rotation=rotation,
        twist=source.twist + weight * (target.twist - source.twist),
        material=source.material + weight[:, None] * material_delta,
        distance=(
            source.distance
            + weight[:, None] * (target.distance - source.distance)
        ),
        # Discrete safety semantics take effect immediately; only continuous
        # task-space fields are interpolated.
        phase=target.phase,
        engaged=target.engaged,
    )


@dataclass
class StepBatch:
    observation: object
    reward: object
    done: object
    info: Dict[str, object]
    ik_feasible_fraction: object
    expert_action: object


class MJWarpScrewEnv:
    """One CUDA-resident vector environment with masked per-world reset."""

    def __init__(
        self,
        config,
        *,
        worlds: Optional[int] = None,
        baseline: str = "expert",
        diffusion_controller=None,
        nominal_planner=None,
        curriculum_progress: float = 1.0,
    ) -> None:
        torch = _torch()
        valid_baselines = (
            "expert",
            "diffusion",
            "external_diffusion",
            "astar_spline",
        )
        if baseline not in valid_baselines:
            raise ValueError(
                "baseline must be expert, diffusion, external_diffusion, "
                "or astar_spline"
            )
        self.config = config
        self.baseline = baseline
        if baseline == "diffusion" and diffusion_controller is None:
            raise ValueError("diffusion baseline requires a controller")
        if baseline == "astar_spline" and nominal_planner is None:
            raise ValueError("astar_spline baseline requires a nominal planner")
        if baseline != "astar_spline" and nominal_planner is not None:
            raise ValueError(
                "nominal_planner is only valid for the astar_spline baseline"
            )
        requested_controller = diffusion_controller
        # Do not sample a Diffusion plan during the reset used to compile the
        # CUDA graph.  The first real plan is created after the final reset.
        self.diffusion_controller = None
        self.backend = MJWarpBackend(config, worlds=worlds)
        self.device = self.backend.device
        self.worlds = self.backend.worlds
        self.supervisor = BatchedSupervisor(
            device=str(self.device), curriculum=config.curriculum
        )
        self.nominal_planner = nominal_planner
        if (
            self.nominal_planner is not None
            and hasattr(self.nominal_planner, "worlds")
            and int(self.nominal_planner.worlds) != self.worlds
        ):
            raise ValueError(
                "nominal planner world count differs from environment"
            )
        self.safety = TaskSpaceSafety(self.supervisor)
        self.ik = BatchedAnalyticIK(
            device=str(self.device),
            linear_samples=config.ik.linear_samples,
        )
        self.state = BatchedTaskState(
            self.backend, self.supervisor, config
        )
        self.curriculum_progress = min(
            max(float(curriculum_progress), 0.0), 1.0
        )
        self.ctrl_velocity = torch.zeros(
            (self.worlds, 16), device=self.device
        )
        self.ctrl_acceleration = torch.zeros_like(self.ctrl_velocity)
        self.last_certified_ctrl = torch.zeros_like(self.ctrl_velocity)
        # Keep the rate-limited nominal cursor separate from the physical
        # ``data.ctrl`` tensor.  The latter contains the previous TQC residual;
        # feeding it back as the next nominal would absorb that residual into
        # IK and then add it a second time on the following tick.
        self.last_nominal_ctrl = torch.zeros_like(self.ctrl_velocity)
        self.actuator_residual = BatchedActuatorResidualMapper(
            self.worlds,
            self.backend.ctrl_range[:, 0],
            self.backend.ctrl_range[:, 1],
            device=self.device,
            dtype=torch.float32,
        )
        self.current_command = self.supervisor.at(
            torch.zeros(self.worlds, device=self.device, dtype=torch.long),
            torch.zeros((self.worlds, 2), device=self.device),
            self.state.socket_angle_deg,
            self.state.socket_mouth[:, :2],
        )
        self.last_certified_command = self.current_command.clone()
        self.diffusion_anchor_blend_source = self.current_command.clone()
        self.diffusion_anchor_blend_step = torch.zeros(
            self.worlds, device=self.device, dtype=torch.long
        )
        self.diffusion_anchor_takeover_active = torch.zeros(
            self.worlds, device=self.device, dtype=torch.bool
        )
        self.obs_history = torch.zeros(
            (self.worlds, 2, OBS_DIM), device=self.device
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.runtime.seed)
        self._all = torch.ones(
            self.worlds, device=self.device, dtype=torch.bool
        )
        self.diffusion_active = torch.zeros(
            self.worlds, device=self.device, dtype=torch.bool
        )
        self.reset(self._all)
        if config.physics.cuda_graph:
            self.backend.prepare_cuda_graph()
            # Graph warmup performs integrations; restore the exact reset.
            self.reset(self._all)
        self.diffusion_controller = requested_controller
        if requested_controller is not None:
            if int(requested_controller.worlds) != self.worlds:
                raise ValueError(
                    "Diffusion controller world count differs from environment"
                )
            # Pickup is scripted, so do not spend a Diffusion sample before
            # any world reaches its measured lifted-spawn handoff.
            requested_controller.reset(self._all, None)

    def _certify_scripted_prefix(
        self,
        spawn_xy,
        socket_angle,
        socket_mouth_xy,
    ):
        """Certify every command through the measured lifted-spawn handoff.

        Checking only PREGRASP allowed rare near-boundary spawns to pass reset
        and fail during CLOSE.  This finite gate follows the exact continuous
        0..175 pickup/lift prefix with the previous IK solution as its branch
        seed.  It returns the step-zero controls used by the physics reset;
        later solutions are qualification-only.
        """

        torch = _torch()
        count = int(spawn_xy.shape[0])
        previous = torch.zeros(
            (count, 16), device=self.device, dtype=torch.float32
        )
        feasible = torch.ones(
            count, device=self.device, dtype=torch.bool
        )
        initial_command = None
        initial_ctrl = None
        initial_passive = None
        for step_index in range(
            self.supervisor.diffusion_handoff_step + 1
        ):
            step = torch.full(
                (count,),
                step_index,
                device=self.device,
                dtype=torch.long,
            )
            command = self.supervisor.at(
                step,
                spawn_xy,
                socket_angle,
                socket_mouth_xy,
            )
            targets = command.fingertip_targets(
                self.supervisor.grasp_height_at(step)
            )
            result = self.ik.solve(targets, previous)
            finite = (
                torch.isfinite(targets).all(dim=(1, 2))
                & torch.isfinite(result.ctrl).all(dim=1)
                & torch.isfinite(result.passive_q).flatten(1).all(dim=1)
            )
            valid = result.success & finite
            if step_index == 0:
                initial_command = command
                initial_ctrl = result.ctrl.clone()
                initial_passive = result.passive_q.clone()
            feasible &= valid
            previous = torch.where(
                valid[:, None], result.ctrl, previous
            )
        if (
            initial_command is None
            or initial_ctrl is None
            or initial_passive is None
        ):
            raise RuntimeError("scripted-prefix IK gate produced no command")
        return (
            feasible,
            initial_command,
            initial_ctrl,
            initial_passive,
        )

    @staticmethod
    def _assign_command_rows(
        destination: CommandBatch,
        destination_rows,
        source: CommandBatch,
        source_rows,
    ) -> None:
        for name in (
            "tip",
            "rotation",
            "twist",
            "material",
            "distance",
            "phase",
            "engaged",
        ):
            getattr(destination, name)[destination_rows] = getattr(
                source, name
            )[source_rows]

    def _sample_reachable(self, mask):
        torch = _torch()
        spawn = self.state.spawn_xy.clone()
        socket_angle = self.state.socket_angle_deg.clone()
        socket_mouth_xy = self.state.socket_mouth[:, :2].clone()
        pending = mask.clone()
        final_command = self.current_command.clone()
        final_ctrl = torch.zeros(
            (self.worlds, 16),
            device=self.device,
            dtype=torch.float32,
        )
        final_passive = None
        # Preserve the declared 60->100 mm curriculum.  At full difficulty,
        # reachability is enforced only by full-prefix rejection and never by
        # silently shrinking or reshaping the final 100 mm task distribution.
        spawn_radius, fixed_socket_angle_deg = (
            self.supervisor.curriculum_parameters(
                self.curriculum_progress
            )
        )
        if abs(
            fixed_socket_angle_deg
            - self.supervisor.fixed_socket_angle_deg
        ) > 1.0e-9:
            raise RuntimeError("curriculum changed the fixed socket angle")
        if (
            self.curriculum_progress >= 1.0 - 1.0e-12
            and abs(float(spawn_radius) - 0.100) > 1.0e-9
        ):
            raise RuntimeError(
                "full-difficulty prefix gate requires the 100 mm spawn radius"
            )
        spawn_radius = float(spawn_radius)
        for _attempt in range(RESET_IK_RESAMPLE_BATCHES):
            slots = torch.nonzero(
                pending, as_tuple=False
            ).flatten()
            needed = int(slots.numel())
            if needed == 0:
                if final_passive is None:
                    raise RuntimeError(
                        "prefix gate completed without passive controls"
                    )
                return (
                    spawn,
                    socket_angle,
                    socket_mouth_xy,
                    final_command,
                    final_ctrl,
                    final_passive,
                )
            mouth_proposal = self.supervisor.sample_socket_mouth(
                needed,
                generator=self.generator,
            )
            angle_proposal = self.supervisor.fixed_angle_batch(needed)
            proposal = self.supervisor.sample_spawn(
                needed,
                socket_mouth_xy=mouth_proposal,
                socket_angle_deg=angle_proposal,
                radius_m=spawn_radius,
                generator=self.generator,
            )
            (
                accepted,
                command,
                initial_ctrl,
                initial_passive,
            ) = self._certify_scripted_prefix(
                proposal,
                angle_proposal,
                mouth_proposal,
            )
            if final_passive is None:
                final_passive = torch.zeros(
                    (self.worlds, *initial_passive.shape[1:]),
                    device=self.device,
                    dtype=initial_passive.dtype,
                )
            accepted_slots = slots[accepted]
            spawn[accepted_slots] = proposal[accepted]
            socket_angle[accepted_slots] = angle_proposal[accepted]
            socket_mouth_xy[accepted_slots] = mouth_proposal[accepted]
            self._assign_command_rows(
                final_command,
                accepted_slots,
                command,
                accepted,
            )
            final_ctrl[accepted_slots] = initial_ctrl[accepted]
            final_passive[accepted_slots] = initial_passive[accepted]
            pending[accepted_slots] = False
        remaining = int(pending.sum().item())
        raise RuntimeError(
            "full-prefix IK rejection sampler exhausted "
            f"{RESET_IK_RESAMPLE_BATCHES} batches with "
            f"{remaining} world(s) still unreachable"
        )

    def set_curriculum_progress(self, progress: float) -> None:
        self.curriculum_progress = min(max(float(progress), 0.0), 1.0)

    def _reset_nominal_planner(self, mask) -> None:
        """Build the exact supervisor endpoints for an A* ALIGN plan."""

        if self.baseline != "astar_spline":
            return
        if self.nominal_planner is None:
            raise RuntimeError("astar_spline nominal planner is unavailable")
        torch = _torch()
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        handoff_step = torch.full(
            (self.worlds,),
            self.supervisor.diffusion_handoff_step,
            device=self.device,
            dtype=torch.long,
        )
        preinsert_step = torch.full_like(
            handoff_step, self.supervisor.preinsert_gate_step
        )
        handoff = self.supervisor.at(
            handoff_step,
            self.state.spawn_xy,
            self.state.socket_angle_deg,
            self.state.socket_mouth[:, :2],
        )
        preinsert = self.supervisor.at(
            preinsert_step,
            self.state.spawn_xy,
            self.state.socket_angle_deg,
            self.state.socket_mouth[:, :2],
        )
        # Planning failure is deliberately fatal here.  A straight supervisor
        # fallback would silently turn a failed A* episode into an expert one.
        self.nominal_planner.reset(
            mask,
            start_command=handoff,
            goal_command=preinsert,
            socket_mouth=self.state.socket_mouth,
            socket_axis=self.state.socket_axis,
        )

    def reset(self, mask=None):
        torch = _torch()
        if mask is None:
            mask = self._all
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if not mask.any():
            return self.obs_history[:, -1]
        (
            spawn,
            socket_angle,
            socket_mouth_xy,
            command,
            ctrl,
            passive,
        ) = self._sample_reachable(mask)
        socket_position, socket_quaternion = (
            self.supervisor.socket_mocap_pose(
                socket_angle, socket_mouth_xy
            )
        )
        # PyTorch 2.7.1's deterministic CUDA advanced-index implementation
        # asserts internally on Blackwell for vector RHS writes such as
        # qpos[ids] = qpos0.  Every reset index below is unique, so using the
        # regular kernel for this tightly scoped state update remains
        # deterministic while avoiding that implementation defect.
        with unique_cuda_index_writes(
            torch, device_type=self.device.type
        ):
            self.backend.reset_worlds(
                mask,
                spawn_xy=spawn,
                socket_position=socket_position,
                socket_quaternion=socket_quaternion,
                ctrl=ctrl,
                passive_q=passive,
            )
            self.ctrl_velocity[mask] = 0.0
            self.ctrl_acceleration[mask] = 0.0
            self.last_certified_ctrl[mask] = ctrl[mask]
            self.last_nominal_ctrl[mask] = ctrl[mask]
            self.current_command = _select_command(
                mask, command, self.current_command
            )
            self.last_certified_command = _select_command(
                mask, command, self.last_certified_command
            )
            self.diffusion_anchor_blend_source = _select_command(
                mask, command, self.diffusion_anchor_blend_source
            )
            self.diffusion_anchor_blend_step[mask] = 0
            self.diffusion_anchor_takeover_active[mask] = False
            self.state.reset(
                mask,
                spawn,
                socket_angle,
                socket_mouth_xy,
                self.current_command,
            )
            observation = self.state.observation()
            self.obs_history[mask, 0] = observation[mask]
            self.obs_history[mask, 1] = observation[mask]
            self.diffusion_active[mask] = False
            self.actuator_residual.reset(mask)
            if self.diffusion_controller is not None:
                self.diffusion_controller.reset(mask, None)
        self._reset_nominal_planner(mask)
        return observation

    def _tqc_controller_state(
        self, blend_active, blend_progress
    ):
        """Expose queued-plan and handover state hidden from physical obs."""

        torch = _torch()
        cursor = torch.zeros(
            self.worlds, device=self.device, dtype=torch.float32
        )
        controller = self.diffusion_controller
        if controller is not None:
            denominator = max(float(controller.replan_interval), 1.0)
            cursor.fill_(
                min(float(controller.cursor), denominator) / denominator
            )
            cursor = torch.where(
                controller.override_active,
                controller.override_cursor.to(torch.float32) / denominator,
                cursor,
            )
        return torch.stack(
            (
                cursor.clamp(0.0, 1.0),
                blend_active.to(torch.float32),
                blend_progress.to(torch.float32).clamp(0.0, 1.0),
            ),
            dim=-1,
        )

    def _tqc_observation(
        self, nominal_ctrl, executable_mask, controller_state
    ):
        """Build the 242-D residual state while preserving base OBS_DIM=191."""

        from .tqc import build_tqc_observation

        torch = _torch()
        low = self.backend.ctrl_range[:, 0]
        high = self.backend.ctrl_range[:, 1]
        midpoint = 0.5 * (low + high)
        halfspan = (0.5 * (high - low)).clamp_min(1.0e-9)
        nominal = ((nominal_ctrl - midpoint) / halfspan).clamp(-1.0, 1.0)
        previous = (
            self.actuator_residual.residual
            / self.actuator_residual.limits[None, :]
        ).clamp(-1.0, 1.0)
        return build_tqc_observation(
            self.obs_history[:, -1],
            nominal,
            previous,
            executable_mask,
            controller_state,
        )

    def _residual_authority(
        self,
        learned_active,
        anchor_guard,
        candidate_success,
        safety_stop,
    ):
        enabled = (
            learned_active
            & ~anchor_guard
            & candidate_success
            & ~safety_stop
        )
        if self.baseline == "astar_spline":
            # The global A*--IK ablation is deliberately residual-free.
            enabled = _torch().zeros_like(enabled)
        return enabled

    def _rate_limit(self, desired):
        torch = _torch()
        # Rate-limit the certified IK stream, not the previously applied
        # (nominal + residual) command.  This preserves the residual as a true
        # post-IK correction instead of integrating it into the nominal path.
        previous = self.last_nominal_ctrl
        dt = self.backend.control_dt
        alpha = 1.0 - math.exp(-2.0 * math.pi * 10.0 * dt)
        filtered = previous + alpha * (desired - previous)
        omega = min(max(2.0 * math.pi * 10.0, 2.0), 12.0)
        linear_mask = torch.zeros(16, device=self.device, dtype=torch.bool)
        linear_mask[torch.as_tensor((0, 3, 6, 9), device=self.device)] = True
        max_step = torch.where(linear_mask, 0.004, 0.15)
        acceleration_limit = torch.where(linear_mask, 4.0, 8.0)
        jerk_limit = torch.where(linear_mask, 40.0, 80.0)
        error = filtered - previous
        desired_acceleration = (
            omega * omega * error - 2.0 * omega * self.ctrl_velocity
        )
        acceleration = torch.maximum(
            torch.minimum(
                desired_acceleration,
                self.ctrl_acceleration + jerk_limit * dt,
            ),
            self.ctrl_acceleration - jerk_limit * dt,
        )
        acceleration = torch.maximum(
            torch.minimum(acceleration, acceleration_limit), -acceleration_limit
        )
        velocity = self.ctrl_velocity + acceleration * dt
        velocity_limit = max_step / dt
        velocity = torch.maximum(
            torch.minimum(velocity, velocity_limit), -velocity_limit
        )
        value = previous + velocity * dt
        crossed = (error != 0.0) & ((value - filtered) * error > 0.0)
        value = torch.where(crossed, filtered, value)
        value = torch.maximum(
            torch.minimum(value, self.backend.ctrl_range[:, 1]),
            self.backend.ctrl_range[:, 0],
        )
        realized_velocity = (value - previous) / dt
        self.ctrl_acceleration = (
            (realized_velocity - self.ctrl_velocity) / dt
        )
        self.ctrl_acceleration = torch.maximum(
            torch.minimum(self.ctrl_acceleration, acceleration_limit),
            -acceleration_limit,
        )
        self.ctrl_velocity = realized_velocity
        self.last_nominal_ctrl.copy_(value)
        return value

    def _diffusion_action(
        self, anchor, external_action=None, *, active=None
    ):
        if self.baseline == "expert":
            return None
        if active is not None and not bool(active.any()):
            return None
        if self.baseline == "external_diffusion":
            if external_action is None:
                raise ValueError("external_diffusion needs a [world,14] action")
            return external_action
        return self.diffusion_controller.act(self.obs_history)

    def _schedule_next_step(self):
        """Advance only after each measured task-space gate is satisfied."""

        torch = _torch()
        proposed = (self.state.step_count + 1).clamp(
            max=self.supervisor.length - 1
        )
        pickup_release_ready = (
            ~self.state.holder_active
            & self.state.grasp_latched
            & self.state.all_four_contact
            & (
                self.state.all_four_contact_stable_steps
                >= PICKUP_RELEASE_GATE_STABLE_STEPS
            )
        )
        pickup_release_hold = (
            self.state.step_count == PICKUP_RELEASE_GATE_STEP
        ) & ~pickup_release_ready
        leaving_scripted_prefix = (
            self.state.step_count
            == self.supervisor.diffusion_handoff_step
        )
        handoff_hold = (
            leaving_scripted_prefix
            & ~self.state.diffusion_handoff_ready()
        )
        leaving_preinsert_gate = (
            self.state.step_count
            == self.supervisor.preinsert_gate_step
        )
        preinsert_hold = (
            leaving_preinsert_gate
            & ~self.state.preinsert_alignment_ready()
        )
        leaving_capture_gate = (
            self.state.step_count
            == self.supervisor.capture_gate_step
        )
        capture_hold = (
            leaving_capture_gate
            & ~(
                self.state.shaft_captured
                & self.state.capture_alignment_ready()
            )
        )
        leaving_thread_gate = (
            self.state.step_count
            == self.supervisor.thread_gate_step
        )
        thread_hold = (
            leaving_thread_gate
            & ~self.state.thread_engagement_ready()
        )
        leaving_regrasp_gate = (
            self.state.step_count
            == self.supervisor.thread_regrasp_step
        )
        regrasp_hold = (
            leaving_regrasp_gate
            & ~self.state.thread_engaged
        )
        contact_recovery_hold = self.state.contact_recovery_pending()
        hold = (
            pickup_release_hold
            | handoff_hold
            | preinsert_hold
            | capture_hold
            | thread_hold
            | regrasp_hold
            | contact_recovery_hold
        )
        return (
            torch.where(hold, self.state.step_count, proposed),
            pickup_release_hold,
            thread_hold,
            handoff_hold,
            preinsert_hold,
            capture_hold,
            regrasp_hold,
            contact_recovery_hold,
        )

    def _limit_insert_anchor(self, anchor: CommandBatch):
        """Limit every pre-thread bore command to 1 mm beyond measurement."""

        torch = _torch()
        insert = (
            (
                (anchor.phase == int(TaskPhase.INSERT))
                | (anchor.phase == int(TaskPhase.ENGAGE))
            )
        ) & ~self.state.thread_engaged
        axis = self.state.socket_axis
        mouth = self.state.socket_mouth
        anchor_depth = torch.einsum(
            "bj,bj->b", axis, anchor.tip - mouth
        )
        depth_limit = self.state.insertion_depth() + 0.001
        limited_depth = torch.minimum(anchor_depth, depth_limit)
        correction = torch.where(
            insert, limited_depth - anchor_depth, torch.zeros_like(anchor_depth)
        )
        return CommandBatch(
            tip=anchor.tip + correction[:, None] * axis,
            rotation=anchor.rotation,
            twist=anchor.twist,
            material=anchor.material,
            distance=anchor.distance,
            phase=anchor.phase,
            engaged=anchor.engaged,
        ), insert

    def _apply_preinsert_feedback(
        self,
        anchor: CommandBatch,
        active,
    ) -> CommandBatch:
        """Bias the command so the measured shaft, not just its anchor, aligns."""

        torch = _torch()
        active = torch.as_tensor(
            active, device=self.device, dtype=torch.bool
        )
        measured = self.state.tip_socket()
        nominal = torch.einsum(
            "bji,bj->bi",
            self.state.socket_frame,
            anchor.tip - self.state.socket_mouth,
        )
        gain = float(self.state.task.preinsert_feedback_gain)
        target_xy = gain * (nominal[:, :2] - measured[:, :2])
        xy_norm = torch.linalg.vector_norm(
            target_xy, dim=-1, keepdim=True
        )
        xy_limit = float(
            self.state.task.preinsert_command_lateral_limit_m
        )
        target_xy = target_xy * torch.minimum(
            torch.ones_like(xy_norm),
            torch.full_like(xy_norm, xy_limit)
            / xy_norm.clamp_min(1.0e-9),
        )
        target_depth = gain * (nominal[:, 2] - measured[:, 2])
        depth_limit = float(
            self.state.task.preinsert_feedback_depth_limit_m
        )
        target_depth = target_depth.clamp(-depth_limit, depth_limit)
        target = torch.cat((target_xy, target_depth[:, None]), dim=-1)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        previous = self.state.preinsert_feedback_offset_socket
        slew = torch.as_tensor(
            (
                self.state.task.preinsert_feedback_lateral_slew_m,
                self.state.task.preinsert_feedback_lateral_slew_m,
                self.state.task.preinsert_feedback_depth_slew_m,
            ),
            device=self.device,
            dtype=previous.dtype,
        )
        updated = previous + (target - previous).clamp(-slew, slew)
        updated = torch.where(
            active[:, None], updated, torch.zeros_like(updated)
        )
        previous.copy_(updated)
        result = anchor.clone()
        result.tip = result.tip + torch.einsum(
            "bij,bj->bi", self.state.socket_frame, updated
        )
        return result

    def _project_prethread_candidate(
        self,
        candidate: CommandBatch,
        anchor: CommandBatch,
    ) -> CommandBatch:
        """Hard-project the final learned command inside bore clearance."""

        torch = _torch()
        result = candidate.clone()
        prethread_insert = (
            (
                (result.phase == int(TaskPhase.INSERT))
                | (result.phase == int(TaskPhase.ENGAGE))
            )
            & ~self.state.thread_engaged
        )
        tip_socket = torch.einsum(
            "bji,bj->bi",
            self.state.socket_frame,
            result.tip - self.state.socket_mouth,
        )
        lateral = tip_socket[:, :2]
        lateral_norm = torch.linalg.vector_norm(
            lateral, dim=-1, keepdim=True
        )
        lateral_limit = float(
            self.state.task.preinsert_command_lateral_limit_m
        )
        lateral_scale = torch.minimum(
            torch.ones_like(lateral_norm),
            torch.full_like(lateral_norm, lateral_limit)
            / lateral_norm.clamp_min(1.0e-9),
        )
        projected_lateral = lateral * lateral_scale
        measured_depth = self.state.insertion_depth()
        projected_depth = torch.minimum(
            tip_socket[:, 2], measured_depth + 0.001
        )
        projected_socket = torch.cat(
            (projected_lateral, projected_depth[:, None]), dim=-1
        )
        projected_tip = self.state.socket_mouth + torch.einsum(
            "bij,bj->bi", self.state.socket_frame, projected_socket
        )
        result.tip = torch.where(
            prethread_insert[:, None], projected_tip, result.tip
        )
        # The bore requires a parallel shaft.  Policies may correct tilt in
        # ALIGN, but once INSERT begins the exact expert orientation is the
        # only safe command; rotation about the shaft resumes after thread
        # engagement through the helical controller.
        result.rotation = torch.where(
            prethread_insert[:, None, None],
            anchor.rotation,
            result.rotation,
        )
        # Before measured thread engagement, the learned policies may move
        # the screw but may not change contact material or the supervisor's
        # certified four-/two-finger holding regime.  In particular,
        # ``anchor.engaged`` is only an intent; it never authorizes a regrasp.
        holding = ~self.state.thread_engaged
        result.material = torch.where(
            holding[:, None], anchor.material, result.material
        )
        result.distance = torch.where(
            holding[:, None], anchor.distance, result.distance
        )
        result.twist = torch.where(
            holding, anchor.twist, result.twist
        )
        return result

    def _smooth_diffusion_to_anchor(
        self,
        baseline: CommandBatch,
        anchor: CommandBatch,
        takeover,
    ):
        """Blend every planned Diffusion-to-anchor takeover over 20 frames."""

        torch = _torch()
        takeover = torch.as_tensor(
            takeover, device=self.device, dtype=torch.bool
        )
        entering = takeover & ~self.diffusion_anchor_takeover_active
        self.diffusion_anchor_blend_source = _select_command(
            entering,
            self.current_command,
            self.diffusion_anchor_blend_source,
        )
        zeros = torch.zeros_like(self.diffusion_anchor_blend_step)
        previous_step = torch.where(
            entering, zeros, self.diffusion_anchor_blend_step
        )
        blend_step = torch.where(
            takeover,
            (previous_step + 1).clamp(
                max=DIFFUSION_TO_ANCHOR_BLEND_STEPS
            ),
            zeros,
        )
        progress = (
            blend_step.to(self.current_command.tip.dtype)
            / float(DIFFUSION_TO_ANCHOR_BLEND_STEPS)
        )
        blended = _blend_command(
            self.diffusion_anchor_blend_source,
            anchor,
            progress,
        )
        result = _select_command(takeover, blended, baseline)
        self.diffusion_anchor_blend_step.copy_(blend_step)
        self.diffusion_anchor_takeover_active.copy_(takeover)
        blend_active = takeover & (
            blend_step < DIFFUSION_TO_ANCHOR_BLEND_STEPS
        )
        return result, blend_active, progress

    def _command_action(
        self,
        previous: CommandBatch,
        following: CommandBatch,
    ):
        """Encode the command actually selected for this environment step."""

        torch = _torch()
        delta_position = torch.einsum(
            "bji,bj->bi",
            self.state.socket_frame,
            following.tip - previous.tip,
        )
        world_delta_rotation = (
            following.rotation
            @ previous.rotation.transpose(-1, -2)
        )
        socket_delta_rotation = (
            self.state.socket_frame.transpose(-1, -2)
            @ world_delta_rotation
            @ self.state.socket_frame
        )
        delta_rotation = rotation_log(socket_delta_rotation)
        material_delta = torch.remainder(
            following.material - previous.material + torch.pi,
            2.0 * torch.pi,
        ) - torch.pi
        return torch.cat(
            (
                delta_position,
                delta_rotation,
                material_delta,
                following.distance - previous.distance,
            ),
            dim=-1,
        )

    def _astar_nominal_command(
        self,
        anchor,
        next_step,
        learned_active,
        safety_guard=None,
    ):
        """Select a prepared global A* command only during ALIGN."""

        torch = _torch()
        if safety_guard is None:
            safety_guard = torch.zeros_like(learned_active)
        else:
            safety_guard = torch.as_tensor(
                safety_guard, device=self.device, dtype=torch.bool
            )
            if tuple(safety_guard.shape) != (self.worlds,):
                raise ValueError(
                    f"A* safety guard must have shape {(self.worlds,)}"
                )
        planner_active = (
            learned_active
            & (anchor.phase == int(TaskPhase.ALIGN))
            & ~safety_guard
        )
        if not bool(planner_active.any()):
            return anchor, planner_active
        if self.nominal_planner is None:
            raise RuntimeError("astar_spline nominal planner is unavailable")
        planned = self.nominal_planner.command(
            anchor, next_step=next_step
        )
        if not isinstance(planned, CommandBatch):
            raise TypeError("nominal planner command must be a CommandBatch")
        expected = {
            "tip": (self.worlds, 3),
            "rotation": (self.worlds, 3, 3),
            "twist": (self.worlds,),
            "material": (self.worlds, 4),
            "distance": (self.worlds, 4),
            "phase": (self.worlds,),
            "engaged": (self.worlds,),
        }
        for name, shape in expected.items():
            value = getattr(planned, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"nominal planner {name} shape {tuple(value.shape)} "
                    f"!= {shape}"
                )
            if value.device != self.device:
                raise ValueError(
                    f"nominal planner {name} is on {value.device}, "
                    f"expected {self.device}"
                )
        for name in ("tip", "rotation", "twist", "material", "distance"):
            if not bool(torch.isfinite(getattr(planned, name)).all()):
                raise FloatingPointError(
                    f"nominal planner {name} contains NaN or infinity"
                )
        # Only the supervisor owns discrete phase and measured engagement
        # intent.  The path planner may choose continuous task-space fields.
        planned = planned.clone()
        planned.phase.copy_(anchor.phase)
        planned.engaged.copy_(anchor.engaged)
        return _select_command(planner_active, planned, anchor), planner_active

    def step(
        self,
        residual_action=None,
        *,
        diffusion_action=None,
        residual_policy=None,
    ) -> StepBatch:
        """Advance one control tick using Diffusion -> IK -> actuator residual.

        ``residual_policy`` is invoked only after the current certified nominal
        IK command and exact 16-channel safety mask have been constructed.  A
        tensor ``residual_action`` remains available for Diffusion--IK-only
        execution, preflight, and deterministic tests.
        """

        torch = _torch()
        if residual_action is not None and residual_policy is not None:
            raise ValueError(
                "provide residual_action or residual_policy, not both"
            )
        (
            next_step,
            pickup_release_schedule_hold,
            thread_schedule_hold,
            handoff_schedule_hold,
            preinsert_schedule_hold,
            capture_schedule_hold,
            regrasp_schedule_hold,
            contact_recovery_hold,
        ) = self._schedule_next_step()
        # If a measured guard clears before its transition has reached the
        # anchor, keep that world at the boundary until all configured blend frames
        # have executed.  This prevents a short guard from producing a second
        # hard switch back to Diffusion.
        anchor_blend_continuation = (
            self.diffusion_anchor_takeover_active
            & (
                self.diffusion_anchor_blend_step
                < DIFFUSION_TO_ANCHOR_BLEND_STEPS
            )
        )
        next_step = torch.where(
            anchor_blend_continuation,
            self.state.step_count,
            next_step,
        )
        self.state.diffusion_handoff_hold_steps += (
            handoff_schedule_hold.to(torch.long)
        )
        learned_active = (
            next_step > self.supervisor.diffusion_handoff_step
        )
        gate_transition = (
            (
                self.state.step_count
                == PICKUP_RELEASE_GATE_STEP
            )
            | (
                self.state.step_count
                == self.supervisor.preinsert_gate_step
            )
            | (
                self.state.step_count
                == self.supervisor.capture_gate_step
            )
            | (
                self.state.step_count
                == self.supervisor.thread_gate_step
            )
            | (
                self.state.step_count
                == self.supervisor.thread_regrasp_step
            )
        )
        safety_gate_hold = (
            pickup_release_schedule_hold
            | preinsert_schedule_hold
            | capture_schedule_hold
            | thread_schedule_hold
            | regrasp_schedule_hold
            | contact_recovery_hold
        )
        preinsert_guard = (
            preinsert_schedule_hold
            | (
                self.state.step_count
                == self.supervisor.preinsert_gate_step
            )
        )
        safety_gate_guard = safety_gate_hold | gate_transition
        regrasp_phase_guard = torch.zeros_like(safety_gate_guard)
        for phase in REGRASP_PHASES:
            regrasp_phase_guard |= (
                self.supervisor.phase[next_step] == phase
            )
        safety_gate_guard |= regrasp_phase_guard
        # An ENGAGE command is still a two-finger hold until the measured
        # engagement latch is true.  Neither learned policy may turn this
        # intent bit into permission to release or regrasp.
        unmeasured_engage = (
            (
                self.supervisor.phase[next_step]
                == int(TaskPhase.ENGAGE)
            )
            & ~self.state.thread_engaged
        )
        safety_gate_guard |= unmeasured_engage
        diffusion_anchor_guard = (
            safety_gate_guard | anchor_blend_continuation
        )
        newly_active = learned_active & ~self.diffusion_active
        anchor = self.supervisor.at(
            next_step,
            self.state.spawn_xy,
            self.state.socket_angle_deg,
            self.state.socket_mouth[:, :2],
        )
        feedback_active = (
            preinsert_schedule_hold
            | (
                (
                    (anchor.phase == int(TaskPhase.INSERT))
                    | (anchor.phase == int(TaskPhase.ENGAGE))
                )
                & ~self.state.thread_engaged
            )
        )
        anchor = self._apply_preinsert_feedback(
            anchor, feedback_active
        )
        anchor, _insert_anchor = self._limit_insert_anchor(anchor)
        anchor_blend_released = (
            learned_active
            & self.diffusion_anchor_takeover_active
            & ~diffusion_anchor_guard
        )
        controller_reset = newly_active | anchor_blend_released
        if (
            self.diffusion_controller is not None
            and bool(controller_reset.any())
        ):
            self.diffusion_controller.reset(
                controller_reset, self.obs_history
            )
        planner_active = torch.zeros_like(learned_active)
        if self.baseline == "astar_spline":
            if diffusion_action is not None:
                raise ValueError(
                    "astar_spline does not accept a Diffusion action"
                )
            proposed_action = None
            baseline, planner_active = self._astar_nominal_command(
                anchor,
                next_step,
                learned_active,
                safety_guard=diffusion_anchor_guard,
            )
        else:
            proposed_action = self._diffusion_action(
                anchor,
                external_action=diffusion_action,
                active=learned_active,
            )
        if proposed_action is not None:
            proposed_action = torch.as_tensor(
                proposed_action, device=self.device, dtype=torch.float32
            )
            if proposed_action.shape != (
                self.worlds,
                ACTION_DIM,
            ):
                raise ValueError(
                    "Diffusion action must have shape "
                    f"{(self.worlds, ACTION_DIM)}"
                )
            self.backend.accumulate_external_health(proposed_action)
            proposed_action = torch.nan_to_num(
                proposed_action, nan=0.0, posinf=1.0, neginf=-1.0
            ).clamp(-1.0, 1.0)
        if proposed_action is None:
            if self.baseline != "astar_spline":
                baseline = anchor
        else:
            diffusion_candidate = self.safety.diffusion(
                self.current_command,
                anchor,
                proposed_action,
                socket_frame=self.state.socket_frame,
                socket_axis=self.state.socket_axis,
                socket_mouth=self.state.socket_mouth,
            )
            baseline = _select_command(
                learned_active, diffusion_candidate, anchor
            )
        diffusion_anchor_takeover = (
            learned_active
            & diffusion_anchor_guard
            & (proposed_action is not None)
        )
        (
            baseline,
            diffusion_anchor_blend_active,
            diffusion_anchor_blend_progress,
        ) = self._smooth_diffusion_to_anchor(
            baseline,
            anchor,
            diffusion_anchor_takeover,
        )
        # The nominal task-space command is projected before analytical IK.
        # Neither Diffusion nor A* can modify the certified command after IK.
        candidate = self._project_prethread_candidate(baseline, anchor)

        grasp_height = self.supervisor.grasp_height_at(next_step)
        candidate_ik = self.ik.solve(
            candidate.fingertip_targets(grasp_height),
            self.backend.tensor("ctrl"),
        )
        anchor_ik = self.ik.solve(
            anchor.fingertip_targets(grasp_height),
            self.backend.tensor("ctrl"),
        )
        (
            use_candidate,
            use_anchor,
            safety_stop,
            planner_ik_failure,
        ) = _select_ik_routes(
            candidate_ik.success,
            anchor_ik.success,
            planner_active,
        )
        feasible_fraction = use_candidate.to(torch.float32)
        self.state.ik_anchor_fallbacks += use_anchor.to(torch.long)
        self.state.ik_safety_stops += safety_stop.to(torch.long)
        self.state.ik_failures += (~use_candidate).to(torch.long)

        command = _select_command(use_candidate, candidate, anchor)
        command = _select_command(
            safety_stop, self.last_certified_command, command
        )
        if proposed_action is None:
            executed_diffusion = torch.zeros(
                (self.worlds, ACTION_DIM),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            diffusion_executable = (
                learned_active
                & ~diffusion_anchor_guard
                & use_candidate
            )
            executed_diffusion = torch.where(
                diffusion_executable[:, None],
                proposed_action,
                torch.zeros_like(proposed_action),
            )
        desired_ctrl = torch.where(
            use_candidate[:, None],
            candidate_ik.ctrl,
            torch.where(
                use_anchor[:, None], anchor_ik.ctrl, self.last_certified_ctrl
            ),
        )
        certified = use_candidate | use_anchor
        self.last_certified_ctrl = torch.where(
            certified[:, None], desired_ctrl, self.last_certified_ctrl
        )
        self.last_certified_command = _select_command(
            certified, command, self.last_certified_command
        )
        # Demonstrations must contain the task-space command that actually
        # generated the transition, including measured feedback, schedule
        # holds and IK fallback.  A nominal table delta is wrong whenever a
        # per-world safety gate keeps or corrects its anchor.
        expert_action = self._command_action(
            self.current_command, command
        )
        nominal_ctrl = self._rate_limit(desired_ctrl)

        # Residual authority exists only for a certified Diffusion command.
        # Scripted pickup, every measured gate/anchor takeover, regrasp, IK
        # fallback and safety stop clear all 16 carried actuator corrections.
        residual_enabled = self._residual_authority(
            learned_active,
            diffusion_anchor_guard,
            use_candidate,
            safety_stop,
        )
        effective_action_mask = residual_enabled[:, None].expand(
            -1, RESIDUAL_ACTION_DIM
        )
        controller_state = self._tqc_controller_state(
            diffusion_anchor_blend_active,
            diffusion_anchor_blend_progress,
        )
        tqc_observation = self._tqc_observation(
            nominal_ctrl,
            effective_action_mask,
            controller_state,
        )
        if residual_policy is not None:
            residual = residual_policy(
                tqc_observation, effective_action_mask
            )
        elif residual_action is None:
            residual = torch.zeros(
                (self.worlds, RESIDUAL_ACTION_DIM),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            residual = residual_action
        residual = torch.as_tensor(
            residual, device=self.device, dtype=torch.float32
        )
        if residual.shape != (self.worlds, RESIDUAL_ACTION_DIM):
            raise ValueError(
                f"residual action must be {(self.worlds, RESIDUAL_ACTION_DIM)}"
            )
        self.backend.accumulate_external_health(residual)
        residual = torch.nan_to_num(
            residual, nan=0.0, posinf=1.0, neginf=-1.0
        ).clamp(-1.0, 1.0)
        residual_step = self.actuator_residual.apply(
            nominal_ctrl,
            residual,
            effective_action_mask,
        )
        accepted_residual = (
            residual_step.requested_residual
            / self.actuator_residual.limits[None, :]
        )
        realised_normalised_residual = (
            residual_step.realised_residual
            / self.actuator_residual.limits[None, :]
        )
        self.backend.tensor("ctrl").copy_(residual_step.applied_ctrl)
        self.current_command = command
        self.state.set_command(command)
        self.backend.step_control()
        self.state.step_count.copy_(next_step)
        self.diffusion_active.copy_(learned_active)
        self.state.elapsed_steps.add_(1)
        self.state.post_physics()
        reward, done, info = self.state.reward_done(
            realised_normalised_residual, certified, safety_stop
        )
        info["tqc_observation"] = tqc_observation.clone()
        info["residual_action_mask"] = effective_action_mask.clone()
        # Replay action is the masked, clipped normalised target requested by
        # the actor.  Physical diagnostics use the post-slew realised offset.
        info["accepted_residual_action"] = accepted_residual.clone()
        info["executed_residual_action"] = accepted_residual.clone()
        info["realised_actuator_residual"] = (
            residual_step.realised_residual.clone()
        )
        info["normalised_realised_actuator_residual"] = (
            realised_normalised_residual.clone()
        )
        info["nominal_ik_ctrl"] = nominal_ctrl.clone()
        info["applied_actuator_ctrl"] = residual_step.applied_ctrl.clone()
        info["residual_ctrl_saturation_mask"] = (
            residual_step.ctrl_saturation_mask.clone()
        )
        info["residual_action_clipped_mask"] = (
            residual_step.action_clipped_mask.clone()
        )
        info["executed_diffusion_action"] = executed_diffusion.clone()
        info["astar_planner_active"] = planner_active.clone()
        info["planner_ik_failure"] = planner_ik_failure.clone()
        info["preinsert_guard_active"] = (
            preinsert_guard.clone()
        )
        info["safety_gate_guard_active"] = (
            safety_gate_guard.clone()
        )
        info["diffusion_anchor_guard_active"] = (
            diffusion_anchor_guard.clone()
        )
        info["diffusion_anchor_blend_active"] = (
            diffusion_anchor_blend_active.clone()
        )
        info["diffusion_anchor_blend_progress"] = (
            diffusion_anchor_blend_progress.clone()
        )
        info["capture_alignment_hold"] = (
            capture_schedule_hold.clone()
        )
        info["pickup_release_hold"] = (
            pickup_release_schedule_hold.clone()
        )
        info["thread_engagement_hold"] = (
            thread_schedule_hold.clone()
        )
        info["thread_regrasp_hold"] = (
            regrasp_schedule_hold.clone()
        )
        info["regrasp_phase_guard"] = regrasp_phase_guard.clone()
        info["contact_recovery_hold"] = (
            contact_recovery_hold.clone()
        )
        info["residual_active_dimensions"] = (
            effective_action_mask.sum(dim=-1)
        )
        info["diffusion_active"] = learned_active.clone()
        info["scripted_prefix_active"] = ~learned_active
        info["diffusion_handoff_ready"] = (
            self.state.diffusion_handoff_ready()
        )
        info["diffusion_handoff_step"] = torch.full_like(
            self.state.step_count,
            self.supervisor.diffusion_handoff_step,
        )
        info["expert_step"] = self.state.step_count.clone()
        info["desired_fingertip_distance_m"] = (
            self.state.desired_command.distance.clone()
        )
        handoff_position_error, handoff_orientation_error = (
            self.state.diffusion_handoff_errors()
        )
        info["diffusion_handoff_position_error_m"] = (
            handoff_position_error
        )
        info["diffusion_handoff_orientation_error_deg"] = (
            torch.rad2deg(handoff_orientation_error)
        )
        info["diffusion_handoff_hold_steps"] = (
            self.state.diffusion_handoff_hold_steps.clone()
        )
        (
            preinsert_lateral_error,
            preinsert_axis_error,
            preinsert_depth_error,
            preinsert_tip_speed,
            preinsert_lateral_speed,
            preinsert_angular_speed,
        ) = self.state.preinsert_alignment_errors()
        info["preinsert_alignment_hold"] = (
            preinsert_schedule_hold.clone()
        )
        info["preinsert_lateral_error_m"] = preinsert_lateral_error
        info["preinsert_axis_error_deg"] = torch.rad2deg(
            preinsert_axis_error
        )
        info["preinsert_depth_error_m"] = preinsert_depth_error
        info["preinsert_tip_speed_m_s"] = preinsert_tip_speed
        info["preinsert_lateral_speed_m_s"] = preinsert_lateral_speed
        info["preinsert_angular_speed_deg_s"] = torch.rad2deg(
            preinsert_angular_speed
        )
        (
            capture_lateral_error,
            capture_axis_error,
            capture_depth_error,
            capture_tip_speed,
            capture_lateral_speed,
            capture_angular_speed,
        ) = self.state.capture_alignment_errors()
        info["capture_lateral_error_m"] = capture_lateral_error
        info["capture_axis_error_deg"] = torch.rad2deg(
            capture_axis_error
        )
        info["capture_depth_error_m"] = capture_depth_error
        info["capture_tip_speed_m_s"] = capture_tip_speed
        info["capture_lateral_speed_m_s"] = capture_lateral_speed
        info["capture_angular_speed_deg_s"] = torch.rad2deg(
            capture_angular_speed
        )
        (
            thread_lateral_error,
            thread_axis_error,
            thread_depth_error,
            thread_tip_speed,
            thread_lateral_speed,
            thread_angular_speed,
        ) = self.state.thread_engagement_errors()
        info["thread_depth_error_m"] = thread_depth_error
        info["thread_lateral_error_m"] = thread_lateral_error
        info["thread_axis_error_deg"] = torch.rad2deg(
            thread_axis_error
        )
        info["thread_tip_speed_m_s"] = thread_tip_speed
        info["thread_lateral_speed_m_s"] = thread_lateral_speed
        info["thread_angular_speed_deg_s"] = torch.rad2deg(
            thread_angular_speed
        )
        feedback_offset = (
            self.state.preinsert_feedback_offset_socket.clone()
        )
        info["preinsert_feedback_lateral_m"] = (
            torch.linalg.vector_norm(
                feedback_offset[:, :2], dim=-1
            )
        )
        info["preinsert_feedback_depth_m"] = feedback_offset[:, 2]
        observation = self.state.observation()
        self.backend.accumulate_external_health(observation, reward)
        # Preserve a clean replay/controller input until the next scheduled
        # health poll aborts the run and keeps the last-good checkpoint.
        observation = torch.nan_to_num(
            observation, nan=0.0, posinf=10.0, neginf=-10.0
        )
        reward = torch.nan_to_num(
            reward, nan=-50.0, posinf=100.0, neginf=-50.0
        )
        self.obs_history[:, 0] = self.obs_history[:, 1]
        self.obs_history[:, 1] = observation
        return StepBatch(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            ik_feasible_fraction=feasible_fraction,
            expert_action=expert_action,
        )

    def state_dict(self):
        """Return a rollout-continuation checkpoint for all vector worlds."""

        result = {
            "version": 4,
            "worlds": int(self.worlds),
            "backend": self.backend.state_dict(),
            "task": self.state.state_dict(),
            "ctrl_velocity": self.ctrl_velocity.detach().clone(),
            "ctrl_acceleration": self.ctrl_acceleration.detach().clone(),
            "last_certified_ctrl": self.last_certified_ctrl.detach().clone(),
            "last_nominal_ctrl": self.last_nominal_ctrl.detach().clone(),
            "actuator_residual": self.actuator_residual.state_dict(),
            "current_command": self.current_command.state_dict(),
            "last_certified_command": (
                self.last_certified_command.state_dict()
            ),
            "diffusion_anchor_blend_source": (
                self.diffusion_anchor_blend_source.state_dict()
            ),
            "diffusion_anchor_blend_step": (
                self.diffusion_anchor_blend_step.detach().clone()
            ),
            "diffusion_anchor_takeover_active": (
                self.diffusion_anchor_takeover_active.detach().clone()
            ),
            "obs_history": self.obs_history.detach().clone(),
            "diffusion_active": self.diffusion_active.detach().clone(),
            "generator_state": self.generator.get_state().clone(),
            "curriculum_progress": float(self.curriculum_progress),
            "diffusion_controller": None,
        }
        if self.diffusion_controller is not None:
            result["diffusion_controller"] = (
                self.diffusion_controller.state_dict()
            )
        return result

    def load_state_dict(self, state) -> None:
        """Restore a complete safe vector-step cursor in-place."""

        version = int(state.get("version", -1))
        if version not in (1, 2, 3, 4):
            raise ValueError("unsupported environment checkpoint version")
        if int(state.get("worlds", -1)) != self.worlds:
            raise ValueError(
                "environment checkpoint world count differs from allocation"
            )
        self.backend.load_state_dict(state.get("backend", {}))
        self.state.load_state_dict(state.get("task", {}))
        for name in (
            "ctrl_velocity",
            "ctrl_acceleration",
            "last_certified_ctrl",
            "obs_history",
            "diffusion_active",
        ):
            source = state.get(name)
            target = getattr(self, name)
            if source is None or tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"environment state {name!r} shape mismatch"
                )
            target.copy_(
                source.to(device=self.device, dtype=target.dtype)
            )
        self.current_command.load_state_dict(
            state.get("current_command", {})
        )
        self.last_certified_command.load_state_dict(
            state.get("last_certified_command", {})
        )
        if version >= 2:
            for name in (
                "diffusion_anchor_blend_step",
                "diffusion_anchor_takeover_active",
            ):
                source = state.get(name)
                target = getattr(self, name)
                if source is None or tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        f"environment state {name!r} shape mismatch"
                    )
                target.copy_(
                    source.to(device=self.device, dtype=target.dtype)
                )
            self.diffusion_anchor_blend_source.load_state_dict(
                state.get("diffusion_anchor_blend_source", {})
            )
        else:
            self.diffusion_anchor_blend_step.zero_()
            self.diffusion_anchor_takeover_active.zero_()
            self.diffusion_anchor_blend_source.load_state_dict(
                self.current_command.state_dict()
            )
        if version >= 3:
            self.actuator_residual.load_state_dict(
                state.get("actuator_residual", {})
            )
        else:
            self.actuator_residual.reset()
        if version >= 4:
            source = state.get("last_nominal_ctrl")
            if (
                source is None
                or tuple(source.shape) != tuple(self.last_nominal_ctrl.shape)
            ):
                raise ValueError(
                    "environment state 'last_nominal_ctrl' shape mismatch"
                )
            self.last_nominal_ctrl.copy_(
                source.to(
                    device=self.device,
                    dtype=self.last_nominal_ctrl.dtype,
                )
            )
        else:
            # v1/v2 carried no actuator residual.  v3 did, and its exact
            # realised offset is sufficient to recover the pre-residual
            # nominal command from the already-restored backend control.
            self.last_nominal_ctrl.copy_(
                self.backend.tensor("ctrl")
                - self.actuator_residual.residual
            )
        generator_state = state.get("generator_state")
        if generator_state is None:
            raise ValueError("environment checkpoint has no generator state")
        self.generator.set_state(generator_state.cpu())
        self.set_curriculum_progress(
            float(state.get("curriculum_progress", 1.0))
        )
        controller_state = state.get("diffusion_controller")
        if self.diffusion_controller is None:
            if controller_state is not None:
                raise ValueError(
                    "checkpoint has a Diffusion controller but environment does not"
                )
        else:
            if controller_state is None:
                raise ValueError(
                    "checkpoint is missing the Diffusion controller state"
                )
            self.diffusion_controller.load_state_dict(controller_state)

    def close(self) -> None:
        # Break task/kernel/backend references before releasing CUDA graph and
        # model allocations, so repeated smoke/evaluation runs in one process
        # do not retain an entire vector world.
        backend = self.backend
        self.diffusion_controller = None
        self.nominal_planner = None
        self.state = None
        self.ik = None
        self.safety = None
        self.supervisor = None
        self.current_command = None
        self.last_certified_command = None
        self.last_nominal_ctrl = None
        self.diffusion_anchor_blend_source = None
        self.diffusion_anchor_blend_step = None
        self.diffusion_anchor_takeover_active = None
        self.actuator_residual = None
        self.ctrl_velocity = None
        self.ctrl_acceleration = None
        self.last_certified_ctrl = None
        self.obs_history = None
        self.diffusion_active = None
        self.backend = None
        if backend is not None:
            backend.close()


__all__ = ["MJWarpScrewEnv", "StepBatch"]
