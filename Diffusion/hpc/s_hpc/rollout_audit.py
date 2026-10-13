"""Task-level certification shared by capacity probes and demo generation."""

from __future__ import annotations

import json
import math

from screw_control import TaskPhase


PHYSICAL_FAILURE_KEYS = (
    "nonfinite",
    "fell_below_table",
    "lateral_escape",
    "safety_stop",
    "contact_constraint_violation",
    "contact_violation_before_capture",
    "contact_violation_before_thread",
    "contact_violation_premature_regrasp",
    "contact_violation_material_drift",
)


class ExpertRolloutAudit:
    """Accumulate first causes without synchronizing the GPU every step."""

    def __init__(self, torch, *, worlds: int, device) -> None:
        self.torch = torch
        self.worlds = int(worlds)
        self.device = device
        self.any_failure = torch.zeros(
            self.worlds, device=device, dtype=torch.bool
        )
        self.any_truncation = torch.zeros_like(self.any_failure)
        self.first_failure_step = torch.full(
            (self.worlds,), -1, device=device, dtype=torch.long
        )
        self.first_truncation_step = torch.full_like(
            self.first_failure_step, -1
        )
        self.reason_seen = {
            key: torch.zeros_like(self.any_failure)
            for key in (*PHYSICAL_FAILURE_KEYS, "unclassified_failure")
        }
        self.first_reason_step = {
            key: torch.full_like(self.first_failure_step, -1)
            for key in self.reason_seen
        }
        self.first_failure_phase = torch.full_like(
            self.first_failure_step, -1
        )
        self.first_failure_expert_step = torch.full_like(
            self.first_failure_step, -1
        )
        self.first_failure_contact_force_n = torch.full(
            (self.worlds, 4),
            float("nan"),
            device=device,
            dtype=torch.float32,
        )
        self.first_failure_holder_active = torch.zeros_like(
            self.any_failure
        )
        self.first_failure_shaft_captured = torch.zeros_like(
            self.any_failure
        )
        self.first_failure_thread_engaged = torch.zeros_like(
            self.any_failure
        )
        self.first_failure_primary_pair_contact = torch.zeros_like(
            self.any_failure
        )
        self.first_failure_two_finger_contact = torch.zeros_like(
            self.any_failure
        )
        self.first_failure_thread_ready = torch.zeros_like(
            self.any_failure
        )
        self.first_failure_before_capture_loss_steps = torch.full_like(
            self.first_failure_step, -1
        )
        self.first_failure_before_thread_loss_steps = torch.full_like(
            self.first_failure_step, -1
        )
        self.first_failure_thread_ready_steps = torch.full_like(
            self.first_failure_step, -1
        )
        self.first_failure_thread_metrics = torch.full(
            (self.worlds, 4),
            float("nan"),
            device=device,
            dtype=torch.float32,
        )
        self.first_failure_desired_distance_m = torch.full(
            (self.worlds, 4),
            float("nan"),
            device=device,
            dtype=torch.float32,
        )

    @staticmethod
    def _record_first(first, active, step_number: int) -> None:
        first.masked_fill_(
            active & (first < 0), int(step_number)
        )

    def update(self, result, *, step_number: int) -> None:
        info = result.info
        failure = self.torch.as_tensor(
            info["failure"], device=self.device, dtype=self.torch.bool
        )
        truncation = self.torch.as_tensor(
            info["truncated"], device=self.device, dtype=self.torch.bool
        )
        first_failure = failure & (self.first_failure_step < 0)
        if "task_phase" in info:
            phase = self.torch.as_tensor(
                info["task_phase"],
                device=self.device,
                dtype=self.torch.long,
            )
            self.first_failure_phase.copy_(
                self.torch.where(
                    first_failure,
                    phase,
                    self.first_failure_phase,
                )
            )
        if "expert_step" in info:
            expert_step = self.torch.as_tensor(
                info["expert_step"],
                device=self.device,
                dtype=self.torch.long,
            )
            self.first_failure_expert_step.copy_(
                self.torch.where(
                    first_failure,
                    expert_step,
                    self.first_failure_expert_step,
                )
            )
        if "fingertip_contact_force_n" in info:
            force = self.torch.as_tensor(
                info["fingertip_contact_force_n"],
                device=self.device,
                dtype=self.torch.float32,
            )
            self.first_failure_contact_force_n.copy_(
                self.torch.where(
                    first_failure[:, None],
                    force,
                    self.first_failure_contact_force_n,
                )
            )
        context_bool = (
            ("holder_active", self.first_failure_holder_active),
            ("shaft_captured", self.first_failure_shaft_captured),
            ("thread_engaged", self.first_failure_thread_engaged),
            (
                "primary_pair_contact",
                self.first_failure_primary_pair_contact,
            ),
            (
                "two_finger_contact",
                self.first_failure_two_finger_contact,
            ),
            (
                "thread_engagement_ready",
                self.first_failure_thread_ready,
            ),
        )
        for key, destination in context_bool:
            if key in info:
                value = self.torch.as_tensor(
                    info[key],
                    device=self.device,
                    dtype=self.torch.bool,
                )
                destination.copy_(
                    self.torch.where(
                        first_failure, value, destination
                    )
                )
        context_long = (
            (
                "before_capture_contact_loss_steps",
                self.first_failure_before_capture_loss_steps,
            ),
            (
                "before_thread_two_finger_loss_steps",
                self.first_failure_before_thread_loss_steps,
            ),
            (
                "thread_ready_stable_steps",
                self.first_failure_thread_ready_steps,
            ),
        )
        for key, destination in context_long:
            if key in info:
                value = self.torch.as_tensor(
                    info[key],
                    device=self.device,
                    dtype=self.torch.long,
                )
                destination.copy_(
                    self.torch.where(
                        first_failure, value, destination
                    )
                )
        metric_keys = (
            "thread_depth_error_m",
            "thread_lateral_error_m",
            "thread_axis_error_deg",
            "thread_tip_speed_m_s",
        )
        if all(key in info for key in metric_keys):
            metrics = self.torch.stack(
                [
                    self.torch.as_tensor(
                        info[key],
                        device=self.device,
                        dtype=self.torch.float32,
                    )
                    for key in metric_keys
                ],
                dim=-1,
            )
            self.first_failure_thread_metrics.copy_(
                self.torch.where(
                    first_failure[:, None],
                    metrics,
                    self.first_failure_thread_metrics,
                )
            )
        if "desired_fingertip_distance_m" in info:
            distance = self.torch.as_tensor(
                info["desired_fingertip_distance_m"],
                device=self.device,
                dtype=self.torch.float32,
            )
            self.first_failure_desired_distance_m.copy_(
                self.torch.where(
                    first_failure[:, None],
                    distance,
                    self.first_failure_desired_distance_m,
                )
            )
        classified = self.torch.zeros_like(failure)
        for key in PHYSICAL_FAILURE_KEYS:
            value = info.get(key)
            if value is None:
                value = self.torch.zeros_like(failure)
            active = self.torch.as_tensor(
                value,
                device=self.device,
                dtype=self.torch.bool,
            )
            classified |= active
            self._record_first(
                self.first_reason_step[key], active, step_number
            )
            self.reason_seen[key] |= active
        unclassified = failure & ~classified
        self._record_first(
            self.first_reason_step["unclassified_failure"],
            unclassified,
            step_number,
        )
        self.reason_seen["unclassified_failure"] |= unclassified
        self._record_first(
            self.first_failure_step, failure, step_number
        )
        self._record_first(
            self.first_truncation_step, truncation, step_number
        )
        self.any_failure |= failure
        self.any_truncation |= truncation

    def assert_clean(self, *, context: str) -> None:
        """Use one host synchronization on clean periodic checks."""

        dirty = self.torch.stack(
            (
                self.any_failure.any(),
                self.any_truncation.any(),
            )
        )
        if not bool(dirty.any().item()):
            return
        summary = self.summary()
        encoded = json.dumps(
            summary, sort_keys=True, separators=(",", ":")
        )
        if summary["failure_worlds"]:
            raise RuntimeError(
                f"{context} had a physical failure in "
                f"{summary['failure_worlds']}/{summary['worlds']} worlds; "
                f"audit={encoded}"
            )
        raise RuntimeError(
            f"{context} truncated in "
            f"{summary['truncation_worlds']}/{summary['worlds']} worlds; "
            f"audit={encoded}"
        )

    @staticmethod
    def _first_value(first) -> int | None:
        selected = first[first >= 0]
        return int(selected.min().item()) if selected.numel() else None

    @staticmethod
    def _finite_value(value) -> float | None:
        result = float(value)
        return result if math.isfinite(result) else None

    def summary(self) -> dict:
        reasons = {}
        for key in self.reason_seen:
            seen = self.reason_seen[key]
            reasons[key] = {
                "worlds": int(seen.sum().item()),
                "first_step": self._first_value(
                    self.first_reason_step[key]
                ),
            }
        summary = {
            "worlds": self.worlds,
            "failure_worlds": int(self.any_failure.sum().item()),
            "truncation_worlds": int(
                self.any_truncation.sum().item()
            ),
            "first_failure_step": self._first_value(
                self.first_failure_step
            ),
            "first_truncation_step": self._first_value(
                self.first_truncation_step
            ),
            "reasons": reasons,
        }
        earliest = summary["first_failure_step"]
        if earliest is not None:
            candidates = self.torch.nonzero(
                self.first_failure_step == earliest,
                as_tuple=False,
            ).flatten()
            world = int(candidates[0].item())
            phase_value = int(
                self.first_failure_phase[world].item()
            )
            try:
                phase_name = TaskPhase(phase_value).name
            except ValueError:
                phase_name = "UNKNOWN"
            summary["first_failure_context"] = {
                "world_index": world,
                "step": earliest,
                "expert_step": int(
                    self.first_failure_expert_step[world].item()
                ),
                "phase": phase_value,
                "phase_name": phase_name,
                "fingertip_contact_force_n": [
                    float(value) if math.isfinite(value) else None
                    for value in self.first_failure_contact_force_n[
                        world
                    ].tolist()
                ],
                "holder_active": bool(
                    self.first_failure_holder_active[world].item()
                ),
                "shaft_captured": bool(
                    self.first_failure_shaft_captured[world].item()
                ),
                "thread_engaged": bool(
                    self.first_failure_thread_engaged[world].item()
                ),
                "primary_pair_contact": bool(
                    self.first_failure_primary_pair_contact[world].item()
                ),
                "two_finger_contact": bool(
                    self.first_failure_two_finger_contact[world].item()
                ),
                "thread_engagement_ready": bool(
                    self.first_failure_thread_ready[world].item()
                ),
                "thread_ready_stable_steps": int(
                    self.first_failure_thread_ready_steps[world].item()
                ),
                "before_capture_contact_loss_steps": int(
                    self.first_failure_before_capture_loss_steps[
                        world
                    ].item()
                ),
                "before_thread_two_finger_loss_steps": int(
                    self.first_failure_before_thread_loss_steps[
                        world
                    ].item()
                ),
                "thread_depth_error_m": self._finite_value(
                    self.first_failure_thread_metrics[world, 0].item()
                ),
                "thread_lateral_error_m": self._finite_value(
                    self.first_failure_thread_metrics[world, 1].item()
                ),
                "thread_axis_error_deg": self._finite_value(
                    self.first_failure_thread_metrics[world, 2].item()
                ),
                "thread_tip_speed_m_s": self._finite_value(
                    self.first_failure_thread_metrics[world, 3].item()
                ),
                "desired_fingertip_distance_m": [
                    float(value) if math.isfinite(value) else None
                    for value in self.first_failure_desired_distance_m[
                        world
                    ].tolist()
                ],
            }
        return summary


def certify_expert_rollout(
    env,
    result,
    audit: ExpertRolloutAudit,
    *,
    context: str,
    require_terminal: bool,
    max_truncation_fraction: float = 0.0,
) -> dict:
    """Reject failures and certify non-truncated worlds at a full horizon.

    The default retains the strict zero-truncation contract.  A non-zero
    allowance is intended for explicit expert-rollout preflight gates:
    physical failures remain unconditionally fatal, while otherwise clean
    time-limit truncations may be excluded from terminal-pose certification.
    """

    max_truncation_fraction = float(max_truncation_fraction)
    if (
        not math.isfinite(max_truncation_fraction)
        or not 0.0 <= max_truncation_fraction <= 1.0
    ):
        raise ValueError("max_truncation_fraction must be in [0, 1]")

    summary = audit.summary()
    truncation_worlds = int(summary["truncation_worlds"])
    truncation_fraction = truncation_worlds / max(audit.worlds, 1)
    summary["truncation_fraction"] = truncation_fraction
    summary["max_truncation_fraction"] = max_truncation_fraction
    summary["admitted_truncation_worlds"] = (
        truncation_worlds
        if truncation_fraction <= max_truncation_fraction
        else 0
    )
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    if summary["failure_worlds"]:
        raise RuntimeError(
            f"{context} had a physical failure in "
            f"{summary['failure_worlds']}/{summary['worlds']} worlds; "
            f"audit={encoded}"
        )
    if truncation_fraction > max_truncation_fraction:
        raise RuntimeError(
            f"{context} truncated in "
            f"{summary['truncation_worlds']}/{summary['worlds']} worlds; "
            f"maximum admitted fraction={max_truncation_fraction:.6g}; "
            f"audit={encoded}"
        )
    if not require_terminal:
        return summary

    info = result.info
    success = info["success"]
    terminal_population = ~audit.any_truncation
    depth_ok = (
        info["depth"] - env.state.task.target_insertion_depth_m
    ).abs() <= 0.002
    twist_ok = (
        info["twist"] - env.state.task.target_rotation_rad
    ).abs() <= math.radians(8.0)
    terminal_checks = {
        "not_success": terminal_population & ~success,
        "thread_not_engaged": (
            terminal_population & ~env.state.thread_engaged
        ),
        "depth_out_of_tolerance": terminal_population & ~depth_ok,
        "twist_out_of_tolerance": terminal_population & ~twist_ok,
    }
    terminal_counts = {
        key: int(value.sum().item())
        for key, value in terminal_checks.items()
    }
    passed = success & env.state.thread_engaged & depth_ok & twist_ok
    failed = terminal_population & ~passed
    failed_worlds = int(failed.sum().item())
    summary["terminal_certified_worlds"] = int(
        terminal_population.sum().item()
    )
    summary["terminal_failure_worlds"] = failed_worlds
    summary["terminal_reasons"] = terminal_counts
    if failed_worlds:
        terminal_encoded = json.dumps(
            summary, sort_keys=True, separators=(",", ":")
        )
        raise RuntimeError(
            f"{context} final-state certification failed in "
            f"{failed_worlds}/{audit.worlds} worlds; "
            f"audit={terminal_encoded}"
        )
    return summary


__all__ = [
    "ExpertRolloutAudit",
    "PHYSICAL_FAILURE_KEYS",
    "certify_expert_rollout",
]
