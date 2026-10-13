"""Paired first-episode evaluation for Diffusion and A* nominal planners.

The comparison consumes the immutable initial-condition manifest emitted by
the formal A* evaluation.  Both controllers are reconstructed with the exact
manifest root seed and must reproduce every reset row before the first physics
step.  Only the first terminal outcome from each world is retained.

The module intentionally keeps the actuator residual at zero for both arms.
It records measured screw-tip paths over the common learned ``ALIGN`` phase,
not nominal commands, and times only the nominal planning/inference operation
with explicit CUDA synchronisation.  This makes the success and path metrics
paired by initial state while keeping the latency scope explicit.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import is_dataclass, replace
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from screw_control import TaskPhase

from .astar_evaluation import (
    EVALUATION_SEED_OFFSET,
    INITIAL_CONDITIONS_SCHEMA,
    _failure_reasons,
    _selected_scalar,
    _terminal_phase,
    canonical_sha256,
    sha256_file,
)
from .contracts import (
    RESIDUAL_ACTION_DIM,
    SCHEMA_HASH,
    TASK_GEOMETRY_HASH,
)
from .monitoring import append_history, atomic_write_json, utc_now


PAIRED_EVALUATION_SCHEMA = "screw-paired-planner-evaluation-v1"
PAIRED_EPISODE_SCHEMA = "screw-paired-planner-episode-v1"
PAIRED_ROW_SCHEMA = "screw-paired-planner-row-v1"
PATH_SCOPE = "measured_screw_tip_during_first_contiguous_learned_align"


def _torch():
    import torch

    return torch


def _evaluation_config_for_manifest(config: Any, root_seed: int) -> Any:
    """Return a detached config whose environment generator uses root_seed."""

    if is_dataclass(config) and is_dataclass(config.runtime):
        return replace(config, runtime=replace(config.runtime, seed=int(root_seed)))
    result = copy.deepcopy(config)
    setattr(result.runtime, "seed", int(root_seed))
    return result


def load_paired_initial_conditions(
    path: str | Path,
    *,
    config: Any,
    episodes: Optional[int] = None,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    """Load and validate the exact formal first-reset distribution.

    The formal manifest stores the canonical hash of the detached evaluation
    config (base seed + 100000).  Requiring that hash prevents a superficially
    paired replay with changed geometry, curriculum, or episode limits.
    """

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("initial-condition manifest must be a JSON object")
    if payload.get("schema") != INITIAL_CONDITIONS_SCHEMA:
        raise ValueError(
            f"initial-condition schema must be {INITIAL_CONDITIONS_SCHEMA!r}"
        )
    root_seed = int(payload.get("root_seed", -1))
    expected_seed = int(config.runtime.seed) + EVALUATION_SEED_OFFSET
    if root_seed != expected_seed:
        raise ValueError(
            f"manifest root_seed {root_seed} != base seed + 100000 "
            f"({expected_seed})"
        )
    evaluation_config = _evaluation_config_for_manifest(config, root_seed)
    expected_config_hash = canonical_sha256(evaluation_config.to_dict())
    manifest_config_hash = str(
        payload.get("evaluation_config_canonical_sha256", "")
    )
    if manifest_config_hash != expected_config_hash:
        raise ValueError(
            "manifest evaluation config hash differs from the requested config"
        )
    rows_value = payload.get("rows")
    if not isinstance(rows_value, list) or not rows_value:
        raise ValueError("initial-condition manifest has no rows")
    requested = len(rows_value) if episodes is None else int(episodes)
    if requested <= 0 or requested > len(rows_value):
        raise ValueError("episodes must lie within the manifest row count")
    rows = [dict(row) for row in rows_value[:requested]]
    required = {
        "episode_index",
        "world_id",
        "root_seed",
        "row_seed",
        "spawn_x_m",
        "spawn_y_m",
        "socket_mouth_x_m",
        "socket_mouth_y_m",
        "socket_angle_deg",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"manifest row {index} is missing {sorted(missing)}"
            )
        if int(row["world_id"]) != index or int(row["episode_index"]) != index:
            raise ValueError("manifest rows must be contiguous and world ordered")
        if int(row["root_seed"]) != root_seed:
            raise ValueError("manifest row root seed differs from the manifest")
        if int(row["row_seed"]) != root_seed + index:
            raise ValueError("manifest row_seed is not its stable row identity")
        numeric = (
            "spawn_x_m",
            "spawn_y_m",
            "socket_mouth_x_m",
            "socket_mouth_y_m",
            "socket_angle_deg",
        )
        if not all(math.isfinite(float(row[key])) for key in numeric):
            raise ValueError(f"manifest row {index} contains a non-finite value")
    return payload, evaluation_config, rows


def assert_environment_matches_manifest(
    env: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    absolute_tolerance_m: float = 5.0e-8,
) -> dict[str, Any]:
    """Fail before physics if a reconstructed environment differs by a row."""

    torch = _torch()
    count = len(rows)
    if int(env.worlds) != count:
        raise ValueError(
            f"environment worlds {env.worlds} != manifest rows {count}"
        )
    expected_spawn = torch.as_tensor(
        [[row["spawn_x_m"], row["spawn_y_m"]] for row in rows],
        dtype=torch.float64,
    )
    expected_mouth = torch.as_tensor(
        [
            [row["socket_mouth_x_m"], row["socket_mouth_y_m"]]
            for row in rows
        ],
        dtype=torch.float64,
    )
    expected_angle = torch.as_tensor(
        [row["socket_angle_deg"] for row in rows], dtype=torch.float64
    )
    actual_spawn = env.state.spawn_xy.detach().to("cpu", torch.float64)
    actual_mouth = env.state.socket_mouth[:, :2].detach().to(
        "cpu", torch.float64
    )
    actual_angle = env.state.socket_angle_deg.detach().to("cpu", torch.float64)
    errors = {
        "spawn_max_abs_error_m": float(
            torch.max(torch.abs(actual_spawn - expected_spawn)).item()
        ),
        "mouth_max_abs_error_m": float(
            torch.max(torch.abs(actual_mouth - expected_mouth)).item()
        ),
        "socket_angle_max_abs_error_deg": float(
            torch.max(torch.abs(actual_angle - expected_angle)).item()
        ),
    }
    if errors["spawn_max_abs_error_m"] > float(absolute_tolerance_m):
        raise RuntimeError(f"spawn rows do not reproduce the manifest: {errors}")
    if errors["mouth_max_abs_error_m"] > float(absolute_tolerance_m):
        raise RuntimeError(f"socket rows do not reproduce the manifest: {errors}")
    if errors["socket_angle_max_abs_error_deg"] > 1.0e-6:
        raise RuntimeError(f"socket angles do not reproduce the manifest: {errors}")
    return {
        **errors,
        "rows": count,
        "absolute_tolerance_m": float(absolute_tolerance_m),
        "matched": True,
    }


def summarise_measured_path(points_xyz_m: np.ndarray) -> dict[str, Any]:
    """Summarise one control-rate measured screw-tip path.

    Straightness is defined in the table plane.  A separate 3-D length makes
    the later rotate/raise motion visible without conflating it with planar
    detour efficiency.
    """

    points = np.asarray(points_xyz_m, dtype=np.float64).reshape(-1, 3)
    if len(points) < 2:
        return {
            "scope": PATH_SCOPE,
            "sample_count": int(len(points)),
            "available": False,
        }
    if not np.isfinite(points).all():
        raise FloatingPointError("measured screw-tip path is non-finite")
    delta = np.diff(points, axis=0)
    path_length_3d = float(np.linalg.norm(delta, axis=1).sum())
    planar = points[:, :2]
    planar_delta = np.diff(planar, axis=0)
    path_length_xy = float(np.linalg.norm(planar_delta, axis=1).sum())
    chord = planar[-1] - planar[0]
    straight_xy = float(np.linalg.norm(chord))
    if straight_xy <= 1.0e-12:
        ratio = None
        maximum_deviation = None
    else:
        relative = planar - planar[0]
        deviation = np.abs(
            chord[0] * relative[:, 1] - chord[1] * relative[:, 0]
        ) / straight_xy
        ratio = path_length_xy / straight_xy
        maximum_deviation = float(np.max(deviation))
    return {
        "scope": PATH_SCOPE,
        "sampling": "one measured post-physics screw-tip sample per control tick",
        "sample_count": int(len(points)),
        "available": True,
        "start_xyz_m": points[0].tolist(),
        "end_xyz_m": points[-1].tolist(),
        "path_length_xy_m": path_length_xy,
        "path_length_xyz_m": path_length_3d,
        "straight_distance_xy_m": straight_xy,
        "path_to_straight_ratio_xy": ratio,
        "max_straight_line_deviation_xy_m": maximum_deviation,
    }


class MeasuredAlignPathRecorder:
    """CUDA-resident control-rate paths for one first episode per world."""

    def __init__(self, worlds: int, max_steps: int, device: Any) -> None:
        torch = _torch()
        self.worlds = int(worlds)
        self.max_points = int(max_steps) + 1
        self.device = torch.device(device)
        self.points = torch.zeros(
            (self.worlds, self.max_points, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self.counts = torch.zeros(
            self.worlds, device=self.device, dtype=torch.long
        )
        self.started = torch.zeros(
            self.worlds, device=self.device, dtype=torch.bool
        )
        self.finished = torch.zeros_like(self.started)
        self.previous_selected = torch.zeros_like(self.started)

    def update(
        self,
        previous_tip: Any,
        current_tip: Any,
        *,
        first_episode_active: Any,
        learned_active: Any,
        phase: Any,
    ) -> None:
        torch = _torch()
        previous = torch.as_tensor(
            previous_tip, device=self.device, dtype=torch.float32
        )
        current = torch.as_tensor(
            current_tip, device=self.device, dtype=torch.float32
        )
        active = torch.as_tensor(
            first_episode_active, device=self.device, dtype=torch.bool
        )
        learned = torch.as_tensor(
            learned_active, device=self.device, dtype=torch.bool
        )
        phase_t = torch.as_tensor(phase, device=self.device, dtype=torch.long)
        if tuple(previous.shape) != (self.worlds, 3) or tuple(current.shape) != (
            self.worlds,
            3,
        ):
            raise ValueError("screw-tip samples have an invalid shape")
        selected = (
            active
            & learned
            & (phase_t == int(TaskPhase.ALIGN))
            & ~self.finished
        )
        leaving = self.previous_selected & ~selected
        self.finished |= leaving
        selected &= ~self.finished
        new = selected & ~self.started
        new_ids = torch.nonzero(new, as_tuple=False).flatten()
        if int(new_ids.numel()):
            self.points[new_ids, 0] = previous.index_select(0, new_ids)
            self.counts[new_ids] = 1
            self.started[new_ids] = True
        ids = torch.nonzero(selected, as_tuple=False).flatten()
        if int(ids.numel()):
            index = self.counts.index_select(0, ids)
            if bool((index >= self.max_points).any()):
                raise RuntimeError("measured path buffer exhausted")
            self.points[ids, index] = current.index_select(0, ids)
            self.counts[ids] = index + 1
        self.previous_selected.copy_(selected)

    def summaries(self) -> list[dict[str, Any]]:
        counts = self.counts.detach().cpu().tolist()
        points = self.points.detach().cpu().numpy()
        return [
            summarise_measured_path(points[index, : int(count)])
            for index, count in enumerate(counts)
        ]


def _cuda_synchronise(device: Any) -> None:
    torch = _torch()
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


class TimedDiffusionController:
    """Transparent controller wrapper timing only calls that sample a plan."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.events: list[dict[str, Any]] = []
        self.first_episode_active = None
        self.background_only_samples_suppressed = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def set_first_episode_active(self, value: Any) -> None:
        self.first_episode_active = value

    def _active_count(self) -> Optional[int]:
        if self.first_episode_active is None:
            return None
        return int(self.first_episode_active.sum().item())

    def _timed(
        self,
        kind: str,
        batch_rows: int,
        function: Callable[[], Any],
        *,
        first_episode_rows_in_sample: Optional[int],
    ):
        before = int(self.inner.sample_count)
        _cuda_synchronise(self.inner.device)
        started = time.perf_counter()
        result = function()
        _cuda_synchronise(self.inner.device)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        after = int(self.inner.sample_count)
        if after > before:
            self.events.append(
                {
                    "kind": str(kind),
                    "sample_count_before": before,
                    "sample_count_after": after,
                    "batch_rows": int(batch_rows),
                    "first_episode_active_worlds": self._active_count(),
                    "first_episode_rows_in_sample": (
                        None
                        if first_episode_rows_in_sample is None
                        else int(first_episode_rows_in_sample)
                    ),
                    "comparison_relevant": (
                        first_episode_rows_in_sample is None
                        or int(first_episode_rows_in_sample) > 0
                    ),
                    "latency_ms": float(elapsed_ms),
                    "explicit_cuda_synchronisation": True,
                }
            )
        return result

    def reset(self, mask: Any, observation_history: Any = None) -> None:
        torch = _torch()
        mask_t = torch.as_tensor(mask, device=self.inner.device, dtype=torch.bool)
        rows = int(mask_t.sum().item())
        if observation_history is None or rows == 0:
            self.inner.reset(mask_t, observation_history)
            return
        tracked_rows = None
        if self.first_episode_active is not None:
            tracked_rows = int(
                (mask_t & self.first_episode_active).sum().item()
            )
            # A terminal row must be reset for shared-simulator health, but a
            # later untracked episode must not advance the global Diffusion
            # sample seed and thereby change a still-active world's first
            # episode.  The inactive row will consume the static full-batch
            # plan; its later outcome is deliberately unobserved.
            if tracked_rows == 0:
                self.background_only_samples_suppressed += 1
                return
        self._timed(
            "masked_reset_sample",
            rows,
            lambda: self.inner.reset(mask_t, observation_history),
            first_episode_rows_in_sample=tracked_rows,
        )

    def act(self, observation_history: Any) -> Any:
        will_sample = int(self.inner.cursor) >= int(self.inner.replan_interval)
        if not will_sample:
            return self.inner.act(observation_history)
        return self._timed(
            "full_batch_replan",
            int(self.inner.worlds),
            lambda: self.inner.act(observation_history),
            first_episode_rows_in_sample=self._active_count(),
        )

    def state_dict(self) -> dict[str, Any]:
        return self.inner.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.inner.load_state_dict(state)


class TimedAStarPlanner:
    """Transparent A* wrapper with synchronised whole-reset latency events.

    A CUDA-graph environment performs one throwaway reset before graph capture
    and a second reset afterwards to establish the actual first-episode state.
    Planning the throwaway state is unnecessary because graph preparation only
    advances the physics backend and never requests a nominal A* command.  At
    large world counts it also doubles the lifetime traffic through the CPU
    MuJoCo collision oracle.  ``skip_initial_resets`` therefore suppresses
    only that non-evaluation planner reset; the environment/backend reset and
    the manifest-matched post-capture A* reset are unchanged.

    During the vector rollout, worlds that have already finished their first
    episode are still reset to keep the shared physics batch healthy.  Those
    later episodes are outside the comparison, so rebuilding an A* plan for a
    reset containing no tracked first-episode row is both unnecessary and can
    perturb the lifetime of the CPU MuJoCo collision oracle.  As with the
    Diffusion wrapper above, only the controller-side reset is suppressed;
    the environment/backend reset remains intact.
    """

    def __init__(self, inner: Any, *, skip_initial_resets: int = 0) -> None:
        if int(skip_initial_resets) < 0:
            raise ValueError("skip_initial_resets must be non-negative")
        self.inner = inner
        self.events: list[dict[str, Any]] = []
        self.skipped_reset_events: list[dict[str, Any]] = []
        self.background_reset_events: list[dict[str, Any]] = []
        self.first_episode_active = None
        self._background_hold = None
        self._background_hold_valid = None
        self._remaining_initial_resets_to_skip = int(skip_initial_resets)
        self.paired_initial_event_index: Optional[int] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def set_first_episode_active(self, value: Any) -> None:
        torch = _torch()
        active = torch.as_tensor(
            value, device=self.inner.device, dtype=torch.bool
        )
        if tuple(active.shape) != (int(self.inner.worlds),):
            raise ValueError(
                "first-episode active mask must have shape "
                f"{(int(self.inner.worlds),)}"
            )
        self.first_episode_active = active

    def _store_background_hold(self, mask: Any, command: Any) -> None:
        """Park unobserved rows at their newly certified handoff pose."""

        torch = _torch()
        mask_t = torch.as_tensor(
            mask, device=self.inner.device, dtype=torch.bool
        )
        if command is None or not hasattr(command, "clone"):
            raise RuntimeError(
                "background A* reset requires the certified start command"
            )
        if self._background_hold is None:
            self._background_hold = command.clone()
            self._background_hold_valid = torch.zeros(
                int(self.inner.worlds),
                device=self.inner.device,
                dtype=torch.bool,
            )
        else:
            for name in (
                "tip",
                "rotation",
                "twist",
                "material",
                "distance",
                "phase",
                "engaged",
            ):
                source = getattr(command, name)
                target = getattr(self._background_hold, name)
                selected = mask_t.reshape(
                    (int(self.inner.worlds),)
                    + (1,) * (source.ndim - 1)
                )
                target.copy_(torch.where(selected, source, target))
        self._background_hold_valid.logical_or_(mask_t)

    def reset(self, mask: Any, **kwargs: Any) -> Any:
        torch = _torch()
        mask_t = torch.as_tensor(mask, device=self.inner.device, dtype=torch.bool)
        rows = int(mask_t.sum().item())
        if rows == 0:
            return None
        if self._remaining_initial_resets_to_skip:
            if rows != int(self.inner.worlds):
                raise RuntimeError(
                    "the CUDA-graph warmup planner skip requires a full reset"
                )
            self._remaining_initial_resets_to_skip -= 1
            self.skipped_reset_events.append(
                {
                    "kind": "planner_reset_skipped",
                    "batch_rows": rows,
                    "reason": "non_evaluation_cuda_graph_warmup_state",
                    "physics_environment_reset_preserved": True,
                }
            )
            return None
        if self.first_episode_active is not None:
            tracked_rows = int(
                (mask_t & self.first_episode_active).sum().item()
            )
            if tracked_rows:
                raise RuntimeError(
                    "post-initial A* reset contains a tracked first-episode "
                    "row; one-shot global planning would be violated"
                )
            self._store_background_hold(mask_t, kwargs.get("start_command"))
            self.background_reset_events.append(
                {
                    "kind": "planner_reset_skipped",
                    "batch_rows": rows,
                    "reason": "background_only_post_first_episode_state",
                    "first_episode_rows_in_reset": 0,
                    "physics_environment_reset_preserved": True,
                    "certified_handoff_hold_installed": True,
                }
            )
            return None
        _cuda_synchronise(self.inner.device)
        started = time.perf_counter()
        result = self.inner.reset(mask_t, **kwargs)
        _cuda_synchronise(self.inner.device)
        self.events.append(
            {
                "kind": "planner_reset",
                "batch_rows": rows,
                "latency_ms": 1000.0 * (time.perf_counter() - started),
                "explicit_cuda_synchronisation": True,
            }
        )
        return result

    def command(self, anchor: Any, *, next_step: Any) -> Any:
        planned = self.inner.command(anchor, next_step=next_step)
        if (
            self.first_episode_active is None
            or self._background_hold is None
            or self._background_hold_valid is None
        ):
            return planned
        torch = _torch()
        selected = planned.clone()
        for name in ("tip", "rotation", "twist", "material", "distance"):
            planned_value = getattr(planned, name)
            hold_value = getattr(self._background_hold, name)
            anchor_value = getattr(anchor, name)
            valid = self._background_hold_valid.reshape(
                (int(self.inner.worlds),)
                + (1,) * (planned_value.ndim - 1)
            )
            active = self.first_episode_active.reshape(
                (int(self.inner.worlds),)
                + (1,) * (planned_value.ndim - 1)
            )
            background_value = torch.where(valid, hold_value, anchor_value)
            getattr(selected, name).copy_(
                torch.where(active, planned_value, background_value)
            )
        return selected

    def mark_paired_initial_state(self) -> None:
        if self._remaining_initial_resets_to_skip:
            raise RuntimeError("A* warmup reset skip was not consumed")
        candidates = [
            index
            for index, event in enumerate(self.events)
            if int(event["batch_rows"]) == int(self.inner.worlds)
        ]
        if not candidates:
            raise RuntimeError("A* environment performed no full reset plan")
        self.paired_initial_event_index = candidates[-1]
        self.events[self.paired_initial_event_index]["paired_initial_state"] = True

    def provenance(self) -> dict[str, Any]:
        result = dict(self.inner.provenance())
        result["evaluation_reset_policy"] = {
            "non_evaluation_cuda_graph_warmup_planning_skipped": bool(
                self.skipped_reset_events
            ),
            "skipped_resets": len(self.skipped_reset_events),
            "background_only_planning_skipped": bool(
                self.background_reset_events
            ),
            "background_only_skipped_resets": len(
                self.background_reset_events
            ),
            "background_only_skipped_rows": sum(
                int(event["batch_rows"])
                for event in self.background_reset_events
            ),
            "physics_environment_resets_preserved": True,
            "first_episode_planning_preserved": True,
            "background_rows_held_at_certified_handoff": bool(
                self.background_reset_events
            ),
            "manifest_matched_reset_planned": (
                self.paired_initial_event_index is not None
            ),
        }
        return result

    def state_dict(self) -> dict[str, Any]:
        return self.inner.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.inner.load_state_dict(state)


def _percentile(values: Sequence[float], fraction: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def aggregate_latency_events(
    events: Sequence[Mapping[str, Any]],
    *,
    selected_event_indices: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Aggregate batch and honest amortised-per-sampled-row latency."""

    if selected_event_indices is None:
        selected = list(events)
    else:
        selected = [events[int(index)] for index in selected_event_indices]
    selected = [
        event
        for event in selected
        if float(event.get("latency_ms", math.nan)) >= 0.0
        and int(event.get("batch_rows", 0)) > 0
    ]
    if not selected:
        return {"events": 0, "available": False}
    batch = [float(event["latency_ms"]) for event in selected]
    total_sampled_rows = sum(int(event["batch_rows"]) for event in selected)
    amortised = [
        float(event["latency_ms"]) / int(event["batch_rows"])
        for event in selected
    ]
    return {
        "available": True,
        "events": len(selected),
        "scope": "explicitly synchronised nominal planning/inference call",
        "total_latency_ms": sum(batch),
        "total_sampled_rows": total_sampled_rows,
        "total_latency_ms_per_sampled_row": (
            sum(batch) / total_sampled_rows
        ),
        "batch_latency_ms": {
            "mean": statistics.fmean(batch),
            "p50": _percentile(batch, 0.50),
            "p95": _percentile(batch, 0.95),
            "min": min(batch),
            "max": max(batch),
        },
        "amortised_latency_ms_per_sampled_row": {
            "mean": statistics.fmean(amortised),
            "p50": _percentile(amortised, 0.50),
            "p95": _percentile(amortised, 0.95),
        },
        "raw_events": [dict(event) for event in selected],
    }


def _terminal_record(
    *,
    stack: str,
    info: Mapping[str, Any],
    world_id: int,
    initial_condition: Mapping[str, Any],
    forced_truncation: bool,
    max_steps: int,
) -> dict[str, Any]:
    if forced_truncation:
        success, failure, truncated = False, False, True
        steps = int(max_steps)
    else:
        success = bool(_selected_scalar(info, "success", world_id, False))
        failure = bool(_selected_scalar(info, "failure", world_id, False))
        truncated = bool(_selected_scalar(info, "truncated", world_id, False))
        steps = int(_selected_scalar(info, "elapsed_steps", world_id, max_steps))
    if sum((success, failure, truncated)) != 1:
        raise RuntimeError("terminal outcome is not mutually exclusive")
    depth = float(_selected_scalar(info, "depth", world_id, 0.0))
    twist = float(_selected_scalar(info, "twist", world_id, 0.0))
    lateral = float(_selected_scalar(info, "lateral_error", world_id, 0.0))
    axis = float(_selected_scalar(info, "axis_error", world_id, 0.0))
    return {
        "schema": PAIRED_EPISODE_SCHEMA,
        "stack": str(stack),
        "world_id": int(world_id),
        "episode_index": int(world_id),
        "initial_condition": dict(initial_condition),
        "outcome": "success" if success else "failure" if failure else "truncated",
        "success": success,
        "failure": failure,
        "truncated": truncated,
        "forced_evaluation_step_limit": bool(forced_truncation),
        "terminal_steps": steps,
        "terminal_phase": _terminal_phase(info, world_id),
        "terminal_depth_m": depth,
        "terminal_turns": twist / (2.0 * math.pi),
        "terminal_lateral_error_m": lateral,
        "terminal_axis_error_rad": axis,
        "terminal_speed_m_s": float(
            _selected_scalar(info, "speed", world_id, 0.0)
        ),
        "ik_failures": int(_selected_scalar(info, "ik_failures", world_id, 0)),
        "ik_anchor_fallbacks": int(
            _selected_scalar(info, "ik_anchor_fallbacks", world_id, 0)
        ),
        "ik_safety_stops": int(
            _selected_scalar(info, "ik_safety_stops", world_id, 0)
        ),
        "planner_ik_failure": bool(
            _selected_scalar(info, "planner_ik_failure", world_id, False)
        ),
        "failure_reasons": _failure_reasons(
            info, world_id, forced=forced_truncation
        ),
    }


def _finite_distribution(
    records: Sequence[Mapping[str, Any]],
    accessor: Callable[[Mapping[str, Any]], Any],
) -> Optional[dict[str, Any]]:
    values: list[float] = []
    for record in records:
        value = accessor(record)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def aggregate_stack_records(
    records: Sequence[Mapping[str, Any]],
    *,
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    outcomes = Counter(str(record["outcome"]) for record in records)
    successes = [record for record in records if bool(record["success"])]
    reached_insert = [
        record for record in records if bool(record.get("reached_insert"))
    ]

    def path_value(record: Mapping[str, Any], key: str):
        return record.get("measured_align_path", {}).get(key)

    return {
        "episodes": len(records),
        "successes": outcomes["success"],
        "failures": outcomes["failure"],
        "truncations": outcomes["truncated"],
        "success_rate": outcomes["success"] / len(records) if records else None,
        "first_insert_reach": {
            "count": len(reached_insert),
            "rate": len(reached_insert) / len(records) if records else None,
            "ticks": _finite_distribution(
                reached_insert, lambda row: row.get("first_insert_tick")
            ),
            "right_censored_count": len(records) - len(reached_insert),
        },
        "completion_ticks_all": _finite_distribution(
            records, lambda row: row["terminal_steps"]
        ),
        "completion_ticks_successes": _finite_distribution(
            successes, lambda row: row["terminal_steps"]
        ),
        "measured_align_path_all": {
            key: _finite_distribution(records, lambda row, k=key: path_value(row, k))
            for key in (
                "path_length_xy_m",
                "path_length_xyz_m",
                "straight_distance_xy_m",
                "path_to_straight_ratio_xy",
                "max_straight_line_deviation_xy_m",
            )
        },
        "measured_align_path_successes": {
            key: _finite_distribution(successes, lambda row, k=key: path_value(row, k))
            for key in (
                "path_length_xy_m",
                "path_length_xyz_m",
                "straight_distance_xy_m",
                "path_to_straight_ratio_xy",
                "max_straight_line_deviation_xy_m",
            )
        },
        "astar_internal_initial_planning_latency_ms_per_world": (
            _finite_distribution(
                records,
                lambda row: row.get("initial_planner_diagnostics", {}).get(
                    "planning_latency_ms"
                ),
            )
        ),
        "nominal_planning_or_inference_latency": dict(timing),
    }


def build_paired_rows(
    diffusion_records: Sequence[Mapping[str, Any]],
    astar_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    diffusion = {int(row["world_id"]): row for row in diffusion_records}
    astar = {int(row["world_id"]): row for row in astar_records}
    if set(diffusion) != set(astar):
        raise ValueError("Diffusion and A* world sets differ")
    result: list[dict[str, Any]] = []
    for world_id in sorted(diffusion):
        first = diffusion[world_id]
        second = astar[world_id]
        if first["initial_condition"] != second["initial_condition"]:
            raise ValueError(f"world {world_id} initial conditions differ")

        def path(record: Mapping[str, Any], key: str) -> Optional[float]:
            value = record.get("measured_align_path", {}).get(key)
            return None if value is None else float(value)

        deltas: dict[str, Any] = {
            "success_diffusion_minus_astar": int(bool(first["success"]))
            - int(bool(second["success"])),
            "reached_insert_diffusion_minus_astar": int(
                bool(first.get("reached_insert"))
            )
            - int(bool(second.get("reached_insert"))),
            "completion_ticks_diffusion_minus_astar": int(first["terminal_steps"])
            - int(second["terminal_steps"]),
        }
        first_insert = first.get("first_insert_tick")
        second_insert = second.get("first_insert_tick")
        deltas["first_insert_tick_diffusion_minus_astar"] = (
            None
            if first_insert is None or second_insert is None
            else int(first_insert) - int(second_insert)
        )
        for key in (
            "path_length_xy_m",
            "path_length_xyz_m",
            "path_to_straight_ratio_xy",
            "max_straight_line_deviation_xy_m",
        ):
            left, right = path(first, key), path(second, key)
            deltas[f"{key}_diffusion_minus_astar"] = (
                None if left is None or right is None else left - right
            )
        result.append(
            {
                "schema": PAIRED_ROW_SCHEMA,
                "world_id": world_id,
                "initial_condition": dict(first["initial_condition"]),
                "diffusion": dict(first),
                "astar_spline": dict(second),
                "paired_delta": deltas,
            }
        )
    return result


def aggregate_paired_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    both_success = [
        row
        for row in rows
        if bool(row["diffusion"]["success"])
        and bool(row["astar_spline"]["success"])
    ]
    both_reached_insert = [
        row
        for row in rows
        if bool(row["diffusion"].get("reached_insert"))
        and bool(row["astar_spline"].get("reached_insert"))
    ]
    discordant = Counter(
        (
            bool(row["diffusion"]["success"]),
            bool(row["astar_spline"]["success"]),
        )
        for row in rows
    )
    keys = (
        "first_insert_tick_diffusion_minus_astar",
        "completion_ticks_diffusion_minus_astar",
        "path_length_xy_m_diffusion_minus_astar",
        "path_length_xyz_m_diffusion_minus_astar",
        "path_to_straight_ratio_xy_diffusion_minus_astar",
        "max_straight_line_deviation_xy_m_diffusion_minus_astar",
    )
    return {
        "episodes": len(rows),
        "both_successes": len(both_success),
        "both_reached_insert": len(both_reached_insert),
        "outcome_pair_counts": {
            "both_success": discordant[(True, True)],
            "diffusion_only_success": discordant[(True, False)],
            "astar_only_success": discordant[(False, True)],
            "neither_success": discordant[(False, False)],
        },
        "both_success_paired_deltas": {
            key: _finite_distribution(
                both_success, lambda row, k=key: row["paired_delta"].get(k)
            )
            for key in keys
        },
        "both_reached_insert_paired_deltas": {
            key: _finite_distribution(
                both_reached_insert,
                lambda row, k=key: row["paired_delta"].get(k),
            )
            for key in keys
        },
        "all_available_paired_deltas": {
            key: _finite_distribution(
                rows, lambda row, k=key: row["paired_delta"].get(k)
            )
            for key in keys
        },
    }


def _initial_astar_diagnostics(planner: Any, worlds: int) -> list[dict[str, Any]]:
    diagnostics = planner.diagnostics()
    rows: list[dict[str, Any]] = [dict() for _ in range(worlds)]
    for key, value in diagnostics.items():
        shape = tuple(getattr(value, "shape", ()))
        if not shape or shape[0] != worlds:
            continue
        cpu = value.detach().cpu()
        if len(shape) != 1:
            continue
        for index in range(worlds):
            scalar = cpu[index].item()
            if isinstance(scalar, float) and not math.isfinite(scalar):
                scalar = None
            rows[index][str(key)] = scalar
    return rows


def _default_health_check(env: Any, *, context: str) -> dict[str, Any]:
    from .workflows import _assert_backend_health

    return dict(_assert_backend_health(env, context=context))


def run_first_episode_stack(
    config: Any,
    *,
    stack: str,
    rows: Sequence[Mapping[str, Any]],
    max_steps: int,
    diffusion_checkpoint: Optional[str | Path] = None,
    health_check: Optional[Callable[..., Mapping[str, Any]]] = None,
    astar_collision_model: str = "exact_mujoco",
) -> dict[str, Any]:
    """Run one controller arm over the already validated manifest rows."""

    torch = _torch()
    from .astar_spline import BatchedAStarSplinePlanner
    from .astar_spline_exact_collision import (
        ExactCollisionBatchedAStarSplinePlanner,
    )
    from .diffusion_runtime import BatchedDiffusionController
    from .env import MJWarpScrewEnv

    worlds = len(rows)
    if stack not in {"diffusion", "astar_spline"}:
        raise ValueError("stack must be diffusion or astar_spline")
    controller = None
    planner = None
    if stack == "diffusion":
        if diffusion_checkpoint is None:
            raise ValueError("Diffusion stack requires a checkpoint")
        inner_controller = BatchedDiffusionController.from_checkpoint(
            diffusion_checkpoint,
            worlds=worlds,
            inference_steps=config.diffusion.inference_steps,
            replan_interval=config.diffusion.replan_interval,
            device=config.runtime.device,
            seed=int(config.runtime.seed),
        )
        controller = TimedDiffusionController(inner_controller)
        env = MJWarpScrewEnv(
            config,
            worlds=worlds,
            baseline="diffusion",
            diffusion_controller=controller,
            curriculum_progress=1.0,
        )
    else:
        if astar_collision_model not in {"conservative_obb", "exact_mujoco"}:
            raise ValueError("unknown A* collision model")
        planner_type = (
            ExactCollisionBatchedAStarSplinePlanner
            if astar_collision_model == "exact_mujoco"
            else BatchedAStarSplinePlanner
        )
        inner_planner = planner_type(
            config, worlds=worlds, device=config.runtime.device
        )
        planner = TimedAStarPlanner(
            inner_planner,
            # MJWarp graph capture consumes only backend state.  The reset
            # after capture is the first state whose A* plan can be executed
            # or included in the paired comparison.
            skip_initial_resets=(1 if bool(config.physics.cuda_graph) else 0),
        )
        env = MJWarpScrewEnv(
            config,
            worlds=worlds,
            baseline="astar_spline",
            nominal_planner=planner,
            curriculum_progress=1.0,
        )
        planner.mark_paired_initial_state()
    backend_health = health_check or _default_health_check
    match = assert_environment_matches_manifest(env, rows)
    initial_astar = (
        _initial_astar_diagnostics(planner, worlds)
        if planner is not None
        else [dict() for _ in range(worlds)]
    )
    zero_residual = torch.zeros(
        (worlds, RESIDUAL_ACTION_DIM),
        device=env.device,
        dtype=torch.float32,
    )
    active = torch.ones(worlds, device=env.device, dtype=torch.bool)
    if controller is not None:
        controller.set_first_episode_active(active)
    if planner is not None:
        planner.set_first_episode_active(active)
    recorder = MeasuredAlignPathRecorder(worlds, max_steps, env.device)
    first_insert_tick = torch.full(
        (worlds,), -1, device=env.device, dtype=torch.long
    )
    previous_tip = env.state.screw_tip().detach().clone()
    records: list[dict[str, Any]] = []
    last_info: Optional[Mapping[str, Any]] = None
    vector_steps = 0
    log_every = max(1, int(config.runtime.log_every_vector_steps))
    last_health = dict(
        backend_health(env, context=f"paired {stack} environment initialization")
    )
    rollout_started = time.perf_counter()
    try:
        for vector_step in range(1, int(max_steps) + 1):
            active_before = active.clone()
            result = env.step(residual_action=zero_residual)
            vector_steps = vector_step
            last_info = result.info
            current_tip = env.state.screw_tip().detach()
            current_phase = env.state.desired_command.phase
            newly_reached_insert = (
                active_before
                & (first_insert_tick < 0)
                & (current_phase >= int(TaskPhase.INSERT))
            )
            first_insert_tick[newly_reached_insert] = result.info[
                "elapsed_steps"
            ][newly_reached_insert]
            recorder.update(
                previous_tip,
                current_tip,
                first_episode_active=active_before,
                learned_active=result.info["diffusion_active"],
                phase=current_phase,
            )
            previous_tip = current_tip.clone()
            first_done = result.done & active
            if bool(first_done.any()):
                ids = (
                    torch.nonzero(first_done, as_tuple=False)
                    .flatten()
                    .detach()
                    .cpu()
                    .tolist()
                )
                for world_id in ids:
                    records.append(
                        _terminal_record(
                            stack=stack,
                            info=result.info,
                            world_id=int(world_id),
                            initial_condition=rows[int(world_id)],
                            forced_truncation=False,
                            max_steps=max_steps,
                        )
                    )
                active[first_done] = False
            if vector_step % log_every == 0:
                last_health = dict(
                    backend_health(
                        env,
                        context=f"paired {stack} vector step {vector_step}",
                    )
                )
            any_active = bool(active.any())
            if bool(result.done.any()) and any_active:
                env.reset(result.done)
                previous_tip = env.state.screw_tip().detach().clone()
            if not any_active:
                break
        if bool(active.any()):
            if last_info is None:
                raise RuntimeError("paired evaluation produced no transition")
            ids = (
                torch.nonzero(active, as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            )
            for world_id in ids:
                records.append(
                    _terminal_record(
                        stack=stack,
                        info=last_info,
                        world_id=int(world_id),
                        initial_condition=rows[int(world_id)],
                        forced_truncation=True,
                        max_steps=max_steps,
                    )
                )
            active.zero_()
        if vector_steps % log_every:
            last_health = dict(
                backend_health(
                    env, context=f"paired {stack} final step {vector_steps}"
                )
            )
        path_summaries = recorder.summaries()
        first_insert_cpu = first_insert_tick.detach().cpu().tolist()
        by_world = {int(record["world_id"]): record for record in records}
        if set(by_world) != set(range(worlds)):
            raise RuntimeError("paired evaluation did not retain one row per world")
        for world_id in range(worlds):
            by_world[world_id]["measured_align_path"] = path_summaries[world_id]
            insert_tick = int(first_insert_cpu[world_id])
            by_world[world_id]["reached_insert"] = insert_tick >= 0
            by_world[world_id]["first_insert_tick"] = (
                insert_tick if insert_tick >= 0 else None
            )
            by_world[world_id]["first_insert_censor_tick"] = int(
                by_world[world_id]["terminal_steps"]
            )
            if stack == "astar_spline":
                by_world[world_id]["initial_planner_diagnostics"] = (
                    initial_astar[world_id]
                )
        records = [by_world[index] for index in range(worlds)]
        if controller is not None:
            relevant_events = [
                event
                for event in controller.events
                if bool(event.get("comparison_relevant", True))
            ]
            background_events = [
                event
                for event in controller.events
                if not bool(event.get("comparison_relevant", True))
            ]
            timing = aggregate_latency_events(relevant_events)
            timing["cold_first_event"] = (
                dict(relevant_events[0]) if relevant_events else None
            )
            timing["steady_state_excluding_first"] = aggregate_latency_events(
                relevant_events[1:]
            )
            timing["by_call_kind"] = {
                kind: aggregate_latency_events(
                    [
                        event
                        for event in relevant_events
                        if event.get("kind") == kind
                    ]
                )
                for kind in ("masked_reset_sample", "full_batch_replan")
            }
            timing["background_only_masked_samples_excluded"] = [
                dict(event) for event in background_events
            ]
            timing["background_only_masked_samples_suppressed"] = int(
                controller.background_only_samples_suppressed
            )
        else:
            assert planner is not None
            index = planner.paired_initial_event_index
            if index != 0 or len(planner.events) != 1:
                raise RuntimeError(
                    "formal one-shot A* evaluation must contain exactly one "
                    "manifest-matched planning event"
                )
            timing = aggregate_latency_events(
                planner.events,
                selected_event_indices=[] if index is None else [index],
            )
            timing["constructor_warmup_events_excluded"] = [
                dict(event)
                for event_index, event in enumerate(planner.events)
                if index is not None and event_index < index
            ]
            timing["constructor_warmup_resets_skipped"] = [
                dict(event) for event in planner.skipped_reset_events
            ]
            timing["background_reset_events_excluded"] = [
                dict(event)
                for event_index, event in enumerate(planner.events)
                if index is not None and event_index > index
            ]
            timing["background_only_planner_resets_suppressed"] = [
                dict(event) for event in planner.background_reset_events
            ]
        duration = max(time.perf_counter() - rollout_started, 1.0e-12)
        return {
            "stack": stack,
            "records": records,
            "summary": aggregate_stack_records(records, timing=timing),
            "latency": timing,
            "initial_condition_match": match,
            "planner_provenance": (
                None if planner is None else planner.provenance()
            ),
            "controller_provenance": (
                None if controller is None else controller.provenance()
            ),
            "execution": {
                "vector_steps": vector_steps,
                "rollout_wall_duration_seconds": duration,
                "throughput_episodes_per_second": worlds / duration,
                "health": last_health,
            },
        }
    finally:
        env.close()


def evaluate_paired_planners(
    config: Any,
    *,
    config_path: str | Path,
    diffusion_checkpoint: str | Path,
    initial_conditions_manifest: str | Path,
    output_dir: str | Path,
    episodes: Optional[int] = None,
    max_steps: Optional[int] = None,
    stack_order: Sequence[str] = ("diffusion", "astar_spline"),
    health_check: Optional[Callable[..., Mapping[str, Any]]] = None,
    astar_collision_model: str = "exact_mujoco",
) -> dict[str, Any]:
    """Run and persist the strict paired comparison on one GPU host."""

    manifest_path = Path(initial_conditions_manifest).expanduser().resolve()
    manifest, evaluation_config, rows = load_paired_initial_conditions(
        manifest_path, config=config, episodes=episodes
    )
    limit = (
        int(evaluation_config.physics.max_episode_steps)
        if max_steps is None
        else int(max_steps)
    )
    if not 1 <= limit <= int(evaluation_config.physics.max_episode_steps):
        raise ValueError("max_steps lies outside the configured episode limit")
    order = tuple(str(value) for value in stack_order)
    if set(order) != {"diffusion", "astar_spline"} or len(order) != 2:
        raise ValueError("stack_order must contain diffusion and astar_spline once")
    if astar_collision_model not in {"conservative_obb", "exact_mujoco"}:
        raise ValueError("unknown A* collision model")
    if int(evaluation_config.diffusion.replan_interval) != 4:
        raise ValueError(
            "formal planning ablation requires Diffusion replanning every 4 ticks"
        )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    occupied = [path for path in output.iterdir() if path.is_file()]
    if occupied:
        raise FileExistsError("paired output directory contains existing files")
    checkpoint = Path(diffusion_checkpoint).expanduser().resolve()
    config_source = Path(config_path).expanduser().resolve()
    if not checkpoint.is_file() or not config_source.is_file():
        raise FileNotFoundError("paired config or Diffusion checkpoint is missing")

    torch = _torch()
    from .contracts import validate_checkpoint
    from .workflows import _validate_frozen_diffusion_config

    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    validate_checkpoint(checkpoint_payload, kind="diffusion")
    # Check against the base training config.  The detached evaluation config
    # intentionally differs only in runtime.seed, which is not checkpoint
    # training state and must not be written back into the checkpoint.
    _validate_frozen_diffusion_config(checkpoint_payload, config)
    del checkpoint_payload
    results: dict[str, Any] = {}
    started = time.perf_counter()
    for stack in order:
        results[stack] = run_first_episode_stack(
            evaluation_config,
            stack=stack,
            rows=rows,
            max_steps=limit,
            diffusion_checkpoint=checkpoint if stack == "diffusion" else None,
            health_check=health_check,
            astar_collision_model=astar_collision_model,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(evaluation_config.runtime.device))
            torch.cuda.empty_cache()
    paired_rows = build_paired_rows(
        results["diffusion"]["records"],
        results["astar_spline"]["records"],
    )
    diffusion_provenance = results["diffusion"]["controller_provenance"]
    astar_provenance = results["astar_spline"]["planner_provenance"]
    if diffusion_provenance.get("replan_interval_ticks") != 4:
        raise RuntimeError("Diffusion runtime violated the 4-tick replan contract")
    if astar_collision_model == "exact_mujoco":
        from .astar_spline_exact_collision import GLOBAL_PLANNER_CONTRACT

        if astar_provenance.get("planner_contract") != GLOBAL_PLANNER_CONTRACT:
            raise RuntimeError("A* runtime violated the one-shot global contract")
        if astar_provenance.get("search", {}).get("replanning") is not False:
            raise RuntimeError("formal A* ablation may not replan during ALIGN")
    episode_paths: dict[str, str] = {}
    for stack in ("diffusion", "astar_spline"):
        path = output / f"{stack}_episodes.jsonl"
        for record in results[stack]["records"]:
            append_history(path, record)
        episode_paths[stack] = str(path)
    paired_path = output / "paired_worlds.jsonl"
    for row in paired_rows:
        append_history(paired_path, row)
    device = torch.device(evaluation_config.runtime.device)
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
        }
    summary = {
        "schema": PAIRED_EVALUATION_SCHEMA,
        "created_utc": utc_now(),
        "comparison": (
            "receding-horizon Diffusion (replan every 4 ticks) versus "
            "one-shot global full-ALIGN A* (zero 16-D residual in both stacks)"
        ),
        "episodes": len(rows),
        "max_steps": limit,
        "first_episode_only": True,
        "same_initial_condition_per_world": True,
        "astar_collision_model": astar_collision_model,
        "residual_contract": {
            "dimensions": RESIDUAL_ACTION_DIM,
            "diffusion": "disabled_zero",
            "astar_spline": "disabled_zero",
        },
        "planning_ablation_contract": {
            "scope": "nominal ALIGN planning strategy",
            "astar": (
                "one global A* plan from nominal supervisor handoff to exact "
                "pre-insertion pose; no replanning"
            ),
            "diffusion": (
                "conditional Diffusion plan refreshed every 4 control ticks"
            ),
            "shared": (
                "paired initial states, zero residual, safety projection, "
                "analytical IK, actuator slew, supervisor gates, and physics"
            ),
        },
        "controller_authority_caveat": (
            "This is the defined planning-strategy ablation: planner family "
            "and planning horizon deliberately co-vary. Shared downstream "
            "supervisor gates, analytical IK, actuator slew limiting, task "
            "physics, initial conditions, and zero residual remain fixed. "
            + (
                "A* uses zero-inflation full-pose filtered MuJoCo "
                "screw--socket contacts with no staging waypoint."
                if astar_collision_model == "exact_mujoco"
                else "A* retains its conservative inflated OBB obstacle."
            )
        ),
        "path_metric_scope": PATH_SCOPE,
        "latency_scope": {
            "host": "same process and GPU host",
            "measurement": "explicit CUDA synchronisation around nominal policy sample/planner reset",
            "diffusion": "batched policy.sample; first event reported separately from steady state",
            "astar": "one-shot full-world global planner.reset that produced the manifest-matched state",
            "warning": (
                "A* runs CPU graph search and uploads its plan; Diffusion runs "
                "batched GPU denoising. Batch latency and amortised sampled-row "
                "latency are reported, not an invented per-world GPU latency."
            ),
        },
        "config": {
            "path": str(config_source),
            "sha256": sha256_file(config_source),
            "base_canonical_sha256": canonical_sha256(config.to_dict()),
            "evaluation_canonical_sha256": canonical_sha256(
                evaluation_config.to_dict()
            ),
        },
        "contracts": {
            "schema_hash": SCHEMA_HASH,
            "task_geometry_hash": TASK_GEOMETRY_HASH,
        },
        "sources": {
            "paired_evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "paired_cli": {
                "path": str(
                    Path(__file__).resolve().parent.parent
                    / "evaluate_paired_planners_gpu.py"
                ),
                "sha256": sha256_file(
                    Path(__file__).resolve().parent.parent
                    / "evaluate_paired_planners_gpu.py"
                ),
            },
        },
        "initial_conditions_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "rows_canonical_sha256": canonical_sha256(rows),
            "source_rows_canonical_sha256": canonical_sha256(manifest["rows"]),
        },
        "diffusion_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "stack_order": list(order),
        "gpu": gpu,
        "diffusion": {
            "summary": results["diffusion"]["summary"],
            "planner_provenance": diffusion_provenance,
            "execution": results["diffusion"]["execution"],
            "initial_condition_match": results["diffusion"][
                "initial_condition_match"
            ],
        },
        "astar_spline": {
            "summary": results["astar_spline"]["summary"],
            "planner_provenance": astar_provenance,
            "execution": results["astar_spline"]["execution"],
            "initial_condition_match": results["astar_spline"][
                "initial_condition_match"
            ],
        },
        "paired": aggregate_paired_rows(paired_rows),
        "wall_duration_seconds": time.perf_counter() - started,
        "outputs": {
            "diffusion_episodes": episode_paths["diffusion"],
            "astar_spline_episodes": episode_paths["astar_spline"],
            "paired_worlds": str(paired_path),
            "summary": str(output / "summary.json"),
        },
    }
    summary_path = atomic_write_json(output / "summary.json", summary)
    atomic_write_json(
        output / "summary.sha256.json",
        {
            "algorithm": "sha256",
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
    )
    for path in (
        Path(episode_paths["diffusion"]),
        Path(episode_paths["astar_spline"]),
        paired_path,
    ):
        atomic_write_json(
            path.with_name(path.name + ".sha256.json"),
            {
                "algorithm": "sha256",
                "path": str(path),
                "sha256": sha256_file(path),
            },
        )
    return summary


__all__ = [
    "PATH_SCOPE",
    "PAIRED_EVALUATION_SCHEMA",
    "MeasuredAlignPathRecorder",
    "TimedAStarPlanner",
    "TimedDiffusionController",
    "aggregate_latency_events",
    "aggregate_paired_rows",
    "assert_environment_matches_manifest",
    "build_paired_rows",
    "evaluate_paired_planners",
    "load_paired_initial_conditions",
    "run_first_episode_stack",
    "summarise_measured_path",
]
