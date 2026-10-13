#!/usr/bin/env python3
"""Evaluate Diffusion--IK--TQC at every integer socket-opening azimuth.

The sweep assigns opening azimuths 1..359 degrees to world IDs 0..358.  The
socket elevation remains fixed at 45 degrees and the screw spawn is sampled
from the configured full-difficulty region: either the opening-facing
half-disc or the full workspace disc with conservative socket exclusion.
Prefix-IK rejection resampling preserves the world-to-azimuth mapping: every
pending slot always reuses the mouth position belonging to that slot.

The residual actor is evaluated deterministically after nominal analytical IK
and with the exact per-tick actuator authority mask supplied by the
environment.  The TQC checkpoint is accepted only when it pins the supplied
Diffusion checkpoint by SHA-256.  Checkpoint/config comparisons are fail-closed
apart from the two deployment-only fields ``name`` and ``physics.worlds``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
FORMAL_ANGLES_DEG = tuple(range(1, 360))
FORMAL_SOCKET_ELEVATION_DEG = 45.0
FORMAL_MOUTH_ORBIT_RADIUS_M = 0.020
FORMAL_SPAWN_RADIUS_M = 0.100
ALLOWED_CONFIG_DIFFERENCES = frozenset({"name", "physics.worlds"})

PHASE_NAMES = {
    0: "pregrasp",
    1: "close",
    2: "lift",
    3: "align",
    4: "insert",
    5: "engage",
    6: "power",
    7: "lr_open",
    8: "lr_reset",
    9: "lr_close",
    10: "ud_open",
    11: "ud_reset",
    12: "ud_close",
    13: "final_hold",
    14: "complete",
}


def _find_hpc_root() -> Path:
    candidates = (
        SCRIPT_DIR.parent / "hpc",  # Diffusion/evaluation -> Diffusion/hpc
        SCRIPT_DIR.parent,  # supports copying evaluation beneath hpc
        SCRIPT_DIR,  # script copied directly into hpc
    )
    for candidate in candidates:
        if (candidate / "s_hpc").is_dir() and (
            candidate / "assets" / "Project_hpc.xml"
        ).is_file():
            return candidate.resolve()
    raise RuntimeError(
        "Cannot locate the matching hpc source tree and assets. Keep this "
        "script in Diffusion/evaluation or beneath hpc."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def _load_torch_checkpoint(torch, path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older Torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a dictionary: {path}")
    return payload


def _checkpoint_config(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    value = payload.get("config")
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} checkpoint does not contain a config object")
    # JSON round-tripping gives a detached, plain mapping and rejects objects
    # that cannot be represented in the immutable configuration contract.
    result = json.loads(json.dumps(dict(value), allow_nan=False))
    curriculum = result.get("curriculum")
    if isinstance(curriculum, dict):
        curriculum.setdefault(
            "restrict_screw_spawn_to_opening_half_disk", True
        )
    if not isinstance(result.get("name"), str):
        raise ValueError(f"{label} checkpoint config has no valid name")
    physics = result.get("physics")
    if not isinstance(physics, dict) or not isinstance(physics.get("worlds"), int):
        raise ValueError(
            f"{label} checkpoint config has no valid physics.worlds"
        )
    return result


def _mapping_difference_paths(
    first: Any,
    second: Any,
    *,
    prefix: str = "",
) -> list[str]:
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        differences: list[str] = []
        for key in sorted(set(first) | set(second), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in first or key not in second:
                differences.append(path)
                continue
            differences.extend(
                _mapping_difference_paths(
                    first[key], second[key], prefix=path
                )
            )
        return differences
    return [] if first == second else [prefix or "<root>"]


def assert_config_compatible(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    reference_label: str,
    candidate_label: str,
) -> list[str]:
    differences = _mapping_difference_paths(reference, candidate)
    forbidden = sorted(set(differences) - ALLOWED_CONFIG_DIFFERENCES)
    if forbidden:
        raise ValueError(
            f"{reference_label} and {candidate_label} configurations differ "
            f"outside name/physics.worlds: {forbidden}"
        )
    return differences


def _tensor_item(info: Mapping[str, Any], key: str, index: int, default: Any) -> Any:
    value = info.get(key)
    if value is None:
        return default
    selected = value[index]
    return selected.item() if hasattr(selected, "item") else selected


def failure_reasons(
    info: Mapping[str, Any],
    index: int,
    phase: str,
    *,
    forced_truncation: bool = False,
) -> list[str]:
    reasons: list[str] = []
    for key in (
        "nonfinite",
        "fell_below_table",
        "lateral_escape",
        "safety_stop",
        "contact_constraint_violation",
        "contact_violation_before_capture",
        "contact_violation_before_thread",
        "contact_violation_premature_regrasp",
        "contact_violation_material_drift",
    ):
        if bool(_tensor_item(info, key, index, False)):
            reasons.append(key)
    if bool(_tensor_item(info, "safety_stop", index, False)):
        reasons.append("ik_exhausted")

    truncated = forced_truncation or bool(
        _tensor_item(info, "truncated", index, False)
    )
    if truncated:
        if (
            int(
                _tensor_item(
                    info, "diffusion_handoff_hold_steps", index, 0
                )
            )
            > 0
            and not bool(
                _tensor_item(
                    info, "diffusion_handoff_ready", index, False
                )
            )
        ):
            reasons.append("handoff_timeout")
        elif bool(
            _tensor_item(info, "preinsert_alignment_hold", index, False)
        ):
            reasons.append("preinsert_gate_timeout")
        elif bool(
            _tensor_item(info, "capture_alignment_hold", index, False)
        ):
            reasons.append("capture_gate_timeout")
        elif bool(
            _tensor_item(info, "thread_engagement_hold", index, False)
        ):
            reasons.append("thread_gate_timeout")
        elif bool(
            _tensor_item(info, "thread_regrasp_hold", index, False)
        ):
            reasons.append("thread_regrasp_gate_timeout")
        elif phase == "insert" and not bool(
            _tensor_item(info, "shaft_captured", index, False)
        ):
            reasons.append("capture_gate_timeout")
        elif phase == "insert":
            reasons.append("thread_gate_timeout")
        elif phase == "engage" and not bool(
            _tensor_item(info, "thread_engaged", index, False)
        ):
            reasons.append("thread_not_engaged")
        elif phase in {
            "power",
            "lr_open",
            "lr_reset",
            "lr_close",
            "ud_open",
            "ud_reset",
            "ud_close",
        }:
            reasons.append("rotation_or_regrasp_stall")
        elif phase in {"final_hold", "complete"}:
            reasons.append("final_tolerance_timeout")
        else:
            reasons.append(f"timeout_{phase}")
        if forced_truncation:
            reasons.append("sweep_max_steps")
    return list(dict.fromkeys(reasons))


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty sweep CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
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
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--tqc", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="evaluation seed (default: config runtime seed + 100000)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="sweep cutoff (default: configured episode limit)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hpc_root = _find_hpc_root()
    if str(hpc_root) not in sys.path:
        sys.path.insert(0, str(hpc_root))

    diffusion_path = args.diffusion.expanduser().resolve()
    tqc_path = args.tqc.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    metadata_path = (
        args.metadata_output.expanduser().resolve()
        if args.metadata_output is not None
        else output_path.with_suffix(".metadata.json")
    )
    for label, path in (
        ("Diffusion checkpoint", diffusion_path),
        ("TQC checkpoint", tqc_path),
        ("configuration", config_path),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} does not exist: {path}")
    if output_path == metadata_path:
        raise SystemExit("--output and --metadata-output must be different files")

    from s_hpc.config import load_config

    config = load_config(config_path)
    evaluation_seed = (
        int(config.runtime.seed) + 100_000
        if args.seed is None
        else int(args.seed)
    )
    if not 0 <= evaluation_seed < 2**63:
        raise SystemExit("--seed must be in [0, 2^63)")
    max_steps = (
        int(config.physics.max_episode_steps)
        if args.max_steps is None
        else int(args.max_steps)
    )
    if max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if (
        abs(
            float(config.curriculum.fixed_socket_angle_deg)
            - FORMAL_SOCKET_ELEVATION_DEG
        )
        > 1.0e-9
    ):
        raise SystemExit("formal sweep requires a fixed 45-degree elevation")
    if (
        abs(
            float(config.curriculum.final_spawn_radius_m)
            - FORMAL_SPAWN_RADIUS_M
        )
        > 1.0e-9
    ):
        raise SystemExit("formal sweep requires the 100 mm spawn radius")

    import torch

    from s_hpc.contracts import (
        TASK_ASSET_HASHES,
        TASK_GEOMETRY_HASH,
        validate_checkpoint,
    )
    from s_hpc.diffusion_runtime import BatchedDiffusionController
    from s_hpc.env import MJWarpScrewEnv, RESET_IK_RESAMPLE_BATCHES
    from s_hpc.tqc import TQCTrainer

    diffusion_payload = _load_torch_checkpoint(torch, diffusion_path)
    tqc_payload = _load_torch_checkpoint(torch, tqc_path)
    validate_checkpoint(diffusion_payload, kind="diffusion")
    validate_checkpoint(tqc_payload, kind="tqc")

    diffusion_sha256 = sha256_file(diffusion_path)
    tqc_sha256 = sha256_file(tqc_path)
    pinned_diffusion_sha256 = tqc_payload.get("diffusion_sha256")
    if (
        not isinstance(pinned_diffusion_sha256, str)
        or len(pinned_diffusion_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in pinned_diffusion_sha256
        )
        or pinned_diffusion_sha256 != diffusion_sha256
    ):
        raise ValueError(
            "TQC checkpoint does not pin the supplied Diffusion checkpoint "
            "by SHA-256"
        )

    selected_config = config.to_dict()
    diffusion_config = _checkpoint_config(
        diffusion_payload, label="Diffusion"
    )
    tqc_config = _checkpoint_config(tqc_payload, label="TQC")
    diffusion_config_differences = assert_config_compatible(
        selected_config,
        diffusion_config,
        reference_label="selected",
        candidate_label="Diffusion checkpoint",
    )
    tqc_config_differences = assert_config_compatible(
        selected_config,
        tqc_config,
        reference_label="selected",
        candidate_label="TQC checkpoint",
    )
    checkpoint_config_differences = assert_config_compatible(
        diffusion_config,
        tqc_config,
        reference_label="Diffusion checkpoint",
        candidate_label="TQC checkpoint",
    )

    xml_path = hpc_root / "assets" / "Project_hpc.xml"
    xml_sha256 = sha256_file(xml_path)
    if TASK_ASSET_HASHES.get("Project_hpc.xml") != xml_sha256:
        raise RuntimeError("runtime Project_hpc.xml hash drifted from contracts")

    angles = np.asarray(FORMAL_ANGLES_DEG, dtype=np.float64)
    azimuth_rad = np.deg2rad(angles)
    # The opening direction is opposite the socket mouth's radial position.
    fixed_mouth_xy = -FORMAL_MOUTH_ORBIT_RADIUS_M * np.column_stack(
        (np.cos(azimuth_rad), np.sin(azimuth_rad))
    )
    worlds = len(FORMAL_ANGLES_DEG)

    class FixedAzimuthEnv(MJWarpScrewEnv):
        def __init__(
            self,
            *env_args,
            fixed_mouths: np.ndarray,
            **env_kwargs,
        ) -> None:
            self._fixed_mouths_cpu = np.asarray(
                fixed_mouths, dtype=np.float32
            ).copy()
            super().__init__(*env_args, **env_kwargs)

        def _sample_reachable(self, mask):
            spawn = self.state.spawn_xy.clone()
            socket_angle = self.state.socket_angle_deg.clone()
            socket_mouth_xy = self.state.socket_mouth[:, :2].clone()
            fixed = torch.as_tensor(
                self._fixed_mouths_cpu,
                device=self.device,
                dtype=self.state.socket_mouth.dtype,
            )
            if tuple(fixed.shape) != (self.worlds, 2):
                raise RuntimeError(
                    f"fixed mouth table has shape {tuple(fixed.shape)}, "
                    f"expected {(self.worlds, 2)}"
                )
            pending = torch.as_tensor(
                mask, device=self.device, dtype=torch.bool
            ).clone()
            final_command = self.current_command.clone()
            final_ctrl = torch.zeros(
                (self.worlds, 16),
                device=self.device,
                dtype=torch.float32,
            )
            final_passive = None
            spawn_radius, fixed_socket_angle_deg = (
                self.supervisor.curriculum_parameters(
                    self.curriculum_progress
                )
            )
            if (
                abs(
                    float(fixed_socket_angle_deg)
                    - FORMAL_SOCKET_ELEVATION_DEG
                )
                > 1.0e-9
            ):
                raise RuntimeError("curriculum changed the socket elevation")
            if abs(float(spawn_radius) - FORMAL_SPAWN_RADIUS_M) > 1.0e-9:
                raise RuntimeError(
                    "full-difficulty sweep requires the 100 mm spawn radius"
                )

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

                # Critical indexing rule: rejected slots keep their original
                # world IDs, so each retry selects that world's fixed mouth.
                mouth_proposal = fixed.index_select(0, slots)
                angle_proposal = self.supervisor.fixed_angle_batch(needed)
                proposal = self.supervisor.sample_spawn(
                    needed,
                    socket_mouth_xy=mouth_proposal,
                    socket_angle_deg=angle_proposal,
                    radius_m=float(spawn_radius),
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
                "fixed-azimuth prefix rejection sampler exhausted "
                f"{RESET_IK_RESAMPLE_BATCHES} batches with "
                f"{remaining} world(s) still unreachable"
            )

    torch.manual_seed(evaluation_seed)
    torch.cuda.manual_seed_all(evaluation_seed)
    controller = BatchedDiffusionController.from_checkpoint(
        diffusion_path,
        worlds=worlds,
        inference_steps=config.diffusion.inference_steps,
        replan_interval=config.diffusion.replan_interval,
        device=config.runtime.device,
        seed=evaluation_seed,
    )
    trainer = TQCTrainer(config, device=config.runtime.device)
    trainer.actor.load_state_dict(tqc_payload["actor"], strict=True)
    trainer.actor.eval()
    trainer.transitions = int(tqc_payload.get("transitions", 0))
    trainer.updates = int(tqc_payload.get("updates", 0))
    if not all(
        bool(torch.isfinite(parameter).all())
        for parameter in trainer.actor.parameters()
    ):
        raise FloatingPointError("TQC actor contains NaN or infinity")

    started = time.monotonic()
    env = FixedAzimuthEnv(
        config,
        worlds=worlds,
        baseline="diffusion",
        diffusion_controller=controller,
        curriculum_progress=1.0,
        fixed_mouths=fixed_mouth_xy,
    )

    expected_mouth = torch.as_tensor(
        fixed_mouth_xy,
        device=env.device,
        dtype=env.state.socket_mouth.dtype,
    )
    initial_mouth_error = float(
        (env.state.socket_mouth[:, :2] - expected_mouth)
        .abs()
        .max()
        .item()
    )
    initial_elevation_error = float(
        (
            env.state.socket_angle_deg
            - FORMAL_SOCKET_ELEVATION_DEG
        )
        .abs()
        .max()
        .item()
    )
    if initial_mouth_error > 1.0e-6 or initial_elevation_error > 1.0e-6:
        env.close()
        raise RuntimeError(
            "fixed-azimuth initialization drifted: "
            f"mouth_error={initial_mouth_error:.3e}, "
            f"elevation_error={initial_elevation_error:.3e}"
        )

    active = torch.ones(worlds, device=env.device, dtype=torch.bool)
    rows: list[dict[str, object] | None] = [None] * worlds
    residual_abs_sum = torch.zeros(
        worlds, device=env.device, dtype=torch.float64
    )
    residual_abs_max = torch.zeros_like(residual_abs_sum)
    channel_tick_count = torch.zeros_like(residual_abs_sum)
    mask_active_count = torch.zeros_like(residual_abs_sum)
    ctrl_saturation_count = torch.zeros_like(residual_abs_sum)
    action_clipped_count = torch.zeros_like(residual_abs_sum)
    last_result = None

    def accumulate_residual_metrics(result) -> None:
        info = result.info
        residual = info["executed_residual_action"]
        action_mask = info["residual_action_mask"]
        saturation = info["residual_ctrl_saturation_mask"]
        clipped = info["residual_action_clipped_mask"]
        expected_shape = (worlds, 16)
        for label, value in (
            ("executed residual", residual),
            ("residual action mask", action_mask),
            ("control saturation mask", saturation),
            ("action clipped mask", clipped),
        ):
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(
                    f"{label} shape {tuple(value.shape)} != {expected_shape}"
                )
        for label, value in (
            ("residual action mask", action_mask),
            ("control saturation mask", saturation),
            ("action clipped mask", clipped),
        ):
            if value.dtype != torch.bool:
                raise TypeError(f"{label} must use boolean dtype")
        if not bool(torch.isfinite(residual).all()):
            raise FloatingPointError("executed residual became non-finite")
        active_columns = active[:, None]
        zeros = torch.zeros_like(residual)
        absolute = torch.where(active_columns, residual.abs(), zeros)
        residual_abs_sum.add_(absolute.sum(dim=1).to(torch.float64))
        residual_abs_max.copy_(
            torch.maximum(
                residual_abs_max,
                absolute.amax(dim=1).to(torch.float64),
            )
        )
        channels = float(expected_shape[1])
        channel_tick_count.add_(active.to(torch.float64) * channels)
        mask_active_count.add_(
            (action_mask & active_columns).sum(dim=1).to(torch.float64)
        )
        ctrl_saturation_count.add_(
            (saturation & active_columns).sum(dim=1).to(torch.float64)
        )
        action_clipped_count.add_(
            (clipped & active_columns).sum(dim=1).to(torch.float64)
        )

    def build_row(
        result,
        world_id: int,
        *,
        forced_truncation: bool,
    ) -> dict[str, object]:
        info = result.info
        phase_id = int(_tensor_item(info, "task_phase", world_id, -1))
        phase = PHASE_NAMES.get(phase_id, f"phase_{phase_id}")
        denominator = max(float(channel_tick_count[world_id].item()), 1.0)
        actual_mouth = env.state.socket_mouth[world_id, :2]
        mouth_error = float(
            (actual_mouth - expected_mouth[world_id]).abs().max().item()
        )
        if mouth_error > 1.0e-6:
            raise RuntimeError(
                f"world {world_id} lost its fixed azimuth "
                f"(mouth error {mouth_error:.3e} m)"
            )
        succeeded = bool(_tensor_item(info, "success", world_id, False))
        failed = bool(_tensor_item(info, "failure", world_id, False))
        truncated = forced_truncation or bool(
            _tensor_item(info, "truncated", world_id, False)
        )
        if forced_truncation:
            succeeded = False
        return {
            "opening_azimuth_deg": FORMAL_ANGLES_DEG[world_id],
            "socket_elevation_deg": float(
                _tensor_item(
                    info,
                    "socket_angle_deg",
                    world_id,
                    FORMAL_SOCKET_ELEVATION_DEG,
                )
            ),
            "socket_mouth_x_m": float(actual_mouth[0].item()),
            "socket_mouth_y_m": float(actual_mouth[1].item()),
            "world_id": world_id,
            "evaluation_seed": evaluation_seed,
            "steps": int(_tensor_item(info, "elapsed_steps", world_id, 0)),
            "final_depth_mm": 1000.0
            * float(_tensor_item(info, "depth", world_id, 0.0)),
            "final_turns": float(
                _tensor_item(info, "twist", world_id, 0.0)
            )
            / (2.0 * math.pi),
            "success": succeeded,
            "failure": failed,
            "truncated": truncated,
            "terminal_phase": phase,
            "return": float(
                _tensor_item(info, "episode_return", world_id, 0.0)
            ),
            "lateral_error_mm": 1000.0
            * float(_tensor_item(info, "lateral_error", world_id, 0.0)),
            "axis_error_deg": math.degrees(
                float(_tensor_item(info, "axis_error", world_id, 0.0))
            ),
            "residual_abs_mean": float(
                residual_abs_sum[world_id].item()
            )
            / denominator,
            "residual_abs_max": float(
                residual_abs_max[world_id].item()
            ),
            "mask_active_fraction": float(
                mask_active_count[world_id].item()
            )
            / denominator,
            "ctrl_saturation_fraction": float(
                ctrl_saturation_count[world_id].item()
            )
            / denominator,
            "action_clipped_fraction": float(
                action_clipped_count[world_id].item()
            )
            / denominator,
            "ik_failures": int(
                _tensor_item(info, "ik_failures", world_id, 0)
            ),
            "ik_anchor_fallbacks": int(
                _tensor_item(info, "ik_anchor_fallbacks", world_id, 0)
            ),
            "ik_safety_stops": int(
                _tensor_item(info, "ik_safety_stops", world_id, 0)
            ),
            "sweep_cutoff": forced_truncation,
            "failure_reasons": ";".join(
                failure_reasons(
                    info,
                    world_id,
                    phase,
                    forced_truncation=forced_truncation,
                )
            ),
        }

    try:
        for vector_step in range(1, max_steps + 1):
            def residual_policy(tqc_state, action_mask):
                return trainer.act(
                    tqc_state,
                    action_mask=action_mask,
                    deterministic=True,
                )

            result = env.step(residual_policy=residual_policy)
            last_result = result
            accumulate_residual_metrics(result)
            newly_done = result.done & active
            done_ids = torch.nonzero(
                newly_done, as_tuple=False
            ).flatten()
            if int(done_ids.numel()):
                for world_id in done_ids.detach().cpu().tolist():
                    rows[world_id] = build_row(
                        result, world_id, forced_truncation=False
                    )
                active[done_ids] = False
                print(
                    json.dumps(
                        {
                            "event": "tqc_azimuth_sweep_progress",
                            "vector_step": vector_step,
                            "completed": worlds - int(active.sum().item()),
                            "total": worlds,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if not bool(active.any()):
                break

        if bool(active.any()):
            if last_result is None:
                raise RuntimeError("sweep produced no environment step")
            cutoff_ids = torch.nonzero(
                active, as_tuple=False
            ).flatten()
            for world_id in cutoff_ids.detach().cpu().tolist():
                rows[world_id] = build_row(
                    last_result, world_id, forced_truncation=True
                )
            active[cutoff_ids] = False
            print(
                json.dumps(
                    {
                        "event": "tqc_azimuth_sweep_cutoff",
                        "vector_step": max_steps,
                        "completed": worlds,
                        "forced_truncations": int(cutoff_ids.numel()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        env.close()

    if any(row is None for row in rows):
        missing = [index for index, row in enumerate(rows) if row is None]
        raise RuntimeError(f"missing terminal records for worlds {missing}")
    completed_rows = [row for row in rows if row is not None]
    if [int(row["opening_azimuth_deg"]) for row in completed_rows] != list(
        FORMAL_ANGLES_DEG
    ):
        raise RuntimeError("world-to-opening-azimuth order drifted")

    _atomic_write_csv(output_path, completed_rows)
    output_sha256 = sha256_file(output_path)
    success_count = sum(bool(row["success"]) for row in completed_rows)
    failure_count = sum(bool(row["failure"]) for row in completed_rows)
    truncated_count = sum(bool(row["truncated"]) for row in completed_rows)
    cutoff_count = sum(bool(row["sweep_cutoff"]) for row in completed_rows)
    mapping = [
        {
            "world_id": index,
            "opening_azimuth_deg": angle,
            "socket_mouth_xy_m": fixed_mouth_xy[index].tolist(),
        }
        for index, angle in enumerate(FORMAL_ANGLES_DEG)
    ]
    metadata = {
        "schema": "screw-tqc-azimuth-sweep-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "control_stack": "diffusion_safety_analytic_ik_post_ik_tqc",
        "deterministic_policy": True,
        "diffusion": {
            "path": str(diffusion_path),
            "sha256": diffusion_sha256,
            "format": diffusion_payload.get("format"),
            "epoch": diffusion_payload.get("epoch"),
            "global_step": diffusion_payload.get("global_step"),
            "config_name": diffusion_config["name"],
            "configured_worlds": diffusion_config["physics"]["worlds"],
            "config_canonical_sha256": canonical_sha256(
                diffusion_config
            ),
        },
        "tqc": {
            "path": str(tqc_path),
            "sha256": tqc_sha256,
            "policy_only": bool(tqc_payload.get("policy_only")),
            "transitions": int(tqc_payload.get("transitions", 0)),
            "updates": int(tqc_payload.get("updates", 0)),
            "config_name": tqc_config["name"],
            "configured_worlds": tqc_config["physics"]["worlds"],
            "config_canonical_sha256": canonical_sha256(tqc_config),
            "pinned_diffusion_sha256": pinned_diffusion_sha256,
        },
        "config": {
            "path": str(config_path),
            "file_sha256": sha256_file(config_path),
            "canonical_sha256": canonical_sha256(selected_config),
            "name": config.name,
            "configured_worlds": config.physics.worlds,
            "allowed_runtime_width_differences": sorted(
                ALLOWED_CONFIG_DIFFERENCES
            ),
            "diffusion_differences": diffusion_config_differences,
            "tqc_differences": tqc_config_differences,
            "checkpoint_pair_differences": (
                checkpoint_config_differences
            ),
        },
        "contracts": {
            "task_geometry_hash": TASK_GEOMETRY_HASH,
            "diffusion_schema_hash": diffusion_payload.get("schema_hash"),
            "tqc_schema_hash": tqc_payload.get("schema_hash"),
            "asset_sha256": TASK_ASSET_HASHES,
        },
        "xml": {
            "path": str(xml_path),
            "sha256": xml_sha256,
        },
        "conditions": {
            "opening_azimuth_deg": [1, 359],
            "opening_azimuth_count": worlds,
            "opening_azimuth_increment_deg": 1,
            "worlds": worlds,
            "one_episode_per_world": True,
            "socket_elevation_deg": FORMAL_SOCKET_ELEVATION_DEG,
            "socket_mouth_orbit_radius_m": FORMAL_MOUTH_ORBIT_RADIUS_M,
            "spawn_region": (
                "opening-facing half-disc"
                if config.curriculum.restrict_screw_spawn_to_opening_half_disk
                else "full workspace disc minus conservative socket exclusion"
            ),
            "restrict_screw_spawn_to_opening_half_disk": bool(
                config.curriculum.restrict_screw_spawn_to_opening_half_disk
            ),
            "spawn_radius_m": FORMAL_SPAWN_RADIUS_M,
            "curriculum_progress": 1.0,
            "evaluation_seed": evaluation_seed,
            "max_steps": max_steps,
            "configured_episode_limit": config.physics.max_episode_steps,
            "diffusion_inference_steps": config.diffusion.inference_steps,
            "diffusion_replan_interval": config.diffusion.replan_interval,
            "diffusion_eta": 0.0,
            "tqc_actor_mode": "deterministic_mean",
            "post_ik_action_mask": True,
            "fixed_mouth_reapplied_by_world_on_prefix_rejection": True,
            "initial_fixed_mouth_max_error_m": initial_mouth_error,
            "initial_fixed_elevation_max_error_deg": (
                initial_elevation_error
            ),
            "world_angle_mapping_sha256": canonical_sha256(mapping),
        },
        "metric_definitions": {
            "residual_abs_mean": (
                "mean absolute accepted normalised TQC residual over all "
                "16 actuator channels and executed episode ticks"
            ),
            "residual_abs_max": (
                "maximum absolute accepted normalised TQC residual"
            ),
            "mask_active_fraction": (
                "fraction of actuator-channel ticks with residual authority"
            ),
            "ctrl_saturation_fraction": (
                "fraction of actuator-channel ticks clipped by ctrlrange"
            ),
            "action_clipped_fraction": (
                "fraction of actuator-channel ticks whose actor action was "
                "clipped to the normalised action range"
            ),
        },
        "summary": {
            "successes": success_count,
            "failures": failure_count,
            "truncations": truncated_count,
            "sweep_cutoffs": cutoff_count,
            "success_rate": success_count / worlds,
            "median_steps": float(
                np.median(
                    [float(row["steps"]) for row in completed_rows]
                )
            ),
            "mean_residual_abs_mean": float(
                np.mean(
                    [
                        float(row["residual_abs_mean"])
                        for row in completed_rows
                    ]
                )
            ),
            "max_residual_abs_max": max(
                float(row["residual_abs_max"])
                for row in completed_rows
            ),
            "wall_time_s": time.monotonic() - started,
        },
        "output_csv": str(output_path),
        "output_csv_sha256": output_sha256,
        "script": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
    }
    _atomic_write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
