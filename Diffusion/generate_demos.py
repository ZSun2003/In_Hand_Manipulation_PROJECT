#!/usr/bin/env python3
"""Generate full-state expert demonstrations for the Diffusion policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from demo_dataset import (
    DatasetMetadata,
    discover_episode_paths,
    save_episode,
    validate_dataset,
    write_metadata,
)
from runtime import PROJECT_ROOT, ensure_runtime_dirs
from screw_config import DEFAULT_CONFIG
from screw_scene import sample_socket_mouth_xy
from screw_sim import SimulationSettings, ScrewSimulator
from screw_control import SCRIPTED_PREFIX_STEPS


def default_dataset_dir() -> Path:
    configured = os.environ.get("SCREW_DATASET_DIR")
    root = Path(configured).resolve() if configured else PROJECT_ROOT / "datasets"
    return root / "demos"


def generate_episode(
    *,
    seed: int,
    random_spawn: bool,
    random_socket_pose: bool,
) -> Dict[str, np.ndarray | Dict[str, float | bool]]:
    rng = np.random.default_rng(seed)
    if random_socket_pose:
        socket_mouth_xy = sample_socket_mouth_xy(rng, DEFAULT_CONFIG)
        socket_angle_deg = 45.0
        config = DEFAULT_CONFIG.with_socket_pose(
            socket_mouth_xy, socket_angle_deg
        )
    else:
        config = DEFAULT_CONFIG
        socket_mouth_xy = config.socket_mouth_xy_m
        socket_angle_deg = config.socket_table_angle_deg
    spawn = (
        ScrewSimulator.sample_reachable_spawn(config=config, seed=seed)
        if random_spawn
        else config.default_screw_xy_m
    )
    simulator = ScrewSimulator(
        config=config,
        screw_xy_m=spawn,
        seed=seed,
        settings=SimulationSettings(
            strict_ik=True,
            guide_mode="mocap",
            guide_scale=1.0,
        ),
    )
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    phases: List[int] = []
    depths: List[float] = []
    twists: List[float] = []
    contacts: List[int] = []
    try:
        while not simulator.expert.complete:
            previous = simulator.current_command.copy()
            observations.append(simulator.observation())
            following = simulator.expert.advance()
            actions.append(
                simulator.expert.command_action(previous, following)
            )
            diagnostics = simulator.apply_command(following)
            if not diagnostics.ik_success:
                raise RuntimeError(
                    f"expert IK failed at step {simulator.control_step}, "
                    f"phase={following.phase.name}"
                )
            phases.append(int(following.phase))
            depths.append(float(diagnostics.insertion_depth_m))
            twists.append(float(diagnostics.twist_rad))
            contacts.append(int(diagnostics.contact_count))
        metrics = simulator.success_metrics()
        if not bool(metrics["success"]):
            raise RuntimeError(
                "expert episode did not finish successfully: "
                + json.dumps(metrics, sort_keys=True)
            )
        return {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "phase_ids": np.asarray(phases, dtype=np.int16),
            "depth_m": np.asarray(depths, dtype=np.float32),
            "twist_rad": np.asarray(twists, dtype=np.float32),
            "contact_count": np.asarray(contacts, dtype=np.int8),
            "spawn_xy": np.asarray(spawn, dtype=np.float32)[None, :],
            "socket_mouth_xy": np.asarray(
                socket_mouth_xy, dtype=np.float32
            )[None, :],
            "socket_angle_deg": np.asarray(
                [socket_angle_deg], dtype=np.float32
            ),
            "success": np.ones((1,), dtype=np.int8),
            "metrics": metrics,
        }
    finally:
        simulator.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic IK/regrasp/two-turn demonstrations. "
            "All output remains inside SCREW."
        )
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", type=Path, default=default_dataset_dir())
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument(
        "--fixed-spawn",
        action="store_true",
        help="Use the canonical spawn instead of random reachable samples.",
    )
    parser.add_argument(
        "--fixed-socket",
        action="store_true",
        help="Use the canonical socket mouth azimuth instead of a random one.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite episodes with matching numeric IDs.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    ensure_runtime_dirs()
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(
            f"dataset output must remain inside SCREW: {output}"
        ) from exc

    # One lightweight instance determines the exact omniscient schema width.
    probe = ScrewSimulator(
        settings=SimulationSettings(
            strict_ik=True,
            guide_mode="mocap",
        )
    )
    try:
        obs_dim = int(probe.observation().shape[0])
        action_dim = int(
            probe.expert.command_action(
                probe.expert.current,
                probe.expert.advance(),
            ).shape[0]
        )
        dt = float(probe.control_dt_s)
    finally:
        probe.close()
    metadata = DatasetMetadata(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_horizon=args.obs_horizon,
        horizon=args.action_horizon,
        policy_start_step=SCRIPTED_PREFIX_STEPS,
        dt=dt,
        observation_description=(
            "omniscient qpos/qvel/forces/contacts/screw/socket/thread state"
        ),
        action_description=(
            "14D socket-frame pose/material-angle/tip-distance increment"
        ),
    )
    write_metadata(output, metadata, overwrite=args.overwrite)

    existing = len(discover_episode_paths(output))
    start_time = time.monotonic()
    for offset in range(args.episodes):
        episode_id = offset if args.overwrite else existing + offset
        episode_seed = int(args.seed + episode_id)
        episode = generate_episode(
            seed=episode_seed,
            random_spawn=not args.fixed_spawn,
            random_socket_pose=not args.fixed_socket,
        )
        path = save_episode(
            output,
            episode_id,
            episode["observations"],
            episode["actions"],
            extras={
                "phase_ids": episode["phase_ids"],
                "depth_m": episode["depth_m"],
                "twist_rad": episode["twist_rad"],
                "contact_count": episode["contact_count"],
                "spawn_xy": episode["spawn_xy"],
                "socket_mouth_xy": episode["socket_mouth_xy"],
                "socket_angle_deg": episode["socket_angle_deg"],
                "success": episode["success"],
            },
            overwrite=args.overwrite,
        )
        elapsed = time.monotonic() - start_time
        metrics = episode["metrics"]
        print(
            f"[{offset + 1}/{args.episodes}] {path.name} "
            f"spawn={np.asarray(episode['spawn_xy']).reshape(-1).round(4).tolist()} "
            f"mouth={np.asarray(episode['socket_mouth_xy']).reshape(-1).round(4).tolist()} "
            f"angle={float(np.asarray(episode['socket_angle_deg']).reshape(-1)[0]):.0f}deg "
            f"turns={float(metrics['turns']):.4f} "
            f"depth={1000.0 * float(metrics['depth_m']):.2f} mm "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    print(json.dumps(validate_dataset(output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
