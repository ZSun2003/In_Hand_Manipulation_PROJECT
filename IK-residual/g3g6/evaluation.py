"""Local deterministic evaluation for a trained TQC checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from . import config
from .env import G3G6ManipulationEnv


def _inside_project(path: Path) -> Path:
    value = (
        (config.PROJECT_ROOT / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )
    try:
        value.relative_to(config.PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("models, normalization data, and reports must be inside IK-residual") from exc
    return value


def evaluate(
    *,
    gripper: str,
    phase: int,
    model_path: Path,
    vecnormalize_path: Path,
    episodes: int,
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    gripper = config.GripperKind(gripper).value
    selected_phase = config.Phase(phase)
    raw = DummyVecEnv(
        [lambda: G3G6ManipulationEnv(gripper=gripper, phase=selected_phase)]
    )
    raw.seed(seed)
    normalized = VecNormalize.load(str(_inside_project(vecnormalize_path)), raw)
    normalized.training = False
    normalized.norm_reward = False
    model = TQC.load(str(_inside_project(model_path)), env=normalized)
    records: list[dict[str, Any]] = []
    try:
        observation = normalized.reset()
        while len(records) < episodes:
            action, _state = model.predict(observation, deterministic=True)
            observation, _reward, done, infos = normalized.step(action)
            if bool(done[0]):
                info = infos[0]
                record = {
                    "episode": len(records),
                    "object": info["object"],
                    "path": info["path"],
                    "rotation_axes": info["rotation_axes"],
                    "rotation_goal_euler_deg": np.asarray(
                        info["rotation_goal_euler_deg"]
                    ).tolist(),
                    "is_success": bool(info.get("is_success", False)),
                    "termination_reason": info["termination_reason"],
                    "position_error_m_per_axis": np.asarray(
                        info["position_error_m_per_axis"]
                    ).tolist(),
                    "angle_error_deg_per_axis": np.asarray(
                        info["angle_error_deg_per_axis"]
                    ).tolist(),
                    "tip_contact_count": int(info["tip_contact_count"]),
                    "max_route_abs_position_error_m": np.asarray(
                        info["max_route_abs_position_error_m"]
                    ).tolist(),
                    "max_route_abs_angle_error_deg": np.asarray(
                        info["max_route_abs_angle_error_deg"]
                    ).tolist(),
                    "ik_exact": bool(info["ik_exact"]),
                    "ik_projection_fraction": float(
                        info["ik_projection_fraction"]
                    ),
                    "endpoint": bool(info["endpoint"]),
                }
                records.append(record)
    finally:
        normalized.close()
    object_counts = {
        name: sum(record["object"] == name for record in records)
        for name in config.OBJECT_NAMES
    }
    report = {
        "schema": "g3g6-local-evaluation-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "gripper": gripper,
        "phase": int(selected_phase),
        "episodes": episodes,
        "successes": sum(record["is_success"] for record in records),
        "success_rate": float(np.mean([record["is_success"] for record in records])),
        "object_counts": object_counts,
        "position_tolerance_m_per_axis": config.POSITION_TOLERANCE_M_PER_AXIS,
        "angle_tolerance_deg_per_axis": config.ANGLE_TOLERANCE_DEG_PER_AXIS,
        "records": records,
    }
    destination = _inside_project(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one local G3/G6 TQC model.")
    parser.add_argument("--gripper", choices=("g3", "g6"), required=True)
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument(
        "--output", type=Path, default=config.PROJECT_ROOT / "evaluation" / "report.json"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = evaluate(
        gripper=args.gripper,
        phase=args.phase,
        model_path=args.model,
        vecnormalize_path=args.vecnormalize,
        episodes=args.episodes,
        seed=args.seed,
        output_path=args.output,
    )
    print(
        f"{report['successes']}/{report['episodes']} successes; "
        f"report={_inside_project(args.output)}"
    )


if __name__ == "__main__":
    main()
