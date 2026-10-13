"""Deterministic matched ablation of only the four G6 linear action channels.

No policy weights, observations, seeds, task targets, IK settings, or rotary
actions are changed.  The intervention multiplies policy outputs at native G6
indices 0, 3, 6, and 9 by 0, 0.5, or 1 before ``env.step``.  Raw traces live in
a temporary directory; the compact episode-level CSV is the retained result.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import config
from .ik_control import ExperimentSpec
from .rotation_sweep import RotationSweepSpec
from . import rotation_residual_sweep as rotation_runner
from . import translation_residual_sweep as translation_runner


DEFAULT_TRAINING_RUN = Path("runs/full_residual/g6")
DEFAULT_OUTPUT = Path(
    "evaluation/api_linear_ablation/linear_scale_ablation.csv"
)
LINEAR_INDICES = np.asarray(config.G6_LINEAR_ACTUATOR_INDICES, dtype=np.int32)


@dataclass(frozen=True)
class AblationCase:
    task_family: str
    object_name: str
    motion: str
    seed: int
    path_name: str | None = None

    @property
    def case_id(self) -> str:
        parts = (self.task_family, self.object_name, self.path_name or "rotation", self.motion)
        return "_".join(parts)


CASES = (
    AblationCase("translation", "cone", "xy", 11, "square"),
    AblationCase("translation", "cone", "xy", 27, "S"),
    AblationCase("translation", "cube", "xyz", 22, "circle"),
    AblationCase("rotation", "cone", "xyz", 34),
    AblationCase("rotation", "cylinder", "yz", 19),
)


class LinearScaledPolicy:
    def __init__(self, policy: Any, scale: float) -> None:
        self.policy = policy
        self.scale = float(scale)

    def predict(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, Any]:
        action, state = self.policy.predict(*args, **kwargs)
        value = np.asarray(action).copy()
        value[..., LINEAR_INDICES] *= self.scale
        return value, state


def _inside_project(path: Path) -> Path:
    value = (
        (config.PROJECT_ROOT / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )
    try:
        value.relative_to(config.PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("ablation inputs and outputs must stay inside IK-residual") from exc
    return value


def _translation_spec(case: AblationCase) -> ExperimentSpec:
    if case.path_name is None:
        raise ValueError("translation ablation requires a path")
    goal = tuple(30.0 if axis in case.motion else 0.0 for axis in "xyz")
    return ExperimentSpec(
        gripper="g6",
        object_name=case.object_name,
        path_name=case.path_name,
        rotation_axes=case.motion,
        rotation_goal_euler_deg=goal,
        seed=case.seed,
    )


def _summarize(
    case: AblationCase, scale: float, summary: dict[str, Any]
) -> dict[str, Any]:
    position = np.asarray(
        summary.get("endpoint_position_error_m_per_axis", (np.nan,) * 3),
        dtype=np.float64,
    )
    angle = np.asarray(
        summary.get("endpoint_angle_error_deg_per_axis", (np.nan,) * 3),
        dtype=np.float64,
    )
    return {
        "task_family": case.task_family,
        "case_id": case.case_id,
        "object": case.object_name,
        "path": case.path_name or "",
        "motion": case.motion,
        "seed": case.seed,
        "linear_action_scale": float(scale),
        "is_success": int(bool(summary["is_success"])),
        "termination_reason": str(summary["termination_reason"]),
        "ended_early": int(bool(summary["ended_early"])),
        "ticks": int(summary["ticks"]),
        "endpoint_position_error_l2_mm": 1000.0 * float(np.linalg.norm(position)),
        "endpoint_euler_error_l2_deg": float(np.linalg.norm(angle)),
        "reached_angle_deg": float(summary.get("reached_angle_deg", np.nan)),
        "maximum_rotation_error_l2_deg": float(
            summary.get("maximum_rotation_error_l2_deg", np.nan)
        ),
        "minimum_height_mm": 1000.0
        * float(
            summary.get(
                "minimum_object_height_m",
                summary.get("_diagnostic_min_height_m", np.nan),
            )
        ),
        "projected_ticks": int(
            summary.get("projected_sweep_ticks", summary.get("ik_projected_ticks", 0))
        ),
    }


def _run_one(
    case: AblationCase,
    scale: float,
    training_run: Path,
    device: str,
) -> dict[str, Any]:
    """Run one intervention in a fresh process/controller lifetime."""

    with tempfile.TemporaryDirectory(prefix="g6_linear_ablation_worker_") as temporary:
        traces = Path(temporary)
        if case.task_family == "translation":
            artifacts = translation_runner.load_training_artifacts(training_run)
            controller = translation_runner.load_controller(artifacts, device=device)
            try:
                controller.model = LinearScaledPolicy(controller.model, scale)
                summary = translation_runner.run_residual_translation_episode(
                    _translation_spec(case),
                    artifacts=artifacts,
                    controller=controller,
                    traces_dir=traces,
                )
                trace = pd.read_csv(
                    traces / f"{_translation_spec(case).episode_id}.csv",
                    usecols=["residual_active", "object_grasp_frame_height_m"],
                )
                trace = trace[trace["residual_active"] == 1]
                summary["_diagnostic_min_height_m"] = float(
                    trace["object_grasp_frame_height_m"].min()
                )
            finally:
                controller.close()
        else:
            artifacts = rotation_runner.load_training_artifacts(training_run)
            controller = rotation_runner.load_controller(artifacts, device=device)
            try:
                controller.model = LinearScaledPolicy(controller.model, scale)
                summary = rotation_runner.run_residual_rotation_sweep_episode(
                    RotationSweepSpec(
                        gripper="g6",
                        object_name=case.object_name,
                        rotation_axes=case.motion,
                        max_angle_deg=90.0,
                        seed=case.seed,
                    ),
                    artifacts=artifacts,
                    controller=controller,
                    traces_dir=traces,
                )
            finally:
                controller.close()
    return _summarize(case, scale, summary)


def run_ablation(
    training_run: Path = DEFAULT_TRAINING_RUN,
    output_csv: Path = DEFAULT_OUTPUT,
    *,
    scales: Sequence[float] = (0.0, 0.5, 1.0),
    device: str = "cpu",
) -> list[dict[str, Any]]:
    run = _inside_project(training_run)
    output = _inside_project(output_csv)
    values = tuple(float(scale) for scale in scales)
    if not values or any(scale < 0.0 or scale > 1.0 for scale in values):
        raise ValueError("linear action scales must lie in [0, 1]")

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="g6_linear_ablation_parent_") as temporary:
        temporary_root = Path(temporary)
        for case in CASES:
            for scale in values:
                print(f"{case.case_id}: linear scale={scale:g}", flush=True)
                result_path = temporary_root / f"{case.case_id}_{scale:g}.json"
                command = [
                    sys.executable,
                    str(config.PROJECT_ROOT / "ablate_api_linear_residual.py"),
                    "--training-run",
                    str(run),
                    "--device",
                    device,
                    "--worker-case",
                    case.case_id,
                    "--worker-scale",
                    str(scale),
                    "--worker-output",
                    str(result_path),
                ]
                completed = subprocess.run(
                    command,
                    cwd=config.PROJECT_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if completed.returncode != 0 or not result_path.is_file():
                    raise RuntimeError(
                        f"ablation worker failed ({completed.returncode}): "
                        f"{case.case_id} scale={scale:g}\n{completed.stdout}"
                    )
                rows.append(json.loads(result_path.read_text(encoding="utf-8")))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(output)
    metadata = {
        "schema": "g3g6-g6-linear-action-scale-ablation-v1",
        "training_run": str(run),
        "model_sha256": translation_runner.load_training_artifacts(run).model_sha256,
        "vec_normalize_sha256": translation_runner.load_training_artifacts(run).vec_normalize_sha256,
        "deterministic_policy": True,
        "device": device,
        "intervention": {
            "indices": LINEAR_INDICES.tolist(),
            "scales": list(values),
            "unchanged": [
                "model weights",
                "VecNormalize statistics",
                "seeds",
                "task targets",
                "nominal feedback IK",
                "all twelve rotary actions",
            ],
        },
        "cases": [case.__dict__ for case in CASES],
        "result_csv": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, default=DEFAULT_TRAINING_RUN)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scales", type=float, nargs="+", default=(0.0, 0.5, 1.0))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--worker-case", choices=tuple(case.case_id for case in CASES))
    parser.add_argument("--worker-scale", type=float)
    parser.add_argument("--worker-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker_case is not None:
        if args.worker_scale is None or args.worker_output is None:
            raise SystemExit("worker mode requires --worker-scale and --worker-output")
        case = next(case for case in CASES if case.case_id == args.worker_case)
        result = _run_one(
            case,
            float(args.worker_scale),
            _inside_project(args.training_run),
            args.device,
        )
        serializable = {
            key: (
                None
                if isinstance(value, float) and not np.isfinite(value)
                else value
            )
            for key, value in result.items()
        }
        args.worker_output.write_text(
            json.dumps(serializable, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return
    rows = run_ablation(
        args.training_run,
        args.output_csv,
        scales=args.scales,
        device=args.device,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
