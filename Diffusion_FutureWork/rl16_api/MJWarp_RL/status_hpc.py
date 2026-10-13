"""Show live Diffusion + MJWarp training metrics without loading CUDA.

Examples::

    python -m MJWarp_RL.status_hpc MJWarp_RL/artifacts/my_run
    python -m MJWarp_RL.status_hpc --run-dir ... --watch --interval 10
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

try:
    from .live_metrics import build_metrics, read_json_object, read_jsonl
except ImportError:  # Direct ``python MJWarp_RL/status_hpc.py`` execution.
    from live_metrics import build_metrics, read_json_object, read_jsonl


def query_gpu_memory(timeout_s: float = 3.0) -> dict[str, Any]:
    """Return lightweight ``nvidia-smi`` telemetry, or an unavailable reason."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "devices": [],
        }
    if result.returncode != 0:
        return {
            "available": False,
            "reason": result.stderr.strip()
            or result.stdout.strip()
            or f"nvidia-smi exited {result.returncode}",
            "devices": [],
        }
    devices: list[dict[str, Any]] = []
    try:
        for row in csv.reader(result.stdout.splitlines()):
            if not row:
                continue
            if len(row) != 5:
                raise ValueError(f"expected 5 columns, got {len(row)}")
            index, name, used, total, utilization = (
                item.strip() for item in row
            )
            devices.append(
                {
                    "index": index,
                    "name": name,
                    "memory_used_mib": float(used),
                    "memory_total_mib": float(total),
                    "utilization_percent": float(utilization),
                }
            )
    except (TypeError, ValueError) as exc:
        return {
            "available": False,
            "reason": f"cannot parse nvidia-smi output: {exc}",
            "devices": [],
        }
    if not devices:
        return {
            "available": False,
            "reason": "nvidia-smi returned no devices",
            "devices": [],
        }
    return {
        "available": True,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": devices,
    }


def build_report(
    output_dir: Path | str,
    *,
    window: int = 20,
    total_steps: int | None = None,
    include_gpu: bool = True,
) -> dict[str, Any]:
    """Read one non-blocking snapshot of a training output directory."""

    output = Path(output_dir).expanduser().resolve()
    training_path = (
        output if output.name == "training.jsonl" else output / "training.jsonl"
    )
    if training_path.name == output.name:
        output = training_path.parent
    status_path = output / "run_status.json"
    jsonl = read_jsonl(training_path)
    run_status, status_error = read_json_object(status_path)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output),
        "files": {
            "training_jsonl": str(training_path),
            "training_jsonl_exists": training_path.is_file(),
            "run_status_json": str(status_path),
            "run_status_json_exists": status_path.is_file(),
        },
        "run_status": run_status,
        "metrics": build_metrics(
            jsonl.records,
            run_status=run_status,
            window=window,
            total_steps=total_steps,
        ),
        "data_quality": {
            "invalid_jsonl_lines": jsonl.invalid_lines,
            "oversized_jsonl_lines": jsonl.oversized_lines,
            "ignored_partial_jsonl_line": jsonl.ignored_partial_line,
            "jsonl_error": jsonl.error,
            "run_status_error": status_error,
        },
        "gpu": (
            query_gpu_memory()
            if include_gpu
            else {"available": False, "reason": "disabled", "devices": []}
        ),
    }


def _number(value: object, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return "N/A"
    if number != number or number in (float("inf"), float("-inf")):
        return "N/A"
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def _percent(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError, OverflowError):
        return "N/A"


def _millimetres(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return _number(1000.0 * float(value), 2)
    except (TypeError, ValueError, OverflowError):
        return "N/A"


def _duration(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        seconds = max(0, int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return "N/A"
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    prefix = f"{days}d " if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def _rate(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "N/A"
    return f"{number:,.0f} step/s" if number >= 1.0 else "N/A"


def _run_state(status: Mapping[str, Any]) -> str:
    for key in ("state", "status", "run_state"):
        value = status.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _render_groups(groups: Mapping[str, Any]) -> str:
    if not groups:
        return "N/A"
    rendered = []
    for name, value in sorted(groups.items()):
        if isinstance(value, Mapping):
            recent = value.get("recent")
            cumulative = value.get("cumulative", value.get("success_rate"))
        else:
            recent = None
            cumulative = None
        rendered.append(
            f"{name}={_percent(recent)}/{_percent(cumulative)}"
        )
    return "  ".join(rendered)


def render_report(report: Mapping[str, Any]) -> str:
    """Render a compact terminal dashboard in Chinese."""

    metrics = report.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    progress = metrics.get("progress", {})
    success = metrics.get("success", {})
    plane = metrics.get("grasp_plane_compliance", {})
    position = metrics.get("position_error_m", {})
    euler = metrics.get("euler_l2_error_deg", {})
    desired_displacement = metrics.get(
        "translation_desired_final_displacement_m", {}
    )
    actual_displacement = metrics.get(
        "translation_actual_final_displacement_m", {}
    )
    displacement_error = metrics.get(
        "translation_final_displacement_error_m", {}
    )
    translation_euler = metrics.get(
        "translation_final_euler_error_deg", {}
    )
    path_rmse = metrics.get("translation_path_tracking_rmse_m", {})
    path_max = metrics.get(
        "translation_path_tracking_max_error_m", {}
    )
    loss = metrics.get("diffusion_loss", {})
    lr = metrics.get("diffusion_lr", {})
    training = metrics.get("training", {})
    status = report.get("run_status", {})
    if not isinstance(status, Mapping):
        status = {}
    files = report.get("files", {})
    quality = report.get("data_quality", {})
    gpu = report.get("gpu", {})

    phase = progress.get("phase") if isinstance(progress, Mapping) else None
    completed = (
        progress.get("completed_steps") if isinstance(progress, Mapping) else None
    )
    target = (
        progress.get("target_steps") if isinstance(progress, Mapping) else None
    )
    progress_text = _number(completed, 0)
    if target is not None:
        progress_text += f" / {_number(target, 0)}"
    fraction = progress.get("fraction") if isinstance(progress, Mapping) else None
    if fraction is not None:
        progress_text += f" ({_percent(fraction)})"
    weighting = (
        success.get("weighting", "log_record")
        if isinstance(success, Mapping)
        else "log_record"
    )
    cumulative_label = (
        "累计(按episode)"
        if weighting == "episode"
        else "累计(日志等权)"
    )
    lines = [
        "GRAB Diffusion + MJWarp 实时训练状态",
        f"时间: {report.get('generated_utc', 'N/A')}",
        f"目录: {report.get('output_dir', 'N/A')}",
        (
            f"运行: state={_run_state(status)}  phase={phase or 'N/A'}  "
            f"记录={_number(metrics.get('record_count'), 0)}"
        ),
        f"进度: {progress_text}  elapsed={_duration(progress.get('elapsed_s'))}",
        (
            f"速度: recent={_rate(progress.get('recent_steps_per_s'))}  "
            f"cumulative={_rate(progress.get('cumulative_steps_per_s'))}  "
            f"ETA={_duration(progress.get('eta_s'))}"
        ),
        (
            f"候选endpoint成功率: recent={_percent(success.get('recent'))}  "
            f"{cumulative_label}={_percent(success.get('cumulative'))}  "
            f"latest={_percent(success.get('latest'))}"
        ),
        (
            "每组至少一条通过认证的IK轨迹: "
            f"recent={_percent(training.get('valid_group_rate'))}  "
            "cumulative="
            f"{_percent(training.get('cumulative_valid_group_rate'))}"
        ),
        "按物体(recent/cumulative): "
        + _render_groups(metrics.get("success_by_object", {})),
        "按任务(recent/cumulative): "
        + _render_groups(metrics.get("success_by_task", {})),
        "旋转目标角°(recent/当前phase累计): "
        + _render_groups(
            metrics.get("success_by_rotation_target_angle_deg", {})
        ),
        (
            "实际hard-tip平面[40%,60%]: "
            f"recent={_percent(plane.get('recent'))}  "
            f"cumulative={_percent(plane.get('cumulative'))}  "
            f"latest={_percent(plane.get('latest'))}"
        ),
        (
            "IK指令平面硬投影合规: "
            f"{_percent(training.get('commanded_grasp_plane_compliance_rate'))}"
        ),
        (
            "位姿误差: "
            f"position recent={_number(position.get('recent_mean'), 5)} m "
            f"latest={_number(position.get('latest'), 5)} m  "
            f"euler recent={_number(euler.get('recent_mean'), 2)}° "
            f"latest={_number(euler.get('latest'), 2)}°"
        ),
        (
            "Translation端点: "
            f"目标位移={_millimetres(desired_displacement.get('latest'))} mm  "
            f"实际位移={_millimetres(actual_displacement.get('latest'))} mm  "
            f"位移误差={_millimetres(displacement_error.get('latest'))} mm  "
            f"Euler={_number(translation_euler.get('latest'), 2)}°"
        ),
        (
            "Translation路径跟踪: "
            f"RMSE recent={_millimetres(path_rmse.get('recent_mean'))} mm "
            f"latest={_millimetres(path_rmse.get('latest'))} mm  "
            f"max recent={_millimetres(path_max.get('recent_mean'))} mm "
            f"latest={_millimetres(path_max.get('latest'))} mm"
        ),
        (
            "Diffusion: "
            f"loss recent={_number(loss.get('recent_mean'), 6)} "
            f"latest={_number(loss.get('latest'), 6)}  "
            f"LR={_number(lr.get('latest'), 8)}"
        ),
        (
            "训练细节: "
            f"stage={training.get('stage') or 'N/A'}  "
            "certified_trajectories="
            f"{_number(training.get('certified_trajectory_count'), 0)}  "
            f"reward={_number(training.get('reward_mean'), 4)}"
        ),
        (
            "Diffusion进度: "
            f"epoch={_number(training.get('epoch'), 0)}/"
            f"{_number(training.get('diffusion_epochs'), 0)}  "
            f"batch={_number(training.get('batch_in_epoch'), 0)}/"
            f"{_number(training.get('batches_per_epoch'), 0)}  "
            f"updates={_number(training.get('diffusion_updates'), 0)}  "
            f"grad={_number(training.get('diffusion_gradient_norm'), 5)}  "
            f"ETA={_duration(training.get('diffusion_eta_s'))}"
        ),
        (
            "进程内显存: "
            f"allocated={_number(training.get('gpu_memory_allocated_gb'), 2)} GiB  "
            f"reserved={_number(training.get('gpu_memory_reserved_gb'), 2)} GiB"
        ),
    ]
    if isinstance(gpu, Mapping) and gpu.get("available"):
        devices = gpu.get("devices", [])
        for device in devices if isinstance(devices, Sequence) else []:
            if not isinstance(device, Mapping):
                continue
            used = device.get("memory_used_mib")
            total = device.get("memory_total_mib")
            memory_fraction = None
            try:
                memory_fraction = float(used) / float(total)  # type: ignore[arg-type]
            except (TypeError, ValueError, ZeroDivisionError):
                pass
            lines.append(
                f"GPU {device.get('index', '?')}: {device.get('name', 'N/A')}  "
                f"VRAM={_number(used, 0)}/{_number(total, 0)} MiB "
                f"({_percent(memory_fraction)})  "
                f"util={_number(device.get('utilization_percent'), 0)}%"
            )
    else:
        reason = gpu.get("reason", "unavailable") if isinstance(gpu, Mapping) else "unavailable"
        lines.append(f"GPU: N/A ({reason})")

    warnings = []
    if isinstance(files, Mapping) and not files.get("training_jsonl_exists"):
        warnings.append("training.jsonl 尚未创建")
    if isinstance(quality, Mapping):
        if quality.get("invalid_jsonl_lines"):
            warnings.append(
                f"跳过 {quality.get('invalid_jsonl_lines')} 条损坏 JSONL"
            )
        if quality.get("oversized_jsonl_lines"):
            warnings.append(
                f"跳过 {quality.get('oversized_jsonl_lines')} 条超大 JSONL"
            )
        if quality.get("ignored_partial_jsonl_line"):
            warnings.append("末行正在写入，本次快照已忽略")
        if quality.get("jsonl_error"):
            warnings.append(f"读取 JSONL 失败: {quality.get('jsonl_error')}")
        if quality.get("run_status_error"):
            warnings.append(
                f"读取 run_status.json 失败: {quality.get('run_status_error')}"
            )
    if warnings:
        lines.append("提示: " + "；".join(warnings))
    return "\n".join(lines)


def _resolve_output(
    positional: Path | None, option: Path | None
) -> Path:
    if positional is not None and option is not None:
        raise ValueError("output path may be given either positionally or by flag")
    return option or positional or Path(".")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="training output directory (or training.jsonl)",
    )
    parser.add_argument(
        "--output",
        "--run-dir",
        dest="output_option",
        type=Path,
        help="training output directory",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="refresh continuously until Ctrl-C",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="watch refresh interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="number of recent JSONL records used for rolling metrics",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="override the target step count used for progress and ETA",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the machine-readable report",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="skip the optional nvidia-smi query",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.interval <= 0.0:
        raise SystemExit("--interval must be positive")
    if args.window <= 0:
        raise SystemExit("--window must be positive")
    if args.total_steps is not None and args.total_steps <= 0:
        raise SystemExit("--total-steps must be positive")
    try:
        output = _resolve_output(args.output, args.output_option)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    first = True
    try:
        while True:
            report = build_report(
                output,
                window=args.window,
                total_steps=args.total_steps,
                include_gpu=not args.no_gpu,
            )
            rendered = (
                json.dumps(
                    report,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                if args.json
                else render_report(report)
            )
            if args.watch and sys.stdout.isatty():
                sys.stdout.write("\033[2J\033[H")
            elif args.watch and not first:
                print("\n" + "=" * 72)
            print(rendered, flush=True)
            first = False
            if not args.watch:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
