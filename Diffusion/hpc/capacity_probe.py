#!/usr/bin/env python3
"""Qualify increasing MJWarp world counts in isolated child processes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.monitoring import atomic_write_json
from s_hpc.paths import add_external_path_flag, resolve_package_path


def _last_json_object(text: str) -> dict | None:
    starts = [index for index, char in enumerate(text) if char == "{"]
    for index in reversed(starts):
        try:
            value = json.loads(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _capacity_summary(results: list[dict], requested: list[int]) -> dict:
    passed = [
        record for record in results if record.get("status") == "pass"
    ]
    memory_points = []
    for record in passed:
        preflight = record.get("preflight")
        if not isinstance(preflight, dict):
            continue
        used = preflight.get("cuda_peak_device_used_bytes")
        total = preflight.get("cuda_total_memory_bytes")
        fraction = preflight.get("cuda_peak_device_used_fraction")
        if not isinstance(used, int) or not isinstance(total, int):
            continue
        memory_points.append(
            {
                "worlds": int(record["worlds"]),
                "peak_device_used_bytes": used,
                "total_device_memory_bytes": total,
                "peak_device_used_fraction": fraction,
            }
        )

    recommendations = {}
    for percent in (70, 80, 90):
        limit = percent / 100.0
        eligible = [
            point["worlds"]
            for point in memory_points
            if isinstance(point["peak_device_used_fraction"], (int, float))
            and point["peak_device_used_fraction"] <= limit
        ]
        recommendations[f"largest_tested_at_or_below_{percent}_percent"] = (
            max(eligible) if eligible else 0
        )

    failed = [
        int(record["worlds"])
        for record in results
        if record.get("status") == "fail"
    ]
    return {
        "qualified_worlds": [int(record["worlds"]) for record in passed],
        "maximum_passed_worlds": (
            max(int(record["worlds"]) for record in passed) if passed else 0
        ),
        "first_failed_worlds": min(failed) if failed else None,
        "all_requested_passed": len(passed) == len(requested),
        "memory_points": memory_points,
        "recommendations": recommendations,
        "recommendation_note": (
            "70% leaves the strongest headroom for Diffusion/TQC/replay; "
            "80% is moderate; 90% is an aggressive simulation-only ceiling. "
            "Only tested, successfully qualified world counts are returned."
        ),
    }


def probe(config: Path, run_dir: Path, worlds: list[int]) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for count in worlds:
        started = time.monotonic()
        command = (
            sys.executable,
            str(ROOT / "preflight.py"),
            "--mode",
            "gpu",
            "--config",
            str(config),
            "--worlds",
            str(count),
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        parsed = _last_json_object(completed.stdout)
        record = {
            "worlds": count,
            "status": "pass" if completed.returncode == 0 else "fail",
            "returncode": completed.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "preflight": parsed,
        }
        if completed.returncode != 0:
            record["stdout_tail"] = completed.stdout[-8000:]
            record["stderr_tail"] = completed.stderr[-8000:]
        results.append(record)
        summary = _capacity_summary(results, worlds)
        snapshot = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(config),
            "requested_worlds": worlds,
            **summary,
            "results": results,
        }
        atomic_write_json(run_dir / "capacity.json", snapshot)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        if completed.returncode != 0:
            break
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "cx3_l40s.json",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs" / "cx3_capacity",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        nargs="+",
        default=(128, 256, 512, 1024),
    )
    parser.add_argument("--require-worlds", type=int, default=1024)
    add_external_path_flag(parser)
    args = parser.parse_args()
    if any(value <= 0 for value in args.worlds):
        parser.error("all world counts must be positive")
    if list(args.worlds) != sorted(set(args.worlds)):
        parser.error("--worlds must be unique and increasing")
    try:
        config = resolve_package_path(
            args.config,
            allow_external=args.allow_external_paths,
            label="config",
        )
        run_dir = resolve_package_path(
            args.run_dir,
            allow_external=args.allow_external_paths,
            label="capacity output",
        )
    except ValueError as exc:
        parser.error(str(exc))
    result = probe(config, run_dir, list(args.worlds))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["maximum_passed_worlds"] < args.require_worlds:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
