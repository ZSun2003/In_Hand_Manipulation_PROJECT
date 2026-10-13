#!/usr/bin/env python3
"""Select and verify a formal world count for the GPU actually allocated."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from s_hpc.contracts import SCHEMA_HASH, TASK_GEOMETRY_HASH
from s_hpc.monitoring import _source_fingerprints, atomic_write_json
from s_hpc.paths import add_external_path_flag, resolve_package_path


SELECTION_FORMAT = "screw-adaptive-capacity-v4"
SCALED_TQC_FIELDS = (
    "total_transitions",
    "learning_starts",
    "replay_capacity",
    "batch_size",
    "checkpoint_every_transitions",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _last_json_object(text: str) -> dict[str, Any] | None:
    starts = [index for index, character in enumerate(text) if character == "{"]
    for index in reversed(starts):
        try:
            value = json.loads(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _gpu_slug(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return value or "gpu"


def derive_config(
    base_payload: dict[str, Any],
    *,
    worlds: int,
    gpu_name: str,
    scale_world_budgets: bool = False,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_payload))
    physics = payload.get("physics")
    if not isinstance(physics, dict):
        raise ValueError("base config has no physics object")
    base_worlds = int(physics.get("worlds", 0))
    if base_worlds <= 0 or int(worlds) <= 0:
        raise ValueError("base and derived world counts must be positive")
    physics["worlds"] = int(worlds)
    payload["name"] = f"cx3_adaptive_{_gpu_slug(gpu_name)}_w{worlds}"
    if scale_world_budgets:
        tqc = payload.get("tqc")
        if not isinstance(tqc, dict):
            raise ValueError("base config has no tqc object")
        for field in SCALED_TQC_FIELDS:
            original = tqc.get(field)
            if not isinstance(original, int) or original <= 0:
                raise ValueError(
                    f"base tqc.{field} must be a positive integer"
                )
            numerator = original * int(worlds)
            if numerator % base_worlds:
                raise ValueError(
                    f"tqc.{field} cannot be scaled exactly from "
                    f"{base_worlds} to {worlds} worlds"
                )
            tqc[field] = numerator // base_worlds
    return payload


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _tail_text(value: str | bytes | None, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def _run_child(
    *,
    config: Path,
    worlds: int,
    memory_ceiling: float,
    physics_steps: int,
    timeout_seconds: int,
    pipeline_mode: str,
) -> dict[str, Any]:
    command = (
        sys.executable,
        str(ROOT / "training_stack_preflight.py"),
        "--config",
        str(config),
        "--worlds",
        str(worlds),
        "--max-memory-fraction",
        str(memory_ceiling),
        "--physics-steps",
        str(physics_steps),
        "--pipeline-mode",
        pipeline_mode,
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "worlds": worlds,
            "status": "fail",
            "failure_kind": "timeout",
            "returncode": None,
            "elapsed_seconds": time.monotonic() - started,
            "stdout_tail": _tail_text(exc.stdout, 12000),
            "stderr_tail": _tail_text(exc.stderr, 12000),
            "preflight": None,
        }
    parsed = _last_json_object(completed.stdout)
    child_status = None if parsed is None else parsed.get("status")
    if completed.returncode == 0 and child_status == "pass":
        status = "safe_pass"
    elif child_status == "over_memory_ceiling":
        status = "raw_pass_over_ceiling"
    else:
        status = "fail"
    record: dict[str, Any] = {
        "worlds": worlds,
        "status": status,
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "preflight": parsed,
    }
    if status == "fail":
        record["stdout_tail"] = completed.stdout[-12000:]
        record["stderr_tail"] = completed.stderr[-12000:]
    return record


def summarize(records: list[dict[str, Any]], requested: list[int]) -> dict[str, Any]:
    safe = [
        int(record["worlds"])
        for record in records
        if record.get("status") == "safe_pass"
    ]
    raw = [
        int(record["worlds"])
        for record in records
        if record.get("status") in {"safe_pass", "raw_pass_over_ceiling"}
    ]
    failed = [
        int(record["worlds"])
        for record in records
        if record.get("status") == "fail"
    ]
    unsafe = [
        int(record["worlds"])
        for record in records
        if record.get("status") != "safe_pass"
    ]
    return {
        "requested_worlds": requested,
        "attempted_worlds": [int(record["worlds"]) for record in records],
        "safe_qualified_worlds": safe,
        "raw_completed_worlds": raw,
        "selected_training_worlds": max(safe) if safe else 0,
        "raw_maximum_completed_worlds": max(raw) if raw else 0,
        "first_failed_worlds": min(failed) if failed else None,
        "first_unsafe_worlds": min(unsafe) if unsafe else None,
        "all_requested_completed": len(raw) == len(requested),
    }


def probe(
    *,
    base_config: Path,
    run_dir: Path,
    worlds: list[int],
    memory_ceiling: float,
    physics_steps: int,
    timeout_seconds: int,
    require_worlds: int,
    scale_world_budgets: bool = False,
    profile_id: str = "generic",
    pipeline_mode: str = "diffusion-tqc",
    stop_after_first_unsafe: bool = False,
) -> dict[str, Any]:
    if pipeline_mode not in {"diffusion-only", "diffusion-tqc"}:
        raise ValueError(
            "pipeline_mode must be diffusion-only or diffusion-tqc"
        )
    if pipeline_mode == "diffusion-only" and scale_world_budgets:
        raise ValueError(
            "Diffusion-only capacity must not scale unused TQC budgets"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_path = run_dir / "selection.json"
    selected_config_path = run_dir / "selected_config.json"
    capacity_path = run_dir / "capacity.json"
    qualified_config_dir = run_dir / "qualified_configs"
    if (
        selection_path.exists()
        or selected_config_path.exists()
        or capacity_path.exists()
        or qualified_config_dir.exists()
    ):
        raise FileExistsError(
            f"adaptive capacity artifacts already exist in {run_dir}; "
            "reuse a complete selection or move the incomplete directory "
            "aside before probing again"
        )
    qualified_config_dir.mkdir(parents=True)
    base_payload = json.loads(base_config.read_text(encoding="utf-8"))
    if not isinstance(base_payload, dict):
        raise ValueError("base config must contain a JSON object")
    source_fingerprints = _source_fingerprints()
    records: list[dict[str, Any]] = []
    snapshot: dict[str, Any] = {}
    for count in worlds:
        candidate_path = qualified_config_dir / f"worlds_{count}.json"
        candidate = derive_config(
            base_payload,
            worlds=count,
            gpu_name="capacity-candidate",
            scale_world_budgets=scale_world_budgets,
        )
        _write_json_file(candidate_path, candidate)
        record = _run_child(
            config=candidate_path,
            worlds=count,
            memory_ceiling=memory_ceiling,
            physics_steps=physics_steps,
            timeout_seconds=timeout_seconds,
            pipeline_mode=pipeline_mode,
        )
        record["candidate_config"] = candidate_path.relative_to(
            run_dir
        ).as_posix()
        record["candidate_config_sha256"] = _sha256(candidate_path)
        records.append(record)
        summary = summarize(records, worlds)
        snapshot = {
            "format": SELECTION_FORMAT,
            "profile_id": profile_id,
            "created_utc": _utc_now(),
            "base_config": str(base_config),
            "base_config_sha256": _sha256(base_config),
            "pipeline_mode": pipeline_mode,
            "budget_policy": (
                "diffusion_only_worlds_v1"
                if pipeline_mode == "diffusion-only"
                else (
                    "scale_tqc_with_worlds_v1"
                    if scale_world_budgets
                    else "worlds_only_v1"
                )
            ),
            "memory_ceiling_fraction": memory_ceiling,
            "physics_steps_per_phase": physics_steps,
            **summary,
            "records": records,
        }
        _write_json_file(capacity_path, snapshot)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        if stop_after_first_unsafe and record.get("status") != "safe_pass":
            break

    if (
        stop_after_first_unsafe
        and snapshot.get("first_unsafe_worlds") is None
    ):
        raise RuntimeError(
            "capacity scan reached its final safety candidate without an "
            "unsafe result; increase the scan ceiling instead of treating "
            "the capped result as a measured maximum"
        )

    selected = int(snapshot.get("selected_training_worlds", 0))
    if selected < require_worlds:
        raise RuntimeError(
            f"no safe capacity at the required {require_worlds} worlds; "
            f"selected={selected}"
        )
    selected_record = next(
        record for record in records if int(record["worlds"]) == selected
    )
    preflight = selected_record.get("preflight")
    if not isinstance(preflight, dict) or not isinstance(
        preflight.get("gpu"), dict
    ):
        raise RuntimeError("selected capacity record has no GPU fingerprint")
    gpu = dict(preflight["gpu"])
    safe_worlds = [
        int(record["worlds"])
        for record in records
        if record.get("status") == "safe_pass"
    ]
    qualified_configs: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "safe_pass":
            continue
        count = int(record["worlds"])
        qualified_configs[str(count)] = {
            "worlds": count,
            "path": str(record["candidate_config"]),
            "sha256": str(record["candidate_config_sha256"]),
        }
    selected_record_config = qualified_config_dir / f"worlds_{selected}.json"
    _write_json_file(
        selected_config_path,
        json.loads(selected_record_config.read_text(encoding="utf-8")),
    )
    try:
        base_config_reference = base_config.relative_to(ROOT).as_posix()
    except ValueError:
        # External paths are only accepted when the caller explicitly opts in
        # through --allow-external-paths.  Preserve that path so a later gate
        # can revalidate the exact source file under the same opt-in.
        base_config_reference = str(base_config)
    selection = {
        "format": SELECTION_FORMAT,
        "profile_id": profile_id,
        "pipeline_mode": pipeline_mode,
        "created_utc": _utc_now(),
        "selected_training_worlds": selected,
        "maximum_qualified_candidate": selected,
        "safe_qualified_worlds": safe_worlds,
        "candidate_cap_reached": selected == max(worlds),
        "attempted_worlds": list(snapshot["attempted_worlds"]),
        "first_unsafe_worlds": snapshot["first_unsafe_worlds"],
        "stopped_after_first_unsafe": stop_after_first_unsafe,
        "raw_maximum_completed_worlds": int(
            snapshot["raw_maximum_completed_worlds"]
        ),
        "budget_policy": snapshot["budget_policy"],
        "memory_ceiling_fraction": memory_ceiling,
        "physics_steps_per_phase": physics_steps,
        "requested_worlds": worlds,
        "gpu_fingerprint": gpu,
        "versions": preflight.get("versions"),
        "schema_hash": SCHEMA_HASH,
        "task_geometry_hash": TASK_GEOMETRY_HASH,
        "training_contract_hash": source_fingerprints[
            "training_contract_hash"
        ],
        "training_contract_files": source_fingerprints[
            "training_contract_files"
        ],
        "source_tree_sha256": source_fingerprints["source_tree_sha256"],
        "source_file_count": source_fingerprints["source_file_count"],
        "base_config": base_config_reference,
        "base_config_sha256": _sha256(base_config),
        "capacity_result": "capacity.json",
        "capacity_result_sha256": _sha256(capacity_path),
        "qualified_configs": qualified_configs,
        "selected_config": selected_config_path.name,
        "selected_config_sha256": _sha256(selected_config_path),
        "probe_script_sha256": _sha256(
            ROOT / "training_stack_preflight.py"
        ),
        "selector_script_sha256": _sha256(ROOT / "adaptive_capacity.py"),
    }
    _write_json_file(selection_path, selection)
    result = dict(snapshot)
    result["selection"] = selection
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def load_and_validate_selection(
    run_dir: Path,
    *,
    allow_external: bool = False,
    expected_profile_id: str | None = None,
    expected_pipeline_mode: str | None = None,
    expected_candidates: list[int] | None = None,
    expected_budget_policy: str | None = None,
    minimum_sweep_physics_steps: int | None = None,
    maximum_memory_ceiling: float | None = None,
) -> tuple[dict[str, Any], Path]:
    selection_path = run_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"missing adaptive selection: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("format") != SELECTION_FORMAT:
        raise ValueError("unsupported adaptive selection format")
    profile_id = selection.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("adaptive selection has no profile_id")
    if (
        expected_profile_id is not None
        and profile_id != expected_profile_id
    ):
        raise ValueError(
            f"adaptive selection profile mismatch: expected "
            f"{expected_profile_id!r}, got {profile_id!r}"
        )
    pipeline_mode = selection.get("pipeline_mode")
    if pipeline_mode not in {"diffusion-only", "diffusion-tqc"}:
        raise ValueError("adaptive selection has no valid pipeline_mode")
    if (
        expected_pipeline_mode is not None
        and pipeline_mode != expected_pipeline_mode
    ):
        raise ValueError(
            "adaptive selection pipeline mode mismatch: expected "
            f"{expected_pipeline_mode!r}, got {pipeline_mode!r}"
        )
    if selection.get("schema_hash") != SCHEMA_HASH:
        raise ValueError("adaptive selection schema hash mismatch")
    if selection.get("task_geometry_hash") != TASK_GEOMETRY_HASH:
        raise ValueError("adaptive selection task geometry hash mismatch")
    source_fingerprints = _source_fingerprints()
    if selection.get("training_contract_hash") != source_fingerprints.get(
        "training_contract_hash"
    ):
        raise ValueError(
            "adaptive selection training contract changed; rerun the "
            "capacity sweep"
        )
    if selection.get("training_contract_files") != source_fingerprints.get(
        "training_contract_files"
    ):
        raise ValueError(
            "adaptive selection training contract file set changed; rerun "
            "the capacity sweep"
        )
    if (
        selection.get("source_tree_sha256")
        != source_fingerprints.get("source_tree_sha256")
        or selection.get("source_file_count")
        != source_fingerprints.get("source_file_count")
    ):
        raise ValueError(
            "adaptive selection source tree changed; rerun the capacity "
            "sweep"
        )
    base_config_name = selection.get("base_config")
    if not isinstance(base_config_name, str):
        raise ValueError("selection has no base_config path")
    base_reference = Path(base_config_name)
    if base_reference.is_absolute():
        if not allow_external:
            raise ValueError(
                "selection uses an external base_config; rerun the gate "
                "with --allow-external-paths"
            )
        base_config_path = base_reference.resolve()
    else:
        if ".." in base_reference.parts:
            raise ValueError("selection has an unsafe base_config path")
        base_config_path = (ROOT / base_reference).resolve()
        try:
            base_config_path.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(
                "selection base_config escaped the package"
            ) from exc
    if (
        not base_config_path.is_file()
        or _sha256(base_config_path) != selection.get("base_config_sha256")
    ):
        raise ValueError(
            "adaptive selection base config is missing or has changed"
        )
    base_payload = json.loads(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(base_payload, dict):
        raise ValueError("adaptive selection base config is not an object")
    budget_policy = selection.get("budget_policy")
    if budget_policy not in {
        "diffusion_only_worlds_v1",
        "scale_tqc_with_worlds_v1",
        "worlds_only_v1",
    }:
        raise ValueError("adaptive selection has an unsupported budget policy")
    if (
        expected_budget_policy is not None
        and budget_policy != expected_budget_policy
    ):
        raise ValueError(
            f"adaptive selection budget policy mismatch: expected "
            f"{expected_budget_policy!r}, got {budget_policy!r}"
        )
    requested_worlds = selection.get("requested_worlds")
    if (
        not isinstance(requested_worlds, list)
        or any(
            not isinstance(value, int) or value <= 0
            for value in requested_worlds
        )
        or requested_worlds != sorted(set(requested_worlds))
    ):
        raise ValueError("adaptive selection has invalid requested_worlds")
    if (
        expected_candidates is not None
        and requested_worlds != expected_candidates
    ):
        raise ValueError(
            f"adaptive selection candidates mismatch: expected "
            f"{expected_candidates}, got {requested_worlds}"
        )
    sweep_physics_steps = selection.get("physics_steps_per_phase")
    if not isinstance(sweep_physics_steps, int) or sweep_physics_steps <= 0:
        raise ValueError("adaptive selection has invalid physics step coverage")
    if (
        minimum_sweep_physics_steps is not None
        and sweep_physics_steps < minimum_sweep_physics_steps
    ):
        raise ValueError(
            f"adaptive selection used only {sweep_physics_steps} physics "
            f"steps; at least {minimum_sweep_physics_steps} are required"
        )
    memory_ceiling = selection.get("memory_ceiling_fraction")
    if (
        not isinstance(memory_ceiling, (int, float))
        or not 0.50 <= float(memory_ceiling) < 1.0
    ):
        raise ValueError("adaptive selection has an invalid memory ceiling")
    if (
        maximum_memory_ceiling is not None
        and float(memory_ceiling) > float(maximum_memory_ceiling) + 1.0e-12
    ):
        raise ValueError(
            f"adaptive selection memory ceiling {memory_ceiling} exceeds "
            f"the allowed {maximum_memory_ceiling}"
        )
    if _sha256(
        ROOT / "training_stack_preflight.py"
    ) != selection.get("probe_script_sha256"):
        raise ValueError(
            "formal-stack probe changed after adaptive selection"
        )
    if _sha256(ROOT / "adaptive_capacity.py") != selection.get(
        "selector_script_sha256"
    ):
        raise ValueError(
            "adaptive selector changed after adaptive selection"
        )
    config_name = selection.get("selected_config")
    if not isinstance(config_name, str) or Path(config_name).name != config_name:
        raise ValueError("selection has an invalid selected_config name")
    config_path = run_dir / config_name
    if not config_path.is_file():
        raise FileNotFoundError(f"selected config is missing: {config_path}")
    if _sha256(config_path) != selection.get("selected_config_sha256"):
        raise ValueError("selected config SHA-256 mismatch")
    capacity_name = selection.get("capacity_result")
    if (
        not isinstance(capacity_name, str)
        or Path(capacity_name).name != capacity_name
    ):
        raise ValueError("selection has an invalid capacity_result name")
    capacity_path = run_dir / capacity_name
    if (
        not capacity_path.is_file()
        or _sha256(capacity_path) != selection.get("capacity_result_sha256")
    ):
        raise ValueError("capacity result SHA-256 mismatch")
    safe_worlds = selection.get("safe_qualified_worlds")
    if (
        not isinstance(safe_worlds, list)
        or not safe_worlds
        or any(
            not isinstance(value, int) or value <= 0
            for value in safe_worlds
        )
        or safe_worlds != sorted(set(safe_worlds))
    ):
        raise ValueError("selection has invalid safe_qualified_worlds")
    qualified_configs = selection.get("qualified_configs")
    if not isinstance(qualified_configs, dict) or set(qualified_configs) != {
        str(value) for value in safe_worlds
    }:
        raise ValueError("selection qualified config set is inconsistent")
    validated_configs: dict[int, Path] = {}
    for worlds in safe_worlds:
        descriptor = qualified_configs[str(worlds)]
        if not isinstance(descriptor, dict):
            raise ValueError(
                f"qualified config descriptor is invalid for {worlds} worlds"
            )
        relative_name = descriptor.get("path")
        if not isinstance(relative_name, str):
            raise ValueError(
                f"qualified config path is missing for {worlds} worlds"
            )
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"qualified config path is unsafe for {worlds} worlds"
            )
        candidate_path = (run_dir / relative_path).resolve()
        try:
            candidate_path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"qualified config escaped the run directory for "
                f"{worlds} worlds"
            ) from exc
        if (
            not candidate_path.is_file()
            or _sha256(candidate_path) != descriptor.get("sha256")
        ):
            raise ValueError(
                f"qualified config is missing or changed for {worlds} worlds"
            )
        candidate_payload = json.loads(
            candidate_path.read_text(encoding="utf-8")
        )
        candidate_worlds = (
            candidate_payload.get("physics", {}).get("worlds")
            if isinstance(candidate_payload, dict)
            else None
        )
        if candidate_worlds != worlds or descriptor.get("worlds") != worlds:
            raise ValueError(
                f"qualified config world count mismatch for {worlds}"
            )
        expected_payload = derive_config(
            base_payload,
            worlds=worlds,
            gpu_name="capacity-candidate",
            scale_world_budgets=(
                budget_policy == "scale_tqc_with_worlds_v1"
            ),
        )
        if candidate_payload != expected_payload:
            raise ValueError(
                f"qualified config does not implement {budget_policy} "
                f"for {worlds} worlds"
            )
        validated_configs[worlds] = candidate_path
    selected_worlds = selection.get("selected_training_worlds")
    if selected_worlds != max(safe_worlds):
        raise ValueError("selection maximum world count is inconsistent")
    if _sha256(config_path) != _sha256(validated_configs[selected_worlds]):
        raise ValueError(
            "selected config does not match the maximum qualified config"
        )
    return selection, config_path


def resolve_qualified_config(
    run_dir: Path,
    requested_worlds: str | int,
    *,
    allow_external: bool = False,
    expected_profile_id: str | None = None,
    expected_pipeline_mode: str | None = None,
    expected_candidates: list[int] | None = None,
    expected_budget_policy: str | None = None,
    minimum_sweep_physics_steps: int | None = None,
    maximum_memory_ceiling: float | None = None,
) -> tuple[dict[str, Any], Path, int]:
    selection, _selected_config = load_and_validate_selection(
        run_dir,
        allow_external=allow_external,
        expected_profile_id=expected_profile_id,
        expected_pipeline_mode=expected_pipeline_mode,
        expected_candidates=expected_candidates,
        expected_budget_policy=expected_budget_policy,
        minimum_sweep_physics_steps=minimum_sweep_physics_steps,
        maximum_memory_ceiling=maximum_memory_ceiling,
    )
    if str(requested_worlds).lower() in {"auto", "max"}:
        worlds = int(selection["selected_training_worlds"])
    else:
        try:
            worlds = int(requested_worlds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"world selection must be 'max' or an integer, got "
                f"{requested_worlds!r}"
            ) from exc
    safe_worlds = [int(value) for value in selection["safe_qualified_worlds"]]
    if worlds not in safe_worlds:
        raise ValueError(
            f"{worlds} worlds was not safely qualified; choose one of "
            f"{safe_worlds}"
        )
    descriptor = selection["qualified_configs"][str(worlds)]
    config_path = (run_dir / str(descriptor["path"])).resolve()
    return selection, config_path, worlds


def gate(
    *,
    run_dir: Path,
    timeout_seconds: int,
    allow_external: bool = False,
    requested_worlds: str | int = "max",
    physics_steps: int | None = None,
    expected_profile_id: str | None = None,
    expected_pipeline_mode: str | None = None,
    expected_candidates: list[int] | None = None,
    expected_budget_policy: str | None = None,
    minimum_sweep_physics_steps: int | None = None,
    maximum_memory_ceiling: float | None = None,
) -> dict[str, Any]:
    selection, config_path, selected = resolve_qualified_config(
        run_dir,
        requested_worlds,
        allow_external=allow_external,
        expected_profile_id=expected_profile_id,
        expected_pipeline_mode=expected_pipeline_mode,
        expected_candidates=expected_candidates,
        expected_budget_policy=expected_budget_policy,
        minimum_sweep_physics_steps=minimum_sweep_physics_steps,
        maximum_memory_ceiling=maximum_memory_ceiling,
    )
    gate_physics_steps = (
        int(selection["physics_steps_per_phase"])
        if physics_steps is None
        else int(physics_steps)
    )
    if gate_physics_steps <= 0:
        raise ValueError("gate physics_steps must be positive")
    record = _run_child(
        config=config_path,
        worlds=selected,
        memory_ceiling=float(selection["memory_ceiling_fraction"]),
        physics_steps=gate_physics_steps,
        timeout_seconds=timeout_seconds,
        pipeline_mode=str(selection["pipeline_mode"]),
    )
    original_gpu = selection.get("gpu_fingerprint") or {}
    current = (record.get("preflight") or {}).get("gpu") or {}
    same_gpu_type = (
        original_gpu.get("pbs_gpu_type")
        == current.get("pbs_gpu_type")
    )
    same_compute_capability = (
        original_gpu.get("compute_capability")
        == current.get("compute_capability")
    )
    original_memory = int(original_gpu.get("total_memory_bytes") or 0)
    current_memory = int(current.get("total_memory_bytes") or 0)
    sufficient_memory = (
        original_memory > 0
        and current_memory >= int(original_memory * 0.99)
    )
    record.update(
        {
            "event": "adaptive_capacity_gate",
            "verified_utc": _utc_now(),
            "selected_config": str(config_path),
            "selected_config_sha256": _sha256(config_path),
            "selected_worlds": selected,
            "physics_steps_per_phase": gate_physics_steps,
            "original_gpu_type": original_gpu.get("pbs_gpu_type"),
            "current_gpu_type": current.get("pbs_gpu_type"),
            "same_gpu_type": same_gpu_type,
            "same_compute_capability": same_compute_capability,
            "sufficient_gpu_memory": sufficient_memory,
        }
    )
    history_path = run_dir / "gate_history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    if (
        record["status"] != "safe_pass"
        or not same_gpu_type
        or not same_compute_capability
        or not sufficient_memory
    ):
        raise RuntimeError(
            "selected world count or GPU fingerprint failed the "
            "formal-stack gate on the currently allocated GPU"
        )
    return record


def _run_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    _run_dir_argument(parser)
    parser.add_argument("--timeout-seconds", type=int, default=5400)


def _selection_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--require-profile-id", default=None)
    parser.add_argument(
        "--require-pipeline-mode",
        choices=("diffusion-only", "diffusion-tqc"),
        default=None,
    )
    parser.add_argument(
        "--require-candidates",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument("--require-budget-policy", default=None)
    parser.add_argument(
        "--require-min-sweep-physics-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--maximum-memory-ceiling",
        type=float,
        default=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_external_path_flag(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe_parser = subparsers.add_parser("probe")
    _common_arguments(probe_parser)
    probe_parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "configs" / "cx3_l40s.json",
    )
    probe_parser.add_argument(
        "--worlds",
        type=int,
        nargs="+",
        default=(512, 1024, 2048, 4096, 8192),
    )
    probe_parser.add_argument("--memory-ceiling", type=float, default=0.90)
    probe_parser.add_argument("--physics-steps", type=int, default=96)
    probe_parser.add_argument("--require-worlds", type=int, default=512)
    probe_parser.add_argument("--profile-id", default="generic")
    probe_parser.add_argument(
        "--pipeline-mode",
        choices=("diffusion-only", "diffusion-tqc"),
        default="diffusion-tqc",
    )
    probe_parser.add_argument(
        "--scale-world-budgets",
        action="store_true",
        help=(
            "scale TQC transitions, warm-up, replay, batch, and checkpoint "
            "budgets in direct proportion to worlds"
        ),
    )
    probe_parser.add_argument(
        "--stop-after-first-unsafe",
        action="store_true",
        help=(
            "stop immediately after the first failed or over-memory "
            "candidate and select the preceding safe candidate"
        ),
    )
    gate_parser = subparsers.add_parser("gate")
    _common_arguments(gate_parser)
    _selection_contract_arguments(gate_parser)
    gate_parser.add_argument("--worlds", default="max")
    gate_parser.add_argument(
        "--physics-steps",
        type=int,
        default=None,
        help="override the sweep's per-phase physics steps for this gate",
    )
    resolve_parser = subparsers.add_parser("resolve")
    _run_dir_argument(resolve_parser)
    _selection_contract_arguments(resolve_parser)
    resolve_parser.add_argument("--worlds", default="max")
    args = parser.parse_args()
    if hasattr(args, "timeout_seconds") and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.command == "probe":
        if (
            any(value <= 0 for value in args.worlds)
            or list(args.worlds) != sorted(set(args.worlds))
        ):
            parser.error("--worlds must be unique, positive, and increasing")
        if not 0.50 <= args.memory_ceiling < 1.0:
            parser.error("--memory-ceiling must be in [0.50, 1.0)")
        if args.physics_steps <= 0:
            parser.error("--physics-steps must be positive")
        if args.require_worlds not in args.worlds:
            parser.error("--require-worlds must be one of --worlds")
        if not args.profile_id:
            parser.error("--profile-id cannot be empty")
    if (
        args.command == "gate"
        and args.physics_steps is not None
        and args.physics_steps <= 0
    ):
        parser.error("--physics-steps must be positive")
    if args.command in {"gate", "resolve"}:
        if (
            args.require_candidates is not None
            and (
                any(value <= 0 for value in args.require_candidates)
                or args.require_candidates
                != sorted(set(args.require_candidates))
            )
        ):
            parser.error(
                "--require-candidates must be unique, positive, and increasing"
            )
        if (
            args.require_min_sweep_physics_steps is not None
            and args.require_min_sweep_physics_steps <= 0
        ):
            parser.error(
                "--require-min-sweep-physics-steps must be positive"
            )
        if (
            args.maximum_memory_ceiling is not None
            and not 0.50 <= args.maximum_memory_ceiling < 1.0
        ):
            parser.error("--maximum-memory-ceiling must be in [0.50, 1.0)")
    try:
        run_dir = resolve_package_path(
            args.run_dir,
            allow_external=args.allow_external_paths,
            label="adaptive capacity output",
        )
        if args.command == "probe":
            base_config = resolve_package_path(
                args.base_config,
                allow_external=args.allow_external_paths,
                label="base config",
            )
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "probe":
        probe(
            base_config=base_config,
            run_dir=run_dir,
            worlds=list(args.worlds),
            memory_ceiling=args.memory_ceiling,
            physics_steps=args.physics_steps,
            timeout_seconds=args.timeout_seconds,
            require_worlds=args.require_worlds,
            scale_world_budgets=args.scale_world_budgets,
            profile_id=args.profile_id,
            pipeline_mode=args.pipeline_mode,
            stop_after_first_unsafe=args.stop_after_first_unsafe,
        )
    elif args.command == "gate":
        gate(
            run_dir=run_dir,
            timeout_seconds=args.timeout_seconds,
            allow_external=args.allow_external_paths,
            requested_worlds=args.worlds,
            physics_steps=args.physics_steps,
            expected_profile_id=args.require_profile_id,
            expected_pipeline_mode=args.require_pipeline_mode,
            expected_candidates=args.require_candidates,
            expected_budget_policy=args.require_budget_policy,
            minimum_sweep_physics_steps=(
                args.require_min_sweep_physics_steps
            ),
            maximum_memory_ceiling=args.maximum_memory_ceiling,
        )
    else:
        _selection, config_path, worlds = resolve_qualified_config(
            run_dir,
            args.worlds,
            allow_external=args.allow_external_paths,
            expected_profile_id=args.require_profile_id,
            expected_pipeline_mode=args.require_pipeline_mode,
            expected_candidates=args.require_candidates,
            expected_budget_policy=args.require_budget_policy,
            minimum_sweep_physics_steps=(
                args.require_min_sweep_physics_steps
            ),
            maximum_memory_ceiling=args.maximum_memory_ceiling,
        )
        print(worlds)
        print(config_path)


if __name__ == "__main__":
    main()
