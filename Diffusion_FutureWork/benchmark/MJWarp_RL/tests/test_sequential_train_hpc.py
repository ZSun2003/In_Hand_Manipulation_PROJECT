from __future__ import annotations

import copy
from contextlib import contextmanager
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from MJWarp_RL import contract
from MJWarp_RL import sequential_train_hpc as sequence
from MJWarp_RL.hpc_config import DEFAULT_CONFIG
from MJWarp_RL.runtime_fingerprint import runtime_code_fingerprint_sha256


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class SequentialTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pins = {
            task: {
                "report_path": f"/reports/{task}/common.json",
                "report_sha256": ("1" if task == sequence.TASK_TRANSLATE else "4")
                * 64,
                "source_report_path": f"/reports/{task}/g3.json",
                "source_report_sha256": (
                    "2" if task == sequence.TASK_TRANSLATE else "5"
                )
                * 64,
                "runtime_code_fingerprint_sha256": (
                    sequence.DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256
                ),
                "task": task,
                "platform_root_heights_m": {
                    platform: sequence.EXPECTED_ROOT_HEIGHT_M[(platform, task)]
                    for platform in ("gripper3", "gripper6")
                },
                "platform_rl_model_fingerprints_sha256": {
                    "gripper3": "a" * 64,
                    "gripper6": "b" * 64,
                },
                "selected_num_envs": sequence.EXPECTED_WORLDS,
            }
            for task in sequence.TASKS
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_r21_control_height_defaults_and_complete_config_are_pinned(self) -> None:
        self.assertEqual(sequence.DEFAULT_RELEASE_ROOT.name, "Diffusion_FutureWork")
        self.assertIn(
            "capacity_reports",
            str(sequence.DEFAULT_TRANSLATION_CAPACITY_REPORT),
        )
        self.assertIn(
            "capacity_reports",
            str(sequence.DEFAULT_ROTATION_CAPACITY_REPORT),
        )
        self.assertEqual(sequence.EXPECTED_WORLDS, 20_000)
        self.assertEqual(sequence.EXPECTED_TOTAL_STEPS, 60_000_000)
        self.assertEqual(
            sequence.EXPECTED_ALGORITHM_SIGNATURE,
            contract.ALGORITHM_SIGNATURE,
        )
        self.assertEqual(
            sequence.DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256,
            runtime_code_fingerprint_sha256(),
        )
        for task in sequence.TASKS:
            with self.subTest(task=task):
                expected = json.loads(
                    json.dumps(replace(DEFAULT_CONFIG, task=task).as_dict())
                )
                self.assertEqual(sequence._expected_training_config(task), expected)
                self.assertEqual(expected["control_ticks"], 3_000)
                self.assertEqual(expected["episode_length"], 3_000)
                self.assertEqual(expected["maximum_demo_episodes"], 2_000)
                self.assertFalse(
                    sequence._legacy_phase_key_paths(expected, path="config")
                )

    def _artifact(
        self,
        platform: str,
        task: str = sequence.TASK_TRANSLATE,
        *,
        hard_failure: int = 0,
        suffix: str = "",
    ) -> tuple[Path, dict[str, object]]:
        output = self.root / f"{platform}_{task}{suffix}"
        output.mkdir()
        model = output / "final_diffusion_model.pt"
        model.write_bytes((platform + "-model").encode() * 64)
        model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
        config = sequence._expected_training_config(task)
        prefix = (
            "G3_Project_RL" if platform == "gripper3" else "_g6_project_rl"
        )
        root_height_m = sequence.EXPECTED_ROOT_HEIGHT_M[(platform, task)]
        model_shards = [
            {
                "object_index": index,
                "object": object_kind,
                "task": task,
                "task_index": sequence.EXPECTED_TASK_INDEX[task],
                "root_height_m": root_height_m,
                "model_xml": f"/release/{prefix}_{task}_{object_kind}.xml",
                "model_xml_sha256": str(index + 6) * 64,
                "nq": 27,
                "nv": 26,
                "nu": 12,
                "worlds": 5_000,
            }
            for index, object_kind in enumerate(sequence.EXPECTED_OBJECTS)
        ]
        fingerprints = {
            "parent_model_fingerprint_sha256": "c" * 64,
            "parent_controller_fingerprint_sha256": "d" * 64,
            "platform_adapter_fingerprint_sha256": "e" * 64,
        }
        capacity = {
            **self.pins[task],
            "platforms": ["gripper3", "gripper6"],
            "task_index": sequence.EXPECTED_TASK_INDEX[task],
            "root_height_m": root_height_m,
            "object_world_counts": [5_000] * 4,
            "remainder_worlds": 0,
            "remainder_object_world_counts": [0] * 4,
            "model_shards": model_shards,
            "passing_candidates": [sequence.EXPECTED_WORLDS],
            "config_canonical_sha256": "1" * 64,
            "candidate_config_set_sha256": "2" * 64,
            "selected_config_sha256": "3" * 64,
            "certified_stage_peak_increment_by_stage": {
                "native_search": 1,
                "diffusion": 2,
            },
            "training_memory_admissions": {
                "native_search": {},
                "diffusion": {},
            },
            **fingerprints,
        }
        counters = {
            "ik_solve_fallback_count": hard_failure,
            "ik_trust_region_projection_count": 4,
            "position_feedback_slew_clip_count": 5,
            "preload_trim_slew_clip_count": 6,
            "ctrl_slew_clip_count": 7,
            "ctrl_certificate_fallback_count": 0,
        }
        memory = {
            "valid": True,
            "closed": True,
            "limit_violated": False,
            "total_memory_changed": False,
            "errors": [],
            "peak_device_memory_fraction": 0.5,
            "peak_device_used_gb": 47.5,
        }
        demo_count = 128
        demo_distribution = {
            f"all/{object_kind}/{task}": 32
            for object_kind in sequence.EXPECTED_OBJECTS
        }
        empty_path = {
            "count": 0,
            "successful_count": 0,
            "seed_unique_count": 0,
            "shape_counts": {},
            "shape_success_counts": {},
            "tracking_rmse_mean_m": None,
            "tracking_rmse_max_m": None,
            "tracking_max_error_m": None,
        }
        full_path = {
            **empty_path,
            "count": 32,
            "successful_count": 32,
            "seed_unique_count": 1,
            "shape_counts": {
                f"shape_{index}": 1 for index in range(5)
            },
            "shape_success_counts": {
                f"shape_{index}": 1 for index in range(5)
            },
            "tracking_rmse_mean_m": 0.001,
            "tracking_rmse_max_m": 0.002,
            "tracking_max_error_m": 0.003,
        }
        empty_rotation = {
            "count": 0,
            "successful_count": 0,
            "angle_min_deg": None,
            "angle_max_deg": None,
            "angle_unique_count": 0,
            "angle_histogram": {},
            "maximum_target_deg": 45,
            "maximum_target_count": 0,
            "maximum_target_success_count": 0,
        }
        full_rotation = {
            **empty_rotation,
            "count": 32,
            "successful_count": 32,
            "angle_min_deg": 10,
            "angle_max_deg": 45,
            "angle_unique_count": 2,
            "angle_histogram": {"10": 31, "45": 1},
            "maximum_target_count": 1,
            "maximum_target_success_count": 1,
        }
        demo_coverage = {
            "run_counts": {"all": demo_count},
            "object_counts": {
                object_kind: 32 for object_kind in sequence.EXPECTED_OBJECTS
            },
            "task_counts": {
                candidate: demo_count if candidate == task else 0
                for candidate in sequence.TASKS
            },
            "template_counts": {
                object_kind: {str(index): 1 for index in range(32)}
                for object_kind in sequence.EXPECTED_OBJECTS
            },
            "source_region_counts": {"all": demo_count},
            "source_region_by_object_task": {},
            "translation_path_by_object": {
                object_kind: dict(
                    full_path if task == sequence.TASK_TRANSLATE else empty_path
                )
                for object_kind in sequence.EXPECTED_OBJECTS
            },
            "rotation_angle": dict(
                full_rotation if task == sequence.TASK_ROTATE else empty_rotation
            ),
            "rotation_angle_by_object": {
                object_kind: dict(
                    full_rotation
                    if task == sequence.TASK_ROTATE
                    else empty_rotation
                )
                for object_kind in sequence.EXPECTED_OBJECTS
            },
            "object_task_counts": demo_distribution,
        }
        expert_final = {
            "phase": "all",
            "steps": sequence.EXPECTED_TOTAL_STEPS,
            "control_ticks": 3_000,
            "remainder_worlds": 0,
            "remainder_object_world_counts": [0] * 4,
            "remainder_backend_health": None,
            "remainder_controller_safety_counters": None,
            "controller_safety_counters_total": counters,
        }
        if task == sequence.TASK_ROTATE:
            expert_final["rotation_target_angle_bounds_deg"] = [10, 45]
        manifest = {
            "schema": sequence.TRAINING_MANIFEST_SCHEMA,
            "created_utc": "2026-08-03T00:00:00+00:00",
            "algorithm": sequence.EXPECTED_ALGORITHM,
            "algorithm_signature": sequence.EXPECTED_ALGORITHM_SIGNATURE,
            "control_hierarchy": (
                "Diffusion imitation -> independent hard-tip IK commands -> "
                "per-tip constrained IK/KKT -> actuators"
            ),
            "physics_backend": "native-mujoco-warp",
            "torch_warp_exchange": "persistent-zero-copy-alias",
            "platform": platform,
            "trained_task": task,
            "task": task,
            "task_index": sequence.EXPECTED_TASK_INDEX[task],
            "root_height_m": root_height_m,
            "expert": sequence.EXPECTED_EXPERT,
            "physics_layout": sequence.EXPECTED_PHYSICS_LAYOUT,
            "runtime_code_fingerprint_sha256": (
                sequence.DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256
            ),
            "total_mjwarp_expert_steps": sequence.EXPECTED_TOTAL_STEPS,
            "expert_steps": sequence.EXPECTED_TOTAL_STEPS,
            "rotation_angle_distribution_deg": {
                "bounds": [10, 45],
                "integer_closed_intervals": True,
                "maximum_successful_demo_required_per_object": True,
            },
            "translation_path_contract": {"frozen": True},
            "diffusion_epochs": 60,
            "success_gate": True,
            "trajectory_certification": (
                sequence.EXPECTED_TRAJECTORY_CERTIFICATION
            ),
            "anti_support_gate_used": False,
            "physical_hard_tip_plane_success_invariant": True,
            "grasp_plane_fraction_bounds": [0.2, 0.6],
            "learned_grasp_plane_objects": list(sequence.EXPECTED_OBJECTS),
            "fixed_grasp_plane_fractions": {},
            "fixed_grasp_plane_yaw_rad": 0.0,
            "fixed_grasp_plane_physical_tolerance_m": 0.002,
            "independent_hard_tip_control": True,
            "diffusion_action_supervision": sequence.EXPECTED_ACTION_SUPERVISION,
            "plate_trained": False,
            "reset_templates_per_object": 32,
            "global_worlds": sequence.EXPECTED_WORLDS,
            "object_world_counts": [5_000] * 4,
            "remainder_worlds": 0,
            "remainder_object_world_counts": [0] * 4,
            "model_shards": model_shards,
            **fingerprints,
            "fruit_manifest_sha256": "f" * 64,
            "maximum_shard_nv": 26,
            "observation_dim": sequence.EXPECTED_OBSERVATION_DIM,
            "action_dim": sequence.EXPECTED_ACTION_DIM,
            "config": config,
            "capacity_selection": capacity,
            "training_device_memory": memory,
            "controller_safety": {
                "fail_closed": True,
                "hard_failure_fields": list(sequence.HARD_COUNTERS),
                "all": counters,
                "combined_totals": counters,
            },
            "expert_final": expert_final,
            "certified_demo_count": demo_count,
            "certified_demo_distribution": demo_distribution,
            "certified_demo_coverage": demo_coverage,
            "final_model": model.name,
            "final_model_sha256": model_sha,
        }
        _write_json(output / "training_manifest.json", manifest)
        first = {
            "schema": "grab-diffusion-live-v1",
            "sequence": 1,
            "state": "running",
            "stage": "ik_expert",
            "phase": "all",
            "target_steps": sequence.EXPECTED_TOTAL_STEPS,
            "total_steps": 1,
        }
        final = {
            "schema": "grab-diffusion-live-v1",
            "sequence": 2,
            "state": "complete",
            "stage": "complete",
            "phase": "complete",
            "target_steps": sequence.EXPECTED_TOTAL_STEPS,
            "total_steps": sequence.EXPECTED_TOTAL_STEPS,
            "controller_safety_counters": counters,
            "training_device_memory": memory,
            "final_model": model.name,
        }
        (output / "training.jsonl").write_text(
            json.dumps(first, sort_keys=True)
            + "\n"
            + json.dumps(final, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        _write_json(output / "run_status.json", final)
        return output, manifest

    def _exercise_foreground(
        self, *, run_id: str, returncodes: list[int]
    ) -> dict[str, object]:
        output_base = self.root / "formal-runs"
        output_base.mkdir(exist_ok=True)
        resolved = {
            "release": Path("/release"),
            "g3": Path("/release/GRAB"),
            "g6": Path("/release/GRAB_G6"),
            "python": Path("/core/bin/python"),
            "capacity_reports": {
                sequence.TASK_TRANSLATE: Path("/reports/translate/common.json"),
                sequence.TASK_ROTATE: Path("/reports/rotate/common.json"),
            },
            "source_reports": {
                sequence.TASK_TRANSLATE: Path("/reports/translate/g3.json"),
                sequence.TASK_ROTATE: Path("/reports/rotate/g3.json"),
            },
            "output_base": output_base,
            "lock_file": self.root / "gpu0.lock",
            "report_pins": self.pins,
        }
        calls: dict[str, list[object]] = {
            "admission": [],
            "command": [],
            "run_log": [],
            "validation": [],
            "archive": [],
        }

        class FakeOwner:
            received_signal = None

            def install_handlers(self) -> None:
                return None

            def terminate_active(self) -> None:
                return None

            def run_log(self, _command: object, **kwargs: object) -> int:
                calls["run_log"].append(Path(str(kwargs["log_path"])).name)
                index = len(calls["run_log"]) - 1
                return returncodes[index] if index < len(returncodes) else 0

        @contextmanager
        def fake_lock(_path: Path):
            yield {"path": str(_path), "pid": 1}

        def admission(**kwargs: object) -> dict[str, object]:
            platform = str(kwargs["expected_platform"])
            task = str(kwargs["expected_task"])
            calls["admission"].append((platform, task))
            pins = self.pins[task]
            return {
                "platform": platform,
                "task": task,
                "task_index": sequence.EXPECTED_TASK_INDEX[task],
                "root_height_m": kwargs["expected_root_height_m"],
                "platform_root_heights_m": pins[
                    "platform_root_heights_m"
                ],
                "platform_rl_model_fingerprints_sha256": pins[
                    "platform_rl_model_fingerprints_sha256"
                ],
                "selected_num_envs": sequence.EXPECTED_WORLDS,
                "runtime_code_fingerprint_sha256": (
                    sequence.DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256
                ),
                "report_sha256": pins["report_sha256"],
                "source_report_sha256": pins["source_report_sha256"],
                "object_world_counts": [5_000] * 4,
                "config_canonical_sha256": "c" * 64,
            }

        def training_command(**kwargs: object) -> list[str]:
            label = Path(str(kwargs["output_dir"])).name
            calls["command"].append(
                (label, str(kwargs["task"]), str(kwargs["project_root"]))
            )
            return ["train", label]

        def validate(**kwargs: object) -> dict[str, object]:
            platform = str(kwargs["expected_platform"])
            task = str(kwargs["expected_task"])
            label = Path(str(kwargs["output_dir"])).name
            calls["validation"].append((label, platform, task))
            pins = dict(kwargs["report_pins"])
            pins.update(
                object_world_counts=[5_000] * 4,
                selected_config_sha256="c" * 64,
            )
            return {
                "platform": platform,
                "task": task,
                "root_height_m": kwargs["expected_root_height_m"],
                "algorithm_signature": "same-algorithm",
                "physics_layout": "same-layout",
                "runtime_code_fingerprint_sha256": (
                    sequence.DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256
                ),
                "config": {
                    "task": task,
                    "seed": sequence.FIXED_SEED,
                    "device": sequence.FIXED_DEVICE,
                },
                "capacity_selection": pins,
            }

        def archive(**kwargs: object) -> dict[str, object]:
            label = str(kwargs["label"])
            calls["archive"].append(label)
            return {"path": f"/packages/{label}.tar.gz", "sha256": "d" * 64}

        args = SimpleNamespace(run_id=run_id, prepared_root=False)
        result: int | None = None
        error: BaseException | None = None
        with (
            mock.patch.object(sequence, "_validate_inputs", return_value=resolved),
            mock.patch.object(sequence, "ChildOwner", return_value=FakeOwner()),
            mock.patch.object(sequence, "_exclusive_gpu_lock", fake_lock),
            mock.patch.object(sequence, "_gpu_idle_snapshot", return_value={"idle": True}),
            mock.patch.object(sequence, "_disk_gate", return_value={"free": True}),
            mock.patch.object(sequence, "_pair_admission", side_effect=admission),
            mock.patch.object(sequence, "_training_command", side_effect=training_command),
            mock.patch.object(
                sequence, "validate_training_artifacts", side_effect=validate
            ),
            mock.patch.object(sequence, "create_models_archive", side_effect=archive),
            mock.patch("builtins.print"),
        ):
            try:
                result = sequence._foreground(args)
            except BaseException as exc:
                error = exc
        status = json.loads(
            (output_base / run_id / "sequence_status.json").read_text(
                encoding="utf-8"
            )
        )
        return {"result": result, "error": error, "calls": calls, "status": status}

    def test_report_pair_is_hash_and_source_path_bound(self) -> None:
        source_path = self.root / "g3.json"
        source = {
            "is_common_report": False,
            "platform": "gripper3",
            "task": sequence.TASK_TRANSLATE,
            "root_height_m": 0.525,
        }
        _write_json(source_path, source)
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        common_path = self.root / "common.json"
        common = {
            "is_common_report": True,
            "common_platforms": ["gripper3", "gripper6"],
            "common_with": str(source_path.resolve()),
            "common_with_sha256": source_sha,
            "selected_num_envs": sequence.EXPECTED_WORLDS,
            "platform": "gripper6",
            "task": sequence.TASK_TRANSLATE,
            "root_height_m": 0.525,
            "platform_root_heights_m": {
                "gripper3": 0.525,
                "gripper6": 0.525,
            },
            "platform_rl_model_fingerprints_sha256": {
                "gripper3": "a" * 64,
                "gripper6": "b" * 64,
            },
            "runtime_code_fingerprint_sha256": (
                sequence.DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256
            ),
        }
        _write_json(common_path, common)
        common_sha = hashlib.sha256(common_path.read_bytes()).hexdigest()
        result = sequence._verify_report_pair(
            capacity_report=common_path,
            capacity_sha256=common_sha,
            source_report=source_path,
            source_sha256=source_sha,
            expected_task=sequence.TASK_TRANSLATE,
        )
        self.assertEqual(result["report_sha256"], common_sha)
        self.assertEqual(result["source_report_sha256"], source_sha)

        source_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(sequence.SequenceFailure, "source.*SHA256"):
            sequence._verify_report_pair(
                capacity_report=common_path,
                capacity_sha256=common_sha,
                source_report=source_path,
                source_sha256=source_sha,
                expected_task=sequence.TASK_TRANSLATE,
            )

    def test_report_pair_rejects_cross_task_reuse(self) -> None:
        source_path = self.root / "g3-rotate.json"
        source = {
            "is_common_report": False,
            "platform": "gripper3",
            "task": sequence.TASK_ROTATE,
            "root_height_m": 0.525,
        }
        _write_json(source_path, source)
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        common_path = self.root / "common-rotate.json"
        common = {
            "is_common_report": True,
            "common_platforms": ["gripper3", "gripper6"],
            "common_with": str(source_path.resolve()),
            "common_with_sha256": source_sha,
            "selected_num_envs": sequence.EXPECTED_WORLDS,
            "platform": "gripper6",
            "task": sequence.TASK_ROTATE,
            "root_height_m": 0.525,
            "platform_root_heights_m": {
                "gripper3": 0.525,
                "gripper6": 0.525,
            },
            "platform_rl_model_fingerprints_sha256": {
                "gripper3": "a" * 64,
                "gripper6": "b" * 64,
            },
            "runtime_code_fingerprint_sha256": (
                sequence.DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256
            ),
        }
        _write_json(common_path, common)
        with self.assertRaisesRegex(sequence.SequenceFailure, "task"):
            sequence._verify_report_pair(
                capacity_report=common_path,
                capacity_sha256=hashlib.sha256(common_path.read_bytes()).hexdigest(),
                source_report=source_path,
                source_sha256=source_sha,
                expected_task=sequence.TASK_TRANSLATE,
            )

    def test_success_artifacts_validate_model_status_and_hard_counters(self) -> None:
        output, _ = self._artifact("gripper3")
        result = sequence.validate_training_artifacts(
            output_dir=output,
            expected_platform="gripper3",
            expected_task=sequence.TASK_TRANSLATE,
            expected_root_height_m=0.525,
            report_pins=self.pins[sequence.TASK_TRANSLATE],
        )
        self.assertEqual(result["training_jsonl_records"], 2)
        self.assertEqual(result["final_model_bytes"], len(b"gripper3-model") * 64)
        self.assertEqual(result["peak_device_memory_fraction"], 0.5)

    def test_rotation_artifact_binds_one_all_phase_and_10_to_45_degrees(self) -> None:
        output, manifest = self._artifact(
            "gripper6", task=sequence.TASK_ROTATE
        )
        result = sequence.validate_training_artifacts(
            output_dir=output,
            expected_platform="gripper6",
            expected_task=sequence.TASK_ROTATE,
            expected_root_height_m=0.525,
            report_pins=self.pins[sequence.TASK_ROTATE],
        )
        self.assertEqual(result["task"], sequence.TASK_ROTATE)
        self.assertEqual(manifest["expert_final"]["phase"], "all")
        self.assertEqual(
            manifest["expert_final"]["rotation_target_angle_bounds_deg"],
            [10, 45],
        )
        self.assertFalse(sequence._legacy_phase_key_paths(manifest))

    def test_artifacts_reject_every_legacy_two_phase_surface(self) -> None:
        output, manifest = self._artifact("gripper3")
        manifest["phase1_expert_steps"] = sequence.EXPECTED_TOTAL_STEPS
        manifest["certified_demo_coverage"]["phase2_counts"] = {}
        _write_json(output / "training_manifest.json", manifest)
        with self.assertRaisesRegex(sequence.SequenceFailure, "legacy phase1/phase2"):
            sequence.validate_training_artifacts(
                output_dir=output,
                expected_platform="gripper3",
                expected_task=sequence.TASK_TRANSLATE,
                expected_root_height_m=0.525,
                report_pins=self.pins[sequence.TASK_TRANSLATE],
            )

    def test_training_jsonl_rejects_legacy_phase_stage(self) -> None:
        output, _ = self._artifact("gripper3")
        jsonl = output / "training.jsonl"
        records = [json.loads(line) for line in jsonl.read_text().splitlines()]
        records[0]["stage"] = "phase1"
        jsonl.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(sequence.SequenceFailure, "legacy two-phase"):
            sequence.validate_training_artifacts(
                output_dir=output,
                expected_platform="gripper3",
                expected_task=sequence.TASK_TRANSLATE,
                expected_root_height_m=0.525,
                report_pins=self.pins[sequence.TASK_TRANSLATE],
            )

    def test_artifacts_reject_task_and_root_height_drift(self) -> None:
        output, manifest = self._artifact("gripper3")
        manifest["trained_task"] = sequence.TASK_ROTATE
        _write_json(output / "training_manifest.json", manifest)
        with self.assertRaisesRegex(sequence.SequenceFailure, "trained_task"):
            sequence.validate_training_artifacts(
                output_dir=output,
                expected_platform="gripper3",
                expected_task=sequence.TASK_TRANSLATE,
                expected_root_height_m=0.525,
                report_pins=self.pins[sequence.TASK_TRANSLATE],
            )

    def test_artifacts_reject_capacity_task_index_root_and_rl_pin_drift(self) -> None:
        mutations = (
            ("task", sequence.TASK_ROTATE),
            ("task_index", 1),
            ("root_height_m", 0.518),
            (
                "platform_rl_model_fingerprints_sha256",
                {"gripper3": "f" * 64, "gripper6": "b" * 64},
            ),
        )
        for index, (name, value) in enumerate(mutations):
            with self.subTest(field=name):
                output, manifest = self._artifact(
                    "gripper3",
                    task=sequence.TASK_TRANSLATE,
                    suffix=f"_{index}",
                )
                manifest["capacity_selection"][name] = value
                _write_json(output / "training_manifest.json", manifest)
                with self.assertRaisesRegex(
                    sequence.SequenceFailure,
                    f"capacity selection mismatch for {name}",
                ):
                    sequence.validate_training_artifacts(
                        output_dir=output,
                        expected_platform="gripper3",
                        expected_task=sequence.TASK_TRANSLATE,
                        expected_root_height_m=0.525,
                        report_pins=self.pins[sequence.TASK_TRANSLATE],
                    )
        manifest["trained_task"] = sequence.TASK_TRANSLATE
        manifest["root_height_m"] = 0.518
        _write_json(output / "training_manifest.json", manifest)
        with self.assertRaisesRegex(sequence.SequenceFailure, "root_height_m"):
            sequence.validate_training_artifacts(
                output_dir=output,
                expected_platform="gripper3",
                expected_task=sequence.TASK_TRANSLATE,
                expected_root_height_m=0.525,
                report_pins=self.pins[sequence.TASK_TRANSLATE],
            )

    def test_failed_artifacts_are_rejected(self) -> None:
        output, _ = self._artifact("gripper3", hard_failure=1)
        with self.assertRaisesRegex(sequence.SequenceFailure, "hard counter"):
            sequence.validate_training_artifacts(
                output_dir=output,
                expected_platform="gripper3",
                expected_task=sequence.TASK_TRANSLATE,
                expected_root_height_m=0.525,
                report_pins=self.pins[sequence.TASK_TRANSLATE],
            )

    def test_model_sha_mismatch_is_rejected(self) -> None:
        output, _ = self._artifact("gripper3")
        (output / "final_diffusion_model.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(sequence.SequenceFailure, "model SHA256"):
            sequence.validate_training_artifacts(
                output_dir=output,
                expected_platform="gripper3",
                expected_task=sequence.TASK_TRANSLATE,
                expected_root_height_m=0.525,
                report_pins=self.pins[sequence.TASK_TRANSLATE],
            )

    def test_pair_consistency_accepts_controlled_pair_and_rejects_drift(self) -> None:
        g3_output, _ = self._artifact("gripper3")
        g6_output, _ = self._artifact("gripper6")
        g3 = sequence.validate_training_artifacts(
            output_dir=g3_output,
            expected_platform="gripper3",
            expected_task=sequence.TASK_TRANSLATE,
            expected_root_height_m=0.525,
            report_pins=self.pins[sequence.TASK_TRANSLATE],
        )
        g6 = sequence.validate_training_artifacts(
            output_dir=g6_output,
            expected_platform="gripper6",
            expected_task=sequence.TASK_TRANSLATE,
            expected_root_height_m=0.525,
            report_pins=self.pins[sequence.TASK_TRANSLATE],
        )
        self.assertTrue(
            sequence.validate_pair_consistency(
                g3, g6, expected_task=sequence.TASK_TRANSLATE
            )["consistent"]
        )
        drifted = copy.deepcopy(g6)
        drifted["config"]["seed"] = 8
        with self.assertRaisesRegex(sequence.SequenceFailure, "pair mismatch"):
            sequence.validate_pair_consistency(
                g3, drifted, expected_task=sequence.TASK_TRANSLATE
            )

    def test_models_archive_excludes_certified_ik_demos(self) -> None:
        output, _ = self._artifact("gripper3")
        for name in ("certified_ik_demos", "elite_demos"):
            demos = output / name
            demos.mkdir()
            (demos / "huge.bin").write_bytes(b"do not package")
        result = sequence.create_models_archive(
            output_dir=output,
            packages_dir=self.root / "packages",
            label="g3",
            run_id="unit",
        )
        archive = Path(result["path"])
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), result["sha256"])
        with tarfile.open(archive, "r:gz") as stream:
            members = stream.getnames()
        self.assertTrue(members)
        self.assertFalse(
            any(
                {"certified_ik_demos", "elite_demos"}.intersection(
                    Path(name).parts
                )
                for name in members
            )
        )
        self.assertTrue(result["certified_ik_demos_excluded"])

    def test_training_command_keeps_formal_controls_fixed(self) -> None:
        command = sequence._training_command(
            project_root=Path("/release/GRAB"),
            capacity_report=Path("/reports/common.json"),
            output_dir=Path("/runs/g3"),
            task=sequence.TASK_TRANSLATE,
        )
        self.assertEqual(
            command,
            [
                "/release/GRAB/MJWarp_RL/run_hpc.sh",
                "-m",
                "MJWarp_RL.train",
                "--seed",
                "7",
                "--device",
                "cuda:0",
                "--task",
                sequence.TASK_TRANSLATE,
                "--num-envs",
                "auto",
                "--capacity-report",
                "/reports/common.json",
                "--output",
                "/runs/g3",
            ],
        )

    def test_final_model_name_is_fixed(self) -> None:
        output, manifest = self._artifact("gripper3")
        manifest["final_model"] = "another_model.pt"
        _write_json(output / "training_manifest.json", manifest)
        with self.assertRaisesRegex(sequence.SequenceFailure, "final model must"):
            sequence.validate_training_artifacts(
                output_dir=output,
                expected_platform="gripper3",
                expected_task=sequence.TASK_TRANSLATE,
                expected_root_height_m=0.525,
                report_pins=self.pins[sequence.TASK_TRANSLATE],
            )

    def test_pair_admission_consistency_checks_world_distribution(self) -> None:
        base = {
            "task": sequence.TASK_TRANSLATE,
            "task_index": sequence.EXPECTED_TASK_INDEX[
                sequence.TASK_TRANSLATE
            ],
            "selected_num_envs": sequence.EXPECTED_WORLDS,
            "runtime_code_fingerprint_sha256": (
                sequence.DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256
            ),
            "report_sha256": "1" * 64,
            "source_report_sha256": "2" * 64,
            "object_world_counts": [5_000] * 4,
            "config_canonical_sha256": "3" * 64,
            "platform_root_heights_m": {
                "gripper3": 0.525,
                "gripper6": 0.525,
            },
            "platform_rl_model_fingerprints_sha256": {
                "gripper3": "a" * 64,
                "gripper6": "b" * 64,
            },
        }
        admissions = {
            "gripper3": {
                **base,
                "platform": "gripper3",
                "root_height_m": 0.525,
            },
            "gripper6": {
                **base,
                "platform": "gripper6",
                "root_height_m": 0.525,
            },
        }
        self.assertTrue(
            sequence._validate_admission_pair(
                admissions, expected_task=sequence.TASK_TRANSLATE
            )["consistent"]
        )
        admissions["gripper6"]["object_world_counts"] = [9_999] * 4
        with self.assertRaisesRegex(sequence.SequenceFailure, "mismatch"):
            sequence._validate_admission_pair(
                admissions, expected_task=sequence.TASK_TRANSLATE
            )

    def test_check_only_never_builds_or_runs_a_training_command(self) -> None:
        base = {
            "selected_num_envs": sequence.EXPECTED_WORLDS,
            "runtime_code_fingerprint_sha256": (
                sequence.DEFAULT_RUNTIME_CODE_FINGERPRINT_SHA256
            ),
            "report_sha256": "1" * 64,
            "source_report_sha256": "2" * 64,
            "object_world_counts": [5_000] * 4,
            "config_canonical_sha256": "3" * 64,
        }
        resolved = {
            "lock_file": self.root / "gpu0.lock",
            "output_base": self.root,
            "python": Path("/python"),
            "capacity_reports": {
                sequence.TASK_TRANSLATE: Path("/translation-common.json"),
                sequence.TASK_ROTATE: Path("/rotation-common.json"),
            },
            "g3": Path("/g3"),
            "g6": Path("/g6"),
            "report_pins": self.pins,
        }

        @contextmanager
        def fake_lock(_path: Path):
            yield {"path": str(_path), "pid": 1}

        def admission(**kwargs: object) -> dict[str, object]:
            platform = str(kwargs["expected_platform"])
            task = str(kwargs["expected_task"])
            return {
                **base,
                "platform": platform,
                "task": task,
                "task_index": sequence.EXPECTED_TASK_INDEX[task],
                "root_height_m": kwargs["expected_root_height_m"],
                "platform_root_heights_m": self.pins[task][
                    "platform_root_heights_m"
                ],
                "platform_rl_model_fingerprints_sha256": self.pins[task][
                    "platform_rl_model_fingerprints_sha256"
                ],
                "report_sha256": self.pins[task]["report_sha256"],
                "source_report_sha256": self.pins[task][
                    "source_report_sha256"
                ],
            }

        with (
            mock.patch.object(sequence, "_validate_inputs", return_value=resolved),
            mock.patch.object(sequence, "_exclusive_gpu_lock", fake_lock),
            mock.patch.object(sequence, "_gpu_idle_snapshot", return_value={"idle": True}),
            mock.patch.object(sequence, "_disk_gate", return_value={"free": True}),
            mock.patch.object(
                sequence, "_pair_admission", side_effect=admission
            ) as admission_mock,
            mock.patch.object(sequence, "_training_command") as training,
            mock.patch("builtins.print"),
        ):
            result = sequence._check_only(SimpleNamespace(run_id="unit"))
        self.assertEqual(result, 0)
        training.assert_not_called()
        self.assertEqual(admission_mock.call_count, 4)

    def test_background_screen_restarts_this_standalone_file(self) -> None:
        output_base = self.root / "runs"
        resolved = {
            "release": Path("/release"),
            "g3": Path("/release/GRAB"),
            "g6": Path("/release/GRAB_G6"),
            "python": Path("/core/bin/python"),
            "capacity_reports": {
                sequence.TASK_TRANSLATE: Path("/reports/translate/common.json"),
                sequence.TASK_ROTATE: Path("/reports/rotate/common.json"),
            },
            "source_reports": {
                sequence.TASK_TRANSLATE: Path("/reports/translate/g3.json"),
                sequence.TASK_ROTATE: Path("/reports/rotate/g3.json"),
            },
            "output_base": output_base,
            "lock_file": Path("/locks/gpu0.lock"),
            "report_pins": self.pins,
        }
        args = SimpleNamespace(
            run_id="unit",
            translation_capacity_report_sha256="1" * 64,
            translation_source_report_sha256="2" * 64,
            rotation_capacity_report_sha256="4" * 64,
            rotation_source_report_sha256="5" * 64,
        )
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with (
            mock.patch.object(sequence, "_validate_inputs", return_value=resolved),
            mock.patch.object(sequence.subprocess, "run", return_value=completed) as run,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(sequence._background(args), 0)
        command = run.call_args.args[0]
        self.assertIn(str(Path(sequence.__file__).resolve()), command)
        self.assertIn("--foreground", command)
        self.assertIn("--prepared-root", command)
        self.assertIn("--translation-capacity-report", command)
        self.assertIn("--rotation-capacity-report", command)
        self.assertNotIn("MJWarp_RL.sequential_train_hpc", command)

    def test_foreground_runs_exact_four_stage_order_and_four_archives(self) -> None:
        outcome = self._exercise_foreground(
            run_id="four-stage-success", returncodes=[0, 0, 0, 0]
        )
        self.assertEqual(outcome["result"], 0)
        self.assertIsNone(outcome["error"])
        calls = outcome["calls"]
        expected_labels = [item[0] for item in sequence.RUN_SPECS]
        self.assertEqual(
            [item[0] for item in calls["command"]], expected_labels
        )
        self.assertEqual(
            calls["run_log"],
            [f"{label}_training.log" for label in expected_labels],
        )
        self.assertEqual(calls["archive"], expected_labels)
        self.assertEqual(
            calls["admission"],
            [
                ("gripper3", sequence.TASK_TRANSLATE),
                ("gripper6", sequence.TASK_TRANSLATE),
                ("gripper3", sequence.TASK_ROTATE),
                ("gripper6", sequence.TASK_ROTATE),
            ],
        )
        status = outcome["status"]
        self.assertEqual(status["state"], "complete")
        self.assertEqual(set(status["runs"]), set(expected_labels))
        self.assertTrue(
            all(value["state"] == "complete" for value in status["runs"].values())
        )
        self.assertEqual(set(status["packages"]), set(expected_labels))
        self.assertEqual(
            set(status["task_pair_consistency"]), set(sequence.TASKS)
        )

    def test_second_stage_failure_prevents_rotation_runs(self) -> None:
        outcome = self._exercise_foreground(
            run_id="four-stage-failure", returncodes=[0, 9]
        )
        self.assertIsNone(outcome["result"])
        self.assertIsInstance(outcome["error"], sequence.SequenceFailure)
        calls = outcome["calls"]
        self.assertEqual(
            [item[0] for item in calls["command"]],
            ["g3_translation", "g6_translation"],
        )
        self.assertEqual(calls["archive"], ["g3_translation"])
        self.assertEqual(
            calls["validation"],
            [("g3_translation", "gripper3", sequence.TASK_TRANSLATE)],
        )
        status = outcome["status"]
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["runs"]["g3_translation"]["state"], "complete")
        self.assertEqual(status["runs"]["g6_translation"]["state"], "failed")
        self.assertEqual(status["runs"]["g3_rotation"]["state"], "admitted")
        self.assertEqual(status["runs"]["g6_rotation"]["state"], "admitted")

    def test_gpu_lock_is_exclusive_and_rejects_symlink(self) -> None:
        canonical_root = self.root.resolve()
        lock = canonical_root / "gpu0.lock"
        with sequence._exclusive_gpu_lock(lock):
            with self.assertRaisesRegex(sequence.SequenceFailure, "lock is held"):
                with sequence._exclusive_gpu_lock(lock):
                    self.fail("second lock unexpectedly succeeded")
        target = canonical_root / "target.lock"
        target.write_text("do not truncate", encoding="utf-8")
        link = canonical_root / "link.lock"
        link.symlink_to(target)
        with self.assertRaisesRegex(sequence.SequenceFailure, "safely open"):
            with sequence._exclusive_gpu_lock(link):
                self.fail("symlink lock unexpectedly succeeded")
        self.assertEqual(target.read_text(encoding="utf-8"), "do not truncate")

    def test_sigterm_exit_code_is_143(self) -> None:
        with mock.patch.object(
            sequence, "_check_only", side_effect=sequence.SequenceInterrupted(signal.SIGTERM)
        ):
            self.assertEqual(sequence.main(["--check-only"]), 143)

    def test_training_log_is_relayed_to_screen_stdout(self) -> None:
        owner = sequence.ChildOwner()
        log = self.root / "training.log"
        displayed = io.StringIO()
        with redirect_stdout(displayed):
            returncode = owner.run_log(
                [sys.executable, "-c", "print('live-success-rate=0.75')"],
                cwd=self.root,
                env=os.environ,
                log_path=log,
            )
        self.assertEqual(returncode, 0)
        self.assertEqual(log.read_text(encoding="utf-8"), "live-success-rate=0.75\n")
        self.assertEqual(displayed.getvalue(), "live-success-rate=0.75\n")

    def test_broken_screen_output_cannot_orphan_training_child(self) -> None:
        owner = sequence.ChildOwner()
        log = self.root / "broken-screen.log"
        program = (
            "import os,time; print(os.getpid(), flush=True); time.sleep(60)"
        )
        with mock.patch.object(
            sys.stdout, "write", side_effect=BrokenPipeError("screen detached badly")
        ):
            with self.assertRaises(BrokenPipeError):
                owner.run_log(
                    [sys.executable, "-c", program],
                    cwd=self.root,
                    env=os.environ,
                    log_path=log,
                )
        child_pid = int(log.read_text(encoding="utf-8").strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)


if __name__ == "__main__":
    unittest.main()
