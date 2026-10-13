from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import job_status as job_status_module
from s_hpc.monitoring import (
    EpisodeTracker,
    RunMonitor,
    _source_fingerprints,
    atomic_write_json,
    heartbeat_age_seconds,
    read_json,
    request_graceful_stop,
    tail_jsonl,
)
from status_hpc import build_report, derive_alerts, parse_qstat, render_report
import system_monitor as system_monitor_module
from system_monitor import query_nvidia_smi


ROOT = Path(__file__).resolve().parents[1]


class Selectable(list):
    """Tiny tensor-like sequence supporting vector index selection."""

    def __getitem__(self, key):
        if isinstance(key, (list, tuple, Selectable)):
            return Selectable([list.__getitem__(self, int(index)) for index in key])
        return list.__getitem__(self, key)


class AtomicStatusTest(unittest.TestCase):
    def test_atomic_json_and_history_are_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "status.json"
            atomic_write_json(target, {"finite": float("nan"), "value": 3})
            self.assertEqual(read_json(target), {"finite": None, "value": 3})
            target.with_name("events.jsonl").write_text(
                '{"sequence":1}\n{"sequence":2}\n{"broken":',
                encoding="utf-8",
            )
            self.assertEqual(
                [item["sequence"] for item in tail_jsonl(
                    target.with_name("events.jsonl"), 2
                )],
                [1, 2],
            )

    def test_run_monitor_progress_throttle_and_terminal_state(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = RunMonitor(
                directory,
                pipeline_mode="full",
                config_name="test",
                interval_seconds=60.0,
            )
            monitor.set_stage(
                "residual_tqc",
                current=100,
                total=1000,
                unit="transition",
                message="training",
            )
            self.assertIsNone(
                monitor.heartbeat(current=101, force=False)
            )
            snapshot = monitor.heartbeat(current=500, force=True)
            self.assertEqual(snapshot["schema"], "screw-hpc-live-v1")
            self.assertEqual(snapshot["pipeline"]["stage_index"], 3)
            self.assertEqual(
                snapshot["stage"]["progress"]["fraction"], 0.5
            )
            monitor.finish("completed", message="done")
            live = read_json(Path(directory) / "status" / "live.json")
            self.assertEqual(live["state"], "completed")

    def test_diffusion_only_pipeline_has_no_residual_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = RunMonitor(
                directory,
                pipeline_mode="diffusion-full",
                config_name="test",
            )
            self.assertEqual(
                monitor.pipeline_order,
                ("demonstrations", "diffusion", "evaluation"),
            )
            monitor.set_stage(
                "evaluation",
                current=0,
                total=2000,
                unit="episode",
                message="evaluating",
            )
            snapshot = read_json(Path(directory) / "status" / "live.json")
            self.assertEqual(snapshot["pipeline"]["stage_index"], 3)
            self.assertEqual(snapshot["pipeline"]["stage_count"], 3)

    def test_failed_live_write_remains_throttled(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = RunMonitor(
                directory,
                pipeline_mode="full",
                config_name="test",
                interval_seconds=60.0,
            )
            with patch(
                "s_hpc.monitoring.atomic_write_json",
                side_effect=OSError("filesystem unavailable"),
            ):
                monitor.heartbeat(force=True)
            self.assertFalse(monitor.due)
            self.assertIn(
                "filesystem unavailable", monitor._last_write_error
            )

    def test_run_manifest_is_immutable_and_rejects_config_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "name": "unit",
                "runtime": {"seed": 17},
                "demos": {"seed": 23},
            }
            first = RunMonitor(
                directory,
                pipeline_mode="diffusion",
                config_name="unit",
                config=config,
            )
            manifest_path = Path(directory) / "run_manifest.json"
            original = manifest_path.read_bytes()
            second = RunMonitor(
                directory,
                pipeline_mode="diffusion",
                config_name="unit",
                config=config,
            )
            self.assertEqual(manifest_path.read_bytes(), original)
            self.assertEqual(
                first.manifest_sha256, second.manifest_sha256
            )
            with self.assertRaisesRegex(ValueError, "resume is refused"):
                RunMonitor(
                    directory,
                    pipeline_mode="diffusion",
                    config_name="unit",
                    config={
                        **config,
                        "runtime": {"seed": 18},
                    },
                )

    def test_environment_manifest_does_not_change_source_fingerprint(self):
        manifest = ROOT / "environment-manifest.txt"
        previous = manifest.read_bytes() if manifest.exists() else None
        try:
            manifest.write_text("created_utc=first\n", encoding="utf-8")
            first = _source_fingerprints()
            manifest.write_text("created_utc=second\n", encoding="utf-8")
            second = _source_fingerprints()
        finally:
            if previous is None:
                manifest.unlink(missing_ok=True)
            else:
                manifest.write_bytes(previous)
        self.assertEqual(
            first["source_tree_sha256"],
            second["source_tree_sha256"],
        )
        self.assertNotIn(
            "environment-manifest.txt",
            first["source_files"],
        )

    def test_wrapper_manifest_and_live_share_one_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            env = dict(os.environ)
            env["SCREW_ALLOCATION_ID"] = "autodl-test-allocation"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "job_status.py"),
                    "--run-dir",
                    str(run_dir),
                    "--allow-external-paths",
                    "--state",
                    "preflight",
                ],
                check=True,
                env=env,
            )
            with patch.dict(
                os.environ,
                {"SCREW_ALLOCATION_ID": "autodl-test-allocation"},
            ):
                monitor = RunMonitor(
                    run_dir,
                    pipeline_mode="full",
                    config_name="test",
                )
            job = read_json(run_dir / "status" / "job.json")
            live = read_json(run_dir / "status" / "live.json")
            manifest = read_json(run_dir / "run_manifest.json")
            self.assertEqual(job["run"]["id"], monitor.run_id)
            self.assertEqual(live["run"]["id"], monitor.run_id)
            self.assertEqual(manifest["run"]["id"], monitor.run_id)

    def test_stop_request_is_bound_to_one_session(self):
        with tempfile.TemporaryDirectory() as directory:
            first = RunMonitor(
                directory,
                pipeline_mode="full",
                config_name="test",
                interval_seconds=10.0,
            )
            request = request_graceful_stop(directory, reason="test")
            self.assertEqual(
                request["target_session_id"], first.session_id
            )
            self.assertTrue(first.stop_requested(poll_seconds=0.1))
            acknowledgement = read_json(
                Path(directory) / "status" / "stop-ack.json"
            )
            self.assertEqual(
                acknowledgement["target_session_id"], first.session_id
            )
            second = RunMonitor(
                directory,
                pipeline_mode="full",
                config_name="test",
                interval_seconds=10.0,
            )
            self.assertNotEqual(second.session_id, first.session_id)
            self.assertFalse(second.stop_requested(poll_seconds=0.1))

    def test_stop_request_rejects_previous_allocation_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = RunMonitor(
                directory,
                pipeline_mode="full",
                config_name="test",
                interval_seconds=10.0,
            )
            live = read_json(Path(directory) / "status" / "live.json")
            live["job"]["scheduler_job_id"] = "old.server"
            atomic_write_json(
                Path(directory) / "status" / "live.json", live
            )
            atomic_write_json(
                Path(directory) / "status" / "job.json",
                {
                    "state": "preflight",
                    "allocation": {
                        "scheduler_job_id": "new.server",
                    },
                },
            )
            with self.assertRaisesRegex(
                RuntimeError, "previous allocation"
            ):
                request_graceful_stop(directory)

    def test_stop_request_rejects_unidentified_old_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = RunMonitor(
                directory,
                pipeline_mode="full",
                config_name="test",
                interval_seconds=10.0,
            )
            live = read_json(Path(directory) / "status" / "live.json")
            self.assertIsNone(
                live["job"]["scheduler_job_id"],
                "unit-test heartbeat should be local",
            )
            atomic_write_json(
                Path(directory) / "status" / "job.json",
                {
                    "state": "preflight",
                    "allocation": {
                        "scheduler_job_id": "new.server",
                    },
                },
            )
            with self.assertRaisesRegex(
                RuntimeError, "previous allocation"
            ):
                request_graceful_stop(directory)

    def test_autodl_stop_rejects_a_previous_local_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {"SCREW_ALLOCATION_ID": "old-autodl-allocation"},
            ):
                RunMonitor(
                    directory,
                    pipeline_mode="full",
                    config_name="test",
                    interval_seconds=10.0,
                )
            live = read_json(Path(directory) / "status" / "live.json")
            self.assertEqual(
                live["job"]["allocation_id"],
                "old-autodl-allocation",
            )
            atomic_write_json(
                Path(directory) / "status" / "job.json",
                {
                    "state": "preflight",
                    "allocation": {
                        "id": "new-autodl-allocation",
                        "identity_source": "SCREW_ALLOCATION_ID",
                        "scheduler": "local",
                        "scheduler_job_id": None,
                    },
                },
            )
            with self.assertRaisesRegex(
                RuntimeError, "previous allocation"
            ):
                request_graceful_stop(directory)


class EpisodeTrackerTest(unittest.TestCase):
    def _info(self):
        return {
            "success": Selectable([True, False, False]),
            "failure": Selectable([False, True, False]),
            "truncated": Selectable([False, False, True]),
            "episode_return": Selectable([100.0, -20.0, 5.0]),
            "depth": Selectable([0.1, 0.02, 0.08]),
            "twist": Selectable([4.0 * 3.141592653589793, 0.0, 3.141592653589793]),
            "ik_failures": Selectable([0, 4, 2]),
            "ik_anchor_fallbacks": Selectable([1, 3, 2]),
            "ik_safety_stops": Selectable([0, 1, 0]),
            "nonfinite": Selectable([False, False, False]),
            "fell_below_table": Selectable([False, True, False]),
            "lateral_escape": Selectable([False, False, False]),
            "safety_stop": Selectable([False, True, False]),
        }

    def test_completed_episode_rates_and_checkpoint_roundtrip(self):
        tracker = EpisodeTracker(recent_episodes=2)
        tracker.add_from_info(
            self._info(),
            Selectable([0, 1, 2]),
            episode_lengths=Selectable([100, 200, 300]),
        )
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["completed"], 3)
        self.assertAlmostEqual(snapshot["success_rate"], 1.0 / 3.0)
        self.assertEqual(snapshot["recent"]["window"], 2)
        self.assertEqual(snapshot["recent"]["successes"], 0)
        self.assertEqual(
            snapshot["failure_reasons"]["fell_below_table"], 1
        )
        restored = EpisodeTracker(recent_episodes=2)
        restored.load_state_dict(tracker.state_dict())
        self.assertEqual(restored.snapshot(), snapshot)

    def test_zero_completed_rate_is_none_not_zero(self):
        snapshot = EpisodeTracker().snapshot()
        self.assertIsNone(snapshot["success_rate"])
        self.assertIsNone(snapshot["recent"]["success_rate"])
        self.assertEqual(
            tuple(snapshot["windows"]), ("128", "512", "2048")
        )

    def test_multi_windows_curriculum_phase_and_jsonl_event(self):
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "episodes.jsonl"
            tracker = EpisodeTracker(event_path=event_path)
            info = self._info()
            info.update(
                {
                    "terminal_phase": Selectable([14, 4, 5]),
                    "curriculum_progress": Selectable([0.01, 0.4, 0.9]),
                    "spawn_radius_m": Selectable([0.0, 0.05, 0.1]),
                    "thread_engaged": Selectable([True, False, False]),
                    "holder_active": Selectable([True, False, True]),
                    "grasp_latched": Selectable([True, False, True]),
                    "diffusion_handoff_ready": Selectable(
                        [True, False, True]
                    ),
                    "diffusion_handoff_hold_steps": Selectable([5, 3, 5]),
                    "world_id": Selectable([4, 5, 6]),
                    "socket_angle_deg": Selectable([45.0, 45.0, 45.0]),
                }
            )
            tracker.add_from_info(
                info,
                Selectable([0, 1, 2]),
                episode_lengths=Selectable([100, 200, 300]),
                event_context={"transitions": 123},
            )
            snapshot = tracker.snapshot()
            self.assertEqual(snapshot["windows"]["128"]["window"], 3)
            interval = snapshot["windows"]["128"][
                "success_rate_ci95"
            ]
            self.assertLess(interval["lower"], 1.0 / 3.0)
            self.assertGreater(interval["upper"], 1.0 / 3.0)
            self.assertEqual(
                snapshot["curriculum"]["warmup"]["completed"], 1
            )
            self.assertEqual(snapshot["curriculum"]["ramp"]["completed"], 1)
            self.assertEqual(snapshot["curriculum"]["full"]["completed"], 1)
            self.assertEqual(
                snapshot["terminal_phases"]["insert"]["completed"], 1
            )
            events = tail_jsonl(event_path, 1)
            self.assertEqual(events[0]["schema"], "screw-hpc-episodes-v1")
            self.assertEqual(events[0]["terminal_count"], 3)
            self.assertEqual(
                events[0]["context"]["transitions"], 123
            )
            self.assertEqual(
                events[0]["episodes"][1]["failure_reasons"],
                ["fell_below_table", "safety_stop", "grasp_lost", "ik_exhausted"],
            )

    def test_terminal_outcomes_must_be_mutually_exclusive(self):
        info = self._info()
        info["failure"][0] = True
        tracker = EpisodeTracker()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            tracker.add_from_info(
                info,
                Selectable([0, 1, 2]),
                episode_lengths=Selectable([100, 200, 300]),
            )
        self.assertEqual(tracker.snapshot()["completed"], 0)


class StatusReaderTest(unittest.TestCase):
    def test_gpu_telemetry_filters_to_allocated_cuda_device(self):
        output = "\n".join(
            (
                "0, A100, GPU-zero, 550.0, P0, 99, 10, 70000, "
                "80000, 90, 300, 400, 1200",
                "1, A100, GPU-one, 550.0, P0, 40, 5, 10000, "
                "80000, 50, 150, 400, 1100",
            )
        )
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout=output,
            stderr="",
        )
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "1"},
            clear=False,
        ), patch("system_monitor.subprocess.run", return_value=completed):
            snapshot = query_nvidia_smi()
        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["selection"]["mode"], "visibility_filter")
        self.assertEqual(
            [device["index"] for device in snapshot["devices"]],
            [1],
        )
        self.assertTrue(snapshot["devices"][0]["allocated"])

    def test_stdlib_sidecar_and_job_writer_run_without_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            common = [
                "--run-dir",
                directory,
                "--allow-external-paths",
            ]
            first_env = dict(os.environ)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
                "SLURM_SUBMIT_DIR",
            ):
                first_env.pop(name, None)
            first_env.update(
                {
                    "PBS_JOBID": "100.server",
                    "PBS_JOBNAME": "telemetry-test",
                    "PBS_QUEUE": "test",
                }
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "job_status.py"),
                    *common,
                    "--state",
                    "preflight",
                ],
                check=True,
                env=first_env,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "system_monitor.py"),
                    *common,
                    "--once",
                ],
                check=True,
                env=first_env,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "job_status.py"),
                    *common,
                    "--state",
                    "running",
                ],
                check=True,
                env=first_env,
            )
            second_env = dict(first_env)
            second_env["PBS_JOBID"] = "101.server"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "job_status.py"),
                    *common,
                    "--state",
                    "starting",
                ],
                check=True,
                env=second_env,
            )
            job = read_json(Path(directory) / "status" / "job.json")
            system = read_json(Path(directory) / "status" / "system.json")
            self.assertEqual(job["state"], "starting")
            self.assertEqual(system["schema"], "screw-hpc-system-v1")
            self.assertIn("gpu", system)
            status_dir = Path(directory) / "status"
            job_lines = (
                status_dir / "job-history.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            system_lines = (
                status_dir / "system-history.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            job_history = [json.loads(line) for line in job_lines]
            system_history = [json.loads(line) for line in system_lines]
            self.assertEqual(
                [event["sequence"] for event in job_history],
                [1, 2, 3],
            )
            self.assertEqual(
                [event["state"] for event in job_history],
                ["preflight", "running", "starting"],
            )
            self.assertEqual(
                job_history[0]["allocation"]["id"],
                job_history[1]["allocation"]["id"],
            )
            self.assertNotEqual(
                job_history[1]["allocation"]["id"],
                job_history[2]["allocation"]["id"],
            )
            self.assertEqual(
                len(
                    {
                        event["identifiers"]["run_id"]
                        for event in job_history
                    }
                ),
                1,
            )
            self.assertEqual(len(system_history), 1)
            identifiers = system_history[0]["identifiers"]
            self.assertEqual(
                identifiers["run_id"],
                job_history[0]["identifiers"]["run_id"],
            )
            self.assertEqual(
                identifiers["allocation_id"],
                job_history[0]["identifiers"]["allocation_id"],
            )
            self.assertTrue(identifiers["session_id"])
            self.assertTrue(identifiers["sidecar_session_id"])
            self.assertTrue(
                system_history[0]["updated_utc"].endswith("+00:00")
            )
            for line in (*job_lines, *system_lines):
                self.assertNotIn("NaN", line)
                self.assertNotIn("Infinity", line)

    def test_history_failure_does_not_prevent_atomic_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory) / "status"
            job_target = status / "job.json"
            job_payload = {
                "schema": "screw-hpc-job-v1",
                "sequence": 1,
                "state": "running",
            }
            with patch.object(
                job_status_module,
                "append_strict_jsonl",
                side_effect=OSError("history unavailable"),
            ):
                job_status_module.persist_job_status(
                    job_target,
                    job_payload,
                )
            self.assertEqual(read_json(job_target), job_payload)

            system_target = status / "system.json"
            system_payload = {
                "schema": "screw-hpc-system-v1",
                "sequence": 1,
                "state": "running",
            }
            with patch.object(
                system_monitor_module,
                "append_strict_jsonl",
                side_effect=OSError("history unavailable"),
            ):
                system_monitor_module.persist_system_sample(
                    system_target,
                    status / "system-history.jsonl",
                    system_payload,
                    append_history=True,
                )
            self.assertEqual(read_json(system_target), system_payload)

    def test_qstat_parser_and_stale_critical_alert(self):
        parsed = parse_qstat(
            """Job Id: 123.server
    Job_Name = screw
    job_state = R
    Resource_List.walltime = 71:50:00
    resources_used.walltime = 01:02:03
"""
        )
        self.assertEqual(parsed["job_state"], "R")
        self.assertEqual(parsed["resources_used.walltime"], "01:02:03")
        stale = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat()
        app = {
            "state": "running",
            "updated_utc": stale,
            "monitor": {"stale_after_seconds": 60},
            "stage": {"name": "residual_tqc"},
            "metrics": {},
            "alerts": [],
        }
        alerts, age, _ = derive_alerts(app, {})
        self.assertGreater(age, 60)
        self.assertIn(
            "app_heartbeat_stale",
            {item["code"] for item in alerts},
        )

    def test_unattributed_physical_gpu_does_not_trigger_memory_alarm(self):
        app = {
            "state": "running",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "monitor": {"stale_after_seconds": 60},
            "session": {"elapsed_seconds": 1},
            "stage": {"name": "residual_tqc"},
            "metrics": {},
            "alerts": [],
        }
        system = {
            "state": "running",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "interval_seconds": 15,
            "gpu": {
                "available": True,
                "selection": {"mode": "allocation_unknown"},
                "devices": [
                    {
                        "allocated": None,
                        "memory_used_mib": 79_500,
                        "memory_total_mib": 80_000,
                        "temperature_c": 90,
                    }
                ],
            },
        }
        alerts, _, _ = derive_alerts(app, system)
        codes = {item["code"] for item in alerts}
        self.assertIn("gpu_allocation_unknown", codes)
        self.assertNotIn("gpu_memory_critical", codes)
        self.assertNotIn("gpu_temperature_high", codes)

    def test_new_allocation_does_not_show_previous_app_as_current(self):
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory) / "status"
            atomic_write_json(
                status / "live.json",
                {
                    "state": "interrupted",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "job": {"scheduler_job_id": "old.server"},
                },
            )
            atomic_write_json(
                status / "job.json",
                {
                    "state": "preflight",
                    "allocation": {
                        "scheduler_job_id": "new.server",
                        "scheduler": "pbs",
                    },
                },
            )
            report, code = build_report(
                Path(directory),
                query_scheduler=False,
                stale_seconds=None,
                history_count=0,
            )
            self.assertEqual(code, 0)
            self.assertEqual(report["state"], "awaiting_application")
            self.assertEqual(report["previous_app"]["state"], "interrupted")

    def test_new_allocation_hides_local_heartbeat_without_job_id(self):
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory) / "status"
            atomic_write_json(
                status / "live.json",
                {
                    "state": "running",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "job": {"scheduler": "local", "scheduler_job_id": None},
                },
            )
            atomic_write_json(
                status / "job.json",
                {
                    "state": "preflight",
                    "allocation": {
                        "scheduler_job_id": "new.server",
                        "scheduler": "pbs",
                    },
                },
            )
            report, code = build_report(
                Path(directory),
                query_scheduler=False,
                stale_seconds=None,
                history_count=0,
            )
            self.assertEqual(code, 0)
            self.assertEqual(report["state"], "awaiting_application")
            self.assertEqual(report["previous_app"]["state"], "running")

    def test_training_alert_needs_minimum_recent_sample(self):
        base = {
            "state": "running",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "monitor": {"stale_after_seconds": 60},
            "stage": {"name": "residual_tqc"},
            "alerts": [],
        }
        for count, expected in ((511, False), (512, True)):
            app = json.loads(json.dumps(base))
            app["metrics"] = {
                "episodes": {
                    "recent": {
                        "window": count,
                        "success_rate": 0.0,
                        "failure_rate": 0.0,
                        "truncation_rate": 1.0,
                    }
                }
            }
            alerts, _, _ = derive_alerts(app, {})
            codes = {item["code"] for item in alerts}
            self.assertEqual(
                "no_recent_training_success" in codes, expected
            )

    def test_socket_orbit_deviation_is_a_critical_alert(self):
        app = {
            "state": "running",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "monitor": {"stale_after_seconds": 60},
            "stage": {"name": "demonstrations"},
            "metrics": {
                "rollout": {
                    "socket_mouth_radius_min_m": 0.020,
                    "socket_mouth_radius_max_m": 0.0205,
                }
            },
            "alerts": [],
        }
        alerts, _, _ = derive_alerts(app, {})
        by_code = {item["code"]: item for item in alerts}
        self.assertEqual(
            by_code["socket_mouth_orbit_invalid"]["severity"], "critical"
        )

    def test_contact_constraint_violation_is_a_warning(self):
        app = {
            "state": "running",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "monitor": {"stale_after_seconds": 60},
            "stage": {"name": "residual_tqc"},
            "metrics": {
                "rollout": {
                    "contact_constraint_violation_fraction": 0.01,
                }
            },
            "alerts": [],
        }
        alerts, _, _ = derive_alerts(app, {})
        by_code = {item["code"]: item for item in alerts}
        self.assertEqual(
            by_code["contact_constraint_violation"]["severity"],
            "warning",
        )

    def test_no_status_renderer_is_actionable(self):
        rendered = render_report(
            {
                "state": "awaiting_application",
                "run_dir": "/tmp/run",
                "message": "preflight",
                "job_status": {
                    "state": "preflight",
                    "message": "checking",
                    "allocation": {
                        "scheduler": "pbs",
                        "scheduler_job_id": "1.server",
                        "hostname": "node",
                    },
                },
                "system": {},
                "existing_artifacts": {},
            }
        )
        self.assertIn("等待 pipeline 心跳", rendered)
        self.assertIn("1.server", rendered)

    def test_full_renderer_handles_training_and_system_metrics(self):
        now = datetime.now(timezone.utc).isoformat()
        report = {
            "run_dir": "/tmp/run",
            "app": {
                "state": "running",
                "updated_utc": now,
                "job": {
                    "scheduler": "pbs",
                    "scheduler_job_id": "2.server",
                    "hostname": "gpu-node",
                    "pid": 42,
                },
                "pipeline": {"stage_index": 3, "stage_count": 4},
                "stage": {
                    "name": "residual_tqc",
                    "message": "training",
                    "progress": {
                        "current": 1000,
                        "total": 5000,
                        "unit": "transition",
                        "fraction": 0.2,
                        "rate_per_second": 100,
                        "eta_seconds": 40,
                        "elapsed_seconds": 10,
                    },
                },
                "metrics": {
                    "episodes": {
                        "scope": "stochastic_training_rollouts",
                        "completed": 10,
                        "successes": 2,
                        "failures": 3,
                        "truncations": 5,
                        "success_rate": 0.2,
                        "recent_capacity": 2048,
                        "recent": {
                            "window": 10,
                            "success_rate": 0.2,
                            "failure_rate": 0.3,
                            "truncation_rate": 0.5,
                            "mean_return": 1,
                            "mean_depth_m": 0.1,
                            "mean_turns": 2,
                            "mean_episode_steps": 100,
                            "failure_reasons": {"safety_stop": 1},
                        },
                    },
                    "tqc": {
                        "transitions": 1000,
                        "total_transitions": 5000,
                        "updates": 4,
                        "mode": "learning",
                        "replay_size": 1000,
                        "replay_capacity": 2000,
                        "replay_fill_fraction": 0.5,
                    },
                    "rollout": {
                        "reward_mean": 1,
                        "depth_mean_m": 0.05,
                        "depth_max_m": 0.1,
                        "turns_mean": 1,
                        "turns_max": 2,
                        "lateral_error_mean_m": 0.001,
                        "lateral_error_max_m": 0.002,
                        "axis_error_mean_rad": 0.01,
                        "axis_error_max_rad": 0.02,
                        "phase_histogram": {"power": 8},
                    },
                },
                "health": {
                    "finite": True,
                    "overflow_worlds": 0,
                    "contacts_total": 100,
                    "contact_capacity_total": 1000,
                    "max_nefc": 10,
                    "njmax": 100,
                },
                "artifacts": {},
                "stop": {},
            },
            "system": {
                "gpu": {
                    "devices": [
                        {
                            "index": 0,
                            "name": "A100",
                            "gpu_utilization_percent": 90,
                            "memory_used_mib": 1000,
                            "memory_total_mib": 80000,
                        }
                    ]
                },
                "application": {"process": {"alive": True, "io": {}}},
                "node": {
                    "load_average": {"1m": 1, "5m": 1, "15m": 1},
                    "run_disk": {
                        "free_bytes": 100 * 1024**3,
                        "free_percent": 50,
                    },
                    "memory": {},
                },
            },
            "job_status": {},
            "scheduler": {"available": False},
            "derived": {
                "app_heartbeat_age_seconds": 1,
                "alerts": [],
                "severity": "ok",
            },
            "history": [],
        }
        rendered = render_report(report)
        self.assertIn("TQC residual", rendered)
        self.assertIn("A100", rendered)
        self.assertIn("20.00%", rendered)


if __name__ == "__main__":
    unittest.main()
