from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from MJWarp_RL.live_metrics import (
    build_metrics,
    read_json_object,
    read_jsonl,
)
from MJWarp_RL.status_hpc import (
    build_report,
    query_gpu_memory,
    render_report,
)


class LiveMetricsTests(unittest.TestCase):
    def test_jsonl_reader_tolerates_corrupt_and_incomplete_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            path.write_bytes(
                b'{"total_steps": 100, "endpoint_success_rate": 0.25}\n'
                b'not-json\n'
                b'["not", "an", "object"]\n'
                b'{"total_steps": 200'
            )
            result = read_jsonl(path)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.invalid_lines, 2)
        self.assertTrue(result.ignored_partial_line)
        self.assertIsNone(result.error)

    def test_missing_files_and_invalid_status_do_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = read_jsonl(root / "missing.jsonl")
            self.assertEqual(result.records, ())
            (root / "run_status.json").write_text("{bad", encoding="utf-8")
            status, error = read_json_object(root / "run_status.json")
        self.assertEqual(status, {})
        self.assertIsNotNone(error)

    def test_metrics_include_weighted_success_groups_and_eta(self) -> None:
        records = [
            {
                "elapsed_s": 10.0,
                "total_steps": 1_000,
                "endpoint_success_count": 2,
                "endpoint_episode_count": 4,
                "object": "banana",
                "task": "translate",
                "success_by_rotation_target_angle_deg": {
                    "30": {"success_count": 1, "episode_count": 2}
                },
                "grasp_plane_fraction": [0.4, 0.5, 0.61],
                "position_error_m": 0.03,
                "euler_l2_error_deg": 12.0,
                "translation_endpoint_desired_final_displacement_l2_mean_m": 0.12,
                "translation_endpoint_actual_final_displacement_l2_mean_m": 0.11,
                "translation_endpoint_final_displacement_error_l2_mean_m": 0.01,
                "translation_endpoint_final_euler_error_l2_mean_deg": 12.0,
                "translation_endpoint_path_tracking_rmse_mean_m": 0.02,
                "translation_endpoint_path_tracking_max_mean_m": 0.04,
                "stage": "diffusion",
                "loss": 0.4,
                "diffusion_lr": 1e-4,
                "certified_trajectory_count": 3,
            },
            {
                "elapsed_s": 20.0,
                "total_steps": 3_000,
                "endpoint_success_count": 6,
                "endpoint_episode_count": 6,
                "object": "pear",
                "task": "rotate",
                "success_by_rotation_target_angle_deg": {
                    "30": {"success_count": 2, "episode_count": 2},
                    "45": {"success_count": 1, "episode_count": 2},
                },
                "phase_cumulative_success_by_rotation_target_angle_deg": {
                    "30": {"success_count": 3, "episode_count": 4},
                    "45": {"success_count": 1, "episode_count": 2},
                },
                "grasp_plane_compliance_rate": 1.0,
                "grasp_plane_sample_count": 6,
                "position_error_m": 0.01,
                "euler_l2_error_deg": 8.0,
                "translation_endpoint_desired_final_displacement_l2_mean_m": 0.18,
                "translation_endpoint_actual_final_displacement_l2_mean_m": 0.175,
                "translation_endpoint_final_displacement_error_l2_mean_m": 0.005,
                "translation_endpoint_final_euler_error_l2_mean_deg": 7.0,
                "translation_endpoint_path_tracking_rmse_mean_m": 0.01,
                "translation_endpoint_path_tracking_max_mean_m": 0.03,
                "diffusion_loss": 0.2,
                "diffusion_lr": 5e-5,
                "certified_trajectory_count": 7,
            },
        ]
        metrics = build_metrics(
            records,
            run_status={"target_steps": 5_000},
            window=2,
        )
        self.assertAlmostEqual(metrics["success"]["cumulative"], 0.8)
        self.assertEqual(metrics["success"]["weighting"], "episode")
        self.assertAlmostEqual(
            metrics["success_by_object"]["banana"]["success_rate"], 0.5
        )
        self.assertAlmostEqual(
            metrics["success_by_task"]["rotate"]["success_rate"], 1.0
        )
        self.assertAlmostEqual(
            metrics["success_by_rotation_target_angle_deg"]["30"][
                "success_rate"
            ],
            0.75,
        )
        self.assertAlmostEqual(
            metrics["success_by_rotation_target_angle_deg"]["45"][
                "success_rate"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            metrics["grasp_plane_compliance"]["cumulative"], 8.0 / 9.0
        )
        self.assertAlmostEqual(
            metrics["position_error_m"]["recent_mean"], 0.02
        )
        self.assertAlmostEqual(
            metrics["translation_path_tracking_rmse_m"]["recent_mean"],
            0.015,
        )
        self.assertAlmostEqual(
            metrics["translation_desired_final_displacement_m"]["recent_mean"],
            0.15,
        )
        self.assertAlmostEqual(
            metrics["translation_actual_final_displacement_m"]["recent_mean"],
            0.1425,
        )
        self.assertAlmostEqual(
            metrics["translation_final_displacement_error_m"]["recent_mean"],
            0.0075,
        )
        self.assertAlmostEqual(
            metrics["translation_final_euler_error_deg"]["recent_mean"],
            9.5,
        )
        self.assertAlmostEqual(
            metrics["translation_path_tracking_max_error_m"]["recent_mean"],
            0.035,
        )
        self.assertAlmostEqual(
            metrics["diffusion_loss"]["recent_mean"], 0.3
        )
        self.assertEqual(
            metrics["training"]["certified_trajectory_count"], 7
        )
        self.assertNotIn("demos_written", metrics["training"])
        self.assertAlmostEqual(
            metrics["progress"]["recent_steps_per_s"], 200.0
        )
        self.assertAlmostEqual(metrics["progress"]["eta_s"], 10.0)

    def test_old_tqc_log_schema_is_accepted_without_fake_metrics(self) -> None:
        metrics = build_metrics(
            [
                {
                    "elapsed_s": 100.0,
                    "total_steps": 50_000,
                    "phase": "phase1",
                    "endpoint_success_rate": None,
                    "position_error_m": 0.02,
                    "euler_l2_error_deg": 18.0,
                    "update": {"critic_loss": 0.3},
                }
            ],
            window=20,
        )
        self.assertIsNone(metrics["success"]["latest"])
        self.assertIsNone(metrics["grasp_plane_compliance"]["latest"])
        self.assertIsNone(metrics["diffusion_loss"]["latest"])
        self.assertEqual(metrics["progress"]["phase"], "phase1")
        self.assertEqual(metrics["position_error_m"]["latest"], 0.02)

    def test_nested_plane_compliance_tensor_is_reduced(self) -> None:
        metrics = build_metrics(
            [
                {
                    "grasp_plane_compliant": [
                        [True, False],
                        [True, True],
                    ]
                }
            ]
        )
        self.assertAlmostEqual(
            metrics["grasp_plane_compliance"]["latest"], 0.75
        )

    @mock.patch("MJWarp_RL.status_hpc.subprocess.run")
    def test_gpu_query_parses_csv_and_failure_is_nonfatal(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="0, NVIDIA RTX PRO 6000, 4096, 97887, 75\n",
            stderr="",
        )
        gpu = query_gpu_memory()
        self.assertTrue(gpu["available"])
        self.assertEqual(gpu["devices"][0]["memory_used_mib"], 4096.0)

        run.side_effect = FileNotFoundError("nvidia-smi")
        unavailable = query_gpu_memory()
        self.assertFalse(unavailable["available"])
        self.assertIn("FileNotFoundError", unavailable["reason"])

    def test_report_renders_when_fields_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training.jsonl").write_text(
                json.dumps(
                    {
                        "elapsed_s": 2.0,
                        "total_steps": 1_000,
                        "endpoint_success_rate": 0.25,
                        "translation_endpoint_desired_final_displacement_l2_mean_m": 0.12,
                        "translation_endpoint_actual_final_displacement_l2_mean_m": 0.11,
                        "translation_endpoint_final_displacement_error_l2_mean_m": 0.01,
                        "translation_endpoint_final_euler_error_l2_mean_deg": 18.0,
                        "translation_endpoint_path_tracking_rmse_mean_m": 0.004,
                        "translation_endpoint_path_tracking_max_mean_m": 0.009,
                        "certified_trajectory_count": 12,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = build_report(
                root, window=5, total_steps=10_000, include_gpu=False
            )
        rendered = render_report(report)
        self.assertIn("成功率", rendered)
        self.assertIn("25.00%", rendered)
        self.assertIn("实际hard-tip平面[40%,60%]", rendered)
        self.assertIn("Translation路径跟踪", rendered)
        self.assertIn("Translation端点", rendered)
        self.assertIn("目标位移=120 mm", rendered)
        self.assertIn("RMSE recent=4 mm", rendered)
        self.assertIn("certified_trajectories=12", rendered)
        self.assertIn("每组至少一条通过认证的IK轨迹", rendered)
        self.assertNotIn("elite_success", rendered.lower())
        self.assertNotIn("demos=", rendered.lower())
        self.assertIn("GPU: N/A", rendered)


if __name__ == "__main__":
    unittest.main()
