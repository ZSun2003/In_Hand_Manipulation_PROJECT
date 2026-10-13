from __future__ import annotations

import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from collect_diagnostics import (
    collect_diagnostics,
    environment_snapshot,
)


class DiagnosticsBundleTest(unittest.TestCase):
    def test_bundle_preserves_source_status_and_log_but_not_checkpoint_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs" / "task.json").write_text(
                '{"worlds": 128}\n', encoding="utf-8"
            )
            (root / "worker.py").write_text(
                "print('exact source')\n", encoding="utf-8"
            )
            (root / "assets").mkdir()
            (root / "assets" / "scene.xml").write_text(
                "<mujoco/>\n", encoding="utf-8"
            )
            (root / "assets" / "mesh.STL").write_bytes(b"binary-mesh")
            (root / "RL_env").mkdir()
            (root / "RL_env" / "secret.py").write_text(
                "must not be archived\n", encoding="utf-8"
            )

            run_dir = root / "runs" / "example"
            (run_dir / "status").mkdir(parents=True)
            (run_dir / "status" / "job.json").write_text(
                '{"state": "failed"}\n', encoding="utf-8"
            )
            (run_dir / "logs").mkdir()
            (run_dir / "logs" / "episodes.jsonl").write_text(
                '{"outcome": "failure"}\n', encoding="utf-8"
            )
            (run_dir / "checkpoints").mkdir()
            (run_dir / "checkpoints" / "large.pt").write_bytes(
                b"checkpoint-payload"
            )
            allocation_log = root / "logs" / "allocation-123.log"
            allocation_log.parent.mkdir(exist_ok=True)
            allocation_log.write_text(
                "Traceback: libffi.so.8 missing\n", encoding="utf-8"
            )

            archive = collect_diagnostics(
                source_root=root,
                run_dir=run_dir,
                exit_code=1,
                job_id="123.pbs-7",
                allocation_log=allocation_log,
            )

            self.assertTrue(archive.is_file())
            pointer = json.loads(
                (run_dir / "diagnostics" / "latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(pointer["exit_code"], 1)
            self.assertEqual(pointer["job_id"], "123.pbs-7")

            with tarfile.open(archive, "r:gz") as stream:
                names = stream.getnames()
                summary_name = next(
                    name for name in names if name.endswith("/SUMMARY.json")
                )
                summary = json.load(stream.extractfile(summary_name))
                source_inventory = json.load(
                    stream.extractfile(
                        next(
                            name
                            for name in names
                            if name.endswith("/source_inventory.json")
                        )
                    )
                )

            self.assertEqual(summary["exit_code"], 1)
            self.assertTrue(
                any(name.endswith("/source_snapshot/worker.py") for name in names)
            )
            self.assertTrue(
                any(
                    name.endswith("/source_snapshot/assets/scene.xml")
                    for name in names
                )
            )
            self.assertFalse(any(name.endswith("/assets/mesh.STL") for name in names))
            mesh_record = next(
                item
                for item in source_inventory["files"]
                if item["path"] == "assets/mesh.STL"
            )
            self.assertEqual(mesh_record["copy_mode"], "inventory_only")
            self.assertEqual(len(mesh_record["sha256"]), 64)
            self.assertTrue(
                any(
                    name.endswith("/run_artifacts/status/job.json")
                    for name in names
                )
            )
            self.assertTrue(
                any(name.endswith("/allocation.log") for name in names)
            )
            self.assertFalse(any("large.pt" in name for name in names))
            self.assertFalse(any("RL_env/secret.py" in name for name in names))

    def test_environment_snapshot_uses_an_allowlist(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "GPU-example",
                "UNRELATED_API_TOKEN": "must-not-be-recorded",
            },
            clear=False,
        ):
            snapshot = environment_snapshot(Path(directory))
        self.assertEqual(
            snapshot["environment"]["CUDA_VISIBLE_DEVICES"], "GPU-example"
        )
        self.assertNotIn("UNRELATED_API_TOKEN", snapshot["environment"])
        self.assertIn(snapshot["ctypes_libffi_check"]["status"], {"pass", "fail"})


if __name__ == "__main__":
    unittest.main()
