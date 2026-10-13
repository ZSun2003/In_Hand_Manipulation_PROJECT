"""Process and archive regressions retained for the r22 G6-only sequencer.

Obsolete r21 dual-gripper certificate/four-stage fixtures were removed;
formal r22 numeric contracts are tested by test_contract and test_single_phase_contract.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import signal
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from MJWarp_RL import sequential_train_hpc as sequence


class SequentialTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()


    def test_models_archive_excludes_certified_ik_demos(self) -> None:
        output = self.root / "g6_training"
        output.mkdir()
        (output / "final_diffusion_model.pt").write_bytes(b"model fixture")
        (output / "training_manifest.json").write_text("{}\n")
        for name in ("certified_ik_demos", "elite_demos"):
            demos = output / name
            demos.mkdir()
            (demos / "huge.bin").write_bytes(b"do not package")
        result = sequence.create_models_archive(
            output_dir=output,
            packages_dir=self.root / "packages",
            label="g6",
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
            project_root=Path("/release/rl16_api"),
            capacity_report=Path("/reports/g6.json"),
            output_dir=Path("/runs/g6"),
            task=sequence.TASK_TRANSLATE,
        )
        self.assertEqual(
            command,
            [
                "/release/rl16_api/MJWarp_RL/run_hpc.sh",
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
                "/reports/g6.json",
                "--output",
                "/runs/g6",
            ],
        )


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
