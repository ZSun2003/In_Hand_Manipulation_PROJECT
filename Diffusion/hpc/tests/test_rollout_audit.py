from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from s_hpc.rollout_audit import (
    ExpertRolloutAudit,
    certify_expert_rollout,
)


def _result(
    *,
    worlds: int,
    failure=None,
    truncated=None,
    safety_stop=None,
    contact_violation=None,
    success=None,
    depth=None,
    twist=None,
):
    zeros = torch.zeros(worlds, dtype=torch.bool)
    info = {
        "failure": zeros.clone() if failure is None else failure,
        "truncated": (
            zeros.clone() if truncated is None else truncated
        ),
        "nonfinite": zeros.clone(),
        "fell_below_table": zeros.clone(),
        "lateral_escape": zeros.clone(),
        "safety_stop": (
            zeros.clone() if safety_stop is None else safety_stop
        ),
        "contact_constraint_violation": (
            zeros.clone()
            if contact_violation is None
            else contact_violation
        ),
        "success": (
            torch.ones(worlds, dtype=torch.bool)
            if success is None
            else success
        ),
        "depth": (
            torch.full((worlds,), 0.080)
            if depth is None
            else depth
        ),
        "twist": (
            torch.full((worlds,), 4.0 * torch.pi)
            if twist is None
            else twist
        ),
    }
    return SimpleNamespace(info=info)


def _env(worlds: int):
    task = SimpleNamespace(
        target_insertion_depth_m=0.080,
        target_rotation_rad=4.0 * torch.pi,
    )
    state = SimpleNamespace(
        task=task,
        thread_engaged=torch.ones(worlds, dtype=torch.bool),
    )
    return SimpleNamespace(state=state)


class ExpertRolloutAuditTest(unittest.TestCase):
    def test_reports_first_step_and_each_physical_reason(self):
        audit = ExpertRolloutAudit(
            torch, worlds=3, device=torch.device("cpu")
        )
        audit.update(_result(worlds=3), step_number=1)
        audit.update(
            _result(
                worlds=3,
                failure=torch.tensor((True, True, False)),
                safety_stop=torch.tensor((True, False, False)),
                contact_violation=torch.tensor(
                    (False, True, False)
                ),
            ),
            step_number=2,
        )
        audit.update(
            _result(
                worlds=3,
                failure=torch.tensor((True, False, False)),
                contact_violation=torch.tensor(
                    (True, False, False)
                ),
            ),
            step_number=4,
        )
        summary = audit.summary()
        self.assertEqual(summary["failure_worlds"], 2)
        self.assertEqual(summary["first_failure_step"], 2)
        self.assertEqual(
            summary["reasons"]["safety_stop"],
            {"worlds": 1, "first_step": 2},
        )
        self.assertEqual(
            summary["reasons"]["contact_constraint_violation"],
            {"worlds": 2, "first_step": 2},
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"physical failure in 2/3 worlds.*first_failure_step.*2",
        ):
            certify_expert_rollout(
                _env(3),
                _result(worlds=3),
                audit,
                context="test expert",
                require_terminal=False,
            )

    def test_partial_clean_rollout_does_not_require_terminal_success(self):
        audit = ExpertRolloutAudit(
            torch, worlds=2, device=torch.device("cpu")
        )
        result = _result(
            worlds=2,
            success=torch.zeros(2, dtype=torch.bool),
        )
        audit.update(result, step_number=1)
        summary = certify_expert_rollout(
            _env(2),
            result,
            audit,
            context="partial probe",
            require_terminal=False,
        )
        self.assertEqual(summary["failure_worlds"], 0)
        audit.assert_clean(context="clean prefix")

    def test_first_failure_context_reports_contact_subcause(self):
        audit = ExpertRolloutAudit(
            torch, worlds=2, device=torch.device("cpu")
        )
        result = _result(
            worlds=2,
            failure=torch.tensor((False, True)),
            contact_violation=torch.tensor((False, True)),
        )
        result.info.update(
            {
                "contact_violation_before_thread": torch.tensor(
                    (False, True)
                ),
                "task_phase": torch.tensor(
                    (4, 4), dtype=torch.long
                ),
                "fingertip_contact_force_n": torch.tensor(
                    ((0.2, 0.2, 0.2, 0.2), (0.0, 0.0, 0.01, 0.02))
                ),
                "holder_active": torch.tensor((False, False)),
                "shaft_captured": torch.tensor((True, True)),
                "thread_engaged": torch.tensor((False, False)),
                "expert_step": torch.tensor((806, 807)),
                "primary_pair_contact": torch.tensor((True, False)),
                "two_finger_contact": torch.tensor((True, False)),
                "thread_engagement_ready": torch.tensor(
                    (False, False)
                ),
                "thread_ready_stable_steps": torch.tensor((4, 0)),
                "before_capture_contact_loss_steps": torch.tensor(
                    (0, 0)
                ),
                "before_thread_two_finger_loss_steps": torch.tensor(
                    (0, 5)
                ),
                "thread_depth_error_m": torch.tensor((0.001, 0.003)),
                "thread_lateral_error_m": torch.tensor(
                    (0.0005, 0.002)
                ),
                "thread_axis_error_deg": torch.tensor((1.0, 4.0)),
                "thread_tip_speed_m_s": torch.tensor((0.01, 0.02)),
                "desired_fingertip_distance_m": torch.tensor(
                    (
                        (0.040, 0.040, 0.040, 0.040),
                        (0.040, 0.040, 0.040, 0.040),
                    )
                ),
            }
        )
        audit.update(result, step_number=700)
        summary = audit.summary()
        self.assertEqual(
            summary["reasons"]["contact_violation_before_thread"],
            {"worlds": 1, "first_step": 700},
        )
        context = summary["first_failure_context"]
        self.assertEqual(context["world_index"], 1)
        self.assertEqual(context["phase_name"], "INSERT")
        self.assertEqual(context["expert_step"], 807)
        self.assertFalse(context["primary_pair_contact"])
        self.assertFalse(context["two_finger_contact"])
        self.assertFalse(context["thread_engagement_ready"])
        self.assertEqual(context["thread_ready_stable_steps"], 0)
        self.assertEqual(
            context["before_thread_two_finger_loss_steps"], 5
        )
        self.assertAlmostEqual(context["thread_depth_error_m"], 0.003)
        self.assertAlmostEqual(context["thread_lateral_error_m"], 0.002)
        self.assertAlmostEqual(context["thread_axis_error_deg"], 4.0)
        self.assertAlmostEqual(context["thread_tip_speed_m_s"], 0.02)
        torch.testing.assert_close(
            torch.tensor(context["desired_fingertip_distance_m"]),
            torch.full((4,), 0.04),
        )
        with self.assertRaisesRegex(
            RuntimeError, "physical failure in 1/2"
        ):
            audit.assert_clean(context="dirty prefix")

    def test_full_rollout_requires_success_thread_depth_and_twist(self):
        audit = ExpertRolloutAudit(
            torch, worlds=3, device=torch.device("cpu")
        )
        result = _result(
            worlds=3,
            success=torch.tensor((True, False, True)),
            depth=torch.tensor((0.080, 0.080, 0.075)),
        )
        audit.update(result, step_number=3000)
        env = _env(3)
        env.state.thread_engaged[1] = False
        with self.assertRaisesRegex(
            RuntimeError, r"final-state certification failed in 2/3"
        ):
            certify_expert_rollout(
                env,
                result,
                audit,
                context="formal gate",
                require_terminal=True,
            )

    def test_canary_can_admit_exactly_half_percent_pure_truncations(self):
        worlds = 200
        audit = ExpertRolloutAudit(
            torch, worlds=worlds, device=torch.device("cpu")
        )
        truncated = torch.zeros(worlds, dtype=torch.bool)
        truncated[17] = True
        success = torch.ones(worlds, dtype=torch.bool)
        success[17] = False
        result = _result(
            worlds=worlds,
            truncated=truncated,
            success=success,
        )
        audit.update(result, step_number=4000)
        summary = certify_expert_rollout(
            _env(worlds),
            result,
            audit,
            context="expert canary",
            require_terminal=True,
            max_truncation_fraction=0.005,
        )
        self.assertEqual(summary["failure_worlds"], 0)
        self.assertEqual(summary["admitted_truncation_worlds"], 1)
        self.assertEqual(summary["terminal_certified_worlds"], 199)
        self.assertEqual(summary["terminal_failure_worlds"], 0)

    def test_canary_rejects_truncations_above_admitted_fraction(self):
        worlds = 200
        audit = ExpertRolloutAudit(
            torch, worlds=worlds, device=torch.device("cpu")
        )
        truncated = torch.zeros(worlds, dtype=torch.bool)
        truncated[:2] = True
        result = _result(worlds=worlds, truncated=truncated)
        audit.update(result, step_number=4000)
        with self.assertRaisesRegex(
            RuntimeError,
            r"truncated in 2/200.*maximum admitted fraction=0.005",
        ):
            certify_expert_rollout(
                _env(worlds),
                result,
                audit,
                context="expert canary",
                require_terminal=True,
                max_truncation_fraction=0.005,
            )

    def test_default_contract_still_rejects_every_truncation(self):
        audit = ExpertRolloutAudit(
            torch, worlds=2, device=torch.device("cpu")
        )
        result = _result(
            worlds=2,
            truncated=torch.tensor((False, True)),
        )
        audit.update(result, step_number=4000)
        with self.assertRaisesRegex(RuntimeError, "truncated in 1/2"):
            certify_expert_rollout(
                _env(2),
                result,
                audit,
                context="strict demonstration",
                require_terminal=True,
            )

    def test_canary_never_admits_physical_failure(self):
        worlds = 200
        audit = ExpertRolloutAudit(
            torch, worlds=worlds, device=torch.device("cpu")
        )
        failure = torch.zeros(worlds, dtype=torch.bool)
        truncated = torch.zeros(worlds, dtype=torch.bool)
        failure[17] = True
        truncated[17] = True
        result = _result(
            worlds=worlds,
            failure=failure,
            truncated=truncated,
            safety_stop=failure.clone(),
        )
        audit.update(result, step_number=4000)
        with self.assertRaisesRegex(RuntimeError, "physical failure in 1/200"):
            certify_expert_rollout(
                _env(worlds),
                result,
                audit,
                context="expert canary",
                require_terminal=True,
                max_truncation_fraction=0.005,
            )


if __name__ == "__main__":
    unittest.main()
