# SPDX-License-Identifier: Apache-2.0
import pytest
from tower_eval.scoring.outcomes import outcome_fields


@pytest.mark.parametrize(
    ("status", "code", "outcome", "valid"),
    [
        ("task_success", 0, "Task Success", True),
        ("task_failure", 0, "Task Failure", True),
        ("duration_reached", 0, "Timeout Failure", True),
        ("step_limit_reached", 0, "Timeout Failure", True),
        ("safety_abort", 0, "Safety Abort", True),
        ("error", 0, "Invalid Trial", False),
        ("task_success", 1, "Invalid Trial", False),
        ("task_success", None, "Invalid Trial", False),
        ("interrupted", 0, "Invalid Trial", False),
        ("robot_stalled", 0, "Invalid Trial", False),
        ("not_started", None, None, False),
    ],
)
def test_outcomes_keep_policy_failures_and_exclude_system_errors(status, code, outcome, valid):
    result = outcome_fields(status, returncode=code)
    assert result["episode_result"] == outcome
    assert result["valid_trial"] == valid
    assert result["task_success"] == (outcome == "Task Success")
