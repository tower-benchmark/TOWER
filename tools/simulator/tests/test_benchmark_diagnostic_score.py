# SPDX-License-Identifier: Apache-2.0
"""Incomplete or qualified trials must not enter the diagnostic-only scorer."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "physical_diagnostic", TOOLS / "benchmark_diagnostic_score.py"
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


@pytest.mark.parametrize(
    "status",
    [
        {
            "status": "infrastructure_failure",
            "valid_trial": False,
            "diagnostic_only": True,
            "evidence_complete": False,
        },
        {"status": "duration_reached", "valid_trial": True, "evidence_complete": True},
        {
            "status": "diagnostic_prefix_complete",
            "valid_trial": False,
            "diagnostic_only": True,
            "evidence_complete": False,
        },
    ],
)
def test_ineligible_input_is_rejected_without_rewriting_status(tmp_path, status):
    path = tmp_path / "status.json"
    encoded = json.dumps(status).encode()
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match="complete, explicitly unqualified"):
        diagnostic.score_physical_diagnostic(tmp_path, "task1-2")
    assert path.read_bytes() == encoded
    assert list(tmp_path.iterdir()) == [path]
