"""The public diagnostic rejects retired execution paths."""

import importlib

import pytest


def test_no_frozen_diagnostic_execution_path():
    diagnostic = importlib.import_module("benchmark_diagnostic")
    assert not hasattr(diagnostic, "run_worker")
    assert not hasattr(diagnostic, "collect_result")
    scorer = importlib.import_module("benchmark_diagnostic_score")
    with pytest.raises(RuntimeError, match="private archive"):
        scorer.score_physical_diagnostic("unused", "task1-2", reference=True)
