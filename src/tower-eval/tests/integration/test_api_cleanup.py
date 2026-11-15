# SPDX-License-Identifier: Apache-2.0
"""Public API, runner compatibility and process-level import guarantees."""

import json
import subprocess
import sys

import tower_eval
from tower_eval import cli
from tower_eval.evaluation.portable import evaluate_policy
from tower_eval.runner.episode import run_episode


def test_public_names_resolve_without_native_imports():
    code = """
import sys
import tower_eval
assert 'tower_eval.envs' not in sys.modules
assert 'tower_eval.simulator' not in sys.modules
assert 'importlib.metadata' not in sys.modules
for name in tower_eval.__all__:
    assert getattr(tower_eval, name) is not None
assert not {'isaacsim', 'omni', 'carb', 'torch', 'pinocchio', 'tower_sim'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True)
    assert set(tower_eval.__all__) == {
        "PolicyAdapter",
        "ActionChunk",
        "TowerEnv",
        "SimulatorBackend",
        "MeasuredState",
        "RobotSpec",
        "GripperSpec",
        "DualModeController",
        "load_config",
        "score_run",
        "StackScorer",
        "BenchmarkMetrics",
        "EpisodeRecord",
        "__version__",
    }


def test_single_episode_runner():
    assert evaluate_policy is run_episode


def test_run_alias_and_provenance_version(tmp_path, capsys):
    config = tmp_path / "run.toml"
    config.write_text('episodes=1\nduration_seconds=0.2\noutput="run"\n')
    assert cli.main(["run", "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    episode = json.loads((tmp_path / "run" / report["episode_reports"][0]).read_text())
    assert episode["steps"] == 2
    assert report["provenance"]["evaluator_version"] == tower_eval.__version__
    assert episode["evaluator_version"] == tower_eval.__version__
