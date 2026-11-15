# SPDX-License-Identifier: Apache-2.0
import json
from contextlib import contextmanager

import pytest
from tower_eval import __version__, cli
from tower_eval.envs import TowerEnv
from tower_eval.evaluation.portable import evaluate_policy
from tower_eval.policies import ReferenceAdapter
from tower_eval.simulator import CpuReferenceBackend


class TrackingPolicy(ReferenceAdapter):
    def __init__(self):
        super().__init__([0.0] * 16)
        self.seeds = []
        self.closes = 0

    def reset(self, *, seed=None, metadata=None):
        self.seeds.append(seed)
        super().reset(seed=seed, metadata=metadata)

    def close(self):
        self.closes += 1


@contextmanager
def local_policy(policy, monkeypatch):
    import sys
    from types import ModuleType

    module = ModuleType("tower_test_policy")
    module.Model = lambda: policy

    # The CLI instantiates classes; bind a factory class with explicit ownership.
    class Model:
        def __new__(cls):
            return policy

    module.Model = Model
    monkeypatch.setitem(sys.modules, "tower_test_policy", module)
    yield "tower_test_policy:Model"


def config_file(tmp_path, *, episodes=2, adapter="reference", server=""):
    path = tmp_path / "config.toml"
    path.write_text(
        f'episodes = {episodes}\nseed = 7\nduration_seconds = 0.2\noutput = "run"\n'
        f'[environment]\nbackend = "cpu"\n[policy]\nadapter = "{adapter}"\n'
        f'server = "{server}"\nentrypoint = "{server}"\n'
    )
    return path


@pytest.mark.parametrize("command", ["smoke", "version"])
def test_cli_discovery(command, capsys):
    assert cli.main([command]) == 0
    output = capsys.readouterr().out
    if command == "version":
        assert output.strip() == __version__
    else:
        result = json.loads(output)
        assert result["status"] == "ok" and result["steps"] == 3
        assert result["metrics"]["valid"] is False


def test_cli_config_check(tmp_path, capsys):
    path = config_file(tmp_path)
    assert cli.main(["config-check", str(path), "--set", "episodes=4"]) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 4


@pytest.mark.parametrize("command", ["config-check", "evaluate"])
def test_cli_invalid_config(tmp_path, capsys, command):
    path = config_file(tmp_path, episodes=0)
    args = (
        ["config-check", str(path)]
        if command == "config-check"
        else ["evaluate", "--config", str(path)]
    )
    assert cli.main(args) == 2
    assert "episodes must be a positive integer" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("adapter,episodes", [("reference", 2), ("local", 3)])
def test_cli_evaluate_multiple_episodes(tmp_path, capsys, adapter, episodes, monkeypatch):
    policy = TrackingPolicy()
    with local_policy(policy, monkeypatch) as server:
        path = config_file(tmp_path, episodes=episodes, adapter=adapter, server=server)
        assert cli.main(["evaluate", "--config", str(path)]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["episodes"]["completed"] == episodes
        assert result["episodes"]["invalid"] == episodes
        assert result == json.loads((tmp_path / "run/summary.json").read_text())
        for index, report in enumerate(result["episode_reports"]):
            episode = json.loads((tmp_path / "run" / report).read_text())
            assert episode["seed"] == 7 + index and episode["steps"] == 2
        if adapter == "local":
            assert policy.seeds == [7, 8, 9]
            assert policy.closes == 1


def test_evaluate_policy_leaves_adapter_owned_by_caller():
    policy = TrackingPolicy()
    try:
        for seed in (1, 2):
            evaluate_policy(TowerEnv(CpuReferenceBackend(), max_steps=1), policy, seed=seed)
        assert policy.seeds == [1, 2]
        assert policy.closes == 0
    finally:
        policy.close()


def test_cli_closes_local_policy_after_reset_failure(tmp_path, capsys, monkeypatch):
    class FailingPolicy(TrackingPolicy):
        def reset(self, **kwargs):
            raise ValueError("reset failed")

    policy = FailingPolicy()
    with local_policy(policy, monkeypatch) as server:
        path = config_file(tmp_path, adapter="local", server=server)
        assert cli.main(["evaluate", "--config", str(path)]) == 2
        assert policy.closes == 1
        assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("phase", ["reset", "act"])
def test_evaluate_failure_closes_environment_but_not_caller_policy(fake_sim, phase):
    class FailingPolicy(TrackingPolicy):
        def reset(self, **kwargs):
            if phase == "reset":
                raise RuntimeError("reset failed")
            super().reset(**kwargs)

        def act(self, observation, info=None):
            raise RuntimeError("act failed")

    policy = FailingPolicy()
    try:
        with pytest.raises(RuntimeError, match=f"{phase} failed"):
            evaluate_policy(TowerEnv(fake_sim, max_steps=1), policy)
        assert fake_sim.closes == 1
        assert policy.closes == 0
    finally:
        policy.close()


def test_cli_overwrites_its_own_report(tmp_path, capsys):
    path = config_file(tmp_path)
    assert cli.main(["evaluate", "--config", str(path)]) == 0
    obsolete = tmp_path / "run/obsolete.txt"
    obsolete.touch()
    assert cli.main(["evaluate", "--config", str(path), "--overwrite"]) == 0
    assert not obsolete.exists()
    assert json.loads((tmp_path / "run/summary.json").read_text())["episodes"]["completed"] == 2


def test_cli_overwrite_recovers_an_interrupted_run(tmp_path, capsys, monkeypatch):
    class FailsOnSecondEpisode(TrackingPolicy):
        def reset(self, *, seed=None, metadata=None):
            if self.seeds:
                raise ValueError("episode 2 failed")
            super().reset(seed=seed, metadata=metadata)

    policy = FailsOnSecondEpisode()
    with local_policy(policy, monkeypatch) as server:
        path = config_file(tmp_path, adapter="local", server=server)
        assert cli.main(["evaluate", "--config", str(path)]) == 2
    assert not (tmp_path / "run/summary.json").exists()
    assert (tmp_path / "run" / cli.RUN_MARKER).is_file()
    path = config_file(tmp_path)
    assert cli.main(["evaluate", "--config", str(path), "--overwrite"]) == 0
    assert json.loads((tmp_path / "run/summary.json").read_text())["episodes"]["completed"] == 2


def test_close_failure_does_not_mask_the_episode_error(fake_sim, tmp_path, capsys):
    class BrokenClose(TrackingPolicy):
        def act(self, observation, info=None):
            raise RuntimeError("act failed")

        def close(self):
            raise OSError("close failed")

    def broken_env_close():
        raise OSError("env close failed")

    fake_sim.close = broken_env_close
    with pytest.raises(RuntimeError, match="act failed"):
        evaluate_policy(TowerEnv(fake_sim, max_steps=1), BrokenClose())
