# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
from tower_eval.config import load_config
from tower_eval.envs import TowerEnv


def test_toml_config_and_override(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('schema = "tower-eval-v1"\nseed = 2\n')
    config = load_config(path, ["seed=9", 'policy.adapter="reference"'])
    assert config.seed == 9
    assert config.policy["adapter"] == "reference"


class _Backend:
    def __init__(self):
        self.robot = SimpleNamespace(lower=np.full(16, -1), upper=np.full(16, 1))
        self.state = SimpleNamespace(qpos=np.zeros(16), qvel=np.zeros(16), time=0.0)

    def reset(self):
        self.state.time = 0.0
        return self.state

    def step(self, action):
        self.state.qpos = np.asarray(action)
        self.state.qvel = np.zeros(16)
        self.state.time += 0.1
        return self.state

    def capture(self):
        return {}

    def close(self):
        pass


def test_gymnasium_env_reset_step():
    env = TowerEnv(_Backend(), max_steps=2)
    observation, info = env.reset(seed=3)
    assert env.observation_space.contains(observation)
    assert info["seed"] == 3
    observation, reward, terminated, truncated, info = env.step(np.zeros(16, dtype=np.float32))
    assert env.observation_space.contains(observation)
    assert reward == 0.0 and not terminated and not truncated
    # A backend without reward/termination methods cannot report a task outcome.
    assert info["task_success"] is None
    env.step(np.zeros(16, dtype=np.float32))
    assert env._steps == 2
    env.close()
