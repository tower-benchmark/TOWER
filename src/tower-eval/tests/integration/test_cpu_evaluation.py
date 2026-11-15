# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from tower_eval.envs import TowerEnv
from tower_eval.evaluation import evaluate
from tower_eval.policies import ActionChunk
from tower_eval.simulator import CpuReferenceBackend


class MinimalPolicy:
    """Third-party example implementing only the documented v1 methods."""

    def reset(self, *, seed=None, metadata=None):
        self.seed = seed

    def act(self, observation, info=None):
        return [0.0] * 16

    def close(self):
        pass


def test_cpu_evaluation_uses_documented_adapter_contract(tmp_path):
    result = evaluate(
        TowerEnv(CpuReferenceBackend(), max_steps=3),
        MinimalPolicy(),
        seed=7,
        output=tmp_path / "summary.json",
    )
    assert result["schema"] == "tower-result-v1"
    assert result["steps"] == 3
    assert (tmp_path / "summary.json").exists()


def test_same_seed_replays_have_identical_observation_fingerprint():
    first = evaluate(
        TowerEnv(CpuReferenceBackend(task="tower_transfer_3"), max_steps=3),
        MinimalPolicy(),
        seed=123,
        max_steps=3,
    )
    second = evaluate(
        TowerEnv(CpuReferenceBackend(task="tower_transfer_3"), max_steps=3),
        MinimalPolicy(),
        seed=123,
        max_steps=3,
    )
    assert first["provenance"]["observation_sha256"] == second["provenance"]["observation_sha256"]


def test_cpu_backend_rejects_invalid_gripper_action():
    env = TowerEnv(CpuReferenceBackend(), max_steps=1)
    env.reset(seed=0)
    action = np.zeros(16, dtype=np.float32)
    action[7] = -0.1
    with pytest.raises(ValueError, match="outside"):
        env.step(action)


def test_cpu_reference_backend_passes_gymnasium_checker():
    check_env(TowerEnv(CpuReferenceBackend(), max_steps=3))


def test_action_chunk_executes_all_rows():
    class ChunkPolicy(MinimalPolicy):
        def act(self, observation, info=None):
            return ActionChunk(np.zeros((3, 16)), dt_s=0.1, metadata={"source": "test"})

    result = evaluate(TowerEnv(CpuReferenceBackend(), max_steps=3), ChunkPolicy(), seed=0)
    assert result["steps"] == 3
