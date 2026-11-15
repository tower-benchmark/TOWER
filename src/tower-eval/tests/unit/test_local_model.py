# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import numpy as np
from tower_eval import ActionChunk
from tower_eval.envs import TowerEnv
from tower_eval.evaluation import evaluate
from tower_eval.policies import LocalPolicyAdapter
from tower_eval.policies.smoke import smoke_observation
from tower_eval.simulator import CpuReferenceBackend


def test_xpolicylab_style_random_model_lifecycle_and_seed():
    path = Path(__file__).resolve().parents[2] / "examples/random_policy/model.py"
    spec = importlib.util.spec_from_file_location("random_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Model()
    policy = LocalPolicyAdapter(model)
    policy.reset(seed=73)
    obs = smoke_observation()
    first = policy.act(obs)
    assert isinstance(first, ActionChunk)
    np.testing.assert_array_equal(model.observation["state"]["joint_encoder"], obs["joint_encoder"])
    policy.reset(seed=73)
    np.testing.assert_array_equal(policy.act(obs).actions, first.actions)
    result = evaluate(
        TowerEnv(CpuReferenceBackend(), max_steps=3), module.CpuSmokePolicy(), seed=73
    )
    assert result["steps"] == 3
    assert result["metrics"]["valid"] is False


def test_local_model_receives_the_same_vision_and_state_contract():
    class EchoModel:
        def update_obs(self, obs):
            self.obs = obs

        def get_action(self):
            joints = self.obs["state"]["joint_encoder"]
            camera = self.obs["vision"]["left_wrist_camera"]["color"]
            assert camera.dtype == np.uint8 and camera[0, 0, 0] == 193
            assert self.obs["instruction"] == "CPU transport smoke"
            return [
                {
                    f"{arm}_{key}": value
                    for arm, index in (("left", 0), ("right", 1))
                    for key, value in (
                        ("arm_joint_state", joints[index]),
                        ("ee_joint_state", [0.05]),
                    )
                }
            ]

    model = EchoModel()
    policy = LocalPolicyAdapter(model)
    obs = smoke_observation()
    obs["joint_encoder"][0, 1] = 0.4321
    result = policy.act(obs)
    assert result.actions[0, 1] == 0.4321  # No J2 conversion in either adapter.
    assert result.metadata["coordinate_frame"] == "encoder_arm_base"
    assert model.obs["state"]["joint_encoder"] is not obs["joint_encoder"]
