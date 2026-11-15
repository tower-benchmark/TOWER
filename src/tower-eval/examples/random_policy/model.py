# SPDX-License-Identifier: Apache-2.0
"""Tiny XPolicyLab-style model for CPU wiring checks; no robot qualification."""

import numpy as np


class Model:
    def __init__(self, model_cfg=None):
        self.seed = int((model_cfg or {}).get("seed", 0))
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.observation = None

    def update_obs(self, obs):
        self.observation = obs

    def get_action(self):
        action = {}
        for arm in ("left", "right"):
            action[f"{arm}_arm_joint_state"] = self.rng.uniform(-0.01, 0.01, 7)
            action[f"{arm}_ee_joint_state"] = np.array([0.05])
        return [action]

    def prepare_case(self, case_meta=None):
        if case_meta and case_meta.get("episode_seed") is not None:
            self.seed = int(case_meta["episode_seed"])


class CpuSmokePolicy:
    """Explicit synthetic fixture for CPU evaluation; never real encoder commands."""

    def __init__(self):
        from tower_eval.policies import LocalPolicyAdapter

        self.model = LocalPolicyAdapter(Model())

    def reset(self, *, seed=None, metadata=None):
        self.model.reset(seed=seed, metadata=metadata)

    def act(self, observation, info=None):
        from tower_eval.policies.smoke import cpu_smoke_chunk, smoke_observation

        return cpu_smoke_chunk(self.model.act(smoke_observation()))

    def close(self):
        self.model.close()
