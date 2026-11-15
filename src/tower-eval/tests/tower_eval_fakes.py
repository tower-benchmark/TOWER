# SPDX-License-Identifier: Apache-2.0
"""Shared deterministic simulator for controller and rollout tests.

A uniquely named module, so `conftest` name collisions across test roots
(for example packages/tower-sim/tests) cannot shadow it.
"""

from dataclasses import replace

import numpy as np
from tower_eval.controllers import RobotSpec
from tower_eval.simulator import MeasuredState


class FakeSim:
    dt = 1 / 30

    def __init__(self, *, robot=None, diagnostics=False):
        self.robot = robot or RobotSpec.from_dict(
            dict(
                names=[f"q{i}" for i in range(16)],
                joint_indices=[*range(7), *range(8, 15)],
                grippers=[dict(index=i, lower=0.0, upper=0.1) for i in (7, 15)],
                lower=[-2.0] * 7 + [0.0] + [-2.0] * 7 + [0.0],
                upper=[2.0] * 7 + [0.1] + [2.0] * 7 + [0.1],
            )
        )
        self.action_space = (self.robot.lower.copy(), self.robot.upper.copy())
        self.capabilities = {"name": "fake", "task_semantics": "smoke"}
        self.commands = []
        self.resets = 0
        self.closes = 0
        self.limit = 3.0
        self.state = MeasuredState(0.0, np.zeros(16), np.zeros(16))
        if diagnostics:
            self.state.effort = np.zeros(16)
            self.state.blocks = np.array([[0, 0, 0, 0, 0, 0, 1.0]])
            self.state.block_velocities = np.zeros((1, 6))
            self.state.eef_world = np.array([[0, 0, 0, 0, 0, 0, 1.0]] * 2)

    def read(self):
        return self.state

    def step(self, target):
        target = np.asarray(target).copy()
        self.commands.append(target)
        self.state = replace(self.state, time=self.state.time + self.dt, qpos=target.copy())
        return self.state

    def capture(self):
        return {"camera": np.full((16, 16, 3), len(self.commands) % 256, np.uint8)}

    def finger_feedback(self, arm):
        return dict(
            positions_m=[0.025, -0.025],
            velocities_m_s=[0.0, 0.0],
            forces_n=[2.0, -2.5],
            effort_limit_n=self.limit,
        )

    def reset(self, *, seed=None, options=None):
        self.resets += 1
        self.commands.clear()
        self.state = replace(self.state, time=0.0, qpos=np.zeros(16), qvel=np.zeros(16))
        return self.state

    def reward(self, state):
        return 0.0

    def terminated(self, state):
        return False

    def close(self):
        self.closes += 1
