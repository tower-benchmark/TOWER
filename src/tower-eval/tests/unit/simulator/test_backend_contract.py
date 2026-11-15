# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from tower_eval.controllers import DualModeController
from tower_eval.envs import TowerEnv
from tower_eval.simulator import CpuReferenceBackend, SimulatorBackend


def test_cpu_backend_supports_controller_read_and_step():
    sim = CpuReferenceBackend()
    controller = DualModeController(sim.robot, sim_dt=sim.dt)
    env = TowerEnv(sim, controller, max_steps=1)
    try:
        observation, _ = env.reset(seed=7)
        np.testing.assert_array_equal(sim.read().qpos, observation["qpos"])
        target = dict(input_space="joint", joints=[[0.2] * 7] * 2, gripper=[0.04] * 2)
        observation, _, _, truncated, _ = env.step(target)
        np.testing.assert_allclose(observation["qpos"][:7], 0.2)
        assert sim.read().time == pytest.approx(sim.dt)
        assert truncated
    finally:
        env.close()


def test_backend_protocol_declares_read():
    assert "read" in SimulatorBackend.__dict__


def test_finger_feedback_is_an_optional_explicit_capability(fake_sim):
    from tower_eval.simulator import FingerFeedbackBackend

    assert isinstance(fake_sim, FingerFeedbackBackend)
    assert not isinstance(CpuReferenceBackend(), FingerFeedbackBackend)
