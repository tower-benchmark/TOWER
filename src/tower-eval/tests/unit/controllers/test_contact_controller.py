# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from tower_eval.controllers.contact import ContactClosure
from tower_eval.controllers.controller import Controller, RobotSpec
from tower_eval_fakes import FakeSim


class Sim(FakeSim):
    def __init__(self):
        super().__init__()
        self.state.qpos[[7, 15]] = 0.05

    def step(self, target):
        # Contact prevents the commanded fingers from reaching their target.
        measured = self.state.qpos.copy()
        state = super().step(target)
        state.qpos = measured
        return state


def controller():
    robot = RobotSpec.from_dict(
        dict(
            names=[str(i) for i in range(16)],
            joint_indices=[*range(7), *range(8, 15)],
            grippers=[dict(index=i, lower=0.0, upper=0.1) for i in (7, 15)],
            lower=[-3.0] * 16,
            upper=[3.0] * 16,
        )
    )
    return Controller(
        robot,
        interpolation_fraction=0.0,
        gripper_control_profile="generator_hysteresis_v1",
        gripper_hysteresis_close_score=0.75,
        gripper_hysteresis_open_score=0.25,
        gripper_model_closed_widths_m=[0.044] * 2,
        gripper_model_open_widths_m=[0.1] * 2,
        gripper_closed_targets_m=[0.03] * 2,
        gripper_open_targets_m=[0.1] * 2,
        gripper_contact_width_m=0.05,
        gripper_contact_close_speed_m_s=0.03,
        gripper_contact_hold_force_n=2.5,
    )


def test_contact_matches_generator_and_holds_until_open():
    c, sim = controller(), Sim()
    ref = ContactClosure(0.05, 0.03, force_n=2.5, speed=0.03, contact_width=0.05)
    target = np.zeros(16)
    target[[7, 15]] = [0.044, 0.1]
    events = []
    for _ in range(40):
        expected = ref.update(0.05, [2.0, -2.5], [0.0, 0.0], finger_positions=[0.025, -0.025])
        c.execute_target(sim, target, action_dt=1 / 30, on_dispatch=lambda **e: events.append(e))
        assert events[-1]["applied_target"][7] == pytest.approx(expected)
        assert 1 not in c._contact_closures
    assert ref.latched and c._contact_closures[0].latched
    assert events[-1]["applied_target"][7] > 0.03
    target[7] = 0.1
    c.execute_target(sim, target, action_dt=1 / 30)
    assert not c._contact_closures
    c.reset()
    assert not c._contact_closures


def test_contact_rejects_hold_above_limit():
    c, sim = controller(), Sim()
    sim.limit = 2.0
    target = np.zeros(16)
    target[[7, 15]] = 0.044
    with pytest.raises(ValueError, match="exceeds physical"):
        c.execute_target(sim, target)


def test_contact_requires_finger_feedback_before_dispatch():
    from tower_eval.simulator import CpuReferenceBackend

    sim = CpuReferenceBackend(dt=1 / 30)
    c = controller()
    target = np.zeros(16)
    target[[7, 15]] = 0.044
    with pytest.raises(ValueError, match="finger_feedback"):
        c.execute_target(sim, target)
    assert sim._state.time == 0.0
