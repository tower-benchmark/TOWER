# SPDX-License-Identifier: Apache-2.0
import contextlib

import numpy as np
import pytest
from tower_eval.controllers.controller import Controller, GripperSpec, RobotSpec
from tower_eval_fakes import FakeSim


def config():
    return dict(
        names=[f"channel{i}" for i in range(16)],
        joint_indices=[*range(7), *range(8, 15)],
        grippers=[dict(index=7, lower=0, upper=0.1), dict(index=15, lower=0, upper=0.2)],
        lower=[-1.0] * 16,
        upper=[1.0] * 16,
    )


class Sim(FakeSim):
    def __init__(self, grip_feedback=None):
        super().__init__(robot=RobotSpec.from_dict(config()))
        self.grip_feedback = grip_feedback

    def step(self, applied):
        state = super().step(applied)
        if self.grip_feedback is not None:
            state.qpos[7] = self.grip_feedback[len(self.commands) - 1]
        return state


def test_explicit_back_arm_hold_preserves_first_measured_pose():
    sim = Sim()
    sim.state.qpos[8:16] = 0.07
    c = Controller(
        RobotSpec.from_dict(config()),
        interpolation_fraction=0.0,
        gripper_control_profile="direct_absolute",
        hold_back_arm=True,
    )
    target = np.full(16, 0.15)
    c.execute_target(sim, target)
    sim.state.qpos[8:16] = 0.09
    c.execute_target(sim, target)
    np.testing.assert_allclose(np.array(sim.commands)[:, 8:16], 0.07)
    np.testing.assert_allclose(np.array(sim.commands)[:, :7], 0.15)
    c.reset()
    sim.state.qpos[8:16] = 0.08
    c.execute_target(sim, target)
    np.testing.assert_allclose(sim.commands[-1][8:16], 0.08)


def test_bimanual_interpolation_fractions_and_observation_identity():
    sim, records = Sim(), []
    controller = Controller(RobotSpec.from_dict(config()))
    target = np.zeros(16)
    target[0], target[8] = 0.9, -0.6
    final = controller.execute_target(sim, target, on_step=lambda **kw: records.append(kw))
    np.testing.assert_allclose(np.array(sim.commands)[:, 0], [0.3, 0.6, 0.9])
    np.testing.assert_allclose(np.array(sim.commands)[:, 8], [-0.2, -0.4, -0.6])
    assert records[-1]["measured"] is final is sim.state
    assert not records[-1]["arm_limit_violations"]
    # A new prediction starts from actual measured positions, not the last target.
    sim.state.qpos[0] = 0.3
    controller.execute_target(sim, target)
    assert sim.commands[-3][0] == pytest.approx(0.5)


def test_gripper_travel_bounds_preserved_without_measured_step_clipping():
    sim, records = Sim(grip_feedback=[0.005, 0.007, 0.009]), []
    target = np.zeros(16)
    target[[7, 15]] = [1.0, -0.3]
    Controller(RobotSpec.from_dict(config())).execute_target(
        sim, target, on_step=lambda **kw: records.append(kw)
    )
    np.testing.assert_allclose(np.array(sim.commands)[:, 7], [0.1 / 3, 0.2 / 3, 0.1])
    np.testing.assert_array_equal(np.array(sim.commands)[:, 15], 0.0)
    np.testing.assert_allclose(
        [r["interpolated_target"][7] for r in records], [0.1 / 3, 0.2 / 3, 0.1]
    )
    assert records[-1]["measured"].qpos[7] == 0.009
    assert records[-1]["raw_target"][7] == 1.0
    assert {m["kind"] for m in records[0]["modifications"]} == {"gripper_target_bounds"}


def test_generator_gripper_directional_rates_and_contact_boundary():
    sim, records = Sim(), []
    sim.state.qpos[[7, 15]] = 0.046
    target = np.zeros(16)
    target[7], target[15] = 0.03, 0.1
    ctrl = Controller(
        RobotSpec.from_dict(config()), gripper_control_profile="generator_directional_v1"
    )
    ctrl.execute_target(sim, target, on_step=lambda **kw: records.append(kw))
    # Closing crosses the 45 mm boundary at the 90 mm/s free-space rate,
    # then continues at 10 mm/s. Opening uses 160 mm/s throughout.
    np.testing.assert_allclose(
        np.array(sim.commands)[:, 7], [0.043, 0.043 - 0.01 / 30, 0.043 - 0.02 / 30]
    )
    np.testing.assert_allclose(
        np.array(sim.commands)[:, 15], [0.046 + 0.16 / 30, 0.046 + 0.32 / 30, 0.046 + 0.48 / 30]
    )
    assert records[0]["modifications"][0]["kind"] == "gripper_generator_close_rate"
    assert records[1]["modifications"][0]["kind"] == "gripper_generator_contact_close_rate"
    assert records[0]["modifications"][1]["kind"] == "gripper_generator_open_rate"
    # Generator-mode rate limiting follows the command stream, not feedback.
    sim.state.qpos[[7, 15]] = 0.09
    ctrl.execute_target(sim, target)
    assert sim.commands[-1][7] == pytest.approx(0.043 - 0.05 / 30)


def test_generator_gripper_reset_reseeds_from_measured_width():
    sim = Sim()
    sim.state.qpos[[7, 15]] = 0.08
    target = np.zeros(16)
    target[[7, 15]] = 0.1
    ctrl = Controller(
        RobotSpec.from_dict(config()), gripper_control_profile="generator_directional_v1"
    )
    ctrl.execute_target(sim, target)
    assert sim.commands[0][7] == pytest.approx(0.08 + 0.16 / 30)
    ctrl.reset()
    sim.state.qpos[[7, 15]] = 0.06
    ctrl.execute_target(sim, target)
    assert sim.commands[-3][7] == pytest.approx(0.06 + 0.16 / 30)


def test_generator_hysteresis_uses_close_score_and_holds_middle_band():
    sim, records = Sim(), []
    sim.state.qpos[[7, 15]] = 0.1
    kwargs = dict(
        gripper_control_profile="generator_hysteresis_v1",
        gripper_hysteresis_close_score=0.75,
        gripper_hysteresis_open_score=0.25,
        gripper_model_closed_widths_m=[0.04, 0.04],
        gripper_model_open_widths_m=[0.1, 0.1],
        gripper_closed_targets_m=[0.044, 0.044],
        gripper_open_targets_m=[0.1, 0.1],
    )
    ctrl = Controller(RobotSpec.from_dict(config()), interpolation_fraction=0.0, **kwargs)
    target = np.zeros(16)
    target[[7, 15]] = 0.07  # Close score .5: retain the initial open state.
    ctrl.execute_target(sim, target)
    np.testing.assert_allclose(np.asarray(sim.commands)[:, [7, 15]], 0.1)
    target[[7, 15]] = 0.055  # Close score .75: enter closed state.
    ctrl.execute_target(sim, target, on_step=lambda **kw: records.append(kw))
    np.testing.assert_allclose(np.asarray(sim.commands)[-3:, 7], [0.097, 0.094, 0.091])
    assert records[0]["modifications"][0]["kind"] == "gripper_hysteresis_decision"
    assert records[0]["modifications"][0]["state"] == "closed"
    target[[7, 15]] = 0.07  # Middle band keeps the closed state.
    ctrl.execute_target(sim, target)
    assert sim.commands[-1][7] == pytest.approx(0.082)
    target[[7, 15]] = 0.086  # Close score below .25: reopen.
    ctrl.execute_target(sim, target)
    assert sim.commands[-1][7] > 0.082


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(gripper_hysteresis_close_score=0.2, gripper_hysteresis_open_score=0.3),
        dict(gripper_hysteresis_close_score=1.1, gripper_hysteresis_open_score=0.3),
        dict(
            gripper_hysteresis_close_score=0.7,
            gripper_hysteresis_open_score=0.3,
            gripper_model_closed_widths_m=[0.04],
            gripper_model_open_widths_m=[0.1, 0.1],
            gripper_closed_targets_m=[0.044, 0.044],
            gripper_open_targets_m=[0.1, 0.1],
        ),
        dict(
            gripper_hysteresis_close_score=0.7,
            gripper_hysteresis_open_score=0.3,
            gripper_model_closed_widths_m=[0.1, 0.04],
            gripper_model_open_widths_m=[0.1, 0.1],
            gripper_closed_targets_m=[0.044, 0.044],
            gripper_open_targets_m=[0.1, 0.1],
        ),
    ],
)
def test_invalid_generator_hysteresis_parameters(kwargs):
    defaults = dict(
        gripper_control_profile="generator_hysteresis_v1",
        gripper_hysteresis_close_score=0.75,
        gripper_hysteresis_open_score=0.25,
        gripper_model_closed_widths_m=[0.04, 0.04],
        gripper_model_open_widths_m=[0.1, 0.1],
        gripper_closed_targets_m=[0.044, 0.044],
        gripper_open_targets_m=[0.1, 0.1],
    )
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        Controller(RobotSpec.from_dict(config()), **defaults)


def test_direct_absolute_profile_only_clamps_physical_gripper_travel():
    sim, records = Sim(), []
    sim.state.qpos[[7, 15]] = [0.08, 0.02]
    target = np.zeros(16)
    target[0], target[8], target[7], target[15] = [0.9, -0.6, 1.0, -0.3]
    Controller(
        RobotSpec.from_dict(config()),
        interpolation_fraction=0.0,
        gripper_control_profile="direct_absolute",
    ).execute_target(sim, target, on_step=lambda **kw: records.append(kw))
    expected = target.copy()
    expected[[7, 15]] = [0.1, 0.0]
    np.testing.assert_allclose(np.asarray(sim.commands), np.tile(expected, (3, 1)))
    assert {m["kind"] for m in records[0]["modifications"]} == {"gripper_target_bounds"}
    assert not any(m["kind"].startswith("arm_") for m in records[0]["modifications"])


def test_out_of_range_arm_targets_pass_through_with_diagnostics():
    sim, records = Sim(), []
    target = np.zeros(16)
    target[0], target[8] = 2.0, -3.0
    Controller(RobotSpec.from_dict(config())).execute_target(
        sim, target, on_step=lambda **kw: records.append(kw)
    )
    np.testing.assert_array_equal(sim.commands[-1], target)
    assert {v["index"] for v in records[-1]["arm_limit_violations"]} == {0, 8}
    assert {v["stage"] for v in records[-1]["arm_limit_violations"]} == {
        "raw_target",
        "applied_target",
    }


@pytest.mark.parametrize(
    "target",
    [
        np.zeros(18),
        np.zeros((1, 16)),
        np.full(16, np.nan),
        np.full(16, np.inf),
        np.ones(16, dtype=complex),
    ],
)
def test_invalid_targets_rejected_before_dispatch(target):
    sim = Sim()
    with pytest.raises(ValueError):
        Controller(RobotSpec.from_dict(config())).execute_target(sim, target)
    assert not sim.commands


@pytest.mark.parametrize("dt", [0, -0.1, 0.05, float("nan"), float("inf")])
def test_invalid_action_time(dt):
    with pytest.raises(ValueError):
        Controller(RobotSpec.from_dict(config())).execute_target(Sim(), np.zeros(16), action_dt=dt)


@pytest.mark.parametrize(
    "key,value",
    [
        ("names", ["x"] * 16),
        ("joint_indices", list(range(14))),
        ("lower", [0.0] * 15),
        ("upper", [float("inf")] * 16),
        ("grippers", [dict(index=7, lower=0.1, upper=0)]),
    ],
)
def test_invalid_robot_config(key, value):
    data = config()
    data[key] = value
    with pytest.raises(ValueError):
        RobotSpec.from_dict(data)


def test_config_isolation_and_missing_fields():
    data = config()
    robot = RobotSpec.from_dict(data)
    data["lower"][0] = -99
    assert robot.lower[0] == -1 and not robot.lower.flags.writeable
    with pytest.raises(ValueError):
        RobotSpec.from_dict({})
    with pytest.raises(ValueError):
        GripperSpec(True, 0, 0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sim_dt=0),
        dict(interpolation_fraction=1.1),
        dict(gripper_step_fraction=-1),
        dict(sim_dt=float("nan")),
        dict(gripper_control_profile="unknown"),
        dict(gripper_open_speed_m_s=0),
        dict(gripper_close_speed_m_s=-1),
        dict(gripper_contact_close_speed_m_s=float("nan")),
    ],
)
def test_invalid_controller_parameters(kwargs):
    deprecated = "gripper_step_fraction" in kwargs
    with (
        pytest.warns(FutureWarning) if deprecated else contextlib.nullcontext(),
        pytest.raises(ValueError),
    ):
        Controller(RobotSpec.from_dict(config()), **kwargs)


def test_generator_gripper_contact_width_inside_each_physical_range():
    with pytest.raises(ValueError, match="contact width"):
        Controller(
            RobotSpec.from_dict(config()),
            gripper_control_profile="generator_directional_v1",
            gripper_contact_width_m=0.3,
        )


def test_invalid_measured_state_and_nonadvancing_clock():
    controller = Controller(RobotSpec.from_dict(config()))
    sim = Sim()
    sim.state.qpos = np.zeros(15)
    with pytest.raises(ValueError):
        controller.execute_target(sim, np.zeros(16))
    sim = Sim()
    sim.step = lambda applied: sim.state
    with pytest.raises(ValueError, match="advance"):
        controller.execute_target(sim, np.zeros(16))


def test_deadband_holds_fixed_reference_and_leaves_gripper_active():
    sim, records = Sim(), []
    ctrl = Controller(RobotSpec.from_dict(config()), arm_deadband_rad=0.05)
    target = np.zeros(16)
    target[0], target[8], target[7] = 0.04, -0.08, 0.03
    ctrl.execute_target(sim, target, on_step=lambda **kw: records.append(kw))
    assert sim.commands[-1][0] == 0.0
    assert sim.commands[-1][8] == -0.08
    assert sim.commands[-1][7] == pytest.approx(0.03)
    assert any(m["kind"] == "arm_deadband" and m["index"] == 0 for m in records[0]["modifications"])
    # Feedback drift must not rebase the held command on every prediction.
    for _ in range(5):
        sim.state.qpos[0] = 0.02
        ctrl.execute_target(sim, target)
        assert sim.commands[-1][0] == 0.0
    # Above-threshold target becomes the new fixed reference.
    target[0] = 0.06
    ctrl.execute_target(sim, target)
    assert sim.commands[-1][0] == 0.06
    target[0] = 0.08
    ctrl.execute_target(sim, target)
    assert sim.commands[-1][0] == 0.06


def test_deadband_boundary_disabled_and_reset():
    sim = Sim()
    ctrl = Controller(RobotSpec.from_dict(config()), arm_deadband_rad=0.05)
    target = np.zeros(16)
    target[0] = 0.05
    ctrl.execute_target(sim, target)
    assert sim.commands[-1][0] == 0.05  # Equality is accepted.
    ctrl.reset()
    sim.state.qpos[0] = 1.0
    target[0] = 1.02
    ctrl.execute_target(sim, target)
    assert sim.commands[-1][0] == 1.0
    ctrl = Controller(RobotSpec.from_dict(config()), arm_deadband_rad=0.0)
    target[0] = 1.001
    ctrl.execute_target(sim, target)
    assert sim.commands[-1][0] == 1.001


@pytest.mark.parametrize("threshold", [-0.01, float("nan"), float("inf")])
def test_invalid_deadband(threshold):
    with pytest.raises(ValueError):
        Controller(RobotSpec.from_dict(config()), arm_deadband_rad=threshold)
