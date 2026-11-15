# SPDX-License-Identifier: Apache-2.0
"""Protocol/atomicity tests use a fake provider; not numerical IK evidence."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tower_eval.controllers import DualModeController, IKFailure
from tower_eval.envs import TowerEnv
from tower_eval.evaluation.portable import evaluate_policy
from tower_eval.policies import ReferenceAdapter
from tower_eval.policies.adapter import ActionChunk, normalize_actions
from tower_eval.policies.serialization import jsonable
from tower_eval_fakes import FakeSim as Sim


class Provider:
    backend = "test_only"

    def __init__(self):
        self.calls = []
        self.fail = None
        self.resets = 0

    def reset(self):
        self.calls.clear()
        self.resets += 1

    def to_base(self, arm, pose, frame):
        return np.asarray(pose)

    def ik(self, arm, pose, seed):
        self.calls.append((arm, seed.copy()))
        return np.r_[pose[:3], Rotation.from_quat(pose[3:]).as_rotvec(), 0.0], arm != self.fail

    def fk(self, arm, q):
        return np.r_[q[:3], Rotation.from_rotvec(q[3:6]).as_quat()]


def action(mode="eef", x=0.1):
    if mode == "joint":
        return dict(input_space="joint", joints=np.full((2, 7), x).tolist(), gripper=[0.04, 0.04])
    return dict(
        input_space="eef", poses=[[x, 0, 0, 0, 0, 0, 1]] * 2, gripper=[0.04, 0.04], frame="arm_base"
    )


def test_public_switch_atomic_failure_reset_and_changed_input():
    sim, provider = Sim(), Provider()
    c = DualModeController(sim.robot, kinematics=provider)
    env = TowerEnv(sim, c)
    env.reset()
    env.step(action("joint"))
    assert not provider.calls
    observation, *_ = env.step(action())
    assert env.observation_space.contains(observation)
    first = sim.commands[-1].copy()
    env.step(action(x=0.2))
    assert np.linalg.norm(first - sim.commands[-1]) > 0.1
    env.step(action("joint", x=0.3))
    assert len(provider.calls) == 4 and sim.resets == 1
    assert c.last_ik_report is None
    before = len(sim.commands)
    provider.fail = 1
    with pytest.raises(IKFailure) as caught:
        env.step(action())
    assert len(sim.commands) == before
    assert caught.value.report["success"] is False
    assert sim.state.time == pytest.approx(4 / 30)
    env.reset()
    assert provider.calls == [] and c.last_ik_report is None
    assert env.action_space.contains(env.action_space.sample())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.update(frame="world"),
        lambda a: a.update(poses=[[0] * 7] * 2),
        lambda a: a.update(poses=[[0, 0, 0, 0, 0, 0, 2]] * 2),
        lambda a: a.update(gripper=[0.2, 0.1]),
        lambda a: a.update(gripper=[float("nan"), 0.1]),
        lambda a: a.update(poses=[[0] * 6] * 2),
        lambda a: a.update(units="mm"),
        lambda a: a.update(input_space="delta"),
        lambda a: a.update(poses=[[float("inf"), 0, 0, 0, 0, 0, 1]] * 2),
    ],
)
def test_rejects_bad_input_without_dispatch(mutate):
    sim = Sim()
    c = DualModeController(sim.robot, kinematics=Provider())
    a = action()
    mutate(a)
    with pytest.raises(ValueError):
        c.execute(sim, a)
    assert sim.commands == []


@pytest.mark.parametrize("mode", ["joint", "eef"])
@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), 0.01, 0.05])
def test_invalid_duration_preserves_previous_reports_without_dispatch(mode, duration):
    sim, provider = Sim(), Provider()
    controller = DualModeController(sim.robot, kinematics=provider)
    controller.execute(sim, action(), action_dt=1 / 30)
    reports, last = controller.ik_reports, controller.last_ik_report
    command_count, call_count = len(sim.commands), len(provider.calls)
    with pytest.raises(ValueError):
        controller.execute(sim, action(mode), action_dt=duration)
    assert controller.ik_reports is reports and controller.last_ik_report is last
    assert len(sim.commands) == command_count and len(provider.calls) == call_count


def test_legacy_arrays_are_joint_and_shared_rate_limiting():
    sim, provider = Sim(), Provider()
    c = DualModeController(sim.robot, kinematics=provider, arm_speed_rad_s=1.0)
    env = TowerEnv(sim, c)
    env.step(np.zeros(16))
    assert not provider.calls
    env.step(action("joint", x=0.5))
    assert sim.commands[-1][0] == pytest.approx(1 / 30)
    env.step(action(x=0.5))
    assert sim.commands[-1][0] == pytest.approx(2 / 30)
    assert c.last_ik_report["arms"][0]["position_error_m"] == 0
    assert c.last_ik_report["limited_target_errors"][0][0] > 0.4


@pytest.mark.parametrize(
    "duration,interpolation,speed,profile",
    [
        (1 / 30, 0.8, None, "direct_absolute"),
        (0.1, 0.8, None, "direct_absolute"),
        (0.1, 0.0, None, "direct_absolute"),
        (0.2, 1.0, None, "direct_absolute"),
        (0.1, 0.8, 1.0, "direct_absolute"),
        (0.1, 0.8, None, "generator_directional_v1"),
    ],
)
def test_eef_preserves_joint_interpolation_and_measured_seeds(
    duration, interpolation, speed, profile
):
    class LaggingSim(Sim):
        def step(self, q):
            state = super().step(q)
            state.qpos *= 0.5
            return state

    joint_sim, eef_sim, provider = LaggingSim(), LaggingSim(), Provider()
    options = dict(
        interpolation_fraction=interpolation, arm_speed_rad_s=speed, gripper_control_profile=profile
    )
    joint_env = TowerEnv(
        joint_sim, DualModeController(joint_sim.robot, **options), action_dt_s=duration
    )
    eef_env = TowerEnv(
        eef_sim,
        DualModeController(eef_sim.robot, kinematics=provider, **options),
        action_dt_s=duration,
    )
    # A second action must start its own schedule from the lagging measurement.
    for x, width in [(0.3, 0.08), (-0.2, 0.01)]:
        joints = np.array([[x, 0.1, -0.1, 0, 0, 0, 0], [-x, 0.05, 0.1, 0, 0, 0, 0]])
        initial = eef_sim.read().qpos.copy()
        first_tick = len(eef_sim.commands)
        joint_env.step(dict(input_space="joint", joints=joints.tolist(), gripper=[width] * 2))
        eef_env.step(
            dict(
                input_space="eef",
                poses=[provider.fk(a, q).tolist() for a, q in enumerate(joints)],
                gripper=[width] * 2,
                frame="arm_base",
            )
        )
        np.testing.assert_allclose(eef_sim.commands, joint_sim.commands, atol=1e-12)
        for tick in range(first_tick, len(eef_sim.commands)):
            measured = initial if tick == first_tick else 0.5 * eef_sim.commands[tick - 1]
            for arm in range(2):
                called_arm, seed = provider.calls[2 * tick + arm]
                assert called_arm == arm
                np.testing.assert_allclose(seed, measured[arm * 8 : arm * 8 + 7])
        assert len(provider.calls) == 2 * len(eef_sim.commands)
        for report in eef_env.controller.ik_reports:
            np.testing.assert_allclose(
                np.asarray(report["ik_target"])[[*range(7), *range(8, 15)]], joints.ravel()
            )


def test_mixed_chunk_serialization_and_evaluator(tmp_path):
    sim, provider = Sim(), Provider()
    env = TowerEnv(sim, DualModeController(sim.robot, kinematics=provider), max_steps=3)
    chunk = ActionChunk([action("joint"), action(), action("joint")], dt_s=1 / 30)
    wire = jsonable(chunk)
    assert wire["protocol"] == "tower-action-chunk-v1"
    assert normalize_actions(wire, env.action_dt_s) == chunk.actions
    result = evaluate_policy(env, ReferenceAdapter(wire), output=tmp_path / "result.json")
    assert result["steps"] == 3 and len(provider.calls) == 2
    assert result["task_success"] is False
    assert result["provenance"]["action_space"]["protocol"] == "tower-absolute-dual-v1"
    with pytest.raises(ValueError, match="dt_s"):
        normalize_actions(chunk, 0.1)


def test_no_implicit_provider_and_negative_quaternion():
    sim = Sim()
    with pytest.raises(RuntimeError, match="explicitly configured"):
        DualModeController(sim.robot).execute(sim, action())
    assert sim.commands == []
    provider = Provider()
    c = DualModeController(sim.robot, kinematics=provider)
    a = action()
    for p in a["poses"]:
        p[6] = -1
    c.execute(sim, a, 1 / 30)
    assert sim.commands[-1][0] == pytest.approx(0.1)


def test_legacy_float32_upper_bound_compatibility():
    sim = Sim()
    env = TowerEnv(sim, DualModeController(sim.robot))
    upper = sim.robot.upper.astype(np.float32)
    assert env.action_space.contains(upper)
    env.step(upper)
    assert sim.commands[-1][7] == pytest.approx(0.1)


def test_failed_tick_preserves_prior_physics_and_reseeds_from_measurement():
    sim, provider = Sim(), Provider()
    c = DualModeController(sim.robot, kinematics=provider)
    original = provider.ik

    def fail_after_one_tick(arm, pose, seed):
        solution, _ = original(arm, pose, seed)
        return solution, not (arm == 1 and len(sim.commands) >= 1)

    provider.ik = fail_after_one_tick
    reports = []
    with pytest.raises(IKFailure):
        c.execute(sim, action(), 0.1, on_ik=reports.append)
    assert len(sim.commands) == 1 and sim.state.time == pytest.approx(1 / 30)
    assert sim.commands[0][0] == pytest.approx(0.1 / 3)
    assert sim.commands[0][7] == pytest.approx(0.04 / 3)
    assert [report["success"] for report in reports] == [True, False]
    np.testing.assert_array_equal(provider.calls[2][1], sim.state.qpos[:7])
    np.testing.assert_array_equal(provider.calls[3][1], sim.state.qpos[8:15])


def test_eef_does_not_use_joint_replay_gateway():
    sim, provider = Sim(), Provider()
    c = DualModeController(sim.robot, kinematics=provider)

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy joint replay gateway was called")

    c.execute_target = forbidden
    env = TowerEnv(sim, c)
    env.step(action())
    assert len(provider.calls) == 2 and len(sim.commands) == 1


def test_direct_backend_honors_explicit_action_duration():
    sim = Sim()
    env = TowerEnv(sim, action_dt_s=0.1)
    _, _, _, _, info = env.step(np.zeros(16))
    assert len(sim.commands) == 3
    assert info["sim_time"] == pytest.approx(0.1)


def test_controller_period_must_match_backend_before_dispatch():
    sim = Sim()
    c = DualModeController(sim.robot, kinematics=Provider(), sim_dt=0.1)
    with pytest.raises(ValueError, match="period"):
        TowerEnv(sim, c)
    assert sim.commands == []


@pytest.mark.parametrize(
    "invalid_fk",
    [
        [float("nan"), 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 2],
        [0] * 6,
    ],
)
def test_independent_fk_must_be_a_finite_unit_pose(invalid_fk):
    sim, provider = Sim(), Provider()
    original = provider.fk
    provider.fk = lambda arm, q: np.asarray(invalid_fk) if arm == 1 else original(arm, q)
    c = DualModeController(sim.robot, kinematics=provider)
    with pytest.raises(IKFailure):
        c.execute(sim, action(x=0), 1 / 30)
    assert sim.commands == []
