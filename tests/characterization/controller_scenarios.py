"""Scripted controller inputs with deterministic time and a lagging fake backend."""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
from tower_eval.controllers import Controller, DualModeController, IKFailure, RobotSpec


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, SimpleNamespace):
        return plain(vars(value))
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


class ScriptedBackend:
    def __init__(self) -> None:
        self.robot = RobotSpec.from_dict(
            dict(
                names=[f"q{i}" for i in range(16)],
                joint_indices=[*range(7), *range(8, 15)],
                grippers=[dict(index=i, lower=0.0, upper=0.1) for i in (7, 15)],
                lower=[-2.0] * 7 + [0.0] + [-2.0] * 7 + [0.0],
                upper=[2.0] * 7 + [0.1] + [2.0] * 7 + [0.1],
            )
        )
        q = np.zeros(16)
        q[[7, 15]] = 0.08
        self.state = SimpleNamespace(qpos=q, qvel=np.zeros(16), time=0.0)

    def read(self) -> SimpleNamespace:
        return self.state

    def step(self, target: Any) -> SimpleNamespace:
        observed = self.state.qpos + 0.5 * (target - self.state.qpos)
        self.state = SimpleNamespace(
            qpos=observed, qvel=np.zeros(16), time=self.state.time + 1 / 30
        )
        return self.state


class ScriptedKinematics:
    """Exercise controller dispatch/atomicity; numerical IK has separate goldens."""

    backend = "scripted-characterization"
    reject_arm: int | None = None

    def to_base(self, arm: int, pose: Any, frame: str) -> Any:
        assert frame == "arm_base"
        return np.asarray(pose).copy()

    def ik(self, arm: int, pose: Any, seed: Any) -> tuple[Any, bool]:
        return np.r_[pose[:3], np.zeros(4)], arm != self.reject_arm

    def fk(self, arm: int, joints: Any) -> Any:
        return np.r_[joints[:3], 0.0, 0.0, 0.0, 1.0]


def controller_streams() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile in ("direct_absolute", "generator_directional_v1", "generator_hysteresis_v1"):
        sim = ScriptedBackend()
        options: dict[str, Any] = dict(
            gripper_control_profile=profile,
            arm_deadband_rad=0.01,
            arm_speed_rad_s=0.6,
            hold_back_arm=True,
        )
        if profile == "generator_hysteresis_v1":
            options.update(
                gripper_hysteresis_close_score=0.7,
                gripper_hysteresis_open_score=0.3,
                gripper_model_closed_widths_m=[0.0, 0.0],
                gripper_model_open_widths_m=[0.1, 0.1],
                gripper_closed_targets_m=[0.01, 0.01],
                gripper_open_targets_m=[0.09, 0.09],
            )
        controller = Controller(sim.robot, **options)
        events = []
        for phase, width in ((0.3, -0.01), (0.305, 0.12), (-0.2, 0.04)):
            target = np.full(16, phase)
            target[[7, 15]] = width
            controller.execute_target(
                sim,
                target,
                action_dt=0.1,
                on_dispatch=lambda events=events, **event: events.append(
                    {"dispatch": plain(event)}
                ),
                on_step=lambda events=events, **event: events.append({"step": plain(event)}),
            )
        controller.reset()
        result[profile] = dict(events=events, reset_target=controller.applied_target)
    sim, provider = ScriptedBackend(), ScriptedKinematics()
    dual = DualModeController(sim.robot, kinematics=provider, arm_speed_rad_s=0.6)
    events = []
    clock = itertools.count()
    with patch("time.perf_counter", side_effect=lambda: next(clock) / 1024):
        for mode, value in (("joint", 0.2), ("eef", 0.3), ("joint", -0.1), ("eef", 0.1)):
            action: dict[str, Any] = dict(input_space=mode, gripper=[0.04, 0.07])
            if mode == "joint":
                action["joints"] = [[value] * 7] * 2
            else:
                action.update(poses=[[value, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2, frame="arm_base")
            dual.execute(
                sim,
                action,
                action_dt=0.1,
                on_dispatch=lambda **event: events.append({"dispatch": plain(event)}),
                on_step=lambda **event: events.append({"step": plain(event)}),
                on_ik=lambda event: events.append({"ik": plain(event)}),
            )
        provider.reject_arm = 1
        before = sim.state.time
        try:
            dual.execute(
                sim, action, on_ik=lambda event: events.append({"ik_failure": plain(event)})
            )
        except IKFailure:
            assert sim.state.time == before
        else:
            raise AssertionError("Scripted rejection unexpectedly dispatched")
    dual.reset()
    result["dual_mode"] = dict(
        events=events,
        reset_report=dual.last_ik_report,
        reset_target=dual.applied_target,
        final_state=plain(sim.state),
    )
    return result
