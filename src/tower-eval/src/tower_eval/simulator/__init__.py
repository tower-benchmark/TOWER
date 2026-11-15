# SPDX-License-Identifier: Apache-2.0
"""Simulator backend interfaces and the deterministic CPU reference backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..controllers import GripperSpec, RobotSpec


@dataclass
class MeasuredState:
    """Core state shared by all backends; diagnostics are optional."""

    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    effort: np.ndarray | None = None
    blocks: np.ndarray | None = None
    block_velocities: np.ndarray | None = None
    eef_world: np.ndarray | None = None


class SimulatorBackend(Protocol):
    """Minimum backend contract required by :class:`tower_eval.envs.TowerEnv`."""

    robot: RobotSpec
    action_space: Any
    capabilities: dict[str, Any]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> MeasuredState: ...
    def read(self) -> MeasuredState: ...
    def step(self, action: np.ndarray) -> MeasuredState: ...
    def capture(self) -> dict[str, np.ndarray]: ...
    def close(self) -> None: ...

    def reward(self, state: MeasuredState) -> float: ...
    def terminated(self, state: MeasuredState) -> bool: ...


@runtime_checkable
class FingerFeedbackBackend(Protocol):
    """Optional contact-control capability, separate from the core backend."""

    def finger_feedback(self, arm: int) -> dict[str, Any]: ...


class CpuReferenceBackend:
    """Deterministic backend for adoption, CI, and policy integration tests."""

    action_coordinate_frame = "physical"

    SUPPORTED_TASKS = {
        "tower_transfer_2",
        "tower_transfer_3",
        "cross_tower_transfer_2",
        "cross_tower_transfer_3",
        "aligned_stack_2",
        "aligned_stack_3",
    }

    def __init__(self, *, dt: float = 0.1, task: str = "tower_transfer_2") -> None:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"unsupported CPU reference task: {task!r}")
        names = tuple(f"channel_{index}" for index in range(16))
        lower = np.full(16, -1.0)
        upper = np.full(16, 1.0)
        for gripper in (GripperSpec(7, 0.0, 0.1), GripperSpec(15, 0.0, 0.1)):
            lower[gripper.index] = gripper.lower
            upper[gripper.index] = gripper.upper
        self.robot = RobotSpec(
            names=names,
            joint_indices=tuple([*range(7), *range(8, 15)]),
            grippers=(GripperSpec(7, 0.0, 0.1), GripperSpec(15, 0.0, 0.1)),
            lower=lower,
            upper=upper,
        )
        self.dt = float(dt)
        self.task = task
        self._state = MeasuredState(0.0, np.zeros(16), np.zeros(16))
        self._seed: int | None = None
        self.action_space = (self.robot.lower.copy(), self.robot.upper.copy())
        self.capabilities = {
            "name": "cpu-reference",
            "deterministic": True,
            "render": False,
            "task_semantics": "smoke",
            "task": task,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> MeasuredState:
        self._seed = seed
        self._state = MeasuredState(0.0, np.zeros(16), np.zeros(16))
        return self._state

    def read(self) -> MeasuredState:
        """Return the latest measurement without advancing simulation."""
        return self._state

    def step(self, action: np.ndarray) -> MeasuredState:
        target = np.asarray(action, dtype=np.float64)
        if target.shape != (16,) or not np.isfinite(target).all():
            raise ValueError("CPU backend expects a finite action vector of shape (16,)")
        if np.any(target < self.robot.lower) or np.any(target > self.robot.upper):
            raise ValueError("CPU backend action exceeds declared robot limits")
        velocity = (target - self._state.qpos) / self.dt
        self._state = MeasuredState(self._state.time + self.dt, target.copy(), velocity)
        return self._state

    def capture(self) -> dict[str, np.ndarray]:
        return {}

    def reward(self, state: MeasuredState) -> float:
        return 0.0

    def terminated(self, state: MeasuredState) -> bool:
        return False

    def close(self) -> None:
        return None


from .commands import (  # noqa: E402 - Pure shared API after backend interfaces.
    joint_commands,
    validate_initial_robot_state,
)

__all__ = [
    "CpuReferenceBackend",
    "FingerFeedbackBackend",
    "MeasuredState",
    "SimulatorBackend",
    "joint_commands",
    "validate_initial_robot_state",
]
