# SPDX-License-Identifier: Apache-2.0
"""Gymnasium environment façade for TOWER simulator backends."""

from __future__ import annotations

import inspect
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..constants import ACTION_DT, CONTROLLER_DT, CONTROLLER_HZ
from ..simulator import SimulatorBackend


class TowerEnv(gym.Env):
    """A thin Gymnasium wrapper around a calibrated TOWER simulator backend."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROLLER_HZ}

    def __init__(
        self,
        backend: SimulatorBackend,
        controller: Any | None = None,
        *,
        max_steps: int = 6000,
        render_mode: str | None = None,
        camera: str | None = None,
        action_dt_s: float | None = None,
        kinematics: Any | None = None,
    ) -> None:
        super().__init__()
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.backend = backend
        self.action_coordinate_frame = getattr(backend, "action_coordinate_frame", "physical")
        from ..controllers import DualModeController
        from .actions import DualActionSpace

        if controller is None and kinematics is not None:
            controller = DualModeController(
                backend.robot, kinematics=kinematics, sim_dt=getattr(backend, "dt", CONTROLLER_DT)
            )
        elif kinematics is not None:
            raise ValueError("Pass kinematics either through the controller or the environment")
        self.controller = controller
        self.action_dt_s = float(
            action_dt_s
            if action_dt_s is not None
            else (
                controller.sim_dt
                if isinstance(controller, DualModeController)
                else ACTION_DT
                if controller is not None
                else getattr(backend, "dt", ACTION_DT)
            )
        )
        period = getattr(controller, "sim_dt", getattr(backend, "dt", self.action_dt_s))
        backend_period = getattr(backend, "dt", period)
        if (
            not np.isfinite([period, backend_period]).all()
            or min(period, backend_period) <= 0
            or not np.isclose(period, backend_period, rtol=0, atol=1e-10)
        ):
            raise ValueError("Controller and backend control periods must be positive and match")
        ratio = self.action_dt_s / period
        if (
            not np.isfinite(ratio)
            or ratio < 1
            or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-10)
        ):
            raise ValueError(
                "action_dt_s must be a positive integer multiple of the control period"
            )
        self._control_steps = round(ratio)
        self.max_steps = max_steps
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode
        self.camera = camera
        self._steps = 0
        lower = np.asarray(backend.robot.lower, dtype=np.float32)
        upper = np.asarray(backend.robot.upper, dtype=np.float32)
        self.joint_space = spaces.Box(lower, upper, dtype=np.float32)
        self.action_space = (
            DualActionSpace(controller)
            if isinstance(controller, DualModeController)
            else self.joint_space
        )
        self.observation_space = spaces.Dict(
            {
                "qpos": spaces.Box(lower, upper, dtype=np.float32),
                "qvel": spaces.Box(-np.inf, np.inf, shape=lower.shape, dtype=np.float32),
                "sim_time": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float64),
            }
        )

    def _observation(self, state: Any) -> dict[str, np.ndarray]:
        qpos = np.asarray(state.qpos, dtype=np.float32)
        qvel = np.asarray(state.qvel, dtype=np.float32)
        if (
            qpos.ndim != 1
            or qpos.shape != self.joint_space.shape
            or qvel.shape != qpos.shape
            or not np.isfinite(qpos).all()
            or not np.isfinite(qvel).all()
            or not np.isfinite(float(state.time))
        ):
            raise ValueError("backend returned invalid finite qpos/qvel state")
        return {
            "qpos": qpos,
            "qvel": qvel,
            "sim_time": np.asarray([state.time], dtype=np.float64),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._steps = 0
        reset = self.backend.reset
        parameters = inspect.signature(reset).parameters
        if "seed" in parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            state = reset(seed=seed, options=options)
        else:
            state = reset()
        if self.controller is not None:
            self.controller.reset()
        return self._observation(state), {
            "seed": seed,
            "options": options or {},
            "backend_seed": seed,
        }

    def step(self, action: Any):
        from ..policies.adapter import validate_action_frame

        validate_action_frame(action, self.action_coordinate_frame)
        if isinstance(action, dict) and not hasattr(self.controller, "execute"):
            raise ValueError("Tagged joint/eef actions require a DualModeController")
        if not isinstance(action, dict):
            action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("action is outside the configured robot space")
        if self.controller is None:
            state = self.backend.step(np.asarray(action, dtype=np.float32))
            for _ in range(1, self._control_steps):
                state = self.backend.step(np.asarray(action, dtype=np.float32))
        elif isinstance(action, dict):
            state = self.controller.execute(self.backend, action, self.action_dt_s)
        else:
            state = self.controller.execute_target(self.backend, action, self.action_dt_s)
        self._steps += 1
        reward_fn = getattr(self.backend, "reward", None)
        terminated_fn = getattr(self.backend, "terminated", None)
        reward = float(reward_fn(state)) if callable(reward_fn) else 0.0  # type: ignore[arg-type]
        terminated = bool(terminated_fn(state)) if callable(terminated_fn) else False
        truncated = self._steps >= self.max_steps
        return (
            self._observation(state),
            reward,
            terminated,
            truncated,
            {
                "sim_time": float(state.time),
                "task_success": terminated and reward > 0
                if callable(terminated_fn) and callable(reward_fn)
                else None,
                "ik_reports": getattr(self.controller, "ik_reports", []),
            },
        )

    def render(self):
        if self.render_mode is None:
            return None
        images = self.backend.capture()
        if self.camera is not None:
            if self.camera not in images:
                raise ValueError(f"requested camera is unavailable: {self.camera}")
            return images[self.camera]
        return images[sorted(images)[0]] if images else None

    def close(self) -> None:
        self.backend.close()
