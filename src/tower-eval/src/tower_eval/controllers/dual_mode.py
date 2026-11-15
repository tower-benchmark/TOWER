# SPDX-License-Identifier: Apache-2.0
"""Explicit Joint/EEF inputs sharing TOWER's absolute-position execution path.

EEF poses are TCP xyz+xyzw in SI units, ordered left then back. IK computes
absolute joint targets; no joint-delta encoding or hardware SDK is involved.
"""

from typing import Any, cast

import numpy as np

from ..constants import ACTION_DT
from .controller import Controller
from .inputs import finite_array, validate_input


class SimulatorKinematics:
    """Bridge to the initialized simulator's verified, TCP-calibrated Lula model.

    The simulator's fk/ik already applies gripper_base <-> TCP. Applying the
    calibration again here would double the offset. No SimulationApp is
    created, no simulator state is set, and no CAN device is opened here.
    """

    backend = "lula"

    def __init__(self, sim):
        if not hasattr(sim, "env") or not hasattr(sim.env, "lula"):
            raise RuntimeError("EEF mode requires initialized simulator kinematics")
        self.env = sim.env

    def to_base(self, arm, pose, frame):
        if frame == "arm_base":
            return pose.copy()
        if frame == "environment_world":
            return self.env.to_base(0, arm, pose.copy())
        raise ValueError("Unsupported EEF frame")

    def ik(self, arm, pose, seed):
        return self.env.lula.ik(pose.copy(), seed.copy())

    def fk(self, arm, joints):
        return self.env.lula.fk(joints.copy())


class IKFailure(RuntimeError):
    """No targets from the failed tick were dispatched; earlier physics remains."""

    def __init__(self, report):
        self.report = report
        super().__init__(f"EEF tick rejected: {report}")


class DualModeController(Controller):
    """Tagged absolute inputs sharing exactly the same joint/gripper driver."""

    def __init__(
        self,
        robot,
        *,
        kinematics=None,
        ik_position_tolerance_m=0.001,
        ik_orientation_tolerance_rad=0.01,
        **kwargs,
    ):
        self.kinematics = kinematics
        tolerances = [ik_position_tolerance_m, ik_orientation_tolerance_rad]
        if not np.isfinite(tolerances).all() or min(tolerances) <= 0:
            raise ValueError("IK tolerances must be positive")
        self.ik_position_tolerance_m, self.ik_orientation_tolerance_rad = tolerances
        super().__init__(robot, **kwargs)

    def reset(self):
        super().reset()
        self.last_ik_report = None
        self.ik_reports = []
        if self.kinematics is not None and hasattr(self.kinematics, "reset"):
            self.kinematics.reset()

    def validate(self, parameters):
        """Return parsed Joint/EEF values after checking this robot's limits."""
        mode, values, gripper = validate_input(parameters)
        for slot, (value, spec) in enumerate(zip(gripper, self.robot.grippers)):
            # HDF5 float32 represents 0.1 m as 0.10000000149 m. Accept only
            # serialization roundoff, then restore the exact physical bound.
            if not spec.lower - 1e-7 <= value <= spec.upper + 1e-7:
                raise ValueError("Gripper width outside physical travel")
            gripper[slot] = np.clip(value, spec.lower, spec.upper)
        if mode == "joint":
            joints = values
            indices = list(self.robot.joint_indices)
            if np.any(joints.ravel() < self.robot.lower[indices] - 1e-7) or np.any(
                joints.ravel() > self.robot.upper[indices] + 1e-7
            ):
                raise ValueError("Joint target outside articulation limits")
            joints = np.clip(
                joints.ravel(), self.robot.lower[indices], self.robot.upper[indices]
            ).reshape(2, 7)
            return mode, joints, gripper
        return mode, values, gripper

    def _target16(self, joints, gripper):
        target = np.empty(16)
        target[list(self.robot.joint_indices)] = joints.ravel()
        target[[g.index for g in self.robot.grippers]] = gripper
        return target

    def execute_target(self, sim, target16, action_dt=ACTION_DT, on_step=None, on_dispatch=None):
        self.last_ik_report = None
        self.ik_reports = []
        return super().execute_target(sim, target16, action_dt, on_step, on_dispatch)

    def execute(
        self, sim, parameters, action_dt=ACTION_DT, *, on_step=None, on_dispatch=None, on_ik=None
    ):
        import time

        from ..kinematics.poses import pose_error

        mode, values, gripper = self.validate(parameters)
        self._action_steps(action_dt)
        for callback in (on_step, on_dispatch, on_ik):
            if callback is not None and not callable(callback):
                raise ValueError("Callbacks must be callable")
        if mode == "joint":
            return self.execute_target(
                sim, self._target16(values, gripper), action_dt, on_step, on_dispatch
            )
        kinematics = self.kinematics
        if kinematics is None:
            raise RuntimeError(
                "EEF requires an explicitly configured kinematics provider; no fallback"
            )
        self.ik_reports = []
        started = 0.0

        def target_for_state(state):
            nonlocal started
            started = time.perf_counter()
            measured, _ = self._state(state)
            report: dict[str, Any] = dict(
                backend=kinematics.backend,
                input_space="eef",
                input=parameters,
                sim_time=float(state.time),
                arms=[],
                success=False,
            )
            self.last_ik_report = report
            self.ik_reports.append(report)
            joints = []
            try:
                for arm in range(2):
                    indices = list(self.robot.joint_indices[arm * 7 : (arm + 1) * 7])
                    seed = measured[indices].copy()
                    target = kinematics.to_base(arm, values[arm], parameters["frame"])
                    arm_started = time.perf_counter()
                    solution, success = kinematics.ik(arm, target, seed)
                    detail: dict[str, Any] = dict(
                        arm=("left", "back")[arm],
                        seed=seed.tolist(),
                        target_base=target.tolist(),
                        success=bool(success),
                        wall_s=time.perf_counter() - arm_started,
                    )
                    if hasattr(kinematics, "report"):
                        detail.update(kinematics.report(arm))
                    candidate = np.asarray(solution, dtype=float)
                    detail["candidate"] = (
                        candidate.tolist() if np.isfinite(candidate).all() else None
                    )
                    report["arms"].append(detail)
                    if not success:
                        raise RuntimeError("Solver rejected target")
                    solution = finite_array(solution, (7,), "IK joint solution")
                    detail["solution"] = solution.tolist()
                    if np.any(solution < self.robot.lower[indices] - 1e-8) or np.any(
                        solution > self.robot.upper[indices] + 1e-8
                    ):
                        raise RuntimeError("IK solution outside articulation limits")
                    position, orientation = pose_error(kinematics.fk(arm, solution), target)
                    detail.update(position_error_m=position, orientation_error_rad=orientation)
                    if (
                        position > self.ik_position_tolerance_m
                        or orientation > self.ik_orientation_tolerance_rad
                    ):
                        raise RuntimeError("Independent FK residual exceeds tolerance")
                    joints.append(solution)
            except Exception as exc:
                report.update(failure=str(exc), ik_wall_s=time.perf_counter() - started)
                if on_ik is not None:
                    on_ik(report)
                raise IKFailure(report) from exc
            report.update(
                success=True,
                ik_wall_s=time.perf_counter() - started,
                ik_target=self._target16(np.asarray(joints), gripper).tolist(),
            )
            return report["ik_target"]

        def dispatched(**event):
            report = cast(dict[str, Any], self.last_ik_report)
            applied = event["applied_target"]
            report["applied_target"] = applied.tolist()
            report["modifications"] = event["modifications"]
            report["limited_target_errors"] = [
                pose_error(
                    kinematics.fk(a, applied[list(self.robot.joint_indices[a * 7 : (a + 1) * 7])]),
                    np.asarray(report["arms"][a]["target_base"]),
                )
                for a in range(2)
            ]
            if on_dispatch is not None:
                on_dispatch(**event)

        for state in self._execute_targets(
            sim, target_for_state, action_dt, on_step=on_step, on_dispatch=dispatched
        ):
            report = cast(dict[str, Any], self.last_ik_report)
            report["tick_wall_s"] = time.perf_counter() - started
            if hasattr(state, "eef"):
                report["measured_tcp_base"] = state.eef.tolist()
                report["tracking_errors"] = [
                    pose_error(state.eef[a], np.asarray(report["arms"][a]["target_base"]))
                    for a in range(2)
                ]
            if on_ik is not None:
                on_ik(report)
        return state
