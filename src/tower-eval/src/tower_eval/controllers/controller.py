# SPDX-License-Identifier: Apache-2.0
"""Bimanual position targets with optional interpolation and legacy gripper profiles.

The default interpolation reproduces RoboDojo's `process_control_info`
(https://github.com/RoboDojo-Benchmark/RoboDojo, MIT, commit
ee67a1468510da7624a089164402359f2afc72c8): targets ramp linearly over the first
80% of control steps, then hold. It is a reimplementation; no RoboDojo code is vendored.
This module has no simulator, task, or legacy dependencies.
"""

import math
import warnings
from dataclasses import dataclass, fields

import numpy as np

from ..constants import ACTION_DT, CONTROLLER_DT


def _index(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("Indices must be integers")
    if not 0 <= value < 16:
        raise ValueError("Indices must be in [0,16)")
    return int(value)


def _vector(value, label):
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    result = np.asarray(value, dtype=float)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite vector (16,)")
    return result.copy()


@dataclass(frozen=True)
class GripperSpec:
    index: int
    lower: float
    upper: float

    def __post_init__(self):
        object.__setattr__(self, "index", _index(self.index))
        lower, upper = float(self.lower), float(self.upper)
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("Gripper travel must be finite and strictly positive")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class RobotSpec:
    names: tuple[str, ...]
    joint_indices: tuple[int, ...]
    grippers: tuple[GripperSpec, ...]
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        names = tuple(self.names)
        if (
            len(names) != 16
            or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != 16
        ):
            raise ValueError("Expected 16 unique nonempty channel names")
        joints = tuple(_index(i) for i in self.joint_indices)
        grips = tuple(self.grippers)
        if (
            len(joints) != 14
            or len(grips) != 2
            or any(not isinstance(g, GripperSpec) for g in grips)
        ):
            raise ValueError("Expected 14 joint indices and two GripperSpec entries")
        if sorted(joints + tuple(g.index for g in grips)) != list(range(16)):
            raise ValueError("Joint and gripper indices must partition all 16 channels")
        lower, upper = _vector(self.lower, "Lower bounds"), _vector(self.upper, "Upper bounds")
        if np.any(lower > upper):
            raise ValueError("Lower bounds exceed upper bounds")
        lower.setflags(write=False)
        upper.setflags(write=False)
        for key, value in [
            ("names", names),
            ("joint_indices", joints),
            ("grippers", grips),
            ("lower", lower),
            ("upper", upper),
        ]:
            object.__setattr__(self, key, value)

    @classmethod
    def from_dict(cls, config):
        """Read explicit names, joint_indices, grippers, lower, and upper keys."""
        if not isinstance(config, dict):
            raise ValueError("Robot configuration must be a dictionary")
        unknown = set(config) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown robot fields: {sorted(unknown)}")
        try:
            return cls(
                names=config["names"],
                joint_indices=config["joint_indices"],
                grippers=tuple(GripperSpec(**g) for g in config["grippers"]),
                lower=config["lower"],
                upper=config["upper"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid robot configuration: {error}") from error


_UNSET = object()


class Controller:
    def __init__(
        self,
        robot,
        interpolation_fraction=0.8,
        gripper_step_fraction=_UNSET,
        sim_dt=CONTROLLER_DT,
        arm_deadband_rad=0.0,
        gripper_control_profile=_UNSET,
        gripper_open_speed_m_s=0.16,
        gripper_close_speed_m_s=0.09,
        gripper_contact_width_m=0.045,
        gripper_contact_close_speed_m_s=0.01,
        gripper_hysteresis_close_score=None,
        gripper_hysteresis_open_score=None,
        gripper_model_closed_widths_m=None,
        gripper_model_open_widths_m=None,
        gripper_closed_targets_m=None,
        gripper_open_targets_m=None,
        gripper_contact_hold_force_n=None,
        hold_back_arm=False,
        arm_speed_rad_s=None,
    ):
        if not isinstance(robot, RobotSpec):
            raise ValueError("Expected RobotSpec")
        if gripper_step_fraction is _UNSET:
            gripper_step_fraction = 0.2
        else:
            warnings.warn(
                "gripper_step_fraction is deprecated and has no effect; omit it.",
                FutureWarning,
                stacklevel=2,
            )
        if gripper_control_profile is _UNSET:
            gripper_control_profile = "measured_fraction"
        elif gripper_control_profile == "measured_fraction":
            warnings.warn(
                "measured_fraction is deprecated; use direct_absolute.", FutureWarning, stacklevel=2
            )
        values = [
            interpolation_fraction,
            gripper_step_fraction,
            sim_dt,
            arm_deadband_rad,
            gripper_open_speed_m_s,
            gripper_close_speed_m_s,
            gripper_contact_width_m,
            gripper_contact_close_speed_m_s,
        ]
        if (
            not np.isfinite(values).all()
            or not 0 <= interpolation_fraction <= 1
            or not 0 < gripper_step_fraction <= 1
            or sim_dt <= 0
            or arm_deadband_rad < 0
        ):
            raise ValueError("Invalid controller fractions or simulation timestep")
        if gripper_control_profile not in (
            "direct_absolute",
            "measured_fraction",
            "generator_directional_v1",
            "generator_hysteresis_v1",
        ):
            raise ValueError("Unknown gripper control profile")
        if (
            gripper_open_speed_m_s <= 0
            or gripper_close_speed_m_s <= 0
            or gripper_contact_close_speed_m_s <= 0
        ):
            raise ValueError("Generator gripper speeds must be positive")
        if gripper_control_profile in (
            "generator_directional_v1",
            "generator_hysteresis_v1",
        ) and any(not grip.lower < gripper_contact_width_m < grip.upper for grip in robot.grippers):
            raise ValueError("Generator gripper contact width must lie inside physical travel")
        hysteresis_values = (
            gripper_model_closed_widths_m,
            gripper_model_open_widths_m,
            gripper_closed_targets_m,
            gripper_open_targets_m,
        )
        if gripper_control_profile == "generator_hysteresis_v1":
            scores = np.asarray(
                [gripper_hysteresis_close_score, gripper_hysteresis_open_score], dtype=float
            )
            if not np.isfinite(scores).all() or not 0 <= scores[1] < scores[0] <= 1:
                raise ValueError("Gripper hysteresis scores must satisfy 0 <= open < close <= 1")
            parsed = []
            for label, value in zip(
                ("model closed widths", "model open widths", "closed targets", "open targets"),
                hysteresis_values,
            ):
                array = np.asarray(value, dtype=float)
                if array.shape != (len(robot.grippers),) or not np.isfinite(array).all():
                    raise ValueError(f"Gripper {label} must contain one finite value per gripper")
                parsed.append(array.copy())
            model_closed, model_open, closed_targets, open_targets = parsed
            if np.any(model_closed >= model_open):
                raise ValueError("Gripper model closed widths must be below open widths")
            for slot, grip in enumerate(robot.grippers):
                if (
                    not grip.lower <= closed_targets[slot] <= grip.upper
                    or not grip.lower <= open_targets[slot] <= grip.upper
                    or closed_targets[slot] >= open_targets[slot]
                ):
                    raise ValueError(
                        "Gripper hysteresis targets must be ordered inside physical travel"
                    )
        else:
            model_closed = model_open = closed_targets = open_targets = None
        if arm_speed_rad_s is not None and (
            not np.isfinite(arm_speed_rad_s) or arm_speed_rad_s <= 0
        ):
            raise ValueError("Arm speed must be positive or None")
        self.arm_speed_rad_s = arm_speed_rad_s
        self._limited_reference = None
        self.robot = robot
        self.interpolation_fraction = float(interpolation_fraction)
        # Retained for configuration compatibility; measured-step clipping is disabled.
        self.gripper_step_fraction = float(gripper_step_fraction)
        self.sim_dt = float(sim_dt)
        self.arm_deadband_rad = float(arm_deadband_rad)
        self.gripper_control_profile = gripper_control_profile
        self.gripper_open_speed_m_s = float(gripper_open_speed_m_s)
        self.gripper_close_speed_m_s = float(gripper_close_speed_m_s)
        self.gripper_contact_width_m = float(gripper_contact_width_m)
        self.gripper_contact_close_speed_m_s = float(gripper_contact_close_speed_m_s)
        self.gripper_hysteresis_close_score = (
            None
            if gripper_hysteresis_close_score is None
            else float(gripper_hysteresis_close_score)
        )
        self.gripper_hysteresis_open_score = (
            None if gripper_hysteresis_open_score is None else float(gripper_hysteresis_open_score)
        )
        self.gripper_model_closed_widths_m = model_closed
        self.gripper_model_open_widths_m = model_open
        self.gripper_closed_targets_m = closed_targets
        self.gripper_open_targets_m = open_targets
        self.gripper_contact_hold_force_n = gripper_contact_hold_force_n
        if not isinstance(hold_back_arm, bool):
            raise ValueError("hold_back_arm must be boolean")
        self.hold_back_arm = hold_back_arm
        if gripper_contact_hold_force_n is not None:
            if (
                gripper_control_profile != "generator_hysteresis_v1"
                or not np.isfinite(gripper_contact_hold_force_n)
                or gripper_contact_hold_force_n <= 0
                or sim_dt != CONTROLLER_DT
            ):
                raise ValueError(
                    "Contact hold requires positive force, hysteresis intent and 30 Hz"
                )
        self.reset()

    def reset(self):
        """Forget held targets between episodes; initialize from first measured state."""
        self._applied_target = None
        self._limited_reference = None
        self._arm_reference = None
        self._gripper_reference = None
        self._gripper_closed_state = None
        self._contact_closures = {}
        self._back_hold = None

    @staticmethod
    def _state(state):
        qpos = _vector(state.qpos, "Measured qpos")
        time = float(state.time)
        if not np.isfinite(time) or time < 0:
            raise ValueError("Measured simulation time must be finite and nonnegative")
        return qpos, time

    @property
    def applied_target(self):
        """Last target submitted to sim.step, or None; step completion is not implied."""
        return None if self._applied_target is None else self._applied_target.copy()

    def execute_target(self, sim, target16, action_dt=ACTION_DT, on_step=None, on_dispatch=None):
        raw = _vector(target16, "Target")
        final_state = None
        for state in self._execute_targets(sim, lambda state: raw, action_dt, on_step, on_dispatch):
            final_state = state
        return final_state

    def _action_steps(self, action_dt):
        """Validate an action's duration and return its simulation tick count."""
        duration = float(action_dt)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Action timestep must be finite and positive")
        ratio = duration / self.sim_dt
        if not np.isfinite(ratio):
            raise ValueError("Action/simulation timestep ratio must be finite")
        count = round(ratio)
        if count < 1 or not math.isclose(ratio, count, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Action timestep must be an integer multiple of simulation timestep")
        return count

    def _execute_targets(self, sim, target_for_state, action_dt, on_step=None, on_dispatch=None):
        """Own one action's schedule; obtain a validated target before each tick.

        Yield after dispatch and time validation so EEF can finish its tick report.
        The interpolation origin and deadband reference stay fixed for the action.
        """
        from ._grippers import _AbsoluteGripper, _DirectionalGripper, _HysteresisGripper

        gripper = {
            "direct_absolute": _AbsoluteGripper,
            "measured_fraction": _AbsoluteGripper,
            "generator_directional_v1": _DirectionalGripper,
            "generator_hysteresis_v1": _HysteresisGripper,
        }[self.gripper_control_profile]()
        count = self._action_steps(action_dt)
        for callback in (on_step, on_dispatch):
            if callback is not None and not callable(callback):
                raise ValueError("Callbacks must be callable")
        if self.gripper_contact_hold_force_n is not None:
            from ..simulator import FingerFeedbackBackend

            if not isinstance(sim, FingerFeedbackBackend):
                raise ValueError("Contact hold requires backend finger_feedback capability")
        state = sim.read()
        start, time = self._state(state)
        if self.hold_back_arm and self._back_hold is None:
            self._back_hold = start[8:16].copy()
        if self._gripper_reference is None:
            self._gripper_reference = start.copy()
            gripper.initialize(self, start)
        reference = start if self._arm_reference is None else self._arm_reference
        interpolated_steps = math.floor(self.interpolation_fraction * count)
        for step in range(count):
            raw = _vector(target_for_state(state), "Target")
            target = raw.copy()
            modifications = []
            for index in self.robot.joint_indices:
                if abs(raw[index] - reference[index]) < self.arm_deadband_rad:
                    target[index] = reference[index]
                    if target[index] != raw[index]:
                        modifications.append(
                            {
                                "index": index,
                                "kind": "arm_deadband",
                                "before": float(raw[index]),
                                "after": float(target[index]),
                                "threshold_rad": self.arm_deadband_rad,
                            }
                        )
            for grip in self.robot.grippers:
                target[grip.index] = np.clip(raw[grip.index], grip.lower, grip.upper)
                if target[grip.index] != raw[grip.index]:
                    modifications.append(
                        {
                            "index": grip.index,
                            "kind": "gripper_target_bounds",
                            "before": float(raw[grip.index]),
                            "after": float(target[grip.index]),
                        }
                    )
            measured, current_time = self._state(state)
            fraction = (step + 1) / (interpolated_steps + 1) if step < interpolated_steps else 1.0
            interpolated = start + fraction * (target - start)
            applied = interpolated.copy()
            for slot, grip in enumerate(self.robot.grippers):
                if not (self.hold_back_arm and slot == 1):
                    gripper.apply(
                        self,
                        sim,
                        slot,
                        grip,
                        target,
                        measured,
                        interpolated,
                        applied,
                        modifications,
                    )
            if self.hold_back_arm:
                modifications.append(
                    dict(
                        kind="diagnostic_back_arm_hold",
                        indices=list(range(8, 16)),
                        before=applied[8:16].tolist(),
                        after=self._back_hold.tolist(),
                    )
                )
                applied[8:16] = self._back_hold
            if self.arm_speed_rad_s is not None:
                indices = list(self.robot.joint_indices)
                ref = measured if self._limited_reference is None else self._limited_reference
                bound = self.arm_speed_rad_s * self.sim_dt
                limited = np.clip(
                    ref[indices] + np.clip(applied[indices] - ref[indices], -bound, bound),
                    self.robot.lower[indices],
                    self.robot.upper[indices],
                )
                if not np.allclose(limited, applied[indices], rtol=0, atol=1e-12):
                    modifications.append(
                        dict(kind="arm_speed_limit", max_speed_rad_s=self.arm_speed_rad_s)
                    )
                applied[indices] = limited
            violations = []
            for stage, values in [("raw_target", raw), ("applied_target", applied)]:
                for index in self.robot.joint_indices:
                    if (
                        values[index] < self.robot.lower[index]
                        or values[index] > self.robot.upper[index]
                    ):
                        violations.append(
                            {
                                "index": index,
                                "name": self.robot.names[index],
                                "stage": stage,
                                "value": float(values[index]),
                                "lower": float(self.robot.lower[index]),
                                "upper": float(self.robot.upper[index]),
                            }
                        )
            event = dict(
                raw_target=raw.copy(),
                interpolated_target=interpolated.copy(),
                applied_target=applied.copy(),
                modifications=modifications,
                arm_limit_violations=violations,
            )
            if on_dispatch is not None:
                on_dispatch(measured=state, **event)
            self._applied_target = applied.copy()
            state = sim.step(applied.copy())
            self._limited_reference = applied.copy()
            gripper.commit(self, applied)
            # Preserve returned state before validation, including simulator failures.
            if on_step is not None:
                on_step(measured=state, **event)
            _, time = self._state(state)
            if not math.isclose(time - current_time, self.sim_dt, rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError("Simulator time must advance by the configured timestep")
            yield state
        self._arm_reference = target.copy()
