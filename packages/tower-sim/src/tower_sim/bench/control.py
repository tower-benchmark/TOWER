"""Episode-local calibrated command scheduling; no simulator or model imports."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from nero_kinematics.fk import urdf
from nero_kinematics.profiles import COMMAND_BOUNDS, SIMULATION_OBSERVATION_TOLERANCE, load_profile

GENERATION_LOWER = np.array(COMMAND_BOUNDS.lower_rad)
GENERATION_UPPER = np.array(COMMAND_BOUNDS.upper_rad)
_PROFILE = load_profile("command")
KILL_LOWER = np.asarray(_PROFILE["kill_line"]["lower_rad"])
KILL_UPPER = np.asarray(_PROFILE["kill_line"]["upper_rad"])
OBSERVED_TRACKING_TOLERANCE_RAD = SIMULATION_OBSERVATION_TOLERANCE.radians
_URDF_LIMITS = urdf.limits()
for _bound in (GENERATION_LOWER, GENERATION_UPPER, KILL_LOWER, KILL_UPPER, _URDF_LIMITS):
    _bound.flags.writeable = False


class IKSolver(Protocol):
    def solve(self, poses: Any, seed: Any, dt_s: float = 0.1) -> dict[str, Any]: ...


def validate_encoder_joints(q, *, kind="Joint state/command"):
    # Storage roundoff only; callers cannot supply observation slack.
    tolerance = 1e-7
    q = np.asarray(q, float)
    if (
        q.shape[-1:] != (7,)
        or not np.isfinite(q).all()
        or np.any(q < GENERATION_LOWER - tolerance)
        or np.any(q > GENERATION_UPPER + tolerance)
    ):
        raise PolicyActionError(kind + " outside generation real-soft intersection bounds")


def validate_observed_encoder_joints(q):
    q = np.asarray(q, float)
    tol = OBSERVED_TRACKING_TOLERANCE_RAD
    if (
        q.shape[-1:] != (7,)
        or not np.isfinite(q).all()
        or np.any(q < KILL_LOWER - tol)
        or np.any(q > KILL_UPPER + tol)
    ):
        raise PolicyActionError(
            "Observed joint state outside shared kill line plus tracking tolerance"
        )


class PolicyActionError(ValueError):
    """A finite model output cannot be executed under the frozen contract."""

    diagnostics: list[dict[str, Any]]


def validate_physical_targets(targets, limits):
    """Check calibrated actuator coordinates against live articulation limits."""
    values, bounds = np.asarray(targets, float), np.asarray(limits, float)
    if (
        bounds.shape != (16, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ValueError("Invalid runtime physical joint limits")
    if values.shape[-1:] != (16,) or not np.isfinite(values).all():
        raise PolicyActionError("Invalid calibrated physical targets")
    # Arm tolerances stay 2e-6 rad; aperture covers two finger travels.
    tolerance = np.full(16, 2e-6)
    tolerance[[7, 15]] = 4e-6
    if np.any(values < bounds[:, 0] - tolerance) or np.any(values > bounds[:, 1] + tolerance):
        raise PolicyActionError("Calibrated physical target outside runtime articulation limits")


def validate_observed_joints(physical, compensation, limits):
    """Check measured arm joints in both physical and encoder coordinates."""
    physical = np.asarray(physical, float)
    arm_ids = np.r_[np.arange(7), np.arange(8, 15)]
    bounds = np.asarray(limits, float)
    if not np.isfinite(physical[arm_ids]).all():
        raise PolicyActionError("Non-finite observed arm joint")
    tolerance = OBSERVED_TRACKING_TOLERANCE_RAD
    if np.any(physical[arm_ids] < bounds[arm_ids, 0] - tolerance) or np.any(
        physical[arm_ids] > bounds[arm_ids, 1] + tolerance
    ):
        raise PolicyActionError(
            "Observed physical arm joint outside simulation URDF bounds plus tracking tolerance"
        )
    encoder = np.stack(
        [
            compensation.encoder_from_physical(
                physical[a * 8 : a * 8 + 7], physical[a * 8 + 7], name
            )
            for a, name in enumerate(("left", "back"))
        ]
    )
    validate_observed_encoder_joints(encoder)
    return encoder


class CommandQueue:
    """10 Hz endpoints, 200 Hz interpolation/fitted delays, plus 80 ms drive delay.

    The first predicted endpoint belongs to observation time + 100 ms. Future
    chunks are never read. Each instance is owned by one episode and starts from
    its prior commanded encoder state. Calibration is applied exactly once.
    """

    def __init__(self, home, delays, physical_from_encoder, physical_limits=None):
        self.times = [0.0]
        self.commands = [np.asarray(home, float).copy()]
        self.delays = np.asarray(delays, float)
        if self.commands[0].shape != (16,) or self.delays.shape != (2, 7):
            raise ValueError("Invalid home/delay shape")
        if (
            not np.isfinite(self.commands).all()
            or not np.isfinite(self.delays).all()
            or np.any(self.delays < 0)
        ):
            raise ValueError("Invalid command/delay values")
        self.convert = physical_from_encoder
        self.physical_limits = physical_limits
        self.cached_tick = None
        self.cached = None
        self._interpolation_times = None
        self._interpolation_columns = None

    def enqueue(self, observation_time, endpoints):
        endpoints = np.asarray(endpoints, float)
        if (
            endpoints.ndim != 2
            or endpoints.shape[1] != 16
            or not len(endpoints)
            or not np.isfinite(endpoints).all()
        ):
            raise PolicyActionError("Expected finite encoder endpoints [N,16]")
        if abs(observation_time - self.times[-1]) > 1e-8:
            raise ValueError("Chunk overlaps or skips the command timeline")
        self.times.extend(observation_time + 0.1 * np.arange(1, len(endpoints) + 1))
        self.commands.extend(endpoints.copy())
        self._interpolation_times = self._interpolation_columns = None

    def target(self, physics_tick):
        command_tick = max(0, physics_tick - 32) // 2
        if self.cached_tick == command_tick:
            assert self.cached is not None
            return tuple(v.copy() for v in self.cached)
        stamp = command_tick / 200.0
        if stamp > self.times[-1] + 1e-8:
            raise ValueError("No known command endpoint for requested time")
        # The history only changes on enqueue, not on every 200 Hz control tick.
        # Contiguous columns also avoid np.interp copying strided joint histories.
        if self._interpolation_times is None or self._interpolation_columns is None:
            self._interpolation_times = np.asarray(self.times)
            self._interpolation_columns = np.ascontiguousarray(np.asarray(self.commands).T)
        encoder = np.array(
            [
                np.interp(
                    stamp - (self.delays[c // 8, c % 8] if c % 8 < 7 else 0.0),
                    self._interpolation_times,
                    self._interpolation_columns[c],
                )
                for c in range(16)
            ]
        )
        validate_encoder_joints(encoder.reshape(2, 8)[:, :7])
        physical = np.asarray(self.convert(encoder.copy()), float)
        if physical.shape != (16,) or not np.isfinite(physical).all():
            raise RuntimeError("Calibration produced invalid physical targets")
        if self.physical_limits is not None:
            validate_physical_targets(physical, self.physical_limits)
        self.cached_tick, self.cached = command_tick, (encoder.copy(), physical.copy())
        return encoder, physical


def encoder_endpoints(response, observation, *, solver: IKSolver | None = None, prefix=16):
    """Validate limits and solve EEF targets in encoder TCP coordinates."""
    if response.get("coordinate_frame") != "encoder_arm_base" or response.get(
        "j8_actuator_mask"
    ) != [False, False]:
        raise ValueError("Unsupported policy coordinate/J8 contract")
    if response.get("control_hz") != 10 or response.get("execute_steps") != prefix:
        raise ValueError("Policy timing differs from frozen protocol")
    # Aperture spans two fingers: each retains its existing 2 micrometre
    # actuator tolerance, hence 4 micrometres on total aperture. This accepts
    # the checkpoint decoder's 100.003771 mm endpoint without clipping commands.
    widths = np.asarray(response["gripper_width_m"], float)[:prefix]
    if (
        widths.shape != (prefix, 2)
        or not np.isfinite(widths).all()
        or np.any(widths < 0)
        or np.any(widths > 0.1 + 2 * 2e-6)
    ):
        raise PolicyActionError("Invalid gripper aperture")
    diagnostics = []
    if response["mode"] == "joint_absolute":
        joints = np.asarray(response["joint_encoder_targets"], float)[:prefix]
    elif response["mode"] == "eef_delta":
        poses = np.asarray(response["eef_encoder_targets"], float)[:prefix]
        if poses.shape != (prefix, 2, 4, 4) or not np.isfinite(poses).all():
            raise PolicyActionError("Malformed EEF targets")
        seed = np.asarray(observation["joint_encoder"], float).copy()
        validate_observed_encoder_joints(seed)
        if solver is None:
            raise ValueError("EEF commands require an explicit IK service")
        result = solver.solve(poses, seed, dt_s=0.1)
        diagnostics = result["diagnostics"]
        if not result["accepted"]:
            error = PolicyActionError(
                "Deployed generation-aligned IK rejected policy target: "
                + str(diagnostics[-1] if diagnostics else "no diagnostics")
            )
            error.diagnostics = diagnostics
            raise error
        joints = np.asarray(result["joints"], float)
    else:
        raise ValueError("Unknown policy action mode")
    if joints.shape != (prefix, 2, 7) or not np.isfinite(joints).all():
        raise PolicyActionError("Malformed joint targets")
    validate_encoder_joints(joints)
    limits = _URDF_LIMITS
    if np.any(joints < limits[:, 0] - 1e-7) or np.any(joints > limits[:, 1] + 1e-7):
        raise PolicyActionError("Joint target outside calibrated URDF limits")
    commands = np.concatenate((joints[:, 0], widths[:, :1], joints[:, 1], widths[:, 1:]), axis=1)
    return commands, diagnostics
