#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared command, dynamics, and video helpers for the live dual-arm runner."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from tower_sim.backend.resources import default_path

RUNNER_ROOT = default_path("replay_runner")
DEFAULT_EPISODE = Path("example_episode.hdf5")
DEFAULT_OUTPUT = RUNNER_ROOT / "outputs" / "episode_000000_viewport"
ARM_NAMES = tuple(f"joint{index}" for index in range(1, 8))
ARM_VELOCITY_LIMITS = np.asarray(
    [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    dtype=np.float64,
)
COMMAND_INTERPOLATIONS = ("zoh", "pchip")


def default_output_dir(episode: str | Path | None = None) -> Path:
    raise ValueError("Standalone replay requires an explicit output directory")


@dataclass(frozen=True)
class CommandSample:
    """One 120 Hz arm command plus a 30 Hz ZOH gripper command."""

    time_seconds: float
    source_index: int
    qpos: np.ndarray
    qvel: np.ndarray
    raw_arm_qvel: np.ndarray
    arm_velocity_clipped: np.ndarray
    arm_source_indices: np.ndarray


class CommandTrajectory:
    """Sample leader/qpos at the physics rate without changing its knots."""

    def __init__(
        self,
        actions: np.ndarray,
        source_hz: float,
        interpolation: str,
        arm_command_delays_seconds: Sequence[float] | None = None,
    ) -> None:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 8 or len(actions) < 1:
            raise ValueError(f"Expected leader/qpos with shape (N, 8), got {actions.shape}")
        if source_hz <= 0:
            raise ValueError("source_hz must be positive")
        if interpolation not in COMMAND_INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {COMMAND_INTERPOLATIONS}")
        self.actions = actions.copy()
        self.source_hz = float(source_hz)
        self.interpolation = interpolation
        if arm_command_delays_seconds is None:
            delays = np.zeros(7, dtype=np.float64)
        else:
            delays = np.asarray(arm_command_delays_seconds, dtype=np.float64)
        if delays.shape != (7,) or not np.isfinite(delays).all() or np.any(delays < 0):
            raise ValueError(
                "arm_command_delays_seconds must contain seven finite, non-negative values"
            )
        self.arm_command_delays_seconds = delays.copy()
        self.knot_times = np.arange(len(actions), dtype=np.float64) / source_hz
        self.end_time = float(self.knot_times[-1])
        self._position_interpolator: Any | None = None
        self._velocity_interpolator: Any | None = None
        if interpolation == "pchip" and len(actions) > 1:
            from scipy.interpolate import PchipInterpolator

            self._position_interpolator = PchipInterpolator(
                self.knot_times,
                self.actions[:, :7],
                axis=0,
                extrapolate=False,
            )
            self._velocity_interpolator = self._position_interpolator.derivative()

    def sample(self, time_seconds: float) -> CommandSample:
        """Return arm position/velocity and gripper ZOH at one time."""

        if not math.isfinite(time_seconds):
            raise ValueError("time_seconds must be finite")
        time_seconds = float(np.clip(time_seconds, 0.0, self.end_time))
        source_position = time_seconds * self.source_hz
        source_index = min(
            int(math.floor(source_position + 1e-9)),
            len(self.actions) - 1,
        )
        arm_times = np.clip(
            time_seconds - self.arm_command_delays_seconds,
            0.0,
            self.end_time,
        )
        arm_source_indices = np.minimum(
            np.floor(arm_times * self.source_hz + 1e-9).astype(np.int64),
            len(self.actions) - 1,
        )
        if self._position_interpolator is None:
            arm_qpos = np.asarray(
                [self.actions[arm_source_indices[index], index] for index in range(7)],
                dtype=np.float64,
            )
            raw_arm_qvel = np.zeros(7, dtype=np.float64)
        else:
            arm_qpos = np.asarray(
                [self._position_interpolator(arm_times[index])[index] for index in range(7)],
                dtype=np.float64,
            )
            raw_arm_qvel = np.asarray(
                [self._velocity_interpolator(arm_times[index])[index] for index in range(7)],
                dtype=np.float64,
            )
        arm_qvel = np.clip(
            raw_arm_qvel,
            -ARM_VELOCITY_LIMITS,
            ARM_VELOCITY_LIMITS,
        )
        qpos = np.empty(8, dtype=np.float64)
        qpos[:7] = arm_qpos
        # Gripper timing is deliberately not interpolated.
        qpos[7] = self.actions[source_index, 7]
        qvel = np.zeros(8, dtype=np.float64)
        qvel[:7] = arm_qvel
        return CommandSample(
            time_seconds=time_seconds,
            source_index=source_index,
            qpos=qpos,
            qvel=qvel,
            raw_arm_qvel=raw_arm_qvel,
            arm_velocity_clipped=(np.abs(raw_arm_qvel) > ARM_VELOCITY_LIMITS),
            arm_source_indices=arm_source_indices,
        )


def command_for_state(
    qpos: np.ndarray,
    qvel: np.ndarray,
    dof_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if qpos.shape != (8,):
        raise ValueError(f"Expected one qpos sample with shape (8,), got {qpos.shape}")
    if qvel.shape != (8,):
        raise ValueError(f"Expected one qvel sample with shape (8,), got {qvel.shape}")
    missing = [name for name in ARM_NAMES if name not in dof_names]
    if missing:
        raise ValueError(f"Robot is missing arm DOFs {missing}; available={dof_names}")

    width = float(np.clip(qpos[7], 0.0, 0.1))
    requested = {
        **{name: float(qpos[index]) for index, name in enumerate(ARM_NAMES)},
        "gripper": width,
        "gripper_joint1": width * 0.5,
        "gripper_joint2": -width * 0.5,
    }
    command = {name: value for name, value in requested.items() if name in dof_names}
    requested_velocities = {
        **{name: float(qvel[index]) for index, name in enumerate(ARM_NAMES)},
        "gripper": 0.0,
        "gripper_joint1": 0.0,
        "gripper_joint2": 0.0,
    }
    indices = np.asarray([dof_names.index(name) for name in command], dtype=np.int32)
    positions = np.asarray(list(command.values()), dtype=np.float64)
    velocities = np.asarray(
        [requested_velocities[name] for name in command],
        dtype=np.float64,
    )
    return indices, positions, velocities, command


class H264VideoWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        import av

        self._av = av
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=Fraction(str(fps)))
        self._stream.width, self._stream.height = size
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {
            "crf": "10",
            "preset": "fast",
            "tune": "film",
        }
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def write(self, bgr_frame: np.ndarray) -> None:
        if not self._opened:
            raise RuntimeError("Cannot write to a closed H.264 writer")
        frame = self._av.VideoFrame.from_ndarray(bgr_frame, format="bgr24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def release(self) -> None:
        if not self._opened:
            return
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._opened = False


def _video_metadata(paths: dict[str, Path]) -> dict[str, Any]:
    import cv2

    metadata = {}
    for name, path in paths.items():
        capture = cv2.VideoCapture(str(path))
        metadata[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        capture.release()
    return metadata


def _controller_gains(
    robot: Any,
    dof_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    kps, kvs = robot.get_articulation_controller().get_gains()
    kps = np.asarray(kps, dtype=np.float64)
    kvs = np.asarray(kvs, dtype=np.float64)
    if kps.shape != (len(dof_names),) or kvs.shape != (len(dof_names),):
        raise RuntimeError(
            "Composed USD gain arrays do not match articulation DOFs: "
            f"kp={kps.shape}, kv={kvs.shape}, dofs={len(dof_names)}"
        )
    report = {
        name: {
            "kp_runtime_per_rad": float(kps[dof_names.index(name)]),
            "kv_runtime_per_rad": float(kvs[dof_names.index(name)]),
        }
        for name in ARM_NAMES
    }
    return kps, kvs, report


def _as_numpy(value: Any) -> np.ndarray:
    """Convert an Isaac tensor backend value to a detached NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _articulation_dynamics_arrays(
    robot: Any,
    dof_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    """Read armature and three-term PhysX joint friction arrays."""

    view = robot._articulation_view
    armatures = _as_numpy(view.get_armatures()).astype(np.float64, copy=True)
    physics_view = view._physics_view
    if not hasattr(physics_view, "get_dof_friction_properties"):
        raise RuntimeError(
            "This Isaac Sim build does not expose three-term DOF friction properties"
        )
    frictions = _as_numpy(physics_view.get_dof_friction_properties()).astype(np.float64, copy=True)
    expected_armatures = (1, len(dof_names))
    expected_frictions = (1, len(dof_names), 3)
    if armatures.shape != expected_armatures:
        raise RuntimeError(
            f"Unexpected armature array shape {armatures.shape}; expected {expected_armatures}"
        )
    if frictions.shape != expected_frictions:
        raise RuntimeError(
            f"Unexpected friction array shape {frictions.shape}; expected {expected_frictions}"
        )
    report = {
        name: {
            "armature_kg_m2": float(armatures[0, dof_names.index(name)]),
            "static_friction": float(frictions[0, dof_names.index(name), 0]),
            "dynamic_friction": float(frictions[0, dof_names.index(name), 1]),
            "viscous_friction": float(frictions[0, dof_names.index(name), 2]),
        }
        for name in ARM_NAMES
    }
    return armatures, frictions, report


def _apply_arm_dynamics_parameters(
    robot: Any,
    dof_names: Sequence[str],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Apply runtime-only armature and PACE-style joint friction values."""

    missing = [name for name in ARM_NAMES if name not in dof_names]
    if missing:
        raise ValueError(f"Robot is missing arm DOFs: {missing}")
    indices = np.asarray([dof_names.index(name) for name in ARM_NAMES], dtype=np.int32)
    original_armatures, original_frictions, original_report = _articulation_dynamics_arrays(
        robot, dof_names
    )
    requested_armatures = original_armatures.copy()
    requested_frictions = original_frictions.copy()
    for name, index in zip(ARM_NAMES, indices, strict=True):
        values = parameters["joints"][name]
        requested_armatures[0, index] = values["armature_kg_m2"]
        # Match robot_alignment's IsaacLab simulator: one fitted Coulomb
        # coefficient is assigned to both PhysX static and dynamic terms.
        requested_frictions[0, index, 0] = values["coulomb_friction"]
        requested_frictions[0, index, 1] = values["coulomb_friction"]
        requested_frictions[0, index, 2] = values["viscous_friction"]

    view = robot._articulation_view
    view.set_armatures(requested_armatures[:, indices], joint_indices=indices)
    view._physics_view.set_dof_friction_properties(
        requested_frictions.astype(np.float32),
        np.asarray([0], dtype=np.int32),
    )
    active_armatures, active_frictions, active_report = _articulation_dynamics_arrays(
        robot, dof_names
    )
    untouched = np.ones(len(dof_names), dtype=bool)
    untouched[indices] = False
    if not (
        np.array_equal(
            active_armatures[:, untouched],
            original_armatures[:, untouched],
        )
        and np.array_equal(
            active_frictions[:, untouched, :],
            original_frictions[:, untouched, :],
        )
    ):
        raise RuntimeError("Applying arm dynamics changed a non-arm DOF")
    if not (
        np.allclose(
            active_armatures[:, indices],
            requested_armatures[:, indices],
            rtol=1e-6,
            atol=1e-7,
        )
        and np.allclose(
            active_frictions[:, indices, :],
            requested_frictions[:, indices, :],
            rtol=1e-6,
            atol=1e-7,
        )
    ):
        raise RuntimeError("PhysX did not retain the requested arm dynamics parameters")
    return {
        "parameter_file": parameters["path"],
        "parameter_file_sha256": parameters["sha256"],
        "mapping": {
            "coulomb_friction": [
                "physx_static_friction",
                "physx_dynamic_friction",
            ],
            "viscous_friction": "physx_viscous_friction",
        },
        "original": original_report,
        "active": active_report,
        "other_dof_dynamics_unchanged": True,
        "save_to_usd": False,
    }


def _articulation_limit_arrays(
    robot: Any,
) -> tuple[np.ndarray, np.ndarray]:
    efforts = np.asarray(
        robot.get_articulation_controller().get_max_efforts(),
        dtype=np.float64,
    )
    velocities = np.asarray(
        robot._articulation_view.get_joint_max_velocities(),
        dtype=np.float64,
    )
    if velocities.ndim == 2 and velocities.shape[0] == 1:
        velocities = velocities[0]
    return efforts, velocities


def _apply_gripper_finger_limits(
    robot: Any,
    dof_names: Sequence[str],
    effort_limit: float,
    velocity_limit: float,
) -> dict[str, Any]:
    """Apply IsaacLab-style runtime limits to the two linear finger DOFs."""

    finger_names = ("gripper_joint1", "gripper_joint2")
    missing = [name for name in finger_names if name not in dof_names]
    if missing:
        raise ValueError(f"Robot is missing gripper finger DOFs: {missing}")
    indices = np.asarray(
        [dof_names.index(name) for name in finger_names],
        dtype=np.int32,
    )
    original_efforts, original_velocities = _articulation_limit_arrays(robot)
    expected_shape = (len(dof_names),)
    if original_efforts.shape != expected_shape or original_velocities.shape != expected_shape:
        raise RuntimeError(
            "Articulation limit arrays do not match DOFs: "
            f"effort={original_efforts.shape}, "
            f"velocity={original_velocities.shape}, dofs={expected_shape}"
        )

    controller = robot.get_articulation_controller()
    controller.set_max_efforts(
        np.full(2, float(effort_limit), dtype=np.float32),
        joint_indices=indices,
    )
    robot._articulation_view.set_max_joint_velocities(
        np.full((1, 2), float(velocity_limit), dtype=np.float32),
        joint_indices=indices,
    )
    active_efforts, active_velocities = _articulation_limit_arrays(robot)
    untouched = np.ones(len(dof_names), dtype=bool)
    untouched[indices] = False
    if not (
        np.array_equal(active_efforts[untouched], original_efforts[untouched])
        and np.array_equal(
            active_velocities[untouched],
            original_velocities[untouched],
        )
    ):
        raise RuntimeError("Applying gripper limits changed another DOF")
    if not (
        np.allclose(active_efforts[indices], effort_limit, rtol=1e-6, atol=1e-6)
        and np.allclose(
            active_velocities[indices],
            velocity_limit,
            rtol=1e-6,
            atol=1e-6,
        )
    ):
        raise RuntimeError("Articulation did not retain requested gripper limits")
    return {
        "dofs": list(finger_names),
        "requested_effort_limit": float(effort_limit),
        "requested_velocity_limit_m_s": float(velocity_limit),
        "original_effort_limits": {
            name: float(original_efforts[index])
            for name, index in zip(finger_names, indices, strict=True)
        },
        "original_velocity_limits_m_s": {
            name: float(original_velocities[index])
            for name, index in zip(finger_names, indices, strict=True)
        },
        "active_effort_limits": {
            name: float(active_efforts[index])
            for name, index in zip(finger_names, indices, strict=True)
        },
        "active_velocity_limits_m_s": {
            name: float(active_velocities[index])
            for name, index in zip(finger_names, indices, strict=True)
        },
        "other_dof_limits_unchanged": True,
        "save_to_usd": False,
    }


class ReplayConfig:
    """Import-compatible placeholder for the retired single-arm replay runner."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Single-arm replay is not part of the public benchmark.")


def replay_hdf5(*args, **kwargs):
    """Import-compatible placeholder for the retired single-arm replay runner."""
    raise NotImplementedError("Single-arm replay is not part of the public benchmark.")
