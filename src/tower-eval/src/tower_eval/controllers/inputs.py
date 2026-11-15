# SPDX-License-Identifier: Apache-2.0
"""Read tagged parameter files and raw HDF5 trajectories without starting a simulator."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

from ..constants import CONTROLLER_DT, GRIPPER_CHANNELS


def finite_array(value, shape, label):
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return array.copy()


def tcp_poses(value):
    poses = finite_array(value, (2, 7), "TCP poses")
    norms = np.linalg.norm(poses[:, 3:], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=0):
        raise ValueError("TCP orientations must be unit xyzw quaternions")
    poses[:, 3:] /= norms[:, None]
    return poses


@dataclass
class ControlSequence:
    actions: list
    dt_s: float
    metadata: dict = field(default_factory=dict)


def load_parameters(path):
    """JSON: {dt_s: seconds, actions: [tagged joint/eef parameter objects]}.

    Each row explicitly selects its input type. A sequence may switch modes.
    Robot-specific limits are checked by DualModeController before execution.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {"dt_s", "actions"}:
        raise ValueError("Parameter file requires exactly dt_s and actions")
    dt = float(data["dt_s"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt_s must be finite and positive")
    if not isinstance(data["actions"], list) or not data["actions"]:
        raise ValueError("actions must be a nonempty list")
    for action in data["actions"]:
        validate_input(action)
    return ControlSequence(
        data["actions"], dt, {"source": str(Path(path).resolve()), "format": "tagged_parameters"}
    )


def validate_input(action):
    if not isinstance(action, dict):
        raise ValueError("Each action must be a tagged dictionary")
    mode = action.get("input_space")
    expected = {
        "joint": {"input_space", "joints", "gripper"},
        "eef": {"input_space", "poses", "gripper", "frame"},
    }
    if not isinstance(mode, str) or mode not in expected or set(action) != expected[mode]:
        raise ValueError("Action must explicitly select joint/eef and its exact fields")
    widths = finite_array(action["gripper"], (2,), "Gripper widths")
    if np.any(widths < -1e-7):
        raise ValueError("Gripper widths must be nonnegative")
    if mode == "joint":
        values = finite_array(action["joints"], (2, 7), "Joint angles")
    else:
        values = tcp_poses(action["poses"])
        if action["frame"] not in ("arm_base", "environment_world"):
            raise ValueError("Unsupported EEF frame")
    return mode, values, widths


def load_raw(
    path,
    *,
    input_space,
    eef_dataset="observations/eef",
    joint_dataset="sim/controller_qpos",
    gripper_dataset="sim/controller_qpos",
):
    """Import raw 30 Hz actions with explicit state/action time alignment.

    Measured EEF uses row i+1 as the desired pose at the end of interval i.
    Leader EEF and joint/width commands use interval-start row i. The terminal
    action is never dispatched. No images or future joint angles feed EEF IK.
    """
    import h5py

    if input_space not in ("joint", "eef"):
        raise ValueError("input_space must explicitly be joint or eef")
    if eef_dataset not in ("observations/eef", "leader/eef"):
        raise ValueError("Unsupported EEF source")
    if joint_dataset not in ("leader/qpos", "sim/controller_qpos"):
        raise ValueError("Joint source must contain absolute commands")
    if gripper_dataset not in ("leader/qpos", "sim/controller_qpos"):
        raise ValueError("Gripper source must contain absolute width commands")
    with h5py.File(path, "r") as h:
        if json.loads(cast(str, h.attrs["arm_order"])) != ["left", "back"]:
            raise ValueError("Raw arm order must be left, back")
        if json.loads(cast(str, h.attrs["joint_names"])) != [f"joint{i}" for i in range(1, 8)] + [
            "gripper"
        ]:
            raise ValueError("Raw joint order mismatch")
        if not h.attrs.get("recording_complete", False):
            raise ValueError("Raw recording is incomplete")
        if (
            h.attrs.get("action_alignment")
            != "row observation precedes action interval; terminal row action_valid=false"
        ):
            raise ValueError("Unsupported raw state/action alignment")
        times = np.asarray(cast(h5py.Dataset, h["sim/time"])[:], dtype=float)
        n = len(times)
        if (
            n < 2
            or int(cast(int, h.attrs["valid_frames"])) != n
            or not np.isfinite(times).all()
            or not np.allclose(np.diff(times), CONTROLLER_DT, rtol=0, atol=1e-8)
        ):
            raise ValueError("Expected a complete uniformly sampled 30 Hz trajectory")
        if not np.array_equal(cast(h5py.Dataset, h["sim/action_valid"])[:], np.arange(n) < n - 1):
            raise ValueError("Raw terminal action must be invalid")
        if np.any(cast(h5py.Dataset, h["lifter/action_mm"])[:] != 0):
            raise ValueError("This replay supports stationary lifters only")
        widths = finite_array(
            cast(h5py.Dataset, h[gripper_dataset])[:], (n, 16), "Gripper command source"
        )[:, list(GRIPPER_CHANNELS)]
        if input_space == "eef":
            ds = cast(h5py.Dataset, h[eef_dataset])
            if ds.attrs.get("frame") != "arm_base":
                raise ValueError("Raw EEF source must declare arm_base")
            expected_fields = [
                f"{arm}_{field}"
                for arm in ("left", "back")
                for field in ("x", "y", "z", "qx", "qy", "qz", "qw")
            ]
            if json.loads(cast(str, ds.attrs["fields"])) != expected_fields:
                raise ValueError("Raw EEF field order mismatch")
            poses = finite_array(ds[:], (n, 14), "Raw EEF").reshape(n, 2, 7)
            offset = 1 if eef_dataset == "observations/eef" else 0
            actions = [
                dict(
                    input_space="eef",
                    poses=tcp_poses(poses[i + offset]).tolist(),
                    gripper=widths[i].tolist(),
                    frame="arm_base",
                )
                for i in range(n - 1)
            ]
        else:
            joints = finite_array(
                cast(h5py.Dataset, h[joint_dataset])[:], (n, 16), "Raw joint commands"
            )
            actions = [
                dict(
                    input_space="joint",
                    joints=joints[i, [*range(7), *range(8, 15)]].reshape(2, 7).tolist(),
                    gripper=widths[i].tolist(),
                )
                for i in range(n - 1)
            ]
        metadata = dict(
            source=str(Path(path).resolve()),
            format="raw_hdf5",
            input_space=input_space,
            source_dataset=eef_dataset if input_space == "eef" else joint_dataset,
            source_time_origin_s=float(times[0]),
            gripper_dataset=gripper_dataset,
            source_frames=n,
            control_intervals=n - 1,
            eef_sampling="interval_end"
            if input_space == "eef" and offset == 1
            else "interval_start",
            initial_qpos=np.asarray(
                cast(h5py.Dataset, h["observations/qpos"])[0], dtype=float
            ).tolist(),
            initial_blocks=np.asarray(
                cast(h5py.Dataset, h["sim/block_poses"])[0], dtype=float
            ).tolist(),
            block_ids=json.loads(cast(str, cast(h5py.Group, h["sim"]).attrs["block_ids"])),
        )
    for action in actions:
        validate_input(action)
    return ControlSequence(actions, CONTROLLER_DT, metadata)
