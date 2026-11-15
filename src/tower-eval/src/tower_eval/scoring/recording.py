# SPDX-License-Identifier: Apache-2.0
"""Two explicit input contracts: persisted files and historical campaign rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from .constants import GRIPPER_CHANNELS
from .geometry import inverse, rotate
from .rules import DEFAULT_RULES


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class Recording:
    spec: dict
    times: np.ndarray
    poses: np.ndarray
    velocities: np.ndarray
    pinch: np.ndarray
    inverse_flange: np.ndarray
    effort: np.ndarray
    opening: np.ndarray
    status: dict
    hashes: dict
    finger_corners: np.ndarray | None = None


def from_files(run, rules=DEFAULT_RULES):
    import h5py

    run = Path(run)
    names = [
        "episode_spec.json",
        "episode/states.hdf5",
        "finger-link-poses.npz",
        "finger-feedback.npz",
        "finger_geometry.json",
        "status.json",
    ]
    hashes = {name: digest(run / name) for name in names}
    spec = json.loads((run / "episode_spec.json").read_text())
    status = json.loads((run / "status.json").read_text())
    with h5py.File(run / "episode/states.hdf5", "r") as f:
        times, poses, vel, qpos = (
            cast(np.ndarray, cast(h5py.Dataset, f[k])[:])
            for k in ("time", "blocks", "block_velocities", "qpos")
        )
    n, b = len(times), len(spec["blocks"])
    if (
        times.shape != (n,)
        or n < 2
        or poses.shape != (n, b, 7)
        or vel.shape != (n, b, 6)
        or qpos.shape != (n, 16)
        or not all(np.isfinite(v).all() for v in (times, poses, vel, qpos))
        or np.any(np.diff(times) <= 0)
        or times[0] < 0
        or times[0] > rules.maximum_sample_gap_s
        or np.diff(times).max() > rules.maximum_sample_gap_s + 1e-8
    ):
        raise ValueError("Malformed, incomplete, or non-contiguous state recording")
    final_time = float(status["simulated_seconds"])
    if not np.isfinite(final_time) or abs(final_time - times[-1]) > rules.maximum_sample_gap_s:
        raise ValueError("Recording does not cover the final evaluator state")
    if np.any(np.linalg.norm(poses[..., 3:], axis=-1) < 1e-8):
        raise ValueError("Invalid block quaternion")
    with np.load(run / "finger-link-poses.npz", allow_pickle=False) as f:
        fingers, finger_times = f["poses_m"], f["time"]
    with np.load(run / "finger-feedback.npz", allow_pickle=False) as f:
        effort, force_times = np.abs(f["forces_n"]), f["time"]
    if fingers.shape != (n, 2, 3, 7) or effort.shape != (n, 2, 2):
        raise ValueError("Incomplete finger evidence")
    for observed in (finger_times, force_times):
        if observed.shape != times.shape or not np.allclose(observed, times, atol=1e-8, rtol=0):
            raise ValueError("Finger evidence timestamps differ from states")
    if not np.isfinite(fingers).all() or not np.isfinite(effort).all():
        raise ValueError("Nonfinite finger evidence")
    finger_geometry = json.loads((run / "finger_geometry.json").read_text())
    offsets = np.asarray([g["pinch_center_in_flange_m"] for g in finger_geometry])
    if offsets.shape != (2, 3) or not np.isfinite(offsets).all():
        raise ValueError("Invalid pinch geometry")
    flange = fingers[:, :, 0]
    pinch = flange[..., :3] + rotate(flange[..., 3:], offsets)
    local_corners = np.asarray(
        [[g["distal_link_corners_m"] for g in arm["fingers"]] for arm in finger_geometry]
    )
    if local_corners.shape != (2, 2, 8, 3) or not np.isfinite(local_corners).all():
        raise ValueError("Invalid distal finger geometry")
    finger_corners = fingers[:, :, 1:, None, :3] + rotate(
        fingers[:, :, 1:, None, 3:], local_corners[None]
    )
    keep = times <= rules.time_limit_s + 1e-8
    return Recording(
        spec,
        times[keep],
        poses[keep],
        vel[keep],
        pinch[keep],
        inverse(flange[keep, :, 3:]),
        effort[keep],
        qpos[keep][:, GRIPPER_CHANNELS],
        status,
        hashes,
        finger_corners[keep],
    )


def from_campaign_rows(spec, rows, finger_geometry):
    """Convert BatchEvidence rows using the offline loader's coordinate mapping."""
    times = np.asarray([row["time"] for row in rows])
    if (
        len(times) < 2
        or times[0] != 0
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or np.max(np.diff(times)) > 0.1 + 1e-8
    ):
        raise ValueError("Online success requires contiguous physical evidence from time zero")
    fingers = np.asarray([row["finger_poses"] for row in rows])
    flange = fingers[:, :, 0]
    offsets = np.asarray([g["pinch_center_in_flange_m"] for g in finger_geometry])
    pinch = flange[..., :3] + rotate(flange[..., 3:], offsets)
    recording = Recording(
        spec,
        times,
        np.asarray([row["blocks"] for row in rows]),
        np.asarray([row["block_velocities"] for row in rows]),
        pinch,
        inverse(flange[..., 3:]),
        np.abs(np.asarray([row["finger_forces"] for row in rows])),
        np.asarray([row["qpos"] for row in rows])[:, GRIPPER_CHANNELS],
        dict(
            status="duration_reached",
            valid_trial=True,
            returncode=0,
            simulated_seconds=float(times[-1]),
        ),
        {},
    )
    if not all(
        np.isfinite(value).all()
        for value in (
            recording.poses,
            recording.velocities,
            recording.pinch,
            recording.inverse_flange,
            recording.effort,
            recording.opening,
        )
    ):
        raise ValueError("Nonfinite online physical evidence")
    return recording


# Compatibility names preserve both constructor contracts.
load_recording = from_files
from_physical_rows = from_campaign_rows


def from_campaign_files(run, task):
    """Reconstruct finalize_trial's rows/status from a persisted campaign recording.

    This deliberately does not call from_files: hashes and distal corners are
    absent in the campaign contract. Timestamp correspondence is still checked.
    """
    import h5py

    from ..tasks import benchmark_legacy_id

    task = benchmark_legacy_id(task)
    run = Path(run)
    with h5py.File(run / "episode/states.hdf5", "r") as stream:
        arrays = {
            name: cast(h5py.Dataset, stream[name])[:]
            for name in ("time", "blocks", "block_velocities", "qpos")
        }
    for filename, key, field in (
        ("finger-link-poses.npz", "finger_poses", "poses_m"),
        ("finger-feedback.npz", "finger_forces", "forces_n"),
    ):
        with np.load(run / filename, allow_pickle=False) as stream:
            if not np.array_equal(stream["time"], arrays["time"]):
                raise ValueError("Finger evidence timestamps differ from states")
            arrays[key] = stream[field]
    rows = [
        {key: value[index] for key, value in arrays.items()} for index in range(len(arrays["time"]))
    ]
    recording = from_campaign_rows(
        json.loads((run / "episode_spec.json").read_text()),
        rows,
        json.loads((run / "finger_geometry.json").read_text()),
    )
    status = json.loads((run / "status.json").read_text())
    # The campaign never scored invalid trials; it recorded them without a score.
    if status.get("valid_trial") is False or status.get("status") == "infrastructure_failure":
        raise ValueError(f"Not a valid campaign trial ({status.get('status')!r}); it has no score")
    if status["status"] == "task_failure" and task != "task1-2":
        recording.status["status"] = "task_failure"
    return recording
