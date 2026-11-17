"""Posture flags inspect TCP evidence independently of action representation."""

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tower_eval.posture_flags import classify


def recording(tmp_path, angles):
    ordinary = np.column_stack(([0, 0, -1], [1, 0, 0], [0, -1, 0]))
    rotations = (
        ordinary @ Rotation.from_euler("x", np.asarray(angles)[:, None], degrees=True).as_matrix()
    )
    poses = np.zeros((len(angles), 2, 7))
    poses[:, :, 3:] = Rotation.from_matrix(rotations).as_quat()[:, None]
    path = tmp_path / "states.hdf5"
    with h5py.File(path, "w") as stream:
        stream["time"] = np.arange(len(angles)) / 20
        stream["eef"] = poses
        stream["eef_world"] = poses
    return path


def report():
    return dict(
        operations=[dict(id="grasp", arm=0, target=1, start=20, end=39)],
        target_intervals=[dict(arm=0, target=1, start=20, end=39)],
    )


@pytest.mark.parametrize(
    "angle,expected", [(0, "ordinary"), (180, "reversed_grasp"), (155, "reversed_grasp")]
)
def test_signed_closing_axis(tmp_path, angle, expected):
    result = classify(recording(tmp_path, np.full(40, angle)), report())
    assert result["status"] == expected
    assert result["operations"][0]["min_approach_down_dot"] == pytest.approx(1)


def test_short_blip_and_approach_window(tmp_path):
    angles = np.zeros(40)
    angles[5:9] = 180
    assert classify(recording(tmp_path, angles), report())["status"] == "ordinary"
    angles[9] = 180
    result = classify(recording(tmp_path, angles), report())
    assert result["status"] == "reversed_grasp"
    assert result["operations"][0]["reversed_consecutive_samples"] == 5


def test_joint_commands_cannot_bypass_recorded_check(tmp_path):
    # Joint actions never enter the runtime EEF exact-half-turn tripwire.
    # The recorded orientation still exposes a near-half-turn about approach.
    result = classify(recording(tmp_path, np.full(40, 170)), report())
    assert result["status"] == "reversed_grasp"
    assert result["operations"][0]["min_closing_dot"] < -0.98


def test_no_engagement(tmp_path):
    value = report()
    value["target_intervals"] = []
    result = classify(recording(tmp_path, np.full(40, 180)), value)
    assert result["status"] == "not_evaluated"
    assert result["operations"][0]["min_closing_dot"] is None


def test_base_drift(tmp_path):
    path = recording(tmp_path, np.zeros(40))
    with h5py.File(path, "r+") as stream:
        poses = stream["eef_world"][:]
        poses[-1, 0, 3:] = Rotation.from_euler("xyz", [1, 2, 3]).as_quat()
        stream["eef_world"][:] = poses
    assert classify(path, report())["status"] == "not_evaluated"


def test_missing_tcp(tmp_path):
    path = tmp_path / "states.hdf5"
    with h5py.File(path, "w"):
        pass
    assert classify(path, report())["status"] == "not_evaluated"
