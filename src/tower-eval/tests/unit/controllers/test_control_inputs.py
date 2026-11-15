# SPDX-License-Identifier: Apache-2.0
import json

import h5py
import numpy as np
import pytest
from tower_eval.controllers.inputs import load_parameters, load_raw


def raw_file(tmp_path):
    path = tmp_path / "raw.hdf5"
    with h5py.File(path, "w") as h:
        h.attrs.update(
            arm_order=json.dumps(["left", "back"]),
            recording_complete=True,
            joint_names=json.dumps([f"joint{i}" for i in range(1, 8)] + ["gripper"]),
            valid_frames=4,
            action_alignment="row observation precedes action interval; terminal row action_valid=false",
        )
        h["sim/time"] = np.arange(4) / 30
        h["sim/action_valid"] = [True, True, True, False]
        h["lifter/action_mm"] = np.zeros((4, 2))
        for key in ("leader/qpos", "sim/controller_qpos", "observations/qpos"):
            h[key] = np.full((4, 16), 0.05)
        h["sim/block_poses"] = np.zeros((4, 10, 7))
        h["sim"].attrs["block_ids"] = json.dumps([str(i) for i in range(10)])
        for key in ("observations/eef", "leader/eef"):
            poses = np.zeros((4, 14))
            poses[:, [6, 13]] = 1
            poses[:, 0] = np.arange(4) * 0.01
            h[key] = poses
            h[key].attrs["frame"] = "arm_base"
            h[key].attrs["fields"] = json.dumps(
                [
                    f"{a}_{f}"
                    for a in ("left", "back")
                    for f in ("x", "y", "z", "qx", "qy", "qz", "qw")
                ]
            )
    return path


def test_observed_eef_uses_endpoint_but_width_and_leader_use_interval_start(tmp_path):
    path = raw_file(tmp_path)
    sequence = load_raw(path, input_space="eef")
    assert len(sequence.actions) == 3
    assert sequence.actions[0]["poses"][0][0] == 0.01
    assert sequence.actions[-1]["poses"][0][0] == 0.03
    assert sequence.actions[0]["gripper"] == [0.05, 0.05]
    leader = load_raw(path, input_space="eef", eef_dataset="leader/eef")
    assert leader.actions[0]["poses"][0][0] == 0.0
    assert leader.metadata["eef_sampling"] == "interval_start"


def test_joint_mode_uses_command_source_and_never_terminal_action(tmp_path):
    sequence = load_raw(raw_file(tmp_path), input_space="joint")
    assert len(sequence.actions) == 3
    np.testing.assert_allclose(sequence.actions[0]["joints"], 0.05)
    assert sequence.dt_s == 1 / 30


def test_joint_order_mismatch_is_rejected_before_import(tmp_path):
    path = raw_file(tmp_path)
    with h5py.File(path, "r+") as h:
        h.attrs["joint_names"] = json.dumps(["gripper"] + [f"joint{i}" for i in range(1, 8)])
    with pytest.raises(ValueError, match="joint order"):
        load_raw(path, input_space="joint")


@pytest.mark.parametrize(
    "field,value",
    [
        ("sim/action_valid", [True] * 4),
        ("sim/time", [0.0, 0.1, 0.2, 0.3]),
        ("lifter/action_mm", np.ones((4, 2))),
    ],
)
def test_invalid_raw_timing_and_lifters_are_rejected(tmp_path, field, value):
    path = raw_file(tmp_path)
    with h5py.File(path, "r+") as h:
        h[field][:] = value
    with pytest.raises(ValueError):
        load_raw(path, input_space="eef")


def test_json_file_can_explicitly_switch_between_two_input_types(tmp_path):
    path = tmp_path / "parameters.json"
    actions = [
        dict(input_space="joint", joints=np.zeros((2, 7)).tolist(), gripper=[0.05, 0.05]),
        dict(
            input_space="eef",
            poses=[[0, 0, 0, 0, 0, 0, 1]] * 2,
            gripper=[0.05, 0.05],
            frame="arm_base",
        ),
    ]
    path.write_text(json.dumps(dict(dt_s=1 / 30, actions=actions)))
    assert load_parameters(path).actions == actions
    actions[0].pop("input_space")
    path.write_text(json.dumps(dict(dt_s=1 / 30, actions=actions)))
    with pytest.raises(ValueError):
        load_parameters(path)
