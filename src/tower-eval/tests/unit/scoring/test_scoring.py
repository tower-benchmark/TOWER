# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tower_eval.scoring import StackScorer


def fixture(offset=0.0, yaw=0.0, layers=2):
    poses = [[0, 0, 0.015, 0, 0, 0, 1]]
    destination_layers = []
    for level in range(1, layers + 1):
        rotation = Rotation.from_euler("z", 90 if level % 2 == 0 else 0, degrees=True).as_quat()
        for side in (-1, 1):
            xy = [side * 0.025, 0] if level % 2 == 0 else [0, side * 0.025]
            poses.append([*xy, 0.015 + 0.03 * level, *rotation])
            destination_layers.append(level)
    poses = np.asarray(poses, dtype=float)
    spec = NS(
        blocks=[
            NS(
                tower="destination",
                fixed_on_table=i == 0,
                dimensions_m=(0.15, 0.15, 0.03) if i == 0 else (0.15, 0.05, 0.03),
            )
            for i in range(len(poses))
        ],
        transfers=[
            NS(block_index=i, goal=poses[i].copy(), destination_layer=destination_layers[i - 1])
            for i in range(1, len(poses))
        ],
    )
    r = Rotation.from_euler("z", yaw, degrees=True)
    poses[1:, :3] = r.apply(poses[1:, :3]) + [offset, 0, 0]
    poses[1:, 3:] = (r * Rotation.from_quat(poses[1:, 3:])).as_quat()
    state = NS(
        blocks=poses, block_velocities=np.zeros((len(poses), 6)), qpos=np.zeros(16), time=0.0
    )
    state.qpos[[7, 15]] = 0.08
    return spec, state


def advance(scorer, state):
    for t in range(41):
        state.time = t / 10
        result = scorer.update(state)
    return result


def test_relaxed_rotated_offset_stack_and_strict_secondary():
    spec, state = fixture(0.015, 15)
    result = advance(StackScorer(spec), state)
    assert result["success"] and not result["strict_success"]
    assert result["layer_counts"] == [2, 2]
    spec, state = fixture()
    assert advance(StackScorer(spec), state)["strict_success"]


def test_three_layer_task13_stack():
    spec, state = fixture(layers=3)
    scorer = StackScorer(spec)
    result = advance(scorer, state)
    assert result["success"] and result["strict_success"]
    assert result["layer_counts"] == [2, 2, 2]
    assert scorer.protocol["required_blocks_per_layer"] == [2, 2, 2]


def test_task22_cross_stack():
    spec, state = fixture()
    selected = [0, 1, 3]
    state.blocks = state.blocks[selected].copy()
    state.blocks[1:, :2] = 0
    state.block_velocities = state.block_velocities[selected]
    spec.blocks = [spec.blocks[i] for i in selected]
    spec.transfers = [
        NS(block_index=i, destination_layer=i, goal=state.blocks[i].copy()) for i in (1, 2)
    ]
    spec.layout_metadata = {"catalog_task": 2}
    scorer = StackScorer(spec)
    assert scorer.required_blocks_per_layer == [1, 1]
    assert advance(scorer, state)["strict_success"]
    state.blocks[2, 2] = 0.045
    assert not advance(StackScorer(spec), state)["success"]
    spec.layout_metadata = {}
    with pytest.raises(ValueError):
        StackScorer(spec)


@pytest.mark.parametrize("failure", ["airborne", "table", "held", "moving", "one_layer", "outside"])
def test_reject_invalid_stack(failure):
    spec, state = fixture()
    if failure == "airborne":
        state.blocks[1:, 2] += 0.02
    if failure == "table":
        state.blocks[1:, 2] = 0.015
    if failure == "held":
        state.qpos[7] = 0.05
    if failure == "moving":
        state.block_velocities[1, 0] = 0.02
    if failure == "one_layer":
        state.blocks[3:, 2] = 0.045
    if failure == "outside":
        state.blocks[1:, 0] += 0.2
    assert not advance(StackScorer(spec), state)["success"]


def test_stability_resets_after_displacement():
    spec, state = fixture()
    scorer = StackScorer(spec)
    for t in range(25):
        state.time = t / 10
        scorer.update(state)
    assert not scorer.result["success"]
    state.blocks[1, 0] += 0.004
    state.time = 2.5
    assert scorer.update(state)["stable_seconds"] == 0


@pytest.mark.parametrize(
    "threshold,value,decision",
    [
        ("dwell_s", 5.0, "success"),
        ("open_gripper_width_m", 0.09, "released"),
        ("strict_position_m", 0.005, "strict_success"),
        ("strict_rotation_deg", 1.0, "strict_success"),
        ("support_height_tolerance_m", -1.0, "success"),
        ("minimum_support_fraction", 1.01, "success"),
        ("support_com_margin_m", -1.0, "success"),
        ("linear_velocity_m_s", -1.0, "success"),
        ("angular_velocity_rad_s", -1.0, "success"),
        ("anchor_translation_m", -1.0, "success"),
        ("anchor_rotation_deg", -1.0, "success"),
        ("required_blocks_per_layer", [3, 3], "success"),
    ],
)
def test_protocol_threshold_changes_decision(threshold, value, decision):
    spec, state = fixture(
        offset=0.007 if threshold == "strict_position_m" else 0.0,
        yaw=3.0 if threshold == "strict_rotation_deg" else 0.0,
    )
    assert advance(StackScorer(spec), state)[decision]
    scorer = StackScorer(spec)
    scorer.protocol[threshold] = value
    assert not advance(scorer, state)[decision]
