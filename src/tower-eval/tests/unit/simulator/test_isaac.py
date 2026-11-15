# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from tower_eval.simulator.commands import joint_commands

# The 2026.11 batch bootstrap maps replayed commands through joint_commands.
ARM = {
    "joint_names": [f"left_joint{i}" for i in range(1, 8)],
    "joint_indices": list(range(7)),
    "gripper_index": 7,
    "gripper_targets": [
        {"name": "left_finger1", "multiplier": 0.5},
        {"name": "left_finger2", "multiplier": -0.5, "offset_m": 0.01},
    ],
}
DOFS = ["right_joint1", "left_finger2", *ARM["joint_names"], "left_finger1"]


def test_joint_commands_maps_named_dofs_and_mimic_fingers():
    actions = np.arange(32, dtype=float).reshape(2, 16) / 100
    indices, values = joint_commands(actions, ARM, DOFS, units_per_meter=100)
    assert indices.dtype == np.int32 and values.dtype == np.float32
    assert indices.tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 1]
    np.testing.assert_array_equal(values[:, :7], actions[:, :7].astype(np.float32))
    width = actions[:, 7]
    np.testing.assert_allclose(values[:, 7], width * 0.5 * 100, rtol=1e-6)
    np.testing.assert_allclose(values[:, 8], (width * -0.5 + 0.01) * 100, rtol=1e-6)


@pytest.mark.parametrize(
    "actions",
    [np.zeros((2, 15)), np.zeros(16), np.full((1, 16), np.nan), np.full((1, 16), np.inf)],
)
def test_joint_commands_rejects_bad_shape_or_nonfinite(actions):
    with pytest.raises(ValueError, match="finite"):
        joint_commands(actions, ARM, DOFS, units_per_meter=1)
