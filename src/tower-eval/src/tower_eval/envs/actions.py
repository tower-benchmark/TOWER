# SPDX-License-Identifier: Apache-2.0
"""Gym space for the explicit union of legacy and tagged absolute actions."""

import numpy as np
from gymnasium import spaces

from ..constants import _SCHEMAS
from ..controllers.dual_mode import DualModeController


class DualActionSpace(spaces.Space):
    """A tagged union (not a fixed-size Box); sample() yields a joint action."""

    def __init__(self, controller: DualModeController):
        super().__init__(shape=None, dtype=None)
        self.controller = controller
        self.legacy = spaces.Box(
            controller.robot.lower.astype(np.float32),
            controller.robot.upper.astype(np.float32),
            dtype=np.float32,
        )

    def contains(self, x):
        try:
            if isinstance(x, dict):
                self.controller.validate(x)
                return True
            value = np.asarray(x, dtype=np.float32)
            return self.legacy.contains(value)
        except (ValueError, TypeError):
            return False

    def sample(self, mask=None, probability=None):
        robot = self.controller.robot
        q = self.np_random.uniform(robot.lower, robot.upper)
        return dict(
            input_space="joint",
            joints=q[list(robot.joint_indices)].reshape(2, 7).tolist(),
            gripper=q[[g.index for g in robot.grippers]].tolist(),
        )

    def to_jsonable(self, sample_n):
        from ..policies.serialization import jsonable

        return jsonable(sample_n)

    def from_jsonable(self, sample_n):
        if not all(self.contains(x) for x in sample_n):
            raise ValueError("Invalid serialized action")
        return sample_n

    def description(self):
        return dict(
            protocol=_SCHEMAS["dual_action"],
            legacy="absolute_joint16",
            input_space=["joint", "eef"],
            arm_order=["left", "back"],
            joints=dict(shape=[2, 7], unit="rad"),
            poses=dict(shape=[2, 7], fields="x y z qx qy qz qw", position_unit="m"),
            frame=["arm_base", "environment_world"],
            gripper=dict(shape=[2], unit="m", meaning="full_width"),
        )
