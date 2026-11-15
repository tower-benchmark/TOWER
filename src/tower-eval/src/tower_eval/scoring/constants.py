# SPDX-License-Identifier: Apache-2.0
"""Shared physical dimensions and recorded channel identities (not IK limits)."""

BLOCK_SIZE_M = (0.15, 0.05, 0.03)
GRIPPER_CHANNELS = [7, 15]
ARM_JOINT_CHANNELS = (tuple(range(7)), tuple(range(8, 15)))
