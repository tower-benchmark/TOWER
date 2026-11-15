"""Pure command mapping and initial-state validation, without backend imports."""

import numpy as np


def joint_commands(actions, arm, dof_names, units_per_meter):
    """Map SI robot commands to named articulation DOFs, including mimic joints."""
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all():
        raise ValueError("Expected finite (N,16) commands")
    names = arm["joint_names"] + [g["name"] for g in arm["gripper_targets"]]
    indices = np.asarray([dof_names.index(n) for n in names], dtype=np.int32)
    values = [actions[:, i] for i in arm["joint_indices"]]
    width = actions[:, arm["gripper_index"]]
    values += [
        (width * g["multiplier"] + g.get("offset_m", 0.0)) * units_per_meter
        for g in arm["gripper_targets"]
    ]
    return indices, np.stack(values, axis=1).astype(np.float32)


def validate_initial_robot_state(saved, robot):
    if tuple(saved.get("names", ())) != tuple(robot.names):
        raise ValueError("Initial robot state channel order mismatch")
    qpos = np.asarray(saved.get("qpos"), dtype=float)
    if qpos.shape != (16,) or not np.isfinite(qpos).all():
        raise ValueError("Initial robot state must contain 16 finite positions")
    if np.any(qpos < robot.lower) or np.any(qpos > robot.upper):
        raise ValueError("Initial robot state exceeds configured articulation limits")
    return qpos.copy()
