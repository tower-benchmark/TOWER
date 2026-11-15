"""Finite TCP poses and numerical residuals, independent of any IK backend."""

import numpy as np
from scipy.spatial.transform import Rotation


def pose_error(actual, target):
    actual, target = pose(actual), pose(target)
    return (
        float(np.linalg.norm(actual[:3] - target[:3])),
        float((Rotation.from_quat(target[3:]).inv() * Rotation.from_quat(actual[3:])).magnitude()),
    )


def pose(value):
    if np.iscomplexobj(value):
        raise ValueError("Pose must be real")
    value = np.asarray(value, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("Pose must be finite xyz + xyzw (7,) in metres")
    norm = np.linalg.norm(value[3:])
    if abs(norm - 1) > 1e-4:
        raise ValueError("Pose requires unit xyzw quaternion")
    return np.r_[value[:3], value[3:] / norm]
