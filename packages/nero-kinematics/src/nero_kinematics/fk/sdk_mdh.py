"""SDK modified-DH FK; preserved operation order for the frozen oracle."""

import numpy as np
from scipy.spatial.transform import Rotation


def sdk_transform(q):
    """SDK modified-DH link7 FK; checked against Pinocchio and scene URDF."""
    rows = [
        (0.138, 0.0, 0.0),
        (0.0, np.pi / 2, np.pi),
        (0.31, np.pi / 2, np.pi),
        (0.0, np.pi / 2, np.pi),
        (0.27001, np.pi / 2, np.pi / 2),
        (0.0, np.pi / 2, np.pi / 2),
        (0.0235, np.pi / 2, 0.0),
    ]
    transform = np.eye(4)
    for angle, (d, alpha, offset) in zip(q, rows):
        fixed, moving = np.eye(4), np.eye(4)
        fixed[:3, :3] = Rotation.from_euler("x", alpha).as_matrix()
        moving[:3, :3] = Rotation.from_euler("z", angle + offset).as_matrix()
        moving[2, 3] = d
        transform = transform @ fixed @ moving
    return transform
