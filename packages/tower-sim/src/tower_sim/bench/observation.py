"""Physical feedback to encoder joints and calibrated encoder TCP poses."""

from types import SimpleNamespace

import numpy as np
from nero_kinematics.fk import urdf


def encoder_observation(qpos, compensation, tcp_rotation, tcp_offset, *, fk=urdf.fk):
    state = SimpleNamespace(qpos=np.asarray(qpos))
    joints, transforms = [], []
    for arm, name in enumerate(("left", "back")):
        q = compensation.encoder_from_physical(
            state.qpos[arm * 8 : arm * 8 + 7], state.qpos[arm * 8 + 7], name
        )
        flange = fk(q, state.qpos[arm * 8 + 7])["gripper_base"]
        tcp = np.eye(4)
        tcp[:3, :3] = tcp_rotation.as_matrix()
        tcp[:3, 3] = tcp_offset
        transform = flange @ tcp
        if not np.allclose(np.linalg.det(transform[:3, :3]), 1.0):
            raise RuntimeError("Encoder FK returned invalid rotation")
        joints.append(q)
        transforms.append(transform)
    return dict(
        joint_encoder=np.array(joints).tolist(),
        eef_encoder=np.array(transforms).tolist(),
        gripper_width_m=state.qpos[[7, 15]].tolist(),
    )
