"""Fast Nero kinematics (base_link -> TCP) with damped least-squares IK.

Chain constants come from runtime/urdf_fk.py (official AgileX URDF); the TCP
offset/rotation is the sim runtime TCP (runtime/tcp.json). All poses are in
the arm base frame, metres.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from ..profiles import load_profile
from . import urdf as urdf_fk

TCP = load_profile("tcp")
SETTINGS = load_profile("numpy-dls")
TCP_OFFSET = np.asarray(TCP["tcp_offset_m"])
TCP_ROT = R.from_quat(TCP["tcp_rotation_xyzw"]).as_matrix()
LIMITS = urdf_fk.limits()


def _origin(j):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", j["rpy"]).as_matrix()
    T[:3, 3] = j["xyz"]
    return T


def _chain(tip="gripper_base"):
    by_child = {j["child"]: (n, j) for n, j in urdf_fk.JOINTS.items()}
    steps, link = [], tip
    sim_name = {v: k for k, v in urdf_fk.SIM_NAME.items()}
    link = sim_name.get(link, link)
    while link != "base_link":
        name, j = by_child[link]
        steps.append((name, j))
        link = j["parent"]
    return steps[::-1]


CHAIN = []
for _name, _j in _chain():
    CHAIN.append(
        (
            _origin(_j),
            int(_name[5:]) - 1 if _j["kind"] == "revolute" else None,
            None if _j["axis"] is None else np.asarray(_j["axis"], float),
        )
    )
TCP_LOCAL = np.eye(4)
TCP_LOCAL[:3, :3] = TCP_ROT
TCP_LOCAL[:3, 3] = TCP_OFFSET


def _rot(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def fk(q, with_axes=False):
    """TCP 4x4 in base frame. With axes: also joint axes/origins in base frame."""
    T = np.eye(4)
    axes, points = [], []
    for origin, index, axis in CHAIN:
        T = T @ origin
        if index is not None:
            if with_axes:
                axes.append(T[:3, :3] @ axis)
                points.append(T[:3, 3].copy())
            M = np.eye(4)
            M[:3, :3] = _rot(axis, q[index])
            T = T @ M
    T = T @ TCP_LOCAL
    if with_axes:
        return T, np.asarray(axes), np.asarray(points)
    return T


def jacobian(q):
    T, axes, points = fk(q, True)
    Jv = np.cross(axes, T[:3, 3] - points).T
    Jw = axes.T
    return T, np.vstack([Jv, Jw])


def pose_error(T, target):
    ep = target[:3, 3] - T[:3, 3]
    er = R.from_matrix(target[:3, :3] @ T[:3, :3].T).as_rotvec()
    return ep, er


def ik(
    target,
    seed,
    rest=None,
    iterations=SETTINGS["iterations"],
    damping=SETTINGS["damping"],
    null_gain=SETTINGS["null_gain"],
    tol_p=SETTINGS["position_tolerance_m"],
    tol_r=SETTINGS["rotation_tolerance_rad"],
):
    """Damped least squares, null-space pull to `rest`. Returns q, pos_err_m, rot_err_rad."""
    q = np.array(seed, float)
    rest = q.copy() if rest is None else np.asarray(rest, float)
    margin = np.radians(SETTINGS["margin_deg"])
    lo, hi = LIMITS[:, 0] + margin, LIMITS[:, 1] - margin
    q = np.clip(q, lo, hi)
    for _ in range(iterations):
        T, J = jacobian(q)
        ep, er = pose_error(T, target)
        if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
            break
        e = np.r_[ep, er]
        JJt = J @ J.T + damping**2 * np.eye(6)
        Jpinv = J.T @ np.linalg.inv(JJt)
        dq = Jpinv @ e
        dq += (np.eye(7) - Jpinv @ J) @ (null_gain * (rest - q))
        step = np.abs(dq).max()
        if step > 0.3:
            dq *= 0.3 / step
        q = np.clip(q + dq, lo, hi)
    T = fk(q)
    ep, er = pose_error(T, target)
    if np.linalg.norm(ep) > 1e-4 or np.linalg.norm(er) > 1e-3:
        # Clipping a pseudoinverse step can stall at a bound. A bounded solve
        # redistributes the remaining motion over the other six joints.
        def residual(x):
            p, r = pose_error(fk(x), target)
            return np.r_[p, r, 1e-5 * (x - rest)]

        solved = least_squares(
            residual, q, bounds=(lo, hi), max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-10
        )
        q = solved.x
        ep, er = pose_error(fk(q), target)
    return q, float(np.linalg.norm(ep)), float(np.linalg.norm(er))


def pose(position, rotation):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T
