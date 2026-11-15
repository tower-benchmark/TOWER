"""Pure EEF FK/model initialization using nero-kinematics; IK is a separate service."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from nero_kinematics.fk.sdk_mdh import sdk_transform
from nero_kinematics.profiles import COMMAND_BOUNDS
from scipy.spatial.transform import Rotation


def fixed_transform(joint):
    origin = joint.find("origin")
    transform = np.eye(4)
    if origin is not None:
        transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return transform


class EefModel:
    """Explicit scene model and TCP calibration, with no vendor configuration or files written."""

    def __init__(
        self, *, scene_urdf, expected_scene_urdf_sha256=None, tcp_offset=None, tcp_rotation=None
    ):
        # The caller must select the scene export explicitly. A cache digest, when
        # supplied, binds these exact bytes; never substitute a packaged model.
        if scene_urdf is None:
            raise ValueError("An explicit scene URDF is required")
        scene_urdf = Path(scene_urdf)
        source = scene_urdf.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if expected_scene_urdf_sha256 is not None and digest != expected_scene_urdf_sha256:
            raise ValueError("Scene URDF does not match the verified cache digest")
        joints = {j.get("name"): j for j in ET.fromstring(source).findall("joint")}
        limits = []
        for i in range(1, 8):
            limit = joints[f"joint{i}"].find("limit")
            if limit is None:
                raise ValueError("Scene joint has no limits")
            if not {"lower", "upper"} <= limit.attrib.keys():
                raise ValueError("Scene joint limit requires lower and upper attributes")
            limits.append([float(limit.attrib["lower"]), float(limit.attrib["upper"])])
        limits = np.asarray(limits)
        if not np.isfinite(limits).all():
            raise ValueError("Expected seven finite joint limits in radians")
        # Preserve the frozen model's hard_limits(scene) GUI intersection, before
        # any caller applies its own inset. These are not command bounds.
        # TOWER 2026.11 GUI_LOWER / GUI_UPPER (unchanged values).
        self.lower = np.maximum(
            limits[:, 0], np.deg2rad([-155.0, -100.0, -158.0, -58.0, -158.0, -42.0, -90.0])
        )
        self.upper = np.minimum(
            limits[:, 1], np.deg2rad([155.0, 100.0, 158.0, 123.0, 158.0, 55.0, 90.0])
        )
        if np.any(self.lower >= self.upper):
            raise ValueError("Empty GUI/model joint limit intersection")
        self.command_lower = np.array(COMMAND_BOUNDS.lower_rad)
        self.command_upper = np.array(COMMAND_BOUNDS.upper_rad)
        for bound in (self.lower, self.upper, self.command_lower, self.command_upper):
            bound.flags.writeable = False
        self.link7_to_gripper = fixed_transform(joints["gripper_flange_joint"]) @ fixed_transform(
            joints["gripper_base_joint"]
        )
        self.tcp_offset = np.zeros(3) if tcp_offset is None else np.array(tcp_offset, float)
        self.tcp_rotation = Rotation.identity() if tcp_rotation is None else tcp_rotation
        if self.tcp_offset.shape != (3,) or not np.isfinite(self.tcp_offset).all():
            raise ValueError("Invalid TCP offset")
        self.identity: dict[str, Any] = dict(
            backend="nero-kinematics-sdk-mdh",
            model_bounds="scene-urdf-intersect-tower-gui-hard",
            command_bounds_profile="command",
            gui_limits_source="tower-2026.11 GUI joint limits (GUI_LOWER/GUI_UPPER)",
            scene_urdf_sha256=digest,
        )
        errors = []
        rng = np.random.default_rng(5)
        for q in [np.zeros(7), *rng.uniform(self.lower + 0.2, self.upper - 0.2, (32, 7))]:
            actual = np.eye(4)
            for i, angle in enumerate(q, 1):
                joint = joints[f"joint{i}"]
                moving = np.eye(4)
                axis = joint.find("axis")
                if axis is None:
                    raise ValueError("Scene joint has no axis")
                moving[:3, :3] = Rotation.from_rotvec(
                    np.fromstring(axis.get("xyz", ""), sep=" ") * angle
                ).as_matrix()
                actual = actual @ fixed_transform(joint) @ moving
            expected = sdk_transform(q)
            errors.append(
                [
                    np.linalg.norm(actual[:3, 3] - expected[:3, 3]),
                    (
                        Rotation.from_matrix(actual[:3, :3]).inv()
                        * Rotation.from_matrix(expected[:3, :3])
                    ).magnitude(),
                ]
            )
        maximum = np.max(errors, axis=0)
        self.identity["sdk_scene_fk_max_error"] = maximum.tolist()
        if maximum[0] > 0.0001 or maximum[1] > 0.0005:
            raise ValueError("SDK model and physical scene kinematics disagree")

    def tcp_transform(self):
        calibration = np.eye(4)
        calibration[:3, 3], calibration[:3, :3] = self.tcp_offset, self.tcp_rotation.as_matrix()
        return self.link7_to_gripper @ calibration

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Expected seven finite radians")
        transform = sdk_transform(q) @ self.tcp_transform()
        return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()]
