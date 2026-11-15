"""Forward kinematics of the official AgileX Nero URDF (base -> gripper fingers).

Parses the three files copied from github.com/agilexrobotics/agx_arm_urdf (HEAD 680fa92):
nero_description.urdf, nero_with_flange_description.xacro, nero_with_gripper_description.xacro.
All poses are 4x4 matrices in the robot base_link frame, metres.
"""

import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1] / "models"
FILES = (
    "nero_description.urdf",
    "nero_with_flange_description.xacro",
    "nero_with_gripper_description.xacro",
)
# Sim USD names the flange link "gripper_flange"; the official repo renamed it to "flange_link".
SIM_NAME = {"flange_link": "gripper_flange"}


def _parse():
    text = "".join((HERE / "agilex" / name).read_text() for name in FILES)
    joints = {}
    for m in re.finditer(r'<joint name="(\w+)" type="(\w+)">(.*?)</joint>', text, re.S):
        name, kind, body = m.groups()
        origin = re.search(r'<origin xyz="([^"]+)" rpy="([^"]+)"', body)
        axis = re.search(r'<axis xyz="([^"]+)"', body)
        mimic = re.search(r'<mimic joint="(\w+)" multiplier="([^"]+)" offset="([^"]+)"', body)
        parent = re.search(r'<parent link="(\w+)"', body)
        child = re.search(r'<child link="(\w+)"', body)
        assert origin is not None and parent is not None and child is not None
        joints[name] = dict(
            kind=kind,
            parent=parent.group(1),
            child=child.group(1),
            xyz=np.array(origin.group(1).split(), float),
            rpy=np.array(origin.group(2).split(), float),
            axis=np.array(axis.group(1).split(), float) if axis else None,
            mimic=(mimic.group(1), float(mimic.group(2)), float(mimic.group(3))) if mimic else None,
        )
    return joints


JOINTS = _parse()


def _origin(j):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", j["rpy"]).as_matrix()
    T[:3, 3] = j["xyz"]
    return T


def fk(q7, width_m=0.07):
    """Return {link_name(sim naming): 4x4 pose in base_link} for 7 arm joints and total gripper width."""
    values = {f"joint{i + 1}": q for i, q in enumerate(q7)}
    values["gripper"] = width_m
    poses = {"base_link": np.eye(4)}
    pending = [n for n in JOINTS if JOINTS[n]["parent"] != "world"]
    while pending:
        for name in list(pending):
            j = JOINTS[name]
            if j["parent"] not in poses:
                continue
            T = poses[j["parent"]] @ _origin(j)
            if j["kind"] in ("revolute", "prismatic"):
                v = values.get(name, 0.0)
                if j["mimic"]:
                    src, mult, off = j["mimic"]
                    v = values[src] * mult + off
                M = np.eye(4)
                if j["kind"] == "revolute":
                    M[:3, :3] = Rotation.from_rotvec(j["axis"] * v).as_matrix()
                else:
                    M[:3, 3] = j["axis"] * v
                T = T @ M
            poses[j["child"]] = T
            pending.remove(name)
    return {SIM_NAME.get(k, k): v for k, v in poses.items()}


def limits():
    out = []
    for i in range(1, 8):
        body = re.search(
            rf'<joint name="joint{i}".*?<limit lower="([^"]+)" upper="([^"]+)"',
            "".join((HERE / "agilex" / n).read_text() for n in FILES),
            re.S,
        )
        assert body is not None
        out.append((float(body.group(1)), float(body.group(2))))
    return np.array(out)
