"""Cache only immutable FK structure; preserve the J2 finite difference and iterations."""

import copy
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


class CachedFK:
    def __init__(self, module):
        module = SimpleNamespace(
            JOINTS=copy.deepcopy(module.JOINTS),
            SIM_NAME=dict(module.SIM_NAME),
            _origin=module._origin,
        )
        self.module = module
        self.records = []
        known = {"base_link"}
        pending = [n for n in module.JOINTS if module.JOINTS[n]["parent"] != "world"]
        while pending:
            progressed = False
            for name in list(pending):
                j = module.JOINTS[name]
                if j["parent"] not in known:
                    continue
                self.records.append(
                    (
                        name,
                        j["kind"],
                        j["parent"],
                        j["child"],
                        module._origin(j),
                        None if j["axis"] is None else j["axis"].copy(),
                        j["mimic"],
                    )
                )
                known.add(j["child"])
                pending.remove(name)
                progressed = True
            if not progressed:
                raise ValueError("URDF tree has unresolved parent")

    def __call__(self, q7, width_m=0.07):
        values = {f"joint{i + 1}": q for i, q in enumerate(q7)}
        values["gripper"] = width_m
        poses = {"base_link": np.eye(4)}
        # Same matrix multiplication and SciPy rotation calls as the source FK.
        for name, kind, parent, child, origin, axis, mimic in self.records:
            T = poses[parent] @ origin
            if kind in ("revolute", "prismatic"):
                v = values.get(name, 0.0)
                if mimic:
                    src, mult, off = mimic
                    v = values[src] * mult + off
                M = np.eye(4)
                if kind == "revolute":
                    M[:3, :3] = Rotation.from_rotvec(axis * v).as_matrix()
                else:
                    M[:3, 3] = axis * v
                T = T @ M
            poses[child] = T
        return {self.module.SIM_NAME.get(k, k): v for k, v in poses.items()}
