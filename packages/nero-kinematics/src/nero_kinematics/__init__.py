# SPDX-License-Identifier: Apache-2.0
"""Pure IK: no simulator, robot connection, or actuator commands."""

import json
from pathlib import Path

import numpy as np

from .factory import create_solver as create_solver
from .posture import posture_evidence as posture_evidence
from .posture import solve_grasp as solve_grasp
from .profiles import COMMAND_BOUNDS as COMMAND_BOUNDS
from .profiles import SIMULATION_OBSERVATION_TOLERANCE as SIMULATION_OBSERVATION_TOLERANCE
from .solvers.ipopt import PinocchioIK, pose

__version__ = "0.2.0"
__all__ = [
    "ArmIK",
    "solve_pair",
    "create_solver",
    "posture_evidence",
    "solve_grasp",
    "COMMAND_BOUNDS",
    "SIMULATION_OBSERVATION_TOLERANCE",
]


class ArmIK:
    """One seven-axis arm; use serially and create one instance per arm."""

    def __init__(self, config_path):
        path = Path(config_path).resolve()
        config = json.loads(path.read_text())
        required = {
            "schema_version",
            "units",
            "urdf",
            "joint_names",
            "lower",
            "upper",
            "base_frame",
            "tcp_frame",
            "tcp_pose",
        }
        optional = {"model_status", "solver", "reference_configuration"}
        if (
            not isinstance(config, dict)
            or not required <= config.keys()
            or config.keys() - required - optional
        ):
            raise ValueError("Missing or unknown configuration fields")
        if config["schema_version"] != 1:
            raise ValueError("Unsupported configuration schema")
        if config["units"] != {"position": "m", "joints": "rad", "quaternion": "xyzw"}:
            raise ValueError("Only metres, radians and xyzw quaternions are supported")
        solver_options = config.get("solver", {})
        allowed = {
            "position_tolerance_m",
            "orientation_tolerance_rad",
            "position_weight",
            "orientation_weight",
            "seed_weight",
            "max_iter",
            "max_time_s",
            "optimizer_tolerance",
            "verified_current_seed",
        }
        if not isinstance(solver_options, dict) or solver_options.keys() - allowed:
            raise ValueError("Unknown solver configuration")
        options = {
            key: config[key] for key in ("lower", "upper", "base_frame", "tcp_frame", "tcp_pose")
        }
        options["reference_configuration"] = config.get("reference_configuration")
        self._solver = PinocchioIK(
            path.parent / config["urdf"], config["joint_names"], **options, **solver_options
        )
        self.model_status = config.get("model_status", "unspecified")

    @property
    def identity(self):
        """JSON-compatible model, limit, calibration and solver identity."""
        return json.loads(json.dumps({**self._solver.identity, "model_status": self.model_status}))

    def _joints(self, values):
        if np.iscomplexobj(values):
            raise ValueError("Joint angles must be real")
        q = np.asarray(values, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Expected seven finite joint angles in model joint-name order")
        if np.any(q < self._solver.lower - 1e-8) or np.any(q > self._solver.upper + 1e-8):
            raise ValueError("Joint angles exceed the configured model/deployment limits")
        return q

    def fk(self, joints):
        """Return TCP xyz+xyzw relative to the configured arm base."""
        return self._solver.fk(self._joints(joints)).tolist()

    def solve(
        self, pose_xyzw, measured_joints, *, frame="arm_base", joint_radius_rad=None, margin_rad=0.0
    ):
        """Return accepted absolute joints or None on numerical failure.

        Invalid input raises ValueError. A rejected candidate is never exposed
        as a joint command. measured_joints must be this arm's current reading.
        """
        if frame != "arm_base":
            raise ValueError("Standalone solver accepts only arm_base; transform world poses first")
        target, seed = pose(pose_xyzw), self._joints(measured_joints)
        joints, success = self._solver.ik(
            target, seed, joint_radius_rad=joint_radius_rad, margin_rad=margin_rad
        )
        assert self._solver.last_report is not None
        return {
            "success": success,
            "joints": joints.tolist() if success else None,
            "joint_names": list(self._solver.identity["joint_names"]),
            "backend": self._solver.backend,
            "diagnostics": dict(self._solver.last_report),
        }

    def reset(self):
        self._solver.reset()


def solve_pair(arms, poses_xyzw, measured_joints):
    """Two-arm all-or-none result, ordered exactly as the supplied arms.

    No actuator calls occur here. A failure withholds both joint targets.
    """
    if len(arms) != 2 or arms[0] is arms[1] or len(poses_xyzw) != 2 or len(measured_joints) != 2:
        raise ValueError("Expected two independent arm instances, poses and measured joint arrays")
    targets = [pose(value) for value in poses_xyzw]
    seeds = [arm._joints(value) for arm, value in zip(arms, measured_joints)]
    reports = [arm.solve(target, seed) for arm, target, seed in zip(arms, targets, seeds)]
    success = all(report["success"] for report in reports)
    return {
        "success": success,
        "joints": [report["joints"] for report in reports] if success else None,
        "backend": "pinocchio_casadi",
        "diagnostics": [report["diagnostics"] for report in reports],
    }
