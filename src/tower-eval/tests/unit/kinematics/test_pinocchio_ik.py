# SPDX-License-Identifier: Apache-2.0
"""Optional real solver tests. TOWER_NERO_URDF selects the deployment model."""

import importlib
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tower_eval.kinematics.pinocchio_ik import PinocchioIK, PinocchioKinematics, pose_error

if os.environ.get("TOWER_REQUIRE_NATIVE_IK") == "1":
    importlib.import_module("pinocchio.casadi")
    importlib.import_module("casadi")
else:
    pytest.importorskip("pinocchio.casadi")
    pytest.importorskip("casadi")


@pytest.fixture
def solver():
    model = os.environ.get("TOWER_NERO_URDF")
    if not model or not Path(model).is_file():
        if os.environ.get("TOWER_REQUIRE_NATIVE_IK") == "1":
            pytest.fail("Native CI requires an existing TOWER_NERO_URDF")
        pytest.skip("Set TOWER_NERO_URDF to calibrated Nero model")
    return PinocchioIK(
        model,
        [f"joint{i}" for i in range(1, 8)],
        lower=[-3] * 7,
        upper=[3] * 7,
        tcp_pose=[0.012, 0.02, -0.03, 0, 0, 0, 1],
        max_time_s=0.2,
    )


def test_reachable_near_limits_unreachable_sign_and_reset(solver):
    rng = np.random.default_rng(110)
    for fraction in [np.full(7, 0.5), np.full(7, 0.92), *rng.uniform(0.2, 0.8, (20, 7))]:
        q = solver.lower + fraction * (solver.upper - solver.lower)
        target = solver.fk(q)
        found, ok = solver.ik(target, q + 0.01)
        assert ok, solver.last_report
        assert pose_error(solver.fk(found), target)[0] <= 0.001
        target[3:] *= -1
        found2, ok = solver.ik(target, q + 0.01)
        assert ok and pose_error(solver.fk(found2), target)[1] <= 0.01
    target[:3] = [10, 10, 10]
    _, ok = solver.ik(target, q)
    assert not ok
    solver.reset()
    assert solver.last_report is None


def test_frames_instances_name_order_and_offset(solver):
    other = PinocchioIK(
        os.environ["TOWER_NERO_URDF"],
        [f"joint{i}" for i in range(7, 0, -1)],
        lower=solver.lower[::-1],
        upper=solver.upper[::-1],
        tcp_pose=[0.012, 0.02, -0.03, 0, 0, 0, 1],
        max_time_s=0.2,
    )
    q = (solver.lower + solver.upper) / 2
    assert np.allclose(solver.fk(q), other.fk(q[::-1]))
    bases = [[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 1, 0]]
    k = PinocchioKinematics([solver, other], bases, [10, 0, 0])
    for a, base in enumerate(bases):
        target = solver.fk(q)
        rotation = Rotation.from_quat(base[3:])
        world = np.r_[
            rotation.apply(target[:3]) + base[:3] - np.array([10, 0, 0]),
            (rotation * Rotation.from_quat(target[3:])).as_quat(),
        ]
        assert pose_error(k.to_base(a, world, "environment_world"), target)[0] < 1e-12
    k.ik(0, solver.fk(q), q)
    assert other.last_report is None
    k.reset()
    assert solver.last_report is None


def test_iteration_and_time_rejection(solver):
    solver.max_time_s = 1e-12
    q = (solver.lower + solver.upper) / 2
    _, ok = solver.ik(solver.fk(q), q + 0.01)
    assert not ok and solver.last_report["solver_status"].startswith("wall_time_exceeded")


def test_continuous_joint_configuration_and_max_iterations(solver, tmp_path):
    import xml.etree.ElementTree as ET

    tree = ET.parse(os.environ["TOWER_NERO_URDF"])
    node = tree.getroot().find("joint[@name='joint1']")
    node.set("type", "continuous")
    path = tmp_path / "continuous.urdf"
    tree.write(path)
    continuous = PinocchioIK(
        path,
        [f"joint{i}" for i in range(1, 8)],
        lower=solver.lower,
        upper=solver.upper,
        tcp_pose=[0.012, 0.02, -0.03, 0, 0, 0, 1],
        max_time_s=0.2,
    )
    assert continuous.model.nq == 8 and continuous.model.nv == 7
    q = (solver.lower + solver.upper) / 2
    assert pose_error(solver.fk(q), continuous.fk(q))[0] < 1e-12
    found, ok = continuous.ik(continuous.fk(q), q + 0.01)
    assert ok and pose_error(continuous.fk(found), continuous.fk(q))[0] < 0.001
    limited = PinocchioIK(
        path,
        [f"joint{i}" for i in range(1, 8)],
        lower=solver.lower,
        upper=solver.upper,
        max_iter=1,
        max_time_s=0.2,
    )
    _, ok = limited.ik(limited.fk(q + 0.1), q)
    assert not ok and limited.last_report["solver_status"] == "Maximum_Iterations_Exceeded"


def test_real_changed_pose_changes_driver_targets(solver):
    from types import SimpleNamespace

    from tower_eval.controllers import DualModeController, RobotSpec
    from tower_eval.envs import TowerEnv

    other = PinocchioIK(
        os.environ["TOWER_NERO_URDF"],
        [f"joint{i}" for i in range(1, 8)],
        lower=solver.lower,
        upper=solver.upper,
        tcp_pose=[0.012, 0.02, -0.03, 0, 0, 0, 1],
        max_time_s=0.2,
    )
    kin = PinocchioKinematics([solver, other], [[0, 0, 0, 0, 0, 0, 1]] * 2)
    robot = RobotSpec.from_dict(
        dict(
            names=[f"q{i}" for i in range(16)],
            joint_indices=[*range(7), *range(8, 15)],
            grippers=[dict(index=i, lower=0.0, upper=0.1) for i in (7, 15)],
            lower=np.r_[solver.lower, 0.0, solver.lower, 0.0],
            upper=np.r_[solver.upper, 0.1, solver.upper, 0.1],
        )
    )
    q = (solver.lower + solver.upper) / 2

    class Drive:
        dt = 1 / 30

        def __init__(self):
            self.robot = robot
            self.state = SimpleNamespace(qpos=np.r_[q, 0.1, q, 0.1], qvel=np.zeros(16), time=0.0)
            self.targets = []

        def read(self):
            return self.state

        def step(self, target):
            self.targets.append(np.asarray(target).copy())
            self.state = SimpleNamespace(
                qpos=np.asarray(target), qvel=np.zeros(16), time=self.state.time + self.dt
            )
            return self.state

    drive = Drive()
    env = TowerEnv(drive, DualModeController(robot, kinematics=kin))
    pose1 = solver.fk(q)
    pose2 = solver.fk(q + np.array([0.1, 0.04, 0, 0.1, 0, 0.03, 0.06]))
    assert pose_error(pose1, pose2)[0] > 0.01 and pose_error(pose1, pose2)[1] > 0.05
    for target in [pose1, pose2]:
        env.step(
            dict(
                input_space="eef", poses=[target.tolist()] * 2, gripper=[0.1, 0.1], frame="arm_base"
            )
        )
    assert np.linalg.norm(drive.targets[1] - drive.targets[0]) > 0.05
    assert pose_error(solver.fk(drive.targets[1][:7]), pose2)[0] < 0.001
